"""Unit tests for the ownership check on GET /conversations/{thread_id}/messages.

The endpoint takes both a thread id and an ``assistant_id``, and until recently
used only the first: any thread whose id a caller could name was returned, under
whatever avatar the caller claimed. That is how a client bug managed to display
one avatar's transcript inside another avatar's chat window without anything
failing — the mismatch was accepted rather than reported.

These tests pin the check down, including the deliberate exception: a thread
created before threads carried ownership metadata is still readable, because
making old conversations disappear would be a worse bug than the one being fixed.
"""

from types import SimpleNamespace

import pytest

from src.api import webapp as webapp_module

THREAD_ID = "thread-abc"
ASSISTANT_ID = "assistant-alpha"
OTHER_ASSISTANT_ID = "assistant-beta"
USER_ID = "6a5e59310832afadd626e583"
OTHER_USER_ID = "another-user"

STORED_MESSAGES = [
    {"type": "human", "content": "hello"},
    {"type": "ai", "content": "hi there"},
]


def _current_user(user_id=USER_ID):
    return {
        "API_KEY": "sk-test-key",
        "identities": [{"user_id": user_id}],
    }


class _ThreadsAPI:
    def __init__(self, thread_metadata):
        self._thread_metadata = thread_metadata
        self.state_reads = []

    async def get(self, thread_id):
        if self._thread_metadata is None:
            raise RuntimeError("no such thread")
        return {"thread_id": thread_id, "metadata": self._thread_metadata}

    async def get_state(self, thread_id):
        self.state_reads.append(thread_id)
        return {"values": {"messages": STORED_MESSAGES}}


def _install_client(monkeypatch, thread_metadata):
    threads_api = _ThreadsAPI(thread_metadata)
    monkeypatch.setattr(
        webapp_module,
        "get_client",
        lambda **kwargs: SimpleNamespace(threads=threads_api),
    )
    return threads_api


def _owned_thread_metadata(assistant_id=ASSISTANT_ID, user_id=USER_ID):
    return {"thread_metadata": {"assistant_id": assistant_id, "user_id": user_id}}


@pytest.mark.asyncio
async def test_a_thread_is_returned_to_the_avatar_it_belongs_to(monkeypatch):
    threads_api = _install_client(monkeypatch, _owned_thread_metadata())

    response = await webapp_module.get_thread_messages(
        request=SimpleNamespace(),
        thread_id=THREAD_ID,
        assistant_id=ASSISTANT_ID,
        current_user=_current_user(),
    )

    assert response.status_code == 200
    assert threads_api.state_reads == [THREAD_ID]


@pytest.mark.asyncio
async def test_a_thread_is_not_served_under_a_different_avatar(monkeypatch):
    """The exact defect: right user, wrong avatar, someone else's transcript."""
    threads_api = _install_client(
        monkeypatch, _owned_thread_metadata(assistant_id=OTHER_ASSISTANT_ID)
    )

    with pytest.raises(webapp_module.HTTPException) as rejection:
        await webapp_module.get_thread_messages(
            request=SimpleNamespace(),
            thread_id=THREAD_ID,
            assistant_id=ASSISTANT_ID,
            current_user=_current_user(),
        )

    assert rejection.value.status_code == 404
    # The messages must not even be read, let alone returned.
    assert threads_api.state_reads == []


@pytest.mark.asyncio
async def test_a_thread_is_not_served_to_a_different_user(monkeypatch):
    threads_api = _install_client(
        monkeypatch, _owned_thread_metadata(user_id=OTHER_USER_ID)
    )

    with pytest.raises(webapp_module.HTTPException) as rejection:
        await webapp_module.get_thread_messages(
            request=SimpleNamespace(),
            thread_id=THREAD_ID,
            assistant_id=ASSISTANT_ID,
            current_user=_current_user(),
        )

    assert rejection.value.status_code == 404
    assert threads_api.state_reads == []


@pytest.mark.asyncio
async def test_a_thread_without_ownership_metadata_is_still_readable(monkeypatch):
    """Threads predating the metadata must not become unreachable."""
    threads_api = _install_client(monkeypatch, {})

    response = await webapp_module.get_thread_messages(
        request=SimpleNamespace(),
        thread_id=THREAD_ID,
        assistant_id=ASSISTANT_ID,
        current_user=_current_user(),
    )

    assert response.status_code == 200
    assert threads_api.state_reads == [THREAD_ID]


@pytest.mark.asyncio
async def test_an_unknown_thread_is_a_404_not_a_500(monkeypatch):
    _install_client(monkeypatch, None)

    with pytest.raises(webapp_module.HTTPException) as rejection:
        await webapp_module.get_thread_messages(
            request=SimpleNamespace(),
            thread_id=THREAD_ID,
            assistant_id=ASSISTANT_ID,
            current_user=_current_user(),
        )

    assert rejection.value.status_code == 404
