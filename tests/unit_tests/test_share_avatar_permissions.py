"""Unit tests for who may share an avatar.

Sharing used to be admin-only: every ordinary user got 401 "Users may only share
avatars of themselves", which meant the one person entitled to publish an avatar
of themselves was the one person who could not. The rule is now the same one the
rest of the API applies to an avatar — the creator decides — resolved from
``metadata.user_id``.
"""

from types import SimpleNamespace

import pytest

from src.api import webapp as webapp_module

ASSISTANT_ID = "assistant-alpha"
CREATOR_ID = "6a5e59310832afadd626e583"
STRANGER_ID = "someone-else"
ADMIN_ID = "the-admin"


def _current_user(user_id):
    return {"API_KEY": "sk-test-key", "identities": [{"user_id": user_id}]}


class _AssistantsAPI:
    def __init__(self, metadata):
        self._metadata = metadata
        self.updates = []

    async def get(self, assistant_id):
        return {"assistant_id": assistant_id, "metadata": self._metadata}

    async def update(self, assistant_id, metadata):
        self.updates.append((assistant_id, metadata))
        return {"assistant_id": assistant_id, "metadata": metadata}


def _install(monkeypatch, metadata):
    assistants_api = _AssistantsAPI(metadata)
    monkeypatch.setattr(
        webapp_module,
        "get_client",
        lambda **kwargs: SimpleNamespace(assistants=assistants_api),
    )
    monkeypatch.setattr(
        webapp_module.app.state,
        "context",
        SimpleNamespace(admin_user_id=ADMIN_ID),
        raising=False,
    )
    return assistants_api


@pytest.mark.asyncio
async def test_the_creator_may_share_their_own_avatar(monkeypatch):
    assistants_api = _install(monkeypatch, {"user_id": CREATOR_ID})

    response = await webapp_module.share_avatar(
        assistant_id=ASSISTANT_ID,
        is_public=True,
        current_user=_current_user(CREATOR_ID),
    )

    assert response.status_code == 200
    assert assistants_api.updates == [(ASSISTANT_ID, {"is_public": True})]


@pytest.mark.asyncio
async def test_the_creator_may_withdraw_a_shared_avatar(monkeypatch):
    assistants_api = _install(monkeypatch, {"user_id": CREATOR_ID, "is_public": True})

    await webapp_module.share_avatar(
        assistant_id=ASSISTANT_ID,
        is_public=False,
        current_user=_current_user(CREATOR_ID),
    )

    assert assistants_api.updates == [(ASSISTANT_ID, {"is_public": False})]


@pytest.mark.asyncio
async def test_a_stranger_may_not_share_someone_elses_avatar(monkeypatch):
    assistants_api = _install(monkeypatch, {"user_id": CREATOR_ID})

    with pytest.raises(webapp_module.HTTPException) as rejection:
        await webapp_module.share_avatar(
            assistant_id=ASSISTANT_ID,
            is_public=True,
            current_user=_current_user(STRANGER_ID),
        )

    assert rejection.value.status_code == 403
    assert assistants_api.updates == []


@pytest.mark.asyncio
async def test_an_avatar_with_no_recorded_creator_is_not_shareable(monkeypatch):
    """Absent ownership fails closed — publishing is not the safe default."""
    assistants_api = _install(monkeypatch, {})

    with pytest.raises(webapp_module.HTTPException) as rejection:
        await webapp_module.share_avatar(
            assistant_id=ASSISTANT_ID,
            is_public=True,
            current_user=_current_user(CREATOR_ID),
        )

    assert rejection.value.status_code == 403
    assert assistants_api.updates == []


@pytest.mark.asyncio
async def test_the_admin_may_still_share_any_avatar(monkeypatch):
    assistants_api = _install(monkeypatch, {"user_id": CREATOR_ID})

    response = await webapp_module.share_avatar(
        assistant_id=ASSISTANT_ID,
        is_public=True,
        current_user=_current_user(ADMIN_ID),
    )

    assert response.status_code == 200
    assert assistants_api.updates == [(ASSISTANT_ID, {"is_public": True})]
