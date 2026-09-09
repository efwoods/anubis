"""Unit tests for the confirmed write behind every learned identity fact.

``store.aput`` returning cleanly is not proof a row landed: the store a graph node is handed
by ``langgraph-api`` queues writes and flushes them on a background task, and a put that never
flushes leaves no row while the tool goes on to reply "Learned". The avatar then keeps the fact
in ``assistant_identity_documents`` graph state — answering from it for the rest of the thread —
while the owner's avatar-settings screen, which reads the store, never shows it.
``_put_fact_document_and_confirm`` reads the key back so that loss becomes a retry and then an
honest refusal.
"""

import pytest

from src.anubis.utils.tools.identity.identity_tools import (
    _put_fact_document_and_confirm,
)

_NAMESPACE = ("creator-1", "assistant-1", "identity_memory")
_KEY = "fact-1"
_VALUE = {"document": {"kwargs": {"page_content": "<FACT>I have twins.</FACT>"}}}


class _FakeStore:
    """A store whose puts land only on the attempts listed in ``lands_on_attempt``."""

    def __init__(self, lands_on_attempt: set[int] | None = None, raises: bool = False):
        self._lands_on_attempt = lands_on_attempt or set()
        self._raises = raises
        self._rows: dict[tuple, dict] = {}
        self.put_calls = 0

    async def aput(self, namespace, key, value):
        self.put_calls += 1
        if self._raises:
            raise RuntimeError("store unavailable")
        if self.put_calls in self._lands_on_attempt:
            self._rows[(tuple(namespace), key)] = value

    async def aget(self, namespace, key):
        return self._rows.get((tuple(namespace), key))


@pytest.mark.asyncio
async def test_confirms_a_write_that_landed():
    store = _FakeStore(lands_on_attempt={1})
    assert await _put_fact_document_and_confirm(store, _NAMESPACE, _KEY, _VALUE) is True
    assert store.put_calls == 1


@pytest.mark.asyncio
async def test_retries_once_when_the_first_put_does_not_land():
    """The exact observed failure: aput returns cleanly, the row is not there."""
    store = _FakeStore(lands_on_attempt={2})
    assert await _put_fact_document_and_confirm(store, _NAMESPACE, _KEY, _VALUE) is True
    assert store.put_calls == 2


@pytest.mark.asyncio
async def test_reports_failure_when_the_row_never_lands():
    """Two silent losses -> the caller is told, rather than replying "Learned" to the user."""
    store = _FakeStore(lands_on_attempt=set())
    assert await _put_fact_document_and_confirm(store, _NAMESPACE, _KEY, _VALUE) is False
    assert store.put_calls == 2


@pytest.mark.asyncio
async def test_reports_failure_when_the_store_raises():
    store = _FakeStore(raises=True)
    assert await _put_fact_document_and_confirm(store, _NAMESPACE, _KEY, _VALUE) is False
    assert store.put_calls == 2
