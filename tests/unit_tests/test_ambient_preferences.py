"""Owner preferences learned from ambient notification cards.

A dismissal or a note on a card is written to the LangGraph store shaped like
an identity document (so the store's index embeds the text), counted up when
the same kind of scene is decided the same way again, and recalled by
similarity as precedent for the next triage. The route that records a decision
accepts the Agent Inbox ``HumanResponse`` types and nothing else.
"""

import json
from types import SimpleNamespace

import pytest

from src.anubis.utils.ambient.preferences import (
    ambient_preference_namespace,
    recall_ambient_preferences,
    record_ambient_preference,
)
from src.api import webapp as webapp_module


class _FakeStore:
    def __init__(self):
        self.items = {}
        self.searches = []

    async def aget(self, namespace, key):
        value = self.items.get((namespace, key))
        return None if value is None else SimpleNamespace(value=value)

    async def aput(self, namespace, key, value):
        self.items[(namespace, key)] = value

    async def asearch(self, namespace, query=None, limit=10):
        self.searches.append((namespace, query, limit))
        return [
            SimpleNamespace(value=value)
            for (item_namespace, _key), value in self.items.items()
            if item_namespace == namespace
        ][:limit]


@pytest.mark.asyncio
async def test_a_decision_is_stored_as_an_embeddable_document_and_counted():
    store = _FakeStore()
    first = await record_ambient_preference(
        store,
        "u1",
        "a1",
        observation_kind="Error Dialog",
        summary="An error dialog is open.",
        decision="ignore",
        note="never tell me about terminal errors",
    )
    assert first["count"] == 1
    assert first["observation_kind"] == "error dialog"
    document = first["document"]
    assert document["kwargs"]["page_content"].startswith(
        "error dialog: An error dialog is open. -> ignore."
    )
    assert "never tell me about terminal errors" in document["kwargs"]["page_content"]
    second = await record_ambient_preference(
        store,
        "u1",
        "a1",
        observation_kind="error dialog",
        summary="Another dialog.",
        decision="ignore",
        note=None,
    )
    assert second["count"] == 2
    # The earlier note survives a later decision without one.
    assert second["note"] == "never tell me about terminal errors"
    assert len(store.items) == 1
    assert list(store.items)[0][0] == ambient_preference_namespace("u1", "a1")


@pytest.mark.asyncio
async def test_recall_returns_one_entry_per_kind_and_decision():
    store = _FakeStore()
    await record_ambient_preference(
        store, "u1", "a1", observation_kind="video_call", summary="s", decision="ignore"
    )
    await record_ambient_preference(
        store,
        "u1",
        "a1",
        observation_kind="video_call",
        summary="s",
        decision="respond",
    )
    recalled = await recall_ambient_preferences(
        store, "u1", "a1", query="a video call", limit=5
    )
    assert {(item["observation_kind"], item["decision"]) for item in recalled} == {
        ("video_call", "ignore"),
        ("video_call", "respond"),
    }
    assert store.searches[0][1] == "a video call"
    assert await recall_ambient_preferences(None, "u1", "a1", query="x") == []
    assert (
        await record_ambient_preference(
            None, "u1", "a1", observation_kind="k", summary="s", decision="ignore"
        )
        is None
    )


@pytest.mark.asyncio
async def test_the_route_records_the_agent_inbox_decision(monkeypatch):
    store = _FakeStore()
    monkeypatch.setattr(webapp_module.app.state, "store", store, raising=False)

    class _Request:
        def __init__(self, body):
            self._body = body

        async def json(self):
            return self._body

    current_user = {"API_KEY": "k", "identities": [{"user_id": "u1"}]}
    response = await webapp_module.record_ambient_preference_route(
        assistant_id="a1",
        request=_Request(
            {
                "observation_id": "obs-1",
                "observation_kind": "error_dialog",
                "summary": "An error dialog is open.",
                "type": "response",
                "args": "only tell me about build failures",
            }
        ),
        current_user=current_user,
    )
    payload = json.loads(response.body)
    assert payload["recorded"] is True
    assert payload["observation_id"] == "obs-1"
    assert payload["preference"]["note"] == "only tell me about build failures"
    assert payload["preference"]["decision"] == "response"

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as error:
        await webapp_module.record_ambient_preference_route(
            assistant_id="a1",
            request=_Request({"type": "edit"}),
            current_user=current_user,
        )
    assert error.value.status_code == 400
