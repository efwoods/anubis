"""Ratings and notes on replies persist per user and avatar.

A thumb on a reply is written to the LangGraph store under
``(user_id, assistant_id, "message_feedback")``, comes back attached to the
reply when the transcript is reloaded, and is listed by the avatar
preferences route the browser reads after every press. Before this, the
browser posted to a route that did not exist and swallowed the 404, so a
rating lived only until the page was refreshed.
"""

import json
from types import SimpleNamespace

import pytest

from src.anubis.utils.ambient.preferences import (
    ambient_decision_namespace,
    list_ambient_decisions,
    record_ambient_decision,
)
from src.anubis.utils.message_feedback import (
    attach_message_feedback,
    list_message_feedback,
    message_feedback_namespace,
    record_message_feedback,
)
from src.api import webapp as webapp_module


class _FakeStore:
    def __init__(self):
        self.items = {}

    async def aget(self, namespace, key):
        value = self.items.get((namespace, key))
        return None if value is None else SimpleNamespace(value=value)

    async def aput(self, namespace, key, value):
        self.items[(namespace, key)] = value

    async def asearch(self, namespace, query=None, filter=None, limit=10):
        matches = []
        for (item_namespace, _key), value in self.items.items():
            if item_namespace != namespace:
                continue
            if filter and any(value.get(k) != v for k, v in filter.items()):
                continue
            matches.append(SimpleNamespace(value=value))
        return matches[:limit]


@pytest.mark.asyncio
async def test_a_thumb_is_stored_under_the_user_and_avatar_and_keeps_its_note():
    store = _FakeStore()
    liked = await record_message_feedback(
        store,
        "u1",
        "a1",
        thread_id="t1",
        message_id="lc_run--1",
        request_id="req-1",
        feedback_type="like",
        comment=None,
        content="Here is the recipe you asked for.",
    )
    assert liked["feedback_type"] == "like"
    assert liked["document"]["kwargs"]["page_content"].startswith(
        'The conversation partner liked this reply: "Here is the recipe'
    )
    assert (message_feedback_namespace("u1", "a1"), "lc_run--1") in store.items

    noted = await record_message_feedback(
        store,
        "u1",
        "a1",
        thread_id="t1",
        message_id="lc_run--1",
        request_id="req-1",
        feedback_type="like",
        comment="more of this",
    )
    assert noted["comment"] == "more of this"
    assert noted["content_excerpt"] == "Here is the recipe you asked for."

    # A later thumb without a note keeps the note.
    disliked = await record_message_feedback(
        store,
        "u1",
        "a1",
        thread_id="t1",
        message_id="lc_run--1",
        request_id="req-1",
        feedback_type="dislike",
    )
    assert disliked["comment"] == "more of this"
    assert disliked["feedback_type"] == "dislike"


@pytest.mark.asyncio
async def test_feedback_is_listed_per_thread_and_attached_to_the_reply():
    store = _FakeStore()
    await record_message_feedback(
        store,
        "u1",
        "a1",
        thread_id="t1",
        message_id="lc_run--1",
        request_id="req-1",
        feedback_type="like",
    )
    await record_message_feedback(
        store,
        "u1",
        "a1",
        thread_id="t2",
        message_id=None,
        request_id="req-2",
        feedback_type="dislike",
        comment="too long",
    )
    assert len(await list_message_feedback(store, "u1", "a1")) == 2
    only_thread_one = await list_message_feedback(store, "u1", "a1", thread_id="t1")
    assert [record["message_id"] for record in only_thread_one] == ["lc_run--1"]

    messages = [
        {"type": "human", "id": "h1", "content": "hi"},
        {"type": "ai", "id": "lc_run--1", "content": "hello"},
        {
            "type": "ai",
            "id": "lc_run--2",
            "content": "a long reply",
            "response_metadata": {"request_id": "req-2"},
        },
        "not a dict",
    ]
    attached = attach_message_feedback(
        messages, await list_message_feedback(store, "u1", "a1")
    )
    assert "feedback" not in attached[0]
    assert attached[1]["feedback"]["type"] == "like"
    assert attached[2]["feedback"] == {
        "type": "dislike",
        "comment": "too long",
        "recorded_at": attached[2]["feedback"]["recorded_at"],
    }
    assert attached[3] == "not a dict"


@pytest.mark.asyncio
async def test_a_card_keeps_its_thumb_and_note_separately():
    store = _FakeStore()
    rated = await record_ambient_decision(
        store,
        "u1",
        "a1",
        observation_id="obs-1",
        observation_kind="scene",
        summary="A pan is smoking.",
        decision="ignore",
    )
    assert rated["rating"] == "ignore"
    noted = await record_ambient_decision(
        store,
        "u1",
        "a1",
        observation_id="obs-1",
        observation_kind="scene",
        summary="A pan is smoking.",
        decision="response",
        note="only fires",
    )
    assert noted["rating"] == "ignore"
    assert noted["note"] == "only fires"
    accepted = await record_ambient_decision(
        store,
        "u1",
        "a1",
        observation_id="obs-1",
        observation_kind="scene",
        summary="A pan is smoking.",
        decision="accept",
    )
    assert accepted["rating"] == "accept"
    assert accepted["note"] == "only fires"
    assert (ambient_decision_namespace("u1", "a1"), "obs-1") in store.items
    # A card without an id cannot be remembered.
    assert (
        await record_ambient_decision(
            store,
            "u1",
            "a1",
            observation_id=None,
            observation_kind="scene",
            summary="",
            decision="accept",
        )
        is None
    )
    listed = await list_ambient_decisions(store, "u1", "a1")
    assert listed == [
        {
            "observation_id": "obs-1",
            "observation_kind": "scene",
            "rating": "accept",
            "note": "only fires",
            "action_taken": None,
            "rated_after_action": None,
            "left_alone": False,
            "decided_at": listed[0]["decided_at"],
        }
    ]


@pytest.mark.asyncio
async def test_the_routes_record_and_return_what_the_browser_shows(monkeypatch):
    store = _FakeStore()
    monkeypatch.setattr(webapp_module.app.state, "store", store, raising=False)
    current_user = {"API_KEY": "k", "identities": [{"user_id": "u1"}]}

    recorded = await webapp_module.record_message_feedback_route(
        feedback=webapp_module.MessageFeedbackRequest(
            assistant_id="a1",
            thread_id="t1",
            message_id="lc_run--1",
            request_id="req-1",
            feedback_type="LIKE",
            comment="  keep this tone ",
            content="hello there",
        ),
        current_user=current_user,
    )
    payload = json.loads(recorded.body)
    assert payload["recorded"] is True
    assert payload["feedback"]["type"] == "like"
    assert payload["feedback"]["comment"] == "keep this tone"

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as rejected:
        await webapp_module.record_message_feedback_route(
            feedback=webapp_module.MessageFeedbackRequest(
                assistant_id="a1", feedback_type="rating", message_id="x"
            ),
            current_user=current_user,
        )
    assert rejected.value.status_code == 400
    with pytest.raises(HTTPException) as unidentified:
        await webapp_module.record_message_feedback_route(
            feedback=webapp_module.MessageFeedbackRequest(
                assistant_id="a1", feedback_type="like"
            ),
            current_user=current_user,
        )
    assert unidentified.value.status_code == 400

    class _Request:
        def __init__(self, body):
            self._body = body

        async def json(self):
            return self._body

    card = await webapp_module.record_ambient_preference_route(
        assistant_id="a1",
        request=_Request(
            {
                "observation_id": "obs-1",
                "observation_kind": "scene",
                "summary": "A pan is smoking.",
                "type": "ignore",
                "args": None,
            }
        ),
        current_user=current_user,
    )
    card_payload = json.loads(card.body)
    assert card_payload["decision"]["rating"] == "ignore"

    preferences = await webapp_module.get_avatar_preferences_route(
        assistant_id="a1", thread_id="t1", current_user=current_user
    )
    preferences_payload = json.loads(preferences.body)
    assert preferences_payload["message_feedback"] == [
        {
            "message_id": "lc_run--1",
            "request_id": "req-1",
            "thread_id": "t1",
            "feedback": {
                "type": "like",
                "comment": "keep this tone",
                "recorded_at": preferences_payload["message_feedback"][0]["feedback"][
                    "recorded_at"
                ],
            },
        }
    ]
    assert preferences_payload["ambient_decisions"][0]["observation_id"] == "obs-1"
    assert preferences_payload["ambient_decisions"][0]["rating"] == "ignore"
