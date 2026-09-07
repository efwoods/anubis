"""A notice can carry an offer, and what the conversation partner does with the
offer is learned.

The triage names what the avatar could do once allowed (``proposed_action``
plus one line of wording); the observation text carries the offer to the
avatar so the heads-up ends by offering it; allowing the offer comes back as
a hidden ``ambient_action`` turn that skips triage and whose reply is stamped
with the observation; and every outcome — allowed, replied in person, left
alone, rated afterwards — is recorded on the card and counted as precedent
per observation kind.
"""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from src.anubis.utils.ambient.observations import (
    NOTIFY_INSTRUCTION,
    NOTIFY_INSTRUCTION_WITH_OFFER,
    OFFER_LINE_PREFIX,
    ambient_details,
    build_ambient_action_additional_kwargs,
    compose_ambient_action_text,
    compose_observation_text,
    is_ambient_action,
    is_ambient_observation,
    normalize_proposed_action,
    proposed_offer,
    split_observation_text,
    strip_instruction,
)
from src.anubis.utils.ambient.preferences import (
    list_ambient_decisions,
    record_ambient_decision,
)
from src.anubis.utils.ambient.triage import (
    describe_ambient_preferences,
    normalize_classification,
)
from src.anubis.utils.ambient.triage_node import route_after_image_resolution
from src.api import webapp as webapp_module

OBSERVATION = {
    "observation_id": "obs-1",
    "sources": ["screen"],
    "captured_at": "t",
    "decision": "notify",
    "proposed_action": "draft",
    "action_description": "Draft a reply to the invoice email",
}


def test_a_notify_with_an_offer_carries_the_offer_to_the_avatar():
    text = compose_observation_text(OBSERVATION, "An invoice email is open.")
    header, body = split_observation_text(text)
    assert "proposed_action=draft" in header
    assert body.startswith(f"{OFFER_LINE_PREFIX} Draft a reply to the invoice email")
    assert text.endswith(NOTIFY_INSTRUCTION_WITH_OFFER)
    # Re-composing (the triage rewrites the turn) does not stack offer lines.
    again = compose_observation_text(OBSERVATION, strip_instruction(body))
    assert again.count(f"\n{OFFER_LINE_PREFIX} ") == 1
    assert strip_instruction(body) == "An invoice email is open."
    plain = compose_observation_text(
        {**OBSERVATION, "proposed_action": "none"}, "An invoice email is open."
    )
    assert plain.endswith(NOTIFY_INSTRUCTION)
    assert OFFER_LINE_PREFIX not in plain
    assert proposed_offer({**OBSERVATION, "decision": "respond"}) is None


def test_an_offer_is_one_verb_on_a_heads_up_and_needs_wording():
    kept = normalize_classification(
        SimpleNamespace(
            decision="notify",
            proposed_action="Draft a reply!",
            action_description=" Draft it ",
        )
    )
    assert (kept.proposed_action, kept.action_description) == ("draft", "Draft it")
    assert normalize_proposed_action("  Research ") == "research"
    assert normalize_proposed_action("123") == "none"
    assert len(normalize_proposed_action("a" * 40)) == 20
    no_wording = normalize_classification(
        SimpleNamespace(decision="notify", proposed_action="reply")
    )
    assert no_wording.proposed_action == "none"
    not_a_notice = normalize_classification(
        SimpleNamespace(
            decision="respond", proposed_action="reply", action_description="Say hi"
        )
    )
    assert (not_a_notice.proposed_action, not_a_notice.action_description) == (
        "none",
        "",
    )
    assert normalize_classification(SimpleNamespace()).proposed_action == "none"


def test_an_allowed_action_is_a_hidden_turn_that_skips_triage():
    kwargs = build_ambient_action_additional_kwargs(
        observation_id="obs-1",
        observation_kind="Email Inbox",
        action="draft",
        action_description="Draft a reply to the invoice email",
        summary="An invoice email is open.",
    )
    turn = HumanMessage(
        id="act-1",
        content=compose_ambient_action_text(kwargs["ambient"]),
        additional_kwargs=kwargs,
    )
    assert kwargs["hidden"] is True
    assert is_ambient_action(turn) and not is_ambient_observation(turn)
    assert ambient_details(turn)["decision"] == "act"
    assert ambient_details(turn)["observation_kind"] == "email inbox"
    assert "[AMBIENT_ACTION id=obs-1 kind=email inbox action=draft]" in turn.content
    assert "Allowed action: Draft a reply to the invoice email" in turn.content
    assert route_after_image_resolution({"messages": [turn]}) == "anubis"


def test_precedent_names_what_the_partner_did_with_earlier_offers():
    rendered = describe_ambient_preferences(
        [
            {
                "observation_kind": "email_inbox",
                "decision": "allowed_action",
                "count": 2,
            },
            {
                "observation_kind": "email_inbox",
                "decision": "disliked_action",
                "count": 1,
            },
            {"observation_kind": "error_dialog", "decision": "left_alone", "count": 3},
        ]
    )
    assert (
        "let the avatar do what the avatar offered ('allowed_action') 2 time(s)"
        in rendered
    )
    assert "disliked what the avatar did after being allowed to act" in rendered
    assert (
        "left the notice alone without choosing anything ('left_alone') 3 time(s)"
        in rendered
    )


class _FakeStore:
    def __init__(self):
        self.items = {}

    async def aget(self, namespace, key):
        value = self.items.get((namespace, key))
        return None if value is None else SimpleNamespace(value=value)

    async def aput(self, namespace, key, value):
        self.items[(namespace, key)] = value

    async def asearch(self, namespace, query=None, filter=None, limit=10):
        return [
            SimpleNamespace(value=value)
            for (item_namespace, _key), value in self.items.items()
            if item_namespace == namespace
        ][:limit]


@pytest.mark.asyncio
async def test_a_card_remembers_the_action_the_outcome_and_being_left_alone():
    store = _FakeStore()
    common = dict(observation_id="obs-1", observation_kind="email_inbox", summary="s")
    alone = await record_ambient_decision(
        store, "u1", "a1", decision="left_alone", **common
    )
    assert alone["left_alone"] is True
    allowed = await record_ambient_decision(
        store, "u1", "a1", decision="act", action_taken="avatar_replied", **common
    )
    assert allowed["action_taken"] == "avatar_replied"
    assert allowed["left_alone"] is False
    rated = await record_ambient_decision(
        store,
        "u1",
        "a1",
        decision="rated_after_action",
        rated_after_action="like",
        **common,
    )
    assert rated["action_taken"] == "avatar_replied"
    assert rated["rated_after_action"] == "like"
    # A later "left alone" cannot undo a card that was acted on.
    later = await record_ambient_decision(
        store, "u1", "a1", decision="left_alone", **common
    )
    assert later["left_alone"] is False
    listed = await list_ambient_decisions(store, "u1", "a1")
    assert listed[0]["action_taken"] == "avatar_replied"
    assert listed[0]["rated_after_action"] == "like"


class _Request:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


@pytest.mark.asyncio
async def test_the_routes_record_allowing_replying_leaving_alone_and_rating(
    monkeypatch,
):
    store = _FakeStore()
    monkeypatch.setattr(webapp_module.app.state, "store", store, raising=False)
    current_user = {"API_KEY": "k", "identities": [{"user_id": "u1"}]}
    card = {
        "observation_id": "obs-1",
        "observation_kind": "email_inbox",
        "summary": "s",
    }

    async def decide(decision_type, args=None, observation_id="obs-1"):
        response = await webapp_module.record_ambient_preference_route(
            assistant_id="a1",
            request=_Request(
                {
                    **card,
                    "observation_id": observation_id,
                    "type": decision_type,
                    "args": args,
                }
            ),
            current_user=current_user,
        )
        return json.loads(response.body)

    allowed = await decide("act", {"action": "avatar"})
    assert allowed["preference"]["decision"] == "allowed_action"
    assert allowed["decision"]["action_taken"] == "avatar_replied"
    replied = await decide("act", "owner", observation_id="obs-2")
    assert replied["preference"]["decision"] == "replied_self"
    assert replied["decision"]["action_taken"] == "owner_replied"
    alone = await decide("left_alone", None, observation_id="obs-3")
    assert alone["preference"]["decision"] == "left_alone"
    assert alone["decision"]["left_alone"] is True

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as bad_actor:
        await decide("act", {"action": "someone"})
    assert bad_actor.value.status_code == 400

    rated = await webapp_module.record_message_feedback_route(
        feedback=webapp_module.MessageFeedbackRequest(
            assistant_id="a1",
            thread_id="t1",
            message_id="lc_run--9",
            feedback_type="dislike",
            comment="too formal",
            observation_id="obs-1",
            observation_kind="email_inbox",
            observation_summary="s",
        ),
        current_user=current_user,
    )
    payload = json.loads(rated.body)
    assert payload["feedback"]["type"] == "dislike"
    assert payload["ambient_decision"]["rated_after_action"] == "dislike"
    assert payload["ambient_decision"]["action_taken"] == "avatar_replied"
    aggregate = store.items[
        (("u1", "a1", "ambient_preference"), "email_inbox:disliked_action")
    ]
    assert aggregate["note"] == "too formal"

    preferences = await webapp_module.get_avatar_preferences_route(
        assistant_id="a1", thread_id=None, current_user=current_user
    )
    decisions = {
        item["observation_id"]: item
        for item in json.loads(preferences.body)["ambient_decisions"]
    }
    assert decisions["obs-1"]["rated_after_action"] == "dislike"
    assert decisions["obs-2"]["action_taken"] == "owner_replied"
    assert decisions["obs-3"]["left_alone"] is True
