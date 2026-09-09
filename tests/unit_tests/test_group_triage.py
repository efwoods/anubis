"""Deciding one message in a room, and the two guarantees around that decision.

The classifier is told which moderation actions the bot can carry out and
whether the owner administers the room, but being told is guidance. These are
enforced after the model answers:

- **A moderation action the platform did not offer never comes back.** A bot
  that received an action it cannot carry out would silently drop the action,
  which reads to the owner as the avatar having handled a message when nothing
  was done. Such a decision becomes ``notify`` instead.
- **A room the owner does not administer offers no moderation at all**,
  whatever permissions the bot happens to hold there.

A model failure is ``notify``: a message the avatar cannot judge goes to the
owner rather than being answered or acted on.
"""

from types import SimpleNamespace

import pytest

from src.anubis.utils.groups import triage
from src.anubis.utils.groups.events import GroupEvent


def _event(text="you are all idiots", mentioned=False):
    return GroupEvent(
        event_id="e1",
        author_id="viewer-1",
        author_name="Dana",
        text=text,
        mentioned=mentioned,
    )


class _FakeModel:
    def __init__(self, answer, *, error=None):
        self.answer = answer
        self.error = error
        self.system_prompts: list[str] = []

    async def ainvoke(self, input=None, **kwargs):
        self.system_prompts.append(str(input[0].content))
        if self.error is not None:
            raise self.error
        return self.answer


def _install(monkeypatch, answer, *, error=None):
    model = _FakeModel(answer, error=error)
    from src.anubis.utils import model as model_module

    monkeypatch.setattr(model_module, "init_model", lambda **kwargs: model)
    return model


async def _classify(monkeypatch, answer, *, available_actions, owns_channel, error=None):
    model = _install(monkeypatch, answer, error=error)
    classification = await triage.classify_group_event(
        SimpleNamespace(),
        event=_event(),
        platform="twitch",
        channel_name="the stream",
        owner_name="Evan",
        recent_events=[],
        policy_rules=[{"rule": "Ban nobody without asking me."}],
        past_decisions=[],
        available_actions=available_actions,
        owns_channel=owns_channel,
    )
    return classification, model


@pytest.mark.asyncio
async def test_a_moderation_action_the_platform_cannot_carry_out_goes_to_the_owner(
    monkeypatch,
):
    answer = SimpleNamespace(
        decision="moderate",
        moderation_action="ban",
        needs_owner_action=False,
        message_kind="harassment",
        summary="abuse",
        salience=0.9,
        confidence=0.99,
        applied_rule="",
        reason="This is abusive.",
    )
    classification, _ = await _classify(
        monkeypatch, answer, available_actions=["warn", "delete"], owns_channel=True
    )
    assert classification.decision == "notify"
    assert classification.moderation_action == "none"
    assert classification.needs_owner_action is True
    assert "not available here" in classification.reason


@pytest.mark.asyncio
async def test_a_room_the_owner_does_not_administer_offers_no_moderation(monkeypatch):
    answer = SimpleNamespace(
        decision="moderate",
        moderation_action="delete",
        needs_owner_action=False,
        message_kind="spam_link",
        summary="a link",
        salience=0.5,
        confidence=0.95,
        applied_rule="",
        reason="Spam.",
    )
    classification, model = await _classify(
        monkeypatch,
        answer,
        available_actions=["warn", "delete", "timeout", "ban"],
        owns_channel=False,
    )
    assert classification.decision == "notify"
    assert classification.moderation_action == "none"
    # The prompt itself says moderation is unavailable, so the model is not
    # invited to propose an action that would then be discarded.
    assert "moderation is not available here" in model.system_prompts[0]


@pytest.mark.asyncio
async def test_an_available_action_is_returned_unchanged(monkeypatch):
    answer = SimpleNamespace(
        decision="moderate",
        moderation_action="delete",
        needs_owner_action=False,
        message_kind="spam_link",
        summary="a link",
        salience=0.5,
        confidence=0.93,
        applied_rule="Never post links in my channel.",
        reason="The rule forbids links.",
    )
    classification, _ = await _classify(
        monkeypatch, answer, available_actions=["warn", "delete"], owns_channel=True
    )
    assert classification.decision == "moderate"
    assert classification.moderation_action == "delete"
    assert classification.applied_rule == "Never post links in my channel."


@pytest.mark.asyncio
async def test_moderate_with_no_action_named_defaults_to_the_mildest(monkeypatch):
    answer = SimpleNamespace(
        decision="moderate",
        moderation_action="none",
        needs_owner_action=False,
        message_kind="rudeness",
        summary="rude",
        salience=0.4,
        confidence=0.8,
        applied_rule="",
        reason="Rude.",
    )
    classification, _ = await _classify(
        monkeypatch, answer, available_actions=["warn"], owns_channel=True
    )
    assert classification.decision == "moderate"
    assert classification.moderation_action == "warn"


@pytest.mark.asyncio
async def test_a_moderation_action_on_a_non_moderation_decision_is_dropped(monkeypatch):
    answer = SimpleNamespace(
        decision="respond",
        moderation_action="ban",
        needs_owner_action=False,
        message_kind="technical_question",
        summary="a question",
        salience=0.6,
        confidence=0.7,
        applied_rule="",
        reason="A question.",
    )
    classification, _ = await _classify(
        monkeypatch, answer, available_actions=["warn", "ban"], owns_channel=True
    )
    assert classification.decision == "respond"
    assert classification.moderation_action == "none"


@pytest.mark.asyncio
async def test_a_failed_classification_reaches_the_owner(monkeypatch):
    classification, _ = await _classify(
        monkeypatch,
        None,
        available_actions=["warn"],
        owns_channel=True,
        error=RuntimeError("the model is down"),
    )
    assert classification.decision == "notify"
    assert classification.confidence == 0.0
    assert "the model is down" in classification.reason


@pytest.mark.asyncio
async def test_the_owner_rules_and_the_room_reach_the_prompt(monkeypatch):
    answer = SimpleNamespace(
        decision="ignore",
        moderation_action="none",
        needs_owner_action=False,
        message_kind="greeting",
        summary="hello",
        salience=0.1,
        confidence=0.9,
        applied_rule="",
        reason="Chatter.",
    )
    model = _install(monkeypatch, answer)
    await triage.classify_group_event(
        SimpleNamespace(),
        event=_event("hi everyone"),
        platform="discord",
        channel_name="general",
        owner_name="Evan",
        recent_events=[
            GroupEvent(event_id="e0", author_id="viewer-2", author_name="Sam", text="anyone here?")
        ],
        policy_rules=[{"rule": "Ban nobody without asking me."}],
        past_decisions=[{"page_content": "EVENT: hi\nDECISION: ignore — chatter"}],
        available_actions=["warn"],
        owns_channel=True,
    )
    prompt = model.system_prompts[0]
    assert "Ban nobody without asking me." in prompt
    assert "anyone here?" in prompt, "the room is context for the decision"
    assert "DECISION: ignore" in prompt, "past decisions are precedent"
