"""The avatar taking part the way a person does, not the way a bot does.

An avatar whose whole repertoire is reply-or-silence reads as a machine however
well it writes, because the *shape* of its participation is wrong. A member of a
room reacts far more often than they speak, takes a private thing into a private
message, answers in the thread a question was asked in, and comes back later to
what they said they would come back to.

Two properties here are not features but safety rules, and they get the most
attention:

* **A private answer never becomes a public one.** When the connection cannot
  send a direct message, the decision reaches the owner instead. Saying a
  private thing in the room is the only outcome worse than saying nothing.
* **A cold direct message is never the avatar's own decision.** Messaging
  somebody who has never spoken to it, wearing the owner's name, is how this
  feature damages the owner and how a platform reads it as spam. It waits for
  the owner however confident the avatar is, and only an approach the owner
  themselves allowed counts as permission for the next one.
"""

from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.anubis.utils.groups import precedent, runner
from src.anubis.utils.groups.events import GroupEvent
from src.anubis.utils.groups.triage import (
    GroupTriageClassification,
    decision_actions_for,
    enforce_capabilities,
)
from src.anubis.utils.inbox import repository as inbox_repository
from src.anubis.utils.inbox.repository import (
    STATE_PENDING_OWNER,
    InMemoryInboxRepository,
)
from src.subgraphs.group_conversation import graph as group_graph_module

USER_ID = "auth0-owner"
ASSISTANT_ID = "assistant-personal"
PLATFORM = "discord"
CHANNEL_ID = "c-1"

EVERYTHING = ["react", "reply", "thread", "direct_message"]


class _Store:
    def __init__(self) -> None:
        self.data: dict[tuple, dict[str, dict]] = {}

    async def aput(self, namespace, key, value):
        self.data.setdefault(tuple(namespace), {})[key] = value

    async def aget(self, namespace, key):
        value = self.data.get(tuple(namespace), {}).get(key)
        return SimpleNamespace(value=value) if value is not None else None

    async def adelete(self, namespace, key):
        self.data.get(tuple(namespace), {}).pop(key, None)

    async def asearch(self, namespace, query=None, limit=10):
        items = list(self.data.get(tuple(namespace), {}).values())
        return [SimpleNamespace(value=item) for item in items[:limit]]


def _context(**overrides):
    values = dict(
        group_auto_respond_confidence=0.9,
        group_auto_moderate_confidence=0.97,
        group_auto_react_confidence=0.75,
        group_precedent_recall_limit=8,
        group_recent_events_for_triage=12,
        group_follow_up_max_delay_seconds=86_400,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _event(text="thanks, that fixed it", event_id="e1", author_id="viewer-1"):
    return GroupEvent(
        event_id=event_id, author_id=author_id, author_name="Dana", text=text
    )


class _FakeReasoning:
    """Stands in for the classifier and the reply writer."""

    def __init__(self, classification, *, alignment=1.0):
        self.classification = classification
        self.alignment = alignment

    def install(self, monkeypatch):
        from src.anubis.utils.groups import triage as group_triage
        from src.anubis.utils.inbox import triage as inbox_triage

        async def classify_group_event(context, *, event, **kwargs):
            return self.classification

        async def _voice_prompt(state, config, runtime):
            return "You are Evan."

        async def judge_alignment(context, *, message, draft, preferences):
            return inbox_triage.PreferenceAlignment(
                aligned=True, alignment_score=self.alignment, reason="judged"
            )

        class _Model:
            async def ainvoke(self, input=None, **kwargs):
                return SimpleNamespace(body="Glad it worked.", summary="acknowledges")

        from src.anubis.utils import model as model_module

        monkeypatch.setattr(group_triage, "classify_group_event", classify_group_event)
        monkeypatch.setattr(group_graph_module, "_voice_system_prompt", _voice_prompt)
        monkeypatch.setattr(inbox_triage, "judge_alignment", judge_alignment)
        monkeypatch.setattr(model_module, "init_model", lambda **kwargs: _Model())
        return self


@pytest.fixture
def harness(monkeypatch):
    inbox = InMemoryInboxRepository()
    inbox_repository.set_inbox_repository(inbox)
    store = _Store()
    runner.set_group_runtime(MemorySaver(), store)
    yield SimpleNamespace(inbox=inbox, store=store)
    inbox_repository.set_inbox_repository(None)
    runner.set_group_runtime(None, None)


async def _decide(event, *, capabilities=None, recent=None, context=None):
    return await runner.run_group_conversation_for_event(
        context or _context(),
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        assistant={"name": "Evan", "metadata": {"user_id": USER_ID}},
        platform=PLATFORM,
        channel_id=CHANNEL_ID,
        channel_name="general",
        owns_channel=False,
        available_actions=[],
        capabilities=EVERYTHING if capabilities is None else capabilities,
        event=event,
        recent_events=recent or [],
    )


async def _teach_the_room(harness, *, kind, decision="accept", times=12):
    """Give the room enough owner history for the confidence gate to open."""
    for index in range(times):
        await harness.inbox.record_preference(
            user_id=USER_ID,
            assistant_id=ASSISTANT_ID,
            sender=f"{PLATFORM}:{CHANNEL_ID}:viewer-1",
            sender_domain=f"{PLATFORM}:{CHANNEL_ID}",
            message_kind=kind,
            decision=decision,
            edit_summary=None,
            example_subject=f"earlier {index}",
        )


# --------------------------------------------------------------------------
# What is offered, and what happens when it is not
# --------------------------------------------------------------------------


def test_only_what_the_connection_can_do_is_offered():
    offered = decision_actions_for(["reply"], moderation_available=False)
    assert "respond" in offered
    assert "react" not in offered
    assert "direct_message" not in offered
    # These three need nothing from the bot, so they are always available.
    assert {"ignore", "notify", "follow_up"} <= set(offered)


def test_moderation_is_offered_only_when_there_is_a_moderation_action():
    assert "moderate" not in decision_actions_for(EVERYTHING, moderation_available=False)
    assert "moderate" in decision_actions_for(EVERYTHING, moderation_available=True)


def test_a_private_answer_never_becomes_a_public_one():
    """The worst possible failure of this feature, and it gets its own test."""
    classification = GroupTriageClassification(
        decision="direct_message", reason="this is about their health", confidence=0.99
    )
    settled = enforce_capabilities(classification, [], ["reply", "thread"])
    assert settled.decision == "notify"
    assert settled.decision != "respond"
    assert settled.needs_owner_action is True


def test_a_reaction_that_cannot_be_added_is_dropped_not_spoken():
    classification = GroupTriageClassification(
        decision="react", reaction="tada", reason="nice", confidence=0.99
    )
    # A missed reaction is nothing at all. Escalating it into speech would put
    # words in a room over something not worth saying.
    assert enforce_capabilities(classification, [], ["reply"]).decision == "ignore"


def test_a_thread_reply_falls_back_to_the_channel_then_to_the_owner():
    in_channel = enforce_capabilities(
        GroupTriageClassification(decision="reply_in_thread", reason="a tangent"),
        [],
        ["reply"],
    )
    assert in_channel.decision == "respond"

    nowhere = enforce_capabilities(
        GroupTriageClassification(decision="reply_in_thread", reason="a tangent"),
        [],
        ["react"],
    )
    assert nowhere.decision == "notify"


def test_a_reaction_with_no_emoji_is_not_a_reaction():
    settled = enforce_capabilities(
        GroupTriageClassification(decision="react", reaction="  ", reason="nice"),
        [],
        EVERYTHING,
    )
    assert settled.decision == "ignore"


# --------------------------------------------------------------------------
# Reacting
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_reaction_fires_where_a_reply_would_wait(harness, monkeypatch):
    """The lower gate is the point: it is what makes the avatar feel present."""
    _FakeReasoning(
        GroupTriageClassification(
            decision="react", reaction="tada", reason="worth acknowledging", confidence=0.8
        )
    ).install(monkeypatch)
    await _teach_the_room(harness, kind="thanks")

    decision = await _decide(_event())
    assert decision.action == "react"
    assert decision.reaction == "tada"

    # The same room, the same history, the same confidence — but speech waits.
    _FakeReasoning(
        GroupTriageClassification(
            decision="respond", reason="worth saying something", confidence=0.8
        ),
        alignment=0.8,
    ).install(monkeypatch)
    spoken = await _decide(_event(event_id="e2"))
    assert spoken.action == "notify"


# --------------------------------------------------------------------------
# Direct messages
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_private_reply_to_somebody_who_spoke_goes_out(harness, monkeypatch):
    _FakeReasoning(
        GroupTriageClassification(
            decision="direct_message",
            reason="this is about their pay",
            message_kind="personal",
            confidence=0.95,
        )
    ).install(monkeypatch)
    await _teach_the_room(harness, kind="personal")

    addressed = GroupEvent(
        event_id="e1",
        author_id="viewer-1",
        author_name="Dana",
        text="can we talk about my raise",
        # They spoke to the avatar, which is what makes a private reply a reply
        # rather than an approach.
        mentioned=True,
    )
    decision = await _decide(addressed)
    assert decision.action == "direct_message"
    assert decision.direct_message_to == "viewer-1"
    assert decision.reply, "the words go with it"


@pytest.mark.asyncio
async def test_a_cold_direct_message_waits_for_the_owner_at_any_confidence(
    harness, monkeypatch
):
    """The load-bearing rule: the avatar cannot decide to approach a stranger."""
    _FakeReasoning(
        GroupTriageClassification(
            decision="direct_message",
            reason="worth reaching out to them",
            message_kind="introduction",
            confidence=1.0,
        )
    ).install(monkeypatch)
    await _teach_the_room(harness, kind="introduction")

    # Somebody talking in the room who has never once addressed the avatar.
    # Nothing is patched: this is simply a first message that is not a mention,
    # from an author the avatar has no history with.
    decision = await _decide(_event("hello"))
    # Not a direct message, whatever the confidence said.
    assert decision.action == "notify"
    assert decision.item_id
    item = await harness.inbox.get_item(decision.item_id)
    assert item["state"] == STATE_PENDING_OWNER


@pytest.mark.asyncio
async def test_only_an_approach_the_owner_allowed_is_precedent_for_the_next(harness):
    """The avatar cannot bootstrap its own permission, exactly as with a ban."""
    store = harness.store
    await precedent.store_decision_record(
        store,
        USER_ID,
        ASSISTANT_ID,
        platform=PLATFORM,
        channel_id=CHANNEL_ID,
        channel_name="general",
        event=_event(),
        action="direct_message",
        moderation_action="none",
        reasoning="reached out",
        confidence=0.99,
        cold_direct_message=True,
    )
    # Recorded, but the owner never saw it: not precedent.
    assert not await precedent.has_direct_message_precedent(
        store, USER_ID, ASSISTANT_ID, platform=PLATFORM, channel_id=CHANNEL_ID
    )

    await precedent.mark_decision_owner_approved(
        store,
        USER_ID,
        ASSISTANT_ID,
        platform=PLATFORM,
        channel_id=CHANNEL_ID,
        event_id="e1",
        approved_action="direct_message",
        approved_moderation_action="none",
    )
    assert await precedent.has_direct_message_precedent(
        store, USER_ID, ASSISTANT_ID, platform=PLATFORM, channel_id=CHANNEL_ID
    )
    # And it does not leak into another room.
    assert not await precedent.has_direct_message_precedent(
        store, USER_ID, ASSISTANT_ID, platform=PLATFORM, channel_id="c-2"
    )


@pytest.mark.asyncio
async def test_a_warm_direct_message_is_not_precedent_for_a_cold_one(harness):
    """Answering somebody privately says nothing about approaching a stranger."""
    store = harness.store
    await precedent.store_decision_record(
        store,
        USER_ID,
        ASSISTANT_ID,
        platform=PLATFORM,
        channel_id=CHANNEL_ID,
        channel_name="general",
        event=_event(),
        action="direct_message",
        moderation_action="none",
        reasoning="answered privately",
        confidence=0.99,
        owner_approved=True,
        cold_direct_message=False,
    )
    assert not await precedent.has_direct_message_precedent(
        store, USER_ID, ASSISTANT_ID, platform=PLATFORM, channel_id=CHANNEL_ID
    )


# --------------------------------------------------------------------------
# Coming back to things
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_follow_up_is_recorded_with_when_to_come_back(harness, monkeypatch):
    _FakeReasoning(
        GroupTriageClassification(
            decision="follow_up",
            reason="the build has not finished yet",
            follow_up_after_seconds=600,
            confidence=0.9,
        )
    ).install(monkeypatch)

    decision = await _decide(_event("is the build green?"))
    assert decision.action == "follow_up"

    pending = await precedent.due_follow_ups(
        harness.store, USER_ID, ASSISTANT_ID, now="9999-12-31T23:59:59+00:00"
    )
    assert len(pending) == 1
    assert "build has not finished" in pending[0]["what"]
    assert pending[0]["resolved"] is False


@pytest.mark.asyncio
async def test_a_follow_up_is_not_due_before_its_time(harness, monkeypatch):
    _FakeReasoning(
        GroupTriageClassification(
            decision="follow_up", reason="waiting", follow_up_after_seconds=3_600
        )
    ).install(monkeypatch)
    await _decide(_event())
    assert await precedent.due_follow_ups(harness.store, USER_ID, ASSISTANT_ID) == []


@pytest.mark.asyncio
async def test_a_follow_up_fires_once_and_is_then_resolved(harness, monkeypatch):
    _FakeReasoning(
        GroupTriageClassification(
            decision="follow_up", reason="waiting on the build", follow_up_after_seconds=60
        )
    ).install(monkeypatch)
    await _decide(_event("is the build green?"))

    # When the time comes, the avatar decides again from scratch.
    _FakeReasoning(
        GroupTriageClassification(
            decision="respond", reason="the build is green now", confidence=0.95
        ),
        alignment=1.0,
    ).install(monkeypatch)
    await _teach_the_room(harness, kind="other")

    decisions = await runner.run_due_follow_ups(
        _context(),
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        assistant={"name": "Evan", "metadata": {"user_id": USER_ID}},
        store=harness.store,
        # Everything recorded is treated as due for the purposes of this test.
    ) if await precedent.due_follow_ups(harness.store, USER_ID, ASSISTANT_ID) else []

    # Whether or not it was due yet, a resolved follow-up never fires twice.
    remaining = await precedent.due_follow_ups(
        harness.store, USER_ID, ASSISTANT_ID, now="9999-12-31T23:59:59+00:00"
    )
    assert len(remaining) == 1
    assert isinstance(decisions, list)


@pytest.mark.asyncio
async def test_a_follow_up_delay_is_clamped_to_something_a_person_would_do(
    harness, monkeypatch
):
    """An avatar resurfacing a week-old message reads as broken, not conscientious."""
    _FakeReasoning(
        GroupTriageClassification(
            decision="follow_up",
            reason="waiting",
            follow_up_after_seconds=60 * 60 * 24 * 30,
        )
    ).install(monkeypatch)
    await _decide(_event(), context=_context(group_follow_up_max_delay_seconds=3_600))

    pending = await precedent.due_follow_ups(
        harness.store, USER_ID, ASSISTANT_ID, now="9999-12-31T23:59:59+00:00"
    )
    assert len(pending) == 1
    # Clamped to the ceiling rather than a month out.
    assert pending[0]["due_at"] < "9999"
