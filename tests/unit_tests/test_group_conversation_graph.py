"""The avatar taking part in a room, end to end through the graph.

Pinned down here:

- **A direct mention is answered**, without triage and without waiting.
- **Ordinary chatter is ignored** and costs nothing further.
- **A reply the avatar proposes on its own waits for the owner** until the
  owner's decisions in that room support sending one.
- **A ban never happens on confidence alone.** Even at confidence 1.0, with the
  action available and the owner administering the room, a timeout or a ban
  goes to the owner until the owner has allowed that same action in that same
  room before. This is the property the whole feature has to get right.
- **A mild moderation action does act** once the score clears the bar, so the
  gate is specific to what cannot be undone rather than blanket timidity.
- **The owner's decision teaches three ways at once** — this person here, this
  room, this kind of message anywhere.
- **A resent message is answered from the row**, not decided a second time.
"""

from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.anubis.utils.groups import precedent, runner
from src.anubis.utils.groups.events import GroupEvent, GroupEventsRequest
from src.anubis.utils.groups.triage import GroupTriageClassification
from src.anubis.utils.inbox import repository as inbox_repository
from src.anubis.utils.inbox.repository import (
    STATE_IGNORED,
    STATE_PENDING_OWNER,
    InMemoryInboxRepository,
)
from src.subgraphs.group_conversation import graph as group_graph_module

USER_ID = "auth0-owner"
ASSISTANT_ID = "assistant-personal"
PLATFORM = "twitch"
CHANNEL_ID = "c-1"
DRAFTED = "Good question — the answer is yes."


class _Store:
    """An in-memory double: namespace → key → value, with substring search."""

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
        group_precedent_recall_limit=8,
        group_recent_events_for_triage=12,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _event(text="is the stream up tomorrow?", mentioned=False, event_id="e1", author_id="viewer-1"):
    return GroupEvent(
        event_id=event_id,
        author_id=author_id,
        author_name="Dana",
        text=text,
        mentioned=mentioned,
    )


class _FakeReasoning:
    """Stands in for the classifier and the reply writer."""

    def __init__(self, classification: GroupTriageClassification | None = None, *, alignment=0.5):
        self.classification = classification
        self.alignment = alignment
        self.classified: list[str] = []
        self.drafted: list[str] = []

    def install(self, monkeypatch):
        from src.anubis.utils.groups import triage as group_triage
        from src.anubis.utils.inbox import triage as inbox_triage

        async def classify_group_event(context, *, event, **kwargs):
            self.classified.append(event.text)
            return self.classification or GroupTriageClassification(
                decision="ignore", reason="chatter", confidence=0.9
            )

        async def _voice_prompt(state, config, runtime):
            return "You are Evan."

        async def judge_alignment(context, *, message, draft, preferences):
            return inbox_triage.PreferenceAlignment(
                aligned=self.alignment >= 0.7,
                alignment_score=self.alignment,
                reason="judged",
            )

        class _Model:
            def __init__(self, drafted):
                self.drafted = drafted

            async def ainvoke(self, input=None, **kwargs):
                self.drafted.append(str(input[-1].content))
                return SimpleNamespace(body=DRAFTED, summary="answers the question")

        from src.anubis.utils import model as model_module

        monkeypatch.setattr(group_triage, "classify_group_event", classify_group_event)
        monkeypatch.setattr(group_graph_module, "_voice_system_prompt", _voice_prompt)
        monkeypatch.setattr(inbox_triage, "judge_alignment", judge_alignment)
        monkeypatch.setattr(model_module, "init_model", lambda **kwargs: _Model(self.drafted))
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


async def _decide(event, *, available_actions=None, owns_channel=True, recent=None, context=None):
    return await runner.run_group_conversation_for_event(
        context or _context(),
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        assistant={"name": "Evan", "metadata": {"user_id": USER_ID}},
        platform=PLATFORM,
        channel_id=CHANNEL_ID,
        channel_name="the stream",
        owns_channel=owns_channel,
        available_actions=available_actions if available_actions is not None else ["warn", "delete", "timeout", "ban"],
        event=event,
        recent_events=recent or [],
    )


# --------------------------------------------------------------------------
# Taking part
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_direct_mention_is_always_answered(harness, monkeypatch):
    """Speaking to the avatar always gets an answer — the classifier picks where.

    A mention used to skip classification entirely, which meant the avatar
    answered a private question in the channel because nothing ever considered
    doing otherwise. It is classified now; what is guaranteed is that a mention
    can never come back as silence.
    """
    reasoning = _FakeReasoning(
        GroupTriageClassification(decision="ignore", reason="chatter", confidence=0.9)
    ).install(monkeypatch)
    decision = await _decide(_event(mentioned=True))
    assert reasoning.classified == ["is the stream up tomorrow?"]
    # The classifier said ignore; a mention is answered anyway.
    assert decision.action == "respond"
    assert decision.reply == DRAFTED


@pytest.mark.asyncio
async def test_ordinary_chatter_is_ignored(harness, monkeypatch):
    _FakeReasoning(
        GroupTriageClassification(decision="ignore", reason="chatter", confidence=0.9)
    ).install(monkeypatch)
    decision = await _decide(_event("lol"))
    assert decision.action == "ignore"
    assert decision.reply is None
    item = (await harness.inbox.list_items(assistant_id=ASSISTANT_ID))[0]
    assert item["state"] == STATE_IGNORED


@pytest.mark.asyncio
async def test_an_unprompted_reply_waits_for_the_owner_until_precedent_exists(
    harness, monkeypatch
):
    _FakeReasoning(
        GroupTriageClassification(
            decision="respond", reason="a question the owner would answer", confidence=0.95
        ),
        alignment=1.0,
    ).install(monkeypatch)
    decision = await _decide(_event())
    # With no decisions recorded for this room, the prior caps the score below
    # the threshold, so nothing is posted unasked.
    assert decision.action == "notify"
    assert decision.reply is None, "nothing is said in the room while the owner decides"
    assert decision.item_id
    item = await harness.inbox.get_item(decision.item_id)
    assert item["state"] == STATE_PENDING_OWNER


# --------------------------------------------------------------------------
# Moderation
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_ban_never_happens_on_confidence_alone(harness, monkeypatch):
    """The property the whole feature has to get right."""
    _FakeReasoning(
        GroupTriageClassification(
            decision="moderate",
            moderation_action="ban",
            reason="threats",
            confidence=1.0,
        )
    ).install(monkeypatch)

    # Make the score itself unimpeachable: a long history of the owner accepting
    # what the avatar does in this room.
    for index in range(12):
        await harness.inbox.record_preference(
            user_id=USER_ID,
            assistant_id=ASSISTANT_ID,
            sender=f"{PLATFORM}:{CHANNEL_ID}:viewer-1",
            sender_domain=f"{PLATFORM}:{CHANNEL_ID}",
            message_kind="harassment",
            decision="accept",
            edit_summary=None,
            example_subject=f"earlier {index}",
        )

    decision = await _decide(_event("i will find you"))
    assert decision.action == "notify", "a ban without the owner's precedent reaches the owner"
    assert decision.moderation_action == "none"


@pytest.mark.asyncio
async def test_a_ban_acts_once_the_owner_has_allowed_one_in_that_room(harness, monkeypatch):
    _FakeReasoning(
        GroupTriageClassification(
            decision="moderate", moderation_action="ban", reason="threats", confidence=1.0
        )
    ).install(monkeypatch)
    for index in range(12):
        await harness.inbox.record_preference(
            user_id=USER_ID,
            assistant_id=ASSISTANT_ID,
            sender=f"{PLATFORM}:{CHANNEL_ID}:viewer-1",
            sender_domain=f"{PLATFORM}:{CHANNEL_ID}",
            message_kind="harassment",
            decision="accept",
            edit_summary=None,
            example_subject=f"earlier {index}",
        )
    # The owner banned somebody here before.
    await precedent.store_decision_record(
        harness.store,
        USER_ID,
        ASSISTANT_ID,
        platform=PLATFORM,
        channel_id=CHANNEL_ID,
        channel_name="the stream",
        event=_event("earlier abuse", event_id="e0"),
        action="moderate",
        moderation_action="ban",
        reasoning="abuse",
        confidence=0.99,
    )
    await precedent.mark_decision_owner_approved(
        harness.store,
        USER_ID,
        ASSISTANT_ID,
        platform=PLATFORM,
        channel_id=CHANNEL_ID,
        event_id="e0",
        approved_action="moderate",
        approved_moderation_action="ban",
    )

    decision = await _decide(_event("i will find you"))
    assert decision.action == "moderate"
    assert decision.moderation_action == "ban"


@pytest.mark.asyncio
async def test_a_mild_action_is_not_held_back_by_the_precedent_gate(harness, monkeypatch):
    """The gate is specific to what cannot be undone, not blanket timidity."""
    _FakeReasoning(
        GroupTriageClassification(
            decision="moderate", moderation_action="delete", reason="a spam link", confidence=1.0
        )
    ).install(monkeypatch)
    for index in range(12):
        await harness.inbox.record_preference(
            user_id=USER_ID,
            assistant_id=ASSISTANT_ID,
            sender=f"{PLATFORM}:{CHANNEL_ID}:viewer-1",
            sender_domain=f"{PLATFORM}:{CHANNEL_ID}",
            message_kind="spam_link",
            decision="accept",
            edit_summary=None,
            example_subject=f"earlier {index}",
        )
    decision = await _decide(_event("buy followers here"))
    assert decision.action == "moderate"
    assert decision.moderation_action == "delete"


# --------------------------------------------------------------------------
# The owner's decision, and what it teaches
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_owner_decision_teaches_the_person_the_room_and_the_kind(
    harness, monkeypatch
):
    _FakeReasoning(
        GroupTriageClassification(
            decision="respond",
            message_kind="technical_question",
            reason="a question",
            confidence=0.5,
        )
    ).install(monkeypatch)
    decision = await _decide(_event())
    assert decision.action == "notify"

    resolved = await runner.resume_group_item(
        _context(),
        item_id=decision.item_id,
        human_response={"type": "accept", "args": None},
    )
    assert resolved["state"] in ("sent", "resolved")

    # This person in this room.
    for_this_person = await harness.inbox.recall_preferences(
        assistant_id=ASSISTANT_ID,
        sender=f"{PLATFORM}:{CHANNEL_ID}:viewer-1",
        sender_domain=f"{PLATFORM}:{CHANNEL_ID}",
        message_kind="technical_question",
    )
    assert any(
        row.get("sender") == f"{PLATFORM}:{CHANNEL_ID}:viewer-1" for row in for_this_person
    )
    # Anyone in this room. The row is keyed by the room itself rather than by an
    # empty sender: the unique key spans (assistant_id, sender, message_kind,
    # decision), so an empty sender here would collide with the kind-only row
    # below and only one of the two would ever be written.
    assert any(
        row.get("sender") == f"{PLATFORM}:{CHANNEL_ID}"
        and row.get("sender_domain") == f"{PLATFORM}:{CHANNEL_ID}"
        for row in for_this_person
    )
    # This kind of message anywhere: a stranger in another room reaches it.
    for_a_stranger = await harness.inbox.recall_preferences(
        assistant_id=ASSISTANT_ID,
        sender="discord:other:viewer-9",
        sender_domain="discord:other",
        message_kind="technical_question",
    )
    assert any(
        not row.get("sender") and not row.get("sender_domain") for row in for_a_stranger
    )


@pytest.mark.asyncio
async def test_the_owner_can_moderate_instead_of_replying(harness, monkeypatch):
    _FakeReasoning(
        GroupTriageClassification(
            decision="respond", message_kind="rudeness", reason="a remark", confidence=0.5
        )
    ).install(monkeypatch)
    decision = await _decide(_event("you are useless"))
    assert decision.action == "notify"

    resolved = await runner.resume_group_item(
        _context(),
        item_id=decision.item_id,
        human_response={
            "type": "edit",
            "args": {
                "action": "moderate",
                "args": {"moderation_action": "timeout"},
                "note": "I time out personal insults.",
            },
        },
    )
    assert resolved["state"] == "sent"
    detail = resolved["confidence_detail"] or {}
    assert detail.get("action") == "moderate"
    assert detail.get("moderation_action") == "timeout"

    # The owner's own words became a rule, and the owner allowing a timeout here
    # is the precedent the next one needs.
    rules = await precedent.list_policy_rules(harness.store, USER_ID, ASSISTANT_ID)
    assert any("time out personal insults" in (rule["rule"] or "") for rule in rules)
    assert await precedent.has_moderation_precedent(
        harness.store,
        USER_ID,
        ASSISTANT_ID,
        platform=PLATFORM,
        channel_id=CHANNEL_ID,
        moderation_action="timeout",
    )


# --------------------------------------------------------------------------
# The batch
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_resent_message_is_answered_without_being_decided_twice(
    harness, monkeypatch
):
    reasoning = _FakeReasoning(
        GroupTriageClassification(decision="ignore", reason="chatter", confidence=0.9)
    ).install(monkeypatch)
    first = await _decide(_event("lol"))
    again = await _decide(_event("lol"))
    assert len(reasoning.classified) == 1, "a retried batch must not run the triage twice"
    assert again.action == first.action
    assert len(await harness.inbox.list_items(assistant_id=ASSISTANT_ID)) == 1


@pytest.mark.asyncio
async def test_every_message_in_a_batch_gets_exactly_one_decision(harness, monkeypatch):
    _FakeReasoning(
        GroupTriageClassification(decision="ignore", reason="chatter", confidence=0.9)
    ).install(monkeypatch)
    request = GroupEventsRequest(
        platform=PLATFORM,
        channel_id=CHANNEL_ID,
        channel_name="the stream",
        owns_channel=True,
        available_actions=["warn"],
        events=[_event(f"message {index}", event_id=f"e{index}") for index in range(5)],
    )
    decisions = await runner.triage_group_events(
        _context(),
        request,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        assistant={"name": "Evan", "metadata": {"user_id": USER_ID}},
        concurrency=3,
    )
    assert len(decisions) == 5
    assert [decision.event_id for decision in decisions] == [f"e{index}" for index in range(5)]


@pytest.mark.asyncio
async def test_one_failed_message_never_fails_the_batch(harness, monkeypatch):
    from src.anubis.utils.groups import triage as group_triage

    _FakeReasoning().install(monkeypatch)

    async def _explode(context, *, event, **kwargs):
        if event.event_id == "e1":
            raise RuntimeError("this one is broken")
        return GroupTriageClassification(decision="ignore", reason="chatter", confidence=0.9)

    monkeypatch.setattr(group_triage, "classify_group_event", _explode)
    request = GroupEventsRequest(
        platform=PLATFORM,
        channel_id=CHANNEL_ID,
        owns_channel=True,
        events=[_event("a", event_id="e0"), _event("b", event_id="e1"), _event("c", event_id="e2")],
    )
    decisions = await runner.triage_group_events(
        _context(),
        request,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        assistant={"name": "Evan", "metadata": {"user_id": USER_ID}},
    )
    assert len(decisions) == 3, "the bot needs one decision back for every message sent"
    broken = next(decision for decision in decisions if decision.event_id == "e1")
    assert broken.action == "ignore"
    assert "broken" in broken.reasoning


@pytest.mark.asyncio
async def test_the_viewer_identity_is_not_the_owner_identity():
    """A stranger's words must never move the owner's own records."""
    viewer = runner.viewer_user_id("twitch", "viewer-1")
    assert viewer != USER_ID
    assert viewer == runner.viewer_user_id("twitch", "viewer-1"), "stable per person"
    assert viewer != runner.viewer_user_id("discord", "viewer-1"), "scoped per platform"
