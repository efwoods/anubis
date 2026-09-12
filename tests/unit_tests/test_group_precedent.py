"""What the avatar remembers about a room, and the gate on irreversible moderation.

Pinned down here:

- **Rules are stored as embeddable documents.** The store's vector index is
  configured in ``langgraph.json`` to embed ``document.kwargs.page_content``,
  so a rule written any other way is saved but never retrieved by similarity —
  which would read as an owner's rule that silently stopped being applied.
- **The same rule is never stored twice**, however often the owner repeats it.
- **A timeout or a ban needs the owner's own precedent in that same room.** A
  decision the avatar made unilaterally is not precedent for the next one;
  only an action the owner allowed or corrected the avatar into counts.
- **A correction rewrites the decision and becomes a rule**, and counts as the
  owner allowing that action.
"""

from types import SimpleNamespace

import pytest

from src.anubis.utils.groups import precedent
from src.anubis.utils.groups.events import DecisionCorrection, GroupEvent

CREATOR = "auth0-owner"
ASSISTANT = "assistant-personal"


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
        if query:
            items = [
                item
                for item in items
                if str(query).casefold() in str(item).casefold()
            ] or items
        return [SimpleNamespace(value=item) for item in items[:limit]]


def _event(text="hello there", author_id="viewer-1", event_id="e1"):
    return GroupEvent(event_id=event_id, author_id=author_id, author_name="Dana", text=text)


@pytest.mark.asyncio
async def test_a_rule_is_stored_where_the_index_can_embed_it():
    store = _Store()
    document = await precedent.store_policy_rule(
        store, CREATOR, ASSISTANT, rule="Never post links in my channel."
    )
    assert document is not None
    namespace = precedent.group_policy_namespace(CREATOR, ASSISTANT)
    stored = next(iter(store.data[namespace].values()))
    # The index configured in langgraph.json embeds this exact path.
    assert stored["document"]["kwargs"]["page_content"]
    assert "Never post links" in stored["document"]["kwargs"]["page_content"]

    rules = await precedent.list_policy_rules(store, CREATOR, ASSISTANT)
    assert [rule["rule"] for rule in rules] == ["Never post links in my channel."]


@pytest.mark.asyncio
async def test_the_same_rule_is_never_stored_twice():
    store = _Store()
    first = await precedent.store_policy_rule(store, CREATOR, ASSISTANT, rule="No spoilers.")
    again = await precedent.store_policy_rule(store, CREATOR, ASSISTANT, rule="no SPOILERS.")
    assert first is not None
    assert again is None, "the same rule in different case is the same rule"
    assert len(await precedent.list_policy_rules(store, CREATOR, ASSISTANT)) == 1


@pytest.mark.asyncio
async def test_a_decision_the_avatar_made_alone_is_not_precedent_for_a_ban():
    """The load-bearing safety property: the avatar cannot bootstrap its own precedent."""
    store = _Store()
    await precedent.store_decision_record(
        store,
        CREATOR,
        ASSISTANT,
        platform="twitch",
        channel_id="c-1",
        channel_name="the stream",
        event=_event("you are all terrible"),
        action="moderate",
        moderation_action="ban",
        reasoning="abuse",
        confidence=0.99,
    )
    assert not await precedent.has_moderation_precedent(
        store,
        CREATOR,
        ASSISTANT,
        platform="twitch",
        channel_id="c-1",
        moderation_action="ban",
    )

    # The owner allowing that action in that room is what creates precedent.
    await precedent.mark_decision_owner_approved(
        store,
        CREATOR,
        ASSISTANT,
        platform="twitch",
        channel_id="c-1",
        event_id="e1",
        approved_action="moderate",
        approved_moderation_action="ban",
    )
    assert await precedent.has_moderation_precedent(
        store,
        CREATOR,
        ASSISTANT,
        platform="twitch",
        channel_id="c-1",
        moderation_action="ban",
    )


@pytest.mark.asyncio
async def test_precedent_does_not_cross_rooms_actions_or_platforms():
    store = _Store()
    await precedent.store_decision_record(
        store,
        CREATOR,
        ASSISTANT,
        platform="twitch",
        channel_id="c-1",
        channel_name="the stream",
        event=_event(),
        action="moderate",
        moderation_action="timeout",
        reasoning="harassment",
        confidence=0.99,
    )
    await precedent.mark_decision_owner_approved(
        store,
        CREATOR,
        ASSISTANT,
        platform="twitch",
        channel_id="c-1",
        event_id="e1",
        approved_action="moderate",
        approved_moderation_action="timeout",
    )

    async def allowed(**overrides):
        arguments = dict(
            platform="twitch", channel_id="c-1", moderation_action="timeout"
        )
        arguments.update(overrides)
        return await precedent.has_moderation_precedent(
            store, CREATOR, ASSISTANT, **arguments
        )

    assert await allowed()
    assert not await allowed(moderation_action="ban"), "a timeout is not a ban"
    assert not await allowed(channel_id="c-2"), "another room is another room"
    assert not await allowed(platform="discord"), "another platform is another platform"


@pytest.mark.asyncio
async def test_a_correction_rewrites_the_decision_and_becomes_a_rule():
    store = _Store()
    await precedent.store_decision_record(
        store,
        CREATOR,
        ASSISTANT,
        platform="discord",
        channel_id="c-9",
        channel_name="general",
        event=_event("check out my link"),
        action="ignore",
        moderation_action="none",
        reasoning="ordinary chatter",
        confidence=0.4,
    )
    result = await precedent.apply_decision_correction(
        store,
        CREATOR,
        ASSISTANT,
        DecisionCorrection(
            platform="discord",
            channel_id="c-9",
            event_id="e1",
            corrected_action="moderate",
            corrected_moderation_action="delete",
            note="I never allow unsolicited links.",
        ),
    )
    assert result["decision_found"] is True
    assert "delete" in (result["learned_rule"] or "")
    assert "I never allow unsolicited links." in (result["learned_rule"] or "")

    # The correction is the owner's own word, so it counts as precedent.
    assert await precedent.has_moderation_precedent(
        store,
        CREATOR,
        ASSISTANT,
        platform="discord",
        channel_id="c-9",
        moderation_action="delete",
    )


@pytest.mark.asyncio
async def test_a_notification_survives_a_resume_without_being_queued_twice():
    """The node that queues a notification re-runs every time the owner resumes."""
    store = _Store()
    event = _event("only you can answer this")
    for _ in range(3):
        await precedent.queue_notification(
            store,
            CREATOR,
            ASSISTANT,
            platform="slack",
            channel_id="c-3",
            channel_name="#ops",
            event=event,
            action="notify",
            moderation_action="none",
            reasoning="the owner decides",
            item_id="item-1",
        )
    waiting = await precedent.list_notifications(store, CREATOR, ASSISTANT)
    assert len(waiting) == 1, "the same message must not stack up on every resume"

    acknowledged = await precedent.acknowledge_notifications(
        store, CREATOR, ASSISTANT, [waiting[0]["notification_id"]]
    )
    assert acknowledged == 1
    assert await precedent.list_notifications(store, CREATOR, ASSISTANT) == []


@pytest.mark.asyncio
async def test_rooms_are_remembered_and_can_be_left():
    store = _Store()
    await precedent.record_channel(
        store,
        CREATOR,
        ASSISTANT,
        platform="slack",
        channel_id="C123",
        channel_name="#general",
        owns_channel=True,
    )
    # Recording the same room again updates rather than duplicates.
    await precedent.record_channel(
        store, CREATOR, ASSISTANT, platform="slack", channel_id="C123", channel_name="#general-renamed"
    )
    channels = await precedent.list_channels(store, CREATOR, ASSISTANT)
    assert len(channels) == 1
    assert channels[0]["channel_name"] == "#general-renamed"

    assert await precedent.forget_channel(
        store, CREATOR, ASSISTANT, platform="slack", channel_id="C123"
    )
    assert await precedent.list_channels(store, CREATOR, ASSISTANT) == []
    assert not await precedent.forget_channel(
        store, CREATOR, ASSISTANT, platform="slack", channel_id="C123"
    )
