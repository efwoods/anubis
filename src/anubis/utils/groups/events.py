"""What a bot sends the API about a group conversation, and what the API sends back.

One shape for every platform. A Slack bot, a Discord bot and a Twitch bot each
post batches of chat messages to ``POST /groups/{assistant_id}/events`` and
each receives one decision per message, so a change to the wire format is one
change here rather than three changes in three languages.

The models are ported from the ``z`` line's live-stream moderation
(``5557c15``), widened from moderation alone to full participation: an event
now says whether the avatar was ``mentioned``, the request says which actions
the bot can actually carry out in that room (``available_actions``) and
whether the owner administers the room (``owns_channel``). Both of those are
load-bearing rather than informational — the classifier is never offered an
action the platform cannot perform, and a moderation action is never proposed
in a room the owner does not administer.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

# What the avatar can decide to do about one message in a room.
#
# This is the whole repertoire of a member of that room, not a moderation menu.
# A person in a channel reacts far more often than they reply, takes a private
# matter to a direct message rather than answering it in front of everybody,
# answers in the thread a question was asked in, and says "I will come back to
# you on that" and then does. An avatar with only reply-or-silence available
# reads as a bot no matter how well it writes, because the shape of its
# participation is wrong.
GROUP_ACTIONS = (
    "ignore",
    "react",
    "respond",
    "reply_in_thread",
    "direct_message",
    "follow_up",
    "notify",
    "moderate",
)

# What a bot reports it can actually do in one room. The avatar is never
# offered a capability the platform or the bot's permissions do not have, and a
# decision that needs a missing one degrades rather than being dropped.
GROUP_CAPABILITIES = (
    "react",
    "reply",
    "thread",
    "direct_message",
    "upload_file",
    "edit_own",
    "delete_own",
    "read_history",
    "typing",
    "presence",
)

# What "moderate" can mean. Which of these a platform actually offers is
# reported per request by the bot: Slack, for example, offers no timeout and no
# ban, so a Slack request never lists them.
MODERATION_ACTIONS = ("none", "warn", "delete", "timeout", "ban")

# The platforms with a bot. Twitch is restricted to the owner's personal avatar
# elsewhere (``resolve_avatar_for_group``); the list itself is open.
GROUP_PLATFORMS = ("slack", "discord", "twitch")


def stable_event_key(platform: str, channel_id: str, event_id: str) -> str:
    """One deterministic identifier for a message, so a retry updates rather than duplicates.

    A bot that resends a batch after a network failure must not create a second
    decision record for the same message, and a correction the owner makes
    later has to find the decision by the platform's own identifiers.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{platform}:{channel_id}:{event_id}"))


class GroupEvent(BaseModel):
    """One message in a room, as the bot saw the message."""

    event_id: str = Field(description="The platform's identifier for this message.")
    author_id: str = Field(description="The platform's identifier for the author.")
    author_name: str = Field(default="", description="The author's display name.")
    text: str = Field(description="The message text.")
    mentioned: bool = Field(
        default=False,
        description=(
            "True when the message addresses the avatar directly — an @mention, a "
            "reply to one of the avatar's own messages, or a direct message."
        ),
    )
    posted_at: str | None = Field(
        default=None, description="ISO 8601 timestamp of when the message was posted."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Anything else the platform knows: badges, reply-to, attachments.",
    )


class GroupEventsRequest(BaseModel):
    """A batch of messages from one room."""

    platform: str = Field(
        description="The platform name: slack, discord, or twitch."
    )
    channel_id: str = Field(
        description="The platform's identifier for the channel, room, or chat."
    )
    channel_name: str = Field(default="", description="A display name for the room.")
    owns_channel: bool = Field(
        default=False,
        description=(
            "True when the owner administers this room. A moderation action is "
            "never taken in a room the owner does not administer."
        ),
    )
    available_actions: list[str] = Field(
        default_factory=list,
        description=(
            "The moderation actions this bot can actually carry out in this room, "
            "given the bot's permissions there: any of warn, delete, timeout, ban. "
            "An empty list means the avatar may take part but never moderate."
        ),
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description=(
            "Everything else this bot can do in this room — react, reply, thread, "
            "direct_message, upload_file, edit_own, delete_own, read_history, typing, "
            "presence. An empty list is read as reply only, which is what a bot that "
            "has not been taught to report its capabilities can always do."
        ),
    )
    events: list[GroupEvent] = Field(default_factory=list)


class GroupDecision(BaseModel):
    """What the bot should do about one message."""

    event_id: str
    author_id: str = ""
    action: Literal[
        "ignore",
        "react",
        "respond",
        "reply_in_thread",
        "direct_message",
        "follow_up",
        "notify",
        "moderate",
    ]
    moderation_action: Literal["none", "warn", "delete", "timeout", "ban"] = "none"
    reply: str | None = None
    reaction: str | None = Field(
        default=None,
        description="The emoji to react with, when the action is react: a name like 'tada' or the character itself.",
    )
    direct_message_to: str | None = Field(
        default=None,
        description="Whose direct message to open, when the action is direct_message.",
    )
    follow_up_after_seconds: int | None = Field(
        default=None,
        description="How long to wait before coming back to this, when the action is follow_up.",
    )
    reasoning: str = ""
    confidence: float = 0.0
    applied_rule: str | None = None
    item_id: str | None = Field(
        default=None,
        description="The inbox item the owner decides on, when the decision waits for the owner.",
    )
    notification_id: str | None = None


class DecisionCorrection(BaseModel):
    """The owner's correction of a decision the avatar already made."""

    platform: str
    channel_id: str
    event_id: str
    corrected_action: Literal["ignore", "respond", "notify", "moderate"]
    corrected_moderation_action: Literal[
        "none", "warn", "delete", "timeout", "ban"
    ] = "none"
    note: str = Field(
        default="", description="Why the owner corrected the decision, in the owner's words."
    )


class PolicyRule(BaseModel):
    """One rule the owner dictated, or one learned from a correction."""

    rule: str = Field(description="One complete standalone rule.")
    rule_context: str = Field(
        default="", description="When the rule applies, or where the rule came from."
    )


def render_event(event: GroupEvent, platform: str, channel_name: str) -> str:
    """Render one message the way the classifier and the stored record both read it."""
    author = event.author_name or event.author_id
    where = channel_name or platform
    mention = " (addressed to the avatar)" if event.mentioned else ""
    return f"[{platform} · {where}] {author}{mention}: {event.text}"


__all__ = [
    "GROUP_ACTIONS",
    "GROUP_CAPABILITIES",
    "GROUP_PLATFORMS",
    "MODERATION_ACTIONS",
    "DecisionCorrection",
    "GroupDecision",
    "GroupEvent",
    "GroupEventsRequest",
    "PolicyRule",
    "render_event",
    "stable_event_key",
]
