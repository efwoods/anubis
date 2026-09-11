"""The chat tools that let the owner run the group conversations in conversation.

Built per turn for the personal avatar only, and modelled directly on
``src/anubis/utils/inbox/inbox_tools.py``. The owner can ask what is waiting
from a room, answer it the way the panel would — the same ``HumanResponse``
reaches the same paused graph — and see or leave the rooms the avatar takes
part in, without opening the panel at all.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.tools import tool

logger = logging.getLogger(__name__)

GROUP_TOOL_NAMES: tuple[str, ...] = (
    "list_group_notifications",
    "resolve_group_notification",
    "list_group_channels",
    "leave_group_channel",
)


def build_group_tools(context: Any, *, user_id: str, assistant_id: str, store: Any) -> list[Any]:
    """Build the group-conversation tools bound to the owner's personal avatar."""
    from src.anubis.utils.groups.precedent import (
        forget_channel,
        list_channels,
        list_notifications,
    )
    from src.anubis.utils.groups.runner import resume_group_item

    @tool
    async def list_group_notifications(limit: int = 10) -> dict[str, Any]:
        """List what is waiting for the owner from a Slack, Discord, or Twitch room.

        Call this when the owner asks what is happening in their rooms, whether
        anybody needs them, or what the avatar has flagged — and at the start of
        a conversation when the GROUP_CONVERSATIONS section says something is
        waiting. Each entry carries the platform, the room, who spoke, what was
        said, and what the avatar proposed to do about the message.

        Args:
            limit: Maximum notifications to return.
        """
        if store is None:
            return {"status": "unavailable", "notifications": []}
        notifications = await list_notifications(
            store, user_id, assistant_id, unread_only=True
        )
        capped = notifications[: max(1, min(int(limit or 10), 50))]
        return {"waiting_count": len(notifications), "notifications": capped}

    @tool
    async def resolve_group_notification(
        item_id: str,
        decision: str,
        message: str = "",
        action: str = "",
        moderation_action: str = "",
        note: str = "",
    ) -> dict[str, Any]:
        """Answer one waiting group message on the owner's behalf.

        Use this when the owner says what to do about something the avatar
        flagged from a room: post the reply, post different words, take a
        moderation action instead, or let the message go.

        Args:
            item_id: The identifier of the waiting item, from list_group_notifications.
            decision: 'accept' to do what the avatar proposed, 'edit' to change
                what is done, or 'ignore' to do nothing.
            message: The words to post, when the owner dictates a different reply.
            action: The action to take instead: 'post_reply', 'moderate', or
                'notify_owner'. Leave empty to keep what the avatar proposed.
            moderation_action: 'warn', 'delete', 'timeout', or 'ban', when the
                action is moderate.
            note: The owner's reason, which becomes a rule the avatar follows
                the next time a message like this appears.
        """
        decision = (decision or "").strip().lower()
        if decision not in ("accept", "edit", "ignore", "response"):
            return {
                "status": "error",
                "error": "The decision must be accept, edit, or ignore.",
            }
        human_response: dict[str, Any]
        if decision in ("edit", "response") or action or moderation_action or message:
            arguments: dict[str, Any] = {}
            if message.strip():
                arguments["body"] = message.strip()
            if moderation_action.strip():
                arguments["moderation_action"] = moderation_action.strip().lower()
            if note.strip():
                arguments["note"] = note.strip()
            human_response = {
                "type": "edit",
                "args": {
                    "action": (action or "").strip().lower() or None,
                    "args": arguments,
                    "note": note.strip(),
                },
            }
        else:
            human_response = {"type": decision, "args": None}
        item = await resume_group_item(
            context, item_id=item_id, human_response=human_response
        )
        if item is None:
            return {"status": "not_found", "item_id": item_id}
        return {"status": "resolved", "item_id": item_id, "state": item.get("state")}

    @tool
    async def list_group_channels() -> dict[str, Any]:
        """List the Slack, Discord, and Twitch rooms this avatar takes part in."""
        if store is None:
            return {"status": "unavailable", "channels": []}
        return {"channels": await list_channels(store, user_id, assistant_id)}

    @tool
    async def leave_group_channel(platform: str, channel_id: str) -> dict[str, Any]:
        """Stop taking part in one room.

        Args:
            platform: 'slack', 'discord', or 'twitch'.
            channel_id: The room's identifier, from list_group_channels.
        """
        if store is None:
            return {"status": "unavailable"}
        left = await forget_channel(
            store,
            user_id,
            assistant_id,
            platform=(platform or "").strip().lower(),
            channel_id=(channel_id or "").strip(),
        )
        return {"status": "left" if left else "not_found", "channel_id": channel_id}

    return [
        list_group_notifications,
        resolve_group_notification,
        list_group_channels,
        leave_group_channel,
    ]


__all__ = ["GROUP_TOOL_NAMES", "build_group_tools"]
