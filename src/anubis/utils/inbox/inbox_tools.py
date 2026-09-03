"""The chat tools that let the owner run the agent inbox in conversation.

Built per turn for the personal avatar only. The avatar reports pending items,
resolves one the way the panel would (the same ``HumanResponse`` reaches the
same paused graph), and can trigger a poll on demand.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.tools import tool

logger = logging.getLogger(__name__)

INBOX_TOOL_NAMES: tuple[str, ...] = (
    "list_inbox_notifications",
    "resolve_inbox_notification",
    "triage_inbox_now",
)


def build_inbox_tools(context: Any, *, user_id: str, assistant_id: str) -> list[Any]:
    """Build the inbox tools bound to the owner's personal avatar."""
    from src.anubis.utils.inbox.repository import (
        OPEN_STATES,
        get_inbox_repository,
        public_item_view,
    )

    @tool
    async def list_inbox_notifications(limit: int = 10) -> dict[str, Any]:
        """List the messages waiting for the owner in the agent inbox.

        Call this when the owner asks whether there is anything to be aware of,
        what needs a reply, or what is pending — and at the start of a
        conversation when the INBOX_NOTIFICATIONS section says items are
        waiting. Each item carries who wrote, the subject, why it was flagged,
        and — for a proposed reply — the draft and its confidence.

        Args:
            limit: Maximum items to return.
        """
        repository = get_inbox_repository()
        if repository is None:
            return {"status": "unavailable", "items": []}
        items = await repository.list_items(
            assistant_id=assistant_id,
            states=OPEN_STATES,
            limit=max(1, min(int(limit or 10), 50)),
        )
        return {
            "pending_count": await repository.count_open(assistant_id),
            "items": [public_item_view(item) for item in items],
        }

    @tool
    async def resolve_inbox_notification(
        item_id: str,
        decision: str,
        reply_text: str | None = None,
    ) -> dict[str, Any]:
        """Resolve one pending inbox item the way the owner decided in chat.

        decision is one of:
        - "accept": send the proposed reply as drafted (reply items) or mark a
          notification as seen.
        - "edit": send the reply with reply_text as its body instead of the draft.
        - "response": send reply_text as the owner's own reply to the message.
        - "ignore": do nothing further with this message.

        Confirm what the owner wants before sending. The owner's decision also
        teaches the inbox how to handle this sender next time.

        Args:
            item_id: The item's id from list_inbox_notifications.
            decision: One of accept, edit, response, ignore.
            reply_text: The reply body for edit or response.
        """
        from src.anubis.utils.inbox.poller import resume_inbox_item

        decision_type = str(decision or "").strip().lower()
        if decision_type not in ("accept", "edit", "response", "ignore"):
            return {
                "status": "invalid_decision",
                "error": "decision must be accept, edit, response, or ignore",
            }
        if decision_type in ("edit", "response") and not (reply_text or "").strip():
            return {
                "status": "missing_reply",
                "error": "reply_text is required for edit or response",
            }
        human_response: dict[str, Any] = {"type": decision_type, "args": None}
        if decision_type == "edit":
            human_response["args"] = {
                "action": "send_reply",
                "args": {"body": reply_text},
            }
        elif decision_type == "response":
            human_response["args"] = reply_text
        item = await resume_inbox_item(
            context, item_id=item_id, human_response=human_response
        )
        if item is None:
            return {"status": "not_found", "item_id": item_id}
        return {"status": item.get("state"), "item": public_item_view(item)}

    @tool
    async def triage_inbox_now() -> dict[str, Any]:
        """Check the owner's connected mailboxes for new mail right now and triage it.

        Use when the owner asks to check their email, sort their inbox, or see
        whether anything new arrived. Returns how many mailboxes were checked
        and how many new items were created; follow with
        list_inbox_notifications to report them.
        """
        from src.anubis.utils.inbox.poller import poll_connected_mailboxes

        try:
            return await poll_connected_mailboxes(context, only_user_id=user_id)
        except Exception as poll_error:  # noqa: BLE001
            logger.exception("triage_inbox_now failed")
            return {"status": "error", "error": str(poll_error)}

    return [list_inbox_notifications, resolve_inbox_notification, triage_inbox_now]
