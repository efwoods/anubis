"""Write a finished phone call into chat and the agent inbox."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from src.anubis.utils.inbox.repository import (
    ACTION_NOTIFY_OWNER,
    STATE_PENDING_OWNER,
    get_inbox_repository,
)

logger = logging.getLogger(__name__)

PHONE_CALL_SOURCE_KIND = "phone_call"


def format_result_body(result: dict[str, Any], *, transcript_snippet: str = "") -> str:
    """Human-readable inbox / chat body for one call result."""
    cost = result.get("cost")
    cost_text = f"${cost:.2f}" if isinstance(cost, (int, float)) else "unknown"
    ready = result.get("ready_at") or "unknown"
    location = result.get("location") or "unknown"
    travel = result.get("travel_minutes")
    if isinstance(travel, (int, float)):
        travel_text = f"{int(round(travel))} min"
    else:
        travel_text = result.get("travel_text") or "unknown"
    destination = result.get("destination_name") or "the restaurant"
    outcome = result.get("outcome") or "unknown"
    lines = [
        f"{destination}: {outcome}.",
        f"Cost: {cost_text}.",
        f"Ready: {ready}.",
        f"Address: {location}.",
        f"Travel: {travel_text}.",
    ]
    if outcome == "ended_payment_required":
        lines.append(
            "The restaurant asked for a card on the line. The call ended. "
            "Pickup and pay at the counter only."
        )
    snippet = (transcript_snippet or "").strip()
    if snippet:
        lines.append("")
        lines.append(snippet[:400])
    return "\n".join(lines)


def format_chat_message(result: dict[str, Any], *, transcript_snippet: str = "") -> str:
    """Assistant message posted on the originating thread."""
    return format_result_body(result, transcript_snippet=transcript_snippet)


async def write_phone_call_result(
    *,
    user_id: str,
    assistant_id: str,
    call_id: str,
    result: dict[str, Any],
    transcript: str = "",
    thread_id: str | None = None,
    send_ios_notification: Any | None = None,
) -> dict[str, Any]:
    """Create an inbox item and a chat-shaped payload for a finished call.

    Chat persistence is the caller's job (the worker or the API has the
    LangGraph client). This function never dials iOS ``place_call``. The
    optional ``send_ios_notification`` callback may fire the existing
    ``send_notification`` MCP tool after the SIP call finishes.
    """
    destination = str(result.get("destination_name") or "Phone call")
    subject = f"{destination} order placed"
    if result.get("outcome") == "ended_payment_required":
        subject = f"{destination} call ended — card required"
    elif result.get("outcome") not in {"placed", None, ""}:
        subject = f"{destination} call: {result.get('outcome')}"
    snippet = (transcript or "").strip()
    body = format_result_body(result, transcript_snippet=snippet)
    item = {
        "item_id": str(uuid4()),
        "user_id": user_id,
        "assistant_id": assistant_id,
        "source_kind": PHONE_CALL_SOURCE_KIND,
        "account_key": "phone",
        "external_id": call_id,
        "external_thread_id": thread_id,
        "sender": "phone_call",
        "recipients": [],
        "subject": subject,
        "body_text": body,
        "received_at": datetime.now(UTC),
        "message_kind": "notification",
        "decision": ACTION_NOTIFY_OWNER,
        "needs_owner_action": False,
        "reason": "Finished phone call",
        "state": STATE_PENDING_OWNER,
        "available_actions": [ACTION_NOTIFY_OWNER],
    }
    inbox = get_inbox_repository()
    stored_item = None
    if inbox is not None:
        try:
            stored_item = await inbox.create_item(item)
        except Exception:
            logger.exception("Could not write the phone-call inbox item")
    chat = {
        "role": "assistant",
        "content": format_chat_message(result, transcript_snippet=snippet),
        "cost": result.get("cost"),
        "ready_at": result.get("ready_at"),
        "location": result.get("location"),
        "travel_minutes": result.get("travel_minutes"),
        "travel_text": result.get("travel_text"),
        "outcome": result.get("outcome"),
        "call_id": call_id,
        "source_kind": PHONE_CALL_SOURCE_KIND,
    }
    if send_ios_notification is not None:
        try:
            await send_ios_notification(subject, body)
        except Exception:
            logger.info("Optional iOS notification after the SIP call failed", exc_info=True)
    return {"inbox_item": stored_item or item, "chat_message": chat}
