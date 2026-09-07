"""The persisted shape of a connect card, and how a reply carries it.

A connect card used to live only inside a paused run: the moment the owner
signed in, the client dropped the card and a reload showed nothing. Now the
card is a RECORD the reply carries in ``response_metadata["connections"]``
(checkpointed with the message, forwarded verbatim on the ``done`` frame), so
the transcript shows "Gmail · ✓ Added · 6 tools" for as long as the thread
lives. The records are built here from the ``connect_account`` tool's results
so the graph and the acknowledgement turn agree on one shape.
"""

from __future__ import annotations

import json
from typing import Any

CARD_STATUS_CONNECTED = "connected"
CARD_STATUS_CANCELLED = "cancelled"
CARD_STATUS_FAILED = "failed"
CARD_STATUS_PENDING_LOGIN = "pending_login"
CARD_STATUS_NOT_CONNECTED = "not_connected"

CONNECT_TOOL_NAMES = frozenset({"connect_account", "connect_mailbox_account"})
ACKNOWLEDGEMENT_MESSAGE_KIND = "connection_acknowledgement"


def connection_card_record(
    provider: Any,
    record: dict[str, Any] | None,
    *,
    status: str,
    error: str | None = None,
    tool_names: list[str] | None = None,
) -> dict[str, Any]:
    """Describe one connection outcome for the transcript."""
    from src.anubis.utils.connected_accounts.store import public_account_view

    view = public_account_view(record) if record else {}
    names = list(tool_names or view.get("tool_names") or [])
    if not names and provider is not None:
        from src.anubis.utils.connected_accounts.tool_factories import tool_names_for

        names = tool_names_for(provider, record)
    return {
        "provider": getattr(provider, "name", None) or (record or {}).get("provider"),
        "display_name": getattr(provider, "display_name", None)
        or (record or {}).get("provider"),
        "icon_key": getattr(provider, "icon_key", "") or "custom",
        "category": getattr(provider, "category", "custom"),
        "login_mode": getattr(provider, "login_mode", "none"),
        "status": status,
        "account_key": view.get("account_key"),
        "display_label": view.get("display_label"),
        "account_address": view.get("account_address"),
        "connected_at": view.get("connected_at"),
        "tool_count": len(names),
        "tool_names": names,
        "error": error,
    }


def card_status_line(card: dict[str, Any]) -> str:
    """One line the voice caption strip and the read-only card show."""
    name = card.get("display_name") or card.get("provider") or "Account"
    status = card.get("status")
    if status == CARD_STATUS_CONNECTED:
        count = int(card.get("tool_count") or 0)
        label = card.get("display_label") or card.get("account_address")
        suffix = f" · {count} tools" if count else ""
        who = f" · Connected as {label}" if label else ""
        return f"{name} · Added{suffix}{who}"
    if status == CARD_STATUS_PENDING_LOGIN:
        return f"{name} · Waiting for sign-in"
    if status == CARD_STATUS_CANCELLED:
        return f"{name} · Not connected"
    if status == CARD_STATUS_FAILED:
        return f"{name} · Sign-in failed"
    return f"{name} · Not connected"


def _tool_message_payload(message: Any) -> dict[str, Any] | None:
    content = getattr(message, "content", None)
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                content = part.get("text")
                break
    if not isinstance(content, str):
        return None
    try:
        parsed = json.loads(content)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def connection_records_from_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Lift every ``card`` a ``connect_account`` tool call produced this turn."""
    cards: list[dict[str, Any]] = []
    for message in messages or []:
        if getattr(message, "type", None) != "tool":
            continue
        if str(getattr(message, "name", "") or "") not in CONNECT_TOOL_NAMES:
            continue
        payload = _tool_message_payload(message)
        if not payload:
            continue
        card = payload.get("card")
        if isinstance(card, dict):
            cards.append(dict(card))
    return cards


def connection_acknowledgement_card(messages: list[Any]) -> list[dict[str, Any]]:
    """Return the card the "+" menu attached to a hidden acknowledgement turn, if any."""
    for message in reversed(list(messages or [])):
        if getattr(message, "type", None) != "human":
            continue
        extra = getattr(message, "additional_kwargs", None) or {}
        if extra.get("kind") == ACKNOWLEDGEMENT_MESSAGE_KIND and isinstance(
            extra.get("connection"), dict
        ):
            return [dict(extra["connection"])]
        return []
    return []


def acknowledgement_instruction(card: dict[str, Any]) -> str:
    """Return the server-authored text of a hidden acknowledgement turn."""
    name = card.get("display_name") or card.get("provider") or "an account"
    label = card.get("display_label") or card.get("account_address") or ""
    as_label = f" as {label}" if label else ""
    tools = ", ".join(str(entry) for entry in (card.get("tool_names") or [])[:8])
    tools_line = f" The tools now available: {tools}." if tools else ""
    return (
        f"The owner just connected {name}{as_label} from the connectors menu."
        f"{tools_line} Acknowledge the connection in one or two sentences and say "
        "what can be done with the account now. Do not ask for any credential."
    )
