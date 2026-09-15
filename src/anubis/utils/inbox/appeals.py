"""Ban appeals that arrive as mail to the Neural Nexus contact addresses.

A banned person is told to write to ``contact@neuralnexus.site``,
``business@neuralnexus.site``, or ``support@neuralnexus.site``. Those
messages reach the administrator's personal-avatar inbox only after that
avatar's mailbox is connected through an email provider: the poller already
reads that mailbox, and this module marks a fetched message as an appeal
when any recipient is one of those addresses. Regular mailbox triage is
skipped so an appeal is never auto-replied.
"""

from __future__ import annotations

from typing import Any, Iterable

from src.anubis.utils.inbox.repository import (
    ACTION_ACCEPT_BAN,
    ACTION_REVOKE_BAN,
    APPEAL_SOURCE_KIND,
    DECISION_NOTIFY,
    STATE_PENDING_OWNER,
)

DEFAULT_BAN_APPEAL_INBOX_ADDRESSES = (
    "contact@neuralnexus.site",
    "business@neuralnexus.site",
    "support@neuralnexus.site",
)


def _parse_address_list(raw: str) -> list[str]:
    addresses: list[str] = []
    for token in raw.replace(";", ",").replace(" or ", ",").split(","):
        address = email_address_from_header(token)
        if address and address not in addresses:
            addresses.append(address)
    return addresses


def email_address_from_header(value: str | None) -> str | None:
    """Return the bare address inside a From/To token, lower-cased."""
    text = str(value or "").strip()
    if not text:
        return None
    if "<" in text and ">" in text:
        text = text.split("<", 1)[1].split(">", 1)[0]
    text = text.strip().strip('"').casefold()
    if "@" not in text or " " in text:
        return None
    return text


def email_addresses_from_headers(*header_values: Any) -> list[str]:
    """Collect distinct addresses from To, Cc, Delivered-To, and similar headers."""
    found: list[str] = []
    for value in header_values:
        parts: Iterable[Any]
        if isinstance(value, (list, tuple)):
            parts = value
        else:
            parts = [value]
        for part in parts:
            for address in _parse_address_list(str(part or "")):
                if address not in found:
                    found.append(address)
    return found


def appeal_inbox_addresses(context: Any | None) -> list[str]:
    """The addresses whose incoming mail is a ban appeal.

    ``BAN_APPEAL_INBOX_ADDRESSES`` wins when set. A single custom
    ``BAN_APPEAL_CONTACT_EMAIL`` (tests, a one-address deploy) stands alone.
    Otherwise the three Neural Nexus contact addresses are used.
    """
    raw = str(getattr(context, "ban_appeal_inbox_addresses", None) or "").strip()
    if raw:
        return _parse_address_list(raw)
    primary = str(getattr(context, "ban_appeal_contact_email", None) or "").strip()
    primary_address = email_address_from_header(primary)
    default_primary = DEFAULT_BAN_APPEAL_INBOX_ADDRESSES[0]
    if primary_address and primary_address != default_primary:
        return [primary_address]
    return list(DEFAULT_BAN_APPEAL_INBOX_ADDRESSES)


def appeal_contact_phrase(context: Any | None) -> str:
    """The phrase a banned person is told to write to."""
    addresses = appeal_inbox_addresses(context)
    if not addresses:
        return DEFAULT_BAN_APPEAL_INBOX_ADDRESSES[0]
    if len(addresses) == 1:
        return addresses[0]
    if len(addresses) == 2:
        return f"{addresses[0]} or {addresses[1]}"
    return f"{', '.join(addresses[:-1])}, or {addresses[-1]}"


def matched_appeal_inbox_address(
    message: dict[str, Any] | None, context: Any | None
) -> str | None:
    """The appeal inbox address this message was addressed to, or None."""
    payload = message or {}
    recipients = email_addresses_from_headers(
        payload.get("recipients"),
        payload.get("cc"),
        payload.get("delivered_to"),
        payload.get("original_to"),
    )
    inbox = set(appeal_inbox_addresses(context))
    for address in recipients:
        if address in inbox:
            return address
    return None


def message_is_ban_appeal(
    message: dict[str, Any] | None,
    *,
    user_id: str | None,
    email: str | None = None,
    context: Any | None = None,
) -> bool:
    """True when this mail is an appeal to the administrator's connected mailbox."""
    from src.security.bans import is_unbannable_administrator

    if not is_unbannable_administrator(user_id=user_id, email=email, context=context):
        return False
    return matched_appeal_inbox_address(message, context) is not None


async def find_ban_for_appellant(
    pool: Any, sender_header: str | None
) -> dict[str, Any] | None:
    """The active ban for the appellant, or the most recent lifted row."""
    from src.security.bans import find_active_ban, list_bans

    sender_email = email_address_from_header(sender_header)
    if pool is None or not sender_email:
        return None
    active = await find_active_ban(pool, email=sender_email)
    if active is not None:
        return active
    for ban in await list_bans(pool, include_lifted=True, limit=200):
        if email_address_from_header(ban.get("email")) == sender_email:
            return ban
    return None


def appeal_item_fields(
    *,
    user_id: str,
    assistant_id: str,
    account_key: str | None,
    message: dict[str, Any],
    recipients: list[str],
    external_id: str | None,
    ban: dict[str, Any] | None,
    appeal_address: str | None,
) -> dict[str, Any]:
    """The inbox row for one appeal email. No triage graph runs on this item."""
    sender = str(message.get("sender") or "")
    ban_reason = str((ban or {}).get("reason") or "").strip()
    reason = (
        ban_reason
        if ban_reason
        else "Appeal received; no matching ban was found for the sender."
    )
    evidence = str((ban or {}).get("excerpt") or "").strip()
    return {
        "user_id": user_id,
        "assistant_id": assistant_id,
        "source_kind": APPEAL_SOURCE_KIND,
        "account_key": account_key,
        "external_id": external_id or None,
        "external_thread_id": message.get("thread_id"),
        "sender": sender,
        "recipients": list(recipients or []),
        "subject": message.get("subject") or "Ban appeal",
        "body_text": str(message.get("body_text") or ""),
        "received_at": message.get("sent_at"),
        "message_kind": APPEAL_SOURCE_KIND,
        "decision": DECISION_NOTIFY,
        "needs_owner_action": True,
        "reason": reason,
        "confidence": 1.0,
        "confidence_detail": {
            "ban_id": (ban or {}).get("ban_id"),
            "enforced": (ban or {}).get("enforced"),
            "banned_email": (ban or {}).get("email")
            or email_address_from_header(sender),
            "banned_user_id": (ban or {}).get("user_id"),
            "appeal_address": appeal_address,
            "supporting_evidence": evidence,
            "source": "appeal_email",
        },
        "available_actions": [ACTION_REVOKE_BAN, ACTION_ACCEPT_BAN],
        "state": STATE_PENDING_OWNER,
    }
