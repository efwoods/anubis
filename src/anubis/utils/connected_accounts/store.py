"""Durable records for the external accounts a user has connected.

Shaped after ``mcp_connection_namespace``
(``src/anubis/utils/tools/data_analysis/backend.py``), deliberately, including
the lesson that namespace learned the hard way.

**Keyed per account from the first commit.** The Model Context Protocol
namespaces originally held a single record under a constant key. That made a
second machine silently overwrite the first, and it let a development daemon
delete production's record, because both wrote the same key into a shared store.
The same mistake here would mean a user's second mailbox evicting the first. The
key is therefore ``"{provider}:{account_address}"`` — unique per account, stable
across reconnects of the same account, and self-describing when read straight
out of the store.

**The plaintext credential is never in a record.** ``encrypted_secret`` holds
Fernet ciphertext produced by ``src/anubis/utils/secret_store.py``. Nothing in
this module decrypts; the tool layer does that at the moment it dials the mail
server, which keeps the plaintext out of anything that logs or serializes a
record.

**Records bind to exactly one avatar.** ``assistant_id`` names the personal
avatar the account was connected for, mirroring how a connection record names
the avatar a device is bound to. A demoted avatar therefore loses access without
the record having to be rewritten, and :func:`bound_accounts_for` is the single
gate the tool layer consults.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

CONNECTED_ACCOUNT_NAMESPACE_KIND = "connected_account"

# Status values a record may carry. "connected" is the only status the tool
# layer acts on; "needs_reconnect" is written when a stored credential stops
# authenticating, so the avatar can tell the owner which account to fix instead
# of failing every mail call with an opaque error.
STATUS_CONNECTED = "connected"
STATUS_NEEDS_RECONNECT = "needs_reconnect"


def connected_account_namespace(user_id: str) -> tuple[str, str]:
    """Store namespace holding one record per connected external account.

    Scoped to the user (a two-element tuple) and keyed INSIDE that namespace by
    :func:`account_key`, so a user who connects a personal Gmail account, a work
    Gmail account, and later a social account has three coexisting records
    rather than three generations of one overwritten record.
    """
    return (user_id, CONNECTED_ACCOUNT_NAMESPACE_KIND)


def account_key(provider_name: str, account_address: str) -> str:
    """Build the per-account store key.

    Lower-cased on both halves because the key must be stable across reconnects,
    and a user who types "Evan@Example.com" one day and "evan@example.com" the
    next means the same mailbox. Without normalization the second connect would
    create a duplicate record rather than refreshing the first.
    """
    return (
        f"{str(provider_name).strip().lower()}:{str(account_address).strip().lower()}"
    )


def derive_display_label(account_address: str) -> str:
    """Derive a short human name for one account from its address.

    The avatar has to name accounts in conversation ("I found it in your work
    mailbox"), and a full address is both long and needlessly discloses the
    domain when the reply is read over someone's shoulder. The local part is
    used, which is what a person would say out loud.
    """
    local_part = str(account_address or "").split("@", 1)[0].strip()
    return local_part or str(account_address or "").strip() or "Mailbox"


def deduplicate_label(
    label: str, existing_records: list[dict[str, Any]], key: str
) -> str:
    """Return a label unique among the user's connected accounts.

    Two accounts can easily share a local part — ``evan@gmail.com`` and
    ``evan@work.com`` both derive "evan" — and an avatar that reports results per
    account cannot distinguish two identically named ones. Later accounts get a
    counted suffix ("evan 2").

    The account's own existing record is skipped while scanning, so reconnecting
    an account keeps the label it already had instead of climbing a counter on
    every reconnect. This mirrors ``deduplicate_label`` in
    ``data_analysis/devices.py``, which solved the same problem for two machines
    both named "Ubuntu".
    """
    taken = {
        str(record.get("display_label") or "")
        for record in existing_records
        if record.get("account_key") != key
    }
    if label not in taken:
        return label
    candidate_index = 2
    while f"{label} {candidate_index}" in taken:
        candidate_index += 1
    return f"{label} {candidate_index}"


def build_account_record(
    *,
    provider: Any,
    account_address: str,
    display_label: str,
    encrypted_secret: str,
    assistant_id: str,
) -> dict[str, Any]:
    """Assemble the stored value for one connected account.

    Connection details are copied from the provider row rather than referenced,
    so a record read years later still describes how to reach the server even if
    the registry row has since changed — the same reason a Model Context
    Protocol connection record stores its own URL instead of recomputing one.
    """
    now = datetime.now(UTC).isoformat()
    return {
        "account_key": account_key(provider.name, account_address),
        "provider": provider.name,
        "kind": provider.kind,
        "credential_mechanism": provider.credential_mechanism,
        "account_address": str(account_address).strip(),
        "display_label": display_label,
        "encrypted_secret": encrypted_secret,
        "imap_host": provider.imap_host,
        "imap_port": provider.imap_port,
        "smtp_host": provider.smtp_host,
        "smtp_port": provider.smtp_port,
        "drafts_mailbox": provider.drafts_mailbox,
        "assistant_id": assistant_id,
        "status": STATUS_CONNECTED,
        "connected_at": now,
        "last_verified_at": now,
    }


def public_account_view(record: dict[str, Any]) -> dict[str, Any]:
    """Project a record down to the fields safe to return over the API.

    SECURITY: this is the ONLY shape an endpoint may return. It exists as a
    whitelist rather than as a "delete the secret key" blacklist so that a field
    added to the record later is excluded by default instead of leaking the
    first time someone forgets to update a removal list. Neither the plaintext
    credential (never stored) nor the ciphertext is included — the ciphertext is
    useless to a caller and its exposure only helps an attacker who later
    obtains the key.
    """
    return {
        "account_key": record.get("account_key"),
        "provider": record.get("provider"),
        "kind": record.get("kind"),
        "account_address": record.get("account_address"),
        "display_label": record.get("display_label"),
        "status": record.get("status"),
        "connected_at": record.get("connected_at"),
        "last_verified_at": record.get("last_verified_at"),
        "assistant_id": record.get("assistant_id"),
    }


async def read_connected_accounts(store: Any, user_id: str) -> list[dict[str, Any]]:
    """Return every connected-account record for a user.

    Returns an empty list when the namespace is empty or the store is
    unreachable: every caller treats "no accounts" and "cannot tell" the same
    way, and a store hiccup must never fail a conversation turn whose real work
    is something else.
    """
    if store is None:
        return []
    namespace = connected_account_namespace(user_id)
    try:
        items = await store.asearch(namespace, limit=100)
    except Exception:
        logger.debug(
            "Could not read connected accounts for user %s", user_id, exc_info=True
        )
        return []
    records: list[dict[str, Any]] = []
    for item in items or []:
        value = getattr(item, "value", None) or {}
        if value:
            records.append(value)
    return records


async def bound_accounts_for(
    store: Any, user_id: str, assistant_id: str
) -> list[dict[str, Any]]:
    """Return the connected accounts this avatar may act on.

    The sole gate the tool layer consults. Two conditions, both required: the
    record is in the connected state, and it is bound to the avatar currently
    answering. Binding to the avatar rather than to the user is what stops a
    second avatar of the same owner — or an avatar demoted out of the personal
    role — from reaching the owner's mail.
    """
    return [
        record
        for record in await read_connected_accounts(store, user_id)
        if record.get("status") == STATUS_CONNECTED
        and record.get("assistant_id") == assistant_id
    ]


async def save_connected_account(
    store: Any, user_id: str, record: dict[str, Any]
) -> None:
    """Write one account record, keyed by its account key.

    Re-saving the same account overwrites only that account's record, which is
    how a reconnect refreshes a rotated credential without disturbing the user's
    other accounts.
    """
    key = record.get("account_key")
    if not key:
        raise ValueError(
            "Cannot save a connected account without an account key; the record "
            "key is the account key."
        )
    await store.aput(connected_account_namespace(user_id), key=key, value=record)


async def clear_connected_account(store: Any, user_id: str, key: str) -> bool:
    """Delete one connected account. Report whether anything was removed.

    Takes a REQUIRED key and deletes exactly one record. It deliberately offers
    no "delete everything" mode: ``/disconnect_mcp`` shipped with an omitted
    identifier meaning "remove every device", which is the behaviour
    ``/mcp/unregister`` had to be changed to refuse after it destroyed a
    production record. Repeating that shape here would repeat the incident.

    Returns:
        True when a record existed and was deleted, False when nothing matched —
        so the endpoint can answer 404 rather than reporting a success that
        removed nothing.
    """
    if store is None or not key:
        return False
    namespace = connected_account_namespace(user_id)
    existing = await store.aget(namespace, key)
    if existing is None:
        return False
    await store.adelete(namespace, key)
    return True


async def mark_account_needs_reconnect(store: Any, user_id: str, key: str) -> None:
    """Flag an account whose stored credential stopped authenticating.

    Best-effort: a failure to record the flag must not turn a already-failing
    mail call into a raised exception. The flag is what lets the avatar say
    which account needs attention instead of silently returning nothing.
    """
    if store is None or not key:
        return
    try:
        namespace = connected_account_namespace(user_id)
        item = await store.aget(namespace, key)
        record = dict(getattr(item, "value", None) or {})
        if not record:
            return
        record["status"] = STATUS_NEEDS_RECONNECT
        await store.aput(namespace, key=key, value=record)
    except Exception:
        logger.debug(
            "Could not flag connected account %s as needing reconnect",
            key,
            exc_info=True,
        )
