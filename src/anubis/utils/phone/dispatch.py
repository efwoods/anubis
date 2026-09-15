"""Inbound caller-ID dispatch to the owner who verified that mobile.

One shared platform number answers every inbound call. The ``From`` header
selects the personal avatar whose Phone connection stores that
``owner_mobile_e164``. Unknown caller IDs are refused; they never mint a
number.
"""

from __future__ import annotations

from typing import Any

from src.anubis.utils.phone.numbers import PhoneNumberError, normalize_e164


async def owner_for_inbound_caller(
    caller_e164: str,
    *,
    store: Any = None,
    repository: Any = None,
) -> dict[str, Any] | None:
    """Return the connected Phone record whose mobile matches ``caller_e164``.

    ``repository`` is a connected-accounts repository that can list every
    telephony record. When that is missing, ``store`` is unused for a full
    scan — inbound dispatch needs the accounts table, not a single-user
    namespace.
    """
    try:
        wanted = normalize_e164(caller_e164)
    except PhoneNumberError:
        return None
    records = await _list_telephony_records(store=store, repository=repository)
    for record in records:
        transport = record.get("transport") or {}
        stored = str(transport.get("owner_mobile_e164") or record.get("account_address") or "")
        try:
            if normalize_e164(stored) == wanted and transport.get("sip_enabled") is True:
                return record
        except PhoneNumberError:
            continue
    return None


async def _list_telephony_records(*, store: Any, repository: Any) -> list[dict[str, Any]]:
    if repository is not None and hasattr(repository, "list_by_kind"):
        try:
            return await repository.list_by_kind("telephony", status="connected")
        except Exception:
            return []
    if repository is not None and hasattr(repository, "list_all"):
        try:
            return [
                record
                for record in await repository.list_all()
                if record.get("kind") == "telephony"
            ]
        except Exception:
            return []
    if store is not None and hasattr(store, "list_telephony_records"):
        try:
            return await store.list_telephony_records()
        except Exception:
            return []
    return []


def inbound_reject_message() -> str:
    """What an unknown caller hears instead of a new number."""
    return (
        "This number belongs to Neural Nexus. Call from the mobile you "
        "verified on your personal avatar, or hang up."
    )
