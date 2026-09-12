"""The accounts deep research turned up, and the loop that asks the owner about them.

Deep research already reads the subject's own pages. When one of those pages is
the subject's channel, profile, or repository host, the connector catalog already
covers that vendor — and the owner is one sign-in away from the avatar reading
that account directly instead of inferring the subject from what the open web
happened to publish.

Nothing here decides anything on the owner's behalf. A page at ``twitter.com``
means only "a page the research read lives at a host the catalog covers"; whether
that account is the owner's is a question only the owner can answer, so every
candidate becomes a question with the page that suggested it named alongside.

The record is durable for the same reason ``asset_bootstrap.record_bootstrap_outcome``
is: research jobs live in an in-process registry with a one-hour lifetime
(``src/api/research_jobs.py``), the progress stream is long gone by the time the
owner next opens a conversation, and the ask has to survive both.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DISCOVERED_ACCOUNTS_NAMESPACE_CATEGORY = "research_discovered_accounts"
"""Store-namespace category holding the accounts one research run turned up."""

ACCOUNT_STATUS_OPEN = "open"
ACCOUNT_STATUS_CONNECTED = "connected"
ACCOUNT_STATUS_DECLINED = "declined"


@dataclass(frozen=True)
class DiscoveredAccount:
    """One account the research suggests may belong to the subject.

    ``evidence_url`` is the page the research actually read. Naming that page when
    the question is put to the owner is what keeps the question honest: the owner
    can see why the account was suggested and say no when the suggestion is wrong.
    """

    provider: str
    display_name: str
    evidence_url: str
    supported_a_verified_fact: bool = False

    def as_record(self) -> dict[str, Any]:
        """Return the storable form of this candidate, opened for the owner."""
        return {
            "provider": self.provider,
            "display_name": self.display_name,
            "evidence_url": self.evidence_url,
            "supported_a_verified_fact": self.supported_a_verified_fact,
            "status": ACCOUNT_STATUS_OPEN,
            "inbox_item_id": None,
        }


@dataclass
class DiscoveredAccountsRecord:
    """Everything one research run found, and where each candidate has got to."""

    job_id: str
    subject_name: str
    created_at: str
    accounts: list[dict[str, Any]] = field(default_factory=list)


def discovered_accounts_namespace(
    creator_id: str, assistant_id: str
) -> tuple[str, str, str]:
    """Return the store namespace holding this avatar's discovered accounts."""
    return (creator_id, assistant_id, DISCOVERED_ACCOUNTS_NAMESPACE_CATEGORY)


def _hostname_of(url: str) -> str:
    """Return the lower-cased hostname of ``url``, or an empty string."""
    try:
        return (urlparse(str(url or "").strip()).hostname or "").lower()
    except Exception:  # noqa: BLE001 - a malformed address suggests nothing
        return ""


def candidate_accounts_from_research(
    summary: dict[str, Any],
    *,
    connected_provider_names: set[str] | None = None,
    maximum: int = 3,
) -> list[DiscoveredAccount]:
    """Return the accounts one research run suggests, best-evidenced first.

    A page that supported a fact the research actually verified outranks a page
    the search merely returned, because the first is evidence the page is about
    this subject and the second is evidence only that the page exists. Within each
    band the run's own ordering is kept.

    One candidate per provider: the owner is asked "is the YouTube channel yours",
    not asked twice because the research read two of its pages. Providers already
    connected are dropped, and so is any provider the catalog lists but cannot yet
    connect, because offering a connection that cannot be completed wastes the
    owner's only answer.
    """
    if maximum <= 0:
        return []

    from src.anubis.utils.connected_accounts.providers import (
        provider_for_public_profile_host,
    )

    already_connected = {
        str(name or "").strip().lower() for name in (connected_provider_names or set())
    }

    # Addresses the run leaned on, in the order the run ranked them, followed by
    # every other page it read.
    verified_urls: list[str] = [
        str(url) for url in (summary.get("media_source_urls") or []) if url
    ]
    verified_urls += [
        str(url) for url in (summary.get("bootstrap_media_urls") or []) if url
    ]
    verified_lookup = {url.strip().rstrip("/").lower() for url in verified_urls}
    other_urls = [
        str(source.get("url") or "")
        for source in (summary.get("sources") or [])
        if isinstance(source, dict) and source.get("url")
    ]

    candidates: list[DiscoveredAccount] = []
    seen_providers: set[str] = set()
    for url in verified_urls + other_urls:
        provider = provider_for_public_profile_host(_hostname_of(url))
        if provider is None:
            continue
        provider_name = provider.name.strip().lower()
        if provider_name in seen_providers or provider_name in already_connected:
            continue
        if not provider.is_available:
            continue
        # A machine pairs itself once the daemon is running and signed in, so
        # there is no account for the owner to connect and no question to ask.
        if provider.name == "desktop_mcp":
            continue
        seen_providers.add(provider_name)
        candidates.append(
            DiscoveredAccount(
                provider=provider.name,
                display_name=provider.display_name,
                evidence_url=url,
                supported_a_verified_fact=(
                    url.strip().rstrip("/").lower() in verified_lookup
                ),
            )
        )
        if len(candidates) >= maximum:
            break
    return candidates


async def record_discovered_accounts(
    store: Any,
    *,
    creator_id: str,
    assistant_id: str,
    job_id: str,
    subject_name: str,
    accounts: list[DiscoveredAccount],
) -> dict[str, Any] | None:
    """Persist what one research run found, merging with anything still open.

    A second research run must not reopen a question the owner has already
    answered, so a candidate whose stored status is no longer ``open`` keeps that
    status and the new run's version of it is discarded.
    """
    if not accounts:
        return None
    namespace = discovered_accounts_namespace(creator_id, assistant_id)
    existing = await read_discovered_accounts(
        store, creator_id=creator_id, assistant_id=assistant_id
    )
    settled_by_provider = {
        str(account.get("provider")): account
        for account in (existing or {}).get("accounts", [])
        if account.get("status") != ACCOUNT_STATUS_OPEN
    }
    merged: list[dict[str, Any]] = list(settled_by_provider.values())
    for candidate in accounts:
        if candidate.provider in settled_by_provider:
            continue
        merged.append(candidate.as_record())

    record = {
        "job_id": job_id,
        "subject_name": subject_name,
        "created_at": datetime.now(UTC).isoformat(),
        "accounts": merged,
    }
    try:
        await store.aput(namespace, key=assistant_id, value=record)
    except Exception as write_error:  # noqa: BLE001 - the research still succeeded
        logger.warning(
            "Could not record the accounts research found for %s: %s",
            assistant_id,
            write_error,
        )
        return None
    return record


async def read_discovered_accounts(
    store: Any, *, creator_id: str, assistant_id: str
) -> dict[str, Any] | None:
    """Return the accounts recorded for this avatar, if any were ever recorded."""
    try:
        item = await store.aget(
            discovered_accounts_namespace(creator_id, assistant_id), assistant_id
        )
    except Exception as read_error:  # noqa: BLE001 - a missing row is not an error
        logger.debug("Discovered-accounts lookup failed (continuing): %s", read_error)
        return None
    if item is None:
        return None
    value = getattr(item, "value", None)
    if value is None and isinstance(item, dict):
        value = item.get("value")
    return dict(value or {}) or None


def open_accounts(record: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the candidates in ``record`` the owner has not yet answered."""
    return [
        account
        for account in (record or {}).get("accounts", [])
        if account.get("status") == ACCOUNT_STATUS_OPEN
    ]


async def mark_discovered_account_resolved(
    store: Any,
    *,
    creator_id: str,
    assistant_id: str,
    provider: str,
    status: str = ACCOUNT_STATUS_CONNECTED,
    inbox_item_id: str | None = None,
) -> dict[str, Any] | None:
    """Close one candidate, leaving every other candidate open.

    Connecting a GitHub account answers the GitHub question and nothing else; the
    YouTube question stays open until the owner answers that one too.
    """
    record = await read_discovered_accounts(
        store, creator_id=creator_id, assistant_id=assistant_id
    )
    if not record:
        return None
    wanted = str(provider or "").strip().lower()
    changed = False
    for account in record.get("accounts", []):
        if str(account.get("provider") or "").strip().lower() != wanted:
            continue
        account["status"] = status
        account["resolved_at"] = datetime.now(UTC).isoformat()
        if inbox_item_id:
            account["inbox_item_id"] = inbox_item_id
        changed = True
    if not changed:
        return record
    try:
        await store.aput(
            discovered_accounts_namespace(creator_id, assistant_id),
            key=assistant_id,
            value=record,
        )
    except Exception as write_error:  # noqa: BLE001 - never fail the connection
        logger.warning(
            "Could not close the %s account question for %s: %s",
            provider,
            assistant_id,
            write_error,
        )
    return record


ACCOUNT_DISCOVERY_SOURCE_KIND = "account_discovery"
"""Inbox ``source_kind`` for an account question this system wrote for the owner."""

INBOX_BODY_CHARACTER_LIMIT = 4000


async def create_account_discovery_inbox_items(
    *,
    creator_id: str,
    assistant_id: str,
    subject_name: str,
    job_id: str,
    accounts: list[DiscoveredAccount],
    store: Any | None = None,
) -> list[str]:
    """Put one question per discovered account into the owner's agent inbox.

    The inbox is already a notify loop, and that is the whole reason the ask lives
    here rather than in a prompt section fired once. An open item is named in
    ``<INBOX_NOTIFICATIONS>`` on every turn until the owner answers, the avatar is
    already told to raise waiting items briefly and not to repeat them once heard,
    and ``resolve_inbox_notification`` already records a decision whenever the
    decision comes — tonight, or in three weeks.

    ``external_id`` is the provider name, so the repository's
    ``UNIQUE (assistant_id, source_kind, account_key, external_id)`` constraint
    means a second research run re-asks nothing.
    """
    from src.anubis.utils.inbox.repository import (
        DECISION_NOTIFY,
        STATE_PENDING_OWNER,
        get_inbox_repository,
    )

    repository = get_inbox_repository()
    if repository is None or not accounts:
        return []

    created_item_ids: list[str] = []
    asked_at = datetime.now(UTC)
    for account in accounts:
        body_text = (
            f"Researching {subject_name} turned up {account.evidence_url}.\n\n"
            f"If that {account.display_name} account belongs to {subject_name}, "
            f"connecting the account lets the avatar read {account.display_name} "
            f"directly instead of inferring {subject_name} from whatever the open "
            "web happens to publish. Connecting is a sign-in on the vendor's own "
            "page, and what the account reveals feeds the next round of research.\n\n"
            f"If that account is not {subject_name}, say so and the question "
            "closes for good."
        )[:INBOX_BODY_CHARACTER_LIMIT]
        try:
            item = await repository.create_item(
                {
                    "user_id": creator_id,
                    "assistant_id": assistant_id,
                    "source_kind": ACCOUNT_DISCOVERY_SOURCE_KIND,
                    "account_key": None,
                    "external_id": account.provider,
                    "external_thread_id": None,
                    "sender": subject_name or "Neural Nexus research",
                    "recipients": [],
                    "subject": (
                        f"Is this {account.display_name} account yours?"
                    ),
                    "body_text": body_text,
                    "received_at": asked_at,
                    "message_kind": ACCOUNT_DISCOVERY_SOURCE_KIND,
                    "decision": DECISION_NOTIFY,
                    "needs_owner_action": True,
                    "reason": (
                        f"Research found a {account.display_name} page for "
                        f"{subject_name}."
                    ),
                    "confidence": 1.0,
                    "confidence_detail": {
                        "provider": account.provider,
                        "evidence_url": account.evidence_url,
                        "job_id": job_id,
                        "supported_a_verified_fact": (
                            account.supported_a_verified_fact
                        ),
                    },
                    "state": STATE_PENDING_OWNER,
                }
            )
        except Exception as create_error:  # noqa: BLE001 - one item must not stop the rest
            logger.warning(
                "Could not raise the %s account question for %s: %s",
                account.provider,
                assistant_id,
                create_error,
            )
            continue
        item_id = str((item or {}).get("item_id") or "")
        if not item_id:
            continue
        created_item_ids.append(item_id)
        if store is not None:
            await _attach_inbox_item_id(
                store,
                creator_id=creator_id,
                assistant_id=assistant_id,
                provider=account.provider,
                inbox_item_id=item_id,
            )
    return created_item_ids


async def _attach_inbox_item_id(
    store: Any,
    *,
    creator_id: str,
    assistant_id: str,
    provider: str,
    inbox_item_id: str,
) -> None:
    """Remember which inbox item carries one candidate's question."""
    record = await read_discovered_accounts(
        store, creator_id=creator_id, assistant_id=assistant_id
    )
    if not record:
        return
    for account in record.get("accounts", []):
        if str(account.get("provider") or "") == provider:
            account["inbox_item_id"] = inbox_item_id
    try:
        await store.aput(
            discovered_accounts_namespace(creator_id, assistant_id),
            key=assistant_id,
            value=record,
        )
    except Exception as write_error:  # noqa: BLE001 - the question was still asked
        logger.debug("Could not attach the inbox item id: %s", write_error)


__all__ = [
    "ACCOUNT_DISCOVERY_SOURCE_KIND",
    "ACCOUNT_STATUS_CONNECTED",
    "ACCOUNT_STATUS_DECLINED",
    "ACCOUNT_STATUS_OPEN",
    "DISCOVERED_ACCOUNTS_NAMESPACE_CATEGORY",
    "DiscoveredAccount",
    "DiscoveredAccountsRecord",
    "candidate_accounts_from_research",
    "create_account_discovery_inbox_items",
    "discovered_accounts_namespace",
    "mark_discovered_account_resolved",
    "open_accounts",
    "read_discovered_accounts",
    "record_discovered_accounts",
]
