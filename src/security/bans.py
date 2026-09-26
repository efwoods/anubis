"""Account bans for terms-of-service and privacy-policy violations.

The ``banned_accounts`` PostgreSQL table is the ONLY source of truth for a
ban. Auth0 is never written: the platform is decoupling from Auth0, and a ban
recorded there would have to be re-created against whatever identity provider
replaces Auth0. Every authentication path (API key, anonymous by hashed IP,
signup by email and IP) asks this module, so a banned person is refused by the
API, the Neural Nexus web application, and every other product that
authenticates through this API.

A ban records the user id, the hashed client IP, and the email address of the
account, so:

* the account's API key is refused (``user_id`` / ``email`` match);
* anonymous traffic from the same IP is refused (``hashed_ip`` match);
* a new signup with the same email or from the same IP is refused.

Banning also cancels the Stripe subscription and refunds the latest paid
invoice (``BAN_REFUND_ENABLED``). Lifting a ban (an appeal) stamps
``lifted_at``; the row stays as an audit record.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from src.anubis.utils.postgres_ddl import execute_ddl_script

logger = logging.getLogger(__name__)

BANNED_ACCOUNTS_TABLE_NAME = "banned_accounts"

_CREATE_BANNED_ACCOUNTS_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {BANNED_ACCOUNTS_TABLE_NAME} (
    ban_id TEXT PRIMARY KEY,
    user_id TEXT,
    hashed_ip TEXT,
    email TEXT,
    reason TEXT NOT NULL,
    violated_clauses TEXT,
    source TEXT NOT NULL,
    excerpt TEXT,
    stripe_customer_id TEXT,
    subscription_id TEXT,
    refund_id TEXT,
    banned_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    lifted_at TIMESTAMPTZ,
    appeal_note TEXT,
    enforced BOOLEAN NOT NULL DEFAULT TRUE,
    skipped_reason TEXT
);
ALTER TABLE {BANNED_ACCOUNTS_TABLE_NAME}
    ADD COLUMN IF NOT EXISTS enforced BOOLEAN NOT NULL DEFAULT TRUE;
ALTER TABLE {BANNED_ACCOUNTS_TABLE_NAME}
    ADD COLUMN IF NOT EXISTS skipped_reason TEXT;
CREATE INDEX IF NOT EXISTS banned_accounts_user_id_idx
    ON {BANNED_ACCOUNTS_TABLE_NAME} (user_id) WHERE lifted_at IS NULL;
CREATE INDEX IF NOT EXISTS banned_accounts_hashed_ip_idx
    ON {BANNED_ACCOUNTS_TABLE_NAME} (hashed_ip) WHERE lifted_at IS NULL;
CREATE INDEX IF NOT EXISTS banned_accounts_email_idx
    ON {BANNED_ACCOUNTS_TABLE_NAME} (lower(email)) WHERE lifted_at IS NULL;
"""

_BAN_COLUMNS = (
    "ban_id, user_id, hashed_ip, email, reason, violated_clauses, source, excerpt, "
    "stripe_customer_id, subscription_id, refund_id, banned_at, lifted_at, appeal_note, "
    "enforced, skipped_reason"
)

# Every parameter is cast to text explicitly. Postgres cannot infer the type of a
# bare parameter that appears only in ``$1 IS NOT NULL``, and refuses the whole
# statement with "could not determine data type of parameter $1" — which
# ``find_active_ban`` then swallows as a lookup failure and treats as "not banned".
# Without the casts no ban is ever matched and nobody is ever refused.
_FIND_ACTIVE_BAN_SQL = f"""
SELECT {_BAN_COLUMNS} FROM {BANNED_ACCOUNTS_TABLE_NAME}
WHERE lifted_at IS NULL
  AND enforced IS DISTINCT FROM FALSE
  AND (
        (%(user_id)s::text IS NOT NULL AND user_id = %(user_id)s::text)
     OR (%(hashed_ip)s::text IS NOT NULL AND hashed_ip = %(hashed_ip)s::text)
     OR (%(email)s::text IS NOT NULL AND lower(email) = lower(%(email)s::text))
  )
ORDER BY banned_at DESC
LIMIT 1;
"""

_INSERT_BAN_SQL = f"""
INSERT INTO {BANNED_ACCOUNTS_TABLE_NAME}
    (ban_id, user_id, hashed_ip, email, reason, violated_clauses, source, excerpt,
     stripe_customer_id, subscription_id, refund_id, enforced, skipped_reason)
VALUES (%(ban_id)s, %(user_id)s, %(hashed_ip)s, %(email)s, %(reason)s,
        %(violated_clauses)s, %(source)s, %(excerpt)s, %(stripe_customer_id)s,
        %(subscription_id)s, %(refund_id)s, %(enforced)s, %(skipped_reason)s);
"""

_LIFT_ADMINISTRATOR_BANS_SQL = f"""
UPDATE {BANNED_ACCOUNTS_TABLE_NAME}
SET lifted_at = now(),
    appeal_note = COALESCE(appeal_note, %(appeal_note)s)
WHERE lifted_at IS NULL
  AND (
        (%(user_id)s::text IS NOT NULL AND user_id = %(user_id)s::text)
     OR (%(email)s::text IS NOT NULL AND lower(email) = lower(%(email)s::text))
  )
RETURNING {_BAN_COLUMNS};
"""

_UPDATE_BAN_REFUND_SQL = f"""
UPDATE {BANNED_ACCOUNTS_TABLE_NAME}
SET subscription_id = %(subscription_id)s, refund_id = %(refund_id)s
WHERE ban_id = %(ban_id)s;
"""

_LIFT_BAN_SQL = f"""
UPDATE {BANNED_ACCOUNTS_TABLE_NAME}
SET lifted_at = now(), appeal_note = %(appeal_note)s
WHERE ban_id = %(ban_id)s AND lifted_at IS NULL
RETURNING {_BAN_COLUMNS};
"""

_LIST_BANS_SQL = f"""
SELECT {_BAN_COLUMNS} FROM {BANNED_ACCOUNTS_TABLE_NAME}
WHERE (%(include_lifted)s::boolean OR lifted_at IS NULL)
ORDER BY banned_at DESC
LIMIT %(limit)s;
"""

# Ban lookups run on every request, so the answer is cached briefly. Writes
# (ban, lift) clear the cache, so a fresh ban is enforced on the very next
# request even before the entry would have expired.
_BAN_CACHE_TTL_SECONDS = 60.0
_ban_cache: dict[
    tuple[str | None, str | None, str | None], tuple[float, dict | None]
] = {}
_ban_cache_lock = asyncio.Lock()


def _clear_ban_cache() -> None:
    _ban_cache.clear()


def shared_hashed_ip_values() -> set[str]:
    """Hashed client addresses that stand for MANY callers rather than one.

    ``resolve_request_hashed_ip`` does not always identify a single visitor:

    * in development every caller resolves to ``DEVELOPMENT_MODE_CLIENT_IP``, so
      one hash covers every local caller;
    * a caller that did not arrive through the proxy forwards no address at all.

    Recording a ban against one of these, or matching one against a ban, would ban
    every caller that shares it. They are therefore never written to a ban row and
    never matched, so a ban on such a caller falls back to the account identifiers
    (user id and email) alone.
    """
    from src.security.auth import DEVELOPMENT_MODE_CLIENT_IP, _hash_key

    return {_hash_key(DEVELOPMENT_MODE_CLIENT_IP)}


def usable_hashed_ip(hashed_ip: str | None) -> str | None:
    """``hashed_ip`` unless it stands for many callers, in which case ``None``."""
    if not hashed_ip:
        return None
    try:
        if hashed_ip in shared_hashed_ip_values():
            return None
    except Exception:  # noqa: BLE001 - a hashing failure must not block the ban path
        return hashed_ip
    return hashed_ip


def _row_to_ban(row: Any) -> dict[str, Any]:
    """Map one ``banned_accounts`` row onto the ban record the callers read.

    The column order is ``_BAN_COLUMNS``, which every statement selects in that
    order. Timestamps are rendered as ISO strings so a ban record can be returned
    from a route without a JSON encoder for datetimes.
    """
    keys = [column.strip() for column in _BAN_COLUMNS.split(",")]
    record = dict(zip(keys, row))
    for timestamp_key in ("banned_at", "lifted_at"):
        value = record.get(timestamp_key)
        if value is not None and hasattr(value, "isoformat"):
            record[timestamp_key] = value.isoformat()
    return record


DEFAULT_ADMIN_ACCOUNT_EMAIL = "e.woods.business@icloud.com"
MODERATION_INBOX_SOURCE_KIND = "moderation"
ACTION_REVOKE_BAN = "revoke_ban"
ACTION_ACCEPT_BAN = "accept_ban"
MODERATION_INBOX_BODY_CHARACTER_LIMIT = 4000


def administrator_account_email(context: Any | None) -> str:
    """The configured administrator email, or the product default."""
    configured = str(getattr(context, "admin_account_email", None) or "").strip()
    return configured or DEFAULT_ADMIN_ACCOUNT_EMAIL


def administrator_user_id(context: Any | None) -> str | None:
    """The configured administrator user id, or None when unset."""
    configured = str(getattr(context, "admin_user_id", None) or "").strip()
    return configured or None


def is_unbannable_administrator(
    *,
    user_id: str | None = None,
    email: str | None = None,
    context: Any | None = None,
) -> bool:
    """Return whether this identity is the unbannable administrator.

    Matches the bare Auth0 user id (``ADMIN_USER_ID``) or the administrator
    email (``ADMIN_ACCOUNT_EMAIL``, default ``e.woods.business@icloud.com``).
    Either match is enough, so a drifted identifier cannot leave the account
    bannable.
    """
    admin_id = administrator_user_id(context)
    if admin_id and user_id and str(user_id).strip() == admin_id:
        return True
    admin_email = administrator_account_email(context).casefold()
    if email and str(email).strip().casefold() == admin_email:
        return True
    return False


def is_ban_exempt_account(
    *,
    user_id: str | None = None,
    email: str | None = None,
    context: Any | None = None,
) -> bool:
    """Return whether a ban on this identity is recorded for audit only.

    Two kinds of account are never refused for a ban: the administrator
    (``is_unbannable_administrator``), and every account on
    ``UNRESTRICTED_METERED_ACCOUNT_IDENTIFIERS``, the accounts set aside for
    demonstrating and testing the product. A demonstration account is driven
    through every edge of the product on purpose, including a Minecraft body
    told to "kill" a zombie, and a false-positive ban there locks the
    demonstration out of every surface at once.

    Kept apart from ``is_unbannable_administrator`` because that function also
    grants administrator powers (appeal handling, the admin view of bans), and
    a demonstration account must not gain those.

    Both identifiers here come from an authenticated account or from the ban
    record written for one, so the email address is treated as verified, which
    is the condition ``is_unrestricted_metered_account`` places on an email
    entry.
    """
    if is_unbannable_administrator(user_id=user_id, email=email, context=context):
        return True
    configured_identifiers = getattr(
        context, "unrestricted_metered_account_identifiers", None
    )
    if not configured_identifiers:
        return False
    from src.anubis.utils.billing.gating import is_unrestricted_metered_account

    bare_user_id = str(user_id or "").split("|")[-1] or None
    account = {
        "user_id": f"auth0|{bare_user_id}" if bare_user_id else None,
        "identities": [{"user_id": bare_user_id}] if bare_user_id else [],
        "email": email,
        "email_verified": bool(email),
    }
    try:
        return is_unrestricted_metered_account(account, configured_identifiers)
    except Exception:  # noqa: BLE001 - an unreadable list exempts nobody
        logger.debug("Could not read the unrestricted account list", exc_info=True)
        return False


def supporting_evidence_quotes(verdict: Mapping[str, Any] | None) -> list[str]:
    """The verbatim content quotes the judge named as the bannable lines."""
    raw = (verdict or {}).get("supporting_evidence")
    if not isinstance(raw, (list, tuple)):
        excerpt = str((verdict or {}).get("excerpt") or "").strip()
        return [excerpt] if excerpt else []
    quotes: list[str] = []
    for candidate in raw:
        quote = str(candidate or "").strip()
        if quote and quote not in quotes:
            quotes.append(quote)
    return quotes


def verdict_is_actionable(
    verdict: Mapping[str, Any] | None, judged_text: str | None = None
) -> bool:
    """True when the judge named a reason and at least one quote from the content."""
    reasoning = str((verdict or {}).get("reasoning") or "").strip()
    quotes = supporting_evidence_quotes(verdict)
    if not reasoning or not quotes:
        return False
    if judged_text is None:
        return True
    haystack = judged_text.casefold()
    return any(quote.casefold() in haystack for quote in quotes)


def _evidence_excerpt(verdict: Mapping[str, Any] | None, fallback: str | None) -> str | None:
    quotes = supporting_evidence_quotes(verdict)
    if quotes:
        return "\n".join(quotes)[:2000]
    fallback_text = str(fallback or "").strip()
    return fallback_text[:2000] if fallback_text else None


@dataclass(frozen=True)
class BanSubject:
    """Everything a ban needs to identify and refund one account."""

    user_id: str | None
    hashed_ip: str | None
    email: str | None
    stripe_customer_id: str | None
    subscription_id: str | None
    is_anonymous: bool


def ban_subject_from_user(
    current_user: Mapping[str, Any] | None, hashed_ip: str | None
) -> BanSubject:
    """Build the ban subject for the authenticated or anonymous caller."""
    from src.anubis.utils.billing.gating import (
        is_anonymous_user,
        resolve_stripe_customer_id,
    )

    current_user = current_user or {}
    anonymous = is_anonymous_user(current_user)
    identities = current_user.get("identities") or [{}]
    bare_user_id = identities[0].get("user_id") if identities else None
    app_metadata = current_user.get("app_metadata") or {}
    subscription_status = app_metadata.get("subscription_status") or {}
    return BanSubject(
        user_id=None if anonymous else bare_user_id,
        # An anonymous visitor IS the hashed IP; an account also carries the IP
        # the violation came from so anonymous traffic from that IP is refused.
        hashed_ip=usable_hashed_ip(hashed_ip or (bare_user_id if anonymous else None)),
        email=None if anonymous else current_user.get("email"),
        stripe_customer_id=resolve_stripe_customer_id(current_user),
        subscription_id=subscription_status.get("subscription_id"),
        is_anonymous=anonymous,
    )


async def ensure_banned_accounts_table(pool: Any, context: Any | None = None) -> None:
    """Create the ``banned_accounts`` table if missing. Best effort at startup."""
    if pool is None:
        return
    try:
        # The script holds a CREATE TABLE, two ALTER TABLE, and three CREATE INDEX
        # statements, and the pool runs with prepare_threshold=0, so it must be
        # split before it is sent.
        await execute_ddl_script(pool, _CREATE_BANNED_ACCOUNTS_TABLE_SQL)
    except Exception as table_error:  # noqa: BLE001 - non-fatal at startup
        logger.error("Could not ensure banned_accounts table exists: %s", table_error)
        return
    await lift_enforced_bans_for_administrator(pool, context)


async def lift_enforced_bans_for_administrator(
    pool: Any, context: Any | None
) -> list[dict[str, Any]]:
    """Lift leftover enforced bans on the administrator so that account can sign in."""
    if pool is None:
        return []
    admin_id = administrator_user_id(context)
    admin_email = administrator_account_email(context)
    if not admin_id and not admin_email:
        return []
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    _LIFT_ADMINISTRATOR_BANS_SQL,
                    {
                        "user_id": admin_id,
                        "email": admin_email,
                        "appeal_note": "unbannable administrator; lifted on startup",
                    },
                )
                rows = await cursor.fetchall()
        _clear_ban_cache()
        return [_row_to_ban(row) for row in rows or []]
    except Exception as lift_error:  # noqa: BLE001 - leftover rows must not block boot
        logger.error("Could not lift leftover administrator bans: %s", lift_error)
        return []


async def find_active_ban(
    pool: Any,
    *,
    user_id: str | None = None,
    hashed_ip: str | None = None,
    email: str | None = None,
) -> dict[str, Any] | None:
    """The active ban matching any of the identifiers, or None (cached briefly)."""
    hashed_ip = usable_hashed_ip(hashed_ip)
    if pool is None or not (user_id or hashed_ip or email):
        return None
    cache_key = (user_id, hashed_ip, email)
    now = time.monotonic()
    async with _ban_cache_lock:
        cached = _ban_cache.get(cache_key)
        if cached is not None and now - cached[0] < _BAN_CACHE_TTL_SECONDS:
            return cached[1]
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    _FIND_ACTIVE_BAN_SQL,
                    {"user_id": user_id, "hashed_ip": hashed_ip, "email": email},
                )
                row = await cursor.fetchone()
    except Exception as lookup_error:  # noqa: BLE001 - fail open on a database hiccup
        logger.error("Ban lookup failed (allowing the request): %s", lookup_error)
        return None
    ban = _row_to_ban(row) if row else None
    async with _ban_cache_lock:
        _ban_cache[cache_key] = (now, ban)
    return ban


async def is_banned(
    pool: Any,
    *,
    user_id: str | None = None,
    hashed_ip: str | None = None,
    email: str | None = None,
) -> bool:
    return (
        await find_active_ban(pool, user_id=user_id, hashed_ip=hashed_ip, email=email)
        is not None
    )


def ban_refusal_detail(
    ban: Mapping[str, Any] | None, appeal_contact: str | None
) -> str:
    contact = appeal_contact or "contact@neuralnexus.site"
    reason = (ban or {}).get("reason") or "a violation of the terms of service"
    return (
        "This account is banned from Neural Nexus and every Afterlife Systems product "
        f"for {reason}. To appeal, contact {contact}."
    )


def _refund_and_cancel_blocking(
    stripe_client: Any, customer_id: str | None, subscription_id: str | None
) -> tuple[str | None, str | None]:
    """Cancel the customer's subscriptions and refund the latest paid invoice.

    Blocking Stripe calls; run in a worker thread. Returns
    ``(cancelled_subscription_id, refund_id)``. Every step is best effort.
    """
    cancelled_subscription_id: str | None = None
    refund_id: str | None = None
    if not customer_id:
        return None, None

    subscription_ids: list[str] = []
    if subscription_id:
        subscription_ids.append(subscription_id)
    try:
        live_subscriptions = stripe_client.Subscription.list(
            customer=customer_id, status="all", limit=20
        )
        for subscription in live_subscriptions.get("data", []):
            if subscription.get("status") in (
                "active",
                "trialing",
                "past_due",
                "unpaid",
            ):
                if subscription["id"] not in subscription_ids:
                    subscription_ids.append(subscription["id"])
    except Exception as list_error:  # noqa: BLE001
        logger.warning(
            "Could not list subscriptions for %s: %s", customer_id, list_error
        )

    for candidate_subscription_id in subscription_ids:
        try:
            stripe_client.Subscription.delete(candidate_subscription_id)
            cancelled_subscription_id = (
                cancelled_subscription_id or candidate_subscription_id
            )
        except Exception as cancel_error:  # noqa: BLE001
            logger.warning(
                "Could not cancel subscription %s: %s",
                candidate_subscription_id,
                cancel_error,
            )

    try:
        paid_invoices = stripe_client.Invoice.list(
            customer=customer_id, status="paid", limit=5
        )
        for invoice in paid_invoices.get("data", []):
            amount_paid = int(invoice.get("amount_paid") or 0)
            charge_id = invoice.get("charge")
            payment_intent_id = invoice.get("payment_intent")
            if amount_paid <= 0 or not (charge_id or payment_intent_id):
                continue
            refund_arguments: dict[str, Any] = {"reason": "requested_by_customer"}
            if charge_id:
                refund_arguments["charge"] = charge_id
            else:
                refund_arguments["payment_intent"] = payment_intent_id
            refund = stripe_client.Refund.create(**refund_arguments)
            refund_id = (
                refund.get("id")
                if isinstance(refund, Mapping)
                else getattr(refund, "id", None)
            )
            break
    except Exception as refund_error:  # noqa: BLE001
        logger.warning(
            "Could not refund the latest invoice for %s: %s", customer_id, refund_error
        )
    return cancelled_subscription_id, refund_id


async def ban_account(
    app_state: Any,
    subject: BanSubject,
    *,
    reason: str,
    violated_clauses: list[str] | None,
    source: str,
    excerpt: str | None,
) -> dict[str, Any] | None:
    """Ban ``subject``: record the ban, refund and cancel in Stripe, drop caches.

    Never raises. Returns the ban record (or None when nothing could be written).
    Idempotent: an already-banned subject gets the existing active ban back.
    """
    pool = getattr(app_state, "pool", None)
    if pool is None:
        logger.error("Cannot record a ban: no database pool on app state")
        return None
    existing = await find_active_ban(
        pool, user_id=subject.user_id, hashed_ip=subject.hashed_ip, email=subject.email
    )
    if existing is not None:
        return existing

    context = getattr(app_state, "context", None)
    unbannable = is_ban_exempt_account(
        user_id=subject.user_id, email=subject.email, context=context
    )
    enforced = not unbannable
    skipped_reason = None
    if unbannable:
        skipped_reason = (
            "unbannable administrator; recorded for audit only"
            if is_unbannable_administrator(
                user_id=subject.user_id, email=subject.email, context=context
            )
            else "unrestricted demonstration account; recorded for audit only"
        )

    ban_id = str(uuid.uuid4())
    clauses_text = "\n".join(violated_clauses or []) or None
    excerpt_text = (excerpt or "")[:2000] or None
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    _INSERT_BAN_SQL,
                    {
                        "ban_id": ban_id,
                        "user_id": subject.user_id,
                        "hashed_ip": subject.hashed_ip,
                        "email": subject.email,
                        "reason": reason,
                        "violated_clauses": clauses_text,
                        "source": source,
                        "excerpt": excerpt_text,
                        "stripe_customer_id": subject.stripe_customer_id,
                        "subscription_id": subject.subscription_id,
                        "refund_id": None,
                        "enforced": enforced,
                        "skipped_reason": skipped_reason,
                    },
                )
    except Exception as insert_error:  # noqa: BLE001
        logger.error("Could not record ban for %s: %s", subject, insert_error)
        return None
    _clear_ban_cache()

    refund_enabled = (
        str(getattr(context, "ban_refund_enabled", "TRUE") or "TRUE").upper() == "TRUE"
    )
    stripe_client = getattr(app_state, "stripe", None)
    cancelled_subscription_id: str | None = None
    refund_id: str | None = None
    if (
        enforced
        and refund_enabled
        and stripe_client is not None
        and subject.stripe_customer_id
        and not subject.is_anonymous
    ):
        try:
            cancelled_subscription_id, refund_id = await asyncio.to_thread(
                _refund_and_cancel_blocking,
                stripe_client,
                subject.stripe_customer_id,
                subject.subscription_id,
            )
            async with pool.connection() as connection:
                async with connection.cursor() as cursor:
                    await cursor.execute(
                        _UPDATE_BAN_REFUND_SQL,
                        {
                            "ban_id": ban_id,
                            "subscription_id": cancelled_subscription_id
                            or subject.subscription_id,
                            "refund_id": refund_id,
                        },
                    )
        except Exception as stripe_error:  # noqa: BLE001
            logger.error("Refund/cancel after ban %s failed: %s", ban_id, stripe_error)

    if subject.user_id and enforced:
        try:
            from src.security.auth import _evict_api_key_cache_for_user

            await _evict_api_key_cache_for_user(subject.user_id)
        except Exception as eviction_error:  # noqa: BLE001
            logger.warning(
                "Could not evict the API-key cache after ban: %s", eviction_error
            )

    logger.warning(
        "%s account (user=%s ip=%s email=%s source=%s): %s",
        "Recorded (not enforced) ban for" if unbannable else "Banned",
        subject.user_id,
        (subject.hashed_ip or "")[:8],
        subject.email,
        source,
        reason,
    )
    ban_record = {
        "ban_id": ban_id,
        "user_id": subject.user_id,
        "hashed_ip": subject.hashed_ip,
        "email": subject.email,
        "reason": reason,
        "violated_clauses": clauses_text,
        "source": source,
        "excerpt": excerpt_text,
        "stripe_customer_id": subject.stripe_customer_id,
        "subscription_id": cancelled_subscription_id or subject.subscription_id,
        "refund_id": refund_id,
        "lifted_at": None,
        "enforced": enforced,
        "skipped_reason": skipped_reason,
    }
    await notify_admin_personal_avatar_of_moderation(app_state, subject, ban_record)
    return ban_record


async def record_moderation_verdict(
    app_state: Any,
    subject: BanSubject,
    verdict: Mapping[str, Any],
    *,
    source: str,
    judged_text: str | None = None,
) -> dict[str, Any] | None:
    """Record a ban only when the judge named a reason and quoted evidence.

    Fast-screen refusals have no quotes from the content. Those wait here until
    the deep judge returns both fields; without them nothing is written.
    """
    if not verdict_is_actionable(verdict, judged_text):
        logger.info(
            "Moderation verdict is not actionable (no reason or quoted evidence); "
            "no ban recorded for %s",
            subject.email or subject.user_id,
        )
        return None
    return await ban_account(
        app_state,
        subject,
        reason=str(verdict.get("reasoning") or "").strip(),
        violated_clauses=list(verdict.get("violated_clauses") or []),
        source=source,
        excerpt=_evidence_excerpt(verdict, None),
    )


async def complete_and_record_moderation_verdict(
    app_state: Any,
    subject: BanSubject,
    verdict: Mapping[str, Any] | None,
    *,
    source: str,
    judged_text: str | None = None,
) -> dict[str, Any] | None:
    """Judge the text when the incoming verdict lacks quotes, then record it."""
    incoming = dict(verdict or {})
    if not verdict_is_actionable(incoming, judged_text) and (judged_text or "").strip():
        from src.anubis.utils.moderation.content_moderation import judge_text

        incoming = await judge_text(judged_text)
    return await record_moderation_verdict(
        app_state, subject, incoming, source=source, judged_text=judged_text
    )


async def notify_admin_personal_avatar_of_moderation(
    app_state: Any, subject: BanSubject, ban: Mapping[str, Any]
) -> None:
    """Put the verdict on the administrator's personal-avatar inbox only."""
    context = getattr(app_state, "context", None)
    pool = getattr(app_state, "pool", None)
    admin_id = administrator_user_id(context)
    if pool is None or not admin_id:
        return
    try:
        from src.anubis.utils.inbox import get_inbox_repository
        from src.anubis.utils.inbox.repository import (
            DECISION_NOTIFY,
            STATE_PENDING_OWNER,
        )
        from src.anubis.utils.personal_avatar import personal_avatar_id_for_owner

        assistant_id = await personal_avatar_id_for_owner(pool, admin_id)
        repository = get_inbox_repository()
        if not assistant_id or repository is None:
            return
        banned_email = subject.email or "anonymous"
        enforced = bool(ban.get("enforced", True))
        subject_line = (
            f"Verdict (not enforced — administrator): {banned_email}"
            if not enforced
            else f"Ban: {banned_email}"
        )
        evidence = str(ban.get("excerpt") or "").strip()
        clauses = str(ban.get("violated_clauses") or "").strip()
        body_parts = [part for part in (evidence, clauses and f"Clauses:\n{clauses}") if part]
        body_text = "\n\n".join(body_parts)[:MODERATION_INBOX_BODY_CHARACTER_LIMIT]
        await repository.create_item(
            {
                "user_id": admin_id,
                "assistant_id": assistant_id,
                "source_kind": MODERATION_INBOX_SOURCE_KIND,
                "account_key": None,
                "external_id": ban.get("ban_id"),
                "external_thread_id": None,
                "sender": banned_email,
                "recipients": [],
                "subject": subject_line,
                "body_text": body_text,
                "received_at": _now_utc(),
                "message_kind": MODERATION_INBOX_SOURCE_KIND,
                "decision": DECISION_NOTIFY,
                "needs_owner_action": True,
                "reason": str(ban.get("reason") or ""),
                "confidence": 1.0,
                "confidence_detail": {
                    "ban_id": ban.get("ban_id"),
                    "enforced": enforced,
                    "violated_clauses": str(ban.get("violated_clauses") or ""),
                    "supporting_evidence": evidence,
                    "source": ban.get("source"),
                    "banned_user_id": subject.user_id,
                    "banned_email": subject.email,
                },
                "state": STATE_PENDING_OWNER,
            }
        )
    except Exception as inbox_error:  # noqa: BLE001 - audit must not fail the ban path
        logger.error(
            "Could not write the moderation inbox item for ban %s: %s",
            ban.get("ban_id"),
            inbox_error,
        )


def _now_utc():
    from datetime import UTC, datetime

    return datetime.now(UTC)


async def lift_ban(
    pool: Any, ban_id: str, appeal_note: str | None
) -> dict[str, Any] | None:
    """Lift one active ban (an accepted appeal). Returns the lifted record or None."""
    if pool is None:
        return None
    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                _LIFT_BAN_SQL, {"ban_id": ban_id, "appeal_note": appeal_note}
            )
            row = await cursor.fetchone()
    _clear_ban_cache()
    return _row_to_ban(row) if row else None


async def list_bans(
    pool: Any, *, include_lifted: bool = False, limit: int = 200
) -> list[dict[str, Any]]:
    if pool is None:
        return []
    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                _LIST_BANS_SQL,
                {"include_lifted": include_lifted, "limit": max(1, min(limit, 1000))},
            )
            rows = await cursor.fetchall()
    return [_row_to_ban(row) for row in rows or []]


__all__ = [
    "ACTION_ACCEPT_BAN",
    "ACTION_REVOKE_BAN",
    "BANNED_ACCOUNTS_TABLE_NAME",
    "DEFAULT_ADMIN_ACCOUNT_EMAIL",
    "MODERATION_INBOX_SOURCE_KIND",
    "shared_hashed_ip_values",
    "usable_hashed_ip",
    "BanSubject",
    "administrator_account_email",
    "administrator_user_id",
    "ban_account",
    "ban_refusal_detail",
    "ban_subject_from_user",
    "ensure_banned_accounts_table",
    "find_active_ban",
    "is_banned",
    "is_unbannable_administrator",
    "lift_ban",
    "lift_enforced_bans_for_administrator",
    "list_bans",
    "notify_admin_personal_avatar_of_moderation",
    "complete_and_record_moderation_verdict",
    "record_moderation_verdict",
    "supporting_evidence_quotes",
    "verdict_is_actionable",
]
