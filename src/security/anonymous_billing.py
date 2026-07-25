# src/security/anonymous_billing.py

"""Free-tier Stripe metering for anonymous (hashed-ip) visitors.

Anonymous users are identified by a hash of the client ip
(``x-forwarded-for``) and are ALWAYS the free tier — they can never subscribe
or trial. Historically they had no Stripe customer at all, so their meter
events were silently dropped and free anonymous usage was invisible to cost
analysis. This module gives each hashed ip a real Stripe customer with a $0
free-tier subscription — and reports that subscription's CURRENT BILLING PERIOD
back to the caller, because the anonymous visitor has no Auth0 record to cache
it in and without it the API would count usage over the calendar month while
Stripe (and the customer portal) count it over the subscription's cycle, which
begins at the visitor's first sighting. Two windows over the same events produce
two different totals, so the anonymous subscription's period travels with the
customer id. The customer is resolved lazily on first use:

1. Process-local TTL cache (hashed ip -> customer id) — hot path, no I/O.
2. ``anonymous_billing_customers`` Postgres row — survives restarts.
3. Stripe Customer Search on ``metadata.anonymous_hashed_ip`` — recovers the
   mapping when a previous create succeeded but the row write failed.
4. ``Customer.create`` + free-tier ``Subscription.create`` — first sighting.

Everything is FAIL-OPEN: a Stripe or database outage must never block
anonymous messaging — metering simply stays a no-op for that request (the
pre-existing behavior). The ``ANONYMOUS_BILLING_ENABLED`` environment variable
turns the whole path off (useful in development, and a guard against customer
fan-out from carrier-grade NAT ip churn).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass

from src.anubis.utils.billing.metering import (
    fetch_anonymous_stripe_customer_id,
    persist_anonymous_stripe_customer_id,
)
from src.anubis.utils.billing.subscription_lifecycle import subscription_period_bounds
from src.anubis.utils.billing.tiers import SubscriptionTier
from src.anubis.utils.context import GlobalContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnonymousBillingRecord:
    """One anonymous visitor's Stripe billing identity and usage window.

    ``current_period_start`` / ``current_period_end`` are epoch seconds copied
    from the visitor's $0 free-tier subscription. They are stamped into the
    anonymous user's ``app_metadata.subscription_status`` so the usage window
    the API counts against is the very window Stripe bills over — the same two
    fields the Stripe webhook caches for authenticated users, resolved the same
    way by ``resolve_usage_period_start_for_user``.
    """

    stripe_customer_id: str
    subscription_id: str | None = None
    subscription_status: str | None = None
    current_period_start: int | None = None
    current_period_end: int | None = None


# Process-local (hashed ip -> (monotonic time cached, billing record)). The TTL
# only bounds memory staleness against manual customer deletion in the Stripe
# dashboard and against the subscription rolling into its next billing period;
# the Postgres row is the durable record of the customer id.
_ANONYMOUS_CUSTOMER_CACHE_MAX_ENTRIES = 4096
_ANONYMOUS_CUSTOMER_CACHE_TTL_SECONDS = 3600.0
_anonymous_customer_cache: OrderedDict[
    str, tuple[float, AnonymousBillingRecord]
] = OrderedDict()

# Once-per-process warnings for silent fail-open paths (avoid log spam).
_warned_anonymous_billing_disabled = False
_warned_anonymous_billing_config_missing = False


def _cache_get(hashed_ip: str) -> AnonymousBillingRecord | None:
    cached_entry = _anonymous_customer_cache.get(hashed_ip)
    if cached_entry is None:
        return None
    cached_at, billing_record = cached_entry
    if time.monotonic() - cached_at >= _ANONYMOUS_CUSTOMER_CACHE_TTL_SECONDS:
        return None
    _anonymous_customer_cache.move_to_end(hashed_ip)
    return billing_record


def _cache_put(hashed_ip: str, billing_record: AnonymousBillingRecord) -> None:
    _anonymous_customer_cache[hashed_ip] = (time.monotonic(), billing_record)
    _anonymous_customer_cache.move_to_end(hashed_ip)
    while len(_anonymous_customer_cache) > _ANONYMOUS_CUSTOMER_CACHE_MAX_ENTRIES:
        _anonymous_customer_cache.popitem(last=False)


def invalidate_anonymous_billing_cache() -> None:
    """Drop every cached anonymous billing record (used by tests)."""
    _anonymous_customer_cache.clear()


def _billing_record_from_subscription(
    stripe_customer_id: str, subscription: dict | None
) -> AnonymousBillingRecord:
    """Build the record for one customer from its free-tier subscription.

    The period bounds come from the shared ``subscription_period_bounds``
    (items-first, top-level fallback for flexible billing mode), which is the
    same reading the Stripe webhook caches for authenticated users and the same
    one the customer portal displays.
    """
    period_start, period_end = (
        subscription_period_bounds(subscription) if subscription else (None, None)
    )
    return AnonymousBillingRecord(
        stripe_customer_id=stripe_customer_id,
        subscription_id=(subscription or {}).get("id"),
        subscription_status=(subscription or {}).get("status"),
        current_period_start=period_start,
        current_period_end=period_end,
    )


async def _fetch_current_subscription(
    stripe_client, stripe_customer_id: str
) -> dict | None:
    """Return the anonymous customer's most recent subscription, or ``None``.

    Reached only when the customer was recovered from Postgres or Customer
    Search (a create returns the subscription directly), and the result is
    cached with the record, so this costs one Stripe call per hashed ip per
    cache time-to-live rather than one per request. Runs on a worker thread
    because the Stripe SDK call is synchronous.
    """

    def _list_subscriptions() -> dict | None:
        subscriptions = stripe_client.Subscription.list(
            customer=stripe_customer_id, status="all", limit=1
        ).to_dict()
        subscription_data = subscriptions.get("data") or []
        return subscription_data[0] if subscription_data else None

    try:
        return await asyncio.to_thread(_list_subscriptions)
    except Exception as subscription_error:  # noqa: BLE001 - fail-open
        logger.warning(
            "Could not read the anonymous free-tier subscription for customer "
            "%s; the usage window falls back to the configured default: %s",
            stripe_customer_id,
            subscription_error,
        )
        return None


def _search_customer_by_hashed_ip(stripe_client, hashed_ip: str) -> str | None:
    """Recover a previously created customer via Stripe Customer Search.

    Customer Search is eventually consistent (up to about a minute), so a
    just-created customer can be missed — acceptable, because the worst case
    is one duplicate anonymous customer, both keyed to the same hashed ip.
    """
    try:
        search_result = stripe_client.Customer.search(
            query=f"metadata['anonymous_hashed_ip']:'{hashed_ip}'", limit=1
        ).to_dict()
        found_customers = search_result.get("data", [])
        if found_customers:
            return found_customers[0]["id"]
    except Exception as search_error:  # noqa: BLE001 - fail-open
        logger.warning(
            "Anonymous customer search failed for %s: %s", hashed_ip, search_error
        )
    return None


async def resolve_or_create_anonymous_billing_record(
    request, hashed_ip: str | None
) -> AnonymousBillingRecord | None:
    """Return the free-tier Stripe billing record for one hashed anonymous ip.

    Lazily creates the customer plus the $0 free-tier subscription on first
    sighting (never a trial — anonymous users use free-tier only), and reports
    that subscription's current billing period alongside the customer id so the
    caller can count usage over Stripe's window instead of the calendar month.
    Returns ``None`` whenever anything is unavailable (fail-open): billing
    disabled, billing objects unprovisioned, or Stripe/database errors.
    """
    global _warned_anonymous_billing_disabled, _warned_anonymous_billing_config_missing

    if not hashed_ip:
        return None
    context = GlobalContext()
    if str(context.anonymous_billing_enabled or "").upper() != "TRUE":
        if not _warned_anonymous_billing_disabled:
            logger.warning(
                "Anonymous billing is disabled (ANONYMOUS_BILLING_ENABLED != TRUE); "
                "anonymous usage will not be metered to Stripe."
            )
            _warned_anonymous_billing_disabled = True
        return None
    billing_config = getattr(request.app.state, "stripe_billing_config", None)
    stripe_client = getattr(request.app.state, "stripe", None)
    if billing_config is None or stripe_client is None:
        if not _warned_anonymous_billing_config_missing:
            logger.warning(
                "Anonymous billing cannot run (stripe_billing_config or stripe "
                "client missing on app.state); anonymous usage will not be metered "
                "to Stripe."
            )
            _warned_anonymous_billing_config_missing = True
        return None

    cached_billing_record = _cache_get(hashed_ip)
    if cached_billing_record is not None:
        return cached_billing_record

    pool = getattr(request.app.state, "pool", None)
    persisted_customer_id = await fetch_anonymous_stripe_customer_id(pool, hashed_ip)
    if persisted_customer_id:
        billing_record = _billing_record_from_subscription(
            persisted_customer_id,
            await _fetch_current_subscription(stripe_client, persisted_customer_id),
        )
        _cache_put(hashed_ip, billing_record)
        return billing_record

    try:
        customer_id = _search_customer_by_hashed_ip(stripe_client, hashed_ip)
        subscription: dict | None = None
        if customer_id is None:
            customer = stripe_client.Customer.create(
                description="Anonymous free-tier visitor",
                metadata={"anonymous_hashed_ip": hashed_ip},
            ).to_dict()
            customer_id = customer["id"]
            # The $0 free-tier subscription is the billing vehicle that makes
            # meter events for this visitor visible in Stripe cost analysis,
            # and its billing cycle is the window usage is counted over.
            # Imported here (not module scope) to avoid a circular import
            # with src.security.auth.
            from src.security.auth import create_free_tier_subscription

            subscription = create_free_tier_subscription(
                stripe_client,
                billing_config,
                customer_id,
                extra_metadata={
                    "anonymous_hashed_ip": hashed_ip,
                    "neural_nexus_tier": SubscriptionTier.FREE.value,
                },
            )
    except Exception as stripe_error:  # noqa: BLE001 - fail-open
        logger.error(
            "Could not create anonymous billing customer for %s: %s",
            hashed_ip,
            stripe_error,
        )
        return None

    if subscription is None:
        # The customer came from Customer Search, or the free-tier subscription
        # create failed and an earlier attempt may already have made one.
        subscription = await _fetch_current_subscription(stripe_client, customer_id)
    billing_record = _billing_record_from_subscription(customer_id, subscription)

    await persist_anonymous_stripe_customer_id(pool, hashed_ip, customer_id)
    _cache_put(hashed_ip, billing_record)
    return billing_record
