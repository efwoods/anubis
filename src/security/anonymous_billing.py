# src/security/anonymous_billing.py

"""Free-tier Stripe metering for anonymous (hashed-ip) visitors.

Anonymous users are identified by a hash of the client ip
(``x-forwarded-for``) and are ALWAYS the free tier — they can never subscribe
or trial. Historically they had no Stripe customer at all, so their meter
events were silently dropped and free anonymous usage was invisible to cost
analysis. This module gives each hashed ip a real Stripe customer with a $0
free-tier subscription, lazily on first use:

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

import logging
import time
from collections import OrderedDict

from src.anubis.utils.billing.metering import (
    fetch_anonymous_stripe_customer_id,
    persist_anonymous_stripe_customer_id,
)
from src.anubis.utils.billing.tiers import SubscriptionTier
from src.anubis.utils.context import GlobalContext

logger = logging.getLogger(__name__)

# Process-local (hashed ip -> (monotonic time cached, customer id)). The TTL
# only bounds memory staleness against manual customer deletion in the Stripe
# dashboard; the Postgres row is the durable record.
_ANONYMOUS_CUSTOMER_CACHE_MAX_ENTRIES = 4096
_ANONYMOUS_CUSTOMER_CACHE_TTL_SECONDS = 3600.0
_anonymous_customer_cache: OrderedDict[str, tuple[float, str]] = OrderedDict()

# Once-per-process warnings for silent fail-open paths (avoid log spam).
_warned_anonymous_billing_disabled = False
_warned_anonymous_billing_config_missing = False


def _cache_get(hashed_ip: str) -> str | None:
    cached_entry = _anonymous_customer_cache.get(hashed_ip)
    if cached_entry is None:
        return None
    cached_at, customer_id = cached_entry
    if time.monotonic() - cached_at >= _ANONYMOUS_CUSTOMER_CACHE_TTL_SECONDS:
        return None
    _anonymous_customer_cache.move_to_end(hashed_ip)
    return customer_id


def _cache_put(hashed_ip: str, customer_id: str) -> None:
    _anonymous_customer_cache[hashed_ip] = (time.monotonic(), customer_id)
    _anonymous_customer_cache.move_to_end(hashed_ip)
    while len(_anonymous_customer_cache) > _ANONYMOUS_CUSTOMER_CACHE_MAX_ENTRIES:
        _anonymous_customer_cache.popitem(last=False)


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


async def resolve_or_create_anonymous_stripe_customer(
    request, hashed_ip: str | None
) -> str | None:
    """Return the free-tier Stripe customer id for one hashed anonymous ip.

    Lazily creates the customer plus the $0 free-tier subscription on first
    sighting (never a trial — anonymous users use free-tier only). Returns
    ``None`` whenever anything is unavailable (fail-open): billing disabled,
    billing objects unprovisioned, or Stripe/database errors.
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

    cached_customer_id = _cache_get(hashed_ip)
    if cached_customer_id:
        return cached_customer_id

    pool = getattr(request.app.state, "pool", None)
    persisted_customer_id = await fetch_anonymous_stripe_customer_id(pool, hashed_ip)
    if persisted_customer_id:
        _cache_put(hashed_ip, persisted_customer_id)
        return persisted_customer_id

    try:
        customer_id = _search_customer_by_hashed_ip(stripe_client, hashed_ip)
        if customer_id is None:
            customer = stripe_client.Customer.create(
                description="Anonymous free-tier visitor",
                metadata={"anonymous_hashed_ip": hashed_ip},
            ).to_dict()
            customer_id = customer["id"]
            # The $0 free-tier subscription is the billing vehicle that makes
            # meter events for this visitor visible in Stripe cost analysis.
            # Imported here (not module scope) to avoid a circular import
            # with src.security.auth.
            from src.security.auth import create_free_tier_subscription

            create_free_tier_subscription(
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

    await persist_anonymous_stripe_customer_id(pool, hashed_ip, customer_id)
    _cache_put(hashed_ip, customer_id)
    return customer_id
