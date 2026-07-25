# src/anubis/utils/billing/stripe_usage.py

"""Read period usage from Stripe Billing Meters — the billing source of truth.

The same usage is recorded twice: once in the local ``api_metrics`` table
(written beside every Stripe meter report, and what allotment gating
historically read on its own) and once in Stripe's Billing Meter aggregation
(what the customer portal displays). Both writes are deliberately fail-open, so
whenever one lands and the other does not the two ledgers drift permanently and
never reconcile. Measured drift on one anonymous visitor's free-tier customer:
342,864 messaging tokens aggregated by Stripe against 100,909 recorded in
``api_metrics`` for the same window — the customer portal showed an exhausted
allotment while the messaging API still reported 159,507 tokens remaining and
kept answering.

Stripe is authoritative, so this module reads the aggregated value for one
customer and one meter straight from Stripe, and ``reconcile_period_usage``
combines that reading with the local sum by taking the LARGER of the two:

* Stripe is normally the larger value and therefore governs, which is what
  makes enforcement agree with the portal.
* The local sum acts as a floor covering Stripe's meter-ingestion lag (an event
  reported seconds ago may not be aggregated yet), so usage recorded locally in
  the last moments of a period can never buy a second free allotment.
* When Stripe cannot be read at all (outage, unprovisioned meter, no customer —
  every one of which yields ``None``), the local sum governs alone, which is
  exactly the pre-existing behavior.

Reads are cached per (customer, meter, period start) for
``STRIPE_USAGE_CACHE_TTL_SECONDS`` because allotment enforcement runs on the
message hot path; ``STRIPE_USAGE_SOURCE_OF_TRUTH_ENABLED=FALSE`` turns the
Stripe read off entirely and returns the ledger to local-only accounting.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from datetime import UTC, datetime
from typing import Any

from src.anubis.utils.billing.tiers import UsageMeter
from src.anubis.utils.context import GlobalContext

logger = logging.getLogger(__name__)

# (stripe customer id, meter event name, period start epoch) -> (cached at, usage)
_STRIPE_USAGE_CACHE_MAX_ENTRIES = 4096
_stripe_usage_cache: OrderedDict[tuple[str, str, int], tuple[float, int]] = (
    OrderedDict()
)

# Once-per-process warning so a disabled or unprovisioned Stripe usage read is
# visible in the logs without spamming every request.
_warned_stripe_usage_unavailable = False


def stripe_usage_source_of_truth_enabled(context: Any | None = None) -> bool:
    """Return whether period usage may be read from Stripe.

    Defaults to enabled: the whole point of reading Stripe is that the local
    table has already been observed to under-count, so the safe default is the
    authoritative source. Only an explicit ``FALSE`` disables it.
    """
    context = context or GlobalContext()
    configured_value = context.stripe_usage_source_of_truth_enabled
    return str(configured_value or "TRUE").strip().upper() != "FALSE"


def _stripe_usage_cache_ttl_seconds(context: Any | None = None) -> float:
    context = context or GlobalContext()
    return float(context.stripe_usage_cache_ttl_seconds or 0)


def _cache_get(cache_key: tuple[str, str, int], ttl_seconds: float) -> int | None:
    cached_entry = _stripe_usage_cache.get(cache_key)
    if cached_entry is None:
        return None
    cached_at, cached_usage = cached_entry
    if ttl_seconds <= 0 or time.monotonic() - cached_at >= ttl_seconds:
        return None
    _stripe_usage_cache.move_to_end(cache_key)
    return cached_usage


def _cache_put(cache_key: tuple[str, str, int], usage: int) -> None:
    _stripe_usage_cache[cache_key] = (time.monotonic(), usage)
    _stripe_usage_cache.move_to_end(cache_key)
    while len(_stripe_usage_cache) > _STRIPE_USAGE_CACHE_MAX_ENTRIES:
        _stripe_usage_cache.popitem(last=False)


def invalidate_stripe_usage_cache() -> None:
    """Drop every cached Stripe usage reading (used by tests and after a reset)."""
    _stripe_usage_cache.clear()


def _align_down_to_minute(epoch_seconds: int) -> int:
    """Both Stripe usage endpoints reject timestamps that are not minute-aligned."""
    return epoch_seconds - (epoch_seconds % 60)


def _sum_meter_event_summaries(
    stripe_client: Any,
    meter_id: str,
    stripe_customer_id: str,
    start_time: int,
    end_time: int,
) -> int:
    """Sum every event-summary bucket Stripe reports for one customer and meter.

    Runs on a worker thread (the synchronous SDK is the only interface that
    paginates event summaries) and is the same call the customer portal makes,
    so both sides read one number from one place.
    """
    total_usage = 0
    summaries = stripe_client.billing.Meter.list_event_summaries(
        meter_id,
        customer=stripe_customer_id,
        start_time=start_time,
        end_time=end_time,
        limit=100,
    )
    for summary_object in summaries.auto_paging_iter():
        summary_document = summary_object.to_dict()
        total_usage += int(summary_document.get("aggregated_value") or 0)
    return total_usage


async def fetch_stripe_period_usage(
    stripe_client: Any,
    billing_config: Any,
    meter: UsageMeter,
    stripe_customer_id: str | None,
    period_start: datetime,
    now: datetime | None = None,
) -> int | None:
    """Return Stripe's aggregated usage for one customer and meter this period.

    ``None`` means "Stripe could not answer" — the reading is disabled, there is
    no Stripe customer (an anonymous visitor while ``ANONYMOUS_BILLING_ENABLED``
    is off), the meter is not provisioned in ``STRIPE_BILLING_CONFIG_JSON``, or
    the call failed. Callers must treat ``None`` as "fall back to the local
    ledger" rather than as zero usage, which would hand out a free allotment on
    every Stripe hiccup.

    The window is ``[period_start, now]`` — usage to date, matching what the
    portal reports and what allotment gating compares against.
    """
    global _warned_stripe_usage_unavailable

    if not stripe_customer_id or stripe_client is None:
        return None
    context = GlobalContext()
    if not stripe_usage_source_of_truth_enabled(context):
        return None
    meter_id = (getattr(billing_config, "meter_ids", None) or {}).get(meter)
    if not meter_id:
        if not _warned_stripe_usage_unavailable:
            logger.warning(
                "Stripe usage cannot be read for meter %s: no meter id in the "
                "Stripe billing config; falling back to local api_metrics "
                "accounting.",
                meter.value,
            )
            _warned_stripe_usage_unavailable = True
        return None

    now = now or datetime.now(UTC)
    start_time = _align_down_to_minute(int(period_start.timestamp()))
    end_time = _align_down_to_minute(int(now.timestamp()))
    if end_time <= start_time:
        # The period began inside the current minute: nothing can be aggregated
        # yet, and Stripe rejects a non-positive window.
        return 0

    cache_key = (stripe_customer_id, meter.value, start_time)
    ttl_seconds = _stripe_usage_cache_ttl_seconds(context)
    cached_usage = _cache_get(cache_key, ttl_seconds)
    if cached_usage is not None:
        return cached_usage

    try:
        stripe_usage = await asyncio.to_thread(
            _sum_meter_event_summaries,
            stripe_client,
            meter_id,
            stripe_customer_id,
            start_time,
            end_time,
        )
    except Exception as usage_error:  # noqa: BLE001 - fall back to local ledger
        logger.error(
            "Could not read Stripe %s usage for customer %s since %s: %s",
            meter.value,
            stripe_customer_id,
            period_start,
            usage_error,
        )
        return None

    _cache_put(cache_key, stripe_usage)
    return stripe_usage


def reconcile_period_usage(local_usage: int, stripe_usage: int | None) -> int:
    """Return the authoritative usage to date from both ledgers.

    Stripe is the source of truth, and in practice reports the larger value
    because the local insert is the one that silently fails; the local sum is
    kept as a floor so usage Stripe has not finished aggregating still counts.
    Taking the maximum satisfies both at once, and degrades to the local sum
    when Stripe could not be read (``stripe_usage`` is ``None``).
    """
    local_usage = max(0, int(local_usage or 0))
    if stripe_usage is None:
        return local_usage
    return max(local_usage, max(0, int(stripe_usage)))
