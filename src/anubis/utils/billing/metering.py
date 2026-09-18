# src/anubis/utils/billing/metering.py

"""Report usage to Stripe Billing Meters, estimate upload cost, and persist metrics.

Every billed operation (a message, a media upload, an adapter-inference pass, an
adapter-training run) reports its consumption to one of the four Stripe Billing
Meters keyed on the customer's ``stripe_customer_id``. Stripe aggregates those
events per billing period and applies the tier's graduated price (included
allotment at zero cost, then pay-per-use overage), so the allotment resets every
month automatically and each dimension is budgeted independently.

Reporting is deliberately best-effort: a metering failure must never break a
user's message. Each helper swallows and logs Stripe errors and returns whether
the report succeeded, so callers can fire-and-forget.

This module also owns the ``api_metrics`` Postgres table described in ``CLAUDE.md``
— one row per billed operation capturing tokens, cost, latency, model, and
inference type — which feeds the Grafana token/cost panels and lets us reconcile
our own accounting against Stripe invoices.
"""

from __future__ import annotations

import asyncio
import calendar
import logging
import uuid
from datetime import datetime, timedelta, timezone, UTC
from typing import Any, Mapping, Sequence

from src.anubis.utils.billing.tiers import UsageMeter

logger = logging.getLogger(__name__)

# Deterministic global anchor for fixed-length usage periods (USAGE_PERIOD_DAYS > 0)
# when a user has no personal period anchor: every process computes the same window
# boundaries across restarts because they all count periods from this instant.
GLOBAL_USAGE_PERIOD_ANCHOR = datetime(2025, 1, 1, tzinfo=UTC)


def _coerce_to_utc(moment: datetime) -> datetime:
    """Interpret a naive datetime as UTC; convert an aware one to UTC."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def _monthly_boundary_for(year: int, month: int, anchor: datetime) -> datetime:
    """Return the anchor's monthly boundary within (year, month).

    The boundary keeps the anchor's day-of-month and time-of-day, clamping the
    day to the target month's length (an anchor on January 31 yields February 28
    or 29) — the same clamping Stripe applies to ``billing_cycle_anchor``.
    """
    last_day_of_month = calendar.monthrange(year, month)[1]
    return anchor.replace(
        year=year, month=month, day=min(anchor.day, last_day_of_month)
    )


def resolve_usage_period_start(
    now: datetime,
    usage_period_days: int,
    period_anchor: datetime | None = None,
) -> datetime:
    """Return the start of the usage period that contains ``now``.

    Two period shapes, selected by the ``USAGE_PERIOD_DAYS`` environment variable:

    * ``usage_period_days == 0`` (default) — calendar-month semantics. Without a
      per-user anchor the period starts on the first of the current UTC month
      (the historical behavior). With a per-user anchor (written on tier upgrade
      or first checkout) the period starts at the most recent monthly boundary
      on the anchor's day-of-month, and never earlier than the anchor itself, so
      an upgrade begins a fresh window at the instant of the upgrade.
    * ``usage_period_days > 0`` — fixed-length windows counted from the anchor
      (per-user anchor when present, otherwise ``GLOBAL_USAGE_PERIOD_ANCHOR``):
      ``anchor + floor((now - anchor) / days) * days``.
    """
    now = _coerce_to_utc(now)
    if period_anchor is not None:
        period_anchor = _coerce_to_utc(period_anchor)
        if period_anchor > now:
            # A future anchor should not happen; treat the anchor as the period
            # start rather than producing a window that has not begun.
            return period_anchor

    if usage_period_days > 0:
        anchor = period_anchor or GLOBAL_USAGE_PERIOD_ANCHOR
        period_length = timedelta(days=usage_period_days)
        elapsed_periods = (now - anchor) // period_length
        return anchor + elapsed_periods * period_length

    if period_anchor is None:
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    boundary_this_month = _monthly_boundary_for(now.year, now.month, period_anchor)
    if boundary_this_month <= now:
        period_start = boundary_this_month
    else:
        previous_month_year = now.year if now.month > 1 else now.year - 1
        previous_month = now.month - 1 if now.month > 1 else 12
        period_start = _monthly_boundary_for(
            previous_month_year, previous_month, period_anchor
        )
    # The first period begins at the anchor itself, never before.
    return max(period_start, period_anchor)


def resolve_usage_period_end(
    period_start: datetime,
    usage_period_days: int,
) -> datetime:
    """Return the exclusive end of the usage period beginning at ``period_start``.

    Companion to ``resolve_usage_period_start`` used by the subscription-status
    endpoint to display when the current allotment resets.
    """
    period_start = _coerce_to_utc(period_start)
    if usage_period_days > 0:
        return period_start + timedelta(days=usage_period_days)
    next_month_year = (
        period_start.year if period_start.month < 12 else period_start.year + 1
    )
    next_month = period_start.month + 1 if period_start.month < 12 else 1
    return _monthly_boundary_for(next_month_year, next_month, period_start)


async def report_meter_event(
    stripe_client: Any,
    meter: UsageMeter,
    stripe_customer_id: str | None,
    value: int,
    idempotency_identifier: str | None = None,
) -> bool:
    """Report ``value`` units of ``meter`` usage for one customer to Stripe.

    ``stripe_client`` is the configured Stripe module/client stored on
    ``app.state.stripe``. ``value`` is the integer number of units consumed
    (tokens for the token meters, a count of trained adapters for the training
    meter). Returns ``True`` when Stripe accepted the event.

    A missing ``stripe_customer_id`` or a non-positive ``value`` is a no-op that
    returns ``False`` — there is nothing meaningful to bill, and reporting a zero
    would still create noise on the meter.
    """
    if not stripe_customer_id:
        logger.warning(
            "Skipping %s meter report: no stripe_customer_id supplied.", meter.value
        )
        return False
    if value <= 0:
        return False

    payload: dict[str, str] = {
        "stripe_customer_id": stripe_customer_id,
        "value": str(int(value)),
    }
    create_kwargs: dict[str, Any] = {
        "event_name": meter.value,
        "payload": payload,
    }
    # A stable identifier makes the event idempotent so a retried request is not
    # double-counted; Stripe deduplicates meter events by identifier within a window.
    if idempotency_identifier:
        create_kwargs["identifier"] = idempotency_identifier

    try:
        meter_event_resource = stripe_client.billing.MeterEvent
        create_async = getattr(meter_event_resource, "create_async", None)
        if create_async is not None:
            await create_async(**create_kwargs)
        else:
            # Fall back to the synchronous SDK call on a worker thread so the event
            # loop is never blocked on network input/output.
            await asyncio.to_thread(meter_event_resource.create, **create_kwargs)
        return True
    except Exception as reporting_error:  # noqa: BLE001 - best-effort metering
        logger.error(
            "Failed to report %s meter event for customer %s: %s",
            meter.value,
            stripe_customer_id,
            reporting_error,
        )
        return False


def billable_tokens_from_metadata(
    response_metadata: Mapping[str, Any] | None,
) -> int:
    """Extract total billable tokens from a model ``response_metadata`` mapping.

    Accepts the ``ResponseMetadata``/``TokenUsage`` shape produced across the
    codebase (``model.py``, ``schema.py``, the media graph): a ``token_usage``
    sub-mapping with ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``.
    Prefers ``total_tokens`` when present, otherwise sums prompt and completion.
    """
    if not response_metadata:
        return 0
    token_usage = response_metadata.get("token_usage") or {}
    total = int(token_usage.get("total_tokens") or 0)
    if total > 0:
        return total
    prompt_tokens = int(token_usage.get("prompt_tokens") or 0)
    completion_tokens = int(token_usage.get("completion_tokens") or 0)
    return prompt_tokens + completion_tokens


API_METRICS_TABLE_NAME = "api_metrics"

_CREATE_API_METRICS_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {API_METRICS_TABLE_NAME} (
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_id TEXT,
    stripe_customer_id TEXT,
    assistant_id TEXT,
    thread_id TEXT,
    inference_type TEXT NOT NULL,
    model_name TEXT,
    prompt_tokens BIGINT NOT NULL DEFAULT 0,
    completion_tokens BIGINT NOT NULL DEFAULT 0,
    total_tokens BIGINT NOT NULL DEFAULT 0,
    cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    latency_ms DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    meter_event_name TEXT,
    media_job_id TEXT
);
"""

# ``media_job_id`` was added after the table shipped, so an existing deployment
# needs the column added rather than the table created. It names the media job
# whose processing made the call, which is what lets one upload's real token
# consumption be summed and reported to Stripe in place of the pre-upload
# estimate.
_ADD_API_METRICS_MEDIA_JOB_COLUMN_SQL = f"""
ALTER TABLE {API_METRICS_TABLE_NAME} ADD COLUMN IF NOT EXISTS media_job_id TEXT;
"""

_CREATE_API_METRICS_MEDIA_JOB_INDEX_SQL = f"""
CREATE INDEX IF NOT EXISTS api_metrics_media_job_idx
    ON {API_METRICS_TABLE_NAME} (media_job_id);
"""

_INSERT_API_METRICS_SQL = f"""
INSERT INTO {API_METRICS_TABLE_NAME}
    (id, user_id, stripe_customer_id, assistant_id, thread_id, inference_type,
     model_name, prompt_tokens, completion_tokens, total_tokens, cost_usd,
     latency_ms, meter_event_name, media_job_id)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

# What one media job actually consumed, summed from the rows its own calls wrote.
_SUM_API_METRICS_FOR_MEDIA_JOBS_SQL = f"""
SELECT COALESCE(SUM(prompt_tokens), 0),
       COALESCE(SUM(completion_tokens), 0),
       COALESCE(SUM(total_tokens), 0),
       COALESCE(SUM(cost_usd), 0.0),
       COUNT(*)
FROM {API_METRICS_TABLE_NAME}
WHERE media_job_id = ANY(%s);
"""


async def ensure_api_metrics_table(pool: Any) -> None:
    """Create the ``api_metrics`` table if it does not yet exist.

    Called once from the FastAPI lifespan startup. Best-effort: a failure here
    (for example, a read-replica connection) must not prevent the app from
    serving, so it logs and returns rather than raising.
    """
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(_CREATE_API_METRICS_TABLE_SQL)
                await cursor.execute(_ADD_API_METRICS_MEDIA_JOB_COLUMN_SQL)
                await cursor.execute(_CREATE_API_METRICS_MEDIA_JOB_INDEX_SQL)
    except Exception as table_error:  # noqa: BLE001 - non-fatal at startup
        logger.error("Could not ensure api_metrics table exists: %s", table_error)


ANONYMOUS_BILLING_CUSTOMERS_TABLE_NAME = "anonymous_billing_customers"

# Anonymous users have no Auth0 record to cache a Stripe customer id in, so
# the (hashed ip -> customer id) mapping lives in Postgres: every anonymous
# visitor's usage lands on a real free-tier Stripe customer for cost analysis.
_CREATE_ANONYMOUS_BILLING_CUSTOMERS_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {ANONYMOUS_BILLING_CUSTOMERS_TABLE_NAME} (
    hashed_ip TEXT PRIMARY KEY,
    stripe_customer_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

_FETCH_ANONYMOUS_BILLING_CUSTOMER_SQL = f"""
SELECT stripe_customer_id FROM {ANONYMOUS_BILLING_CUSTOMERS_TABLE_NAME}
WHERE hashed_ip = %s;
"""

_PERSIST_ANONYMOUS_BILLING_CUSTOMER_SQL = f"""
INSERT INTO {ANONYMOUS_BILLING_CUSTOMERS_TABLE_NAME} (hashed_ip, stripe_customer_id)
VALUES (%s, %s)
ON CONFLICT (hashed_ip) DO UPDATE SET stripe_customer_id = EXCLUDED.stripe_customer_id;
"""


async def ensure_anonymous_billing_customers_table(pool: Any) -> None:
    """Create the ``anonymous_billing_customers`` table if it does not yet exist.

    Called once from the FastAPI lifespan startup, beside
    ``ensure_api_metrics_table``. Best-effort: logs and returns on failure.
    """
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(_CREATE_ANONYMOUS_BILLING_CUSTOMERS_TABLE_SQL)
    except Exception as table_error:  # noqa: BLE001 - non-fatal at startup
        logger.error(
            "Could not ensure anonymous_billing_customers table exists: %s",
            table_error,
        )


async def fetch_anonymous_stripe_customer_id(
    pool: Any, hashed_ip: str | None
) -> str | None:
    """Return the Stripe customer id recorded for one hashed anonymous ip.

    Fail-open: any database problem returns ``None`` (the caller may create a
    duplicate customer in the worst case, recovered by Customer Search).
    """
    if pool is None or not hashed_ip:
        return None
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    _FETCH_ANONYMOUS_BILLING_CUSTOMER_SQL, (hashed_ip,)
                )
                row = await cursor.fetchone()
                return row[0] if row else None
    except Exception as fetch_error:  # noqa: BLE001 - fail-open
        logger.error(
            "Could not fetch anonymous billing customer for %s: %s",
            hashed_ip,
            fetch_error,
        )
        return None


async def persist_anonymous_stripe_customer_id(
    pool: Any, hashed_ip: str | None, stripe_customer_id: str | None
) -> bool:
    """Record the (hashed ip -> Stripe customer) mapping; fail-open on error."""
    if pool is None or not hashed_ip or not stripe_customer_id:
        return False
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    _PERSIST_ANONYMOUS_BILLING_CUSTOMER_SQL,
                    (hashed_ip, stripe_customer_id),
                )
        return True
    except Exception as persist_error:  # noqa: BLE001 - fail-open
        logger.error(
            "Could not persist anonymous billing customer for %s: %s",
            hashed_ip,
            persist_error,
        )
        return False


_USAGE_SINCE_SQL = f"""
SELECT COALESCE(SUM(total_tokens), 0)
FROM {API_METRICS_TABLE_NAME}
WHERE user_id = %s
  AND meter_event_name = %s
  AND created_at >= %s;
"""

# Customer-keyed variant. The Stripe customer id is the DURABLE billing
# identity: a delete-and-re-signup mints a fresh Auth0 subject (the ``user_id``
# above) but deliberately reattaches the same Stripe customer (delete_user keeps
# it to block free-trial re-harvesting). Aggregating paid users by customer id
# therefore carries usage across the churn, closing the allotment-reset hole,
# while anonymous/free users with no customer still fall back to ``user_id``.
_USAGE_SINCE_BY_CUSTOMER_SQL = f"""
SELECT COALESCE(SUM(total_tokens), 0)
FROM {API_METRICS_TABLE_NAME}
WHERE stripe_customer_id = %s
  AND meter_event_name = %s
  AND created_at >= %s;
"""

_USAGE_BY_METER_SINCE_SQL = f"""
SELECT meter_event_name, COALESCE(SUM(total_tokens), 0)
FROM {API_METRICS_TABLE_NAME}
WHERE user_id = %s
  AND created_at >= %s
  AND meter_event_name IS NOT NULL
GROUP BY meter_event_name;
"""

_USAGE_BY_METER_SINCE_BY_CUSTOMER_SQL = f"""
SELECT meter_event_name, COALESCE(SUM(total_tokens), 0)
FROM {API_METRICS_TABLE_NAME}
WHERE stripe_customer_id = %s
  AND created_at >= %s
  AND meter_event_name IS NOT NULL
GROUP BY meter_event_name;
"""


async def fetch_usage_since(
    pool: Any,
    user_id: str | None,
    meter_event_name: str,
    period_start: datetime,
    stripe_customer_id: str | None = None,
) -> int:
    """Return the user's usage for one meter since ``period_start``, from ``api_metrics``.

    This is the enforcement-side counterpart to Stripe's meter aggregation:
    allotment gating reads the locally persisted usage instead of calling Stripe
    on the hot path. When ``stripe_customer_id`` is supplied the sum is keyed on
    that durable billing identity (which survives a delete-and-re-signup, unlike
    the Auth0 ``user_id``); otherwise it keys on ``user_id``, the identifier that
    also covers anonymous users (hashed-IP). ``period_start`` comes from
    ``resolve_usage_period_start`` (or the cached Stripe billing period for paid
    tiers), so the window follows the ``USAGE_PERIOD_DAYS`` configuration and
    per-user upgrade anchors.

    Best-effort and fail-open: any database error returns zero so a metrics outage
    degrades to "not gated" rather than blocking every message.
    """
    if pool is None:
        return 0
    if stripe_customer_id:
        sql = _USAGE_SINCE_BY_CUSTOMER_SQL
        params: tuple[Any, ...] = (stripe_customer_id, meter_event_name, period_start)
    elif user_id:
        sql = _USAGE_SINCE_SQL
        params = (user_id, meter_event_name, period_start)
    else:
        return 0
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(sql, params)
                row = await cursor.fetchone()
                return int(row[0]) if row and row[0] is not None else 0
    except Exception as usage_error:  # noqa: BLE001 - fail-open gating
        logger.error(
            "Could not read usage since %s for user %s meter %s: %s",
            period_start,
            user_id,
            meter_event_name,
            usage_error,
        )
        return 0


async def fetch_usage_by_meter_since(
    pool: Any,
    user_id: str | None,
    period_start: datetime,
    stripe_customer_id: str | None = None,
) -> dict[str, int]:
    """Return the user's usage since ``period_start`` for every meter at once.

    One grouped query backing the subscription-status endpoint, which displays
    used-versus-allotment for all four meters; meters with no usage in the window
    are simply absent from the mapping. Keys on ``stripe_customer_id`` when
    supplied (durable across re-signup) and otherwise on ``user_id``, matching
    ``fetch_usage_since``. Fail-open like ``fetch_usage_since``.
    """
    if pool is None:
        return {}
    if stripe_customer_id:
        sql = _USAGE_BY_METER_SINCE_BY_CUSTOMER_SQL
        params: tuple[Any, ...] = (stripe_customer_id, period_start)
    elif user_id:
        sql = _USAGE_BY_METER_SINCE_SQL
        params = (user_id, period_start)
    else:
        return {}
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(sql, params)
                rows = await cursor.fetchall()
                return {
                    str(meter_event_name): int(total)
                    for meter_event_name, total in rows
                    if meter_event_name is not None and total is not None
                }
    except Exception as usage_error:  # noqa: BLE001 - fail-open display
        logger.error(
            "Could not read per-meter usage since %s for user %s: %s",
            period_start,
            user_id,
            usage_error,
        )
        return {}


_ROLLING_WINDOW_USAGE_SQL = f"""
SELECT COALESCE(SUM(total_tokens), 0), MIN(created_at)
FROM {API_METRICS_TABLE_NAME}
WHERE user_id = %s
  AND created_at >= now() - make_interval(secs => %s);
"""

_ROLLING_WINDOW_USAGE_FILTERED_SQL = f"""
SELECT COALESCE(SUM(total_tokens), 0), MIN(created_at)
FROM {API_METRICS_TABLE_NAME}
WHERE user_id = %s
  AND created_at >= now() - make_interval(secs => %s)
  AND meter_event_name = ANY(%s);
"""


async def fetch_rolling_window_usage(
    pool: Any,
    user_id: str | None,
    window_seconds: int,
    meter_event_names: Sequence[str] | None = None,
) -> tuple[int, datetime | None]:
    """Return ``(total tokens, oldest usage timestamp)`` inside a rolling window.

    Backs the per-period token rate limit: unlike the monthly allotment (which is
    a billing budget), the rate limit is an abuse guard that caps how fast tokens
    can be consumed regardless of tier or pay-per-use, so a runaway client cannot
    burn an entire month's budget (or an unbounded overage bill) in minutes.
    ``meter_event_names`` narrows the sum to specific meters so message traffic
    and media-upload traffic are limited independently; ``None`` sums every meter.
    The oldest timestamp lets the caller compute a Retry-After — the limit clears
    when that row ages out of the window.

    Best-effort and fail-open like ``fetch_usage_since``: a database error
    returns zero usage so a metrics outage degrades to "not rate limited" rather
    than refusing every request.
    """
    if pool is None or not user_id or window_seconds <= 0:
        return 0, None
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                if meter_event_names:
                    await cursor.execute(
                        _ROLLING_WINDOW_USAGE_FILTERED_SQL,
                        (user_id, int(window_seconds), list(meter_event_names)),
                    )
                else:
                    await cursor.execute(
                        _ROLLING_WINDOW_USAGE_SQL, (user_id, int(window_seconds))
                    )
                row = await cursor.fetchone()
                if not row:
                    return 0, None
                total = int(row[0]) if row[0] is not None else 0
                oldest_usage_at = row[1]
                return total, oldest_usage_at
    except Exception as usage_error:  # noqa: BLE001 - fail-open rate limiting
        logger.error(
            "Could not read rolling-window usage for user %s: %s",
            user_id,
            usage_error,
        )
        return 0, None


def token_rate_limit_retry_after_seconds(
    window_usage: int,
    tokens_per_window: int,
    window_seconds: int,
    oldest_usage_at: datetime | None,
    now: datetime | None = None,
) -> int | None:
    """Return ``None`` when the request is allowed, else a Retry-After in seconds.

    Pure decision logic for the token rate limit (the endpoint helper turns a
    non-``None`` result into HTTP 429). A ``tokens_per_window`` of zero or less
    disables the limit entirely. When the window usage has already reached the
    cap, the wait is the time until the oldest contributing row ages out of the
    rolling window, clamped to ``[1, window_seconds]`` so the client always
    receives a sane, bounded hint even if timestamps are missing or skewed.
    """
    if tokens_per_window <= 0 or window_seconds <= 0:
        return None
    if window_usage < tokens_per_window:
        return None
    if oldest_usage_at is None:
        return window_seconds
    now = _coerce_to_utc(now or datetime.now(UTC))
    oldest_usage_at = _coerce_to_utc(oldest_usage_at)
    seconds_until_oldest_expires = (
        oldest_usage_at + timedelta(seconds=window_seconds) - now
    ).total_seconds()
    return max(1, min(window_seconds, int(seconds_until_oldest_expires) + 1))


async def persist_api_metrics_row(
    pool: Any,
    inference_type: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    cost_usd: float = 0.0,
    latency_ms: float = 0.0,
    user_id: str | None = None,
    stripe_customer_id: str | None = None,
    assistant_id: str | None = None,
    thread_id: str | None = None,
    model_name: str | None = None,
    meter_event_name: str | None = None,
    media_job_id: str | None = None,
) -> bool:
    """Insert one row into ``api_metrics`` describing a single billed operation.

    Best-effort persistence for observability and invoice reconciliation; returns
    whether the row was written and never raises into the request path.

    A ``False`` return is the drift signal: the Stripe meter event for the same
    operation has already been accepted, so a lost row leaves the local ledger
    permanently behind Stripe's aggregation (observed as 241,955 messaging tokens
    of Stripe-only usage on one anonymous customer). Callers should log the
    mismatch, and usage reads treat Stripe as authoritative precisely because
    this write is the one that can vanish.
    """
    if pool is None:
        # Graph-side callers (the media pipeline, the speech helpers, the model
        # callback) cannot import the web application to reach ``app.state.pool``,
        # so they pass no pool and the shared one the lifespan published is used.
        from src.anubis.utils.runtime_handles import get_postgres_pool

        pool = get_postgres_pool()
    if pool is None:
        logger.warning(
            "No database pool available to record an api_metrics row for a %s "
            "operation; local usage accounting will lag the Stripe meter for "
            "this request.",
            inference_type,
        )
        return False
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    _INSERT_API_METRICS_SQL,
                    (
                        str(uuid.uuid4()),
                        user_id,
                        stripe_customer_id,
                        assistant_id,
                        thread_id,
                        inference_type,
                        model_name,
                        int(prompt_tokens),
                        int(completion_tokens),
                        int(total_tokens),
                        float(cost_usd),
                        float(latency_ms),
                        meter_event_name,
                        media_job_id,
                    ),
                )
        return True
    except Exception as insert_error:  # noqa: BLE001 - non-fatal metering
        logger.error("Could not persist api_metrics row: %s", insert_error)
        return False


# Which ``api_metrics.inference_type`` a media-processing model call is recorded
# as, chosen by the graph node that made the call. A node with no entry here is
# recorded under its own name, so a new analysis node shows up in the ledger as
# itself rather than disappearing into a catch-all.
MEDIA_PROCESSING_INFERENCE_TYPES = {
    "convert_media_list_to_text_document": "media_conversion",
    "index_docs": "document_indexing",
    "deep_judge": "moderation",
    "fast_screen": "moderation",
    "process_adapter_documents": "adapter_dataset",
}


def media_processing_inference_type(node_name: str | None) -> str:
    """Return the ``inference_type`` for a model call made by a media graph node."""
    node = str(node_name or "").strip()
    if not node:
        return "media_processing"
    if node in MEDIA_PROCESSING_INFERENCE_TYPES:
        return MEDIA_PROCESSING_INFERENCE_TYPES[node]
    if node.startswith("analyze_"):
        return "psychological_analysis"
    return node


def build_api_metrics_callback_handler() -> Any:
    """Return a LangChain callback handler that records model calls in ``api_metrics``.

    This is the one place every model call in the application passes through
    (``init_model`` attaches the handler to each model it builds), which is what
    makes the recorded cost of a media upload match the vendor's invoice. Before
    it, an upload's model calls — classification, dialogue segmentation,
    first-person rewriting, characteristic extraction, the twelve
    psycho-analysis dimensions, the moderation judgement — were paid for and
    recorded nowhere: on 2026-09-17 one account's uploads cost about $9 at the
    vendors while ``api_metrics`` recorded $0.38.

    A call is recorded when its run carries ``media_job_id``, which the media job
    stamps on the graph's configurable. A chat reply is deliberately left alone:
    the message endpoint already writes that row together with its Stripe meter
    event, and recording the reply twice would overstate every per-message cost
    the billing portal and the unit-economics tools report.

    The handler is built lazily because importing ``langchain_core`` at module
    scope costs seconds of cold start (see the import rule in ``CLAUDE.md``).
    """
    from langchain_core.callbacks.base import AsyncCallbackHandler

    class ApiMetricsCallbackHandler(AsyncCallbackHandler):
        """Write one ``api_metrics`` row per model call made for a media job."""

        raise_error = False

        def __init__(self) -> None:
            super().__init__()
            # Run metadata arrives on the start event and the token usage on the
            # end event, so the start event's metadata is held by run id until
            # its end event arrives.
            self._metadata_by_run: dict[str, dict[str, Any]] = {}

        async def on_chat_model_start(
            self, serialized: Any, messages: Any, **kwargs: Any
        ) -> None:
            self._remember_run(kwargs)

        async def on_llm_start(
            self, serialized: Any, prompts: Any, **kwargs: Any
        ) -> None:
            self._remember_run(kwargs)

        def _remember_run(self, kwargs: dict[str, Any]) -> None:
            run_id = str(kwargs.get("run_id") or "")
            if not run_id:
                return
            metadata = kwargs.get("metadata") or {}
            if metadata.get("media_job_id"):
                self._metadata_by_run[run_id] = dict(metadata)

        async def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
            self._metadata_by_run.pop(str(kwargs.get("run_id") or ""), None)

        async def on_llm_end(self, response: Any, **kwargs: Any) -> None:
            metadata = self._metadata_by_run.pop(str(kwargs.get("run_id") or ""), None)
            if not metadata:
                return
            try:
                from src.anubis.utils.context import GlobalContext

                context = GlobalContext()
                for reading in _model_call_token_readings(response):
                    cost_usd = price_model_token_usage(
                        reading["model_name"],
                        context,
                        prompt_tokens=reading["prompt_tokens"],
                        completion_tokens=reading["completion_tokens"],
                        cached_prompt_tokens=reading["cached_prompt_tokens"],
                        cache_write_tokens=reading["cache_write_tokens"],
                    )
                    await persist_api_metrics_row(
                        None,
                        inference_type=media_processing_inference_type(
                            metadata.get("langgraph_node")
                        ),
                        prompt_tokens=reading["prompt_tokens"],
                        completion_tokens=reading["completion_tokens"],
                        total_tokens=reading["prompt_tokens"]
                        + reading["completion_tokens"],
                        cost_usd=cost_usd,
                        user_id=metadata.get("user_id"),
                        assistant_id=metadata.get("assistant_id"),
                        model_name=reading["model_name"],
                        media_job_id=str(metadata.get("media_job_id")),
                    )
            except Exception as recording_error:  # noqa: BLE001 - never break a run
                logger.warning(
                    "Could not record a media-processing model call: %s",
                    recording_error,
                )

    return ApiMetricsCallbackHandler()


def _model_call_token_readings(result: Any) -> list[dict[str, Any]]:
    """Pull per-generation token counts and the model name out of an LLMResult.

    LangChain reports usage in two places and neither is universal:
    ``usage_metadata`` on the generation's message (provider-independent, and the
    only place the cached-token detail appears) and ``llm_output["token_usage"]``
    (only some OpenAI client paths). The per-generation reading is preferred and
    the aggregate is the fallback.
    """
    llm_output = getattr(result, "llm_output", None) or {}
    fallback_model = llm_output.get("model_name") or llm_output.get("model")
    readings: list[dict[str, Any]] = []
    for generation_list in getattr(result, "generations", None) or []:
        for generation in generation_list or []:
            message = getattr(generation, "message", None)
            usage_metadata = getattr(message, "usage_metadata", None) or {}
            if not usage_metadata:
                continue
            input_details = usage_metadata.get("input_token_details") or {}
            response_metadata = getattr(message, "response_metadata", None) or {}
            readings.append(
                {
                    "model_name": response_metadata.get("model_name")
                    or response_metadata.get("model")
                    or fallback_model,
                    "prompt_tokens": int(usage_metadata.get("input_tokens") or 0),
                    "completion_tokens": int(usage_metadata.get("output_tokens") or 0),
                    "cached_prompt_tokens": int(input_details.get("cache_read") or 0),
                    "cache_write_tokens": int(input_details.get("cache_creation") or 0),
                }
            )
    if readings:
        return readings
    aggregate_usage = llm_output.get("token_usage") or {}
    if not aggregate_usage:
        return []
    cached_tokens = (aggregate_usage.get("prompt_tokens_details") or {}).get(
        "cached_tokens"
    ) or 0
    return [
        {
            "model_name": fallback_model,
            "prompt_tokens": int(aggregate_usage.get("prompt_tokens") or 0),
            "completion_tokens": int(aggregate_usage.get("completion_tokens") or 0),
            "cached_prompt_tokens": int(cached_tokens),
            "cache_write_tokens": 0,
        }
    ]


def resolve_model_token_prices(
    model_name: str | None, context: Any
) -> tuple[float, float, float, float]:
    """Return ``(prompt, completion, cached prompt, cache write)`` prices per token.

    Prices come from ``GlobalContext`` — never a hard-coded rate — so a vendor
    price change is an env edit. The comparison is case-insensitive and
    prefix-based because a provider reports a dated model name
    (``gpt-5.4-nano-2026-03-17``) for a model configured without the date. An
    unrecognized model name falls back to the inference model's prices, which
    keeps an unknown model's spend visible: a wrong rate is recoverable
    arithmetic, a silent zero is not.
    """
    requested = str(model_name or "").casefold()

    def names_match(configured: Any) -> bool:
        configured_name = str(configured or "").casefold()
        if not configured_name or not requested:
            return False
        return requested.startswith(configured_name) or configured_name.startswith(
            requested
        )

    def price(field_name: str) -> float:
        return float(getattr(context, field_name, 0.0) or 0.0)

    if names_match(getattr(context, "image_model", None)):
        return (
            price("image_model_prompt_cost"),
            price("image_model_completion_cost"),
            price("image_model_cached_prompt_cost"),
            0.0,
        )
    if names_match(getattr(context, "classification_model", None)):
        return (
            price("classification_model_prompt_cost"),
            price("classification_model_completion_cost"),
            price("classification_model_cached_prompt_cost"),
            price("classification_model_cache_write_cost"),
        )
    if names_match(getattr(context, "llama_model", None)):
        return (
            price("llama_model_prompt_cost"),
            price("llama_model_completion_cost"),
            0.0,
            0.0,
        )
    return (
        price("model_prompt_cost"),
        price("model_completion_cost"),
        price("model_cached_prompt_cost"),
        price("model_cache_write_cost"),
    )


def price_model_token_usage(
    model_name: str | None,
    context: Any,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_prompt_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Return what one model call cost, in United States dollars.

    ``prompt_tokens`` is the provider's total input count, which already
    *includes* the cached and cache-written tokens, so those two are subtracted
    before the uncached rate applies and then billed at their own rates. Prompt
    caching is not a rounding error: on 2026-09-17 cache writes were the largest
    single line of the inference model's invoice, and pricing every input token
    at the uncached rate is how a recorded cost drifts from the invoice.
    """
    prompt_price, completion_price, cached_price, cache_write_price = (
        resolve_model_token_prices(model_name, context)
    )
    prompt_tokens = max(0, int(prompt_tokens or 0))
    completion_tokens = max(0, int(completion_tokens or 0))
    cached_prompt_tokens = max(0, int(cached_prompt_tokens or 0))
    cache_write_tokens = max(0, int(cache_write_tokens or 0))
    uncached_prompt_tokens = max(
        0, prompt_tokens - cached_prompt_tokens - cache_write_tokens
    )
    return (
        uncached_prompt_tokens * prompt_price
        + cached_prompt_tokens * cached_price
        + cache_write_tokens * cache_write_price
        + completion_tokens * completion_price
    )


async def read_recorded_media_job_usage(
    pool: Any, media_job_ids: Sequence[str]
) -> dict[str, Any]:
    """Return what the named media jobs actually consumed, summed from their rows.

    Every paid call a media job makes writes its own ``api_metrics`` row stamped
    with the job's id, so the sum of those rows is the upload's real consumption
    — the number reported to the Stripe meter in place of the pre-upload
    estimate, and the number the billing portal then shows the customer.
    Best-effort: an unreadable sum returns zeros, and the caller falls back to
    the estimate rather than billing nothing.
    """
    empty_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "call_count": 0,
    }
    job_ids = [str(job_id) for job_id in media_job_ids if job_id]
    if not job_ids:
        return empty_usage
    if pool is None:
        from src.anubis.utils.runtime_handles import get_postgres_pool

        pool = get_postgres_pool()
    if pool is None:
        return empty_usage
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(_SUM_API_METRICS_FOR_MEDIA_JOBS_SQL, (job_ids,))
                row = await cursor.fetchone()
    except Exception as sum_error:  # noqa: BLE001 - non-fatal accounting read
        logger.error("Could not sum api_metrics rows for a media job: %s", sum_error)
        return empty_usage
    if not row:
        return empty_usage
    return {
        "prompt_tokens": int(row[0] or 0),
        "completion_tokens": int(row[1] or 0),
        "total_tokens": int(row[2] or 0),
        "cost_usd": float(row[3] or 0.0),
        "call_count": int(row[4] or 0),
    }


async def report_adapter_training_usage(
    stripe_client: Any,
    pool: Any,
    *,
    stripe_customer_id: str | None,
    metering_user_id: str | None,
    trained_adapter_count: int = 1,
    assistant_id: str | None = None,
    idempotency_identifier: str | None = None,
) -> bool:
    """Report one adapter-training run against the ``adapter_training_units`` meter.

    The adapter-training job endpoint does not exist yet (Phase 7 of the media
    pipeline); this helper is the complete, ready-to-call metering path so the
    future training job only has to invoke one function. The unit is a count of
    trained adapters, carried in ``total_tokens`` so ``fetch_usage_since`` sums
    every meter uniformly for allotment gating and the subscription-status
    endpoint. Best-effort like every other metering path; returns whether the
    Stripe meter event was accepted.
    """
    if trained_adapter_count <= 0:
        return False
    stripe_accepted = await report_meter_event(
        stripe_client,
        UsageMeter.ADAPTER_TRAINING_UNITS,
        stripe_customer_id,
        trained_adapter_count,
        idempotency_identifier=idempotency_identifier,
    )
    await persist_api_metrics_row(
        pool,
        inference_type="adapter_training",
        total_tokens=trained_adapter_count,
        user_id=metering_user_id,
        stripe_customer_id=stripe_customer_id,
        assistant_id=assistant_id,
        meter_event_name=UsageMeter.ADAPTER_TRAINING_UNITS.value,
    )
    return stripe_accepted
