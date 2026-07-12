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
import logging
import uuid
from typing import Any, Mapping

from src.anubis.utils.billing.tiers import UsageMeter

logger = logging.getLogger(__name__)

# Rough token-per-unit heuristics used to distill an upload into a token-equivalent
# for the document_upload_tokens meter. Grounded in research/04_token_workload_cost_model.md:
# an hour of audio transcribes to ~9k words ≈ 12k tokens, and each media item is
# additionally run through structured-output identity-analysis passes that read the
# transcript back in, so we multiply by a pipeline-amplification factor.
_ESTIMATED_TOKENS_PER_AUDIO_SECOND = 3  # ≈ 12k tokens per hour of speech
_ESTIMATED_TOKENS_PER_TEXT_CHARACTER = 0.25  # ≈ 4 characters per token
_ESTIMATED_TOKENS_PER_IMAGE = 1_000  # vision description + downstream analysis
_ESTIMATED_TOKENS_PER_URL = 1_000  # fetched page/video distilled through the pipeline
_UPLOAD_PIPELINE_AMPLIFICATION = 3.0  # transcription + classification + analysis passes


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


def estimate_upload_token_units(
    audio_seconds: float = 0.0,
    text_character_count: int = 0,
    image_count: int = 0,
    url_count: int = 0,
) -> int:
    """Distill an upload into an integer token-equivalent for the upload meter.

    Uploads are billed against ``document_upload_tokens`` because the identity
    pipeline (transcription, diarization, classification, and structured-output
    analysis) consumes model tokens roughly proportional to the raw material.
    This estimate is intentionally conservative and pipeline-amplified so an
    upload draws down the same monthly budget it will actually cost to process,
    preventing a heavy upload from silently overrunning the tier allotment.
    """
    raw_tokens = (
        audio_seconds * _ESTIMATED_TOKENS_PER_AUDIO_SECOND
        + text_character_count * _ESTIMATED_TOKENS_PER_TEXT_CHARACTER
        + image_count * _ESTIMATED_TOKENS_PER_IMAGE
        + url_count * _ESTIMATED_TOKENS_PER_URL
    )
    return int(raw_tokens * _UPLOAD_PIPELINE_AMPLIFICATION)


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
    meter_event_name TEXT
);
"""

_INSERT_API_METRICS_SQL = f"""
INSERT INTO {API_METRICS_TABLE_NAME}
    (id, user_id, stripe_customer_id, assistant_id, thread_id, inference_type,
     model_name, prompt_tokens, completion_tokens, total_tokens, cost_usd,
     latency_ms, meter_event_name)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
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
    except Exception as table_error:  # noqa: BLE001 - non-fatal at startup
        logger.error("Could not ensure api_metrics table exists: %s", table_error)


_MONTH_TO_DATE_USAGE_SQL = f"""
SELECT COALESCE(SUM(total_tokens), 0)
FROM {API_METRICS_TABLE_NAME}
WHERE user_id = %s
  AND meter_event_name = %s
  AND created_at >= date_trunc('month', now());
"""


async def fetch_month_to_date_usage(
    pool: Any, user_id: str | None, meter_event_name: str
) -> int:
    """Return the user's calendar-month-to-date usage for one meter, from ``api_metrics``.

    This is the enforcement-side counterpart to Stripe's meter aggregation: the
    free tier (including anonymous users) has no Stripe subscription and therefore
    no billing period or payable overage, so allotment gating reads the locally
    persisted usage instead of calling Stripe on the hot path. The calendar month
    approximates the billing period; paid tiers are never blocked (overage is
    billable), so the approximation only ever affects free-tier users.

    Best-effort and fail-open: any database error returns zero so a metrics outage
    degrades to "not gated" rather than blocking every free-tier message.
    """
    if pool is None or not user_id:
        return 0
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    _MONTH_TO_DATE_USAGE_SQL, (user_id, meter_event_name)
                )
                row = await cursor.fetchone()
                return int(row[0]) if row and row[0] is not None else 0
    except Exception as usage_error:  # noqa: BLE001 - fail-open gating
        logger.error(
            "Could not read month-to-date usage for user %s meter %s: %s",
            user_id,
            meter_event_name,
            usage_error,
        )
        return 0


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
) -> bool:
    """Insert one row into ``api_metrics`` describing a single billed operation.

    Best-effort persistence for observability and invoice reconciliation; returns
    whether the row was written and never raises into the request path.
    """
    if pool is None:
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
                    ),
                )
        return True
    except Exception as insert_error:  # noqa: BLE001 - non-fatal metering
        logger.error("Could not persist api_metrics row: %s", insert_error)
        return False
