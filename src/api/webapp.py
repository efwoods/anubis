# src/anubis/webapp.py

import asyncio
import base64
import functools
import json
import os
import re
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager

# from src.url_loading_graph.graph import url_loading_graph
from datetime import datetime, timezone, UTC

# Add metrics imports
from time import time_ns
from typing import Annotated, Any, List, Literal, Optional
from uuid import UUID, uuid4

import httpx
from fastapi import (
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.base import IndexConfig
from langgraph.store.postgres import AsyncPostgresStore
from langgraph_sdk import get_client

# NOTE: ``PyPDFLoader`` is imported lazily inside ``process_files_for_message``
# (the only call site) — eager import of ``langchain_community`` adds ~7.3 s to
# every cold start because the umbrella package eagerly registers many
# integrations. The first PDF upload pays the import cost once.
# Prometheus metrics
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, BeforeValidator

from src.anubis.graph import message_workflow
from src.anubis.utils.billing import (
    TIER_DEFINITIONS,
    SubscriptionTier,
    TierCapability,
    UsageMeter,
    ESTIMATED_AUDIO_FALLBACK_DURATION_SECONDS,
    TokenEstimationError,
    billable_tokens_from_metadata,
    count_words,
    LIVE_SUBSCRIPTION_STATUSES,
    SubscribeAction,
    clear_pending_cancellation,
    customer_has_payment_method,
    ensure_anonymous_billing_customers_table,
    ensure_api_metrics_table,
    TokenEstimateBreakdown,
    estimate_media_item_tokens,
    estimate_message_request_token_breakdown,
    estimate_text_tokens_from_words,
    exhausted_allotment_block_reason,
    fetch_or_measure_deep_agent_tool_schema_token_estimate,
    fetch_system_prompt_token_estimate,
    fetch_rolling_window_usage,
    fetch_usage_by_meter_since,
    fetch_usage_since,
    is_admin_metering_bypass,
    load_stripe_billing_config,
    persist_api_metrics_row,
    plan_subscribe_action,
    plan_tier_change,
    report_meter_event,
    resolve_metering_user_id,
    resolve_pay_per_use_enabled,
    resolve_stripe_customer_id,
    resolve_tier,
    resolve_usage_period_anchor,
    resolve_usage_period_end,
    resolve_usage_period_start,
    resolve_use_adapter_inference,
    resolve_checkout_trial_period_days,
    resolve_effective_monthly_allotment,
    resolve_trial_context,
    release_pending_subscription_schedule,
    subscription_has_pending_downgrade_schedule,
    subscription_period_bounds,
    tier_allotment_for_meter,
    tier_from_value,
    token_rate_limit_retry_after_seconds,
)
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.graph_interrupts import collect_pending_interrupts
from src.anubis.utils.huggingface_prefetch import ensure_huggingface_models_cached
from src.anubis.utils.store_cache import (
    invalidate_store_cache_entry,
    invalidate_store_cache_for_assistant,
)
from src.api.media_jobs import (
    MediaJob,
    create_child_job,
    create_master_job,
    get_job,
    request_cancel,
    run_batch_media_job,
)
from src.security.auth import (
    _tier_from_subscription,
    check_subscription_status,
    get_current_user,
    get_current_user_or_anonymous_user,
    get_user_with_api_key,
    security_route,
    update_assistant_config,
    update_user_app_metadata_fields,
    update_user_subscription_status,
)


def tier_from_value_or_400(value: str) -> SubscriptionTier:
    """Coerce a request-supplied tier string into a SubscriptionTier or raise 400.

    Unlike the defensive ``tier_from_value`` (which silently falls back to free),
    an explicit tier chosen by the caller must be a real tier name, so an unknown
    value is a client error rather than a silent downgrade.
    """
    try:
        return SubscriptionTier(str(value).strip().lower())
    except ValueError:
        raise HTTPException(
            detail=f"Unknown subscription tier '{value}'. Expected one of: "
            + ", ".join(t.value for t in SubscriptionTier),
            status_code=400,
        )


# Human-readable reason returned when a tier lacks a capability, so the client can
# prompt the user to upgrade to the tier that unlocks it.
_CAPABILITY_REQUIRED_TIER = {
    TierCapability.UPLOAD: SubscriptionTier.PRO,
    TierCapability.TRAIN_ADAPTER: SubscriptionTier.PREMIUM,
}


def enforce_tier_capability(current_user: dict, capability: TierCapability) -> SubscriptionTier:
    """Raise HTTP 403 unless the user's resolved tier unlocks ``capability``.

    This is the enforcement layer that gates billable work by tier: every tier can
    message, pro adds uploads, premium adds adapter training. Anonymous users
    resolve to free and therefore reach only the message capability. Returns the
    resolved tier so callers can reuse it without recomputing.
    """
    tier = resolve_tier(current_user)
    if capability in TIER_DEFINITIONS[tier].capabilities:
        return tier
    required = _CAPABILITY_REQUIRED_TIER.get(capability)
    required_text = f" Upgrade to the {required.value} tier." if required else ""
    raise HTTPException(
        status_code=403,
        detail=f"Your '{tier.value}' tier does not permit this action.{required_text}",
    )


def resolve_usage_period_start_for_user(current_user: dict) -> datetime:
    """Return the start of the usage period governing this user's allotment.

    The most recent of the available period signals wins, so a mid-period tier
    upgrade always begins a fresh local usage window:

    1. The Stripe billing period start cached into
       ``app_metadata.subscription_status.current_period_start`` by the webhook
       (paid tiers only) — keeps local gating aligned with the invoice period
       without a Stripe call on the hot path.
    2. The user's ``usage_period_anchor`` (written at tier upgrade and first
       checkout), expanded into a recurring window by
       ``resolve_usage_period_start`` per the USAGE_PERIOD_DAYS configuration.
    3. The environment-configured default period (calendar month when
       USAGE_PERIOD_DAYS is zero).
    """
    context = GlobalContext()
    now = datetime.now(UTC)
    usage_period_days = int(context.usage_period_days or 0)
    period_anchor = resolve_usage_period_anchor(current_user)
    period_start = resolve_usage_period_start(now, usage_period_days, period_anchor)

    app_metadata = (current_user or {}).get("app_metadata") or {}
    subscription_status = app_metadata.get("subscription_status") or {}
    cached_stripe_period_start = subscription_status.get("current_period_start")
    if cached_stripe_period_start:
        try:
            stripe_period_start = datetime.fromtimestamp(
                int(cached_stripe_period_start), tz=UTC
            )
            period_start = max(period_start, stripe_period_start)
        except (TypeError, ValueError, OSError):
            pass
    return period_start


async def enforce_remaining_allotment(
    app_state,
    current_user: dict,
    meter: UsageMeter,
    estimated_request_tokens: int = 0,
) -> None:
    """Block a user of ANY tier whose ``meter`` allotment cannot cover this request.

    The block decision is ``exhausted_allotment_block_reason``: period usage
    PLUS the pre-call estimate of this request under the allotment is allowed;
    at or past the allotment, only pay-per-use (a payment method on file lets
    the Stripe graduated metered price bill the overage) allows the request.
    Otherwise the request is refused with HTTP 402 until the period resets,
    pay-per-use is enabled, or the user upgrades tiers. Usage is read from the
    local ``api_metrics`` table (which also covers anonymous users via their
    hashed-IP identifier) over the period resolved by
    ``resolve_usage_period_start_for_user``. The admin testing account skips
    enforcement entirely.
    """
    if is_admin_metering_bypass(current_user, GlobalContext().admin_user_id):
        return
    tier = resolve_tier(current_user)
    allotment = tier_allotment_for_meter(tier, meter)
    if allotment is None:
        # The capability gate is the authority for dimensions the tier lacks.
        return
    # A user inside a free-trial window keeps the trial tier's allotment as a
    # floor after changing tiers (e.g. trialing premium then downgrading to
    # pro keeps the premium allotment until trial_end), so the gate must judge
    # against the trial-aware effective allotment, not the plain tier value.
    effective_allotment = resolve_effective_monthly_allotment(
        tier, meter, resolve_trial_context(current_user)
    )
    metering_user_id = resolve_metering_user_id(current_user)
    period_start = resolve_usage_period_start_for_user(current_user)
    period_usage = await fetch_usage_since(
        getattr(app_state, "pool", None), metering_user_id, meter.value, period_start
    )
    block_reason = exhausted_allotment_block_reason(
        tier,
        meter,
        period_usage,
        resolve_pay_per_use_enabled(current_user),
        estimated_request_tokens=estimated_request_tokens,
        allotment_override=effective_allotment,
    )
    if block_reason:
        raise HTTPException(status_code=402, detail=block_reason)


async def enforce_token_rate_limit(
    app_state,
    current_user: dict,
    meter_event_names: list[str],
    window_seconds: int,
    tokens_per_window: int,
    estimated_request_tokens: int = 0,
) -> None:
    """Refuse the request with HTTP 429 when the user's token rate cap is met.

    A tokens-per-window abuse guard (in the spirit of the OpenAI rate-limit
    guide) independent of the monthly allotment and of pay-per-use: it caps how
    fast tokens can be consumed so a runaway client cannot burn a month's budget
    or an unbounded overage bill in minutes. The window usage is checked WITH
    this request's pre-call estimate added, so one huge request is refused
    before burning the cap rather than after. The Retry-After header tells the
    client when the oldest usage row ages out of the rolling window. A cap of
    zero or less disables the limit entirely; the admin testing account skips
    the limit.
    """
    if tokens_per_window <= 0 or window_seconds <= 0:
        return
    if is_admin_metering_bypass(current_user, GlobalContext().admin_user_id):
        return
    window_usage, oldest_usage_at = await fetch_rolling_window_usage(
        getattr(app_state, "pool", None),
        resolve_metering_user_id(current_user),
        window_seconds,
        meter_event_names=meter_event_names,
    )
    projected_window_usage = window_usage + max(0, estimated_request_tokens)
    retry_after_seconds = token_rate_limit_retry_after_seconds(
        projected_window_usage, tokens_per_window, window_seconds, oldest_usage_at
    )
    if retry_after_seconds is not None:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Token rate limit reached: {window_usage:,} tokens used in the "
                f"last {window_seconds} seconds (plus an estimated "
                f"{max(0, estimated_request_tokens):,} for this request) against "
                f"a cap of {tokens_per_window:,}. "
                f"Retry after {retry_after_seconds} seconds."
            ),
            headers={"Retry-After": str(retry_after_seconds)},
        )


def _resolve_usage_period_bounds_for_user(
    current_user: dict,
) -> tuple[datetime, datetime]:
    """Return the (start, end) of the usage period governing this user.

    Start comes from ``resolve_usage_period_start_for_user``; the end prefers
    the Stripe billing period end cached by the webhook, falling back to the
    environment-configured period shape. Shared by enforcement-adjacent
    display: the subscription-status endpoint and the per-response usage
    snapshots.
    """
    context = GlobalContext()
    usage_period_days = int(context.usage_period_days or 0)
    period_start = resolve_usage_period_start_for_user(current_user)

    app_metadata = (current_user or {}).get("app_metadata") or {}
    cached_subscription_status = app_metadata.get("subscription_status") or {}
    cached_period_end = cached_subscription_status.get("current_period_end")
    if cached_period_end:
        try:
            return period_start, datetime.fromtimestamp(int(cached_period_end), tz=UTC)
        except (TypeError, ValueError, OSError):
            pass
    return period_start, resolve_usage_period_end(period_start, usage_period_days)


async def _build_meter_usage_snapshot(
    app_state, current_user: dict, meter: UsageMeter
) -> dict:
    """Return one meter's usage-versus-allotment view for endpoint responses.

    The same shape a customer portal polls from ``/verify_subscription_status``,
    scoped to the single meter governing the current request, so every metered
    endpoint can show the caller where they stand: allotment, usage to date in
    the current period, remaining budget, the pay-per-use flag, and the period
    bounds.
    """
    tier = resolve_tier(current_user)
    # Trial-aware allotment so the streamed snapshot matches what enforcement
    # actually gates against during a free-trial window (see
    # resolve_effective_monthly_allotment).
    allotment = resolve_effective_monthly_allotment(
        tier, meter, resolve_trial_context(current_user)
    )
    period_start, period_end = _resolve_usage_period_bounds_for_user(current_user)
    used_to_date = await fetch_usage_since(
        getattr(app_state, "pool", None),
        resolve_metering_user_id(current_user),
        meter.value,
        period_start,
    )
    monthly_allotment = allotment.monthly_allotment if allotment else None
    return {
        "meter": meter.value,
        "tier": tier.value,
        "monthly_allotment": monthly_allotment,
        "used_to_date": used_to_date,
        "remaining": (
            max(0, monthly_allotment - used_to_date)
            if monthly_allotment is not None
            else None
        ),
        "pay_per_use_enabled": resolve_pay_per_use_enabled(current_user),
        "usage_period_start": period_start.isoformat(),
        "usage_period_end": period_end.isoformat(),
    }


async def _measure_system_prompt_tokens_for_request(
    app_state, estimation_config: dict, message: str | None
) -> int:
    """Return the MEASURED token estimate of this request's real system prompt.

    Reads the process-wide estimate cache first (populated every time
    ``load_consciousness`` builds the prompt); on a miss or stale entry the
    REAL prompt is built through ``build_system_prompt_text_for_estimation``
    (store reads only — no model call, and the build records a fresh cache
    entry as a side effect) and measured with the same word-ratio arithmetic.
    Raises on any failure — the caller treats estimation as fail-closed.
    """
    context = GlobalContext()
    configurable = (estimation_config or {}).get("configurable") or {}
    user_id = configurable.get("user_id")
    assistant_id = configurable.get("assistant_id")
    if user_id and assistant_id:
        cached_estimate = fetch_system_prompt_token_estimate(
            user_id,
            assistant_id,
            max_age_seconds=float(
                context.system_prompt_token_estimate_cache_ttl_seconds or 0
            ),
        )
        if cached_estimate is not None:
            return cached_estimate

    # Lazy import: nodes pulls the full prompt-building stack, which must not
    # load at webapp import time (cold-start convention in CLAUDE.md).
    from src.anubis.utils.nodes import build_system_prompt_text_for_estimation

    system_prompt_text = await build_system_prompt_text_for_estimation(
        getattr(app_state, "store", None), estimation_config, message or ""
    )
    return estimate_text_tokens_from_words(count_words(system_prompt_text))


async def _estimate_message_request_tokens(
    app_state,
    estimation_config: dict,
    message: str | None,
    file_text_content: str | None,
    multimodal_content: list | None,
) -> TokenEstimateBreakdown:
    """Manually estimate one message turn's tokens before the model runs.

    Manual computation (no tokenizer, no counting endpoint — arithmetic over
    known quantities is the fastest path), split into input versus output:

    * INPUT — the MEASURED system prompt for this (user, avatar) pair (from
      the estimate cache, or built fresh on a miss), the MEASURED schemas of
      every tool the deep agent binds (enumerated from the compiled agent,
      so newly added tools are included automatically — see
      ``tool_schema_estimate_cache.py``; the provider bills the serialized
      tool definitions as input on the initial model call), the word-ratio
      estimate of the variable user text (message plus attached-file text),
      and every attached image's vision patches. There is NO fixed or
      guessed input overhead — every input component is measured. (The tool
      loop may run the model more than once before the final reply; the loop
      count is unknowable in advance, so the estimate covers the initial
      call and recorded actual totals govern accrual for the loop.) The
      allotment gate consumes ``input_tokens``.
    * OUTPUT — the expected reply budget
      (``MESSAGE_EXPECTED_OUTPUT_TOKENS_ESTIMATE``) plus each attached
      image's expected description. Output is not gated in advance; actual
      TOTAL usage from the model API accrues after the turn.

    ``estimation_config`` is the request's LangGraph config whose
    ``configurable`` carries ``user_id``, ``assistant_id``, and
    ``assistant_ctx`` — the same identifiers ``load_consciousness`` builds
    the prompt from, so the estimate reflects the prompt the model will read.

    Fail-closed: an unreadable attached image, a failed system-prompt build,
    or any other estimation failure raises HTTP 422 — nothing unestimated may
    reach the model.
    """
    context = GlobalContext()
    try:
        system_prompt_tokens = await _measure_system_prompt_tokens_for_request(
            app_state, estimation_config, message
        )
        # First call per process compiles the deep agent graph to enumerate
        # the bound tools, so the measurement runs on a worker thread; every
        # later call returns the cached integer.
        tool_schema_tokens = await asyncio.to_thread(
            fetch_or_measure_deep_agent_tool_schema_token_estimate
        )

        image_dimensions: list[tuple[int, int]] = []
        for content_block in multimodal_content or []:
            if (
                not isinstance(content_block, dict)
                or content_block.get("type") != "image_url"
            ):
                continue
            image_url_value = (content_block.get("image_url") or {}).get("url") or ""
            if "," in image_url_value:
                image_bytes = base64.b64decode(image_url_value.split(",", 1)[1])
            else:
                image_bytes = base64.b64decode(image_url_value)
            image_dimensions.append(_image_dimensions_from_bytes(image_bytes))

        variable_text = " ".join(
            part for part in (message or "", file_text_content or "") if part
        )
        return estimate_message_request_token_breakdown(
            count_words(variable_text),
            image_dimensions,
            system_prompt_tokens=system_prompt_tokens,
            tool_schema_tokens=tool_schema_tokens,
            expected_output_tokens=int(
                context.message_expected_output_tokens_estimate or 0
            ),
        )
    except HTTPException:
        raise
    except Exception as estimation_error:  # noqa: BLE001 - fail-closed estimation
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not estimate token usage for this message: "
                f"{estimation_error} The request was not processed."
            ),
        ) from estimation_error


async def _meter_message_usage(
    app_state,
    current_user: dict,
    response_metadata: Optional[dict],
    thread_id: Optional[str],
    assistant_id: Optional[str],
    latency_ms: float,
    request_id: Optional[str] = None,
) -> dict | None:
    """Report messaging (or adapter-inference) token usage; return the usage block.

    Best-effort and non-fatal: extracts billable tokens from the model response
    metadata, reports them to the correct Stripe meter keyed on the customer,
    increments the Prometheus token/cost counters, and writes one
    ``api_metrics`` row. A ``None`` customer id (anonymous user) makes the
    Stripe report a no-op while still recording local metrics. The admin
    testing account skips both billing writes (Prometheus observability is
    kept). Returns the turn's usage summary — this turn's actual token counts
    plus the caller's usage-versus-allotment snapshot for the governing meter —
    for endpoint responses; returns ``None`` when metering fails (the model
    already ran; post-response reporting stays fail-open).
    """
    try:
        token_usage = (response_metadata or {}).get("token_usage") or {}
        prompt_tokens = int(token_usage.get("prompt_tokens") or 0)
        completion_tokens = int(token_usage.get("completion_tokens") or 0)
        total_tokens = billable_tokens_from_metadata(response_metadata)
        model_name = (response_metadata or {}).get("model_name")
        cost_usd = float((response_metadata or {}).get("total_cost") or 0.0)

        stripe_customer_id = resolve_stripe_customer_id(current_user)
        tier = resolve_tier(current_user)
        admin_metering_bypass = is_admin_metering_bypass(
            current_user, GlobalContext().admin_user_id
        )

        # Adapter inference is billed against a separate meter at a different rate;
        # the think node sets is_adapter_inference when the client requested
        # adapter=True and the user is Premium (see use_adapter_inference in config).
        is_adapter_inference = bool((response_metadata or {}).get("is_adapter_inference"))
        if is_adapter_inference and tier == SubscriptionTier.PREMIUM:
            meter = UsageMeter.ADAPTER_INFERENCE_TOKENS
            inference_type = "adapter_inference"
        else:
            meter = UsageMeter.MESSAGING_TOKENS
            inference_type = "message"

        # The request id keys Stripe's meter-event deduplication so a retried
        # request cannot double-bill the same turn.
        if not admin_metering_bypass:
            await report_meter_event(
                app_state.stripe,
                meter,
                stripe_customer_id,
                total_tokens,
                idempotency_identifier=(
                    f"{request_id}:{meter.value}" if request_id else None
                ),
            )

        if model_name and total_tokens > 0:
            if prompt_tokens:
                MODEL_TOKENS_TOTAL.labels(model=model_name, type="prompt").inc(prompt_tokens)
            if completion_tokens:
                MODEL_TOKENS_TOTAL.labels(model=model_name, type="completion").inc(
                    completion_tokens
                )
            if cost_usd:
                MODEL_COST_TOTAL.labels(model=model_name).inc(cost_usd)

        if not admin_metering_bypass:
            await persist_api_metrics_row(
                getattr(app_state, "pool", None),
                inference_type=inference_type,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                user_id=resolve_metering_user_id(current_user),
                stripe_customer_id=stripe_customer_id,
                assistant_id=assistant_id,
                thread_id=thread_id,
                model_name=model_name,
                meter_event_name=meter.value,
            )

        # The insert above was awaited, so the snapshot's used_to_date already
        # includes this turn.
        usage_snapshot = await _build_meter_usage_snapshot(
            app_state, current_user, meter
        )
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            **usage_snapshot,
            **({"admin_metering_bypass": True} if admin_metering_bypass else {}),
        }
    except Exception as metering_error:  # noqa: BLE001 - metering must never break a reply
        logger.error("Failed to meter message usage: %s", metering_error)
        return None


def _drop_empty_file_fields(value: Any) -> Any:
    """Normalize the multipart ``files`` field so an absent upload is treated as
    "no files" instead of raising a 422.

    Swagger UI (and some HTTP clients) submit an *empty* file field as a form
    value of ``""`` rather than omitting it. FastAPI then receives ``[""]`` and,
    while validating each element against ``UploadFile``, fails with
    ``Expected UploadFile, received: <class 'str'>`` before the endpoint body
    ever runs. Stripping the stray string(s) here turns that into an empty list.
    """
    if isinstance(value, list):
        return [item for item in value if not isinstance(item, str)]
    if isinstance(value, str):
        return []
    return value


# Reusable annotation for optional multipart file uploads. Use this instead of
# ``Optional[List[UploadFile]] = File(None)`` so empty file fields don't 422.
OptionalUploadFiles = Annotated[
    Optional[List[UploadFile]],
    BeforeValidator(_drop_empty_file_fields),
    File(),
]

import logging
from urllib.parse import quote

import stripe
from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from langgraph_sdk.schema import Assistant
from psycopg.rows import class_row

from src.anubis.utils import runtime_handles

load_dotenv()


logger = logging.getLogger(__name__)

from uuid import NAMESPACE_URL, uuid5


def _namespace_safe_formatted_filename(u: str) -> str:
    formatted_name = str(uuid5(NAMESPACE_URL, u))
    return formatted_name


def _document_label_and_key(metadata: dict) -> tuple[str | None, str | None]:
    """Map a stored Document's metadata to the pair (human label, storage key).

    The *label* is what ``/list_avatar_documents`` displays; the *key* is the
    ``namespace_filename`` that ``/delete_avatar_document`` matches store rows on.
    Keeping both in one place is what lets the two endpoints round-trip: a string
    a user copies out of the list resolves back to the exact key delete needs.

    Playlist videos carry playlist_url / playlist_title / video_title (see
    ``URLDocumentLoaderClass._load_youtube_playlist``); they are labeled
    ``{playlist_title} :: {video_title}`` but keyed by an opaque uuid5
    namespace_filename (hashed over ``{playlist_ns}::{video_ns}``) — the label is
    human-readable titles, the key is a hash, so the two never coincide. Everything
    else is both labeled and keyed by its plain filename. Titles fall back to
    URLs/filenames when yt_dlp couldn't resolve them.
    """
    filename = metadata.get("filename")
    namespace_filename = metadata.get("namespace_filename")
    key = (
        namespace_filename
        if isinstance(namespace_filename, str) and namespace_filename
        else None
    )
    playlist_url = metadata.get("playlist_url")
    if isinstance(playlist_url, str) and playlist_url:
        playlist_label = (metadata.get("playlist_title") or playlist_url).strip()
        video_label = (
            metadata.get("video_title")
            or (filename if isinstance(filename, str) else "")
            or "untitled"
        ).strip()
        return f"{playlist_label} :: {video_label}", key
    if isinstance(filename, str) and filename:
        return filename, key
    return None, key


def _iter_document_labels(store_items) -> Iterator[tuple[str, str | None]]:
    """Yield (label, key) for each stored Document, de-structuring the same
    value.document.kwargs.metadata path /list and /delete read. Multiple
    Documents per source (quote / identity / analysis) yield the same pair; the
    caller de-dupes."""
    for item in store_items or []:
        value = getattr(item, "value", None)
        if value is None and isinstance(item, dict):
            value = item.get("value")
        if not isinstance(value, dict):
            continue
        document = value.get("document")
        kwargs_blob = document.get("kwargs") if isinstance(document, dict) else None
        metadata = (
            kwargs_blob.get("metadata") if isinstance(kwargs_blob, dict) else None
        )
        if not isinstance(metadata, dict):
            continue
        label, key = _document_label_and_key(metadata)
        if label:
            yield label, key


def _latest_ai_from_stream_update(payload: dict) -> AIMessage | None:
    """Pick the last AIMessage from a LangGraph ``updates`` chunk (any node)."""
    last_ai: AIMessage | None = None
    for _node, v in payload.items():
        if not isinstance(v, dict):
            continue
        msgs = v.get("messages")
        if not msgs:
            continue
        tail = msgs[-1]
        if isinstance(tail, AIMessage):
            last_ai = tail
    return last_ai


async def message_graph_sse(
    graph,
    human_message: HumanMessage,
    config: dict,
    context: GlobalContext,
    *,
    thread_id: str,
    user_id: str,
    assistant_id: str,
    conversation_title_value: str | None,
    start_time_ns: int,
    request_id: str,
    langgraph_client_headers: dict,
    resume_command: Optional[Command] = None,
    app_state=None,
    current_user: Optional[dict] = None,
    estimated_request_tokens: TokenEstimateBreakdown | None = None,
    estimate_meter: UsageMeter | None = None,
    include_usage_metrics: bool = True,
):
    """Stream assistant tokens (SSE) then a terminal event with full metadata.

    When ``include_usage_metrics`` is true, the FIRST frame is a
    ``usage_estimate`` event carrying this request's pre-call ``input_tokens``
    estimate (what the allotment gate consumed) and the caller's
    usage-versus-allotment snapshot. Actual meter accrual always uses the
    provider's returned token usage after the turn, regardless of
    ``include_usage_metrics``. The terminal event is ``done`` on completion
    (optionally carrying the turn's actual ``usage``), or ``interrupt`` when
    the graph pauses for human approval. Pass ``resume_command``
    (``Command(resume=...)``) to continue a paused run instead of sending a
    fresh ``human_message``.
    """
    accumulated_chunks: list[str] = []
    last_ai: AIMessage | None = None

    # Pre-call estimate + allotment snapshot (reporting only).
    if (
        include_usage_metrics
        and estimated_request_tokens is not None
        and app_state is not None
        and current_user is not None
    ):
        usage_estimate_event = {
            "type": "usage_estimate",
            "input_tokens": estimated_request_tokens.input_tokens,
            "usage": await _build_meter_usage_snapshot(
                app_state, current_user, estimate_meter or UsageMeter.MESSAGING_TOKENS
            ),
            "thread_id": thread_id,
            "request_id": request_id,
        }
        yield f"data: {json.dumps(usage_estimate_event, default=str)}\n\n"

    graph_input = (
        resume_command if resume_command is not None else {"messages": [human_message]}
    )

    # #region agent log
    if resume_command is not None:
        try:
            import time
            from pathlib import Path

            _log_path = (
                Path(__file__).resolve().parents[2] / ".cursor" / "debug-ba8488.log"
            )
            _log_path.parent.mkdir(parents=True, exist_ok=True)
            with _log_path.open("a", encoding="utf-8") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "ba8488",
                            "location": "webapp.py:message_graph_sse:resume_start",
                            "message": "outer graph resume requested",
                            "data": {
                                "thread_id": thread_id,
                                "resume_payload": getattr(
                                    resume_command, "resume", None
                                ),
                            },
                            "hypothesisId": "D",
                            "timestamp": int(time.time() * 1000),
                        },
                        default=str,
                    )
                    + "\n"
                )
        except Exception:
            pass
    # #endregion

    async for item in graph.astream(
        input=graph_input,
        config=config,
        context=context,
        stream_mode=["custom", "updates"],
        subgraphs=True,
    ):
        if not isinstance(item, tuple) or len(item) != 3:
            continue
        _ns, mode, payload = item
        if mode == "custom" and isinstance(payload, dict):
            if payload.get("type") == "assistant_token":
                accumulated_chunks.append(payload.get("text") or "")
                yield f"data: {json.dumps(payload)}\n\n"
        elif mode == "updates" and isinstance(payload, dict):
            ai = _latest_ai_from_stream_update(payload)
            if ai is not None:
                last_ai = ai

    thread_metadata = {
        "thread_metadata": {
            "user_id": user_id,
            "assistant_id": assistant_id,
            "most_recent_message": datetime.now(UTC).isoformat(),
            "conversation_title": conversation_title_value,
        },
        "graph_id": "Anubis",
    }
    langgraph_client = get_client(headers=langgraph_client_headers)
    await langgraph_client.threads.update(thread_id=thread_id, metadata=thread_metadata)

    # If the graph paused for human approval, surface the preview instead of ``done``.
    # The client resumes via ``POST /message/{assistant_id}/resume`` on this thread_id.
    snapshot = await graph.aget_state(config)
    pending_interrupts = collect_pending_interrupts(snapshot)
    if pending_interrupts:
        interrupt_event: dict = {
            "type": "interrupt",
            "thread_id": thread_id,
            "request_id": request_id,
            "interrupt": getattr(pending_interrupts[0], "value", None),
            "total_response_time_ms": (time_ns() - start_time_ns) // 1_000_000,
        }
        yield f"data: {json.dumps(interrupt_event, default=str)}\n\n"
        return

    content = last_ai.content if last_ai is not None else "".join(accumulated_chunks)
    done: dict = {
        "type": "done",
        "content": content,
        "thread_id": thread_id,
        "request_id": request_id,
        "total_response_time_ms": (time_ns() - start_time_ns) // 1_000_000,
    }
    response_metadata = (
        last_ai.response_metadata
        if last_ai is not None and getattr(last_ai, "response_metadata", None)
        else None
    )
    if response_metadata:
        done["response_metadata"] = response_metadata

    # Always accrue actual model API usage BEFORE the terminal frame. Reporting
    # the usage block on ``done`` is optional (``include_usage_metrics``);
    # metering failure is fail-open and just omits the usage block.
    if app_state is not None and current_user is not None:
        turn_usage = await _meter_message_usage(
            app_state=app_state,
            current_user=current_user,
            response_metadata=response_metadata,
            thread_id=thread_id,
            assistant_id=assistant_id,
            latency_ms=(time_ns() - start_time_ns) / 1_000_000,
            request_id=request_id,
        )
        if include_usage_metrics and turn_usage:
            done["usage"] = turn_usage
    yield f"data: {json.dumps(done, default=str)}\n\n"


class MessagePayload(BaseModel):
    message: str = "Hey! Please tell me about yourself and what you can do for me."
    your_name: Optional[str] = None
    your_description: Optional[str] = None
    conversation_title: Optional[str] = None


class FeedbackData(BaseModel):
    """Feedback data for human-in-the-loop responses"""

    feedback_type: str  # 'like', 'dislike', 'rating', 'edit'
    rating: Optional[float] = None  # 1-5 scale for 'rating' type
    comment: Optional[str] = None
    edited_response: Optional[str] = None  # User edited the response


class MessageResponse(BaseModel):
    """Response model for message endpoints with feedback support"""

    content: str
    response_metadata: Optional[dict] = None
    total_response_time_ms: int
    thread_id: str
    request_id: str  # For feedback submission
    feedback: Optional[FeedbackData] = None


# Create a custom registry for metrics
registry = CollectorRegistry()

# Define metrics
REQUEST_COUNT = Counter(
    "anubis_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"],
    registry=registry,
)

REQUEST_LATENCY = Histogram(
    "anubis_request_duration_seconds",
    "Request duration in seconds",
    ["method", "endpoint"],
    registry=registry,
)

ACTIVE_REQUESTS = Gauge(
    "anubis_active_requests", "Number of active requests", registry=registry
)

MODEL_TOKENS_TOTAL = Counter(
    "anubis_model_tokens_total",
    "Total number of tokens used by model",
    ["model", "type"],  # type: prompt or completion
    registry=registry,
)

MODEL_COST_TOTAL = Counter(
    "anubis_model_cost_total_usd",
    "Total cost in USD for model usage",
    ["model"],
    registry=registry,
)

API_RESPONSE_STATUS = Counter(
    "anubis_api_response_status_total",
    "Response status codes",
    ["status"],
    registry=registry,
)


class ASSISTANT_QUERY(BaseModel):
    assistant_id: UUID
    graph_id: str
    created_at: datetime
    updated_at: datetime
    config: dict[str, Any]
    metadata: dict[str, Any]
    version: int
    name: str
    description: str | None
    context: dict[str, Any]

    def to_assistant(self) -> Assistant:
        return self.model_dump(mode="json")


AUTH_CATCH_ALL_PATTERNS = (
    # assistants
    ("POST", re.compile(r"^/assistants$")),
    ("POST", re.compile(r"^/assistants/search$")),
    ("POST", re.compile(r"^/assistants/count$")),
    ("GET", re.compile(r"^/assistants/[^/]+$")),
    ("DELETE", re.compile(r"^/assistants/[^/]+$")),
    ("PATCH", re.compile(r"^/assistants/[^/]+$")),
    ("GET", re.compile(r"^/assistants/[^/]+/graph$")),
    ("GET", re.compile(r"^/assistants/[^/]+/subgraphs$")),
    ("GET", re.compile(r"^/assistants/[^/]+/subgraphs/[^/]+$")),
    ("GET", re.compile(r"^/assistants/[^/]+/schemas$")),
    ("POST", re.compile(r"^/assistants/[^/]+/versions$")),
    ("POST", re.compile(r"^/assistants/[^/]+/latest$")),
    # threads
    ("POST", re.compile(r"^/threads$")),
    ("POST", re.compile(r"^/threads/search$")),
    ("POST", re.compile(r"^/threads/count$")),
    ("POST", re.compile(r"^/threads/prune$")),
    ("GET", re.compile(r"^/threads/[^/]+/state$")),
    ("POST", re.compile(r"^/threads/[^/]+/state$")),
    ("GET", re.compile(r"^/threads/[^/]+/state/[^/]+$")),
    ("POST", re.compile(r"^/threads/[^/]+/state/checkpoint$")),
    ("GET", re.compile(r"^/threads/[^/]+/history$")),
    ("POST", re.compile(r"^/threads/[^/]+/history$")),
    ("POST", re.compile(r"^/threads/[^/]+/copy$")),
    ("GET", re.compile(r"^/threads/[^/]+$")),
    ("DELETE", re.compile(r"^/threads/[^/]+$")),
    ("PATCH", re.compile(r"^/threads/[^/]+$")),
    ("GET", re.compile(r"^/threads/[^/]+/stream$")),
    # thread runs
    ("GET", re.compile(r"^/threads/[^/]+/runs$")),
    ("POST", re.compile(r"^/threads/[^/]+/runs$")),
    ("POST", re.compile(r"^/threads/[^/]+/runs/stream$")),
    ("POST", re.compile(r"^/threads/[^/]+/runs/wait$")),
    ("GET", re.compile(r"^/threads/[^/]+/runs/[^/]+$")),
    ("DELETE", re.compile(r"^/threads/[^/]+/runs/[^/]+$")),
    ("GET", re.compile(r"^/threads/[^/]+/runs/[^/]+/join$")),
    ("GET", re.compile(r"^/threads/[^/]+/runs/[^/]+/stream$")),
    ("POST", re.compile(r"^/threads/[^/]+/runs/[^/]+/cancel$")),
    # runs
    ("POST", re.compile(r"^/runs/cancel$")),
    ("POST", re.compile(r"^/runs/stream$")),
    ("POST", re.compile(r"^/runs/wait$")),
    ("POST", re.compile(r"^/runs$")),
    ("POST", re.compile(r"^/runs/batch$")),
    # crons
    ("POST", re.compile(r"^/threads/[^/]+/runs/crons$")),
    ("POST", re.compile(r"^/runs/crons$")),
    ("POST", re.compile(r"^/runs/crons/search$")),
    ("POST", re.compile(r"^/runs/crons/count$")),
    ("PATCH", re.compile(r"^/runs/crons/[^/]+$")),
    ("DELETE", re.compile(r"^/runs/crons/[^/]+$")),
    # store
    ("PUT", re.compile(r"^/store/items$")),
    ("DELETE", re.compile(r"^/store/items$")),
    ("GET", re.compile(r"^/store/items$")),
    ("POST", re.compile(r"^/store/items/search$")),
    ("POST", re.compile(r"^/store/namespaces$")),
    # a2a
    ("POST", re.compile(r"^/a2a/[^/]+$")),
    # mcp
    ("POST", re.compile(r"^/mcp$")),
    ("GET", re.compile(r"^/mcp$")),
    ("DELETE", re.compile(r"^/mcp$")),
)


def _is_auth_catch_all_target(method: str, path: str) -> bool:
    normalized_path = path.rstrip("/") or "/"
    for expected_method, pattern in AUTH_CATCH_ALL_PATTERNS:
        if method == expected_method and pattern.match(normalized_path):
            return True
    return False


async def get_public_avatars(
    assistant_id: Optional[str] = None, user_id: Optional[str] = None
):
    pool = app.state.pool

    if assistant_id:
        # Retrieve the public avatar matching the assistant_id
        search_query = """
        SELECT * FROM assistant 
        WHERE (metadata->>'is_public')::boolean = TRUE
        AND assistant_id = %s
        """
    elif user_id:
        # Retrieve all public avatars not owned by the current user.
        search_query = """
        SELECT * FROM assistant
        WHERE (metadata->>'is_public')::boolean = TRUE
        AND (metadata->>'user_id') != %s
        """
    else:
        # Retrieve all public avatars
        search_query = """
        SELECT * FROM assistant
        WHERE (metadata->>'is_public')::boolean = TRUE
        """

    async with pool.connection() as conn:
        async with conn.cursor(row_factory=class_row(ASSISTANT_QUERY)) as cur:
            if assistant_id:
                await cur.execute(search_query, (assistant_id,))
            elif user_id:
                await cur.execute(search_query, (user_id,))
            else:
                await cur.execute(search_query)
            data = await cur.fetchall()

            return [assistant_query.to_assistant() for assistant_query in data]


def _assistant_without_metadata_if_public(assistant: dict[str, Any]) -> dict[str, Any]:
    meta = assistant.get("metadata")
    if isinstance(meta, dict):
        pub = meta.get("is_public")
        if pub is True or (isinstance(pub, str) and pub.lower() == "true"):
            return {k: v for k, v in assistant.items() if k != "metadata"}
    return assistant


import debugpy

logger.info(f"DEBUG_PORT: {os.getenv('DEBUG_PORT', 5678)}")
logger.info(f"DEV: {os.getenv('DEV', 'false')}")

if os.getenv("DEV", "false").lower() == "true":
    debugpy.listen(("0.0.0.0", int(os.getenv("DEBUG_PORT", 5678))))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup: Preload the Whisper model pipeline
    global context
    global store_context_manager

    # Initialize context / context
    app.state.context = GlobalContext()
    ensure_huggingface_models_cached(app.state.context)
    # Explicit timeouts instead of httpx's silent 5 s default: a short connect
    # timeout fails fast on an unreachable host, while a generous read timeout
    # tolerates a slow-but-alive upstream.
    app.state.httpx_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0)
    )
    app.state.stripe = stripe
    app.state.stripe.api_key = app.state.context.stripe_secret_key

    async_postgres_store_uri = app.state.context.async_postgres_store_uri
    logger.warning(f"app.state.context.dev: {app.state.context.dev}")
    pool = AsyncConnectionPool(
        conninfo=async_postgres_store_uri,
        min_size=1,
        max_size=5,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        open=False,  # do not open on create
    )
    app.state.pool = pool
    await app.state.pool.open()

    # Ensure the api_metrics table exists and cache the parsed Stripe billing config
    # (meter + tier price ids) so metering and subscription endpoints can use them.
    await ensure_api_metrics_table(app.state.pool)
    await ensure_anonymous_billing_customers_table(app.state.pool)
    try:
        app.state.stripe_billing_config = load_stripe_billing_config(
            app.state.context.stripe_billing_config_json
        )
    except ValueError as billing_config_error:
        logger.error("Invalid STRIPE_BILLING_CONFIG_JSON: %s", billing_config_error)
        app.state.stripe_billing_config = None
    try:
        embed = "huggingface:" + app.state.context.embedding_model
        # IndexConfig key must be ``fields`` (plural). Using ``field`` is ignored and
        # LangGraph falls back to embedding the entire JSON value ("$") — catastrophic
        # when values include multi‑MB reference_image_data URIs on store.aput.
        store = AsyncPostgresStore(
            app.state.pool,
            index=IndexConfig(
                dims=640,
                embed=embed,
                fields=["document.kwargs.page_content"],
            ),
        )

        await store.setup()
        logger.info("Store setup complete")
        app.state.store = store
        # Registry for background media-processing jobs (see src/api/media_jobs.py).
        app.state.media_jobs = {}
        checkpointer = AsyncPostgresSaver(app.state.pool)
        await checkpointer.setup()
        app.state.checkpointer = checkpointer
        # Publish the shared checkpointer so the deep agent (rebuilt each turn inside
        # the ``think`` node) can reuse it and make HITL ``interrupt``s durable.
        runtime_handles.set_deep_agent_checkpointer(checkpointer)
        app.state.graph = message_workflow.compile(
            store=store, checkpointer=checkpointer
        )
        logger.info("Application startup: lifecycle complete")
        yield
    finally:
        await pool.close()


app = FastAPI(
    title="Neural Nexus API",
    description="LangGraph-based API",
    version="1.0.0",
    lifespan=lifespan,
)


# Middleware for request metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time_ns()
    request_id = str(uuid.uuid4())

    request.state.request_id = request_id
    ACTIVE_REQUESTS.inc()

    try:
        if _is_auth_catch_all_target(method=request.method, path=request.url.path):
            try:
                await get_current_user(
                    request=request, api_key=request.headers.get("API-KEY")
                )
            except HTTPException as exc:
                return JSONResponse(
                    status_code=exc.status_code, content={"detail": exc.detail}
                )

        response = await call_next(request)
        latency_ms = (time_ns() - start_time) // 1_000_000

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=str(request.url.path),
            status=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method, endpoint=str(request.url.path)
        ).observe(latency_ms / 1000)
        API_RESPONSE_STATUS.labels(status=response.status_code).inc()

        return response
    except Exception as e:
        latency_ms = (time_ns() - start_time) // 1_000_000
        API_RESPONSE_STATUS.labels(status=500).inc()
        raise
    finally:
        ACTIVE_REQUESTS.dec()


@app.get("/metrics")
async def prometheus_metrics():
    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


@app.get("/*", include_in_schema=False)
async def documentation():
    return RedirectResponse(url="/docs")


@app.get("/", include_in_schema=False)
async def documentation():
    return RedirectResponse(url="/docs")


app.include_router(router=security_route)


def _checkout_line_items_for_tier(
    billing_config, tier: SubscriptionTier
) -> list[dict]:
    """Build Stripe Checkout line items: the flat base price plus each metered price.

    The licensed base price carries ``quantity=1``; metered prices are reported via
    usage events and therefore carry no quantity.
    """
    identifiers = billing_config.identifiers_for_tier(tier)
    line_items: list[dict] = [{"price": identifiers.base_price_id, "quantity": 1}]
    for meter, price_id in identifiers.metered_price_ids.items():
        line_items.append({"price": price_id})
    return line_items


_LIVE_SUBSCRIPTION_STATUSES = LIVE_SUBSCRIPTION_STATUSES


def _reactivate_subscription_for_user(stripe_client, subscription: dict) -> None:
    """Undo a pending period-end cancellation or scheduled downgrade.

    The caller has already retrieved a live subscription; this only releases any
    pending schedule and clears ``cancel_at_period_end``
    (``clear_pending_cancellation`` in the billing package). Stripe errors
    become HTTP 502.
    """
    subscription_id = subscription.get("id")
    try:
        clear_pending_cancellation(stripe_client, subscription)
    except HTTPException:
        raise
    except Exception as reactivate_error:
        logger.error(
            "Could not reactivate subscription %s: %s",
            subscription_id,
            reactivate_error,
        )
        raise HTTPException(
            status_code=502, detail="Could not reactivate the subscription."
        )


async def _change_subscription_tier_for_user(
    request: Request,
    current_user: dict,
    requested_tier: SubscriptionTier,
    currently_trialing: bool,
    pay_per_use: bool | None = None,
) -> dict:
    """Switch an existing subscription to a different tier via the Subscription API.

    The Stripe customer portal cannot switch plans that contain metered prices, so
    tier changes go through POST /subscribe. The direction decides the timing, per
    the retained/cleared usage rules (``plan_tier_change``):

    * **Upgrades** take effect immediately — every subscription item's price is
      replaced with the target tier's prices (base + metered) in one prorated
      update. Outside a trial the local usage window restarts so the new tier
      begins with a fresh allotment; while trialing the usage-period anchor is
      left alone so trial free-usage is retained.
    * **Downgrades** take effect at the period end via a Subscription Schedule —
      the user already paid for the higher tier through the period, so billing
      and allotment keep the higher tier until the boundary ("unused allotment
      continues"). Downgrading to free cancels the subscription at period end.

    ``pay_per_use`` optionally sets the overage flag in the same call (same
    validation as POST /set_pay_per_use). Same-tier requests are the caller's
    responsibility (``plan_subscribe_action`` routes them away first).
    """
    billing_config = getattr(request.app.state, "stripe_billing_config", None)
    if billing_config is None:
        raise HTTPException(
            detail="Billing is not configured; cannot change tier.", status_code=503
        )

    stripe_client = request.app.state.stripe
    status = await check_subscription_status(request=request, current_user=current_user)
    subscription_id = status.get("subscription_id")
    if not subscription_id:
        raise HTTPException(
            detail="No active subscription to change. Use POST /subscribe first.",
            status_code=404,
        )
    current_tier = tier_from_value(status.get("tier"))
    tier_change_plan = plan_tier_change(
        current_tier, requested_tier, currently_trialing=currently_trialing
    )

    if requested_tier == SubscriptionTier.FREE:
        # Downgrade to free = cancellation at period end; the
        # customer.subscription.deleted webhook pins the tier to free at the
        # boundary, so the paid allotment continues until then.
        try:
            subscription = stripe_client.Subscription.retrieve(subscription_id).to_dict()
            _release_pending_subscription_schedule(stripe_client, subscription)
            stripe_client.Subscription.modify(
                subscription_id, cancel_at_period_end=True
            )
        except Exception as cancel_error:
            logger.error("Could not schedule downgrade to free: %s", cancel_error)
            raise HTTPException(detail="Could not change tier.", status_code=502)
        if pay_per_use is not None:
            await _apply_pay_per_use_setting(request, current_user, pay_per_use)
        return {
            "message": "Subscription will end at the period boundary; you will drop to the free tier."
        }

    try:
        subscription = stripe_client.Subscription.retrieve(subscription_id).to_dict()
        _release_pending_subscription_schedule(stripe_client, subscription)
        existing_items = subscription.get("items", {}).get("data", [])
        target_price_ids = billing_config.identifiers_for_tier(
            requested_tier
        ).all_price_ids()

        if tier_change_plan.schedule_change_at_period_end:
            # Downgrade: keep the paid-for tier (billing AND allotment) until the
            # period boundary, then switch. Phase one restates the current items
            # through the period end; phase two runs the target tier for one
            # period and then releases, leaving the subscription running on the
            # target tier's items.
            schedule = stripe_client.SubscriptionSchedule.create(
                from_subscription=subscription_id
            ).to_dict()
            current_phase = (schedule.get("phases") or [{}])[0]
            _, current_period_end = _subscription_period_bounds(subscription)
            stripe_client.SubscriptionSchedule.modify(
                schedule["id"],
                end_behavior="release",
                phases=[
                    {
                        "items": [
                            {"price": (item.get("price") or {}).get("id")}
                            for item in existing_items
                        ],
                        "start_date": current_phase.get("start_date"),
                        "end_date": current_period_end
                        or current_phase.get("end_date"),
                    },
                    {
                        "items": [
                            {"price": target_price_id}
                            for target_price_id in target_price_ids
                        ],
                        # Stripe flexible billing rejects legacy ``iterations``;
                        # ``duration`` is the supported one-period phase shape.
                        "duration": {"interval": "month", "interval_count": 1},
                    },
                ],
            )
        else:
            # Upgrade: immediate switch. Stripe forbids changing a subscription
            # item between licensed and metered usage types, so the safe universal
            # move is: delete every existing item and add the target tier's prices
            # in the same atomic modify call. Tier price ids never overlap across
            # tiers (per-tier lookup keys), so a delete+add of the same price
            # cannot occur; the same-tier case is rejected by the planner above.
            #
            # clear_usage applies only to classic-billing-mode subscriptions with
            # legacy usage-record metered items; flexible-mode subscriptions (the
            # default for newly created ones) reject the parameter, and
            # Billing-Meter usage lives on the meter rather than the item, so
            # nothing needs clearing there.
            billing_mode_type = (subscription.get("billing_mode") or {}).get("type")
            supports_clear_usage = billing_mode_type != "flexible"
            items_payload: list[dict] = []
            for existing_item in existing_items:
                deletion: dict = {"id": existing_item["id"], "deleted": True}
                usage_type = (
                    (existing_item.get("price") or {}).get("recurring") or {}
                ).get("usage_type")
                if usage_type == "metered" and supports_clear_usage:
                    deletion["clear_usage"] = True
                items_payload.append(deletion)
            for target_price_id in target_price_ids:
                items_payload.append({"price": target_price_id})

            stripe_client.Subscription.modify(
                subscription_id,
                items=items_payload,
                proration_behavior="always_invoice",
            )
    except Exception as change_error:
        logger.error("Could not change subscription tier: %s", change_error)
        raise HTTPException(detail="Could not change tier.", status_code=502)

    if tier_change_plan.reset_usage_period_anchor:
        # Upgrade clears local usage: the new tier starts with a fresh allotment.
        await _write_usage_period_anchor(request, current_user)

    if pay_per_use is not None:
        await _apply_pay_per_use_setting(request, current_user, pay_per_use)

    if tier_change_plan.schedule_change_at_period_end:
        return {
            "message": (
                f"Subscription will switch to the {requested_tier.value} tier at the "
                "period boundary; your current allotment continues until then."
            )
        }
    return {"message": f"Subscription changed to the {requested_tier.value} tier."}


@app.post("/subscribe")
async def subscribe(
    request: Request,
    tier: SubscriptionTier = Query(
        default=SubscriptionTier.PRO, description="Chosen subscription tier."
    ),
    pay_per_use: Optional[bool] = Query(
        default=None,
        description="Optional pay-per-use flag applied when the action does not start checkout.",
    ),
    current_user: dict = Depends(get_current_user),
):
    """Single subscription entry point: checkout, tier change, or reactivate.

    Dispatches via ``plan_subscribe_action``:

    * **start_checkout** — no live subscription; create a Stripe Checkout session
      for the requested tier (base + metered prices). The free tier always
      collects a payment method (pay-per-use vehicle).
    * **no_change_required** — already on the requested tier with nothing pending;
      subscription and trial untouched.
    * **reactivate** — undo pending period-end cancellation or scheduled downgrade.
    * **change_tier** — switch tiers (upgrades immediate; downgrades at period end).
      While trialing, the usage-period anchor is not rewritten so trial free-usage
      is retained.
    * **reactivate_and_change_tier** — reactivate then change tier in one call.

    Cancellation for end users goes through GET /manage_subscription (billing
    portal). Anonymous users can never reach this endpoint.
    """
    verified_email = current_user.get("email_verified", None)
    if not verified_email:
        raise HTTPException(
            detail="Please verify your email before subscribing.", status_code=401
        )

    requested_tier = tier_from_value_or_400(
        tier.value if isinstance(tier, SubscriptionTier) else tier
    )

    billing_config = getattr(request.app.state, "stripe_billing_config", None)
    email = current_user.get("email")
    if billing_config is None:
        # Billing objects not provisioned yet — fall back to the legacy payment link.
        redirect_url = (
            f"{app.state.context.stripe_payment_url}"
            f"?locked_prefilled_email={email}"
        )
        return {
            "action": "start_checkout",
            "url": redirect_url,
            "message": "Follow this link to subscribe.",
        }

    stripe_client = request.app.state.stripe
    status = await check_subscription_status(
        request=request, current_user=current_user
    )
    current_status = status.get("status")
    current_tier = tier_from_value(status.get("tier"))
    cancel_at_period_end = False
    has_pending_downgrade_schedule = False
    subscription: dict | None = None
    subscription_id = status.get("subscription_id")
    if subscription_id and current_status in _LIVE_SUBSCRIPTION_STATUSES:
        try:
            subscription = stripe_client.Subscription.retrieve(subscription_id).to_dict()
        except Exception as retrieve_error:
            logger.error(
                "Could not retrieve subscription %s: %s",
                subscription_id,
                retrieve_error,
            )
            raise HTTPException(
                detail="Could not read the current subscription. Please try again.",
                status_code=502,
            )
        cancel_at_period_end = bool(subscription.get("cancel_at_period_end"))
        has_pending_downgrade_schedule = subscription_has_pending_downgrade_schedule(
            subscription
        )

    action = plan_subscribe_action(
        current_status,
        current_tier,
        requested_tier,
        cancel_at_period_end,
        has_pending_downgrade_schedule,
    )

    if action is SubscribeAction.NO_CHANGE_REQUIRED:
        if pay_per_use is not None:
            await _apply_pay_per_use_setting(request, current_user, pay_per_use)
        return {
            "action": "no_change_required",
            "message": f"Already subscribed to the {requested_tier.value} tier.",
            "subscription_status": status,
        }

    if action is SubscribeAction.REACTIVATE:
        assert subscription is not None
        _reactivate_subscription_for_user(stripe_client, subscription)
        return {
            "action": "reactivate",
            "message": "Subscription reactivated.",
            "cancel_at_period_end": False,
            "subscription_status": status,
        }

    currently_trialing = current_status == "trialing"
    if action is SubscribeAction.CHANGE_TIER:
        result = await _change_subscription_tier_for_user(
            request,
            current_user,
            requested_tier,
            currently_trialing=currently_trialing,
            pay_per_use=pay_per_use,
        )
        return {"action": "change_tier", **result}

    if action is SubscribeAction.REACTIVATE_AND_CHANGE_TIER:
        assert subscription is not None
        _reactivate_subscription_for_user(stripe_client, subscription)
        result = await _change_subscription_tier_for_user(
            request,
            current_user,
            requested_tier,
            currently_trialing=currently_trialing,
            pay_per_use=pay_per_use,
        )
        return {"action": "reactivate_and_change_tier", **result}

    # START_CHECKOUT
    customer_id = resolve_stripe_customer_id(current_user)
    tier_definition = TIER_DEFINITIONS[requested_tier]
    subscription_data: dict = {}
    # One free trial per Stripe customer, ever — a returning user re-selecting
    # a paid tier through Checkout pays from day one instead of harvesting a
    # second trial (resolve_checkout_trial_period_days).
    checkout_trial_period_days = resolve_checkout_trial_period_days(
        stripe_client, customer_id, tier_definition.trial_period_days
    )
    if checkout_trial_period_days > 0:
        subscription_data["trial_period_days"] = checkout_trial_period_days
        # Stripe forbids the legacy payment link's "pause" end behavior on
        # subscriptions containing metered prices; "cancel" achieves the same
        # product outcome — the customer.subscription.deleted webhook pins the
        # user back to the free tier when a trial lapses without a payment method.
        subscription_data["trial_settings"] = {
            "end_behavior": {"missing_payment_method": "cancel"}
        }

    base_url = str(request.base_url).rstrip("/")
    checkout_kwargs: dict = {
        "mode": "subscription",
        "line_items": _checkout_line_items_for_tier(billing_config, requested_tier),
        "success_url": f"{base_url}/verify_subscription_status",
        "cancel_url": f"{base_url}/docs",
        "metadata": {
            "auth0_user_id": current_user.get("user_id", ""),
            "neural_nexus_tier": requested_tier.value,
        },
    }
    if requested_tier == SubscriptionTier.FREE:
        # The first free-tier invoice totals $0, which would let Checkout skip
        # payment-method collection — but a payment method is the entire point
        # of the free subscription (billing pay-per-use overage), so force
        # collection.
        checkout_kwargs["payment_method_collection"] = "always"
    if subscription_data:
        checkout_kwargs["subscription_data"] = subscription_data
    if customer_id:
        checkout_kwargs["customer"] = customer_id
    elif email:
        checkout_kwargs["customer_email"] = email

    try:
        session = stripe_client.checkout.Session.create(**checkout_kwargs)
    except Exception as checkout_error:
        logger.error("Could not create Checkout session: %s", checkout_error)
        raise HTTPException(
            detail="Could not start checkout. Please try again.", status_code=502
        )
    return {
        "action": "start_checkout",
        "url": session["url"],
        "message": "Follow this link to subscribe.",
    }


# Moved to src/anubis/utils/billing/subscription_lifecycle.py so the auth layer
# (delete-and-re-signup adoption) can share the same logic; the underscore
# aliases keep the existing call sites in this module unchanged.
_subscription_period_bounds = subscription_period_bounds
_release_pending_subscription_schedule = release_pending_subscription_schedule


async def _write_usage_period_anchor(request: Request, current_user: dict) -> None:
    """Restart the user's local usage window at this instant (upgrade semantics).

    Writes ``app_metadata.usage_period_anchor`` so allotment gating and the
    subscription-status endpoint count usage from the tier change forward —
    "usage cleared on upgrade". Also updates the in-memory user so the current
    request already sees the fresh window.
    """
    anchor_value = datetime.now(UTC).isoformat()
    await update_user_app_metadata_fields(
        request,
        current_user.get("user_id"),
        {"usage_period_anchor": anchor_value},
    )
    current_user.setdefault("app_metadata", {})["usage_period_anchor"] = anchor_value


async def _apply_pay_per_use_setting(
    request: Request, current_user: dict, enabled: bool
) -> None:
    """Persist the explicit pay-per-use flag, requiring a payment method to enable.

    Disabling is always allowed (a user capping their own spend). Enabling
    requires a Stripe customer with a payment method on file, because pay-per-use
    means the graduated metered price bills overage — without a card there is
    nothing to bill and the allotment gate would silently become an unbounded
    free pass. Raises HTTPException with actionable guidance when the
    requirement is not met.
    """
    if enabled:
        stripe_client = request.app.state.stripe
        customer_id = resolve_stripe_customer_id(current_user)
        if not customer_id:
            status = await check_subscription_status(
                request=request, current_user=current_user
            )
            customer_id = status.get("customer_id")
        if not customer_id:
            raise HTTPException(
                status_code=402,
                detail=(
                    "Pay-per-use requires a payment method on file. Subscribe first "
                    "(POST /subscribe?tier=free adds a card without a paid plan)."
                ),
            )
        try:
            customer_document = stripe_client.Customer.retrieve(
                customer_id, expand=["invoice_settings.default_payment_method"]
            ).to_dict()
            payment_methods = []
            if not customer_has_payment_method(customer_document):
                payment_methods = (
                    stripe_client.PaymentMethod.list(customer=customer_id, limit=1)
                    .to_dict()
                    .get("data", [])
                )
        except Exception as customer_error:
            logger.error(
                "Could not verify payment method for customer %s: %s",
                customer_id,
                customer_error,
            )
            raise HTTPException(
                status_code=502, detail="Could not verify payment method with Stripe."
            )
        if not customer_has_payment_method(customer_document, payment_methods):
            raise HTTPException(
                status_code=402,
                detail=(
                    "No payment method on file. Add one via GET /manage_subscription "
                    "before enabling pay-per-use."
                ),
            )

    updated = await update_user_app_metadata_fields(
        request, current_user.get("user_id"), {"pay_per_use_enabled": enabled}
    )
    if not updated:
        raise HTTPException(
            status_code=502, detail="Could not persist the pay-per-use setting."
        )
    current_user.setdefault("app_metadata", {})["pay_per_use_enabled"] = enabled


@app.post("/set_pay_per_use")
async def set_pay_per_use(
    request: Request,
    enabled: bool = True,
    current_user: dict = Depends(get_current_user),
):
    """Enable or disable billing overage past the monthly allotment (pay-per-use).

    With pay-per-use enabled, usage past a meter's allotment continues and the
    tier's graduated metered price bills the overage; disabled, requests are
    refused with HTTP 402 at the allotment. Enabling requires a payment method on
    file (subscribe — the free tier's $0 subscription qualifies — or add a card
    through the billing portal). Trialing users with a card may enable pay-per-use.
    """
    await _apply_pay_per_use_setting(request, current_user, enabled)
    return {"pay_per_use_enabled": enabled}


@app.get("/manage_subscription")
async def manage_subscription(
    request: Request, current_user: dict = Depends(get_current_user)
):
    """Return a Stripe billing-portal session URL for this customer.

    A billing-portal session is created per request, so the link works in both
    the test and live Stripe environments (the static
    STRIPE_MANAGE_SUBSCRIPTION_URL login page remains only as a degraded-mode
    fallback when billing objects are not provisioned). The portal covers
    invoices, payment methods, cancellation, and billing information; tier
    switching stays on POST /subscribe because the portal cannot switch plans
    that contain metered prices.
    """
    billing_config = getattr(request.app.state, "stripe_billing_config", None)
    if billing_config is None:
        return {
            "url": request.app.state.context.stripe_manage_subscription_url,
            "message": "Follow this link to manage your subscription.",
        }

    stripe_client = request.app.state.stripe
    customer_id = resolve_stripe_customer_id(current_user)
    if not customer_id:
        status = await check_subscription_status(
            request=request, current_user=current_user
        )
        customer_id = status.get("customer_id")
    if not customer_id:
        raise HTTPException(
            status_code=404,
            detail="No billing account yet. Use POST /subscribe to create one.",
        )

    portal_kwargs: dict = {
        "customer": customer_id,
        "return_url": f"{str(request.base_url).rstrip('/')}/docs",
    }
    portal_configuration_id = getattr(billing_config, "portal_configuration_id", None)
    if portal_configuration_id:
        portal_kwargs["configuration"] = portal_configuration_id
    try:
        session = stripe_client.billing_portal.Session.create(**portal_kwargs).to_dict()
    except Exception as portal_error:
        logger.error("Could not create billing-portal session: %s", portal_error)
        raise HTTPException(
            status_code=502, detail="Could not open the billing portal. Try again."
        )
    return {
        "url": session["url"],
        "message": "Follow this link to manage your subscription.",
    }


def _auth0_user_id_for_customer(stripe_client, customer_id: Optional[str]) -> Optional[str]:
    """Look up the Auth0 user id stored on a Stripe customer's metadata."""
    if not customer_id:
        return None
    try:
        customer = stripe_client.Customer.retrieve(customer_id).to_dict()
        return customer.get("metadata", {}).get("auth0_user_id") or None
    except Exception as lookup_error:
        logger.error("Could not retrieve Stripe customer %s: %s", customer_id, lookup_error)
        return None


async def _handle_stripe_event(
    request: Request, stripe_client, event_type: str, data_object: dict
) -> None:
    """Sync tier/status into Auth0 for the subscription-lifecycle events we care about."""
    if event_type == "checkout.session.completed":
        customer_id = data_object.get("customer")
        auth0_user_id = data_object.get("metadata", {}).get(
            "auth0_user_id"
        ) or _auth0_user_id_for_customer(stripe_client, customer_id)
        subscription_id = data_object.get("subscription")
        tier = data_object.get("metadata", {}).get("neural_nexus_tier", "free")
        status_value = "active"
        current_period_start = None
        current_period_end = None
        cancel_at_period_end = False
        if subscription_id:
            try:
                subscription = stripe_client.Subscription.retrieve(
                    subscription_id
                ).to_dict()
                status_value = subscription.get("status", "active")
                tier = _tier_from_subscription(stripe_client, subscription)
                current_period_start, current_period_end = _subscription_period_bounds(
                    subscription
                )
                cancel_at_period_end = bool(subscription.get("cancel_at_period_end"))
                if status_value == "trialing" and customer_id:
                    # A Checkout-granted trial must stamp the same one-trial-
                    # per-customer flag the auto-enrollment path stamps, so a
                    # later delete-and-re-signup can never harvest a second
                    # trial through Checkout.
                    stripe_client.Customer.modify(
                        customer_id,
                        metadata={"neural_nexus_trial_used": "true"},
                    )
            except Exception as retrieve_error:
                logger.error("Could not retrieve subscription: %s", retrieve_error)
        await update_user_subscription_status(
            request,
            auth0_user_id,
            {
                "status": status_value,
                "subscription_id": subscription_id,
                "customer_id": customer_id,
                "email": data_object.get("customer_details", {}).get("email"),
                "tier": tier,
                "current_period_start": current_period_start,
                "current_period_end": current_period_end,
                "cancel_at_period_end": cancel_at_period_end,
            },
        )
        # First checkout starts a fresh local usage window (a free→paid upgrade,
        # or the free $0 subscription's first period).
        await update_user_app_metadata_fields(
            request,
            auth0_user_id,
            {"usage_period_anchor": datetime.now(UTC).isoformat()},
        )

    elif event_type in (
        "customer.subscription.created",
        "customer.subscription.updated",
        "customer.subscription.deleted",
    ):
        # ``created`` flows through the same non-deleted sync path as
        # ``updated`` so a subscription created outside Checkout (for example
        # a server-side ``Subscription.create``) syncs its tier/status into
        # Auth0 immediately instead of waiting for a later ``updated`` bump.
        customer_id = data_object.get("customer")
        auth0_user_id = _auth0_user_id_for_customer(stripe_client, customer_id)
        current_period_start, current_period_end = _subscription_period_bounds(
            data_object
        )
        if event_type == "customer.subscription.deleted":
            tier = "free"
            status_value = "canceled"
        else:
            tier = _tier_from_subscription(stripe_client, data_object)
            status_value = data_object.get("status", "active")
        await update_user_subscription_status(
            request,
            auth0_user_id,
            {
                "status": status_value,
                "subscription_id": data_object.get("id"),
                "customer_id": customer_id,
                "email": None,
                "tier": tier,
                "current_period_start": current_period_start,
                "current_period_end": current_period_end,
                "cancel_at_period_end": bool(data_object.get("cancel_at_period_end")),
            },
        )
        if event_type == "customer.subscription.deleted":
            # A stale explicit pay-per-use flag must not grant overage after the
            # subscription (the billing vehicle) is gone.
            await update_user_app_metadata_fields(
                request, auth0_user_id, {"pay_per_use_enabled": False}
            )

    elif event_type == "invoice.payment_failed":
        customer_id = data_object.get("customer")
        auth0_user_id = _auth0_user_id_for_customer(stripe_client, customer_id)
        await update_user_subscription_status(
            request,
            auth0_user_id,
            {
                "status": "past_due",
                "subscription_id": data_object.get("subscription"),
                "customer_id": customer_id,
                "email": None,
                "tier": "free",
            },
        )


def _resolve_stripe_webhook_secret(context) -> Optional[str]:
    """Return the webhook signing secret from env, or from the CLI-shared file.

    Prod sets ``STRIPE_WEBHOOK_SECRET`` once from a Dashboard "Your account"
    endpoint. Local docker-compose leaves that empty and the ``stripe-cli``
    service writes ``STRIPE_WEBHOOK_SECRET_FILE`` on each start — read on every
    request so the API can come up before the CLI finishes printing the secret.
    """
    explicit = getattr(context, "stripe_webhook_secret", None)
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    secret_file = getattr(context, "stripe_webhook_secret_file", None)
    if not secret_file:
        return None
    try:
        with open(secret_file, encoding="utf-8") as handle:
            value = handle.read().strip()
        return value or None
    except OSError:
        return None


@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    """Verify and process Stripe subscription-lifecycle webhooks.

    This is the real-time source of truth for a user's tier/status: it keeps the
    cached ``app_metadata.subscription_status`` in sync on checkout completion,
    subscription updates/cancellation, and payment failure. The endpoint verifies
    the Stripe signature against ``stripe_webhook_secret`` before trusting any data.
    """
    stripe_client = request.app.state.stripe
    webhook_secret = _resolve_stripe_webhook_secret(request.app.state.context)
    if not webhook_secret:
        logger.error(
            "Stripe webhook secret not configured "
            "(set STRIPE_WEBHOOK_SECRET or wait for STRIPE_WEBHOOK_SECRET_FILE); "
            "rejecting webhook."
        )
        raise HTTPException(status_code=503, detail="Webhook not configured.")

    payload = await request.body()
    signature = request.headers.get("stripe-signature")
    try:
        event = stripe_client.Webhook.construct_event(
            payload, signature, webhook_secret
        )
    except Exception as verify_error:
        logger.error("Stripe webhook verification failed: %s", verify_error)
        raise HTTPException(status_code=400, detail="Invalid webhook signature.")

    # construct_event returns a StripeObject (not a dict subclass in stripe-python
    # 15, so ``.get`` is unavailable); convert once here so the handler works on
    # plain dicts.
    event_document = event.to_dict()
    try:
        await _handle_stripe_event(
            request,
            stripe_client,
            event_document["type"],
            event_document["data"]["object"],
        )
    except Exception as handling_error:
        # Return 200 so Stripe does not retry indefinitely on a non-transient bug;
        # the event is logged for manual reconciliation.
        logger.error(
            "Error handling Stripe event %s: %s",
            event_document["type"],
            handling_error,
        )

    return {"received": True}


@app.get("/verify_subscription_status")
async def verify_subscription_status(
    request: Request, current_user: dict = Depends(get_current_user)
):
    """Return subscription status plus per-meter allotment, usage, and remaining.

    The single endpoint a customer portal polls: subscription identity/status,
    the pay-per-use flag, the current usage period bounds, and — for every meter
    the tier grants — the period allotment, usage to date (from the local
    ``api_metrics`` accounting), remaining budget, and the overage rate that
    applies when pay-per-use is enabled.
    """
    status = await check_subscription_status(request=request, current_user=current_user)
    tier = tier_from_value(status.get("tier"))

    context = GlobalContext()
    usage_period_days = int(context.usage_period_days or 0)
    period_start = resolve_usage_period_start_for_user(current_user)

    app_metadata = (current_user or {}).get("app_metadata") or {}
    cached_subscription_status = app_metadata.get("subscription_status") or {}
    cached_period_end = cached_subscription_status.get("current_period_end")
    if cached_period_end:
        try:
            period_end = datetime.fromtimestamp(int(cached_period_end), tz=UTC)
        except (TypeError, ValueError, OSError):
            period_end = resolve_usage_period_end(period_start, usage_period_days)
    else:
        period_end = resolve_usage_period_end(period_start, usage_period_days)

    usage_by_meter = await fetch_usage_by_meter_since(
        getattr(request.app.state, "pool", None),
        resolve_metering_user_id(current_user),
        period_start,
    )

    # Trial-aware per-meter view: within a free-trial window the trial tier's
    # allotment is a floor over the current tier's (resolve_effective_monthly_
    # allotment), and any meter the trial tier grants but the current tier does
    # not stays visible until trial_end. The meter order is the current tier's
    # definition order, then any trial-only meters, kept deterministic via an
    # insertion-ordered dict.
    trial_context = resolve_trial_context(current_user)
    ordered_meters: dict = dict.fromkeys(
        TIER_DEFINITIONS[tier].meter_allotments.keys()
    )
    if trial_context is not None:
        for trial_meter in TIER_DEFINITIONS[
            trial_context.trial_tier
        ].meter_allotments:
            ordered_meters.setdefault(trial_meter, None)

    meters: dict = {}
    for meter in ordered_meters:
        allotment = resolve_effective_monthly_allotment(tier, meter, trial_context)
        if allotment is None:
            continue
        used_to_date = int(usage_by_meter.get(meter.value, 0))
        meters[meter.value] = {
            "monthly_allotment": allotment.monthly_allotment,
            "used_to_date": used_to_date,
            "remaining": max(0, allotment.monthly_allotment - used_to_date),
            "overage_price_per_million": allotment.overage_price_per_million,
            "overage_price_per_unit_usd": allotment.overage_price_per_unit_usd,
        }

    return {
        "status": status.get("status"),
        "tier": tier.value,
        "subscription_id": status.get("subscription_id"),
        "customer_id": status.get("customer_id"),
        "email": status.get("email"),
        "pay_per_use_enabled": resolve_pay_per_use_enabled(current_user),
        "cancel_at_period_end": bool(
            cached_subscription_status.get("cancel_at_period_end")
        ),
        "usage_period_start": period_start.isoformat(),
        "usage_period_end": period_end.isoformat(),
        "meters": meters,
    }


async def _demote_other_personal_avatars(
    client: Any, user_id: str, keep_assistant_id: Optional[str]
) -> None:
    """Enforce "at most one personal avatar per user" by demoting the rest.

    A user may flag exactly one avatar as their ``PERSONAL_AVATAR_OF_THE_CREATOR``
    (the only avatar that can reach their desktop MCP server and future personal
    analytics). When a new avatar is flagged, any *other* avatar of the same user
    that still holds the flag is cleared, so the newest choice wins without an
    error. ``keep_assistant_id`` is the avatar just flagged (never demoted).
    """
    try:
        owned_avatars = await client.assistants.search(
            metadata={"user_id": user_id}, limit=1000
        )
    except Exception:
        logger.warning(
            "Could not enumerate avatars to demote prior personal avatar for user %s",
            user_id,
            exc_info=True,
        )
        return

    for avatar in owned_avatars or []:
        avatar_id = avatar.get("assistant_id")
        metadata = avatar.get("metadata") or {}
        if avatar_id == keep_assistant_id:
            continue
        if metadata.get("is_personal_avatar_of_creator") is True:
            try:
                await client.assistants.update(
                    assistant_id=avatar_id,
                    metadata={"is_personal_avatar_of_creator": False},
                )
            except Exception:
                logger.warning(
                    "Failed to demote prior personal avatar %s for user %s",
                    avatar_id,
                    user_id,
                    exc_info=True,
                )


@app.post("/create_avatar")
async def create_avatar(
    name: str,
    description: Optional[str] = None,
    is_public: bool = False,
    is_personal_avatar_of_creator: bool = False,
    current_user: dict = Depends(get_current_user),
):

    # If the avatar is of the individual, then the avatar is allowed to be made public.
    # Reference image, audio, and third-party authenticated account is required to create a shareable avatar. Limited to one shareable avatar of themselves.
    # Include reference image, reference audio

    logger.info(f"breakpoint")
    context = app.state.context

    if current_user["identities"][0]["user_id"] == context.anonymous_user_id:
        return JSONResponse(
            content="User must be logged in to create avatars.", status_code=400
        )

    try:
        assistant_id = str(uuid4())
        user_id = current_user["identities"][0]["user_id"]
        metadata = {
            "user_id": user_id,
            "is_public": False,
            "is_personal_avatar_of_creator": is_personal_avatar_of_creator,
        }

        if user_id == context.admin_user_id:
            metadata["is_public"] = is_public

        token = current_user["API_KEY"]
        headers = {"API-KEY": f"{token}"}
        client = get_client(headers=headers)

        create_avatar_response = await client.assistants.create(
            graph_id="Anubis",
            description=description,
            name=name,
            assistant_id=assistant_id,
            metadata=metadata,
        )

        # store the creator of the assistant
        # The langgraph_sdk StoreClient exposes put_item (HTTP API), not the
        # BaseStore aput method used elsewhere on in-process store objects.
        await client.store.put_item(
            (assistant_id, "creator_id"), key="creator_id", value={"value": user_id}
        )

        # At most one personal avatar per user: flagging this one demotes any other.
        if is_personal_avatar_of_creator:
            await _demote_other_personal_avatars(
                client, user_id, keep_assistant_id=assistant_id
            )

        return JSONResponse(content=create_avatar_response, status_code=200)
    except Exception as creation_error:
        logger.exception(f"Error creating avatar {name}")
        raise HTTPException(
            detail=f"Error creating avatar {name}: {creation_error}", status_code=500
        )


@app.post("/share_avatar")
async def share_avatar(
    assistant_id: str,
    is_public: bool = True,
    current_user: dict = Depends(get_current_user),
):
    context = app.state.context
    user_id = current_user["identities"][0]["user_id"]

    if user_id == context.admin_user_id:
        """verify users are creating avatars of their own likeness in the future"""
        metadata = {"is_public": is_public}

    # Only admins may share avatars;
    # Users will authenticate and share avatars in the near future.
    if user_id == context.admin_user_id:
        try:
            token = current_user["API_KEY"]
            client = get_client(headers={"API-KEY": f"{token}"})
            result = await client.assistants.update(
                assistant_id=assistant_id, metadata=metadata
            )
            return JSONResponse(result, status_code=200)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error during update of sharing avatar: {e}"
            )
    raise HTTPException(
        status_code=401, detail="Users may only share avatars of themselves."
    )


# @app.post("/user_is_creator")
# async def user_is_creator(
#     assistant_id: str,
#     current_user: dict = Depends(get_current_user),
# ):
#     """Used to establish creator in vectorstore due to update in code. Unnecessary, already implemented.

#     Args:
#         assistant_id (str): _description_
#         current_user (dict, optional): _description_. Defaults to Depends(get_current_user).

#     Raises:
#         HTTPException: _description_
#         HTTPException: _description_

#     Returns:
#         _type_: _description_
#     """
#     context = app.state.context
#     user_id = current_user["identities"][0]["user_id"]
#     if user_id == context.admin_user_id:
#         try:
#             token = current_user["API_KEY"]
#             client = get_client(headers={"API-KEY": f"{token}"})
#             namespace = (assistant_id, 'creator_id')
#             await client.store.put_item(namespace, key='creator_id', value={"value": user_id}) 
#             return JSONResponse(content="stored creator_id", status_code=200)
#         except Exception as e:
#             raise HTTPException(
#                 status_code=500, detail=f"Error during update of sharing avatar: {e}"
#             )
        
#     raise HTTPException(
#         status_code=401, detail="Users may only share avatars of themselves."
#     )

@app.patch("/modify_avatar")
async def modify_avatar(
    request: Request,
    assistant_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    new_avatar_name: Optional[str] = None,
    new_avatar_description: Optional[str] = None,
    is_personal_avatar_of_creator: bool = False,
):
    # Avatar name changes also need to be applied to the db for consistent identities
    logger.info("breakpoint")
    update_personal_avatar_flag = (
        "is_personal_avatar_of_creator" in request.query_params
    )
    if not assistant_id:
        raise HTTPException(
            detail="Supply assistant_id for the assistant to modify.", status_code=400
        )
    if (
        not new_avatar_name
        and not new_avatar_description
        and not update_personal_avatar_flag
    ):
        raise HTTPException(
            detail=(
                "Supply at least one of: new avatar name, new avatar description, "
                "or is_personal_avatar_of_creator."
            ),
            status_code=400,
        )

    if not current_user:
        raise HTTPException(
            content="User must be logged in to modify avatar avatars.", status_code=401
        )

    token = current_user["API_KEY"]
    client = get_client(headers={"API-KEY": f"{token}"})

    # Build a single update from only the supplied fields. ``metadata`` is merged
    # (not replaced) by the assistant update, so sending only the flag preserves
    # ``user_id`` / ``is_public`` — matching the share_avatar merge pattern above.
    update_kwargs: dict[str, Any] = {
        "graph_id": "Anubis",
        "assistant_id": assistant_id,
    }
    if new_avatar_name:
        update_kwargs["name"] = new_avatar_name
    if new_avatar_description:
        update_kwargs["description"] = new_avatar_description
    if update_personal_avatar_flag:
        update_kwargs["metadata"] = {
            "is_personal_avatar_of_creator": is_personal_avatar_of_creator
        }

    try:
        result = await client.assistants.update(**update_kwargs)
    except Exception:
        raise HTTPException(status_code=500, detail="Error updating assistant.")

    # At most one personal avatar per user: flagging this one demotes any other.
    if update_personal_avatar_flag and is_personal_avatar_of_creator:
        user_id = current_user["identities"][0]["user_id"]
        await _demote_other_personal_avatars(
            client, user_id, keep_assistant_id=assistant_id
        )

    return JSONResponse(content=result, status_code=200)


@app.post("/disconnect_mcp")
async def disconnect_mcp(
    current_user: dict = Depends(get_current_user),
):
    """Forget the user's saved MCP data-server connection (disconnect).

    Deletes the single per-user connection record. The next turn on any owned
    avatar re-enters the discovery/consent flow, so the user can re-connect
    (and re-bind the connection to whichever avatar they choose). There is no
    enable/disable switch — removing the connection is the disconnect.
    """
    from src.anubis.utils.tools.data_analysis.backend import (
        mcp_connection_namespace,
    )
    from src.anubis.utils.tools.data_analysis.discovery import CONNECTION_KEY

    user_id = current_user["identities"][0]["user_id"]
    token = current_user["API_KEY"]
    client = get_client(headers={"API-KEY": f"{token}"})
    try:
        await client.store.delete_item(
            list(mcp_connection_namespace(user_id)), key=CONNECTION_KEY
        )
        return JSONResponse(
            content={"disconnected": True, "user_id": user_id}, status_code=200
        )
    except Exception as disconnect_error:
        raise HTTPException(
            status_code=500,
            detail=f"Error disconnecting MCP server: {disconnect_error}",
        )


# ---------------------------------------------------------------------------
# MCP local-daemon relay + registration (see anubis-mcp-server-ubuntu).
#
# The user's local MCP server has no inbound port. It reaches this API in one of
# two directions:
#   - relay (default): one outbound WebSocket to ``/mcp/relay`` that tunnels HTTP
#     both ways (this API sends ``proxy`` frames; the daemon replays them against
#     its localhost MCP server and returns ``proxy_response`` frames);
#   - registration: ``POST /mcp/register`` / ``/mcp/heartbeat`` / ``/mcp/unregister``
#     announce presence + reachable URL (relay presence fallback, and the only
#     channel for the tunnel/local advanced modes).
# The user's API-KEY authenticates the daemon→API direction; a per-device secret
# (``Authorization: Bearer``) authenticates the API→MCP direction.
# ---------------------------------------------------------------------------


@app.websocket("/mcp/relay")
async def mcp_relay(websocket: WebSocket):
    """Accept a local MCP daemon's outbound relay socket and tunnel HTTP over it.

    Authenticates the user from the ``API-KEY`` handshake header, consumes the
    daemon's first ``register`` frame (device id + secret + announced server
    metadata), records the live session in the in-process relay registry, then
    forwards every subsequent ``proxy_response`` frame to the awaiting
    ``proxy_request`` call. The registry is what the graph's ``mcp_discovery``
    node and the ``/mcp/relay/{device_id}`` bridge read to reach this device.
    """
    from src.anubis.utils.tools.data_analysis import relay as relay_registry

    api_key = websocket.headers.get("API-KEY")
    if not api_key:
        await websocket.close(code=1008)
        return
    try:
        # A ``WebSocket`` carries ``.app`` just like a ``Request``, which is all
        # ``get_user_with_api_key`` needs (httpx client + management token).
        user = await get_user_with_api_key(api_key, websocket)  # type: ignore[arg-type]
    except Exception:
        logger.warning("MCP relay authentication error", exc_info=True)
        user = None
    if not user:
        await websocket.close(code=1008)
        return

    user_id = user["identities"][0]["user_id"]
    await websocket.accept()

    device_id: str | None = None
    try:
        register_message = json.loads(await websocket.receive_text())
        if register_message.get("type") != relay_registry.FRAME_REGISTER:
            await websocket.close(code=1008)
            return
        device_id = register_message.get("device_id")
        device_secret = register_message.get("device_secret")
        if not device_id or not device_secret:
            await websocket.close(code=1008)
            return

        relay_registry.register_session(
            device_id=device_id,
            user_id=user_id,
            device_secret=device_secret,
            server_name=register_message.get("server_name") or "Ubuntu-OS-Filesystem",
            allowed_roots=tuple(register_message.get("allowed_roots") or []),
            websocket=websocket,
        )
        await websocket.send_text(
            json.dumps(
                {"type": relay_registry.FRAME_REGISTERED, "device_id": device_id}
            )
        )

        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                continue
            relay_registry.handle_incoming(device_id, message)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.warning(
            "MCP relay socket error for device %s", device_id, exc_info=True
        )
    finally:
        if device_id:
            relay_registry.drop_session(device_id, websocket)


@app.api_route("/mcp/relay/{device_id}", methods=["GET", "POST", "DELETE"])
async def mcp_relay_bridge(device_id: str, request: Request):
    """Tunnel one MCP HTTP call to a device over its live relay socket.

    This is the URL stored as a relay connection's ``McpConnection.url``; the
    avatar's ``MultiServerMCPClient`` calls it exactly like a normal
    streamable-HTTP MCP endpoint. Access is gated by the per-device Bearer
    secret (the publicly routable path must not be callable by anyone who merely
    guesses a device id). The request is re-pathed to the daemon's local MCP
    endpoint (``/mcp``) — the daemon forwards to ``local_mcp_url + path``.

    A streamable-HTTP client also opens a *standalone* ``GET`` to this endpoint
    to receive server-initiated (server→client) messages — an unbounded
    Server-Sent-Events stream. The relay tunnels discrete request/response
    pairs, not open-ended streams: a tunneled ``GET`` would block the daemon's
    single WebSocket message loop reading a body that never ends, wedging the
    whole relay for that device. The MCP specification allows a server to
    decline the server-push stream by answering the standalone ``GET`` with
    ``405 Method Not Allowed``; the client then operates request/response-only
    (every filesystem tool call is a ``POST`` whose response is finite and
    buffered), which the relay fully supports. So short-circuit ``GET`` here and
    never forward it over the socket.
    """
    from src.anubis.utils.tools.data_analysis import relay as relay_registry

    if request.method == "GET":
        return JSONResponse(
            content={"error": "Server-push stream is not offered over the relay."},
            status_code=405,
        )

    session = relay_registry.get_session(device_id)
    if session is None:
        return JSONResponse(
            content={"error": "MCP relay device is offline."}, status_code=503
        )

    if request.headers.get("Authorization") != f"Bearer {session.device_secret}":
        return JSONResponse(content={"error": "Unauthorized."}, status_code=401)

    forwarded_headers: dict[str, str] = {}
    for header_name in (
        "content-type",
        "accept",
        "mcp-session-id",
        "mcp-protocol-version",
    ):
        header_value = request.headers.get(header_name)
        if header_value is not None:
            forwarded_headers[header_name] = header_value
    forwarded_headers["Authorization"] = f"Bearer {session.device_secret}"

    try:
        status_code, response_headers, response_body = await relay_registry.proxy_request(
            device_id,
            method=request.method,
            path=relay_registry.LOCAL_MCP_PATH,
            headers=forwarded_headers,
            body=await request.body(),
            timeout_seconds=float(
                app.state.context.data_analysis_relay_request_timeout_seconds
            ),
        )
    except TimeoutError:
        return JSONResponse(
            content={"error": "MCP relay timed out."}, status_code=504
        )
    except Exception as proxy_error:
        return JSONResponse(
            content={"error": f"MCP relay failed: {proxy_error}"}, status_code=502
        )

    # Preserve only the MCP-meaningful response headers; the media type governs
    # how the streamable-HTTP client parses the body (JSON vs. text/event-stream).
    passthrough_headers = {
        key: value
        for key, value in response_headers.items()
        if key.lower() in ("mcp-session-id", "mcp-protocol-version")
    }
    return Response(
        content=response_body,
        status_code=status_code,
        headers=passthrough_headers,
        media_type=response_headers.get("content-type"),
    )


@app.post("/mcp/register")
async def mcp_register(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Record a local MCP daemon's pushed presence for the authenticated user.

    Stores a single per-user ``pending_consent`` registration; the next turn on
    the user's personal avatar reads it (``mcp_discovery``) and offers the
    connection. This does not itself connect anything — user consent produces
    the separate, avatar-bound ``mcp_connection`` record.
    """
    from src.anubis.utils.tools.data_analysis.backend import (
        mcp_registration_namespace,
    )
    from src.anubis.utils.tools.data_analysis.discovery import REGISTRATION_KEY

    body = await request.json()
    user_id = current_user["identities"][0]["user_id"]
    token = current_user["API_KEY"]
    client = get_client(headers={"API-KEY": f"{token}"})

    connection_mode = body.get("connection_mode") or "relay"
    device_id = body.get("device_id")
    # In relay mode the daemon's announced mcp_url may point at a different
    # host (e.g. production) than the API instance that accepted the register
    # call. Always rewrite to this request's own relay bridge so consent and
    # tool calls stay on the same process that holds the WebSocket.
    mcp_url = body.get("mcp_url")
    if connection_mode == "relay" and device_id:
        mcp_url = f"{str(request.base_url).rstrip('/')}/mcp/relay/{device_id}"

    record = {
        "status": "pending_consent",
        "connection_mode": connection_mode,
        "server_name": body.get("server_name") or "Ubuntu-OS-Filesystem",
        # Every mode is driven as a streamable-HTTP client; in relay mode the
        # ``mcp_url`` points at this API's own ``/mcp/relay/<device_id>`` bridge.
        "transport": "streamable_http",
        "device_id": device_id,
        "device_secret": body.get("device_secret"),
        "mcp_url": mcp_url,
        "discovery_url": body.get("discovery_url"),
        "allowed_roots": body.get("allowed_roots") or [],
        "last_seen_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        await client.store.put_item(
            list(mcp_registration_namespace(user_id)),
            key=REGISTRATION_KEY,
            value=record,
        )
        return JSONResponse(
            content={"registered": True, "device_id": record["device_id"]},
            status_code=200,
        )
    except Exception as register_error:
        raise HTTPException(
            status_code=500,
            detail=f"Error registering MCP server: {register_error}",
        )


@app.post("/mcp/heartbeat")
async def mcp_heartbeat(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Refresh a registration's presence fields so it keeps counting as online.

    Heartbeats also sync ``device_id`` / ``mcp_url`` / ``connection_mode`` /
    ``device_secret`` from the daemon body. Without that, a new local daemon
    (different device id) can keep an older registration's ``last_seen_at``
    fresh while the live relay socket belongs to a different device — leaving
    ``resolve_available_connection`` unable to see the online session.
    """
    from src.anubis.utils.tools.data_analysis.backend import (
        mcp_registration_namespace,
    )
    from src.anubis.utils.tools.data_analysis.discovery import REGISTRATION_KEY

    user_id = current_user["identities"][0]["user_id"]
    token = current_user["API_KEY"]
    client = get_client(headers={"API-KEY": f"{token}"})

    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    namespace = list(mcp_registration_namespace(user_id))
    try:
        existing = await client.store.get_item(namespace, key=REGISTRATION_KEY)
    except Exception:
        # A missing item surfaces as an error from the HTTP store client; a
        # heartbeat before/without a registration is simply a no-op.
        existing = None
    record = (existing or {}).get("value") if isinstance(existing, dict) else None
    if not record:
        # No prior registration (e.g. endpoint was down during startup);
        # a heartbeat alone is not enough to reconstruct one.
        return JSONResponse(content={"acknowledged": False}, status_code=200)

    try:
        record["last_seen_at"] = datetime.now(timezone.utc).isoformat()
        if body.get("device_id"):
            record["device_id"] = body["device_id"]
        if body.get("connection_mode"):
            record["connection_mode"] = body["connection_mode"]
        if body.get("device_secret"):
            record["device_secret"] = body["device_secret"]
        connection_mode = record.get("connection_mode") or "relay"
        device_id = record.get("device_id")
        if connection_mode == "relay" and device_id:
            record["mcp_url"] = (
                f"{str(request.base_url).rstrip('/')}/mcp/relay/{device_id}"
            )
        elif body.get("mcp_url"):
            record["mcp_url"] = body["mcp_url"]
        await client.store.put_item(namespace, key=REGISTRATION_KEY, value=record)
        return JSONResponse(content={"acknowledged": True}, status_code=200)
    except Exception as heartbeat_error:
        raise HTTPException(
            status_code=500,
            detail=f"Error recording MCP heartbeat: {heartbeat_error}",
        )


@app.post("/mcp/unregister")
async def mcp_unregister(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Delete the user's pending registration (daemon shutdown).

    Distinct from ``/disconnect_mcp``, which forgets the *consented*, avatar-bound
    connection; unregister only removes the presence record.
    """
    from src.anubis.utils.tools.data_analysis.backend import (
        mcp_registration_namespace,
    )
    from src.anubis.utils.tools.data_analysis.discovery import REGISTRATION_KEY

    user_id = current_user["identities"][0]["user_id"]
    token = current_user["API_KEY"]
    client = get_client(headers={"API-KEY": f"{token}"})
    try:
        await client.store.delete_item(
            list(mcp_registration_namespace(user_id)), key=REGISTRATION_KEY
        )
        return JSONResponse(content={"unregistered": True}, status_code=200)
    except Exception as unregister_error:
        raise HTTPException(
            status_code=500,
            detail=f"Error unregistering MCP server: {unregister_error}",
        )


@app.delete("/delete_avatar")
async def delete_avatar(
    assistant_id: str, request: Request, current_user: dict = Depends(get_current_user)
):
    # TODO: Delete avatar in database
    logger.info("breakpoint")
    token = current_user["API_KEY"]
    user_id = current_user["identities"][0]["user_id"]
    client = get_client(headers={"API-KEY": f"{token}"})

    metadata = {"user_id": user_id}
    metadata.update({"assistant_id": assistant_id})
    # Delete all entries in the store and store vectors for the created avatars
    pool = request.app.state.pool
    SQL_STORE_DELETE_QUERY = """DELETE FROM store WHERE prefix = %s OR prefix LIKE %s or prefix LIKE %s or prefix LIKE %s;"""
    SQL_STORE_VECTOR_DELETE_QUERY = """DELETE FROM store WHERE prefix = %s OR prefix LIKE %s or prefix LIKE %s or prefix LIKE %s;"""
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                params = (
                    assistant_id,
                    f"{assistant_id}.%",
                    f"%.{assistant_id}.%",
                    f"%.{assistant_id}",
                )
                await cur.execute(SQL_STORE_DELETE_QUERY, params)
                await cur.execute(SQL_STORE_VECTOR_DELETE_QUERY, params)
    except Exception as e:
        raise HTTPException(
            detail="Error deleting items from store and store vectors during delete avatar.",
            status_code=500,
        )

    # Every store row mentioning the assistant was just removed by raw SQL,
    # bypassing the store client — drop every cached entry for the assistant
    # from the load_consciousness read-through cache.
    invalidate_store_cache_for_assistant(assistant_id)

    try:
        await client.assistants.delete(assistant_id=assistant_id, delete_threads=True)
    except Exception as e:
        raise HTTPException(detail="Error Deleting Assistant", status_code=500)
    return JSONResponse("Deleted Avatar Successfully", status_code=200)


@app.get("/list_public_avatars")
async def list_public_avatars(assistant_id: Optional[str] = None):
    public_avatars_result = await get_public_avatars(assistant_id=assistant_id)
    return [
        {k: v for k, v in assistant.items() if k != "metadata"}
        for assistant in public_avatars_result
    ]


@app.get("/list_user_avatars")
async def list_user_avatars(
    current_user: dict = Depends(get_current_user),
):
    logger.info("breakpoint")
    if not current_user:
        public_avatars_result = await get_public_avatars()
        return [_assistant_without_metadata_if_public(a) for a in public_avatars_result]
    try:
        public_avatars_result = await get_public_avatars(
            user_id=current_user["identities"][0]["user_id"]
        )
        token = current_user["API_KEY"]
        client = get_client(headers={"API-KEY": f"{token}"})
        response = await client.assistants.search(
            metadata={"user_id": current_user["identities"][0]["user_id"]}
        )
        if len(response) > 0:
            avatar_list = response
            public_avatars_result.extend(avatar_list)  # public and private avatars
        sanitized = [
            _assistant_without_metadata_if_public(a) for a in public_avatars_result
        ]
        return JSONResponse(sanitized, status_code=200)
    except Exception as e:
        error = f"Error in listing avatars: {e}"
        raise HTTPException(detail=error, status_code=500)


@app.post("/select_avatar")
async def select_avatar(
    request: Request,
    response: Response,
    current_user: dict = Depends(get_current_user),
    assistant_id: Optional[str] = None,
    assistant_name: Optional[str] = None,
):
    logger.info("breakpoint")
    if not current_user and not assistant_id:
        return HTTPException(
            status_code=400,
            detail="Unauthenticated users must log in to use the select avatars via name feature. Please log in or use an assistant_id for selection.",
        )

    assistant_config = {"configurable": {"assistant_id": assistant_id}}

    public_avatar_result = await get_public_avatars(assistant_id=assistant_id)

    # if not current_user['identities'][0]['user_id'] is request.app.state.context['anonymous_user_id']: # anonymous user case
    if not current_user:
        if len(public_avatar_result) > 0:
            assistant_config["configurable"].update(
                {
                    "assistant_ctx": {
                        "name": public_avatar_result[0].get("name", None),
                        "description": public_avatar_result[0].get("description", None),
                    }
                }
            )

        public_avatar_result = await update_assistant_config(
            assistant_config=assistant_config, request=request
        )
        return assistant_config
    else:
        token = current_user["API_KEY"]
        client = get_client(headers={"API-KEY": token})
        user_id = current_user["identities"][0]["user_id"]
        if assistant_id:
            try:
                if len(public_avatar_result) == 0:  # the avatar was not public
                    result = await client.assistants.get(
                        assistant_id=assistant_id
                    )  # attempt to get user-specific avatar with api key
                    if not result:
                        raise HTTPException(
                            detail="Assistant not found: {assistant_id}",
                            status_code=500,
                        )
                        # assistant = {"name": None, "description": None}
                    else:
                        assistant = result
                    logger.info(f"result:{result}")
                    assistant_config = {
                        "configurable": {
                            "assistant_id": assistant_id,
                            "assistant_ctx": {
                                "name": assistant.get("name", ""),
                                "description": assistant.get("description", ""),
                                "metadata": assistant.get("metadata", {}),
                            },
                        }
                    }
                else:
                    assistant_config["configurable"].update(
                        {
                            "assistant_ctx": {
                                "name": public_avatar_result[0].get("name", None),
                                "description": public_avatar_result[0].get(
                                    "description", None
                                ),
                                "metadata": public_avatar_result[0].get("metadata", {}),
                            }
                        }
                    )
                provider_encoded_user_id = quote(current_user["user_id"], safe="")

                hashed_api_key = current_user["app_metadata"]["api_key"]
                _ = await update_assistant_config(
                    hashed_api_key=hashed_api_key,
                    provider_encoded_user_id=provider_encoded_user_id,
                    assistant_config=assistant_config,
                    request=request,
                )
                return assistant_config
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error using assistant_id for logged in user {e}",
                )
        elif assistant_name:
            try:
                result = await client.assistants.search(name=assistant_name)
                try:
                    if len(result) == 0:
                        raise HTTPException(
                            detail="Assistant not found.", status_code=400
                        )
                    assistant = result[0]
                    is_public = assistant.get("metadata", {}).get("is_public", False)
                    if not is_public and (
                        current_user["identities"][0]["user_id"]
                        != assistant.get("metadata", {}).get("user_id", None)
                    ):
                        raise HTTPException(
                            detail="Non-public avatar id.", status_code=401
                        )
                    else:
                        assistant_config = {
                            "configurable": {
                                "assistant_ctx": {
                                    "name": assistant.get("name", None),
                                    "description": assistant.get("description", None),
                                    "metadata": assistant.get("metadata", {}),
                                },
                                "assistant_id": assistant.get("assistant_id", None),
                            }
                        }
                    hashed_api_key = current_user["app_metadata"]["api_key"]
                    provider_encoded_user_id = quote(current_user["user_id"], safe="")
                    result = await update_assistant_config(
                        hashed_api_key=hashed_api_key,
                        provider_encoded_user_id=provider_encoded_user_id,
                        assistant_config=assistant_config,
                        request=request,
                    )

                    return JSONResponse(content=assistant_config, status_code=200)
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error during avatar selection via assistant_name: {e}",
                    )
            except Exception as e:
                error_str = "{error}".format(error=e)
                return HTTPException(detail=error_str, status_code=500)
        else:
            return HTTPException(
                detail="Error: either assistant_id or assistant_name is required.",
                status_code=400,
            )


async def process_files_for_message(
    files: OptionalUploadFiles = None,
    message: str = "",
) -> tuple:
    """Process uploaded files and return content for inclusion in messages.

    Returns:
        tuple: (text_content, multimodal_content, image_filenames)
        - text_content: str - concatenated text from text files (non-image)
        - multimodal_content: list or None - multimodal content (text + image blocks)
        - image_filenames: filenames for each image block, in order
    """
    if not files:
        return "", None, []

    text_contents = []
    multimodal_parts = []
    image_filenames: List[str] = []
    has_images = False

    for file in files:
        try:
            content = await file.read()
            filename = file.filename or "unknown_file"
            content_type = file.content_type or ""

            if content_type.startswith("image/"):
                base64_image = base64.b64encode(content).decode("utf-8")
                image_url = f"data:{content_type};base64,{base64_image}"

                multimodal_parts.append(
                    {"type": "image_url", "image_url": {"url": image_url}}
                )
                image_filenames.append(filename)
                has_images = True
                text_contents.append(f"[Image: {filename}]")

            elif content_type.startswith("text/") or content_type == "application/pdf":
                # Handle text files and PDFs
                if content_type == "application/pdf":
                    try:
                        from langchain_community.document_loaders import (
                            PyPDFLoader,
                        )

                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".pdf"
                        ) as temp_pdf:
                            temp_pdf.write(content)
                            temp_pdf.flush()
                            pdf_loader = PyPDFLoader(temp_pdf.name)
                            pdf_docs = pdf_loader.load()

                        pdf_text = "\n\n".join(
                            [
                                doc.page_content
                                for doc in pdf_docs
                                if hasattr(doc, "page_content")
                            ]
                        )
                        if pdf_text:
                            text_contents.append(f"[PDF File: {filename}]\n{pdf_text}")
                        else:
                            text_contents.append(
                                f"[PDF File: {filename} - no extractable text]"
                            )
                    except Exception as pdf_error:
                        logger.error(
                            f"Failed to extract PDF text from {filename}: {pdf_error}"
                        )
                        text_contents.append(f"[PDF File: {filename}]")
                    finally:
                        try:
                            os.unlink(temp_pdf.name)
                        except Exception:
                            pass
                else:
                    # Text files
                    try:
                        text_content = content.decode("utf-8")
                        text_contents.append(f"[File: {filename}]\n{text_content}")
                    except UnicodeDecodeError:
                        text_contents.append(f"[Binary Text File: {filename}]")

            elif content_type.startswith("audio/"):
                # Audio files - describe that audio was uploaded
                text_contents.append(f"[Audio File: {filename} - {content_type}]")

            else:
                # Other file types
                text_contents.append(f"[File: {filename} - {content_type}]")

        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            text_contents.append(f"[Error processing file: {file.filename}]")

    # Combine text content (file-derived only; caller message is merged below for images)
    combined_text = "\n\n".join(text_contents) if text_contents else ""

    # Return multimodal content if images are present
    if has_images:
        text_segments = []
        if (message or "").strip():
            text_segments.append(message.strip())
        if combined_text:
            text_segments.append(combined_text)
        full_text = "\n\n".join(text_segments)
        multimodal_content = [{"type": "text", "text": full_text}] + multimodal_parts
        return combined_text, multimodal_content, image_filenames

    return combined_text, None, []


@app.post("/message")
async def message_selected_avatar(
    request: Request,
    message: str = Form(""),
    your_name: Optional[str] = Form(None),
    your_description: Optional[str] = Form(None),
    conversation_title: Optional[str] = Form(None),
    files: OptionalUploadFiles = None,
    thread_id: Optional[str] = Form(None),
    stream: bool = Form(True),
    feedback: bool = Form(False),
    like: bool = Form(False),
    dislike: bool = Form(False),
    user_timezone: Optional[str] = Form(None),
    include_quality_metrics: bool = Form(True),
    include_usage_metrics: bool = Form(True),
    adapter: bool = Form(False),
    current_user: dict = Depends(get_current_user),
):
    # NOTE: ``feedback`` / ``like`` / ``dislike`` are inert placeholders. The
    # data-collection / preference-learning pipeline is intentionally deferred
    # while the upload + evaluation pipeline ships first; the parameters exist
    # now so the frontend can wire its UI without a breaking API change later.
    langgraph_client_headers = {"API-KEY": request.headers.get("api-key")}
    # allow for select avatar in query and anonymous user for a dedicated endpoint
    start_time = time_ns()
    config = current_user.get("app_metadata", {}).get("assistant_config", {})
    if not config:
        raise HTTPException(
            detail="Error retrieving assistant information.", status_code=400
        )

    # This turn bills the adapter-inference meter only when the client asked for
    # the adapter AND the user's tier grants adapter inference; otherwise the
    # messaging meter governs. Attached files are processed FIRST so the
    # pre-call estimate covers message text, attached file text, and attached
    # images; enforcement then verifies the estimate against the allotment
    # period BEFORE any model call (and before a thread is created). Any tier
    # without pay-per-use is blocked at the allotment; a token rate cap guards
    # against runaway clients.
    message_meter = (
        UsageMeter.ADAPTER_INFERENCE_TOKENS
        if resolve_use_adapter_inference(current_user, adapter)
        else UsageMeter.MESSAGING_TOKENS
    )
    (
        file_text_content,
        multimodal_content,
        image_filenames,
    ) = await process_files_for_message(files, message=message)

    user_name = your_name
    user_description = your_description
    user_id = current_user["identities"][0]["user_id"]
    config_update = {
        "configurable": {
            "user_ctx": {"name": user_name, "description": user_description},
            "user_id": user_id,
        }
    }
    assistant_id = config["configurable"].get("assistant_id")

    # The pre-call estimate measures the REAL system prompt for this
    # (user, avatar) pair, so estimation reads the merged configurable the
    # prompt builder needs (user_id, assistant_id, assistant_ctx). The merge
    # happens on a copy: enforcement below still runs before a thread is
    # created and before any model call.
    estimation_config = {
        "configurable": {
            **config.get("configurable", {}),
            **config_update["configurable"],
        }
    }
    estimated_request_tokens = await _estimate_message_request_tokens(
        request.app.state,
        estimation_config,
        message,
        file_text_content,
        multimodal_content,
    )
    await enforce_remaining_allotment(
        request.app.state,
        current_user,
        message_meter,
        estimated_request_tokens=estimated_request_tokens.input_tokens,
    )
    message_rate_limit_context = GlobalContext()
    await enforce_token_rate_limit(
        request.app.state,
        current_user,
        meter_event_names=[
            UsageMeter.MESSAGING_TOKENS.value,
            UsageMeter.ADAPTER_INFERENCE_TOKENS.value,
        ],
        window_seconds=int(message_rate_limit_context.message_rate_limit_window_seconds or 0),
        tokens_per_window=int(
            message_rate_limit_context.message_rate_limit_tokens_per_window or 0
        ),
        estimated_request_tokens=estimated_request_tokens.total_tokens,
    )

    # Handle thread_id
    if not thread_id:
        thread_id = str(uuid4())
        thread_metadata = {
            "thread_metadata": {"user_id": user_id, "assistant_id": assistant_id},
            "graph_id": "Anubis",
        }
        # create thread_id
        try:
            langgraph_client = get_client(headers=langgraph_client_headers)
            thread_create_response = await langgraph_client.threads.create(
                thread_id=thread_id, metadata=thread_metadata
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail="Error creating new conversation thread."
            )

    # update with user information
    config_update["configurable"]["thread_id"] = thread_id
    config["configurable"].update(config_update["configurable"])
    # client-supplied IANA timezone (e.g. "America/New_York") used to localize system_time
    config["configurable"]["user_timezone"] = user_timezone
    config["configurable"]["include_quality_metrics"] = include_quality_metrics
    config["configurable"]["use_adapter_inference"] = resolve_use_adapter_inference(
        current_user, adapter
    )

    # store = app.state.store
    graph = app.state.graph

    # Uploaded files were already processed above (before enforcement) so the
    # pre-call estimate could cover them; reuse those results here.

    # Create the human message content
    if multimodal_content:
        human_message = HumanMessage(
            id=str(uuid4()),
            content=multimodal_content,
            additional_kwargs={"image_filenames": image_filenames},
        )
    else:
        # Use text-only content
        if file_text_content:
            if (message or "").strip():
                human_message_content = message.strip() + "\n\n" + file_text_content
            else:
                human_message_content = file_text_content
        else:
            human_message_content = message
        human_message = HumanMessage(id=str(uuid4()), content=human_message_content)

    if stream:
        return StreamingResponse(
            message_graph_sse(
                graph,
                human_message,
                config,
                app.state.context,
                thread_id=thread_id,
                user_id=user_id,
                assistant_id=assistant_id,
                conversation_title_value=conversation_title,
                start_time_ns=start_time,
                request_id=request.state.request_id,
                langgraph_client_headers=langgraph_client_headers,
                app_state=request.app.state,
                current_user=current_user,
                estimated_request_tokens=estimated_request_tokens,
                estimate_meter=message_meter,
                include_usage_metrics=include_usage_metrics,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    result = await graph.ainvoke(
        input={"messages": [human_message]},
        config=config,
        context=app.state.context,
    )

    # Update most_recent_message
    langgraph_client = get_client(headers=langgraph_client_headers)
    thread_metadata = {
        "thread_metadata": {
            "user_id": user_id,
            "assistant_id": assistant_id,
            "most_recent_message": datetime.now(UTC).isoformat(),
            "conversation_title": conversation_title,
        },
        "graph_id": "Anubis",
    }
    await langgraph_client.threads.update(thread_id=thread_id, metadata=thread_metadata)

    response_data = {}
    response_data["content"] = result["messages"][-1].content
    response_metadata = result["messages"][-1].response_metadata
    if response_metadata:
        response_data["response_metadata"] = response_metadata

    response_data["total_response_time_ms"] = (time_ns() - start_time) // 1000000
    logger.warning(f"RESPONSE_DATA: {response_data}")
    response_data["thread_id"] = thread_id
    response_data["request_id"] = request.state.request_id
    turn_usage = await _meter_message_usage(
        app_state=request.app.state,
        current_user=current_user,
        response_metadata=response_metadata,
        thread_id=thread_id,
        assistant_id=assistant_id,
        latency_ms=(time_ns() - start_time) / 1_000_000,
        request_id=request.state.request_id,
    )
    if include_usage_metrics:
        response_data["input_tokens"] = estimated_request_tokens.input_tokens
        if turn_usage:
            response_data["usage"] = turn_usage
    return JSONResponse(response_data, status_code=200)


@app.post("/message/{assistant_id}")
async def message_avatar(
    request: Request,
    assistant_id: str,
    message: str = Form(""),
    your_name: Optional[str] = Form(None),
    your_description: Optional[str] = Form(None),
    conversation_title: Optional[str] = Form(None),
    files: OptionalUploadFiles = None,
    thread_id: Optional[str] = Form(None),
    stream: bool = Form(True),
    feedback: bool = Form(False),
    like: bool = Form(False),
    dislike: bool = Form(False),
    user_timezone: Optional[str] = Form(None),
    include_quality_metrics: bool = Form(True),
    include_usage_metrics: bool = Form(True),
    adapter: bool = Form(False),
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    # NOTE: ``feedback`` / ``like`` / ``dislike`` are inert placeholders. The
    # data-collection / preference-learning pipeline is intentionally deferred
    # while the upload + evaluation pipeline ships first; the parameters exist
    # now so the frontend can wire its UI without a breaking API change later.

    # allow for select avatar in query and anonymous user for a dedicated endpoint

    logger.warning(f"stream:{stream}")
    start_time = time_ns()
    config = current_user.get("app_metadata", {}).get("assistant_config", {})
    if not config:
        raise HTTPException(
            detail="Error retrieving assistant information.", status_code=400
        )

    # This turn bills the adapter-inference meter only when the client asked for
    # the adapter AND the user's tier grants adapter inference; otherwise the
    # messaging meter governs. Attached files are processed FIRST so the
    # pre-call estimate covers message text, attached file text, and attached
    # images; enforcement then verifies the estimate against the allotment
    # period BEFORE any model call (and before a thread is created). Any tier
    # without pay-per-use is blocked at the allotment; a token rate cap guards
    # against runaway clients.
    message_meter = (
        UsageMeter.ADAPTER_INFERENCE_TOKENS
        if resolve_use_adapter_inference(current_user, adapter)
        else UsageMeter.MESSAGING_TOKENS
    )
    (
        file_text_content,
        multimodal_content,
        image_filenames,
    ) = await process_files_for_message(files, message=message)

    user_name = your_name
    user_description = your_description
    user_id = current_user["identities"][0]["user_id"]
    assistant_id = assistant_id.strip()
    if request.headers.get("api-key", "") != "":
        langgraph_client_headers = {"API-KEY": request.headers.get("api-key")}
        try:
            langgraph_client = get_client(headers=langgraph_client_headers)
            assistant = await langgraph_client.assistants.get(assistant_id=assistant_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error selecting avatar.")

        config_update = {
            "configurable": {
                "user_ctx": {"name": user_name, "description": user_description},
                "user_id": user_id,
                "assistant_id": assistant_id,
                "assistant_ctx": {
                    "name": assistant.get("name", None),
                    "description": assistant.get("description", None),
                    "metadata": assistant.get("metadata", {}),
                },
            }
        }

    else:
        # anonymous user_id and assistant_id is handled in the current_user dependency function
        langgraph_client_headers = {"API-KEY": app.state.context.anonymous_api_key}
        config_update = {
            "configurable": {
                "user_ctx": {"name": user_name, "description": user_description},
            }
        }

    # The pre-call estimate measures the REAL system prompt for this
    # (user, avatar) pair, so estimation reads the merged configurable the
    # prompt builder needs (user_id, assistant_id, assistant_ctx — the path
    # avatar's context just fetched above, or the anonymous dependency's).
    # The merge happens on a copy: enforcement below still runs before a
    # thread is created and before any model call.
    estimation_config = {
        "configurable": {
            **config.get("configurable", {}),
            **config_update["configurable"],
            "user_id": user_id,
            "assistant_id": assistant_id,
        }
    }
    estimated_request_tokens = await _estimate_message_request_tokens(
        request.app.state,
        estimation_config,
        message,
        file_text_content,
        multimodal_content,
    )
    await enforce_remaining_allotment(
        request.app.state,
        current_user,
        message_meter,
        estimated_request_tokens=estimated_request_tokens.input_tokens,
    )
    message_rate_limit_context = GlobalContext()
    await enforce_token_rate_limit(
        request.app.state,
        current_user,
        meter_event_names=[
            UsageMeter.MESSAGING_TOKENS.value,
            UsageMeter.ADAPTER_INFERENCE_TOKENS.value,
        ],
        window_seconds=int(message_rate_limit_context.message_rate_limit_window_seconds or 0),
        tokens_per_window=int(
            message_rate_limit_context.message_rate_limit_tokens_per_window or 0
        ),
        estimated_request_tokens=estimated_request_tokens.total_tokens,
    )

    # Handle thread_id
    if not thread_id:
        thread_id = str(uuid4())
        thread_metadata = {
            "thread_metadata": {"user_id": user_id, "assistant_id": assistant_id},
            "graph_id": "Anubis",
        }
        # create thread_id
        try:
            langgraph_client = get_client(headers=langgraph_client_headers)
            thread_create_response = await langgraph_client.threads.create(
                thread_id=thread_id, metadata=thread_metadata
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail="Error creating new conversation thread."
            )

    # update with user information
    config_update["configurable"]["thread_id"] = thread_id
    config["configurable"].update(config_update["configurable"])
    # client-supplied IANA timezone (e.g. "America/New_York") used to localize system_time
    config["configurable"]["user_timezone"] = user_timezone
    config["configurable"]["include_quality_metrics"] = include_quality_metrics
    config["configurable"]["use_adapter_inference"] = resolve_use_adapter_inference(
        current_user, adapter
    )

    # store = app.state.store
    graph = app.state.graph

    # Uploaded files were already processed above (before enforcement) so the
    # pre-call estimate could cover them; reuse those results here.

    # Create the human message content
    if multimodal_content:
        human_message = HumanMessage(
            id=str(uuid4()),
            content=multimodal_content,
            additional_kwargs={"image_filenames": image_filenames},
        )
    else:
        # Use text-only content
        if file_text_content:
            if (message or "").strip():
                human_message_content = message.strip() + "\n\n" + file_text_content
            else:
                human_message_content = file_text_content
        else:
            human_message_content = message
        human_message = HumanMessage(id=str(uuid4()), content=human_message_content)

    conversation_title_data = (
        conversation_title if conversation_title != "" else thread_id
    )

    if stream:
        return StreamingResponse(
            message_graph_sse(
                graph,
                human_message,
                config,
                app.state.context,
                thread_id=thread_id,
                user_id=user_id,
                assistant_id=assistant_id,
                conversation_title_value=conversation_title_data,
                start_time_ns=start_time,
                request_id=request.state.request_id,
                langgraph_client_headers=langgraph_client_headers,
                app_state=request.app.state,
                current_user=current_user,
                estimated_request_tokens=estimated_request_tokens,
                estimate_meter=message_meter,
                include_usage_metrics=include_usage_metrics,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    result = await graph.ainvoke(
        input={"messages": [human_message]},
        config=config,
        context=app.state.context,
    )

    # Update most_recent_message
    langgraph_client = get_client(headers=langgraph_client_headers)
    thread_metadata = {
        "thread_metadata": {
            "user_id": user_id,
            "assistant_id": assistant_id,
            "most_recent_message": datetime.now(UTC).isoformat(),
            "conversation_title": conversation_title_data,
        },
        "graph_id": "Anubis",
    }
    await langgraph_client.threads.update(thread_id=thread_id, metadata=thread_metadata)

    response_data = {}
    response_data["content"] = result["messages"][-1].content
    response_metadata = result["messages"][-1].response_metadata
    if response_metadata:
        response_data["response_metadata"] = response_metadata

    response_data["total_response_time_ms"] = (time_ns() - start_time) // 1000000
    response_data["thread_id"] = thread_id
    response_data["request_id"] = request.state.request_id
    turn_usage = await _meter_message_usage(
        app_state=request.app.state,
        current_user=current_user,
        response_metadata=response_metadata,
        thread_id=thread_id,
        assistant_id=assistant_id,
        latency_ms=(time_ns() - start_time) / 1_000_000,
        request_id=request.state.request_id,
    )
    if include_usage_metrics:
        response_data["input_tokens"] = estimated_request_tokens.input_tokens
        if turn_usage:
            response_data["usage"] = turn_usage
    return JSONResponse(response_data, status_code=200)


@app.post("/message/{assistant_id}/resume")
async def resume_avatar_message(
    request: Request,
    assistant_id: str,
    thread_id: str = Form(...),
    decision: str = Form("apply"),
    items: Optional[str] = Form(None),
    your_name: Optional[str] = Form(None),
    your_description: Optional[str] = Form(None),
    user_timezone: Optional[str] = Form(None),
    include_quality_metrics: bool = Form(True),
    include_usage_metrics: bool = Form(True),
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    """Resume a run paused for human approval (edit/delete identity fact).

    ``decision`` is ``apply`` | ``cancel``. ``items`` (JSON list) carries the owner's
    per-document decisions — one entry per matched document with ``index`` and an ``action``
    ∈ ``skip`` | ``accept`` | ``edit`` | ``remove`` (plus ``corrected_text`` /
    ``correction_context`` when the action is ``edit``). Any matched document the owner did
    not act on defaults to ``skip`` in the tool, so a missing/empty list changes nothing.
    Older clients' ``approve`` / ``reject`` are accepted as aliases for ``apply`` / ``cancel``.
    Streams the continuation as SSE (same ``assistant_token`` → ``done``/``interrupt`` shape as
    ``/message/{assistant_id}``).
    """
    start_time = time_ns()

    # Map legacy spellings so an older panel still resolves to the current vocabulary.
    decision_aliases = {"approve": "apply", "reject": "cancel"}
    raw_decision = (decision or "apply").strip().lower()
    decision_value = decision_aliases.get(raw_decision, raw_decision)
    if decision_value not in ("apply", "cancel"):
        raise HTTPException(status_code=400, detail="decision must be apply or cancel.")

    config = current_user.get("app_metadata", {}).get("assistant_config", {})
    if not config:
        raise HTTPException(
            detail="Error retrieving assistant information.", status_code=400
        )

    # The resumed continuation bills the messaging meter (resume has no adapter
    # form field) and IS a model call, so the same allotment and rate-limit
    # gates apply as on the initial message endpoints. A resume carries no new
    # user text or images, so the input estimate is the measured system prompt
    # recorded when the interrupted turn built the prompt moments ago (zero on
    # a cold cache — the recorded actual usage still governs accrual) plus the
    # measured bound tool schemas (billed as input on every model call); the
    # output estimate is the expected reply budget. No fixed or guessed input
    # overhead — every input component is measured.
    message_rate_limit_context = GlobalContext()
    estimated_request_tokens = TokenEstimateBreakdown(
        input_tokens=(
            fetch_system_prompt_token_estimate(
                current_user["identities"][0]["user_id"],
                assistant_id,
                max_age_seconds=float(
                    message_rate_limit_context.system_prompt_token_estimate_cache_ttl_seconds
                    or 0
                ),
            )
            or 0
        )
        + await asyncio.to_thread(
            fetch_or_measure_deep_agent_tool_schema_token_estimate
        ),
        output_tokens=int(
            message_rate_limit_context.message_expected_output_tokens_estimate or 0
        ),
    )
    await enforce_remaining_allotment(
        request.app.state,
        current_user,
        UsageMeter.MESSAGING_TOKENS,
        estimated_request_tokens=estimated_request_tokens.input_tokens,
    )
    await enforce_token_rate_limit(
        request.app.state,
        current_user,
        meter_event_names=[
            UsageMeter.MESSAGING_TOKENS.value,
            UsageMeter.ADAPTER_INFERENCE_TOKENS.value,
        ],
        window_seconds=int(message_rate_limit_context.message_rate_limit_window_seconds or 0),
        tokens_per_window=int(
            message_rate_limit_context.message_rate_limit_tokens_per_window or 0
        ),
        estimated_request_tokens=estimated_request_tokens.total_tokens,
    )

    user_id = current_user["identities"][0]["user_id"]
    if request.headers.get("api-key", "") != "":
        langgraph_client_headers = {"API-KEY": request.headers.get("api-key")}
        try:
            langgraph_client = get_client(headers=langgraph_client_headers)
            assistant = await langgraph_client.assistants.get(assistant_id=assistant_id)
        except Exception:
            raise HTTPException(status_code=500, detail="Error selecting avatar.")
        config_update = {
            "configurable": {
                "user_ctx": {"name": your_name, "description": your_description},
                "user_id": user_id,
                "assistant_id": assistant_id,
                "assistant_ctx": {
                    "name": assistant.get("name", None),
                    "description": assistant.get("description", None),
                    "metadata": assistant.get("metadata", {}),
                },
            }
        }
    else:
        langgraph_client_headers = {"API-KEY": app.state.context.anonymous_api_key}
        config_update = {
            "configurable": {
                "user_ctx": {"name": your_name, "description": your_description},
            }
        }

    config_update["configurable"]["thread_id"] = thread_id
    config["configurable"].update(config_update["configurable"])
    config["configurable"]["user_timezone"] = user_timezone
    config["configurable"]["include_quality_metrics"] = include_quality_metrics

    graph = app.state.graph

    # The resume value is the decision dict the paused tool's ``interrupt`` expects; it flows
    # outer-``interrupt`` → ``think`` → the deep-agent tool unchanged. ``cancel`` abandons the
    # whole correction; ``apply`` carries the owner's per-item decisions. The tool defaults any
    # un-acted item to ``skip``, so an empty/missing list is a safe no-op.
    resume_payload: dict = {"type": decision_value}
    if items:
        try:
            parsed_items = json.loads(items)
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="items must be a JSON list.")
        if not isinstance(parsed_items, list):
            raise HTTPException(status_code=400, detail="items must be a JSON list.")
        resume_payload["items"] = parsed_items

    # The outer graph exposes a single think-level interrupt per pause. Multi-interrupt
    # resume maps are built inside ``think`` for the checkpointed deep agent only.
    return StreamingResponse(
        message_graph_sse(
            graph,
            None,
            config,
            app.state.context,
            thread_id=thread_id,
            user_id=user_id,
            assistant_id=assistant_id,
            conversation_title_value=None,
            start_time_ns=start_time,
            request_id=request.state.request_id,
            langgraph_client_headers=langgraph_client_headers,
            resume_command=Command(resume=resume_payload),
            app_state=request.app.state,
            current_user=current_user,
            estimated_request_tokens=estimated_request_tokens,
            estimate_meter=UsageMeter.MESSAGING_TOKENS,
            include_usage_metrics=include_usage_metrics,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/conversations")
async def get_all_conversations(
    request: Request,
    assistant_id: str,
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    """Return all threads for this user + assistant, newest-first."""
    user_id = current_user["identities"][0]["user_id"]
    if request.headers.get("api-key", "") != "":
        langgraph_client_headers = {"API-KEY": request.headers.get("api-key")}
    else:
        langgraph_client_headers = {
            "API-KEY": request.app.state.context.anonymous_api_key
        }
    try:
        langgraph_client = get_client(headers=langgraph_client_headers)
        threads = await langgraph_client.threads.search(
            metadata={
                "thread_metadata": {"user_id": user_id, "assistant_id": assistant_id}
            },
            sort_by="updated_at",
            sort_order="desc",
        )
        return JSONResponse(threads)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error loading threads: {exc}")


@app.get("/conversations/{thread_id}/messages")
async def get_thread_messages(
    request: Request,
    thread_id: str,
    assistant_id: str,
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    """Return the message history for a single thread."""
    if request.headers.get("api-key", "") != "":
        langgraph_client_headers = {"API-KEY": request.headers.get("api-key")}
    else:
        langgraph_client_headers = {
            "API-KEY": request.app.state.context.anonymous_api_key
        }
    try:
        langgraph_client = get_client(headers=langgraph_client_headers)
        state = await langgraph_client.threads.get_state(thread_id=thread_id)
        messages = state.get("values", {}).get("messages", []) if state else []
        return JSONResponse({"messages": messages})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error loading messages: {exc}")


ALLOWED_IMAGE_MIMES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})


def normalize_declared_image_mime(ct: str) -> str:
    ct = (ct or "").split(";")[0].strip().lower()
    if ct == "image/jpg":
        return "image/jpeg"
    return ct


def _sniff_media_category_from_bytes(chunk: bytes) -> Optional[str]:
    """Infer image/audio/video/pdf from magic bytes when Content-Type is unhelpful."""
    if not chunk:
        return None
    if chunk[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if chunk[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if chunk[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if chunk[:4] == b"RIFF" and len(chunk) >= 12 and chunk[8:12] == b"WEBP":
        return "image/webp"
    if chunk[:4] == b"%PDF":
        return "application/pdf"
    if chunk[:3] == b"ID3" or chunk[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio/mpeg"
    if chunk[:4] == b"OggS":
        return "audio/ogg"
    if chunk[:4] == b"RIFF" and len(chunk) >= 12 and chunk[8:12] == b"WAVE":
        return "audio/wav"
    if chunk[:4] == b"\x1a\x45\xdf\xa3":
        return "video/webm"
    if len(chunk) >= 12 and chunk[4:8] == b"ftyp":
        return "video/mp4"
    return None


def _gif_image_descriptor_count(data: bytes) -> int:
    if len(data) < 13:
        return 0
    if data[:6] not in (b"GIF87a", b"GIF89a"):
        return 0
    packed = data[10]
    i = 13
    if packed & 0x80:
        i += 3 * (1 << ((packed & 0x07) + 1))
    count = 0
    n = len(data)
    while i < n:
        tag = data[i]
        if tag == 0x3B:
            break
        if tag == 0x21:
            i += 1
            if i >= n:
                break
            i += 1
            while i < n:
                bsize = data[i]
                i += 1
                if bsize == 0:
                    break
                i += bsize
        elif tag == 0x2C:
            count += 1
            i += 1
            if i + 8 > n:
                break
            i += 8
            local = data[i - 1]
            if local & 0x80:
                i += 3 * (1 << ((local & 0x07) + 1))
            if i >= n:
                break
            i += 1
            while i < n:
                bsize = data[i]
                i += 1
                if bsize == 0:
                    break
                i += bsize
        else:
            i += 1
    return count


def _gif_is_animated(data: bytes) -> bool:
    return _gif_image_descriptor_count(data) > 1


def _webp_is_animated(data: bytes) -> bool:
    cap = min(len(data), 65536)
    return b"ANMF" in data[:cap]


def validate_upload_image_bytes(declared_mime: str, body: bytes) -> str:
    """Return normalized image MIME; raises HTTPException if not an allowed still image."""
    mime = normalize_declared_image_mime(declared_mime)
    sniff = _sniff_media_category_from_bytes(body[:512])
    if mime in ("", "application/octet-stream"):
        if sniff not in ALLOWED_IMAGE_MIMES:
            raise HTTPException(
                status_code=400,
                detail="Could not determine an allowed image type from the file or URL.",
            )
        mime = sniff
    if mime not in ALLOWED_IMAGE_MIMES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image type not allowed (got {mime!r}); "
                "allowed: image/jpeg, image/png, image/gif (non-animated), image/webp."
            ),
        )
    if sniff and normalize_declared_image_mime(sniff) != mime:
        raise HTTPException(
            status_code=400,
            detail="Declared Content-Type does not match image file contents.",
        )
    if mime == "image/gif" and _gif_is_animated(body):
        raise HTTPException(
            status_code=400, detail="Animated GIF is not allowed; use a still frame."
        )
    if mime == "image/webp" and _webp_is_animated(body):
        raise HTTPException(
            status_code=400, detail="Animated WebP is not allowed; use a still image."
        )
    return mime


async def probe_remote_url_content_type(url: str) -> str:
    """Best-effort Content-Type for a remote URL (HEAD, then ranged GET + sniff)."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
        head_ct = ""
        try:
            head = await client.head(url)
            head_ct = (
                (head.headers.get("content-type") or "").split(";")[0].strip().lower()
            )
        except Exception:
            pass
        if head_ct and head_ct != "application/octet-stream":
            return head_ct
        resp = await client.get(url, headers={"Range": "bytes=0-511"})
        resp.raise_for_status()
        body_ct = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        if body_ct and body_ct != "application/octet-stream":
            return body_ct
        sniffed = _sniff_media_category_from_bytes(resp.content[:512])
        return sniffed or body_ct or "application/octet-stream"


async def require_url_content_type_prefix(url: str, prefix: str, label: str) -> None:
    ct = await probe_remote_url_content_type(url)
    if not ct.startswith(prefix):
        raise HTTPException(
            status_code=400,
            detail=f"{label} URL must resolve to {prefix}* (got {ct!r}).",
        )


def _is_youtube_url(url: str) -> bool:
    """Recognize URLs whose Content-Type is HTML but whose payload is video/audio."""
    from urllib.parse import urlparse

    from src.anubis.utils.classes.URLDocumentLoaderClass import _YOUTUBE_HOSTS

    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    return host in _YOUTUBE_HOSTS


MAX_REMOTE_URL_DOWNLOAD_BYTES = 25 * 1024 * 1024


async def fetch_remote_url_bytes(
    url: str,
    max_bytes: int = MAX_REMOTE_URL_DOWNLOAD_BYTES,
) -> tuple[bytes, str]:
    """Download a URL and return (body, Content-Type without parameters)."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=120.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        body = r.content
        if len(body) > max_bytes:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Remote resource exceeds maximum download size ({max_bytes} bytes)."
                ),
            )
        header_ct = (r.headers.get("content-type") or "").split(";")[0].strip().lower()
        return body, header_ct or "application/octet-stream"


def make_data_uri(mime: str, body: bytes) -> str:
    """RFC 2397 data URI: ``data:<mime>;base64,<payload>``."""
    return f"data:{mime};base64,{base64.b64encode(body).decode('ascii')}"


# ---------------------------------------------------------------------------
# CSV ingest preprocessing
#
# Tabular uploads are converted at the API edge into a JSON ``statements``
# document so the rest of the pipeline only ever has to handle media types it
# already knows about. Each CSV row becomes one statement with the shape
# requested by the avatar-identity ingest contract:
#
#     {
#         "messages": [{"role": "assistant", "content": "<row text>"}],
#         "metadata": {"target": "<name>", "source": "<filename>"}
#     }
#
# The text column and target name are picked once per upload by
# ``CSVUserTextColumnIdentificationClass`` (model-driven, schema-constrained).
# Detection happens HERE so the process_media graph never sees raw CSV bytes.
# ---------------------------------------------------------------------------


_CSV_MIME_HINTS = frozenset(
    {
        "text/csv",
        "application/csv",
        "application/vnd.ms-excel",
        "application/x-csv",
    }
)
_CSV_NAME_HINT_RE = re.compile(
    r"\b(user[_-]?name|user|name|author|screen[_-]?name|"
    r"handle|username|creator|speaker|full[_-]?name)\b",
    re.IGNORECASE,
)
_CSV_BOOLEAN_VALUES = frozenset({"true", "false", "yes", "no", "0", "1"})
_CSV_PREVIEW_ROW_LIMIT = 8
_CSV_STATS_SAMPLE_VALUES = 3


def _is_csv_upload(filename: str, content_type: str) -> bool:
    """True when the upload looks like a CSV by MIME type or filename."""
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct in _CSV_MIME_HINTS:
        return True
    name = (filename or "").strip().lower()
    return name.endswith(".csv") or name.endswith(".tsv")


def _decode_csv_bytes(raw: bytes) -> str:
    """Decode CSV bytes preferring UTF-8, falling back to latin-1 then replace.

    BOM is stripped because ``csv.reader`` treats it as part of the first
    header otherwise.
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _parse_csv_to_rows(raw: bytes, filename: str) -> tuple[list[str], list[dict]]:
    """Return ``(headers, rows)`` from CSV bytes.

    Uses ``csv.Sniffer`` for delimiter detection (handles ``,`` and ``\\t``
    files), falls back to comma when sniffing fails on tiny / malformed
    samples. ``rows`` is a list of OrderedDicts keyed by header.
    """
    import csv as _csv
    from io import StringIO

    text = _decode_csv_bytes(raw)
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail=f"CSV upload {filename!r} is empty.",
        )

    sample = text[:8192]
    dialect: Any
    try:
        dialect = _csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except _csv.Error:
        dialect = _csv.excel
        if filename.lower().endswith(".tsv"):
            dialect = _csv.excel_tab

    reader = _csv.DictReader(StringIO(text), dialect=dialect)
    headers = list(reader.fieldnames or [])
    if not headers:
        raise HTTPException(
            status_code=400,
            detail=f"CSV upload {filename!r} has no header row.",
        )
    rows: list[dict] = []
    for row in reader:
        rows.append(
            {h: (row.get(h) if row.get(h) is not None else "") for h in headers}
        )
    if not rows:
        raise HTTPException(
            status_code=400,
            detail=f"CSV upload {filename!r} has no data rows.",
        )
    return headers, rows


def _looks_numeric(value: str) -> bool:
    v = value.strip()
    if not v:
        return False
    try:
        float(v.replace(",", ""))
        return True
    except ValueError:
        return False


def _build_csv_column_stats(
    headers: list[str], rows: list[dict]
) -> dict[str, dict[str, Any]]:
    """Per-column summary used as model context for column identification."""
    stats: dict[str, dict[str, Any]] = {}
    total = len(rows)
    for header in headers:
        values = [str(r.get(header) or "").strip() for r in rows]
        non_empty = [v for v in values if v]
        non_empty_count = len(non_empty)
        if non_empty_count == 0:
            stats[header] = {
                "non_empty_count": 0,
                "non_empty_ratio": 0.0,
                "avg_len": 0.0,
                "max_len": 0,
                "distinct_count": 0,
                "distinct_ratio": 0.0,
                "looks_numeric": False,
                "looks_boolean": False,
                "name_hint": bool(_CSV_NAME_HINT_RE.search(header or "")),
                "sample_values": [],
            }
            continue
        avg_len = sum(len(v) for v in non_empty) / non_empty_count
        max_len = max(len(v) for v in non_empty)
        distinct = sorted(set(non_empty), key=non_empty.index)
        distinct_count = len(distinct)
        looks_numeric = (
            sum(1 for v in non_empty if _looks_numeric(v)) / non_empty_count
        ) >= 0.9
        looks_boolean = (
            sum(1 for v in non_empty if v.lower() in _CSV_BOOLEAN_VALUES)
            / non_empty_count
        ) >= 0.9
        stats[header] = {
            "non_empty_count": non_empty_count,
            "non_empty_ratio": non_empty_count / total if total else 0.0,
            "avg_len": round(avg_len, 2),
            "max_len": max_len,
            "distinct_count": distinct_count,
            "distinct_ratio": distinct_count / non_empty_count,
            "looks_numeric": looks_numeric,
            "looks_boolean": looks_boolean,
            "name_hint": bool(_CSV_NAME_HINT_RE.search(header or "")),
            "sample_values": distinct[:_CSV_STATS_SAMPLE_VALUES],
        }
    return stats


def _csv_dominant_value(values: list[str]) -> tuple[Optional[str], float]:
    """Return the dominant non-empty value and its share of non-empty rows."""
    cleaned = [v for v in (s.strip() for s in values) if v]
    if not cleaned:
        return None, 0.0
    counts: dict[str, int] = {}
    for v in cleaned:
        counts[v] = counts.get(v, 0) + 1
    top_value, top_count = max(counts.items(), key=lambda kv: kv[1])
    return top_value, top_count / len(cleaned)


def _filename_target_hint(filename: str) -> str:
    """Title-case a filename stem when it looks like a person's name namespace_safe_formatted_filename."""
    stem = (filename or "").rsplit(".", 1)[0]
    stem = re.sub(r"[_\-]+", " ", stem).strip()
    if not stem:
        return ""
    parts = [p for p in stem.split() if p and p.isalpha()]
    if 1 <= len(parts) <= 4:
        return " ".join(p.capitalize() for p in parts)
    return ""


async def _csv_to_statements_payload(
    *, raw: bytes, source_filename: str
) -> dict[str, Any]:
    """Convert CSV bytes into the avatar-identity statements JSON document.

    Output shape passed downstream to the JSON media handler:

        {
            "statements": [
                {
                    "messages": [{"role": "assistant", "content": "<text>"}],
                    "metadata": {"target": "<name>", "source": "<filename>"}
                },
                ...
            ],
            "metadata": {
                "target": "<dominant target name or null>",
                "source": "<source_filename>",
                "csv_text_column": "<column>",
                "csv_target_column": "<column or null>",
                "csv_row_count": <int>,
                "csv_classifier_reasoning": "<llm reasoning>"
            }
        }
    """
    headers, rows = _parse_csv_to_rows(raw, source_filename)
    return await _rows_to_statements_payload(
        headers=headers, rows=rows, source_filename=source_filename
    )


async def _rows_to_statements_payload(
    *, headers: list[str], rows: list[dict], source_filename: str
) -> dict[str, Any]:
    """Convert parsed tabular ``(headers, rows)`` into the statements document.

    The shared core behind every tabular upload format (CSV/TSV bytes via
    ``_parse_csv_to_rows``, tabular JSON via ``_normalize_tabular_json_to_rows``):
    build per-column stats, have ``CSVUserTextColumnIdentificationClass`` pick
    the free-text and target-name columns once per upload, then emit one
    statement per non-empty row. Output shape documented on
    ``_csv_to_statements_payload``.
    """
    from src.anubis.utils.classes.CSVUserTextColumnIdentificationClass import (
        CSVUserTextColumnIdentificationClass,
    )

    column_stats = _build_csv_column_stats(headers, rows)

    sample_rows = rows[:_CSV_PREVIEW_ROW_LIMIT]
    classifier = CSVUserTextColumnIdentificationClass()
    classifier_response = await classifier.classify(
        filename=source_filename,
        headers=headers,
        sample_rows=sample_rows,
        column_stats=column_stats,
    )

    text_column: str = classifier_response.get("text_column") or ""
    if text_column not in headers:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Could not identify a text column in {source_filename!r}; "
                "the tabular upload does not appear to contain a free-text column."
            ),
        )

    target_column: Optional[str] = classifier_response.get("target_name_column")
    target_name_value: Optional[str] = classifier_response.get("target_name_value")

    dominant_target: Optional[str] = None
    if target_column and target_column in headers:
        candidate, share = _csv_dominant_value(
            [str(r.get(target_column) or "") for r in rows]
        )
        if candidate and share >= 0.8:
            dominant_target = candidate

    if not target_name_value:
        target_name_value = dominant_target or _filename_target_hint(source_filename)

    statements: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.get(text_column) or "").strip()
        if not text:
            continue
        if target_column and target_column in headers:
            row_target = str(row.get(target_column) or "").strip() or target_name_value
        else:
            row_target = target_name_value
        statements.append(
            {
                "messages": [{"role": "assistant", "content": text}],
                "metadata": {
                    "target": row_target or None,
                    "source": source_filename,
                },
            }
        )

    if not statements:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Tabular upload {source_filename!r} produced no non-empty rows "
                f"in column {text_column!r}."
            ),
        )

    return {
        "statements": statements,
        "metadata": {
            "target": target_name_value or None,
            "source": source_filename,
            "csv_text_column": text_column,
            "csv_target_column": target_column,
            "csv_row_count": len(statements),
            "csv_classifier_reasoning": classifier_response.get("reasoning", ""),
        },
    }


def _build_csv_statements_media_entry(
    *,
    payload: dict[str, Any],
    source_filename: str,
    user_id: str,
    assistant_id: str,
) -> dict[str, Any]:
    """Render the CSV preprocessing payload as a JSON-typed media_files entry.

    The downstream process_media_graph already routes ``application/json``
    files with a ``.json`` suffix through the JSON handler in
    ``process_media_graph/utils/nodes.py``, which now understands the
    ``{"statements": [...]}`` shape produced by CSV preprocessing.
    """
    statements_blob = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    metadata = payload.get("metadata") or {}
    return {
        "filename": source_filename,
        "content_type": "application/json",
        "content": statements_blob,
        "user_id": user_id,
        "assistant_id": assistant_id,
        "reference_audio": False,
        "reference_image": False,
        "base64_encoded_str": make_data_uri("application/json", statements_blob),
        "csv_target_name": metadata.get("target"),
        "csv_text_column": metadata.get("csv_text_column"),
        "csv_target_column": metadata.get("csv_target_column"),
        "csv_row_count": metadata.get("csv_row_count"),
        "namespace_filename": source_filename
        if not "." in source_filename
        else _namespace_safe_formatted_filename(source_filename),
    }


# ---------------------------------------------------------------------------
# Tabular JSON ingest preprocessing
#
# A JSON upload can be the SAME table a CSV would carry — e.g. a pandas
# ``DataFrame.to_json()`` dump (orient="columns": ``{column: {row_key: value}}``)
# or orient="records" (a list of flat dicts). Those are detected here and pushed
# through the exact CSV pipeline (``_rows_to_statements_payload``) so the
# process_media graph only ever sees the ``{"statements": [...]}`` contract.
# JSON that is already contract-shaped (``{"statements": [...]}`` or
# ``{"messages": [...]}``, including JSON-Lines files of statement objects)
# passes through untouched — the graph's JSON handler owns those shapes.
# ---------------------------------------------------------------------------

_JSON_MIME_HINTS = frozenset(
    {
        "application/json",
        "application/x-ndjson",
        "application/jsonl",
        "application/json-lines",
    }
)


def _is_json_upload(filename: str, content_type: str) -> bool:
    """True when the upload looks like JSON / JSON-Lines by MIME type or filename."""
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct in _JSON_MIME_HINTS:
        return True
    name = (filename or "").strip().lower()
    return name.endswith((".json", ".jsonl", ".ndjson"))


def _is_tabular_scalar(value: Any) -> bool:
    """True for cell values a table can hold (no nested containers)."""
    return value is None or isinstance(value, (str, int, float, bool))


def _tabular_cell_to_str(value: Any) -> str:
    """Render a scalar cell the way ``_parse_csv_to_rows`` renders CSV cells."""
    return "" if value is None else str(value)


def _normalize_tabular_json_to_rows(
    parsed: Any,
) -> Optional[tuple[list[str], list[dict]]]:
    """Return ``(headers, rows)`` when ``parsed`` JSON is a flat table, else None.

    Recognized tabular shapes (both produced by ``pandas.DataFrame.to_json``):

    * orient="columns" — ``{column_name: {row_key: scalar}}``;
    * orient="records" — ``[{column_name: scalar}, ...]``.

    A dict carrying the avatar-identity contract keys (``statements`` /
    ``messages`` lists) is never treated as a table, and any nested container
    cell disqualifies the shape — those payloads pass through to the
    process_media graph unchanged.
    """
    if isinstance(parsed, dict):
        if isinstance(parsed.get("statements"), list) or isinstance(
            parsed.get("messages"), list
        ):
            return None
        if not parsed or not all(
            isinstance(column_values, dict) for column_values in parsed.values()
        ):
            return None
        for column_values in parsed.values():
            if not all(
                _is_tabular_scalar(cell_value)
                for cell_value in column_values.values()
            ):
                return None
        headers = [str(column_name) for column_name in parsed.keys()]
        # Row keys in first-seen order across columns (columns may be sparse).
        row_keys: list[Any] = []
        seen_row_keys: set[Any] = set()
        for column_values in parsed.values():
            for row_key in column_values.keys():
                if row_key not in seen_row_keys:
                    seen_row_keys.add(row_key)
                    row_keys.append(row_key)
        if not row_keys:
            return None
        rows = [
            {
                header: _tabular_cell_to_str(column_values.get(row_key))
                for header, column_values in zip(headers, parsed.values())
            }
            for row_key in row_keys
        ]
        return headers, rows

    if isinstance(parsed, list):
        if not parsed or not all(isinstance(record, dict) for record in parsed):
            return None
        headers = []
        seen_headers: set[str] = set()
        for record in parsed:
            for column_name, cell_value in record.items():
                if not _is_tabular_scalar(cell_value):
                    return None
                if column_name not in seen_headers:
                    seen_headers.add(column_name)
                    headers.append(str(column_name))
        if not headers:
            return None
        rows = [
            {header: _tabular_cell_to_str(record.get(header)) for header in headers}
            for record in parsed
        ]
        return headers, rows

    return None


async def _tabular_json_to_statements_payload(
    *, raw: bytes, source_filename: str
) -> Optional[dict[str, Any]]:
    """Convert a tabular JSON / JSON-Lines upload into the statements document.

    Returns None when the payload is not a flat table (contract-shaped JSON,
    arbitrary JSON, undecodable bytes) so the caller passes the file through to
    the process_media graph unchanged.
    """
    text = _decode_csv_bytes(raw)
    parsed: Any
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Possibly JSON-Lines: one object per line. A records-of-scalars file is
        # a table; statement-shaped lines come back None from the normalizer and
        # the graph's JSON-Lines parser handles them instead.
        line_objects: list[Any] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                line_objects.append(json.loads(line))
            except json.JSONDecodeError:
                return None
        if not line_objects:
            return None
        parsed = line_objects

    normalized = _normalize_tabular_json_to_rows(parsed)
    if normalized is None:
        return None
    headers, rows = normalized
    return await _rows_to_statements_payload(
        headers=headers, rows=rows, source_filename=source_filename
    )


@app.get("/avatar_reference_image")
async def get_avatar_reference_image(
    request: Request,
    assistant_id: str,
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    """Return stored reference image data URI or image URL string for UI avatars.

    Lookup uses the assistant owner's store namespace so anonymous chatters see the
    same portrait that the chat-time consciousness loader reads.
    """
    store = app.state.store
    if request.headers.get("api-key", "") != "":
        langgraph_client_headers = {"API-KEY": request.headers.get("api-key")}
    else:
        langgraph_client_headers = {"API-KEY": app.state.context.anonymous_api_key}
    try:
        langgraph_client = get_client(headers=langgraph_client_headers)
        assistant = await langgraph_client.assistants.get(assistant_id=assistant_id)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not load assistant: {exc}"
        ) from exc
    assistant_owner_user_id = (assistant.get("metadata") or {}).get("user_id")
    if not assistant_owner_user_id:
        return JSONResponse({"reference_image_data": None})
    namespace = (assistant_owner_user_id, assistant_id, "reference_image")
    item = await store.aget(namespace, assistant_id)
    if item is None:
        return JSONResponse({"reference_image_data": None})
    if isinstance(item, dict):
        value = item.get("value") or {}
    else:
        value = getattr(item, "value", None) or {}
    return JSONResponse({"reference_image_data": value.get("reference_image_data")})


from typing import Optional

_MANIFEST_TEXT_MIMES = frozenset(
    {"text/plain", "text/markdown", "application/octet-stream"}
)


def _looks_like_manifest_candidate(filename: str, mime_type: str) -> bool:
    """True for uploads that could be a newline-delimited URL list (.txt/.md).

    CSVs are handled separately and excluded by the caller. Octet-stream is
    allowed because browsers often send .txt/.md that way.
    """
    name = (filename or "").lower()
    return (
        mime_type in _MANIFEST_TEXT_MIMES
        or mime_type.startswith("text/")
        or name.endswith(".txt")
        or name.endswith(".md")
    )


def _extract_manifest_urls(raw: bytes) -> List[str]:
    """Pull the http(s) URLs out of a text/markdown manifest.

    A line counts only if, stripped, it is *itself* a single URL — name/header
    lines (e.g. ``Gracie Abrams``) and prose are ignored, so the same parser
    handles a pure list (``confirmed_search_results_list.txt``) and a
    name+URL playlist list. Returns ``[]`` when no bare-URL line is present, in
    which case the caller ingests the file as an ordinary text document.
    Order is preserved and duplicates dropped.
    """
    from urllib.parse import urlparse

    text = raw.decode("utf-8", errors="replace")
    seen: set[str] = set()
    urls: List[str] = []
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate or candidate.startswith("#"):
            continue
        # Markdown bullets / numbering before a bare URL: strip common prefixes.
        candidate = candidate.lstrip("-*•").strip()
        try:
            parsed = urlparse(candidate)
        except Exception:
            continue
        if (
            parsed.scheme in ("http", "https")
            and parsed.netloc
            and " " not in candidate
        ):
            if candidate not in seen:
                seen.add(candidate)
                urls.append(candidate)
    return urls


def _collect_input_urls(raw_url_field: Any) -> List[str]:
    """Flatten the ``url`` form field into an ordered, de-duplicated list of URLs.

    The field may arrive as a single string or as repeated ``url=`` form fields
    (FastAPI gives us ``list[str]``), and each value may itself hold several URLs
    pasted one-per-line (or whitespace-separated). Non-URL tokens — e.g. a name
    line above a playlist link — are ignored, mirroring ``_extract_manifest_urls``
    so the same paste works in the field or in an uploaded manifest. Order is
    preserved and duplicates dropped.

    The ``url`` field is typed as an array (``Optional[List[str]]``), so API
    browsers (Swagger "Try it out", the LangGraph API explorer, etc.) often
    present and submit it as a JSON-array literal — e.g. ``["https://…"]`` or a
    bare ``[https://…]`` — rather than a plain value. We strip the surrounding
    ``[ ]`` brackets, quotes and stray commas per token so a single URL works
    whether or not the user wraps it in quotes/brackets. We deliberately do *not*
    strip ``) }`` (keeps Wikipedia-style ``…_(disambiguation)`` URLs intact) and
    split on whitespace only (keeps comma-bearing query strings intact).
    """
    from urllib.parse import urlparse

    if raw_url_field is None:
        values: List[str] = []
    elif isinstance(raw_url_field, str):
        values = [raw_url_field]
    else:
        values = [v for v in raw_url_field if isinstance(v, str)]

    seen: set[str] = set()
    urls: List[str] = []
    for value in values:
        for token in re.split(r"\s+", value or ""):
            candidate = token.strip().strip("<>\"'[],").lstrip("-*•").strip()
            if not candidate:
                continue
            try:
                parsed = urlparse(candidate)
            except Exception:
                continue
            if parsed.scheme in ("http", "https") and parsed.netloc:
                if candidate not in seen:
                    seen.add(candidate)
                    urls.append(candidate)
    return urls


def _lightweight_url_media_entry(
    url_clean: str, *, user_id: str, assistant_id: str
) -> dict:
    """A generic ``page_url`` entry for bulk URLs — no upfront content-type probe.

    The media graph's ``URLDocumentLoaderClass`` classifies and expands it
    (youtube/playlist/twitter/instagram/twitch/linktree/article). Used for every
    URL in a multi-input or manifest request so the endpoint can return 202 fast
    instead of probing hundreds of URLs serially.
    """
    return {
        "filename": url_clean,
        "content_type": "text/html",
        "content": b"",
        "page_url": url_clean,
        "user_id": user_id,
        "assistant_id": assistant_id,
        "reference_audio": False,
        "reference_image": False,
        "namespace_filename": _namespace_safe_formatted_filename(url_clean),
    }


async def _build_media_entries_for_file(
    raw_name: str,
    content: bytes,
    mime_type: str,
    *,
    reference_image: bool,
    reference_audio: bool,
    user_id: str,
    assistant_id: str,
) -> list:
    """Build the ``media_files`` entries for a single uploaded file.

    Extracted verbatim from the original single-file branch so it can run per
    file in a multi-file request. Raises ``HTTPException`` on unsupported types.
    """
    entries: list = []
    if (
        not reference_image
        and not reference_audio
        and _is_csv_upload(raw_name, mime_type)
    ):
        csv_payload = await _csv_to_statements_payload(
            raw=content, source_filename=raw_name
        )
        entries.append(
            _build_csv_statements_media_entry(
                payload=csv_payload,
                source_filename=raw_name,
                user_id=user_id,
                assistant_id=assistant_id,
            )
        )
    elif not reference_image and not reference_audio and _is_json_upload(
        raw_name, mime_type
    ):
        # A JSON upload may be the same table a CSV would carry (a pandas
        # to_json dump). Convert tabular shapes through the CSV statements
        # pipeline; anything else (contract-shaped statements/messages JSON,
        # JSON-Lines of statement objects, arbitrary JSON) passes through as a
        # plain upload for the process_media graph's JSON handler.
        tabular_statements_payload = await _tabular_json_to_statements_payload(
            raw=content, source_filename=raw_name
        )
        if tabular_statements_payload is not None:
            entries.append(
                _build_csv_statements_media_entry(
                    payload=tabular_statements_payload,
                    source_filename=raw_name,
                    user_id=user_id,
                    assistant_id=assistant_id,
                )
            )
        else:
            entries.append(
                {
                    "filename": raw_name,
                    "content_type": mime_type,
                    "content": content,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "reference_audio": False,
                    "reference_image": False,
                    "base64_encoded_str": make_data_uri(mime_type, content),
                    "namespace_filename": raw_name if not "." in raw_name else _namespace_safe_formatted_filename(raw_name),
                }
            )
    elif reference_image:
        if mime_type.startswith("audio/"):
            raise HTTPException(
                status_code=400,
                detail="reference_image requires an image file, not audio.",
            )
        mime = validate_upload_image_bytes(mime_type, content)
        entries.append(
            {
                "filename": raw_name,
                "content_type": mime,
                "content": content,
                "user_id": user_id,
                "assistant_id": assistant_id,
                "reference_audio": False,
                "reference_image": True,
                "base64_encoded_str": make_data_uri(mime, content),
                "namespace_filename": raw_name
                if not "." in raw_name
                else _namespace_safe_formatted_filename(raw_name),
            }
        )
    elif reference_audio:
        if mime_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="reference_audio requires an audio file, not an image.",
            )
        sniff = _sniff_media_category_from_bytes(content[:512])
        effective = mime_type
        if mime_type == "application/octet-stream":
            if not sniff or not sniff.startswith("audio/"):
                raise HTTPException(
                    status_code=400,
                    detail="Could not determine an audio type from the upload.",
                )
            effective = sniff
        elif not mime_type.startswith("audio/") and not mime_type.startswith("video/"):
            raise HTTPException(
                status_code=400,
                detail="reference_audio requires an audio or video Content-Type.",
            )
        entries.append(
            {
                "filename": raw_name,
                "content_type": effective,
                "content": content,
                "user_id": user_id,
                "assistant_id": assistant_id,
                "reference_audio": True,
                "reference_image": False,
                "base64_encoded_str": make_data_uri(effective, content),
                "namespace_filename": raw_name
                if not "." in raw_name
                else _namespace_safe_formatted_filename(raw_name),
            }
        )
    else:
        sniff = _sniff_media_category_from_bytes(content[:512])
        if mime_type.startswith("image/") or (
            mime_type == "application/octet-stream" and sniff in ALLOWED_IMAGE_MIMES
        ):
            mime = validate_upload_image_bytes(mime_type, content)
            entries.append(
                {
                    "filename": raw_name,
                    "content_type": mime,
                    "content": content,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "reference_audio": False,
                    "reference_image": False,
                    "base64_encoded_str": make_data_uri(mime, content),
                    "namespace_filename": raw_name
                    if not "." in raw_name
                    else _namespace_safe_formatted_filename(raw_name),
                }
            )
        elif mime_type.startswith("audio/") or (
            mime_type == "application/octet-stream"
            and sniff
            and sniff.startswith("audio/")
        ):
            effective = (
                mime_type if mime_type.startswith("audio/") else (sniff or mime_type)
            )
            if not effective.startswith("audio/"):
                raise HTTPException(
                    status_code=400,
                    detail="Expected an audio upload.",
                )
            entries.append(
                {
                    "filename": raw_name,
                    "content_type": effective,
                    "content": content,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "reference_audio": False,
                    "reference_image": False,
                    "base64_encoded_str": make_data_uri(effective, content),
                    "namespace_filename": raw_name
                    if not "." in raw_name
                    else _namespace_safe_formatted_filename(raw_name),
                }
            )
        elif mime_type.startswith("video/"):
            entries.append(
                {
                    "filename": raw_name,
                    "content_type": mime_type,
                    "content": content,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "reference_audio": False,
                    "reference_image": False,
                    "base64_encoded_str": make_data_uri(mime_type, content),
                    "namespace_filename": raw_name
                    if not "." in raw_name
                    else _namespace_safe_formatted_filename(raw_name),
                }
            )
        elif mime_type == "application/pdf":
            entries.append(
                {
                    "filename": raw_name,
                    "content_type": mime_type,
                    "content": content,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "reference_audio": False,
                    "reference_image": False,
                    "base64_encoded_str": make_data_uri(mime_type, content),
                    "namespace_filename": raw_name
                    if not "." in raw_name
                    else _namespace_safe_formatted_filename(raw_name),
                }
            )
        elif (
            mime_type
            in (
                "text/plain",
                "application/json",
                "text/markdown",
                "application/octet-stream",
            )
            or mime_type.startswith("text/")
            or (raw_name or "").lower().endswith(".log")
        ):
            entries.append(
                {
                    "filename": raw_name,
                    "content_type": mime_type,
                    "content": content,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "reference_audio": False,
                    "reference_image": False,
                    "base64_encoded_str": make_data_uri(mime_type, content),
                    "namespace_filename": raw_name
                    if not "." in raw_name
                    else _namespace_safe_formatted_filename(raw_name),
                }
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported upload Content-Type {mime_type!r}.",
            )
    return entries


def _is_youtube_playlist_url_str(url_clean: str) -> bool:
    """Cheap, network-free check for a YouTube **playlist** URL.

    Used on the request path to *detect* playlists (so they can be enumerated
    later, in the background) without paying for ``yt_dlp``. The enumeration
    itself lives in ``_expand_youtube_playlist_to_media_entries``.
    """
    from src.anubis.utils.classes.URLDocumentLoaderClass import _classify_url

    return _classify_url(url_clean) == "youtube_playlist"


async def _expand_youtube_playlist_to_media_entries(
    url_clean: str,
    *,
    user_id: str,
    assistant_id: str,
    create_reference_media_from_playlist: bool = False,
) -> Optional[list]:
    """Expand a YouTube **playlist** URL into one media entry per video.

    Each video becomes its own top-level item — and therefore its own child job
    with its own progress/cancel id — keyed by a single uuid5 over
    ``{playlist_ns}::{video_ns}`` and named ``{playlist}::{video}`` so the videos
    list, dedupe, and cancel individually rather than collapsing into a single
    playlist job. Playlist context (``playlist_url`` / title / ns) rides on each
    entry so the produced Documents get stamped, ``/list_avatar_documents`` groups
    every video under its playlist, and a whole-playlist delete can match them by
    ``playlist_namespace_filename``.

    Returns ``None`` for any non-playlist URL so the caller falls back to the
    normal single-URL path; returns ``[]`` if the playlist resolves to no videos.
    """
    # Lazy import (heavy yt_dlp path + cold-start convention). _classify_url is
    # pure; _extract_playlist_entries does the flat yt_dlp enumeration.
    from src.anubis.utils.classes.URLDocumentLoaderClass import (
        _classify_url,
        _extract_playlist_entries,
    )

    if _classify_url(url_clean) != "youtube_playlist":
        return None

    entries, playlist_title = await _extract_playlist_entries(url_clean)
    if not entries:
        logger.warning("YouTube playlist produced no entries: %s", url_clean)
        return []

    # playlist_ns mirrors URLDocumentLoaderClass._namespace_for so the composite
    # keys built here match what the graph would have produced — dedup stays
    # consistent across upload paths.
    playlist_ns = _namespace_safe_formatted_filename(url_clean)
    playlist_label = (playlist_title or url_clean).strip()
    # Per-video token estimate from the flat entry's duration — the SAME
    # duration source and formula the endpoint billed against at submit, so
    # child-job estimates match the batch's billed total.
    playlist_analysis_passes = int(
        GlobalContext().estimated_analysis_passes_per_document or 0
    )
    media_entries: list = []
    for entry in entries:
        video_id = entry.get("id")
        watch_url = entry.get("url") or (
            f"https://www.youtube.com/watch?v={video_id}" if video_id else None
        )
        if not watch_url:
            continue
        video_duration_seconds = float(entry.get("duration") or 0.0)
        video_estimated_tokens = estimate_media_item_tokens(
            "video",
            duration_seconds=(
                video_duration_seconds
                if video_duration_seconds > 0
                else ESTIMATED_AUDIO_FALLBACK_DURATION_SECONDS
            ),
            include_analysis=True,
            analysis_passes=playlist_analysis_passes,
        )
        video_ns = _namespace_safe_formatted_filename(watch_url)
        video_title = (entry.get("title") or "").strip()
        media_entries.append(
            {
                "filename": f"{playlist_label}::{video_title or watch_url}",
                "estimated_tokens": video_estimated_tokens,
                "content_type": "text/html",
                "content": b"",
                "page_url": watch_url,
                "user_id": user_id,
                "assistant_id": assistant_id,
                "reference_audio": False,
                "reference_image": False,
                "create_reference_media_from_playlist": create_reference_media_from_playlist,
                # Single opaque uuid5 over the composite so the store key carries
                # no ``::`` separator. The playlist a video belongs to is recovered
                # from playlist_namespace_filename below (and from playlist_url /
                # title for the listing), not by parsing this key.
                "namespace_filename": _namespace_safe_formatted_filename(
                    f"{playlist_ns}::{video_ns}"
                ),
                "playlist_url": url_clean,
                "playlist_namespace_filename": playlist_ns,
                "playlist_title": playlist_title,
                "video_title": video_title,
                "url_kind": "youtube_playlist_entry",
            }
        )
    logger.info(
        "Expanded YouTube playlist %s (%s) into %d per-video upload items",
        url_clean,
        playlist_title or "untitled",
        len(media_entries),
    )
    return media_entries


async def _build_media_entries_for_url(
    url_clean: str,
    *,
    reference_image: bool,
    reference_audio: bool,
    user_id: str,
    assistant_id: str,
    rich: bool,
) -> list:
    """Build the ``media_files`` entries for a single URL.

    ``rich=True`` runs the original per-URL content-type probing path (handles
    direct image/audio/video/csv URLs and reference flags) — used for a lone URL
    request. ``rich=False`` returns one lightweight ``page_url`` entry that the
    media graph classifies, avoiding an upfront probe per URL in bulk requests.
    """
    if not rich:
        return [
            _lightweight_url_media_entry(
                url_clean, user_id=user_id, assistant_id=assistant_id
            )
        ]

    entries: list = []
    namespace_safe_formatted_filename = _namespace_safe_formatted_filename(url_clean)

    if reference_image:
        body, header_ct = await fetch_remote_url_bytes(url_clean)
        img_mime = validate_upload_image_bytes(header_ct, body)
        entries.append(
            {
                "filename": url_clean,
                "content_type": img_mime,
                "content": b"",
                "image_url": url_clean,
                "user_id": user_id,
                "assistant_id": assistant_id,
                "reference_audio": False,
                "reference_image": True,
                "base64_encoded_str": make_data_uri(img_mime, body),
                "namespace_filename": namespace_safe_formatted_filename
                if not "." in namespace_safe_formatted_filename
                else _namespace_safe_formatted_filename(
                    namespace_safe_formatted_filename
                ),
            }
        )
    elif reference_audio:
        # YouTube watch pages report Content-Type: text/html. Bypass the
        # audio/* guard for those by pulling the audio track via yt_dlp.
        if _is_youtube_url(url_clean):
            from src.anubis.utils.classes.URLDocumentLoaderClass import (
                _download_youtube_audio_b64,
            )

            audio_data_uri, _suffix = await _download_youtube_audio_b64(url_clean)
            if not audio_data_uri:
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract audio from YouTube URL.",
                )
            entries.append(
                {
                    "filename": url_clean,
                    "content_type": "audio/mp3",
                    "content": b"",
                    "audio_url": url_clean,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "reference_audio": True,
                    "reference_image": False,
                    "base64_encoded_str": audio_data_uri,
                    "namespace_filename": _namespace_safe_formatted_filename(url_clean),
                }
            )
        else:
            await require_url_content_type_prefix(
                url_clean, "audio/", "Reference audio"
            )
            body, header_ct = await fetch_remote_url_bytes(url_clean)
            sniff = _sniff_media_category_from_bytes(body[:512])
            audio_mime = (
                header_ct
                if header_ct.startswith("audio/")
                else (sniff if sniff.startswith("audio/") else header_ct)
            )
            entries.append(
                {
                    "filename": url_clean,
                    "content_type": audio_mime,
                    "content": b"",
                    "audio_url": url_clean,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "reference_audio": True,
                    "reference_image": False,
                    "base64_encoded_str": make_data_uri(audio_mime, body),
                    "namespace_filename": url_clean
                    if not "." in url_clean
                    else _namespace_safe_formatted_filename(url_clean),
                }
            )
    else:
        # YouTube URLs probe as text/html but their payload is video/audio.
        # Route them directly to the URL pipeline so URLDocumentLoaderClass
        # can pull subtitles or audio via yt_dlp without us first
        # downloading the HTML page.
        if _is_youtube_url(url_clean):
            entries.append(
                {
                    "filename": url_clean,
                    "content_type": "text/html",
                    "content": b"",
                    "page_url": url_clean,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "reference_audio": False,
                    "reference_image": False,
                    "namespace_filename": _namespace_safe_formatted_filename(url_clean),
                }
            )
            ct = ""  # skip the per-Content-Type branches below
        else:
            ct = await probe_remote_url_content_type(url_clean)
        if not ct:
            pass
        elif _is_csv_upload(namespace_safe_formatted_filename or url_clean, ct):
            body, _header_ct = await fetch_remote_url_bytes(url_clean)
            csv_filename = (
                namespace_safe_formatted_filename
                if namespace_safe_formatted_filename.endswith((".csv", ".tsv"))
                else (f"{namespace_safe_formatted_filename or 'remote_table'}.csv")
            )
            csv_payload = await _csv_to_statements_payload(
                raw=body, source_filename=csv_filename
            )
            entries.append(
                _build_csv_statements_media_entry(
                    payload=csv_payload,
                    source_filename=csv_filename,
                    user_id=user_id,
                    assistant_id=assistant_id,
                )
            )
        elif ct.startswith("image/"):
            body, header_ct = await fetch_remote_url_bytes(url_clean)
            img_mime = validate_upload_image_bytes(header_ct, body)
            entries.append(
                {
                    "filename": url_clean,
                    "content_type": img_mime,
                    "content": b"",
                    "image_url": url_clean,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "reference_audio": False,
                    "reference_image": False,
                    "base64_encoded_str": make_data_uri(img_mime, body),
                    "namespace_filename": url_clean
                    if not "." in url_clean
                    else _namespace_safe_formatted_filename(url_clean),
                }
            )
        elif ct.startswith("audio/"):
            body, header_ct = await fetch_remote_url_bytes(url_clean)
            sniff = _sniff_media_category_from_bytes(body[:512])
            audio_mime = (
                header_ct
                if header_ct.startswith("audio/")
                else (sniff if sniff.startswith("audio/") else ct)
            )
            entries.append(
                {
                    "filename": url_clean,
                    "content_type": audio_mime,
                    "content": b"",
                    "audio_url": url_clean,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "reference_audio": False,
                    "reference_image": False,
                    "base64_encoded_str": make_data_uri(audio_mime, body),
                    "namespace_filename": url_clean
                    if not "." in url_clean
                    else _namespace_safe_formatted_filename(url_clean),
                }
            )
        elif ct.startswith("video/"):
            body, header_ct = await fetch_remote_url_bytes(url_clean)
            video_mime = header_ct if header_ct.startswith("video/") else ct
            entries.append(
                {
                    "filename": url_clean,
                    "content_type": video_mime,
                    "content": b"",
                    "video_url": url_clean,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "reference_audio": False,
                    "reference_image": False,
                    "base64_encoded_str": make_data_uri(video_mime, body),
                    "namespace_filename": url_clean
                    if not "." in url_clean
                    else _namespace_safe_formatted_filename(url_clean),
                }
            )
        elif ct.startswith("text/") or ct in (
            "application/json",
            "application/xml",
            "application/xhtml+xml",
            "application/javascript",
            "application/ld+json",
        ):
            body, header_ct = await fetch_remote_url_bytes(url_clean)
            doc_mime = (
                header_ct
                if (
                    header_ct.startswith("text/")
                    or header_ct
                    in (
                        "application/json",
                        "application/xml",
                        "application/xhtml+xml",
                        "application/javascript",
                        "application/ld+json",
                    )
                )
                else ct
            )
            entries.append(
                {
                    "filename": url_clean,
                    "content_type": doc_mime,
                    "content": b"",
                    "page_url": url_clean,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "reference_audio": False,
                    "reference_image": False,
                    "base64_encoded_str": make_data_uri(doc_mime, body),
                    "namespace_filename": url_clean
                    if not "." in url_clean
                    else _namespace_safe_formatted_filename(url_clean),
                }
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Could not map URL to a supported media type (Content-Type: {ct!r}).",
            )
    return entries


def _decode_entry_media_bytes(entry: dict) -> bytes | None:
    """Return an entry's media bytes from ``content`` or its base64 data URI."""
    content = entry.get("content")
    if content:
        return content
    base64_encoded_str = entry.get("base64_encoded_str")
    if not base64_encoded_str:
        return None
    payload = base64_encoded_str
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload)
    except Exception:  # noqa: BLE001 - caller treats undecodable as absent
        return None


def _image_dimensions_from_bytes(image_bytes: bytes) -> tuple[int, int]:
    """Return (width, height) pixels by parsing the image header with Pillow.

    ``Image.open`` reads only the header (no full raster decode) and Pillow's
    default decompression-bomb guard stays active. Any failure is an
    estimation error — image dimensions are always knowable from real image
    bytes, so an unparsable image must not reach the vision model unmetered.
    """
    import io

    from PIL import Image

    try:
        with Image.open(io.BytesIO(image_bytes)) as image_handle:
            width_pixels, height_pixels = image_handle.size
        return int(width_pixels), int(height_pixels)
    except Exception as image_error:  # noqa: BLE001 - fail-closed estimation
        raise TokenEstimationError(
            f"Could not read image dimensions: {image_error}"
        ) from image_error


def _extract_text_from_html_bytes(body: bytes) -> str:
    """Return the visible text of an HTML page for word-count estimation."""
    decoded = body.decode("utf-8", errors="ignore")
    try:
        from bs4 import BeautifulSoup

        return BeautifulSoup(decoded, "html.parser").get_text(" ")
    except ImportError:
        # Crude tag strip when bs4 is unavailable — good enough for a word count.
        return re.sub(r"<[^>]+>", " ", decoded)


def _extract_pdf_text_from_bytes(pdf_bytes: bytes, filename: str) -> str:
    """Extract a PDF's text (PyPDFLoader over a temp file) for estimation.

    The same loader the message path uses, run BEFORE any documents are
    created so the extracted text itself is what gets estimated. Raises
    ``TokenEstimationError`` when the PDF cannot be parsed — fail-closed.
    """
    from langchain_community.document_loaders import PyPDFLoader

    temp_pdf_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf.flush()
            temp_pdf_path = temp_pdf.name
        pdf_documents = PyPDFLoader(temp_pdf_path).load()
        return "\n\n".join(
            document.page_content
            for document in pdf_documents
            if hasattr(document, "page_content")
        )
    except Exception as pdf_error:  # noqa: BLE001 - fail-closed estimation
        raise TokenEstimationError(
            f"Could not extract text from PDF {filename!r}: {pdf_error}"
        ) from pdf_error
    finally:
        if temp_pdf_path:
            try:
                os.unlink(temp_pdf_path)
            except OSError:
                pass


_REMOTE_DURATION_PROBE_TIMEOUT_SECONDS = 8.0


async def _estimate_tokens_for_media_entry(entry: dict) -> int:
    """Estimate one media entry's total model tokens before any model call.

    Dispatches on the entry shape produced by ``_build_media_entries_for_file``
    and ``_build_media_entries_for_url`` and applies the estimation formulas in
    ``src/anubis/utils/billing/estimation.py``. Identity analysis is included
    only for entries the pipeline will actually analyze (reference-image and
    reference-audio uploads skip analysis), so estimates track the pipeline's
    real model usage.

    Fail-closed: anything that cannot be estimated raises
    ``TokenEstimationError`` (the endpoint answers 422 naming the item). The
    ONLY sanctioned fallback is an unknown duration on a confirmed audio/video
    item, which assumes ``ESTIMATED_AUDIO_FALLBACK_DURATION_SECONDS``.
    """
    context = GlobalContext()
    analysis_passes = int(context.estimated_analysis_passes_per_document or 0)
    include_analysis = not (
        entry.get("reference_audio") or entry.get("reference_image")
    )
    content_type = (entry.get("content_type") or "").split(";")[0].strip().lower()
    filename = entry.get("filename") or entry.get("page_url") or "item"

    def _estimate_for_duration(kind: str, duration_seconds: float | None) -> int:
        effective_duration = (
            duration_seconds
            if duration_seconds and duration_seconds > 0
            else ESTIMATED_AUDIO_FALLBACK_DURATION_SECONDS
        )
        return estimate_media_item_tokens(
            kind,
            duration_seconds=effective_duration,
            include_analysis=include_analysis,
            analysis_passes=analysis_passes,
        )

    # Images: bytes are always in hand (uploads carry ``content``; image URLs
    # were downloaded into the data URI when the entry was built).
    if content_type.startswith("image/") or entry.get("image_url"):
        image_bytes = _decode_entry_media_bytes(entry)
        if image_bytes is None and entry.get("image_url"):
            image_bytes, _header = await fetch_remote_url_bytes(entry["image_url"])
        if not image_bytes:
            raise TokenEstimationError(f"No image bytes available for {filename!r}.")
        width_pixels, height_pixels = _image_dimensions_from_bytes(image_bytes)
        return estimate_media_item_tokens(
            "image",
            width_pixels=width_pixels,
            height_pixels=height_pixels,
            include_analysis=include_analysis,
            analysis_passes=analysis_passes,
        )

    # Audio: measure the clip we already hold; duration-probe failure takes the
    # sanctioned fallback (the type IS confirmed audio).
    if content_type.startswith("audio/") or entry.get("audio_url"):
        from src.anubis.utils.utility import get_audio_duration_seconds

        duration_seconds = None
        media_payload = entry.get("base64_encoded_str")
        if media_payload:
            try:
                duration_seconds = await get_audio_duration_seconds(
                    media_payload, entry.get("filename")
                )
            except Exception as duration_error:  # noqa: BLE001 - sanctioned fallback
                logger.warning(
                    "Audio duration probe failed for %s (%s); assuming %.0f seconds.",
                    filename,
                    duration_error,
                    ESTIMATED_AUDIO_FALLBACK_DURATION_SECONDS,
                )
        return _estimate_for_duration("audio", duration_seconds)

    # Video: processed as its audio track — same formula over the video duration.
    if content_type.startswith("video/") or entry.get("video_url"):
        from src.anubis.utils.utility import get_video_duration_seconds

        duration_seconds = None
        media_payload = entry.get("base64_encoded_str")
        if media_payload:
            try:
                duration_seconds = await get_video_duration_seconds(
                    media_payload, entry.get("filename")
                )
            except Exception as duration_error:  # noqa: BLE001 - sanctioned fallback
                logger.warning(
                    "Video duration probe failed for %s (%s); assuming %.0f seconds.",
                    filename,
                    duration_error,
                    ESTIMATED_AUDIO_FALLBACK_DURATION_SECONDS,
                )
        return _estimate_for_duration("video", duration_seconds)

    # URLs: probe what the URL actually yields and estimate that media type.
    if entry.get("page_url"):
        page_url = entry["page_url"]

        # YouTube pages probe as text/html but their payload is video/audio —
        # read the duration from metadata only (nothing downloaded).
        if _is_youtube_url(page_url):
            from src.anubis.utils.utility import get_remote_video_duration_seconds

            duration_seconds = None
            try:
                duration_seconds = await asyncio.wait_for(
                    get_remote_video_duration_seconds(page_url),
                    timeout=_REMOTE_DURATION_PROBE_TIMEOUT_SECONDS,
                )
            except Exception as duration_error:  # noqa: BLE001 - sanctioned fallback
                logger.warning(
                    "Remote duration probe failed for %s (%s); assuming %.0f seconds.",
                    page_url,
                    duration_error,
                    ESTIMATED_AUDIO_FALLBACK_DURATION_SECONDS,
                )
            return _estimate_for_duration("video", duration_seconds)

        # Rich single-URL entries downloaded the page body into the data URI;
        # bulk lightweight entries carry nothing yet — probe the content type
        # and estimate the real media behind the URL.
        page_bytes = _decode_entry_media_bytes(entry)
        if page_bytes is None:
            probed_content_type = await probe_remote_url_content_type(page_url)
            if probed_content_type.startswith("image/"):
                image_bytes, _header = await fetch_remote_url_bytes(page_url)
                width_pixels, height_pixels = _image_dimensions_from_bytes(image_bytes)
                return estimate_media_item_tokens(
                    "image",
                    width_pixels=width_pixels,
                    height_pixels=height_pixels,
                    include_analysis=include_analysis,
                    analysis_passes=analysis_passes,
                )
            if probed_content_type.startswith("audio/"):
                from src.anubis.utils.utility import get_audio_duration_seconds

                audio_bytes, _header = await fetch_remote_url_bytes(page_url)
                duration_seconds = None
                try:
                    duration_seconds = await get_audio_duration_seconds(
                        base64.b64encode(audio_bytes).decode("ascii"), page_url
                    )
                except Exception:  # noqa: BLE001 - sanctioned fallback
                    pass
                return _estimate_for_duration("audio", duration_seconds)
            if probed_content_type.startswith("video/"):
                from src.anubis.utils.utility import get_remote_video_duration_seconds

                duration_seconds = None
                try:
                    duration_seconds = await asyncio.wait_for(
                        get_remote_video_duration_seconds(page_url),
                        timeout=_REMOTE_DURATION_PROBE_TIMEOUT_SECONDS,
                    )
                except Exception:  # noqa: BLE001 - sanctioned fallback
                    pass
                return _estimate_for_duration("video", duration_seconds)
            page_bytes, _header = await fetch_remote_url_bytes(page_url)
        page_text = _extract_text_from_html_bytes(page_bytes)
        return estimate_media_item_tokens(
            "text",
            word_count=count_words(page_text),
            include_analysis=include_analysis,
            analysis_passes=analysis_passes,
        )

    # PDFs: extract the text FIRST and estimate the extracted words.
    if content_type == "application/pdf":
        pdf_bytes = _decode_entry_media_bytes(entry) or b""
        pdf_text = _extract_pdf_text_from_bytes(pdf_bytes, filename)
        return estimate_media_item_tokens(
            "text",
            word_count=count_words(pdf_text),
            include_analysis=include_analysis,
            analysis_passes=analysis_passes,
        )

    # Everything else is text-shaped (plain text, markdown, CSV-statements
    # JSON, arbitrary JSON): decode and count the actual words.
    text_bytes = _decode_entry_media_bytes(entry) or b""
    text_content = text_bytes.decode("utf-8", errors="ignore")
    return estimate_media_item_tokens(
        "text",
        word_count=count_words(text_content),
        include_analysis=include_analysis,
        analysis_passes=analysis_passes,
    )


async def _estimate_media_entries_tokens(media_files: list) -> int:
    """Stamp ``estimated_tokens`` on every entry (probing in parallel) and sum.

    Bulk requests probe their URLs concurrently. Fail-closed: the first entry
    that cannot be estimated aborts the request with HTTP 422 naming the item —
    nothing unestimated may proceed to a model.
    """

    async def _estimate_one(entry: dict) -> int:
        try:
            estimated = await _estimate_tokens_for_media_entry(entry)
        except (TokenEstimationError, HTTPException) as estimation_error:
            item_name = entry.get("filename") or entry.get("page_url") or "item"
            detail = (
                estimation_error.detail
                if isinstance(estimation_error, HTTPException)
                else str(estimation_error)
            )
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Could not estimate token usage for {item_name!r}: {detail} "
                    "The request was not processed."
                ),
            ) from estimation_error
        entry["estimated_tokens"] = estimated
        return estimated

    estimates = await asyncio.gather(
        *(_estimate_one(entry) for entry in media_files)
    )
    return sum(estimates)


@app.post("/update_avatar_identity_with_media")
async def update_avatar_identity_with_media(
    files: OptionalUploadFiles = None,
    url: Annotated[Optional[List[str]], Form()] = None,
    assistant_id: Annotated[Optional[str], Form()] = None,
    reference_audio: Annotated[bool, Form()] = False,
    reference_image: Annotated[bool, Form()] = False,
    create_reference_media_from_playlist: Annotated[bool, Form()] = False,
    current_user: dict = Depends(get_current_user),
):
    # Context user_id, assistant_id
    """
    Upload media for processing and indexing.

    Accepts **any mix** of files and URLs in a single request — multiple files in
    ``files`` and/or one or more URLs in ``url`` (repeated ``url=`` fields and/or
    several URLs pasted one-per-line into a single field both work). A ``.txt`` /
    ``.md`` file whose lines are bare URLs is treated as a **manifest**: its URLs
    are expanded and processed individually (name/header lines are ignored), so a
    saved list like ``confirmed_search_results_list.txt`` works the same as pasting
    the URLs. A YouTube **playlist** URL is enumerated in the background (so the
    202 isn't blocked on yt_dlp) into one item per video — each its own upload
    with its own progress/cancel id, listed individually as ``{playlist}::{video}``;
    those child ids appear on the master's progress stream as
    ``playlist_child_added`` events. Every item is processed in parallel (bounded
    by ``media_processing_concurrency``); items whose key already exists for this
    avatar (see ``/list_avatar_documents``) are **skipped**, so re-uploading a
    large playlist only processes new videos. The endpoint returns ``202`` with a
    ``job_id`` immediately; progress streams from
    ``GET /media_job/{job_id}/progress``.

    Images must use real MIME types: ``image/jpeg``, ``image/png``, ``image/gif`` (non-animated),
    or ``image/webp`` (non-animated). Proprietary vs biographical classification is done inside
    the processing pipeline via structured model output (no ``proprietary_content`` flag).

    With **reference_image=true** or **reference_audio=true** the request must carry
    **exactly one** file or URL (a reference clip/image is a single item): the file
    or URL must be an allowed still image, or resolve to ``audio/*``, respectively.

    With **create_reference_media_from_playlist=true** the batch has **no single target speaker**:
    every detected speaker is the avatar. Audio/video items are still diarized (so
    no stored reference-audio clip is required and known-speaker labelling is
    skipped). With **multiple speakers**, each statement becomes one ``quote``
    training example whose question is the **preceding statement** (the first
    statement, having no predecessor, gets a synthesized question). With a
    **single speaker** the transcript is a monologue: it is classified normally
    (monologue / tweets_or_quotes), which stores it in the vectorstore, marks it
    analysis-acceptable, and makes it adapter-acceptable with a synthesized prompt.
    YouTube items are forced onto the audio/diarize path (subtitles, which carry no
    speaker turns, are skipped). Use it for playlists/recordings where all voices
    belong to the avatar. It is mutually exclusive with ``reference_image`` /
    ``reference_audio`` (which designate a single target) and applies to every item
    in the request, including expanded playlist children.
    """
    try:
        # Gate: only pro/premium tiers may update avatar identity with media.
        # Allotment and rate-limit enforcement run AFTER the media entries are
        # built and estimated, so the decision uses this request's real
        # estimated token cost — see the enforcement block below.
        enforce_tier_capability(current_user, TierCapability.UPLOAD)

        user_id = current_user["identities"][0]["user_id"]
        if not assistant_id:
            raise HTTPException(status_code=400, detail="assistant_id is required")

        token = current_user["API_KEY"]
        client = get_client(headers={"API-KEY": f"{token}"})
        try:
            assistant = await client.assistants.get(assistant_id)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Could not load assistant: {exc}"
            ) from exc
        assistant_meta = assistant.get("metadata") or {}
        creator_id = assistant_meta.get("user_id")
        if not creator_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Assistant metadata is missing the creator's user_id; "
                    "cannot verify upload permissions."
                ),
            )
        if user_id != creator_id:
            raise HTTPException(
                status_code=403,
                detail=(
                    "Only the creator of this avatar may upload media for it. "
                    "The signed-in user is not the assistant's creator."
                ),
            )

        config = {
            "configurable": {
                "user_id": user_id,
                "user_ctx": {"name": None, "description": None},
                "assistant_id": assistant_id,
                "assistant_ctx": {
                    "name": assistant.get("name"),
                    "description": assistant.get("description"),
                    "assistant_id": assistant_id,
                    "metadata": assistant_meta,
                },
            }
        }

        upload_list = [f for f in (files or []) if f is not None]
        non_empty_files = [
            f for f in upload_list if (getattr(f, "filename", None) or "").strip()
        ]
        # ``url`` may arrive as one field, repeated ``url=`` fields, or several URLs
        # pasted one-per-line into a single field. Flatten to an ordered,
        # de-duplicated list of bare URLs (name/header lines ignored).
        input_urls = _collect_input_urls(url)

        if not non_empty_files and not input_urls:
            raise HTTPException(
                status_code=400,
                detail="Send at least one file or url.",
            )
        if reference_image and reference_audio:
            raise HTTPException(
                status_code=400,
                detail="Use only one of reference_image or reference_audio.",
            )
        if create_reference_media_from_playlist and (
            reference_image or reference_audio
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "create_reference_media_from_playlist cannot be combined with "
                    "reference_image/reference_audio: a reference clip designates a "
                    "single target, while create_reference_media_from_playlist treats every detected "
                    "speaker as the target."
                ),
            )

        reference_mode = reference_image or reference_audio
        if reference_mode and (len(non_empty_files) + len(input_urls)) != 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    "reference_image/reference_audio requires exactly one file or "
                    "url (a reference clip or image is a single item)."
                ),
            )

        media_files: list = []
        # Playlist URLs detected on the request path but enumerated later, in the
        # background master task, so the 202 isn't blocked on yt_dlp. Each yields
        # one child job per video once expanded.
        playlist_urls: list[str] = []

        if reference_mode:
            # Exactly one input (guarded above). No manifest expansion in reference
            # mode — a reference must be a single image/audio item.
            if non_empty_files:
                uf = non_empty_files[0]
                content = await uf.read()
                mime_type = (
                    (uf.content_type or "application/octet-stream")
                    .split(";")[0]
                    .strip()
                    .lower()
                )
                media_files.extend(
                    await _build_media_entries_for_file(
                        uf.filename,
                        content,
                        mime_type,
                        reference_image=reference_image,
                        reference_audio=reference_audio,
                        user_id=user_id,
                        assistant_id=assistant_id,
                    )
                )
            else:
                media_files.extend(
                    await _build_media_entries_for_url(
                        input_urls[0],
                        reference_image=reference_image,
                        reference_audio=reference_audio,
                        user_id=user_id,
                        assistant_id=assistant_id,
                        rich=True,
                    )
                )
        else:
            # General path: build entries for every file (expanding any URL-manifest
            # .txt/.md into its URLs) and every URL. A lone file/URL keeps the rich
            # single-item path (direct image/audio/video/csv links + content
            # probing); anything larger defers URL classification/expansion to the
            # media graph so the endpoint returns 202 fast instead of probing
            # hundreds of URLs serially. Each item is processed in parallel there.
            file_entries: list = []
            manifest_urls: list[str] = []
            for uf in non_empty_files:
                content = await uf.read()
                raw_name = uf.filename
                mime_type = (
                    (uf.content_type or "application/octet-stream")
                    .split(";")[0]
                    .strip()
                    .lower()
                )
                # A .txt/.md (non-CSV) whose lines are bare URLs is a manifest:
                # expand it into URLs instead of ingesting it as a text document.
                if not _is_csv_upload(raw_name, mime_type) and (
                    _looks_like_manifest_candidate(raw_name, mime_type)
                ):
                    extracted = _extract_manifest_urls(content)
                    if extracted:
                        manifest_urls.extend(extracted)
                        continue
                file_entries.extend(
                    await _build_media_entries_for_file(
                        raw_name,
                        content,
                        mime_type,
                        reference_image=False,
                        reference_audio=False,
                        user_id=user_id,
                        assistant_id=assistant_id,
                    )
                )

            # Merge explicit + manifest URLs, de-duplicated, order preserved.
            all_urls: list[str] = []
            seen_urls: set[str] = set()
            for u in (*input_urls, *manifest_urls):
                if u not in seen_urls:
                    seen_urls.add(u)
                    all_urls.append(u)

            # Probe each URL up front only for a single lone URL; bulk requests use
            # lightweight entries the media graph classifies and expands.
            rich_urls = len(file_entries) == 0 and len(all_urls) == 1
            url_entries: list = []
            for u in all_urls:
                # A playlist is set aside for background enumeration (one child job
                # per video, expanded off the request path); non-playlist URLs take
                # the normal single-URL path here.
                if _is_youtube_playlist_url_str(u):
                    playlist_urls.append(u)
                    continue
                url_entries.extend(
                    await _build_media_entries_for_url(
                        u,
                        reference_image=False,
                        reference_audio=False,
                        user_id=user_id,
                        assistant_id=assistant_id,
                        rich=rich_urls,
                    )
                )

            media_files = [*file_entries, *url_entries]

        # A playlist-only upload has no ready media_files yet (its videos are
        # enumerated in the background), so only reject when nothing at all — no
        # files and no playlists — was found.
        if not media_files and not playlist_urls:
            raise HTTPException(
                status_code=400,
                detail="No processable media found in the request.",
            )

        # Stamp the batch-wide "no single target" flag onto every entry (top
        # level, alongside reference_audio/reference_image). convert_uploaded_
        # files_to_media reads it for audio/video/url items and threads it into
        # their metadata; expanded playlist children inherit it downstream.
        if create_reference_media_from_playlist:
            for entry in media_files:
                entry["create_reference_media_from_playlist"] = True

        # ------------------------------------------------------------------
        # Pre-request token estimation (fail-closed), then enforcement, then
        # metering — all BEFORE any model call happens. Every entry gets a
        # typed estimate (image dimensions, audio/video durations, extracted
        # text word counts); playlists are enumerated now so their videos'
        # durations are billed accurately at submit.
        # ------------------------------------------------------------------
        estimated_tokens_total = await _estimate_media_entries_tokens(media_files)

        playlist_estimated_tokens = 0
        if playlist_urls:
            from src.anubis.utils.utility import get_remote_playlist_video_durations

            estimation_context = GlobalContext()
            playlist_analysis_passes = int(
                estimation_context.estimated_analysis_passes_per_document or 0
            )
            for playlist_url in playlist_urls:
                try:
                    playlist_video_durations = await get_remote_playlist_video_durations(
                        playlist_url
                    )
                except Exception as playlist_error:  # noqa: BLE001 - fail-closed
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"Could not enumerate playlist {playlist_url!r} to "
                            f"estimate token usage: {playlist_error} "
                            "The request was not processed."
                        ),
                    ) from playlist_error
                if not playlist_video_durations:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"Playlist {playlist_url!r} contains no videos to "
                            "estimate. The request was not processed."
                        ),
                    )
                for video_duration_seconds in playlist_video_durations:
                    playlist_estimated_tokens += estimate_media_item_tokens(
                        "video",
                        duration_seconds=(
                            video_duration_seconds
                            if video_duration_seconds > 0
                            else ESTIMATED_AUDIO_FALLBACK_DURATION_SECONDS
                        ),
                        include_analysis=True,
                        analysis_passes=playlist_analysis_passes,
                    )
        estimated_tokens_total += playlist_estimated_tokens

        # Enforce allotment and token rate with this request's estimate so an
        # over-budget upload is refused before anything is spent (admin
        # testing bypass happens inside the helpers).
        await enforce_remaining_allotment(
            app.state,
            current_user,
            UsageMeter.DOCUMENT_UPLOAD_TOKENS,
            estimated_request_tokens=estimated_tokens_total,
        )
        media_upload_rate_limit_context = GlobalContext()
        await enforce_token_rate_limit(
            app.state,
            current_user,
            meter_event_names=[UsageMeter.DOCUMENT_UPLOAD_TOKENS.value],
            window_seconds=int(
                media_upload_rate_limit_context.media_upload_rate_limit_window_seconds
                or 0
            ),
            tokens_per_window=int(
                media_upload_rate_limit_context.media_upload_rate_limit_tokens_per_window
                or 0
            ),
            estimated_request_tokens=estimated_tokens_total,
        )

        # Report the estimate against the document-upload meter (Stripe +
        # local api_metrics). Billing WRITES stay fail-open — only estimation
        # is fail-closed. The admin testing account is never metered.
        upload_admin_metering_bypass = is_admin_metering_bypass(
            current_user, GlobalContext().admin_user_id
        )
        if not upload_admin_metering_bypass and estimated_tokens_total > 0:
            try:
                await report_meter_event(
                    app.state.stripe,
                    UsageMeter.DOCUMENT_UPLOAD_TOKENS,
                    resolve_stripe_customer_id(current_user),
                    estimated_tokens_total,
                )
                await persist_api_metrics_row(
                    getattr(app.state, "pool", None),
                    inference_type="document_upload",
                    total_tokens=estimated_tokens_total,
                    user_id=resolve_metering_user_id(current_user),
                    stripe_customer_id=resolve_stripe_customer_id(current_user),
                    assistant_id=assistant_id,
                    meter_event_name=UsageMeter.DOCUMENT_UPLOAD_TOKENS.value,
                )
            except Exception as upload_metering_error:  # noqa: BLE001 - non-fatal
                logger.error("Failed to meter upload usage: %s", upload_metering_error)

        store = app.state.store

        # Collect every namespace_filename already indexed for this avatar. The
        # store layout ((user_id, assistant_id, <category>)) mirrors what
        # /list_avatar_documents exposes; keys are read from
        # value.document.kwargs.metadata.namespace_filename. This set is handed to
        # the media graph, which skips any incoming item — or expanded playlist /
        # linktree child — whose key is already present, so re-uploading a large
        # playlist only processes new entries (the user's "skip what's already
        # uploaded" requirement). To refresh an existing item, delete it first via
        # DELETE /delete_avatar_document, then re-upload.
        try:
            existing_items = await store.asearch(
                (user_id, assistant_id), limit=1_000_000
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Could not read this avatar's existing media to skip "
                    f"already-indexed items: {exc}"
                ),
            ) from exc

        existing_namespaces: set[str] = set()
        for item in existing_items or []:
            value = getattr(item, "value", None)
            if value is None and isinstance(item, dict):
                value = item.get("value")
            if not isinstance(value, dict):
                continue
            document = value.get("document")
            if not isinstance(document, dict):
                continue
            kwargs_blob = document.get("kwargs")
            if not isinstance(kwargs_blob, dict):
                continue
            metadata = kwargs_blob.get("metadata")
            if not isinstance(metadata, dict):
                continue
            stored_filename = metadata.get("namespace_filename")
            if isinstance(stored_filename, str) and stored_filename.strip():
                existing_namespaces.add(stored_filename.strip())

        incoming_filenames = [
            name
            for name in (
                (entry.get("namespace_filename") or "").strip() for entry in media_files
            )
            if name
        ]
        already_indexed = sorted(
            {name for name in incoming_filenames if name in existing_namespaces}
        )
        if already_indexed:
            logger.info(
                "Skipping %d top-level item(s) already indexed for this avatar: %s",
                len(already_indexed),
                already_indexed,
            )

        # Media processing (diarization, PDFs, YouTube playlists, indexing) can run
        # well past the request timeout, so start it as a background job and return
        # immediately. Each top-level item gets its own child job (its own progress
        # stream + independently cancellable); a master job aggregates them and is
        # the handle to cancel the whole batch. Progress is streamed via
        # GET /media_job/{job_id}/progress for either id; cancel via
        # POST /media_job/{job_id}/cancel. Bytes are already in ``media_files`` and
        # ``store`` / ``context`` are long-lived app resources, so the task is safe
        # after return. ``existing_namespaces`` lets the graph skip already-indexed
        # items and the children that expand from playlists/linktrees.
        registry = app.state.media_jobs
        master = create_master_job(registry, user_id, assistant_id)

        items: list = []
        item_descriptors: list = []
        for media_file in media_files:
            child = create_child_job(
                registry,
                user_id=user_id,
                assistant_id=assistant_id,
                parent_id=master.job_id,
                filename=media_file.get("filename"),
                namespace_filename=media_file.get("namespace_filename"),
                estimated_tokens=media_file.get("estimated_tokens"),
            )
            master.child_ids.append(child.job_id)
            items.append({"child": child, "media_file": media_file})
            item_descriptors.append(
                {
                    "job_id": child.job_id,
                    "filename": child.filename,
                    "status": child.status,
                    "estimated_tokens": media_file.get("estimated_tokens"),
                    "status_url": f"/media_job/{child.job_id}",
                    "progress_url": f"/media_job/{child.job_id}/progress",
                    "cancel_url": f"/media_job/{child.job_id}/cancel",
                }
            )

        # Playlists are enumerated inside the background task (off the request
        # path); each binds its URL + flags into an async expander that mints one
        # child job per video under this master once it resolves.
        deferred_expanders = [
            functools.partial(
                _expand_youtube_playlist_to_media_entries,
                playlist_url,
                user_id=user_id,
                assistant_id=assistant_id,
                create_reference_media_from_playlist=create_reference_media_from_playlist,
            )
            for playlist_url in playlist_urls
        ]

        master.task = asyncio.create_task(
            run_batch_media_job(
                master,
                items,
                config,
                store,
                app.state.context,
                concurrency=max(1, app.state.context.media_processing_concurrency),
                existing_namespaces=sorted(existing_namespaces),
                registry=registry,
                deferred_expanders=deferred_expanders,
            )
        )

        # Media now runs as a background job, so per-file indexing failures can
        # no longer be reported synchronously here. The failed-file logic that
        # fixed the silent-success bug lives in ``run_media_job``: it captures
        # ``failed_to_index_files`` from the graph and surfaces it on the job
        # result, delivered to clients via the SSE ``done`` event on
        # ``/media_job/{job_id}/progress``.
        return JSONResponse(
            status_code=202,
            content={
                "job_id": master.job_id,
                "status": master.status,
                "status_url": f"/media_job/{master.job_id}",
                "progress_url": f"/media_job/{master.job_id}/progress",
                "cancel_url": f"/media_job/{master.job_id}/cancel",
                "items_accepted": len(media_files),
                "filenames": [m.get("filename") for m in media_files],
                "items": item_descriptors,
                # Playlists resolve to their per-video child jobs in the background;
                # those child ids surface on the master's progress stream as
                # ``playlist_child_added`` events rather than in this response.
                "playlists_expanding": len(playlist_urls),
                # The full pre-request estimate (playlist videos included) that
                # was checked against the allotment and reported to the meter,
                # plus where the caller now stands against the allotment.
                "estimated_tokens_total": estimated_tokens_total,
                "usage": await _build_meter_usage_snapshot(
                    app.state, current_user, UsageMeter.DOCUMENT_UPLOAD_TOKENS
                ),
                **(
                    {"admin_metering_bypass": True}
                    if upload_admin_metering_bypass
                    else {}
                ),
                "message": (
                    "Media processing started; enumerating "
                    f"{len(playlist_urls)} playlist(s) in the background"
                    if playlist_urls
                    else "Media processing started"
                ),
            },
        )

    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing media: {str(e)}")


@app.get("/media_jobs")
async def list_media_jobs(
    include_finished: bool = False,
    assistant_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    """List the current user's media jobs — by default only the **active** ones.

    Returns one entry per top-level batch (the **master** job each upload creates),
    newest first, with rolled-up child status counts so a client can render an
    "uploads in progress" view without polling every ``/media_job/{job_id}``. A job
    is "active" while it is still ``queued`` or ``running`` (``done`` not yet set);
    finished jobs linger in the registry for ``_FINISHED_TTL_SECONDS`` and are only
    included when ``include_finished=true``. Pass ``assistant_id`` to scope the list
    to one avatar. The registry is per-process (see media_jobs.py), so this reflects
    jobs owned by the worker handling the request.
    """
    user_id = current_user["identities"][0]["user_id"]
    registry = app.state.media_jobs

    masters = [
        job
        for job in registry.values()
        if job.is_master
        and job.user_id == user_id
        and (assistant_id is None or job.assistant_id == assistant_id)
        and (include_finished or not job.done.is_set())
    ]
    masters.sort(key=lambda j: j.created_at, reverse=True)

    def _summary(job: MediaJob) -> dict:
        children = [registry[cid] for cid in job.child_ids if cid in registry]
        statuses = [c.status for c in children]
        return {
            "job_id": job.job_id,
            "assistant_id": job.assistant_id,
            "status": job.status,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "duration_seconds": job.duration_seconds,
            "children_total": len(children),
            "children_completed": statuses.count("completed"),
            "children_error": statuses.count("error"),
            "children_cancelled": statuses.count("cancelled"),
            "children_running": statuses.count("running"),
            "children_queued": statuses.count("queued"),
            "status_url": f"/media_job/{job.job_id}",
            "progress_url": f"/media_job/{job.job_id}/progress",
            "cancel_url": f"/media_job/{job.job_id}/cancel",
        }

    jobs = [_summary(job) for job in masters]
    return {"count": len(jobs), "jobs": jobs}


@app.get("/media_job/{job_id}")
async def media_job_status(
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Point-in-time snapshot of a media job — a pollable alternative to the SSE
    ``/progress`` stream.

    For a **master** job this lists its current child jobs, **including videos a
    YouTube playlist enumerated in the background** after the upload returned 202
    (those don't appear in the upload response because they don't exist yet at
    request time). Poll this to watch the queue fill in and drain; for a child job
    it returns that single item's status/result.
    """
    user_id = current_user["identities"][0]["user_id"]
    job_id = job_id.strip()
    registry = app.state.media_jobs
    job: Optional[MediaJob] = get_job(registry, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown or expired job_id")
    if job.user_id != user_id:
        raise HTTPException(status_code=403, detail="This job belongs to another user.")

    def _descriptor(j: MediaJob) -> dict:
        return {
            "job_id": j.job_id,
            "filename": j.filename,
            "namespace_filename": j.namespace_filename,
            "status": j.status,
            "estimated_tokens": j.estimated_tokens,
            "error": j.error,
            "created_at": j.created_at,
            "started_at": j.started_at,
            "finished_at": j.finished_at,
            "duration_seconds": j.duration_seconds,
            "progress_url": f"/media_job/{j.job_id}/progress",
            "cancel_url": f"/media_job/{j.job_id}/cancel",
        }

    snapshot: dict = {
        **_descriptor(job),
        "is_master": job.is_master,
        "result": job.result,
    }
    if job.is_master:
        children = [registry[cid] for cid in job.child_ids if cid in registry]
        statuses = [c.status for c in children]
        snapshot["children"] = [_descriptor(c) for c in children]
        snapshot["estimated_tokens_total"] = sum(
            c.estimated_tokens or 0 for c in children
        )
        snapshot["children_total"] = len(children)
        snapshot["children_completed"] = statuses.count("completed")
        snapshot["children_error"] = statuses.count("error")
        snapshot["children_cancelled"] = statuses.count("cancelled")
        snapshot["children_running"] = statuses.count("running")
        snapshot["children_queued"] = statuses.count("queued")
    return snapshot


@app.get("/media_job/{job_id}/progress")
async def media_job_progress(
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Stream progress (SSE) for a background media job started by
    ``/update_avatar_identity_with_media``.

    Replays any buffered ``media_progress`` events, then streams live ones with
    periodic keep-alive comments, ending with a ``done`` event carrying the final
    status and result (or error).
    """
    user_id = current_user["identities"][0]["user_id"]
    job_id = job_id.strip()
    job: Optional[MediaJob] = get_job(app.state.media_jobs, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown or expired job_id")
    if job.user_id != user_id:
        raise HTTPException(status_code=403, detail="This job belongs to another user.")

    def _with_timing(payload: dict) -> dict:
        """Return a copy of ``payload`` stamped with the job's start time and the
        wall-clock seconds elapsed since processing began, so every SSE ``data:``
        frame carries timing. ``started_at`` is epoch seconds (set when the job
        flipped to running); fall back to ``created_at`` if it hasn't yet."""
        started = job.started_at or job.created_at
        return {
            **payload,
            "started_at": job.started_at,
            "elapsed_seconds": round(time_ns() / 1_000_000_000 - started, 3),
        }

    async def event_stream(job: MediaJob):
        yield f"data: {json.dumps(_with_timing({'type': 'status', 'status': job.status}), default=str)}\n\n"
        last_index = 0
        while True:
            # Drain everything appended since we last yielded.
            while last_index < len(job.events):
                yield f"data: {json.dumps(_with_timing(job.events[last_index]), default=str)}\n\n"
                last_index += 1

            if job.done.is_set() and last_index >= len(job.events):
                break

            # Clear-then-recheck guards against a wakeup lost between the length
            # check above and the wait below.
            job._updated.clear()
            if last_index < len(job.events) or job.done.is_set():
                continue
            try:
                await asyncio.wait_for(job._updated.wait(), timeout=15)
            except asyncio.TimeoutError:
                # Keep the connection alive AND report timing so clients can show
                # how long the current stage has been running.
                yield f"data: {json.dumps(_with_timing({'type': 'keep_alive'}), default=str)}\n\n"

        done = _with_timing(
            {
                "type": "done",
                "status": job.status,
                "result": job.result,
                "error": job.error,
                "finished_at": job.finished_at,
                "duration_seconds": job.duration_seconds,
            }
        )
        yield f"data: {json.dumps(done, default=str)}\n\n"

    return StreamingResponse(
        event_stream(job),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _rollback_media_job_rows(
    *,
    user_id: str,
    assistant_id: Optional[str],
    item_job_ids: Optional[List[str]] = None,
    master_job_id: Optional[str] = None,
    namespace_filenames: Optional[List[str]] = None,
) -> int:
    """Best-effort delete of store rows a cancelled job/item already indexed.

    Documents are stamped with ``master_job_id`` / ``item_job_id`` at conversion
    time (see convert_media_list_to_text_document), so a cancel deletes exactly the
    rows that run wrote — including expanded playlist/linktree children whose
    ``namespace_filename`` differs from the top-level item. ``namespace_filenames``
    is an extra fallback for the top-level item key. Store rows removed CASCADE the
    matching store_vectors embeddings. Returns the number of rows deleted; never
    raises — rollback is best-effort ("attempt to delete").
    """
    if not assistant_id:
        return 0
    prefix_like = f"{user_id}.{assistant_id}.%"
    pool = app.state.pool
    total_deleted = 0
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                if master_job_id:
                    await cur.execute(
                        "DELETE FROM store WHERE prefix LIKE %s AND "
                        "value #>> '{document,kwargs,metadata,master_job_id}' = %s",
                        (prefix_like, master_job_id),
                    )
                    total_deleted += cur.rowcount or 0
                for item_job_id in item_job_ids or []:
                    await cur.execute(
                        "DELETE FROM store WHERE prefix LIKE %s AND "
                        "value #>> '{document,kwargs,metadata,item_job_id}' = %s",
                        (prefix_like, item_job_id),
                    )
                    total_deleted += cur.rowcount or 0
                for namespace_filename in namespace_filenames or []:
                    if not namespace_filename:
                        continue
                    await cur.execute(
                        "DELETE FROM store WHERE prefix LIKE %s AND "
                        "value #>> '{document,kwargs,metadata,namespace_filename}' = %s",
                        (prefix_like, namespace_filename),
                    )
                    total_deleted += cur.rowcount or 0
    except Exception as exc:  # noqa: BLE001 - rollback is best-effort
        logger.warning("Rollback for cancelled media job failed: %s", exc)
    return total_deleted


@app.post("/media_job/{job_id}/cancel")
async def cancel_media_job(
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Cancel a background media job and roll back what it already wrote.

    Send a **child** ``job_id`` to cancel one document: its processing stops and any
    store rows it already indexed are deleted (rolled back), leaving the rest of the
    batch running. Send the **master** ``job_id`` to do the same for the whole batch.
    Rollback is best-effort — typically a cancel lands before indexing completes, so
    there may be nothing to delete.
    """
    user_id = current_user["identities"][0]["user_id"]
    job_id = job_id.strip() # remove spaces and newline characters
    registry = app.state.media_jobs
    job: Optional[MediaJob] = get_job(registry, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown or expired job_id")
    if job.user_id != user_id:
        raise HTTPException(status_code=403, detail="This job belongs to another user.")

    # Flag + cancel the running task(s); returns the affected child jobs.
    targets = request_cancel(registry, job)

    deleted = await _rollback_media_job_rows(
        user_id=user_id,
        assistant_id=job.assistant_id,
        master_job_id=job.job_id if job.is_master else None,
        item_job_ids=[c.job_id for c in targets],
        namespace_filenames=[c.namespace_filename for c in targets],
    )

    return JSONResponse(
        status_code=200,
        content={
            "job_id": job.job_id,
            "scope": "batch" if job.is_master else "item",
            "status": "cancelled",
            "cancelled_items": [
                {"job_id": c.job_id, "filename": c.filename} for c in targets
            ],
            "rows_rolled_back": deleted,
            "message": (
                "Batch cancelled; processing stopped and indexed rows rolled back."
                if job.is_master
                else "Item cancelled; processing stopped and indexed rows rolled back."
            ),
        },
    )


@app.get("/list_avatar_documents")
async def list_avatar_documents(current_user: dict = Depends(get_current_user)):
    user_id = current_user["identities"][0]["user_id"]
    assistant_id = (
        current_user["app_metadata"]
        .get("assistant_config", {})
        .get("configurable", {})
        .get("assistant_id", None)
    )
    if assistant_id is None:
        raise HTTPException(
            detail="Please select an avatar before continuing.", status_code=400
        )

    # Read the avatar's store namespace in-process via app.state.store rather than
    # the LangGraph SDK HTTP client. The HTTP round-trip ConnectTimeouts while a
    # long media job is occupying the API process; this same in-process path is
    # what the upload endpoint's dedup uses.
    store = app.state.store
    try:
        all_document_items = await store.asearch(
            (user_id, assistant_id), limit=1_000_000
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read this avatar's documents: {exc}",
        ) from exc

    # Each source produces several Documents (quote / identity / analysis); the
    # set de-dupes them down to one entry per source. Playlist videos are listed
    # as ``{playlist} :: {video}`` and everything else by plain filename — see
    # _document_label_and_key, shared with /delete_avatar_document so a label
    # copied out of this list resolves back to the key delete needs.
    uploaded_documents: set[str] = {
        label for label, _key in _iter_document_labels(all_document_items)
    }

    return {"uploaded_documents": sorted(uploaded_documents)}


@app.delete("/delete_avatar_document")
async def delete_avatar_documents(
    source_document_name: str, current_user: dict = Depends(get_current_user)
):

    # Strip wrappers from copied SQL tuple/list output, e.g. ('Mom.m4a',) or "Mom.m4a",
    # leaving only the filename or already-derived namespace id.
    source_document_name = source_document_name.strip(" \t\n\r\"'`(),[]")
    # Keep the user-facing name for the response; source_document_name itself may
    # be rewritten below into an opaque hashed/composite store key.
    display_name = source_document_name
    user_id = current_user["identities"][0]["user_id"]
    assistant_id = (
        current_user["app_metadata"]
        .get("assistant_config", {})
        .get("configurable", {})
        .get("assistant_id", None)
    )
    if assistant_id is None:
        raise HTTPException(
            detail="Please select an avatar before continuing.", status_code=400
        )

    # Users delete by pasting a string straight out of /list_avatar_documents.
    # For a plain file that string IS the stored key (filename), but a playlist
    # video is listed by the human-readable ``{playlist_title} :: {video_title}``
    # label, whose words never equal the uuid5-hashed
    # ``{playlist_ns}::{video_ns}`` namespace_filename it's stored under — so the
    # raw label matches no row and delete 404s. Resolve the label back to its key
    # via the same helper /list builds labels with, so the two round-trip. Only
    # when nothing matches do we treat the input as a filename/URL and hash it
    # (this also avoids mangling a label that happens to contain a ".").
    try:
        existing_items = await app.state.store.asearch(
            (user_id, assistant_id), limit=1_000_000
        )
        label_to_key = {
            label: key for label, key in _iter_document_labels(existing_items) if key
        }
    except Exception:
        label_to_key = {}
    resolved_key = label_to_key.get(source_document_name)
    if resolved_key:
        source_document_name = resolved_key
    elif "." in source_document_name:
        source_document_name = _namespace_safe_formatted_filename(source_document_name)

    pool = app.state.pool

    # LangGraph store: prefix = namespace tuple dot-joined.
    # Match either chunk keys built from the filename prefix, or reference_* namespaces
    # (reference_image, reference_audio, …) where the serialized LangChain Document holds the
    # basename under value.document.kwargs.metadata.filename (same path as list_documents).
    # Rows removed from store CASCADE-delete matching store_vectors embeddings.
    # Playlist videos are keyed by a single opaque namespace_filename (a uuid5 over
    # ``{playlist_ns}::{video_ns}``) and carry their playlist's id under
    # value.document.kwargs.metadata.playlist_namespace_filename. Passing a bare
    # playlist namespace id (or its URL, hashed above) deletes the WHOLE playlist
    # via the playlist_namespace_filename value-match clause; passing a single
    # video's namespace_filename (resolved from its list label above) deletes that
    # one video via the prefix clauses. A plain (non-playlist) id matches no
    # playlist_namespace_filename, so that clause is inert for it.
    SQL_DELETE_DOCUMENT_QUERY = """
DELETE FROM store
WHERE (
    prefix = %s
    OR prefix LIKE %s
    OR prefix LIKE %s
    OR prefix LIKE %s
)
OR (
    prefix LIKE %s
    AND value #>> '{document,kwargs,metadata,playlist_namespace_filename}' = %s
)
OR (
    prefix LIKE %s
    AND value #>> '{document,kwargs,metadata,namespace_filename}' = %s
)
RETURNING value #>> '{document,kwargs,metadata,document_id}' AS document_id
"""
    total_deleted = 0
    # document_ids of the rows just deleted; used below to prune the stylometric
    # "direct quote" feature corpus so it no longer reflects removed documents.
    deleted_document_ids: set[str] = set()
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                params = (
                    f"{user_id}.{assistant_id}.{source_document_name}",
                    f"{user_id}.{assistant_id}.{source_document_name}.%",
                    f"{user_id}.{assistant_id}.%.{source_document_name}",
                    f"{user_id}.{assistant_id}.%.{source_document_name}.%",
                    # Whole playlist: every video whose playlist_namespace_filename
                    # equals this id. Scoped to this user/assistant via the prefix —
                    # playlist_ns is a deterministic hash of the playlist URL and is
                    # therefore shared across users who uploaded the same playlist,
                    # so an unscoped value match would cross avatars.
                    f"{user_id}.{assistant_id}.%",
                    source_document_name,
                    f"{user_id}.{assistant_id}.reference_%",
                    source_document_name,
                )
                await cur.execute(SQL_DELETE_DOCUMENT_QUERY, params)
                returned_rows = await cur.fetchall()
                total_deleted += len(returned_rows)
                deleted_document_ids = {
                    row[0] for row in returned_rows if row and row[0]
                }
    except Exception:
        raise HTTPException(detail="Error deleting documents.", status_code=500)

    if total_deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No stored rows matched document: {display_name}",
        )

    # The raw SQL above can remove the avatar's reference image (the
    # ``reference_%`` prefix clause) without going through the store client, so
    # drop the reference-image entry from the load_consciousness read-through
    # cache. Unconditional because the deleted rows are not inspected per
    # namespace; the invalidation is a dictionary pop either way.
    invalidate_store_cache_entry(
        (user_id, assistant_id, "reference_image"), assistant_id
    )

    # Prune the deleted documents' rows from the stylometric "direct quote"
    # feature corpus (a {document_id: [len(FEATURE_NAMES) floats]} dict in the store), then
    # recalibrate the empirical threshold + IsolationForest from what remains.
    # Best-effort: a failure here must not fail the (already committed) delete.
    try:
        await _prune_ground_truth_features_for_deleted_docs(
            user_id, assistant_id, deleted_document_ids
        )
    except Exception as exc:  # pragma: no cover - operator log only
        logger.warning(
            "ground-truth feature prune failed for %s: %s", assistant_id, exc
        )

    return JSONResponse(
        content=f"Successfully deleted: {display_name}", status_code=200
    )


async def _prune_ground_truth_features_for_deleted_docs(
    user_id: str | None, assistant_id: str | None, deleted_document_ids: set[str]
) -> None:
    """Remove deleted documents from the per-document stylometric feature corpus.

    Reads the ``{document_id: [len(FEATURE_NAMES) floats]}`` dict the avatar's "direct quote"
    corpus is stored under, drops every ``deleted_document_ids`` entry, then:

    * if rows remain — rebuilds the ``(n_docs, len(FEATURE_NAMES))`` array and recalibrates the
      empirical threshold + IsolationForest from it, persisting all the derived
      artifacts (dict, threshold, model, style profile);
    * if the corpus is now empty — deletes those keys so a later re-upload
      starts clean.

    All artifacts live under the owner-scoped ``(user_id, assistant_id,
    <artifact_name>)`` namespaces that ``calibrate_ground_truth`` writes.
    No-op when there is no user/assistant or nothing was deleted.
    """
    if not user_id or not assistant_id or not deleted_document_ids:
        return

    from src.anubis.utils.dataset.style_features import (
        GROUND_TRUTH_FEATURES_DICT_KEY,
        deserialize_features_by_doc_id,
        features_by_doc_id_to_arr,
        recompute_ground_truth_artifacts,
        serialize_features_by_doc_id,
    )

    store = app.state.store

    dict_namespace = (user_id, assistant_id, GROUND_TRUTH_FEATURES_DICT_KEY)
    threshold_namespace = (user_id, assistant_id, "ground_truth_text_empirical_threshold_list_str")
    model_namespace = (user_id, assistant_id, "ground_truth_text_features_model_b64_pkl")
    style_profile_namespace = (user_id, assistant_id, "style_profile")

    item = await store.aget(dict_namespace, key=GROUND_TRUTH_FEATURES_DICT_KEY)
    features_by_doc_id_str = (getattr(item, "value", None) or {}).get("value", None)
    features_by_doc_id = deserialize_features_by_doc_id(features_by_doc_id_str)
    if not features_by_doc_id:
        return

    # Drop the deleted documents' rows.
    removed = False
    for document_id in deleted_document_ids:
        if features_by_doc_id.pop(document_id, None) is not None:
            removed = True
    if not removed:
        return

    if not features_by_doc_id:
        # Corpus is now empty: clear the derived keys.
        await store.adelete(dict_namespace, key=GROUND_TRUTH_FEATURES_DICT_KEY)
        await store.adelete(threshold_namespace, key="ground_truth_text_empirical_threshold_list_str")
        await store.adelete(model_namespace, key="ground_truth_text_features_model_b64_pkl")
        await store.adelete(style_profile_namespace, key="style_profile")
        return

    # Rebuild the corpus array and recalibrate the derived artifacts.
    ground_truth_text_features_arr = features_by_doc_id_to_arr(features_by_doc_id)
    threshold_list_str, model_b64_pkl = recompute_ground_truth_artifacts(
        ground_truth_text_features_arr
    )

    from src.anubis.utils.dataset.style_features import build_style_profile_str
    style_profile_str = await build_style_profile_str(ground_truth_text_features_arr)

    await store.aput(
        dict_namespace,
        key=GROUND_TRUTH_FEATURES_DICT_KEY,
        value={"value": serialize_features_by_doc_id(features_by_doc_id)},
    )
    await store.aput(
        threshold_namespace,
        key="ground_truth_text_empirical_threshold_list_str",
        value={"value": threshold_list_str},
    )
    await store.aput(
        model_namespace,
        key="ground_truth_text_features_model_b64_pkl",
        value={"value": model_b64_pkl},
    )
    await store.aput(
        style_profile_namespace,
        key="style_profile",
        value={"value": style_profile_str},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
