# src/anubis/webapp.py

import asyncio
import time
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
from types import SimpleNamespace
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
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
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
    is_anonymous_user,
    canceled_tier_allotment_floor_applies,
    current_stripe_billing_config,
    initialize_stripe_billing_config,
    persist_api_metrics_row,
    plan_resubscribe_usage_window,
    plan_subscribe_action,
    plan_tier_change,
    report_meter_event,
    schedule_usage_notification,
    fetch_stripe_period_usage,
    reconcile_period_usage,
    resolve_canceled_tier_context,
    resolve_metering_bypass,
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
    minimum_tier_from_value,
    tier_allotment_for_meter,
    tier_from_value,
    tier_meets_minimum,
    token_rate_limit_retry_after_seconds,
)
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.geo import (
    GEO_LOCATION_METADATA_KEY,
    CheckinThrottle,
    GeoLocationError,
    avatars_near,
    build_geo_location,
    geo_location_of,
    validate_coordinates,
    within_bounds,
)
from src.anubis.utils.graph_interrupts import collect_pending_interrupts
from src.anubis.utils.avatar_deletion import (
    purge_avatar_data,
    search_all_avatars_for_user,
)
from src.anubis.utils.huggingface_prefetch import ensure_huggingface_models_cached
from src.anubis.utils.nltk_prefetch import ensure_nltk_corpora_cached
from src.anubis.utils.store_cache import (
    invalidate_store_cache_entry,
    invalidate_store_cache_for_assistant,
)
from src.anubis.utils.conversation_titles import (
    CONVERSATION_TITLE_SOURCE_KEY,
    MANUAL_TITLE_SOURCE,
    name_conversation_thread,
)
from src.api.message_stops import (
    AMBIENT_BUSY_RETRY_AFTER_SECONDS,
    STOP_REQUESTED,
    STREAM_ENDED,
    ActiveMessageTurnRegistry,
    GraphStreamPump,
    build_stopped_reply_metadata,
    discard_ambient_observation,
    persist_stopped_reply,
    schedule_background,
    yield_in_flight_ambient_observations,
)
from src.api.media_jobs import (
    MediaJob,
    create_child_job,
    create_master_job,
    estimate_processing_seconds,
    get_job,
    job_estimated_media_seconds,
    job_estimated_processing_seconds,
    request_cancel,
    run_batch_media_job,
)
from src.security.auth import (
    _tier_from_subscription,
    bearer_credentials_from_request,
    check_subscription_status,
    get_current_user,
    get_current_user_or_anonymous_user,
    get_current_user_or_anonymous_user_id,
    get_user,
    get_user_with_api_key,
    security_route,
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
    TierCapability.AUDIO_RESPONSES: SubscriptionTier.PRO,
    TierCapability.VIDEO_RESPONSES: SubscriptionTier.PREMIUM,
}


def enforce_tier_capability(
    current_user: dict, capability: TierCapability
) -> SubscriptionTier:
    """Raise HTTP 403 unless the user's resolved tier unlocks ``capability``.

    This is the enforcement layer that gates billable work by tier: every tier can
    message, pro adds uploads, premium adds adapter training. Anonymous users
    resolve to free and therefore reach only the message capability. Returns the
    resolved tier so callers can reuse it without recomputing.

    Every tier keeps its own feature set here, with NO exemptions. An account
    listed in ``UNRESTRICTED_METERED_ACCOUNT_IDENTIFIERS`` is uncapped WITHIN the
    tier the account holds — the allotment and the rate limit stop applying — but
    the tier still decides which capabilities exist at all, so a listed account on
    the free tier is refused uploads exactly like any other free-tier account and
    reaches uploads by changing tier. Keeping this gate tier-only is what makes a
    demonstration account a faithful demonstration of the tier being shown.
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


def resolve_emotion_media_minimum_tier() -> SubscriptionTier:
    """Return the lowest tier whose owner may generate an avatar's emotion media.

    Emotion media generation is priced by the vendor per image and per second
    of video, so which tier unlocks the feature is a deployment decision rather
    than a fixed capability: ``EMOTION_MEDIA_MINIMUM_TIER`` names the tier, and
    premium — the most capable tier in the catalog — is both the default and
    what a misconfigured value falls back to.
    """
    context = getattr(app.state, "context", None) or GlobalContext()
    return minimum_tier_from_value(
        getattr(context, "emotion_media_minimum_tier", None)
    )


def user_may_generate_emotion_media(current_user: dict) -> bool:
    """Whether this user's tier reaches ``EMOTION_MEDIA_MINIMUM_TIER``."""
    return tier_meets_minimum(
        resolve_tier(current_user), resolve_emotion_media_minimum_tier()
    )


def enforce_emotion_media_tier(current_user: dict) -> SubscriptionTier:
    """Raise HTTP 403 unless the user's tier reaches the emotion-media minimum."""
    tier = resolve_tier(current_user)
    minimum_tier = resolve_emotion_media_minimum_tier()
    if tier_meets_minimum(tier, minimum_tier):
        return tier
    raise HTTPException(
        status_code=403,
        detail=(
            f"Your '{tier.value}' tier does not permit generating emotion media "
            f"from a reference image. Upgrade to the {minimum_tier.value} tier."
        ),
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

    One deliberate exception overrides rule 1: while a canceled paid period is
    still being retained (``canceled_tier_allotment_floor_applies`` — the user
    was refunded mid-period and resubscribed to the same or a lower tier inside
    that period), the restored anchor is INTENTIONALLY older than the brand-new
    subscription's Stripe period start. Taking the maximum there would discard
    the retained window and hand the user a second full allotment for a period
    they were refunded for, so the anchor alone governs until the retained
    period closes.
    """
    context = GlobalContext()
    now = datetime.now(UTC)
    usage_period_days = int(context.usage_period_days or 0)
    period_anchor = resolve_usage_period_anchor(current_user)
    period_start = resolve_usage_period_start(now, usage_period_days, period_anchor)

    if canceled_tier_allotment_floor_applies(
        resolve_tier(current_user), resolve_canceled_tier_context(current_user), now
    ):
        return period_start

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


async def resolve_period_usage_to_date(
    app_state,
    current_user: dict,
    meter: UsageMeter,
    period_start: datetime,
) -> int:
    """Return this caller's authoritative usage for ``meter`` since ``period_start``.

    STRIPE IS THE SOURCE OF TRUTH. Usage is recorded in two places — Stripe's
    Billing Meter aggregation (what the customer portal displays) and the local
    ``api_metrics`` table (what gating historically read alone) — and because
    both writes are fail-open, the two ledgers drift permanently whenever one
    lands without the other. That drift is what let the portal show an anonymous
    visitor an exhausted allotment while this API still reported budget
    remaining and kept answering. ``reconcile_period_usage`` resolves the two
    into one number: Stripe governs, the local sum stands as a floor for usage
    Stripe has not finished aggregating, and the local sum governs alone when
    Stripe cannot be read.

    Every allotment decision and every usage display goes through here, so the
    402 boundary and the number the customer sees are always the same number.
    """
    local_usage = await fetch_usage_since(
        getattr(app_state, "pool", None),
        resolve_metering_user_id(current_user),
        meter.value,
        period_start,
        stripe_customer_id=resolve_stripe_customer_id(current_user),
    )
    stripe_usage = await fetch_stripe_period_usage(
        getattr(app_state, "stripe", None),
        current_stripe_billing_config(app_state),
        meter,
        resolve_stripe_customer_id(current_user),
        period_start,
    )
    return reconcile_period_usage(local_usage, stripe_usage)


async def enforce_remaining_allotment(
    app_state,
    current_user: dict,
    meter: UsageMeter,
    estimated_request_tokens: int = 0,
    assistant_id: str | None = None,
) -> None:
    """Block a user of ANY tier whose ``meter`` allotment cannot cover this request.

    The block decision is ``exhausted_allotment_block_reason``: period usage
    PLUS the pre-call estimate of this request under the allotment is allowed;
    at or past the allotment, only pay-per-use (a payment method on file lets
    the Stripe graduated metered price bill the overage) allows the request.
    Otherwise the request is refused with HTTP 402 until the period resets,
    pay-per-use is enabled, or the user upgrades tiers. Usage comes from
    ``resolve_period_usage_to_date`` — Stripe's meter aggregation (the source of
    truth the customer portal displays) reconciled with the local
    ``api_metrics`` table, which also covers anonymous users via their hashed-IP
    identifier — over the period resolved by
    ``resolve_usage_period_start_for_user``. The admin testing account, any
    identifier listed in ``ADMIN_METERING_BYPASS_IDENTIFIERS``, and (in
    development only) any identifier listed in
    ``DEV_METERED_ENFORCEMENT_BYPASS_IDENTIFIERS`` skip enforcement entirely;
    only the first two also stop being metered. An account listed in
    ``UNRESTRICTED_METERED_ACCOUNT_IDENTIFIERS`` likewise skips this refusal, in
    production as well as development, and may run past the allotment of whatever
    tier the account currently holds while every token stays metered; that
    account's tier still governs which capabilities exist at all.
    ``assistant_id`` is the avatar
    the request is aimed at: an anonymous visitor messaging an avatar listed in
    ``UNRESTRICTED_ANONYMOUS_MESSAGING_AVATAR_IDENTIFIERS`` also skips this
    refusal while remaining fully metered.
    """
    if resolve_metering_bypass(
        current_user, assistant_id=assistant_id
    ).skips_enforcement:
        return
    tier = resolve_tier(current_user)
    allotment = tier_allotment_for_meter(tier, meter)
    if allotment is None:
        # The capability gate is the authority for dimensions the tier lacks.
        return
    # A user inside a free-trial window keeps the trial tier's allotment as a
    # floor after changing tiers (e.g. trialing premium then downgrading to
    # pro keeps the premium allotment until trial_end), and a user who was
    # refunded mid-period and resubscribed to the same or a lower tier keeps
    # the allotment they already paid for until that period closes. The gate
    # must judge against that effective allotment, not the plain tier value.
    effective_allotment = resolve_effective_monthly_allotment(
        tier,
        meter,
        resolve_trial_context(current_user),
        canceled_tier_context=resolve_canceled_tier_context(current_user),
    )
    period_start = resolve_usage_period_start_for_user(current_user)
    period_usage = await resolve_period_usage_to_date(
        app_state, current_user, meter, period_start
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
    assistant_id: str | None = None,
) -> None:
    """Refuse the request with HTTP 429 when the user's token rate cap is met.

    A tokens-per-window abuse guard (in the spirit of the OpenAI rate-limit
    guide) independent of the monthly allotment and of pay-per-use: it caps how
    fast tokens can be consumed so a runaway client cannot burn a month's budget
    or an unbounded overage bill in minutes. The window usage is checked WITH
    this request's pre-call estimate added, so one huge request is refused
    before burning the cap rather than after. The Retry-After header tells the
    client when the oldest usage row ages out of the rolling window. A cap of
    zero or less disables the limit entirely; every requester that
    ``resolve_metering_bypass`` marks as skipping enforcement also skips the
    limit — an account listed in ``UNRESTRICTED_METERED_ACCOUNT_IDENTIFIERS``,
    and an anonymous visitor messaging an avatar listed in
    ``UNRESTRICTED_ANONYMOUS_MESSAGING_AVATAR_IDENTIFIERS``, which is why
    ``assistant_id`` is passed through: an unlimited demonstration avatar that
    still answered only until the rolling token cap was met would not be
    unlimited.
    """
    if tokens_per_window <= 0 or window_seconds <= 0:
        return
    if resolve_metering_bypass(
        current_user, assistant_id=assistant_id
    ).skips_enforcement:
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
    bounds. ``used_to_date`` is the reconciled, Stripe-authoritative figure
    (``resolve_period_usage_to_date``), so what a client is told matches both
    the customer portal and the number the 402 gate judges against.
    """
    tier = resolve_tier(current_user)
    # Trial- and refund-aware allotment so the streamed snapshot matches what
    # enforcement actually gates against during a free-trial window or a
    # retained paid period (see resolve_effective_monthly_allotment).
    allotment = resolve_effective_monthly_allotment(
        tier,
        meter,
        resolve_trial_context(current_user),
        canceled_tier_context=resolve_canceled_tier_context(current_user),
    )
    period_start, period_end = _resolve_usage_period_bounds_for_user(current_user)
    used_to_date = await resolve_period_usage_to_date(
        app_state, current_user, meter, period_start
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
    kept); a dev enforcement-only bypass keeps both writes, so its usage stays
    visible everywhere usage is read. Returns the turn's usage summary — this turn's actual token counts
    plus the caller's usage-versus-allotment snapshot for the governing meter —
    for endpoint responses; returns ``None`` when metering fails (the model
    already ran; post-response reporting stays fail-open).
    """
    try:
        token_usage = (response_metadata or {}).get("token_usage") or {}
        prompt_tokens = int(token_usage.get("prompt_tokens") or 0)
        completion_tokens = int(token_usage.get("completion_tokens") or 0)
        total_tokens = billable_tokens_from_metadata(response_metadata)
        # langchain_openai stamps model_name on the streamed reply; other providers
        # may not, so fall back to the configured model rather than dropping the
        # turn from the per-model Prometheus counters below.
        model_name = (response_metadata or {}).get("model_name") or getattr(
            getattr(app_state, "context", None), "model", None
        )
        cost_usd = float((response_metadata or {}).get("total_cost") or 0.0)

        stripe_customer_id = resolve_stripe_customer_id(current_user)
        tier = resolve_tier(current_user)
        # The avatar is passed so an unrestricted demonstration turn is LABELLED
        # as one on the usage payload. The bypass it resolves to skips no
        # metering write, so both ledgers below run exactly as for any other turn.
        metering_bypass = resolve_metering_bypass(
            current_user, assistant_id=assistant_id
        )

        # Adapter inference is billed against a separate meter at a different rate;
        # the think node sets is_adapter_inference when the client requested
        # adapter=True and the user is Premium (see use_adapter_inference in config).
        is_adapter_inference = bool(
            (response_metadata or {}).get("is_adapter_inference")
        )
        if is_adapter_inference and tier == SubscriptionTier.PREMIUM:
            meter = UsageMeter.ADAPTER_INFERENCE_TOKENS
            inference_type = "adapter_inference"
        else:
            meter = UsageMeter.MESSAGING_TOKENS
            inference_type = "message"

        # The request id keys Stripe's meter-event deduplication so a retried
        # request cannot double-bill the same turn.
        stripe_meter_event_accepted = False
        if not metering_bypass.skips_metering_writes:
            stripe_meter_event_accepted = await report_meter_event(
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
                MODEL_TOKENS_TOTAL.labels(model=model_name, type="prompt").inc(
                    prompt_tokens
                )
            if completion_tokens:
                MODEL_TOKENS_TOTAL.labels(model=model_name, type="completion").inc(
                    completion_tokens
                )
            if cost_usd:
                MODEL_COST_TOTAL.labels(model=model_name).inc(cost_usd)

        if not metering_bypass.skips_metering_writes:
            local_row_written = await persist_api_metrics_row(
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
            # Stripe accepted the usage but the local ledger did not record it:
            # the two ledgers have just diverged for this customer, permanently
            # and by exactly these tokens. Usage reads treat Stripe as the source
            # of truth so the divergence cannot mislead the customer, but it must
            # still be greppable when reconciling api_metrics against invoices.
            if stripe_meter_event_accepted and not local_row_written:
                logger.warning(
                    "Usage ledger divergence: Stripe accepted %s tokens on the %s "
                    "meter for customer %s (request %s) but the api_metrics row "
                    "was not written; the local ledger now under-counts this "
                    "customer by that amount.",
                    total_tokens,
                    meter.value,
                    stripe_customer_id,
                    request_id,
                )

        # The insert above was awaited, so the snapshot's used_to_date already
        # includes this turn.
        usage_snapshot = await _build_meter_usage_snapshot(
            app_state, current_user, meter
        )
        # Push that same reconciled figure to the customer portal so its meters
        # move now instead of waiting for Stripe's aggregation. Fire-and-forget
        # and fail-open: it never delays or breaks the reply, and a lost event is
        # corrected by the portal's own Stripe read.
        schedule_usage_notification(
            stripe_customer_id=stripe_customer_id,
            meter_event_name=meter.value,
            cumulative_period_usage=usage_snapshot.get("used_to_date"),
            usage_period_start=usage_snapshot.get("usage_period_start"),
            usage_period_end=usage_snapshot.get("usage_period_end"),
        )
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            # Dollar cost of this turn at the configured MODEL_PROMPT_COST /
            # MODEL_COMPLETION_COST rates (derived in graph._attach_token_usage_metadata).
            "cost_usd": cost_usd,
            **usage_snapshot,
            **metering_bypass.usage_response_fields(),
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


# The two store categories that hold an avatar's reference assets, in the order
# they are probed. A source is one or the other, never both: the reference image
# is an image upload, the reference audio an audio/video upload.
REFERENCE_DOCUMENT_CATEGORIES = ("reference_image", "reference_audio")


def indexed_namespace_filenames(existing_items: Any) -> set[str]:
    """The namespace_filename of every document already indexed for an avatar.

    Read from store rows laid out as ((user_id, assistant_id, <category>)),
    the same layout /list_avatar_documents exposes, taking the key from
    value.document.kwargs.metadata.namespace_filename. The set is handed to the
    media graph, which skips any incoming item — or expanded playlist / linktree
    child — whose key is already present, so re-uploading a large playlist only
    processes new entries. To refresh an existing item, delete it first via
    DELETE /delete_avatar_document, then re-upload.

    The reference categories are excluded. A reference clip or portrait
    DESIGNATES the avatar's voice sample or face; it is not indexed source
    material, and /list_avatar_documents does not list it. Counting one as
    already-indexed made the recording a video was used for permanently
    un-ingestable: uploading a YouTube interview as the reference audio wrote
    (user_id, assistant_id, "reference_audio") under that URL's
    namespace_filename, so every later upload of the same URL as identity media
    was skipped as a duplicate — the audio reached the voice model and the
    transcript never reached the graph, while the job still reported success.
    The media graph already exempts an incoming item that designates a reference
    asset (see determine_media_type); this is that exemption on the other side,
    for the row such an upload leaves behind.
    """
    indexed: set[str] = set()
    for item in existing_items or []:
        item_namespace = getattr(item, "namespace", None)
        if item_namespace is None and isinstance(item, dict):
            item_namespace = item.get("namespace")
        category = (
            item_namespace[2]
            if isinstance(item_namespace, (list, tuple)) and len(item_namespace) > 2
            else None
        )
        if category in REFERENCE_DOCUMENT_CATEGORIES:
            continue
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
            indexed.add(stored_filename.strip())
    return indexed


def _document_reference_role(
    metadata: dict, item_namespace: tuple | list | None
) -> str | None:
    """Name the reference role of one stored Document: the avatar's portrait
    ("reference_image"), its voice sample ("reference_audio"), or None for an
    ordinary source document.

    Two independent signals are read because neither one alone is present on
    every row that represents a reference asset:

    * ``metadata["reference_image"] / ["reference_audio"]`` — the boolean the
      upload carried. Set on the reference Document by ``process_media_graph``
      (``src/subgraphs/process_media_graph/utils/nodes.py``), but also set to
      *False* on every ordinary image, hence the truthiness test.
    * a corroborating category, taken either from ``metadata["namespace"]`` or
      from the namespace tuple the row itself lives under. The reference
      Document is written twice — once under ``(user_id, assistant_id,
      "reference_image" | "reference_audio")`` keyed by assistant_id, and once
      through the normal indexing path — and the copy stored under the
      reference namespace is serialized before the node stamps
      ``metadata["namespace"]``, so that copy is recognised by its namespace
      tuple while the indexed copy is recognised by its metadata.

    Requiring the flag *and* a matching category is what keeps an ordinary
    transcript that merely carries a stale ``reference_audio`` flag from being
    presented to the user as the avatar's voice sample.
    """
    namespace_category = metadata.get("namespace")
    trailing_namespace_element = (
        item_namespace[-1]
        if isinstance(item_namespace, (tuple, list)) and item_namespace
        else None
    )
    for reference_category in REFERENCE_DOCUMENT_CATEGORIES:
        if not metadata.get(reference_category):
            continue
        if (
            namespace_category == reference_category
            or trailing_namespace_element == reference_category
        ):
            return reference_category
    return None


def _iter_document_labels(
    store_items,
) -> Iterator[tuple[str, str | None, str | None]]:
    """Yield (label, key, reference_role) for each stored Document, de-structuring
    the same value.document.kwargs.metadata path /list and /delete read. Multiple
    Documents per source (quote / identity / analysis) yield the same triple; the
    caller de-dupes. ``reference_role`` is "reference_image", "reference_audio",
    or None — see _document_reference_role."""
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
        item_namespace = getattr(item, "namespace", None)
        if item_namespace is None and isinstance(item, dict):
            item_namespace = item.get("namespace")
        label, key = _document_label_and_key(metadata)
        if label:
            yield label, key, _document_reference_role(metadata, item_namespace)


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


async def _meter_image_description_usage(
    app_state,
    current_user: dict,
    payload: dict,
    *,
    assistant_id: Optional[str],
    thread_id: Optional[str],
    request_id: Optional[str],
) -> None:
    """Record one image-description call from the graph (``api_metrics`` + Stripe).

    Emitted by ``resolve_human_message_images`` as an ``image_description_usage``
    stream event. Best-effort like every other metering write; the admin
    testing account and the dev enforcement-only bypass skip the writes the
    same way the messaging meter does.
    """
    try:
        metering_bypass = resolve_metering_bypass(
            current_user, assistant_id=assistant_id
        )
        if metering_bypass.skips_metering_writes:
            return
        total_tokens = int(payload.get("total_tokens") or 0)
        stripe_customer_id = resolve_stripe_customer_id(current_user)
        if total_tokens > 0:
            await report_meter_event(
                app_state.stripe,
                UsageMeter.MESSAGING_TOKENS,
                stripe_customer_id,
                total_tokens,
                idempotency_identifier=(
                    f"{request_id}:image_description:{payload.get('source') or ''}"
                    if request_id
                    else None
                ),
            )
        await persist_api_metrics_row(
            getattr(app_state, "pool", None),
            inference_type="image_description",
            prompt_tokens=int(payload.get("input_tokens") or 0),
            completion_tokens=int(payload.get("output_tokens") or 0),
            total_tokens=total_tokens,
            cost_usd=float(payload.get("total_cost") or 0.0),
            latency_ms=float(payload.get("latency_ms") or 0.0),
            user_id=resolve_metering_user_id(current_user),
            stripe_customer_id=stripe_customer_id,
            assistant_id=assistant_id,
            thread_id=thread_id,
            model_name=payload.get("model_name"),
            meter_event_name=UsageMeter.MESSAGING_TOKENS.value,
        )
    except Exception:  # noqa: BLE001 - metering never fails the stream
        logger.debug("Could not meter an image description", exc_info=True)


async def _meter_stopped_turn(
    response_metadata: dict,
    *,
    app_state,
    current_user: Optional[dict],
    thread_id: str,
    assistant_id: str,
    request_id: str,
    start_time_ns: int,
) -> dict | None:
    """Meter the estimated usage of a reply that ended early.

    Split from ``_finalize_stopped_turn`` because the Stripe meter-event call
    can take seconds, and a person who pressed Stop is waiting on the stream's
    terminal frame: the stream sends ``done`` first and runs this afterwards
    in a background task. Returns the usage block, or ``None``.
    """
    if app_state is None or current_user is None:
        return None
    return await _meter_message_usage(
        app_state=app_state,
        current_user=current_user,
        response_metadata=response_metadata,
        thread_id=thread_id,
        assistant_id=assistant_id,
        latency_ms=(time_ns() - start_time_ns) / 1_000_000,
        request_id=request_id,
    )


def _thread_metadata_updates(
    *,
    user_id: str,
    assistant_id: str,
    thread_id: str,
    conversation_title_value: str | None,
) -> dict:
    """Build the thread_metadata keys one ``/message`` turn owns.

    ``conversation_title`` is written only when the caller actually sent a name.
    A caller that sends none leaves the key out entirely rather than writing
    ``None`` over whatever name the thread carries — writing ``None`` is how the
    name the conversation namer stored used to disappear on the reader's next
    message. A name the caller did send is the caller's own choice and is
    stamped as such, so the namer leaves that name alone.
    """
    updates: dict = {
        "user_id": user_id,
        "assistant_id": assistant_id,
        "most_recent_message": datetime.now(UTC).isoformat(),
    }
    caller_title = (conversation_title_value or "").strip()
    if caller_title and caller_title != thread_id:
        updates["conversation_title"] = caller_title
        updates[CONVERSATION_TITLE_SOURCE_KEY] = MANUAL_TITLE_SOURCE
    return updates


async def _write_thread_metadata(
    langgraph_client: Any, thread_id: str, updates: dict
) -> None:
    """Merge ``updates`` into a thread's nested ``thread_metadata`` and store it.

    The platform patches thread metadata one top-level key at a time, and
    ``thread_metadata`` is a single top-level key holding an object. Writing a
    freshly built object therefore replaces every key inside it, not just the
    keys being written. Each ``/message`` turn writes that object, so a name the
    conversation namer stored after the previous turn — and the ``shared`` flag
    a reader set on the conversation — were erased by the next message the
    reader sent. Reading the thread first and merging is what makes those keys
    outlive a turn.
    """
    existing_metadata: dict = {}
    existing_thread_metadata: dict = {}
    try:
        thread = await langgraph_client.threads.get(thread_id=thread_id)
        if isinstance(thread, dict) and isinstance(thread.get("metadata"), dict):
            existing_metadata = thread["metadata"]
            nested = existing_metadata.get("thread_metadata")
            if isinstance(nested, dict):
                existing_thread_metadata = nested
    except Exception:  # noqa: BLE001 - the write still goes out
        # A thread that cannot be read is usually a thread that does not exist
        # yet. Writing the update alone is right in that case; losing the keys
        # of a thread that does exist is the only cost of a read that failed.
        logger.debug(
            "Could not read thread %s before writing its metadata", thread_id, exc_info=True
        )
    await langgraph_client.threads.update(
        thread_id=thread_id,
        metadata={
            "thread_metadata": {**existing_thread_metadata, **updates},
            "graph_id": "Anubis",
        },
    )


async def _name_new_conversation(
    thread_id: str,
    *,
    langgraph_client_headers: dict,
    user_id: str,
    assistant_id: str,
) -> str:
    """Name a conversation that has no name yet, swallowing every failure.

    Run after the reply has been delivered, never before: the reader is waiting
    on the avatar's words, and a classification call made for the sidebar's
    benefit must not sit in front of them. A conversation that already carries a
    name is left alone, so this costs one call per conversation rather than one
    per turn, and it returns the name it wrote or an empty string.
    """
    try:
        return await name_conversation_thread(
            thread_id,
            langgraph_client_headers=langgraph_client_headers,
            only_when_unnamed=True,
            user_id=user_id,
            assistant_id=assistant_id,
        )
    except Exception:  # noqa: BLE001 - a name is a convenience, the reply is not
        logger.warning("Could not name conversation %s", thread_id, exc_info=True)
        return ""


async def _finalize_stopped_turn(
    partial_text: str,
    *,
    stopped_by: str,
    graph,
    config: dict,
    app_state,
    current_user: Optional[dict],
    thread_id: str,
    assistant_id: str,
    user_id: str,
    conversation_title_value: str | None,
    request_id: str,
    langgraph_client_headers: dict,
    estimated_prompt_tokens: int,
    start_time_ns: int,
    meter_inline: bool = True,
    discard_observation_id: str | None = None,
) -> dict:
    """Close out a reply that ended before the avatar finished.

    Records the partial reply on the thread (unless nothing was said), stamps
    the thread's ``most_recent_message``, and returns the terminal ``done``
    frame for the stream. The estimated token usage is metered here when
    ``meter_inline`` is true (nobody is waiting: a disconnected client); the
    stop route's stream passes false, sends the frame, and meters afterwards.
    ``stopped_by`` is ``"user"`` for the stop route and ``"disconnect"`` for a
    client that went away. ``discard_observation_id`` names an ambient
    observation that was stopped before the graph triaged it; that hidden
    message is removed from the thread rather than kept as context.
    """
    response_metadata = build_stopped_reply_metadata(
        partial_text,
        estimated_prompt_tokens=estimated_prompt_tokens,
        model_name=getattr(getattr(app_state, "context", None), "model", None),
        stopped_by=stopped_by,
    )
    if discard_observation_id:
        await discard_ambient_observation(graph, config, discard_observation_id)
    await persist_stopped_reply(
        graph,
        config,
        partial_text,
        request_id=request_id,
        response_metadata=response_metadata,
    )
    try:
        langgraph_client = get_client(headers=langgraph_client_headers)
        await _write_thread_metadata(
            langgraph_client,
            thread_id,
            _thread_metadata_updates(
                user_id=user_id,
                assistant_id=assistant_id,
                thread_id=thread_id,
                conversation_title_value=conversation_title_value,
            ),
        )
    except Exception:  # noqa: BLE001 - metadata is a convenience, the stop is not
        logger.warning(
            "Could not update thread %s metadata after a stopped reply",
            thread_id,
            exc_info=True,
        )
    done: dict = {
        "type": "done",
        "stopped": True,
        "stopped_by": stopped_by,
        "content": partial_text,
        "thread_id": thread_id,
        "request_id": request_id,
        "total_response_time_ms": (time_ns() - start_time_ns) // 1_000_000,
        "response_metadata": response_metadata,
    }
    if meter_inline:
        turn_usage = await _meter_stopped_turn(
            response_metadata,
            app_state=app_state,
            current_user=current_user,
            thread_id=thread_id,
            assistant_id=assistant_id,
            request_id=request_id,
            start_time_ns=start_time_ns,
        )
        if turn_usage:
            done["usage"] = turn_usage
    return done


async def _finalize_disconnected_turn(
    pump: GraphStreamPump,
    partial_text: str,
    active_turn=None,
    **stopped_turn_arguments,
) -> None:
    """Background finalizer for a reply whose client disconnected.

    ``active_turn`` is marked finished once the graph run is down and the
    record is written, so a typed turn waiting on the same thread can start.
    """
    try:
        await pump.aclose()
        await _finalize_stopped_turn(
            partial_text, stopped_by="disconnect", **stopped_turn_arguments
        )
    except Exception:  # noqa: BLE001 - nothing is listening any more
        logger.warning(
            "Could not finalize the disconnected reply %s",
            stopped_turn_arguments.get("request_id"),
            exc_info=True,
        )
    finally:
        if active_turn is not None:
            active_turn.mark_finished()


# What the model vendors say when the operator's own account is out of credit.
# None of these are the reader's doing, so they are never reported as a 402 —
# that status sends the reader to billing, and the reader's allotment is fine.
_VENDOR_CREDIT_EXHAUSTED_MARKERS = (
    "insufficient_quota",
    "exceeded your current quota",
    "credit balance is too low",
    "billing_hard_limit_reached",
    "insufficient credits",
    "insufficient_credits",
    "out of credits",
)


def _stream_error_frame(
    run_error: BaseException, *, request_id: str, thread_id: str | None
) -> dict:
    """The last frame of a message stream whose run failed after it began.

    The frame carries a ``status`` the client can act on the way it acts on an
    HTTP status — a 402 is a spent allotment and opens billing — and a
    ``message`` fit to show. A refusal the run itself raised as an
    ``HTTPException`` (a metering call inside the loop) keeps its status and
    sentence. The operator's model account being out of credit is said plainly
    and marked as the service's fault (503), never as the reader's. Anything
    else is a plain failure; its internals stay in the server log.
    """
    if isinstance(run_error, HTTPException):
        detail = run_error.detail
        code = None
        message = None
        if isinstance(detail, dict):
            code = detail.get("error")
            message = detail.get("detail") or detail.get("message")
        elif isinstance(detail, str):
            message = detail
        return {
            "type": "error",
            "status": int(run_error.status_code),
            "code": code,
            "message": message or "The avatar could not finish that reply.",
            "request_id": request_id,
            "thread_id": thread_id,
        }
    error_text = f"{type(run_error).__name__}: {run_error}".lower()
    if any(marker in error_text for marker in _VENDOR_CREDIT_EXHAUSTED_MARKERS):
        return {
            "type": "error",
            "status": 503,
            "code": "model_provider_credit_exhausted",
            "message": (
                "The avatar's model provider refused this reply because the "
                "service's credit with it is used up. This is on our side, not "
                "yours; please try again later."
            ),
            "request_id": request_id,
            "thread_id": thread_id,
        }
    return {
        "type": "error",
        "status": 500,
        "code": "turn_failed",
        "message": "The avatar could not finish that reply. Please try again.",
        "request_id": request_id,
        "thread_id": thread_id,
    }


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
    spoken_turn_frame: dict | None = None,
    ambient: bool = False,
):
    """Stream assistant tokens (SSE) then a terminal event with full metadata.

    ``ambient`` marks a hidden webcam / screen observation. Observations yield
    to the person: a typed or spoken turn (``ambient`` false) first stops any
    observation still running on the same thread and waits for the run behind
    that observation to wind down, so the two never write the thread's
    checkpoint at the same time. An observation stopped before the graph
    triaged the observation is removed from the thread.

    ``spoken_turn_frame`` (a ``diarize=true`` turn) is sent right after
    ``turn_started`` as a ``spoken_turn`` frame: the speaker-labelled script and
    segments of what the microphone heard, so the client can replace its
    placeholder bubble before any reply token arrives.

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
    ambient_decision: dict | None = None

    # The turn is registered before anything is sent so that
    # ``POST /message/{assistant_id}/stop`` can find it from the moment the
    # client learns its id; a stop that arrives before the graph stream exists
    # is remembered and honoured the instant the pump below is attached.
    turn_registry = getattr(app_state, "active_message_turns", None)
    active_turn = (
        turn_registry.register(
            request_id=request_id,
            thread_id=thread_id,
            assistant_id=assistant_id,
            user_id=user_id,
            ambient=ambient,
        )
        if turn_registry is not None
        else None
    )

    try:
        # The first frame names the turn. The client keeps ``request_id`` so it
        # can end the reply early through the stop route; ``thread_id`` rides
        # along because a first message on a new conversation has no other way
        # to learn its thread before the terminal frame.
        turn_started_event = {
            "type": "turn_started",
            "request_id": request_id,
            "thread_id": thread_id,
        }
        yield f"data: {json.dumps(turn_started_event, default=str)}\n\n"
        if spoken_turn_frame:
            yield f"data: {json.dumps({'type': 'spoken_turn', **spoken_turn_frame}, default=str)}\n\n"

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
                    app_state,
                    current_user,
                    estimate_meter or UsageMeter.MESSAGING_TOKENS,
                ),
                "thread_id": thread_id,
                "request_id": request_id,
            }
            yield f"data: {json.dumps(usage_estimate_event, default=str)}\n\n"
    except BaseException:
        # Nothing has been streamed from the graph yet (the client left, or the
        # usage snapshot failed): forget the turn so no stale entry lingers.
        if turn_registry is not None:
            turn_registry.unregister(request_id)
        raise

    graph_input = (
        resume_command if resume_command is not None else {"messages": [human_message]}
    )

    # The person's turn comes first: an ambient observation still being
    # processed on this thread is stopped and allowed to wind down before this
    # run touches the thread. Observations never wait for each other here (the
    # route refuses a second observation on a busy thread with 409 instead).
    if not ambient:
        try:
            await yield_in_flight_ambient_observations(turn_registry, thread_id)
        except BaseException:
            if turn_registry is not None:
                turn_registry.unregister(request_id)
            raise

    # The graph is drained on the pump's own task and consumed here through a
    # queue, so a stop request can wake this generator between frames and the
    # graph run can be cancelled from outside it (see ``message_stops``).
    # The identifier LangSmith will give this turn's root run. Pinning it here,
    # rather than letting the tracer mint one, is what lets the reply carry a
    # link that opens THIS message's run inside the conversation's thread —
    # without it the client can only link to the thread and the reader has to
    # hunt for the turn.
    langsmith_run_id = str(uuid4())
    config = {**config, "run_id": langsmith_run_id}

    pump = GraphStreamPump(
        graph.astream(
            input=graph_input,
            config=config,
            context=context,
            stream_mode=["custom", "updates"],
            subgraphs=True,
        )
    )
    if active_turn is not None:
        active_turn.attach_wake(pump.request_stop)
    stopped_by_user = False
    disconnected = False
    stopped_turn_arguments = dict(
        graph=graph,
        config=config,
        app_state=app_state,
        current_user=current_user,
        thread_id=thread_id,
        assistant_id=assistant_id,
        user_id=user_id,
        conversation_title_value=conversation_title_value,
        request_id=request_id,
        langgraph_client_headers=langgraph_client_headers,
        estimated_prompt_tokens=(
            estimated_request_tokens.input_tokens
            if estimated_request_tokens is not None
            else 0
        ),
        start_time_ns=start_time_ns,
    )

    try:
        while True:
            item = await pump.next_item()
            if item is STREAM_ENDED:
                break
            if item is STOP_REQUESTED:
                stopped_by_user = True
                break
            if not isinstance(item, tuple) or len(item) != 3:
                continue
            _ns, mode, payload = item
            if mode == "custom" and isinstance(payload, dict):
                if payload.get("type") == "assistant_token":
                    accumulated_chunks.append(payload.get("text") or "")
                    yield f"data: {json.dumps(payload)}\n\n"
                elif payload.get("type") == "status":
                    # The deep agent started or finished a tool. Forwarded so the
                    # client can say what the avatar is doing during a token-less
                    # stretch such as a data-analysis turn. The frame carries the
                    # human-readable phrase and the tool name, never tool arguments.
                    yield f"data: {json.dumps(payload, default=str)}\n\n"
                elif payload.get("type") == "media_job_started":
                    # The in-chat identity-update tool started a media batch; the
                    # client follows it on GET /media_job/{job_id}/progress.
                    yield f"data: {json.dumps(payload, default=str)}\n\n"
                elif payload.get("type") == "keepalive":
                    # SSE comment frame emitted during post-reply analysis (Go Emotions
                    # + SHAP), which yields no tokens yet gates the terminal ``done``
                    # frame. Comment lines are ignored by SSE parsers but the bytes
                    # reset the client's idle-read timer, preventing a premature
                    # "Error in input stream" while the metadata is computed.
                    yield ": keepalive\n\n"
                elif payload.get("type") == "ambient_decision":
                    # The graph triaged an ambient observation (ignore / respond /
                    # notify). Forwarded as its own frame so the client knows the
                    # decision before any reply tokens, and repeated on ``done``.
                    ambient_decision = {
                        key: value for key, value in payload.items() if key != "type"
                    }
                    yield f"data: {json.dumps(payload, default=str)}\n\n"
                elif payload.get("type") == "image_description_usage":
                    # One vision call described an attached image (a typed turn's
                    # attachment or an ambient snapshot). Metered here because the
                    # graph node has no request context; never surfaced as a frame.
                    if app_state is not None and current_user is not None:
                        await _meter_image_description_usage(
                            app_state,
                            current_user,
                            payload,
                            assistant_id=assistant_id,
                            thread_id=thread_id,
                            request_id=request_id,
                        )
            elif mode == "updates" and isinstance(payload, dict):
                ai = _latest_ai_from_stream_update(payload)
                if ai is not None:
                    last_ai = ai
    except (asyncio.CancelledError, GeneratorExit):
        # The client went away mid-reply (tab closed, network dropped, or the
        # browser aborted the fetch because the stop route was unreachable).
        # This task is being torn down and its own awaits would be cancelled
        # again, so winding the graph down and recording what was streamed is
        # handed to a task the cancellation does not reach.
        pump.cancel()
        disconnected = True
        schedule_background(
            _finalize_disconnected_turn(
                pump,
                "".join(accumulated_chunks),
                active_turn=active_turn,
                discard_observation_id=(
                    human_message.id if ambient and ambient_decision is None else None
                ),
                **stopped_turn_arguments,
            )
        )
        raise
    except Exception as run_error:  # noqa: BLE001 - reported to the client below
        # The run failed after the response had begun: the model vendor
        # refusing for want of credit, a metering call inside the loop refused
        # with 402, the graph raising. A 200 and some frames are already on the
        # wire, so there is no status code left to say why. Raising here would
        # cut the connection with no terminating chunk, and the browser reports
        # that as a network error with no reason attached — the client showed
        # "the response stream ended unexpectedly" for a spent credit balance.
        # Say why in a last ``error`` frame and end the stream cleanly instead.
        logger.exception(
            "Message turn %s on thread %s failed mid-stream", request_id, thread_id
        )
        pump.cancel()
        yield f"data: {json.dumps(_stream_error_frame(run_error, request_id=request_id, thread_id=thread_id), default=str)}\n\n"
        return
    finally:
        if turn_registry is not None:
            turn_registry.unregister(request_id)
        # A run that ended on its own (or failed) is over now; a stopped run is
        # marked once the pump has been closed and the record written, and a
        # disconnected one by the background finalizer.
        if active_turn is not None and not disconnected and not stopped_by_user:
            active_turn.mark_finished()

    if stopped_by_user:
        # The person pressed Stop. Cancel the graph run, keep whatever the
        # avatar had said so far on the thread, meter the estimated usage, and
        # end the stream with a ``done`` frame flagged ``stopped`` so the client
        # finalizes the bubble exactly as it would a completed reply.
        # Metering runs after the frame is sent: the Stripe meter-event call
        # can take seconds, and the person is waiting for the composer to
        # unlock. A stopped ``done`` therefore carries no ``usage`` block.
        await pump.aclose()
        try:
            done = await _finalize_stopped_turn(
                "".join(accumulated_chunks),
                stopped_by="user",
                meter_inline=False,
                discard_observation_id=(
                    human_message.id if ambient and ambient_decision is None else None
                ),
                **stopped_turn_arguments,
            )
        finally:
            if active_turn is not None:
                active_turn.mark_finished()
        yield f"data: {json.dumps(done, default=str)}\n\n"
        schedule_background(
            _meter_stopped_turn(
                done["response_metadata"],
                app_state=app_state,
                current_user=current_user,
                thread_id=thread_id,
                assistant_id=assistant_id,
                request_id=request_id,
                start_time_ns=start_time_ns,
            )
        )
        return

    langgraph_client = get_client(headers=langgraph_client_headers)
    await _write_thread_metadata(
        langgraph_client,
        thread_id,
        _thread_metadata_updates(
            user_id=user_id,
            assistant_id=assistant_id,
            thread_id=thread_id,
            conversation_title_value=conversation_title_value,
        ),
    )

    # If the graph paused for human approval, surface the preview instead of ``done``.
    # The client resumes via ``POST /message/{assistant_id}/resume`` on this thread_id.
    snapshot = await graph.aget_state(config)
    pending_interrupts = collect_pending_interrupts(snapshot)
    if pending_interrupts:
        interrupt_event: dict = {
            "type": "interrupt",
            "thread_id": thread_id,
            "request_id": request_id,
            "run_id": langsmith_run_id,
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
        # The turn's LangSmith run, so a reader can open this reply's own trace
        # rather than the whole conversation's thread.
        "run_id": langsmith_run_id,
        # The id the reply is stored under, so feedback pressed on the live
        # bubble is recorded against the same row the transcript reloads.
        "message_id": getattr(last_ai, "id", None) if last_ai is not None else None,
        "total_response_time_ms": (time_ns() - start_time_ns) // 1_000_000,
    }
    response_metadata = (
        last_ai.response_metadata
        if last_ai is not None and getattr(last_ai, "response_metadata", None)
        else None
    )
    if response_metadata:
        done["response_metadata"] = response_metadata
    if ambient_decision is not None:
        done["ambient"] = ambient_decision

    # Always accrue actual model API usage BEFORE the terminal frame. Reporting
    # the usage block on ``done`` is optional (``include_usage_metrics``);
    # metering failure is fail-open and just omits the usage block. An ignored
    # ambient observation produced no reply and no model usage, so the
    # messaging meter is left alone (the vision calls were metered above).
    ignored_ambient_turn = (
        ambient_decision is not None
        and ambient_decision.get("decision") == "ignore"
        and last_ai is None
    )
    if app_state is not None and current_user is not None and not ignored_ambient_turn:
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

    # The conversation is named after the reply has been delivered, never
    # before: the reader has the avatar's words in full by this frame, and the
    # naming call costs the reader nothing but a still-open connection. A
    # conversation that already carries a name is left alone, so this is one
    # call per conversation rather than one per turn.
    #
    # The name is sent as its own frame instead of being left for the next
    # listing of conversations, because a conversation is named on the turn
    # that starts it — exactly the turn on which the browser is putting a new
    # row in the sidebar. Handing the name over now is what stops that row
    # from reading "Conversation Sep 9, 10:14" until something else refreshes.
    conversation_name = await _name_new_conversation(
        thread_id,
        langgraph_client_headers=langgraph_client_headers,
        user_id=user_id,
        assistant_id=assistant_id,
    )
    if conversation_name:
        conversation_name_frame = {
            "type": "conversation_title",
            "thread_id": thread_id,
            "conversation_title": conversation_name,
        }
        yield f"data: {json.dumps(conversation_name_frame, default=str)}\n\n"


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


class ConnectMcpRequest(BaseModel):
    """Bind a Neural Nexus machine to the personal avatar immediately."""

    device_id: Optional[str] = None
    device_label: Optional[str] = None


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


async def resolve_assistant_for_creator(
    assistant_id: str,
    current_user: dict,
    action_description: str = "perform this action",
) -> tuple[dict, str]:
    """Load an avatar and confirm the signed-in caller is the creator of that avatar.

    Returns ``(assistant, creator_user_id)``.

    The creator is read from the avatar's ``metadata.user_id``, which ``create_avatar``
    stamps with the bare Auth0 identifier (``identities[0]["user_id"]``) — the same value
    every other ownership check in this module compares against, and the same value the
    avatar's store namespaces are keyed under. Callers therefore use the returned
    ``creator_user_id`` to scope store reads and writes rather than re-deriving it.

    ``action_description`` completes the sentence "Only the creator of this avatar may
    ___" in the 403 detail, so each caller reports the action the caller was refused.

    The administrator is exempt, the same way ``/delete_avatar`` and ``/share_avatar``
    are: that account may read and correct what any avatar has learned (and the other
    creator-gated settings this helper protects). Store rows stay keyed by the original
    ``creator_user_id``, so an admin acting on someone else's avatar does not move the
    data under the admin's id.

    Raises 400 when the avatar cannot be loaded or carries no creator, and 403 when the
    signed-in caller is not that creator and is not the administrator.
    """
    user_id = current_user["identities"][0]["user_id"]
    token = current_user["API_KEY"]
    client = get_client(headers={"API-KEY": f"{token}"})
    try:
        assistant = await client.assistants.get(assistant_id)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not load assistant: {exc}"
        ) from exc
    assistant_metadata = assistant.get("metadata") or {}
    creator_id = assistant_metadata.get("user_id")
    if not creator_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "Assistant metadata is missing the creator's user_id; "
                "cannot verify permissions for this avatar."
            ),
        )
    admin_user_id = getattr(getattr(app.state, "context", None), "admin_user_id", None)
    if user_id != creator_id and user_id != admin_user_id:
        # Named in the log because the 403 body deliberately says only that the
        # caller is not the creator: the two identifiers are what distinguishes a
        # correct refusal from a credential resolving to the wrong account, and
        # without them that distinction cannot be made after the fact.
        logger.warning(
            "Refusing to let %s %s: assistant %s was created by %s",
            user_id,
            action_description,
            assistant_id,
            creator_id,
        )
        raise HTTPException(
            status_code=403,
            detail=(
                f"Only the creator of this avatar may {action_description}. "
                "The signed-in user is not the assistant's creator."
            ),
        )
    return assistant, creator_id


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


def _assistant_without_metadata_if_public(
    assistant: dict[str, Any], viewer_user_id: str | None = None
) -> dict[str, Any]:
    """Hide a public avatar's metadata from everyone except its creator.

    The metadata carries ``user_id``, so stripping it is what keeps one user's
    identifier out of another user's listing. It must NOT be stripped from the
    creator's own copy: ``metadata.user_id`` is the only thing a client can
    compare against the signed-in user to decide whether that user may
    administer the avatar. Stripping it from the owner too made every avatar the
    owner had shared read as someone else's, so the Avatar Settings tab vanished
    the moment an avatar was made public — including from the personal avatar,
    the one avatar the product expects a user to share.

    ``viewer_user_id`` is the caller. Passing None keeps the unconditional
    behaviour, which is what an unauthenticated public listing wants.
    """
    meta = assistant.get("metadata")
    if isinstance(meta, dict):
        pub = meta.get("is_public")
        if pub is True or (isinstance(pub, str) and pub.lower() == "true"):
            if viewer_user_id is not None and meta.get("user_id") == viewer_user_id:
                return assistant
            return _assistant_without_metadata(assistant)
    return assistant


def _assistant_without_metadata(assistant: dict[str, Any]) -> dict[str, Any]:
    """Drop the metadata, keeping the real-world pin as a public top-level field.

    A geo-located avatar's pin is the one piece of metadata that is meant to be
    public: the pin is what places the avatar on the world map and what tells a
    passer-by that an avatar stands here. Everything else in the metadata — the
    creator's ``user_id`` above all — stays private, so the pin is lifted out
    rather than the metadata being kept.
    """
    stripped = {k: v for k, v in assistant.items() if k != "metadata"}
    pin = geo_location_of(assistant)
    if pin is not None:
        stripped[GEO_LOCATION_METADATA_KEY] = pin
    return stripped


def _build_geo_location_or_400(
    *,
    latitude: Optional[float],
    longitude: Optional[float],
    location_name: Optional[str],
    geofence_radius_meters: Optional[int],
) -> Optional[dict[str, Any]]:
    """Build the validated pin for a create or modify request, or None when unpinned.

    A caller that sends one half of a coordinate pair has made a mistake worth
    reporting rather than a pin worth guessing, so half a pair is a 400.
    """
    if latitude is None and longitude is None:
        return None
    if latitude is None or longitude is None:
        raise HTTPException(
            status_code=400, detail="Send both latitude and longitude."
        )
    try:
        return build_geo_location(
            latitude,
            longitude,
            location_name=location_name,
            geofence_radius_meters=geofence_radius_meters,
        )
    except GeoLocationError as invalid_location:
        raise HTTPException(status_code=400, detail=str(invalid_location)) from invalid_location


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
    # Same rationale as the Hub prefetch above: the stylometric feature
    # extractor's corpora are ~20 MB, and paying for them on the first scored
    # reply stalls that request behind the download.
    ensure_nltk_corpora_cached()
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
    # Connected accounts (mailboxes, custom connectors) live in their own table,
    # keyed by user and personal avatar rather than by an identity provider's
    # namespace. Publish the repository process-wide so graph nodes and tools —
    # which cannot import this module — read the same table the routes write,
    # then copy any legacy store records across once (never overwriting).
    from src.anubis.utils.connected_accounts import (
        repository as connected_accounts_repository,
    )

    await connected_accounts_repository.ensure_connected_accounts_table(app.state.pool)
    connected_accounts_repository.set_repository(
        connected_accounts_repository.PostgresConnectedAccountRepository(app.state.pool)
    )
    await connected_accounts_repository.migrate_store_connected_accounts_to_table(
        app.state.pool
    )
    # Popup sign-ins (Google, GitHub, X, Plaid, live browser) keep their
    # in-flight state server-side, keyed by the nonce the signed state carries.
    from src.anubis.utils.connected_accounts import pending_logins as pending_logins_module

    await pending_logins_module.ensure_pending_logins_table(app.state.pool)
    pending_logins_module.set_pending_login_repository(
        pending_logins_module.PostgresPendingLoginRepository(app.state.pool)
    )
    # Website audits keep a content hash per page so "what changed" can be answered.
    from src.anubis.utils.connected_accounts.website_tools import (
        ensure_website_crawls_table,
    )

    await ensure_website_crawls_table(app.state.pool)
    # Graph-side analytics tools reach the application tables through the
    # same pool; the graph cannot import this module.
    runtime_handles.set_postgres_pool(app.state.pool)
    # Signed-in browser sessions (sites with no OAuth) are revisited on a
    # schedule so the owner does not sign in again days later.
    from src.anubis.utils.connected_accounts import browser_sessions as browser_sessions_module

    app.state.browser_session_keepalive = asyncio.create_task(
        browser_sessions_module.keepalive_forever(app.state.context, None)
    )
    try:
        from src.anubis.utils import analytics as analytics_package

        await analytics_package.ensure_analytics_tables(app.state.pool)
        analytics_package.publish_analytics_repositories(app.state.pool)
    except ImportError:
        logger.info("Analytics package not installed; reports and schedules are off.")
    except Exception as analytics_boot_error:  # noqa: BLE001 - startup must not fail
        logger.error("Analytics tables could not be prepared: %s", analytics_boot_error)
    # Opt-in usage analytics: consent, the action log, and described page
    # captures live in their own tables; old rows are forgotten on a schedule.
    app.state.usage_analytics_purge_task = None
    try:
        from src.anubis.utils import usage_analytics as usage_analytics_package
        from src.anubis.utils.usage_analytics import retention as usage_analytics_retention

        await usage_analytics_package.ensure_usage_analytics_tables(app.state.pool)
        usage_analytics_repository = (
            usage_analytics_package.publish_usage_analytics_repository(app.state.pool)
        )
        app.state.usage_analytics_purge_task = asyncio.create_task(
            usage_analytics_retention.purge_forever(
                usage_analytics_repository,
                int(GlobalContext().usage_analytics_retention_days or 0),
            )
        )
    except Exception as usage_analytics_boot_error:  # noqa: BLE001 - startup must not fail
        logger.error(
            "Usage analytics tables could not be prepared: %s", usage_analytics_boot_error
        )
    # Generated avatar media (emotion stills, idle loops, lip-sync clips, voice
    # clips) and the durable media jobs live in their own BYTEA tables.
    from src.anubis.utils import media_assets as media_assets_package

    await media_assets_package.ensure_media_asset_tables(app.state.pool)
    media_assets_package.set_media_asset_repository(
        media_assets_package.PostgresMediaAssetRepository(app.state.pool)
    )
    # A professional voice clone trains for hours; its state lives in the
    # avatar_voice table and is refreshed on a schedule that survives restarts.
    app.state.voice_training_poller = asyncio.create_task(
        _poll_training_voice_clones(app.state.context)
    )
    # The agent inbox: items, learned preferences, and poll cursors in their own
    # tables; the triage graph runs in-process on the shared checkpointer.
    from src.anubis.utils import inbox as inbox_package

    await inbox_package.ensure_inbox_tables(app.state.pool)
    inbox_package.set_inbox_repository(
        inbox_package.PostgresInboxRepository(app.state.pool)
    )

    # Geo-located avatars. Two throttles keep a moving phone from flooding the
    # visit records and from notifying the same visitor about the same avatar on
    # every position update. Both live in this process, so a deployment with
    # several workers dedupes per worker rather than per fleet.
    app.state.geo_checkin_throttle = CheckinThrottle(
        app.state.context.geo_checkin_min_interval_seconds
    )
    app.state.geo_notify_throttle = CheckinThrottle(
        app.state.context.geo_notify_cooldown_seconds
    )
    # Resolve from STRIPE_BILLING_CONFIG_JSON, else the file written by the compose
    # stripe-provision service (STRIPE_BILLING_CONFIG_FILE) — so a reprovision never
    # requires pasting JSON into the env or a manual edit. This is only the INITIAL
    # load: current_stripe_billing_config re-reads the file when a reprovision
    # changes it, so edited prices take effect without restarting the API.
    initialize_stripe_billing_config(app.state)
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
        # Registry for background deep-research jobs (see src/api/research_jobs.py).
        app.state.research_jobs = {}
        # The in-chat update_avatar_identity_with_media tool starts media
        # batches through this published starter (see runtime_handles).
        runtime_handles.set_identity_media_job_starter(
            start_identity_media_job_from_chat
        )
        checkpointer = AsyncPostgresSaver(app.state.pool)
        await checkpointer.setup()
        app.state.checkpointer = checkpointer
        # Publish the shared checkpointer so the deep agent (rebuilt each turn inside
        # the ``think`` node) can reuse it and make HITL ``interrupt``s durable.
        runtime_handles.set_deep_agent_checkpointer(checkpointer)
        from src.anubis.utils.inbox import poller as inbox_poller

        inbox_poller.set_inbox_runtime(checkpointer, store)
        app.state.inbox_poller = asyncio.create_task(
            inbox_poller.poll_forever(app.state.context)
        )
        app.state.graph = message_workflow.compile(
            store=store, checkpointer=checkpointer
        )
        # Scheduled reports (weekly sprint digest, monthly spend digest, and
        # any report the owner schedules in conversation) run through the
        # graph on a fresh hidden thread and land in the inbox as notify items.
        try:
            from src.anubis.utils.analytics import schedules as report_schedules

            app.state.report_scheduler = asyncio.create_task(
                report_schedules.run_due_schedules_forever(
                    app.state.context, run_question=_run_scheduled_report_question
                )
            )
        except ImportError:
            app.state.report_scheduler = None
        except Exception as scheduler_error:  # noqa: BLE001 - startup must not fail
            logger.error("Report scheduler could not start: %s", scheduler_error)
            app.state.report_scheduler = None
        # Keep the unmodified-inference-model style baseline in step with MODEL:
        # a boot whose MODEL differs from the model recorded with the committed
        # baseline retrains it ONCE (coordinated through the shared store) or adopts
        # a sibling container's published result. Never blocks or fails startup.
        try:
            from src.anubis.utils.dataset.baseline_provenance import (
                ensure_baseline_matches_model,
            )

            app.state.baseline_retrain_task = await ensure_baseline_matches_model(
                app.state
            )
        except Exception as baseline_check_error:  # noqa: BLE001 - startup must not fail
            logger.error("Baseline provenance check failed: %s", baseline_check_error)
            app.state.baseline_retrain_task = None
        # Continuous learning: the background sweep that finalizes conversation
        # sentiment history, folds ratings into learned preferences, and infers
        # preferences once an account has gone idle (see learning/bulk_learning.py).
        app.state.learning_sweeper_task = None
        if str(app.state.context.learning_sweep_enabled or "").upper() == "TRUE":
            from src.anubis.utils.learning.bulk_learning import run_learning_sweeper

            app.state.learning_sweeper_task = asyncio.create_task(
                run_learning_sweeper(app)
            )
        logger.info("Application startup: lifecycle complete")
        yield
    finally:
        sweeper_task = getattr(app.state, "learning_sweeper_task", None)
        if sweeper_task is not None:
            sweeper_task.cancel()
            try:
                await sweeper_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - shutdown
                pass
        purge_task = getattr(app.state, "usage_analytics_purge_task", None)
        if purge_task is not None:
            purge_task.cancel()
        try:
            from src.anubis.utils.connected_accounts.browser_sessions import (
                shutdown_browser_sessions,
            )

            await shutdown_browser_sessions()
        except Exception:
            logger.debug("Browser sessions did not shut down cleanly", exc_info=True)
        await pool.close()


app = FastAPI(
    title="Neural Nexus API",
    description="LangGraph-based API",
    version="1.0.0",
    lifespan=lifespan,
)
# Replies being generated right now, keyed by request id, so
# ``POST /message/{assistant_id}/stop`` can end one (see ``message_stops``).
app.state.active_message_turns = ActiveMessageTurnRegistry()


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
                # Called directly rather than through dependency injection, so
                # BOTH credentials have to be handed over explicitly: an omitted
                # argument would leave the parameter holding FastAPI's Depends
                # sentinel instead of None.
                await get_current_user(
                    request=request,
                    api_key=request.headers.get("API-KEY"),
                    bearer_credentials=bearer_credentials_from_request(request),
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


def _checkout_line_items_for_tier(billing_config, tier: SubscriptionTier) -> list[dict]:
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
    billing_config = current_stripe_billing_config(request.app.state)
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
            subscription = stripe_client.Subscription.retrieve(
                subscription_id
            ).to_dict()
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
                        "end_date": current_period_end or current_phase.get("end_date"),
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

    billing_config = current_stripe_billing_config(request.app.state)
    email = current_user.get("email")
    if billing_config is None:
        # Billing objects not provisioned yet — fall back to the legacy payment link.
        redirect_url = (
            f"{app.state.context.stripe_payment_url}?locked_prefilled_email={email}"
        )
        return {
            "action": "start_checkout",
            "url": redirect_url,
            "message": "Follow this link to subscribe.",
        }

    stripe_client = request.app.state.stripe
    status = await check_subscription_status(request=request, current_user=current_user)
    current_status = status.get("status")
    current_tier = tier_from_value(status.get("tier"))
    cancel_at_period_end = False
    has_pending_downgrade_schedule = False
    subscription: dict | None = None
    subscription_id = status.get("subscription_id")
    if subscription_id and current_status in _LIVE_SUBSCRIPTION_STATUSES:
        try:
            subscription = stripe_client.Subscription.retrieve(
                subscription_id
            ).to_dict()
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
    if customer_id:
        # Without this, a returning customer sees an EMPTY card form even with a
        # card already on file: subscription-mode Checkout saves cards with
        # allow_redisplay="limited", and Checkout only prefills cards marked
        # "always". Widening the filter shows the cards this customer already
        # has, and payment_method_save lets them consent to reuse for next time
        # — which is also what stops a duplicate PaymentMethod being created for
        # a card the customer already gave us.
        checkout_kwargs["saved_payment_method_options"] = {
            "payment_method_save": "enabled",
            "allow_redisplay_filters": ["always", "limited", "unspecified"],
        }
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
    billing_config = current_stripe_billing_config(request.app.state)
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


def _auth0_user_id_for_customer(
    stripe_client, customer_id: Optional[str]
) -> Optional[str]:
    """Look up the Auth0 user id stored on a Stripe customer's metadata."""
    if not customer_id:
        return None
    try:
        customer = stripe_client.Customer.retrieve(customer_id).to_dict()
        return customer.get("metadata", {}).get("auth0_user_id") or None
    except Exception as lookup_error:
        logger.error(
            "Could not retrieve Stripe customer %s: %s", customer_id, lookup_error
        )
        return None


async def _read_user_app_metadata(request: Request, auth0_user_id: str | None) -> dict:
    """Return one Auth0 user's ``app_metadata``, or an empty mapping.

    Webhook handlers receive only a Stripe object, so any rule that depends on
    what the account already holds (the usage window it was counting against,
    a retained paid period) has to read it back from Auth0. Best-effort: a
    lookup failure yields ``{}`` and the caller falls back to its default,
    because a webhook must never fail on a read.
    """
    if not auth0_user_id:
        return {}
    try:
        auth0_user = await get_user(auth0_user_id, request)
    except Exception as lookup_error:  # noqa: BLE001 - best-effort webhook read
        logger.error(
            "Could not read app_metadata for %s while handling a Stripe event: %s",
            auth0_user_id,
            lookup_error,
        )
        return {}
    return auth0_user.get("app_metadata") or {}


async def _build_canceled_tier_context_fields(
    request: Request,
    auth0_user_id: str | None,
    canceled_tier: SubscriptionTier,
    period_end_epoch: int | None,
) -> dict:
    """Return the app_metadata patch recording a mid-period paid cancellation.

    Produces two keys together, because they are two halves of one rule:

    * ``usage_period_anchor`` moves to this instant, which is what makes
      "a refunded subscription immediately switches to the free-tier allotment"
      literally true — usage to date stops counting against the new window.
    * ``canceled_tier_context`` remembers the tier, the period end, and the
      window that was just abandoned, so a resubscribe before ``period_end``
      can restore it (see ``plan_resubscribe_usage_window``) instead of handing
      out a second allotment for a period the customer was refunded for.

    A canceled FREE subscription records no context — there is no paid period
    to retain, and the allotment floor never applies to the free tier anyway.
    """
    fresh_anchor = datetime.now(UTC).isoformat()
    if canceled_tier == SubscriptionTier.FREE or not period_end_epoch:
        return {"usage_period_anchor": fresh_anchor}
    app_metadata = await _read_user_app_metadata(request, auth0_user_id)
    return {
        "usage_period_anchor": fresh_anchor,
        "canceled_tier_context": {
            "tier": canceled_tier.value,
            "period_end": int(period_end_epoch),
            "previous_usage_period_anchor": app_metadata.get("usage_period_anchor"),
        },
    }


async def _write_resubscribe_usage_period_anchor(
    request: Request, auth0_user_id: str | None, subscribed_tier: SubscriptionTier
) -> None:
    """Set the usage window a completed checkout starts, honoring a retained period.

    Ordinary checkouts restart the window at this instant. A checkout that
    resubscribes to the same or a lower tier inside a period the customer was
    refunded for instead RESTORES the window they were counting against, so the
    usage they already accrued carries over and (via the allotment floor in
    ``resolve_effective_monthly_allotment``) the canceled tier's limits govern
    the rest of that period.

    The ``canceled_tier_context`` is cleared whenever it is not being honored —
    an upgrade past the canceled tier, or a window that has since expired — so a
    stale record can never resurrect a paid allotment later.
    """
    app_metadata = await _read_user_app_metadata(request, auth0_user_id)
    canceled_tier_context = resolve_canceled_tier_context(
        {"app_metadata": app_metadata}
    )
    restored_anchor = plan_resubscribe_usage_window(
        subscribed_tier, canceled_tier_context
    )
    if restored_anchor is not None:
        await update_user_app_metadata_fields(
            request, auth0_user_id, {"usage_period_anchor": restored_anchor.isoformat()}
        )
        return
    await update_user_app_metadata_fields(
        request,
        auth0_user_id,
        {
            "usage_period_anchor": datetime.now(UTC).isoformat(),
            "canceled_tier_context": None,
        },
    )


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
        # Checkout normally starts a fresh local usage window (a free→paid
        # upgrade, or the free $0 subscription's first period). Resubscribing
        # inside a period the customer was refunded for is the exception:
        # ``plan_resubscribe_usage_window`` restores the pre-cancellation window
        # for a same-or-lower tier so accrued usage carries over, and returns
        # None (anchor at this instant, usage cleared) for an upgrade or an
        # expired retention window.
        await _write_resubscribe_usage_period_anchor(
            request, auth0_user_id, tier_from_value(tier)
        )

    elif event_type == "customer.updated":
        # The customer portal cannot evict this API's api-key cache directly, so
        # it mirrors the pay-per-use switch into Stripe customer metadata and
        # lets this webhook carry it the rest of the way. Writing the flag into
        # Auth0 here evicts the cache (update_user_app_metadata_fields), which
        # is what makes a portal-side toggle take effect on the very next
        # request instead of after the five-minute TTL. Stripe metadata values
        # are always strings, so the boolean is parsed rather than trusted.
        raw_flag = (data_object.get("metadata") or {}).get("pay_per_use_enabled")
        if raw_flag is not None:
            customer_id = data_object.get("id")
            auth0_user_id = (data_object.get("metadata") or {}).get(
                "auth0_user_id"
            ) or _auth0_user_id_for_customer(stripe_client, customer_id)
            await update_user_app_metadata_fields(
                request,
                auth0_user_id,
                {
                    "pay_per_use_enabled": str(raw_flag).strip().lower()
                    in ("true", "1", "yes", "enabled")
                },
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
            # Read the tier the ended subscription represented BEFORE pinning
            # the account to free — a refund-driven immediate cancellation has
            # to remember what was paid for in order to honor it on a
            # resubscribe inside the same period.
            canceled_tier = tier_from_value(
                _tier_from_subscription(stripe_client, data_object)
            )
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
            # subscription (the billing vehicle) is gone. The same patch drops
            # the account to the free-tier allotment AT ONCE by restarting the
            # usage window, and records what was paid for so a resubscribe
            # inside the paid-for period can retain it — see
            # _build_canceled_tier_context_fields.
            await update_user_app_metadata_fields(
                request,
                auth0_user_id,
                {
                    "pay_per_use_enabled": False,
                    **(
                        await _build_canceled_tier_context_fields(
                            request, auth0_user_id, canceled_tier, current_period_end
                        )
                    ),
                },
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
    request: Request,
    current_user: dict = Depends(get_current_user_or_anonymous_user_id),
):
    """Return subscription status plus per-meter allotment, usage, and remaining.

    The single endpoint a customer portal polls: subscription identity/status,
    the pay-per-use flag, the current usage period bounds, and — for every meter
    the tier grants — the period allotment, usage to date, remaining budget, and
    the overage rate that applies when pay-per-use is enabled. ``used_to_date``
    is the reconciled, Stripe-authoritative figure
    (``resolve_period_usage_to_date``), the same number allotment enforcement
    judges against.

    ANONYMOUS CALLERS ARE ACCEPTED (no API key): an anonymous visitor is
    identified by the hash of their network address, always resolves to the free
    tier, and reports the usage accumulating on their per-hashed-ip Stripe
    free-tier customer — the same customer, window, and aggregation the customer
    portal reads, so an anonymous visitor can be shown exactly how much of the
    free allotment they have spent. The response flags them with ``anonymous``
    so a client can render their view read-only (they cannot subscribe, hold a
    trial, or enable pay-per-use until they create an account).
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
        stripe_customer_id=resolve_stripe_customer_id(current_user),
    )

    # Trial-aware per-meter view: within a free-trial window the trial tier's
    # allotment is a floor over the current tier's (resolve_effective_monthly_
    # allotment), and any meter the trial tier grants but the current tier does
    # not stays visible until trial_end. The meter order is the current tier's
    # definition order, then any trial-only meters, kept deterministic via an
    # insertion-ordered dict.
    trial_context = resolve_trial_context(current_user)
    canceled_tier_context = resolve_canceled_tier_context(current_user)
    ordered_meters: dict = dict.fromkeys(TIER_DEFINITIONS[tier].meter_allotments.keys())
    if trial_context is not None:
        for trial_meter in TIER_DEFINITIONS[trial_context.trial_tier].meter_allotments:
            ordered_meters.setdefault(trial_meter, None)
    # A retained paid period can likewise grant meters the current tier lacks
    # (refunded premium, resubscribed pro keeps premium's adapter meters until
    # the paid-for period closes), so those stay visible too.
    if canceled_tier_allotment_floor_applies(tier, canceled_tier_context):
        assert canceled_tier_context is not None
        for retained_meter in TIER_DEFINITIONS[
            canceled_tier_context.canceled_tier
        ].meter_allotments:
            ordered_meters.setdefault(retained_meter, None)

    granted_allotments: dict = {}
    for meter in ordered_meters:
        allotment = resolve_effective_monthly_allotment(
            tier,
            meter,
            trial_context,
            canceled_tier_context=canceled_tier_context,
        )
        if allotment is not None:
            granted_allotments[meter] = allotment

    # Stripe's aggregation per granted meter, read CONCURRENTLY: a premium tier
    # grants four meters and a portal polls this endpoint, so four sequential
    # Stripe round trips would be four times the wait for one screen. Cached
    # readings return without any call at all.
    stripe_usage_by_meter = dict(
        zip(
            granted_allotments,
            await asyncio.gather(
                *(
                    fetch_stripe_period_usage(
                        getattr(request.app.state, "stripe", None),
                        current_stripe_billing_config(request.app.state),
                        meter,
                        resolve_stripe_customer_id(current_user),
                        period_start,
                    )
                    for meter in granted_allotments
                )
            ),
        )
    )

    meters: dict = {}
    for meter, allotment in granted_allotments.items():
        # Stripe reconciled against the local per-meter sum read above, so the
        # portal, this response, and the 402 gate all report one figure.
        # ``over_allotment`` is stated explicitly because a customer billing
        # pay-per-use overage needs to see how far past the allotment they have
        # run, not a bar pinned at zero remaining.
        used_to_date = reconcile_period_usage(
            int(usage_by_meter.get(meter.value, 0)), stripe_usage_by_meter.get(meter)
        )
        meters[meter.value] = {
            "monthly_allotment": allotment.monthly_allotment,
            "used_to_date": used_to_date,
            "remaining": max(0, allotment.monthly_allotment - used_to_date),
            "over_allotment": max(0, used_to_date - allotment.monthly_allotment),
            "overage_price_per_million": allotment.overage_price_per_million,
            "overage_price_per_unit_usd": allotment.overage_price_per_unit_usd,
        }

    # A requester that skips enforcement is never refused at the allotment, so a
    # client rendering this payload has to be told: the same bypass flags the SSE
    # usage frames and the upload response already carry are spread here too, and
    # a portal reading unrestricted_metered_account knows not to draw an exhausted
    # allotment as a wall. One vocabulary across all three usage payloads.
    metering_bypass = resolve_metering_bypass(current_user)
    return {
        "status": status.get("status"),
        "tier": tier.value,
        "subscription_id": status.get("subscription_id"),
        "customer_id": status.get("customer_id"),
        "email": status.get("email"),
        "anonymous": is_anonymous_user(current_user),
        "pay_per_use_enabled": resolve_pay_per_use_enabled(current_user),
        "cancel_at_period_end": bool(
            cached_subscription_status.get("cancel_at_period_end")
        ),
        "usage_period_start": period_start.isoformat(),
        "usage_period_end": period_end.isoformat(),
        "meters": meters,
        **metering_bypass.usage_response_fields(),
    }


# The "at most one personal avatar per user" enforcement lives in
# src/anubis/utils/personal_avatar.py so that /create_avatar, /modify_avatar, and
# post-verification auto-provisioning all enforce the invariant identically.
from src.anubis.utils.personal_avatar import (  # noqa: E402
    demote_other_personal_avatars as _demote_other_personal_avatars,
)


@app.get("/personal_avatar")
async def get_personal_avatar(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Return the caller's one personal avatar plus the capabilities exclusive to it.

    Every signed-up user always has exactly one personal avatar, so a missing
    avatar is a provisioning gap this route closes silently rather than an error
    reported back to the owner: resolution provisions one when none is found.

    Each capability reports a live status where the capability has landed and
    ``not_configured`` where the connection has not been made yet, so the owner
    can see the full set of what the personal avatar is for.
    """
    from src.anubis.utils.personal_avatar import (
        PERSONAL_AVATAR_CAPABILITIES,
        resolve_personal_avatar,
    )

    token = current_user["API_KEY"]
    user_id = current_user["identities"][0]["user_id"]
    client = get_client(headers={"API-KEY": f"{token}"})

    try:
        personal_avatar = await resolve_personal_avatar(
            client, request, current_user, token
        )
    except Exception as resolution_error:
        raise HTTPException(
            status_code=500,
            detail=f"Error resolving the personal avatar: {resolution_error}",
        )

    if personal_avatar is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "The personal avatar could not be provisioned. Retry shortly; "
                "provisioning is retried on every request until it succeeds."
            ),
        )

    capability_statuses = await _resolve_personal_avatar_capability_statuses(
        client, user_id, personal_avatar
    )

    return JSONResponse(
        content={
            "personal_avatar": personal_avatar,
            "capabilities": [
                {
                    "name": capability.name,
                    "summary": capability.summary,
                    "status": capability_statuses.get(
                        capability.status_key, "not_configured"
                    ),
                }
                for capability in PERSONAL_AVATAR_CAPABILITIES
            ],
        },
        status_code=200,
    )


async def _search_namespace_records(
    client: Any, namespace: tuple[str, ...]
) -> list[dict[str, Any]]:
    """List every record in one per-user store namespace, best effort.

    Used by two namespace families that share the same shape — one record per
    thing, keyed by that thing's identifier. The Model Context Protocol
    namespaces hold one record per connected machine keyed by ``device_id``; the
    connected-account namespace holds one record per connected external account
    keyed by ``"{provider}:{address}"``. This wraps the SDK ``StoreClient``
    search so endpoints (which authenticate as the user and therefore use the
    HTTP store client rather than the in-process ``BaseStore``) can read a whole
    record set in one call.

    Returns an empty list when the namespace is empty or unreachable — every
    caller treats "no devices" and "cannot tell" the same way, and a store hiccup
    must never fail an endpoint whose real job is something else.
    """
    expected_namespace = list(namespace)
    try:
        response = await client.store.search_items(expected_namespace, limit=100)
    except Exception:
        logger.debug("Could not search store namespace %s", namespace, exc_info=True)
        return []
    items = (response or {}).get("items") or []
    records: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_namespace = item.get("namespace")
        if item_namespace is not None and list(item_namespace) != expected_namespace:
            continue
        value = item.get("value") or {}
        if not isinstance(value, dict):
            continue
        owner = value.get("owner_user_id")
        if owner and expected_namespace and owner != expected_namespace[0]:
            continue
        records.append(value)
    return records


async def _resolve_personal_avatar_capability_statuses(
    client: Any, user_id: str, personal_avatar: dict
) -> dict[str, Any]:
    """Collect the live status of each personal-avatar capability.

    Capabilities that have not been built yet are simply absent from the returned
    mapping, and the caller renders those as ``not_configured``. Every lookup is
    best-effort: a capability whose backing store is unreachable must not fail the
    whole listing.
    """
    from src.anubis.utils.tools.data_analysis.backend import mcp_connection_namespace

    statuses: dict[str, Any] = {}

    connections = await _search_namespace_records(
        client, mcp_connection_namespace(user_id)
    )
    connected_data_servers = [
        {
            "server_name": connection.get("server_name"),
            "device_label": connection.get("device_label"),
            "platform": connection.get("platform"),
            "bound_to_this_avatar": (
                connection.get("assistant_id") == personal_avatar.get("assistant_id")
            ),
            "connected_at": connection.get("connected_at"),
        }
        for connection in connections
        if connection.get("status") == "connected"
    ]
    if connected_data_servers:
        statuses["connected_data_servers"] = connected_data_servers

    # Connected external accounts fill two capabilities from one namespace,
    # split by the provider's kind. The split is deliberate and is a security
    # boundary rather than presentation: `social_accounts` is the capability an
    # identity-verification gate reads, and a mailbox must never be able to
    # discharge "the account behind this likeness is verified". See
    # `connected_accounts/providers.py`.
    from src.anubis.utils.connected_accounts import (
        STATUS_CONNECTED,
        public_account_view,
    )

    accounts = await _connected_account_records(client, user_id)
    connected_mailboxes = [
        public_account_view(account)
        for account in accounts
        if account.get("kind") == "mailbox"
    ]
    if connected_mailboxes:
        statuses["connected_mailboxes"] = connected_mailboxes

    connected_social_accounts = [
        public_account_view(account)
        for account in accounts
        if account.get("kind") == "social" and account.get("status") == STATUS_CONNECTED
    ]
    if connected_social_accounts:
        statuses["connected_social_accounts"] = connected_social_accounts

    # The personal avatar's adapter is trained from the owner's messages across
    # every avatar; no connection step gates that, so it is always active.
    statuses["adapter_training"] = "active"
    return statuses


async def _resolve_personal_avatar_for_connection(
    client: Any, request: Any, current_user: dict, token: str
) -> dict:
    """Return the caller's personal avatar or fail the request.

    Every connected-account route is exclusive to the personal avatar: these
    credentials reach the owner's private mail, and no shared or secondary
    avatar may be bound to them. Resolution is self-healing, so a user who has
    somehow lost their personal avatar gets one provisioned here rather than an
    error telling them to create one first.
    """
    from src.anubis.utils.personal_avatar import resolve_personal_avatar

    try:
        personal_avatar = await resolve_personal_avatar(
            client, request, current_user, token
        )
    except Exception as resolution_error:
        raise HTTPException(
            status_code=500,
            detail=f"Error resolving the personal avatar: {resolution_error}",
        )
    if personal_avatar is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "The personal avatar could not be provisioned. Retry shortly; "
                "provisioning is retried on every request until it succeeds."
            ),
        )
    return personal_avatar


async def _connected_account_records(client: Any, user_id: str) -> list[dict[str, Any]]:
    """Every connected-account record for a user, from the table or the legacy store.

    The repository is published by the lifespan; when it is absent (the dev
    server without the lifespan, or unit tests) the legacy store namespace is
    read through the SDK client the route already holds.
    """
    from src.anubis.utils.connected_accounts import connected_account_namespace
    from src.anubis.utils.connected_accounts.repository import get_repository

    repository = get_repository()
    if repository is not None:
        try:
            return await repository.list_for_user(user_id)
        except Exception:
            logger.debug(
                "Could not list connected accounts for %s", user_id, exc_info=True
            )
            return []
    return await _search_namespace_records(client, connected_account_namespace(user_id))


async def _get_connected_account_record(
    client: Any, user_id: str, key: str
) -> dict[str, Any] | None:
    from src.anubis.utils.connected_accounts import connected_account_namespace
    from src.anubis.utils.connected_accounts.repository import get_repository

    repository = get_repository()
    if repository is not None:
        return await repository.get(user_id, key)
    item = await client.store.get_item(
        list(connected_account_namespace(user_id)), key=key
    )
    value = (item or {}).get("value") if isinstance(item, dict) else None
    return dict(value) if value else None


async def _put_connected_account_record(
    client: Any, user_id: str, record: dict[str, Any]
) -> None:
    from src.anubis.utils.connected_accounts import connected_account_namespace
    from src.anubis.utils.connected_accounts.repository import get_repository

    repository = get_repository()
    if repository is not None:
        await repository.upsert(user_id, record)
        return
    await client.store.put_item(
        list(connected_account_namespace(user_id)),
        key=record["account_key"],
        value=record,
    )


async def _delete_connected_account_record(client: Any, user_id: str, key: str) -> None:
    from src.anubis.utils.connected_accounts import connected_account_namespace
    from src.anubis.utils.connected_accounts.repository import get_repository

    repository = get_repository()
    if repository is not None:
        await repository.delete(user_id, key)
        return
    await client.store.delete_item(list(connected_account_namespace(user_id)), key=key)


async def _connect_account_from_fields(
    request: Request,
    current_user: dict,
    provider_name: str,
    fields: dict[str, Any],
) -> JSONResponse:
    """Shared body of ``/connect_account`` and its ``/connect_mailbox`` alias.

    The provider's credential mechanism chooses the handler that PROVES the
    connection (a real mail login, a real tool listing) before anything is
    stored; the plaintext credential is encrypted by the handler and never
    persisted or logged in the clear. The cap is enforced here, once, for every
    mechanism, and reconnecting an account the owner already has refreshes that
    record rather than counting against the cap — otherwise changing a password
    would eventually lock the owner out of their own mailbox.
    """
    from src.anubis.utils.connected_accounts import get_provider, public_account_view
    from src.anubis.utils.connected_accounts.connect_handlers import (
        ConnectNeedsLogin,
        ConnectRefused,
        ConnectRequest,
        connect_account,
    )

    provider = get_provider(provider_name)
    if provider is None:
        raise HTTPException(
            status_code=400, detail=f"Unknown provider {provider_name!r}."
        )

    token = current_user["API_KEY"]
    user_id = current_user["identities"][0]["user_id"]
    client = get_client(headers={"API-KEY": f"{token}"})
    personal_avatar = await _resolve_personal_avatar_for_connection(
        client, request, current_user, token
    )

    existing_records = await _connected_account_records(client, user_id)
    try:
        record = await connect_account(
            ConnectRequest(
                provider=provider,
                fields=fields,
                assistant_id=personal_avatar.get("assistant_id"),
                context=app.state.context,
                existing_records=existing_records,
            )
        )
    except ConnectNeedsLogin as needs_login:
        # Not a failure: the provider signs in through a popup. The card opens
        # the login endpoint with the request carried here.
        return JSONResponse(content=needs_login.as_response(), status_code=200)
    except ConnectRefused as refused:
        raise HTTPException(status_code=refused.status_code, detail=refused.detail)

    record["user_id"] = user_id
    _enforce_connection_caps(existing_records, provider, record)
    await _put_connected_account_record(client, user_id, record)
    _after_record_stored(provider, record)

    return JSONResponse(
        content={"connected": True, "account": public_account_view(record)},
        status_code=200,
    )


def _enforce_connection_caps(
    existing_records: list[dict[str, Any]], provider: Any, record: dict[str, Any]
) -> None:
    """Refuse a NEW account beyond the per-user caps; reconnects always pass.

    Shared by the form route, the OAuth callback, the Plaid exchange, and the
    live-browser finish so every way in counts the same accounts.
    """
    from src.anubis.utils.connected_accounts.providers import KIND_MCP_SERVER

    key = record["account_key"]
    if any(existing.get("account_key") == key for existing in existing_records):
        return
    if provider.kind == KIND_MCP_SERVER:
        maximum = int(
            getattr(app.state.context, "max_custom_mcp_connectors_per_user", 10) or 10
        )
        already = sum(
            1 for existing in existing_records if existing.get("kind") == KIND_MCP_SERVER
        )
        if already >= maximum:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"A maximum of {maximum} custom connectors may be connected. "
                    "Disconnect one before connecting another."
                ),
            )
    maximum = int(app.state.context.max_connected_accounts_per_user)
    if len(existing_records) >= maximum:
        raise HTTPException(
            status_code=409,
            detail=(
                f"A maximum of {maximum} accounts may be connected. "
                "Disconnect one before connecting another."
            ),
        )


def _after_record_stored(provider: Any, record: dict[str, Any]) -> None:
    """Housekeeping once a record is written (cache resets, schedule seeds)."""
    from src.anubis.utils.connected_accounts.providers import KIND_MCP_SERVER

    if provider.kind == KIND_MCP_SERVER:
        from src.anubis.utils.connected_accounts.mcp_server_tools import (
            forget_cached_tools,
        )

        forget_cached_tools((record.get("transport") or {}).get("server_url") or "")
    # The first business-category connection seeds the weekly sprint digest
    # and the monthly spend digest; both are deliverable through the inbox.
    if provider.category in ("finance", "development", "vendor"):
        try:
            from src.anubis.utils.analytics.schedules import (
                get_schedule_repository,
                seed_default_schedules,
            )

            repository = get_schedule_repository()
            if repository is not None:
                schedule_background(
                    seed_default_schedules(
                        repository,
                        user_id=str(record.get("user_id") or ""),
                        assistant_id=str(record.get("assistant_id") or ""),
                    )
                )
        except ImportError:
            pass
        except Exception:
            logger.debug("Could not seed default report schedules", exc_info=True)


async def _connected_account_records_without_session(user_id: str) -> list[dict[str, Any]]:
    """Every record for a user when no SDK session exists (popup callbacks)."""
    from src.anubis.utils.connected_accounts import read_connected_accounts

    return await read_connected_accounts(getattr(app.state, "store", None), user_id)


async def _store_connected_record_without_session(
    user_id: str, record: dict[str, Any]
) -> None:
    """Persist a record when the request carries no owner session (popup callbacks)."""
    from src.anubis.utils.connected_accounts import save_connected_account

    await save_connected_account(getattr(app.state, "store", None), user_id, record)


def _popup_result_response(result: dict[str, Any]) -> HTMLResponse:
    """The HTML a popup ends on; posts a non-secret result to the opener."""
    from src.anubis.utils.connected_accounts.oauth_flow import (
        allowed_popup_origins,
        render_popup_result_html,
    )

    return HTMLResponse(
        render_popup_result_html(result, allowed_popup_origins(app.state.context))
    )


def _login_token_from_request(request: Request) -> str:
    """Return the signed login token a popup presents (header or query)."""
    token = request.headers.get("X-Login-Token") or request.query_params.get("t") or ""
    if not token:
        raise HTTPException(status_code=401, detail="A login token is required.")
    return str(token)


def _verify_login_token(token: str, *, mode: str) -> dict:
    from src.anubis.utils.connected_accounts.oauth_state import (
        OAuthStateError,
        state_secret,
        verify_state,
    )

    try:
        payload = verify_state(token, state_secret(app.state.context))
    except OAuthStateError as state_error:
        raise HTTPException(status_code=401, detail=str(state_error))
    if payload.get("mode") != mode:
        raise HTTPException(status_code=401, detail="The login token is for another flow.")
    return payload


@app.post("/connect_account/plaid/link_token")
async def connect_account_plaid_link_token(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Begin a bank connection through Plaid Link.

    Returns ``{link_url, nonce, expires_in}``; the card opens ``link_url`` (a
    path on this API) in the window it opened on the click. 503 when Plaid
    is not configured on this server.
    """
    from src.anubis.utils.connected_accounts import get_provider
    from src.anubis.utils.connected_accounts.oauth_state import OAuthStateError
    from src.anubis.utils.connected_accounts.plaid_link import (
        PlaidLinkError,
        start_plaid_link,
    )

    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    provider = get_provider(str((body or {}).get("provider") or "plaid"))
    if provider is None or provider.login_mode != "plaid_link":
        raise HTTPException(status_code=400, detail="This provider does not connect through Plaid.")
    token = current_user["API_KEY"]
    user_id = current_user["identities"][0]["user_id"]
    client = get_client(headers={"API-KEY": f"{token}"})
    personal_avatar = await _resolve_personal_avatar_for_connection(
        client, request, current_user, token
    )
    try:
        started = await start_plaid_link(
            app.state.context,
            user_id=user_id,
            assistant_id=str(personal_avatar.get("assistant_id")),
            provider=provider,
        )
    except PlaidLinkError as plaid_error:
        # Financial accounts connect ONLY through Plaid Link (the bank's own
        # secure login inside Plaid, with multiple institutions and accounts).
        # There is no browser fallback for a bank; a 503 says Plaid must be
        # configured on the server.
        raise HTTPException(status_code=plaid_error.status_code, detail=plaid_error.detail)
    except OAuthStateError as state_error:
        raise HTTPException(status_code=503, detail=str(state_error))
    return JSONResponse(content=started, status_code=200)


@app.get("/connect_account/plaid/link")
async def connect_account_plaid_link_page(request: Request, t: str = ""):
    """The popup page that runs Plaid Link (the login token is the auth)."""
    from src.anubis.utils.connected_accounts.oauth_flow import allowed_popup_origins
    from src.anubis.utils.connected_accounts.pending_logins import (
        get_pending_login_repository,
    )
    from src.anubis.utils.connected_accounts.plaid_link import (
        PLAID_EXCHANGE_PATH,
        render_link_page_html,
    )

    payload = _verify_login_token(t, mode="plaid")
    pending = await get_pending_login_repository().peek(str(payload.get("nonce") or ""))
    if pending is None or pending.get("status") != "started":
        return _popup_result_response(
            {"ok": False, "nonce": payload.get("nonce"), "provider": "plaid", "error": "This bank sign-in has expired. Start the connection again."}
        )
    link_token = str((pending.get("payload") or {}).get("link_token") or "")
    return HTMLResponse(
        render_link_page_html(
            link_token=link_token,
            nonce=str(payload.get("nonce")),
            login_token=t,
            allowed_origins=allowed_popup_origins(app.state.context),
            exchange_path=PLAID_EXCHANGE_PATH,
        )
    )


@app.post("/connect_account/plaid/exchange")
async def connect_account_plaid_exchange(request: Request):
    """Finish a bank connection: exchange the public token and store the item."""
    from src.anubis.utils.connected_accounts import get_provider, public_account_view
    from src.anubis.utils.connected_accounts.pending_logins import (
        get_pending_login_repository,
    )
    from src.anubis.utils.connected_accounts.plaid_link import (
        PlaidLinkError,
        exchange_public_token,
    )
    from src.anubis.utils.connected_accounts.tool_factories import tool_names_for

    payload = _verify_login_token(_login_token_from_request(request), mode="plaid")
    body = await request.json()
    public_token = str((body or {}).get("public_token") or "")
    nonce = str(payload.get("nonce") or "")
    if not public_token:
        raise HTTPException(status_code=400, detail="A public_token is required.")
    pending = await get_pending_login_repository().consume(nonce)
    if pending is None:
        return JSONResponse(
            content={"ok": False, "nonce": nonce, "provider": "plaid", "error": "This bank sign-in was already completed or has expired."},
            status_code=200,
        )
    user_id = str(pending["user_id"])
    existing_records = await _connected_account_records_without_session(user_id)
    try:
        record = await exchange_public_token(
            app.state.context,
            public_token=public_token,
            pending=pending,
            existing_records=existing_records,
        )
        provider = get_provider("plaid")
        _enforce_connection_caps(existing_records, provider, record)
        await _store_connected_record_without_session(user_id, record)
        _after_record_stored(provider, record)
    except PlaidLinkError as plaid_error:
        return JSONResponse(content={"ok": False, "nonce": nonce, "provider": "plaid", "error": plaid_error.detail}, status_code=200)
    except HTTPException as http_error:
        return JSONResponse(content={"ok": False, "nonce": nonce, "provider": "plaid", "error": str(http_error.detail)}, status_code=200)
    view = public_account_view(record)
    # The first transaction sync runs in the background so the card answers at once.
    try:
        from src.anubis.utils.analytics.finance import sync_transactions

        schedule_background(sync_transactions(app.state.context, app.state.pool, user_id, record))
    except ImportError:
        pass
    except Exception:
        logger.debug("Could not schedule the first Plaid sync", exc_info=True)
    return JSONResponse(
        content={
            "ok": True,
            "nonce": nonce,
            "provider": "plaid",
            "account_key": view.get("account_key"),
            "display_label": view.get("display_label"),
            "account_address": view.get("account_address"),
            "tool_count": len(tool_names_for(provider, record)),
        },
        status_code=200,
    )


@app.post("/connect_account/browser/start")
async def connect_account_browser_start(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Open a live browser at a site's sign-in page for the owner to sign in on.

    Body: ``provider`` (langsmith, openai, anthropic, a social
    site, custom_site, or website) plus ``site_url`` and ``name`` for a custom
    site. Returns ``{login_id, view_url, nonce, expires_in}``; the card opens
    ``view_url`` (a path on this API) in the window it opened on the click.
    """
    from src.anubis.utils.connected_accounts import get_provider
    from src.anubis.utils.connected_accounts.browser_login import start_login
    from src.anubis.utils.connected_accounts.browser_sessions import BrowserSessionError
    from src.anubis.utils.connected_accounts.oauth_state import OAuthStateError

    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="A JSON object body is required.")
    provider = get_provider(str(body.get("provider") or ""))
    if provider is None:
        raise HTTPException(status_code=400, detail="Unknown provider.")
    if provider.kind == "bank":
        raise HTTPException(
            status_code=400,
            detail="Financial accounts connect through Plaid, not a browser sign-in.",
        )
    site_url = str(body.get("site_url") or "").strip() or None
    # Any provider can be signed in to on a page the owner names (a bank's
    # own website when Plaid is not configured, a vendor without an OAuth
    # app); providers with their own login page need no address.
    if provider.login_mode != "browser_session" and not provider.login_url and not site_url:
        raise HTTPException(
            status_code=400,
            detail=f"{provider.display_name} needs the address of the page to sign in on (site_url).",
        )
    token = current_user["API_KEY"]
    user_id = current_user["identities"][0]["user_id"]
    client = get_client(headers={"API-KEY": f"{token}"})
    personal_avatar = await _resolve_personal_avatar_for_connection(
        client, request, current_user, token
    )
    reconnect_key = str(body.get("reconnect_account_key") or "").strip() or None
    if reconnect_key and reconnect_key.startswith("account:"):
        reconnect_key = reconnect_key[len("account:"):]
    try:
        started = await start_login(
            app.state.context,
            user_id=user_id,
            assistant_id=str(personal_avatar.get("assistant_id")),
            provider=provider,
            site_url=site_url or provider.login_url,
            name=body.get("name"),
            reconnect_account_key=reconnect_key,
        )
    except BrowserSessionError as session_error:
        raise HTTPException(status_code=session_error.status_code, detail=session_error.detail)
    except OAuthStateError as state_error:
        raise HTTPException(status_code=503, detail=str(state_error))
    return JSONResponse(content=started, status_code=200)


@app.get("/connect_account/browser/{login_id}")
async def connect_account_browser_view(request: Request, login_id: str, t: str = ""):
    """The popup page showing the live sign-in browser (login token is the auth)."""
    from src.anubis.utils.connected_accounts import get_provider
    from src.anubis.utils.connected_accounts.browser_login import (
        get_live_login,
        render_login_page_html,
        verify_login_token,
    )
    from src.anubis.utils.connected_accounts.browser_sessions import BrowserSessionError
    from src.anubis.utils.connected_accounts.oauth_flow import allowed_popup_origins

    try:
        payload = verify_login_token(app.state.context, t, login_id)
    except BrowserSessionError as session_error:
        return _popup_result_response({"ok": False, "error": session_error.detail})
    login = get_live_login(login_id)
    if login is None or login.user_id != payload.get("user_id"):
        return _popup_result_response(
            {"ok": False, "nonce": payload.get("nonce"), "error": "This sign-in window has expired. Start the sign-in again."}
        )
    provider = get_provider(login.provider_name)
    return HTMLResponse(
        render_login_page_html(
            login_id=login_id,
            token=t,
            nonce=login.nonce,
            provider_name=login.provider_name,
            display_name=provider.display_name if provider else login.provider_name,
            site_url=login.site_url,
            allowed_origins=allowed_popup_origins(app.state.context),
            finish_path=f"/connect_account/browser/{login_id}/finish",
            cancel_path=f"/connect_account/browser/{login_id}/cancel",
            stream_path=f"/connect_account/browser/{login_id}/stream",
        )
    )


@app.websocket("/connect_account/browser/{login_id}/stream")
async def connect_account_browser_stream(websocket: WebSocket, login_id: str):
    """Stream frames of the sign-in browser to the popup and forward input."""
    from src.anubis.utils.connected_accounts.browser_login import (
        get_live_login,
        stream_login,
        verify_login_token,
    )
    from src.anubis.utils.connected_accounts.browser_sessions import BrowserSessionError

    token = websocket.query_params.get("t") or ""
    try:
        payload = verify_login_token(app.state.context, token, login_id)
    except BrowserSessionError:
        await websocket.close(code=4401)
        return
    login = get_live_login(login_id)
    if login is None or login.user_id != payload.get("user_id") or login.finished:
        await websocket.close(code=4404)
        return
    if login.streaming:
        await websocket.close(code=4409)
        return
    await websocket.accept()
    try:
        await stream_login(websocket, login, app.state.context)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.post("/connect_account/browser/{login_id}/finish")
async def connect_account_browser_finish(request: Request, login_id: str):
    """Save the signed-in session as a connected account."""
    from src.anubis.utils.connected_accounts import get_provider, public_account_view
    from src.anubis.utils.connected_accounts.browser_login import (
        finish_login,
        verify_login_token,
    )
    from src.anubis.utils.connected_accounts.browser_sessions import BrowserSessionError
    from src.anubis.utils.connected_accounts.tool_factories import tool_names_for

    token = _login_token_from_request(request)
    try:
        payload = verify_login_token(app.state.context, token, login_id)
    except BrowserSessionError as session_error:
        raise HTTPException(status_code=session_error.status_code, detail=session_error.detail)
    user_id = str(payload.get("user_id") or "")
    nonce = payload.get("nonce")
    existing_records = await _connected_account_records_without_session(user_id)
    try:
        finished = await finish_login(
            app.state.context, login_id=login_id, user_id=user_id, existing_records=existing_records
        )
        record = finished["record"]
        provider = get_provider(str(record.get("provider") or ""))
        _enforce_connection_caps(existing_records, provider, record)
        await _store_connected_record_without_session(user_id, record)
        _after_record_stored(provider, record)
    except BrowserSessionError as session_error:
        return JSONResponse(content={"ok": False, "nonce": nonce, "error": session_error.detail}, status_code=200)
    except HTTPException as http_error:
        return JSONResponse(content={"ok": False, "nonce": nonce, "error": str(http_error.detail)}, status_code=200)
    view = public_account_view(record)
    return JSONResponse(
        content={
            "ok": True,
            "nonce": nonce,
            "provider": record.get("provider"),
            "account_key": view.get("account_key"),
            "display_label": view.get("display_label"),
            "account_address": view.get("account_address"),
            "tool_count": len(tool_names_for(provider, record)),
            "heuristic_signed_in": finished.get("heuristic_signed_in"),
        },
        status_code=200,
    )


@app.post("/connect_account/browser/{login_id}/cancel")
async def connect_account_browser_cancel(request: Request, login_id: str):
    """Close a sign-in window without saving anything."""
    from src.anubis.utils.connected_accounts.browser_login import (
        cancel_login,
        verify_login_token,
    )
    from src.anubis.utils.connected_accounts.browser_sessions import BrowserSessionError

    token = _login_token_from_request(request)
    try:
        payload = verify_login_token(app.state.context, token, login_id)
    except BrowserSessionError as session_error:
        raise HTTPException(status_code=session_error.status_code, detail=session_error.detail)
    cancelled = await cancel_login(login_id, str(payload.get("user_id") or ""))
    return JSONResponse(content={"ok": False, "cancelled": cancelled}, status_code=200)


@app.post("/connect_account/oauth/start")
async def connect_account_oauth_start(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Begin a popup sign-in with a vendor's own OAuth page.

    Body: ``provider`` (gmail, google_calendar, google_analytics, youtube,
    github, x, vercel, or custom_mcp with ``server_url`` and ``name``).
    Returns ``{authorization_url, nonce, expires_in}``; the card opens the URL
    in a window it opened synchronously on the click. 503 when the vendor's
    client id is not configured on this server.
    """
    from src.anubis.utils.connected_accounts import get_provider
    from src.anubis.utils.connected_accounts.mcp_oauth import McpOAuthError
    from src.anubis.utils.connected_accounts.oauth_flow import (
        OAuthFlowError,
        start_oauth,
    )
    from src.anubis.utils.connected_accounts.oauth_state import OAuthStateError

    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="A JSON object body is required.")
    provider = get_provider(str(body.get("provider") or ""))
    if provider is None:
        raise HTTPException(status_code=400, detail="Unknown provider.")
    if provider.login_mode != "oauth_popup" and provider.name != "custom_mcp":
        raise HTTPException(
            status_code=400,
            detail=f"{provider.display_name} does not sign in through OAuth.",
        )

    token = current_user["API_KEY"]
    user_id = current_user["identities"][0]["user_id"]
    client = get_client(headers={"API-KEY": f"{token}"})
    personal_avatar = await _resolve_personal_avatar_for_connection(
        client, request, current_user, token
    )
    try:
        started = await start_oauth(
            app.state.context,
            user_id=user_id,
            assistant_id=str(personal_avatar.get("assistant_id")),
            provider=provider,
            server_url=body.get("server_url"),
            name=body.get("name"),
        )
    except OAuthFlowError as flow_error:
        if flow_error.status_code == 503 and provider.login_url:
            # No OAuth app for this vendor on this server: the owner signs in
            # on the vendor's own login page in the live browser instead, and
            # the account is used through the signed-in session. Every
            # account stays connectable without a vendor console step.
            return JSONResponse(
                content=await _start_browser_fallback(
                    request,
                    current_user,
                    provider,
                    personal_avatar,
                    name=body.get("name"),
                    reconnect_account_key=body.get("reconnect_account_key"),
                ),
                status_code=200,
            )
        raise HTTPException(status_code=flow_error.status_code, detail=flow_error.detail)
    except McpOAuthError as flow_error:
        raise HTTPException(status_code=flow_error.status_code, detail=flow_error.detail)
    except OAuthStateError as state_error:
        raise HTTPException(status_code=503, detail=str(state_error))
    return JSONResponse(content=started, status_code=200)


async def _start_browser_fallback(
    request: Request,
    current_user: dict,
    provider: Any,
    personal_avatar: dict,
    *,
    name: str | None = None,
    reconnect_account_key: str | None = None,
) -> dict:
    """Open the live sign-in browser for a vendor whose OAuth app is not configured."""
    from src.anubis.utils.connected_accounts.browser_login import start_login
    from src.anubis.utils.connected_accounts.browser_sessions import BrowserSessionError

    user_id = current_user["identities"][0]["user_id"]
    reconnect_key = str(reconnect_account_key or "").strip() or None
    if reconnect_key and reconnect_key.startswith("account:"):
        reconnect_key = reconnect_key[len("account:"):]
    try:
        started = await start_login(
            app.state.context,
            user_id=user_id,
            assistant_id=str(personal_avatar.get("assistant_id")),
            provider=provider,
            site_url=provider.login_url,
            name=str(name or "").strip() or provider.display_name,
            reconnect_account_key=reconnect_key,
        )
    except BrowserSessionError as session_error:
        raise HTTPException(status_code=session_error.status_code, detail=session_error.detail)
    started["login_mode"] = "browser_session"
    started["fallback"] = "browser_session"
    return started


@app.get("/connect_account/oauth/callback")
async def connect_account_oauth_callback(
    request: Request,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    error_description: Optional[str] = None,
):
    """Where a vendor's sign-in page returns. Public: the signed state is the auth.

    Exchanges the code, proves the token, stores the encrypted result, and
    renders a page that posts a non-secret result to the window that opened
    the popup, then closes. Every failure renders the same page with
    ``ok: false`` and a plain message; token-endpoint text never reaches the
    browser.
    """
    from src.anubis.utils.connected_accounts import get_provider, public_account_view
    from src.anubis.utils.connected_accounts.mcp_oauth import McpOAuthError
    from src.anubis.utils.connected_accounts.oauth_flow import (
        OAuthFlowError,
        complete_oauth,
    )
    from src.anubis.utils.connected_accounts.oauth_state import (
        OAuthStateError,
        state_secret,
        verify_state,
    )
    from src.anubis.utils.connected_accounts.tool_factories import tool_names_for

    nonce = None
    provider_name = None
    try:
        payload = verify_state(state or "", state_secret(app.state.context))
        nonce = payload.get("nonce")
        provider_name = payload.get("provider")
    except OAuthStateError as state_error:
        return _popup_result_response({"ok": False, "error": str(state_error)})
    if error:
        return _popup_result_response(
            {
                "ok": False,
                "nonce": nonce,
                "provider": provider_name,
                "error": "The sign-in was cancelled or refused."
                if error == "access_denied"
                else f"The sign-in did not complete ({error}).",
            }
        )
    if not code:
        return _popup_result_response(
            {"ok": False, "nonce": nonce, "provider": provider_name, "error": "No code was returned."}
        )

    user_id = str(payload.get("user_id") or "")
    existing_records = await _connected_account_records_without_session(user_id)
    try:
        record = await complete_oauth(
            app.state.context,
            code=code,
            state=state or "",
            existing_records=existing_records,
        )
        provider = get_provider(str(record.get("provider") or ""))
        _enforce_connection_caps(existing_records, provider, record)
        await _store_connected_record_without_session(user_id, record)
        _after_record_stored(provider, record)
    except (OAuthFlowError, McpOAuthError) as flow_error:
        return _popup_result_response(
            {"ok": False, "nonce": nonce, "provider": provider_name, "error": flow_error.detail}
        )
    except HTTPException as http_error:
        return _popup_result_response(
            {"ok": False, "nonce": nonce, "provider": provider_name, "error": str(http_error.detail)}
        )
    except Exception:
        logger.exception("OAuth callback failed for provider %s", provider_name)
        return _popup_result_response(
            {
                "ok": False,
                "nonce": nonce,
                "provider": provider_name,
                "error": "The sign-in could not be completed. Start the sign-in again.",
            }
        )
    view = public_account_view(record)
    return _popup_result_response(
        {
            "ok": True,
            "nonce": nonce,
            "provider": record.get("provider"),
            "account_key": view.get("account_key"),
            "display_label": view.get("display_label"),
            "account_address": view.get("account_address"),
            "tool_count": len(tool_names_for(provider, record)),
        }
    )


@app.post("/connect_account")
async def connect_account_route(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Connect one of the owner's accounts to their personal avatar.

    Body: ``provider`` plus the fields that provider's connect card declares,
    either flat (``email_address``, ``app_password``) or under ``fields``. The
    catalog (``GET /connectable_providers``) says which fields each provider
    needs, so a new provider is connected through this same route with no client
    change. Coming-soon providers answer 501 with a plain message; providers
    connected by running a daemon answer 400 with the pairing instructions.
    """
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="A JSON object body is required.")
    provider_name = str(body.get("provider") or "").strip().lower()
    if not provider_name:
        raise HTTPException(status_code=400, detail="A provider is required.")
    fields = dict(body.get("fields") or {})
    for name, value in body.items():
        if name not in ("provider", "fields", "assistant_id"):
            fields.setdefault(name, value)
    return await _connect_account_from_fields(
        request, current_user, provider_name, fields
    )


@app.post("/connect_mailbox")
async def connect_mailbox(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Connect one of the owner's email accounts to their personal avatar.

    Body: ``provider`` (default "email_account"), ``email_address``,
    ``password``, and optionally ``imap_host`` / ``smtp_host`` for a domain that
    publishes no settings. Kept as an alias of ``POST /connect_account`` for
    clients that predate the generic route; the behaviour — prove by real login,
    encrypt, store, never log the plaintext — is identical because both routes
    share one body. ``app_password`` is still read so a browser tab opened
    before the field was renamed still connects; it is never offered.
    """
    body = await request.json()
    provider_name = str(body.get("provider") or "email_account").strip().lower()
    fields = {
        "email_address": body.get("email_address"),
        "password": body.get("password") or body.get("app_password"),
        "imap_host": body.get("imap_host"),
        "smtp_host": body.get("smtp_host"),
    }
    return await _connect_account_from_fields(
        request, current_user, provider_name, fields
    )


@app.get("/connectable_providers")
async def connectable_providers(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Describe every account that can be connected, and the form each one needs.

    The same card description the avatar raises mid-conversation, served for the
    settings screen and the New Connector picker. Both surfaces read this one
    description so a provider's labels, fields, and help text cannot drift into
    two versions — which matters most for the app-password explanation, the only
    place an owner is told that a Google account password will never
    authenticate. Providers come back in catalog order (featured first, then by
    category) with their availability, so a coming-soon row renders disabled.

    Carries no user data: it is the static registry, projected for a client.
    """
    from src.anubis.utils.connected_accounts.connection_tools import (
        build_connect_card,
    )
    from src.anubis.utils.connected_accounts.providers import (
        CATEGORY_ORDER,
        catalog_providers,
    )

    return JSONResponse(
        content={
            "categories": list(CATEGORY_ORDER),
            "providers": [
                build_connect_card(provider) for provider in catalog_providers()
            ],
        },
        status_code=200,
    )


@app.get("/list_connected_accounts")
async def list_connected_accounts(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """List the external accounts connected to the caller's personal avatar.

    Returns the projection in ``public_account_view`` only. Neither the password
    (never stored) nor its ciphertext is included: the ciphertext is useless to a
    caller and returning it would only help an attacker who later obtained the
    encryption key.
    """
    from src.anubis.utils.connected_accounts import public_account_view

    token = current_user["API_KEY"]
    user_id = current_user["identities"][0]["user_id"]
    client = get_client(headers={"API-KEY": f"{token}"})
    await _resolve_personal_avatar_for_connection(client, request, current_user, token)

    records = await _connected_account_records(client, user_id)
    return JSONResponse(
        content={"accounts": [public_account_view(record) for record in records]},
        status_code=200,
    )


async def _device_rows_for_user(client: Any, user_id: str) -> list[dict[str, Any]]:
    """The device rows ``GET /list_mcp_connections`` returns, as a reusable helper."""
    from src.anubis.utils.tools.data_analysis import relay as relay_registry
    from src.anubis.utils.tools.data_analysis.backend import (
        mcp_connection_namespace,
        mcp_registration_namespace,
    )

    registrations = await _search_namespace_records(
        client, mcp_registration_namespace(user_id)
    )
    connections = await _search_namespace_records(
        client, mcp_connection_namespace(user_id)
    )
    connection_by_device = {
        record.get("device_id"): record
        for record in connections
        if record.get("device_id")
    }

    devices: list[dict[str, Any]] = []
    seen_device_ids: set[str] = set()
    for registration in registrations:
        registration_device_id = registration.get("device_id")
        if not registration_device_id:
            continue
        live_session = relay_registry.get_session(registration_device_id)
        if live_session is not None and live_session.user_id != user_id:
            # Leftover store row from a bind that was never this account's
            # daemon. The socket names the real owner.
            continue
        seen_device_ids.add(registration_device_id)
        connection = connection_by_device.get(registration_device_id) or {}
        devices.append(
            {
                "device_id": registration_device_id,
                "device_label": registration.get("device_label")
                or connection.get("device_label"),
                "platform": registration.get("platform") or connection.get("platform"),
                "server_name": registration.get("server_name"),
                "connection_mode": registration.get("connection_mode"),
                "online": relay_registry.is_online(registration_device_id, user_id),
                "last_seen_at": registration.get("last_seen_at"),
                "connected": connection.get("status") == "connected",
                "bound_assistant_id": connection.get("assistant_id"),
                "connected_at": connection.get("connected_at"),
                "owner_user_id": registration.get("owner_user_id")
                or connection.get("owner_user_id")
                or user_id,
            }
        )

    # A connection whose registration record is gone (the daemon unregistered on
    # shutdown but the avatar keeps the adopted connection) still belongs in the
    # listing, otherwise a user sees a machine vanish while the avatar still
    # holds tools for the machine.
    for device_identifier, connection in connection_by_device.items():
        if device_identifier in seen_device_ids:
            continue
        live_session = relay_registry.get_session(device_identifier)
        if live_session is not None and live_session.user_id != user_id:
            continue
        devices.append(
            {
                "device_id": device_identifier,
                "device_label": connection.get("device_label"),
                "platform": connection.get("platform"),
                "server_name": connection.get("server_name"),
                "connection_mode": None,
                "online": relay_registry.is_online(device_identifier, user_id),
                "last_seen_at": None,
                "connected": connection.get("status") == "connected",
                "bound_assistant_id": connection.get("assistant_id"),
                "connected_at": connection.get("connected_at"),
                "owner_user_id": connection.get("owner_user_id") or user_id,
            }
        )

    devices.sort(key=lambda device: str(device.get("device_label") or ""))
    return devices


@app.get("/list_connections")
async def list_connections(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Everything the personal avatar is connected to, in one row shape.

    Accounts (mailboxes, custom connectors) and devices (machines running the
    Neural Nexus daemon) are stored differently because their lifecycles differ,
    but the owner should not have to know that: the "+" menu and the settings
    screen render this one list. Keys are prefixed ``account:`` / ``device:`` so
    ``POST /set_connection_state`` can route a toggle to the right store, and
    every row names its own ``disconnect_endpoint``.
    """
    from src.anubis.utils.connected_accounts.listing import (
        account_connection_view,
        device_connection_view,
    )

    token = current_user["API_KEY"]
    user_id = current_user["identities"][0]["user_id"]
    client = get_client(headers={"API-KEY": f"{token}"})
    personal_avatar = await _resolve_personal_avatar_for_connection(
        client, request, current_user, token
    )

    account_records = await _connected_account_records(client, user_id)
    device_rows = await _device_rows_for_user(client, user_id)
    connections = [account_connection_view(record) for record in account_records] + [
        device_connection_view(device) for device in device_rows
    ]
    return JSONResponse(
        content={
            "personal_avatar_id": personal_avatar.get("assistant_id"),
            "connection_count": len(connections),
            "connections": connections,
        },
        status_code=200,
    )


@app.post("/set_connection_state")
async def set_connection_state(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Toggle one connection on or off from the "+" menu or the settings screen.

    Body: ``connection_key`` (``account:...`` or ``device:...``) and
    ``connected`` (boolean). Off means DISCONNECT: the account's record and its
    encrypted credential are deleted, or the device's adopted connection is
    dropped. On cannot restore a deleted credential, so for an account it
    returns the provider's connect card description for the client to open —
    the same card the avatar raises in chat — and for a device it clears the
    per-avatar suppression marker so ``mcp_auto_adopt`` binds the machine on
    the next turn.
    """
    from src.anubis.utils.connected_accounts import get_provider
    from src.anubis.utils.connected_accounts.connection_tools import build_connect_card
    from src.anubis.utils.connected_accounts.listing import split_connection_key

    body = await request.json()
    connection_key = str((body or {}).get("connection_key") or "").strip()
    connected = bool((body or {}).get("connected"))
    try:
        source_kind, identifier = split_connection_key(connection_key)
    except ValueError as key_error:
        raise HTTPException(status_code=400, detail=str(key_error))

    token = current_user["API_KEY"]
    user_id = current_user["identities"][0]["user_id"]
    client = get_client(headers={"API-KEY": f"{token}"})
    personal_avatar = await _resolve_personal_avatar_for_connection(
        client, request, current_user, token
    )

    if source_kind == "account":
        existing = await _get_connected_account_record(client, user_id, identifier)
        if not connected:
            if existing is None:
                raise HTTPException(
                    status_code=404, detail=f"No connected account {identifier!r}."
                )
            await _delete_connected_account_record(client, user_id, identifier)
            if existing.get("kind") == "mcp_server":
                from src.anubis.utils.connected_accounts.mcp_server_tools import (
                    forget_cached_tools,
                )

                forget_cached_tools(
                    (existing.get("transport") or {}).get("server_url") or ""
                )
            return JSONResponse(
                content={"connection_key": connection_key, "connected": False},
                status_code=200,
            )
        provider_name = identifier.split(":", 1)[0]
        provider = get_provider(provider_name)
        if provider is None:
            raise HTTPException(
                status_code=400, detail=f"Unknown provider {provider_name!r}."
            )
        if existing is not None and existing.get("status") == "connected":
            return JSONResponse(
                content={"connection_key": connection_key, "connected": True},
                status_code=200,
            )
        return JSONResponse(
            content={
                "connection_key": connection_key,
                "connected": False,
                "action": "open_connect_card",
                "card": build_connect_card(provider, [existing] if existing else []),
            },
            status_code=200,
        )

    # Devices.
    from src.anubis.utils.tools.data_analysis.backend import (
        mcp_connection_declined_namespace,
        mcp_connection_namespace,
    )

    assistant_id = personal_avatar.get("assistant_id")
    declined_namespace = list(mcp_connection_declined_namespace(user_id, assistant_id))
    if not connected:
        await client.store.delete_item(
            list(mcp_connection_namespace(user_id)), key=identifier
        )
        # Mark the device declined for this avatar so auto-adopt does not
        # silently re-bind the machine on the next turn — that is what made an
        # explicit disconnect appear not to work before the marker existed.
        await client.store.put_item(
            declined_namespace,
            key=identifier,
            value={
                "device_id": identifier,
                "declined_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return JSONResponse(
            content={"connection_key": connection_key, "connected": False},
            status_code=200,
        )
    try:
        await client.store.delete_item(declined_namespace, key=identifier)
    except Exception:
        logger.debug("No suppression marker to clear for device %s", identifier)
    return JSONResponse(
        content={
            "connection_key": connection_key,
            "connected": True,
            "message": (
                "The machine will be connected on the next conversation turn if the "
                "Neural Nexus daemon is running on the machine."
            ),
        },
        status_code=200,
    )


@app.delete("/disconnect_account")
async def disconnect_account(
    request: Request,
    account_key: str,
    current_user: dict = Depends(get_current_user),
):
    """Disconnect one external account, deleting its stored credential.

    The account key is REQUIRED and exactly one record is deleted. There is
    deliberately no "disconnect everything" mode: ``/disconnect_mcp`` shipped
    with an omitted identifier meaning "remove every device", and that shape is
    what let one caller destroy every record it could see. Repeating it here
    would put every mailbox a user owns behind a single missing argument.
    """
    token = current_user["API_KEY"]
    user_id = current_user["identities"][0]["user_id"]
    client = get_client(headers={"API-KEY": f"{token}"})
    await _resolve_personal_avatar_for_connection(client, request, current_user, token)

    if not str(account_key or "").strip():
        raise HTTPException(
            status_code=400,
            detail="An account_key is required; name the account to disconnect.",
        )

    # Existence is checked before deleting rather than inferred from the delete
    # call: a delete alone would answer "disconnected" for an account that was
    # never connected — and an owner who mistyped a key would believe a live
    # mailbox had been removed.
    existing = await _get_connected_account_record(client, user_id, account_key)
    if not existing:
        raise HTTPException(
            status_code=404,
            detail=f"No connected account {account_key!r} to disconnect.",
        )
    await _delete_connected_account_record(client, user_id, account_key)
    if existing.get("kind") == "mcp_server":
        from src.anubis.utils.connected_accounts.mcp_server_tools import (
            forget_cached_tools,
        )

        forget_cached_tools((existing.get("transport") or {}).get("server_url") or "")

    return JSONResponse(
        content={"disconnected": True, "account_key": account_key},
        status_code=200,
    )


def _writing_sample_text_from_sent_messages(messages: list[dict[str, Any]]) -> str:
    """Render the owner's sent messages as one plain-text document for ingestion.

    Quoted replies and signatures are trimmed so the identity pipeline learns
    from the owner's own sentences rather than from whatever the owner was
    replying to. Each message keeps its subject and recipient as context lines,
    which the quote-extraction pass uses as the prompt that elicited the text.
    """
    import re as regular_expressions

    sections: list[str] = []
    for message in messages:
        body = str(message.get("body_text") or "")
        kept_lines: list[str] = []
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith(">"):
                continue
            if regular_expressions.match(
                r"^On .+ wrote:$|^-{2,}\s*$|^Sent from my ", stripped
            ):
                break
            kept_lines.append(line.rstrip())
        text = "\n".join(kept_lines).strip()
        if not text:
            continue
        sections.append(
            f"Subject: {message.get('subject') or ''}\n"
            f"To: {', '.join(message.get('recipients') or []) if isinstance(message.get('recipients'), list) else message.get('recipients') or ''}\n"
            f"Date: {message.get('sent_at') or ''}\n\n{text}"
        )
    return "\n\n---\n\n".join(sections)


@app.post("/import_mailbox_writing_samples")
async def import_mailbox_writing_samples(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Pull the owner's sent mail into the personal avatar's identity.

    Body: ``account_key`` (optional when exactly one mailbox is connected),
    ``limit`` (default 50, capped by ``MAILBOX_FETCH_MAX_MESSAGES`` × 4).

    This is the "pull data to update the identity" lever for a mailbox: the
    owner's own sent messages are the best evidence of how the owner writes, so
    they are fetched from the provider's sent folder, stripped of quoted text and
    signatures, and fed through the SAME background media job an uploaded text
    file takes. That way they land in the ``quote`` / ``identity`` namespaces the
    avatar already grounds on, are metered as an upload, and show up in the
    documents list under one namespace per import — never a second ingestion
    path to keep in step with the first.
    """
    from src.anubis.utils.connected_accounts import get_provider
    from src.anubis.utils.secret_store import SecretDecryptionError, decrypt_secret
    from src.anubis.utils.tools.email.imap_client import (
        MailboxAuthenticationError,
        MailboxCredentials,
        MailboxUnreachableError,
        search_messages,
    )

    enforce_tier_capability(current_user, TierCapability.UPLOAD)
    body = await request.json()
    body = body if isinstance(body, dict) else {}
    requested_key = str(body.get("account_key") or "").strip()
    fetch_ceiling = int(app.state.context.mailbox_fetch_max_messages or 25) * 4
    limit = max(1, min(int(body.get("limit") or 50), fetch_ceiling))

    token = current_user["API_KEY"]
    user_id = current_user["identities"][0]["user_id"]
    client = get_client(headers={"API-KEY": f"{token}"})
    personal_avatar = await _resolve_personal_avatar_for_connection(
        client, request, current_user, token
    )
    assistant_id = personal_avatar.get("assistant_id")

    mailboxes = [
        record
        for record in await _connected_account_records(client, user_id)
        if record.get("kind") == "mailbox"
        and record.get("status") == "connected"
        and record.get("assistant_id") == assistant_id
    ]
    if requested_key:
        mailboxes = [
            record for record in mailboxes if record.get("account_key") == requested_key
        ]
    if not mailboxes:
        raise HTTPException(
            status_code=404,
            detail="No connected mailbox matches; connect a mailbox first.",
        )
    if len(mailboxes) > 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "Several mailboxes are connected; pass account_key to say which "
                f"one to import from: {[record.get('account_key') for record in mailboxes]}"
            ),
        )
    record = mailboxes[0]
    provider = get_provider(str(record.get("provider") or ""))
    sent_folder = record.get("sent_mailbox") or (
        provider.sent_mailbox if provider else None
    )
    if not sent_folder:
        raise HTTPException(
            status_code=400, detail="This mailbox provider exposes no sent folder."
        )

    try:
        from src.anubis.utils.connected_accounts.mailbox_credentials import (
            mailbox_credentials_for,
        )
        from src.anubis.utils.connected_accounts.oauth_flow import (
            OAuthReconnectRequired,
        )

        credentials = await mailbox_credentials_for(
            record, app.state.context, store=None, user_id=user_id
        )
    except SecretDecryptionError:
        raise HTTPException(
            status_code=409,
            detail="The stored mailbox credential could not be read; reconnect the mailbox.",
        )
    except OAuthReconnectRequired:
        raise HTTPException(
            status_code=409,
            detail="Google no longer accepts the saved sign-in; sign in with Google again.",
        )
    try:
        messages = await asyncio.to_thread(
            search_messages, credentials, None, limit, sent_folder
        )
    except MailboxAuthenticationError:
        raise HTTPException(
            status_code=409,
            detail="The mailbox rejected its saved password; reconnect it.",
        )
    except MailboxUnreachableError as unreachable_error:
        raise HTTPException(status_code=503, detail=str(unreachable_error))

    text = _writing_sample_text_from_sent_messages(messages)
    if not text.strip():
        raise HTTPException(
            status_code=404, detail="No sent messages with readable text were found."
        )

    content = text.encode("utf-8")
    label = str(record.get("display_label") or "mailbox")
    stamp = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S")
    filename = f"sent_mail_{label}_{stamp}.txt"
    media_entries = await _build_media_entries_for_file(
        filename,
        content,
        "text/plain",
        reference_image=False,
        reference_audio=False,
        user_id=user_id,
        assistant_id=assistant_id,
    )
    for entry in media_entries:
        entry["estimated_tokens"] = max(1, len(content) // 4)

    config = {
        "configurable": {
            "user_id": user_id,
            "user_ctx": {"name": None, "description": None},
            "assistant_id": assistant_id,
            "assistant_ctx": {
                "name": personal_avatar.get("name"),
                "description": personal_avatar.get("description"),
                "assistant_id": assistant_id,
                "metadata": personal_avatar.get("metadata") or {},
            },
        }
    }
    registry = app.state.media_jobs
    master = create_master_job(registry, user_id, assistant_id)
    items: list = []
    for media_file in media_entries:
        child = create_child_job(
            registry,
            user_id=user_id,
            assistant_id=assistant_id,
            parent_id=master.job_id,
            filename=media_file.get("filename"),
            namespace_filename=media_file.get("namespace_filename"),
            estimated_tokens=media_file.get("estimated_tokens"),
            estimated_media_seconds=media_file.get("estimated_media_seconds"),
            estimated_processing_seconds=media_file.get("estimated_processing_seconds"),
        )
        master.child_ids.append(child.job_id)
        items.append({"child": child, "media_file": media_file})
    master.task = asyncio.create_task(
        run_batch_media_job(
            master,
            items,
            config,
            app.state.store,
            app.state.context,
            concurrency=max(1, app.state.context.media_processing_concurrency),
            existing_namespaces=[],
            registry=registry,
            deferred_expanders=[],
        )
    )
    return JSONResponse(
        status_code=202,
        content={
            "job_id": master.job_id,
            "status": master.status,
            "status_url": f"/media_job/{master.job_id}",
            "progress_url": f"/media_job/{master.job_id}/progress",
            "cancel_url": f"/media_job/{master.job_id}/cancel",
            "account_key": record.get("account_key"),
            "messages_imported": len(messages),
            "filename": filename,
        },
    )


@app.post("/create_avatar")
async def create_avatar(
    name: str,
    description: Optional[str] = None,
    is_public: bool = False,
    is_personal_avatar_of_creator: bool = False,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    location_name: Optional[str] = None,
    geofence_radius_meters: Optional[int] = None,
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

    # An avatar may be pinned to a real-world place at creation. The pin is
    # optional here and can be added, moved, or removed later through
    # PATCH /modify_avatar, so a creator is never forced to know the place up
    # front. Validate before the try block below: that block turns every
    # exception into a 500, and a bad coordinate is a 400 the caller can fix.
    geo_location = _build_geo_location_or_400(
        latitude=latitude,
        longitude=longitude,
        location_name=location_name,
        geofence_radius_meters=geofence_radius_meters,
    )

    try:
        assistant_id = str(uuid4())
        user_id = current_user["identities"][0]["user_id"]
        metadata = {
            "user_id": user_id,
            "is_public": False,
            "is_personal_avatar_of_creator": is_personal_avatar_of_creator,
        }
        if geo_location is not None:
            metadata[GEO_LOCATION_METADATA_KEY] = geo_location

        if user_id == context.admin_user_id:
            # or is_personal_avatar_of_creator == True
            # verify there is only a single personal avatar of the creator;
            # verfiy the personal avatar of the creator against social media accounts
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
    """List an avatar publicly, or withdraw it again.

    A user may share an avatar they created, and nothing else. That is the same
    rule /delete_avatar and /update_avatar_identity_with_media enforce, resolved
    from the same ``metadata.user_id``, and it satisfies the product rule that a
    person shares only their own likeness: the avatar someone made of themselves
    is theirs to publish. The admin account may still share any avatar.

    Sharing is reversible — ``is_public=false`` withdraws the avatar from
    /list_public_avatars — so this grants no permanent exposure.
    """
    context = app.state.context
    user_id = current_user["identities"][0]["user_id"]
    token = current_user["API_KEY"]
    client = get_client(headers={"API-KEY": f"{token}"})

    try:
        assistant = await client.assistants.get(assistant_id)
    except Exception as lookup_error:
        raise HTTPException(
            status_code=404, detail=f"Could not load assistant: {lookup_error}"
        ) from lookup_error

    assistant_metadata = assistant.get("metadata") or {}
    creator_id = assistant_metadata.get("user_id")
    is_admin = user_id == context.admin_user_id
    if not is_admin and (not creator_id or creator_id != user_id):
        raise HTTPException(
            status_code=403,
            detail="Only the creator of this avatar may share it.",
        )

    # Sharing is limited to the avatar that depicts its creator. That is the
    # entire basis on which a user is entitled to publish an avatar: it is their
    # own likeness. An avatar they invented, or assembled from someone else's
    # material, carries no such entitlement and stays private.
    if not is_admin and not assistant_metadata.get("is_personal_avatar_of_creator"):
        raise HTTPException(
            status_code=403,
            detail=(
                "Only your personal avatar — the one that depicts you — may be "
                "shared publicly."
            ),
        )

    try:
        # LangGraph merges metadata by key, so this leaves user_id and the
        # personal-avatar flag alone and changes only the sharing state.
        result = await client.assistants.update(
            assistant_id=assistant_id, metadata={"is_public": is_public}
        )
        return JSONResponse(result, status_code=200)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during update of sharing avatar: {e}"
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
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    location_name: Optional[str] = None,
    geofence_radius_meters: Optional[int] = None,
    clear_geo_location: bool = False,
):
    # Avatar name changes also need to be applied to the db for consistent identities
    logger.info("breakpoint")
    update_personal_avatar_flag = (
        "is_personal_avatar_of_creator" in request.query_params
    )
    # The pin may be added, moved, or removed long after the avatar was created,
    # so this route — not only /create_avatar — is where a place is set.
    update_geo_location = latitude is not None or longitude is not None
    if not assistant_id:
        raise HTTPException(
            detail="Supply assistant_id for the assistant to modify.", status_code=400
        )
    if (
        not new_avatar_name
        and not new_avatar_description
        and not update_personal_avatar_flag
        and not update_geo_location
        and not clear_geo_location
    ):
        raise HTTPException(
            detail=(
                "Supply at least one of: new avatar name, new avatar description, "
                "is_personal_avatar_of_creator, a latitude and longitude, or "
                "clear_geo_location."
            ),
            status_code=400,
        )

    if not current_user:
        raise HTTPException(
            content="User must be logged in to modify avatar avatars.", status_code=401
        )

    token = current_user["API_KEY"]
    client = get_client(headers={"API-KEY": f"{token}"})

    # Moving an avatar in the physical world is the creator's call alone: an
    # avatar that stands at a memorial must not be re-pinned to a storefront by
    # a stranger. The name and description paths lean on the LangGraph API key,
    # but a pin change is gated here the way /share_avatar and /delete_avatar
    # gate theirs.
    geo_location = _build_geo_location_or_400(
        latitude=latitude,
        longitude=longitude,
        location_name=location_name,
        geofence_radius_meters=geofence_radius_meters,
    )
    if update_geo_location or clear_geo_location:
        await resolve_assistant_for_creator(
            assistant_id,
            current_user,
            action_description="change that avatar's real-world location",
        )

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
    if geo_location is not None or clear_geo_location:
        # The assistant update merges metadata key by key, so writing None here
        # removes the pin and leaves user_id and is_public untouched.
        update_kwargs.setdefault("metadata", {})[GEO_LOCATION_METADATA_KEY] = (
            None if clear_geo_location else geo_location
        )

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
    device_id: str | None = None,
    current_user: dict = Depends(get_current_user),
):
    """Forget saved MCP data-server connections (disconnect).

    Deletes one machine's connection record when ``device_id`` is given, or every
    machine's when it is omitted. There is no enable/disable switch — removing the
    connection is the disconnect.

    Note that the avatar re-adopts a reachable machine automatically on the next
    turn. Suppressing that re-adoption is a per-avatar decision made in
    conversation ("disconnect my phone"), which writes the suppression marker;
    this endpoint is the account-level equivalent of unplugging, not a permanent
    opt-out.
    """
    from src.anubis.utils.tools.data_analysis.backend import (
        mcp_connection_namespace,
    )

    user_id = current_user["identities"][0]["user_id"]
    token = current_user["API_KEY"]
    client = get_client(headers={"API-KEY": f"{token}"})
    namespace = list(mcp_connection_namespace(user_id))
    try:
        if device_id:
            await client.store.delete_item(namespace, key=device_id)
            removed = [device_id]
        else:
            records = await _search_namespace_records(
                client, mcp_connection_namespace(user_id)
            )
            removed = []
            for record in records:
                record_device_id = record.get("device_id")
                if not record_device_id:
                    continue
                await client.store.delete_item(namespace, key=record_device_id)
                removed.append(record_device_id)
        return JSONResponse(
            content={
                "disconnected": True,
                "user_id": user_id,
                "disconnected_device_ids": removed,
            },
            status_code=200,
        )
    except Exception as disconnect_error:
        raise HTTPException(
            status_code=500,
            detail=f"Error disconnecting MCP server: {disconnect_error}",
        )


@app.post("/connect_mcp")
async def connect_mcp(
    request: Request,
    body: ConnectMcpRequest | None = None,
    current_user: dict = Depends(get_current_user),
):
    """Bind a reachable Neural Nexus machine to the personal avatar now.

    ``mcp_auto_adopt`` only runs at the start of a graph turn. A Connect-now
    click cannot wait for the next message: this endpoint clears suppression
    and writes the connection immediately so the daemon is attached this
    instant if it is reachable.
    """
    from src.anubis.utils.personal_avatar import resolve_personal_avatar
    from src.anubis.utils.tools.data_analysis.discovery import (
        clear_declined,
        resolve_available_connections,
        save_user_connection,
    )

    store = request.app.state.store
    context = request.app.state.context
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="The connection store is not available right now.",
        )

    token = current_user["API_KEY"]
    user_id = current_user["identities"][0]["user_id"]
    client = get_client(headers={"API-KEY": f"{token}"})
    try:
        personal_avatar = await resolve_personal_avatar(
            client, request, current_user, token
        )
    except Exception as resolution_error:
        raise HTTPException(
            status_code=500,
            detail=f"Error resolving the personal avatar: {resolution_error}",
        )
    if personal_avatar is None:
        raise HTTPException(
            status_code=500,
            detail="The personal avatar could not be provisioned.",
        )
    assistant_id = personal_avatar.get("assistant_id")
    if not assistant_id:
        raise HTTPException(
            status_code=500,
            detail="The personal avatar has no identifier.",
        )

    available = await resolve_available_connections(
        store, user_id, context, ignore_failure_backoff=True
    )
    requested_device_id = (body.device_id if body else None) or None
    requested_label = (body.device_label if body else None) or None
    if requested_device_id:
        requested_device_id = requested_device_id.removeprefix("device:")

    selected = available
    if requested_device_id:
        selected = [
            connection
            for connection in available
            if connection.device_id == requested_device_id
        ]
    elif requested_label:
        lowered = requested_label.strip().lower()
        selected = [
            connection
            for connection in available
            if (connection.device_label or "").strip().lower() == lowered
        ]

    if not selected:
        raise HTTPException(
            status_code=409,
            detail=(
                "No Neural Nexus machine is reachable right now. "
                "Confirm the daemon is running, then connect again."
            ),
        )

    connected_labels: list[str] = []
    for connection in selected:
        await save_user_connection(
            store,
            user_id,
            connection=connection,
            assistant_id=assistant_id,
        )
        await clear_declined(store, user_id, assistant_id, connection.device_id)
        connected_labels.append(connection.device_label or connection.device_id)

    return JSONResponse(
        content={
            "connected": True,
            "connected_devices": connected_labels,
            "message": ("Connected to " + ", ".join(connected_labels) + " now."),
        },
        status_code=200,
    )


@app.get("/list_mcp_connections")
async def list_mcp_connections(
    current_user: dict = Depends(get_current_user),
):
    """List every machine the user has registered, with live presence.

    Registrations and connections are separate records: a machine that is running
    the daemon appears as a registration, and gains a connection once the avatar
    adopts the machine. Both are merged here so one call answers "what do I have
    connected, and is each machine up right now?".

    Presence comes from the in-process relay registry rather than from the stored
    heartbeat, because the live socket is the authoritative signal for relay-mode
    machines. Host directory paths are deliberately omitted. The same rows feed
    ``GET /list_connections`` through ``_device_rows_for_user``.
    """
    user_id = current_user["identities"][0]["user_id"]
    token = current_user["API_KEY"]
    client = get_client(headers={"API-KEY": f"{token}"})

    devices = await _device_rows_for_user(client, user_id)
    return JSONResponse(
        content={"user_id": user_id, "device_count": len(devices), "devices": devices},
        status_code=200,
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
    ``proxy_request`` call. The registry is what the graph's ``mcp_auto_adopt``
    node and the ``/mcp/relay/{device_id}`` bridge read to reach this device.
    """
    from src.anubis.utils.tools.data_analysis import relay as relay_registry
    from src.anubis.utils.tools.data_analysis.devices import derive_device_identity

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

        # Older daemons announce no device label; derive one from the announced
        # server name so every machine is nameable in conversation whether or not
        # the daemon on that machine has been updated.
        device_label, platform = derive_device_identity(register_message)

        relay_registry.register_session(
            device_id=device_id,
            user_id=user_id,
            device_secret=device_secret,
            server_name=register_message.get("server_name") or "Ubuntu-OS-Filesystem",
            allowed_roots=tuple(register_message.get("allowed_roots") or []),
            websocket=websocket,
            device_label=device_label,
            platform=platform,
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
        logger.warning("MCP relay socket error for device %s", device_id, exc_info=True)
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
        (
            status_code,
            response_headers,
            response_body,
        ) = await relay_registry.proxy_request(
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
        return JSONResponse(content={"error": "MCP relay timed out."}, status_code=504)
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
    """Record one local MCP daemon's pushed presence for the authenticated user.

    Stores a registration record keyed by ``device_id``, so a user who runs the
    daemon on several machines accumulates one record per machine rather than
    each machine overwriting the last. The next turn on the user's personal
    avatar reads every record (``mcp_auto_adopt``) and binds the machines that
    are reachable and not explicitly suppressed.
    """
    from src.anubis.utils.tools.data_analysis.backend import (
        mcp_registration_namespace,
    )
    from src.anubis.utils.tools.data_analysis.devices import (
        deduplicate_label,
        derive_device_identity,
    )

    body = await request.json()
    user_id = current_user["identities"][0]["user_id"]
    token = current_user["API_KEY"]
    client = get_client(headers={"API-KEY": f"{token}"})

    connection_mode = body.get("connection_mode") or "relay"
    device_id = body.get("device_id")
    if not device_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "A device_id is required to register a Model Context Protocol "
                "server; the registration record is keyed by the device."
            ),
        )
    # In relay mode the daemon's announced mcp_url may point at a different
    # host (e.g. production) than the API instance that accepted the register
    # call. Always rewrite to this request's own relay bridge so adoption and
    # tool calls stay on the same process that holds the WebSocket.
    mcp_url = body.get("mcp_url")
    if connection_mode == "relay":
        mcp_url = f"{str(request.base_url).rstrip('/')}/mcp/relay/{device_id}"

    existing_records = await _search_namespace_records(
        client, mcp_registration_namespace(user_id)
    )
    # The cap counts only OTHER devices, so a machine that is already registered
    # can always re-register (every daemon restart does) even at the limit.
    other_device_count = sum(
        1 for record in existing_records if record.get("device_id") != device_id
    )
    max_devices = int(app.state.context.data_analysis_max_devices_per_user)
    if other_device_count >= max_devices:
        raise HTTPException(
            status_code=409,
            detail=(
                f"This account already has {other_device_count} registered "
                f"Model Context Protocol devices, which is the configured "
                f"maximum of {max_devices}. Disconnect a device before "
                f"registering another."
            ),
        )

    derived_label, platform = derive_device_identity(body)
    device_label = deduplicate_label(derived_label, existing_records, device_id)

    record = {
        "status": "pending_consent",
        "owner_user_id": user_id,
        "connection_mode": connection_mode,
        "server_name": body.get("server_name") or "Ubuntu-OS-Filesystem",
        # Every mode is driven as a streamable-HTTP client; in relay mode the
        # ``mcp_url`` points at this API's own ``/mcp/relay/<device_id>`` bridge.
        "transport": "streamable_http",
        "device_id": device_id,
        "device_label": device_label,
        "platform": platform,
        "device_secret": body.get("device_secret"),
        "mcp_url": mcp_url,
        "discovery_url": body.get("discovery_url"),
        "allowed_roots": body.get("allowed_roots") or [],
        "last_seen_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        await client.store.put_item(
            list(mcp_registration_namespace(user_id)),
            key=device_id,
            value=record,
        )
        return JSONResponse(
            content={
                "registered": True,
                "device_id": device_id,
                "device_label": device_label,
                "platform": platform,
            },
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
    """Refresh one device's registration so the device keeps counting as online.

    The record is looked up by the body's ``device_id``, so each machine
    heartbeats its own record. Before records were device-keyed, a second machine
    kept the first machine's ``last_seen_at`` fresh, which made an offline
    machine look reachable and hid the machine that really was online.

    Heartbeats also sync ``connection_mode`` / ``device_secret`` / ``mcp_url``
    and re-derive the device label, so a daemon that gains an explicit label in a
    later release adopts the label without needing to re-register.
    """
    from src.anubis.utils.tools.data_analysis.backend import (
        mcp_registration_namespace,
    )
    from src.anubis.utils.tools.data_analysis.devices import (
        deduplicate_label,
        derive_device_identity,
    )

    user_id = current_user["identities"][0]["user_id"]
    token = current_user["API_KEY"]
    client = get_client(headers={"API-KEY": f"{token}"})

    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    device_id = body.get("device_id")
    if not device_id:
        # A heartbeat that does not say which machine it came from cannot be
        # applied to any record without guessing, and guessing is exactly what
        # let one machine refresh another machine's presence.
        return JSONResponse(content={"acknowledged": False}, status_code=200)

    namespace = list(mcp_registration_namespace(user_id))
    try:
        existing = await client.store.get_item(namespace, key=device_id)
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
        record["device_id"] = device_id
        if body.get("connection_mode"):
            record["connection_mode"] = body["connection_mode"]
        if body.get("device_secret"):
            record["device_secret"] = body["device_secret"]
        if body.get("server_name"):
            record["server_name"] = body["server_name"]

        # Re-derive identity from the heartbeat body, falling back to whatever
        # the record already holds, so an updated daemon can start supplying an
        # explicit label mid-session.
        derived_label, platform = derive_device_identity(
            {
                "device_label": body.get("device_label"),
                "platform": body.get("platform") or record.get("platform"),
                "server_name": body.get("server_name") or record.get("server_name"),
            }
        )
        if body.get("device_label"):
            other_records = [
                other
                for other in await _search_namespace_records(
                    client, mcp_registration_namespace(user_id)
                )
                if other.get("device_id") != device_id
            ]
            record["device_label"] = deduplicate_label(
                derived_label, other_records, device_id
            )
        else:
            record["device_label"] = record.get("device_label") or derived_label
        record["platform"] = platform

        connection_mode = record.get("connection_mode") or "relay"
        if connection_mode == "relay":
            record["mcp_url"] = (
                f"{str(request.base_url).rstrip('/')}/mcp/relay/{device_id}"
            )
        elif body.get("mcp_url"):
            record["mcp_url"] = body["mcp_url"]
        await client.store.put_item(namespace, key=device_id, value=record)
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
    """Delete ONE device's registration record (that daemon is shutting down).

    Distinct from ``/disconnect_mcp``, which forgets the *adopted*, avatar-bound
    connection; unregister only removes the presence record.

    The ``device_id`` is REQUIRED, and only that device's record is deleted. This
    is the fix for a recorded production incident: while every machine shared one
    registration record under a constant key, stopping a development daemon
    deleted production's registration, because both daemons wrote and deleted the
    same key in a shared store. Deleting every record when no device is named
    would reproduce exactly that failure, so a body without a ``device_id`` is
    rejected rather than treated as "all".
    """
    from src.anubis.utils.tools.data_analysis.backend import (
        mcp_registration_namespace,
    )

    user_id = current_user["identities"][0]["user_id"]
    token = current_user["API_KEY"]
    client = get_client(headers={"API-KEY": f"{token}"})
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    device_id = body.get("device_id")
    if not device_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "A device_id is required to unregister a Model Context Protocol "
                "server, so that one machine's shutdown never removes another "
                "machine's registration."
            ),
        )

    try:
        await client.store.delete_item(
            list(mcp_registration_namespace(user_id)), key=device_id
        )
        return JSONResponse(
            content={"unregistered": True, "device_id": device_id}, status_code=200
        )
    except Exception as unregister_error:
        raise HTTPException(
            status_code=500,
            detail=f"Error unregistering MCP server: {unregister_error}",
        )


@app.delete("/delete_avatar")
async def delete_avatar(
    assistant_id: str, request: Request, current_user: dict = Depends(get_current_user)
):
    context = app.state.context
    token = current_user["API_KEY"]
    user_id = current_user["identities"][0]["user_id"]
    client = get_client(headers={"API-KEY": f"{token}"})

    # Ownership gate. Without this check any authenticated caller could delete
    # any avatar by passing an arbitrary assistant_id, including a public avatar
    # or one belonging to another user. Mirrors the creator check that
    # /update_avatar_identity_with_media already performs.
    try:
        assistant = await client.assistants.get(assistant_id)
    except Exception as lookup_error:
        raise HTTPException(
            status_code=404, detail=f"Could not load assistant: {lookup_error}"
        ) from lookup_error
    creator_id = (assistant.get("metadata") or {}).get("user_id")
    if not creator_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "Assistant metadata is missing the creator's user_id; "
                "cannot verify deletion permissions."
            ),
        )
    if user_id != creator_id and user_id != context.admin_user_id:
        raise HTTPException(
            status_code=403,
            detail=(
                "Only the creator of this avatar may delete the avatar. "
                "The signed-in user is not the assistant's creator."
            ),
        )

    try:
        deleted_counts = await purge_avatar_data(
            pool=request.app.state.pool,
            langgraph_sdk_client=client,
            assistant_id=assistant_id,
        )
    except Exception as purge_error:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting avatar {assistant_id}: {purge_error}",
        ) from purge_error
    return JSONResponse(
        {"message": "Deleted Avatar Successfully", "deleted": deleted_counts},
        status_code=200,
    )


@app.get("/list_public_avatars")
async def list_public_avatars(assistant_id: Optional[str] = None):
    public_avatars_result = await get_public_avatars(assistant_id=assistant_id)
    return [
        _assistant_without_metadata(assistant) for assistant in public_avatars_result
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
        # Paged: assistants.search defaults to limit=10, which silently hid
        # every avatar past the tenth from the owner's own listing.
        response = await search_all_avatars_for_user(
            client, current_user["identities"][0]["user_id"]
        )
        if len(response) > 0:
            avatar_list = response
            public_avatars_result.extend(avatar_list)  # public and private avatars
        # The caller sees their OWN avatars in full, public or not; only other
        # people's public avatars are stripped.
        sanitized = [
            _assistant_without_metadata_if_public(
                a, viewer_user_id=current_user["identities"][0]["user_id"]
            )
            for a in public_avatars_result
        ]
        return JSONResponse(sanitized, status_code=200)
    except Exception as e:
        error = f"Error in listing avatars: {e}"
        raise HTTPException(detail=error, status_code=500)


""" Geo-located avatars """


async def _public_geo_candidates() -> list[dict[str, Any]]:
    """Every public avatar, as the map and the proximity search see them.

    Only public avatars are placed on a map or announced to a passer-by: a pin
    is an invitation to walk up to an avatar, and an avatar its creator has not
    shared is not inviting anyone. The public listing already strips the
    metadata and keeps the pin, so the entries returned here carry no creator
    identifier.
    """
    return [
        _assistant_without_metadata(assistant)
        for assistant in await get_public_avatars()
    ]


def _resolve_search_radius(requested_radius_meters: Optional[float]) -> float:
    """Resolve the radius to search, defaulted and capped by the configured limits."""
    context = app.state.context
    if requested_radius_meters is None or requested_radius_meters <= 0:
        return float(context.geo_nearby_default_radius_meters)
    return float(
        min(requested_radius_meters, context.geo_nearby_max_radius_meters)
    )


@app.get("/avatars/geo")
async def list_geo_avatars(
    min_latitude: Optional[float] = None,
    min_longitude: Optional[float] = None,
    max_latitude: Optional[float] = None,
    max_longitude: Optional[float] = None,
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    """Every public avatar pinned to a place, optionally inside a map viewport.

    This is what the world globe and the local map draw. Omitting the bounding
    box returns every public pin on Earth; supplying part of the box leaves the
    remaining sides open, and a box whose minimum longitude exceeds the maximum
    wraps across the antimeridian the way a map viewport does.
    """
    avatars = []
    for assistant in await _public_geo_candidates():
        pin = geo_location_of(assistant)
        if pin is None:
            continue
        if not within_bounds(
            pin,
            min_latitude=min_latitude,
            min_longitude=min_longitude,
            max_latitude=max_latitude,
            max_longitude=max_longitude,
        ):
            continue
        avatars.append(
            {
                "assistant_id": assistant.get("assistant_id"),
                "name": assistant.get("name"),
                "description": assistant.get("description"),
                "geo_location": pin,
            }
        )
    return {"avatars": avatars, "count": len(avatars)}


@app.get("/avatars/nearby")
async def list_nearby_avatars(
    latitude: float,
    longitude: float,
    radius_meters: Optional[float] = None,
    accuracy_meters: Optional[float] = None,
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    """Public geo-located avatars around a point, nearest first.

    ``accuracy_meters`` is the accuracy the device reported for the position; a
    reading that is only accurate to eighty meters widens each geofence by that
    much, so a visitor who is genuinely standing at the place is not told they
    are outside because the phone is unsure where the phone is.
    """
    try:
        latitude_value, longitude_value = validate_coordinates(latitude, longitude)
    except GeoLocationError as invalid_location:
        raise HTTPException(
            status_code=400, detail=str(invalid_location)
        ) from invalid_location
    radius = _resolve_search_radius(radius_meters)
    return {
        "latitude": latitude_value,
        "longitude": longitude_value,
        "radius_meters": radius,
        "avatars": avatars_near(
            latitude_value,
            longitude_value,
            await _public_geo_candidates(),
            radius_meters=radius,
            accuracy_meters=accuracy_meters,
        ),
    }


class GeoCheckin(BaseModel):
    """One position report from a device that is watching for nearby avatars."""

    latitude: float
    longitude: float
    accuracy_meters: Optional[float] = None


@app.post("/geo/checkin")
async def geo_checkin(
    checkin: GeoCheckin,
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    """Record that a person is standing near a geo-located avatar.

    The web application posts here as the device moves. The response says which
    avatars are nearby, which geofences the person is inside, which of those the
    application should raise a notification for (at most one notification per
    person per avatar per cooldown window), and which avatar to open in
    camera-backed voice mode.
    """
    try:
        latitude_value, longitude_value = validate_coordinates(
            checkin.latitude, checkin.longitude
        )
    except GeoLocationError as invalid_location:
        raise HTTPException(
            status_code=400, detail=str(invalid_location)
        ) from invalid_location

    visitor_id = current_user["identities"][0]["user_id"]
    radius = _resolve_search_radius(None)
    nearby = avatars_near(
        latitude_value,
        longitude_value,
        await _public_geo_candidates(),
        radius_meters=radius,
        accuracy_meters=checkin.accuracy_meters,
    )
    inside = [entry for entry in nearby if entry["inside_geofence"]]

    notify = [
        entry
        for entry in inside
        if app.state.geo_notify_throttle.allow(visitor_id, entry["assistant_id"])
    ]

    visits_recorded: list[str] = []
    for entry in inside:
        assistant_id = entry["assistant_id"]
        if not app.state.geo_checkin_throttle.allow(visitor_id, assistant_id):
            continue
        # A visit is best effort: the notification and the offer to talk matter
        # to the person standing there, and a failed write must not take those
        # away.
        try:
            visited_at = datetime.now(tz=timezone.utc).isoformat()
            await app.state.store.aput(
                (visitor_id, assistant_id, "geo_visit"),
                key=visited_at,
                value={
                    "value": {
                        "visited_at": visited_at,
                        "latitude": latitude_value,
                        "longitude": longitude_value,
                        "accuracy_meters": checkin.accuracy_meters,
                        "distance_meters": entry["distance_meters"],
                    }
                },
            )
            visits_recorded.append(assistant_id)
        except Exception:  # noqa: BLE001 - a visit record must never fail a check-in
            logger.warning(
                "Could not record a geo visit for avatar %s", assistant_id, exc_info=True
            )

    return {
        "inside_geofence": inside,
        "nearby": nearby,
        "notify": notify,
        "visits_recorded": visits_recorded,
        "open_avatar_id": inside[0]["assistant_id"] if inside else None,
    }


async def label_spoken_turn_files(
    app_state,
    current_user: dict,
    *,
    files: OptionalUploadFiles,
    message: str,
    assistant_id: str,
    thread_id: Optional[str],
    your_name: Optional[str],
    request_id: Optional[str],
) -> tuple[list, str, dict | None, dict | None]:
    """Turn a ``diarize=true`` turn's audio attachment into a speaker-labelled script.

    Returns the remaining (non-audio) files, the message text (the script when
    the turn carried audio), the ``additional_kwargs`` for the human message and
    the ``spoken_turn`` stream frame. A turn without audio passes through
    unchanged. The diarization call is metered as ``api_metrics`` inference type
    ``diarization``.
    """
    from src.anubis.utils.ambient.observations import (
        SOURCE_MICROPHONE,
        build_ambient_additional_kwargs,
    )
    from src.anubis.utils.media_assets import get_media_asset_repository
    from src.anubis.utils.voice.speakers import (
        diarize_spoken_turn,
        speaker_labels_enabled,
    )

    speaker_context = GlobalContext()
    if not speaker_labels_enabled(speaker_context):
        raise HTTPException(
            status_code=404,
            detail="Speaker labelling is not enabled on this deployment.",
        )
    if is_anonymous_user(current_user):
        raise HTTPException(
            status_code=403, detail="Speaker labelling requires a signed-in user."
        )
    attached = list(files or [])
    audio_uploads = []
    remaining = []
    for upload in attached:
        content_type = (
            (getattr(upload, "content_type", None) or "").split(";")[0].strip().lower()
        )
        if content_type.startswith("audio/"):
            audio_uploads.append(upload)
        else:
            remaining.append(upload)
    if not audio_uploads:
        return attached, message, None, None

    owner_label = (your_name or "").strip()
    langgraph_client = get_client(headers={"API-KEY": current_user["API_KEY"]})
    if not owner_label:
        try:
            assistant = await langgraph_client.assistants.get(assistant_id=assistant_id)
            owner_label = str(assistant.get("name") or "").strip()
        except Exception:  # noqa: BLE001 - the label falls back to "Owner"
            owner_label = ""
    recent_avatar_replies = await _recent_avatar_reply_texts(langgraph_client, thread_id)
    user_id = current_user["identities"][0]["user_id"]
    repository = get_media_asset_repository()

    scripts: list[str] = []
    segments: list[dict] = []
    other_speakers: list[str] = []
    owner_spoke = False
    avatar_spoke = False
    owner_identified = False
    duration_seconds = 0.0
    for upload in audio_uploads:
        await upload.seek(0)
        raw = await upload.read()
        if not raw:
            continue
        try:
            spoken = await diarize_spoken_turn(
                raw,
                mime_type=upload.content_type or "audio/webm",
                filename=upload.filename or "utterance.webm",
                context=speaker_context,
                repository=repository,
                user_id=user_id,
                assistant_id=assistant_id,
                thread_id=thread_id,
                owner_label=owner_label,
                recent_avatar_replies=recent_avatar_replies,
            )
        except Exception as diarization_error:  # noqa: BLE001
            logger.exception("Speaker labelling failed")
            raise HTTPException(
                status_code=400,
                detail=f"The recording could not be transcribed: {diarization_error}",
            )
        try:
            await persist_api_metrics_row(
                app_state.pool,
                inference_type="diarization",
                prompt_tokens=int(spoken.usage.get("input_tokens") or 0),
                completion_tokens=int(spoken.usage.get("output_tokens") or 0),
                total_tokens=int(spoken.usage.get("total_tokens") or 0),
                cost_usd=float(spoken.total_cost or 0.0),
                latency_ms=float(spoken.latency_ms or 0.0),
                user_id=user_id,
                assistant_id=assistant_id,
                thread_id=thread_id,
                model_name=spoken.model,
            )
        except Exception:  # noqa: BLE001
            logger.debug("Could not record diarization metrics", exc_info=True)
        if spoken.script:
            scripts.append(spoken.script)
        record = spoken.additional_kwargs()["speakers"]
        segments.extend(record["segments"])
        for label in record["other_speakers"]:
            if label not in other_speakers:
                other_speakers.append(label)
        owner_spoke = owner_spoke or bool(record["owner_spoke"])
        avatar_spoke = avatar_spoke or bool(record.get("avatar_spoke"))
        owner_identified = owner_identified or bool(record["owner_identified"])
        duration_seconds += float(record["duration_seconds"] or 0.0)
        owner_label = record["owner_label"]

    script = "\n".join(scripts).strip()
    typed = (message or "").strip()
    combined = "\n\n".join(part for part in (typed, script) if part)
    speakers_record = {
        "owner_label": owner_label,
        "avatar_label": f"{owner_label} (avatar)",
        "owner_identified": owner_identified,
        "owner_spoke": owner_spoke,
        "avatar_spoke": avatar_spoke,
        "others_spoke": bool(other_speakers),
        "other_speakers": other_speakers,
        "segments": segments,
        "duration_seconds": round(duration_seconds, 3),
    }
    # Only the person talking, alone, is a direct turn. Anything else heard in
    # the room (other people, another avatar, this avatar's own playback, or
    # nothing intelligible) is triaged so the avatar does not answer itself.
    nothing_was_heard = not combined
    if nothing_was_heard:
        combined = "(nothing intelligible was heard)"
    if other_speakers or not owner_spoke:
        additional_kwargs = build_ambient_additional_kwargs(
            sources=[SOURCE_MICROPHONE],
            captured_at=datetime.now(UTC).isoformat(),
            voice_mode=True,
            # A spoken turn is shown so the owner can see what the room said.
            # A turn where nothing intelligible was heard has nothing to show,
            # and showing it put a bare internal header in the transcript as
            # though the owner had typed it.
            hidden=nothing_was_heard,
        )
        additional_kwargs["speakers"] = speakers_record
    else:
        additional_kwargs = {"kind": "spoken_turn", "speakers": speakers_record}
    frame = {"content": combined, "speakers": speakers_record}
    return remaining, combined, additional_kwargs, frame


def refuse_ambient_observation_on_busy_thread(
    turn_registry: ActiveMessageTurnRegistry | None, thread_id: str | None
) -> None:
    """Answer 409 to an observation arriving while the thread has a turn running.

    The browser skips a capture while a typed turn is in flight; this is the
    server's own guard against a client that does not. An observation is
    disposable context, so the observation is refused rather than queued:
    ``Retry-After`` tells the client when to look again, and the client treats
    the refusal as pacing, not as a failure.
    """
    if turn_registry is None or not thread_id:
        return
    if turn_registry.has_turn_on_thread(thread_id):
        raise HTTPException(
            status_code=409,
            detail=(
                "Another turn is in flight on this conversation; the ambient "
                "observation was skipped."
            ),
            headers={"Retry-After": str(AMBIENT_BUSY_RETRY_AFTER_SECONDS)},
        )


async def _recent_avatar_reply_texts(
    langgraph_client, thread_id: Optional[str], limit: int = 6
) -> list[str]:
    """The avatar's last spoken replies in a thread, for echo detection."""
    if not thread_id:
        return []
    try:
        state = await langgraph_client.threads.get_state(thread_id=thread_id)
    except Exception:  # noqa: BLE001 - echo detection is best-effort
        return []
    values = state.get("values") if isinstance(state, dict) else None
    messages = (values or {}).get("messages") or []
    texts: list[str] = []
    for message in reversed(list(messages)):
        message_type = (
            message.get("type") if isinstance(message, dict) else getattr(message, "type", None)
        )
        if message_type != "ai":
            continue
        content = (
            message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
        )
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
        text = str(content or "").strip()
        if text:
            texts.append(text)
        if len(texts) >= limit:
            break
    return texts


def enforce_ambient_request(
    current_user: dict,
    *,
    files: OptionalUploadFiles,
    image_filenames: list,
    sources: Optional[str],
    captured_at: Optional[str],
    voice_mode: bool,
    thread_id: Optional[str],
) -> dict:
    """Gate an ``ambient=true`` turn and build the hidden message's tag.

    Ambient observations are webcam / screen snapshots the browser sends on a
    timer, so the checks here are the ones a typed turn does not need: the
    deployment switch, a signed-in caller (an anonymous visitor on a shared
    avatar never watches through the owner's allotment), at least one attached
    file, a per-image size ceiling, and a per-thread minimum interval that a
    misconfigured or hostile client cannot beat (429 with ``Retry-After``).
    Returns the ``additional_kwargs`` that mark the ``HumanMessage`` hidden and
    carry the observation record the graph's triage node completes.
    """
    from src.anubis.utils.ambient.observations import (
        ambient_throttle,
        build_ambient_additional_kwargs,
        resolve_sources,
    )

    ambient_context = GlobalContext()
    if str(ambient_context.ambient_capture_enabled or "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        raise HTTPException(
            status_code=404, detail="Ambient vision is not enabled on this deployment."
        )
    if is_anonymous_user(current_user):
        raise HTTPException(
            status_code=403, detail="Ambient vision requires a signed-in user."
        )
    attached = list(files or [])
    if not attached:
        raise HTTPException(
            status_code=422,
            detail="An ambient observation needs at least one attached snapshot.",
        )
    max_bytes = int(ambient_context.ambient_capture_max_image_bytes or 0)
    for attached_file in attached:
        size = getattr(attached_file, "size", None)
        if max_bytes and size and int(size) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Snapshot {attached_file.filename or ''} is larger than the "
                    f"{max_bytes} byte ceiling for ambient observations."
                ),
            )
    seconds_to_wait = ambient_throttle.check_and_mark(
        thread_id, float(ambient_context.ambient_capture_min_interval_seconds or 0)
    )
    if seconds_to_wait is not None:
        raise HTTPException(
            status_code=429,
            detail=(
                "Ambient observations on this conversation are limited to one every "
                f"{ambient_context.ambient_capture_min_interval_seconds:g} seconds."
            ),
            headers={"Retry-After": str(int(seconds_to_wait) + 1)},
        )
    resolved_sources = resolve_sources(
        [attached_file.filename or "" for attached_file in attached], sources
    )
    return build_ambient_additional_kwargs(
        sources=resolved_sources,
        captured_at=captured_at,
        voice_mode=voice_mode,
        image_filenames=list(image_filenames or []),
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
            # The bytes decide what the attachment is, not the client's
            # declaration: a WebP picture arrives declared
            # ``application/octet-stream`` from any client whose extension
            # table lacks ``.webp``, and routing that by the declaration would
            # drop the picture into the "other file types" line below, leaving
            # the avatar to answer from the filename alone.
            content_type = effective_upload_mime_type(file.content_type or "", content)

            if content_type.startswith("image/"):
                # The picture is prepared the same way an identity upload is:
                # the bytes are checked against the still-image types, and a
                # HEIC photo, an AVIF export, a TIFF scan or a BMP is
                # transcoded here so the model receives a format it reads. A
                # picture that cannot be prepared is described in words rather
                # than failing the whole turn, so the avatar can say what
                # happened instead of the person's message disappearing.
                try:
                    content_type, content = prepare_still_image_upload(
                        content_type, content
                    )
                except HTTPException as unreadable_image:
                    text_contents.append(
                        f"[Image: {filename} - could not be read: "
                        f"{unreadable_image.detail}]"
                    )
                    continue
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


async def _remember_turn_attachments_for_identity_tool(
    thread_id: str, files: list[UploadFile] | None, current_user: dict
) -> None:
    """Re-read this turn's uploads and record them for the identity-update tool."""
    from src.api.chat_attachments import TurnAttachment, remember_turn_attachments

    attachments: list[TurnAttachment] = []
    for upload in files or []:
        if upload is None or not (getattr(upload, "filename", None) or "").strip():
            continue
        try:
            await upload.seek(0)
            content = await upload.read()
        except Exception:  # noqa: BLE001 - a file that cannot be re-read is skipped
            logger.debug("Could not re-read %r for the identity tool", upload.filename)
            continue
        if not content:
            continue
        attachments.append(
            TurnAttachment(
                filename=upload.filename,
                # Resolved against the bytes for the same reason the message
                # content is: the identity tool tells the model what it may
                # learn from, and a WebP picture declared
                # ``application/octet-stream`` must be offered as the image
                # it is.
                mime_type=effective_upload_mime_type(
                    upload.content_type or "application/octet-stream", content
                )
                or "application/octet-stream",
                content=content,
            )
        )
    remember_turn_attachments(thread_id, attachments, current_user)


async def _connection_acknowledgement_turn(
    user_id: str, connection_key: Optional[str]
) -> tuple[str, dict]:
    """Build the hidden acknowledgement turn for a connector the "+" menu opened."""
    from src.anubis.utils.connected_accounts import get_connected_account, get_provider
    from src.anubis.utils.connected_accounts.connection_cards import (
        ACKNOWLEDGEMENT_MESSAGE_KIND,
        CARD_STATUS_CONNECTED,
        acknowledgement_instruction,
        connection_card_record,
    )
    from src.anubis.utils.connected_accounts.listing import split_connection_key

    key = str(connection_key or "").strip()
    if key.startswith("account:"):
        _source, key = split_connection_key(key)
    if not key:
        raise HTTPException(status_code=400, detail="connection_key is required.")
    record = await get_connected_account(getattr(app.state, "store", None), user_id, key)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No connected account {key!r}.")
    provider = get_provider(str(record.get("provider") or ""))
    card = connection_card_record(provider, record, status=CARD_STATUS_CONNECTED)
    return acknowledgement_instruction(card), {
        "hidden": True,
        "kind": ACKNOWLEDGEMENT_MESSAGE_KIND,
        "connection": card,
    }


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
    ambient: bool = Form(False),
    sources: Optional[str] = Form(None),
    captured_at: Optional[str] = Form(None),
    voice_mode: bool = Form(False),
    diarize: bool = Form(False),
    at_place: bool = Form(False),
    ambient_action_observation_id: Optional[str] = Form(None),
    ambient_action: Optional[str] = Form(None),
    ambient_action_description: Optional[str] = Form(None),
    ambient_action_kind: Optional[str] = Form(None),
    ambient_action_summary: Optional[str] = Form(None),
    turn_kind: Optional[str] = Form(None),
    connection_key: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    # ``ambient_action_observation_id`` names a notification card whose offer
    # the conversation partner just allowed: the turn is a hidden instruction
    # to carry out ``ambient_action_description`` (``ambient_action`` is the
    # offer kind, ``reply`` or ``act``); the avatar's reply is stamped with the
    # observation so a thumb on the reply is learned against the card.
    # ``diarize=true`` marks the attached audio as one live-voice utterance to
    # be transcribed WITH speaker labels (the owner recognised from the voice
    # recordings, other people as "Speaker N" remembered per thread). The
    # script becomes the human turn's text. When someone other than the owner
    # spoke, the turn is triaged like an ambient observation heard through the
    # microphone (ignore / respond / notify) while staying visible in the
    # transcript; an owner-only utterance goes straight to the avatar.
    # ``ambient=true`` marks this turn as an ambient observation: the attached
    # files are webcam / screen snapshots (``sources`` names each file's origin,
    # aligned with ``files``; ``captured_at`` is the browser's timestamp;
    # ``voice_mode`` says whether the conversation partner is in voice mode).
    # The turn is stored hidden in the thread, described into text, and
    # triaged by the graph as ignore / respond / notify. A future microphone
    # clip rides the same call as another source, so the avatar always receives
    # one message per capture tick. Signed-in callers only; the API enforces a
    # minimum interval per thread.
    # ``like`` / ``dislike`` rate the avatar's previous reply on ``thread_id``;
    # ``feedback`` marks this message as feedback about the avatar (stored and
    # used from the very next reply). Both are collected by the continuous
    # learning pipeline (src/anubis/utils/learning/) and folded into learned
    # preferences by the background sweep once the account has gone idle.

    # allow for select avatar in query and anonymous user for a dedicated endpoint

    logger.warning(f"stream:{stream}")
    start_time = time_ns()
    config = current_user.get("app_metadata", {}).get("assistant_config", {})
    if not config:
        # This endpoint identifies the avatar by the URL path parameter. An
        # authenticated (api-key) caller rebuilds the full configurable from the
        # path ``assistant_id`` below (name/description/metadata fetched fresh),
        # so an ``assistant_config`` on the account is not required. Only the
        # anonymous branch depends on the dependency-populated
        # ``assistant_config``, which ``get_anonymous_user_with_anonymous_api_key``
        # builds in memory from the same path ``assistant_id`` after confirming
        # the avatar is public (that is where the public-avatar gate lives, so
        # the anonymous branch must not fall back to an empty configurable);
        # start from an empty configurable that the api-key branch fills in.
        if not is_anonymous_user(current_user):
            config = {"configurable": {}}
        else:
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
    spoken_turn_additional_kwargs: dict | None = None
    spoken_turn_frame: dict | None = None
    if diarize:
        (
            files,
            message,
            spoken_turn_additional_kwargs,
            spoken_turn_frame,
        ) = await label_spoken_turn_files(
            request.app.state,
            current_user,
            files=files,
            message=message,
            assistant_id=assistant_id,
            thread_id=thread_id,
            your_name=your_name,
            request_id=request.state.request_id,
        )
    (
        file_text_content,
        multimodal_content,
        image_filenames,
    ) = await process_files_for_message(files, message=message)
    ambient_additional_kwargs: dict | None = None
    if spoken_turn_additional_kwargs and "ambient" in spoken_turn_additional_kwargs:
        # Others spoke: the spoken turn is triaged like an observation heard
        # through the microphone. The tag already carries ``speakers``.
        ambient_additional_kwargs = spoken_turn_additional_kwargs
    if ambient:
        ambient_additional_kwargs = enforce_ambient_request(
            current_user,
            files=files,
            image_filenames=image_filenames,
            sources=sources,
            captured_at=captured_at,
            voice_mode=voice_mode,
            thread_id=thread_id,
        )
        refuse_ambient_observation_on_busy_thread(
            getattr(request.app.state, "active_message_turns", None), thread_id
        )
    ambient_action_additional_kwargs: dict | None = None
    if (ambient_action_observation_id or "").strip():
        from src.anubis.utils.ambient.observations import (
            build_ambient_action_additional_kwargs,
        )

        if is_anonymous_user(current_user):
            raise HTTPException(
                status_code=401,
                detail="Only a signed-in conversation partner can allow an action.",
            )
        if not (ambient_action_description or "").strip():
            raise HTTPException(
                status_code=400,
                detail="ambient_action_description names what the avatar may do.",
            )
        ambient_action_additional_kwargs = build_ambient_action_additional_kwargs(
            observation_id=ambient_action_observation_id.strip(),
            observation_kind=ambient_action_kind or "other",
            action=ambient_action or "reply",
            action_description=ambient_action_description,
            summary=ambient_action_summary or "",
        )
    # ``turn_kind=connection_acknowledgement`` is the hidden turn the "+" menu
    # sends once a connector it opened finishes signing in: the avatar
    # acknowledges the connection in the transcript and the card persists on
    # that reply. The text is server-authored from the stored record.
    connection_acknowledgement_additional_kwargs: dict | None = None
    connection_acknowledgement_text: str | None = None
    if (turn_kind or "").strip() == "connection_acknowledgement":
        if is_anonymous_user(current_user):
            raise HTTPException(
                status_code=401, detail="Only a signed-in owner can connect accounts."
            )
        (
            connection_acknowledgement_text,
            connection_acknowledgement_additional_kwargs,
        ) = await _connection_acknowledgement_turn(
            current_user["identities"][0]["user_id"], connection_key
        )

    user_name = your_name
    user_description = your_description
    user_id = current_user["identities"][0]["user_id"]
    assistant_id = assistant_id.strip()
    if not is_anonymous_user(current_user):
        langgraph_client_headers = {"API-KEY": current_user["API_KEY"]}
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
    # This is the endpoint anonymous visitors message a public avatar through,
    # so passing the path avatar into enforcement is what lets an avatar listed
    # in UNRESTRICTED_ANONYMOUS_MESSAGING_AVATAR_IDENTIFIERS keep answering an
    # anonymous visitor past the free-tier allotment and past the token rate
    # cap. Both turns are still metered; only the refusals are lifted.
    await enforce_remaining_allotment(
        request.app.state,
        current_user,
        message_meter,
        estimated_request_tokens=estimated_request_tokens.input_tokens,
        assistant_id=assistant_id,
    )
    message_rate_limit_context = GlobalContext()
    await enforce_token_rate_limit(
        request.app.state,
        current_user,
        meter_event_names=[
            UsageMeter.MESSAGING_TOKENS.value,
            UsageMeter.ADAPTER_INFERENCE_TOKENS.value,
        ],
        window_seconds=int(
            message_rate_limit_context.message_rate_limit_window_seconds or 0
        ),
        tokens_per_window=int(
            message_rate_limit_context.message_rate_limit_tokens_per_window or 0
        ),
        estimated_request_tokens=estimated_request_tokens.total_tokens,
        assistant_id=assistant_id,
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

    # Keep this turn's raw files for the in-chat identity-update tool. The
    # content built above is what the model reads; the tool needs the bytes as
    # uploaded (the media graph classifies and converts them itself). The graph
    # offers the tool to the avatar's creator only, and the tool's starter
    # re-checks ownership, tier, and allotment before anything is processed.
    if not is_anonymous_user(current_user):
        await _remember_turn_attachments_for_identity_tool(
            thread_id, files, current_user
        )
    # client-supplied IANA timezone (e.g. "America/New_York") used to localize system_time
    config["configurable"]["user_timezone"] = user_timezone
    # ``at_place=true`` says the person sending this message is standing at the
    # real-world place the avatar is pinned to, watching that place through a
    # camera. This changes only how the avatar speaks (the YOUR PLACE section of
    # the system prompt) — anyone may talk to a geo-located avatar from anywhere,
    # so this is never a permission check.
    config["configurable"]["visitor_present_at_place"] = bool(at_place)
    config["configurable"]["include_quality_metrics"] = include_quality_metrics
    config["configurable"]["use_adapter_inference"] = resolve_use_adapter_inference(
        current_user, adapter
    )

    # store = app.state.store
    graph = app.state.graph

    # Inline continuous-learning signals carried on the message itself. Best
    # effort: a failed store write must never cost the user the reply.
    if (like or dislike or feedback) and thread_id:
        try:
            await _apply_inline_message_feedback(
                request.app.state,
                langgraph_client_headers=langgraph_client_headers,
                user_id=user_id,
                assistant_id=assistant_id,
                thread_id=thread_id,
                like=like,
                dislike=dislike,
                feedback_text=message if feedback else None,
            )
        except Exception as inline_feedback_error:  # noqa: BLE001
            logger.warning("Inline message feedback failed: %s", inline_feedback_error)

    # Uploaded files were already processed above (before enforcement) so the
    # pre-call estimate could cover them; reuse those results here.

    # Create the human message content
    if multimodal_content:
        human_message = HumanMessage(
            id=str(uuid4()),
            content=multimodal_content,
            additional_kwargs=ambient_additional_kwargs
            or {"image_filenames": image_filenames},
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
        if ambient_additional_kwargs is not None:
            # An ambient observation without an image (for example a
            # microphone-only capture): keep the observation header so the
            # triage node still recognises the turn.
            from src.anubis.utils.ambient.observations import observation_header

            human_message_content = (
                observation_header(ambient_additional_kwargs["ambient"])
                + "\n"
                + (human_message_content or "")
            )
        if ambient_action_additional_kwargs is not None:
            from src.anubis.utils.ambient.observations import (
                compose_ambient_action_text,
            )

            # The allowed action replaces whatever text rode along: the turn
            # is the instruction, hidden like an observation.
            human_message_content = compose_ambient_action_text(
                ambient_action_additional_kwargs["ambient"]
            )
        if connection_acknowledgement_additional_kwargs is not None:
            human_message_content = connection_acknowledgement_text or ""
        human_message_additional_kwargs = (
            connection_acknowledgement_additional_kwargs
            or ambient_action_additional_kwargs
            or ambient_additional_kwargs
            or spoken_turn_additional_kwargs
            or {}
        )
        if (
            ambient_additional_kwargs is None
            and spoken_turn_additional_kwargs is None
            and ambient_action_additional_kwargs is None
            and connection_acknowledgement_additional_kwargs is None
        ):
            from src.anubis.utils.client_harvest_turns import (
                CLIENT_HARVEST_MESSAGE_KIND,
                is_client_harvest_marker_text,
            )

            # A browser harvest (follow-up chips, a profile description) is
            # machine traffic: hidden from the transcript like an ambient
            # observation, and dropped from later turns' model input.
            if is_client_harvest_marker_text(human_message_content):
                human_message_additional_kwargs = {
                    "hidden": True,
                    "kind": CLIENT_HARVEST_MESSAGE_KIND,
                }
        human_message = HumanMessage(
            id=str(uuid4()),
            content=human_message_content,
            additional_kwargs=human_message_additional_kwargs,
        )

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
                spoken_turn_frame=spoken_turn_frame,
                ambient=ambient,
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
    await _write_thread_metadata(
        langgraph_client,
        thread_id,
        _thread_metadata_updates(
            user_id=user_id,
            assistant_id=assistant_id,
            thread_id=thread_id,
            conversation_title_value=conversation_title_data,
        ),
    )

    # A conversation this turn just started has no name. Naming it is a
    # classification call the caller is not waiting on, so it runs after the
    # reply has been assembled and its result reaches the browser on the next
    # listing of conversations rather than in this response.
    schedule_background(
        _name_new_conversation(
            thread_id,
            langgraph_client_headers=langgraph_client_headers,
            user_id=user_id,
            assistant_id=assistant_id,
        )
    )

    response_data = {}
    response_data["content"] = result["messages"][-1].content
    response_metadata = result["messages"][-1].response_metadata
    if response_metadata:
        response_data["response_metadata"] = response_metadata

    response_data["total_response_time_ms"] = (time_ns() - start_time) // 1000000
    response_data["thread_id"] = thread_id
    response_data["request_id"] = request.state.request_id
    # The id the reply is stored under, so a client can rate exactly this
    # reply through POST /message_feedback.
    response_data["message_id"] = getattr(result["messages"][-1], "id", None)
    if spoken_turn_frame:
        response_data["spoken_turn"] = spoken_turn_frame
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


@app.post("/message/{assistant_id}/stop")
async def stop_avatar_message(
    request: Request,
    assistant_id: str,
    request_id: Optional[str] = Form(None),
    thread_id: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    """End a reply that is still being generated.

    The stream announced its ``request_id`` in its first frame
    (``turn_started``); pass that, or the ``thread_id`` when the first frame
    was never seen. The running stream cancels the model call, records what
    the avatar had said so far on the thread, and ends with a ``done`` frame
    flagged ``stopped``. Only the caller who started the turn may stop it.

    404 means no reply is being generated for that id in this process — the
    turn already finished, or another API process is streaming it. A client
    that gets 404 should abort its own stream; the streaming process treats
    the disconnect exactly like a stop.
    """
    if not (request_id or thread_id):
        raise HTTPException(
            status_code=422, detail="Pass request_id or thread_id to stop a reply."
        )
    turn_registry = getattr(request.app.state, "active_message_turns", None)
    active_turn = (
        turn_registry.find(request_id=request_id, thread_id=thread_id)
        if turn_registry is not None
        else None
    )
    caller_user_id = current_user["identities"][0]["user_id"]
    if (
        active_turn is None
        or active_turn.user_id != caller_user_id
        or (active_turn.assistant_id or "") != assistant_id.strip()
    ):
        raise HTTPException(
            status_code=404, detail="No reply is being generated for that request."
        )
    active_turn.request_stop()
    return JSONResponse(
        {
            "status": "stopping",
            "request_id": active_turn.request_id,
            "thread_id": active_turn.thread_id,
        }
    )


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
        # Same rule as ``POST /message/{assistant_id}``: this endpoint identifies
        # the avatar by the URL path parameter. An authenticated (api-key) caller
        # rebuilds the full configurable from the path ``assistant_id`` below
        # (name/description/metadata fetched fresh), so an ``assistant_config`` on
        # the account is not required — a client that messaged an avatar by id
        # must be able to resume the run that message paused, otherwise the
        # approve/edit/reject panel renders but can never be acted on. Only the
        # anonymous branch depends on the dependency-populated
        # ``assistant_config``, which carries the public-avatar gate applied when
        # the anonymous visitor was resolved; start from an empty configurable
        # that the api-key branch fills in.
        if not is_anonymous_user(current_user):
            config = {"configurable": {}}
        else:
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
    # The resumed continuation is aimed at the same avatar as the message that
    # paused, so the demonstration-avatar exemption applies here too: an
    # anonymous visitor whose turn interrupted for approval must be able to
    # complete that turn, not be refused halfway through. ``assistant_id`` is
    # trimmed the same way the lookup below trims the path parameter.
    await enforce_remaining_allotment(
        request.app.state,
        current_user,
        UsageMeter.MESSAGING_TOKENS,
        estimated_request_tokens=estimated_request_tokens.input_tokens,
        assistant_id=assistant_id.strip(),
    )
    await enforce_token_rate_limit(
        request.app.state,
        current_user,
        meter_event_names=[
            UsageMeter.MESSAGING_TOKENS.value,
            UsageMeter.ADAPTER_INFERENCE_TOKENS.value,
        ],
        window_seconds=int(
            message_rate_limit_context.message_rate_limit_window_seconds or 0
        ),
        tokens_per_window=int(
            message_rate_limit_context.message_rate_limit_tokens_per_window or 0
        ),
        estimated_request_tokens=estimated_request_tokens.total_tokens,
        assistant_id=assistant_id.strip(),
    )

    user_id = current_user["identities"][0]["user_id"]
    # Trim the path parameter the same way ``POST /message/{assistant_id}`` does, so a
    # stray trailing space in the id surfaces as a miss on our side rather than a 500
    # from the assistant lookup.
    assistant_id = assistant_id.strip()
    if not is_anonymous_user(current_user):
        langgraph_client_headers = {"API-KEY": current_user["API_KEY"]}
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


CONVERSATION_LISTING_DEFAULT_LIMIT = 100
CONVERSATION_LISTING_MAXIMUM_LIMIT = 1000


@app.get("/conversations")
async def get_all_conversations(
    request: Request,
    assistant_id: str,
    limit: int = CONVERSATION_LISTING_DEFAULT_LIMIT,
    offset: int = 0,
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    """Return this user + assistant's threads, newest-first.

    ``limit`` and ``offset`` are passed to the LangGraph software development
    kit explicitly rather than left to their defaults. ``threads.search``
    defaults to ``limit=10``, and omitting the argument silently truncated every
    caller's history to the ten most recent conversations: an account holding
    forty-nine threads with one avatar was handed ten of them and shown no sign
    that the other thirty-nine existed. A listing that quietly discards most of
    its rows is worse than one that refuses, because the client cannot tell the
    difference between "you have ten conversations" and "you were given ten".

    The ceiling keeps one request from asking the database for an unbounded
    page; a caller with more threads than the ceiling pages through them with
    ``offset``.
    """
    user_id = current_user["identities"][0]["user_id"]
    langgraph_client_headers = {"API-KEY": current_user["API_KEY"]}
    requested_limit = max(1, min(limit, CONVERSATION_LISTING_MAXIMUM_LIMIT))
    requested_offset = max(0, offset)
    try:
        langgraph_client = get_client(headers=langgraph_client_headers)
        threads = await langgraph_client.threads.search(
            metadata={
                "thread_metadata": {"user_id": user_id, "assistant_id": assistant_id}
            },
            limit=requested_limit,
            offset=requested_offset,
            sort_by="updated_at",
            sort_order="desc",
        )
        return JSONResponse(threads)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error loading threads: {exc}")


def _message_without_observation_header(message: Any) -> Any:
    """Drop an observation's internal header line from a message bound for a transcript.

    Observations are written with an ``[AMBIENT_OBSERVATION id=... ]`` first
    line so the avatar can tell a scene it noticed from a turn the person
    typed. That line is addressed to the model. A visible spoken turn was
    handing it to the reader as well, who saw the raw header attributed to
    themselves.
    """
    from src.anubis.utils.ambient.observations import (
        is_ambient_observation,
        split_observation_text,
    )

    if not isinstance(message, dict) or not is_ambient_observation(message):
        return message
    content = message.get("content")
    if not isinstance(content, str):
        return message
    header, body = split_observation_text(content)
    if header is None:
        return message
    return {**message, "content": body}


async def _run_scheduled_report_question(schedule: dict) -> dict:
    """Run one scheduled report's question through the graph on a fresh thread.

    The ``run_question`` callable the scheduler is wired with: the config has
    the same shape as a ``/message`` turn for the owner and the personal
    avatar, plus ``scheduled=True`` so no tool pauses on a sign-in card. The
    reply text and the thread id are returned; the scheduler then finds the
    report the avatar saved for that thread and creates the inbox item.
    """
    from langchain_core.messages import HumanMessage

    user_id = str(schedule.get("user_id") or "")
    assistant_id = str(schedule.get("assistant_id") or "")
    question = str(schedule.get("question") or "")
    thread_id = str(uuid4())
    assistant_name = ""
    assistant_metadata: dict = {}
    try:
        async with app.state.pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT name, metadata FROM assistant WHERE assistant_id = %s",
                (assistant_id,),
            )
            row = await cursor.fetchone()
        if row:
            assistant_name = str(row[0] or "")
            assistant_metadata = dict(row[1] or {}) if isinstance(row[1], dict) else {}
    except Exception:
        logger.debug("Could not read the assistant row for a scheduled report", exc_info=True)
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            "assistant_id": assistant_id,
            "assistant_ctx": {
                "assistant_id": assistant_id,
                "name": assistant_name,
                "metadata": assistant_metadata,
            },
            "user_name": None,
            "user_description": None,
            "scheduled": True,
            "include_quality_metrics": False,
            "use_adapter_inference": False,
        },
        "run_id": str(uuid4()),
    }
    instruction = (
        f"{question}\n\nThis is a scheduled report titled "
        f"{schedule.get('title') or schedule.get('kind') or 'report'!r}. Use the connected "
        "accounts and tools, chart every time series with make_chart, and finish by "
        "calling save_report with kind "
        f"{schedule.get('kind') or 'custom'!r} and a summary the owner can read in the inbox."
    )
    human_message = HumanMessage(
        id=str(uuid4()),
        content=instruction,
        additional_kwargs={"hidden": True, "kind": "scheduled_report", "schedule_id": str(schedule.get("schedule_id") or "")},
    )
    result = await app.state.graph.ainvoke(
        {"messages": [human_message]}, config=config, context=app.state.context
    )
    reply_text = ""
    try:
        messages = (result or {}).get("messages") or []
        last = messages[-1] if messages else None
        content = getattr(last, "content", "") if last is not None else ""
        reply_text = content if isinstance(content, str) else str(content)
    except Exception:
        reply_text = ""
    return {"thread_id": thread_id, "reply_text": reply_text}


def _report_repository_or_503():
    try:
        from src.anubis.utils.analytics.reports import get_report_repository
    except ImportError:
        raise HTTPException(status_code=503, detail="Reports are not available on this server.")
    repository = get_report_repository()
    if repository is None:
        raise HTTPException(status_code=503, detail="Reports are not available on this server.")
    return repository


@app.get("/reports")
async def list_reports_route(
    request: Request,
    assistant_id: Optional[str] = None,
    q: Optional[str] = None,
    kind: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
):
    """List the owner's saved reports, newest first, searchable and filterable.

    ``q`` searches the title and summary; ``kind`` filters by report kind;
    ``since`` / ``until`` bound the creation date (ISO). Every analysis the
    avatar saves with save_report, and every scheduled digest, appears here.
    """
    from src.anubis.utils.analytics.reports import public_report_view

    repository = _report_repository_or_503()
    user_id = current_user["identities"][0]["user_id"]
    rows = await repository.list(
        user_id,
        assistant_id=assistant_id or None,
        query=(q or "").strip() or None,
        kind=(kind or "").strip() or None,
        since=since or None,
        until=until or None,
        limit=max(1, min(int(limit or 20), 100)),
        offset=max(0, int(offset or 0)),
    )
    kinds = []
    if assistant_id:
        try:
            kinds = await repository.kinds(user_id, assistant_id)
        except Exception:
            kinds = []
    return JSONResponse(
        content={"reports": [public_report_view(row) for row in rows], "kinds": kinds, "limit": limit, "offset": offset},
        status_code=200,
    )


@app.get("/reports/{report_id}")
async def get_report_route(
    request: Request,
    report_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return one saved report (charts inline)."""
    from src.anubis.utils.analytics.reports import public_report_view

    repository = _report_repository_or_503()
    user_id = current_user["identities"][0]["user_id"]
    row = await repository.get(user_id, report_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No such report.")
    return JSONResponse(content={"report": public_report_view(row)}, status_code=200)


@app.delete("/reports/{report_id}")
async def delete_report_route(
    request: Request,
    report_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete one saved report of the owner."""
    repository = _report_repository_or_503()
    user_id = current_user["identities"][0]["user_id"]
    deleted = await repository.delete(user_id, report_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="No such report.")
    return JSONResponse(content={"deleted": True, "report_id": report_id}, status_code=200)


@app.get("/report_schedules")
async def list_report_schedules_route(
    request: Request,
    assistant_id: str,
    current_user: dict = Depends(get_current_user),
):
    """List the scheduled reports of one avatar."""
    try:
        from src.anubis.utils.analytics.schedules import get_schedule_repository
    except ImportError:
        raise HTTPException(status_code=503, detail="Schedules are not available on this server.")
    repository = get_schedule_repository()
    if repository is None:
        raise HTTPException(status_code=503, detail="Schedules are not available on this server.")
    user_id = current_user["identities"][0]["user_id"]
    rows = await repository.list_for_avatar(user_id, assistant_id)
    return JSONResponse(content={"schedules": [dict(row, next_run_at=str(row.get("next_run_at")), last_run_at=str(row.get("last_run_at")) if row.get("last_run_at") else None, created_at=str(row.get("created_at"))) for row in rows]}, status_code=200)


async def _pending_interrupt_for_thread(thread_id: str) -> Optional[dict]:
    """The interrupt a thread is paused on, so a reload re-arms the card.

    Read from the graph's checkpoint; ``None`` when the thread is not paused
    or the state cannot be read (a reload must never fail over this).
    """
    graph_instance = getattr(app.state, "graph", None)
    if graph_instance is None:
        return None
    try:
        from src.anubis.utils.graph_interrupts import collect_pending_interrupts

        snapshot = await graph_instance.aget_state({"configurable": {"thread_id": thread_id}})
        pending = collect_pending_interrupts(snapshot)
    except Exception:
        return None
    if not pending:
        return None
    value = getattr(pending[0], "value", None)
    if not isinstance(value, dict):
        return None
    return {"thread_id": thread_id, "interrupt": value}


@app.get("/conversations/{thread_id}/messages")
async def get_thread_messages(
    request: Request,
    thread_id: str,
    assistant_id: str,
    include_hidden: bool = False,
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    """Return the message history for a single thread.

    Hidden turns — ambient webcam/screen observations the conversation partner
    never typed — are dropped unless ``include_hidden=true``; the avatar's
    replies to them stay in the transcript.

    ``assistant_id`` is not decoration: the thread is verified to belong to that
    avatar AND to this caller before any message is returned. Without the check
    this endpoint would hand over any thread whose id the caller could name,
    which is how a client bug once served one avatar's transcript under another
    avatar's chat window — silently, because the mismatched id was accepted.
    """
    user_id = current_user["identities"][0]["user_id"]
    langgraph_client_headers = {"API-KEY": current_user["API_KEY"]}
    try:
        langgraph_client = get_client(headers=langgraph_client_headers)
        thread = await langgraph_client.threads.get(thread_id=thread_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"No such conversation: {exc}")

    # Threads are stamped by /message with the avatar and user they belong to
    # (the same shape /conversations searches on). A thread predating that
    # stamping carries neither, and is left alone rather than made unreadable.
    thread_metadata = (thread.get("metadata") or {}).get("thread_metadata") or {}
    owning_assistant_id = thread_metadata.get("assistant_id")
    owning_user_id = thread_metadata.get("user_id")
    if owning_assistant_id is not None and owning_assistant_id != assistant_id:
        raise HTTPException(
            status_code=404,
            detail="That conversation does not belong to this avatar.",
        )
    if owning_user_id is not None and owning_user_id != user_id:
        raise HTTPException(
            status_code=404,
            detail="That conversation does not belong to this user.",
        )

    try:
        state = await langgraph_client.threads.get_state(thread_id=thread_id)
        messages = state.get("values", {}).get("messages", []) if state else []
        if not include_hidden:
            from src.anubis.utils.ambient.observations import is_hidden_message
            from src.anubis.utils.client_harvest_turns import (
                without_stale_client_harvest_turns,
            )

            # Harvest turns from earlier browser builds were stored without
            # the hidden flag; drop them and their JSON replies here too.
            messages = without_stale_client_harvest_turns(
                [message for message in messages if not is_hidden_message(message)]
            )
            # A spoken turn that IS shown carries the same
            # ``[AMBIENT_OBSERVATION ...]`` header the avatar reads, which is
            # framing for the model and not something the owner typed. The
            # transcript shows what the room said, not the framing.
            messages = [_message_without_observation_header(m) for m in messages]
        messages = await _with_stored_message_feedback(
            messages, user_id=user_id, assistant_id=assistant_id, thread_id=thread_id
        )
        pending_interrupt = await _pending_interrupt_for_thread(thread_id)
        return JSONResponse({"messages": messages, "pending_interrupt": pending_interrupt})
    except Exception as exc:
        # With the traceback the reader learns which frame failed; without it the
        # browser's "Error loading messages" is the only evidence there is, and a
        # failure raised inside the platform's own state read is invisible here.
        logger.exception("Could not load the messages of thread %s", thread_id)
        raise HTTPException(status_code=500, detail=f"Error loading messages: {exc}")


@app.post("/conversations/{thread_id}/title")
async def name_conversation_route(
    request: Request,
    thread_id: str,
    assistant_id: str,
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    """Name a conversation from its whole transcript.

    Called by the browser when the reader leaves a conversation — switching to
    another conversation, starting a new one, or closing the page. A
    conversation is named once when it starts, from an opening exchange that is
    all there is to read at that point; by the time the reader leaves, the
    conversation has usually moved on to the subject it was actually about, and
    this is where that subject becomes the name.

    A name the reader typed in the sidebar is never replaced. The response
    carries the name the conversation now has, and ``"renamed"`` says whether
    this request is what wrote it, so a browser that fired this request while
    navigating away can ignore the answer without wondering what it missed.
    """
    user_id = current_user["identities"][0]["user_id"]
    langgraph_client_headers = {"API-KEY": current_user["API_KEY"]}
    try:
        conversation_name = await name_conversation_thread(
            thread_id,
            langgraph_client_headers=langgraph_client_headers,
            only_when_unnamed=False,
            user_id=user_id,
            assistant_id=assistant_id,
        )
    except Exception:  # noqa: BLE001 - a name is a convenience
        logger.warning("Could not name conversation %s", thread_id, exc_info=True)
        conversation_name = ""
    return JSONResponse(
        {
            "thread_id": thread_id,
            "conversation_title": conversation_name,
            "renamed": bool(conversation_name),
        }
    )


@app.post("/ambient_preferences/{assistant_id}")
async def record_ambient_preference_route(
    assistant_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Record the conversation partner's decision on an ambient notification card.

    Body, in the Agent Inbox ``HumanResponse`` shape plus the observation it
    answers: ``{"observation_id", "observation_kind", "summary",
    "type": "ignore" | "response" | "accept", "args": text | null}``. The
    decision is stored as a preference in the LangGraph store and recalled as
    precedent the next time a similar scene is triaged; nothing is resumed and
    no reply is generated (a reply is simply the next ordinary turn).
    """
    from src.anubis.utils.ambient.preferences import (
        ACTION_TAKEN_AVATAR,
        ACTION_TAKEN_OWNER,
        ambient_decision_view,
        record_ambient_decision,
        record_ambient_preference,
    )

    body = await request.json()
    body = body if isinstance(body, dict) else {}
    decision_type = str(body.get("type") or "").strip().lower()
    if decision_type not in ("accept", "ignore", "response", "act", "left_alone"):
        raise HTTPException(
            status_code=400,
            detail="type must be accept, ignore, response, act, or left_alone.",
        )
    args = body.get("args")
    note = args.strip() if isinstance(args, str) and decision_type == "response" else None
    # ``act`` says who acted on the card: the avatar (the conversation partner
    # allowed the offer) or the conversation partner (replied in person).
    action_taken: str | None = None
    if decision_type == "act":
        actor = args.get("action") if isinstance(args, dict) else args
        actor = str(actor or "").strip().lower()
        if actor not in ("avatar", "owner"):
            raise HTTPException(
                status_code=400, detail="act needs args.action of avatar or owner."
            )
        action_taken = ACTION_TAKEN_AVATAR if actor == "avatar" else ACTION_TAKEN_OWNER
    # The aggregated precedent the next triage reads, per observation kind.
    aggregate_decision = {
        "act": "allowed_action" if action_taken == ACTION_TAKEN_AVATAR else "replied_self",
        "left_alone": "left_alone",
    }.get(decision_type, decision_type)
    store = getattr(app.state, "store", None)
    user_id = current_user["identities"][0]["user_id"]
    observation_kind = str(body.get("observation_kind") or "other")
    summary = str(body.get("summary") or "")
    recorded = await record_ambient_preference(
        store,
        user_id,
        assistant_id.strip(),
        observation_kind=observation_kind,
        summary=summary,
        decision=aggregate_decision,
        note=note,
    )
    # The card's own record: which thumb, note, and action this observation
    # received, so the card shows them again after a reload.
    decision = await record_ambient_decision(
        store,
        user_id,
        assistant_id.strip(),
        observation_id=(
            str(body.get("observation_id") or "").strip() or None
        ),
        observation_kind=observation_kind,
        summary=summary,
        decision=decision_type,
        note=note,
        action_taken=action_taken,
    )
    return JSONResponse(
        {
            "recorded": recorded is not None,
            "observation_id": body.get("observation_id"),
            "preference": recorded,
            "decision": ambient_decision_view(decision),
        }
    )


class MessageFeedbackRequest(BaseModel):
    """A thumb or a note on one avatar reply."""

    assistant_id: str
    thread_id: str | None = None
    # The id the reply is stored under (``done.message_id`` on a live turn, the
    # message ``id`` on a reloaded transcript).
    message_id: str | None = None
    # The request the reply streamed under, for replies rated before their
    # stored id was known.
    request_id: str | None = None
    # 'like' | 'dislike' | 'rating' | 'comment' | 'feels_real' | 'feels_fake'
    feedback_type: str
    # For 'rating': 1-5, where 3 and above counts as positive.
    rating: float | None = None
    comment: str | None = None
    # The rated reply's text, quoted in the embedded record.
    content: str | None = None
    # When the reply carried out an allowed action (or answered an
    # observation), the observation it belongs to: the thumb is also learned
    # as precedent for that kind of scene.
    observation_id: str | None = None
    observation_kind: str | None = None
    observation_summary: str | None = None


MESSAGE_FEEDBACK_TYPES = ("like", "dislike", "rating", "comment", "feels_real", "feels_fake")


def _feedback_langgraph_headers(request: Request, current_user: dict) -> dict:
    """The LangGraph credential for reading the rated thread back.

    A signed-in caller reads with the caller's own key; an anonymous visitor
    (a shared avatar chat) reads with the anonymous key, the same key the
    visitor's turns were run under.
    """
    api_key = current_user.get("API_KEY") if isinstance(current_user, dict) else None
    if not api_key:
        api_key = getattr(
            getattr(request.app.state, "context", None), "anonymous_api_key", None
        )
    return {"API-KEY": api_key} if api_key else {}


@app.post("/message_feedback")
async def record_message_feedback_route(
    request: Request,
    feedback: MessageFeedbackRequest,
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    """Record the conversation partner's reaction to one avatar reply.

    Open to anonymous visitors as well as signed-in users: a visitor chatting
    with a shared avatar is identified by the same hashed address the
    visitor's turns run under, so the visitor's ratings, notes, and learned
    preferences follow the visitor across conversations with that avatar.

    ``feedback_type`` is ``like`` or ``dislike`` (a thumb), ``rating`` (with
    ``rating`` on a 1-5 scale; 3 and above counts as positive), ``comment``
    (a written note; ``comment`` required), or ``feels_real`` / ``feels_fake``
    (what feels real or off about this reply; ``comment`` optional, the reply
    itself is the example). A thumb with a ``comment`` records both.

    The row the browser restores lives under ``(user_id, assistant_id,
    "message_feedback")``. The same press also teaches the avatar: thumbs and
    ratings become rated-message records the background sweep folds into
    learned preferences, and notes and feels-real statements are stored for
    the very next reply (see src/anubis/utils/learning/). Notes and marks that
    answer a notification card are learned as ambient precedent as well.
    """
    from src.anubis.utils.message_feedback import (
        feedback_view,
        record_message_feedback,
    )

    feedback_type = (feedback.feedback_type or "").strip().lower()
    if feedback_type not in MESSAGE_FEEDBACK_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"feedback_type must be one of: {', '.join(MESSAGE_FEEDBACK_TYPES)}.",
        )
    if not (feedback.message_id or "").strip() and not (feedback.request_id or "").strip():
        raise HTTPException(
            status_code=400, detail="message_id or request_id is required."
        )
    if feedback_type == "rating" and feedback.rating is None:
        raise HTTPException(status_code=400, detail="rating is required for feedback_type=rating.")
    if feedback_type == "comment" and not (feedback.comment or "").strip():
        raise HTTPException(status_code=400, detail="comment is required for feedback_type=comment.")

    store = getattr(app.state, "store", None)
    user_id = current_user["identities"][0]["user_id"]
    assistant_id = feedback.assistant_id.strip()
    thread_id = (feedback.thread_id or "").strip() or None
    message_id = (feedback.message_id or "").strip() or None
    request_id = (feedback.request_id or "").strip() or None

    thumb: str | None = None
    feels: str | None = None
    rating_score: float | None = None
    if feedback_type in ("like", "dislike"):
        thumb = feedback_type
    elif feedback_type == "rating":
        rating_score = float(feedback.rating)
        thumb = "like" if rating_score >= 3.0 else "dislike"
    elif feedback_type in ("feels_real", "feels_fake"):
        feels = feedback_type

    # A press made before the browser knew the reply's stored id (only the
    # request id, or only the quoted text) is still filed under the stored
    # id when the thread can name the reply, so the rating comes back on the
    # reply after a reload; stored replies carry no request id to match on.
    langgraph_client_headers = _feedback_langgraph_headers(request, current_user)
    resolved_reply: tuple[dict | None, dict | None] | None = None
    if message_id is None and thread_id:
        resolved_reply = await _resolve_rated_reply(
            langgraph_client_headers,
            thread_id=thread_id,
            message_id=None,
            request_id=request_id,
            content=feedback.content,
        )
        resolved_avatar_message = resolved_reply[0]
        if resolved_avatar_message and resolved_avatar_message.get("id"):
            message_id = str(resolved_avatar_message["id"])

    recorded = await record_message_feedback(
        store,
        user_id,
        assistant_id,
        thread_id=thread_id,
        message_id=message_id,
        request_id=request_id,
        feedback_type=thumb,
        comment=feedback.comment,
        content=feedback.content,
        feels=feels,
        rating_score=rating_score,
    )

    # Teach the avatar from the same press (rated message, note, feels-real).
    learning: dict | None = None
    try:
        learning = await _learn_from_message_feedback(
            request.app.state,
            langgraph_client_headers=langgraph_client_headers,
            user_id=user_id,
            assistant_id=assistant_id,
            thread_id=thread_id,
            message_id=message_id,
            request_id=request_id,
            thumb=thumb,
            rating_score=rating_score,
            feels=feels,
            comment=feedback.comment,
            content=feedback.content,
            resolved_reply=resolved_reply,
        )
    except Exception as learning_error:  # noqa: BLE001 - the row is saved; learning is best effort
        logger.warning("Message feedback learning failed: %s", learning_error)

    ambient_decision = None
    if (feedback.observation_id or "").strip() and (thumb or feels):
        from src.anubis.utils.ambient.preferences import (
            ambient_decision_view,
            record_ambient_decision,
            record_ambient_preference,
        )

        observation_kind = feedback.observation_kind or "other"
        summary = feedback.observation_summary or ""
        liked = thumb == "like" or (thumb is None and feels == "feels_real")
        await record_ambient_preference(
            store,
            user_id,
            assistant_id,
            observation_kind=observation_kind,
            summary=summary,
            decision="liked_action" if liked else "disliked_action",
            note=feedback.comment,
        )
        ambient_decision = ambient_decision_view(
            await record_ambient_decision(
                store,
                user_id,
                assistant_id,
                observation_id=feedback.observation_id.strip(),
                observation_kind=observation_kind,
                summary=summary,
                decision="rated_after_action",
                rated_after_action="like" if liked else "dislike",
            )
        )
    return JSONResponse(
        {
            "recorded": recorded is not None,
            "feedback_type": feedback_type,
            "ambient_decision": ambient_decision,
            "message_id": recorded.get("message_id") if recorded else message_id,
            "request_id": recorded.get("request_id") if recorded else request_id,
            "feedback": feedback_view(recorded),
            "learning": learning,
        }
    )


@app.get("/avatar_preferences/{assistant_id}")
async def get_avatar_preferences_route(
    assistant_id: str,
    thread_id: str | None = None,
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    """Everything this caller has told one avatar through ratings and notes.

    Open to anonymous visitors as well as signed-in users (see
    ``record_message_feedback_route``).

    ``message_feedback`` lists the thumbs and notes on replies (narrowed to
    one thread when ``thread_id`` is given); ``ambient_decisions`` lists the
    thumb and note each notification card received. The browser reads this
    after every press so what is on screen is what is stored.
    """
    from src.anubis.utils.ambient.preferences import list_ambient_decisions
    from src.anubis.utils.learning.feedback import list_learned_records
    from src.anubis.utils.message_feedback import feedback_view, list_message_feedback

    store = getattr(app.state, "store", None)
    user_id = current_user["identities"][0]["user_id"]
    feedback_records = await list_message_feedback(
        store, user_id, assistant_id.strip(), thread_id=(thread_id or "").strip() or None
    )
    decisions = await list_ambient_decisions(store, user_id, assistant_id.strip())
    learned = await list_learned_records(store, user_id, assistant_id.strip())
    return JSONResponse(
        {
            "assistant_id": assistant_id.strip(),
            "thread_id": (thread_id or "").strip() or None,
            # What the avatar has learned about this person: dictated and
            # inferred preferences, feedback messages, and what feels real.
            "learned_preferences": learned.get("preferences", []),
            "feedback_messages": learned.get("feedback_messages", []),
            "what_feels_real": learned.get("what_feels_real", []),
            "message_feedback": [
                {
                    "message_id": record.get("message_id"),
                    "request_id": record.get("request_id"),
                    "thread_id": record.get("thread_id"),
                    "feedback": feedback_view(record),
                }
                for record in feedback_records
            ],
            "ambient_decisions": decisions,
        }
    )


async def _with_stored_message_feedback(
    messages: list, *, user_id: str, assistant_id: str, thread_id: str
) -> list:
    """Attach each stored rating to its reply; a store problem changes nothing."""
    from src.anubis.utils.message_feedback import (
        attach_message_feedback,
        list_message_feedback,
    )

    try:
        records = await list_message_feedback(
            getattr(app.state, "store", None),
            user_id,
            assistant_id,
            thread_id=thread_id,
        )
        return attach_message_feedback(messages, records)
    except Exception:  # noqa: BLE001 - a rating must never hide a transcript
        logger.debug("Could not attach message feedback", exc_info=True)
        return messages


async def _load_thread_message_dicts(langgraph_client, thread_id: str) -> list[dict]:
    """The checkpointed messages of one thread as plain dicts (``id``, ``type``, ``content``)."""
    state = await langgraph_client.threads.get_state(thread_id=thread_id)
    messages = state.get("values", {}).get("messages", []) if state else []
    return [message for message in messages if isinstance(message, dict)]


def _message_dict_type(message: dict) -> str:
    return str(message.get("type") or message.get("role") or "")


def _message_dict_request_id(message: dict) -> str:
    response_metadata = message.get("response_metadata") or {}
    request_id = message.get("request_id") or (
        response_metadata.get("request_id") if isinstance(response_metadata, dict) else None
    )
    return str(request_id or "")


def _find_avatar_message_and_prompt(
    messages: list[dict],
    message_id: str | None,
    request_id: str | None = None,
    content_excerpt: str | None = None,
) -> tuple[dict | None, dict | None]:
    """The avatar message ``message_id`` names (else the one streamed under
    ``request_id``, else the newest one whose text starts with
    ``content_excerpt``, else the latest one when nothing was named) and the
    user message before it."""
    from src.anubis.utils.learning.sentiment import message_text

    avatar_index: int | None = None
    excerpt = (content_excerpt or "").strip()[:200]
    for index, message in enumerate(messages):
        if _message_dict_type(message) not in ("ai", "assistant"):
            continue
        if message.get("tool_calls"):
            continue
        if message_id is not None:
            if message.get("id") == message_id:
                avatar_index = index
                break
            continue
        if request_id:
            if _message_dict_request_id(message) == request_id:
                avatar_index = index
                break
            if excerpt and message_text(message.get("content")).strip().startswith(excerpt):
                avatar_index = index
            continue
        avatar_index = index
    if avatar_index is None:
        return None, None
    preceding_user_message = None
    for message in reversed(messages[:avatar_index]):
        if _message_dict_type(message) in ("human", "user"):
            preceding_user_message = message
            break
    return messages[avatar_index], preceding_user_message


async def _resolve_rated_reply(
    langgraph_client_headers: dict,
    *,
    thread_id: str | None,
    message_id: str | None,
    request_id: str | None,
    content: str | None,
) -> tuple[dict | None, dict | None]:
    """The stored reply a feedback press names, and the user message before it.

    Read from the thread when the thread is known: by stored id, else by the
    request id the reply streamed under, else by the text the browser quoted.
    A thread that cannot be read resolves to nothing, and the caller falls
    back to what the browser sent.
    """
    if not thread_id:
        return None, None
    try:
        langgraph_client = get_client(headers=langgraph_client_headers)
        messages = await _load_thread_message_dicts(langgraph_client, thread_id)
    except Exception as thread_error:  # noqa: BLE001 - fall back to the browser's text
        logger.debug("Could not load the rated thread %s: %s", thread_id, thread_error)
        return None, None
    return _find_avatar_message_and_prompt(messages, message_id, request_id, content)


async def _learn_from_message_feedback(
    app_state,
    *,
    langgraph_client_headers: dict,
    user_id: str,
    assistant_id: str,
    thread_id: str | None,
    message_id: str | None,
    request_id: str | None,
    thumb: str | None,
    rating_score: float | None,
    feels: str | None,
    comment: str | None,
    content: str | None,
    resolved_reply: tuple[dict | None, dict | None] | None = None,
) -> dict:
    """Turn one feedback press into continuous-learning records.

    A thumb (or a rating) becomes a rated-message record holding the reply and
    the user message before it; a note becomes a feedback message; a
    feels-real mark becomes a what-feels-real statement. The reply text comes
    from the thread when the thread is known, else from the text the browser
    sent. Returns what was learned, for the response.
    """
    from src.anubis.utils.learning.feedback import (
        store_feedback_message,
        store_message_rating,
        store_what_feels_real,
    )
    from src.anubis.utils.learning.namespaces import RATING_NEGATIVE, RATING_POSITIVE
    from src.anubis.utils.learning.sentiment import message_text

    store = getattr(app_state, "store", None)
    if store is None:
        return {}
    avatar_message, preceding_user_message = (
        resolved_reply
        if resolved_reply is not None
        else await _resolve_rated_reply(
            langgraph_client_headers,
            thread_id=thread_id,
            message_id=message_id,
            request_id=request_id,
            content=content,
        )
    )
    avatar_text = (
        message_text(avatar_message.get("content")) if avatar_message else ""
    ).strip() or (content or "").strip()
    preceding_text = (
        message_text(preceding_user_message.get("content")).strip()
        if preceding_user_message
        else None
    )
    resolved_message_id = (
        str(avatar_message.get("id"))
        if avatar_message and avatar_message.get("id")
        else (message_id or request_id)
    )
    learned: dict = {}
    if thumb in ("like", "dislike") and resolved_message_id and avatar_text:
        await store_message_rating(
            store,
            user_id,
            assistant_id,
            rating=RATING_POSITIVE if thumb == "like" else RATING_NEGATIVE,
            rating_score=rating_score,
            thread_id=thread_id or "",
            message_id=resolved_message_id,
            avatar_message_text=avatar_text,
            preceding_user_message_text=preceding_text,
            request_id=request_id,
        )
        learned["rating"] = "collected; folded into learned preferences by the background sweep"
    if (comment or "").strip():
        document = await store_feedback_message(
            store,
            user_id,
            assistant_id,
            comment=comment,
            thread_id=thread_id,
            message_id=resolved_message_id,
            related_avatar_message=avatar_text or None,
        )
        learned["feedback_message"] = (
            "immediate; used in the next reply" if document is not None else "duplicate"
        )
    if feels in ("feels_real", "feels_fake"):
        statement = (comment or "").strip()
        if not statement and avatar_text:
            label = "feels real" if feels == "feels_real" else "feels fake or off"
            statement = f"A reply like this one {label}: {avatar_text[:400]}"
        if statement:
            document = await store_what_feels_real(
                store,
                user_id,
                assistant_id,
                statement=statement,
                statement_context=(
                    f"About this avatar message: {avatar_text[:400]}" if avatar_text else None
                ),
                polarity=feels,
                source="dictated",
                thread_id=thread_id,
                message_id=resolved_message_id,
            )
            learned["what_feels_real"] = (
                "immediate; used in the next reply" if document is not None else "duplicate"
            )
    return learned


async def _apply_inline_message_feedback(
    app_state,
    *,
    langgraph_client_headers: dict,
    user_id: str,
    assistant_id: str,
    thread_id: str,
    like: bool,
    dislike: bool,
    feedback_text: str | None,
) -> None:
    """Apply the ``like`` / ``dislike`` / ``feedback`` fields of one /message call.

    The thumb lands on the thread's latest avatar reply (both the row the
    browser restores and the learning record); ``feedback_text`` is stored as
    a feedback message about that reply and used from the very next turn.
    """
    from src.anubis.utils.message_feedback import record_message_feedback

    thumb = None
    if like or dislike:
        thumb = "like" if like and not dislike else "dislike"
    if not thumb and not (feedback_text or "").strip():
        return
    langgraph_client = get_client(headers=langgraph_client_headers)
    messages = await _load_thread_message_dicts(langgraph_client, thread_id)
    avatar_message, _preceding = _find_avatar_message_and_prompt(messages, None)
    latest_message_id = (
        str(avatar_message.get("id")) if avatar_message and avatar_message.get("id") else None
    )
    if thumb and latest_message_id:
        await record_message_feedback(
            app_state.store,
            user_id,
            assistant_id,
            thread_id=thread_id,
            message_id=latest_message_id,
            request_id=_message_dict_request_id(avatar_message) or None,
            feedback_type=thumb,
            comment=feedback_text,
        )
    await _learn_from_message_feedback(
        app_state,
        langgraph_client_headers=langgraph_client_headers,
        user_id=user_id,
        assistant_id=assistant_id,
        thread_id=thread_id,
        message_id=latest_message_id,
        request_id=None,
        thumb=thumb,
        rating_score=None,
        feels=None,
        comment=feedback_text,
        content=None,
    )


# Still images the platform accepts from a person. The first four are the ones
# every model vendor reads directly; the rest are what phones and design tools
# actually produce (an iPhone photo is HEIC, a modern export is AVIF, a scan is
# TIFF), and they are transcoded on the way in by
# ``transcode_still_image_to_model_readable`` so no reader downstream has to
# know about them.
MODEL_READABLE_IMAGE_MIMES = frozenset(
    {"image/jpeg", "image/png", "image/gif", "image/webp"}
)
TRANSCODED_IMAGE_MIMES = frozenset(
    {"image/heic", "image/heif", "image/avif", "image/bmp", "image/tiff"}
)
ALLOWED_IMAGE_MIMES = MODEL_READABLE_IMAGE_MIMES | TRANSCODED_IMAGE_MIMES

# Spellings of the same still image formats that clients and web servers send.
# Each name on the left is the same bytes as the name on the right, so they are
# folded together before anything is decided from the declaration.
_IMAGE_MIME_ALIASES = {
    "image/jpg": "image/jpeg",
    "image/pjpeg": "image/jpeg",
    "image/x-png": "image/png",
    "image/x-bmp": "image/bmp",
    "image/x-ms-bmp": "image/bmp",
    "image/tif": "image/tiff",
    "image/x-tiff": "image/tiff",
    "image/x-heic": "image/heic",
    "image/heic-sequence": "image/heic",
    "image/x-heif": "image/heif",
    "image/heif-sequence": "image/heif",
    "image/avif-sequence": "image/avif",
    "image/x-avif": "image/avif",
}


def normalize_declared_image_mime(ct: str) -> str:
    ct = (ct or "").split(";")[0].strip().lower()
    return _IMAGE_MIME_ALIASES.get(ct, ct)


# The ``ftyp`` brands that make an ISO base media file a still picture rather
# than a movie. ``heic``/``heix``/``hevc``/``hevx`` are HEVC-coded photos,
# ``mif1``/``msf1`` are the generic HEIF forms an iPhone also writes, and
# ``avif``/``avis`` are the AV1-coded ones; every other brand (``isom``,
# ``mp42``, ``qt  ``...) stays video.
_ISO_BASE_MEDIA_BRAND_MIMES = {
    b"heic": "image/heic",
    b"heix": "image/heic",
    b"heim": "image/heic",
    b"heis": "image/heic",
    b"hevc": "image/heic",
    b"hevx": "image/heic",
    b"hevm": "image/heic",
    b"hevs": "image/heic",
    b"mif1": "image/heif",
    b"msf1": "image/heif",
    b"avif": "image/avif",
    b"avis": "image/avif",
}


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
    if chunk[:2] == b"BM":
        return "image/bmp"
    if chunk[:4] in (b"II\x2a\x00", b"MM\x00\x2a"):
        return "image/tiff"
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
        # An iPhone photo and an iPhone movie are both ISO base media files;
        # only the brand in the ``ftyp`` box says which one this is.
        return _ISO_BASE_MEDIA_BRAND_MIMES.get(chunk[8:12], "video/mp4")
    return None


# Content-Type values that name no media type at all. A client that cannot map
# the filename extension sends one of these instead of admitting it does not
# know, so a declaration in this set carries no information about the bytes.
UNINFORMATIVE_UPLOAD_MIMES = frozenset(
    {
        "",
        "application/octet-stream",
        "binary/octet-stream",
        "application/x-binary",
        "application/download",
        "application/unknown",
    }
)


def effective_upload_mime_type(declared_mime: str, body: bytes) -> str:
    """Return the media type an upload should be routed by: contents over declaration.

    A multipart client derives the part's Content-Type from the filename
    extension, and no client's extension table is complete: curl, several
    mobile file pickers, and every library that does not guess send a
    ``.webp`` upload as ``application/octet-stream`` while sending the same
    picture as ``.png`` correctly. Routing by the declaration alone therefore
    drops WebP images that are perfectly readable, so the magic bytes decide
    whenever the declaration is uninformative or names a different category
    than the bytes (``text/plain`` for a WebP). A declaration that agrees with
    the bytes on the category is kept as sent, so a QuickTime recording stays
    ``video/quicktime`` instead of being flattened to the sniffer's coarser
    ``video/mp4``.
    """
    declared_upload_mime = normalize_declared_image_mime(declared_mime)
    sniffed_upload_mime = normalize_declared_image_mime(
        _sniff_media_category_from_bytes(body[:512]) or ""
    )
    if not sniffed_upload_mime:
        return declared_upload_mime
    if declared_upload_mime in UNINFORMATIVE_UPLOAD_MIMES:
        return sniffed_upload_mime
    if declared_upload_mime.split("/")[0] != sniffed_upload_mime.split("/")[0]:
        return sniffed_upload_mime
    return declared_upload_mime


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
    """Return normalized image MIME; raises HTTPException if not an allowed still image.

    The declared Content-Type is a hint, never evidence. Clients derive the
    multipart part's Content-Type from the filename extension (curl and browsers
    both do), so a JPEG that someone saved as ``screenshot.PNG`` arrives declared
    ``image/png``. The magic bytes are the only authority on what the file
    actually is, so a recognized sniff always wins over the declaration: the
    declaration is used only when the magic bytes are unrecognized.
    """
    declared_image_mime = normalize_declared_image_mime(declared_mime)
    sniffed_image_mime = normalize_declared_image_mime(
        _sniff_media_category_from_bytes(body[:512]) or ""
    )

    if sniffed_image_mime:
        # Magic bytes recognized — they decide, whatever the caller declared.
        if sniffed_image_mime not in ALLOWED_IMAGE_MIMES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"File contents are {sniffed_image_mime!r}, which is not an "
                    "allowed still image; allowed: image/jpeg, image/png, image/gif (non-animated), image/webp (non-animated), image/heic, image/heif, image/avif, image/bmp, image/tiff."
                ),
            )
        if declared_image_mime not in ("", "application/octet-stream") and (
            declared_image_mime != sniffed_image_mime
        ):
            logger.info(
                "Upload declared Content-Type %r but the file contents are %r; "
                "using the sniffed type.",
                declared_image_mime,
                sniffed_image_mime,
            )
        mime = sniffed_image_mime
    else:
        # Magic bytes unrecognized (an image format this sniffer does not cover):
        # the declaration is all there is to go on.
        if declared_image_mime in ("", "application/octet-stream"):
            raise HTTPException(
                status_code=400,
                detail="Could not determine an allowed image type from the file or URL.",
            )
        if declared_image_mime not in ALLOWED_IMAGE_MIMES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Image type not allowed (got {declared_image_mime!r}); "
                    "allowed: image/jpeg, image/png, image/gif (non-animated), image/webp (non-animated), image/heic, image/heif, image/avif, image/bmp, image/tiff."
                ),
            )
        mime = declared_image_mime

    if mime == "image/gif" and _gif_is_animated(body):
        raise HTTPException(
            status_code=400, detail="Animated GIF is not allowed; use a still frame."
        )
    if mime == "image/webp" and _webp_is_animated(body):
        raise HTTPException(
            status_code=400, detail="Animated WebP is not allowed; use a still image."
        )
    return mime


def transcode_still_image_to_model_readable(
    mime: str, body: bytes
) -> tuple[str, bytes]:
    """Return ``(mime, bytes)`` in a format every model vendor can read.

    The image formats a person actually has are wider than the four formats the
    vision models accept: an iPhone photo is HEIC, a modern web export is AVIF,
    a scan is TIFF, a screenshot pasted out of an old tool is BMP. Rejecting
    those asks the person to go and convert their own picture, so they are
    decoded here once, at the edge, and everything downstream — the description
    model, the reference-portrait store, the emotion-media vendor — only ever
    sees JPEG, PNG, GIF or WebP.

    Transparency decides the target: an image carrying an alpha channel becomes
    PNG so the transparent parts do not turn black, and everything else becomes
    JPEG, which is far smaller than PNG for a photograph. A frame is taken from
    a sequence (a Live Photo, an animated AVIF) because a still is what the
    identity pipeline reads.
    """
    import io

    if mime in MODEL_READABLE_IMAGE_MIMES:
        return mime, body

    from PIL import Image

    if mime in ("image/heic", "image/heif"):
        try:
            import pillow_heif
        except ImportError as heif_support_missing:
            raise HTTPException(
                status_code=400,
                detail=(
                    "HEIC/HEIF images cannot be read by this deployment "
                    "(pillow-heif is not installed). Please upload the photo as "
                    "JPEG, PNG or WebP."
                ),
            ) from heif_support_missing
        pillow_heif.register_heif_opener()

    try:
        with Image.open(io.BytesIO(body)) as opened_image:
            opened_image.load()
            keeps_transparency = opened_image.mode in ("RGBA", "LA", "PA") or (
                "transparency" in opened_image.info
            )
            converted_image = opened_image.convert(
                "RGBA" if keeps_transparency else "RGB"
            )
    except Exception as undecodable_image:  # noqa: BLE001 - a client error
        raise HTTPException(
            status_code=400,
            detail=(
                f"The {mime} image could not be read. Please upload the picture "
                "as JPEG, PNG or WebP."
            ),
        ) from undecodable_image

    rewritten = io.BytesIO()
    if keeps_transparency:
        converted_image.save(rewritten, format="PNG", optimize=True)
        return "image/png", rewritten.getvalue()
    converted_image.save(rewritten, format="JPEG", quality=90, optimize=True)
    return "image/jpeg", rewritten.getvalue()


def prepare_still_image_upload(declared_mime: str, body: bytes) -> tuple[str, bytes]:
    """Validate one uploaded still image and return bytes a model can read.

    The single entry point every image upload goes through: the magic bytes
    decide the type, an animated GIF or WebP is refused, and a format no model
    vendor reads is transcoded. Callers store and forward the returned bytes,
    not the ones the person sent, so the declared type, the stored type and the
    bytes always agree.
    """
    return transcode_still_image_to_model_readable(
        validate_upload_image_bytes(declared_mime, body), body
    )


async def probe_remote_url_content_type(url: str) -> str:
    """Best-effort Content-Type for a remote URL (HEAD, then ranged GET + sniff).

    Known MediaWiki article URLs (Fandom, Wikipedia, wiki.gg) are ``text/html``
    even when the HTML skin is Cloudflare-challenged: the parse API on the
    same host is what ``fetch_remote_url_bytes`` will actually download.
    A failed HEAD (403 challenge page) must not be treated as the type.
    """
    from src.anubis.utils.classes.URLDocumentLoaderClass import (
        mediawiki_article_api_target,
    )

    if mediawiki_article_api_target(url):
        return "text/html"
    async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
        head_ct = ""
        try:
            head = await client.head(url)
            if head.status_code < 400:
                head_ct = (
                    (head.headers.get("content-type") or "")
                    .split(";")[0]
                    .strip()
                    .lower()
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
    """Download a URL and return (body, Content-Type without parameters).

    MediaWiki article URLs are fetched through the parse API so Fandom
    ``/wiki/…`` pages (Cloudflare-challenged HTML skins) still ingest.
    """
    from src.anubis.utils.classes.URLDocumentLoaderClass import (
        ARTICLE_FETCH_USER_AGENT,
        fetch_mediawiki_article_html,
    )

    mediawiki = await fetch_mediawiki_article_html(url)
    if mediawiki is not None:
        body, header_ct = mediawiki
        if len(body) > max_bytes:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Remote resource exceeds maximum download size ({max_bytes} bytes)."
                ),
            )
        return body, header_ct
    async with httpx.AsyncClient(follow_redirects=True, timeout=120.0) as client:
        try:
            r = await client.get(url, headers={"User-Agent": ARTICLE_FETCH_USER_AGENT})
            r.raise_for_status()
        except httpx.HTTPStatusError as fetch_error:
            status = (
                fetch_error.response.status_code
                if fetch_error.response is not None
                else "unknown"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Could not fetch URL (HTTP {status}).",
            ) from fetch_error
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
                _is_tabular_scalar(cell_value) for cell_value in column_values.values()
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


async def _assistant_owner_for_media(
    assistant_id: str, current_user: dict
) -> str | None:
    """The owner of an avatar, for reading its public media as any chatter."""
    langgraph_client = get_client(headers={"API-KEY": current_user["API_KEY"]})
    try:
        assistant = await langgraph_client.assistants.get(assistant_id=assistant_id)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not load assistant: {exc}"
        ) from exc
    return (assistant.get("metadata") or {}).get("user_id")


@app.get("/avatar_emotion_media")
async def get_avatar_emotion_media(
    assistant_id: str,
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    """The avatar's emotion media manifest: one still and one idle loop per emotion.

    Read by the chat once per avatar and cached in memory on the client; the
    entries name asset URLs served by ``GET /avatar_emotion_media/{asset_id}``.
    Anyone who may chat with the avatar may read this, like the portrait.
    """
    from src.anubis.utils.media_assets import get_media_asset_repository
    from src.anubis.utils.media_generation.emotion_media import build_manifest

    repository = get_media_asset_repository()
    if repository is None:
        return JSONResponse(
            {
                "emotions": {},
                "complete": False,
                "missing": [],
                "last_generation": None,
                "generation": None,
            }
        )
    owner_user_id = await _assistant_owner_for_media(assistant_id, current_user)
    assets = await repository.list_emotion_assets(assistant_id)
    manifest = build_manifest(assets)
    manifest["last_generation"] = await _last_emotion_media_generation(
        repository, assistant_id
    )
    manifest["generation"] = _emotion_media_generation_permission(
        current_user, owner_user_id, manifest
    )
    return JSONResponse(manifest)


def _emotion_media_generation_permission(
    current_user: dict, owner_user_id: str | None, manifest: dict[str, Any]
) -> dict[str, Any] | None:
    """Whether this caller may generate the avatar's emotion media, and at what cost.

    The avatar's settings screen shows the generate button behind this answer:
    the button is offered to the owner alone, is disabled with the required
    tier named when the owner's tier is below ``EMOTION_MEDIA_MINIMUM_TIER``,
    and is disabled with no upgrade prompt when the deployment has generation
    switched off entirely. ``None`` for anyone who is not the owner — a chatter
    reading the manifest is never shown a control.

    The two cost estimates come with the answer, because the owner is asked to
    confirm the spend before a run starts: ``cost_full_rebuild`` replaces every
    generated still and idle loop, and ``cost_missing_only`` builds just what
    the newest run left absent.
    """
    from src.anubis.utils.media_generation.emotion_media import (
        emotion_media_cost_estimate,
        emotion_media_enabled,
        full_build_asset_counts,
        missing_asset_counts,
    )

    identities = current_user.get("identities") or []
    caller_user_id = identities[0].get("user_id") if identities else None
    if not owner_user_id or caller_user_id != owner_user_id:
        return None
    tier = resolve_tier(current_user)
    minimum_tier = resolve_emotion_media_minimum_tier()
    configured = emotion_media_enabled(app.state.context)
    full_stills, full_loops = full_build_asset_counts()
    missing_stills, missing_loops = missing_asset_counts(manifest.get("missing"))
    return {
        "tier": tier.value,
        "required_tier": minimum_tier.value,
        "tier_allows": tier_meets_minimum(tier, minimum_tier),
        "configured": configured,
        "allowed": configured and tier_meets_minimum(tier, minimum_tier),
        "cost_full_rebuild": emotion_media_cost_estimate(
            app.state.context, still_count=full_stills, idle_loop_count=full_loops
        ),
        "cost_missing_only": emotion_media_cost_estimate(
            app.state.context,
            still_count=missing_stills,
            idle_loop_count=missing_loops,
        ),
    }


async def _last_emotion_media_generation(
    repository: Any, assistant_id: str
) -> dict[str, Any] | None:
    """Why the newest generation run left assets missing, for the settings screen.

    The newest ``emotion_media`` job row (an upload or a regeneration), reduced
    to its state, the per-asset failures with their ``error_code``, and the
    summary sentence. ``None`` when no run has been recorded.
    """
    from src.anubis.utils.media_generation.emotion_media import summarize_failures

    try:
        jobs = await repository.list_jobs(
            assistant_id=assistant_id, job_kind="emotion_media"
        )
    except Exception:  # noqa: BLE001 - the manifest must not fail on history
        logger.debug("Could not read emotion media jobs", exc_info=True)
        return None
    if not jobs:
        return None
    job = jobs[0]
    detail = job.get("detail") or {}
    failures = list(detail.get("failures") or [])
    summary = detail.get("summary") or summarize_failures(failures)
    return {
        "job_id": job.get("job_id"),
        "state": job.get("state"),
        "finished_at": job.get("updated_at"),
        "subject": detail.get("subject"),
        "withheld": bool(detail.get("withheld", False)),
        "moderation_risk": detail.get("moderation_risk"),
        "moderation_reasons": detail.get("moderation_reasons") or [],
        "error": detail.get("error"),
        "failures": [
            {
                "emotion": f.get("emotion"),
                "asset_kind": f.get("asset_kind"),
                "error_code": f.get("error_code"),
                "message": f.get("message"),
            }
            for f in failures
        ],
        "summary": summary,
    }


@app.get("/avatar_emotion_media/{asset_id}")
async def get_avatar_emotion_media_asset(
    asset_id: str,
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    """Stream one generated asset's bytes (an emotion still or an idle loop)."""
    from src.anubis.utils.media_assets import get_media_asset_repository

    repository = get_media_asset_repository()
    if repository is None:
        raise HTTPException(status_code=404, detail="No media is available.")
    asset = await repository.get_emotion_asset(asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="No such asset.")
    # Asset ids are unguessable and rows are immutable once written, so the
    # response is cacheable indefinitely; a regenerated asset gets a new id.
    return Response(
        content=asset["bytes"],
        media_type=asset.get("mime_type") or "application/octet-stream",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


async def _run_emotion_media_job(
    job_id: str,
    user_id: str,
    assistant_id: str,
    reference_image_data_uri: str,
    only_missing: bool,
    emotions: tuple[str, ...] | None = None,
    asset_kinds: tuple[str, ...] | None = None,
    extra_prompt: str | None = None,
    subject: str | None = None,
    assessment: dict[str, Any] | None = None,
    proceed_despite_moderation_risk: bool = False,
) -> None:
    from src.anubis.utils.billing.metering import persist_api_metrics_row
    from src.anubis.utils.media_assets import get_media_asset_repository
    from src.anubis.utils.media_assets.repository import (
        JOB_STATE_COMPLETED,
        JOB_STATE_FAILED,
        JOB_STATE_RUNNING,
    )
    from src.anubis.utils.media_generation.emotion_media import (
        generate_emotion_media_for_avatar,
        summarize_failures,
    )

    repository = get_media_asset_repository()
    if repository is None:
        return
    await repository.update_job(job_id, state=JOB_STATE_RUNNING)

    async def _record_metric(inference_type, cost_usd, model_name, request_id):
        try:
            await persist_api_metrics_row(
                app.state.pool,
                inference_type=inference_type,
                cost_usd=cost_usd,
                user_id=user_id,
                assistant_id=assistant_id,
                model_name=model_name,
            )
        except Exception:  # noqa: BLE001
            logger.debug("Could not record %s cost", inference_type, exc_info=True)

    def _progress(stage, fields):
        asyncio.create_task(
            repository.update_job(job_id, detail={"stage": stage, **fields})
        )

    try:
        manifest = await generate_emotion_media_for_avatar(
            app.state.context,
            repository,
            user_id=user_id,
            assistant_id=assistant_id,
            reference_image_data_uri=reference_image_data_uri,
            only_missing=only_missing,
            emotions=emotions,
            asset_kinds=asset_kinds,
            extra_prompt=extra_prompt,
            subject=subject,
            assessment=assessment,
            proceed_despite_moderation_risk=proceed_despite_moderation_risk,
            progress=_progress,
            metrics=_record_metric,
        )
        failures = manifest.get("failures") or []
        await repository.update_job(
            job_id,
            state=JOB_STATE_COMPLETED if not failures else JOB_STATE_FAILED,
            detail={
                "complete": manifest.get("complete"),
                "failures": failures,
                "summary": summarize_failures(failures),
                "subject": manifest.get("subject"),
                "withheld": manifest.get("withheld", False),
                "moderation_risk": manifest.get("moderation_risk"),
                "moderation_reasons": manifest.get("moderation_reasons"),
            },
        )
    except Exception as job_error:  # noqa: BLE001
        logger.exception("Emotion media job %s failed: %s", job_id, job_error)
        await repository.update_job(
            job_id, state=JOB_STATE_FAILED, detail={"error": str(job_error)}
        )


@app.post("/avatar_emotion_media/regenerate")
async def regenerate_avatar_emotion_media(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """(Re)build an avatar's emotion stills and idle loops from its reference image.

    Body: ``assistant_id``, optional ``only_missing`` (default true — retry what
    failed rather than paying for the whole set again), optional ``emotion`` and
    ``asset_kind`` (``still`` or ``idle_loop``) to redo one asset, optional
    ``prompt`` appended to that generation so the owner can improve a video or
    portrait. Runs as a durable job in ``avatar_media_jobs``; poll
    ``GET /avatar_media_jobs/{job_id}``.
    """
    from src.anubis.utils.media_assets import get_media_asset_repository
    from src.anubis.utils.media_assets.repository import (
        ASSET_KIND_IDLE_LOOP,
        ASSET_KIND_STILL,
    )
    from src.anubis.utils.media_generation.emotion_media import emotion_media_enabled
    from src.anubis.utils.media_generation.prompts import BASE_EMOTIONS

    body = await request.json()
    body = body if isinstance(body, dict) else {}
    assistant_id = str(body.get("assistant_id") or "").strip()
    emotion = str(body.get("emotion") or "").strip() or None
    asset_kind = str(body.get("asset_kind") or "").strip() or None
    extra_prompt = str(body.get("prompt") or "").strip() or None
    # The owner's explicit "generate anyway" after a predicted moderation
    # refusal: attempt the vendor calls at their own cost.
    proceed_despite_moderation_risk = bool(
        body.get("proceed_despite_moderation_risk", False)
    )
    targeted = bool(emotion or asset_kind or extra_prompt)
    only_missing = bool(body.get("only_missing", not targeted))
    if not assistant_id:
        raise HTTPException(status_code=400, detail="assistant_id is required")
    if emotion and emotion not in BASE_EMOTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown emotion {emotion!r}.")
    if asset_kind and asset_kind not in (ASSET_KIND_STILL, ASSET_KIND_IDLE_LOOP):
        raise HTTPException(
            status_code=400, detail="asset_kind must be still or idle_loop."
        )
    enforce_tier_capability(current_user, TierCapability.UPLOAD)
    enforce_emotion_media_tier(current_user)
    await resolve_assistant_for_creator(
        assistant_id,
        current_user,
        action_description="regenerate media for that avatar",
    )
    repository = get_media_asset_repository()
    if repository is None or not emotion_media_enabled(app.state.context):
        raise HTTPException(
            status_code=503, detail="Emotion media generation is not configured."
        )
    user_id = current_user["identities"][0]["user_id"]
    item = await app.state.store.aget(
        (user_id, assistant_id, "reference_image"), assistant_id
    )
    value = (getattr(item, "value", None) or {}) if item is not None else {}
    reference_image_data_uri = value.get("reference_image_data")
    if not reference_image_data_uri:
        raise HTTPException(
            status_code=404,
            detail="Upload a reference image before generating emotion media.",
        )
    # References stored before they were assessed carry no subject or
    # moderation risk: assess now and keep the answer with the image.
    from src.anubis.utils.media_generation.reference_subject import (
        assessment_from_store_value,
        assessment_store_fields,
        classify_reference_subject,
    )

    assessment = assessment_from_store_value(value)
    if assessment is None:
        assessment = await classify_reference_subject(
            reference_image_data_uri, app.state.context
        )
        try:
            from src.anubis.utils.store_cache import invalidate_store_cache_entry

            reference_namespace = (user_id, assistant_id, "reference_image")
            await app.state.store.aput(
                reference_namespace,
                assistant_id,
                {**value, **assessment_store_fields(assessment)},
            )
            invalidate_store_cache_entry(reference_namespace, assistant_id)
        except Exception:  # noqa: BLE001 - remembering the assessment is optional
            logger.debug("Could not store the reference assessment", exc_info=True)
    subject = assessment["subject"]
    job_id = await repository.create_job(
        user_id=user_id,
        assistant_id=assistant_id,
        job_kind="emotion_media",
        detail={
            "only_missing": only_missing,
            "emotion": emotion,
            "asset_kind": asset_kind,
            "prompt": extra_prompt,
            "subject": subject,
            "moderation_risk": assessment.get("moderation_risk"),
            "moderation_reasons": assessment.get("moderation_reasons"),
            "proceed_despite_moderation_risk": proceed_despite_moderation_risk,
        },
    )
    asyncio.create_task(
        _run_emotion_media_job(
            job_id,
            user_id,
            assistant_id,
            reference_image_data_uri,
            only_missing,
            emotions=(emotion,) if emotion else None,
            asset_kinds=(asset_kind,) if asset_kind else None,
            extra_prompt=extra_prompt,
            subject=subject,
            assessment=assessment,
            proceed_despite_moderation_risk=proceed_despite_moderation_risk,
        )
    )
    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status_url": f"/avatar_media_jobs/{job_id}"},
    )


@app.delete("/avatar_emotion_media/{asset_id}")
async def delete_avatar_emotion_media_asset(
    asset_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete one generated emotion still or idle loop."""
    from src.anubis.utils.media_assets import get_media_asset_repository
    from src.anubis.utils.media_assets.repository import ASSET_KIND_LIP_SYNC

    repository = get_media_asset_repository()
    if repository is None:
        raise HTTPException(status_code=404, detail="No such asset.")
    asset = await repository.get_emotion_asset(asset_id)
    if asset is None or asset.get("asset_kind") == ASSET_KIND_LIP_SYNC:
        raise HTTPException(status_code=404, detail="No such asset.")
    await resolve_assistant_for_creator(
        asset["assistant_id"],
        current_user,
        action_description="delete generated media for that avatar",
    )
    await repository.delete_emotion_asset(asset_id)
    return Response(status_code=204)


@app.get("/avatar_media_jobs")
async def list_avatar_media_jobs(
    assistant_id: str,
    job_kind: str | None = None,
    current_user: dict = Depends(get_current_user),
):
    """List this avatar's durable media jobs, newest first."""
    from src.anubis.utils.media_assets import get_media_asset_repository

    await resolve_assistant_for_creator(
        assistant_id, current_user, action_description="list media jobs for that avatar"
    )
    repository = get_media_asset_repository()
    if repository is None:
        return JSONResponse({"jobs": []})
    jobs = await repository.list_jobs(assistant_id=assistant_id, job_kind=job_kind)
    user_id = current_user["identities"][0]["user_id"]
    return JSONResponse(
        {"jobs": [job for job in jobs if job.get("user_id") == user_id]}
    )


@app.get("/avatar_media_jobs/{job_id}")
async def get_avatar_media_job(
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    """State and progress of one durable media job (emotion media, voice clone)."""
    from src.anubis.utils.media_assets import get_media_asset_repository

    repository = get_media_asset_repository()
    if repository is None:
        raise HTTPException(status_code=404, detail="No such job.")
    job = await repository.get_job(job_id)
    if job is None or job.get("user_id") != current_user["identities"][0]["user_id"]:
        raise HTTPException(status_code=404, detail="No such job.")
    return JSONResponse(job)


async def _poll_training_voice_clones(context: Any) -> None:
    """Refresh every training professional clone until it reports a result."""
    from src.anubis.utils.media_assets import get_media_asset_repository
    from src.anubis.utils.voice.corpus import refresh_training_state

    interval = float(
        getattr(context, "professional_voice_clone_poll_interval_seconds", None)
        or 300.0
    )
    while True:
        try:
            await asyncio.sleep(interval)
            repository = get_media_asset_repository()
            if (
                repository is None
                or not hasattr(repository, "pool")
                or repository.pool is None
            ):
                continue
            async with repository.pool.connection() as connection:
                async with connection.cursor() as cursor:
                    await cursor.execute(
                        "SELECT assistant_id, user_id FROM avatar_voice "
                        "WHERE professional_state = 'training';"
                    )
                    rows = await cursor.fetchall()
            for assistant_id, user_id in rows:
                try:
                    await refresh_training_state(
                        repository, context, user_id=user_id, assistant_id=assistant_id
                    )
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "Training poll failed for %s", assistant_id, exc_info=True
                    )
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001
            logger.debug("Voice training poller iteration failed", exc_info=True)


def _voice_repository_or_503() -> Any:
    from src.anubis.utils.media_assets import get_media_asset_repository
    from src.anubis.utils.voice.corpus import voice_configured

    repository = get_media_asset_repository()
    if repository is None or not voice_configured(app.state.context):
        raise HTTPException(
            status_code=503, detail="Voice features are not configured."
        )
    return repository


# Metadata keys the avatar carries once ElevenLabs has banned its cloned voice.
# Written on the assistant rather than only on the voice row so every reader of
# the avatar knows, without a second request: the chat screen hides the speak
# button and live voice mode answers in text, exactly as they do for an avatar
# that has no voice audio model at all.
VOICE_MODEL_BLOCKED_METADATA_KEY = "voice_model_blocked"
VOICE_MODEL_BLOCKED_REASON_METADATA_KEY = "voice_model_blocked_reason"


async def note_blocked_voice_on_avatar(
    assistant_id: str, current_user: dict, reason: str | None = None
) -> None:
    """Record on the avatar itself that its cloned voice has been banned.

    Called wherever the ban is confirmed — the settings status read and the
    speak route. Idempotent: an avatar already carrying the note is left alone,
    so the common path costs one metadata read and no write. Never raises: the
    note makes the refusal legible, and failing to write it must not turn a
    handled refusal into a failed request.
    """
    from src.anubis.utils.voice.corpus import BLOCKED_VOICE_MESSAGE

    try:
        token = current_user["API_KEY"]
        client = get_client(headers={"API-KEY": f"{token}"})
        assistant = await client.assistants.get(assistant_id)
        metadata = assistant.get("metadata") or {}
        if metadata.get(VOICE_MODEL_BLOCKED_METADATA_KEY) is True:
            return
        # LangGraph merges metadata by key, so this adds the note and leaves
        # user_id, the personal-avatar flag and the sharing state alone.
        await client.assistants.update(
            assistant_id=assistant_id,
            metadata={
                VOICE_MODEL_BLOCKED_METADATA_KEY: True,
                VOICE_MODEL_BLOCKED_REASON_METADATA_KEY: (
                    reason or BLOCKED_VOICE_MESSAGE
                ),
            },
        )
        logger.warning(
            "Noted on avatar %s that its cloned voice is blocked by ElevenLabs",
            assistant_id,
        )
    except Exception:  # noqa: BLE001 - the note is advisory, never fatal
        logger.warning(
            "Could not note the blocked voice on avatar %s",
            assistant_id,
            exc_info=True,
        )


async def _owned_assistant_for_voice(
    assistant_id: str, current_user: dict, action: str
) -> tuple[dict, bool]:
    """Resolve an avatar the caller owns; report whether it is their personal avatar."""
    assistant, _creator = await resolve_assistant_for_creator(
        assistant_id, current_user, action_description=action
    )
    metadata = assistant.get("metadata") or {}
    return assistant, metadata.get("is_personal_avatar_of_creator") is True


@app.get("/avatar_voice")
async def get_avatar_voice(
    assistant_id: str,
    current_user: dict = Depends(get_current_user),
):
    """The avatar's voice status: collected seconds, clones, thresholds, active voice."""
    from src.anubis.utils.voice.corpus import voice_status_for

    repository = _voice_repository_or_503()
    _assistant, is_personal = await _owned_assistant_for_voice(
        assistant_id, current_user, "read that avatar's voice"
    )
    status = await voice_status_for(
        repository,
        app.state.context,
        user_id=current_user["identities"][0]["user_id"],
        assistant_id=assistant_id,
        is_personal_avatar=is_personal,
        store=app.state.store,
    )
    if status.instant_voice_blocked:
        await note_blocked_voice_on_avatar(
            assistant_id, current_user, status.instant_voice_blocked_reason
        )
    return JSONResponse(status.as_dict())


@app.post("/avatar_voice/samples")
async def add_avatar_voice_sample(
    assistant_id: str = Form(...),
    audio: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Add a recording of the avatar speaking to its voice corpus.

    The settings recorder posts each take here (the two-minute script, or a
    dropped file). The dominant speaker is isolated first, so a take with
    background voices contributes only the avatar's speech; the isolated clip
    is stored, the running total updated, and — once the minimum is reached —
    the instant clone is created. When no diarizer reference clip exists yet,
    the take also becomes it.
    """
    from src.anubis.utils.utility import isolate_dominant_speaker_audio_b64
    from src.anubis.utils.voice.corpus import add_voice_clip, voice_status_for
    from src.anubis.utils.voice.reference_audio import (
        read_usable_reference_audio,
        store_reference_audio,
    )
    from src.anubis.utils.voice.reference_eligibility import reference_clip_rejection

    repository = _voice_repository_or_503()
    enforce_tier_capability(current_user, TierCapability.UPLOAD)
    assistant, is_personal = await _owned_assistant_for_voice(
        assistant_id, current_user, "add voice samples to that avatar"
    )
    user_id = current_user["identities"][0]["user_id"]
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The recording is empty.")
    mime_type = audio.content_type or "audio/webm"
    data_uri = make_data_uri(mime_type, raw)
    sample_filename = audio.filename or "voice-sample.webm"

    try:
        # Recorder takes are one person talking; keep that single speaker so
        # the take yields a clip with a duration instead of the passthrough.
        isolated = await isolate_dominant_speaker_audio_b64(
            data_uri,
            context=app.state.context,
            filename=sample_filename,
            content_type=mime_type,
            reference_audio=False,
            allow_single_speaker=True,
        )
    except Exception as isolation_error:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"The recording could not be processed: {isolation_error}",
        )
    seconds = float(isolated.get("duration") or 0.0)
    clip_uri = isolated.get("audio_base64_preprocessed") or ""
    if seconds <= 0 or not clip_uri:
        raise HTTPException(
            status_code=400, detail="No speech was found in the recording."
        )

    await add_voice_clip(
        repository,
        app.state.context,
        user_id=user_id,
        assistant_id=assistant_id,
        audio_data_uri=clip_uri,
        duration_seconds=seconds,
        source="recorder",
        source_document_name=sample_filename,
        is_personal_avatar=is_personal,
        avatar_name=assistant.get("name") or "",
    )

    # The diarizer needs a short single-speaker anchor for later uploads; the
    # first take supplies it when nothing has been stored yet. The helper
    # stores a listable Document under the same filename as the clip above so
    # the document list can show which recording is the reference.
    stored_anchor, _anchor_problem = await read_usable_reference_audio(
        app.state.store, user_id, assistant_id, context=app.state.context
    )
    if stored_anchor is None:
        try:
            anchor = await isolate_dominant_speaker_audio_b64(
                data_uri,
                context=app.state.context,
                filename=sample_filename,
                content_type=mime_type,
                reference_audio=True,
            )
            # A take the isolation could not cut a single-speaker clip from is
            # not stored: an anchor the diarizer rejects is worse than no anchor,
            # because every later upload then fails to label anyone.
            anchor_rejection = reference_clip_rejection(
                audio_data_uri=anchor.get("audio_base64_preprocessed"),
                transcript_text=anchor.get("text"),
                duration_seconds=anchor.get("duration"),
                maximum_seconds=getattr(
                    app.state.context, "reference_audio_clip_max_seconds", None
                ),
                minimum_seconds=getattr(
                    app.state.context, "reference_audio_minimum_seconds", None
                ),
            )
            if anchor_rejection is not None:
                logger.info(
                    "No diarizer reference stored from %s for %s: %s",
                    sample_filename,
                    assistant_id,
                    anchor_rejection,
                )
            else:
                await store_reference_audio(
                    app.state.store,
                    user_id=user_id,
                    assistant_id=assistant_id,
                    audio_data_uri=anchor.get("audio_base64_preprocessed") or "",
                    transcript_text=anchor.get("text") or "",
                    filename=sample_filename,
                    namespace_filename=_namespace_safe_formatted_filename(
                        sample_filename
                    ),
                    duration_seconds=anchor.get("duration"),
                    source="recorder",
                    replace=False,
                )
        except Exception:  # noqa: BLE001
            logger.debug(
                "Could not store a diarizer reference from the recording", exc_info=True
            )

    status = await voice_status_for(
        repository,
        app.state.context,
        user_id=user_id,
        assistant_id=assistant_id,
        is_personal_avatar=is_personal,
        store=app.state.store,
    )
    return JSONResponse({"added_seconds": seconds, **status.as_dict()})


@app.post("/avatar_voice/reference")
async def set_avatar_voice_reference(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Point the avatar's reference clip at a different upload.

    Body: ``assistant_id``, ``source_document_name``. The named upload must
    already have speech in the voice corpus; the longest stored clip cut from
    that upload is re-isolated into a short single-speaker reference and
    replaces the stored reference. Nothing else changes: the avatar's identity
    documents and the trained voice are untouched.
    """
    from src.anubis.utils.utility import isolate_dominant_speaker_audio_b64
    from src.anubis.utils.voice.corpus import (
        clip_data_uri,
        longest_clip_for_document,
        voice_status_for,
    )
    from src.anubis.utils.voice.reference_audio import store_reference_audio
    from src.anubis.utils.voice.reference_eligibility import reference_clip_rejection

    repository = _voice_repository_or_503()
    body = await request.json()
    body = body if isinstance(body, dict) else {}
    assistant_id = str(body.get("assistant_id") or "").strip()
    source_document_name = str(body.get("source_document_name") or "").strip()
    if not assistant_id or not source_document_name:
        raise HTTPException(
            status_code=400,
            detail="assistant_id and source_document_name are required.",
        )
    _assistant, is_personal = await _owned_assistant_for_voice(
        assistant_id, current_user, "change that avatar's reference audio"
    )
    user_id = current_user["identities"][0]["user_id"]

    clip = await longest_clip_for_document(
        repository, assistant_id, source_document_name
    )
    if clip is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No stored speech for {source_document_name}. Only uploads that "
                "contributed to the voice can become the reference."
            ),
        )
    try:
        anchor = await isolate_dominant_speaker_audio_b64(
            clip_data_uri(clip),
            context=app.state.context,
            filename=source_document_name,
            content_type=clip.get("mime_type") or "audio/mpeg",
            reference_audio=True,
            allow_single_speaker=True,
        )
    except Exception as isolation_error:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"The clip could not be prepared as a reference: {isolation_error}",
        )
    anchor_uri = anchor.get("audio_base64_preprocessed") or ""
    anchor_rejection = reference_clip_rejection(
        audio_data_uri=anchor_uri,
        transcript_text=anchor.get("text"),
        duration_seconds=anchor.get("duration"),
        maximum_seconds=getattr(
            app.state.context, "reference_audio_clip_max_seconds", None
        ),
        minimum_seconds=getattr(
            app.state.context, "reference_audio_minimum_seconds", None
        ),
    )
    if anchor_rejection is not None:
        raise HTTPException(status_code=400, detail=anchor_rejection)
    await store_reference_audio(
        app.state.store,
        user_id=user_id,
        assistant_id=assistant_id,
        audio_data_uri=anchor_uri,
        transcript_text=anchor.get("text") or "",
        filename=source_document_name,
        namespace_filename=_namespace_safe_formatted_filename(source_document_name),
        duration_seconds=anchor.get("duration"),
        source="owner_selection",
        replace=True,
    )
    status = await voice_status_for(
        repository,
        app.state.context,
        user_id=user_id,
        assistant_id=assistant_id,
        is_personal_avatar=is_personal,
        store=app.state.store,
    )
    return JSONResponse(status.as_dict())


@app.get("/avatar_voice/verification")
async def get_avatar_voice_verification(
    assistant_id: str,
    current_user: dict = Depends(get_current_user),
):
    """The CAPTCHA the owner reads aloud to verify the professional voice."""
    from src.anubis.utils.voice import elevenlabs_client
    from src.anubis.utils.voice.corpus import prepare_professional_voice

    repository = _voice_repository_or_503()
    assistant, is_personal = await _owned_assistant_for_voice(
        assistant_id, current_user, "verify that avatar's voice"
    )
    if not is_personal:
        raise HTTPException(
            status_code=403,
            detail="Professional voice cloning is for the personal avatar.",
        )
    user_id = current_user["identities"][0]["user_id"]
    record = await prepare_professional_voice(
        repository,
        app.state.context,
        user_id=user_id,
        assistant_id=assistant_id,
        avatar_name=assistant.get("name") or "",
    )
    if record.get("professional_state") != "awaiting_verification":
        raise HTTPException(
            status_code=409,
            detail=(
                f"The professional voice is {record.get('professional_state')}; "
                "verification is only offered while it awaits verification."
            ),
        )
    try:
        captcha = await elevenlabs_client.get_verification_captcha(
            app.state.context, voice_id=record["professional_voice_id"]
        )
    except elevenlabs_client.ElevenLabsError as vendor_error:
        raise HTTPException(status_code=502, detail=str(vendor_error))
    return JSONResponse(
        {"voice_id": record["professional_voice_id"], "captcha": captcha}
    )


@app.post("/avatar_voice/verification")
async def submit_avatar_voice_verification(
    assistant_id: str = Form(...),
    recording: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Submit the owner's spoken CAPTCHA and start professional training."""
    from src.anubis.utils.voice import elevenlabs_client
    from src.anubis.utils.voice.corpus import submit_verification_and_train

    repository = _voice_repository_or_503()
    _assistant, is_personal = await _owned_assistant_for_voice(
        assistant_id, current_user, "verify that avatar's voice"
    )
    if not is_personal:
        raise HTTPException(
            status_code=403,
            detail="Professional voice cloning is for the personal avatar.",
        )
    raw = await recording.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The recording is empty.")
    try:
        record = await submit_verification_and_train(
            repository,
            app.state.context,
            user_id=current_user["identities"][0]["user_id"],
            assistant_id=assistant_id,
            recording=(
                recording.filename or "captcha.webm",
                raw,
                recording.content_type or "audio/webm",
            ),
        )
    except ValueError as state_error:
        raise HTTPException(status_code=409, detail=str(state_error))
    except elevenlabs_client.ElevenLabsError as vendor_error:
        raise HTTPException(status_code=502, detail=str(vendor_error))
    return JSONResponse(
        {
            "professional_state": record.get("professional_state"),
            "training_started_at": record.get("training_started_at"),
        }
    )


@app.post("/avatar_voice/professional/retry")
async def retry_avatar_professional_voice(
    assistant_id: str = Form(...),
    current_user: dict = Depends(get_current_user),
):
    """Retry professional clone preparation after the vendor refused it.

    ElevenLabs offers professional voice cloning to accounts on the Creator plan
    or above; a refusal parks the voice in ``plan_required`` and nothing retries
    by itself. Once the ElevenLabs account is upgraded (or a transient vendor
    failure has passed) the owner retries from the Voice panel.
    """
    from src.anubis.utils.voice.corpus import retry_professional_voice

    repository = _voice_repository_or_503()
    assistant, is_personal = await _owned_assistant_for_voice(
        assistant_id, current_user, "retry that avatar's professional voice"
    )
    if not is_personal:
        raise HTTPException(
            status_code=403,
            detail="Professional voice cloning is for the personal avatar.",
        )
    record = await retry_professional_voice(
        repository,
        app.state.context,
        user_id=current_user["identities"][0]["user_id"],
        assistant_id=assistant_id,
        avatar_name=assistant.get("name") or "",
    )
    return JSONResponse(
        {
            "professional_state": record.get("professional_state"),
            "detail": {
                key: value
                for key, value in (record.get("detail") or {}).items()
                if key.startswith("professional_")
            },
        }
    )


@app.post("/avatar_voice/rebuild")
async def rebuild_avatar_voice(
    assistant_id: str = Form(...),
    current_user: dict = Depends(get_current_user),
):
    """Delete the avatar's cloned voice and train a new one from the speech held now.

    The first instant clone is otherwise final, so a voice trained from the
    wrong speech — a clip cut from an arbitrary video, a recording of somebody
    else, or a clone the vendor has since banned — stays the avatar's voice for
    good. This deletes the vendor's copy and trains a replacement from the
    clips the avatar holds at this moment, which is why deleting the uploads
    that fed the bad voice first is how the owner chooses what the new voice is
    trained from.

    The response is the ordinary voice status. When the remaining speech is
    below the minimum the reply simply carries no voice model: the clone is
    then built by itself as soon as enough speech is collected again.
    """
    from src.anubis.utils.voice.corpus import rebuild_instant_voice, voice_status_for

    repository = _voice_repository_or_503()
    enforce_tier_capability(current_user, TierCapability.UPLOAD)
    assistant, is_personal = await _owned_assistant_for_voice(
        assistant_id, current_user, "rebuild that avatar's voice"
    )
    user_id = current_user["identities"][0]["user_id"]
    await rebuild_instant_voice(
        repository,
        app.state.context,
        user_id=user_id,
        assistant_id=assistant_id,
        avatar_name=assistant.get("name") or "",
    )
    status = await voice_status_for(
        repository,
        app.state.context,
        user_id=user_id,
        assistant_id=assistant_id,
        is_personal_avatar=is_personal,
        store=app.state.store,
    )
    return JSONResponse(status.as_dict())


@app.post("/transcribe")
async def transcribe_recording(
    assistant_id: str = Form(...),
    audio: UploadFile = File(...),
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    """Turn one spoken utterance into text (dictation and live-audio turns)."""
    from src.anubis.utils.utility import transcribe_audio

    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The recording is empty.")
    mime_type = audio.content_type or "audio/webm"
    started = time.perf_counter()
    try:
        result = await transcribe_audio(
            make_data_uri(mime_type, raw),
            app.state.context,
            filename=audio.filename or "utterance.webm",
            reference_audio=False,
            max_duration_seconds=None,
            live_voice=True,
        )
    except Exception as transcription_error:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"The recording could not be transcribed: {transcription_error}",
        )
    text = str(result.get("text") or "").strip()
    try:
        await persist_api_metrics_row(
            app.state.pool,
            inference_type="transcription",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            user_id=current_user["identities"][0]["user_id"],
            assistant_id=assistant_id,
            model_name=getattr(app.state.context, "audio_transcription_model", None),
        )
    except Exception:  # noqa: BLE001
        logger.debug("Could not record transcription metrics", exc_info=True)
    return JSONResponse(
        {"text": text, "duration_seconds": result.get("file_duration_s")}
    )


@app.post("/speak")
async def speak_text(
    request: Request,
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    """Render text in the avatar's cloned voice and return the audio.

    Body: ``assistant_id``, ``text``. Uses the professional clone once it is
    fine-tuned, otherwise the instant clone; with neither, answers 409
    ``voice_not_ready`` and the collected seconds so the client can prompt the
    owner to record. A clone ElevenLabs has banned answers 409 ``voice_blocked``
    — a distinct condition from having no clone, and one no amount of further
    recording fixes. Characters spoken are recorded in ``api_metrics`` and, when
    the meter exists, reported to Stripe.
    """
    from src.anubis.utils.voice import elevenlabs_client
    from src.anubis.utils.voice.corpus import (
        BLOCKED_VOICE_MESSAGE,
        mark_voice_blocked,
        resolve_active_voice_id,
        voice_record_blocked,
        voice_record_blocked_reason,
        voice_status_for,
    )

    repository = _voice_repository_or_503()
    body = await request.json()
    body = body if isinstance(body, dict) else {}
    assistant_id = str(body.get("assistant_id") or "").strip()
    text = str(body.get("text") or "").strip()
    if not assistant_id or not text:
        raise HTTPException(
            status_code=400, detail="assistant_id and text are required."
        )
    if len(text) > 5000:
        text = text[:5000]
    enforce_tier_capability(current_user, TierCapability.AUDIO_RESPONSES)

    user_id = current_user["identities"][0]["user_id"]

    def _blocked_voice_response(reason: str | None) -> JSONResponse:
        """Build the one answer every blocked-voice path returns."""
        return JSONResponse(
            status_code=409,
            content={
                "error": "voice_blocked",
                "detail": reason or BLOCKED_VOICE_MESSAGE,
            },
        )

    kind, voice_id = await resolve_active_voice_id(repository, assistant_id)
    # A voice already known to be banned is refused without calling the vendor:
    # the answer cannot change, and the call would be billed for a 403.
    stored_voice = await repository.get_voice(assistant_id) or {}
    if voice_id is not None and voice_record_blocked(stored_voice):
        reason = voice_record_blocked_reason(stored_voice)
        await note_blocked_voice_on_avatar(assistant_id, current_user, reason)
        return _blocked_voice_response(reason)
    if voice_id is None:
        status = await voice_status_for(
            repository,
            app.state.context,
            user_id=user_id,
            assistant_id=assistant_id,
            is_personal_avatar=False,
        )
        return JSONResponse(
            status_code=409,
            content={
                "error": "voice_not_ready",
                "detail": (
                    "This avatar has no cloned voice yet. Record about two minutes of "
                    "the avatar speaking in settings to create one."
                ),
                "collected_seconds": status.collected_seconds,
                "instant_minimum_seconds": status.instant_minimum_seconds,
            },
        )

    model_id = str(
        getattr(app.state.context, "elevenlabs_text_to_speech_model", None)
        or "eleven_flash_v2_5"
    )
    started = time.perf_counter()
    try:
        audio_bytes = await elevenlabs_client.synthesize_speech(
            app.state.context, voice_id=voice_id, text=text, model_id=model_id
        )
    except elevenlabs_client.ElevenLabsVoiceBlockedError as blocked_error:
        # The ban was applied between the last safety check and this request.
        # Record it so the settings Voice panel stops advertising the voice and
        # later speak attempts are refused without a vendor round trip.
        await mark_voice_blocked(
            repository, user_id, assistant_id, reason=str(blocked_error)
        )
        await note_blocked_voice_on_avatar(
            assistant_id, current_user, str(blocked_error)
        )
        return _blocked_voice_response(str(blocked_error))
    except elevenlabs_client.ElevenLabsError as vendor_error:
        raise HTTPException(status_code=502, detail=str(vendor_error))

    cost_per_thousand = float(
        getattr(
            app.state.context,
            "elevenlabs_text_to_speech_cost_per_1000_characters_usd",
            None,
        )
        or 0.05
    )
    await _meter_speech_characters(
        current_user,
        assistant_id=assistant_id,
        characters=len(text),
        cost_usd=cost_per_thousand * len(text) / 1000.0,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        model_name=model_id,
    )
    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={"X-Voice-Kind": kind, "Cache-Control": "no-store"},
    )


async def _meter_speech_characters(
    current_user: dict,
    *,
    assistant_id: str,
    characters: int,
    cost_usd: float,
    latency_ms: float,
    model_name: str,
) -> None:
    """Record speech spend locally and, when the meter exists, to Stripe."""
    try:
        await persist_api_metrics_row(
            app.state.pool,
            inference_type="speech_synthesis",
            total_tokens=characters,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            user_id=current_user["identities"][0]["user_id"],
            assistant_id=assistant_id,
            model_name=model_name,
            meter_event_name=getattr(
                getattr(UsageMeter, "SPEECH_CHARACTERS", None), "value", None
            ),
        )
    except Exception:  # noqa: BLE001
        logger.debug("Could not record speech metrics", exc_info=True)
    speech_meter = getattr(UsageMeter, "SPEECH_CHARACTERS", None)
    if speech_meter is None:
        return
    try:
        stripe_customer_id = await resolve_stripe_customer_id(app.state, current_user)
        await report_meter_event(
            app.state.stripe, speech_meter, stripe_customer_id, int(characters)
        )
    except Exception:  # noqa: BLE001
        logger.debug("Could not report speech meter event", exc_info=True)


@app.post("/lip_sync")
async def start_lip_sync_clip(
    request: Request,
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    """Render a lip-synced video of the avatar saying a reply.

    Body: ``assistant_id``, ``text``, ``emotion``. Premium capability
    (``VIDEO_RESPONSES``) and the ``LIP_SYNC_ENABLED`` switch both apply. A
    phrase already rendered for this emotion is answered ``completed`` at once;
    otherwise a generation starts and ``GET /lip_sync/{generation_id}`` is
    polled. Seconds are metered when the clip completes.
    """
    from src.anubis.utils.media_generation.lip_sync import (
        lip_sync_enabled,
        start_lip_sync,
    )
    from src.anubis.utils.voice import elevenlabs_client
    from src.anubis.utils.voice.corpus import resolve_active_voice_id

    repository = _voice_repository_or_503()
    if not lip_sync_enabled(app.state.context):
        raise HTTPException(status_code=503, detail="Video replies are not enabled.")
    body = await request.json()
    body = body if isinstance(body, dict) else {}
    assistant_id = str(body.get("assistant_id") or "").strip()
    text = str(body.get("text") or "").strip()[:2000]
    emotion = str(body.get("emotion") or "neutral").strip().lower() or "neutral"
    if not assistant_id or not text:
        raise HTTPException(
            status_code=400, detail="assistant_id and text are required."
        )
    enforce_tier_capability(current_user, TierCapability.VIDEO_RESPONSES)

    _kind, voice_id = await resolve_active_voice_id(repository, assistant_id)
    if voice_id is None:
        raise HTTPException(
            status_code=409,
            detail="This avatar has no cloned voice yet; record one in settings.",
        )
    try:
        result = await start_lip_sync(
            app.state.context,
            repository,
            user_id=current_user["identities"][0]["user_id"],
            assistant_id=assistant_id,
            text=text,
            emotion=emotion,
            voice_id=voice_id,
        )
    except elevenlabs_client.ElevenLabsError as vendor_error:
        raise HTTPException(status_code=502, detail=str(vendor_error))
    if result["status"] == "completed":
        return JSONResponse(
            {
                "status": "completed",
                "video_url": f"/avatar_emotion_media/{result['asset_id']}",
                "cached": True,
            }
        )
    return JSONResponse(
        status_code=202,
        content={"status": "pending", "generation_id": result["job_id"]},
    )


@app.get("/lip_sync/{generation_id}")
async def get_lip_sync_clip(
    generation_id: str,
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    """Status of a lip-sync generation; ``video_url`` once the clip is stored."""
    from src.anubis.utils.media_generation.lip_sync import poll_lip_sync
    from src.anubis.utils.voice import elevenlabs_client

    repository = _voice_repository_or_503()
    job = await repository.get_job(generation_id)
    if job is None or job.get("job_kind") != "lip_sync":
        raise HTTPException(status_code=404, detail="No such generation.")
    try:
        result = await poll_lip_sync(app.state.context, repository, job=job)
    except elevenlabs_client.ElevenLabsError as vendor_error:
        raise HTTPException(status_code=502, detail=str(vendor_error))
    if result.get("newly_completed"):
        detail = job.get("detail") or {}
        seconds = float(detail.get("estimated_seconds") or 0.0)
        await _meter_video_seconds(
            current_user,
            assistant_id=job["assistant_id"],
            seconds=seconds,
            cost_usd=float(
                getattr(
                    app.state.context, "elevenlabs_lip_sync_cost_per_second_usd", None
                )
                or 0.14
            )
            * seconds,
        )
    if result["status"] == "completed":
        return JSONResponse(
            {
                "status": "completed",
                "video_url": f"/avatar_emotion_media/{result['asset_id']}",
            }
        )
    return JSONResponse({"status": result["status"]})


async def _meter_video_seconds(
    current_user: dict, *, assistant_id: str, seconds: float, cost_usd: float
) -> None:
    """Record lip-sync spend locally and, when the meter exists, to Stripe."""
    video_meter = getattr(UsageMeter, "VIDEO_GENERATION_SECONDS", None)
    try:
        await persist_api_metrics_row(
            app.state.pool,
            inference_type="lip_sync",
            total_tokens=int(round(seconds)),
            cost_usd=cost_usd,
            user_id=current_user["identities"][0]["user_id"],
            assistant_id=assistant_id,
            model_name=getattr(app.state.context, "elevenlabs_lip_sync_model", None),
            meter_event_name=getattr(video_meter, "value", None),
        )
    except Exception:  # noqa: BLE001
        logger.debug("Could not record lip-sync metrics", exc_info=True)
    if video_meter is None:
        return
    try:
        stripe_customer_id = await resolve_stripe_customer_id(app.state, current_user)
        await report_meter_event(
            app.state.stripe,
            video_meter,
            stripe_customer_id,
            max(1, int(round(seconds))),
        )
    except Exception:  # noqa: BLE001
        logger.debug("Could not report video meter event", exc_info=True)


def _inbox_repository_or_503() -> Any:
    from src.anubis.utils.inbox import get_inbox_repository

    repository = get_inbox_repository()
    if repository is None:
        raise HTTPException(
            status_code=503, detail="The agent inbox is not configured."
        )
    return repository


@app.get("/inbox/items")
async def list_inbox_items(
    request: Request,
    state: str = "open",
    limit: int = 50,
    current_user: dict = Depends(get_current_user),
):
    """The owner's inbox items: ``state=open`` (default), ``all``, or one state."""
    from src.anubis.utils.inbox.repository import OPEN_STATES, public_item_view

    repository = _inbox_repository_or_503()
    token = current_user["API_KEY"]
    client = get_client(headers={"API-KEY": f"{token}"})
    personal_avatar = await _resolve_personal_avatar_for_connection(
        client, request, current_user, token
    )
    assistant_id = personal_avatar.get("assistant_id")
    states = None
    if state == "open":
        states = OPEN_STATES
    elif state and state != "all":
        states = (state,)
    items = await repository.list_items(
        assistant_id=assistant_id, states=states, limit=max(1, min(int(limit), 200))
    )
    return JSONResponse(
        {
            "personal_avatar_id": assistant_id,
            "pending_count": await repository.count_open(assistant_id),
            "items": [public_item_view(item) for item in items],
        }
    )


@app.get("/inbox/count")
async def inbox_count(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """How many items await the owner — the badge."""
    repository = _inbox_repository_or_503()
    token = current_user["API_KEY"]
    client = get_client(headers={"API-KEY": f"{token}"})
    personal_avatar = await _resolve_personal_avatar_for_connection(
        client, request, current_user, token
    )
    return JSONResponse(
        {
            "pending_count": await repository.count_open(
                personal_avatar.get("assistant_id")
            )
        }
    )


@app.post("/inbox/items/{item_id}/decide")
async def decide_inbox_item(
    item_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Deliver the owner's decision to a pending item.

    Body: ``{"type": "accept"|"edit"|"ignore"|"response", "args": ...}`` — the
    Agent Inbox ``HumanResponse`` shape. ``edit`` carries
    ``{"action":"send_reply","args":{"subject","body"}}``; ``response`` carries
    free text. The paused graph resumes, sends when the decision says so, and
    records the decision as a preference for this sender and kind of message.
    """
    from src.anubis.utils.inbox.poller import resume_inbox_item
    from src.anubis.utils.inbox.repository import public_item_view

    repository = _inbox_repository_or_503()
    token = current_user["API_KEY"]
    client = get_client(headers={"API-KEY": f"{token}"})
    personal_avatar = await _resolve_personal_avatar_for_connection(
        client, request, current_user, token
    )
    item = await repository.get_item(item_id)
    if item is None or item.get("assistant_id") != personal_avatar.get("assistant_id"):
        raise HTTPException(status_code=404, detail="No such inbox item.")
    body = await request.json()
    body = body if isinstance(body, dict) else {}
    decision_type = str(body.get("type") or "").strip().lower()
    if decision_type not in ("accept", "edit", "ignore", "response"):
        raise HTTPException(
            status_code=400, detail="type must be accept, edit, ignore, or response."
        )
    human_response = {"type": decision_type, "args": body.get("args")}
    updated = await resume_inbox_item(
        app.state.context, item_id=item_id, human_response=human_response
    )
    return JSONResponse({"item": public_item_view(updated or item)})


@app.post("/inbox/poll")
async def poll_inbox_now(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Check the owner's connected mailboxes now and triage anything new."""
    from src.anubis.utils.inbox.poller import poll_connected_mailboxes

    _inbox_repository_or_503()
    token = current_user["API_KEY"]
    client = get_client(headers={"API-KEY": f"{token}"})
    await _resolve_personal_avatar_for_connection(client, request, current_user, token)
    result = await poll_connected_mailboxes(
        app.state.context, only_user_id=current_user["identities"][0]["user_id"]
    )
    return JSONResponse(result)


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
    langgraph_client_headers = {"API-KEY": current_user["API_KEY"]}
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


def _rejected_media_item(source_name: str, error: BaseException) -> dict:
    """Describe one upload item that could not be turned into a media entry.

    Entry building validates each item (media type, CSV/JSON shape, remote
    fetch) and raises on the ones it cannot process. In a multi-item request
    every other item is still perfectly processable, so the failure is recorded
    as data and returned to the caller instead of aborting the whole batch.
    """
    if isinstance(error, HTTPException):
        return {
            "filename": source_name,
            "reason": str(error.detail),
            "status_code": error.status_code,
        }
    return {
        "filename": source_name,
        "reason": f"{type(error).__name__}: {error}",
        "status_code": 400,
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

    The declared type is resolved against the file's magic bytes first, so an
    upload whose client could not name the extension (a ``.webp`` picture sent
    as ``application/octet-stream``) is routed as the image it is rather than
    rejected as an unsupported Content-Type.
    """
    entries: list = []
    mime_type = effective_upload_mime_type(mime_type, content)
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
    elif (
        not reference_image
        and not reference_audio
        and _is_json_upload(raw_name, mime_type)
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
                    "namespace_filename": raw_name
                    if not "." in raw_name
                    else _namespace_safe_formatted_filename(raw_name),
                }
            )
    elif reference_image:
        if mime_type.startswith("audio/"):
            raise HTTPException(
                status_code=400,
                detail="reference_image requires an image file, not audio.",
            )
        mime, content = prepare_still_image_upload(mime_type, content)
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
        effective = mime_type
        if not effective.startswith("audio/") and not effective.startswith("video/"):
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
        if mime_type.startswith("image/"):
            mime, content = prepare_still_image_upload(mime_type, content)
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
        elif mime_type.startswith("audio/"):
            effective = mime_type
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
            effective = mime_type
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
        # A PDF uploaded without a usable declaration (extension-less file, or
        # a client that sends application/octet-stream) resolves to
        # application/pdf on the %PDF magic above, so it is ingested as a PDF
        # instead of falling through to the plain-text branch as binary text.
        elif mime_type == "application/pdf":
            effective = "application/pdf"
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
        video_media_seconds = (
            round(video_duration_seconds, 1)
            if video_duration_seconds > 0
            else ESTIMATED_AUDIO_FALLBACK_DURATION_SECONDS
        )
        media_entries.append(
            {
                "filename": f"{playlist_label}::{video_title or watch_url}",
                "estimated_tokens": video_estimated_tokens,
                "estimated_media_seconds": video_media_seconds,
                "estimated_processing_seconds": estimate_processing_seconds(
                    video_media_seconds,
                    GlobalContext().media_preprocessing_seconds_per_media_second,
                ),
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
        img_mime, body = prepare_still_image_upload(header_ct, body)
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
            img_mime, body = prepare_still_image_upload(header_ct, body)
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


def _stamp_media_time_estimate(
    entry: dict, duration_seconds: float | None, context: GlobalContext
) -> None:
    """Record a speech item's probed length and expected processing time.

    ``estimated_media_seconds`` is the audio/video length the token estimate was
    computed from (the sanctioned fallback when the probe failed);
    ``estimated_processing_seconds`` is that length times
    ``media_preprocessing_seconds_per_media_second``. Both ride on the entry into
    ``create_child_job`` so the upload card and every progress frame can show
    "about N min" and time remaining.
    """
    entry["estimated_media_seconds"] = (
        round(float(duration_seconds), 1) if duration_seconds else None
    )
    entry["estimated_processing_seconds"] = estimate_processing_seconds(
        entry["estimated_media_seconds"],
        context.media_preprocessing_seconds_per_media_second,
    )


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
        _stamp_media_time_estimate(entry, effective_duration, context)
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
            probe_timeout_seconds = float(
                context.remote_duration_probe_timeout_seconds or 30.0
            )
            try:
                duration_seconds = await asyncio.wait_for(
                    get_remote_video_duration_seconds(page_url),
                    timeout=probe_timeout_seconds,
                )
            except Exception as duration_error:  # noqa: BLE001 - sanctioned fallback
                # ``str(asyncio.TimeoutError())`` is empty, so name the failure
                # class as well; an empty "()" in this line is a probe timeout.
                logger.warning(
                    "Remote duration probe failed for %s (%s: %s; limit %.0f s); assuming %.0f seconds.",
                    page_url,
                    type(duration_error).__name__,
                    duration_error,
                    probe_timeout_seconds,
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
                        timeout=float(
                            context.remote_duration_probe_timeout_seconds or 30.0
                        ),
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

    estimates = await asyncio.gather(*(_estimate_one(entry) for entry in media_files))
    return sum(estimates)


async def _start_media_batch(
    *,
    user_id: str,
    assistant_id: str,
    config: dict,
    media_files: list,
    playlist_urls: list[str],
    rejected_items: list[dict],
    current_user: dict,
    create_reference_media_from_playlist: bool = False,
) -> dict:
    """Estimate, enforce, meter, and start one media batch; return the 202 body.

    Shared by ``POST /update_avatar_identity_with_media`` and the in-chat
    ``update_avatar_identity_with_media`` tool (``start_identity_media_job_from_chat``)
    so both paths bill and run media identically. Raises ``HTTPException`` when
    the allotment or rate limit refuses the batch.
    """
    # A reference image in this batch stores the portrait only. Emotion
    # stills and idle loops wait for POST /avatar_emotion_media/regenerate.
    # The submitter's tier still travels with the run so an explicit
    # generate later can read it.
    config.setdefault("configurable", {})["subscription_tier"] = resolve_tier(
        current_user
    ).value
    config["configurable"]["emotion_media_minimum_tier"] = (
        resolve_emotion_media_minimum_tier().value
    )

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
            media_upload_rate_limit_context.media_upload_rate_limit_window_seconds or 0
        ),
        tokens_per_window=int(
            media_upload_rate_limit_context.media_upload_rate_limit_tokens_per_window
            or 0
        ),
        estimated_request_tokens=estimated_tokens_total,
    )

    # Report the estimate against the document-upload meter (Stripe +
    # local api_metrics). Billing WRITES stay fail-open — only estimation
    # is fail-closed. The admin testing account is never metered; a dev
    # enforcement-only bypass still is.
    upload_metering_bypass = resolve_metering_bypass(current_user)
    if not upload_metering_bypass.skips_metering_writes and estimated_tokens_total > 0:
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

    # Read what this avatar already holds so the media graph can skip items it
    # has seen before — see indexed_namespace_filenames for which rows count.
    try:
        existing_items = await store.asearch((user_id, assistant_id), limit=1_000_000)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not read this avatar's existing media to skip "
                f"already-indexed items: {exc}"
            ),
        ) from exc

    existing_namespaces = indexed_namespace_filenames(existing_items)

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
            estimated_media_seconds=media_file.get("estimated_media_seconds"),
            estimated_processing_seconds=media_file.get("estimated_processing_seconds"),
        )
        master.child_ids.append(child.job_id)
        items.append({"child": child, "media_file": media_file})
        item_descriptors.append(
            {
                "job_id": child.job_id,
                "filename": child.filename,
                "status": child.status,
                "estimated_tokens": media_file.get("estimated_tokens"),
                "estimated_media_seconds": child.estimated_media_seconds,
                "estimated_processing_seconds": child.estimated_processing_seconds,
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
    return {
        "job_id": master.job_id,
        "status": master.status,
        "status_url": f"/media_job/{master.job_id}",
        "progress_url": f"/media_job/{master.job_id}/progress",
        "cancel_url": f"/media_job/{master.job_id}/cancel",
        "items_accepted": len(media_files),
        "filenames": [m.get("filename") for m in media_files],
        "items": item_descriptors,
        # Items that could not be turned into a job (mislabeled media
        # type, unreachable URL, malformed CSV). The rest of the batch
        # still runs; these are reported so the caller can fix and
        # re-upload just the ones that were skipped.
        "items_rejected": len(rejected_items),
        "rejected": rejected_items,
        # Playlists resolve to their per-video child jobs in the background;
        # those child ids surface on the master's progress stream as
        # ``playlist_child_added`` events rather than in this response.
        "playlists_expanding": len(playlist_urls),
        # The full pre-request estimate (playlist videos included) that
        # was checked against the allotment and reported to the meter,
        # plus where the caller now stands against the allotment.
        "estimated_tokens_total": estimated_tokens_total,
        "estimated_media_seconds_total": job_estimated_media_seconds(registry, master),
        "estimated_processing_seconds_total": job_estimated_processing_seconds(
            registry, master
        ),
        "usage": await _build_meter_usage_snapshot(
            app.state, current_user, UsageMeter.DOCUMENT_UPLOAD_TOKENS
        ),
        **upload_metering_bypass.usage_response_fields(),
        "message": (
            "Media processing started"
            + (
                f"; enumerating {len(playlist_urls)} playlist(s) in the background"
                if playlist_urls
                else ""
            )
            + (
                f"; skipped {len(rejected_items)} unprocessable item(s)"
                if rejected_items
                else ""
            )
        ),
    }


async def start_identity_media_job_from_chat(
    *,
    user_id: str,
    assistant_id: str,
    assistant_ctx: dict,
    current_user: dict,
    attachments: list,
    urls: list[str],
    reference_image: bool = False,
    reference_audio: bool = False,
) -> dict:
    """Start the media batch the in-chat identity-update tool asked for.

    ``attachments`` are ``TurnAttachment`` records of the current turn; ``urls``
    come from the conversation. Builds the same media entries the upload
    endpoint builds, then hands them to ``_start_media_batch``. Refusals
    (allotment, rate limit, every item rejected) come back as a ``status``
    dictionary rather than raising, because the caller is a tool inside a turn.
    """
    config = {
        "configurable": {
            "user_id": user_id,
            "user_ctx": {"name": None, "description": None},
            "assistant_id": assistant_id,
            "assistant_ctx": {
                "name": assistant_ctx.get("name"),
                "description": assistant_ctx.get("description"),
                "assistant_id": assistant_id,
                "metadata": assistant_ctx.get("metadata") or {},
            },
        }
    }
    # The same two gates ``POST /update_avatar_identity_with_media`` applies:
    # only the avatar's creator, on a tier with the UPLOAD capability. The
    # graph already offers the tool to the creator alone; this is the check
    # that holds even if a tool call arrives some other way.
    if (assistant_ctx.get("metadata") or {}).get("user_id") != user_id:
        return {
            "status": "refused",
            "status_code": 403,
            "detail": "Only the avatar's creator can update its identity.",
        }
    try:
        enforce_tier_capability(current_user, TierCapability.UPLOAD)
    except HTTPException as refusal:
        return {
            "status": "refused",
            "status_code": refusal.status_code,
            "detail": refusal.detail,
        }
    reference_mode = bool(reference_image or reference_audio)
    media_files: list = []
    rejected_items: list[dict] = []
    playlist_urls: list[str] = []
    for attachment in attachments:
        try:
            media_files.extend(
                await _build_media_entries_for_file(
                    attachment.filename,
                    attachment.content,
                    attachment.mime_type,
                    reference_image=bool(reference_image),
                    reference_audio=bool(reference_audio),
                    user_id=user_id,
                    assistant_id=assistant_id,
                )
            )
        except Exception as file_entry_error:  # noqa: BLE001 - per-item skip
            rejected_items.append(
                _rejected_media_item(attachment.filename, file_entry_error)
            )
    single_url = len(attachments) == 0 and len(urls) == 1
    for url in urls:
        if _is_youtube_playlist_url_str(url) and not reference_mode:
            playlist_urls.append(url)
            continue
        try:
            media_files.extend(
                await _build_media_entries_for_url(
                    url,
                    reference_image=bool(reference_image),
                    reference_audio=bool(reference_audio),
                    user_id=user_id,
                    assistant_id=assistant_id,
                    rich=single_url or reference_mode,
                )
            )
        except Exception as url_entry_error:  # noqa: BLE001 - per-item skip
            rejected_items.append(_rejected_media_item(url, url_entry_error))
    if not media_files and not playlist_urls:
        return {
            "status": "rejected",
            "detail": "Nothing could be processed.",
            "rejected": rejected_items,
        }
    try:
        started = await _start_media_batch(
            user_id=user_id,
            assistant_id=assistant_id,
            config=config,
            media_files=media_files,
            playlist_urls=playlist_urls,
            rejected_items=rejected_items,
            current_user=current_user,
        )
    except HTTPException as refusal:
        return {
            "status": "refused",
            "status_code": refusal.status_code,
            "detail": refusal.detail,
        }
    return {
        "status": "started",
        "job_id": started.get("job_id"),
        "items_accepted": started.get("items_accepted", 0),
        "filenames": started.get("filenames", []),
        "items_rejected": started.get("items_rejected", 0),
        "rejected": started.get("rejected", []),
        "playlists_expanding": started.get("playlists_expanding", 0),
        "message": started.get("message"),
    }


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

    Images are identified by their own bytes, not by the Content-Type the client
    declares (a client with no ``.webp`` or ``.heic`` row in its extension table
    sends ``application/octet-stream``). Accepted still images are ``image/jpeg``,
    ``image/png``, ``image/gif`` (non-animated), ``image/webp`` (non-animated),
    ``image/heic``, ``image/heif``, ``image/avif``, ``image/bmp`` and
    ``image/tiff``; the last five are transcoded to JPEG (or PNG when they carry
    transparency) on the way in, so what is stored and what the models read is
    always one of the first four. Proprietary vs biographical classification is
    done inside the processing pipeline via structured model output (no
    ``proprietary_content`` flag).

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

        # Only the creator of an avatar may add media to that avatar's identity.
        # Shared with /list_avatar_documents and /delete_avatar_document, which read
        # and remove the very rows this endpoint writes.
        assistant, _creator_id = await resolve_assistant_for_creator(
            assistant_id,
            current_user,
            action_description="upload media for that avatar",
        )
        assistant_meta = assistant.get("metadata") or {}

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
        # Items that failed validation while their siblings succeeded. A batch is
        # a set of independent jobs, so one unusable file (a mislabeled media
        # type, an unreachable URL, a malformed CSV) is reported here and skipped
        # rather than failing the whole request. Reference mode is exempt: it
        # carries exactly one item, so its failure is the request's failure.
        rejected_items: list[dict] = []
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
                raw_name = uf.filename
                # Each file is its own job: a failure here rejects that one item
                # and the loop moves on, so a single mislabeled or unreadable
                # upload cannot discard its siblings.
                try:
                    content = await uf.read()
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
                except Exception as file_entry_error:  # noqa: BLE001 - per-item skip
                    rejected_items.append(
                        _rejected_media_item(raw_name, file_entry_error)
                    )
                    logger.warning(
                        "Skipping upload %r: %s",
                        raw_name,
                        rejected_items[-1]["reason"],
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
                # Same per-item isolation as the file loop: an unreachable or
                # unsupported URL is skipped, not fatal to the batch.
                try:
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
                except Exception as url_entry_error:  # noqa: BLE001 - per-item skip
                    rejected_items.append(_rejected_media_item(u, url_entry_error))
                    logger.warning(
                        "Skipping url %r: %s", u, rejected_items[-1]["reason"]
                    )

            media_files = [*file_entries, *url_entries]

        # A playlist-only upload has no ready media_files yet (its videos are
        # enumerated in the background), so only reject when nothing at all — no
        # files and no playlists — was found.
        if not media_files and not playlist_urls:
            # Nothing survived. When items were rejected individually, return why
            # — a bare "no processable media" hides the actual per-item reasons.
            if rejected_items:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": (
                            "No processable media found in the request; every "
                            "item was rejected."
                        ),
                        "rejected": rejected_items,
                    },
                )
            raise HTTPException(
                status_code=400,
                detail="No processable media found in the request.",
            )
        if rejected_items:
            logger.info(
                "Continuing with %d item(s); %d rejected: %s",
                len(media_files),
                len(rejected_items),
                [item["filename"] for item in rejected_items],
            )

        # Stamp the batch-wide "no single target" flag onto every entry (top
        # level, alongside reference_audio/reference_image). convert_uploaded_
        # files_to_media reads it for audio/video/url items and threads it into
        # their metadata; expanded playlist children inherit it downstream.
        if create_reference_media_from_playlist:
            for entry in media_files:
                entry["create_reference_media_from_playlist"] = True

        return JSONResponse(
            status_code=202,
            content=await _start_media_batch(
                user_id=user_id,
                assistant_id=assistant_id,
                config=config,
                media_files=media_files,
                playlist_urls=playlist_urls,
                rejected_items=rejected_items,
                current_user=current_user,
                create_reference_media_from_playlist=create_reference_media_from_playlist,
            ),
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
            "estimated_media_seconds": job_estimated_media_seconds(registry, job),
            "estimated_processing_seconds": job_estimated_processing_seconds(
                registry, job
            ),
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
            "estimated_media_seconds": job_estimated_media_seconds(registry, j),
            "estimated_processing_seconds": job_estimated_processing_seconds(
                registry, j
            ),
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

    registry = app.state.media_jobs

    def _with_timing(payload: dict) -> dict:
        """Return a copy of ``payload`` stamped with the job's start time, the
        wall-clock seconds elapsed since processing began, and the processing-time
        estimate (probed media length x ``media_preprocessing_seconds_per_media_second``)
        with the seconds still expected, so every SSE ``data:`` frame carries
        timing. ``started_at`` is epoch seconds (set when the job flipped to
        running); fall back to ``created_at`` if it hasn't yet. The estimate keys
        are ``None`` for items that are not audio or video.
        """
        started = job.started_at or job.created_at
        elapsed_seconds = round(time_ns() / 1_000_000_000 - started, 3)
        estimated_processing = job_estimated_processing_seconds(registry, job)
        return {
            **payload,
            "started_at": job.started_at,
            "elapsed_seconds": elapsed_seconds,
            "estimated_media_seconds": job_estimated_media_seconds(registry, job),
            "estimated_processing_seconds": estimated_processing,
            "estimated_remaining_seconds": (
                round(max(0.0, estimated_processing - elapsed_seconds), 1)
                if estimated_processing is not None
                else None
            ),
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
    job_id = job_id.strip()  # remove spaces and newline characters
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
async def list_avatar_documents(
    assistant_id: str, current_user: dict = Depends(get_current_user)
):
    # The caller names the avatar explicitly. Only the creator of that avatar may
    # see the source documents that built the avatar's identity, so this endpoint
    # applies the same creator check /update_avatar_identity_with_media applies
    # when writing those documents. resolve_assistant_for_creator returns the
    # creator's user id, which is the user id the avatar's store rows were written
    # under and therefore what scopes the read below. After the creator check the
    # creator's user id and the caller's user id necessarily hold the same value;
    # reading the creator's makes the store scoping self-documenting.
    _assistant, user_id = await resolve_assistant_for_creator(
        assistant_id,
        current_user,
        action_description="list the documents of that avatar",
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
    # mapping de-dupes them down to one entry per source. Playlist videos are
    # listed as ``{playlist} :: {video}`` and everything else by plain filename —
    # see _document_label_and_key, shared with /delete_avatar_document so a label
    # copied out of this list resolves back to the key delete needs.
    #
    # The reference role is folded in per source rather than per Document: of the
    # several Documents one source produces, only the reference copies carry the
    # role, so the first non-None role wins and the ordinary siblings never
    # overwrite it back to None.
    reference_role_by_document_label: dict[str, str | None] = {}
    for label, _key, reference_role in _iter_document_labels(all_document_items):
        if reference_role_by_document_label.get(label) is None:
            reference_role_by_document_label[label] = reference_role

    uploaded_document_labels = sorted(reference_role_by_document_label)

    # Seconds of the avatar's speech stored per upload, so the client can show
    # which uploads feed the voice model. Best-effort: no repository or a read
    # failure leaves every document at zero.
    from src.anubis.utils.media_assets import get_media_asset_repository

    voice_seconds_by_label: dict[str, float] = {}
    voice_repository = get_media_asset_repository()
    if voice_repository is not None:
        try:
            from src.anubis.utils.voice.corpus import voice_seconds_by_document

            voice_seconds_by_label = voice_seconds_by_document(
                await voice_repository.list_voice_clips(assistant_id)
            )
        except Exception as voice_error:  # noqa: BLE001
            logger.debug(
                "Voice seconds unavailable for %s: %s", assistant_id, voice_error
            )

    return {
        # Unchanged shape: the plain label list every existing caller reads, and
        # the exact strings /delete_avatar_document accepts back.
        "uploaded_documents": uploaded_document_labels,
        # Same sources, same order, with the reference role attached. A client
        # that wants to mark which upload is the avatar's portrait or its voice
        # sample reads this list instead of the bare labels above.
        "documents": [
            {
                "label": label,
                "reference_role": reference_role_by_document_label[label],
                "is_reference_image": reference_role_by_document_label[label]
                == "reference_image",
                "is_reference_audio": reference_role_by_document_label[label]
                == "reference_audio",
                "voice_seconds": round(voice_seconds_by_label.get(label, 0.0), 1),
                "in_voice_corpus": label in voice_seconds_by_label,
            }
            for label in uploaded_document_labels
        ],
    }


@app.delete("/delete_avatar_document")
async def delete_avatar_documents(
    assistant_id: str,
    source_document_name: str,
    current_user: dict = Depends(get_current_user),
):

    # Strip wrappers from copied SQL tuple/list output, e.g. ('Mom.m4a',) or "Mom.m4a",
    # leaving only the filename or already-derived namespace id.
    source_document_name = source_document_name.strip(" \t\n\r\"'`(),[]")
    # Keep the user-facing name for the response; source_document_name itself may
    # be rewritten below into an opaque hashed/composite store key.
    display_name = source_document_name
    # Same rule as /list_avatar_documents: the caller names the avatar, and only
    # the creator of that avatar may delete the documents that built the avatar's
    # identity. The creator's user id scopes every store prefix built below.
    _assistant, user_id = await resolve_assistant_for_creator(
        assistant_id,
        current_user,
        action_description="delete the documents of that avatar",
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
            label: key
            for label, key, _reference_role in _iter_document_labels(existing_items)
            if key
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
    invalidate_store_cache_entry(
        (user_id, assistant_id, "reference_audio"), assistant_id
    )

    # A deleted upload's speech leaves the voice corpus too, so the seconds the
    # settings panel reports stay truthful. The trained voice is never rebuilt
    # or removed here. Best-effort, like the feature prune below.
    from src.anubis.utils.media_assets import get_media_asset_repository

    voice_repository = get_media_asset_repository()
    if voice_repository is not None:
        try:
            from src.anubis.utils.voice.corpus import forget_document_clips

            await forget_document_clips(
                voice_repository,
                user_id=user_id,
                assistant_id=assistant_id,
                source_document_name=display_name,
            )
        except Exception as voice_error:  # noqa: BLE001
            logger.warning(
                "voice clip prune failed for %s / %s: %s",
                assistant_id,
                display_name,
                voice_error,
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


# ---------------------------------------------------------------------------
# Deep research with web-based fact verification (creator-only)
# ---------------------------------------------------------------------------


def _start_deep_research_job(
    app_state,
    current_user: dict,
    *,
    assistant_id: str,
    creator_id: str,
    subject_name: str,
    subject_description: str | None,
    research_hint: str | None,
    assistant_metadata: dict | None = None,
):
    """Register and launch one research job; the pipeline runs as a background task."""
    from src.anubis.utils.research.deep_research import run_deep_research
    from src.api.research_jobs import add_event as add_research_event
    from src.api.research_jobs import create_research_job
    from src.api.research_jobs import finish_job as finish_research_job

    context = app_state.context
    if str(getattr(context, "deep_research_enabled", "true") or "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        raise HTTPException(
            status_code=503, detail="Deep research is disabled (DEEP_RESEARCH_ENABLED)."
        )
    registry = app_state.research_jobs
    job = create_research_job(
        registry,
        user_id=creator_id,
        assistant_id=assistant_id,
        subject_name=subject_name,
    )

    async def _run() -> None:
        job.started_at = time.time()
        job.status = "running"
        try:
            summary = await run_deep_research(
                app_state.store,
                context,
                creator_id=creator_id,
                assistant_id=assistant_id,
                subject_name=subject_name,
                subject_description=subject_description,
                research_hint=research_hint,
                emit=lambda payload: add_research_event(job, payload),
                is_cancelled=lambda: job.cancelled,
            )
            if summary.get("cancelled"):
                finish_research_job(
                    job, cancelled=True, result={"message": "Research cancelled."}
                )
                return
            # Facts written into the avatar's identity change what
            # load_consciousness reads on the next turn.
            if summary.get("applied"):
                invalidate_store_cache_for_assistant(assistant_id)

            # Media the research verified goes through the same pipeline an
            # uploaded link takes, so a video or a PDF behind a verified fact is
            # transcribed, described, and indexed rather than reduced to the
            # sentences the page happened to expose. The batch is itself a
            # background job, so this returns at once and the two run alongside
            # each other; the pipeline skips anything this avatar already holds,
            # so re-running the research re-ingests nothing.
            media_source_urls = list(summary.get("media_source_urls") or [])
            if media_source_urls:
                add_research_event(
                    job,
                    {
                        "type": "research_progress",
                        "stage": "learning_from_media",
                        "media_sources": len(media_source_urls),
                    },
                )
                try:
                    media_batch = await start_identity_media_job_from_chat(
                        user_id=creator_id,
                        assistant_id=assistant_id,
                        assistant_ctx={
                            "name": subject_name,
                            "description": subject_description,
                            "metadata": assistant_metadata or {},
                        },
                        current_user=current_user,
                        attachments=[],
                        urls=media_source_urls,
                    )
                except Exception as media_error:  # noqa: BLE001 - research still succeeded
                    logger.exception(
                        "Deep research job %s could not start the media batch: %s",
                        job.job_id,
                        media_error,
                    )
                    media_batch = {"status": "refused", "detail": str(media_error)}
                summary["media_batch"] = media_batch
                add_research_event(
                    job,
                    {
                        "type": "research_progress",
                        "stage": "media_started",
                        "media_batch": media_batch,
                    },
                )
            # The source pages read are billed like uploaded documents.
            tokens_read = int(summary.get("tokens_read") or 0)
            if tokens_read > 0:
                metering_bypass = resolve_metering_bypass(
                    current_user, assistant_id=assistant_id
                )
                if not metering_bypass.skips_metering_writes:
                    stripe_customer_id = resolve_stripe_customer_id(current_user)
                    await report_meter_event(
                        app_state.stripe,
                        UsageMeter.DOCUMENT_UPLOAD_TOKENS,
                        stripe_customer_id,
                        tokens_read,
                        idempotency_identifier=f"deep-research:{job.job_id}",
                    )
                    await persist_api_metrics_row(
                        getattr(app_state, "pool", None),
                        inference_type="deep_research",
                        prompt_tokens=tokens_read,
                        completion_tokens=0,
                        total_tokens=tokens_read,
                        cost_usd=0.0,
                        latency_ms=(time.time() - (job.started_at or job.created_at))
                        * 1000.0,
                        user_id=resolve_metering_user_id(current_user),
                        stripe_customer_id=stripe_customer_id,
                        assistant_id=assistant_id,
                        thread_id=None,
                        model_name=None,
                        meter_event_name=UsageMeter.DOCUMENT_UPLOAD_TOKENS.value,
                    )
            finish_research_job(job, result=summary)
        except asyncio.CancelledError:
            finish_research_job(
                job, cancelled=True, result={"message": "Research cancelled."}
            )
            raise
        except Exception as research_error:  # noqa: BLE001 - surfaced through the job
            logger.exception(
                "Deep research job %s failed: %s", job.job_id, research_error
            )
            finish_research_job(job, error=str(research_error))

    job.task = asyncio.create_task(_run())
    return job


def _get_owned_research_job(request: Request, job_id: str, user_id: str):
    from src.api.research_jobs import get_research_job

    job = get_research_job(request.app.state.research_jobs, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown or expired research job.")
    if job.user_id != user_id:
        raise HTTPException(
            status_code=403, detail="This research job belongs to another user."
        )
    return job


@app.post("/avatar/{assistant_id}/deep_research", status_code=202)
async def start_avatar_deep_research(
    request: Request,
    assistant_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Research the avatar's subject on the web and verify the facts against sources.

    Pro and premium (the upload capability). Body: optional ``research_hint``,
    words that narrow the subject ("the sculptor, not the footballer").

    The research reads what the avatar already holds, searches the web topic by
    topic, and verifies every claim across sources. Facts the sources do not
    contradict are added to the avatar's identity as soon as the job finishes;
    the contradictions wait for the creator at
    ``GET /avatar/{assistant_id}/research/proposals``. Returns ``202`` with the
    job id; follow ``GET /research_job/{job_id}/progress``.
    """
    enforce_tier_capability(current_user, TierCapability.UPLOAD)
    assistant, creator_id = await resolve_assistant_for_creator(
        assistant_id, current_user, action_description="research that avatar"
    )
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 - an empty body is a valid request
        body = {}
    body = body if isinstance(body, dict) else {}
    research_hint = str(body.get("research_hint") or "").strip() or None
    job = _start_deep_research_job(
        request.app.state,
        current_user,
        assistant_id=assistant_id,
        creator_id=creator_id,
        subject_name=assistant.get("name") or assistant_id,
        subject_description=assistant.get("description"),
        research_hint=research_hint,
        assistant_metadata=assistant.get("metadata") or {},
    )
    return JSONResponse(
        {
            "job_id": job.job_id,
            "status": job.status,
            "status_url": f"/research_job/{job.job_id}",
            "progress_url": f"/research_job/{job.job_id}/progress",
            "proposals_url": f"/avatar/{assistant_id}/research/proposals",
        },
        status_code=202,
    )


@app.get("/research_job/{job_id}")
async def research_job_status(
    request: Request, job_id: str, current_user: dict = Depends(get_current_user)
):
    """One research job's status, result, and latest stage."""
    job = _get_owned_research_job(
        request, job_id, current_user["identities"][0]["user_id"]
    )
    return JSONResponse(job.snapshot())


@app.get("/research_job/{job_id}/progress")
async def research_job_progress(
    request: Request, job_id: str, current_user: dict = Depends(get_current_user)
):
    """Server-sent progress of one research job: every stage so far, then live, then ``research_done``."""
    job = _get_owned_research_job(
        request, job_id, current_user["identities"][0]["user_id"]
    )

    async def _progress():
        next_index = 0
        while True:
            job.updated.clear()
            while next_index < len(job.events):
                yield f"data: {json.dumps(job.events[next_index], default=str)}\n\n"
                next_index += 1
            if job.done.is_set() and next_index >= len(job.events):
                yield (
                    "data: "
                    + json.dumps(
                        {"type": "research_done", **job.snapshot()}, default=str
                    )
                    + "\n\n"
                )
                return
            await job.updated.wait()

    return StreamingResponse(
        _progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/research_job/{job_id}/cancel")
async def cancel_research_job(
    request: Request, job_id: str, current_user: dict = Depends(get_current_user)
):
    """Ask a running research job to stop at the next stage boundary."""
    from src.api.research_jobs import request_cancel as request_research_cancel

    job = _get_owned_research_job(
        request, job_id, current_user["identities"][0]["user_id"]
    )
    if job.done.is_set():
        return JSONResponse(
            {"job_id": job_id, "status": job.status, "message": "Job already finished."}
        )
    request_research_cancel(job)
    return JSONResponse(
        {"job_id": job_id, "status": job.status, "message": "Cancellation requested."}
    )


@app.get("/avatar/{assistant_id}/research/proposals")
async def list_avatar_research_proposals(
    request: Request, assistant_id: str, current_user: dict = Depends(get_current_user)
):
    """List the researched facts the sources disagree on, for the creator's decision."""
    from src.anubis.utils.research.deep_research import list_proposals

    _assistant, creator_id = await resolve_assistant_for_creator(
        assistant_id, current_user, action_description="review that avatar's research"
    )
    proposals = await list_proposals(request.app.state.store, creator_id, assistant_id)
    return JSONResponse({"assistant_id": assistant_id, "proposals": proposals})


class ResearchResolutionItem(BaseModel):
    """One decision the creator made about one researched fact."""

    fact_id: str
    action: Literal["accept", "edit", "ignore"]
    corrected_text: str | None = None


class ResearchResolutionRequest(BaseModel):
    """Every decision the creator made in one pass of the review."""

    items: List[ResearchResolutionItem]


@app.post("/avatar/{assistant_id}/research/resolve")
async def resolve_avatar_research_proposals(
    request: Request,
    assistant_id: str,
    resolution: ResearchResolutionRequest,
    current_user: dict = Depends(get_current_user),
):
    """Apply the creator's decisions on the contradicted facts.

    Accepted and edited facts become identity documents the avatar reads on the
    next turn and the settings screen lists under what the avatar has learned;
    ignored facts are dropped.
    """
    from src.anubis.utils.research.deep_research import (
        ProposalResolution,
        resolve_proposals,
    )

    _assistant, creator_id = await resolve_assistant_for_creator(
        assistant_id, current_user, action_description="resolve that avatar's research"
    )
    result = await resolve_proposals(
        request.app.state.store,
        creator_id,
        assistant_id,
        [
            ProposalResolution(
                fact_id=item.fact_id,
                action=item.action,
                corrected_text=item.corrected_text,
            )
            for item in resolution.items
        ],
    )
    if result["accepted"] or result["edited"]:
        invalidate_store_cache_for_assistant(assistant_id)
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# What an avatar has learned about its own identity (owner-only)
# ---------------------------------------------------------------------------

# The four store groups that hold what an avatar has learned about ITSELF, keyed
# by the store namespace's third element and valued by the ``learned_from`` name
# the API reports. Each group lives under ``(creator_id, assistant_id, <group>)``
# — the same namespaces ``load_consciousness`` reads every turn
# (src/anubis/utils/nodes.py) and the conversational correction tools sweep
# (identity_tools._correction_namespaces). Facts about the *user* live under
# ``(assistant_id, user_id, "identity")`` and are deliberately not listed here.
#
# ``identity`` is a prefix: the media pipeline writes atomic facts under
# ``(creator_id, assistant_id, "identity", <filename uuid5>)``, and a prefix
# search sweeps those sub-namespaces along with the root.
IDENTITY_FACT_GROUPS: dict[str, str] = {
    "identity_memory": "conversation",
    "identity": "media",
    "analysis": "analysis",
    "memory": "memory",
}

# A fifth group has no namespace of its own: deep research writes into the
# ``identity`` namespace (so the avatar reads researched facts exactly like
# uploaded ones) and marks each document with ``RESEARCH_FACT_SOURCE``, which
# ``_identity_fact_row`` reports as the ``research`` group.

_FACT_CONTEXT_TAG_PATTERN = re.compile(r"<FACT_CONTEXT>(.*?)</FACT_CONTEXT>", re.DOTALL)


def _store_item_namespace(item) -> tuple:
    namespace = getattr(item, "namespace", None)
    if namespace is None and isinstance(item, dict):
        namespace = item.get("namespace")
    return tuple(namespace) if namespace else ()


def _store_item_key(item) -> str | None:
    key = getattr(item, "key", None)
    if key is None and isinstance(item, dict):
        key = item.get("key")
    return key


def _store_item_updated_at(item) -> str | None:
    updated_at = getattr(item, "updated_at", None)
    if updated_at is None and isinstance(item, dict):
        updated_at = item.get("updated_at")
    if isinstance(updated_at, datetime):
        return updated_at.isoformat()
    return str(updated_at) if updated_at else None


def _identity_fact_row(item, *, learned_from: str) -> dict | None:
    """Shape one store item into a ``/avatar_identity_facts`` row.

    Returns ``None`` when the item carries no recoverable fact — a raw transcript
    chunk or reference row that shares the ``identity`` prefix with the atomic
    biographical facts, or an analysis/memory row with empty content.
    """
    from urllib.parse import urlparse

    from src.anubis.utils.research.deep_research import RESEARCH_FACT_SOURCE
    from src.anubis.utils.tools.identity.identity_tools import (
        _extract_clean_fact,
        _item_document_id,
        _item_document_kwargs,
        _page_content_is_wrapped,
    )

    document_kwargs = _item_document_kwargs(item)
    metadata = document_kwargs.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    page_content = document_kwargs.get("page_content") or ""
    is_wrapped = _page_content_is_wrapped(page_content)

    if learned_from == "media":
        # Only the atomic first-person facts the media pipeline extracted count
        # as "learned"; the verbatim transcript and description chunks stored
        # beside them are sources, and Data Uploaded already lists those.
        is_biographical_fact = (
            metadata.get("classified_situation") == "biographical_facts"
        )
        if not (is_biographical_fact or is_wrapped):
            return None

    feature = metadata.get("feature") if learned_from == "analysis" else None
    feature_statement = metadata.get(feature) if isinstance(feature, str) else None
    if isinstance(feature_statement, str) and feature_statement.strip():
        fact = feature_statement.strip()
    else:
        fact = _extract_clean_fact(item)
    if not fact:
        return None

    context = metadata.get("fact_context")
    if not context and is_wrapped:
        context_match = _FACT_CONTEXT_TAG_PATTERN.search(page_content)
        context = context_match.group(1).strip() if context_match else None
    if not context:
        context = (
            metadata.get("supporting_reason")
            or metadata.get("concise_context_summary")
            or None
        )

    source_label = None
    if learned_from in ("media", "analysis"):
        source_label, _key = _document_label_and_key(metadata)

    # A researched fact lives under the same ``identity`` prefix as the facts
    # extracted from uploads — the avatar must read every fact the same way —
    # but the owner needs to tell a fact the web supplied from a fact an upload
    # supplied, so the metadata mark deep research writes renames the group and
    # the first source page names the row.
    if metadata.get("source") == RESEARCH_FACT_SOURCE:
        learned_from = "research"
        source_urls = metadata.get("source_urls") or []
        if source_urls:
            try:
                source_label = urlparse(str(source_urls[0])).netloc or str(
                    source_urls[0]
                )
            except Exception:  # noqa: BLE001 - the label is a convenience
                source_label = str(source_urls[0])

    namespace = _store_item_namespace(item)
    key = _store_item_key(item)
    created_at = metadata.get("created_at") or _store_item_updated_at(item)
    return {
        "fact_id": _item_document_id(item) or key,
        "fact": fact,
        "context": context,
        "learned_from": learned_from,
        "feature": feature if isinstance(feature, str) else None,
        "source_label": source_label,
        "namespace": list(namespace),
        "key": key,
        "created_at": created_at,
        "corrected_from": metadata.get("corrected_from"),
        "source_urls": list(metadata.get("source_urls") or []),
        "verification_status": metadata.get("verification_status"),
    }


def _identity_fact_sort_key(row: dict) -> str:
    return row.get("created_at") or ""


@app.get("/avatar_identity_facts")
async def list_avatar_identity_facts(
    assistant_id: str, current_user: dict = Depends(get_current_user)
):
    """Everything the avatar has learned about its own identity, newest first.

    Five groups, reported as ``learned_from``: ``conversation`` (facts the owner
    told the avatar in chat), ``media`` (first-person biographical facts extracted
    from uploads), ``research`` (facts deep research verified on the web, which
    live in the same namespace as the media facts and are told apart by their
    ``source`` mark), ``analysis`` (derived traits such as beliefs and values),
    and ``memory`` (episodic memories from the owner's own conversations).

    Only the creator of the avatar may read this — the same creator check
    ``/list_avatar_documents`` applies — so a visitor chatting with a public
    avatar learns nothing about what the avatar was taught.
    """
    _assistant, creator_id = await resolve_assistant_for_creator(
        assistant_id,
        current_user,
        action_description="view what that avatar has learned",
    )
    store = app.state.store
    facts: list[dict] = []
    counts = {learned_from: 0 for learned_from in IDENTITY_FACT_GROUPS.values()}
    # Researched facts share the media namespace and are counted under their own
    # name, so the settings screen can filter them apart.
    counts.setdefault("research", 0)
    for namespace_group, learned_from in IDENTITY_FACT_GROUPS.items():
        namespace = (creator_id, assistant_id, namespace_group)
        try:
            items = await store.asearch(namespace, limit=1_000_000)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not read what this avatar has learned: {exc}",
            ) from exc
        for item in items or []:
            row = _identity_fact_row(item, learned_from=learned_from)
            if row is None:
                continue
            facts.append(row)
            # The row names its own group: a researched fact is read out of the
            # media namespace but reported as research.
            counts[row["learned_from"]] = counts.get(row["learned_from"], 0) + 1
    facts.sort(key=_identity_fact_sort_key, reverse=True)
    return {"facts": facts, "counts": counts}


def _owned_fact_namespace(
    namespace_value, *, creator_id: str, assistant_id: str
) -> tuple:
    """Validate a namespace the client sends back from the facts listing.

    The tuple must be one of the avatar's own fact groups under the creator's id,
    so a caller cannot reach another avatar's rows by naming a different tuple.
    """
    if not isinstance(namespace_value, (list, tuple)) or len(namespace_value) < 3:
        raise HTTPException(
            status_code=400,
            detail="namespace must be the list returned by /avatar_identity_facts.",
        )
    namespace = tuple(str(part) for part in namespace_value)
    if (
        namespace[0] != creator_id
        or namespace[1] != assistant_id
        or namespace[2] not in IDENTITY_FACT_GROUPS
    ):
        raise HTTPException(
            status_code=403,
            detail="That fact does not belong to this avatar.",
        )
    return namespace


async def _load_owned_fact_item(
    store, *, creator_id: str, assistant_id: str, body: dict
) -> tuple[tuple, str, object]:
    namespace = _owned_fact_namespace(
        body.get("namespace"), creator_id=creator_id, assistant_id=assistant_id
    )
    key = str(body.get("key") or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="key is required.")
    item = await store.aget(namespace, key)
    if item is None:
        raise HTTPException(status_code=404, detail="No such fact.")
    return namespace, key, item


@app.delete("/avatar_identity_facts")
async def delete_avatar_identity_fact(
    request: Request,
    assistant_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Forget one learned fact. Body: ``{namespace, key}`` from the listing."""
    _assistant, creator_id = await resolve_assistant_for_creator(
        assistant_id,
        current_user,
        action_description="delete what that avatar has learned",
    )
    body = await request.json()
    body = body if isinstance(body, dict) else {}
    store = app.state.store
    namespace, key, _item = await _load_owned_fact_item(
        store, creator_id=creator_id, assistant_id=assistant_id, body=body
    )
    await store.adelete(namespace, key)
    # load_consciousness reads these namespaces through a process-wide
    # read-through cache; drop the avatar's entries so the forgotten fact does
    # not ground the next reply.
    invalidate_store_cache_for_assistant(assistant_id)
    return Response(status_code=204)


@app.put("/avatar_identity_facts")
async def update_avatar_identity_fact(
    request: Request,
    assistant_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Rewrite one learned fact in place.

    Body: ``{namespace, key, fact, context?}``. The rewrite goes through the same
    helper the conversational ``edit_identity_fact`` tool uses, so the stored
    format (the ``<FACT>`` wrapper for identity facts, the plain event and
    context paragraphs for episodic memories) is preserved, the row re-embeds
    under the same key,
    and ``corrected_from`` / ``correction_origin`` record the owner's edit.
    """
    from src.anubis.utils.tools.identity.identity_tools import (
        FactMatch,
        _apply_fact_rewrite,
        _item_document_kwargs,
    )

    _assistant, creator_id = await resolve_assistant_for_creator(
        assistant_id,
        current_user,
        action_description="edit what that avatar has learned",
    )
    body = await request.json()
    body = body if isinstance(body, dict) else {}
    corrected_fact = str(body.get("fact") or "").strip()
    if not corrected_fact:
        raise HTTPException(status_code=400, detail="fact is required.")

    store = app.state.store
    namespace, key, item = await _load_owned_fact_item(
        store, creator_id=creator_id, assistant_id=assistant_id, body=body
    )
    learned_from = IDENTITY_FACT_GROUPS[namespace[2]]
    current_row = _identity_fact_row(item, learned_from=learned_from) or {}

    corrected_context = body.get("context")
    if corrected_context is None:
        corrected_context = current_row.get("context") or ""
    corrected_context = str(corrected_context).strip()

    match = FactMatch(
        item=item,
        namespace=namespace,
        key=key,
        kind="fact",
        matched_text=current_row.get("fact"),
        score=1.0,
    )
    change = await _apply_fact_rewrite(
        store, match, corrected_fact, corrected_context, user_modified=True
    )

    updated_document = change.document
    feature = (_item_document_kwargs(item).get("metadata") or {}).get("feature")
    extra_metadata: dict[str, str] = {}
    if updated_document is not None:
        # Media-ingested facts carry the fact only inside the <FACT> span (no
        # ``metadata.fact``), so the rewrite helper cannot record what it
        # replaced; record it here from the fact the listing showed.
        if updated_document.metadata.get("corrected_from") is None and current_row.get(
            "fact"
        ):
            extra_metadata["corrected_from"] = current_row["fact"]
        # Analysis rows keep the statement under ``metadata[<feature>]`` too
        # (that is what the trait loaders read), so mirror the edit there.
        if learned_from == "analysis" and isinstance(feature, str):
            extra_metadata[feature] = corrected_fact
    if extra_metadata:
        updated_document.metadata.update(extra_metadata)
        await store.aput(
            namespace, key=key, value={"document": updated_document.to_json()}
        )

    invalidate_store_cache_for_assistant(assistant_id)

    updated_item = SimpleNamespace(
        namespace=namespace,
        key=key,
        value={"document": updated_document.to_json()} if updated_document else {},
        updated_at=datetime.now(tz=UTC),
    )
    return _identity_fact_row(updated_item, learned_from=learned_from) or {}


@app.post("/avatar/{assistant_id}/recalibrate_ground_truth")
async def recalibrate_avatar_ground_truth(
    assistant_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Refit an avatar's direct-quote cloud from the quotes already in the store.

    The cloud an avatar's replies are scored against — the empirical Mahalanobis
    threshold and the IsolationForest behind
    ``comparison_to_direct_quote_response_analysis`` — is derived state, refitted
    after each media upload that adds direct quotes. Two cases leave an avatar
    without one: quotes ingested before that refit was wired to every upload path,
    and a refit that was skipped because the batch timed out or failed. In both,
    the corpus is present and only the fit is missing, so re-uploading the source
    media to trigger a refit would be pure waste.

    Runs through the media-job registry rather than inline: the fit is quadratic
    in corpus size, so a large avatar would hold an HTTP connection open far past
    any sensible request timeout. Returns 202 with the same job/progress/cancel
    URLs a media upload returns, so callers can reuse the upload progress stream.
    """
    _, creator_user_id = await resolve_assistant_for_creator(
        assistant_id,
        current_user,
        action_description="recalibrate the authenticity model of that avatar",
    )

    from src.api.media_jobs import (
        _calibrate_ground_truth_after_batch,
        create_master_job,
        finish_job,
    )

    registry = app.state.media_jobs
    # Owner-scoped, matching every other per-avatar artifact: the cloud must be
    # written under the id the message path reads back, which is the avatar's
    # creator, never the caller if the two ever diverge.
    master = create_master_job(registry, creator_user_id, assistant_id)

    async def _run_recalibration() -> None:
        try:
            # force: this is an explicit request to fit a corpus already in the
            # store, so the "did this batch index new quotes" gate does not apply.
            await _calibrate_ground_truth_after_batch(
                master, app.state.store, app.state.context, force=True
            )
            finish_job(
                master,
                result={
                    "assistant_id": assistant_id,
                    "message": "Direct-quote calibration finished",
                },
            )
        except asyncio.CancelledError:
            finish_job(
                master,
                cancelled=True,
                result={"message": "Recalibration cancelled"},
            )
            raise
        except Exception as exc:  # noqa: BLE001 - surface every failure on the job
            logger.exception("Recalibration failed for %s: %s", assistant_id, exc)
            finish_job(master, error=str(exc))

    master.task = asyncio.create_task(_run_recalibration())

    return JSONResponse(
        status_code=202,
        content={
            "job_id": master.job_id,
            "assistant_id": assistant_id,
            "status_url": f"/media_job/{master.job_id}",
            "progress_url": f"/media_job/{master.job_id}/progress",
            "cancel_url": f"/media_job/{master.job_id}/cancel",
            "message": "Direct-quote calibration started",
        },
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
    threshold_namespace = (
        user_id,
        assistant_id,
        "ground_truth_text_empirical_threshold_list_str",
    )
    model_namespace = (
        user_id,
        assistant_id,
        "ground_truth_text_features_model_b64_pkl",
    )
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
        await store.adelete(
            threshold_namespace, key="ground_truth_text_empirical_threshold_list_str"
        )
        await store.adelete(
            model_namespace, key="ground_truth_text_features_model_b64_pkl"
        )
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


# ── Usage analytics (opt-in action log and described page captures) ───────────
#
# A person who opted in has the browser record every action taken in the web
# application and capture the page itself (rendered from the document, never a
# window picker) on an interval and on route changes. Events and captures are
# accepted only for a signed-in account whose consent row says enabled, and
# only while USAGE_ANALYTICS_ENABLED is on. Captures are described after the
# response (a vision call takes seconds); the browser never waits on the call.


class UsageAnalyticsConsentRequest(BaseModel):
    """The body of POST /usage_analytics/consent."""

    enabled: bool
    # Where the choice was made: signup, avatar_settings, account_settings.
    source: str | None = None


class UsageAnalyticsEventsRequest(BaseModel):
    """The body of POST /usage_analytics/events: one batch of browser events."""

    session_id: str | None = None
    assistant_id: str | None = None
    thread_id: str | None = None
    route: str | None = None
    events: list[dict] = []


def _usage_analytics_repository_or_503():
    """Return the published repository, or raise 503 when the lifespan did not bind one."""
    from src.anubis.utils.usage_analytics import get_usage_analytics_repository

    repository = get_usage_analytics_repository()
    if repository is None:
        raise HTTPException(
            status_code=503, detail="Usage analytics is not available on this server."
        )
    return repository


def _usage_analytics_feature_enabled(context: GlobalContext) -> bool:
    return str(context.usage_analytics_enabled or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


async def _usage_analytics_recording_user(current_user: dict, repository) -> str:
    """Return the user id a recording request records under, or raise the refusal.

    404 when the deployment turned the feature off, 403 for an anonymous
    caller or an account that has not opted in: the browser stops recording
    on either answer rather than retrying.
    """
    if not _usage_analytics_feature_enabled(GlobalContext()):
        raise HTTPException(
            status_code=404, detail="Usage analytics is not enabled on this deployment."
        )
    if is_anonymous_user(current_user):
        raise HTTPException(
            status_code=403, detail="Usage analytics requires a signed-in user."
        )
    user_id = current_user["identities"][0]["user_id"]
    if not await repository.is_enabled(user_id):
        raise HTTPException(
            status_code=403,
            detail="This account has not opted in to usage analytics.",
        )
    return user_id


def _parse_iso_datetime_or_400(value: Optional[str], name: str) -> Optional[datetime]:
    if not (value or "").strip():
        return None
    from src.anubis.utils.usage_analytics.repository import _as_datetime

    parsed = _as_datetime(value)
    if parsed is None:
        raise HTTPException(status_code=400, detail=f"{name} must be an ISO 8601 timestamp.")
    return parsed


@app.get("/usage_analytics/consent")
async def get_usage_analytics_consent_route(
    current_user: dict = Depends(get_current_user),
):
    """Return whether this account has opted in to usage analytics, and when the choice was made."""
    from src.anubis.utils.usage_analytics.repository import consent_view

    repository = _usage_analytics_repository_or_503()
    user_id = current_user["identities"][0]["user_id"]
    row = await repository.get_consent(user_id)
    return JSONResponse(
        content={
            **consent_view(row, user_id),
            "feature_enabled": _usage_analytics_feature_enabled(GlobalContext()),
        },
        status_code=200,
    )


@app.post("/usage_analytics/consent")
async def set_usage_analytics_consent_route(
    body: UsageAnalyticsConsentRequest,
    current_user: dict = Depends(get_current_user),
):
    """Opt this account in to, or out of, usage analytics.

    Opting out stops recording at once; the rows already stored stay until
    the retention purge or ``DELETE /usage_analytics/data`` removes them.
    """
    from src.anubis.utils.usage_analytics.repository import consent_view

    repository = _usage_analytics_repository_or_503()
    user_id = current_user["identities"][0]["user_id"]
    row = await repository.set_consent(user_id, bool(body.enabled), source=body.source)
    return JSONResponse(
        content={
            **consent_view(row, user_id),
            "feature_enabled": _usage_analytics_feature_enabled(GlobalContext()),
        },
        status_code=200,
    )


@app.post("/usage_analytics/events", status_code=202)
async def record_usage_analytics_events_route(
    body: UsageAnalyticsEventsRequest,
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    """Store a batch of actions the browser recorded for a consenting account."""
    from src.anubis.utils.usage_analytics.events import (
        UsageEventError,
        normalise_event_batch,
    )

    repository = _usage_analytics_repository_or_503()
    user_id = await _usage_analytics_recording_user(current_user, repository)
    context = GlobalContext()
    try:
        rows = normalise_event_batch(
            body.events,
            user_id=user_id,
            session_id=body.session_id,
            assistant_id=body.assistant_id,
            thread_id=body.thread_id,
            route=body.route,
            max_events=int(context.usage_analytics_max_events_per_request or 200),
        )
    except UsageEventError as event_error:
        raise HTTPException(status_code=422, detail=str(event_error)) from event_error
    recorded = await repository.record_events(rows) if rows else 0
    return JSONResponse(content={"recorded": recorded}, status_code=202)


# Per-session floor between captures, so a misconfigured client cannot flood
# the vision model. Reuses the ambient throttle's bookkeeping, keyed by the
# browser session rather than by a conversation thread.
_usage_analytics_capture_throttle = None


def _usage_capture_throttle():
    """Return the process-wide capture throttle, created on first use."""
    global _usage_analytics_capture_throttle
    if _usage_analytics_capture_throttle is None:
        from src.anubis.utils.ambient.observations import AmbientThrottle

        _usage_analytics_capture_throttle = AmbientThrottle()
    return _usage_analytics_capture_throttle


@app.post("/usage_analytics/screenshots", status_code=202)
async def record_usage_analytics_screenshot_route(
    request: Request,
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    assistant_id: Optional[str] = Form(None),
    thread_id: Optional[str] = Form(None),
    route: Optional[str] = Form(None),
    trigger: Optional[str] = Form(None),
    recent_actions: Optional[str] = Form(None),
    occurred_at: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    """Store one capture of the web page and describe the capture in the background.

    ``recent_actions`` is the browser's short list of what the person did
    before the capture; the describing model reads the list beside the image.
    The full-size image is discarded once the thumbnail and the description
    exist.
    """
    from src.anubis.utils.usage_analytics import screenshots as capture_module

    repository = _usage_analytics_repository_or_503()
    user_id = await _usage_analytics_recording_user(current_user, repository)
    context = GlobalContext()
    seconds_to_wait = _usage_capture_throttle().check_and_mark(
        (session_id or "").strip() or user_id,
        float(context.usage_analytics_capture_min_interval_seconds or 0),
    )
    if seconds_to_wait is not None:
        raise HTTPException(
            status_code=429,
            detail=f"Captures are accepted at most every {context.usage_analytics_capture_min_interval_seconds} seconds per session.",
            headers={"Retry-After": str(max(1, int(seconds_to_wait + 0.999)))},
        )
    image_bytes = await file.read()
    max_bytes = int(context.usage_analytics_max_image_bytes or 0)
    if not image_bytes:
        raise HTTPException(status_code=422, detail="The capture is empty.")
    if max_bytes and len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"The capture is larger than the {max_bytes} byte ceiling.",
        )
    image_mime = (file.content_type or "image/png").split(";")[0].strip() or "image/png"
    try:
        thumbnail, thumbnail_mime, width, height = capture_module.make_thumbnail(
            image_bytes, max_width=int(context.usage_analytics_thumbnail_width or 640)
        )
    except Exception as thumbnail_error:  # noqa: BLE001 - an unreadable image is a client error
        raise HTTPException(
            status_code=422, detail="The capture is not a readable image."
        ) from thumbnail_error
    when = _parse_iso_datetime_or_400(occurred_at, "occurred_at") or datetime.now(UTC)
    recent = (recent_actions or "").strip()[:6000] or None
    stored = await repository.create_screenshot(
        {
            "user_id": user_id,
            "session_id": (session_id or "").strip() or None,
            "assistant_id": (assistant_id or "").strip() or None,
            "thread_id": (thread_id or "").strip() or None,
            "route": (route or "").strip()[:400] or None,
            "trigger": (trigger or "").strip()[:60] or None,
            "recent_actions": recent,
            "thumbnail": thumbnail,
            "thumbnail_mime": thumbnail_mime,
            "width": width,
            "height": height,
            "occurred_at": when,
            "status": capture_module.STATUS_PENDING,
        }
    )
    asyncio.create_task(
        capture_module.describe_and_store(
            repository=repository,
            store=getattr(app.state, "store", None),
            screenshot_id=stored["id"],
            user_id=user_id,
            image_bytes=image_bytes,
            image_mime=image_mime,
            recent_actions=recent,
            route=stored.get("route"),
            assistant_id=stored.get("assistant_id"),
            session_id=stored.get("session_id"),
            occurred_at=when,
        )
    )
    return JSONResponse(
        content={"screenshot_id": stored["id"], "status": capture_module.STATUS_PENDING},
        status_code=202,
    )


def _usage_analytics_subject_user_id(current_user: dict, user_id: Optional[str]) -> str:
    """Return whose rows a read covers: the caller's, or anyone's for the administrator."""
    caller_id = current_user["identities"][0]["user_id"]
    requested = (user_id or "").strip()
    if not requested or requested == caller_id:
        return caller_id
    admin_id = (GlobalContext().admin_user_id or "").strip()
    if admin_id and caller_id == admin_id:
        return requested
    raise HTTPException(
        status_code=403, detail="Only the administrator may read another account's usage analytics."
    )


@app.get("/usage_analytics/summary")
async def usage_analytics_summary_route(
    since: Optional[str] = None,
    until: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
):
    """Return recent action events and described captures, newest first.

    The administrator may pass ``user_id`` to read another account's rows for
    product analysis; everyone else reads their own.
    """
    from src.anubis.utils.usage_analytics.repository import (
        consent_view,
        event_view,
        screenshot_view,
    )

    repository = _usage_analytics_repository_or_503()
    subject = _usage_analytics_subject_user_id(current_user, user_id)
    since_at = _parse_iso_datetime_or_400(since, "since")
    until_at = _parse_iso_datetime_or_400(until, "until")
    bounded_limit = max(1, min(int(limit or 100), 500))
    bounded_offset = max(0, int(offset or 0))
    events = await repository.list_events(
        subject,
        since=since_at,
        until=until_at,
        session_id=(session_id or "").strip() or None,
        limit=bounded_limit,
        offset=bounded_offset,
    )
    screenshots = await repository.list_screenshots(
        subject,
        since=since_at,
        until=until_at,
        session_id=(session_id or "").strip() or None,
        limit=bounded_limit,
        offset=bounded_offset,
    )
    consent = await repository.get_consent(subject)
    return JSONResponse(
        content={
            "user_id": subject,
            "consent": consent_view(consent, subject),
            "events": [event_view(row) for row in events],
            "screenshots": [screenshot_view(row) for row in screenshots],
            "limit": bounded_limit,
            "offset": bounded_offset,
        },
        status_code=200,
    )


@app.get("/usage_analytics/screenshots/{screenshot_id}/thumbnail")
async def usage_analytics_thumbnail_route(
    screenshot_id: str,
    user_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    """Return the JPEG thumbnail kept beside one capture's description."""
    repository = _usage_analytics_repository_or_503()
    subject = _usage_analytics_subject_user_id(current_user, user_id)
    found = await repository.get_thumbnail(subject, screenshot_id)
    if found is None:
        raise HTTPException(status_code=404, detail="No such capture.")
    thumbnail, mime = found
    return Response(content=thumbnail, media_type=mime)


@app.delete("/usage_analytics/data")
async def delete_usage_analytics_data_route(
    current_user: dict = Depends(get_current_user),
):
    """Delete every stored event and capture of this account; consent is left as set."""
    repository = _usage_analytics_repository_or_503()
    user_id = current_user["identities"][0]["user_id"]
    removed = await repository.delete_user_data(user_id)
    store = getattr(app.state, "store", None)
    if store is not None:
        try:
            from src.anubis.utils.usage_analytics.screenshots import usage_analytics_namespace

            items = await store.asearch(usage_analytics_namespace(user_id), limit=1000)
            for item in items or []:
                key = getattr(item, "key", None)
                if key:
                    await store.adelete(usage_analytics_namespace(user_id), key)
        except Exception:  # noqa: BLE001 - the tables are the record of truth
            logger.debug("Usage analytics store rows were not deleted", exc_info=True)
    return JSONResponse(content={"deleted": removed}, status_code=200)



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
