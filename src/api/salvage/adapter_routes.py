"""Adapter training and lookup endpoints. INERT SALVAGE — this router is never registered.

Salvaged from the ``z-anubis`` branch (commit ``d1b8473``) on 2026-09-09,
where these four handlers and their helpers were declared inline in
``src/api/webapp.py``. They are moved onto a standalone ``APIRouter`` here so
the salvage costs the live application nothing: ``webapp.py`` does not import
this module and never calls ``include_router`` on ``adapter_route``, so none
of the paths are served and no background training watcher is ever started.

Everything the handlers need from ``webapp`` is imported INSIDE the function
that uses it, the way the live routes already lazy-import (see
``record_message_feedback_route`` in ``webapp.py``), so this module imports on
its own for review and testing without pulling ``webapp`` in.

The billing side of adapters is already live on f-anubis:
``UsageMeter.ADAPTER_TRAINING_UNITS``, ``UsageMeter.ADAPTER_INFERENCE_TOKENS``
and ``TierCapability.TRAIN_ADAPTER`` all exist in ``billing/tiers.py``, and
``resolve_use_adapter_inference`` already gates the ``adapter`` flag on
``/message``. What is missing is the call to the adapter service, which is
what this module and ``src/anubis/utils/adapters/client.py`` supply.

Note on the message path: ``z-anubis`` also changed
``resolve_use_adapter_inference`` in ``billing/gating.py`` to prefer a trained
adapter automatically (a three-state ``adapter_requested`` of
``None``/``True``/``False`` plus an ``adapter_available`` argument). That
change is NOT applied here, because editing ``gating.py`` would alter live
behaviour. ``resolve_adapter_preference`` below carries the same decision as a
standalone function; on activation, fold it into ``gating.py`` exactly as the
``z-anubis`` diff has it.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from time import time_ns
from typing import Any, Mapping

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from src.anubis.utils.adapters.client import (
    ADAPTER_STATUS_ERROR,
    ADAPTER_STATUS_TRAINED,
    ADAPTER_STATUS_TRAINING,
    AdapterServerClient,
    AdapterTrainingDataMissing,
    adapter_inference_enabled,
    avatar_adapter_metadata,
    avatar_has_trained_adapter,
)
from src.anubis.utils.billing.tiers import SubscriptionTier, TierCapability, UsageMeter
from src.security.auth import get_current_user

logger = logging.getLogger(__name__)

adapter_route = APIRouter(tags=["adapter (inert salvage)"])

_ADAPTER_TRAINING_POLL_SECONDS = 15.0
_ADAPTER_TRAINING_MAX_WAIT_SECONDS = 6 * 3600.0


def resolve_adapter_preference(
    user: Mapping[str, Any] | None,
    adapter_requested: bool | None,
    adapter_available: bool = False,
) -> bool:
    """Whether this turn should answer through the trained adapter.

    The ``z-anubis`` replacement for ``resolve_use_adapter_inference``, kept
    here rather than applied to ``billing/gating.py`` so that salvaging it
    changes no live behaviour. Adapter inference is a Premium-only capability
    and the adapter is PREFERRED once trained: with ``adapter_requested`` left
    ``None``, a Premium user whose avatar has a completed adapter uses the
    adapter automatically; ``False`` opts the turn out; ``True`` asks for the
    adapter explicitly (Premium still required, and the ``think`` node still
    falls back to the standard model when no adapter is actually available). A
    non-Premium user never uses the adapter, and the request falls back to
    standard inference and ``messaging_tokens`` metering without an error.
    """
    from src.anubis.utils.billing.gating import resolve_tier

    if adapter_requested is False:
        return False
    if resolve_tier(user) != SubscriptionTier.PREMIUM:
        return False
    if adapter_requested is True:
        return True
    return bool(adapter_available)


async def count_quote_documents(
    store: Any, creator_id: str, assistant_id: str, ceiling: int
) -> int:
    """How many direct-quote documents the avatar holds, capped at ``ceiling``."""
    try:
        items = await store.asearch(
            (creator_id, assistant_id, "quote"), limit=max(1, ceiling)
        )
    except Exception as count_error:  # noqa: BLE001
        logger.warning(
            "Could not count quote documents for %s: %s", assistant_id, count_error
        )
        return 0
    return len(items or [])


async def write_avatar_adapter_metadata(
    api_key: str, assistant_id: str, adapter_block: dict
) -> None:
    from langgraph_sdk import get_client

    client = get_client(headers={"API-KEY": api_key})
    await client.assistants.update(
        assistant_id=assistant_id, metadata={"adapter": adapter_block}
    )


async def watch_adapter_training(
    app_state: Any,
    current_user: dict,
    assistant_id: str,
    job_id: str,
    base_model_label: str | None,
) -> None:
    """Poll the adapter service until the job settles, then record the outcome on the avatar.

    A completed job flips the avatar's ``metadata.adapter.status`` to
    ``trained`` (which is what makes the message path prefer the adapter) and
    meters one adapter-training unit; an error or cancellation records that
    instead. Never raises.
    """
    from src.anubis.utils.billing.gating import (
        resolve_metering_bypass,
        resolve_metering_user_id,
        resolve_stripe_customer_id,
    )
    from src.anubis.utils.billing.metering import (
        persist_api_metrics_row,
        report_meter_event,
    )
    from src.security.auth import _evict_api_key_cache_for_user

    client = AdapterServerClient(app_state.context, current_user["API_KEY"])
    started = time_ns()
    final_status: dict = {}
    try:
        while (time_ns() - started) / 1e9 < _ADAPTER_TRAINING_MAX_WAIT_SECONDS:
            await asyncio.sleep(_ADAPTER_TRAINING_POLL_SECONDS)
            try:
                final_status = await client.training_status(job_id)
            except Exception as poll_error:  # noqa: BLE001 - keep polling
                logger.info(
                    "Adapter training poll failed for %s: %s", job_id, poll_error
                )
                continue
            if final_status.get("status") in ("completed", "error", "cancelled"):
                break
        status = final_status.get("status")
        now = datetime.now(UTC).isoformat()
        if status == "completed":
            result = final_status.get("result") or {}
            adapter_block = {
                "status": ADAPTER_STATUS_TRAINED,
                "job_id": job_id,
                "trained_at": now,
                "base_model": final_status.get("model") or base_model_label,
                "adapter_s3_prefix": final_status.get("adapter_s3_prefix"),
                "training_examples": result.get("training_examples"),
                "training_duration_seconds": final_status.get("duration_seconds"),
            }
            stripe_customer_id = resolve_stripe_customer_id(current_user)
            metering_bypass = resolve_metering_bypass(
                current_user, assistant_id=assistant_id
            )
            if not metering_bypass.skips_metering_writes:
                await report_meter_event(
                    app_state.stripe,
                    UsageMeter.ADAPTER_TRAINING_UNITS,
                    stripe_customer_id,
                    1,
                    idempotency_identifier=f"adapter-training:{job_id}",
                )
                await persist_api_metrics_row(
                    getattr(app_state, "pool", None),
                    inference_type="adapter_training",
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=1,
                    cost_usd=0.0,
                    latency_ms=float(final_status.get("duration_seconds") or 0.0)
                    * 1000.0,
                    user_id=resolve_metering_user_id(current_user),
                    stripe_customer_id=stripe_customer_id,
                    assistant_id=assistant_id,
                    thread_id=None,
                    model_name=adapter_block["base_model"],
                    meter_event_name=UsageMeter.ADAPTER_TRAINING_UNITS.value,
                )
        else:
            adapter_block = {
                "status": ADAPTER_STATUS_ERROR,
                "job_id": job_id,
                "finished_at": now,
                "error": final_status.get("error")
                or (
                    "cancelled"
                    if status == "cancelled"
                    else "training did not finish in time"
                ),
            }
        await write_avatar_adapter_metadata(
            current_user["API_KEY"], assistant_id, adapter_block
        )
        await _evict_api_key_cache_for_user(current_user.get("user_id"))
    except Exception as watch_error:  # noqa: BLE001
        logger.exception(
            "Adapter training watcher for %s failed: %s", job_id, watch_error
        )
    finally:
        watchers = getattr(app_state, "adapter_training_watchers", None)
        if isinstance(watchers, dict):
            watchers.pop(assistant_id, None)


async def start_adapter_training(
    app_state: Any,
    current_user: dict,
    assistant_id: str,
    assistant: dict,
    creator_id: str,
    *,
    trigger: str,
    parameters: dict | None = None,
) -> dict:
    """Start one adapter-training job for the avatar and watch it to completion.

    Shared by the explicit endpoint (``trigger="manual"``) and the automatic
    start after a media upload (``trigger="auto"``). Raises ``HTTPException``
    for every refusal so the endpoint reports the reason and the automatic
    path can log it.
    """
    from src.api.webapp import enforce_remaining_allotment, enforce_tier_capability

    context = app_state.context
    if not adapter_inference_enabled(context):
        raise HTTPException(
            status_code=503,
            detail=(
                "The adapter service is not configured "
                "(ADAPTER_SERVER_BASE_URL / ADAPTER_INFERENCE_ENABLED)."
            ),
        )
    enforce_tier_capability(current_user, TierCapability.TRAIN_ADAPTER)
    await enforce_remaining_allotment(
        app_state,
        current_user,
        UsageMeter.ADAPTER_TRAINING_UNITS,
        estimated_request_tokens=1,
        assistant_id=assistant_id,
    )
    existing = avatar_adapter_metadata(assistant.get("metadata"))
    watchers = getattr(app_state, "adapter_training_watchers", None)
    if not isinstance(watchers, dict):
        watchers = {}
        app_state.adapter_training_watchers = watchers
    if existing.get("status") == ADAPTER_STATUS_TRAINING and assistant_id in watchers:
        raise HTTPException(
            status_code=409,
            detail=f"An adapter is already training for this avatar (job {existing.get('job_id')}).",
        )
    minimum_quotes = int(
        getattr(context, "adapter_training_min_quote_documents", 200) or 0
    )
    quote_count = await count_quote_documents(
        app_state.store, creator_id, assistant_id, minimum_quotes
    )
    if quote_count < minimum_quotes:
        raise HTTPException(
            status_code=422,
            detail=(
                f"The avatar holds {quote_count} direct-quote documents; at least "
                f"{minimum_quotes} are needed to train an adapter. Upload more of the "
                "target's own words (transcripts, posts, messages) first."
            ),
        )

    client = AdapterServerClient(context, current_user["API_KEY"])
    try:
        started = await client.start_training(assistant_id, parameters=parameters)
    except AdapterTrainingDataMissing as missing_error:
        raise HTTPException(
            status_code=422, detail=str(missing_error)
        ) from missing_error
    except Exception as start_error:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"The adapter service could not start training: {start_error}",
        ) from start_error
    job_id = started.get("job_id")
    adapter_block = {
        "status": ADAPTER_STATUS_TRAINING,
        "job_id": job_id,
        "started_at": datetime.now(UTC).isoformat(),
        "trigger": trigger,
        "previous_trained_at": existing.get("trained_at"),
        "previous_adapter_s3_prefix": existing.get("adapter_s3_prefix"),
    }
    await write_avatar_adapter_metadata(
        current_user["API_KEY"], assistant_id, adapter_block
    )
    watchers[assistant_id] = asyncio.create_task(
        watch_adapter_training(
            app_state, current_user, assistant_id, job_id, started.get("model")
        )
    )
    return {
        "job_id": job_id,
        "status": ADAPTER_STATUS_TRAINING,
        "trigger": trigger,
        "adapter": adapter_block,
    }


class AdapterTrainingParameters(BaseModel):
    base_model: str | None = None
    learning_rate: float | None = None
    lora_rank: int | None = None
    lora_alpha: int | None = None
    num_generations: int | None = None
    max_completion_length: int | None = None


@adapter_route.post("/avatar/{assistant_id}/train_adapter", status_code=202)
async def train_avatar_adapter(
    request: Request,
    assistant_id: str,
    parameters: AdapterTrainingParameters | None = None,
    current_user: dict = Depends(get_current_user),
) -> JSONResponse:
    """Train (or retrain) the avatar's LoRA adapter on the adapter service.

    Premium only (``TRAIN_ADAPTER`` capability; one adapter-training unit is
    metered when the job completes). The avatar needs at least
    ``ADAPTER_TRAINING_MIN_QUOTE_DOCUMENTS`` direct-quote documents. Returns
    ``202`` with the job id; the avatar's ``metadata.adapter`` tracks the job
    and flips to ``trained`` on success, after which ``/message/{assistant_id}``
    prefers the adapter.
    """
    from src.api.webapp import resolve_assistant_for_creator

    assistant, creator_id = await resolve_assistant_for_creator(
        assistant_id,
        current_user,
        action_description="train an adapter for that avatar",
    )
    started = await start_adapter_training(
        request.app.state,
        current_user,
        assistant_id,
        assistant,
        creator_id,
        trigger="manual",
        parameters=parameters.model_dump(exclude_none=True) if parameters else None,
    )
    return JSONResponse(started, status_code=202)


@adapter_route.get("/avatar/{assistant_id}/adapter")
async def get_avatar_adapter(
    request: Request, assistant_id: str, current_user: dict = Depends(get_current_user)
) -> JSONResponse:
    """The avatar's adapter state: the recorded block plus the live job status when training."""
    from src.api.webapp import resolve_assistant_for_creator

    assistant, _creator_id = await resolve_assistant_for_creator(
        assistant_id, current_user, action_description="read that avatar's adapter"
    )
    adapter_block = avatar_adapter_metadata(assistant.get("metadata"))
    live_status = None
    if (
        adapter_block.get("status") == ADAPTER_STATUS_TRAINING
        and adapter_block.get("job_id")
        and adapter_inference_enabled(request.app.state.context)
    ):
        try:
            live_status = await AdapterServerClient(
                request.app.state.context, current_user["API_KEY"]
            ).training_status(adapter_block["job_id"])
        except Exception as status_error:  # noqa: BLE001
            live_status = {"error": str(status_error)}
    return JSONResponse(
        {
            "assistant_id": assistant_id,
            "adapter": adapter_block,
            "trained": avatar_has_trained_adapter(assistant.get("metadata")),
            "adapter_service_configured": adapter_inference_enabled(
                request.app.state.context
            ),
            "training_status": live_status,
        }
    )


@adapter_route.get("/avatar/{assistant_id}/adapter/training_progress")
async def stream_avatar_adapter_training_progress(
    request: Request, assistant_id: str, current_user: dict = Depends(get_current_user)
) -> StreamingResponse:
    """Pass the adapter service's training progress stream through, as server-sent events."""
    from src.api.webapp import resolve_assistant_for_creator

    assistant, _creator_id = await resolve_assistant_for_creator(
        assistant_id,
        current_user,
        action_description="watch that avatar's adapter training",
    )
    adapter_block = avatar_adapter_metadata(assistant.get("metadata"))
    job_id = adapter_block.get("job_id")
    if not job_id:
        raise HTTPException(
            status_code=404, detail="No adapter training job for this avatar."
        )
    client = AdapterServerClient(request.app.state.context, current_user["API_KEY"])

    async def _relay():
        async for line in client.stream_training_progress(job_id):
            yield line + "\n"

    return StreamingResponse(
        _relay(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@adapter_route.post("/avatar/{assistant_id}/adapter/cancel")
async def cancel_avatar_adapter_training(
    request: Request, assistant_id: str, current_user: dict = Depends(get_current_user)
) -> JSONResponse:
    """Cancel the avatar's in-progress adapter training job."""
    from src.api.webapp import resolve_assistant_for_creator

    assistant, _creator_id = await resolve_assistant_for_creator(
        assistant_id,
        current_user,
        action_description="cancel that avatar's adapter training",
    )
    adapter_block = avatar_adapter_metadata(assistant.get("metadata"))
    if adapter_block.get("status") != ADAPTER_STATUS_TRAINING or not adapter_block.get(
        "job_id"
    ):
        raise HTTPException(
            status_code=409,
            detail="No adapter training is in progress for this avatar.",
        )
    client = AdapterServerClient(request.app.state.context, current_user["API_KEY"])
    result = await client.cancel_training(adapter_block["job_id"])
    return JSONResponse(result)


__all__ = [
    "AdapterTrainingParameters",
    "adapter_route",
    "cancel_avatar_adapter_training",
    "count_quote_documents",
    "get_avatar_adapter",
    "resolve_adapter_preference",
    "start_adapter_training",
    "stream_avatar_adapter_training_progress",
    "train_avatar_adapter",
    "watch_adapter_training",
    "write_avatar_adapter_metadata",
]
