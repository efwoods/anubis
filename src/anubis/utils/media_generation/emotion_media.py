"""Build an avatar's emotion media set from its reference image.

One reference image in; six stills and seven idle loops out, persisted to
``avatar_emotion_media`` so runtime is a pure lookup:

1. The reference is stored as the ``neutral`` still.
2. Six image edits run concurrently, one per generated emotion.
3. Seven image-to-video generations run concurrently — the neutral loop from
   the reference itself, the others from their stills.
4. Every completed asset is written the moment it completes, so a failure in
   one generation never loses the rest, and the missing ones can be retried by
   ``regenerate_missing_emotion_media``.

Spend is recorded per call in ``api_metrics`` (``image_generation`` /
``video_generation``) with the configured unit costs. Progress is reported
through the ``progress`` callback so the media-processing graph can forward the
stages to the upload toast.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from src.anubis.utils.media_assets.repository import (
    ASSET_KIND_IDLE_LOOP,
    ASSET_KIND_STILL,
)
from src.anubis.utils.media_generation import xai_client
from src.anubis.utils.media_generation.prompts import (
    BASE_EMOTIONS,
    GENERATED_EMOTIONS,
    NEUTRAL_EMOTION,
    idle_loop_prompt_for,
    still_prompt_for,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, dict[str, Any]], None]
MetricsCallback = Callable[[str, float, str, str | None], Awaitable[None]]

STAGE_STILLS = "emotion_stills"
STAGE_LOOPS = "idle_loops"
STAGE_COMPLETE = "emotion_media_complete"


def emotion_media_enabled(context: Any) -> bool:
    """Whether generation is switched on and a key is configured."""
    flag = str(getattr(context, "emotion_media_generation_enabled", None) or "true")
    return flag.strip().lower() in ("1", "true", "yes", "on") and bool(
        str(getattr(context, "xai_api_key", None) or "").strip()
    )


def _noop_progress(stage: str, fields: dict[str, Any]) -> None:
    return None


def _with_extra_prompt(base_prompt: str, extra_prompt: str | None) -> str:
    """Append the owner's improvement note, when they gave one."""
    extra = (extra_prompt or "").strip()
    if not extra:
        return base_prompt
    return f"{base_prompt} Additional direction from the owner: {extra}"


async def _noop_metrics(
    inference_type: str, cost_usd: float, model_name: str, request_id: str | None
) -> None:
    return None


def build_manifest(assets: list[dict[str, Any]]) -> dict[str, Any]:
    """Shape the stored assets into the manifest the client caches.

    ``{"emotions": {emotion: {"still": {...}, "idle_loop": {...}}}, "complete": bool}``
    where each entry carries ``asset_id``, ``mime_type``, ``url``, and for loops
    ``duration_seconds``. ``complete`` is true when every base emotion has both.
    """
    emotions: dict[str, dict[str, Any]] = {emotion: {} for emotion in BASE_EMOTIONS}
    for asset in assets:
        kind = asset.get("asset_kind")
        if kind not in (ASSET_KIND_STILL, ASSET_KIND_IDLE_LOOP):
            continue
        emotion = str(asset.get("emotion") or "")
        if emotion not in emotions:
            continue
        emotions[emotion][kind] = {
            "asset_id": asset.get("asset_id"),
            "mime_type": asset.get("mime_type"),
            "url": f"/avatar_emotion_media/{asset.get('asset_id')}",
            "duration_seconds": asset.get("duration_seconds"),
            "created_at": asset.get("created_at"),
        }
    complete = all(
        ASSET_KIND_STILL in entry and ASSET_KIND_IDLE_LOOP in entry
        for entry in emotions.values()
    )
    missing = [
        f"{emotion}:{kind}"
        for emotion, entry in emotions.items()
        for kind in (ASSET_KIND_STILL, ASSET_KIND_IDLE_LOOP)
        if kind not in entry
    ]
    return {"emotions": emotions, "complete": complete, "missing": missing}


async def _store_still(
    repository: Any,
    *,
    user_id: str,
    assistant_id: str,
    emotion: str,
    image_bytes: bytes,
    mime_type: str,
    vendor: str | None,
    request_id: str | None,
    prompt: str | None,
) -> str:
    return await repository.upsert_emotion_asset(
        {
            "user_id": user_id,
            "assistant_id": assistant_id,
            "emotion": emotion,
            "asset_kind": ASSET_KIND_STILL,
            "mime_type": mime_type,
            "bytes": image_bytes,
            "vendor": vendor,
            "vendor_request_id": request_id,
            "prompt": prompt,
        }
    )


async def generate_emotion_media_for_avatar(
    context: Any,
    repository: Any,
    *,
    user_id: str,
    assistant_id: str,
    reference_image_data_uri: str,
    only_missing: bool = False,
    emotions: tuple[str, ...] | None = None,
    asset_kinds: tuple[str, ...] | None = None,
    extra_prompt: str | None = None,
    progress: ProgressCallback | None = None,
    metrics: MetricsCallback | None = None,
) -> dict[str, Any]:
    """Generate and persist the full emotion set for one avatar.

    Args:
        context: The ``GlobalContext`` with the xAI settings and unit costs.
        repository: A media-asset repository (Postgres or in-memory).
        user_id: The avatar's owner.
        assistant_id: The avatar.
        reference_image_data_uri: The neutral reference as a data URI.
        only_missing: Skip emotions whose asset already exists (a retry).
        emotions: Limit generation to these base emotions. ``None`` means all.
        asset_kinds: Limit generation to ``still`` and/or ``idle_loop``.
        extra_prompt: Owner note appended to each generation prompt, for a
            targeted redo ("make the blink slower").
        progress: Called with ``(stage, fields)`` as assets complete.
        metrics: Awaited with ``(inference_type, cost_usd, model, request_id)``
            per vendor call, for the ``api_metrics`` ledger.

    Returns:
        The manifest (see :func:`build_manifest`) plus ``"failures"`` — a list
        of ``{"emotion", "asset_kind", "error"}`` for anything that did not
        generate. Never raises for a vendor failure; raises
        ``XaiNotConfiguredError`` when no key is configured.
    """
    report_progress = progress or _noop_progress
    record_metric = metrics or _noop_metrics
    failures: list[dict[str, str]] = []

    target_emotions = tuple(emotions) if emotions else BASE_EMOTIONS
    kinds = tuple(asset_kinds) if asset_kinds else (ASSET_KIND_STILL, ASSET_KIND_IDLE_LOOP)
    generate_stills = ASSET_KIND_STILL in kinds
    generate_loops = ASSET_KIND_IDLE_LOOP in kinds
    is_full_build = emotions is None and asset_kinds is None

    existing = await repository.list_emotion_assets(assistant_id)
    have = {(asset["emotion"], asset["asset_kind"]) for asset in existing}

    image_cost = float(getattr(context, "xai_image_cost_per_image_usd", None) or 0.04)
    video_cost_per_second = float(
        getattr(context, "xai_video_cost_per_second_usd", None) or 0.08
    )

    still_uris: dict[str, str] = {NEUTRAL_EMOTION: reference_image_data_uri}

    async def _load_existing_still_uri(emotion: str) -> None:
        existing_asset = next(
            (
                asset
                for asset in existing
                if asset["emotion"] == emotion and asset["asset_kind"] == ASSET_KIND_STILL
            ),
            None,
        )
        if existing_asset is None:
            return
        full = await repository.get_emotion_asset(existing_asset["asset_id"])
        if full and full.get("bytes"):
            still_uris[emotion] = xai_client._data_uri(
                full.get("mime_type") or "image/jpeg", full["bytes"]
            )

    # 1. The reference IS the neutral still. A targeted redo of one emotion
    #    leaves it alone; a full build always writes it so the manifest has it.
    if is_full_build or (
        generate_stills and NEUTRAL_EMOTION in target_emotions
    ):
        neutral_mime, neutral_bytes = xai_client._decode_data_uri(
            reference_image_data_uri
        )
        if not (only_missing and (NEUTRAL_EMOTION, ASSET_KIND_STILL) in have):
            await _store_still(
                repository,
                user_id=user_id,
                assistant_id=assistant_id,
                emotion=NEUTRAL_EMOTION,
                image_bytes=neutral_bytes,
                mime_type=neutral_mime,
                vendor=None,
                request_id=None,
                prompt=None,
            )
            have.add((NEUTRAL_EMOTION, ASSET_KIND_STILL))

    stills_to_make = (
        [emotion for emotion in GENERATED_EMOTIONS if emotion in target_emotions]
        if generate_stills
        else []
    )
    loops_to_make = (
        [emotion for emotion in BASE_EMOTIONS if emotion in target_emotions]
        if generate_loops
        else []
    )

    # Loops need the still they animate, even when this job is not remaking it.
    for emotion in loops_to_make:
        if emotion != NEUTRAL_EMOTION and emotion not in still_uris:
            await _load_existing_still_uri(emotion)

    # 2. Stills, concurrently.
    stills_done = 0
    stills_total = len(stills_to_make)

    async def _make_still(emotion: str) -> None:
        nonlocal stills_done
        if only_missing and (emotion, ASSET_KIND_STILL) in have:
            await _load_existing_still_uri(emotion)
            stills_done += 1
            report_progress(
                STAGE_STILLS,
                {
                    "current": stills_done,
                    "total": stills_total,
                    "emotion": emotion,
                    "asset_kind": ASSET_KIND_STILL,
                },
            )
            return
        prompt = _with_extra_prompt(still_prompt_for(emotion), extra_prompt)
        try:
            result = await xai_client.edit_image(
                context,
                reference_image_data_uri=reference_image_data_uri,
                prompt=prompt,
            )
        except xai_client.XaiGenerationError as generation_error:
            failures.append(
                {
                    "emotion": emotion,
                    "asset_kind": ASSET_KIND_STILL,
                    "error": str(generation_error),
                }
            )
            logger.warning(
                "Emotion still %s failed for %s: %s",
                emotion,
                assistant_id,
                generation_error,
            )
            return
        await record_metric(
            "image_generation", image_cost, result["model"], result.get("request_id")
        )
        await _store_still(
            repository,
            user_id=user_id,
            assistant_id=assistant_id,
            emotion=emotion,
            image_bytes=result["bytes"],
            mime_type=result["mime_type"],
            vendor="xai",
            request_id=result.get("request_id"),
            prompt=prompt,
        )
        still_uris[emotion] = xai_client._data_uri(result["mime_type"], result["bytes"])
        have.add((emotion, ASSET_KIND_STILL))
        stills_done += 1
        report_progress(
            STAGE_STILLS,
            {
                "current": stills_done,
                "total": stills_total,
                "emotion": emotion,
                "asset_kind": ASSET_KIND_STILL,
            },
        )

    if stills_to_make:
        report_progress(
            STAGE_STILLS,
            {"current": 0, "total": stills_total},
        )
        await asyncio.gather(*(_make_still(emotion) for emotion in stills_to_make))

    # 3. Idle loops, concurrently, each from its still.
    loops_done = 0
    loops_total = len(loops_to_make)

    async def _make_loop(emotion: str) -> None:
        nonlocal loops_done
        if only_missing and (emotion, ASSET_KIND_IDLE_LOOP) in have:
            loops_done += 1
            report_progress(
                STAGE_LOOPS,
                {
                    "current": loops_done,
                    "total": loops_total,
                    "emotion": emotion,
                    "asset_kind": ASSET_KIND_IDLE_LOOP,
                },
            )
            return
        still_uri = still_uris.get(emotion)
        if not still_uri:
            failures.append(
                {
                    "emotion": emotion,
                    "asset_kind": ASSET_KIND_IDLE_LOOP,
                    "error": "No still was available to animate.",
                }
            )
            return
        prompt = _with_extra_prompt(idle_loop_prompt_for(emotion), extra_prompt)
        try:
            result = await xai_client.generate_idle_loop(
                context, still_image_data_uri=still_uri, prompt=prompt
            )
        except xai_client.XaiGenerationError as generation_error:
            failures.append(
                {
                    "emotion": emotion,
                    "asset_kind": ASSET_KIND_IDLE_LOOP,
                    "error": str(generation_error),
                }
            )
            logger.warning(
                "Idle loop %s failed for %s: %s",
                emotion,
                assistant_id,
                generation_error,
            )
            return
        await record_metric(
            "video_generation",
            video_cost_per_second * float(result.get("duration_seconds") or 0.0),
            result["model"],
            result.get("request_id"),
        )
        await repository.upsert_emotion_asset(
            {
                "user_id": user_id,
                "assistant_id": assistant_id,
                "emotion": emotion,
                "asset_kind": ASSET_KIND_IDLE_LOOP,
                "mime_type": result["mime_type"],
                "bytes": result["bytes"],
                "duration_seconds": result.get("duration_seconds"),
                "vendor": "xai",
                "vendor_request_id": result.get("request_id"),
                "prompt": prompt,
            }
        )
        loops_done += 1
        report_progress(
            STAGE_LOOPS,
            {
                "current": loops_done,
                "total": loops_total,
                "emotion": emotion,
                "asset_kind": ASSET_KIND_IDLE_LOOP,
            },
        )

    if loops_to_make:
        report_progress(
            STAGE_LOOPS,
            {"current": 0, "total": loops_total},
        )
        await asyncio.gather(*(_make_loop(emotion) for emotion in loops_to_make))

    manifest = build_manifest(await repository.list_emotion_assets(assistant_id))
    manifest["failures"] = failures
    report_progress(
        STAGE_COMPLETE,
        {"complete": manifest["complete"], "failures": len(failures)},
    )
    return manifest
