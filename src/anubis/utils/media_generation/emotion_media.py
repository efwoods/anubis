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

The prompt family is chosen by the reference **subject** (a person, a stylized
character, or a non-human image with no face — see ``prompts.py``); the caller
classifies the reference once with ``reference_subject.classify_reference_subject``
and passes the answer in. Each failure carries an ``error_code`` and a
``message`` (``describe_failure``) so a moderation refusal — which retrying
only repeats, at the same charge — is told apart from a transient error.

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
    normalize_reference_subject,
    still_prompt_for,
)
from src.anubis.utils.media_generation.reference_subject import (
    moderation_blocks_generation,
    moderation_warning,
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


def describe_failure(
    emotion: str, asset_kind: str, error: Exception | str
) -> dict[str, str]:
    """One failure entry: what did not generate, the vendor's words, and why.

    ``error_code`` is one of the ``xai_client.ERROR_CODE_*`` values so a client
    can tell a moderation refusal (retrying repeats the charge) from a
    transient vendor error (retrying is reasonable); ``message`` is the
    sentence to show a person.
    """
    error_text = str(error)
    error_code, message = xai_client.failure_reason(error_text, asset_kind)
    return {
        "emotion": emotion,
        "asset_kind": asset_kind,
        "error": error_text,
        "error_code": error_code,
        "message": message,
    }


def describe_predicted_failure(
    emotion: str, asset_kind: str, warning: str
) -> dict[str, str]:
    """Describe an asset withheld because the vendor's refusal was predicted."""
    return {
        "emotion": emotion,
        "asset_kind": asset_kind,
        "error": "Withheld: the reference image would be refused by content moderation.",
        "error_code": xai_client.ERROR_CODE_MODERATION_PREDICTED,
        "message": warning,
    }


def summarize_failures(failures: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse per-asset failures into the counts and sentence a toast shows.

    Returns ``{"failed_stills", "failed_loops", "moderated", "predicted",
    "message"}``. ``predicted`` counts assets withheld before any call was
    made (nothing charged); ``moderated`` counts assets the vendor rendered
    and then refused (charged). ``message`` is empty when nothing failed.
    """
    failed_stills = sum(1 for f in failures if f.get("asset_kind") == ASSET_KIND_STILL)
    failed_loops = sum(
        1 for f in failures if f.get("asset_kind") == ASSET_KIND_IDLE_LOOP
    )
    moderated = sum(
        1
        for f in failures
        if f.get("error_code") == xai_client.ERROR_CODE_CONTENT_MODERATED
    )
    predicted = sum(
        1
        for f in failures
        if f.get("error_code") == xai_client.ERROR_CODE_MODERATION_PREDICTED
    )
    if not failures:
        return {
            "failed_stills": 0,
            "failed_loops": 0,
            "moderated": 0,
            "predicted": 0,
            "message": "",
        }
    stopped = next(
        (f for f in failures if f.get("error_code") in xai_client.STOP_RUN_ERROR_CODES),
        None,
    )
    if stopped is not None:
        # The first stop-condition failure explains the whole run; the rest
        # were not attempted because of it.
        return {
            "failed_stills": failed_stills,
            "failed_loops": failed_loops,
            "moderated": moderated,
            "predicted": predicted,
            "message": str(stopped.get("message") or ""),
        }
    if predicted == len(failures):
        # Every entry carries the same warning sentence; it already says
        # nothing was charged and what to change.
        return {
            "failed_stills": failed_stills,
            "failed_loops": failed_loops,
            "moderated": 0,
            "predicted": predicted,
            "message": str(failures[0].get("message") or ""),
        }
    parts = []
    if failed_stills:
        parts.append(f"{failed_stills} portrait{'s' if failed_stills != 1 else ''}")
    if failed_loops:
        parts.append(f"{failed_loops} emotion video{'s' if failed_loops != 1 else ''}")
    what = " and ".join(parts)
    if moderated == len(failures):
        message = (
            f"xAI's content moderation refused {what}. The rendering charge "
            "stands, and retrying with the same reference image repeats both "
            "the charge and the refusal. Use a different reference image: a "
            "calm head-and-shoulders portrait passes far more often than a "
            "full-body action pose, a weapon, or a well-known trademarked "
            "character."
        )
    elif moderated:
        message = (
            f"{what} could not be generated; xAI's content moderation refused "
            f"{moderated} of them. Retrying repeats the charge for the refused "
            "ones, so consider a calmer, head-and-shoulders reference image."
        )
    else:
        message = f"{what} could not be generated. Retrying is reasonable."
    return {
        "failed_stills": failed_stills,
        "failed_loops": failed_loops,
        "moderated": moderated,
        "predicted": predicted,
        "message": message,
    }


def _motion_prompt_for(motion_prompts: dict[str, str] | None, emotion: str) -> str | None:
    """Return the measured motion block for ``emotion``, falling back to the neutral block."""
    if not motion_prompts:
        return None
    return motion_prompts.get(emotion) or motion_prompts.get("neutral") or None


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


def emotion_media_cost_estimate(
    context: Any, *, still_count: int, idle_loop_count: int
) -> dict[str, Any]:
    """Estimate what one generation run of this size costs at the vendor, in US dollars.

    Images are priced per generated still and video per second of idle loop, at
    the configured unit costs — the same numbers the run records per call in
    ``api_metrics``. The neutral still is never counted: the reference image is
    the neutral still, so storing that one costs nothing.

    Args:
        context: The ``GlobalContext`` holding the xAI unit costs.
        still_count: Stills that would be generated (never the neutral one).
        idle_loop_count: Idle loops that would be rendered.

    Returns:
        The counts, the unit costs, and the totals, so the settings screen can
        show the owner the arithmetic before spending anything.
    """
    image_cost = float(getattr(context, "xai_image_cost_per_image_usd", None) or 0.04)
    video_cost_per_second = float(
        getattr(context, "xai_video_cost_per_second_usd", None) or 0.08
    )
    idle_loop_seconds = int(getattr(context, "xai_idle_loop_duration_seconds", None) or 6)
    stills_usd = still_count * image_cost
    loops_usd = idle_loop_count * idle_loop_seconds * video_cost_per_second
    return {
        "stills": still_count,
        "idle_loops": idle_loop_count,
        "image_cost_usd": round(image_cost, 4),
        "video_cost_per_second_usd": round(video_cost_per_second, 4),
        "idle_loop_seconds": idle_loop_seconds,
        "stills_usd": round(stills_usd, 2),
        "idle_loops_usd": round(loops_usd, 2),
        "total_usd": round(stills_usd + loops_usd, 2),
    }


def full_build_asset_counts() -> tuple[int, int]:
    """Return the stills and idle loops a complete build generates.

    One still per emotion the vendor draws (every base emotion except neutral,
    which the reference image already is) and one idle loop per base emotion.
    """
    return (len(GENERATED_EMOTIONS), len(BASE_EMOTIONS))


def missing_asset_counts(missing: list[str] | tuple[str, ...] | None) -> tuple[int, int]:
    """Return the stills and idle loops an ``only_missing`` run would generate.

    ``missing`` is the manifest's list of ``"<emotion>:<asset_kind>"`` entries.
    A missing neutral still is not counted: that one is copied from the
    reference image rather than generated.
    """
    still_count = 0
    idle_loop_count = 0
    for entry in missing or ():
        emotion, _, kind = str(entry).partition(":")
        if kind == ASSET_KIND_IDLE_LOOP:
            idle_loop_count += 1
        elif kind == ASSET_KIND_STILL and emotion != NEUTRAL_EMOTION:
            still_count += 1
    return (still_count, idle_loop_count)


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
    subject: str | None = None,
    assessment: dict[str, Any] | None = None,
    proceed_despite_moderation_risk: bool = False,
    progress: ProgressCallback | None = None,
    metrics: MetricsCallback | None = None,
    motion_prompts: dict[str, str] | None = None,
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
        motion_prompts: The person's measured motion block per emotion (see
            ``src/anubis/utils/motion/motion_prompt.py``), keyed by emotion
            with ``neutral`` as the fallback. A still takes the carriage lines;
            an idle loop replaces its generic breathe-blink-fidget clause with
            the block. ``None`` or an empty dict means the generic prompts.
        subject: What the reference depicts (``person``,
            ``stylized_character``, ``non_human``); picks the prompt family.
            ``None`` means ``person``.
        assessment: The reference assessment from
            ``reference_subject.classify_reference_subject``. When its
            ``moderation_risk`` is high, **no vendor call is made**: every
            requested asset is reported as a ``moderation_predicted`` failure
            carrying the warning sentence, so nothing is charged for a video
            the vendor would refuse after rendering.
        proceed_despite_moderation_risk: The owner's explicit choice to
            attempt generation anyway, at their own cost.
        progress: Called with ``(stage, fields)`` as assets complete.
        metrics: Awaited with ``(inference_type, cost_usd, model, request_id)``
            per vendor call, for the ``api_metrics`` ledger.

    Returns:
        The manifest (see :func:`build_manifest`) plus ``"failures"`` — a list
        of ``{"emotion", "asset_kind", "error", "error_code", "message"}`` for
        anything that did not generate (see :func:`describe_failure`) — and
        ``"subject"``, the prompt family used. Never raises for a vendor failure; raises
        ``XaiNotConfiguredError`` when no key is configured.
    """
    report_progress = progress or _noop_progress
    record_metric = metrics or _noop_metrics
    failures: list[dict[str, str]] = []
    reference_subject = normalize_reference_subject(
        subject if subject is not None else (assessment or {}).get("subject")
    )

    target_emotions = tuple(emotions) if emotions else BASE_EMOTIONS
    kinds = (
        tuple(asset_kinds) if asset_kinds else (ASSET_KIND_STILL, ASSET_KIND_IDLE_LOOP)
    )
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
    # Set by the first failure whose code means every further call would fail
    # the same way (the xAI team is out of credits); the remaining assets are
    # then reported as not attempted instead of each being tried and billed.
    stop_reason: dict[str, str] = {}

    def _record_failure(failure: dict[str, str]) -> None:
        failures.append(failure)
        if (
            failure.get("error_code") in xai_client.STOP_RUN_ERROR_CODES
            and not stop_reason
        ):
            stop_reason.update(failure)
            logger.warning(
                "Stopping emotion media for %s after %s: %s",
                assistant_id,
                failure.get("error_code"),
                failure.get("error"),
            )

    def _not_attempted(emotion: str, asset_kind: str) -> dict[str, str]:
        return {
            "emotion": emotion,
            "asset_kind": asset_kind,
            "error": "Not attempted: an earlier call in this run failed with "
            f"{stop_reason.get('error_code')}.",
            "error_code": xai_client.ERROR_CODE_NOT_ATTEMPTED,
            "message": str(stop_reason.get("message") or ""),
        }

    async def _load_existing_still_uri(emotion: str) -> None:
        existing_asset = next(
            (
                asset
                for asset in existing
                if asset["emotion"] == emotion
                and asset["asset_kind"] == ASSET_KIND_STILL
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
    if is_full_build or (generate_stills and NEUTRAL_EMOTION in target_emotions):
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

    # Pre-flight: a reference the vendor's moderation would refuse is caught
    # here, before the first call, and the owner is told what to change. The
    # neutral still is the owner's own image and has already been stored.
    if moderation_blocks_generation(assessment) and not proceed_despite_moderation_risk:
        warning = moderation_warning(assessment)
        for emotion in stills_to_make:
            failures.append(
                describe_predicted_failure(emotion, ASSET_KIND_STILL, warning)
            )
        for emotion in loops_to_make:
            failures.append(
                describe_predicted_failure(emotion, ASSET_KIND_IDLE_LOOP, warning)
            )
        logger.warning(
            "Emotion media withheld for %s: %s",
            assistant_id,
            (assessment or {}).get("moderation_reasons"),
        )
        return await _finish(
            repository,
            assistant_id,
            failures,
            reference_subject,
            report_progress,
            assessment=assessment,
            withheld=True,
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
        if stop_reason:
            failures.append(_not_attempted(emotion, ASSET_KIND_STILL))
            return
        prompt = _with_extra_prompt(
            still_prompt_for(
                emotion, reference_subject, motion_prompt=_motion_prompt_for(motion_prompts, emotion)
            ),
            extra_prompt,
        )
        try:
            result = await xai_client.edit_image(
                context,
                reference_image_data_uri=reference_image_data_uri,
                prompt=prompt,
            )
        except xai_client.XaiGenerationError as generation_error:
            _record_failure(
                describe_failure(emotion, ASSET_KIND_STILL, generation_error)
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
        if stop_reason:
            failures.append(_not_attempted(emotion, ASSET_KIND_IDLE_LOOP))
            return
        still_uri = still_uris.get(emotion)
        if not still_uri:
            failures.append(
                describe_failure(
                    emotion, ASSET_KIND_IDLE_LOOP, "No still was available to animate."
                )
            )
            return
        prompt = _with_extra_prompt(
            idle_loop_prompt_for(
                emotion, reference_subject, motion_prompt=_motion_prompt_for(motion_prompts, emotion)
            ),
            extra_prompt,
        )
        try:
            result = await xai_client.generate_idle_loop(
                context, still_image_data_uri=still_uri, prompt=prompt
            )
        except xai_client.XaiGenerationError as generation_error:
            _record_failure(
                describe_failure(emotion, ASSET_KIND_IDLE_LOOP, generation_error)
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

    return await _finish(
        repository,
        assistant_id,
        failures,
        reference_subject,
        report_progress,
        assessment=assessment,
        withheld=False,
    )


async def _finish(
    repository: Any,
    assistant_id: str,
    failures: list[dict[str, str]],
    reference_subject: str,
    report_progress: ProgressCallback,
    *,
    assessment: dict[str, Any] | None,
    withheld: bool,
) -> dict[str, Any]:
    """Build the manifest and emit the completion frame.

    The frame carries why things failed, not only how many: the upload toast
    marks the step in error with the reason instead of ticking seven refused
    videos as done, and says when nothing was charged.
    """
    manifest = build_manifest(await repository.list_emotion_assets(assistant_id))
    manifest["failures"] = failures
    manifest["subject"] = reference_subject
    manifest["withheld"] = withheld
    manifest["moderation_risk"] = (assessment or {}).get("moderation_risk")
    manifest["moderation_reasons"] = list(
        (assessment or {}).get("moderation_reasons") or []
    )
    summary = summarize_failures(failures)
    report_progress(
        STAGE_COMPLETE,
        {
            "complete": manifest["complete"],
            "failures": len(failures),
            "failed_stills": summary["failed_stills"],
            "failed_loops": summary["failed_loops"],
            "moderated": summary["moderated"],
            "predicted": summary["predicted"],
            "withheld": withheld,
            "failure_message": summary["message"],
            "failed_assets": [
                {
                    "emotion": f["emotion"],
                    "asset_kind": f["asset_kind"],
                    "error_code": f.get("error_code"),
                }
                for f in failures
            ],
            "subject": reference_subject,
            "moderation_reasons": manifest["moderation_reasons"],
        },
    )
    return manifest
