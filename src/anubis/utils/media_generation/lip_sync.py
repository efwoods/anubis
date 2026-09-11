"""Lip-synced video replies: the emotion still plus the cloned voice → a clip.

Per reply, in voice mode with video enabled:

1. The reply is spoken with the active clone (bytes).
2. The emotion's still is uploaded to ElevenLabs once and its ``asset_id``
   cached on the asset row; the speech is uploaded as a second asset.
3. A ``/v1/flows/video`` generation is created and recorded as a durable job.
4. Polling downloads the finished clip and stores it as an
   ``avatar_emotion_media`` row of kind ``lip_sync``, keyed by the emotion and a
   digest of the text, so a repeated phrase (a greeting, a refusal) is served
   from the table instead of rendered again.

Cost is recorded per clip (``lip_sync``) using the configured per-second rate
and the clip's estimated duration, and reported to the video meter.

The generation carries a ``prompt`` — the vendor's behavioural channel — built
from a stable cinematic line plus the person's measured motion block
(``src/anubis/utils/motion/motion_prompt.py``), so the clip moves the way the
person moves. The clip cache key includes a digest of that prompt: a matured
motion profile renders a new clip rather than serving the old motion.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any

from src.anubis.utils.media_assets.repository import (
    ASSET_KIND_LIP_SYNC,
    ASSET_KIND_STILL,
    JOB_STATE_COMPLETED,
    JOB_STATE_FAILED,
    JOB_STATE_RUNNING,
)
from src.anubis.utils.voice import elevenlabs_client

logger = logging.getLogger(__name__)

JOB_KIND_LIP_SYNC = "lip_sync"

# Conversational pace, for estimating a clip's length before the vendor says.
_WORDS_PER_SECOND = 2.5


def lip_sync_enabled(context: Any) -> bool:
    """Whether lip-sync generation is switched on process-wide and a key exists."""
    flag = str(getattr(context, "lip_sync_enabled", None) or "true").strip().lower()
    return flag in ("1", "true", "yes", "on") and bool(
        str(
            getattr(context, "elevenlabs_api_key", None)
            or getattr(context, "nn_elevenlabs_api_key", None)
            or ""
        ).strip()
    )


def text_digest(text: str, prompt: str | None = None) -> str:
    """Return a stable key for one spoken text (whitespace- and case-insensitive).

    When a behavioural ``prompt`` drove the clip, the key carries a digest of
    that prompt too, so a clip rendered under an older motion profile is not
    served for a newer one.
    """
    normalized = " ".join(str(text or "").lower().split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
    if prompt and str(prompt).strip():
        prompt_normalized = " ".join(str(prompt).split())
        digest += "-" + hashlib.sha256(prompt_normalized.encode("utf-8")).hexdigest()[:8]
    return digest


def lip_sync_prompt_enabled(context: Any) -> bool:
    """Report whether the lip-sync generation may send the behavioural prompt."""
    flag = str(getattr(context, "lip_sync_prompt_enabled", None) or "true").strip().lower()
    return flag in ("1", "true", "yes", "on")


def build_lip_sync_prompt(context: Any, motion_prompt: str | None) -> str | None:
    """Build the layered prompt: cinematic foundation, then the person's measured motion."""
    if not lip_sync_prompt_enabled(context):
        return None
    from src.anubis.utils.motion.motion_prompt import compose_video_prompt

    cinematic = str(getattr(context, "lip_sync_cinematic_prompt", None) or "").strip()
    composed = compose_video_prompt(cinematic, motion_prompt or "")
    return composed or None


def estimate_duration_seconds(text: str) -> float:
    """Roughly how long the clip will run, from the word count."""
    words = len(str(text or "").split())
    return max(1.0, math.ceil(words / _WORDS_PER_SECOND))


async def find_cached_clip(
    repository: Any, *, assistant_id: str, emotion: str, text: str, prompt: str | None = None
) -> dict[str, Any] | None:
    """Return the stored clip for this emotion + text (+ prompt), if one exists."""
    digest = text_digest(text, prompt)
    for asset in await repository.list_emotion_assets(assistant_id):
        if (
            asset.get("asset_kind") == ASSET_KIND_LIP_SYNC
            and asset.get("emotion") == emotion
            and asset.get("variant_key") == digest
        ):
            return asset
    return None


async def _still_asset(
    repository: Any, assistant_id: str, emotion: str
) -> dict[str, Any] | None:
    for asset in await repository.list_emotion_assets(assistant_id):
        if (
            asset.get("asset_kind") == ASSET_KIND_STILL
            and asset.get("emotion") == emotion
        ):
            return asset
    return None


async def ensure_still_uploaded(
    context: Any, repository: Any, *, assistant_id: str, emotion: str
) -> str | None:
    """Upload the emotion still to ElevenLabs once; return its asset id."""
    still = await _still_asset(repository, assistant_id, emotion)
    if still is None:
        still = await _still_asset(repository, assistant_id, "neutral")
    if still is None:
        return None
    if still.get("elevenlabs_asset_id"):
        return str(still["elevenlabs_asset_id"])
    full = await repository.get_emotion_asset(still["asset_id"])
    if not full or not full.get("bytes"):
        return None
    extension = "png" if "png" in str(full.get("mime_type") or "") else "jpg"
    asset_id = await elevenlabs_client.upload_asset(
        context,
        payload=full["bytes"],
        name=f"{assistant_id}-{still.get('emotion')}.{extension}",
        mime_type=full.get("mime_type") or "image/jpeg",
    )
    await repository.upsert_emotion_asset({**full, "elevenlabs_asset_id": asset_id})
    return asset_id


async def start_lip_sync(
    context: Any,
    repository: Any,
    *,
    user_id: str,
    assistant_id: str,
    text: str,
    emotion: str,
    voice_id: str,
    motion_prompt: str | None = None,
) -> dict[str, Any]:
    """Begin (or short-circuit) a lip-sync clip for one reply.

    ``motion_prompt`` is the person's measured motion block for this emotion;
    it becomes the behavioural layer of the generation prompt.

    Returns ``{"status": "completed", "asset_id"}`` when a cached clip exists,
    otherwise ``{"status": "pending", "job_id", "generation_id"}``.
    """
    prompt = build_lip_sync_prompt(context, motion_prompt)
    cached = await find_cached_clip(
        repository, assistant_id=assistant_id, emotion=emotion, text=text, prompt=prompt
    )
    if cached is not None:
        return {"status": "completed", "asset_id": cached["asset_id"], "cached": True}

    image_asset_id = await ensure_still_uploaded(
        context, repository, assistant_id=assistant_id, emotion=emotion
    )
    if image_asset_id is None:
        raise elevenlabs_client.ElevenLabsError(
            "The avatar has no emotion still to animate; generate emotion media first."
        )
    model_id = str(
        getattr(context, "elevenlabs_text_to_speech_model", None) or "eleven_flash_v2_5"
    )
    speech_bytes = await elevenlabs_client.synthesize_speech(
        context, voice_id=voice_id, text=text, model_id=model_id
    )
    audio_asset_id = await elevenlabs_client.upload_asset(
        context,
        payload=speech_bytes,
        name=f"{assistant_id}-speech.mp3",
        mime_type="audio/mpeg",
    )
    model_id = str(getattr(context, "elevenlabs_lip_sync_model", None) or "creatify-aurora")
    resolution = str(getattr(context, "elevenlabs_lip_sync_resolution", None) or "720p")
    prompt_sent = bool(prompt)
    # The field is passed only when there is one, so a vendor client (or a
    # test double) that predates it is called exactly as before.
    prompt_arguments = {"prompt": prompt} if prompt else {}
    try:
        generation_id = await elevenlabs_client.create_lip_sync_video(
            context,
            model_id=model_id,
            image_asset_id=image_asset_id,
            audio_asset_id=audio_asset_id,
            resolution=resolution,
            **prompt_arguments,
        )
    except elevenlabs_client.ElevenLabsError as vendor_error:
        # A vendor model that does not know the field refuses the whole
        # request; retry once without it and record that on the job so the
        # fallback is not rediscovered per clip.
        if not prompt or "prompt" not in str(vendor_error).lower():
            raise
        logger.info("Lip-sync vendor refused the prompt field; retrying without it: %s", vendor_error)
        prompt_sent = False
        generation_id = await elevenlabs_client.create_lip_sync_video(
            context,
            model_id=model_id,
            image_asset_id=image_asset_id,
            audio_asset_id=audio_asset_id,
            resolution=resolution,
        )
    job_id = await repository.create_job(
        user_id=user_id,
        assistant_id=assistant_id,
        job_kind=JOB_KIND_LIP_SYNC,
        vendor_reference=generation_id,
        state=JOB_STATE_RUNNING,
        detail={
            "emotion": emotion,
            "text_digest": text_digest(text, prompt),
            "characters": len(text),
            "estimated_seconds": estimate_duration_seconds(text),
            "prompt": prompt or "",
            "prompt_sent": prompt_sent,
        },
    )
    return {"status": "pending", "job_id": job_id, "generation_id": generation_id}


async def poll_lip_sync(
    context: Any, repository: Any, *, job: dict[str, Any]
) -> dict[str, Any]:
    """Check one running lip-sync job; store the clip when it has completed.

    Returns ``{"status": "pending"|"completed"|"failed", "asset_id"?}``. The
    caller records spend when ``completed`` is first reached.
    """
    if job.get("state") == JOB_STATE_COMPLETED:
        return {
            "status": "completed",
            "asset_id": (job.get("detail") or {}).get("asset_id"),
        }
    if job.get("state") == JOB_STATE_FAILED:
        return {"status": "failed"}
    generation = await elevenlabs_client.get_lip_sync_video(
        context, generation_id=str(job.get("vendor_reference"))
    )
    status = generation.get("status")
    if status in ("pending", "generating", "processing", "queued", ""):
        return {"status": "pending"}
    if status != "completed" or not generation.get("content_url"):
        await repository.update_job(
            job["job_id"], state=JOB_STATE_FAILED, detail={"vendor_status": status}
        )
        return {"status": "failed"}
    video_bytes, content_type = await elevenlabs_client.download(
        generation["content_url"]
    )
    detail = job.get("detail") or {}
    asset_id = await repository.upsert_emotion_asset(
        {
            "user_id": job["user_id"],
            "assistant_id": job["assistant_id"],
            "emotion": detail.get("emotion") or "neutral",
            "asset_kind": ASSET_KIND_LIP_SYNC,
            "variant_key": detail.get("text_digest") or "",
            "mime_type": (content_type.split(";")[0] if content_type else None)
            or generation.get("content_mime_type")
            or "video/mp4",
            "bytes": video_bytes,
            "duration_seconds": detail.get("estimated_seconds"),
            "vendor": "elevenlabs",
            "vendor_request_id": job.get("vendor_reference"),
            "prompt": detail.get("prompt") or None,
        }
    )
    await repository.update_job(
        job["job_id"],
        state=JOB_STATE_COMPLETED,
        detail={"asset_id": asset_id, "newly_completed": True},
    )
    return {"status": "completed", "asset_id": asset_id, "newly_completed": True}
