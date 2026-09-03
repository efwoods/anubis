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


def text_digest(text: str) -> str:
    """Stable key for one spoken text (whitespace- and case-insensitive)."""
    normalized = " ".join(str(text or "").lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def estimate_duration_seconds(text: str) -> float:
    """Roughly how long the clip will run, from the word count."""
    words = len(str(text or "").split())
    return max(1.0, math.ceil(words / _WORDS_PER_SECOND))


async def find_cached_clip(
    repository: Any, *, assistant_id: str, emotion: str, text: str
) -> dict[str, Any] | None:
    """Return the stored clip for this emotion + text, if one exists."""
    digest = text_digest(text)
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
) -> dict[str, Any]:
    """Begin (or short-circuit) a lip-sync clip for one reply.

    Returns ``{"status": "completed", "asset_id"}`` when a cached clip exists,
    otherwise ``{"status": "pending", "job_id", "generation_id"}``.
    """
    cached = await find_cached_clip(
        repository, assistant_id=assistant_id, emotion=emotion, text=text
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
    generation_id = await elevenlabs_client.create_lip_sync_video(
        context,
        model_id=str(
            getattr(context, "elevenlabs_lip_sync_model", None) or "creatify-aurora"
        ),
        image_asset_id=image_asset_id,
        audio_asset_id=audio_asset_id,
        resolution=str(
            getattr(context, "elevenlabs_lip_sync_resolution", None) or "720p"
        ),
    )
    job_id = await repository.create_job(
        user_id=user_id,
        assistant_id=assistant_id,
        job_kind=JOB_KIND_LIP_SYNC,
        vendor_reference=generation_id,
        state=JOB_STATE_RUNNING,
        detail={
            "emotion": emotion,
            "text_digest": text_digest(text),
            "characters": len(text),
            "estimated_seconds": estimate_duration_seconds(text),
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
        }
    )
    await repository.update_job(
        job["job_id"],
        state=JOB_STATE_COMPLETED,
        detail={"asset_id": asset_id, "newly_completed": True},
    )
    return {"status": "completed", "asset_id": asset_id, "newly_completed": True}
