"""The voice-clone corpus: collecting the avatar's own speech and cloning from it.

Every avatar gets an **instant** clone from its reference audio once at least
``ELEVENLABS_INSTANT_VOICE_CLONE_MINIMUM_SECONDS`` (60 s) of target-only speech
has been collected; the clone is rebuilt once the corpus reaches the target
(120 s) if the first clone used less. The **personal** avatar keeps collecting
from every audio/video upload — only the owner's turns, as isolated by the
diarizer — until ``ELEVENLABS_PROFESSIONAL_VOICE_CLONE_MINIMUM_SECONDS`` (30 min)
is reached, at which point a professional voice is created and its samples
attached; the owner then verifies with a spoken CAPTCHA and training starts
(three to six hours). The professional voice replaces the instant one once it
reports ``fine_tuned``.

One collection process therefore serves both clones: the first two minutes of
the personal avatar's corpus are its instant clone.

Clips live in ``avatar_voice_clips`` (bytes) and the running state in
``avatar_voice`` (see ``media_assets/repository.py``); nothing here talks to the
store.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from src.anubis.utils.media_assets.repository import (
    VOICE_STATE_AWAITING_VERIFICATION,
    VOICE_STATE_COLLECTING,
    VOICE_STATE_FAILED,
    VOICE_STATE_FINE_TUNED,
    VOICE_STATE_NOT_STARTED,
    VOICE_STATE_TRAINING,
)
from src.anubis.utils.voice import elevenlabs_client

logger = logging.getLogger(__name__)

CLIP_SOURCE_RECORDER = "recorder"
CLIP_SOURCE_REFERENCE_UPLOAD = "reference_upload"
CLIP_SOURCE_MEDIA_UPLOAD = "media_upload"


@dataclass
class VoiceThresholds:
    """The second counts that gate each clone, read from the context."""

    instant_minimum: float
    instant_target: float
    professional_minimum: float
    professional_maximum: float

    @classmethod
    def from_context(cls, context: Any) -> VoiceThresholds:
        """Read the four thresholds, with the cost report's defaults."""
        return cls(
            instant_minimum=float(
                getattr(context, "elevenlabs_instant_voice_clone_minimum_seconds", None)
                or 60
            ),
            instant_target=float(
                getattr(context, "elevenlabs_instant_voice_clone_target_seconds", None)
                or 120
            ),
            professional_minimum=float(
                getattr(
                    context, "elevenlabs_professional_voice_clone_minimum_seconds", None
                )
                or 1800
            ),
            professional_maximum=float(
                getattr(
                    context, "elevenlabs_professional_voice_clone_maximum_seconds", None
                )
                or 10800
            ),
        )


@dataclass
class VoiceStatus:
    """What ``GET /avatar_voice`` reports."""

    assistant_id: str
    collected_seconds: float
    instant_voice_id: str | None
    instant_voice_seconds: float
    professional_voice_id: str | None
    professional_state: str
    active_voice: str  # instant | professional | none
    active_voice_id: str | None
    instant_minimum_seconds: float
    instant_target_seconds: float
    professional_minimum_seconds: float
    professional_maximum_seconds: float
    professional_eligible: bool
    clip_count: int
    verification_requested_at: str | None = None
    training_started_at: str | None = None
    detail: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Plain dictionary for a JSON response."""
        return asdict(self)


def _decode_data_uri(data_uri: str) -> tuple[bytes, str]:
    header, _, encoded = str(data_uri).partition(",")
    mime_type = header[5:].split(";", 1)[0] or "audio/mpeg"
    if mime_type == "audio/mp3":
        mime_type = "audio/mpeg"
    return base64.b64decode(encoded), mime_type


def voice_configured(context: Any) -> bool:
    """Whether an ElevenLabs key is present."""
    return bool(
        str(
            getattr(context, "elevenlabs_api_key", None)
            or getattr(context, "nn_elevenlabs_api_key", None)
            or ""
        ).strip()
    )


async def _voice_record(
    repository: Any, user_id: str, assistant_id: str
) -> dict[str, Any]:
    record = await repository.get_voice(assistant_id) or {}
    record.setdefault("assistant_id", assistant_id)
    record.setdefault("user_id", user_id)
    record.setdefault("professional_state", VOICE_STATE_NOT_STARTED)
    record.setdefault("collected_seconds", 0.0)
    record.setdefault("instant_voice_id", None)
    record.setdefault("instant_voice_seconds", 0.0)
    record.setdefault("professional_voice_id", None)
    record.setdefault("detail", {})
    return record


async def _clips_for_clone(
    repository: Any, assistant_id: str, *, max_seconds: float
) -> tuple[list[tuple[str, bytes, str]], float]:
    """Return the oldest clips up to ``max_seconds`` total, as SDK file tuples."""
    clips = await repository.list_voice_clips(assistant_id, include_bytes=True)
    files: list[tuple[str, bytes, str]] = []
    total = 0.0
    for index, clip in enumerate(clips):
        duration = float(clip.get("duration_seconds") or 0.0)
        if total >= max_seconds:
            break
        mime_type = clip.get("mime_type") or "audio/mpeg"
        extension = "mp3" if "mpeg" in mime_type or "mp3" in mime_type else "wav"
        files.append((f"clip_{index:03d}.{extension}", clip["bytes"], mime_type))
        total += duration
    return files, total


async def ensure_instant_voice(
    repository: Any,
    context: Any,
    *,
    user_id: str,
    assistant_id: str,
    avatar_name: str = "",
) -> dict[str, Any]:
    """Create (or upgrade) the instant clone when the corpus allows it.

    Idempotent: nothing happens below the minimum, or when a clone already
    exists that was built from at least the target seconds.
    """
    thresholds = VoiceThresholds.from_context(context)
    record = await _voice_record(repository, user_id, assistant_id)
    collected = float(await repository.total_voice_seconds(assistant_id))
    record["collected_seconds"] = collected

    if collected < thresholds.instant_minimum or not voice_configured(context):
        await repository.upsert_voice(record)
        return record

    existing_id = record.get("instant_voice_id")
    built_from = float(record.get("instant_voice_seconds") or 0.0)
    should_build = existing_id is None or (
        built_from < thresholds.instant_target
        and collected >= thresholds.instant_target
    )
    if not should_build:
        await repository.upsert_voice(record)
        return record

    files, used_seconds = await _clips_for_clone(
        repository, assistant_id, max_seconds=thresholds.instant_target
    )
    if not files:
        await repository.upsert_voice(record)
        return record
    label = avatar_name or assistant_id
    try:
        voice_id = await elevenlabs_client.create_instant_voice(
            context,
            name=f"{label} (instant)"[:80],
            clips=files,
            description="Neural Nexus instant voice clone",
        )
    except elevenlabs_client.ElevenLabsError as clone_error:
        logger.warning("Instant clone failed for %s: %s", assistant_id, clone_error)
        record["detail"] = {
            **record.get("detail", {}),
            "instant_error": str(clone_error),
        }
        await repository.upsert_voice(record)
        return record

    if existing_id and existing_id != voice_id:
        await elevenlabs_client.delete_voice(context, existing_id)
    record["instant_voice_id"] = voice_id
    record["instant_voice_seconds"] = used_seconds
    record["detail"] = {
        k: v for k, v in record.get("detail", {}).items() if k != "instant_error"
    }
    await repository.upsert_voice(record)
    logger.info(
        "Instant voice %s created for %s from %.0fs",
        voice_id,
        assistant_id,
        used_seconds,
    )
    return record


async def prepare_professional_voice(
    repository: Any,
    context: Any,
    *,
    user_id: str,
    assistant_id: str,
    avatar_name: str = "",
    language: str = "en",
) -> dict[str, Any]:
    """Create the professional voice and attach the corpus, once the minimum is met.

    Leaves the record in ``awaiting_verification``: the owner must read the
    CAPTCHA aloud (``submit_verification_and_train``) before training starts.
    """
    thresholds = VoiceThresholds.from_context(context)
    record = await _voice_record(repository, user_id, assistant_id)
    collected = float(await repository.total_voice_seconds(assistant_id))
    record["collected_seconds"] = collected
    if (
        record.get("professional_state")
        not in (VOICE_STATE_NOT_STARTED, VOICE_STATE_COLLECTING, VOICE_STATE_FAILED)
        or collected < thresholds.professional_minimum
        or not voice_configured(context)
    ):
        if (
            collected < thresholds.professional_minimum
            and record.get("professional_state") == VOICE_STATE_NOT_STARTED
        ):
            record["professional_state"] = VOICE_STATE_COLLECTING
        await repository.upsert_voice(record)
        return record

    files, used_seconds = await _clips_for_clone(
        repository, assistant_id, max_seconds=thresholds.professional_maximum
    )
    label = avatar_name or assistant_id
    try:
        voice_id = record.get(
            "professional_voice_id"
        ) or await elevenlabs_client.create_professional_voice(
            context,
            name=f"{label} (professional)"[:80],
            language=language,
            description="Neural Nexus professional voice clone",
        )
        await elevenlabs_client.add_professional_samples(
            context, voice_id=voice_id, clips=files
        )
    except elevenlabs_client.ElevenLabsError as clone_error:
        logger.warning(
            "Professional clone preparation failed for %s: %s",
            assistant_id,
            clone_error,
        )
        record["professional_state"] = VOICE_STATE_FAILED
        record["detail"] = {
            **record.get("detail", {}),
            "professional_error": str(clone_error),
        }
        await repository.upsert_voice(record)
        return record

    record["professional_voice_id"] = voice_id
    record["professional_state"] = VOICE_STATE_AWAITING_VERIFICATION
    record["verification_requested_at"] = datetime.now(UTC).isoformat()
    record["detail"] = {
        **{
            k: v
            for k, v in record.get("detail", {}).items()
            if k != "professional_error"
        },
        "professional_sample_seconds": used_seconds,
    }
    await repository.upsert_voice(record)
    return record


async def submit_verification_and_train(
    repository: Any,
    context: Any,
    *,
    user_id: str,
    assistant_id: str,
    recording: tuple[str, bytes, str],
) -> dict[str, Any]:
    """Submit the owner's CAPTCHA recording, then start training."""
    record = await _voice_record(repository, user_id, assistant_id)
    voice_id = record.get("professional_voice_id")
    if (
        not voice_id
        or record.get("professional_state") != VOICE_STATE_AWAITING_VERIFICATION
    ):
        raise ValueError("The professional voice is not awaiting verification.")
    verification = await elevenlabs_client.submit_verification_recording(
        context, voice_id=voice_id, recording=recording
    )
    model_id = str(
        getattr(context, "elevenlabs_professional_voice_clone_training_model", None)
        or "eleven_multilingual_v2"
    )
    training = await elevenlabs_client.train_professional_voice(
        context, voice_id=voice_id, model_id=model_id
    )
    record["professional_state"] = VOICE_STATE_TRAINING
    record["training_started_at"] = datetime.now(UTC).isoformat()
    record["detail"] = {
        **record.get("detail", {}),
        "verification": verification,
        "training": training,
        "training_model": model_id,
    }
    await repository.upsert_voice(record)
    return record


async def refresh_training_state(
    repository: Any, context: Any, *, user_id: str, assistant_id: str
) -> dict[str, Any]:
    """Poll the vendor once and record the professional voice's training state."""
    record = await _voice_record(repository, user_id, assistant_id)
    voice_id = record.get("professional_voice_id")
    if not voice_id or record.get("professional_state") != VOICE_STATE_TRAINING:
        return record
    model_id = (record.get("detail") or {}).get("training_model")
    state = await elevenlabs_client.get_voice_fine_tuning_state(
        context, voice_id=voice_id, model_id=model_id
    )
    lowered = state.lower()
    if lowered == "fine_tuned":
        record["professional_state"] = VOICE_STATE_FINE_TUNED
    elif lowered in ("failed", "fine_tuning_failed"):
        record["professional_state"] = VOICE_STATE_FAILED
    record["detail"] = {**record.get("detail", {}), "vendor_state": state}
    await repository.upsert_voice(record)
    return record


async def add_voice_clip(
    repository: Any,
    context: Any,
    *,
    user_id: str,
    assistant_id: str,
    audio_data_uri: str,
    duration_seconds: float,
    source: str,
    source_document_name: str | None = None,
    is_personal_avatar: bool = False,
    avatar_name: str = "",
) -> dict[str, Any]:
    """Store one target-only clip and advance the clone state machine.

    Every avatar's corpus feeds its instant clone. Only the personal avatar's
    corpus keeps growing past the instant target toward the professional
    minimum; other avatars stop collecting once their instant clone is built
    from the target seconds, because nothing further would be used.

    Returns the updated voice record.
    """
    thresholds = VoiceThresholds.from_context(context)
    payload, mime_type = _decode_data_uri(audio_data_uri)
    duration = max(0.0, float(duration_seconds or 0.0))
    if not payload or duration <= 0:
        return await _voice_record(repository, user_id, assistant_id)

    collected_before = float(await repository.total_voice_seconds(assistant_id))
    ceiling = (
        thresholds.professional_maximum
        if is_personal_avatar
        else thresholds.instant_target
    )
    if collected_before >= ceiling:
        logger.info(
            "Voice corpus for %s already holds %.0fs (ceiling %.0fs); clip not stored",
            assistant_id,
            collected_before,
            ceiling,
        )
        return await ensure_instant_voice(
            repository,
            context,
            user_id=user_id,
            assistant_id=assistant_id,
            avatar_name=avatar_name,
        )

    await repository.add_voice_clip(
        {
            "user_id": user_id,
            "assistant_id": assistant_id,
            "source": source,
            "source_document_name": source_document_name,
            "mime_type": mime_type,
            "bytes": payload,
            "duration_seconds": duration,
        }
    )
    record = await ensure_instant_voice(
        repository,
        context,
        user_id=user_id,
        assistant_id=assistant_id,
        avatar_name=avatar_name,
    )
    if is_personal_avatar:
        record = await prepare_professional_voice(
            repository,
            context,
            user_id=user_id,
            assistant_id=assistant_id,
            avatar_name=avatar_name,
        )
    return record


async def resolve_active_voice_id(
    repository: Any, assistant_id: str
) -> tuple[str, str | None]:
    """Which cloned voice speaks for the avatar: ``("professional"|"instant"|"none", id)``."""
    record = await repository.get_voice(assistant_id) or {}
    if record.get("professional_state") == VOICE_STATE_FINE_TUNED and record.get(
        "professional_voice_id"
    ):
        return "professional", record["professional_voice_id"]
    if record.get("instant_voice_id"):
        return "instant", record["instant_voice_id"]
    return "none", None


async def voice_status_for(
    repository: Any,
    context: Any,
    *,
    user_id: str,
    assistant_id: str,
    is_personal_avatar: bool,
) -> VoiceStatus:
    """Assemble the status the settings Voice panel renders."""
    thresholds = VoiceThresholds.from_context(context)
    record = await _voice_record(repository, user_id, assistant_id)
    collected = float(await repository.total_voice_seconds(assistant_id))
    clips = await repository.list_voice_clips(assistant_id)
    active, active_id = await resolve_active_voice_id(repository, assistant_id)
    return VoiceStatus(
        assistant_id=assistant_id,
        collected_seconds=collected,
        instant_voice_id=record.get("instant_voice_id"),
        instant_voice_seconds=float(record.get("instant_voice_seconds") or 0.0),
        professional_voice_id=record.get("professional_voice_id"),
        professional_state=str(
            record.get("professional_state") or VOICE_STATE_NOT_STARTED
        ),
        active_voice=active,
        active_voice_id=active_id,
        instant_minimum_seconds=thresholds.instant_minimum,
        instant_target_seconds=thresholds.instant_target,
        professional_minimum_seconds=thresholds.professional_minimum,
        professional_maximum_seconds=thresholds.professional_maximum,
        professional_eligible=bool(is_personal_avatar),
        clip_count=len(clips),
        verification_requested_at=record.get("verification_requested_at"),
        training_started_at=record.get("training_started_at"),
        detail={
            k: v
            for k, v in (record.get("detail") or {}).items()
            if k
            in ("instant_error", "professional_error", "vendor_state", "training_model")
        },
    )
