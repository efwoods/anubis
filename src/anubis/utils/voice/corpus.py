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
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.anubis.utils.media_assets.repository import (
    VOICE_STATE_AWAITING_VERIFICATION,
    VOICE_STATE_COLLECTING,
    VOICE_STATE_FAILED,
    VOICE_STATE_FINE_TUNED,
    VOICE_STATE_NOT_STARTED,
    VOICE_STATE_PLAN_REQUIRED,
    VOICE_STATE_TRAINING,
)
from src.anubis.utils.voice import elevenlabs_client

# A failed instant clone (a vendor 400, a plan limit) is retried when the Voice
# panel next reads the status, but no more often than this, so a persistent
# refusal does not hit the vendor on every settings refresh.
INSTANT_CLONE_RETRY_SECONDS = 300.0

# How long a "this voice is still in good standing" answer from ElevenLabs is
# trusted before the settings Voice panel asks again. ElevenLabs applies its
# moderation ban asynchronously — a clone is created, reports healthy, and is
# banned minutes or hours later — so a one-time check at creation is not enough
# to keep the panel honest; it is re-read on a status read this often.
VOICE_SAFETY_RECHECK_SECONDS = 900.0

logger = logging.getLogger(__name__)

CLIP_SOURCE_RECORDER = "recorder"
CLIP_SOURCE_REFERENCE_UPLOAD = "reference_upload"
CLIP_SOURCE_MEDIA_UPLOAD = "media_upload"

# Professional voice cloning is only offered to ElevenLabs accounts on the
# Creator plan or above; the vendor says so in the refusal text.
PROFESSIONAL_VOICE_PLAN_HELP_URL = (
    "https://elevenlabs.io/docs/eleven-api/guides/how-to/voices/"
    "professional-voice-cloning"
)
_PLAN_REFUSAL_MARKERS = ("creator plan", "requires you to be on", "upgrade your plan")


def is_plan_refusal(error_text: str) -> bool:
    """Whether a vendor error says the ElevenLabs plan is too low for the request."""
    lowered = (error_text or "").lower()
    return any(marker in lowered for marker in _PLAN_REFUSAL_MARKERS)


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
    instant_voice_blocked: bool
    instant_voice_blocked_reason: str | None
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
    clips: list[dict[str, Any]] = field(default_factory=list)
    reference_audio_document: str | None = None
    # Whether the stored reference clip can actually anchor the diarizer, and
    # the sentence naming what is wrong when the clip cannot. The Voice panel
    # asks the owner for a better recording on the strength of these two.
    reference_audio_usable: bool = False
    reference_audio_problem: str | None = None
    # Whether talking to this avatar in voice mode grows its voice, and the one
    # thing standing in the way when it does not. ``accrual_consent`` is the
    # owner's answer, or ``None`` when they have not been asked yet.
    accrual_enabled: bool = False
    accrual_blocked_reason: str | None = None
    accrual_consent: str | None = None
    # The vendor's stock voice the owner chose for the avatar to speak with
    # while it has no usable clone: ``{"voice_id", "name", "gender"}`` or
    # ``None``. Reported whether or not a clone exists, so the panel can show
    # the pick without a second read.
    standard_voice: dict[str, Any] | None = None
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


BLOCKED_VOICE_MESSAGE = (
    "ElevenLabs has blocked this avatar's cloned voice for violating its terms "
    "of service, which it applies to clones of public figures. The block is "
    "permanent and belongs to the voice: re-cloning the same recording produces "
    "the same block. Delete the voice model and record or upload speech the "
    "avatar's owner is entitled to clone."
)


def voice_record_blocked(record: dict[str, Any]) -> bool:
    """Whether the stored record already knows the instant clone is banned."""
    return bool((record.get("detail") or {}).get("instant_blocked"))


def voice_record_blocked_reason(record: dict[str, Any]) -> str | None:
    """Return the message to show for a banned instant clone, else None."""
    detail = record.get("detail") or {}
    if not detail.get("instant_blocked"):
        return None
    return str(detail.get("instant_blocked_reason") or BLOCKED_VOICE_MESSAGE)


def _mark_record_blocked(record: dict[str, Any], reason: str | None = None) -> None:
    """Record the vendor's permanent ban of this avatar's instant clone."""
    record["detail"] = {
        **(record.get("detail") or {}),
        "instant_blocked": True,
        "instant_blocked_reason": reason or BLOCKED_VOICE_MESSAGE,
        "instant_blocked_at": datetime.now(tz=UTC).timestamp(),
        "instant_safety_checked_at": datetime.now(tz=UTC).timestamp(),
    }


async def mark_voice_blocked(
    repository: Any, user_id: str, assistant_id: str, *, reason: str | None = None
) -> dict[str, Any]:
    """Persist that ElevenLabs has banned this avatar's cloned voice.

    Called both by the pre-emptive safety check and by ``POST /speak`` when the
    ban is met at synthesis time, so the settings Voice panel stops advertising
    a voice model that cannot speak.
    """
    record = await _voice_record(repository, user_id, assistant_id)
    _mark_record_blocked(record, reason)
    await repository.upsert_voice(record)
    logger.warning(
        "Voice %s for %s is blocked by ElevenLabs",
        record.get("instant_voice_id"),
        assistant_id,
    )
    return record


async def _refresh_voice_safety(
    repository: Any,
    context: Any,
    record: dict[str, Any],
    *,
    assistant_id: str,
) -> dict[str, Any]:
    """Ask ElevenLabs whether the instant clone is still allowed to speak.

    The point of asking here rather than at synthesis time is that the settings
    Voice panel is where someone finds out: without this, a banned clone still
    reads "Voice model trained and available" and the ban only ever surfaces as
    a speak button that does nothing. Skipped when the answer is already known
    or was asked for recently — see VOICE_SAFETY_RECHECK_SECONDS.
    """
    voice_id = record.get("instant_voice_id")
    if not voice_id or not voice_configured(context):
        return record
    if voice_record_blocked(record):
        return record
    detail = record.get("detail") or {}
    checked_at = float(detail.get("instant_safety_checked_at") or 0.0)
    now = datetime.now(tz=UTC).timestamp()
    if now - checked_at < VOICE_SAFETY_RECHECK_SECONDS:
        return record
    blocked = await elevenlabs_client.voice_is_blocked(context, voice_id=str(voice_id))
    if blocked:
        _mark_record_blocked(record)
        logger.warning(
            "Voice %s for %s is blocked by ElevenLabs", voice_id, assistant_id
        )
    else:
        record["detail"] = {**detail, "instant_safety_checked_at": now}
    await repository.upsert_voice(record)
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
    """Create the instant clone once the corpus reaches the minimum.

    Idempotent: nothing happens below the minimum, and a clone that exists is
    never rebuilt — the first instant voice is final. Later clips only grow
    the corpus (toward the personal avatar's professional clone).
    """
    thresholds = VoiceThresholds.from_context(context)
    record = await _voice_record(repository, user_id, assistant_id)
    collected = float(await repository.total_voice_seconds(assistant_id))
    record["collected_seconds"] = collected

    if collected < thresholds.instant_minimum or not voice_configured(context):
        await repository.upsert_voice(record)
        return record

    if record.get("instant_voice_id"):
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
    except elevenlabs_client.ElevenLabsVoiceBlockedError as blocked_error:
        # A ban is permanent and belongs to the recording, so this must not be
        # written as ``instant_error``: that key marks a transient failure the
        # status read retries every few minutes, and retrying a banned clone
        # only repeats the refusal.
        logger.warning(
            "Instant clone for %s was blocked by ElevenLabs: %s",
            assistant_id,
            blocked_error,
        )
        _mark_record_blocked(record, str(blocked_error))
        await repository.upsert_voice(record)
        return record
    except elevenlabs_client.ElevenLabsError as clone_error:
        logger.warning("Instant clone failed for %s: %s", assistant_id, clone_error)
        record["detail"] = {
            **record.get("detail", {}),
            "instant_error": str(clone_error),
            "instant_error_at": datetime.now(tz=UTC).timestamp(),
        }
        await repository.upsert_voice(record)
        return record

    record["instant_voice_id"] = voice_id
    record["instant_voice_seconds"] = used_seconds
    record["detail"] = {
        k: v
        for k, v in record.get("detail", {}).items()
        if k not in ("instant_error", "instant_error_at")
    }
    await repository.upsert_voice(record)
    logger.info(
        "Instant voice %s created for %s from %.0fs",
        voice_id,
        assistant_id,
        used_seconds,
    )
    # Ask the vendor immediately whether the clone it just minted is allowed to
    # speak. ElevenLabs accepts the clone and bans it separately, so the answer
    # here is "not yet banned" rather than "never will be" — the recheck on
    # every status read (see _refresh_voice_safety) is what catches the rest.
    return await _refresh_voice_safety(
        repository, context, record, assistant_id=assistant_id
    )


async def rebuild_instant_voice(
    repository: Any,
    context: Any,
    *,
    user_id: str,
    assistant_id: str,
    avatar_name: str = "",
) -> dict[str, Any]:
    """Delete the avatar's instant clone and train a new one from the corpus as it stands.

    ``ensure_instant_voice`` builds the first clone and then never rebuilds, so
    a clone trained from the wrong speech — a recording of somebody else, a clip
    cut from an arbitrary video, speech the vendor went on to ban — is otherwise
    the avatar's voice forever. This is the way out: the vendor's copy is
    deleted, the stored clone is cleared along with the errors and the ban mark
    that belonged to it, and a fresh clone is trained from the clips the avatar
    holds now. Deleting an upload first (which removes that upload's clips) is
    therefore how the owner chooses what the new voice is trained from.

    The vendor copy is deleted before the new clone is requested, because a
    vendor plan allows only so many voices and the old one is being replaced.
    Should the new clone then fail, the avatar is left with no instant voice and
    the ordinary retry on the next status read builds one.

    The professional clone, its verification, and the collected clips are all
    left exactly as they are.

    Returns:
        The stored voice record after the rebuild attempt.
    """
    record = await _voice_record(repository, user_id, assistant_id)
    previous_voice_id = record.get("instant_voice_id")
    if previous_voice_id and voice_configured(context):
        try:
            await elevenlabs_client.delete_voice(context, previous_voice_id)
            logger.info(
                "Deleted instant voice %s for %s before rebuilding",
                previous_voice_id,
                assistant_id,
            )
        except elevenlabs_client.ElevenLabsError as delete_error:
            # A voice already gone at the vendor, or a vendor outage, must not
            # strand the avatar with a stored id that no longer speaks. The
            # stored clone is cleared either way and a new one is trained.
            logger.warning(
                "Could not delete instant voice %s for %s: %s",
                previous_voice_id,
                assistant_id,
                delete_error,
            )

    record["instant_voice_id"] = None
    record["instant_voice_seconds"] = 0.0
    record["detail"] = {
        key: value
        for key, value in (record.get("detail") or {}).items()
        # Every one of these described the clone being deleted: a transient
        # failure to build it, and the vendor's ban on the voice it produced.
        # Carrying them onto the next clone would either suppress the rebuild
        # or report the new voice as banned before the vendor has judged it.
        if key
        not in (
            "instant_error",
            "instant_error_at",
            "instant_blocked",
            "instant_blocked_reason",
            "instant_blocked_at",
            "instant_safety_checked_at",
        )
    }
    if previous_voice_id:
        record["detail"]["instant_replaced_voice_id"] = previous_voice_id
    record["detail"]["instant_rebuilt_at"] = datetime.now(tz=UTC).timestamp()
    await repository.upsert_voice(record)

    # ``ensure_instant_voice`` re-reads the record, so the cleared row above is
    # what it sees: no clone, and the corpus as the owner has left it.
    return await ensure_instant_voice(
        repository,
        context,
        user_id=user_id,
        assistant_id=assistant_id,
        avatar_name=avatar_name,
    )


INSTANT_REFRESHED_DETAIL_KEY = "instant_refreshed_at_target"
"""``detail`` marker so the one refresh at the target happens exactly once."""


def instant_refresh_enabled(context: Any) -> bool:
    """Whether a clone built from less than the target is rebuilt at the target."""
    return str(
        getattr(context, "instant_voice_refresh_at_target_enabled", "") or ""
    ).strip().lower() in ("1", "true", "yes", "on")


async def refresh_instant_voice_at_target(
    repository: Any,
    context: Any,
    *,
    user_id: str,
    assistant_id: str,
    avatar_name: str = "",
    record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild an instant clone once, the first time the corpus reaches the target.

    ``ensure_instant_voice`` builds the first clone at the minimum and never
    rebuilds, so an avatar whose corpus crossed sixty seconds mid-conversation
    keeps a clone trained on sixty seconds however much it goes on to hear. That
    is the right default for an upload — the owner chose what to upload — but
    not for speech accrued as a side effect of talking, where waiting a few more
    minutes produces a markedly better voice for free.

    This replaces such a clone exactly once, when the corpus first reaches
    ``ELEVENLABS_INSTANT_VOICE_CLONE_TARGET_SECONDS``, which is the behaviour
    that field's own description has always claimed. A clone already built from
    the target seconds or more is left alone, as is one the vendor has blocked —
    rebuilding a banned voice only earns the same ban.

    Returns the voice record, unchanged when no refresh was due.
    """
    record = record or await _voice_record(repository, user_id, assistant_id)
    if not instant_refresh_enabled(context) or not voice_configured(context):
        return record
    detail = record.get("detail") or {}
    if detail.get(INSTANT_REFRESHED_DETAIL_KEY) or voice_record_blocked(record):
        return record
    if not record.get("instant_voice_id"):
        return record

    thresholds = VoiceThresholds.from_context(context)
    if float(record.get("instant_voice_seconds") or 0.0) >= thresholds.instant_target:
        return record
    collected = float(await repository.total_voice_seconds(assistant_id))
    if collected < thresholds.instant_target:
        return record

    # Marked before the rebuild, not after: a rebuild that fails leaves the
    # avatar with no instant voice and the ordinary retry on the next status
    # read builds one, and marking afterwards would let a failing vendor be
    # asked to rebuild on every later clip.
    record["detail"] = {
        **detail,
        INSTANT_REFRESHED_DETAIL_KEY: datetime.now(tz=UTC).timestamp(),
    }
    await repository.upsert_voice(record)
    logger.info(
        "Refreshing the instant voice for %s: clone built from %.0fs, corpus now %.0fs",
        assistant_id,
        float(record.get("instant_voice_seconds") or 0.0),
        collected,
    )
    return await rebuild_instant_voice(
        repository,
        context,
        user_id=user_id,
        assistant_id=assistant_id,
        avatar_name=avatar_name,
    )


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
        plan_refused = is_plan_refusal(str(clone_error))
        # A plan refusal is not transient: every later clip would hit the same
        # wall, so the record parks in ``plan_required`` until the owner asks
        # for a retry (``retry_professional_voice``) after upgrading the
        # ElevenLabs account. The instant voice keeps working meanwhile.
        record["professional_state"] = (
            VOICE_STATE_PLAN_REQUIRED if plan_refused else VOICE_STATE_FAILED
        )
        record["detail"] = {
            **record.get("detail", {}),
            "professional_error": str(clone_error),
            "professional_error_kind": "plan_required" if plan_refused else "vendor",
            "professional_help_url": PROFESSIONAL_VOICE_PLAN_HELP_URL,
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
            if k
            not in (
                "professional_error",
                "professional_error_kind",
                "professional_help_url",
            )
        },
        "professional_sample_seconds": used_seconds,
    }
    await repository.upsert_voice(record)
    return record


async def retry_professional_voice(
    repository: Any,
    context: Any,
    *,
    user_id: str,
    assistant_id: str,
    avatar_name: str = "",
) -> dict[str, Any]:
    """Retry professional clone preparation after a plan refusal or vendor failure.

    Resets a ``plan_required`` or ``failed`` record to ``collecting`` and runs
    ``prepare_professional_voice`` once. Any other state is returned untouched.
    """
    record = await _voice_record(repository, user_id, assistant_id)
    if record.get("professional_state") not in (
        VOICE_STATE_PLAN_REQUIRED,
        VOICE_STATE_FAILED,
    ):
        return record
    record["professional_state"] = VOICE_STATE_COLLECTING
    await repository.upsert_voice(record)
    return await prepare_professional_voice(
        repository,
        context,
        user_id=user_id,
        assistant_id=assistant_id,
        avatar_name=avatar_name,
    )


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
    # Every avatar keeps every clip: the clips are the pool the owner can pick
    # a reference from and the seconds the settings panel reports. The
    # professional maximum bounds storage for every avatar; the instant clone
    # is built once and never rebuilt (see ensure_instant_voice).
    ceiling = thresholds.professional_maximum
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


async def resolve_speaking_voice(
    repository: Any, assistant_id: str
) -> tuple[str, str | None]:
    """Which voice actually speaks: a usable clone first, else the standard voice.

    ``("professional"|"instant"|"standard"|"none", id)``. A clone the vendor
    has banned does not count as usable, so an avatar with a banned clone and
    a standard voice speaks with the standard voice.
    """
    from src.anubis.utils.voice.standard_voices import standard_voice_of

    record = await repository.get_voice(assistant_id) or {}
    kind, voice_id = await resolve_active_voice_id(repository, assistant_id)
    if voice_id is not None and not voice_record_blocked(record):
        return kind, voice_id
    standard = standard_voice_of(record)
    if standard is not None:
        return "standard", standard["voice_id"]
    return "none", None


def voice_seconds_by_document(clips: list[dict[str, Any]]) -> dict[str, float]:
    """Seconds of stored speech per source document name."""
    seconds_by_document: dict[str, float] = {}
    for clip in clips:
        document_name = clip.get("source_document_name")
        if not document_name:
            continue
        seconds_by_document[document_name] = seconds_by_document.get(
            document_name, 0.0
        ) + float(clip.get("duration_seconds") or 0.0)
    return seconds_by_document


async def forget_document_clips(
    repository: Any,
    *,
    user_id: str,
    assistant_id: str,
    source_document_name: str,
) -> dict[str, Any]:
    """Drop a deleted document's clips and recompute the collected seconds.

    The trained voice is never touched: deleting speech lowers the count the
    panel shows, nothing more.
    """
    removed = await repository.delete_voice_clips_for_document(
        assistant_id, source_document_name
    )
    record = await _voice_record(repository, user_id, assistant_id)
    record["collected_seconds"] = float(
        await repository.total_voice_seconds(assistant_id)
    )
    await repository.upsert_voice(record)
    if removed:
        logger.info(
            "Removed %d voice clip(s) of %s for %s", removed, source_document_name, assistant_id
        )
    return record


async def longest_clip_for_document(
    repository: Any, assistant_id: str, source_document_name: str
) -> dict[str, Any] | None:
    """Return the longest stored clip cut from one document, with bytes, or ``None``."""
    clips = await repository.list_voice_clips(assistant_id, include_bytes=True)
    matching = [
        clip for clip in clips if clip.get("source_document_name") == source_document_name
    ]
    if not matching:
        return None
    return max(matching, key=lambda clip: float(clip.get("duration_seconds") or 0.0))


def clip_data_uri(clip: dict[str, Any]) -> str:
    """Return a data URI for a stored clip's bytes."""
    mime_type = clip.get("mime_type") or "audio/mpeg"
    return f"data:{mime_type};base64," + base64.b64encode(clip.get("bytes") or b"").decode()


def _instant_clone_retry_due(
    record: dict[str, Any],
    collected_seconds: float,
    thresholds: VoiceThresholds,
    context: Any,
) -> bool:
    """Whether a status read should retry a failed instant clone now."""
    if record.get("instant_voice_id") or not voice_configured(context):
        return False
    if collected_seconds < thresholds.instant_minimum:
        return False
    detail = record.get("detail") or {}
    if not detail.get("instant_error"):
        return False
    failed_at = float(detail.get("instant_error_at") or 0.0)
    return datetime.now(tz=UTC).timestamp() - failed_at >= INSTANT_CLONE_RETRY_SECONDS


async def voice_readiness(
    repository: Any,
    context: Any,
    *,
    user_id: str,
    assistant_id: str,
) -> dict[str, Any]:
    """A cheap read of whether the avatar can speak yet, and how far along it is.

    ``voice_status_for`` is what the settings panel renders and it asks the
    vendor whether the clone is still allowed to speak. That is far too much for
    something reported on every spoken turn, so this reads the stored row and
    the corpus total only. It is what tells a live conversation that the voice
    has just become usable, so the avatar starts speaking without a reload.
    """
    from src.anubis.utils.voice.standard_voices import standard_voice_of

    record = await _voice_record(repository, user_id, assistant_id)
    thresholds = VoiceThresholds.from_context(context)
    active, active_id = await resolve_speaking_voice(repository, assistant_id)
    standard_voice = standard_voice_of(record)
    readiness = {
        "active_voice": active,
        "has_voice": active_id is not None,
        # A banned clone silences the avatar only while no standard voice
        # stands in for it; the client latches ``blocked`` into text-only replies.
        "blocked": voice_record_blocked(record) and standard_voice is None,
        "standard_voice": standard_voice,
    }
    # Whether an avatar can speak is plain to anyone who presses speak, so it is
    # reported to whoever is talking. How much speech it holds is the owner's
    # business, and the row already names the owner, so telling them apart costs
    # no extra lookup. A row that does not exist yet defaults to the caller and
    # holds nothing to disclose.
    if str(record.get("user_id") or "") != str(user_id or ""):
        return readiness
    readiness.update(
        {
            "collected_seconds": float(
                await repository.total_voice_seconds(assistant_id)
            ),
            "instant_minimum_seconds": thresholds.instant_minimum,
            "professional_state": str(
                record.get("professional_state") or VOICE_STATE_NOT_STARTED
            ),
        }
    )
    return readiness


async def voice_status_for(
    repository: Any,
    context: Any,
    *,
    user_id: str,
    assistant_id: str,
    is_personal_avatar: bool,
    store: Any | None = None,
) -> VoiceStatus:
    """Assemble the status the settings Voice panel renders.

    With ``store`` given, the status also names the document the reference
    clip was cut from.
    """
    thresholds = VoiceThresholds.from_context(context)
    record = await _voice_record(repository, user_id, assistant_id)
    collected = float(await repository.total_voice_seconds(assistant_id))
    # The clone is normally built when the clip that crosses the minimum is
    # stored. When that attempt failed (the panel shows ``instant_error``) the
    # corpus already holds enough speech, so a status read retries it instead
    # of waiting for yet another upload.
    if _instant_clone_retry_due(record, collected, thresholds, context):
        record = await ensure_instant_voice(
            repository,
            context,
            user_id=user_id,
            assistant_id=assistant_id,
        )
    record = await _refresh_voice_safety(
        repository, context, record, assistant_id=assistant_id
    )
    clips = await repository.list_voice_clips(assistant_id)
    active, active_id = await resolve_active_voice_id(repository, assistant_id)
    from src.anubis.utils.voice.capture import accrual_blocked_reason, consent_state
    from src.anubis.utils.voice.standard_voices import standard_voice_of

    accrual_consent = consent_state(record)
    accrual_blocked = await accrual_blocked_reason(
        repository,
        context,
        store,
        user_id=user_id,
        assistant_id=assistant_id,
        is_personal_avatar=is_personal_avatar,
    )
    reference_audio_document: str | None = None
    reference_audio_usable = False
    reference_audio_problem: str | None = None
    if store is not None:
        from src.anubis.utils.voice.reference_audio import read_reference_audio
        from src.anubis.utils.voice.reference_eligibility import (
            stored_reference_rejection,
        )

        stored_reference = await read_reference_audio(store, user_id, assistant_id)
        if stored_reference is not None:
            reference_audio_document = stored_reference.get("filename")
        # A row can exist and still be unable to anchor the diarizer — rows
        # written before that was checked hold the isolation's passthrough
        # fallback. The owner is told which of the two situations this is.
        reference_audio_problem = stored_reference_rejection(
            stored_reference,
            maximum_seconds=getattr(context, "reference_audio_clip_max_seconds", None),
            minimum_seconds=getattr(context, "reference_audio_minimum_seconds", None),
        )
        reference_audio_usable = reference_audio_problem is None
    return VoiceStatus(
        assistant_id=assistant_id,
        collected_seconds=collected,
        instant_voice_id=record.get("instant_voice_id"),
        instant_voice_seconds=float(record.get("instant_voice_seconds") or 0.0),
        instant_voice_blocked=voice_record_blocked(record),
        instant_voice_blocked_reason=voice_record_blocked_reason(record),
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
        clips=[
            {
                "clip_id": clip.get("clip_id"),
                "source": clip.get("source"),
                "source_document_name": clip.get("source_document_name"),
                "duration_seconds": float(clip.get("duration_seconds") or 0.0),
                "created_at": clip.get("created_at"),
            }
            for clip in clips
        ],
        reference_audio_document=reference_audio_document,
        reference_audio_usable=reference_audio_usable,
        reference_audio_problem=reference_audio_problem,
        accrual_enabled=accrual_blocked is None,
        accrual_blocked_reason=accrual_blocked,
        accrual_consent=accrual_consent,
        standard_voice=standard_voice_of(record),
        detail={
            k: v
            for k, v in (record.get("detail") or {}).items()
            if k
            in (
                "instant_error",
                "instant_blocked",
                "instant_blocked_reason",
                "professional_error",
                "professional_error_kind",
                "professional_help_url",
                "vendor_state",
                "training_model",
            )
        },
    )
