"""Grow a personal avatar's voice corpus from the speech it is already hearing.

Voice mode transcribes the owner's speech on every hands-free utterance and
then throws the audio away. That audio is the same speech the voice clone needs,
so this module hands it to ``add_voice_clip`` instead — the corpus grows while
the person simply talks to their avatar, the instant clone is created the moment
sixty seconds have accrued, and collection continues toward the professional
clone.

**Accrual is gated on the avatar already holding a reference audio clip.** The
reference is the anchor the diarizer matches voices against, and therefore the
only thing that can tell the owner apart from a third party or from background
chatter in the room. Without one, nothing here stores anything: voice mode still
transcribes and the avatar still answers, but the voice model does not grow. The
reference is supplied deliberately — while creating the avatar, through the
settings recorder, with a media upload carrying the owner's speech, or from
verified research — and never inferred from a live utterance. A first utterance
must not seed it: a quiet owner in a room with a television would otherwise make
a stranger's voice the avatar's voice, and ``ensure_instant_voice`` builds the
instant clone once and never rebuilds it.

Three kinds of voice reach a microphone, and any of them may be absent — often
there is no avatar voice at all. Only the first accrues:

* the person, matched against the reference clip;
* the avatar's own playback heard back through a speaker, which
  ``mark_avatar_echo`` relabels;
* third parties, which keep their ``Speaker N`` labels.

``claim_lone_speaker_as_owner`` deliberately relabels a lone voice as the owner
so a monologue is answered rather than triaged. That is right for answering and
wrong for accrual, so this module reads ``SpokenTurn.owner_matched_reference``,
which records whether the diarizer itself matched the reference, rather than
trusting the claimed labels.

Neither path accrues anything the diarizer has not attributed to the owner.
A speaker-labelled turn arrives already diarized against the reference, so that
attribution is free; a dictated utterance was transcribed by whisper, which
reports words and not who spoke them, so it is diarized here against the same
reference before any of it is kept. Being the signed-in owner is not evidence
that the voice on the microphone is the owner's.

Nothing here is on the reply path: every entry point is called through
``schedule_background`` after the turn has streamed, and every failure is logged
and swallowed.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


CAPTURE_SOURCE_SPOKEN_TURN = "voice_mode"
"""``avatar_voice_clips.source`` for speech cut from a speaker-labelled turn."""

CAPTURE_SOURCE_DICTATION = "voice_mode_dictation"
"""``avatar_voice_clips.source`` for a dictated utterance."""

CONSENT_DETAIL_KEY = "implicit_capture_consent"
"""``avatar_voice.detail`` key holding ``"granted"`` or ``"declined"``."""

CONSENT_AT_DETAIL_KEY = "implicit_capture_consent_at"

CONSENT_GRANTED = "granted"
CONSENT_DECLINED = "declined"

BLOCKED_NOT_PERSONAL = "not_personal"
BLOCKED_DISABLED = "capture_disabled"
BLOCKED_UNCONFIGURED = "voice_not_configured"
BLOCKED_CONSENT_MISSING = "consent_missing"
BLOCKED_CONSENT_DECLINED = "consent_declined"
BLOCKED_REFERENCE_MISSING = "reference_audio_missing"
BLOCKED_CEILING_REACHED = "ceiling_reached"


def capture_enabled(context: Any) -> bool:
    """Whether this deployment lets voice mode accrue speech at all."""
    return str(
        getattr(context, "voice_mode_capture_enabled", "") or ""
    ).strip().lower() in ("1", "true", "yes", "on")


def consent_state(record: dict[str, Any] | None) -> str | None:
    """The owner's answer to the capture question, or ``None`` when never asked."""
    detail = (record or {}).get("detail") or {}
    answer = str(detail.get(CONSENT_DETAIL_KEY) or "").strip().lower()
    return answer if answer in (CONSENT_GRANTED, CONSENT_DECLINED) else None


async def set_consent(
    repository: Any,
    *,
    user_id: str,
    assistant_id: str,
    granted: bool,
) -> dict[str, Any]:
    """Record the owner's answer on the avatar's voice row.

    Kept on ``avatar_voice.detail`` rather than in the LangGraph store because
    the store embeds every value it is given; a consent flag has nothing worth
    embedding and would cost a vector row per answer.
    """
    from src.anubis.utils.voice.corpus import _voice_record

    record = await _voice_record(repository, user_id, assistant_id)
    record["detail"] = {
        **(record.get("detail") or {}),
        CONSENT_DETAIL_KEY: CONSENT_GRANTED if granted else CONSENT_DECLINED,
        CONSENT_AT_DETAIL_KEY: datetime.now(tz=UTC).isoformat(),
    }
    await repository.upsert_voice(record)
    return record


async def accrual_blocked_reason(
    repository: Any,
    context: Any,
    store: Any,
    *,
    user_id: str,
    assistant_id: str,
    is_personal_avatar: bool,
) -> str | None:
    """Why this avatar cannot accrue speech from voice mode, or ``None`` when it can."""
    _reference, reason = await accrual_gate(
        repository,
        context,
        store,
        user_id=user_id,
        assistant_id=assistant_id,
        is_personal_avatar=is_personal_avatar,
    )
    return reason


async def accrual_gate(
    repository: Any,
    context: Any,
    store: Any,
    *,
    user_id: str,
    assistant_id: str,
    is_personal_avatar: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    """The reference clip accrual may use, or the reason there is none.

    Returns ``(clip, None)`` when this avatar may accrue and ``(None, reason)``
    otherwise. The clip comes back with the verdict because both accrual paths
    need the audio itself — it is what the diarizer is given to recognise the
    owner with — and reading it twice is a wasted store round trip.

    The order of the checks is the order the owner can act on: what they cannot
    change at all first, then the consent answer and the reference clip.
    """
    from src.anubis.utils.voice.corpus import VoiceThresholds, voice_configured

    if not is_personal_avatar:
        return None, BLOCKED_NOT_PERSONAL
    if not capture_enabled(context):
        return None, BLOCKED_DISABLED
    if repository is None or not voice_configured(context):
        return None, BLOCKED_UNCONFIGURED

    from src.anubis.utils.voice.corpus import _voice_record

    record = await _voice_record(repository, user_id, assistant_id)
    answer = consent_state(record)
    if answer is None:
        return None, BLOCKED_CONSENT_MISSING
    if answer == CONSENT_DECLINED:
        return None, BLOCKED_CONSENT_DECLINED

    # The gate. The reference clip is what makes a voice in the room
    # attributable; without it there is no safe way to know whose speech this is.
    from src.anubis.utils.voice.reference_audio import read_usable_reference_audio

    reference, _problem = await read_usable_reference_audio(
        store, user_id, assistant_id, context=context
    )
    if reference is None:
        return None, BLOCKED_REFERENCE_MISSING

    thresholds = VoiceThresholds.from_context(context)
    collected = float(await repository.total_voice_seconds(assistant_id))
    if collected >= thresholds.professional_maximum:
        return None, BLOCKED_CEILING_REACHED
    return reference, None


def _windows_from_spans(
    spans: list[tuple[float, float]],
    *,
    minimum_seconds: float,
    maximum_seconds: float,
) -> list[dict[str, Any]]:
    """Shape the owner's stretches for ``cut_target_turns_to_mp3_data_uri``.

    Windows shorter than ``minimum_seconds`` are dropped — they are mostly onset
    and release, and they make a clone worse rather than better — and the turn
    stops contributing at ``maximum_seconds`` so one long monologue cannot come
    to dominate the voice.
    """
    windows: list[dict[str, Any]] = []
    kept = 0.0
    for start, end in spans:
        start = float(start or 0.0)
        end = float(end or 0.0)
        length = end - start
        if length < minimum_seconds:
            continue
        if kept + length > maximum_seconds:
            end = start + max(0.0, maximum_seconds - kept)
            length = end - start
            if length < minimum_seconds:
                break
        windows.append({"is_target": True, "start": start, "end": end})
        kept += length
        if kept >= maximum_seconds:
            break
    return windows


def _owner_spans_of_turn(turn: Any) -> list[tuple[float, float]]:
    """The owner's stretches of a speaker-labelled turn.

    Only segments the diarizer attributed to the owner are taken. Avatar echo
    and every ``Speaker N`` segment are left behind, so a conversation held in a
    room with other people contributes the owner's half and nothing else.
    """
    return [
        (
            float(getattr(segment, "start", 0.0) or 0.0),
            float(getattr(segment, "end", 0.0) or 0.0),
        )
        for segment in turn.segments
        if getattr(segment, "is_owner", False)
        and not getattr(segment, "is_avatar", False)
    ]


def _owner_spans_of_diarized_segments(
    segments: list[Any], target_label: str
) -> list[tuple[float, float]]:
    """The owner's stretches of a freshly diarized recording.

    The diarizer returns the reference's given name for a voice that matched it
    and an anonymous letter for every other voice, so the owner is the segments
    carrying that name. A recording where the owner never spoke yields nothing,
    which is the point: whoever else was talking is not this avatar's voice.
    """
    wanted = (target_label or "").strip().lower()
    if not wanted:
        return []
    spans: list[tuple[float, float]] = []
    for segment in segments or []:
        if not isinstance(segment, dict):
            continue
        if wanted not in str(segment.get("speaker") or "").lower():
            continue
        if not str(segment.get("text") or "").strip():
            continue
        start = segment.get("start")
        end = segment.get("end")
        if start is None or end is None:
            continue
        spans.append((float(start), float(end)))
    return spans


async def accrue_voice_from_spoken_turn(
    turn: Any,
    repository: Any,
    context: Any,
    store: Any,
    *,
    user_id: str,
    assistant_id: str,
    is_personal_avatar: bool,
    avatar_name: str = "",
    thread_id: str | None = None,
) -> dict[str, Any] | None:
    """Add the owner's speech in one live utterance to the avatar's voice corpus.

    Costs no vendor call: the utterance was already diarized to produce the
    speaker script, so the owner's windows are cut straight out of the audio
    that diarization preprocessed.

    Returns the updated voice record when something was stored, else ``None``.
    """
    try:
        blocked = await accrual_blocked_reason(
            repository,
            context,
            store,
            user_id=user_id,
            assistant_id=assistant_id,
            is_personal_avatar=is_personal_avatar,
        )
        if blocked is not None:
            logger.debug("Voice accrual skipped for %s: %s", assistant_id, blocked)
            return None

        # Only a real match counts. ``claim_lone_speaker_as_owner`` relabels a
        # lone voice as the owner so a monologue is answered instead of triaged,
        # but that lone voice may be a television.
        if not getattr(turn, "owner_matched_reference", False):
            logger.debug(
                "Voice accrual skipped for %s: the diarizer did not match the owner",
                assistant_id,
            )
            return None

        audio_data_uri = str(getattr(turn, "audio_data_uri", "") or "")
        if not audio_data_uri:
            return None

        windows = _windows_from_spans(
            _owner_spans_of_turn(turn),
            minimum_seconds=float(
                getattr(context, "voice_mode_capture_min_segment_seconds", None) or 1.0
            ),
            maximum_seconds=float(
                getattr(context, "voice_mode_capture_max_seconds_per_turn", None) or 30.0
            ),
        )
        if not windows:
            return None

        from src.anubis.utils.voice.clips import cut_target_turns_to_mp3_data_uri

        clip_uri, seconds = await cut_target_turns_to_mp3_data_uri(audio_data_uri, windows)
        if not clip_uri or seconds <= 0:
            return None

        return await _store_clip(
            repository,
            context,
            user_id=user_id,
            assistant_id=assistant_id,
            clip_uri=clip_uri,
            seconds=seconds,
            source=CAPTURE_SOURCE_SPOKEN_TURN,
            thread_id=thread_id,
            is_personal_avatar=is_personal_avatar,
            avatar_name=avatar_name,
        )
    except Exception:  # noqa: BLE001 - accrual must never disturb the conversation
        logger.exception("Could not accrue voice from a spoken turn for %s", assistant_id)
        return None


async def accrue_voice_from_utterance(
    audio_data_uri: str,
    duration_seconds: float,
    repository: Any,
    context: Any,
    store: Any,
    *,
    user_id: str,
    assistant_id: str,
    is_personal_avatar: bool,
    avatar_name: str = "",
    thread_id: str | None = None,
    filename: str | None = None,
    content_type: str | None = None,
    on_diarized: Any = None,
) -> dict[str, Any] | None:
    """Add the owner's speech in one dictated utterance to the voice corpus.

    ``POST /transcribe`` transcribes with whisper, which reports words and not
    who said them, so this path has no attribution of its own. It does not
    accrue on the strength of whose account is signed in: a microphone picks up
    whoever is near it, and a clone is permanent. The recording is diarized here
    against the avatar's stored reference clip, and only the stretches the
    diarizer matches to that reference are kept — the same treatment the
    speaker-labelled path gets for free.

    That costs one diarization call per accrued utterance, which is the price of
    knowing whose voice is being learned. It is paid off the reply path, after
    the transcription has already been returned, and only while the avatar is
    still collecting: consent, the reference clip and the corpus ceiling all gate
    it first, and ``VOICE_MODE_CAPTURE_ENABLED`` switches it off outright.

    ``on_diarized`` is called with the diarization result so the caller can meter
    it; this module has no access to the metrics pool and must not import the
    web application to get one.
    """
    try:
        reference, blocked = await accrual_gate(
            repository,
            context,
            store,
            user_id=user_id,
            assistant_id=assistant_id,
            is_personal_avatar=is_personal_avatar,
        )
        if blocked is not None or reference is None:
            logger.debug("Voice accrual skipped for %s: %s", assistant_id, blocked)
            return None

        minimum = float(
            getattr(context, "voice_mode_capture_min_segment_seconds", None) or 1.0
        )
        maximum = float(
            getattr(context, "voice_mode_capture_max_seconds_per_turn", None) or 30.0
        )
        # Nothing long enough to be worth a diarization call, or nothing at all.
        if not audio_data_uri or float(duration_seconds or 0.0) < minimum:
            return None

        from src.anubis.utils.utility import transcribe_audio_diarize

        diarized = await transcribe_audio_diarize(
            audio_data_uri,
            context,
            encoded_reference_audio=str(reference.get("audio_data_uri") or ""),
            filename=filename or "utterance.mp3",
            content_type=content_type or "audio/mpeg",
        )
        if on_diarized is not None:
            try:
                on_diarized(diarized)
            except Exception:  # noqa: BLE001 - metering must not lose the clip
                logger.debug("Could not meter the accrual diarization", exc_info=True)

        spans = _owner_spans_of_diarized_segments(
            diarized.get("segments") or [],
            str(getattr(context, "audio_diarization_known_speaker_name", "") or ""),
        )
        if not spans:
            # Somebody talked, and it was not this avatar's person.
            logger.debug(
                "Voice accrual for %s found no owner speech in the utterance",
                assistant_id,
            )
            return None

        windows = _windows_from_spans(
            spans, minimum_seconds=minimum, maximum_seconds=maximum
        )
        if not windows:
            return None

        from src.anubis.utils.voice.clips import cut_target_turns_to_mp3_data_uri

        clip_uri, seconds = await cut_target_turns_to_mp3_data_uri(
            audio_data_uri, windows
        )
        if not clip_uri or seconds <= 0:
            return None

        return await _store_clip(
            repository,
            context,
            user_id=user_id,
            assistant_id=assistant_id,
            clip_uri=clip_uri,
            seconds=seconds,
            source=CAPTURE_SOURCE_DICTATION,
            thread_id=thread_id,
            is_personal_avatar=is_personal_avatar,
            avatar_name=avatar_name,
        )
    except Exception:  # noqa: BLE001 - accrual must never disturb the conversation
        logger.exception("Could not accrue voice from an utterance for %s", assistant_id)
        return None


async def _store_clip(
    repository: Any,
    context: Any,
    *,
    user_id: str,
    assistant_id: str,
    clip_uri: str,
    seconds: float,
    source: str,
    thread_id: str | None,
    is_personal_avatar: bool,
    avatar_name: str,
) -> dict[str, Any]:
    """Hand one clip to the clone state machine and report what changed.

    The document name carries the thread and the moment, so a conversation the
    owner would rather the voice had not learned from can be dropped with
    ``forget_document_clips`` and the clone rebuilt from what is left.
    """
    from src.anubis.utils.voice.corpus import add_voice_clip

    document_name = f"{source} {thread_id or 'thread'} {datetime.now(tz=UTC).isoformat()}"
    record = await add_voice_clip(
        repository,
        context,
        user_id=user_id,
        assistant_id=assistant_id,
        audio_data_uri=clip_uri,
        duration_seconds=seconds,
        source=source,
        source_document_name=document_name,
        is_personal_avatar=is_personal_avatar,
        avatar_name=avatar_name,
    )
    logger.info(
        "Voice accrual for %s added %.1fs from %s (corpus now %.0fs)",
        assistant_id,
        seconds,
        source,
        float(record.get("collected_seconds") or 0.0),
    )

    from src.anubis.utils.voice.corpus import refresh_instant_voice_at_target

    return await refresh_instant_voice_at_target(
        repository,
        context,
        user_id=user_id,
        assistant_id=assistant_id,
        avatar_name=avatar_name,
        record=record,
    )
