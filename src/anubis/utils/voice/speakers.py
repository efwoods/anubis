"""Who is speaking: label a live-voice utterance by speaker.

A personal avatar in the real world hears three kinds of voice: the owner
(the person the avatar is), other people in the room, and its own replies.
The avatar's own replies never reach this module: the avatar knows what the
avatar said because those turns are the assistant messages of the thread, and
the microphone is paused while the avatar speaks. This module tells the other
two apart.

One diarization call (``gpt-4o-transcribe-diarize``) receives the utterance
together with reference clips of the voices the avatar already knows:

* the owner, cut from the voice recordings the owner made for the voice clone;
* up to ``voice_speaker_memory_max_speakers`` other people heard earlier in
  the same conversation thread, remembered as short clips in the
  ``avatar_voice_speakers`` table so "Speaker 2" keeps meaning the same person
  from one utterance to the next.

Voices matched to a reference come back labelled with that reference's name.
Any other voice is given the next free ``Speaker N`` label, and a clip of that
voice is remembered for the following utterances when the person spoke long
enough to make a usable reference.

The result is a script (``Evan: …`` / ``Speaker 2: …``) that becomes the human
turn's text, plus a ``speakers`` record kept in the message's
``additional_kwargs`` so the web app can paint speaker chips and the graph can
tell whether the owner spoke at all.
"""

from __future__ import annotations

import base64
import difflib
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

from src.anubis.utils.context import GlobalContext

logger = logging.getLogger(__name__)

SPOKEN_TURN_KIND = "spoken_turn"
DEFAULT_OWNER_LABEL = "Owner"
OTHER_SPEAKER_LABEL_PREFIX = "Speaker"
AVATAR_LABEL_SUFFIX = "(avatar)"
# An owner-attributed line this similar to a recent reply is the avatar's own
# cloned voice coming out of a speaker, not the person talking.
AVATAR_ECHO_MIN_CHARACTERS = 20
AVATAR_ECHO_MIN_RATIO = 0.75
# The diarizer accepts references between two and ten seconds long.
REFERENCE_CLIP_MIN_SECONDS = 2.0
REFERENCE_CLIP_MAX_SECONDS = 10.0
# The diarizer accepts at most four known speakers per call.
DIARIZER_MAX_KNOWN_SPEAKERS = 4

_OWNER_CLIP_CACHE_SECONDS = 600.0
_owner_clip_cache: dict[str, tuple[float, str, str | None]] = {}


@dataclass
class LabelledSegment:
    """One stretch of speech attributed to a single speaker."""

    speaker: str
    text: str
    start: float
    end: float
    is_owner: bool = False
    is_new_speaker: bool = False
    is_avatar: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Serialise for ``additional_kwargs`` and the stream frame."""
        return {
            "speaker": self.speaker,
            "text": self.text,
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "is_owner": bool(self.is_owner),
            "is_avatar": bool(self.is_avatar),
        }


@dataclass
class SpokenTurn:
    """The outcome of diarizing one utterance."""

    script: str
    segments: list[LabelledSegment]
    owner_label: str
    owner_identified: bool
    other_speakers: list[str]
    duration_seconds: float
    remembered_new_speakers: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    total_cost: float = 0.0
    latency_ms: float = 0.0
    model: str = ""

    @property
    def owner_spoke(self) -> bool:
        """Whether any segment belongs to the owner."""
        return any(segment.is_owner for segment in self.segments)

    @property
    def others_spoke(self) -> bool:
        """Whether anyone other than the owner and the avatar's own echo spoke."""
        return any(
            not segment.is_owner and not segment.is_avatar for segment in self.segments
        )

    @property
    def avatar_spoke(self) -> bool:
        """Whether the avatar's own voice was heard coming out of a speaker."""
        return any(segment.is_avatar for segment in self.segments)

    def additional_kwargs(self) -> dict[str, Any]:
        """The ``speakers`` record stored on the human message."""
        return {
            "kind": SPOKEN_TURN_KIND,
            "speakers": {
                "owner_label": self.owner_label,
                "avatar_label": avatar_label_for(self.owner_label),
                "owner_identified": self.owner_identified,
                "owner_spoke": self.owner_spoke,
                "avatar_spoke": self.avatar_spoke,
                "others_spoke": self.others_spoke,
                "other_speakers": list(self.other_speakers),
                "segments": [segment.as_dict() for segment in self.segments],
                "duration_seconds": round(float(self.duration_seconds), 3),
            },
        }


def speaker_labels_enabled(context: GlobalContext) -> bool:
    """Whether the deployment labels live-voice utterances by speaker."""
    return str(context.voice_speaker_labels_enabled or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def next_other_speaker_label(existing_labels: list[str]) -> str:
    """The next free ``Speaker N`` label; the owner is speaker one by convention."""
    taken: set[int] = set()
    for label in existing_labels:
        parts = str(label).split()
        if len(parts) == 2 and parts[0] == OTHER_SPEAKER_LABEL_PREFIX and parts[1].isdigit():
            taken.add(int(parts[1]))
    number = 2
    while number in taken:
        number += 1
    return f"{OTHER_SPEAKER_LABEL_PREFIX} {number}"


def label_segments(
    raw_segments: list[Any],
    *,
    owner_label: str,
    owner_reference_given: bool,
    remembered_labels: list[str],
) -> tuple[list[LabelledSegment], dict[str, str]]:
    """Map the diarizer's speaker names onto owner / remembered / new labels.

    The diarizer returns a known speaker's given name when a voice matches a
    reference and an anonymous letter (``A``, ``B`` …) otherwise. Anonymous
    voices are given fresh ``Speaker N`` labels in order of first appearance.
    Returns the labelled segments and the mapping from each anonymous diarizer
    name to the label chosen for that voice.
    """
    # Without an owner reference the diarizer cannot return the owner's name;
    # a raw label that happens to equal the owner label is then just another
    # unknown voice.
    known = set(remembered_labels)
    if owner_reference_given:
        known.add(owner_label)
    new_label_by_raw_name: dict[str, str] = {}
    labelled: list[LabelledSegment] = []
    for raw in raw_segments:
        speaker = str(_attribute(raw, "speaker") or "").strip()
        text = str(_attribute(raw, "text") or "").strip()
        if not text:
            continue
        start = float(_attribute(raw, "start") or 0.0)
        end = float(_attribute(raw, "end") or start)
        if speaker == owner_label and owner_reference_given:
            labelled.append(LabelledSegment(owner_label, text, start, end, is_owner=True))
            continue
        if speaker in known:
            labelled.append(LabelledSegment(speaker, text, start, end))
            continue
        if speaker not in new_label_by_raw_name:
            new_label_by_raw_name[speaker] = next_other_speaker_label(
                [*remembered_labels, *new_label_by_raw_name.values()]
            )
        labelled.append(
            LabelledSegment(
                new_label_by_raw_name[speaker], text, start, end, is_new_speaker=True
            )
        )
    return labelled, new_label_by_raw_name


def avatar_label_for(owner_label: str) -> str:
    """The label of the avatar's own voice heard in the room."""
    return f"{owner_label} {AVATAR_LABEL_SUFFIX}"


def _normalise_words(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(text or "").lower()).strip()


def _normalised_words_collapsed(text: str) -> str:
    return " ".join(_normalise_words(text).split())


def is_avatar_echo(text: str, recent_avatar_replies: list[str]) -> bool:
    """Whether a heard line repeats something the avatar recently said aloud.

    The avatar's cloned voice is the person's voice, so the diarizer attributes
    the avatar's own playback (from another device, or a speaker the echo
    canceller did not catch) to the person. The words give the echo away: a
    line that is a long enough fragment of a recent reply, or close to one, is
    the avatar and not the person.
    """
    heard = _normalised_words_collapsed(text)
    if len(heard) < AVATAR_ECHO_MIN_CHARACTERS:
        return False
    for reply in recent_avatar_replies or []:
        said = _normalised_words_collapsed(reply)
        if not said:
            continue
        if heard in said:
            return True
        ratio = difflib.SequenceMatcher(None, heard, said).ratio()
        if ratio >= AVATAR_ECHO_MIN_RATIO:
            return True
    return False


def mark_avatar_echo(
    segments: list[LabelledSegment],
    *,
    owner_label: str,
    recent_avatar_replies: list[str],
) -> list[LabelledSegment]:
    """Relabel owner-attributed lines that repeat the avatar's recent replies."""
    if not recent_avatar_replies:
        return segments
    relabelled: list[LabelledSegment] = []
    for segment in segments:
        if segment.is_owner and is_avatar_echo(segment.text, recent_avatar_replies):
            relabelled.append(
                LabelledSegment(
                    avatar_label_for(owner_label),
                    segment.text,
                    segment.start,
                    segment.end,
                    is_owner=False,
                    is_avatar=True,
                )
            )
        else:
            relabelled.append(segment)
    return relabelled


def render_speaker_script(segments: list[LabelledSegment]) -> str:
    """Render ``Name: words`` lines, merging consecutive lines of one speaker."""
    lines: list[str] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if lines and lines[-1][0] == segment.speaker:
            lines[-1] = (segment.speaker, f"{lines[-1][1]} {text}")
        else:
            lines.append((segment.speaker, text))
    return "\n".join(f"{speaker}: {text}" for speaker, text in lines)


def _attribute(record: Any, name: str) -> Any:
    if isinstance(record, dict):
        return record.get(name)
    return getattr(record, name, None)


def _ffmpeg_executable() -> str:
    from src.anubis.utils.utility import _ffmpeg_executable as utility_ffmpeg

    return utility_ffmpeg()


def cut_mp3_clip(
    source_path: str, output_path: str, *, start: float, duration: float
) -> None:
    """Cut ``duration`` seconds from ``start`` of an MP3 into a new MP3 file."""
    command = [
        _ffmpeg_executable(),
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, float(start)):.3f}",
        "-t",
        f"{max(0.1, float(duration)):.3f}",
        "-i",
        source_path,
        "-vn",
        "-codec:a",
        "mp3",
        output_path,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not os.path.getsize(output_path):
        detail = (completed.stderr or "").strip()[-1000:] or "no ffmpeg output"
        raise RuntimeError(f"ffmpeg could not cut the speaker clip: {detail}")


def _data_uri(mime_type: str, payload: bytes) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(payload).decode('ascii')}"


async def owner_reference_clip(
    repository: Any,
    assistant_id: str,
    *,
    max_seconds: float,
) -> str | None:
    """A data URI of the owner's voice, cut to ``max_seconds`` from the longest stored clip.

    The clip comes from the recordings the owner made for the voice clone. The
    cut is cached in-process for a few minutes: an utterance arrives every few
    seconds in a live conversation and the recordings rarely change.
    """
    if repository is None or not assistant_id:
        return None
    try:
        clips = await repository.list_voice_clips(assistant_id, include_bytes=False)
    except Exception:  # noqa: BLE001 - no clips means no owner reference
        logger.debug("Could not list voice clips for %s", assistant_id, exc_info=True)
        return None
    if not clips:
        return None
    longest = max(clips, key=lambda clip: float(clip.get("duration_seconds") or 0.0))
    if float(longest.get("duration_seconds") or 0.0) < REFERENCE_CLIP_MIN_SECONDS:
        return None
    clip_id = str(longest.get("clip_id") or "")
    cached = _owner_clip_cache.get(assistant_id)
    now = time.monotonic()
    if cached and cached[1] == clip_id and now - cached[0] < _OWNER_CLIP_CACHE_SECONDS:
        return cached[2]

    with_bytes = await repository.list_voice_clips(assistant_id, include_bytes=True)
    record = next((clip for clip in with_bytes if str(clip.get("clip_id")) == clip_id), None)
    if record is None or not record.get("bytes"):
        return None
    mime_type = str(record.get("mime_type") or "audio/mpeg")
    extension = ".mp3" if "mpeg" in mime_type or "mp3" in mime_type else ".wav"
    source_path = output_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as source_file:
            source_file.write(record["bytes"])
            source_path = source_file.name
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as output_file:
            output_path = output_file.name
        from src.anubis.utils.utility import _transcode_audio_to_mp3

        clip_seconds = min(float(max_seconds), REFERENCE_CLIP_MAX_SECONDS)
        _transcode_audio_to_mp3(source_path, output_path, max_seconds=clip_seconds)
        with open(output_path, "rb") as output_file:
            data_uri = _data_uri("audio/mp3", output_file.read())
    except Exception:  # noqa: BLE001 - a broken clip disables owner labelling
        logger.exception("Could not prepare the owner's reference clip for %s", assistant_id)
        data_uri = None
    finally:
        for path in (source_path, output_path):
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
    _owner_clip_cache[assistant_id] = (now, clip_id, data_uri)
    return data_uri


def forget_owner_reference_clip(assistant_id: str) -> None:
    """Drop the cached owner clip (after the owner's recordings change)."""
    _owner_clip_cache.pop(assistant_id, None)


def _diarize_mp3_with_known_speakers(
    mp3_path: str,
    upload_name: str,
    context: GlobalContext,
    *,
    known_speaker_names: list[str],
    known_speaker_references: list[str],
) -> Any:
    """One synchronous diarizer call with up to four known speakers.

    The configured live-voice language hint rides along so the diarizer does
    not invent a caption in another language for a noisy clip. The diarizer
    accepts no prompt, so none is sent.
    """
    from src.anubis.utils.utility import (
        _openai_client_for_speech,
        _speech_call_with_retry,
        live_voice_transcription_language,
    )

    client = _openai_client_for_speech(context)
    model = context.audio_diarization_model or "gpt-4o-transcribe-diarize"
    optional_arguments: dict[str, Any] = {}
    language = live_voice_transcription_language(context)
    if language:
        optional_arguments["language"] = language
    extra_body: dict[str, Any] = {}
    if known_speaker_names:
        extra_body = {
            "known_speaker_names": list(known_speaker_names),
            "known_speaker_references": list(known_speaker_references),
        }
    if extra_body:
        optional_arguments["extra_body"] = extra_body
    with open(mp3_path, "rb") as audio_file:
        return _speech_call_with_retry(
            lambda: client.audio.transcriptions.create(
                model=model,
                file=(upload_name, audio_file),
                response_format="diarized_json",
                chunking_strategy="auto",
                **optional_arguments,
            ),
            context,
            description=f"diarization({upload_name})",
        )


async def diarize_spoken_turn(
    audio_bytes: bytes,
    *,
    mime_type: str,
    filename: str,
    context: GlobalContext,
    repository: Any,
    user_id: str,
    assistant_id: str,
    thread_id: str | None,
    owner_label: str,
    diarizer: Any = None,
    recent_avatar_replies: list[str] | None = None,
) -> SpokenTurn:
    """Transcribe one utterance and label every line by speaker.

    ``recent_avatar_replies`` (the avatar's last spoken replies) lets the
    avatar's own voice, heard through a speaker and attributed to the person by
    the diarizer, be relabelled as the avatar. ``diarizer`` overrides the OpenAI
    call (tests pass a fake); the default runs
    ``_diarize_mp3_with_known_speakers`` in a worker thread.
    """
    import asyncio

    from src.anubis.utils.utility import (
        _diarize_token_cost,
        _diarize_usage_tokens_dict,
        preprocess_audio,
    )
    from src.anubis.utils.voice.transcript_hygiene import (
        clip_is_silent,
        is_known_hallucination,
    )

    started = time.perf_counter()
    owner_label = (owner_label or DEFAULT_OWNER_LABEL).strip() or DEFAULT_OWNER_LABEL
    max_remembered = int(context.voice_speaker_memory_max_speakers or 3)
    reference_seconds = float(context.voice_speaker_reference_max_seconds or 9.0)
    min_segment_seconds = float(context.voice_speaker_min_segment_seconds or 2.0)

    owner_reference = await owner_reference_clip(
        repository, assistant_id, max_seconds=reference_seconds
    )
    remembered: list[dict[str, Any]] = []
    if repository is not None and thread_id:
        try:
            remembered = await repository.list_thread_speakers(
                assistant_id, thread_id, include_bytes=True
            )
        except Exception:  # noqa: BLE001 - memory is an aid, not a requirement
            logger.debug("Could not list remembered speakers", exc_info=True)
            remembered = []

    known_names: list[str] = []
    known_references: list[str] = []
    if owner_reference:
        known_names.append(owner_label)
        known_references.append(owner_reference)
    for record in remembered:
        if len(known_names) >= DIARIZER_MAX_KNOWN_SPEAKERS:
            break
        payload = record.get("bytes") or b""
        if not payload:
            continue
        known_names.append(str(record["label"]))
        known_references.append(_data_uri(record.get("mime_type") or "audio/mpeg", payload))
    remembered_labels = [name for name in known_names if name != owner_label]

    preprocessed = await preprocess_audio(
        _data_uri(mime_type or "audio/webm", audio_bytes),
        truncate_only=False,
        reference_audio=False,
        filename=filename,
        max_duration_seconds=None,
    )
    mp3_bytes = base64.b64decode(preprocessed["audio_base64"].split(",", 1)[1])
    duration_seconds = float(preprocessed.get("duration_seconds") or 0.0)

    mp3_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_file:
            mp3_file.write(mp3_bytes)
            mp3_path = mp3_file.name
        upload_name = "utterance.mp3"
        # A clip with no speech in it (a cough, a click, trailing room tone) is
        # never sent: the diarizer would invent a caption for the silence.
        if await asyncio.to_thread(
            clip_is_silent,
            mp3_path,
            _ffmpeg_executable(),
            silence_max_volume_db=getattr(context, "voice_silence_max_volume_db", None),
        ):
            return SpokenTurn(
                script="",
                segments=[],
                owner_label=owner_label,
                owner_identified=bool(owner_reference),
                other_speakers=[],
                duration_seconds=duration_seconds,
                remembered_new_speakers=[],
                usage={},
                total_cost=0.0,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                model=str(context.audio_diarization_model or "gpt-4o-transcribe-diarize"),
            )
        if diarizer is not None:
            response = await diarizer(
                mp3_path,
                known_speaker_names=known_names,
                known_speaker_references=known_references,
            )
        else:
            response = await asyncio.to_thread(
                _diarize_mp3_with_known_speakers,
                mp3_path,
                upload_name,
                context,
                known_speaker_names=known_names,
                known_speaker_references=known_references,
            )
        raw_segments = [
            segment
            for segment in (_attribute(response, "segments") or [])
            if not is_known_hallucination(str(_attribute(segment, "text") or ""))
        ]
        segments, new_label_by_raw_name = label_segments(
            raw_segments,
            owner_label=owner_label,
            owner_reference_given=bool(owner_reference),
            remembered_labels=remembered_labels,
        )
        segments = mark_avatar_echo(
            segments,
            owner_label=owner_label,
            recent_avatar_replies=list(recent_avatar_replies or []),
        )
        remembered_new = await _remember_new_speakers(
            repository,
            mp3_path,
            segments,
            new_label_by_raw_name,
            user_id=user_id,
            assistant_id=assistant_id,
            thread_id=thread_id,
            already_remembered=len(remembered_labels),
            max_remembered=max_remembered,
            min_segment_seconds=min_segment_seconds,
        )
    finally:
        if mp3_path:
            try:
                os.unlink(mp3_path)
            except OSError:
                pass

    usage = _diarize_usage_tokens_dict(_attribute(response, "usage"))
    total_cost = _diarize_token_cost(usage, context)
    if total_cost == 0 and duration_seconds:
        total_cost = duration_seconds * float(
            context.audio_diarization_estimated_price_per_minute or 0.0
        )
    other_speakers = sorted(
        {
            segment.speaker
            for segment in segments
            if not segment.is_owner and not segment.is_avatar
        },
        key=lambda label: (len(label), label),
    )
    return SpokenTurn(
        script=render_speaker_script(segments),
        segments=segments,
        owner_label=owner_label,
        owner_identified=bool(owner_reference),
        other_speakers=other_speakers,
        duration_seconds=duration_seconds,
        remembered_new_speakers=remembered_new,
        usage=usage,
        total_cost=float(total_cost or 0.0),
        latency_ms=(time.perf_counter() - started) * 1000.0,
        model=str(context.audio_diarization_model or "gpt-4o-transcribe-diarize"),
    )


async def _remember_new_speakers(
    repository: Any,
    mp3_path: str,
    segments: list[LabelledSegment],
    new_label_by_raw_name: dict[str, str],
    *,
    user_id: str,
    assistant_id: str,
    thread_id: str | None,
    already_remembered: int,
    max_remembered: int,
    min_segment_seconds: float,
) -> list[str]:
    """Store a reference clip for each new voice that spoke long enough."""
    if repository is None or not thread_id or not new_label_by_raw_name:
        return []
    room = max(0, max_remembered - already_remembered)
    if room <= 0:
        return []
    remembered: list[str] = []
    for label in list(new_label_by_raw_name.values())[:room]:
        own_segments = [segment for segment in segments if segment.speaker == label]
        if not own_segments:
            continue
        longest = max(own_segments, key=lambda segment: segment.end - segment.start)
        length = longest.end - longest.start
        if length < min_segment_seconds:
            continue
        clip_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as clip_file:
                clip_path = clip_file.name
            cut_seconds = min(length, REFERENCE_CLIP_MAX_SECONDS)
            cut_mp3_clip(mp3_path, clip_path, start=longest.start, duration=cut_seconds)
            with open(clip_path, "rb") as clip_file:
                clip_bytes = clip_file.read()
            await repository.add_thread_speaker(
                {
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "thread_id": thread_id,
                    "label": label,
                    "mime_type": "audio/mpeg",
                    "bytes": clip_bytes,
                    "duration_seconds": cut_seconds,
                    "sample_text": longest.text[:300],
                }
            )
            remembered.append(label)
        except Exception:  # noqa: BLE001 - a lost memory only costs a label next time
            logger.exception("Could not remember speaker %s", label)
        finally:
            if clip_path:
                try:
                    os.unlink(clip_path)
                except OSError:
                    pass
    return remembered


def spoken_turn_of(message: Any) -> dict[str, Any] | None:
    """The ``speakers`` record of a message, or ``None`` when the turn was typed."""
    if isinstance(message, dict):
        additional_kwargs = message.get("additional_kwargs") or {}
    else:
        additional_kwargs = getattr(message, "additional_kwargs", None) or {}
    if additional_kwargs.get("kind") not in (SPOKEN_TURN_KIND, "ambient_observation"):
        return None
    record = additional_kwargs.get("speakers")
    return dict(record) if isinstance(record, dict) else None
