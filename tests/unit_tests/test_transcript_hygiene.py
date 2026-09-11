"""Live-voice speech hygiene: invented captions and silent clips never become turns.

Whisper-style models transcribe silence as memorised subtitles
(``MBC 뉴스 이덕영입니다``, ``Thank you for watching``). In voice mode those
captions were shown as what the person said and answered by the avatar.
"""

from __future__ import annotations

import asyncio
import base64
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import src.anubis.utils.utility as utility_module  # noqa: E402
import src.anubis.utils.voice.speakers as speakers_module  # noqa: E402
from src.anubis.utils.utility import (  # noqa: E402
    _ffmpeg_executable,
    live_voice_transcription_language,
    transcribe_audio,
)
from src.anubis.utils.voice.transcript_hygiene import (  # noqa: E402
    clip_is_silent,
    drop_hallucinated_text,
    is_known_hallucination,
    keep_confident_segments,
    measure_peak_volume_db,
)


def _ffmpeg_available() -> bool:
    executable = _ffmpeg_executable()
    return bool(shutil.which(executable) or Path(executable).exists())


requires_ffmpeg = pytest.mark.skipif(
    not _ffmpeg_available(), reason="ffmpeg is not available in this environment"
)


# --- caption recognition ------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "MBC 뉴스 이덕영입니다.",
        "  MBC 뉴스 이덕영입니다  ",
        "mbc 뉴스 이덕영입니다",
        "MBC 뉴스 이덕영입니다. MBC 뉴스 이덕영입니다.",
        "Thank you for watching!",
        "Thanks for watching.",
        "Subtitles by the Amara.org community",
        "字幕由Amara.org社区提供",
        "ご視聴ありがとうございました",
        "Learn more at www.plastics-car.com",
        "Visit www.Flydreamers.com to learn more.",
        "Please see the complete disclaimer at https://sites.google.com",
        "Thank you for joining us.",
        "We'll see you next time. Bye for now.",
        "Subtitles by the Amara.org community",
        "",
        "   ",
    ],
)
def test_memorised_captions_are_recognised(text):
    assert is_known_hallucination(text)


@pytest.mark.parametrize(
    "text",
    [
        "Thank you for watching my garden grow this summer.",
        "Can you order me a pizza?",
        "I saw the MBC news this morning.",
        "Bye, see you tomorrow.",
        "Thanks for watching the kids while I was out.",
        "Thank you.",
        "Love you. Bye.",
        "Can you hear me? This is a microphone test.",
        "Tell me a story you would tell to your kids",
    ],
)
def test_real_speech_is_kept(text):
    assert not is_known_hallucination(text)
    assert drop_hallucinated_text(text) == text


def test_drop_hallucinated_text_returns_empty_for_a_caption():
    assert drop_hallucinated_text("MBC 뉴스 이덕영입니다.") == ""


# --- whisper confidence filter ----------------------------------------------------


def _whisper_segment(text, *, no_speech=0.01, logprob=-0.2, compression=1.3):
    return SimpleNamespace(
        text=text,
        no_speech_prob=no_speech,
        avg_logprob=logprob,
        compression_ratio=compression,
    )


def test_confident_segments_survive_and_noisy_ones_are_dropped():
    segments = [
        _whisper_segment(" Can you hear me?"),
        _whisper_segment(" MBC 뉴스 이덕영입니다.", no_speech=0.92, logprob=-0.9),
        _whisper_segment(" Learn more at www.plastics-car.com", no_speech=0.2),
        _whisper_segment(" you you you you you", compression=3.1),
        _whisper_segment(" mumble mumble", logprob=-1.4),
        {
            "text": " This is a microphone test.",
            "no_speech_prob": 0.1,
            "avg_logprob": -0.3,
        },
    ]
    text, dropped = keep_confident_segments(
        segments,
        no_speech_probability_max=0.6,
        average_logprob_min=-1.0,
        compression_ratio_max=2.4,
    )
    assert text == "Can you hear me? This is a microphone test."
    assert dropped == 4


def test_confidence_checks_can_be_disabled():
    segments = [
        _whisper_segment(" hello", no_speech=0.99, logprob=-3.0, compression=9.0)
    ]
    text, dropped = keep_confident_segments(
        segments,
        no_speech_probability_max=0.0,
        average_logprob_min=0.0,
        compression_ratio_max=0.0,
    )
    assert text == "hello"
    assert dropped == 0


# --- language hint --------------------------------------------------------------


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("en", "en"),
        ("EN", "en"),
        ("fr", "fr"),
        ("none", None),
        ("auto", None),
        ("", None),
        (None, None),
    ],
)
def test_language_hint_resolution(configured, expected):
    context = SimpleNamespace(voice_transcription_language=configured)
    assert live_voice_transcription_language(context) == expected


# --- silence gate ---------------------------------------------------------------


def _synth_mp3(source: str) -> bytes:
    return subprocess.run(
        [
            _ffmpeg_executable(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            source,
            "-codec:a",
            "mp3",
            "-f",
            "mp3",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout


def _silent_mp3(seconds: float = 1.2) -> bytes:
    return _synth_mp3(f"anullsrc=r=48000:cl=mono:d={seconds}")


def _tone_mp3(seconds: float = 1.2) -> bytes:
    return _synth_mp3(f"sine=frequency=330:duration={seconds}")


def _data_uri(payload: bytes, mime_type: str = "audio/mpeg") -> str:
    return f"data:{mime_type};base64,{base64.b64encode(payload).decode('ascii')}"


@requires_ffmpeg
def test_peak_volume_separates_silence_from_a_tone(tmp_path):
    silent = tmp_path / "silent.mp3"
    silent.write_bytes(_silent_mp3())
    tone = tmp_path / "tone.mp3"
    tone.write_bytes(_tone_mp3())

    silent_peak, _ = measure_peak_volume_db(str(silent), _ffmpeg_executable())
    tone_peak, _ = measure_peak_volume_db(str(tone), _ffmpeg_executable())
    assert silent_peak < -60.0
    assert tone_peak > -30.0

    assert clip_is_silent(
        str(silent), _ffmpeg_executable(), silence_max_volume_db=-45.0
    )
    assert not clip_is_silent(
        str(tone), _ffmpeg_executable(), silence_max_volume_db=-45.0
    )
    # A floor of zero or higher disables the gate.
    assert not clip_is_silent(
        str(silent), _ffmpeg_executable(), silence_max_volume_db=0.0
    )
    assert not clip_is_silent(
        str(silent), _ffmpeg_executable(), silence_max_volume_db=None
    )


def test_unmeasurable_clip_is_not_treated_as_silent(tmp_path):
    missing = tmp_path / "missing.mp3"
    assert measure_peak_volume_db(str(missing), _ffmpeg_executable()) is None
    assert not clip_is_silent(
        str(missing), _ffmpeg_executable(), silence_max_volume_db=-45.0
    )


# --- transcribe_audio on the live-voice path --------------------------------------


def _speech_context(**overrides):
    values = {
        "audio_transcription_model": "whisper-test",
        "audio_transcription_price_per_minute": 0.006,
        "whisper_max_bytes": 26214400,
        "chunk_source_bytes_target": 20971520,
        "voice_transcription_language": "en",
        "voice_transcription_prompt": "",
        "voice_no_speech_probability_max": 0.6,
        "voice_average_logprob_min": -1.0,
        "voice_compression_ratio_max": 2.4,
        "voice_silence_max_volume_db": -45.0,
        "openai_speech_max_retries": 0,
        "openai_speech_retry_base_seconds": 0.0,
        "dev": "FALSE",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeTranscriptions:
    """Answers ``text`` for response_format=text and a verbose object otherwise."""

    def __init__(self, text: str, segments=None):
        self.text = text
        self.segments = segments
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("response_format") == "text":
            return self.text
        segments = self.segments
        if segments is None:
            segments = [_whisper_segment(self.text)]
        return SimpleNamespace(
            text=self.text, segments=segments, language="en", duration=1.2
        )


def _install_fake_speech_client(
    monkeypatch, text: str, segments=None
) -> _FakeTranscriptions:
    transcriptions = _FakeTranscriptions(text, segments)
    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=transcriptions))
    monkeypatch.setattr(
        utility_module, "_openai_client_for_speech", lambda context: client
    )
    return transcriptions


@requires_ffmpeg
def test_silent_live_voice_clip_never_reaches_the_speech_model(monkeypatch):
    transcriptions = _install_fake_speech_client(monkeypatch, "MBC 뉴스 이덕영입니다.")
    result = asyncio.run(
        transcribe_audio(
            _data_uri(_silent_mp3()),
            _speech_context(),
            filename="utterance.mp3",
            reference_audio=False,
            max_duration_seconds=None,
            live_voice=True,
        )
    )
    assert result["text"] == ""
    assert result["skipped_silent"] is True
    assert result["total_cost"] == 0.0
    assert transcriptions.calls == []


@requires_ffmpeg
def test_live_voice_caption_is_dropped_and_language_hint_is_sent(monkeypatch):
    transcriptions = _install_fake_speech_client(monkeypatch, "MBC 뉴스 이덕영입니다.")
    result = asyncio.run(
        transcribe_audio(
            _data_uri(_tone_mp3()),
            _speech_context(voice_transcription_prompt="Casual conversation."),
            filename="utterance.mp3",
            reference_audio=False,
            max_duration_seconds=None,
            live_voice=True,
        )
    )
    assert result["text"] == ""
    assert result["dropped_segments"] == 1
    assert len(transcriptions.calls) == 1
    call = transcriptions.calls[0]
    assert call["language"] == "en"
    assert call["prompt"] == "Casual conversation."
    assert call["model"] == "whisper-test"
    assert call["response_format"] == "verbose_json"


@requires_ffmpeg
def test_low_confidence_live_voice_segments_are_dropped(monkeypatch):
    _install_fake_speech_client(
        monkeypatch,
        "Thank you. Learn more at www.plastics-car.com",
        segments=[
            _whisper_segment(" Thank you.", no_speech=0.85),
            _whisper_segment(" Learn more at www.plastics-car.com", no_speech=0.3),
        ],
    )
    result = asyncio.run(
        transcribe_audio(
            _data_uri(_tone_mp3()),
            _speech_context(),
            filename="utterance.mp3",
            reference_audio=False,
            max_duration_seconds=None,
            live_voice=True,
        )
    )
    assert result["text"] == ""
    assert result["dropped_segments"] == 2


@requires_ffmpeg
def test_real_live_voice_words_pass_through(monkeypatch):
    _install_fake_speech_client(monkeypatch, "Order me a pizza, please.")
    result = asyncio.run(
        transcribe_audio(
            _data_uri(_tone_mp3()),
            _speech_context(),
            filename="utterance.mp3",
            reference_audio=False,
            max_duration_seconds=None,
            live_voice=True,
        )
    )
    assert result["text"] == "Order me a pizza, please."


@requires_ffmpeg
def test_uploaded_media_is_not_gated_or_language_forced(monkeypatch):
    transcriptions = _install_fake_speech_client(monkeypatch, "Thank you for watching.")
    result = asyncio.run(
        transcribe_audio(
            _data_uri(_silent_mp3()),
            _speech_context(),
            filename="lecture.mp3",
            reference_audio=False,
            max_duration_seconds=None,
        )
    )
    assert result["text"] == "Thank you for watching."
    assert len(transcriptions.calls) == 1
    assert "language" not in transcriptions.calls[0]
    assert "prompt" not in transcriptions.calls[0]
    assert transcriptions.calls[0]["response_format"] == "text"


# --- diarized spoken turn --------------------------------------------------------


def _speaker_context(**overrides):
    values = {
        "voice_speaker_labels_enabled": "true",
        "voice_speaker_memory_max_speakers": 2,
        "voice_speaker_min_segment_seconds": 1.0,
        "voice_speaker_reference_max_seconds": 3.0,
        "voice_transcription_language": "en",
        "voice_silence_max_volume_db": -45.0,
        "audio_diarization_model": "diarize-test",
        "audio_diarization_estimated_price_per_minute": 0.006,
        "audio_diarization_price_per_million_tokens_input": 0.0,
        "audio_diarization_price_per_million_tokens_output": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeDiarizer:
    def __init__(self, segments):
        self.segments = segments
        self.calls = 0

    async def __call__(
        self, mp3_path, *, known_speaker_names, known_speaker_references
    ):
        self.calls += 1
        return SimpleNamespace(
            segments=self.segments,
            usage={
                "type": "tokens",
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        )


def _segment(speaker, text, start, end):
    return SimpleNamespace(speaker=speaker, text=text, start=start, end=end)


def _no_owner_clip(monkeypatch):
    async def none_clip(repository, assistant_id, **kwargs):
        return None

    monkeypatch.setattr(speakers_module, "owner_reference_clip", none_clip)


@requires_ffmpeg
def test_silent_spoken_turn_skips_the_diarizer(monkeypatch):
    _no_owner_clip(monkeypatch)
    diarizer = _FakeDiarizer([_segment("A", "MBC 뉴스 이덕영입니다.", 0.0, 1.0)])
    spoken = asyncio.run(
        speakers_module.diarize_spoken_turn(
            _silent_mp3(),
            mime_type="audio/mpeg",
            filename="utterance.mp3",
            context=_speaker_context(),
            repository=None,
            user_id="user",
            assistant_id="assistant",
            thread_id=None,
            owner_label="Evan",
            diarizer=diarizer,
        )
    )
    assert diarizer.calls == 0
    assert spoken.script == ""
    assert spoken.segments == []
    assert spoken.total_cost == 0.0


@requires_ffmpeg
def test_hallucinated_segments_are_dropped_from_a_spoken_turn(monkeypatch):
    _no_owner_clip(monkeypatch)
    diarizer = _FakeDiarizer(
        [
            _segment("A", "MBC 뉴스 이덕영입니다.", 0.0, 0.6),
            _segment("A", "Can you order me a pizza?", 0.6, 1.2),
            _segment("A", "Thank you for watching.", 1.2, 1.4),
        ]
    )
    spoken = asyncio.run(
        speakers_module.diarize_spoken_turn(
            _tone_mp3(),
            mime_type="audio/mpeg",
            filename="utterance.mp3",
            context=_speaker_context(),
            repository=None,
            user_id="user",
            assistant_id="assistant",
            thread_id=None,
            owner_label="Evan",
            diarizer=diarizer,
        )
    )
    assert diarizer.calls == 1
    assert [segment.text for segment in spoken.segments] == [
        "Can you order me a pizza?"
    ]
    assert spoken.segments[0].is_owner
    assert spoken.script == "Can you order me a pizza?"
    assert "MBC" not in spoken.script


def test_diarizer_call_carries_the_language_hint(monkeypatch, tmp_path):
    captured: dict = {}

    class _Transcriptions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(segments=[], usage={})

    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=_Transcriptions()))
    monkeypatch.setattr(
        utility_module, "_openai_client_for_speech", lambda context: client
    )
    monkeypatch.setattr(
        utility_module,
        "_speech_call_with_retry",
        lambda call, context, description: call(),
    )
    clip = tmp_path / "utterance.mp3"
    clip.write_bytes(b"not really audio")
    speakers_module._diarize_mp3_with_known_speakers(
        str(clip),
        "utterance.mp3",
        _speaker_context(),
        known_speaker_names=["Evan"],
        known_speaker_references=["data:audio/mpeg;base64,AAAA"],
    )
    assert captured["language"] == "en"
    assert captured["extra_body"]["known_speaker_names"] == ["Evan"]
    assert "prompt" not in captured

    captured.clear()
    speakers_module._diarize_mp3_with_known_speakers(
        str(clip),
        "utterance.mp3",
        _speaker_context(voice_transcription_language="none"),
        known_speaker_names=[],
        known_speaker_references=[],
    )
    assert "language" not in captured
    assert "extra_body" not in captured
