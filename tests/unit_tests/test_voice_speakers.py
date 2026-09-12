"""Who is speaking: labelling one live-voice utterance by speaker.

Pure label mapping and script rendering, then ``diarize_spoken_turn`` with a
fake diarizer over a real (ffmpeg-made) utterance so the reference clips, the
speaker memory and the ``speakers`` record are exercised end to end without
OpenAI.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.anubis.utils.media_assets.repository import (  # noqa: E402
    InMemoryMediaAssetRepository,
)
from src.anubis.utils.voice import speakers as speakers_module  # noqa: E402
from src.anubis.utils.voice.speakers import (  # noqa: E402
    LabelledSegment,
    SpokenTurn,
    claim_lone_speaker_as_owner,
    diarize_spoken_turn,
    label_segments,
    next_other_speaker_label,
    render_speaker_script,
    spoken_turn_of,
)


def _segment(speaker, text, start, end):
    return SimpleNamespace(speaker=speaker, text=text, start=start, end=end)


def test_next_other_speaker_label_skips_taken_numbers():
    assert next_other_speaker_label([]) == "Speaker 2"
    assert next_other_speaker_label(["Speaker 2"]) == "Speaker 3"
    assert next_other_speaker_label(["Speaker 3"]) == "Speaker 2"
    assert next_other_speaker_label(["Speaker 2", "Speaker 3", "Evan"]) == "Speaker 4"


def test_label_segments_maps_owner_remembered_and_new_voices():
    raw = [
        _segment("Evan", "Meet my friend.", 0.0, 1.5),
        _segment("A", "Hi there.", 1.6, 2.4),
        _segment("Speaker 2", "Long time no see.", 2.5, 4.0),
        _segment("B", "Who is this?", 4.1, 5.0),
        _segment("A", "I am Maria.", 5.1, 6.0),
        _segment("A", "   ", 6.1, 6.2),
    ]
    labelled, new_by_raw = label_segments(
        raw,
        owner_label="Evan",
        owner_reference_given=True,
        remembered_labels=["Speaker 2"],
    )
    assert [segment.speaker for segment in labelled] == [
        "Evan",
        "Speaker 3",
        "Speaker 2",
        "Speaker 4",
        "Speaker 3",
    ]
    assert labelled[0].is_owner and not labelled[1].is_owner
    assert new_by_raw == {"A": "Speaker 3", "B": "Speaker 4"}
    assert labelled[1].is_new_speaker and not labelled[2].is_new_speaker


def test_label_segments_without_owner_reference_never_claims_the_owner():
    raw = [_segment("Evan", "Hello", 0.0, 1.0), _segment("A", "Hi", 1.0, 2.0)]
    labelled, _ = label_segments(
        raw, owner_label="Evan", owner_reference_given=False, remembered_labels=[]
    )
    # "Evan" cannot come back from the diarizer without a reference; if a raw
    # label happens to collide it is treated like any other unknown voice.
    assert [segment.speaker for segment in labelled] == ["Speaker 2", "Speaker 3"]
    assert not any(segment.is_owner for segment in labelled)


def test_claim_lone_speaker_as_owner_relabels_a_monologue_and_forgets_the_new_label():
    segments = [
        LabelledSegment("Speaker 2", "A little bit.", 0.0, 1.0, is_new_speaker=True),
        LabelledSegment("Speaker 2", "Oh, boy.", 1.1, 1.8, is_new_speaker=True),
    ]
    claimed, remaining_new = claim_lone_speaker_as_owner(
        segments,
        owner_label="Shivon",
        new_label_by_raw_name={"A": "Speaker 2"},
    )
    assert [segment.speaker for segment in claimed] == ["Shivon", "Shivon"]
    assert all(segment.is_owner for segment in claimed)
    assert remaining_new == {}


def test_claim_lone_speaker_as_owner_leaves_two_living_voices_alone():
    segments = [
        LabelledSegment("Evan", "Say hi.", 0.0, 1.0, is_owner=True),
        LabelledSegment("Speaker 2", "Hello.", 1.1, 2.0, is_new_speaker=True),
    ]
    claimed, remaining_new = claim_lone_speaker_as_owner(
        segments,
        owner_label="Evan",
        new_label_by_raw_name={"A": "Speaker 2"},
    )
    assert claimed is segments
    assert remaining_new == {"A": "Speaker 2"}


def test_claim_lone_speaker_as_owner_ignores_avatar_echo_when_counting():
    segments = [
        LabelledSegment("Evan (avatar)", "Give him a moment.", 0.0, 1.5, is_avatar=True),
        LabelledSegment("Speaker 2", "And just that individual version of myself.", 1.6, 3.0),
    ]
    claimed, remaining_new = claim_lone_speaker_as_owner(
        segments,
        owner_label="Evan",
        new_label_by_raw_name={"A": "Speaker 2"},
    )
    assert [segment.speaker for segment in claimed] == [
        "Evan (avatar)",
        "Evan",
    ]
    assert claimed[0].is_avatar and claimed[1].is_owner
    assert remaining_new == {}


def test_render_speaker_script_merges_consecutive_lines():
    segments = [
        LabelledSegment("Evan", "Hello.", 0, 1, is_owner=True),
        LabelledSegment("Evan", "How are you?", 1, 2, is_owner=True),
        LabelledSegment("Speaker 2", "Fine, thanks.", 2, 3),
        LabelledSegment("Evan", "Good.", 3, 4, is_owner=True),
    ]
    # Only the third voice is named. The person at the microphone is the one
    # person the avatar is talking to, so naming their lines added nothing and
    # named them wrongly on every avatar that is not a portrait of them.
    assert render_speaker_script(segments) == (
        "Hello. How are you?\nSpeaker 2: Fine, thanks.\nGood."
    )


def test_spoken_turn_of_reads_both_kinds():
    record = {"segments": []}
    assert spoken_turn_of({"additional_kwargs": {"kind": "spoken_turn", "speakers": record}}) == record
    assert (
        spoken_turn_of({"additional_kwargs": {"kind": "ambient_observation", "speakers": record}})
        == record
    )
    assert spoken_turn_of({"additional_kwargs": {"kind": "ambient_observation"}}) is None
    assert spoken_turn_of({"additional_kwargs": {}}) is None


# --- diarize_spoken_turn over a real utterance ---------------------------------


def _ffmpeg_available() -> bool:
    executable = speakers_module._ffmpeg_executable()
    return bool(shutil.which(executable) or Path(executable).exists())


def _sine_webm(seconds: float) -> bytes:
    return subprocess.run(
        [
            speakers_module._ffmpeg_executable(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=330:duration={seconds}",
            "-c:a",
            "libopus",
            "-f",
            "webm",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout


def _sine_mp3(seconds: float) -> bytes:
    return subprocess.run(
        [
            speakers_module._ffmpeg_executable(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=220:duration={seconds}",
            "-codec:a",
            "mp3",
            "-f",
            "mp3",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout


def _context():
    return SimpleNamespace(
        voice_speaker_labels_enabled="true",
        voice_speaker_memory_max_speakers=2,
        voice_speaker_min_segment_seconds=1.0,
        voice_speaker_reference_max_seconds=3.0,
        audio_diarization_model="diarize-test",
        audio_diarization_estimated_price_per_minute=0.006,
        audio_diarization_price_per_million_tokens_input=0.0,
        audio_diarization_price_per_million_tokens_output=0.0,
    )


class _FakeDiarizer:
    """Records the known speakers handed in and plays back scripted segments."""

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.calls = []

    async def __call__(self, mp3_path, *, known_speaker_names, known_speaker_references):
        assert Path(mp3_path).exists()
        assert len(known_speaker_names) == len(known_speaker_references)
        for reference in known_speaker_references:
            assert reference.startswith("data:audio/")
        self.calls.append(list(known_speaker_names))
        segments = self.scripts.pop(0)
        return SimpleNamespace(
            segments=segments,
            usage={"type": "tokens", "input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg is not available")
def test_diarize_spoken_turn_labels_owner_remembers_others_and_keeps_labels_stable(monkeypatch):
    monkeypatch.setattr(speakers_module, "_diarize_token_cost", lambda usage, context: 0.0, raising=False)
    repository = InMemoryMediaAssetRepository()
    asyncio.run(
        repository.add_voice_clip(
            {
                "user_id": "u1",
                "assistant_id": "a1",
                "source": "recording",
                "mime_type": "audio/mpeg",
                "bytes": _sine_mp3(4.0),
                "duration_seconds": 4.0,
            }
        )
    )
    utterance = _sine_webm(6.0)
    diarizer = _FakeDiarizer(
        [
            [
                _segment("Evan", "This is my colleague.", 0.0, 1.8),
                _segment("A", "Nice to meet you, I am Maria.", 2.0, 4.5),
                _segment("B", "Hey.", 4.6, 5.0),
            ],
            [
                _segment("Speaker 2", "Can you tell me about the project?", 0.0, 2.5),
                _segment("Evan", "Go ahead.", 2.6, 3.4),
            ],
        ]
    )
    common = {
        "mime_type": "audio/webm",
        "filename": "utterance.webm",
        "context": _context(),
        "repository": repository,
        "user_id": "u1",
        "assistant_id": "a1",
        "thread_id": "t1",
        "owner_label": "Evan",
        "diarizer": diarizer,
    }

    first = asyncio.run(diarize_spoken_turn(utterance, **common))
    assert diarizer.calls[0] == ["Evan"], "the owner's clip is the only known voice at first"
    assert first.owner_identified and first.owner_spoke and first.others_spoke
    assert first.script == (
        "This is my colleague.\nSpeaker 2: Nice to meet you, I am Maria.\nSpeaker 3: Hey."
    )
    assert first.other_speakers == ["Speaker 2", "Speaker 3"]
    # Maria spoke long enough to be remembered; the short "Hey." did not.
    assert first.remembered_new_speakers == ["Speaker 2"]
    remembered = asyncio.run(repository.list_thread_speakers("a1", "t1", include_bytes=True))
    assert [record["label"] for record in remembered] == ["Speaker 2"]
    assert remembered[0]["bytes"] and remembered[0]["duration_seconds"] == pytest.approx(2.5, abs=0.05)
    assert first.duration_seconds == pytest.approx(6.0, abs=0.3)
    assert first.usage["total_tokens"] == 15
    assert first.total_cost > 0  # falls back to the per-minute estimate

    record = first.additional_kwargs()
    assert record["kind"] == "spoken_turn"
    assert record["speakers"]["owner_label"] == "Evan"
    assert record["speakers"]["segments"][0]["is_owner"] is True
    assert record["speakers"]["segments"][1]["speaker"] == "Speaker 2"

    second = asyncio.run(diarize_spoken_turn(utterance, **common))
    assert diarizer.calls[1] == ["Evan", "Speaker 2"], "the remembered voice rides along"
    assert second.script == "Speaker 2: Can you tell me about the project?\nGo ahead."
    assert second.remembered_new_speakers == []


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg is not available")
def test_diarize_spoken_turn_without_owner_recordings_treats_a_lone_speaker_as_the_owner():
    repository = InMemoryMediaAssetRepository()
    utterance = _sine_webm(2.0)
    diarizer = _FakeDiarizer([[_segment("A", "Hello there.", 0.0, 1.5)]])
    turn = asyncio.run(
        diarize_spoken_turn(
            utterance,
            mime_type="audio/webm",
            filename="utterance.webm",
            context=_context(),
            repository=repository,
            user_id="u1",
            assistant_id="a1",
            thread_id="t1",
            owner_label="Evan",
            diarizer=diarizer,
        )
    )
    assert diarizer.calls == [[]]
    assert turn.owner_spoke and not turn.others_spoke
    assert turn.script == "Hello there."
    assert turn.remembered_new_speakers == []
    remembered = asyncio.run(repository.list_thread_speakers("a1", "t1", include_bytes=True))
    assert remembered == []


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg is not available")
def test_diarize_spoken_turn_treats_a_lone_unmatched_voice_as_the_owner(monkeypatch):
    monkeypatch.setattr(speakers_module, "_diarize_token_cost", lambda usage, context: 0.0, raising=False)
    repository = InMemoryMediaAssetRepository()
    asyncio.run(
        repository.add_voice_clip(
            {
                "user_id": "u1",
                "assistant_id": "a1",
                "source": "recording",
                "mime_type": "audio/mpeg",
                "bytes": _sine_mp3(4.0),
                "duration_seconds": 4.0,
            }
        )
    )
    diarizer = _FakeDiarizer([[_segment("A", "A little bit.", 0.0, 1.4)]])
    turn = asyncio.run(
        diarize_spoken_turn(
            _sine_webm(2.0),
            mime_type="audio/webm",
            filename="utterance.webm",
            context=_context(),
            repository=repository,
            user_id="u1",
            assistant_id="a1",
            thread_id="t1",
            owner_label="Shivon",
            diarizer=diarizer,
        )
    )
    assert diarizer.calls == [["Shivon"]], "the owner's clip was handed in"
    assert turn.owner_spoke and not turn.others_spoke
    assert turn.script == "A little bit."
    assert turn.remembered_new_speakers == []


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg is not available")
def test_diarize_spoken_turn_does_not_keep_a_remembered_lone_voice_as_someone_else(monkeypatch):
    monkeypatch.setattr(speakers_module, "_diarize_token_cost", lambda usage, context: 0.0, raising=False)
    repository = InMemoryMediaAssetRepository()
    asyncio.run(
        repository.add_voice_clip(
            {
                "user_id": "u1",
                "assistant_id": "a1",
                "source": "recording",
                "mime_type": "audio/mpeg",
                "bytes": _sine_mp3(4.0),
                "duration_seconds": 4.0,
            }
        )
    )
    asyncio.run(
        repository.add_thread_speaker(
            {
                "user_id": "u1",
                "assistant_id": "a1",
                "thread_id": "t1",
                "label": "Speaker 2",
                "mime_type": "audio/mpeg",
                "bytes": _sine_mp3(3.0),
                "duration_seconds": 3.0,
                "sample_text": "A little bit.",
            }
        )
    )
    diarizer = _FakeDiarizer(
        [[_segment("Speaker 2", "to pay your wall, you know.", 0.0, 2.0)]]
    )
    turn = asyncio.run(
        diarize_spoken_turn(
            _sine_webm(3.0),
            mime_type="audio/webm",
            filename="utterance.webm",
            context=_context(),
            repository=repository,
            user_id="u1",
            assistant_id="a1",
            thread_id="t1",
            owner_label="Shivon",
            diarizer=diarizer,
        )
    )
    assert diarizer.calls == [["Shivon", "Speaker 2"]]
    assert turn.owner_spoke and not turn.others_spoke
    assert turn.script == "to pay your wall, you know."


# --- the avatar's own voice heard through a speaker --------------------------------

from src.anubis.utils.voice.speakers import (  # noqa: E402
    is_avatar_echo,
    mark_avatar_echo,
)


def test_is_avatar_echo_matches_fragments_and_near_repeats_only_when_long_enough():
    replies = ["Sure, the project kicks off next Monday and I will send the plan tonight."]
    assert is_avatar_echo("the project kicks off next Monday", replies)
    assert is_avatar_echo("Sure the project kicks off next Monday and I'll send the plan tonight", replies)
    assert not is_avatar_echo("Yes.", replies), "short lines are never treated as echo"
    assert not is_avatar_echo("What time does the meeting start tomorrow morning?", replies)
    assert not is_avatar_echo("the project kicks off next Monday", [])


def test_mark_avatar_echo_relabels_owner_lines_that_repeat_replies():
    segments = [
        LabelledSegment("Evan", "the project kicks off next Monday and I will send the plan", 0, 3, is_owner=True),
        LabelledSegment("Evan", "Right, and remind me to call Maria.", 3, 5, is_owner=True),
        LabelledSegment("Speaker 2", "the project kicks off next Monday and I will send the plan", 5, 8),
    ]
    marked = mark_avatar_echo(
        segments,
        avatar_label="Evan",
        recent_avatar_replies=["The project kicks off next Monday and I will send the plan tonight."],
    )
    assert [segment.speaker for segment in marked] == ["Evan (avatar)", "Evan", "Speaker 2"]
    assert marked[0].is_avatar and not marked[0].is_owner
    assert marked[1].is_owner
    assert not marked[2].is_avatar, "only owner-attributed lines can be the clone's echo"
    turn = SpokenTurn(
        script=render_speaker_script(marked[:2]),
        segments=marked[:2],
        owner_label="Evan",
        owner_identified=True,
        other_speakers=[],
        duration_seconds=5.0,
        avatar_label="Evan",
    )
    assert turn.avatar_spoke and turn.owner_spoke and not turn.others_spoke
    record = turn.additional_kwargs()["speakers"]
    assert record["avatar_label"] == "Evan (avatar)"
    assert record["segments"][0]["is_avatar"] is True
