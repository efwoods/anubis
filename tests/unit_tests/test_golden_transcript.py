"""Unit tests for the golden-transcript reformatter (Quest 3).

Covers:
* ``build_golden_transcript`` — coalescing, target relabeling, recomputed text,
  validated against the hand-annotated golden dataset fixture.
* ``_reconcile_speakers`` — collapsing diarizer over-split labels via a stubbed
  SpeakerReconciliation judge.
* ``_window_text`` — bounded windowing of long-form prose.
"""

import json
from pathlib import Path

import pytest

from src.subgraphs.process_media_graph.utils.helper_functions import (
    _reconcile_speakers,
    _window_text,
    build_golden_transcript,
)

_FIXTURE_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "dbe60d13-89c5-4206-aa8d-8dd10592c559"
    / "transcriptions"
)
_RAW = _FIXTURE_DIR / "https___www.youtube.com_watch_v_CkUcCcRq_eM_1782152050694843865.json"
_GOLDEN = (
    _FIXTURE_DIR
    / "https___www.youtube.com_watch_v_CkUcCcRq_eM_1782152050694843865_golden_dataset_hand_annotated.json"
)


# --------------------------------------------------------------------------- #
# build_golden_transcript
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_build_golden_transcript_coalesces_and_matches_format():
    raw = json.loads(_RAW.read_text())
    golden = json.loads(_GOLDEN.read_text())

    # reconcile=False keeps the diarizer's own labels (no LLM in unit tests).
    out = await build_golden_transcript(
        raw, target_speaker_label="avatar", reference_audio=False, reconcile=False
    )

    # Same top-level shape as the hand-annotated golden dataset.
    assert set(out.keys()) == {
        "source",
        "filename",
        "model",
        "duration",
        "text",
        "segments",
    }
    assert out["model"] == raw["model"]

    # Coalescing collapses the 494 raw segments to a small number of turns
    # (the hand-annotated reference has 23).
    assert len(out["segments"]) < len(raw["segments"])
    assert len(out["segments"]) <= 2 * len(golden["segments"])

    # The target is labeled "avatar" and those turns carry is_target.
    avatar_segments = [s for s in out["segments"] if s["speaker"] == "avatar"]
    assert avatar_segments
    assert all(s["is_target"] for s in avatar_segments)
    assert not any(
        s["is_target"] for s in out["segments"] if s["speaker"] != "avatar"
    )

    # Top-level text is recomputed from the coalesced turns.
    assert out["text"] == "\n".join(s["text"] for s in out["segments"])


@pytest.mark.asyncio
async def test_reference_audio_makes_single_avatar_turn():
    raw = {
        "model": "m",
        "segments": [
            {"speaker": "A", "text": "one", "start": 0, "end": 1},
            {"speaker": "B", "text": "two", "start": 1, "end": 2},
        ],
    }
    out = await build_golden_transcript(
        raw, target_speaker_label="avatar", reference_audio=True, reconcile=False
    )
    # A reference upload IS the target: every turn becomes the avatar and merges.
    assert {s["speaker"] for s in out["segments"]} == {"avatar"}
    assert all(s["is_target"] for s in out["segments"])


# --------------------------------------------------------------------------- #
# _reconcile_speakers — fixes diarizer over-splitting
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_reconcile_merges_oversplit_target(monkeypatch):
    """A target split across labels 'avatar' and 'C' is merged under 'avatar'."""
    import src.anubis.utils.classes.SpeakerReconciliationClass as recon_module

    class _FakeReconciler:
        async def reconcile(self, turns, target_speaker_label=None):
            return {
                "label_map": [
                    {"raw_speaker": "avatar", "canonical_speaker": "avatar", "is_target": True},
                    {"raw_speaker": "C", "canonical_speaker": "avatar", "is_target": True},
                    {"raw_speaker": "B", "canonical_speaker": "B", "is_target": False},
                ],
                "canonical_speaker_count": 2,
                "reasoning": "C and avatar are the same speaker.",
            }

    monkeypatch.setattr(
        recon_module, "SpeakerReconciliationClass", _FakeReconciler
    )

    turns = [
        {"speaker": "avatar", "text": "Hello.", "is_target": True, "start": 0, "end": 1},
        {"speaker": "B", "text": "Hi there.", "is_target": False, "start": 1, "end": 2},
        {"speaker": "C", "text": "Good to be here.", "is_target": False, "start": 2, "end": 3},
    ]
    reconciled = await _reconcile_speakers(turns, "avatar")
    speakers = [t["speaker"] for t in reconciled]
    # 'C' was remapped to 'avatar'; both target turns are flagged.
    assert "C" not in speakers
    assert speakers.count("avatar") >= 1
    target_text = " ".join(t["text"] for t in reconciled if t.get("is_target"))
    assert "Hello." in target_text and "Good to be here." in target_text


@pytest.mark.asyncio
async def test_reconcile_returns_input_on_failure(monkeypatch):
    import src.anubis.utils.classes.SpeakerReconciliationClass as recon_module

    class _BoomReconciler:
        async def reconcile(self, turns, target_speaker_label=None):
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(
        recon_module, "SpeakerReconciliationClass", _BoomReconciler
    )
    turns = [{"speaker": "A", "text": "x", "is_target": False}]
    assert await _reconcile_speakers(turns, "avatar") == turns


# --------------------------------------------------------------------------- #
# _window_text
# --------------------------------------------------------------------------- #


def test_window_text_respects_cap_and_boundaries():
    text = ("Sentence one. " * 100) + "\n\n" + ("Sentence two. " * 100)
    windows = _window_text(text, window_chars=500, max_windows=3)
    assert 1 <= len(windows) <= 3
    assert all(len(w) <= 500 for w in windows[:-1])


def test_window_text_small_input_single_window():
    assert _window_text("just a little text", 500, 5) == ["just a little text"]
