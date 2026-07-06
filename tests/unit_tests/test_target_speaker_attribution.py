"""Unit tests for Part A: post-diarization target-speaker attribution.

Covers the deterministic per-chunk label namespacing in
``_merge_diarized_segments_from_chunks`` and the adjudication application
(``adjudicate_target_speaker_labels`` with the model stubbed), including the
union-with-diarizer-votes rule, the low-confidence-stays-non-target rule, and
the character-limit per-chunk fallback.
"""

from types import SimpleNamespace

import pytest

import src.subgraphs.process_media_graph.utils.target_attribution as target_attribution
from src.anubis.utils.utility import _merge_diarized_segments_from_chunks
from src.subgraphs.process_media_graph.utils.helper_functions import (
    coalesce_segments_by_speaker,
)


class _FakeSegment:
    def __init__(self, speaker, text, start=0.0, end=1.0):
        self._data = {"speaker": speaker, "text": text, "start": start, "end": end}

    def model_dump(self):
        return dict(self._data)


class _FakeChunkResponse:
    def __init__(self, segments):
        self.segments = segments


# --------------------------------------------------------------------------- #
# _merge_diarized_segments_from_chunks — per-chunk label namespacing
# --------------------------------------------------------------------------- #


def test_merge_namespaces_non_target_labels_when_known_label_given():
    chunk0 = _FakeChunkResponse(
        [_FakeSegment("avatar", "target speech"), _FakeSegment("speaker_1", "other")]
    )
    chunk1 = _FakeChunkResponse(
        [_FakeSegment("speaker_0", "someone"), _FakeSegment("avatar", "target again")]
    )
    merged = _merge_diarized_segments_from_chunks(
        [chunk0, chunk1], [0.0, 100.0], known_speaker_label="avatar"
    )
    labels = [seg["speaker"] for seg in merged]
    # "avatar" is kept verbatim; other labels are chunk-namespaced.
    assert labels == ["avatar", "chunk_0.speaker_1", "chunk_1.speaker_0", "avatar"]
    # chunk_idx retained and offsets applied.
    assert [seg["chunk_idx"] for seg in merged] == [0, 0, 1, 1]
    assert merged[2]["start"] == 100.0


def test_merge_keeps_labels_verbatim_without_known_label():
    chunk0 = _FakeChunkResponse([_FakeSegment("speaker_0", "a")])
    chunk1 = _FakeChunkResponse([_FakeSegment("speaker_0", "b")])
    merged = _merge_diarized_segments_from_chunks(
        [chunk0, chunk1], [0.0, 10.0], known_speaker_label=None
    )
    assert [seg["speaker"] for seg in merged] == ["speaker_0", "speaker_0"]
    assert [seg["chunk_idx"] for seg in merged] == [0, 1]


# --------------------------------------------------------------------------- #
# adjudicate_target_speaker_labels — application rules (model stubbed)
# --------------------------------------------------------------------------- #


def _stub_model(monkeypatch, attributions):
    """Stub init_model so ainvoke returns a canned attribution response."""

    class _Response:
        def __init__(self):
            self.attributions = [SimpleNamespace(**a) for a in attributions]

    class _Model:
        async def ainvoke(self, input):
            return _Response()

    monkeypatch.setattr(
        target_attribution, "init_model", lambda **kwargs: _Model(), raising=False
    )
    # init_model is imported lazily inside the function; patch the source module.
    import src.anubis.utils.model as model_module

    monkeypatch.setattr(model_module, "init_model", lambda **kwargs: _Model())


@pytest.mark.asyncio
async def test_adjudication_promotes_medium_and_high_not_low(monkeypatch):
    turns = [
        {"speaker": "avatar", "text": "confirmed target", "chunk_idx": 0},
        {"speaker": "chunk_0.speaker_1", "text": "long answer", "chunk_idx": 0},
        {"speaker": "chunk_1.speaker_0", "text": "maybe target", "chunk_idx": 1},
        {"speaker": "chunk_1.speaker_2", "text": "host question", "chunk_idx": 1},
    ]
    _stub_model(
        monkeypatch,
        [
            {
                "speaker_label": "chunk_0.speaker_1",
                "belongs_to_target": True,
                "confidence": "high",
                "evidence_summary": "role continuity",
            },
            {
                "speaker_label": "chunk_1.speaker_0",
                "belongs_to_target": True,
                "confidence": "low",
                "evidence_summary": "ambiguous",
            },
            {
                "speaker_label": "chunk_1.speaker_2",
                "belongs_to_target": False,
                "confidence": "high",
                "evidence_summary": "asks questions",
            },
        ],
    )
    context = SimpleNamespace(
        target_speaker_attribution_transcript_character_limit=100000
    )
    result = await target_attribution.adjudicate_target_speaker_labels(
        turns,
        reference_transcript_text="the quick fox",
        target_name="avatar",
        target_speaker_label="avatar",
        context=context,
    )
    # avatar confirmed by voice matcher; high-confidence label promoted;
    # low-confidence stays non-target; explicit False stays False.
    assert result == {
        "avatar": True,
        "chunk_0.speaker_1": True,
        "chunk_1.speaker_0": False,
        "chunk_1.speaker_2": False,
    }


@pytest.mark.asyncio
async def test_adjudication_result_applied_and_recoalesced(monkeypatch):
    """Promoted labels relabel to avatar and re-coalesce into long turns."""
    turns = [
        {"speaker": "avatar", "text": "A", "is_target": True, "chunk_idx": 0},
        {"speaker": "chunk_0.speaker_1", "text": "B", "is_target": False, "chunk_idx": 0},
        {"speaker": "chunk_1.speaker_0", "text": "C", "is_target": False, "chunk_idx": 1},
    ]
    attribution_map = {
        "avatar": True,
        "chunk_0.speaker_1": True,
        "chunk_1.speaker_0": False,
    }
    for turn in turns:
        if attribution_map.get(turn["speaker"]):
            turn["is_target"] = True
            turn["speaker"] = "avatar"
    coalesced = coalesce_segments_by_speaker(turns)
    # A and B merge under avatar; C stays separate as a non-target speaker.
    assert [t["speaker"] for t in coalesced] == ["avatar", "chunk_1.speaker_0"]
    assert coalesced[0]["text"] == "A B"
    assert coalesced[0]["is_target"] is True


@pytest.mark.asyncio
async def test_adjudication_returns_none_on_empty_response(monkeypatch):
    turns = [{"speaker": "chunk_0.speaker_0", "text": "x", "chunk_idx": 0}]
    _stub_model(monkeypatch, [])  # empty attributions -> None
    context = SimpleNamespace(
        target_speaker_attribution_transcript_character_limit=100000
    )
    result = await target_attribution.adjudicate_target_speaker_labels(
        turns,
        reference_transcript_text="",
        target_name="avatar",
        target_speaker_label="avatar",
        context=context,
    )
    assert result is None
