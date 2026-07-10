"""Unit tests for Part B: long-form text dialogue segmentation.

Covers the deterministic pieces (window splitting, narrator folding, target
relabeling) and the sequential window loop with the segmentation model stubbed.
"""

import pytest

import src.subgraphs.process_media_graph.utils.text_dialogue_segmentation as tds
from src.subgraphs.process_media_graph.utils.helper_functions import (
    coalesce_segments_by_speaker,
)

# --------------------------------------------------------------------------- #
# split_text_into_dialogue_windows
# --------------------------------------------------------------------------- #


def test_windows_never_split_mid_line_and_preserve_content():
    text = "line one\nline two\nline three\nline four\n"
    windows = tds.split_text_into_dialogue_windows(text, window_characters=20)
    assert "".join(windows) == text
    for window in windows:
        # No window ends in the middle of a line (each ends on a newline or is
        # the final window).
        assert window.endswith("\n") or window == windows[-1]


def test_single_oversized_line_emitted_alone():
    text = "x" * 100 + "\n" + "short\n"
    windows = tds.split_text_into_dialogue_windows(text, window_characters=10)
    assert "".join(windows) == text
    assert windows[0] == "x" * 100 + "\n"


# --------------------------------------------------------------------------- #
# fold_narrator_segments
# --------------------------------------------------------------------------- #


def test_fold_trailing_and_leading_narration():
    segments = [
        {"speaker": "narrator", "text": "[music]", "is_speech": False},
        {"speaker": "Denny", "text": "Agent Miranda?", "is_speech": True},
        {"speaker": "Miranda", "text": "Speaking.", "is_speech": True},
        {"speaker": "narrator", "text": "[line clicks]", "is_speech": False},
    ]
    folded = tds.fold_narrator_segments(segments)
    # Leading narration folds onto the following speaker; trailing onto the
    # preceding speaker.
    assert folded[0]["speaker"] == "Denny"
    assert folded[3]["speaker"] == "Miranda"


# --------------------------------------------------------------------------- #
# relabel_target_segments
# --------------------------------------------------------------------------- #


def test_relabel_target_by_alias_case_insensitive():
    segments = [
        {"speaker": "Denny", "text": "Q", "is_speech": True},
        {"speaker": "Miranda", "text": "A1", "is_speech": True},
        {"speaker": "Agent Miranda", "text": "A2", "is_speech": True},
    ]
    relabeled = tds.relabel_target_segments(
        segments, target_roster_names=["miranda", "agent miranda"]
    )
    assert relabeled[0]["speaker"] == "Denny" and relabeled[0]["is_target"] is False
    assert relabeled[1]["speaker"] == "avatar" and relabeled[1]["is_target"] is True
    assert relabeled[2]["speaker"] == "avatar" and relabeled[2]["is_target"] is True


# --------------------------------------------------------------------------- #
# segment_text_into_speaker_turns — sequential window loop (model stubbed)
# --------------------------------------------------------------------------- #


def _stub_window_segmenter(monkeypatch, results_by_window_prefix):
    """Stub segment_dialogue_window to return canned results keyed by content."""

    async def _fake_segment(window_text, *, roster, last_attributed_speaker, previous_turn_tail):
        for prefix, result in results_by_window_prefix.items():
            if window_text.lstrip().startswith(prefix):
                return result
        # Default: one narrator turn.
        return tds.WindowSegmentationResult(
            reasoning="",
            segments=[
                tds.SegmentedSpeakerTurn(
                    speaker="narrator", text=window_text.strip(), is_speech=False
                )
            ],
            updated_roster=[],
            final_attributed_speaker="narrator",
        )

    monkeypatch.setattr(tds, "segment_dialogue_window", _fake_segment)


@pytest.mark.asyncio
async def test_sequential_windows_carry_roster_and_merge_boundary(monkeypatch):
    text = "WINDOW_A content\nWINDOW_B content\n"
    window_a_result = tds.WindowSegmentationResult(
        reasoning="",
        segments=[
            tds.SegmentedSpeakerTurn(speaker="Denny", text="Q?", is_speech=True),
            tds.SegmentedSpeakerTurn(speaker="Miranda", text="Part1", is_speech=True),
        ],
        updated_roster=[tds.SpeakerRosterEntry(name="Miranda", description="agent")],
        final_attributed_speaker="Miranda",
    )
    window_b_result = tds.WindowSegmentationResult(
        reasoning="",
        segments=[
            tds.SegmentedSpeakerTurn(speaker="Miranda", text="Part2", is_speech=True),
        ],
        updated_roster=[],
        final_attributed_speaker="Miranda",
    )
    _stub_window_segmenter(
        monkeypatch,
        {"WINDOW_A": window_a_result, "WINDOW_B": window_b_result},
    )
    segments, roster = await tds.segment_text_into_speaker_turns(
        text, window_characters=20, max_characters=250000
    )
    assert [s["speaker"] for s in segments] == ["Denny", "Miranda", "Miranda"]
    assert any(entry["name"] == "Miranda" for entry in roster)
    # Miranda's two turns across the window boundary merge under coalescing.
    coalesced = coalesce_segments_by_speaker(segments)
    assert [t["speaker"] for t in coalesced] == ["Denny", "Miranda"]
    assert coalesced[1]["text"] == "Part1 Part2"


@pytest.mark.asyncio
async def test_convert_returns_error_document_when_no_target(monkeypatch):
    async def _fake_segment_turns(text, *, window_characters, max_characters):
        return (
            [{"speaker": "A", "text": "hello", "is_speech": True}],
            [{"name": "A", "description": ""}],
        )

    async def _fake_infer(*, roster, segments, classification_target_name, filename):
        return {
            "has_identifiable_target": False,
            "target_name": None,
            "matching_roster_names": [],
        }

    monkeypatch.setattr(tds, "segment_text_into_speaker_turns", _fake_segment_turns)
    monkeypatch.setattr(tds, "infer_target_speaker", _fake_infer)

    documents = await tds.convert_text_dialogue_to_documents(
        "some text",
        user_id="u",
        assistant_id="a",
        media_item={"metadata": {"filename": "f.txt", "namespace_filename": "ns"}},
        classification_target_name=None,
    )
    assert len(documents) == 1
    assert documents[0].metadata["error"] == "dialogue_text_target_not_identifiable"
