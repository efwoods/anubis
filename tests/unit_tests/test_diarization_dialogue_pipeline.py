"""Unit tests for the diarized-dialogue refactor.

Covers the deterministic pieces of the pipeline:

* ``coalesce_segments_by_speaker`` — turn coalescing by speaker label.
* ``_build_target_quote_documents_from_dialogue`` — per-target quote Documents
  carrying the preceding non-target turn as ``adapter_prompt``.
* ``build_adapter_and_langsmith_for_quotes`` — real prompt vs synthetic prompt.
* ``build_langsmith_for_conversation`` — Q&A pairs from a role-converted chat.

The synthetic-question generator (an LLM call) is stubbed so these tests stay
deterministic and offline.
"""

import pytest

import src.anubis.utils.dataset.formatting as formatting
from src.anubis.utils.dataset.formatting import (
    build_adapter_and_langsmith_for_quotes,
    build_langsmith_for_conversation,
)
import src.subgraphs.process_media_graph.utils.helper_functions as helper_functions
from src.subgraphs.process_media_graph.utils.helper_functions import (
    _build_adapter_dialogue_document,
    _build_target_quote_documents_from_dialogue,
    coalesce_segments_by_speaker,
    process_dialogue_json_to_documents,
)
from src.anubis.utils.utility import _select_dominant_speaker_segments
from langchain_core.documents import Document


@pytest.fixture
def stub_synthetic_questions(monkeypatch):
    """Replace the LLM-backed synthetic question generator with a stub."""

    async def _fake_create_question_list(messages):
        return [f"SYNTH::{m}" for m in messages]

    monkeypatch.setattr(
        formatting, "create_question_list", _fake_create_question_list
    )


# --------------------------------------------------------------------------- #
# coalesce_segments_by_speaker
# --------------------------------------------------------------------------- #


def test_coalesce_merges_consecutive_same_speaker():
    segments = [
        {"speaker": "A", "text": "hi", "start": 0.0, "end": 1.0, "is_target": False},
        {"speaker": "A", "text": "there", "start": 1.0, "end": 2.0, "is_target": False},
        {"speaker": "B", "text": "hello", "start": 2.0, "end": 3.0, "is_target": True},
        {"speaker": "A", "text": "bye", "start": 3.0, "end": 4.0, "is_target": False},
    ]
    turns = coalesce_segments_by_speaker(segments)
    assert [t["speaker"] for t in turns] == ["A", "B", "A"]
    assert turns[0]["text"] == "hi there"
    assert turns[0]["start"] == 0.0 and turns[0]["end"] == 2.0
    assert turns[1]["text"] == "hello" and turns[1]["is_target"] is True
    assert turns[2]["text"] == "bye"


def test_coalesce_is_target_ored_within_a_turn():
    segments = [
        {"speaker": "A", "text": "one", "is_target": False},
        {"speaker": "A", "text": "two", "is_target": True},
    ]
    turns = coalesce_segments_by_speaker(segments)
    assert len(turns) == 1
    assert turns[0]["is_target"] is True
    assert turns[0]["text"] == "one two"


def test_coalesce_single_speaker_collapses_to_one_turn():
    segments = [
        {"speaker": "A", "text": "a"},
        {"speaker": "A", "text": "b"},
        {"speaker": "A", "text": "c"},
    ]
    turns = coalesce_segments_by_speaker(segments)
    assert len(turns) == 1
    assert turns[0]["text"] == "a b c"


def test_coalesce_is_idempotent():
    segments = [
        {"speaker": "A", "text": "a", "start": 0, "end": 1, "is_target": False},
        {"speaker": "B", "text": "b", "start": 1, "end": 2, "is_target": True},
        {"speaker": "B", "text": "c", "start": 2, "end": 3, "is_target": True},
    ]
    once = coalesce_segments_by_speaker(segments)
    twice = coalesce_segments_by_speaker(once)
    assert once == twice


def test_coalesce_skips_empty_and_non_dict():
    segments = [
        {"speaker": "A", "text": "  "},
        "not-a-dict",
        {"speaker": "A", "text": "real"},
    ]
    turns = coalesce_segments_by_speaker(segments)
    assert len(turns) == 1
    assert turns[0]["text"] == "real"


# --------------------------------------------------------------------------- #
# _build_target_quote_documents_from_dialogue
# --------------------------------------------------------------------------- #


def _media_item():
    return {"metadata": {"filename": "talk.wav", "source": "talk.wav"}}


def test_target_quote_uses_preceding_nontarget_as_prompt():
    turns = [
        {"speaker": "A", "text": "What did you build?", "is_target": False},
        {"speaker": "B", "text": "I built a rocket.", "is_target": True},
    ]
    docs = _build_target_quote_documents_from_dialogue(
        turns,
        user_id="u",
        assistant_id="a",
        media_item=_media_item(),
        target_name="B",
    )
    assert len(docs) == 1
    md = docs[0].metadata
    assert docs[0].page_content == "I built a rocket."
    assert md["adapter_prompt"] == "What did you build?"
    assert md["classified_situation"] == "tweets_or_quotes"
    assert md["namespace"] == "quote"
    assert md["adapter_acceptable"] is True
    assert md["is_target"] is True


def test_target_quote_first_turn_has_no_prompt():
    turns = [
        {"speaker": "B", "text": "I lead with a statement.", "is_target": True},
        {"speaker": "A", "text": "Interesting.", "is_target": False},
    ]
    docs = _build_target_quote_documents_from_dialogue(
        turns,
        user_id="u",
        assistant_id="a",
        media_item=_media_item(),
        target_name="B",
    )
    assert len(docs) == 1
    assert docs[0].metadata["adapter_prompt"] is None


def test_target_quote_only_emits_target_turns():
    turns = [
        {"speaker": "A", "text": "q1", "is_target": False},
        {"speaker": "B", "text": "a1", "is_target": True},
        {"speaker": "A", "text": "q2", "is_target": False},
        {"speaker": "B", "text": "a2", "is_target": True},
    ]
    docs = _build_target_quote_documents_from_dialogue(
        turns,
        user_id="u",
        assistant_id="a",
        media_item=_media_item(),
        target_name="B",
    )
    assert [d.page_content for d in docs] == ["a1", "a2"]
    assert [d.metadata["adapter_prompt"] for d in docs] == ["q1", "q2"]
    assert [d.metadata["chunk_index"] for d in docs] == [0, 1]
    assert all(d.metadata["total_chunks"] == 2 for d in docs)


# --------------------------------------------------------------------------- #
# build_adapter_and_langsmith_for_quotes (prompt-aware)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_quotes_use_real_prompt_when_present(stub_synthetic_questions):
    quotes = ["I built a rocket.", "I like coffee."]
    prompts = ["What did you build?", None]
    adapter_rows, langsmith_rows = await build_adapter_and_langsmith_for_quotes(
        quotes=quotes,
        dataset_source_filename="talk.wav",
        prompts=prompts,
    )
    # Real prompt verbatim; missing prompt synthesized from the answer.
    assert adapter_rows[0]["messages"][0]["content"] == "What did you build?"
    assert adapter_rows[0]["messages"][1]["content"] == "I built a rocket."
    assert adapter_rows[1]["messages"][0]["content"] == "SYNTH::I like coffee."
    assert langsmith_rows[0]["inputs"]["question"] == "What did you build?"
    assert langsmith_rows[1]["inputs"]["question"] == "SYNTH::I like coffee."


@pytest.mark.asyncio
async def test_quotes_default_prompts_all_synthetic(stub_synthetic_questions):
    quotes = ["one", "two"]
    adapter_rows, _ = await build_adapter_and_langsmith_for_quotes(
        quotes=quotes,
        dataset_source_filename="src",
    )
    assert [r["messages"][0]["content"] for r in adapter_rows] == [
        "SYNTH::one",
        "SYNTH::two",
    ]


# --------------------------------------------------------------------------- #
# build_langsmith_for_conversation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_conversation_langsmith_pairs(stub_synthetic_questions):
    messages = [
        {"role": "user", "content": "What did you build?"},
        {"role": "assistant", "content": "A rocket."},
        {"role": "user", "content": "Why?"},
        {"role": "assistant", "content": "To go to Mars."},
    ]
    rows = await build_langsmith_for_conversation(messages, "talk.wav")
    assert [r["inputs"]["question"] for r in rows] == ["What did you build?", "Why?"]
    assert [r["outputs"]["answer"] for r in rows] == ["A rocket.", "To go to Mars."]


@pytest.mark.asyncio
async def test_conversation_langsmith_synthesizes_for_leading_assistant(
    stub_synthetic_questions,
):
    messages = [
        {"role": "assistant", "content": "I open with a statement."},
        {"role": "user", "content": "Tell me more."},
        {"role": "assistant", "content": "Sure, here it is."},
    ]
    rows = await build_langsmith_for_conversation(messages, "talk.wav")
    assert rows[0]["inputs"]["question"] == "SYNTH::I open with a statement."
    assert rows[1]["inputs"]["question"] == "Tell me more."


# --------------------------------------------------------------------------- #
# _build_adapter_dialogue_document — dataset shape is preserved (no speaker field)
# --------------------------------------------------------------------------- #


def test_adapter_messages_have_only_role_and_content():
    """Adapter / LangSmith dataset rows must keep the {role, content} shape.

    Per-speaker identity is preserved in identity Documents (covered below);
    the role-converted dataset format used by adapter trainers and the
    LangSmith eval builder is intentionally unchanged.
    """
    segments = [
        {"speaker": "A", "text": "q1", "is_target": False, "start": 0, "end": 1},
        {"speaker": "B", "text": "q2", "is_target": False, "start": 1, "end": 2},
        {"speaker": "T", "text": "a1", "is_target": True, "start": 2, "end": 3},
        {"speaker": "C", "text": "q3", "is_target": False, "start": 3, "end": 4},
    ]
    doc = _build_adapter_dialogue_document(
        segments,
        user_id="u",
        assistant_id="a",
        media_item=_media_item(),
        target_name="T",
    )
    assert doc is not None
    msgs = doc.metadata["messages"]
    for m in msgs:
        assert set(m.keys()) == {"role", "content"}, f"unexpected keys: {m.keys()}"
    # Distinct non-target speakers' content is coalesced into a single user
    # message, which is the legitimate adapter dataset shape.
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "q1\nq2"
    assert msgs[1] == {"role": "assistant", "content": "a1"}
    assert msgs[2] == {"role": "user", "content": "q3"}
    # The distinct speakers do still survive in document metadata.
    assert doc.metadata["speakers"] == ["A", "B", "C", "T"]


# --------------------------------------------------------------------------- #
# process_dialogue_json_to_documents — per-statement non-target Documents
# --------------------------------------------------------------------------- #


@pytest.fixture
def stub_biographical_identity(monkeypatch):
    """Stub ``_build_biographical_identity_documents`` to return one Document
    per call so per-statement emission becomes directly observable."""

    async def _fake(*, text_content, user_id, assistant_id, media_item, target_name=None):
        return [
            Document(
                page_content=text_content,
                metadata={
                    "namespace": "identity",
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "target_name": target_name,
                },
            )
        ]

    monkeypatch.setattr(
        helper_functions, "_build_biographical_identity_documents", _fake
    )


@pytest.mark.asyncio
async def test_each_nontarget_statement_emits_its_own_identity_document(
    stub_biographical_identity,
):
    """Two distinct non-target speakers must produce two identity Documents,
    each tagged with its own speaker label and timestamps — not one merged
    blob.
    """
    payload = {
        "segments": [
            {"speaker": "A", "text": "Grant built robots.", "is_target": False, "start": 0.0, "end": 1.5},
            {"speaker": "T", "text": "Yes, several.", "is_target": True, "start": 1.5, "end": 2.5},
            {"speaker": "B", "text": "Grant won championships.", "is_target": False, "start": 2.5, "end": 4.0},
        ],
        "target_name": "T",
    }
    docs = await process_dialogue_json_to_documents(
        payload, user_id="u", assistant_id="a", media_item=_media_item()
    )
    identity_docs = [d for d in docs if d.metadata.get("namespace") == "identity"]
    assert len(identity_docs) == 2, "expected one Document per non-target statement"
    by_speaker = {d.metadata["speaker"]: d for d in identity_docs}
    assert set(by_speaker.keys()) == {"A", "B"}
    assert by_speaker["A"].page_content == "Grant built robots."
    assert by_speaker["A"].metadata["start"] == 0.0
    assert by_speaker["A"].metadata["end"] == 1.5
    assert by_speaker["B"].page_content == "Grant won championships."
    assert by_speaker["B"].metadata["start"] == 2.5
    assert by_speaker["B"].metadata["end"] == 4.0


@pytest.mark.asyncio
async def test_empty_nontarget_statements_emit_no_documents(stub_biographical_identity):
    payload = {
        "segments": [
            {"speaker": "A", "text": "   ", "is_target": False, "start": 0, "end": 1},
            {"speaker": "T", "text": "Hi.", "is_target": True, "start": 1, "end": 2},
        ],
        "target_name": "T",
    }
    docs = await process_dialogue_json_to_documents(
        payload, user_id="u", assistant_id="a", media_item=_media_item()
    )
    assert [d for d in docs if d.metadata.get("namespace") == "identity"] == []


# --------------------------------------------------------------------------- #
# _select_dominant_speaker_segments — pure selection logic
# --------------------------------------------------------------------------- #


def test_select_dominant_speaker_filters_silence_only_winners():
    """A non-text-bearing segment (silence / music / applause) must not
    contribute to a speaker's total — even a long one. Otherwise a long
    crowd-applause segment labelled `crowd` would beat a real speaker.
    """
    segs = [
        {"speaker": "crowd", "text": "", "start": 0.0, "end": 60.0},
        {"speaker": "target", "text": "My name is Grant.", "start": 60.0, "end": 65.0},
        {"speaker": "host", "text": "Welcome.", "start": 65.0, "end": 67.0},
    ]
    result = _select_dominant_speaker_segments(segs)
    assert result is not None
    target_speaker, target_segs, totals, total = result
    assert target_speaker == "target"
    assert "crowd" not in totals  # silence-only speaker dropped entirely
    assert [s["speaker"] for s in target_segs] == ["target"]
    assert total == pytest.approx(5.0)


def test_select_dominant_speaker_single_speaker_returns_none():
    """One-speaker input means the caller should fall through to passthrough."""
    segs = [
        {"speaker": "A", "text": "x", "start": 0, "end": 1},
        {"speaker": "A", "text": "y", "start": 1, "end": 2},
    ]
    assert _select_dominant_speaker_segments(segs) is None


def test_select_dominant_speaker_no_text_returns_none():
    """If nothing has text the caller should fall through to passthrough."""
    segs = [
        {"speaker": "A", "text": "", "start": 0, "end": 60},
        {"speaker": "B", "text": "   ", "start": 60, "end": 120},
    ]
    assert _select_dominant_speaker_segments(segs) is None


def test_select_dominant_speaker_short_target_returns_none():
    """Combined target speech below the short-fallback threshold → None."""
    segs = [
        {"speaker": "A", "text": "a", "start": 0.0, "end": 0.3},
        {"speaker": "B", "text": "b", "start": 0.3, "end": 0.6},
    ]
    assert _select_dominant_speaker_segments(segs, short_fallback_s=1.0) is None


def test_select_dominant_speaker_tie_break_by_first_seen():
    """When two speakers tie, the first-seen label wins for determinism."""
    segs = [
        {"speaker": "B", "text": "x", "start": 0.0, "end": 5.0},
        {"speaker": "A", "text": "y", "start": 5.0, "end": 10.0},
    ]
    result = _select_dominant_speaker_segments(segs)
    assert result is not None
    assert result[0] == "B"


def test_select_dominant_speaker_ignores_invalid_timestamps():
    """Segments with missing or backwards timestamps are skipped entirely."""
    segs = [
        {"speaker": "A", "text": "ok", "start": 0.0, "end": 5.0},
        {"speaker": "B", "text": "bad", "start": 10.0, "end": 5.0},
        {"speaker": "C", "text": "no-end", "start": 0.0},
        {"speaker": "A", "text": "more", "start": 5.0, "end": 7.0},
        {"speaker": "D", "text": "short", "start": 7.0, "end": 9.0},
    ]
    result = _select_dominant_speaker_segments(segs)
    assert result is not None
    target_speaker, _segs, totals, _total = result
    assert target_speaker == "A"  # 7s vs D's 2s; B/C dropped as invalid
    assert set(totals.keys()) == {"A", "D"}
