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
from src.subgraphs.process_media_graph.utils.helper_functions import (
    _build_target_quote_documents_from_dialogue,
    coalesce_segments_by_speaker,
)


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
