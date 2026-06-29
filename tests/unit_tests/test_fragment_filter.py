"""Unit tests for the fragment filter (Quest 1 — stop creating fragment Documents).

Covers:
* ``classify_fragment_heuristic`` — junk / useful / indeterminant triage.
* ``is_useful_content`` — synchronous gate (drops only clear junk).
* ``filter_fragment_documents`` — heuristic-only and LLM-fallback paths.
"""

import pytest
from langchain_core.documents import Document

from src.subgraphs.process_media_graph.utils.fragment_filter import (
    classify_fragment_heuristic,
    deweld_glued_text,
    filter_fragment_documents,
    is_useful_content,
)

# --------------------------------------------------------------------------- #
# classify_fragment_heuristic
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "Page 13 of 10",
        "11/5/25, 11:46 PMPage 12 of 10",
        "12",
        "- 4 -",
        "11:46 PM",
        "© 2025 Acme Corp",
        "Copyright 2024",
        "Skip to content",
        "......",
        "   ",
    ],
)
def test_heuristic_flags_junk(text):
    assert classify_fragment_heuristic(text) == "junk"


@pytest.mark.parametrize(
    "text",
    [
        "Follow me, and I will make you fishers of men.",
        "It is written, Man shall not live by bread alone.",
        "She is the project director at Neuralink's Office of the CEO.",
    ],
)
def test_heuristic_keeps_useful(text):
    assert classify_fragment_heuristic(text) == "useful"


def test_heuristic_short_real_line_is_indeterminant():
    # A genuine short line (a one-word answer) is not obvious junk; it is left
    # for the LLM fallback to decide.
    assert classify_fragment_heuristic("Yeah.") == "indeterminant"


def test_is_useful_content_drops_junk_keeps_indeterminant():
    assert is_useful_content("Page 13 of 10") is False
    assert is_useful_content("Yeah.") is True  # indeterminant kept by the sync gate
    assert is_useful_content("This is a complete and meaningful sentence.") is True


# --------------------------------------------------------------------------- #
# deweld_glued_text
# --------------------------------------------------------------------------- #


def test_deweld_splits_heading_welded_to_paragraph():
    # The reported case: a Title-Case heading fused to the first body word.
    line = '"Magic Wands" Seamlessly Fix A WorkflowThese are SaaS tools.'
    out = deweld_glued_text(line)
    # The weld becomes a line break, so the heading and the sentence separate.
    assert "Workflow\nThese" in out
    assert "WorkflowThese" not in out
    # The heading sits on its own line once the paragraph is broken off.
    assert "A Workflow" in out.splitlines()[0]
    assert out.splitlines()[1].startswith("These are SaaS tools.")


@pytest.mark.parametrize(
    "welded,expected_break",
    [
        ("For The Physical WorldMachine intelligence", "World\nMachine"),
        ("To Help With Virtual TasksSometimes the best", "Tasks\nSometimes"),
    ],
)
def test_deweld_splits_camelcase_welds(welded, expected_break):
    assert expected_break in deweld_glued_text(welded)


def test_deweld_restores_sentence_punctuation_space():
    assert deweld_glued_text("bread alone.It is written") == "bread alone. It is written"
    assert deweld_glued_text("Why even bot-her?Ah") == "Why even bot-her? Ah"


@pytest.mark.parametrize(
    "brand_line",
    [
        "The new iPhone shipped today.",
        "She posted it on YouTube last night.",
        "The repo lives on GitHub now.",
        "He paid with PayPal at checkout.",
    ],
)
def test_deweld_preserves_known_brand_tokens(brand_line):
    # CamelCase brands must not be shattered ("iPhone" -> "i\nPhone").
    assert deweld_glued_text(brand_line) == brand_line


def test_deweld_leaves_clean_prose_unchanged():
    prose = "Follow me, and I will make you fishers of men.\nIt is written."
    assert deweld_glued_text(prose) == prose


def test_deweld_does_not_recover_lowercase_welds():
    # Documented limitation: all-lowercase welds are not shape-recoverable.
    assert deweld_glued_text("integrating into yourdaily workflow") == (
        "integrating into yourdaily workflow"
    )


def test_deweld_preserves_indentation_and_blank_lines():
    text = "  Heading\n\n  BodyText here"
    out = deweld_glued_text(text)
    # Leading whitespace and the blank line survive; only the weld is broken.
    assert out == "  Heading\n\n  Body\nText here"


# --------------------------------------------------------------------------- #
# classify_fragment_heuristic — title-aware branch
# --------------------------------------------------------------------------- #


class _FakeEmbedder:
    """Stand-in for the sentence embedder returning a fixed title similarity."""

    def __init__(self, similarity):
        self._similarity = similarity

    def encode(self, texts, convert_to_numpy=True):
        return [object(), object()]  # opaque; only similarity() is inspected

    def similarity(self, message_embedding, fact_embedding):
        return [[self._similarity]]


def _patch_embedder(monkeypatch, similarity):
    import src.anubis.utils.runtime_handles as runtime_handles

    monkeypatch.setattr(
        runtime_handles, "get_sentence_embedder", lambda: _FakeEmbedder(similarity)
    )


def test_title_high_similarity_drops_without_llm(monkeypatch):
    """A near-identical title restatement is dropped on similarity alone."""
    import src.anubis.utils.runtime_handles as runtime_handles

    _patch_embedder(monkeypatch, 0.95)
    monkeypatch.setattr(
        runtime_handles,
        "get_title_fragment_classifier",
        lambda: pytest.fail("LLM judge must not run above the drop threshold"),
    )
    assert (
        classify_fragment_heuristic("The iPhone Era", title="The iPhone Era") == "junk"
    )


def test_title_low_similarity_skips_llm_and_keeps(monkeypatch):
    """Below the suspect floor the line is clearly unrelated; no LLM call."""
    import src.anubis.utils.runtime_handles as runtime_handles

    _patch_embedder(monkeypatch, 0.1)
    monkeypatch.setattr(
        runtime_handles,
        "get_title_fragment_classifier",
        lambda: pytest.fail("LLM judge must not run below the suspect floor"),
    )
    verdict = classify_fragment_heuristic(
        "She joined the company in 2014 and led the design team.",
        title="The iPhone Era",
    )
    assert verdict == "useful"


def test_title_suspect_band_consults_llm(monkeypatch):
    """The ambiguous band hands off to the LLM, which can drop or keep the line."""
    import src.anubis.utils.runtime_handles as runtime_handles

    _patch_embedder(monkeypatch, 0.75)

    class _FakeJudge:
        def __init__(self, is_title):
            self._is_title = is_title

        def classify(self, fragment, title):
            return {"is_title": self._is_title}

    monkeypatch.setattr(
        runtime_handles, "get_title_fragment_classifier", lambda: _FakeJudge(True)
    )
    assert (
        classify_fragment_heuristic("iPhone iPhone iPhone", title="The iPhone Era")
        == "junk"
    )

    monkeypatch.setattr(
        runtime_handles, "get_title_fragment_classifier", lambda: _FakeJudge(False)
    )
    # Judge says it is real content -> falls through to the normal length gate.
    assert (
        classify_fragment_heuristic(
            "The iPhone changed how the team shipped products that year.",
            title="The iPhone Era",
        )
        == "useful"
    )


def test_title_judge_failure_keeps_line(monkeypatch):
    """A judge exception must never drop the line (favor recall)."""
    import src.anubis.utils.runtime_handles as runtime_handles

    _patch_embedder(monkeypatch, 0.75)

    class _BoomJudge:
        def classify(self, fragment, title):
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(
        runtime_handles, "get_title_fragment_classifier", lambda: _BoomJudge()
    )
    assert (
        classify_fragment_heuristic(
            "The iPhone changed how the team shipped products that year.",
            title="The iPhone Era",
        )
        == "useful"
    )



# --------------------------------------------------------------------------- #
# filter_fragment_documents
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_filter_drops_junk_without_llm():
    docs = [
        Document(page_content="Page 13 of 10"),
        Document(page_content="A genuine sentence with real meaning here."),
        Document(page_content="   "),
    ]
    kept = await filter_fragment_documents(docs, use_llm_fallback=False)
    assert len(kept) == 1
    assert "genuine sentence" in kept[0].page_content


@pytest.mark.asyncio
async def test_filter_indeterminant_uses_llm_judge(monkeypatch):
    """indeterminant chunks consult the LLM judge; a 'not useful' verdict drops them."""
    import src.anubis.utils.classes.UsefulContentClassificationClass as ucc_module

    class _FakeJudge:
        async def classify(self, text):
            # Accept the genuine short line, reject the stray fragment.
            return {"is_useful": "love" in text.lower()}

    # ``filter_fragment_documents`` imports the judge lazily from its defining
    # module, so patch the class there.
    monkeypatch.setattr(
        ucc_module, "UsefulContentClassificationClass", _FakeJudge
    )

    kept_line = "Love you"  # short -> indeterminant -> judge accepts
    dropped_line = "x y"  # short -> indeterminant -> judge rejects
    assert classify_fragment_heuristic(kept_line) == "indeterminant"
    assert classify_fragment_heuristic(dropped_line) == "indeterminant"

    docs = [Document(page_content=kept_line), Document(page_content=dropped_line)]
    kept = await filter_fragment_documents(docs, use_llm_fallback=True)
    contents = [d.page_content for d in kept]
    assert kept_line in contents
    assert dropped_line not in contents
