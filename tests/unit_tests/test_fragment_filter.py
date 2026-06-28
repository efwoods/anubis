"""Unit tests for the fragment filter (Quest 1 — stop creating fragment Documents).

Covers:
* ``classify_fragment_heuristic`` — junk / useful / borderline triage.
* ``is_useful_content`` — synchronous gate (drops only clear junk).
* ``strip_repeated_lines`` — running-header/footer boilerplate removal.
* ``filter_fragment_documents`` — heuristic-only and LLM-fallback paths.
"""

import pytest
from langchain_core.documents import Document

from src.subgraphs.process_media_graph.utils.fragment_filter import (
    classify_fragment_heuristic,
    filter_fragment_documents,
    is_useful_content,
    strip_repeated_lines,
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


def test_heuristic_short_real_line_is_borderline():
    # A genuine short line (a one-word answer) is not obvious junk; it is left
    # for the LLM fallback to decide.
    assert classify_fragment_heuristic("Yeah.") == "borderline"


def test_is_useful_content_drops_junk_keeps_borderline():
    assert is_useful_content("Page 13 of 10") is False
    assert is_useful_content("Yeah.") is True  # borderline kept by the sync gate
    assert is_useful_content("This is a complete and meaningful sentence.") is True


# --------------------------------------------------------------------------- #
# strip_repeated_lines
# --------------------------------------------------------------------------- #


def test_strip_repeated_lines_removes_running_headers():
    pages = [
        "ACME REPORT\nReal content one\nPage 1",
        "ACME REPORT\nReal content two\nPage 2",
        "ACME REPORT\nReal content three\nPage 3",
        "ACME REPORT\nReal content four\nPage 4",
    ]
    cleaned = strip_repeated_lines(pages)
    assert all("ACME REPORT" not in page for page in cleaned)
    assert "Real content one" in cleaned[0]


def test_strip_repeated_lines_noop_for_short_docs():
    pages = ["HEADER\nbody a", "HEADER\nbody b"]
    # Fewer than min_pages -> untouched.
    assert strip_repeated_lines(pages) == pages


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
async def test_filter_borderline_uses_llm_judge(monkeypatch):
    """Borderline chunks consult the LLM judge; a 'not useful' verdict drops them."""
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

    kept_line = "Love you"  # short -> borderline -> judge accepts
    dropped_line = "x y"  # short -> borderline -> judge rejects
    assert classify_fragment_heuristic(kept_line) == "borderline"
    assert classify_fragment_heuristic(dropped_line) == "borderline"

    docs = [Document(page_content=kept_line), Document(page_content=dropped_line)]
    kept = await filter_fragment_documents(docs, use_llm_fallback=True)
    contents = [d.page_content for d in kept]
    assert kept_line in contents
    assert dropped_line not in contents
