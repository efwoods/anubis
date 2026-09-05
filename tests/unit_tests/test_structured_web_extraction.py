"""Unit tests for Part B: parser-first structured web extraction.

Live fandom pages are bot-challenge protected, so these tests use trimmed HTML
snapshots that reproduce the load-bearing structures: a character-wiki page with
a biographical section and a ``Context / Comment(s)`` quote table, and a
personal-homepage page with an ``<h1>`` name and a "Research interests:" bio
paragraph. The parser and deterministic extractors are exercised offline; the
subject-inference model call is not invoked by these tests.
"""

import pytest

import src.subgraphs.process_media_graph.utils.structured_web_extraction as swe

WIKI_HTML = """
<html><head><title>Curie - Fallout Wiki</title></head><body>
<div class="mw-parser-output">
  <h1 class="page-header__title">Curie</h1>
  <h2><span class="mw-headline">Personality</span></h2>
  <p>Because she was programmed with the knowledge of the top scientific minds,
     Curie sees herself foremostly as a scientist and doctor.</p>
  <p>Having been isolated within Vault 81 for decades, Curie is entirely naive
     to the true nature of the wasteland.</p>
  <h2><span class="mw-headline">Notable quotes</span></h2>
  <table class="wikitable">
    <tr><th>Context</th><th>Comment(s)</th></tr>
    <tr><td>When Freedom Calls</td>
        <td>"It is good you were here. The offensive capabilities of this
            Deathclaw are quite advanced."</td></tr>
    <tr><td>Unlikely Valentine</td>
        <td>"A real private investigator. I hope he can find your little boy."</td></tr>
  </table>
</div></body></html>
"""

HOMEPAGE_HTML = """
<html><head><title>Lex Fridman</title></head><body>
  <h1>Lex Fridman</h1>
  <p>Research Scientist at MIT. Host of Lex Fridman Podcast.</p>
  <p>Research interests: Human-AI interaction, robotics, and machine learning.
     Podcast interests: History, philosophy, physics, and astronomy.</p>
  <h3>Research &amp; Publications</h3>
  <ul><li>Some paper (2020)</li></ul>
</body></html>
"""

ARTICLE_HTML = """
<html><head><title>Some News Story</title></head><body>
  <div class="mw-parser-output">
  <p>A generic news article with no biographical structure and no quote table.</p>
  <p>Just more prose about an event that happened somewhere.</p>
  </div>
</body></html>
"""


# --------------------------------------------------------------------------- #
# parse_html_into_structured_blocks
# --------------------------------------------------------------------------- #


def test_parse_wiki_headings_infobox_and_table():
    parsed = swe.parse_html_into_structured_blocks(
        WIKI_HTML, url="https://fallout.fandom.com/wiki/Curie"
    )
    assert parsed["page_title"] == "Curie - Fallout Wiki"
    # h1 fallback supplies the subject name when no portable infobox exists.
    assert parsed["infobox_subject_name"] == "Curie"
    kinds = [block["kind"] for block in parsed["blocks"]]
    assert "paragraph" in kinds and "table" in kinds
    headings = {block["heading_path"] for block in parsed["blocks"]}
    assert "Personality" in headings and "Notable quotes" in headings


def test_extract_biographical_prose_blocks_selects_personality():
    parsed = swe.parse_html_into_structured_blocks(WIKI_HTML, url="x")
    bio = swe.extract_biographical_prose_blocks(parsed["blocks"])
    assert len(bio) == 2
    assert all(block["heading_path"] == "Personality" for block in bio)
    assert "scientist and doctor" in bio[0]["text"]


def test_extract_direct_quotes_verbatim_with_context():
    parsed = swe.parse_html_into_structured_blocks(WIKI_HTML, url="x")
    quotes = swe.extract_direct_quotes_from_blocks(
        parsed["blocks"], target_name="Curie"
    )
    quote_texts = [q["quote_text"] for q in quotes]
    assert (
        "It is good you were here. The offensive capabilities of this "
        "Deathclaw are quite advanced." in quote_texts
    )
    # Quotation marks stripped; context cell carried as the prompt.
    freedom = next(q for q in quotes if q["context_prompt"] == "When Freedom Calls")
    assert not freedom["quote_text"].startswith('"')


def test_homepage_bio_paragraph_detected():
    parsed = swe.parse_html_into_structured_blocks(
        HOMEPAGE_HTML, url="https://lexfridman.com/"
    )
    assert parsed["infobox_subject_name"] == "Lex Fridman"
    assert swe.page_looks_like_subject_page(parsed) is True
    bio = swe.extract_biographical_prose_blocks(parsed["blocks"])
    assert any("Research interests" in block["text"] for block in bio)


def test_generic_article_is_not_a_subject_page():
    parsed = swe.parse_html_into_structured_blocks(ARTICLE_HTML, url="x")
    assert swe.page_looks_like_subject_page(parsed) is False


# --------------------------------------------------------------------------- #
# convert_structured_web_page_to_documents — orchestrator (model stubbed)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_convert_produces_quote_and_identity_documents(monkeypatch):
    async def _fake_infer(**kwargs):
        return {
            "has_identifiable_target": True,
            "target_name": "Curie",
            "matching_roster_names": ["Curie"],
        }

    async def _fake_bio(text_content, **kwargs):
        from langchain_core.documents import Document

        return [Document(page_content=f"IDENTITY::{text_content[:20]}", metadata={})]

    captured = {}

    async def _fake_dialogue(*, dialogue_payload, user_id, assistant_id, media_item):
        from langchain_core.documents import Document

        captured["payload"] = dialogue_payload
        return [Document(page_content="QUOTE_DOC", metadata={})]

    monkeypatch.setattr(swe, "infer_target_from_structured_page", _fake_infer)
    import src.subgraphs.process_media_graph.utils.helper_functions as hf

    monkeypatch.setattr(hf, "_build_biographical_identity_documents", _fake_bio)
    monkeypatch.setattr(hf, "process_dialogue_json_to_documents", _fake_dialogue)

    documents = await swe.convert_structured_web_page_to_documents(
        WIKI_HTML,
        url="https://fallout.fandom.com/wiki/Curie",
        user_id="u",
        assistant_id="a",
        media_item={"metadata": {"filename": "Curie", "namespace_filename": "ns"}},
    )
    page_contents = [d.page_content for d in documents]
    assert "QUOTE_DOC" in page_contents
    assert any(pc.startswith("IDENTITY::") for pc in page_contents)
    # The quote segments were packaged as golden-format avatar turns.
    avatar_turns = [
        seg for seg in captured["payload"]["segments"] if seg.get("speaker") == "avatar"
    ]
    assert avatar_turns and all(seg.get("is_target") for seg in avatar_turns)


# --------------------------------------------------------------------------- #
# Encyclopedic biography (Wikipedia-style article about one person)
# --------------------------------------------------------------------------- #

WIKIPEDIA_HTML = """
<html><head><title>Liv Boeree</title></head><body>
<div class="mw-parser-output">
  <table class="infobox"><tr><th>Nickname</th><td>"Bakes", "ODB"</td></tr></table>
  <p>Olivia "Liv" Boeree (born 18 July 1984) is a British science communicator
     and former professional poker player.</p>
  <div class="mw-heading mw-heading2"><h2 id="Early_life">Early life</h2></div>
  <p>Boeree was born in Kent and studied astrophysics at the University of
     Manchester.</p>
  <div class="mw-heading mw-heading2"><h2 id="Poker_career">Poker career</h2></div>
  <p>She won the "$10,000 Tag Team No-Limit Hold'em Championship" and was
     named "Europe's Leading Lady" at the European Poker Awards.</p>
  <p>Of the win she said, "It was the best day of my poker career."</p>
  <div class="mw-heading mw-heading2"><h2 id="Writing">Writing</h2></div>
  <p>She co-wrote the paper "Dissolving the Fermi Paradox".</p>
  <div class="mw-heading mw-heading2"><h2 id="References">References</h2></div>
  <ol class="references">
    <li>"Liv Boeree and Igor Kurganov Tie the Knot". PokerNews.</li>
    <li>"WSOP NEWS: LIV BOEREE WINS TAG TEAM EVENT". WSOP.com.</li>
  </ol>
  <div class="mw-heading mw-heading2"><h2 id="External_links">External links</h2></div>
  <ul><li>"Official website"</li></ul>
</div></body></html>
"""

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/Liv_Boeree"


def test_wikipedia_biography_is_an_encyclopedic_subject_page():
    parsed = swe.parse_html_into_structured_blocks(WIKIPEDIA_HTML, url=WIKIPEDIA_URL)
    assert parsed["url"] == WIKIPEDIA_URL
    assert swe.page_is_encyclopedic_biography(parsed) is True
    assert swe.page_looks_like_subject_page(parsed) is True
    # The heading signal works without the Wikimedia host as well.
    parsed_elsewhere = swe.parse_html_into_structured_blocks(
        WIKIPEDIA_HTML, url="https://example.org/people/liv"
    )
    assert swe.page_is_encyclopedic_biography(parsed_elsewhere) is True
    # A character wiki with only "Personality" is not encyclopedic.
    curie = swe.parse_html_into_structured_blocks(
        WIKI_HTML, url="https://fallout.fandom.com/wiki/Curie"
    )
    assert swe.page_is_encyclopedic_biography(curie) is False


def test_wikipedia_prose_covers_the_whole_body_except_boilerplate():
    parsed = swe.parse_html_into_structured_blocks(WIKIPEDIA_HTML, url=WIKIPEDIA_URL)
    # Without encyclopedic mode no section heading matches the wiki hints and
    # the whole biography would be lost.
    assert swe.extract_biographical_prose_blocks(parsed["blocks"]) == []
    bio = swe.extract_biographical_prose_blocks(parsed["blocks"], encyclopedic=True)
    headings = [block["heading_path"] for block in bio]
    assert headings == ["", "Early life", "Poker career", "Poker career", "Writing"]
    assert "science communicator" in bio[0]["text"]
    assert not any(swe._is_boilerplate_section(block["heading_path"]) for block in bio)


def test_wikipedia_quotes_skip_titles_nicknames_and_references():
    parsed = swe.parse_html_into_structured_blocks(WIKIPEDIA_HTML, url=WIKIPEDIA_URL)
    candidates = swe.extract_direct_quotes_from_blocks(
        parsed["blocks"], target_name="Liv Boeree", encyclopedic=True
    )
    quote_texts = [candidate["quote_text"] for candidate in candidates]
    # Reference titles, the infobox nickname cells, the lead nickname, and the
    # award / event names (no speech verb in the paragraph) are gone.
    assert "Bakes" not in quote_texts
    assert "Liv" not in quote_texts
    assert "Europe's Leading Lady" not in quote_texts
    assert not any("Tie the Knot" in text for text in quote_texts)
    assert "Official website" not in quote_texts
    # Paragraphs with a speech verb survive as ambiguous candidates that carry
    # their paragraph for the attribution pass.
    assert "It was the best day of my poker career." in quote_texts
    assert "Dissolving the Fermi Paradox" in quote_texts
    assert all(candidate.get("context_text") for candidate in candidates)


def test_boilerplate_section_detection():
    assert swe._is_boilerplate_section("References") is True
    assert swe._is_boilerplate_section("Notes and references") is True
    assert swe._is_boilerplate_section("External links") is True
    assert swe._is_boilerplate_section("Notable quotes") is False
    assert swe._is_boilerplate_section("") is False


def test_group_prose_blocks_into_sections_joins_and_splits():
    blocks = [
        {"heading_path": "", "text": "Lead one.", "kind": "paragraph"},
        {"heading_path": "", "text": "Lead two.", "kind": "paragraph"},
        {"heading_path": "Career", "text": "a" * 30, "kind": "paragraph"},
        {"heading_path": "Career", "text": "b" * 30, "kind": "paragraph"},
        {"heading_path": "Career", "text": "c" * 30, "kind": "paragraph"},
    ]
    groups = swe.group_prose_blocks_into_sections(blocks, character_limit=70)
    assert [group["heading_path"] for group in groups] == ["", "Career", "Career"]
    assert groups[0]["text"] == "Lead one.\n\nLead two."
    assert groups[1]["text"] == "a" * 30 + "\n\n" + "b" * 30
    assert groups[2]["text"] == "c" * 30


@pytest.mark.asyncio
async def test_attribute_quote_candidates_keeps_only_spoken_blocks(monkeypatch):
    class _Response:
        def __init__(self):
            self.attributions = [
                swe.QuoteBlockAttribution(
                    block_index=1, quotes_spoken_by_target=True, reasoning="said"
                ),
                swe.QuoteBlockAttribution(
                    block_index=2, quotes_spoken_by_target=False, reasoning="title"
                ),
            ]

    class _Model:
        async def ainvoke(self, input):
            return _Response()

    import src.anubis.utils.model as model_module

    monkeypatch.setattr(model_module, "init_model", lambda **kwargs: _Model())
    candidates = [
        {
            "quote_text": "kept table quote",
            "context_prompt": "ctx",
            "heading_path": "Quotes",
        },
        {
            "quote_text": "It was the best day.",
            "context_prompt": None,
            "heading_path": "Career",
            "block_index": 4,
            "context_text": 'She said, "It was the best day."',
        },
        {
            "quote_text": "Dissolving the Fermi Paradox",
            "context_prompt": None,
            "heading_path": "Writing",
            "block_index": 7,
            "context_text": 'She co-wrote "Dissolving the Fermi Paradox".',
        },
    ]
    kept = await swe.attribute_quote_candidates_to_target(
        target_name="Liv Boeree", candidates=candidates
    )
    assert [candidate["quote_text"] for candidate in kept] == [
        "kept table quote",
        "It was the best day.",
    ]


@pytest.mark.asyncio
async def test_convert_wikipedia_groups_sections_and_attributes_quotes(monkeypatch):
    async def _fake_infer(**kwargs):
        return {
            "has_identifiable_target": True,
            "target_name": "Liv Boeree",
            "matching_roster_names": ["Liv Boeree", "Liv"],
        }

    bio_calls = []

    async def _fake_bio(text_content, **kwargs):
        from langchain_core.documents import Document

        bio_calls.append(text_content)
        return [Document(page_content=f"IDENTITY::{text_content[:20]}", metadata={})]

    async def _fake_attribute(*, target_name, candidates):
        return [
            candidate
            for candidate in candidates
            if candidate["quote_text"].startswith("It was the best day")
        ]

    captured = {}

    async def _fake_dialogue(*, dialogue_payload, user_id, assistant_id, media_item):
        from langchain_core.documents import Document

        captured["payload"] = dialogue_payload
        return [Document(page_content="QUOTE_DOC", metadata={})]

    monkeypatch.setattr(swe, "infer_target_from_structured_page", _fake_infer)
    monkeypatch.setattr(swe, "attribute_quote_candidates_to_target", _fake_attribute)
    import src.subgraphs.process_media_graph.utils.helper_functions as hf

    monkeypatch.setattr(hf, "_build_biographical_identity_documents", _fake_bio)
    monkeypatch.setattr(hf, "process_dialogue_json_to_documents", _fake_dialogue)

    documents = await swe.convert_structured_web_page_to_documents(
        WIKIPEDIA_HTML,
        url=WIKIPEDIA_URL,
        user_id="u",
        assistant_id="a",
        media_item={
            "metadata": {"filename": WIKIPEDIA_URL, "namespace_filename": "ns"}
        },
    )
    # One fact-extraction call per section: lead, Early life, Poker career, Writing.
    assert len(bio_calls) == 4
    assert bio_calls[0].startswith('Olivia "Liv" Boeree')
    assert "best day of my poker career" in bio_calls[2]
    sections = [
        document.metadata["source_section"]
        for document in documents
        if document.page_content.startswith("IDENTITY::")
    ]
    assert sections == ["", "Early life", "Poker career", "Writing"]
    avatar_turns = [
        segment
        for segment in captured["payload"]["segments"]
        if segment.get("speaker") == "avatar"
    ]
    assert [turn["text"] for turn in avatar_turns] == [
        "It was the best day of my poker career."
    ]
