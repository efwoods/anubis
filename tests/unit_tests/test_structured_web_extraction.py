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
    quotes = swe.extract_direct_quotes_from_blocks(parsed["blocks"], target_name="Curie")
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
        seg
        for seg in captured["payload"]["segments"]
        if seg.get("speaker") == "avatar"
    ]
    assert avatar_turns and all(seg.get("is_target") for seg in avatar_turns)
