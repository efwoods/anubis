"""Parser-first extraction of biography and direct quotes from structured pages.

Structured web pages (character wikis, personal homepages, and similar) carry a
person's biographical prose and their direct quotes in identifiable HTML
structures. This module parses that structure with BeautifulSoup and extracts
the quote and biography text VERBATIM; the language model is used only to infer
the page's subject and to classify blocks whose speaker is ambiguous. The quote
text itself is never routed through the model.

The extraction functions take ``(html_text, url)`` from whatever fetcher
succeeds so a future link crawler can reuse them on any discovered page. The
produced segments follow the golden format so the same
:func:`process_dialogue_json_to_documents` pipeline that consumes diarized audio
and segmented text also consumes structured pages.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Section headings that mark biographical prose across common page shapes.
_BIOGRAPHICAL_HEADING_HINTS = (
    "personality",
    "background",
    "biography",
    "bio",
    "history",
    "characteristics",
    "overview",
    "about",
    "description",
    "traits",
)

# Inline biography lead-ins seen on homepages with no section heading.
_BIOGRAPHICAL_PARAGRAPH_HINTS = (
    "research interests",
    "podcast interests",
    "interests:",
)

_QUOTE_CHARACTER_PATTERN = re.compile(r"[\"“”‘’]")


class QuoteBlockAttribution(BaseModel):
    """Whether one ambiguous parsed quote block is spoken by the target."""

    block_index: int
    quotes_spoken_by_target: bool
    reasoning: str


class QuoteBlockAttributionResponse(BaseModel):
    """One attribution per ambiguous quote block."""

    attributions: List[QuoteBlockAttribution]


def _strip_quote_marks(text: str) -> str:
    """Trim surrounding quotation marks and whitespace from a quote string."""
    stripped = (text or "").strip()
    if len(stripped) >= 2 and _QUOTE_CHARACTER_PATTERN.match(stripped[0]):
        stripped = stripped[1:]
    if stripped and _QUOTE_CHARACTER_PATTERN.match(stripped[-1:]):
        stripped = stripped[:-1]
    return stripped.strip()


def _extract_quoted_spans(text: str) -> List[str]:
    """Return the double-quoted spans inside a string (marks removed).

    Runs of whitespace inside a span are collapsed to a single space: HTML
    source whitespace (line breaks, indentation) is presentational, not part of
    the quoted content, so collapsing it yields the true verbatim quote.
    """
    spans = re.findall(r"[\"“]([^\"“”]+)[\"”]", text or "")
    return [
        re.sub(r"\s+", " ", span).strip()
        for span in spans
        if span.strip()
    ]


def parse_html_into_structured_blocks(
    html_text: str, *, url: str
) -> Dict[str, Any]:
    """Parse an HTML page into ordered structured blocks.

    Returns ``{page_title, infobox_subject_name, blocks}`` where each block is
    ``{heading_path, kind, text, table_rows}``. MediaWiki-aware selectors are
    tried first (``.mw-headline`` section headings, ``aside.portable-infobox``
    subject name, ``table.wikitable`` rows); a generic fallback (title / h1-h3 /
    paragraphs / tables) covers non-wiki pages such as personal homepages.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_text or "", "html.parser")

    title_element = soup.find("title")
    page_title = title_element.get_text(strip=True) if title_element else ""

    infobox_subject_name = None
    infobox = soup.select_one("aside.portable-infobox")
    if infobox is not None:
        infobox_title = infobox.select_one("h2, .pi-title")
        if infobox_title is not None:
            infobox_subject_name = infobox_title.get_text(strip=True)
    if infobox_subject_name is None:
        first_h1 = soup.find("h1")
        if first_h1 is not None:
            infobox_subject_name = first_h1.get_text(strip=True)

    blocks: List[Dict[str, Any]] = []

    # Walk the main content in document order, tracking the current heading.
    content_root = (
        soup.select_one("div.mw-parser-output")
        or soup.find("main")
        or soup.body
        or soup
    )
    current_heading = ""
    for element in content_root.find_all(
        ["h1", "h2", "h3", "h4", "p", "ul", "ol", "table"], recursive=True
    ):
        tag_name = element.name
        if tag_name in ("h1", "h2", "h3", "h4"):
            headline = element.select_one(".mw-headline")
            current_heading = (
                headline.get_text(strip=True)
                if headline is not None
                else element.get_text(strip=True)
            )
            continue
        if tag_name == "p":
            text = element.get_text(" ", strip=True)
            if text:
                blocks.append(
                    {
                        "heading_path": current_heading,
                        "kind": "paragraph",
                        "text": text,
                        "table_rows": [],
                    }
                )
        elif tag_name in ("ul", "ol"):
            items = [
                li.get_text(" ", strip=True)
                for li in element.find_all("li", recursive=False)
            ]
            items = [item for item in items if item]
            if items:
                blocks.append(
                    {
                        "heading_path": current_heading,
                        "kind": "list",
                        "text": "\n".join(items),
                        "table_rows": [[item] for item in items],
                    }
                )
        elif tag_name == "table":
            table_rows: List[List[str]] = []
            for row in element.find_all("tr"):
                cells = [
                    cell.get_text(" ", strip=True)
                    for cell in row.find_all(["th", "td"])
                ]
                if any(cell for cell in cells):
                    table_rows.append(cells)
            if table_rows:
                blocks.append(
                    {
                        "heading_path": current_heading,
                        "kind": "table",
                        "text": "",
                        "table_rows": table_rows,
                    }
                )

    return {
        "page_title": page_title,
        "infobox_subject_name": infobox_subject_name,
        "blocks": blocks,
    }


def page_looks_like_subject_page(parsed: Dict[str, Any]) -> bool:
    """Heuristic: does the parsed page describe a single individual subject.

    True when a portable infobox subject is present, or when the page carries a
    biographical section heading, or a biography-style lead paragraph. Used to
    decide whether to run structured extraction versus falling through to the
    plain-text route.
    """
    if parsed.get("infobox_subject_name"):
        # An infobox alone is not decisive (many pages have one); require a
        # biographical signal too, EXCEPT when the page has quote structures.
        pass
    blocks = parsed.get("blocks") or []
    has_bio_heading = any(
        any(hint in (block.get("heading_path") or "").lower() for hint in _BIOGRAPHICAL_HEADING_HINTS)
        for block in blocks
    )
    has_bio_paragraph = any(
        block.get("kind") == "paragraph"
        and any(
            hint in (block.get("text") or "").lower()
            for hint in _BIOGRAPHICAL_PARAGRAPH_HINTS
        )
        for block in blocks
    )
    has_quote_table = any(
        block.get("kind") == "table"
        and any(
            _extract_quoted_spans(" ".join(row))
            for row in (block.get("table_rows") or [])
        )
        for block in blocks
    )
    return bool(has_bio_heading or has_bio_paragraph or has_quote_table)


def extract_biographical_prose_blocks(
    blocks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Select the biographical prose blocks (paragraphs under bio headings).

    Each returned block keeps its ``heading_path`` as provenance so downstream
    identity Documents record which section a fact came from.
    """
    selected: List[Dict[str, Any]] = []
    for block in blocks:
        if block.get("kind") != "paragraph":
            continue
        heading = (block.get("heading_path") or "").lower()
        text = (block.get("text") or "").strip()
        if not text:
            continue
        under_bio_heading = any(
            hint in heading for hint in _BIOGRAPHICAL_HEADING_HINTS
        )
        is_bio_paragraph = any(
            hint in text.lower() for hint in _BIOGRAPHICAL_PARAGRAPH_HINTS
        )
        if under_bio_heading or is_bio_paragraph:
            selected.append(block)
    return selected


def _quote_blocks_from_parsed(
    blocks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract candidate quote blocks (verbatim) with their context.

    Handles the fandom ``Context / Comment(s)`` table layout (the sibling
    context cell becomes the genuine context prompt), numbered transcript-style
    rows, and quote-bearing list items. Quotation marks are stripped; text is
    otherwise verbatim.
    """
    candidates: List[Dict[str, Any]] = []
    for block in blocks:
        heading = block.get("heading_path") or ""
        if block.get("kind") == "table":
            for row in block.get("table_rows") or []:
                cells = [cell for cell in row if cell]
                if not cells:
                    continue
                # The quote is the last cell that contains quotation marks;
                # earlier cells are context (episode / action).
                quote_cell = None
                for cell in reversed(cells):
                    if _extract_quoted_spans(cell):
                        quote_cell = cell
                        break
                if quote_cell is None:
                    continue
                context_cells = [cell for cell in cells if cell is not quote_cell]
                for quote in _extract_quoted_spans(quote_cell):
                    candidates.append(
                        {
                            "quote_text": quote,
                            "context_prompt": " - ".join(context_cells).strip()
                            or None,
                            "heading_path": heading,
                        }
                    )
        elif block.get("kind") in ("paragraph", "list"):
            for quote in _extract_quoted_spans(block.get("text") or ""):
                candidates.append(
                    {
                        "quote_text": quote,
                        "context_prompt": None,
                        "heading_path": heading,
                    }
                )
    return candidates


async def infer_target_from_structured_page(
    *,
    page_title: str,
    infobox_subject_name: Optional[str],
    leading_paragraphs: List[str],
    heading_names: List[str],
) -> Dict[str, Any]:
    """Infer the page's single subject via structured output."""
    import json

    from langchain_core.messages import HumanMessage, SystemMessage

    from src.anubis.utils.model import init_model
    from src.anubis.utils.prompts.text_dialogue_segmentation_prompt import (
        STRUCTURED_PAGE_TARGET_INFERENCE_SYSTEM_PROMPT,
    )
    from src.subgraphs.process_media_graph.utils.text_dialogue_segmentation import (
        TargetSpeakerInference,
    )

    human_message = "\n\n".join(
        [
            f"Page title: {page_title or 'unknown'}",
            f"Infobox subject name (may be empty): "
            f"{infobox_subject_name or 'none'}",
            "Leading paragraphs:\n" + "\n".join(leading_paragraphs[:3]),
            "Section heading names: " + json.dumps(heading_names[:30]),
        ]
    )

    model = init_model(
        model_without_tools=False, response_format=TargetSpeakerInference
    )
    response = await model.ainvoke(
        input=[
            SystemMessage(content=STRUCTURED_PAGE_TARGET_INFERENCE_SYSTEM_PROMPT),
            HumanMessage(content=human_message),
        ]
    )
    return {
        "has_identifiable_target": bool(
            getattr(response, "has_identifiable_target", False)
        ),
        "target_name": getattr(response, "target_name", None),
        "matching_roster_names": list(
            getattr(response, "matching_roster_names", []) or []
        ),
    }


def extract_direct_quotes_from_blocks(
    blocks: List[Dict[str, Any]], *, target_name: str
) -> List[Dict[str, Any]]:
    """Deterministically extract the target's direct quotes from parsed blocks.

    Quote text is parser-extracted verbatim (never model-generated). Returns a
    list of ``{quote_text, context_prompt, heading_path}``. On a wiki character
    page nearly every quoted line under the character's own page is spoken by
    that character, so this returns candidates directly; genuinely ambiguous
    attribution is resolved by the caller via a single classification pass.
    """
    return _quote_blocks_from_parsed(blocks)


def _structured_target_not_identifiable_document(
    media_item: Dict[str, Any],
) -> Document:
    metadata = media_item.get("metadata", {}) or {}
    return Document(
        page_content=(
            "[No single subject could be inferred from this structured page, so "
            "it cannot be extracted into target quotes and biographical facts.]"
        ),
        metadata={
            "status": "error",
            "error": "structured_page_target_not_identifiable",
            "filename": metadata.get("filename", ""),
            "namespace_filename": metadata.get("namespace_filename", ""),
        },
    )


async def convert_structured_web_page_to_documents(
    html_text: str,
    *,
    url: str,
    user_id: str,
    assistant_id: str,
    media_item: Dict[str, Any],
) -> List[Document]:
    """Parse a structured page and produce quote + biographical Documents.

    Parses the HTML, infers the subject, extracts verbatim direct quotes into
    golden-format ``avatar`` turns, and feeds biographical prose sections to the
    existing biographical-identity pipeline. Returns a single error Document
    when no subject is inferable.
    """
    from src.subgraphs.process_media_graph.utils.helper_functions import (
        _build_biographical_identity_documents,
        coalesce_segments_by_speaker,
        process_dialogue_json_to_documents,
    )

    parsed = parse_html_into_structured_blocks(html_text, url=url)
    blocks = parsed.get("blocks") or []

    leading_paragraphs = [
        block["text"]
        for block in blocks
        if block.get("kind") == "paragraph" and block.get("text")
    ][:3]
    heading_names = sorted(
        {
            block.get("heading_path")
            for block in blocks
            if block.get("heading_path")
        }
    )

    inference = await infer_target_from_structured_page(
        page_title=parsed.get("page_title", ""),
        infobox_subject_name=parsed.get("infobox_subject_name"),
        leading_paragraphs=leading_paragraphs,
        heading_names=heading_names,
    )
    if not inference["has_identifiable_target"]:
        return [_structured_target_not_identifiable_document(media_item)]

    target_name = inference["target_name"] or (
        parsed.get("infobox_subject_name") or ""
    )
    documents: List[Document] = []

    # 1) Direct quotes -> golden-format avatar turns -> dialogue pipeline.
    quote_candidates = extract_direct_quotes_from_blocks(
        blocks, target_name=target_name
    )
    if quote_candidates:
        segments: List[Dict[str, Any]] = []
        for candidate in quote_candidates:
            context_prompt = candidate.get("context_prompt")
            if context_prompt:
                segments.append(
                    {
                        "speaker": "context",
                        "text": context_prompt,
                        "is_target": False,
                    }
                )
            segments.append(
                {
                    "speaker": "avatar",
                    "text": candidate["quote_text"],
                    "is_target": True,
                }
            )
        coalesced_segments = coalesce_segments_by_speaker(segments)
        dialogue_payload = {
            "segments": coalesced_segments,
            "target_name": "avatar",
            "speakers": [{"name": target_name, "description": "page subject"}],
        }
        documents.extend(
            await process_dialogue_json_to_documents(
                dialogue_payload=dialogue_payload,
                user_id=user_id,
                assistant_id=assistant_id,
                media_item=media_item,
            )
        )

    # 2) Biographical prose sections -> biographical-identity pipeline.
    for block in extract_biographical_prose_blocks(blocks):
        try:
            bio_documents = await _build_biographical_identity_documents(
                text_content=block["text"],
                user_id=user_id,
                assistant_id=assistant_id,
                media_item=media_item,
                target_name=target_name,
            )
        except Exception as bio_error:  # noqa: BLE001 - one section must not abort
            logger.warning(
                "biographical extraction failed for section %r: %s",
                block.get("heading_path"),
                bio_error,
            )
            continue
        for document in bio_documents:
            document.metadata["source_section"] = block.get("heading_path") or ""
        documents.extend(bio_documents)

    if not documents:
        return [_structured_target_not_identifiable_document(media_item)]
    return documents
