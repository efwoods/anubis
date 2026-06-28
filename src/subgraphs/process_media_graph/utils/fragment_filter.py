"""Drop useless fragment Documents before they reach the vectorstore/adapter pipeline.

Targets page numbers, running headers/footers, navigation text, and timestamps.

Two layers, cheapest first (cost-bounded):

1. ``classify_fragment_heuristic`` — a free regex + token-length + special-char
   ratio gate that returns ``"useful"`` / ``"junk"`` / ``"borderline"``.
2. ``filter_fragment_documents`` — drops ``junk``, keeps ``useful``, and for
   ``borderline`` chunks (short but not obviously junk — e.g. a real one-line
   quote vs a stray ``"12"``) consults the ``UsefulContentClassification`` LLM
   judge when ``use_llm_fallback`` is on. Judgements are cached by content hash
   within a single call to bound cost.

``strip_repeated_lines`` is a loader-agnostic cleaning pass that removes lines
recurring across many PDF pages (running headers/footers/page numbers) — the
root cause of fragments like ``"Page 13 of 10"``.
"""

import logging
import re
from hashlib import sha1
from typing import Dict, List, Literal

from langchain_core.documents import Document

from src.anubis.utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)

FragmentVerdict = Literal["useful", "junk", "borderline"]

# Lines/chunks that are pure page furniture. Matched against the STRIPPED text.
_JUNK_PATTERNS: List[re.Pattern] = [
    re.compile(r"^page\s+\d+\s*(of\s*\d+)?$", re.IGNORECASE),          # "Page 13 of 10"
    re.compile(r"^[-–—•·\s]*\d+[-–—•·\s]*$"),                            # "- 4 -", "12"
    re.compile(r"^\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}.*$"),                  # "11/5/25, 11:46 PM..."
    re.compile(r"^\d{1,2}:\d{2}\s*(am|pm)?$", re.IGNORECASE),           # "11:46 PM"
    re.compile(r"^[\s\W_]+$"),                                          # only punctuation/whitespace
    re.compile(r"^(©|\(c\)|copyright\b|all rights reserved\b)", re.IGNORECASE),
    re.compile(
        r"^(skip to (main )?content|share|subscribe|read more|menu|home|next|previous|"
        r"sign in|log in|accept( all)?( cookies)?|cookie(s)? (policy|settings))$",
        re.IGNORECASE,
    ),
]

# Below this token count a chunk is "short"; short-but-not-junk = borderline.
_DEFAULT_MIN_USEFUL_TOKENS = 4
# Above this fraction of non-alphanumeric, non-space characters, treat as junk.
_SPECIAL_CHAR_RATIO_LIMIT = 0.5


def _special_char_ratio(text: str) -> float:
    if not text:
        return 1.0
    non_word = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return non_word / len(text)


def classify_fragment_heuristic(
    text: str, min_tokens: int = _DEFAULT_MIN_USEFUL_TOKENS
) -> FragmentVerdict:
    """Free, synchronous triage of a chunk into useful / junk / borderline."""
    content = (text or "").strip()
    if not content:
        return "junk"
    if any(pattern.match(content) for pattern in _JUNK_PATTERNS):
        return "junk"
    if _special_char_ratio(content) > _SPECIAL_CHAR_RATIO_LIMIT:
        return "junk"
    if not any(c.isalpha() for c in content):
        return "junk"
    if count_tokens(content) >= max(1, min_tokens):
        return "useful"
    # Short but not obviously junk — a genuine one-liner (tweet/quote) or a stray
    # fragment. Let the LLM fallback decide; callers without a fallback keep it.
    return "borderline"


def is_useful_content(text: str, min_tokens: int = _DEFAULT_MIN_USEFUL_TOKENS) -> bool:
    """Return False only for clear junk; keep useful and borderline content.

    Synchronous gate used where an async LLM call is undesirable (e.g. the
    index-time safety net). Favors recall — borderline short lines are kept.
    """
    return classify_fragment_heuristic(text, min_tokens) != "junk"


async def _llm_is_useful(text: str, cache: Dict[str, bool]) -> bool:
    """Borderline-chunk judge via UsefulContentClassification, cached by hash."""
    key = sha1(text.strip().encode("utf-8")).hexdigest()
    if key in cache:
        return cache[key]
    try:
        # Lazy import keeps model/SDK off the cold-start path.
        from src.anubis.utils.classes.UsefulContentClassificationClass import (
            UsefulContentClassificationClass,
        )

        classifier = UsefulContentClassificationClass()
        response = await classifier.classify(text)
        verdict = bool(response.get("is_useful", True))
    except Exception as exc:  # pragma: no cover - never block ingestion on judge failure
        logger.warning("UsefulContent LLM judge failed (%s); keeping chunk", exc)
        verdict = True
    cache[key] = verdict
    return verdict


async def filter_fragment_documents(
    documents: List[Document],
    *,
    min_tokens: int = _DEFAULT_MIN_USEFUL_TOKENS,
    use_llm_fallback: bool = True,
) -> List[Document]:
    """Return only the Documents whose ``page_content`` is real content."""
    if not documents:
        return []
    kept: List[Document] = []
    cache: Dict[str, bool] = {}
    for document in documents:
        content = document.page_content or ""
        verdict = classify_fragment_heuristic(content, min_tokens)
        if verdict == "junk":
            logger.debug("fragment_filter dropped junk: %r", content[:60])
            continue
        if verdict == "useful":
            kept.append(document)
            continue
        # borderline
        if use_llm_fallback and await _llm_is_useful(content, cache):
            kept.append(document)
        elif use_llm_fallback:
            logger.debug("fragment_filter LLM dropped borderline: %r", content[:60])
        else:
            kept.append(document)
    if len(kept) != len(documents):
        logger.info(
            "fragment_filter kept %d/%d documents", len(kept), len(documents)
        )
    return kept


def strip_repeated_lines(
    page_texts: List[str], *, min_repeat_fraction: float = 0.5, min_pages: int = 3
) -> List[str]:
    """Remove running headers/footers/page-number lines repeated across pages.

    A line that appears (stripped) on at least ``min_repeat_fraction`` of pages is
    treated as page furniture and dropped from every page. Only applied when there
    are at least ``min_pages`` pages, so short documents are left untouched.
    """
    if len(page_texts) < min_pages:
        return page_texts

    from collections import Counter

    line_page_counts: Counter = Counter()
    for page in page_texts:
        unique_lines = {
            line.strip() for line in (page or "").splitlines() if line.strip()
        }
        for line in unique_lines:
            line_page_counts[line] += 1

    threshold = max(2, int(len(page_texts) * min_repeat_fraction))
    boilerplate = {
        line for line, count in line_page_counts.items() if count >= threshold
    }
    if not boilerplate:
        return page_texts

    logger.info("strip_repeated_lines removing %d boilerplate lines", len(boilerplate))
    cleaned: List[str] = []
    for page in page_texts:
        cleaned_lines = [
            line
            for line in (page or "").splitlines()
            if line.strip() not in boilerplate
        ]
        cleaned.append("\n".join(cleaned_lines))
    return cleaned
