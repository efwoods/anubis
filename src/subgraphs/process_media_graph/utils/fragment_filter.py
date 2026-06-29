"""Drop useless fragment Documents before they reach the vectorstore/adapter pipeline.

Targets page numbers, running headers/footers, navigation text, and timestamps.

Two layers, cheapest first (cost-bounded):

1. ``classify_fragment_heuristic`` — a free regex + token-length + special-char
   ratio gate that returns ``"useful"`` / ``"junk"`` / ``"indeterminant"``. It also
   drops fused HTML nav/title bars ("AboutMachine IntelligencePress") that plain
   PDF extraction welds into one line.
2. ``filter_fragment_documents`` — drops ``junk``, keeps ``useful``, and for
   ``indeterminant`` chunks (short but not obviously junk — e.g. a real one-line
   quote vs a stray ``"12"``) consults the ``UsefulContentClassification`` LLM
   judge when ``use_llm_fallback`` is on. Judgements are cached by content hash
   within a single call to bound cost.

"""
import logging
import re
from hashlib import sha1
from typing import Dict, List, Literal

from langchain_core.documents import Document

from src.anubis.utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)

FragmentVerdict = Literal["useful", "junk", "indeterminant"]

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

# Below this token count a chunk is "short"; short-but-not-junk = indeterminant.
_DEFAULT_MIN_USEFUL_TOKENS = 4
# Above this fraction of non-alphanumeric, non-space characters, treat as junk.
_SPECIAL_CHAR_RATIO_LIMIT = 0.5

# A short "title bar" scraped from an HTML page and rendered to PDF often fuses
# adjacent link labels with no separating space, leaving capital letters mid-run
# ("AboutMachine IntelligencePress" from the nav links About | Machine
# Intelligence | Press). A genuine human statement — even a very short one —
# separates its words with spaces, so a short line whose words are mostly such
# fused camel-runs is page chrome, not authored content. We require BOTH multiple
# fused boundaries and that most words are fused, so single CamelCase brand tokens
# ("iPhone", "GitHub") in real prose are not mistaken for nav furniture. This only
# catches the *fused* form; a normally-spaced title ("Machine Intelligence") is
# indistinguishable from a real short phrase by shape alone and is left to the LLM
# fallback / cross-page repetition signal instead.
_TITLE_MAX_WORDS = 8
# A weld boundary: a lowercase letter immediately followed by an uppercase letter
# that itself starts a new lowercase-continued word ("...workflow" + "These..."').
# Requiring the trailing ``[a-z]`` keeps acronyms intact — "SaaS"/"PaaS" end in a
# lone capital, so ``a→S`` has no following lowercase and is left alone — while
# still catching a Title-Case heading fused to a sentence.
_GLUED_WORD_BOUNDARY = re.compile(r"([a-z])([A-Z][a-z])")

# CamelCase brand/product tokens whose internal lowercase→uppercase boundary is
# intentional, not a weld. ``deweld_glued_text`` leaves a standalone token in this
# set intact so it is not shattered ("iPhone" must not become "i\nPhone"). Matched
# case-insensitively against the token's alphanumeric core (surrounding quotes /
# punctuation stripped). Welded forms ("iPhoneThese") are a rare accepted miss.
_CAMEL_BRAND_ALLOWLIST = frozenset(
    brand.lower()
    for brand in (
        "iPhone",
        "iPad",
        "iPod",
        "iOS",
        "iMac",
        "macOS",
        "iCloud",
        "YouTube",
        "GitHub",
        "GitLab",
        "PayPal",
        "LinkedIn",
        "eBay",
        "PowerPoint",
        "JavaScript",
        "TypeScript",
        "WordPress",
        "DeepMind",
        "OpenAI",
        "PlayStation",
        "WhatsApp",
        "DoorDash",
        "SoundCloud",
        "SpaceX",
        "TikTok",
    )
)

# A run of one-or-more whitespace characters; ``re.split`` with a capturing group
# keeps these separators so de-welding preserves the original line/spacing layout.
_WHITESPACE_RUN = re.compile(r"(\s+)")

# Sentence-ending punctuation immediately welded to the next capitalized word with
# no space ("bread alone.It is" / "bot-her?Ah"). Plain PDF extraction drops the
# space the glyph spacing implied; we restore it.
_SENTENCE_PUNCT_WELD = re.compile(r"([.?!])([A-Z])")

# Cosine similarity (against the page title) above which a line is treated as a
# title restatement and dropped outright; between the suspect floor and this, the
# line is ambiguous and the LLM safeguard decides; below the floor it is clearly
# unrelated and skips the LLM entirely (keeps per-document judge calls bounded).
_TITLE_SIMILARITY_DROP = 0.9
_TITLE_SIMILARITY_SUSPECT = 0.6


def _special_char_ratio(text: str) -> float:
    if not text:
        return 1.0
    non_word = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return non_word / len(text)


def _deweld_token(token: str) -> str:
    r"""Break a single whitespace-delimited token at intentional-looking welds.

    Inserts a ``\\n`` at every lowercase→uppercase boundary so a heading welded to
    the following sentence ("WorkflowThese") splits onto its own line — unless the
    token's alphanumeric core is a known CamelCase brand ("iPhone", "YouTube"),
    which is left whole.
    """
    core = token.strip("\"'“”‘’()[]{}.,;:!?-")
    if core.lower() in _CAMEL_BRAND_ALLOWLIST:
        return token
    return _GLUED_WORD_BOUNDARY.sub(r"\1\n\2", token)


def deweld_glued_text(text: str) -> str:
    """Repair PDF word-welds that ``pypdf`` plain extraction leaves behind.

    Two recoverable boundaries (see ``nodes.py`` PDF path):

    * **camelCase** — a Title-Case heading fused to a sentence ("WorkflowThese")
      is split with a newline so the heading becomes its own line and can be
      title-classified; known brand tokens are protected via
      ``_CAMEL_BRAND_ALLOWLIST``.
    * **sentence punctuation** — ``.?!`` welded to the next capital ("alone.It")
      gets its space back.

    All-lowercase welds ("yourdaily") are NOT recoverable by shape and are left
    untouched. Already-spaced text is returned unchanged. Existing whitespace
    (including newlines) is preserved by splitting on whitespace runs and
    rejoining with the original separators.
    """
    if not text:
        return text
    spaced = _SENTENCE_PUNCT_WELD.sub(r"\1 \2", text)
    # Keep the captured whitespace separators (odd indices) verbatim; de-weld only
    # the content pieces (even indices) so line/indent structure survives.
    pieces = _WHITESPACE_RUN.split(spaced)
    pieces[::2] = [_deweld_token(piece) for piece in pieces[::2]]
    return "".join(pieces)


def classify_fragment_heuristic(
    text: str, min_tokens: int = _DEFAULT_MIN_USEFUL_TOKENS, title: str | None = None
) -> FragmentVerdict:
    """Free, synchronous triage of a chunk into useful / junk / indeterminant."""
    content = (text or "").strip()
    if not content:
        return "junk"
    if any(pattern.match(content) for pattern in _JUNK_PATTERNS):
        return "junk"
    if _special_char_ratio(content) > _SPECIAL_CHAR_RATIO_LIMIT:
        return "junk"
    if not any(c.isalpha() for c in content):
        return "junk"
    # Running headers / cover titles repeat the document title on every page and
    # are page furniture, not content. When the caller knows the page title we
    # compare this line against it. Embedding similarity alone is brittle: a line
    # that merely repeats a brand/keyword from the title (e.g. a paragraph that
    # says "iPhone" several times) scores title-like even though it is real
    # content — the corner case the previous algorithm missed. So similarity only
    # decides the unambiguous ends; the ambiguous middle is handed to an LLM judge
    # that reasons about meaning and role rather than lexical overlap.
    if title:
        from src.anubis.utils.runtime_handles import get_sentence_embedder

        model = get_sentence_embedder()
        message_embedding, fact_embedding = model.encode(
            [content, title],
            convert_to_numpy=True,
        )
        similarity = float(model.similarity(message_embedding, fact_embedding)[0][0])
        if similarity > _TITLE_SIMILARITY_DROP:
            # Near-identical to the title -> page furniture.
            return "junk"
        if similarity > _TITLE_SIMILARITY_SUSPECT:
            # Overlaps the title enough to be suspicious but not a clear copy. Ask
            # the LLM safeguard (passing the actual title) to break the tie; keep
            # the line on any failure so we never drop real content on judge error.
            try:
                from src.anubis.utils.runtime_handles import (
                    get_title_fragment_classifier,
                )

                verdict = get_title_fragment_classifier().classify(content, title)
                if bool(verdict.get("is_title", False)):
                    return "junk"
            except Exception as exc:  # pragma: no cover - never block on judge failure
                logger.warning(
                    "Title-fragment LLM judge failed (%s); keeping line", exc
                )
        # Below the suspect band, or judged real content: fall through to the
        # normal length/usefulness gates.

    if count_tokens(content) >= max(1, min_tokens):
        return "useful"
    # Short but not obviously junk — a genuine one-liner (tweet/quote) or a stray
    # fragment. Let the LLM fallback decide; callers without a fallback keep it.

    # Greater than the minimum number of tokens, not classified as a title per the model, llm, nor regex. Inderterminant classification of text.
    return "indeterminant"


def is_useful_content(text: str, min_tokens: int = _DEFAULT_MIN_USEFUL_TOKENS) -> bool:
    """Return False only for clear junk; keep useful and indeterminant content.

    Synchronous gate used where an async LLM call is undesirable (e.g. the
    index-time safety net). Favors recall — indeterminant short lines are kept.
    """
    return classify_fragment_heuristic(text, min_tokens) != "junk"


async def _llm_is_useful(text: str, cache: Dict[str, bool]) -> bool:
    """indeterminant-chunk judge via UsefulContentClassification, cached by hash."""
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
        # indeterminant
        if use_llm_fallback and await _llm_is_useful(content, cache):
            kept.append(document)
        elif use_llm_fallback:
            logger.debug("fragment_filter LLM dropped indeterminant: %r", content[:60])
        else:
            kept.append(document)
    if len(kept) != len(documents):
        logger.info(
            "fragment_filter kept %d/%d documents", len(kept), len(documents)
        )
    return kept