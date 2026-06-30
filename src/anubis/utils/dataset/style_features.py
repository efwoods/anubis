"""Shared stylometric feature extractor — the single source of truth.

This module computes a flat dictionary of **33 scalar stylometric features** for
one text. The SAME function is called by

* the production authenticity evaluator
  (:mod:`src.anubis.utils.dataset.authenticity_evaluator`), and
* the validation notebook (``style.ipynb``),

so the notebook genuinely exercises the production code path rather than a
parallel re-implementation that could drift.

Design constraints (from ``features/prompt_drafts/style/style.md``):

* **No spaCy** — part-of-speech information comes from ``nltk.pos_tag`` only.
  Features that would require a dependency parse (clause density, parse-tree
  depth, T-units) are intentionally omitted; in their place we use a
  gzip-compressibility proxy of the POS-tag stream as a syntactic-diversity
  signal, plus ``textstat`` readability indices.
* **No VADER / sentiment** — sentiment lives elsewhere (Go-Emotions on the
  reply); style is measured purely from form.
* Every feature returns a finite ``float`` where possible and ``nan`` on texts
  too short to support the metric. No exception ever propagates out of
  :func:`extract_style_features`.

The feature names are deliberately self-commenting (``mean_sentence_length_words``
rather than ``MLS``) so downstream profile JSON and evaluator reports read
without a glossary.
"""

from __future__ import annotations

import gzip
import html
import json
import math
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Canonical, ORDERED feature list. The order is the column order of every
# feature matrix and feature vector in the pipeline, so it must stay stable:
# the stored covariance / inverse-covariance matrices are indexed by it.
# ---------------------------------------------------------------------------
FEATURE_NAMES: List[str] = [
    # ── Lexical diversity (7) ──────────────────────────────────────────────
    "type_token_ratio",  # unique words / total words (length-biased)
    "moving_average_ttr",  # TTR averaged over a sliding window (length-robust)
    "mtld_lexical_diversity",  # Measure of Textual Lexical Diversity
    "hdd_lexical_diversity",  # Hypergeometric Distribution Diversity (HD-D)
    "maas_lexical_diversity",  # Maas index a^2 (log-curve fit of TTR vs length)
    "yule_characteristic_k",  # Yule's K — vocabulary repetition, length-stable
    "lexical_density_content_word_ratio",  # content words / all words
    # ── Part-of-speech density via nltk.pos_tag (8) ────────────────────────
    "noun_density",  # share of tokens tagged noun
    "verb_density",  # share of tokens tagged verb
    "adjective_density",  # share of tokens tagged adjective
    "adverb_density",  # share of tokens tagged adverb
    "pronoun_density",  # share of tokens tagged pronoun
    "preposition_density",  # share of tokens tagged preposition
    "noun_to_verb_ratio",  # nominal (high) vs verbal/conversational (low) style
    "pos_sequence_compressibility",  # gzip ratio of the POS-tag stream (template reuse proxy)
    # ── Sentence shape (4) ─────────────────────────────────────────────────
    "mean_sentence_length_words",  # average words per sentence
    "stdev_sentence_length_words",  # sentence-length variability (rhythm)
    "interrogative_sentence_ratio",  # share of sentences ending in '?'
    "exclamatory_sentence_ratio",  # share of sentences ending in '!'
    # ── Punctuation fingerprint, marks per 1,000 words (7) ─────────────────
    "comma_rate_per_1k",
    "semicolon_rate_per_1k",
    "colon_rate_per_1k",
    "dash_rate_per_1k",
    "ellipsis_rate_per_1k",
    "exclamation_rate_per_1k",
    "question_mark_rate_per_1k",
    # ── Surface / flow (3) ─────────────────────────────────────────────────
    "all_caps_word_ratio",  # SHOUTING / emphasis habit
    "words_per_paragraph",  # internet writing = short paragraphs
    "transition_word_rate_per_1k",  # logical-bridge words per 1k (however, therefore, …)
    # ── Readability composites via textstat (3) ────────────────────────────
    "flesch_kincaid_grade",  # words/sentence + syllables/word -> US grade
    "gunning_fog_index",  # sentence length + % complex words
    "smog_index",  # polysyllable-count grade estimate
    # ── Information theory (1) ─────────────────────────────────────────────
    "lexical_entropy_bits",  # Shannon entropy of the word distribution
]

assert len(FEATURE_NAMES) == 33, f"expected 33 features, found {len(FEATURE_NAMES)}"


# Human-legible display title per feature. Keyed by the snake_case FEATURE_NAMES
# so build_style_profile_str() can render `LEGIBLE[name]`. The titles are what the
# LLM sees in the <STYLE> block, so they spell out acronyms (MATTR, HD-D, SMOG)
# rather than leaving the raw variable name.
FEATURE_NAMES_HUMAN_LEGIBLE: Dict[str, str] = {
    # ── Lexical diversity (7) ──────────────────────────────────────────────
    "type_token_ratio": "Type-Token Ratio (TTR)",
    "moving_average_ttr": "Moving-Average Type-Token Ratio (MATTR)",
    "mtld_lexical_diversity": "Measure of Textual Lexical Diversity (MTLD)",
    "hdd_lexical_diversity": "Hypergeometric Distribution Diversity (HD-D)",
    "maas_lexical_diversity": "Maas Lexical Diversity Index (a²)",
    "yule_characteristic_k": "Yule's Characteristic K",
    "lexical_density_content_word_ratio": "Lexical Density (Content-Word Ratio)",
    # ── Part-of-speech density via nltk.pos_tag (8) ────────────────────────
    "noun_density": "Noun Density",
    "verb_density": "Verb Density",
    "adjective_density": "Adjective Density",
    "adverb_density": "Adverb Density",
    "pronoun_density": "Pronoun Density",
    "preposition_density": "Preposition Density",
    "noun_to_verb_ratio": "Noun-to-Verb Ratio",
    "pos_sequence_compressibility": "Part-of-Speech Sequence Diversity",
    # ── Sentence shape (4) ─────────────────────────────────────────────────
    "mean_sentence_length_words": "Mean Sentence Length (words)",
    "stdev_sentence_length_words": "Sentence-Length Variability (std dev, words)",
    "interrogative_sentence_ratio": "Question-Sentence Ratio",
    "exclamatory_sentence_ratio": "Exclamation-Sentence Ratio",
    # ── Punctuation fingerprint, marks per 1,000 words (7) ─────────────────
    "comma_rate_per_1k": "Commas per 1,000 Words",
    "semicolon_rate_per_1k": "Semicolons per 1,000 Words",
    "colon_rate_per_1k": "Colons per 1,000 Words",
    "dash_rate_per_1k": "Dashes per 1,000 Words",
    "ellipsis_rate_per_1k": "Ellipses per 1,000 Words",
    "exclamation_rate_per_1k": "Exclamation Marks per 1,000 Words",
    "question_mark_rate_per_1k": "Question Marks per 1,000 Words",
    # ── Surface / flow (3) ─────────────────────────────────────────────────
    "all_caps_word_ratio": "ALL-CAPS Word Ratio",
    "words_per_paragraph": "Words per Paragraph",
    "transition_word_rate_per_1k": "Transition Words per 1,000 Words",
    # ── Readability composites via textstat (3) ────────────────────────────
    "flesch_kincaid_grade": "Flesch-Kincaid Grade Level",
    "gunning_fog_index": "Gunning Fog Index",
    "smog_index": "SMOG Index",
    # ── Information theory (1) ─────────────────────────────────────────────
    "lexical_entropy_bits": "Lexical Entropy (bits)",
}

assert len(FEATURE_NAMES) == len(FEATURE_NAMES_HUMAN_LEGIBLE), (
    f"expected {len(FEATURE_NAMES)} features, found {len(FEATURE_NAMES_HUMAN_LEGIBLE)}"
)

# One-line plain-language description per feature, written for the LLM that reads
# the style profile. Each states the unit/range, the typical band, and crucially
# which DIRECTION means what — because the numbers alone are meaningless to the
# model (and several features are inverse: Maas and Yule's K go DOWN as vocabulary
# gets richer). Ranges are read straight off the computations in
# extract_style_features (POS/diversity shares are 0–1, punctuation is per-1k,
# Yule's K is scaled to 10⁴ words, etc.).
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    # ── Lexical diversity (7) ──────────────────────────────────────────────
    "type_token_ratio": "Unique words divided by total words. Ranges 0–1. Higher means more varied vocabulary, lower means more repetition. Length-biased (drops as a text grows), so it is most comparable at similar lengths.",
    "moving_average_ttr": "Type-token ratio averaged over a sliding ~50-word window. Ranges 0–1. Higher means richer vocabulary. Length-robust, so it stays comparable across short and long texts.",
    "mtld_lexical_diversity": "Mean length of word runs that stay above a 0.72 type-token threshold. Unbounded above; typically ~20–120 (most prose 40–100). Higher means sustained lexical variety, lower means vocabulary that repeats quickly.",
    "hdd_lexical_diversity": "Hypergeometric (HD-D) diversity: the type-token ratio a random fixed-size sample is expected to show. Ranges 0–1, typically ~0.70–0.90. Higher means more diverse word choice.",
    "maas_lexical_diversity": "Maas index a², a log-curve fit of type-token ratio versus length. Small positive value, typically ~0.0–0.2. Measure: LOWER means richer/more diverse vocabulary, HIGHER means more repetitive vocabulary (same words are used).",
    "yule_characteristic_k": "Yule's K, vocabulary repetition rate scaled to 10⁴ words; length-stable. Greater than 0, typically ~50–200. Measure: HIGHER means more repetitive vocabulary, LOWER means more varied vocabulary (a variety of words are used).",
    "lexical_density_content_word_ratio": "Content words (noun/verb/adjective/adverb) over all tagged tokens. Ranges 0–1, typically ~0.4–0.6. Higher means dense, informational, nominal writing; lower means more function words and a conversational feel.",
    # ── Part-of-speech density via nltk.pos_tag (8) ────────────────────────
    "noun_density": "Share of tokens tagged as nouns. Ranges 0–1, typically ~0.20–0.35. Higher means a nominal, topic-heavy style.",
    "verb_density": "Share of tokens tagged as verbs. Ranges 0–1, typically ~0.15–0.25. Higher means an active, event-driven style.",
    "adjective_density": "Share of tokens tagged as adjectives. Ranges 0–1, typically ~0.05–0.10. Higher means more descriptive, modifier-heavy writing.",
    "adverb_density": "Share of tokens tagged as adverbs. Ranges 0–1, typically ~0.03–0.08. Higher means more hedging/intensifying ('really', 'very', 'just').",
    "pronoun_density": "Share of tokens tagged as pronouns. Ranges 0–1, typically ~0.05–0.15. Higher means a personal, conversational voice (I/you/we).",
    "preposition_density": "Share of tokens tagged as prepositions (including 'to'). Ranges 0–1, typically ~0.10–0.15. Higher means more elaborated, phrase-stacked syntax.",
    "noun_to_verb_ratio": "Nouns divided by verbs (+1 smoothed so it stays finite). Greater than 0, typically ~1–3. Higher means a nominal, formal register; lower (near 1) means a verbal, conversational register.",
    "pos_sequence_compressibility": "Inverse gzip-compression ratio of the part-of-speech tag stream. Roughly ~1–3. Higher means more varied sentence shapes; lower means a few repeated grammatical templates (more formulaic). Very short texts read low from compression overhead.",
    # ── Sentence shape (4) ─────────────────────────────────────────────────
    "mean_sentence_length_words": "Average words per sentence. Greater than 0, typically ~10–25. Higher means longer, more complex sentences; lower means short, punchy ones.",
    "stdev_sentence_length_words": "Standard deviation of sentence length, in words. 0 or greater, typically ~4–15. Higher means rhythmic variety (mixing long and short sentences); 0 means uniform sentence length.",
    "interrogative_sentence_ratio": "Fraction of sentences ending in '?'. Ranges 0–1. Higher means a questioning, rhetorical, engaging style.",
    "exclamatory_sentence_ratio": "Fraction of sentences ending in '!'. Ranges 0–1. Higher means an emphatic, excited tone.",
    # ── Punctuation fingerprint, marks per 1,000 words (7) ─────────────────
    "comma_rate_per_1k": "Commas per 1,000 words. 0 or greater, typically ~40–80. Higher means more clause-chaining and parenthetical phrasing.",
    "semicolon_rate_per_1k": "Semicolons per 1,000 words. 0 or greater, usually ~0–5 (rare). Higher means deliberate, formal joining of independent clauses.",
    "colon_rate_per_1k": "Colons per 1,000 words. 0 or greater, typically ~0–10. Higher means frequent setups, lists, or explanatory pauses.",
    "dash_rate_per_1k": "Em dashes, en dashes, and hyphens per 1,000 words. 0 or greater, typically ~0–30. Higher means an interruptive, aside-heavy, informal rhythm.",
    "ellipsis_rate_per_1k": "Ellipsis characters ('…') per 1,000 words. 0 or greater, typically ~0–10. Higher means trailing-off, hesitant, or suspenseful phrasing. Counts only the single '…' glyph, not three dots.",
    "exclamation_rate_per_1k": "Exclamation marks per 1,000 words. 0 or greater. Higher means an emphatic, high-energy tone.",
    "question_mark_rate_per_1k": "Question marks per 1,000 words. 0 or greater. Higher means a more inquisitive, rhetorical style.",
    # ── Surface / flow (3) ─────────────────────────────────────────────────
    "all_caps_word_ratio": "Fraction of multi-letter tokens written in ALL CAPS. Ranges 0–1, usually near 0. Higher means a habit of SHOUTING or capitalized emphasis.",
    "words_per_paragraph": "Words divided by number of paragraphs (blank-line separated). Greater than 0. Higher means long, blocky paragraphs; lower means short, internet-style chunks.",
    "transition_word_rate_per_1k": "Logical-bridge words (however, therefore, moreover, …) per 1,000 words. 0 or greater, typically ~0–20. Higher means explicit, essayistic argument structure.",
    # ── Readability composites via textstat (3) ────────────────────────────
    "flesch_kincaid_grade": "Flesch-Kincaid US grade level from words-per-sentence and syllables-per-word. Typically ~1–15 (can go negative for very simple text). Higher means harder to read / a more educated register.",
    "gunning_fog_index": "Gunning Fog index: years of formal education a reader needs, from sentence length and the share of complex words. Typically ~6–17. Higher means denser, harder prose.",
    "smog_index": "SMOG grade estimate from the count of polysyllabic words. Typically ~6–14. Higher means more complex, multi-syllable vocabulary.",
    # ── Information theory (1) ─────────────────────────────────────────────
    "lexical_entropy_bits": "Shannon entropy of the word-frequency distribution, in bits. 0 or greater and grows with vocabulary size (~4–10+ bits common). Higher means less predictable, more varied word choice; lower means repetitive, predictable wording.",
}

assert len(FEATURE_NAMES) == len(FEATURE_DESCRIPTIONS), (
    f"expected {len(FEATURE_NAMES)} features, found {len(FEATURE_DESCRIPTIONS)}"
)


# ---------------------------------------------------------------------------
# Lexicons / regexes built once at import (cheap, no model downloads).
# ---------------------------------------------------------------------------

# Logical "bridge" words counted for transition density.
_TRANSITION_WORDS = frozenset(
    {
        "however",
        "therefore",
        "furthermore",
        "moreover",
        "nevertheless",
        "consequently",
        "meanwhile",
        "conversely",
        "thus",
        "hence",
        "accordingly",
        "additionally",
        "similarly",
        "instead",
        "otherwise",
        "subsequently",
    }
)

# Penn-Treebank tag prefixes -> coarse POS class. nltk.pos_tag emits PTB tags.
_NOUN_TAGS = ("NN",)  # NN, NNS, NNP, NNPS
_VERB_TAGS = ("VB",)  # VB, VBD, VBG, VBN, VBP, VBZ
_ADJECTIVE_TAGS = ("JJ",)  # JJ, JJR, JJS
_ADVERB_TAGS = ("RB",)  # RB, RBR, RBS
_PRONOUN_TAGS = ("PRP", "WP")  # PRP, PRP$, WP, WP$
_PREPOSITION_TAGS = ("IN", "TO")  # IN (prep/subord-conj), TO

_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")
_WORD_RE = re.compile(r"[A-Za-z']+")
_SENTENCE_FALLBACK_RE = re.compile(r"(?<=[.!?])\s+")

# Punctuation fingerprint: label -> the character(s) that count toward it.
_PUNCTUATION_MARKS: Dict[str, str] = {
    "comma_rate_per_1k": ",",
    "semicolon_rate_per_1k": ";",
    "colon_rate_per_1k": ":",
    "dash_rate_per_1k": "—–-",  # em dash, en dash, hyphen-minus
    "ellipsis_rate_per_1k": "…",
    "exclamation_rate_per_1k": "!",
    "question_mark_rate_per_1k": "?",
}


def _ensure_nltk_resources() -> None:
    """Lazily download the nltk data the extractor needs.

    Kept inside a function (not at import) so importing this module never pays a
    download/cold-start cost — consistent with the repo's lazy-import convention.
    Downloads are no-ops once the data is cached.
    """
    import nltk

    for resource, locator in (
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
        ("stopwords", "corpora/stopwords"),
    ):
        try:
            nltk.data.find(locator)
        except LookupError:
            nltk.download(resource, quiet=True)


def clean_text(text: str) -> str:
    """Light, stylometry-preserving normalisation.

    Ground-truth tweets carry HTML entities (``&amp;``), URLs and @mentions that
    would distort token and punctuation counts. We unescape entities and drop
    URLs + @mentions, but deliberately KEEP casing and punctuation because those
    ARE the stylistic signal we want to measure.
    """
    text = html.unescape(text or "")
    text = _URL_RE.sub("", text)
    text = _MENTION_RE.sub("", text)
    return text.strip()


def _word_tokens(text: str) -> List[str]:
    """Lowercased alphabetic tokens (apostrophes kept so ``it's`` stays whole)."""
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _sentences(text: str) -> List[str]:
    """Sentence split via nltk; regex fallback if the model is unavailable."""
    try:
        from nltk.tokenize import sent_tokenize

        sents = [s for s in sent_tokenize(text or "") if s.strip()]
    except Exception:
        sents = [
            s.strip() for s in _SENTENCE_FALLBACK_RE.split(text or "") if s.strip()
        ]
    return sents or ([text.strip()] if (text or "").strip() else [])


def _nan_features() -> Dict[str, float]:
    """All-NaN feature row for empty/degenerate input (callers impute later)."""
    return {name: math.nan for name in FEATURE_NAMES}


def extract_style_features(text: str) -> Dict[str, float]:
    """Return the 33 stylometric scalars for one document.

    The returned dict is keyed by :data:`FEATURE_NAMES`. Values are floats; a
    metric that cannot be computed on the given text yields ``nan`` rather than
    raising, so a single short document never breaks a batch.

    FUTURE DIRECTION asdf :
        character n-grams
        word n-grams
        function word frequencies
        punctuation frequencies
        lexical richness (vocab size): unique_words vs. total_words
        (FIND AND STORE KEY PHRASES) such as "you know" "got it." "what do ya mean?"
        average sentence length (words_per_sentence)
        average word length (characters per word)

    """
    _ensure_nltk_resources()

    cleaned = clean_text(text)
    if not cleaned:
        return _nan_features()

    from lexicalrichness import LexicalRichness
    from nltk.tokenize import word_tokenize

    features: Dict[str, float] = {}

    # Token views: `words` keeps punctuation as separate tokens (needed for
    # ALL-CAPS detection); `alpha_words` is the lowercased alphabetic stream
    # that most lexical metrics operate on.
    words = word_tokenize(cleaned)
    alpha_words = _word_tokens(cleaned)
    alpha_count = len(alpha_words) or 1  # guard divisions by zero
    per_thousand = 1000.0 / alpha_count
    sentences = _sentences(cleaned)
    sentence_count = len(sentences) or 1

    # ── A. LEXICAL DIVERSITY ───────────────────────────────────────────────
    # lexicalrichness implements the length-robust diversity indices; raw TTR is
    # length-biased and kept only as a baseline. Short texts make several of
    # these undefined, so each is guarded individually.
    lex = LexicalRichness(cleaned)
    features["type_token_ratio"] = _safe(lambda: lex.ttr)
    features["moving_average_ttr"] = _safe(
        lambda: lex.mattr(window_size=min(50, max(1, lex.words)))
    )
    features["mtld_lexical_diversity"] = _safe(lambda: lex.mtld(threshold=0.72))
    features["hdd_lexical_diversity"] = _safe(
        lambda: lex.hdd(draws=min(42, max(1, lex.words)))
    )
    features["maas_lexical_diversity"] = _safe(lambda: lex.Maas)

    # Yule's K = 10^4 * (Σ m^2 · V_m − N) / N^2, where V_m is the number of word
    # types occurring exactly m times. Length-stable measure of repetition.
    word_frequencies = Counter(alpha_words)
    frequency_spectrum = Counter(word_frequencies.values())
    yule_k = (
        1e4
        * (sum(m * m * v_m for m, v_m in frequency_spectrum.items()) - alpha_count)
        / (alpha_count * alpha_count)
    )
    features["yule_characteristic_k"] = float(yule_k)

    # ── B. PART-OF-SPEECH DENSITY (nltk.pos_tag, Penn Treebank) ────────────
    # One tagging pass feeds every POS feature plus lexical density and the
    # POS-stream compressibility proxy.
    pos_tags = [tag for _, tag in _safe_pos_tag(words)]
    pos_total = len(pos_tags) or 1
    noun_count = _count_tags(pos_tags, _NOUN_TAGS)
    verb_count = _count_tags(pos_tags, _VERB_TAGS)
    adjective_count = _count_tags(pos_tags, _ADJECTIVE_TAGS)
    adverb_count = _count_tags(pos_tags, _ADVERB_TAGS)
    pronoun_count = _count_tags(pos_tags, _PRONOUN_TAGS)
    preposition_count = _count_tags(pos_tags, _PREPOSITION_TAGS)

    features["noun_density"] = noun_count / pos_total
    features["verb_density"] = verb_count / pos_total
    features["adjective_density"] = adjective_count / pos_total
    features["adverb_density"] = adverb_count / pos_total
    features["pronoun_density"] = pronoun_count / pos_total
    features["preposition_density"] = preposition_count / pos_total
    # +1 smoothing keeps the ratio finite when a class is absent.
    features["noun_to_verb_ratio"] = (noun_count + 1) / (verb_count + 1)

    # Lexical density = content words (noun/verb/adj/adv) / all tagged tokens.
    content_word_count = noun_count + verb_count + adjective_count + adverb_count
    features["lexical_density_content_word_ratio"] = content_word_count / pos_total

    # POS-stream compressibility: LLM output reuses a few syntactic templates, so
    # its tag stream compresses more than varied human prose. We report 1/ratio
    # so HIGHER = more syntactically diverse (less template-y).
    pos_blob = " ".join(pos_tags).encode()
    compression_ratio = len(gzip.compress(pos_blob)) / (len(pos_blob) or 1)
    features["pos_sequence_compressibility"] = (
        1.0 / compression_ratio if compression_ratio else math.nan
    )

    # ── C. SENTENCE SHAPE ──────────────────────────────────────────────────
    sentence_lengths = [len(_word_tokens(s)) for s in sentences]
    mean_sentence_length = sum(sentence_lengths) / sentence_count
    features["mean_sentence_length_words"] = mean_sentence_length
    features["stdev_sentence_length_words"] = _population_stdev(
        sentence_lengths, mean_sentence_length
    )
    features["interrogative_sentence_ratio"] = (
        sum(1 for s in sentences if s.rstrip().endswith("?")) / sentence_count
    )
    features["exclamatory_sentence_ratio"] = (
        sum(1 for s in sentences if s.rstrip().endswith("!")) / sentence_count
    )

    # ── D. PUNCTUATION FINGERPRINT (marks per 1,000 words) ─────────────────
    for feature_name, characters in _PUNCTUATION_MARKS.items():
        features[feature_name] = (
            sum(cleaned.count(ch) for ch in characters) * per_thousand
        )

    # ── E. SURFACE / FLOW ──────────────────────────────────────────────────
    features["all_caps_word_ratio"] = sum(
        1 for w in words if w.isupper() and len(w) > 1
    ) / (len(words) or 1)
    paragraphs = [p for p in re.split(r"\n\s*\n", cleaned) if p.strip()] or [cleaned]
    features["words_per_paragraph"] = alpha_count / len(paragraphs)
    features["transition_word_rate_per_1k"] = (
        sum(1 for w in alpha_words if w in _TRANSITION_WORDS) * per_thousand
    )

    # ── F. READABILITY COMPOSITES (textstat) ───────────────────────────────
    import textstat

    features["flesch_kincaid_grade"] = _safe(
        lambda: float(textstat.flesch_kincaid_grade(cleaned))
    )
    features["gunning_fog_index"] = _safe(lambda: float(textstat.gunning_fog(cleaned)))
    features["smog_index"] = _safe(lambda: float(textstat.smog_index(cleaned)))

    # ── G. INFORMATION THEORY ──────────────────────────────────────────────
    # Shannon entropy of the unigram distribution, in bits: how unpredictable the
    # next word is. Computed from the same frequency table as Yule's K.
    entropy_bits = 0.0
    for count in word_frequencies.values():
        probability = count / alpha_count
        entropy_bits -= probability * math.log2(probability)
    features["lexical_entropy_bits"] = entropy_bits

    # Guarantee exactly the declared keys, in the declared order.
    return {name: float(features.get(name, math.nan)) for name in FEATURE_NAMES}


# ---------------------------------------------------------------------------
# Small numeric helpers (kept module-private and self-documenting).
# ---------------------------------------------------------------------------


def _safe(metric_fn: Callable[[], Any], default: float = 0.0) -> float:
    """Run a metric, swallowing short-text / zero-division errors.

    Returns ``default`` on the ``ValueError`` / ``ZeroDivisionError`` that
    lexicalrichness and textstat raise on tiny inputs.
    """
    try:
        value = metric_fn()
    except (ValueError, ZeroDivisionError, IndexError, KeyError):
        return default
    if value is None:
        return default
    value = float(value)
    return default if math.isnan(value) else value


def _safe_pos_tag(tokens: List[str]) -> List[Tuple[str, str]]:
    """``nltk.pos_tag`` with a defensive fallback to an empty tagging."""
    try:
        from nltk import pos_tag

        return list(pos_tag(tokens))
    except Exception:
        return []


def _count_tags(pos_tags: List[str], prefixes: Tuple[str, ...]) -> int:
    """Count tags whose Penn-Treebank label starts with any of ``prefixes``."""
    return sum(1 for tag in pos_tags if tag.startswith(prefixes))


def _population_stdev(values: Sequence[float], mean: float) -> float:
    """Return the population standard deviation (ddof=0); 0.0 if < 2 values."""
    if len(values) < 2:
        return 0.0
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler


def compute_mahalanobis_distance(synthetic_features, reference_feature_array):
    """Compute Mahalanobis distance between synthetic features and reference features of a dataset.

    Args:
        synthetic_features (np.array): shape is (n_observations, n_features); n_observations are one or more
        reference_feature_array (np.array): shape is (n_observations, n_features)
    """
    scaler = StandardScaler()

    corpus_scaled = scaler.fit_transform(reference_feature_array)

    corpus_scaled_mean = np.mean(corpus_scaled, axis=0)

    corpus_scaled_reg = LedoitWolf().fit(corpus_scaled)
    corpus_scaled_reg_cov_inv = np.linalg.inv(corpus_scaled_reg.covariance_)
    corpus_scaled_mean_series = pd.Series(corpus_scaled_mean)

    M_d_arr = []

    if synthetic_features.ndim > 1:
        for synthetic_feature_index in range(0, synthetic_features.shape[0]):
            synthetic_feature = synthetic_features[synthetic_feature_index, :]
            synth_scaled = scaler.transform(synthetic_feature.reshape(1, -1))
            synth_scaled_flattened = synth_scaled.flatten()
            synth_scaled_series = pd.Series(synth_scaled_flattened)

            M_d = np.dot(
                np.dot(
                    (synth_scaled_series - corpus_scaled_mean_series).T,
                    corpus_scaled_reg_cov_inv,
                ),
                (synth_scaled_series - corpus_scaled_mean_series),
            )
            M_d_arr.append(M_d)
    else:
        synthetic_feature = synthetic_features
        synth_scaled = scaler.transform(synthetic_feature.reshape(1, -1))
        synth_scaled_flattened = synth_scaled.flatten()
        synth_scaled_series = pd.Series(synth_scaled_flattened)
        M_d = np.dot(
            np.dot(
                (synth_scaled_series - corpus_scaled_mean_series).T,
                corpus_scaled_reg_cov_inv,
            ),
            (synth_scaled_series - corpus_scaled_mean_series),
        )
        M_d_arr.append(M_d)

    return M_d_arr


def compute_empirical_distribution(reference_dataset_arr):
    """Compute the empirical distribution using the leave-one-out method for comparison.

    Args:
        reference_dataset_arr (np.arr): expected shape (n_observations, n_features)
    """

    scaler = StandardScaler()
    M_d_squared_arr = []

    reference_dataset_df = pd.DataFrame(reference_dataset_arr)

    for i in range(0, reference_dataset_df.values.shape[0]):
        data = reference_dataset_df.values[i, :]
        corpus = reference_dataset_df.drop(reference_dataset_df.index[i])
        corpus_scaled = scaler.fit_transform(corpus.values)
        corpus_scaled_mean = np.mean(corpus_scaled, axis=0)
        corpus_scaled_reg = LedoitWolf().fit(corpus_scaled)
        corpus_scaled_reg_cov_inv = np.linalg.inv(corpus_scaled_reg.covariance_)

        data_scaled = scaler.transform(data.reshape(1, -1))
        data_scaled = data_scaled.flatten()

        data_scaled_series = pd.Series(data_scaled)
        corpus_scaled_mean_series = pd.Series(corpus_scaled_mean)

        M_d_squared_arr.append(
            np.dot(
                np.dot(
                    (data_scaled_series - corpus_scaled_mean_series).T,
                    corpus_scaled_reg_cov_inv,
                ),
                (data_scaled_series - corpus_scaled_mean_series),
            )
        )

    return M_d_squared_arr


# ---------------------------------------------------------------------------
# Ground-truth corpus persistence helpers.
#
# The "direct quote" corpus is persisted in the LangGraph store as a dict
# {document_id: [33 floats]} rather than a flat (n_docs, 33) array, so that an
# individual source document's rows can be pruned when that document is deleted.
# Wherever the corpus is consumed (Mahalanobis distance, IsolationForest fit,
# SHAP background) it is reconstructed into a single (n_docs, 33) array; those
# consumers are all set-based, so ROW ORDER is irrelevant.
# ---------------------------------------------------------------------------

# Store key (and second namespace element) for the per-document feature dict.
# The writer (process_text_to_document), the reader
# (graph._attach_analyzed_features), and the deleter (delete_avatar_documents)
# must all agree on this name.
GROUND_TRUTH_FEATURES_DICT_KEY = "ground_truth_text_features_by_doc_id_dict_str"


def serialize_features_by_doc_id(features_by_doc_id: Dict[str, Any]) -> str:
    """Serialize ``{document_id: 1-D 33-feature row}`` to a JSON string.

    Each row is coerced via ``.tolist()`` (numpy) or ``list`` so the whole dict
    is JSON-serializable for the store's ``{"value": <str>}`` envelope.
    """
    return json.dumps(
        {
            doc_id: (row.tolist() if hasattr(row, "tolist") else list(row))
            for doc_id, row in features_by_doc_id.items()
        }
    )


def deserialize_features_by_doc_id(features_by_doc_id_str: Any) -> Dict[str, Any]:
    """Inverse of :func:`serialize_features_by_doc_id`.

    Returns ``{}`` for any falsy input (key missing / never written) so callers
    can treat "no corpus yet" and "empty corpus" identically. Each value is
    rehydrated into a 1-D numpy array.
    """
    if not features_by_doc_id_str:
        return {}
    raw = json.loads(features_by_doc_id_str)
    return {doc_id: np.array(row) for doc_id, row in raw.items()}


def features_by_doc_id_to_arr(features_by_doc_id: Dict[str, Any]) -> Any:
    """Recombine per-document feature rows into one ``(n_docs, 33)`` array.

    Row order is irrelevant: every downstream consumer (empirical distribution,
    Mahalanobis covariance, IsolationForest) is a set statistic over the rows.
    Returns an empty ``(0, 33)`` array when the dict is empty.
    """
    if not features_by_doc_id:
        return np.empty((0, len(FEATURE_NAMES)))
    return np.vstack([np.asarray(row) for row in features_by_doc_id.values()])


# Cap on the number of corpus rows used to (re)calibrate the empirical threshold
# and IsolationForest. ``compute_empirical_distribution`` is leave-one-out: it
# refits StandardScaler + LedoitWolf + a matrix inverse once PER ROW, so it is
# O(n^2) in the corpus size. The corpus accumulates across every upload
# (``calibrate_ground_truth`` merges new rows into the persisted per-doc dict and
# recomputes over the UNION), so a large source like a ~68k-row tweet CSV makes
# calibration dominate processing time — and grow quadratically on each
# subsequent upload. The threshold and model are statistical calibrations, so a
# representative random subsample yields an equivalent Tukey fence and
# IsolationForest at bounded cost. Raise this for a tighter fit at quadratically
# higher calibration time.
MAX_CALIBRATION_ROWS = 1500


def recompute_ground_truth_artifacts(
    ground_truth_text_features_arr: Any,
) -> Tuple[str, str]:
    """Recalibrate the empirical threshold and IsolationForest from the corpus.

    Given the reconstructed ``(n_docs, 33)`` corpus array, returns
    ``(empirical_threshold_list_str, model_b64_pkl)`` serialized exactly as the
    store expects:

    * threshold — Tukey upper fence ``Q3 + 1.5*(Q3-Q1)`` of the leave-one-out
      squared-Mahalanobis empirical distribution, then ``json.dumps``'d.
    * model — an ``IsolationForest`` fit on the corpus, pickled then
      base64-encoded to a ``str`` (orjson in the store cannot serialize raw
      bytes).

    sklearn/pickle/base64 are imported lazily to keep module import cheap.
    """
    import base64
    import pickle

    from sklearn.ensemble import IsolationForest

    # Bound the O(n^2) leave-one-out calibration: above MAX_CALIBRATION_ROWS use a
    # deterministic random subsample (fixed seed so re-uploads of the same corpus
    # stay stable) instead of the full accumulated corpus.
    calibration_arr = ground_truth_text_features_arr
    if calibration_arr.shape[0] > MAX_CALIBRATION_ROWS:
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(
            calibration_arr.shape[0], size=MAX_CALIBRATION_ROWS, replace=False
        )
        calibration_arr = calibration_arr[sample_idx]

    # Recalibrate the empirical distribution and comparison threshold.
    ground_truth_empirical_arr = compute_empirical_distribution(calibration_arr)
    ground_truth_Q3 = np.percentile(ground_truth_empirical_arr, 75)
    ground_truth_Q1 = np.percentile(ground_truth_empirical_arr, 25)
    ground_truth_text_empirical_threshold = ground_truth_Q3 + 1.5 * (
        ground_truth_Q3 - ground_truth_Q1
    )

    # Recalibrate the Isolation Forest for prediction and explainable values.
    model = IsolationForest().fit(calibration_arr)

    ground_truth_text_empirical_threshold_list_str = json.dumps(
        ground_truth_text_empirical_threshold.tolist()
    )
    model_b64_pkl = base64.b64encode(pickle.dumps(model)).decode("utf-8")

    return ground_truth_text_empirical_threshold_list_str, model_b64_pkl


async def build_style_profile_str(ground_truth_text_features_arr) -> str:
    """Build the LLM interpretable string to allow for the
    median calculated features of the direct quotes of the
    target to influence the writing of the avatar.
    Allows the features to be LLM legible.
    """
    import numpy as np
    import pandas as pd

    ground_truth_text_features_median = np.array(
        list(pd.DataFrame(ground_truth_text_features_arr).median(axis=0).values)
    )

    # ground_truth_text_features_median expected shape: (n_features, )

    style_profile_str = ""

    idx = 0
    for name in FEATURE_NAMES:
        style_profile_str += f"{FEATURE_NAMES_HUMAN_LEGIBLE[name]}: {ground_truth_text_features_median[idx]}; Description: {FEATURE_DESCRIPTIONS[name]}\n\n"
        idx += 1

    return style_profile_str
