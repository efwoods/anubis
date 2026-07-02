"""Shared stylometric feature extractor — the single source of truth.

This module computes a flat dictionary of **28 scalar stylometric features** for
one text (feature-vector version 3; see :data:`STYLE_FEATURE_VECTOR_VERSION`).
The SAME function is called by the production authenticity evaluator
(``graph._attach_analyzed_features``), the per-avatar calibration
(``calibrate_ground_truth``), the bundled-baseline builder
(``data/build_baseline_features_arr.py``), and the validation notebook
(``style.ipynb``), so every path exercises the same code rather than a parallel
re-implementation that could drift.

Design constraints (from ``features/prompt_drafts/style/style.md``):

* **No spaCy** — part-of-speech information comes from ``nltk.pos_tag`` only.
  Features that would require a dependency parse (clause density, parse-tree
  depth, T-units) are intentionally omitted.
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

import html
import json
import logging
import math
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical, ORDERED feature list. The order is the column order of every
# feature matrix and feature vector in the pipeline, so it must stay stable:
# the stored covariance / inverse-covariance matrices are indexed by it.
# ---------------------------------------------------------------------------
FEATURE_NAMES: List[str] = [
    # ── Lexical diversity (3) ──────────────────────────────────────────────
    # Only the length-ROBUST diversity indices are kept. Raw TTR, Maas a², and
    # Yule's K were removed in vector version 3 as multicollinear: TTR is length-
    # biased and its two raw components (vocabulary size, total word count) were
    # also dropped, while Maas and Yule's K measure the same repetition signal
    # MATTR/MTLD/HD-D already carry length-robustly.
    "moving_average_ttr",                  # TTR averaged over a sliding window (length-robust)
    "mtld_lexical_diversity",              # Measure of Textual Lexical Diversity
    "hdd_lexical_diversity",               # Hypergeometric Distribution Diversity (HD-D)
    "lexical_density_content_word_ratio",  # content words / all words
    # ── Part-of-speech density via nltk.pos_tag (7) ────────────────────────
    # pos_sequence_compressibility was removed in v3 as redundant with
    # lexical_entropy_bits (both measure sequence predictability/variety).
    "noun_density",                        # share of tokens tagged noun
    "verb_density",                        # share of tokens tagged verb
    "adjective_density",                   # share of tokens tagged adjective
    "adverb_density",                      # share of tokens tagged adverb
    "pronoun_density",                     # share of tokens tagged pronoun
    "preposition_density",                 # share of tokens tagged preposition
    "noun_to_verb_ratio",                  # nominal (high) vs verbal/conversational (low) style
    # ── Sentence shape (4) ─────────────────────────────────────────────────
    "mean_sentence_length_words",          # average words per sentence
    "stdev_sentence_length_words",         # sentence-length variability (rhythm)
    "interrogative_sentence_ratio",        # share of sentences ending in '?'
    "exclamatory_sentence_ratio",          # share of sentences ending in '!'
    # ── Punctuation fingerprint, marks per word (7) ────────────────────────
    # Renamed from *_rate_per_1k in v4: the same counts normalized per WORD
    # (the per-1k value divided by 1,000) so every rate lives on a 0–1 scale.
    "comma_rate_per_word",
    "semicolon_rate_per_word",
    "colon_rate_per_word",
    "dash_rate_per_word",
    "ellipsis_rate_per_word",
    "exclamation_rate_per_word",
    "question_mark_rate_per_word",
    # ── Surface / flow (3) ─────────────────────────────────────────────────
    "all_caps_word_ratio",                 # SHOUTING / emphasis habit
    "words_per_paragraph",                 # internet writing = short paragraphs
    "transition_word_rate_per_word",       # logical-bridge words per word (however, therefore, …)
    # The readability composites (Flesch-Kincaid, Gunning Fog, SMOG) were removed
    # in v3: all three are deterministic functions of sentence length + syllable/
    # complex-word counts, so they are mutually collinear and add no signal beyond
    # the sentence-shape and word-length features already present.
    # ── Information theory (1) ─────────────────────────────────────────────
    "lexical_entropy_bits",                # Shannon entropy of the word distribution
    # ── Word & vocabulary shape (1) ────────────────────────────────────────
    # vocabulary_size_unique_words and total_word_count were removed in v3 (they
    # are the raw numerator/denominator of TTR — captured, and length-dependent).
    "average_word_length_characters",      # mean characters per word (orthographic length habit)
    # ── Signature key phrases (1) ──────────────────────────────────────────
    # Occurrences of the avatar's auto-discovered signature key-phrases per total
    # word. Unlike the character n-grams / function-word vectors (which were
    # capture-only and are now dropped), this collapses the key-phrase signal to a
    # single scalar so it can enter the Mahalanobis vector. The phrase set is
    # avatar-specific and passed into extract_style_features; see key_phrases.py.
    "key_phrase_rate",
]

# Bump whenever the composition or order of FEATURE_NAMES changes. The width of
# this vector is baked into persisted artifacts (the per-document ground-truth
# corpus rows, the bundled ChatGPT baseline matrix + IsolationForest, and any
# stored covariance matrices). Readers use `len(FEATURE_NAMES)` to detect and
# discard rows written under an older version — see deserialize_features_by_doc_id
# and the baseline staleness guards in graph._attach_analyzed_features /
# utility.load_baseline_features_explainer_model.
#   v1: the original 33 features.
#   v2: appended average_word_length_characters, vocabulary_size_unique_words,
#       total_word_count (width 33 -> 36).
#   v3: removed 9 multicollinear features (type_token_ratio, maas_lexical_diversity,
#       yule_characteristic_k, pos_sequence_compressibility, flesch_kincaid_grade,
#       gunning_fog_index, smog_index, vocabulary_size_unique_words,
#       total_word_count) and appended key_phrase_rate (width 36 -> 28).
#   v4: renamed the eight *_rate_per_1k features to *_rate_per_word and rescaled
#       their VALUES from marks-per-1,000-words to marks-per-word (divided by
#       1,000, a 0–1 scale). Width unchanged (28) — which is exactly why the
#       persisted corpus is now VERSION-TAGGED by serialize_features_by_doc_id:
#       a width check alone cannot tell v3 rows (per-1k scale) from v4 rows
#       (per-word scale), and mixing the two scales in one Mahalanobis /
#       IsolationForest corpus silently corrupts both comparisons.
STYLE_FEATURE_VECTOR_VERSION = 4

assert len(FEATURE_NAMES) == 28, f"expected 28 features, found {len(FEATURE_NAMES)}"


# Human-legible display title per feature. Keyed by the snake_case FEATURE_NAMES
# so build_style_profile_str() can render `LEGIBLE[name]`. The titles are what the
# LLM sees in the <STYLE> block, so they spell out acronyms (MATTR, HD-D, SMOG)
# rather than leaving the raw variable name.
FEATURE_NAMES_HUMAN_LEGIBLE: Dict[str, str] = {
    # ── Lexical diversity (3) ──────────────────────────────────────────────
    "moving_average_ttr": "Moving-Average Type-Token Ratio (MATTR)",
    "mtld_lexical_diversity": "Measure of Textual Lexical Diversity (MTLD)",
    "hdd_lexical_diversity": "Hypergeometric Distribution Diversity (HD-D)",
    "lexical_density_content_word_ratio": "Lexical Density (Content-Word Ratio)",
    # ── Part-of-speech density via nltk.pos_tag (7) ────────────────────────
    "noun_density": "Noun Density",
    "verb_density": "Verb Density",
    "adjective_density": "Adjective Density",
    "adverb_density": "Adverb Density",
    "pronoun_density": "Pronoun Density",
    "preposition_density": "Preposition Density",
    "noun_to_verb_ratio": "Noun-to-Verb Ratio",
    # ── Sentence shape (4) ─────────────────────────────────────────────────
    "mean_sentence_length_words": "Mean Sentence Length (words)",
    "stdev_sentence_length_words": "Sentence-Length Variability (std dev, words)",
    "interrogative_sentence_ratio": "Question-Sentence Ratio",
    "exclamatory_sentence_ratio": "Exclamation-Sentence Ratio",
    # ── Punctuation fingerprint, marks per word (7) ────────────────────────
    "comma_rate_per_word": "Commas per Word",
    "semicolon_rate_per_word": "Semicolons per Word",
    "colon_rate_per_word": "Colons per Word",
    "dash_rate_per_word": "Dashes per Word",
    "ellipsis_rate_per_word": "Ellipses per Word",
    "exclamation_rate_per_word": "Exclamation Marks per Word",
    "question_mark_rate_per_word": "Question Marks per Word",
    # ── Surface / flow (3) ─────────────────────────────────────────────────
    "all_caps_word_ratio": "ALL-CAPS Word Ratio",
    "words_per_paragraph": "Words per Paragraph",
    "transition_word_rate_per_word": "Transition Words per Word",
    # ── Information theory (1) ─────────────────────────────────────────────
    "lexical_entropy_bits": "Lexical Entropy (bits)",
    # ── Word & vocabulary shape (1) ────────────────────────────────────────
    "average_word_length_characters": "Average Word Length (characters)",
    # ── Signature key phrases (1) ──────────────────────────────────────────
    "key_phrase_rate": "Signature Key-Phrase Rate (per word)",
}

assert len(FEATURE_NAMES) == len(FEATURE_NAMES_HUMAN_LEGIBLE), f"expected {len(FEATURE_NAMES)} features, found {len(FEATURE_NAMES_HUMAN_LEGIBLE)}"

# One-line plain-language description per feature, written for the LLM that reads
# the style profile. Each states the unit/range, the typical band, and crucially
# which DIRECTION means what — because the numbers alone are meaningless to the
# model (and several features are inverse: Maas and Yule's K go DOWN as vocabulary
# gets richer). Ranges are read straight off the computations in
# extract_style_features (POS/diversity shares are 0–1, punctuation is per-word
# on a 0–1 scale, etc.).
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    # ── Lexical diversity (3) ──────────────────────────────────────────────
    "moving_average_ttr": "Type-token ratio averaged over a sliding ~50-word window. Ranges 0–1. Higher means richer vocabulary. Length-robust, so it stays comparable across short and long texts.",
    "mtld_lexical_diversity": "Mean length of word runs that stay above a 0.72 type-token threshold. Unbounded above; typically ~20–120 (most prose 40–100). Higher means sustained lexical variety, lower means vocabulary that repeats quickly.",
    "hdd_lexical_diversity": "Hypergeometric (HD-D) diversity: the type-token ratio a random fixed-size sample is expected to show. Ranges 0–1, typically ~0.70–0.90. Higher means more diverse word choice.",
    "lexical_density_content_word_ratio": "Content words (noun/verb/adjective/adverb) over all tagged tokens. Ranges 0–1, typically ~0.4–0.6. Higher means dense, informational, nominal writing; lower means more function words and a conversational feel.",
    # ── Part-of-speech density via nltk.pos_tag (7) ────────────────────────
    "noun_density": "Share of tokens tagged as nouns. Ranges 0–1, typically ~0.20–0.35. Higher means a nominal, topic-heavy style.",
    "verb_density": "Share of tokens tagged as verbs. Ranges 0–1, typically ~0.15–0.25. Higher means an active, event-driven style.",
    "adjective_density": "Share of tokens tagged as adjectives. Ranges 0–1, typically ~0.05–0.10. Higher means more descriptive, modifier-heavy writing.",
    "adverb_density": "Share of tokens tagged as adverbs. Ranges 0–1, typically ~0.03–0.08. Higher means more hedging/intensifying ('really', 'very', 'just').",
    "pronoun_density": "Share of tokens tagged as pronouns. Ranges 0–1, typically ~0.05–0.15. Higher means a personal, conversational voice (I/you/we).",
    "preposition_density": "Share of tokens tagged as prepositions (including 'to'). Ranges 0–1, typically ~0.10–0.15. Higher means more elaborated, phrase-stacked syntax.",
    "noun_to_verb_ratio": "Nouns divided by verbs (+1 smoothed so it stays finite). Greater than 0, typically ~1–3. Higher means a nominal, formal register; lower (near 1) means a verbal, conversational register.",
    # ── Sentence shape (4) ─────────────────────────────────────────────────
    "mean_sentence_length_words": "Average words per sentence. Greater than 0, typically ~10–25. Higher means longer, more complex sentences; lower means short, punchy ones.",
    "stdev_sentence_length_words": "Standard deviation of sentence length, in words. 0 or greater, typically ~4–15. Higher means rhythmic variety (mixing long and short sentences); 0 means uniform sentence length.",
    "interrogative_sentence_ratio": "Fraction of sentences ending in '?'. Ranges 0–1. Higher means a questioning, rhetorical, engaging style.",
    "exclamatory_sentence_ratio": "Fraction of sentences ending in '!'. Ranges 0–1. Higher means an emphatic, excited tone.",
    # ── Punctuation fingerprint, marks per word (7) ────────────────────────
    "comma_rate_per_word": "Commas per word (occurrences divided by total words). Ranges 0–1, typically ~0.04–0.08. Higher means more clause-chaining and parenthetical phrasing.",
    "semicolon_rate_per_word": "Semicolons per word (occurrences divided by total words). Ranges 0–1, usually ~0.0–0.005 (rare). Higher means deliberate, formal joining of independent clauses.",
    "colon_rate_per_word": "Colons per word (occurrences divided by total words). Ranges 0–1, typically ~0.0–0.01. Higher means frequent setups, lists, or explanatory pauses.",
    "dash_rate_per_word": "Em dashes, en dashes, and hyphens per word (occurrences divided by total words). Ranges 0–1, typically ~0.0–0.03. Higher means an interruptive, aside-heavy, informal rhythm.",
    "ellipsis_rate_per_word": "Ellipsis characters ('…') per word (occurrences divided by total words). Ranges 0–1, typically ~0.0–0.01. Higher means trailing-off, hesitant, or suspenseful phrasing. Counts only the single '…' glyph, not three dots.",
    "exclamation_rate_per_word": "Exclamation marks per word (occurrences divided by total words). Ranges 0–1. Higher means an emphatic, high-energy tone.",
    "question_mark_rate_per_word": "Question marks per word (occurrences divided by total words). Ranges 0–1. Higher means a more inquisitive, rhetorical style.",
    # ── Surface / flow (3) ─────────────────────────────────────────────────
    "all_caps_word_ratio": "Fraction of multi-letter tokens written in ALL CAPS. Ranges 0–1, usually near 0. Higher means a habit of SHOUTING or capitalized emphasis.",
    "words_per_paragraph": "Words divided by number of paragraphs (blank-line separated). Greater than 0. Higher means long, blocky paragraphs; lower means short, internet-style chunks.",
    "transition_word_rate_per_word": "Logical-bridge words (however, therefore, moreover, …) per word (occurrences divided by total words). Ranges 0–1, typically ~0.0–0.02. Higher means explicit, essayistic argument structure.",
    # ── Information theory (1) ─────────────────────────────────────────────
    "lexical_entropy_bits": "Shannon entropy of the word-frequency distribution, in bits. 0 or greater and grows with vocabulary size (~4–10+ bits common). Higher means less predictable, more varied word choice; lower means repetitive, predictable wording.",
    # ── Word & vocabulary shape (1) ────────────────────────────────────────
    "average_word_length_characters": "Mean number of characters per word (apostrophes counted, e.g. \"it's\" is 4). Greater than 0, typically ~4–5 for English prose. Higher means a preference for longer, often more formal or Latinate words; lower means shorter, plainer words.",
    # ── Signature key phrases (1) ──────────────────────────────────────────
    "key_phrase_rate": "Occurrences of the avatar's auto-discovered signature key-phrases (2–4 word recurring expressions like 'you know', 'got it') per total word in the text. 0 or greater, usually small (~0.0–0.1). Higher means the writing leans on the speaker's characteristic fixed phrasings; 0 means none of the signature phrases appear.",
}

assert len(FEATURE_NAMES) == len(FEATURE_DESCRIPTIONS), f"expected {len(FEATURE_NAMES)} features, found {len(FEATURE_DESCRIPTIONS)}"


# ---------------------------------------------------------------------------
# Lexicons / regexes built once at import (cheap, no model downloads).
# ---------------------------------------------------------------------------

# Logical "bridge" words counted for transition density.
_TRANSITION_WORDS = frozenset(
    {
        "however", "therefore", "furthermore", "moreover", "nevertheless",
        "consequently", "meanwhile", "conversely", "thus", "hence",
        "accordingly", "additionally", "similarly", "instead", "otherwise",
        "subsequently",
    }
)

# Penn-Treebank tag prefixes -> coarse POS class. nltk.pos_tag emits PTB tags.
_NOUN_TAGS = ("NN",)                       # NN, NNS, NNP, NNPS
_VERB_TAGS = ("VB",)                       # VB, VBD, VBG, VBN, VBP, VBZ
_ADJECTIVE_TAGS = ("JJ",)                  # JJ, JJR, JJS
_ADVERB_TAGS = ("RB",)                     # RB, RBR, RBS
_PRONOUN_TAGS = ("PRP", "WP")              # PRP, PRP$, WP, WP$
_PREPOSITION_TAGS = ("IN", "TO")           # IN (prep/subord-conj), TO

_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")
_WORD_RE = re.compile(r"[A-Za-z']+")
# Unicode curly-apostrophe family -> ASCII, so "don’t" stays ONE word token
# under the ASCII-only _WORD_RE instead of splitting into "don" + "t" (mirrors
# burrows_delta._APOSTROPHE_VARIANTS_RE — the two tokenisers must agree).
_APOSTROPHE_VARIANTS_RE = re.compile(r"[‘’ʼ`´]")
_SENTENCE_FALLBACK_RE = re.compile(r"(?<=[.!?])\s+")

# Punctuation fingerprint: label -> the character(s) that count toward it.
_PUNCTUATION_MARKS: Dict[str, str] = {
    "comma_rate_per_word": ",",
    "semicolon_rate_per_word": ";",
    "colon_rate_per_word": ":",
    "dash_rate_per_word": "—–-",   # em dash, en dash, hyphen-minus
    "ellipsis_rate_per_word": "…",
    "exclamation_rate_per_word": "!",
    "question_mark_rate_per_word": "?",
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
    text = _APOSTROPHE_VARIANTS_RE.sub("'", text)
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
        sents = [s.strip() for s in _SENTENCE_FALLBACK_RE.split(text or "") if s.strip()]
    return sents or ([text.strip()] if (text or "").strip() else [])


def _nan_features() -> Dict[str, float]:
    """All-NaN feature row for empty/degenerate input (callers impute later)."""
    return {name: math.nan for name in FEATURE_NAMES}

def extract_style_features(
    text: str,
    *,
    key_phrases: Sequence[str] | None = None,
    update_key_phrases_only: bool = False,
    features_dict: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Return the 28 stylometric scalars for one document.

    The returned dict is keyed by :data:`FEATURE_NAMES`. Values are floats; a
    metric that cannot be computed on the given text yields ``nan`` rather than
    raising, so a single short document never breaks a batch.

    Args:
        text: The document to fingerprint.
        key_phrases: The avatar's auto-discovered signature phrases (from
            :func:`src.anubis.utils.dataset.key_phrases.discover_key_phrases`).
            Used ONLY to compute ``key_phrase_rate`` (occurrences per total word).
            When ``None`` or empty — e.g. a text with no calibrated phrase set —
            ``key_phrase_rate`` is ``0.0``. This is the one avatar-relative
            feature in the vector; every other feature depends on ``text`` alone.
        update_key_phrases_only: Recompute ONLY ``key_phrase_rate`` against the
            given ``key_phrases``, reusing every other value from
            ``features_dict``. Used when the same text must be scored against a
            second phrase set (e.g. baseline phrases first, then the avatar's)
            without paying for a full re-extraction. Returns a NEW dict; the
            passed ``features_dict`` is not mutated.
        features_dict: Pre-computed features dictionary the update-only path
            copies its non-key-phrase values from. Required when
            ``update_key_phrases_only`` is True.

    Coverage of the scalar features requested in
    ``features/statistical_significance.md``:

    * average sentence length (words per sentence) — ``mean_sentence_length_words``.
    * average word length (characters per word) — ``average_word_length_characters``.
    * punctuation frequencies — the seven ``*_rate_per_word`` marks above.
    * signature key-phrase reliance — ``key_phrase_rate``.

    The character n-gram and function-word VECTORS that used to be captured in the
    nested :mod:`src.anubis.utils.dataset.stylistic_profile` profile were dropped;
    the key-phrase signal is now carried here as the single ``key_phrase_rate``
    scalar and, separately, as the prompt-injected signature-phrase list.
    """

    if update_key_phrases_only:
        if not features_dict:
            raise ValueError(
                "Must send a pre-computed features_dict from which to update key_phrase_rate."
            )
        # Copy (never mutate the caller's dict) and re-measure only the
        # key-phrase rate against the new phrase set below.
        features: Dict[str, float] = dict(features_dict)
        cleaned = clean_text(text)
    else:
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
        alpha_count = len(alpha_words) or 1            # guard divisions by zero
        per_word = 1.0 / alpha_count                   # marks-per-word (0–1 scale)
        sentences = _sentences(cleaned)
        sentence_count = len(sentences) or 1

        # ── A. LEXICAL DIVERSITY ───────────────────────────────────────────────
        # Only the length-ROBUST diversity indices are kept (v3). Raw TTR, Maas a²,
        # and Yule's K were removed as multicollinear with MATTR/MTLD/HD-D. Short
        # texts make several of these undefined, so each is guarded individually.
        lex = LexicalRichness(cleaned)
        features["moving_average_ttr"] = _safe(
            lambda: lex.mattr(window_size=min(50, max(1, lex.words)))
        )
        features["mtld_lexical_diversity"] = _safe(lambda: lex.mtld(threshold=0.72))
        features["hdd_lexical_diversity"] = _safe(
            lambda: lex.hdd(draws=min(42, max(1, lex.words)))
        )

        # Word-frequency table, reused below for lexical entropy. (Yule's K, which
        # also derived from this table, was removed in v3.)
        word_frequencies = Counter(alpha_words)

        # ── B. PART-OF-SPEECH DENSITY (nltk.pos_tag, Penn Treebank) ────────────
        # One tagging pass feeds every POS feature plus lexical density.
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

        # ── D. PUNCTUATION FINGERPRINT (marks per word, 0–1 scale) ─────────────
        for feature_name, characters in _PUNCTUATION_MARKS.items():
            features[feature_name] = (
                sum(cleaned.count(ch) for ch in characters) * per_word
            )

        # ── E. SURFACE / FLOW ──────────────────────────────────────────────────
        features["all_caps_word_ratio"] = (
            sum(1 for w in words if w.isupper() and len(w) > 1) / (len(words) or 1)
        )
        paragraphs = [p for p in re.split(r"\n\s*\n", cleaned) if p.strip()] or [cleaned]
        features["words_per_paragraph"] = alpha_count / len(paragraphs)
        features["transition_word_rate_per_word"] = (
            sum(1 for w in alpha_words if w in _TRANSITION_WORDS) * per_word
        )

        # (Readability composites — Flesch-Kincaid, Gunning Fog, SMOG — were removed
        # in v3 as mutually collinear functions of sentence length + syllable counts.)

        # ── F. INFORMATION THEORY ──────────────────────────────────────────────
        # Shannon entropy of the unigram distribution, in bits: how unpredictable the
        # next word is. Computed from the word-frequency table built above.
        entropy_bits = 0.0
        for count in word_frequencies.values():
            probability = count / alpha_count
            entropy_bits -= probability * math.log2(probability)
        features["lexical_entropy_bits"] = entropy_bits

        # ── G. WORD SHAPE ──────────────────────────────────────────────────────
        # `total_words` is the TRUE token count (len(alpha_words)); `alpha_count`
        # above was floored to 1 only to guard divisions, so it must not be reused
        # here. Average word length divides total characters by that true count and
        # is NaN when there are no word tokens (all-punctuation input). (The raw
        # vocabulary-size and total-word-count features were removed in v3 as the
        # length-dependent components of TTR.)
        total_words = len(alpha_words)
        total_characters = sum(len(word) for word in alpha_words)
        features["average_word_length_characters"] = (
            total_characters / total_words if total_words else math.nan
        )

    # ── H. SIGNATURE KEY-PHRASE RATE ───────────────────────────────────────
    # Occurrences of the avatar's signature phrases per total word. The phrase set
    # is avatar-specific (passed in); with no set the rate is 0.0. Delegated to
    # key_phrases so the tokenisation matches how the phrases were discovered.
    from src.anubis.utils.dataset.key_phrases import key_phrase_occurrence_rate

    features["key_phrase_rate"] = key_phrase_occurrence_rate(cleaned, key_phrases)

    # Guarantee exactly the declared keys, in the declared order.
    return {name: float(features.get(name, math.nan)) for name in FEATURE_NAMES}


# ---------------------------------------------------------------------------
# Small numeric helpers (kept module-private and self-documenting).
# ---------------------------------------------------------------------------


def _safe(metric_fn: Callable[[], Any], default: float = 0.0) -> float:
    """Run a metric, swallowing short-text / zero-division errors.

    Returns ``default`` on the ``ValueError`` / ``ZeroDivisionError`` that
    lexicalrichness raises on tiny inputs.
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
            synth_scaled = scaler.transform(synthetic_feature.reshape(1,-1))
            synth_scaled_flattened = synth_scaled.flatten()
            synth_scaled_series = pd.Series(synth_scaled_flattened)

            M_d = np.dot(np.dot((synth_scaled_series - corpus_scaled_mean_series).T, corpus_scaled_reg_cov_inv), (synth_scaled_series - corpus_scaled_mean_series))
            M_d_arr.append(M_d)
    else:
        synthetic_feature = synthetic_features
        synth_scaled = scaler.transform(synthetic_feature.reshape(1,-1))
        synth_scaled_flattened = synth_scaled.flatten()
        synth_scaled_series = pd.Series(synth_scaled_flattened)
        M_d = np.dot(np.dot((synth_scaled_series - corpus_scaled_mean_series).T, corpus_scaled_reg_cov_inv), (synth_scaled_series - corpus_scaled_mean_series))
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
        
        M_d_squared_arr.append(np.dot(np.dot((data_scaled_series - corpus_scaled_mean_series).T, corpus_scaled_reg_cov_inv), (data_scaled_series - corpus_scaled_mean_series)))

    return M_d_squared_arr


# ---------------------------------------------------------------------------
# Ground-truth corpus persistence helpers.
#
# The "direct quote" corpus is persisted in the LangGraph store as a dict
# {document_id: [len(FEATURE_NAMES) floats]} rather than a flat (n_docs, F) array,
# so that an individual source document's rows can be pruned when that document is
# deleted. Wherever the corpus is consumed (Mahalanobis distance, IsolationForest
# fit, SHAP background) it is reconstructed into a single (n_docs, F) array; those
# consumers are all set-based, so ROW ORDER is irrelevant. F is the current vector
# width (see STYLE_FEATURE_VECTOR_VERSION); rows stored at an older width are
# dropped on read by deserialize_features_by_doc_id.
# ---------------------------------------------------------------------------

# Store key (and final namespace element) for the per-document feature dict,
# stored owner-scoped at (user_id, assistant_id, GROUND_TRUTH_FEATURES_DICT_KEY).
# The writer (calibrate_ground_truth), the reader
# (graph._attach_analyzed_features), and the deleter (delete_avatar_documents)
# must all agree on this name.
GROUND_TRUTH_FEATURES_DICT_KEY = "ground_truth_text_features_by_doc_id_dict_str"


# ---------------------------------------------------------------------------
# Bundled ChatGPT-baseline artifacts (the "unmodified LLM" cloud).
#
# These are regenerated by data/build_baseline_features_arr.py whenever the
# feature vector width changes (see STYLE_FEATURE_VECTOR_VERSION). Both readers
# (graph._attach_analyzed_features and utility.load_baseline_features_explainer_model)
# cache the array/model in the LangGraph store on first use. On an EXISTING
# deployment that store still holds the previous-width artifacts, so both readers
# call baseline_feature_array_is_current() and, when it is stale, reload from the
# freshly-bundled .npy/.pkl and overwrite the store — the deployment self-heals
# without a manual store wipe. Paths are cwd-relative (the app runs from repo root).
# ---------------------------------------------------------------------------
BASELINE_FEATURES_ARR_PATH = "src/anubis/utils/dataset/baseline_features_arr.npy"
BASELINE_FEATURES_MODEL_PATH = "src/anubis/utils/dataset/baseline_features_model_b64.pkl"
# Pre-built SHAP KernelExplainer over the baseline IsolationForest, so the runtime
# loads it instead of rebuilding a KernelExplainer (kmeans + repeated model.predict)
# on first use. Regenerated alongside the model by build_baseline_features_arr.py.
BASELINE_FEATURES_EXPLAINER_PATH = "src/anubis/utils/dataset/baseline_features_explainer_b64.pkl"
# The ChatGPT baseline's self-discovered signature phrases — the reference set
# the baseline matrix's key_phrase_rate column was measured against. The runtime
# loads these to score a candidate message's key_phrase_rate consistently with
# the baseline cloud (store-cached as "baseline_key_phrase_profile").
BASELINE_KEY_PHRASES_PATH = "src/anubis/utils/dataset/baseline_key_phrases.json"


def load_bundled_baseline_features_arr() -> Any:
    """Load the current-width baseline feature matrix from the bundled ``.npy``."""
    return np.load(BASELINE_FEATURES_ARR_PATH, allow_pickle=False)


def baseline_feature_array_is_current(baseline_features_arr: Any) -> bool:
    """True when a baseline matrix has this build's feature-vector WIDTH.

    A matrix cached under an older :data:`STYLE_FEATURE_VECTOR_VERSION` has the
    wrong number of columns; feeding it to StandardScaler / IsolationForest /
    Mahalanobis against a current candidate row raises on the shape mismatch, so
    callers must detect the staleness and reload the bundled artifact first.
    """
    return (
        getattr(baseline_features_arr, "ndim", 0) == 2
        and baseline_features_arr.shape[1] == len(FEATURE_NAMES)
    )


# Envelope keys for the version-tagged serialized corpus. The blob is
# ``{VERSION_ENVELOPE_KEY: <int>, ROWS_ENVELOPE_KEY: {document_id: [cells]}}``.
# Version tagging is REQUIRED (not just a width check) because two feature-vector
# versions can share the same WIDTH while carrying incompatible SCALES — v3 stored
# punctuation as marks-per-1,000-words and v4 stores marks-per-word (÷1000), both
# 28 wide. Mixing the two scales in one Mahalanobis / IsolationForest corpus
# silently corrupts every comparison, so a version mismatch drops the whole
# stored corpus (it rebuilds at the current version as quotes are re-ingested).
_VECTOR_VERSION_ENVELOPE_KEY = "style_feature_vector_version"
_ROWS_ENVELOPE_KEY = "rows"


def serialize_features_by_doc_id(features_by_doc_id: Dict[str, Any]) -> str:
    """Serialize ``{document_id: 1-D feature row}`` to a version-tagged JSON string.

    Output shape: ``{"style_feature_vector_version": <int>, "rows":
    {document_id: [cells]}}``. The version tag lets
    :func:`deserialize_features_by_doc_id` reject a corpus written under a
    different :data:`STYLE_FEATURE_VECTOR_VERSION` even when the row WIDTH is
    unchanged (a width check alone cannot distinguish v3's per-1k scale from v4's
    per-word scale).

    Each row is coerced via ``.tolist()`` (numpy) or ``list``. Non-finite cells
    (the NaN a partial-NaN feature row legitimately carries — see
    ``extract_style_features``) are written as ``null`` so the blob is STRICT
    JSON: Python's ``json.dumps`` would otherwise emit the bare ``NaN`` token,
    which strict parsers (PostgreSQL ``::jsonb``, orjson) reject — breaking any
    SQL diagnostics over the stored corpus. ``deserialize_features_by_doc_id``
    maps ``null`` back to ``nan``.
    """

    def _strict_json_cell(cell_value: Any) -> Any:
        return cell_value if math.isfinite(cell_value) else None

    return json.dumps(
        {
            _VECTOR_VERSION_ENVELOPE_KEY: STYLE_FEATURE_VECTOR_VERSION,
            _ROWS_ENVELOPE_KEY: {
                doc_id: [
                    _strict_json_cell(cell_value)
                    for cell_value in (
                        row.tolist() if hasattr(row, "tolist") else list(row)
                    )
                ]
                for doc_id, row in features_by_doc_id.items()
            },
        }
    )


def deserialize_features_by_doc_id(features_by_doc_id_str: Any) -> Dict[str, Any]:
    """Inverse of :func:`serialize_features_by_doc_id`.

    Returns ``{}`` for any falsy input (key missing / never written) so callers
    can treat "no corpus yet" and "empty corpus" identically. Each value is
    rehydrated into a 1-D numpy array.

    **Feature-version migration.** The whole stored corpus is discarded unless
    it was written under the CURRENT :data:`STYLE_FEATURE_VECTOR_VERSION`:

    * A version-tagged envelope (v4+) whose ``style_feature_vector_version`` does
      not match the current version is dropped wholesale — its cells may be on an
      incompatible SCALE (v3 per-1k vs v4 per-word) even at the same width.
    * A legacy bare ``{document_id: [row]}`` blob (written before version tagging
      existed, i.e. v3 or earlier) has no version tag, so it is treated as
      pre-current and dropped wholesale.
    * Within a matching-version envelope, any row whose WIDTH is not
      ``len(FEATURE_NAMES)`` is still dropped defensively.

    This is the single chokepoint every reader (calibrate_ground_truth,
    graph._attach_analyzed_features, webapp deletion pruning) passes through. The
    corpus re-accumulates at the current version as fresh quotes are ingested,
    and the merge/re-serialize in calibrate_ground_truth persists the pruning.
    Callers that reach an empty dict simply skip ground-truth comparison until
    enough current-version rows exist.
    """
    if not features_by_doc_id_str:
        return {}
    raw = json.loads(features_by_doc_id_str)

    # Distinguish the version-tagged envelope from a legacy bare doc_id dict. Only
    # the envelope carries the version key; a legacy blob's keys are all
    # document_ids, so a missing/mismatched version tag means "not the current
    # version" and the entire corpus is dropped.
    is_versioned_envelope = (
        isinstance(raw, dict) and _VECTOR_VERSION_ENVELOPE_KEY in raw
    )
    if not is_versioned_envelope:
        logger.warning(
            "deserialize_features_by_doc_id: dropping untagged (pre-v%d) stored "
            "corpus; it predates version tagging and may be on an incompatible "
            "scale. The corpus will recalibrate as new quotes are ingested.",
            STYLE_FEATURE_VECTOR_VERSION,
        )
        return {}

    stored_version = raw.get(_VECTOR_VERSION_ENVELOPE_KEY)
    if stored_version != STYLE_FEATURE_VECTOR_VERSION:
        logger.warning(
            "deserialize_features_by_doc_id: dropping stored corpus written under "
            "feature-vector version %r (current is %d); scales/widths may differ. "
            "The corpus will recalibrate as new quotes are ingested.",
            stored_version,
            STYLE_FEATURE_VECTOR_VERSION,
        )
        return {}

    rows_by_doc_id = raw.get(_ROWS_ENVELOPE_KEY) or {}
    expected_width = len(FEATURE_NAMES)
    kept: Dict[str, Any] = {}
    dropped = 0
    for doc_id, row in rows_by_doc_id.items():
        # ``null`` cells are the strict-JSON encoding of NaN (see
        # serialize_features_by_doc_id).
        row_array = np.array(
            [math.nan if cell_value is None else cell_value for cell_value in row],
            dtype=np.float64,
        )
        if row_array.shape == (expected_width,):
            kept[doc_id] = row_array
        else:
            dropped += 1
    if dropped:
        logger.warning(
            "deserialize_features_by_doc_id: dropped %d/%d stored feature rows "
            "whose width != current %d-feature vector.",
            dropped,
            len(rows_by_doc_id),
            expected_width,
        )
    return kept


def features_by_doc_id_to_arr(features_by_doc_id: Dict[str, Any]) -> Any:
    """Recombine per-document feature rows into one ``(n_docs, F)`` array.

    Row order is irrelevant: every downstream consumer (empirical distribution,
    Mahalanobis covariance, IsolationForest) is a set statistic over the rows.
    ``F`` is the current vector width ``len(FEATURE_NAMES)``; the dict passed in
    has already been width-filtered by :func:`deserialize_features_by_doc_id`, so
    every row is guaranteed to have this width. Returns an empty ``(0, F)`` array
    when the dict is empty.
    """
    if not features_by_doc_id:
        return np.empty((0, len(FEATURE_NAMES)))
    return np.vstack([np.asarray(row) for row in features_by_doc_id.values()])


def feature_row_is_all_nan(feature_row: Any) -> bool:
    """True when every value in a single feature row is NaN.

    ``extract_style_features`` returns an all-NaN row for text that
    ``clean_text`` reduces to nothing (a URL-only or emoji-only quote line, for
    example). Such a row carries no stylometric signal at all, so writers use
    this predicate to keep the row out of the persisted corpus entirely.
    """
    row_array = np.asarray(feature_row, dtype=np.float64)
    return bool(np.isnan(row_array).all())


def sanitize_ground_truth_feature_matrix(feature_matrix: Any) -> Any:
    """Make a feature matrix safe for StandardScaler / LedoitWolf / IsolationForest.

    ``extract_style_features`` deliberately yields ``nan`` for metrics that
    cannot be computed on a given text ("callers impute later" — see
    ``_nan_features``); this function is that promised imputation chokepoint.
    scikit-learn's StandardScaler / LedoitWolf / IsolationForest all raise
    ``ValueError: Input X contains NaN`` otherwise.

    Method:

    * rows where EVERY value is NaN (empty/degenerate source text) are dropped —
      such a row describes nothing and would only drag the imputed medians;
    * each remaining NaN cell is imputed with the column's ``np.nanmedian`` —
      the median is the same location statistic ``build_style_profile_str``
      summarizes the corpus with, so imputation cannot shift a column's median;
    * a column that is NaN in every remaining row falls back to ``0.0`` (there
      is no observed value to take a median of).

    Returns a new ``(n_kept_rows, n_features)`` float array; the input is not
    mutated. An empty input passes through unchanged.
    """
    matrix = np.asarray(feature_matrix, dtype=np.float64)
    if matrix.size == 0:
        return matrix

    kept_row_mask = ~np.isnan(matrix).all(axis=1)
    sanitized_matrix = matrix[kept_row_mask].copy()
    if sanitized_matrix.size == 0:
        return sanitized_matrix

    nan_cell_mask = np.isnan(sanitized_matrix)
    if nan_cell_mask.any():
        # Suppress the "All-NaN slice" RuntimeWarning for columns with no
        # observed values; those medians come back NaN and are floored to 0.0.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            column_medians = np.nanmedian(sanitized_matrix, axis=0)
        column_medians = np.nan_to_num(column_medians, nan=0.0)
        sanitized_matrix[nan_cell_mask] = np.broadcast_to(
            column_medians, sanitized_matrix.shape
        )[nan_cell_mask]
    return sanitized_matrix


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


def recompute_ground_truth_artifacts(ground_truth_text_features_arr: Any) -> Tuple[str, str]:
    """Recalibrate the empirical threshold and IsolationForest from the corpus.

    Given the reconstructed ``(n_docs, len(FEATURE_NAMES))`` corpus array, returns
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

    # NaN-proof the corpus first (drop all-NaN rows, median-impute the rest):
    # StandardScaler / LedoitWolf / IsolationForest below all raise on NaN, and
    # both callers (calibrate_ground_truth and
    # webapp._prune_ground_truth_features_for_deleted_docs) can hold rows from
    # degenerate quote lines. This chokepoint covers them both.
    calibration_arr = sanitize_ground_truth_feature_matrix(
        ground_truth_text_features_arr
    )

    # Bound the O(n^2) leave-one-out calibration: above MAX_CALIBRATION_ROWS use a
    # deterministic random subsample (fixed seed so re-uploads of the same corpus
    # stay stable) instead of the full accumulated corpus.
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
    ground_truth_text_empirical_threshold = ground_truth_Q3 + 1.5 * (ground_truth_Q3 - ground_truth_Q1)

    # Recalibrate the Isolation Forest for prediction and explainable values.
    model = IsolationForest().fit(calibration_arr)

    ground_truth_text_empirical_threshold_list_str = json.dumps(
        ground_truth_text_empirical_threshold.tolist()
    )
    model_b64_pkl = base64.b64encode(pickle.dumps(model)).decode("utf-8")

    return ground_truth_text_empirical_threshold_list_str, model_b64_pkl

async def build_style_profile_str(ground_truth_text_features_arr) -> str:
    """ Build the LLM interpretable string to allow for the 
    median calculated features of the direct quotes of the 
    target to influence the writing of the avatar. 
    Allows the features to be LLM legible. 
    """
    import numpy as np
    import pandas as pd

    ground_truth_text_features_median = np.array(list(pd.DataFrame(ground_truth_text_features_arr).median(axis=0).values))
    
    # ground_truth_text_features_median expected shape: (n_features, )
    
    style_profile_str = ""

    idx = 0
    for name in FEATURE_NAMES:
        style_profile_str += f"{FEATURE_NAMES_HUMAN_LEGIBLE[name]}: {ground_truth_text_features_median[idx]}; Description: {FEATURE_DESCRIPTIONS[name]}\n\n"
        idx +=1

    return style_profile_str
