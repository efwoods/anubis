"""Auto-discovered key phrases + function-word frequencies (capture-only).

These are two of the VECTOR-valued stylometric features requested in
``features/statistical_significance.md``. They do not reduce to a single scalar,
so they live here (and are surfaced through the nested profile in
:mod:`src.anubis.utils.dataset.stylistic_profile`) rather than in the fixed
Mahalanobis vector of :mod:`src.anubis.utils.dataset.style_features`. Nothing in
this module is wired into the authenticity evaluator yet — the goal for now is to
CAPTURE and persist these signals per avatar so a later pass can score against
them.

Two public functions:

* :func:`function_word_frequencies` — Mosteller-and-Wallace-style closed-class
  function-word rates for one text. Function words (articles, pronouns,
  prepositions, conjunctions, auxiliaries, particles) carry almost no topic, so
  their relative rates are a classic, topic-independent authorship fingerprint.

* :func:`discover_key_phrases` — over a corpus of the target's direct quotes,
  finds recurring multi-word expressions (2–4 words) that are OVER-REPRESENTED
  relative to a generic-English baseline. The score is a keyness ratio: the
  phrase's observed relative frequency in the corpus divided by the frequency a
  generic English writer would produce by chaining the same words INDEPENDENTLY
  (the product of the words' generic unigram frequencies). Because fixed
  collocations ("you know", "got it", "what do ya mean") recur far more than
  independence predicts, they rise to the top — which is exactly the behaviour
  asked for. This is a pointwise-mutual-information keyness against a bundled
  generic baseline, so it needs no corpus download and is fully deterministic.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List

from src.anubis.utils.dataset.burrows_delta import tokenize

# ---------------------------------------------------------------------------
# Closed-class English function words (topic-free authorship signal).
#
# Deliberately a closed class — determiners, pronouns, prepositions,
# conjunctions, auxiliary/modal verbs, wh-words, and common particles. Content
# words are excluded on purpose so the rates reflect HOW someone writes, not WHAT
# about. Apostrophe forms (it's, don't) are kept because contraction habits are
# themselves stylistic.
# ---------------------------------------------------------------------------
FUNCTION_WORDS: frozenset = frozenset(
    {
        # Articles / determiners
        "a", "an", "the", "this", "that", "these", "those", "some", "any",
        "no", "every", "each", "either", "neither", "all", "both", "such",
        # Personal / possessive / reflexive pronouns
        "i", "me", "my", "mine", "myself", "we", "us", "our", "ours",
        "ourselves", "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself", "she", "her", "hers", "herself",
        "it", "its", "itself", "they", "them", "their", "theirs",
        "themselves", "who", "whom", "whose", "which", "what",
        # Demonstratives / quantifiers already partly above; add indefinite
        "one", "none", "many", "much", "more", "most", "few", "less", "least",
        "other", "another", "someone", "somebody", "something", "anyone",
        "anybody", "anything", "everyone", "everybody", "everything",
        "nobody", "nothing",
        # Prepositions
        "of", "in", "on", "at", "by", "for", "with", "about", "against",
        "between", "into", "through", "during", "before", "after", "above",
        "below", "to", "from", "up", "down", "over", "under", "again",
        "further", "then", "once", "out", "off", "near", "within", "without",
        "toward", "towards", "upon", "among", "across", "behind", "beyond",
        # Conjunctions
        "and", "but", "or", "nor", "so", "yet", "because", "as", "until",
        "while", "although", "though", "unless", "since", "whereas", "if",
        "whether", "than",
        # Auxiliary / modal verbs
        "am", "is", "are", "was", "were", "be", "been", "being", "have",
        "has", "had", "having", "do", "does", "did", "doing", "can", "could",
        "will", "would", "shall", "should", "may", "might", "must", "ought",
        # Common particles / adverbs of degree & negation
        "not", "only", "just", "very", "too", "also", "even", "still", "here",
        "there", "when", "where", "why", "how", "all", "both", "own", "same",
    }
)


# ---------------------------------------------------------------------------
# Bundled generic-English unigram relative frequencies (fraction of running
# text). Approximate values for the most common words; anything absent is
# treated as the floor frequency below. This is the "generic English baseline"
# the key-phrase keyness score is measured against — a phrase's expected
# frequency under a generic writer is the PRODUCT of its words' values here, so
# only the ratios (not exact magnitudes) matter, and the long tail collapsing to
# a shared floor is fine.
# ---------------------------------------------------------------------------
GENERIC_ENGLISH_UNIGRAM_RELATIVE_FREQUENCY: Dict[str, float] = {
    "the": 0.0700, "of": 0.0360, "and": 0.0290, "to": 0.0260, "a": 0.0230,
    "in": 0.0210, "is": 0.0110, "it": 0.0100, "you": 0.0100, "that": 0.0100,
    "he": 0.0095, "was": 0.0090, "for": 0.0090, "on": 0.0075, "are": 0.0070,
    "with": 0.0070, "as": 0.0065, "i": 0.0064, "his": 0.0060, "they": 0.0058,
    "be": 0.0056, "at": 0.0054, "one": 0.0052, "have": 0.0050, "this": 0.0050,
    "from": 0.0048, "or": 0.0047, "had": 0.0046, "by": 0.0045, "not": 0.0044,
    "word": 0.0043, "but": 0.0042, "what": 0.0041, "some": 0.0040, "we": 0.0040,
    "can": 0.0039, "out": 0.0038, "other": 0.0037, "were": 0.0036, "all": 0.0035,
    "there": 0.0034, "when": 0.0033, "up": 0.0032, "use": 0.0031, "your": 0.0030,
    "how": 0.0030, "said": 0.0029, "an": 0.0028, "each": 0.0028, "she": 0.0027,
    "which": 0.0026, "do": 0.0026, "their": 0.0025, "time": 0.0025, "if": 0.0024,
    "will": 0.0024, "way": 0.0023, "about": 0.0023, "many": 0.0022, "then": 0.0022,
    "them": 0.0021, "would": 0.0021, "so": 0.0020, "these": 0.0020, "her": 0.0020,
    "him": 0.0019, "has": 0.0019, "look": 0.0018, "two": 0.0018, "more": 0.0018,
    "day": 0.0017, "could": 0.0017, "go": 0.0017, "come": 0.0016, "did": 0.0016,
    "my": 0.0016, "no": 0.0015, "get": 0.0015, "know": 0.0015, "just": 0.0014,
    "than": 0.0014, "like": 0.0014, "into": 0.0013, "our": 0.0013, "over": 0.0013,
    "think": 0.0012, "also": 0.0012, "back": 0.0012, "after": 0.0011, "well": 0.0011,
    "want": 0.0011, "because": 0.0011, "any": 0.0010, "good": 0.0010,
    "man": 0.0010, "here": 0.0010, "very": 0.0010, "mean": 0.0009, "got": 0.0009,
    "me": 0.0012, "us": 0.0009, "am": 0.0007, "yeah": 0.0004, "okay": 0.0004,
    "ya": 0.0002, "gonna": 0.0002, "kinda": 0.0001, "wanna": 0.0002,
}

# Frequency assigned to any word not in the table above (a rare word). Small
# enough that a phrase built from distinctive/content words gets a high keyness.
_GENERIC_FLOOR_RELATIVE_FREQUENCY = 5e-5


def function_word_frequencies(text: str, *, per_thousand: bool = True) -> Dict[str, Any]:
    """Closed-class function-word rates for one text (authorship fingerprint).

    Returns a JSON-serialisable dict with the per-word rates (only for function
    words that actually appear, to keep the payload compact), the overall share
    of tokens that are function words, and the token total the rates were
    computed over. Rates are per-1,000 tokens when ``per_thousand`` (the default),
    else relative frequencies in ``[0, 1]``.
    """
    tokens = tokenize(text)
    token_total = len(tokens)
    if token_total == 0:
        return {
            "token_total": 0,
            "function_word_token_share": 0.0,
            "rates_are_per_1k_tokens": per_thousand,
            "function_word_rates": {},
        }

    function_word_counts = Counter(t for t in tokens if t in FUNCTION_WORDS)
    scale = 1000.0 if per_thousand else 1.0
    rates = {
        word: (count / token_total) * scale
        for word, count in function_word_counts.items()
    }
    return {
        "token_total": token_total,
        "function_word_token_share": sum(function_word_counts.values()) / token_total,
        "rates_are_per_1k_tokens": per_thousand,
        # Sorted high-to-low so the dominant function words read first.
        "function_word_rates": dict(
            sorted(rates.items(), key=lambda kv: kv[1], reverse=True)
        ),
    }


def _generic_expected_relative_frequency(phrase_tokens: List[str]) -> float:
    """Frequency a generic English writer would emit this phrase by chance.

    Independence model: the product of each word's generic unigram relative
    frequency (floor for out-of-table words). This under-predicts fixed
    collocations, which is what makes the keyness ratio surface them.
    """
    expected = 1.0
    for token in phrase_tokens:
        expected *= GENERIC_ENGLISH_UNIGRAM_RELATIVE_FREQUENCY.get(
            token, _GENERIC_FLOOR_RELATIVE_FREQUENCY
        )
    return expected


def discover_key_phrases(
    documents: List[str],
    *,
    ngram_sizes: tuple = (2, 3, 4),
    min_count: int = 3,
    top_k: int = 40,
) -> List[Dict[str, Any]]:
    """Find recurring phrases over-represented vs generic English.

    Args:
        documents: The target's direct-quote corpus (one string per document).
        ngram_sizes: Phrase lengths in words to consider (default 2-, 3-, 4-grams).
        min_count: A phrase must occur at least this many times across the corpus
            to be a candidate — what makes a phrase "recurring" not a one-off.
        top_k: Maximum number of phrases to return, ranked by keyness (most
            distinctive first).

    Returns:
        A list of ``{"phrase", "count", "corpus_relative_frequency",
        "keyness_log2_over_generic_english"}`` dicts, JSON-serialisable, ordered
        by keyness descending. ``keyness`` is ``log2(observed / expected)``: 0
        means "as frequent as generic English predicts", positive means
        over-represented (the interesting direction), larger means more distinctive.
    """
    corpus_tokens: List[List[str]] = [tokenize(document) for document in documents]
    token_grand_total = sum(len(tokens) for tokens in corpus_tokens)
    if token_grand_total == 0:
        return []

    scored: List[Dict[str, Any]] = []
    for ngram_size in ngram_sizes:
        phrase_counts: Counter = Counter()
        for tokens in corpus_tokens:
            for start_index in range(len(tokens) - ngram_size + 1):
                phrase_tokens = tokens[start_index : start_index + ngram_size]
                phrase_counts[tuple(phrase_tokens)] += 1

        for phrase_tokens, count in phrase_counts.items():
            if count < min_count:
                continue
            # Normalise by the corpus token total (not the per-n phrase total) so
            # keyness is comparable across the different n-gram sizes.
            observed_relative_frequency = count / token_grand_total
            expected_relative_frequency = _generic_expected_relative_frequency(
                list(phrase_tokens)
            )
            keyness = math.log2(
                observed_relative_frequency / expected_relative_frequency
            )
            scored.append(
                {
                    "phrase": " ".join(phrase_tokens),
                    "count": count,
                    "corpus_relative_frequency": observed_relative_frequency,
                    "keyness_log2_over_generic_english": keyness,
                }
            )

    scored.sort(
        key=lambda item: item["keyness_log2_over_generic_english"], reverse=True
    )
    return scored[:top_k]
