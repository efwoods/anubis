"""Auto-discovered signature key phrases.

Key phrases are the surviving key-phrase signal from
``features/statistical_significance.md``. The character n-gram and function-word
vectors that used to live here alongside them were dropped (they were
capture-only and never scored); the key-phrase signal is now used two ways:

* as the scalar ``key_phrase_rate`` in the fixed Mahalanobis vector of
  :mod:`src.anubis.utils.dataset.style_features` (via
  :func:`key_phrase_occurrence_rate`), and
* as a separately-stored, separately prompt-injected list of the avatar's
  signature phrases.

Two public functions:

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

* :func:`key_phrase_occurrence_rate` — given a text and a set of already-
  discovered phrases, counts how many times those phrases occur (as contiguous
  token runs) per total word. This is the scalar the avatar's ``key_phrase_rate``
  feature is built from, so it tokenises with the SAME :func:`tokenize` the
  discovery step uses to keep the two consistent.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Sequence

from src.anubis.utils.dataset.burrows_delta import tokenize

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

# The only single-character tokens that are real English words. Any other
# single-character token inside a candidate phrase is tokenizer shrapnel
# (URL path characters, stray initials from stripped handles) — such phrases
# are rejected by phrase_is_well_formed.
_VALID_SINGLE_CHARACTER_TOKENS = frozenset({"a", "i"})

# Tokens that only ever appear as debris of stripped markup, never as speech.
# "https"/"http"/"co" cover the ``https://t.co/...`` link shrapnel that
# dominated discovery before the corpus was cleaned; "amp" is the unescaped
# ``&amp;``. Kept as a read-time guard so phrase sets stored BEFORE the
# corpus-cleaning fix self-heal when reloaded and re-unioned.
_MARKUP_DEBRIS_TOKENS = frozenset({"https", "http", "www", "co", "amp"})


def build_corpus_phrase_attestation_set(
    documents: Sequence[str], *, ngram_sizes: tuple = (2, 3, 4)
) -> set:
    """Every 2–4-word phrase that actually occurs in the CLEANED corpus.

    Used to validate a previously-stored signature-phrase set against the
    current quote corpus: a signature phrase, by definition, must occur in the
    avatar's own quotes. Phrase sets stored before discovery cleaned its corpus
    hold artifacts of RAW text (@mention chains such as "cb doge tesla
    mayemusk") whose tokens look like real words, so no shape-based filter can
    reject them — but they never occur in the cleaned corpus, so attestation
    drops them. Tokenisation matches :func:`discover_key_phrases` exactly
    (clean_text then tokenize), so any discovered phrase is attested by
    construction.
    """
    from src.anubis.utils.dataset.style_features import clean_text

    attested_phrases: set = set()
    for document in documents:
        tokens = tokenize(clean_text(document))
        for ngram_size in ngram_sizes:
            for start_index in range(len(tokens) - ngram_size + 1):
                attested_phrases.add(
                    " ".join(tokens[start_index : start_index + ngram_size])
                )
    return attested_phrases


def phrase_is_well_formed(phrase: str) -> bool:
    """True when a signature phrase looks like real speech, not markup debris.

    Applied in three places so the same rule governs the phrase set everywhere:
    on the output of :func:`discover_key_phrases`, when a previously-stored
    phrase set is reloaded for re-union (healing sets polluted before the
    corpus-cleaning fix), and when the set is rendered into the
    <SIGNATURE PHRASES> system prompt section.
    """
    tokens = (phrase or "").split()
    if not tokens:
        return False
    for token in tokens:
        if token in _MARKUP_DEBRIS_TOKENS:
            return False
        if len(token) == 1 and token not in _VALID_SINGLE_CHARACTER_TOKENS:
            return False
    return True


def key_phrase_occurrence_rate(
    text: str, key_phrases: Sequence[str] | None
) -> float:
    """Signature-phrase occurrences per total word in ``text``.

    This is the scalar behind the ``key_phrase_rate`` stylometric feature. For
    each phrase in ``key_phrases`` we count how many times it appears in ``text``
    as a CONTIGUOUS run of tokens, sum those counts across all phrases, and divide
    by the text's total token count. Overlapping/repeated matches are counted (the
    "occurrence count" definition), so a text that leans heavily on the speaker's
    fixed collocations scores high.

    ``text`` is tokenised with the same :func:`tokenize` used by
    :func:`discover_key_phrases`, so a stored phrase (whose words were produced by
    that tokeniser and re-joined with single spaces) matches token-for-token here.

    Returns ``0.0`` when ``key_phrases`` is empty/``None`` or the text has no
    tokens — the neutral value for an avatar with no calibrated phrase set.
    """
    tokens = tokenize(text)
    token_total = len(tokens)
    if token_total == 0 or not key_phrases:
        return 0.0

    # Pre-split each phrase into its token sequence once. Skip empties defensively.
    phrase_token_sequences = [phrase.split() for phrase in key_phrases]
    phrase_token_sequences = [seq for seq in phrase_token_sequences if seq]
    if not phrase_token_sequences:
        return 0.0

    total_occurrences = 0
    for phrase_tokens in phrase_token_sequences:
        phrase_length = len(phrase_tokens)
        if phrase_length > token_total:
            continue
        for start_index in range(token_total - phrase_length + 1):
            if tokens[start_index : start_index + phrase_length] == phrase_tokens:
                total_occurrences += 1

    return total_occurrences / token_total


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
    # Discovery must mine the SAME text the key_phrase_rate feature is later
    # measured on: extract_style_features scores clean_text()'d text, so the
    # corpus is cleaned identically here (HTML entities unescaped, URLs and
    # @mentions dropped, apostrophes normalized). Mining raw tweets instead is
    # what produced the "https t co ..." / "amp ..." junk phrase sets — phrases
    # that could then NEVER match cleaned text, pinning key_phrase_rate to 0.
    # Imported lazily: style_features lazily imports this module in the other
    # direction, so a module-scope import here would be circular.
    from src.anubis.utils.dataset.style_features import clean_text

    corpus_tokens: List[List[str]] = [
        tokenize(clean_text(document)) for document in documents
    ]
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
            if not phrase_is_well_formed(" ".join(phrase_tokens)):
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
