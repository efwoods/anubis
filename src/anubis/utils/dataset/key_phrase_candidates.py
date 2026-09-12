"""Stage one of signature key-phrase discovery: statistical recall.

This module answers "which words and expressions could plausibly mark how this
person talks?" and deliberately does NOT answer "which of them actually do" —
that judgement belongs to :mod:`src.anubis.utils.dataset.key_phrase_judgement`,
because no statistic can separate a stance marker ("concerning") from a topic
word ("civilization") when both are content words over-represented against a
reference corpus.

Why this module exists at all, when
:func:`src.anubis.utils.dataset.key_phrases.discover_key_phrases` already mines
phrases: that function scores a phrase as
``log2(observed / product-of-per-word-generic-frequencies)`` against a hardcoded
hundred-word table with a ``5e-5`` floor for every other word. Because an
out-of-table word always contributes the floor, the score is really a proxy for
"is this phrase built out of rare words", so topic nouns and proper nouns
dominate. Measured over 5,904 tweets, the resulting top-40 was almost entirely
subject matter and advertising copy, one advertisement alone occupied eight of
the forty slots as eight overlapping n-grams, and — because the phrase lengths
were fixed at two-to-four words — a single distinctive word could never be
discovered at all.

The replacement rests on four ideas:

1. **Contrast against a real reference corpus, not an independence model.** A
   phrase is interesting when the target says it far more than a reference
   speaker would. The reference is the bundled ChatGPT baseline corpus.
2. **Back off rather than count directly above one word.** Measured on that
   reference, 96% of its three-word phrases and 98.6% of its four-word phrases
   occur exactly once, so a direct count comparison at those lengths would
   report "unseen in the reference" for nearly every candidate — reproducing the
   very rarity artifact being removed. An interpolated bigram back-off model
   gives every phrase a small but strictly positive expectation instead.
3. **Rank by effect size, gate on significance.** The Dunning log-likelihood
   grows with raw frequency, so ranking by the Dunning log-likelihood floods the
   list with "is", "of", "the", "that". Ranking by the log ratio of relative
   frequencies and using the Dunning log-likelihood only as a significance gate
   keeps the frequent-but-unremarkable words out.
4. **Dispersion, not raw count, decides what "recurring" means.** A phrase
   concentrated in one document is that document's subject; a phrase spread
   through the person's media is a habit.

Everything here is pure Python, deterministic, and free of numpy and of any
network access, so the caller can keep running it inside ``asyncio.to_thread``.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

from src.anubis.utils.dataset.burrows_delta import tokenize

# The reference corpus. This is the same file the ChatGPT baseline artifacts are
# fitted from, reused here as the "generic assistant prose" pole. Resolved
# relative to this module so the path holds under both the Docker layout and a
# local checkout.
_REPOSITORY_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
DEFAULT_REFERENCE_CORPUS_PATH = os.path.join(
    _REPOSITORY_ROOT, "data", "unmodified_inference_model_baseline_corpus.jsonl"
)

# Case-preserving word pattern for the proper-noun pass. clean_text keeps casing
# on purpose, and burrows_delta.tokenize lowercases, so the capitalisation
# evidence has to be gathered in a separate pass over the cleaned-but-uncased
# text with a pattern that mirrors the tokeniser's notion of a word.
_CASE_PRESERVING_WORD_PATTERN = re.compile(r"[A-Za-z']+")

# A sentence boundary for the purposes of "is this token sentence-initial".
# Deliberately a cheap regex rather than style_features._sentences: that helper
# reaches for nltk, and this module is imported inside a worker thread where a
# cold nltk download would be a surprising cost for a question this crude.
_SENTENCE_TERMINATOR_CHARACTERS = frozenset(".!?\n\r")

# Chi-square critical value at one degree of freedom, p < 0.001. A candidate
# whose Dunning log-likelihood falls below this is not distinguishable from the
# reference at that confidence and is discarded before ranking.
DUNNING_SIGNIFICANCE_THRESHOLD_P_001 = 10.83

# Closed-class grammatical words. A phrase is rejected when EVERY one of its
# tokens is drawn from this set, which covers both the single bare function word
# ("what", "the") and the all-grammatical collocation ("will be", "there is",
# "we are", "should be"). A phrase containing at least one open-class word keeps
# its function words freely, which is what lets "what do ya mean", "you know"
# and "got it" survive.
#
# The reason all-grammatical phrases are dropped rather than judged is that they
# carry no lexical signal of their own AND are already measured elsewhere: the
# function-word channel of Burrows' Delta is precisely a fingerprint of how this
# person distributes exactly these words. Keeping them here would duplicate that
# measurement while diluting the shortlist the judge has to read.
#
# The exclusions matter more than the inclusions. This set deliberately omits
# stance and discourse adverbs — "just", "really", "actually", "honestly",
# "literally", "maybe", "definitely", "absolutely", "yeah", "okay", "kinda",
# "lol", "anyway" and their kin — because those ARE the idiolect this module
# exists to find. For the same reason neither of the two lists already available
# in the codebase is usable here: GENERIC_ENGLISH_UNIGRAM_RELATIVE_FREQUENCY is a
# frequency table that contains "yeah", "okay", "well" and "just", and the nltk
# stopword list contains "just", "very", "so" and "too".
GRAMMATICAL_FUNCTION_WORDS: FrozenSet[str] = (
    frozenset(
        {
            # determiners and quantifiers of the closed class
            "a", "an", "the", "this", "that", "these", "those", "each", "every",
            "either", "neither", "both", "another",
            # pronouns and possessives
            "i", "me", "my", "mine", "myself",
            "you", "your", "yours", "yourself", "yourselves",
            "he", "him", "his", "himself",
            "she", "her", "hers", "herself",
            "it", "its", "itself",
            "we", "us", "our", "ours", "ourselves",
            "they", "them", "their", "theirs", "themselves",
            "who", "whom", "whose", "which",
            # auxiliaries and copulas
            "am", "is", "are", "was", "were", "be", "been", "being",
            "do", "does", "did", "done", "doing",
            "have", "has", "had", "having",
            "will", "would", "shall", "should", "can", "could", "may", "might",
            "must", "ought",
            # contracted auxiliaries the tokeniser keeps whole
            "i'm", "it's", "that's", "there's", "he's", "she's", "we're",
            "they're", "you're", "i've", "we've", "they've", "you've",
            "i'll", "we'll", "they'll", "you'll", "it'll", "that'll",
            "i'd", "we'd", "they'd", "you'd", "he'd", "she'd", "it'd",
            "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't",
            "didn't", "haven't", "hasn't", "hadn't", "won't", "wouldn't",
            "can't", "cannot", "couldn't", "shouldn't", "mustn't",
            # prepositions
            "about", "above", "across", "after", "against", "along", "among",
            "around", "as", "at", "before", "behind", "below", "beneath",
            "beside", "besides", "between", "beyond", "by", "down", "during",
            "except", "for", "from", "in", "inside", "into", "near", "of",
            "off", "on", "onto", "out", "outside", "over", "past", "per",
            "since", "through", "throughout", "to", "toward", "towards",
            "under", "underneath", "until", "up", "upon", "with", "within",
            "without",
            # conjunctions and complementizers
            "and", "or", "but", "nor", "yet", "so", "if", "because", "although",
            "though", "unless", "whereas", "while", "whether", "than", "that",
            # negators and bare wh-words
            "not", "no", "what", "when", "where", "why", "how",
            # numerals of the closed class
            "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "first", "second", "third",
        }
    )
)



def phrase_is_entirely_grammatical(phrase_tokens: Sequence[str]) -> bool:
    """True when every token is a closed-class grammatical word.

    Such a phrase says nothing about this person's vocabulary, and the
    function-word channel of Burrows' Delta already measures how heavily the
    person leans on these words. Rejecting them here keeps the shortlist the
    judge reads about a third shorter on real corpora.
    """
    if not phrase_tokens:
        return True
    return all(token in GRAMMATICAL_FUNCTION_WORDS for token in phrase_tokens)


@dataclass(frozen=True)
class KeyPhraseDiscoveryConfiguration:
    """Every tunable of the statistical stage, in one place.

    The defaults were chosen against two real corpora — 5,904 Musk tweets
    (49,070 tokens) and the 21,566-token reference — and each is annotated with
    the behaviour it produces there.
    """

    # One-word candidates are the point of this rewrite: a person's most
    # characteristic marker is often a single word they reach for.
    ngram_sizes: Tuple[int, ...] = (1, 2, 3, 4)

    # None means "derive from corpus size". A fixed floor is wrong in both
    # directions: three occurrences in 49,070 tokens is a topic burst, while the
    # same three in a small upload is genuinely all the evidence there is.
    minimum_occurrence_count: Optional[int] = None
    occurrence_floor_tokens_per_increment: int = 10000
    minimum_occurrence_count_floor: int = 3

    # Dispersion: in how many distinct units must a phrase appear. The floor of
    # two is the operative rule — a phrase confined to a single unit is that
    # unit's subject, not a habit.
    minimum_dispersion_unit_count: Optional[int] = None
    dispersion_unit_fraction: float = 0.01
    minimum_dispersion_unit_count_floor: int = 2
    maximum_adaptive_dispersion_unit_count: int = 5

    # A long transcript is one document but many occasions. Splitting into
    # windows makes "spread through the media" mean the same thing whether the
    # corpus is six thousand tweets or three two-hour transcripts.
    dispersion_window_size_tokens: int = 400

    dunning_log_likelihood_threshold: float = DUNNING_SIGNIFICANCE_THRESHOLD_P_001

    # Additive smoothing on the REFERENCE side. This is the single mechanism
    # that removes the rarity artifact: an unseen rare word no longer earns a
    # huge score for being rare, because every unseen word is floored alike.
    smoothing_count: float = 0.5
    # Weight on the bigram term of the back-off. Low enough that the model stays
    # close to a unigram product (the reference is too small to trust its bigram
    # conditionals heavily), high enough to reward genuine collocation.
    backoff_interpolation_weight: float = 0.4

    # A token capitalised in at least this share of its non-sentence-initial
    # appearances is treated as a name, and any phrase containing it is dropped.
    proper_noun_capitalisation_ratio_threshold: float = 0.5
    proper_noun_minimum_observation_count: int = 2

    # Overlap collapse: how much of the shorter phrase must sit inside the
    # longer one before the shorter is considered the same expression.
    subsumption_containment_threshold: float = 0.70

    candidate_pool_size: int = 250
    shortlist_size: int = 200

    concordance_line_count: int = 3
    concordance_context_words: int = 6


@dataclass(frozen=True)
class KeyPhraseCandidate:
    """One surviving phrase plus every number the judge and the ranking need."""

    phrase: str
    ngram_size: int
    occurrence_count: int
    dispersion_unit_count: int
    target_relative_frequency: float
    reference_expected_count: float
    dunning_log_likelihood: float
    log_ratio_over_reference: float
    concordance_lines: Tuple[str, ...] = ()

    def as_storable_dictionary(self) -> Dict[str, object]:
        """JSON-serialisable view for the stored profile detail envelope."""
        return {
            "phrase": self.phrase,
            "ngram_size": self.ngram_size,
            "occurrence_count": self.occurrence_count,
            "dispersion_unit_count": self.dispersion_unit_count,
            "target_relative_frequency": self.target_relative_frequency,
            "reference_expected_count": self.reference_expected_count,
            "dunning_log_likelihood": self.dunning_log_likelihood,
            "log_ratio_over_reference": self.log_ratio_over_reference,
        }


@dataclass(frozen=True)
class ReferenceLanguageModel:
    """Interpolated bigram back-off model over the reference corpus.

    The expectation for a phrase is the chain
    ``P(first) * P(second|first) * P(third|second) * …``, where each conditional
    interpolates the reference's bigram estimate with the smoothed unigram
    estimate. For a one-word phrase the chain collapses to the smoothed unigram
    probability, so there is no separate code path for single words.
    """

    unigram_counts: Mapping[str, int]
    bigram_counts: Mapping[Tuple[str, str], int]
    unigram_token_total: int
    vocabulary_size: int
    document_count: int
    smoothing_count: float
    backoff_interpolation_weight: float

    def unigram_probability(self, token: str) -> float:
        """Additively smoothed unigram probability; strictly positive always."""
        numerator = self.unigram_counts.get(token, 0) + self.smoothing_count
        denominator = self.unigram_token_total + self.smoothing_count * (
            self.vocabulary_size + 1
        )
        if denominator <= 0.0:
            return 0.0
        return numerator / denominator

    def conditional_probability(self, previous_token: str, token: str) -> float:
        """Interpolation of the bigram estimate with the smoothed unigram one."""
        unigram_estimate = self.unigram_probability(token)
        previous_count = self.unigram_counts.get(previous_token, 0)
        if previous_count <= 0:
            return unigram_estimate
        bigram_estimate = (
            self.bigram_counts.get((previous_token, token), 0) / previous_count
        )
        weight = self.backoff_interpolation_weight
        return weight * bigram_estimate + (1.0 - weight) * unigram_estimate

    def expected_relative_frequency(self, phrase_tokens: Sequence[str]) -> float:
        """Relative frequency a reference speaker would produce this phrase at."""
        if not phrase_tokens:
            return 0.0
        expected = self.unigram_probability(phrase_tokens[0])
        for position in range(1, len(phrase_tokens)):
            expected *= self.conditional_probability(
                phrase_tokens[position - 1], phrase_tokens[position]
            )
        return expected

    def ngram_total(self, ngram_size: int) -> int:
        """How many n-gram positions of this size the reference corpus holds.

        Approximated as the token total, which overstates by at most
        ``(ngram_size - 1)`` per document. The overstatement is uniform across
        candidates of the same length, so the ranking is unaffected.
        """
        return max(self.unigram_token_total, 1)


def load_reference_corpus_texts(
    reference_corpus_path: str = DEFAULT_REFERENCE_CORPUS_PATH,
) -> Tuple[str, ...]:
    """Final assistant reply from every conversation in the baseline corpus.

    Duplicates the few lines of ``data/build_baseline_features_arr.py``'s
    ``_baseline_assistant_texts`` rather than importing them: ``data/`` is a
    directory of scripts, not an importable package at runtime (that script
    inserts the repository root onto ``sys.path`` in order to run at all).

    A missing or unreadable corpus returns an empty tuple rather than raising,
    so a deployment whose data directory was trimmed degrades to "no candidates"
    instead of failing the whole upload calibration.
    """
    texts: List[str] = []
    try:
        with open(reference_corpus_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                conversation = json.loads(line)
                assistant_messages = [
                    message["content"]
                    for message in conversation.get("messages", [])
                    if message.get("role") == "assistant" and message.get("content")
                ]
                if assistant_messages:
                    texts.append(assistant_messages[-1])
    except (OSError, ValueError):
        return ()
    return tuple(texts)


@lru_cache(maxsize=4)
def build_reference_language_model(
    reference_texts: Tuple[str, ...],
    *,
    smoothing_count: float = 0.5,
    backoff_interpolation_weight: float = 0.4,
) -> ReferenceLanguageModel:
    """Count the reference once per process and cache the result.

    Cached on the texts tuple so tests can build a model over a synthetic
    reference without disturbing the bundled one.
    """
    from src.anubis.utils.dataset.style_features import clean_text

    unigram_counts: Counter[str] = Counter()
    bigram_counts: Counter[Tuple[str, str]] = Counter()
    for text in reference_texts:
        tokens = tokenize(clean_text(text))
        unigram_counts.update(tokens)
        for position in range(len(tokens) - 1):
            bigram_counts[(tokens[position], tokens[position + 1])] += 1

    return ReferenceLanguageModel(
        unigram_counts=dict(unigram_counts),
        bigram_counts=dict(bigram_counts),
        unigram_token_total=sum(unigram_counts.values()),
        vocabulary_size=len(unigram_counts),
        document_count=len(reference_texts),
        smoothing_count=smoothing_count,
        backoff_interpolation_weight=backoff_interpolation_weight,
    )


@lru_cache(maxsize=1)
def load_default_reference_language_model(
    smoothing_count: float = 0.5, backoff_interpolation_weight: float = 0.4
) -> ReferenceLanguageModel:
    """The bundled reference model, built at most once per process."""
    return build_reference_language_model(
        load_reference_corpus_texts(),
        smoothing_count=smoothing_count,
        backoff_interpolation_weight=backoff_interpolation_weight,
    )


def build_dispersion_units(
    documents: Sequence[str], *, window_size_tokens: int
) -> List[List[str]]:
    """Split the corpus into the units dispersion is counted over.

    A document longer than ``window_size_tokens`` becomes several consecutive
    windows; a shorter one is a single unit. One rule covers both corpus shapes
    the product actually sees: six thousand tweets stay six thousand units, and
    three two-hour transcripts become roughly a hundred and fifty units rather
    than three. Without this, "appears in at least two documents" would be
    unsatisfiable for anyone who uploaded a single long recording.

    Windows never straddle a document boundary, so a phrase repeated in two
    different transcripts is credited to two different sources.
    """
    from src.anubis.utils.dataset.style_features import clean_text

    units: List[List[str]] = []
    for document in documents:
        tokens = tokenize(clean_text(document))
        if not tokens:
            continue
        if len(tokens) <= window_size_tokens:
            units.append(tokens)
            continue
        for start in range(0, len(tokens), window_size_tokens):
            window = tokens[start : start + window_size_tokens]
            if window:
                units.append(window)
    return units


def build_probable_proper_noun_token_set(
    documents: Sequence[str],
    *,
    capitalisation_ratio_threshold: float,
    minimum_observation_count: int,
) -> FrozenSet[str]:
    """Tokens that behave like names, judged by mid-sentence capitalisation.

    A word that is capitalised where capitalisation is not merely positional is
    a name, a product or a place — subject matter rather than style. Two
    positions are excluded from the evidence because capitalisation there means
    something else:

    * **Sentence-initial**, where every word is capitalised regardless.
    * **Inside a run of two or more all-capital tokens**, because sustained
      capitals are shouting, and shouting is a style signal this pipeline should
      keep rather than a naming signal.

    Runs over ``clean_text`` output with a case-preserving pattern, since the
    shared tokeniser lowercases and would erase the evidence entirely.
    """
    from src.anubis.utils.dataset.style_features import clean_text

    capitalised_observations: Counter[str] = Counter()
    total_observations: Counter[str] = Counter()

    for document in documents:
        text = clean_text(document)
        matches = list(_CASE_PRESERVING_WORD_PATTERN.finditer(text))
        all_capital_flags = [
            match.group(0).isupper() and len(match.group(0)) > 1 for match in matches
        ]
        for index, match in enumerate(matches):
            word = match.group(0)
            # Sentence-initial: nothing before this word, or the nearest
            # preceding non-space character terminates a sentence.
            preceding = text[: match.start()].rstrip()
            if not preceding or preceding[-1] in _SENTENCE_TERMINATOR_CHARACTERS:
                continue
            # Inside a shouted run: this token and a neighbour are both all-caps.
            neighbour_is_all_capital = (
                index > 0 and all_capital_flags[index - 1]
            ) or (
                index + 1 < len(all_capital_flags) and all_capital_flags[index + 1]
            )
            if all_capital_flags[index] and neighbour_is_all_capital:
                continue
            lowered = word.lower()
            total_observations[lowered] += 1
            if word[:1].isupper():
                capitalised_observations[lowered] += 1

    probable_proper_nouns: Set[str] = {
        token
        for token, observations in total_observations.items()
        if observations >= minimum_observation_count
        and capitalised_observations[token] / observations
        >= capitalisation_ratio_threshold
    }
    return frozenset(probable_proper_nouns)


def dunning_log_likelihood(
    target_count: int,
    reference_expected_count: float,
    target_ngram_total: int,
    reference_ngram_total: int,
) -> float:
    """Two-way Dunning log-likelihood for one phrase, target versus reference.

    Used only as a SIGNIFICANCE GATE, never as the ranking key. The statistic
    scales with raw frequency, so ranking by it promotes the most common
    function words above every distinctive marker — measured, not assumed.

    The reference side is an expected count from the back-off model rather than
    an observed count, so it is fractional and may be arbitrarily small; the
    corresponding term is dropped when it reaches zero, which is the standard
    convention for the zero cell.
    """
    if target_count <= 0 or target_ngram_total <= 0 or reference_ngram_total <= 0:
        return 0.0

    observed_total = target_count + reference_expected_count
    combined_total = target_ngram_total + reference_ngram_total
    expected_target = target_ngram_total * observed_total / combined_total
    expected_reference = reference_ngram_total * observed_total / combined_total

    log_likelihood = 0.0
    if expected_target > 0.0:
        log_likelihood += target_count * math.log(target_count / expected_target)
    if reference_expected_count > 0.0 and expected_reference > 0.0:
        log_likelihood += reference_expected_count * math.log(
            reference_expected_count / expected_reference
        )
    return 2.0 * log_likelihood


def log_ratio_over_reference(
    target_count: int,
    reference_expected_count: float,
    target_ngram_total: int,
    reference_ngram_total: int,
    *,
    smoothing_count: float,
) -> float:
    """Effect size: how many doublings more often the target says this phrase.

    This is the RANKING key. Being a ratio of relative frequencies it is
    independent of how common the phrase is in absolute terms, which is exactly
    the property the Dunning log-likelihood lacks.
    """
    if target_ngram_total <= 0 or reference_ngram_total <= 0:
        return 0.0
    target_rate = (target_count + smoothing_count) / target_ngram_total
    reference_rate = (
        reference_expected_count + smoothing_count
    ) / reference_ngram_total
    if reference_rate <= 0.0 or target_rate <= 0.0:
        return 0.0
    return math.log2(target_rate / reference_rate)


def _occupied_token_positions(
    phrase: str, occurrence_positions: Sequence[Tuple[int, int]]
) -> Set[Tuple[int, int]]:
    """Every ``(unit, token index)`` this phrase's occurrences physically cover."""
    phrase_length = len(phrase.split())
    occupied: Set[Tuple[int, int]] = set()
    for unit_index, start_index in occurrence_positions:
        for offset in range(phrase_length):
            occupied.add((unit_index, start_index + offset))
    return occupied


def overlap_fraction_with_occupied_positions(
    phrase: str,
    occurrence_positions: Sequence[Tuple[int, int]],
    occupied_positions: Set[Tuple[int, int]],
) -> float:
    """Share of this phrase's occurrences that sit on already-claimed tokens.

    Overlap is measured on TOKEN POSITIONS rather than by asking whether one
    phrase's text contains the other's, because the redundancy that actually
    occurs is not always nesting. A repeated sentence produces a row of
    same-length phrases sliding along it — "sign up via web", "up via web
    browser", "via web browser at" — none of which contains any other, yet all
    of which are the same underlying string. Positional overlap catches those
    and ordinary nesting alike.

    A fraction near one means this phrase is almost entirely explained by
    phrases already kept, so it adds nothing.
    """
    if not occurrence_positions:
        return 0.0
    phrase_length = len(phrase.split())
    overlapping = 0
    for unit_index, start_index in occurrence_positions:
        if any(
            (unit_index, start_index + offset) in occupied_positions
            for offset in range(phrase_length)
        ):
            overlapping += 1
    return overlapping / len(occurrence_positions)


def collapse_subsumed_candidates(
    candidates: Sequence[KeyPhraseCandidate],
    occurrence_positions: Mapping[str, Sequence[Tuple[int, int]]],
    *,
    containment_threshold: float,
) -> List[KeyPhraseCandidate]:
    """Keep one phrase per underlying expression.

    Which member of a family survives is decided BEFORE the walk, and not by
    score. The walk visits the LONGEST candidates first, so a full expression
    claims its token positions before any fragment of it is considered, and a
    shorter phrase is then dropped only when the phrases already kept cover most
    of its occurrences — that is, only when it never appears on its own.

    The rule is "a shorter phrase survives if it lives independently", and both
    interesting cases fall out of it. A speaker who says "what do ya mean" makes
    "ya" and "mean" occur ONLY inside that expression, so both fragments are
    dropped and the expression is kept, which is right — "ya" alone is not the
    habit. A speaker who says "gets me every time" also says "gets" in dozens of
    unrelated sentences, so only a small share of "gets" is claimed and both
    survive on their own merits, which is also right.

    Ordering by score instead would keep the longest member of every family
    unconditionally, because more content words mean a higher ratio against the
    reference; a speaker whose habit is simply the word "tricky" would then get
    whichever sentence they most often surrounded it with.

    The returned list is re-sorted into the caller's ranking order, so this
    function changes WHICH phrases survive and never their final order.
    """
    walk_order = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.ngram_size,
            -candidate.dispersion_unit_count,
            -candidate.log_ratio_over_reference,
            candidate.phrase,
        ),
    )

    kept: List[KeyPhraseCandidate] = []
    occupied_positions: Set[Tuple[int, int]] = set()
    for candidate in walk_order:
        positions = occurrence_positions.get(candidate.phrase) or ()
        if (
            overlap_fraction_with_occupied_positions(
                candidate.phrase, positions, occupied_positions
            )
            >= containment_threshold
        ):
            continue
        kept.append(candidate)
        occupied_positions |= _occupied_token_positions(candidate.phrase, positions)

    kept.sort(
        key=lambda candidate: (
            -candidate.log_ratio_over_reference,
            -candidate.occurrence_count,
            candidate.phrase,
        )
    )
    return kept


def build_concordance_lines(
    phrase: str,
    dispersion_units: Sequence[Sequence[str]],
    occurrence_positions: Sequence[Tuple[int, int]],
    *,
    line_count: int,
    context_words: int,
) -> Tuple[str, ...]:
    """Keyword-in-context lines showing how the phrase is actually used.

    Occurrences are sampled EVENLY across the corpus (first, middle, last)
    rather than taking the first few, so the judge sees the phrase's range of
    use instead of three lines from whichever document happened to be read
    first — which matters most for exactly the phrases the judge finds hardest,
    where one usage reads as style and another as subject matter.
    """
    if not occurrence_positions or line_count <= 0:
        return ()

    total = len(occurrence_positions)
    if total <= line_count:
        sampled_indices = list(range(total))
    else:
        sampled_indices = [
            round(step * (total - 1) / (line_count - 1)) if line_count > 1 else 0
            for step in range(line_count)
        ]
        sampled_indices = sorted(set(sampled_indices))

    phrase_length = len(phrase.split())
    lines: List[str] = []
    for index in sampled_indices:
        unit_index, start_index = occurrence_positions[index]
        if unit_index >= len(dispersion_units):
            continue
        unit = dispersion_units[unit_index]
        left = unit[max(0, start_index - context_words) : start_index]
        right = unit[
            start_index + phrase_length : start_index + phrase_length + context_words
        ]
        lines.append(
            f"{' '.join(left)} [{phrase}] {' '.join(right)}".strip()
        )
    return tuple(lines)


def resolve_minimum_occurrence_count(
    target_token_total: int, configuration: KeyPhraseDiscoveryConfiguration
) -> int:
    """How many total occurrences make a phrase "recurring", scaled to the corpus.

    A fixed floor is wrong in both directions. Three occurrences in 49,070
    tokens is a topic burst that the old fixed floor of three admitted; the same
    three in a two-thousand-token upload is all the evidence that exists.
    """
    if configuration.minimum_occurrence_count is not None:
        return max(1, configuration.minimum_occurrence_count)
    scaled = round(
        target_token_total / max(1, configuration.occurrence_floor_tokens_per_increment)
    )
    return max(configuration.minimum_occurrence_count_floor, int(scaled))


def resolve_minimum_dispersion_unit_count(
    dispersion_unit_total: int, configuration: KeyPhraseDiscoveryConfiguration
) -> int:
    """In how many distinct units a phrase must appear.

    The floor of two is the rule that answers "a phrase may persist only once":
    a phrase confined to one document or one passage is that passage's subject,
    while a phrase found in two or more separate places is a habit — even if it
    occurs only once in each.
    """
    if configuration.minimum_dispersion_unit_count is not None:
        return max(1, configuration.minimum_dispersion_unit_count)
    proportional = math.ceil(
        dispersion_unit_total * configuration.dispersion_unit_fraction
    )
    return min(
        configuration.maximum_adaptive_dispersion_unit_count,
        max(configuration.minimum_dispersion_unit_count_floor, proportional),
    )


def _count_ngrams_with_dispersion(
    dispersion_units: Sequence[Sequence[str]], ngram_size: int
) -> Tuple[Counter[str], Counter[str]]:
    """Occurrence counts and dispersion-unit counts for one phrase length."""
    occurrence_counts: Counter[str] = Counter()
    dispersion_counts: Counter[str] = Counter()
    for unit in dispersion_units:
        seen_in_unit = set()
        for start in range(len(unit) - ngram_size + 1):
            phrase = " ".join(unit[start : start + ngram_size])
            occurrence_counts[phrase] += 1
            seen_in_unit.add(phrase)
        for phrase in seen_in_unit:
            dispersion_counts[phrase] += 1
    return occurrence_counts, dispersion_counts


def _collect_occurrence_positions(
    dispersion_units: Sequence[Sequence[str]],
    phrases_by_ngram_size: Mapping[int, FrozenSet[str]],
) -> Dict[str, List[Tuple[int, int]]]:
    """Positions of the surviving phrases only.

    Deliberately a second pass. Recording positions for every n-gram during
    counting would hold roughly two hundred thousand tuples for a corpus the
    size of the tweet set, nearly all of them for phrases about to be discarded.
    """
    positions: Dict[str, List[Tuple[int, int]]] = {}
    for unit_index, unit in enumerate(dispersion_units):
        for ngram_size, phrases in phrases_by_ngram_size.items():
            if not phrases:
                continue
            for start in range(len(unit) - ngram_size + 1):
                phrase = " ".join(unit[start : start + ngram_size])
                if phrase in phrases:
                    positions.setdefault(phrase, []).append((unit_index, start))
    return positions


def discover_key_phrase_candidates(
    documents: Sequence[str],
    *,
    configuration: Optional[KeyPhraseDiscoveryConfiguration] = None,
    reference_language_model: Optional[ReferenceLanguageModel] = None,
    protected_phrases: Optional[FrozenSet[str]] = None,
) -> List[KeyPhraseCandidate]:
    """Statistical shortlist of phrases that could mark how this person talks.

    The output is deliberately a SHORTLIST, not an answer: it maximises recall
    of stylistic markers while removing the four things that are provably not
    style (markup debris, names, bare grammatical words, and phrase fragments
    that duplicate a longer phrase). Deciding which of the survivors are style
    rather than subject matter is the judge's job.

    ``protected_phrases`` are phrases already stored for this speaker. They are
    exempt from the pool truncation only — never from the filters or the
    significance gate — so an incumbent is re-measured against the CURRENT
    corpus and keeps its place only if the evidence still supports it. Without
    the exemption an incumbent could be dropped merely because a large corpus
    produced more candidates than the pool holds, which would make the stored
    set churn for reasons unrelated to the speaker.

    Ordering is fully deterministic — the sort key ends in the phrase text
    itself — so two runs over the same corpus produce byte-identical output.
    That matters downstream: an unstable phrase set forces the calibration to
    recompute every stored feature row.
    """
    from src.anubis.utils.dataset.key_phrases import phrase_is_well_formed

    configuration = configuration or KeyPhraseDiscoveryConfiguration()
    if reference_language_model is None:
        reference_language_model = load_default_reference_language_model(
            configuration.smoothing_count, configuration.backoff_interpolation_weight
        )

    dispersion_units = build_dispersion_units(
        documents, window_size_tokens=configuration.dispersion_window_size_tokens
    )
    target_token_total = sum(len(unit) for unit in dispersion_units)
    if target_token_total == 0 or reference_language_model.unigram_token_total == 0:
        return []

    minimum_occurrence_count = resolve_minimum_occurrence_count(
        target_token_total, configuration
    )
    minimum_dispersion_unit_count = resolve_minimum_dispersion_unit_count(
        len(dispersion_units), configuration
    )
    probable_proper_nouns = build_probable_proper_noun_token_set(
        documents,
        capitalisation_ratio_threshold=(
            configuration.proper_noun_capitalisation_ratio_threshold
        ),
        minimum_observation_count=configuration.proper_noun_minimum_observation_count,
    )
    reference_ngram_total = reference_language_model.ngram_total(1)

    scored_candidates: List[KeyPhraseCandidate] = []
    surviving_phrases_by_ngram_size: Dict[int, FrozenSet[str]] = {}

    for ngram_size in configuration.ngram_sizes:
        occurrence_counts, dispersion_counts = _count_ngrams_with_dispersion(
            dispersion_units, ngram_size
        )
        surviving: Set[str] = set()
        for phrase, occurrence_count in occurrence_counts.items():
            # Cheap rejections first: the arithmetic below is the expensive part.
            if occurrence_count < minimum_occurrence_count:
                continue
            if dispersion_counts[phrase] < minimum_dispersion_unit_count:
                continue
            if not phrase_is_well_formed(phrase):
                continue
            phrase_tokens = phrase.split()
            if phrase_is_entirely_grammatical(phrase_tokens):
                continue
            if any(token in probable_proper_nouns for token in phrase_tokens):
                continue

            expected_relative_frequency = (
                reference_language_model.expected_relative_frequency(phrase_tokens)
            )
            reference_expected_count = (
                expected_relative_frequency * reference_ngram_total
            )
            target_relative_frequency = occurrence_count / target_token_total
            # Only over-representation is interesting; a phrase the reference
            # uses MORE than the target says nothing about this person's style.
            if target_relative_frequency <= expected_relative_frequency:
                continue

            dunning = dunning_log_likelihood(
                occurrence_count,
                reference_expected_count,
                target_token_total,
                reference_ngram_total,
            )
            if dunning < configuration.dunning_log_likelihood_threshold:
                continue

            scored_candidates.append(
                KeyPhraseCandidate(
                    phrase=phrase,
                    ngram_size=ngram_size,
                    occurrence_count=occurrence_count,
                    dispersion_unit_count=dispersion_counts[phrase],
                    target_relative_frequency=target_relative_frequency,
                    reference_expected_count=reference_expected_count,
                    dunning_log_likelihood=dunning,
                    log_ratio_over_reference=log_ratio_over_reference(
                        occurrence_count,
                        reference_expected_count,
                        target_token_total,
                        reference_ngram_total,
                        smoothing_count=configuration.smoothing_count,
                    ),
                )
            )
            surviving.add(phrase)
        surviving_phrases_by_ngram_size[ngram_size] = frozenset(surviving)

    if not scored_candidates:
        return []

    # Rank by effect size. The trailing phrase text makes ties resolve the same
    # way on every run, which is what keeps the stored profile stable.
    scored_candidates.sort(
        key=lambda candidate: (
            -candidate.log_ratio_over_reference,
            -candidate.occurrence_count,
            candidate.phrase,
        )
    )
    protected_phrases = protected_phrases or frozenset()
    pooled = scored_candidates[: configuration.candidate_pool_size]
    pooled_phrases = {candidate.phrase for candidate in pooled}
    # Re-admit any incumbent that cleared every filter but fell outside the pool
    # purely on rank, keeping the overall ordering intact.
    reinstated = [
        candidate
        for candidate in scored_candidates[configuration.candidate_pool_size :]
        if candidate.phrase in protected_phrases
        and candidate.phrase not in pooled_phrases
    ]
    if reinstated:
        pooled = sorted(
            pooled + reinstated,
            key=lambda candidate: (
                -candidate.log_ratio_over_reference,
                -candidate.occurrence_count,
                candidate.phrase,
            ),
        )
    scored_candidates = pooled

    mutable_pool_phrases: Dict[int, Set[str]] = {}
    for candidate in scored_candidates:
        mutable_pool_phrases.setdefault(candidate.ngram_size, set()).add(
            candidate.phrase
        )
    pool_phrases_by_ngram_size: Dict[int, FrozenSet[str]] = {
        ngram_size: frozenset(phrases)
        for ngram_size, phrases in mutable_pool_phrases.items()
    }
    occurrence_positions = _collect_occurrence_positions(
        dispersion_units, pool_phrases_by_ngram_size
    )

    collapsed = collapse_subsumed_candidates(
        scored_candidates,
        occurrence_positions,
        containment_threshold=configuration.subsumption_containment_threshold,
    )
    shortlisted = collapsed[: configuration.shortlist_size]
    shortlisted_phrases = {candidate.phrase for candidate in shortlisted}
    shortlisted += [
        candidate
        for candidate in collapsed[configuration.shortlist_size :]
        if candidate.phrase in protected_phrases
        and candidate.phrase not in shortlisted_phrases
    ]
    collapsed = shortlisted

    return [
        KeyPhraseCandidate(
            phrase=candidate.phrase,
            ngram_size=candidate.ngram_size,
            occurrence_count=candidate.occurrence_count,
            dispersion_unit_count=candidate.dispersion_unit_count,
            target_relative_frequency=candidate.target_relative_frequency,
            reference_expected_count=candidate.reference_expected_count,
            dunning_log_likelihood=candidate.dunning_log_likelihood,
            log_ratio_over_reference=candidate.log_ratio_over_reference,
            concordance_lines=build_concordance_lines(
                candidate.phrase,
                dispersion_units,
                occurrence_positions.get(candidate.phrase, []),
                line_count=configuration.concordance_line_count,
                context_words=configuration.concordance_context_words,
            ),
        )
        for candidate in collapsed
    ]
