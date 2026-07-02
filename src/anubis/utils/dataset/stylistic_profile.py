"""Per-avatar stylistic profile builder + per-text feature extractor.

The build phase runs over the corpus of direct quotes ONCE
(:func:`compute_profile_from_quotes`) and produces a JSON-serialisable
profile dict. The eval phase extracts the same feature shape for a single
candidate text (:func:`compute_features_for_text`) so the authenticity
evaluator can compare two same-shape feature dicts without ever touching the
corpus again.

Feature families implemented today:

* Authorship reference distribution (Burrows Delta inputs).
* Lexicon: vocabulary set with frequencies, distinctive bigrams/trigrams,
  TF-IDF-ish keyness against a stored English unigram baseline.
* Sentence: average length in tokens, length distribution moments,
  declarative/interrogative/exclamatory ratios.
* Syntax (best-effort): POS ratios, average parse-tree depth, clausal
  density, passive voice rate, T-units per sentence, dependency distance —
  computed via spaCy if available, otherwise NLTK fallback (POS only).
* Style: contraction rate, hedge / intensifier / modal counts, punctuation
  rates per 1k tokens.
* Prosody (basic): mean syllables per word, stress pattern entropy via
  ``pronouncing`` if available.
* Consistency: sliding-window chi-squared of word distribution at build
  time only; the candidate-text stage skips this metric (single text).

All features stored numerically so JSON round-trips are lossless.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List

from src.anubis.utils.dataset.burrows_delta import (
    build_reference_distribution,
    tokenize,
)

logger = logging.getLogger(__name__)

# Mahalanobis-distance-squared is chi-squared distributed (with degrees of
# freedom = number of features) IF the features are multivariate normal. A point
# is flagged a distributional outlier when its distance exceeds the radius at
# this upper-tail probability.
_OUTLIER_TAIL_PROBABILITY = 0.975
# Per-feature KS goodness-of-fit p-value above which we DON'T reject the
# chi-squared hypothesis for that feature (i.e. the Mahalanobis assumption is
# credible for it).
_CHI_SQUARED_FIT_ALPHA = 0.05


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PUNCT_TO_TRACK = ".,;:!?-—\"'()[]"
_HEDGES = frozenset(
    {
        "maybe", "perhaps", "possibly", "probably", "likely", "seems",
        "appears", "somewhat", "kind", "sort", "i think", "i guess",
        "i suppose",
    }
)
_INTENSIFIERS = frozenset(
    {
        "very", "really", "extremely", "totally", "absolutely", "completely",
        "incredibly", "literally", "super",
    }
)
_MODALS = frozenset(
    {"can", "could", "may", "might", "must", "shall", "should", "will",
     "would", "ought"}
)
_CONTRACTION_RE = re.compile(
    r"\b\w+(?:'s|'re|'ve|'ll|'d|'m|n't)\b", flags=re.IGNORECASE
)


""" ------------------------------------------------------------------ """
""" Helpers                                                            """
""" ------------------------------------------------------------------ """


def _split_sentences(text: str) -> List[str]:
    """Cheap sentence split that doesn't require a model download."""
    if not text:
        return []
    raw = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return raw or [text.strip()]


def _ngram_counts(tokens: List[str], n: int, top_k: int) -> List[List[Any]]:
    counts: Counter = Counter()
    if len(tokens) < n:
        return []
    for i in range(len(tokens) - n + 1):
        counts[" ".join(tokens[i : i + n])] += 1
    return [list(item) for item in counts.most_common(top_k)]


def _try_spacy():
    """Return a loaded spaCy pipeline or ``None`` if unavailable."""
    try:
        import spacy

        try:
            return spacy.load("en_core_web_sm")
        except Exception:
            try:
                from spacy.cli.download import download as _spacy_download

                _spacy_download("en_core_web_sm")
                return spacy.load("en_core_web_sm")
            except Exception:
                return None
    except Exception:
        return None


def _try_pronouncing():
    """Return the ``pronouncing`` module or ``None`` if unavailable."""
    try:
        import pronouncing

        return pronouncing
    except Exception:
        return None


def _safe_mean(values: Iterable[float]) -> float:
    vs = list(values)
    return sum(vs) / len(vs) if vs else 0.0


def _safe_stdev(values: Iterable[float], mean: float) -> float:
    vs = list(values)
    if len(vs) < 2:
        return 0.0
    return math.sqrt(sum((v - mean) ** 2 for v in vs) / (len(vs) - 1))


""" ------------------------------------------------------------------ """
""" Feature families                                                   """
""" ------------------------------------------------------------------ """


def _lexical_features(text: str, top_k_ngrams: int = 50) -> Dict[str, Any]:
    tokens = tokenize(text)
    token_count = len(tokens)
    types = set(tokens)
    type_count = len(types)
    type_token_ratio = type_count / token_count if token_count else 0.0
    word_freq = Counter(tokens)
    return {
        "token_count": token_count,
        "type_count": type_count,
        "type_token_ratio": type_token_ratio,
        "top_unigrams": [list(item) for item in word_freq.most_common(top_k_ngrams)],
        "top_bigrams": _ngram_counts(tokens, 2, top_k_ngrams),
        "top_trigrams": _ngram_counts(tokens, 3, top_k_ngrams),
    }


def _sentence_features(text: str) -> Dict[str, Any]:
    sentences = _split_sentences(text)
    lengths = [len(tokenize(s)) for s in sentences]
    decl = sum(1 for s in sentences if s.endswith("."))
    interrog = sum(1 for s in sentences if s.endswith("?"))
    excl = sum(1 for s in sentences if s.endswith("!"))
    mean_len = _safe_mean(lengths)
    return {
        "sentence_count": len(sentences),
        "avg_sentence_length_tokens": mean_len,
        "stdev_sentence_length_tokens": _safe_stdev(lengths, mean_len),
        "declarative_ratio": decl / len(sentences) if sentences else 0.0,
        "interrogative_ratio": interrog / len(sentences) if sentences else 0.0,
        "exclamatory_ratio": excl / len(sentences) if sentences else 0.0,
    }


def _style_features(text: str) -> Dict[str, Any]:
    tokens = tokenize(text)
    n = len(tokens) or 1

    contraction_count = len(_CONTRACTION_RE.findall(text or ""))
    hedge_count = sum(1 for t in tokens if t in _HEDGES)
    intensifier_count = sum(1 for t in tokens if t in _INTENSIFIERS)
    modal_count = sum(1 for t in tokens if t in _MODALS)

    punct_counts = {
        ch: (text or "").count(ch) for ch in _PUNCT_TO_TRACK
    }
    punct_rate_per_1k = {
        ch: (count / n) * 1000.0 for ch, count in punct_counts.items()
    }

    return {
        "contraction_rate_per_1k": (contraction_count / n) * 1000.0,
        "hedge_rate_per_1k": (hedge_count / n) * 1000.0,
        "intensifier_rate_per_1k": (intensifier_count / n) * 1000.0,
        "modal_rate_per_1k": (modal_count / n) * 1000.0,
        "punctuation_rate_per_1k": punct_rate_per_1k,
    }


def _syntax_features(text: str) -> Dict[str, Any]:
    nlp = _try_spacy()
    if nlp is None:
        # NLTK fallback: POS only.
        try:
            import nltk
            from nltk import pos_tag, word_tokenize

            try:
                tags = pos_tag(word_tokenize(text or ""))
            except LookupError:
                nltk.download("punkt", quiet=True)
                nltk.download("averaged_perceptron_tagger", quiet=True)
                tags = pos_tag(word_tokenize(text or ""))
            tag_counts: Counter = Counter(tag for _, tag in tags)
            total = sum(tag_counts.values()) or 1
            return {
                "pos_ratios": {tag: c / total for tag, c in tag_counts.items()},
                "avg_parse_tree_depth": None,
                "clausal_density": None,
                "passive_voice_rate": None,
                "tunits_per_sentence": None,
                "avg_dependency_distance": None,
                "engine": "nltk",
            }
        except Exception as exc:
            logger.warning("Syntax fallback (NLTK) failed: %s", exc)
            return {
                "pos_ratios": {},
                "avg_parse_tree_depth": None,
                "clausal_density": None,
                "passive_voice_rate": None,
                "tunits_per_sentence": None,
                "avg_dependency_distance": None,
                "engine": "none",
            }

    doc = nlp(text or "")
    pos_counts: Counter = Counter(tok.pos_ for tok in doc if not tok.is_space)
    total_tokens = sum(pos_counts.values()) or 1
    pos_ratios = {pos: c / total_tokens for pos, c in pos_counts.items()}

    sentences = list(doc.sents)
    parse_depths: List[int] = []
    clauses_per_sent: List[int] = []
    tunits_per_sent: List[int] = []
    passive_count = 0
    dep_distances: List[int] = []

    for sent in sentences:
        head_for_root = sent.root

        def _depth(token, _h=head_for_root):
            d = 0
            cur = token
            while cur.head is not cur and cur.head is not _h:
                cur = cur.head
                d += 1
                if d > 1000:
                    break
            return d

        depths = [_depth(tok) for tok in sent if not tok.is_space]
        if depths:
            parse_depths.append(max(depths))
        clauses = sum(
            1 for tok in sent
            if tok.dep_ in {"ccomp", "xcomp", "advcl", "relcl", "acl", "csubj"}
        ) + 1
        clauses_per_sent.append(clauses)
        tunits = sum(
            1 for tok in sent if tok.dep_ in {"ROOT", "conj"}
        ) or 1
        tunits_per_sent.append(tunits)
        if any(tok.dep_ == "auxpass" or tok.dep_ == "nsubjpass" for tok in sent):
            passive_count += 1
        for tok in sent:
            if tok.head is not None and tok.head is not tok:
                dep_distances.append(abs(tok.i - tok.head.i))

    return {
        "pos_ratios": pos_ratios,
        "avg_parse_tree_depth": _safe_mean(parse_depths),
        "clausal_density": _safe_mean(clauses_per_sent),
        "passive_voice_rate": (passive_count / len(sentences)) if sentences else 0.0,
        "tunits_per_sentence": _safe_mean(tunits_per_sent),
        "avg_dependency_distance": _safe_mean(dep_distances),
        "engine": "spacy",
    }


def _prosody_features(text: str) -> Dict[str, Any]:
    pronouncing = _try_pronouncing()
    if pronouncing is None:
        return {
            "mean_syllables_per_word": None,
            "stress_pattern_entropy": None,
            "engine": "none",
        }

    tokens = tokenize(text)
    syll_counts: List[int] = []
    stress_patterns: List[str] = []
    for tok in tokens:
        phones_list = pronouncing.phones_for_word(tok)
        if not phones_list:
            continue
        phones = phones_list[0]
        syll_counts.append(pronouncing.syllable_count(phones))
        stress_patterns.append(pronouncing.stresses(phones))

    if not stress_patterns:
        return {
            "mean_syllables_per_word": None,
            "stress_pattern_entropy": None,
            "engine": "pronouncing",
        }

    counts: Counter = Counter(stress_patterns)
    total = sum(counts.values()) or 1
    entropy = -sum(
        (c / total) * math.log((c / total) + 1e-12) for c in counts.values()
    )
    return {
        "mean_syllables_per_word": _safe_mean(syll_counts),
        "stress_pattern_entropy": entropy,
        "engine": "pronouncing",
    }


def _consistency_features(corpus_documents: List[str]) -> Dict[str, Any]:
    """Sliding-window chi-squared drift across the corpus (build-time only)."""
    if len(corpus_documents) < 4:
        return {"sliding_window_chi_squared": None, "windows_evaluated": 0}

    try:
        from scipy.stats import chisquare
    except Exception:
        return {"sliding_window_chi_squared": None, "windows_evaluated": 0}

    midpoint = len(corpus_documents) // 2
    first_half_tokens = [t for d in corpus_documents[:midpoint] for t in tokenize(d)]
    second_half_tokens = [t for d in corpus_documents[midpoint:] for t in tokenize(d)]
    fh_counts = Counter(first_half_tokens)
    sh_counts = Counter(second_half_tokens)
    shared_vocab = [w for w in fh_counts if fh_counts[w] >= 5 and sh_counts.get(w, 0) >= 5]
    if not shared_vocab:
        return {"sliding_window_chi_squared": None, "windows_evaluated": 0}

    observed = [sh_counts[w] for w in shared_vocab]
    expected_total = sum(observed)
    fh_total = sum(fh_counts[w] for w in shared_vocab) or 1
    expected = [
        max(1.0, fh_counts[w] / fh_total * expected_total) for w in shared_vocab
    ]
    try:
        stat, p = chisquare(f_obs=observed, f_exp=expected)
    except Exception:
        return {"sliding_window_chi_squared": None, "windows_evaluated": 0}
    return {
        "sliding_window_chi_squared": float(stat),
        "p_value": float(p),
        "windows_evaluated": 2,
        "shared_vocab_size": len(shared_vocab),
    }


""" ------------------------------------------------------------------ """
""" Public API                                                         """
""" ------------------------------------------------------------------ """


def compute_features_for_text(
    text: str,
    *,
    reference_distribution: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Extract the same feature shape from a single candidate text.

    The returned dict can be diffed against the corpus-side
    ``stylistic_profile['features']`` to produce per-feature distance
    scores in the authenticity evaluator. Burrows Delta is computed here
    against the supplied ``reference_distribution`` (loaded once from the
    stored profile).
    """
    features: Dict[str, Any] = {
        "lexical": _lexical_features(text),
        "sentence": _sentence_features(text),
        "style": _style_features(text),
        "syntax": _syntax_features(text),
        "prosody": _prosody_features(text),
    }

    if reference_distribution:
        from src.anubis.utils.dataset.burrows_delta import (
            burrows_delta_against_reference,
        )

        delta, contributions = burrows_delta_against_reference(
            text, reference_distribution
        )
        features["burrows_delta"] = {
            "delta": delta,
            "top_contributions": dict(
                sorted(
                    contributions.items(), key=lambda kv: abs(kv[1]), reverse=True
                )[:25]
            ),
        }
    return features


def compute_profile_from_quotes(
    quote_documents: List[str],
    *,
    target_name: str | None = None,
) -> Dict[str, Any]:
    """Build the stylistic profile JSON from a corpus of direct quotes.

    Returns a dict that contains both the aggregate ``features`` (averaged
    across the corpus where averaging makes sense) and the per-document
    distribution moments needed by the evaluator. The ``reference``
    sub-dict is the Burrows Delta reference profile.
    """
    quotes = [q for q in quote_documents if (q or "").strip()]
    document_count = len(quotes)
    aggregate_text = "\n".join(quotes)

    # Note: key-phrase discovery no longer happens here. Signature key phrases are
    # discovered, stored, and prompt-injected separately by calibrate_ground_truth
    # (the (creator_id, assistant_id, "key_phrase") vectorstore + key_phrase_profile
    # blob), and their scalar summary rides in style_features' key_phrase_rate. The
    # character n-gram and function-word vectors were dropped entirely (capture-only,
    # never scored).
    aggregate_features = {
        "lexical": _lexical_features(aggregate_text, top_k_ngrams=200),
        "sentence": _sentence_features(aggregate_text),
        "style": _style_features(aggregate_text),
        "syntax": _syntax_features(aggregate_text),
        "prosody": _prosody_features(aggregate_text),
        "consistency": _consistency_features(quotes),
    }

    reference = build_reference_distribution(quotes)

    # Per-document moments for the Burrows-style features the evaluator
    # consumes most often (mean / stdev of sentence length, TTR, etc.).
    sentence_lengths: List[float] = []
    ttrs: List[float] = []
    for q in quotes:
        sentence_features = _sentence_features(q)
        sentence_lengths.append(
            sentence_features.get("avg_sentence_length_tokens", 0.0) or 0.0
        )
        ttrs.append(_lexical_features(q)["type_token_ratio"])
    sl_mean = _safe_mean(sentence_lengths)
    ttr_mean = _safe_mean(ttrs)
    moments = {
        "avg_sentence_length_tokens": {
            "mean": sl_mean,
            "stdev": _safe_stdev(sentence_lengths, sl_mean),
        },
        "type_token_ratio": {
            "mean": ttr_mean,
            "stdev": _safe_stdev(ttrs, ttr_mean),
        },
    }

    return {
        "version": 1,
        "target_name": target_name,
        "document_count": document_count,
        "features": aggregate_features,
        "reference_distribution": reference,
        "per_document_moments": moments,
    }


