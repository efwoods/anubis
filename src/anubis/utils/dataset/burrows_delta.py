"""Custom Burrows Delta for authorship attribution.

Two-phase API so the corpus is touched exactly once:

* :func:`build_reference_distribution` — runs over the full corpus of target
  quotes and produces a JSON-serialisable reference profile (top-N most
  common words, their relative frequencies, mean and stdev across documents).

* :func:`burrows_delta_against_reference` — at evaluation time, takes a
  candidate text and the precomputed reference profile and returns the Delta
  score (lower is closer to the reference author). Never re-reads the
  corpus.

References:

* https://github.com/fastdatascience/faststylometry/blob/main/Burrows%20Delta%20Walkthrough.ipynb
* Burrows, J. (2002). Delta: A measure of stylistic difference and a guide
  to likely authorship. Literary and Linguistic Computing, 17(3), 267-287.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple

_DEFAULT_TOP_N = 150
_TOKEN_RE = re.compile(r"[A-Za-z']+")


def tokenize(text: str) -> List[str]:
    """Lowercase alpha tokens; apostrophes preserved (function words like ``it's``)."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _document_word_distribution(text: str, vocabulary: List[str]) -> Dict[str, float]:
    """Relative-frequency vector of ``vocabulary`` over ``text``."""
    tokens = tokenize(text)
    n = len(tokens) or 1
    counts = Counter(tokens)
    return {w: counts.get(w, 0) / n for w in vocabulary}


def build_reference_distribution(
    corpus_documents: Iterable[str],
    top_n: int = _DEFAULT_TOP_N,
) -> Dict[str, object]:
    """Compute and serialise the corpus-side reference distribution.

    Output shape (JSON-serialisable):

    .. code-block:: json

        {
            "top_n": 150,
            "vocabulary": ["the", "and", ...],
            "mean_relative_frequencies": {"the": 0.05, ...},
            "stdev_relative_frequencies": {"the": 0.01, ...},
            "document_count": 27
        }

    The Delta function is z-score-based, so we need the per-word mean and
    standard deviation across reference documents.
    """
    docs: List[str] = [d for d in corpus_documents if (d or "").strip()]
    if not docs:
        return {
            "top_n": top_n,
            "vocabulary": [],
            "mean_relative_frequencies": {},
            "stdev_relative_frequencies": {},
            "document_count": 0,
        }

    aggregate_counts: Counter = Counter()
    for doc in docs:
        aggregate_counts.update(tokenize(doc))
    vocabulary: List[str] = [
        word for word, _ in aggregate_counts.most_common(top_n)
    ]

    per_doc_distributions: List[Dict[str, float]] = [
        _document_word_distribution(doc, vocabulary) for doc in docs
    ]

    mean_freq: Dict[str, float] = {}
    stdev_freq: Dict[str, float] = {}
    n_docs = len(per_doc_distributions)
    for word in vocabulary:
        values = [d[word] for d in per_doc_distributions]
        mean = sum(values) / n_docs
        if n_docs > 1:
            variance = sum((v - mean) ** 2 for v in values) / (n_docs - 1)
        else:
            variance = 0.0
        mean_freq[word] = mean
        stdev_freq[word] = math.sqrt(variance) if variance > 0 else 0.0

    return {
        "top_n": top_n,
        "vocabulary": vocabulary,
        "mean_relative_frequencies": mean_freq,
        "stdev_relative_frequencies": stdev_freq,
        "document_count": n_docs,
    }


def burrows_delta_against_reference(
    candidate_text: str,
    reference_distribution: Dict[str, object],
) -> Tuple[float, Dict[str, float]]:
    """Compute Burrows Delta of ``candidate_text`` vs. a stored reference.

    Implementation:

    1. Project the candidate onto the same vocabulary the reference used.
    2. For each vocabulary word, compute the candidate's z-score using the
       reference's mean/stdev.
    3. Delta = mean of absolute z-scores. Lower = more like the reference
       author.

    Returns ``(delta, contribution_per_word)`` so callers can highlight which
    words drove the score.
    """
    vocabulary: List[str] = list(reference_distribution.get("vocabulary") or [])
    means: Dict[str, float] = reference_distribution.get("mean_relative_frequencies") or {}
    stds: Dict[str, float] = reference_distribution.get("stdev_relative_frequencies") or {}

    if not vocabulary:
        return float("nan"), {}

    cand_dist = _document_word_distribution(candidate_text, vocabulary)
    contributions: Dict[str, float] = {}
    abs_z_total = 0.0
    counted = 0
    for word in vocabulary:
        sigma = stds.get(word, 0.0) or 0.0
        if sigma <= 0:
            continue
        z = (cand_dist[word] - means.get(word, 0.0)) / sigma
        contributions[word] = z
        abs_z_total += abs(z)
        counted += 1

    delta = abs_z_total / counted if counted else float("nan")
    return delta, contributions
