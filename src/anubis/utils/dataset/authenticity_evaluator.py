"""Authenticity evaluator: compares a candidate response against the stored stylistic profile.

The evaluator NEVER reads the corpus directly. It loads the stylistic
profile that :mod:`build_profile` wrote to the LangGraph store and compares
the candidate's same-shape feature dict against the stored aggregate
features and the stored Burrows Delta reference distribution.

Outputs a dict containing:

* ``burrows_delta`` — float, lower = closer to target author.
* ``authorship_score`` — bounded [0, 1] mapping of Delta (1 - tanh(Delta)).
* ``feature_distances`` — per-family numeric distances (sentence length
  z-score, lexical jaccard against top unigrams, syntax POS divergence,
  style rate gaps).
* ``per_feature_diffs`` — raw arithmetic diffs the operator can audit.
* ``overall_authenticity`` — equally weighted mean of the bounded scores
  for the families we have features for.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from src.anubis.utils.dataset.stylistic_profile import compute_features_for_text


def _safe(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _pos_kl_like_distance(
    candidate_pos: Dict[str, float], reference_pos: Dict[str, float]
) -> float:
    """Symmetric KL-flavoured distance over POS ratio dicts."""
    keys = set(candidate_pos) | set(reference_pos)
    if not keys:
        return 0.0
    eps = 1e-9
    total = 0.0
    for k in keys:
        p = max(candidate_pos.get(k, 0.0), eps)
        q = max(reference_pos.get(k, 0.0), eps)
        total += (p - q) * math.log(p / q)
    return total / 2.0


def evaluate_authenticity_against_profile(
    candidate_text: str,
    stylistic_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """Score ``candidate_text`` against the stored stylistic profile."""
    if not stylistic_profile:
        return {
            "status": "skipped",
            "reason": "no_stylistic_profile_available",
        }

    reference = stylistic_profile.get("reference_distribution") or {}
    aggregate = stylistic_profile.get("features") or {}
    moments = stylistic_profile.get("per_document_moments") or {}

    candidate_features = compute_features_for_text(
        candidate_text, reference_distribution=reference
    )

    delta = _safe(
        candidate_features.get("burrows_delta", {}).get("delta"), default=float("nan")
    )
    authorship_score = (
        1.0 - math.tanh(delta) if not math.isnan(delta) else 0.0
    )
    authorship_score = max(0.0, min(1.0, authorship_score))

    candidate_sl = _safe(
        candidate_features.get("sentence", {}).get("avg_sentence_length_tokens")
    )
    sl_mean = _safe(moments.get("avg_sentence_length_tokens", {}).get("mean"))
    sl_stdev = _safe(moments.get("avg_sentence_length_tokens", {}).get("stdev"))
    sl_z = (candidate_sl - sl_mean) / sl_stdev if sl_stdev > 0 else 0.0
    sl_score = max(0.0, 1.0 - min(abs(sl_z) / 3.0, 1.0))

    candidate_top_unigrams = [
        item[0]
        for item in (
            candidate_features.get("lexical", {}).get("top_unigrams") or []
        )
    ]
    reference_top_unigrams = [
        item[0]
        for item in (aggregate.get("lexical", {}).get("top_unigrams") or [])
    ]
    lex_jaccard = _jaccard(
        candidate_top_unigrams[:50], reference_top_unigrams[:50]
    )

    cand_pos = candidate_features.get("syntax", {}).get("pos_ratios") or {}
    ref_pos = aggregate.get("syntax", {}).get("pos_ratios") or {}
    pos_distance = _pos_kl_like_distance(cand_pos, ref_pos)
    syntax_score = max(0.0, 1.0 - min(pos_distance, 1.0))

    cand_style = candidate_features.get("style", {})
    ref_style = aggregate.get("style", {})
    style_rate_keys = (
        "contraction_rate_per_1k",
        "hedge_rate_per_1k",
        "intensifier_rate_per_1k",
        "modal_rate_per_1k",
    )
    rate_gaps = []
    for key in style_rate_keys:
        c = _safe(cand_style.get(key))
        r = _safe(ref_style.get(key))
        rate_gaps.append(abs(c - r))
    avg_rate_gap = sum(rate_gaps) / len(rate_gaps) if rate_gaps else 0.0
    style_score = max(0.0, 1.0 - min(avg_rate_gap / 50.0, 1.0))

    component_scores = [
        authorship_score,
        sl_score,
        lex_jaccard,
        syntax_score,
        style_score,
    ]
    overall = sum(component_scores) / len(component_scores)

    return {
        "status": "ok",
        "burrows_delta": delta,
        "authorship_score": authorship_score,
        "sentence_length_z": sl_z,
        "sentence_length_score": sl_score,
        "lexical_top50_jaccard": lex_jaccard,
        "pos_kl_distance": pos_distance,
        "syntax_score": syntax_score,
        "style_avg_rate_gap": avg_rate_gap,
        "style_score": style_score,
        "overall_authenticity": overall,
        "candidate_features_summary": {
            "lexical": {
                "type_token_ratio": _safe(
                    candidate_features.get("lexical", {}).get("type_token_ratio")
                ),
            },
            "sentence": candidate_features.get("sentence", {}),
            "style": candidate_features.get("style", {}),
            "syntax": {
                "engine": candidate_features.get("syntax", {}).get("engine"),
                "avg_dependency_distance": _safe(
                    candidate_features.get("syntax", {}).get("avg_dependency_distance")
                ),
                "passive_voice_rate": _safe(
                    candidate_features.get("syntax", {}).get("passive_voice_rate")
                ),
            },
            "burrows_top_contributions": (
                candidate_features.get("burrows_delta", {}).get("top_contributions") or {}
            ),
        },
    }


async def evaluate_authenticity(
    *,
    candidate_text: str,
    creator_id: str,
    assistant_id: str,
    store,
) -> Dict[str, Any]:
    """Convenience wrapper that loads the stored profile and scores ``candidate_text``."""
    from src.anubis.utils.dataset.build_profile import load_stylistic_profile

    profile: Optional[Dict[str, Any]] = await load_stylistic_profile(
        creator_id=creator_id, assistant_id=assistant_id, store=store
    )
    return evaluate_authenticity_against_profile(candidate_text, profile or {})
