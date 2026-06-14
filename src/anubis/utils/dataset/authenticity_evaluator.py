"""Authenticity evaluator — place a candidate text on the style axis.

    CHATGPT —— AVATAR —— REAL PERSON

We want the avatar's writing to be **dissimilar** from the base ChatGPT voice and
**similar** to the real person's primary-source writing. This module quantifies
that with the **Mahalanobis distance** of a candidate text to two reference
clouds, each described by a flat feature-matrix profile built once by
:func:`src.anubis.utils.dataset.stylistic_profile.compute_feature_matrix_profile`:

* the **baseline** profile  — the generic ChatGPT voice (bundled artifact); and
* the **ground-truth** profile — the real person's quotes (per-avatar, optional;
  may be absent before any primary-source quotes have been ingested).

For each cloud we report the actual distance, the cloud's outlier threshold, and
whether the candidate is an outlier to that cloud — so a caller knows, in raw
numbers, how far the text sits from ChatGPT and from the real person. Burrows
Delta is computed as a secondary, independent authorship signal.

The corpus itself is NEVER touched here: we read only the precomputed profiles.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from src.anubis.utils.dataset.burrows_delta import burrows_delta_against_reference
from src.anubis.utils.dataset.style_features import (
    FEATURE_NAMES,
    extract_style_features,
)

logger = logging.getLogger(__name__)


def _is_mahalanobis_ready(profile: Optional[Dict[str, Any]]) -> bool:
    """Return True only if a profile carries the matrices Mahalanobis needs."""
    return bool(
        profile
        and profile.get("inverse_covariance_matrix")
        and profile.get("normalized_feature_mean")
        and profile.get("normalization_min")
        and profile.get("normalization_max")
    )


def _normalize_candidate_vector(
    raw_features: Dict[str, float], profile: Dict[str, Any]
) -> List[float]:
    """Apply the profile's stored min-max 0-1 scaling to the candidate.

    Missing/NaN candidate features are imputed with the profile's *raw* feature
    mean, which after scaling lands near the cloud centroid — a neutral position
    that neither inflates nor deflates the distance.
    """
    feature_min = profile["normalization_min"]
    feature_max = profile["normalization_max"]
    feature_means = profile.get("feature_means", {})

    normalized: List[float] = []
    for name in FEATURE_NAMES:
        value = raw_features.get(name, math.nan)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            value = float(feature_means.get(name, feature_min.get(name, 0.0)))
        lower = float(feature_min.get(name, 0.0))
        upper = float(feature_max.get(name, lower))
        span = upper - lower
        scaled = (value - lower) / span if span != 0.0 else 0.0
        normalized.append(scaled)
    return normalized


def _mahalanobis_distance_to_cloud(
    candidate_normalized: List[float], profile: Dict[str, Any]
) -> float:
    """Mahalanobis distance from the candidate to a profile's cloud centroid.

    Uses the cloud's stored mean and pseudo-inverse covariance, so the distance
    is in covariance-scaled standard deviations and accounts for the correlation
    between same-family features.
    """
    import numpy as np
    from scipy.spatial.distance import mahalanobis

    centroid = np.asarray(
        [profile["normalized_feature_mean"][name] for name in FEATURE_NAMES],
        dtype=float,
    )
    inverse_covariance = np.asarray(profile["inverse_covariance_matrix"], dtype=float)
    point = np.asarray(candidate_normalized, dtype=float)
    return float(mahalanobis(point, centroid, inverse_covariance))


def _distance_report(
    raw_features: Dict[str, float], profile: Optional[Dict[str, Any]], cloud_label: str
) -> Optional[Dict[str, Any]]:
    """Build the distance + outlier verdict for one reference cloud.

    Returns ``None`` if the profile can't support Mahalanobis (e.g. it was built
    from too few documents to estimate a covariance).
    """
    if not _is_mahalanobis_ready(profile):
        return None
    assert profile is not None  # narrowed by _is_mahalanobis_ready above

    normalized = _normalize_candidate_vector(raw_features, profile)
    distance = _mahalanobis_distance_to_cloud(normalized, profile)
    threshold = profile.get("mahalanobis_outlier_threshold")

    return {
        "cloud": cloud_label,
        "mahalanobis_distance": distance,
        "outlier_threshold": threshold,
        "is_outlier": (
            bool(distance > threshold) if threshold is not None else None
        ),
        "chi_squared_consistent_feature_count": profile.get(
            "chi_squared_consistent_feature_count"
        ),
        "feature_dimensionality": len(FEATURE_NAMES),
    }


def evaluate_authenticity_against_profile(
    candidate_text: str,
    baseline_profile: Optional[Dict[str, Any]],
    ground_truth_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score one candidate text against the baseline and ground-truth clouds.

    Parameters
    ----------
    candidate_text:
        The avatar-generated text to position on the style axis.
    baseline_profile:
        Flat feature-matrix profile of the ChatGPT baseline voice (the cloud we
        want to be UNLIKE). ``None`` / empty disables baseline distance.
    ground_truth_profile:
        Flat feature-matrix profile of the real person's primary-source writing
        (the cloud we want to be LIKE). Optional — absent until quotes exist.

    Returns:
        A self-commenting report dict. Distances are ``None`` when the
        corresponding profile is missing or too small for a covariance.
    """
    raw_features = extract_style_features(candidate_text or "")

    baseline_report = _distance_report(raw_features, baseline_profile, "chatgpt_baseline")
    ground_truth_report = _distance_report(
        raw_features, ground_truth_profile, "ground_truth_real_person"
    )

    # Burrows Delta (secondary signal) straight from the candidate text against
    # each profile's stored reference distribution.
    def _burrows(profile: Optional[Dict[str, Any]]) -> float:
        reference = (profile or {}).get("reference_distribution")
        if not reference:
            return math.nan
        delta, _contributions = burrows_delta_against_reference(candidate_text or "", reference)
        return delta

    distance_to_chatgpt_baseline = (
        baseline_report["mahalanobis_distance"] if baseline_report else None
    )
    distance_to_ground_truth = (
        ground_truth_report["mahalanobis_distance"] if ground_truth_report else None
    )

    # Axis verdict: we want the candidate CLOSER to the real person than to
    # ChatGPT. Only decidable when both distances exist.
    if distance_to_chatgpt_baseline is not None and distance_to_ground_truth is not None:
        if distance_to_ground_truth < distance_to_chatgpt_baseline:
            axis_verdict = "closer to REAL person"
        else:
            axis_verdict = "still closer to ChatGPT"
    else:
        axis_verdict = "undetermined (need both baseline and ground-truth profiles)"

    return {
        "candidate_feature_values": raw_features,
        "distance_to_chatgpt_baseline": distance_to_chatgpt_baseline,
        "distance_to_ground_truth": distance_to_ground_truth,
        "baseline_outlier_threshold": (
            baseline_report["outlier_threshold"] if baseline_report else None
        ),
        "ground_truth_outlier_threshold": (
            ground_truth_report["outlier_threshold"] if ground_truth_report else None
        ),
        "is_outlier_to_chatgpt_baseline": (
            baseline_report["is_outlier"] if baseline_report else None
        ),
        "is_outlier_to_ground_truth": (
            ground_truth_report["is_outlier"] if ground_truth_report else None
        ),
        "burrows_delta_to_baseline": _burrows(baseline_profile),
        "burrows_delta_to_ground_truth": _burrows(ground_truth_profile),
        # How many of the 33 features are chi-squared-consistent in each cloud
        # (i.e. how trustworthy the parametric Mahalanobis assumption is).
        "baseline_chi_squared_consistent_feature_count": (
            (baseline_profile or {}).get("chi_squared_consistent_feature_count")
        ),
        "ground_truth_chi_squared_consistent_feature_count": (
            (ground_truth_profile or {}).get("chi_squared_consistent_feature_count")
        ),
        "feature_dimensionality": len(FEATURE_NAMES),
        "axis_verdict": axis_verdict,
    }
