"""Layer C: learning — how measurements accumulate, and what gets kept.

A signature is a per-measurement running mean **with variance and sample
count**, weighted by the seconds of motion each reading rested on. It
converges across sessions and uploads rather than being replaced, the way
the psychological profile accumulates. A measurement is *reliable* only once
it has enough seconds behind it and its spread is small relative to its
value; anything else stays in the signature but is kept out of every prompt.

The same module decides what survives the byte budget: which tracks to prune
(oldest first, per emotion) and which golden segments to keep (curated for
identity confidence and emotion coverage, not recency, because the golden
set's job is to outlive any change of basis or landmark set).
"""

from __future__ import annotations

from typing import Any

from src.anubis.utils.motion.measurements import Measurement

CATEGORICAL_KEYS = frozenset({"dominant_gesture_hand", "hands_rest_position"})


def fold_measurements(
    signature: dict[str, Any] | None, measurements: dict[str, Measurement]
) -> dict[str, Any]:
    """Return a new signature with ``measurements`` folded in by seconds."""
    result: dict[str, Any] = {key: dict(value) for key, value in (signature or {}).items()}
    for key, measurement in measurements.items():
        weight = max(float(measurement.seconds), 1e-6)
        entry = result.get(key) or {}
        if key in CATEGORICAL_KEYS or isinstance(measurement.value, str):
            counts = dict(entry.get("counts") or {})
            label = str(measurement.value)
            counts[label] = float(counts.get(label, 0.0)) + weight
            result[key] = {
                "counts": counts,
                "seconds": float(entry.get("seconds") or 0.0) + weight,
                "samples": int(entry.get("samples") or 0) + int(measurement.samples or 1),
            }
            continue
        value = float(measurement.value)
        previous_weight = float(entry.get("seconds") or 0.0)
        previous_mean = float(entry.get("mean") or 0.0)
        previous_m2 = float(entry.get("m2") or 0.0)
        total_weight = previous_weight + weight
        delta = value - previous_mean
        mean = previous_mean + delta * (weight / total_weight)
        m2 = previous_m2 + weight * delta * (value - mean)
        result[key] = {
            "mean": mean,
            "m2": m2,
            "variance": (m2 / total_weight) if total_weight > 0 else 0.0,
            "seconds": total_weight,
            "samples": int(entry.get("samples") or 0) + int(measurement.samples or 1),
        }
    return result


def signature_value(signature: dict[str, Any], key: str) -> float | str | None:
    """Return the mean, or the most common label, for a measurement."""
    entry = signature.get(key)
    if not entry:
        return None
    if "counts" in entry:
        counts = entry["counts"] or {}
        if not counts:
            return None
        return max(counts, key=counts.get)
    return float(entry.get("mean", 0.0))


def is_reliable(
    entry: dict[str, Any] | None,
    *,
    min_seconds: float,
    max_coefficient_of_variation: float = 0.75,
    absolute_tolerance: float = 0.0,
) -> bool:
    """Report whether an entry has enough seconds and a stable enough spread to speak."""
    if not entry:
        return False
    if float(entry.get("seconds") or 0.0) < min_seconds:
        return False
    if "counts" in entry:
        counts = entry["counts"] or {}
        total = sum(counts.values()) or 0.0
        if total <= 0:
            return False
        return (max(counts.values()) / total) >= 0.6
    mean = abs(float(entry.get("mean") or 0.0))
    variance = float(entry.get("variance") or 0.0)
    spread = variance ** 0.5
    if spread <= absolute_tolerance:
        return True
    if mean < 1e-9:
        return spread <= absolute_tolerance
    return (spread / mean) <= max_coefficient_of_variation


def tracks_to_prune(
    tracks: list[dict[str, Any]], *, retention_seconds: float
) -> list[str]:
    """Return track ids to delete so each emotion's total stays under the budget, oldest first."""
    if retention_seconds <= 0:
        return []
    by_emotion: dict[str, list[dict[str, Any]]] = {}
    for track in tracks:
        by_emotion.setdefault(str(track.get("emotion") or "neutral"), []).append(track)
    doomed: list[str] = []
    for group in by_emotion.values():
        group.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        total = 0.0
        for track in group:
            total += float(track.get("duration_seconds") or 0.0)
            if total > retention_seconds:
                doomed.append(str(track["track_id"]))
    return doomed


def golden_segments_to_prune(
    segments: list[dict[str, Any]], *, budget_seconds: float
) -> list[str]:
    """Return which golden segments to drop so the set stays under budget.

    Scores identity confidence highest, then rewards the emotions with the
    least coverage so the set stays broad, and drops the lowest scores first.
    """
    if budget_seconds <= 0 or not segments:
        return []
    total = sum(float(segment.get("duration_seconds") or 0.0) for segment in segments)
    if total <= budget_seconds:
        return []
    coverage: dict[str, float] = {}
    for segment in segments:
        emotion = str(segment.get("emotion") or "neutral")
        coverage[emotion] = coverage.get(emotion, 0.0) + float(segment.get("duration_seconds") or 0.0)
    max_coverage = max(coverage.values()) or 1.0

    def score(segment: dict[str, Any]) -> float:
        confidence = float(segment.get("identity_confidence") or 0.0)
        emotion = str(segment.get("emotion") or "neutral")
        breadth = 1.0 - (coverage[emotion] / max_coverage)
        return confidence * 2.0 + breadth

    ordered = sorted(segments, key=score)
    doomed: list[str] = []
    for segment in ordered:
        if total <= budget_seconds:
            break
        total -= float(segment.get("duration_seconds") or 0.0)
        doomed.append(str(segment["segment_id"]))
    return doomed


__all__ = [
    "CATEGORICAL_KEYS",
    "fold_measurements",
    "golden_segments_to_prune",
    "is_reliable",
    "signature_value",
    "tracks_to_prune",
]
