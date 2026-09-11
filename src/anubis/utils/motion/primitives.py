"""Tier 2: motion primitives — how the person moves *through time*.

A mean cannot say that a hand leaves rest just before a stressed syllable,
peaks in 380 ms and retracts over 600 ms. A primitive can: the timeline is
cut into **events** (a gesture bout, a head movement, an expression), each
event's trajectory is resampled to a fixed length and normalized in
amplitude, recurring events are clustered, and each cluster keeps its
**prototype trajectory** — the actual curve over normalized time, per named
coordinate — together with how often it happens, how long it lasts, how big
it is, and in what context.

Channels:

* ``body_left_hand`` / ``body_right_hand`` — the wrist in shoulder widths
  relative to the mid-shoulder point (x, y, z);
* ``head`` — yaw, pitch, roll in degrees;
* ``face`` — basis coefficients, when the window carries a face stream and a
  basis is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import numpy as np

from src.anubis.utils.motion.codec import (
    FACE_ENCODING_BASIS,
    FACE_ENCODING_DENSE,
    MotionWindow,
)
from src.anubis.utils.motion.landmarks import (
    BODY_JOINT_INDEX,
    BODY_VALUES_PER_JOINT,
    HEAD_POSE_NAMES,
)
from src.anubis.utils.motion.normalize import normalize_body_frames

PROTOTYPE_LENGTH = 32

CHANNEL_LEFT_HAND = "body_left_hand"
CHANNEL_RIGHT_HAND = "body_right_hand"
CHANNEL_HEAD = "head"
CHANNEL_FACE = "face"

_HAND_SPEED_ONSET = 0.6  # shoulder widths per second
_HAND_SPEED_OFFSET = 0.25
_HEAD_SPEED_ONSET = 25.0  # degrees per second
_HEAD_SPEED_OFFSET = 8.0
_FACE_SPEED_ONSET = 0.35  # coefficient units per second
_FACE_SPEED_OFFSET = 0.12
_MIN_EVENT_SECONDS = 0.12
_MAX_EVENT_SECONDS = 3.0
# A movement that pauses at its apex (the hand reaches out, holds, comes back)
# dips below the offset speed for a few frames; a gap shorter than this joins
# the two halves into one event.
_JOIN_GAP_SECONDS = 0.2
_MERGE_DISTANCE = 0.55


@dataclass
class MotionEvent:
    """MotionEvent."""
    channel: str
    start_seconds: float
    end_seconds: float
    trajectory: np.ndarray  # [frames, dims], raw units
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Return the stream's length in seconds."""
        return self.end_seconds - self.start_seconds

    @property
    def amplitude(self) -> float:
        """Return the largest displacement from the event's first frame."""
        offset = self.trajectory - self.trajectory[0]
        return float(np.linalg.norm(offset, axis=1).max()) if offset.shape[0] else 0.0


@dataclass
class Primitive:
    """A recurring movement: its prototype curve and its statistics."""

    primitive_id: str
    channel: str
    prototype: np.ndarray  # [PROTOTYPE_LENGTH, dims], amplitude-normalized, starts at zero
    dimension: int
    occurrences: int
    duration_mean: float
    duration_std: float
    amplitude_mean: float
    amplitude_std: float
    context: dict[str, int] = field(default_factory=dict)
    emotion: str = "neutral"
    landmark_set_version: str = ""

    def to_record(self) -> dict[str, Any]:
        """Return the storable record for this object."""
        return {
            "primitive_id": self.primitive_id,
            "channel": self.channel,
            "emotion": self.emotion,
            "landmark_set_version": self.landmark_set_version,
            "prototype": np.ascontiguousarray(self.prototype, dtype=np.float32).tobytes(),
            "prototype_length": int(self.prototype.shape[0]),
            "dimension": int(self.dimension),
            "occurrences": int(self.occurrences),
            "duration_mean": float(self.duration_mean),
            "duration_std": float(self.duration_std),
            "amplitude_mean": float(self.amplitude_mean),
            "amplitude_std": float(self.amplitude_std),
            "context": dict(self.context),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Primitive:
        """Rebuild the object from a stored record."""
        length = int(record["prototype_length"])
        dimension = int(record["dimension"])
        prototype = np.frombuffer(record["prototype"], dtype=np.float32).copy().reshape(length, dimension)
        return cls(
            primitive_id=str(record["primitive_id"]),
            channel=str(record["channel"]),
            prototype=prototype,
            dimension=dimension,
            occurrences=int(record.get("occurrences") or 0),
            duration_mean=float(record.get("duration_mean") or 0.0),
            duration_std=float(record.get("duration_std") or 0.0),
            amplitude_mean=float(record.get("amplitude_mean") or 0.0),
            amplitude_std=float(record.get("amplitude_std") or 0.0),
            context={str(key): int(value) for key, value in (record.get("context") or {}).items()},
            emotion=str(record.get("emotion") or "neutral"),
            landmark_set_version=str(record.get("landmark_set_version") or ""),
        )

    def prototype_as_lists(self) -> list[list[float]]:
        """Return the prototype trajectory as nested lists, for JSON."""
        return [[float(value) for value in row] for row in self.prototype]


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def _segments_from_speed(
    speed: np.ndarray, rate: float, *, onset: float, offset: float
) -> list[tuple[int, int]]:
    """Segment by hysteresis: start above ``onset``, end when below ``offset``."""
    segments: list[tuple[int, int]] = []
    inside = False
    start = 0
    for index, value in enumerate(speed):
        if not inside and value > onset:
            inside = True
            start = index
        elif inside and value < offset:
            inside = False
            segments.append((start, index + 1))
    if inside:
        segments.append((start, speed.size + 1))
    join_frames = max(1, int(_JOIN_GAP_SECONDS * rate))
    joined: list[tuple[int, int]] = []
    for segment in segments:
        if joined and segment[0] - joined[-1][1] <= join_frames:
            joined[-1] = (joined[-1][0], segment[1])
        else:
            joined.append(segment)
    min_frames = max(2, int(_MIN_EVENT_SECONDS * rate))
    max_frames = int(_MAX_EVENT_SECONDS * rate)
    return [(a, b) for a, b in joined if min_frames <= (b - a) <= max_frames]


def _context_for(window: MotionWindow, start_seconds: float, end_seconds: float) -> dict[str, Any]:
    speaking = False
    for segment in window.speech:
        if str(segment.get("kind") or "speaking") != "speaking":
            continue
        seg_start = float(segment.get("start") or 0.0)
        seg_end = float(segment.get("end") or 0.0)
        if seg_start < end_seconds and seg_end > start_seconds:
            speaking = True
            break
    return {"speaking": speaking, "emotion": window.emotion}


def _events_for_series(
    window: MotionWindow,
    channel: str,
    series: np.ndarray,
    rate: float,
    *,
    onset: float,
    offset: float,
    valid: np.ndarray | None = None,
) -> list[MotionEvent]:
    if series.shape[0] < 4:
        return []
    speed = np.linalg.norm(np.diff(series, axis=0), axis=1) * rate
    if valid is not None:
        speed = np.where(valid[1:] & valid[:-1], speed, 0.0)
    events: list[MotionEvent] = []
    for start, end in _segments_from_speed(speed, rate, onset=onset, offset=offset):
        end = min(end, series.shape[0])
        start_seconds = start / rate
        end_seconds = end / rate
        events.append(
            MotionEvent(
                channel=channel,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                trajectory=series[start:end].astype(np.float32),
                context=_context_for(window, start_seconds, end_seconds),
            )
        )
    return events


def extract_events(window: MotionWindow, basis: Any | None = None) -> list[MotionEvent]:
    """Cut a window into movement events on every channel it carries."""
    events: list[MotionEvent] = []
    body = window.streams.get("body")
    if body is not None and body.frames.shape[0] >= 4:
        normalized, width = normalize_body_frames(body.frames)
        joints = normalized.reshape(normalized.shape[0], -1, BODY_VALUES_PER_JOINT)
        seen = width > 1e-4
        for channel, name in ((CHANNEL_LEFT_HAND, "left_wrist"), (CHANNEL_RIGHT_HAND, "right_wrist")):
            index = BODY_JOINT_INDEX[name]
            visible = (joints[:, index, 3] > 0.5) & seen
            events.extend(
                _events_for_series(
                    window, channel, joints[:, index, :3], float(body.sample_rate_hz),
                    onset=_HAND_SPEED_ONSET, offset=_HAND_SPEED_OFFSET, valid=visible,
                )
            )
    head = window.streams.get("head_pose")
    if head is not None and head.frames.shape[0] >= 4:
        angles = head.frames[:, :3]
        events.extend(
            _events_for_series(
                window, CHANNEL_HEAD, angles, float(head.sample_rate_hz),
                onset=_HEAD_SPEED_ONSET, offset=_HEAD_SPEED_OFFSET,
            )
        )
    face = window.streams.get("face")
    if face is not None and face.frames.shape[0] >= 4:
        coefficients: np.ndarray | None = None
        if window.face_encoding == FACE_ENCODING_BASIS:
            coefficients = face.frames
        elif window.face_encoding == FACE_ENCODING_DENSE and basis is not None:
            coefficients = basis.encode(face.frames)
        if coefficients is not None:
            events.extend(
                _events_for_series(
                    window, CHANNEL_FACE, coefficients, float(face.sample_rate_hz),
                    onset=_FACE_SPEED_ONSET, offset=_FACE_SPEED_OFFSET,
                )
            )
    return events


# ---------------------------------------------------------------------------
# Prototypes and clustering
# ---------------------------------------------------------------------------


def resample_trajectory(trajectory: np.ndarray, length: int = PROTOTYPE_LENGTH) -> np.ndarray:
    """Resample ``[frames, dims]`` to ``[length, dims]``, zeroed at the start, unit amplitude."""
    array = np.asarray(trajectory, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    frames, dims = array.shape
    if frames == 0:
        return np.zeros((length, dims), dtype=np.float32)
    source = np.linspace(0.0, 1.0, num=frames)
    target = np.linspace(0.0, 1.0, num=length)
    resampled = np.stack([np.interp(target, source, array[:, d]) for d in range(dims)], axis=1)
    resampled = resampled - resampled[0]
    amplitude = float(np.linalg.norm(resampled, axis=1).max())
    if amplitude > 1e-9:
        resampled = resampled / amplitude
    return resampled.astype(np.float32)


def _prototype_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float(np.sqrt(np.mean((a - b) ** 2)) * np.sqrt(a.shape[1]))


def _cluster(vectors: np.ndarray, cluster_count: int) -> np.ndarray:
    """Cluster by k-means then pick medoids; returns a label per row."""
    if vectors.shape[0] <= cluster_count:
        return np.arange(vectors.shape[0])
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=cluster_count, n_init=4, random_state=0)
    return model.fit_predict(vectors)


def events_to_primitives(
    events: list[MotionEvent],
    *,
    emotion: str,
    landmark_set_version: str,
    max_per_channel: int = 6,
) -> list[Primitive]:
    """Cluster a window's events into primitives, per channel."""
    primitives: list[Primitive] = []
    by_channel: dict[str, list[MotionEvent]] = {}
    for event in events:
        by_channel.setdefault(event.channel, []).append(event)
    for channel, channel_events in by_channel.items():
        prototypes = np.stack([resample_trajectory(event.trajectory) for event in channel_events])
        dims = prototypes.shape[2]
        flat = prototypes.reshape(prototypes.shape[0], -1)
        cluster_count = max(1, min(max_per_channel, int(np.ceil(len(channel_events) / 3))))
        labels = _cluster(flat, cluster_count)
        for label in np.unique(labels):
            members = [event for event, member_label in zip(channel_events, labels) if member_label == label]
            member_prototypes = prototypes[labels == label]
            centroid = member_prototypes.mean(axis=0)
            medoid_index = int(np.argmin([_prototype_distance(p, centroid) for p in member_prototypes]))
            durations = np.array([event.duration_seconds for event in members])
            amplitudes = np.array([event.amplitude for event in members])
            context: dict[str, int] = {}
            for event in members:
                key = "speaking" if event.context.get("speaking") else "silent"
                context[key] = context.get(key, 0) + 1
            primitives.append(
                Primitive(
                    primitive_id=str(uuid4()),
                    channel=channel,
                    prototype=member_prototypes[medoid_index],
                    dimension=dims,
                    occurrences=len(members),
                    duration_mean=float(durations.mean()),
                    duration_std=float(durations.std()),
                    amplitude_mean=float(amplitudes.mean()),
                    amplitude_std=float(amplitudes.std()),
                    context=context,
                    emotion=emotion,
                    landmark_set_version=landmark_set_version,
                )
            )
    return primitives


def _weighted(a: float, wa: int, b: float, wb: int) -> float:
    total = max(wa + wb, 1)
    return (a * wa + b * wb) / total


def merge_primitives(
    existing: list[Primitive],
    incoming: list[Primitive],
    *,
    max_per_channel: int = 8,
) -> list[Primitive]:
    """Fold new primitives into the dictionary: merge near ones, add the rest, prune rare ones."""
    merged = [Primitive(**{**p.__dict__, "prototype": p.prototype.copy(), "context": dict(p.context)}) for p in existing]
    for candidate in incoming:
        best: Primitive | None = None
        best_distance = _MERGE_DISTANCE
        for current in merged:
            if current.channel != candidate.channel or current.emotion != candidate.emotion:
                continue
            distance = _prototype_distance(current.prototype, candidate.prototype)
            if distance < best_distance:
                best, best_distance = current, distance
        if best is None:
            merged.append(candidate)
            continue
        total = best.occurrences + candidate.occurrences
        weight_existing = best.occurrences / max(total, 1)
        best.prototype = (best.prototype * weight_existing + candidate.prototype * (1 - weight_existing)).astype(np.float32)
        best.duration_mean = _weighted(best.duration_mean, best.occurrences, candidate.duration_mean, candidate.occurrences)
        best.duration_std = _weighted(best.duration_std, best.occurrences, candidate.duration_std, candidate.occurrences)
        best.amplitude_mean = _weighted(best.amplitude_mean, best.occurrences, candidate.amplitude_mean, candidate.occurrences)
        best.amplitude_std = _weighted(best.amplitude_std, best.occurrences, candidate.amplitude_std, candidate.occurrences)
        for key, value in candidate.context.items():
            best.context[key] = best.context.get(key, 0) + value
        best.occurrences = total
    # Keep the most frequent per (channel, emotion).
    kept: list[Primitive] = []
    groups: dict[tuple[str, str], list[Primitive]] = {}
    for primitive in merged:
        groups.setdefault((primitive.channel, primitive.emotion), []).append(primitive)
    for group in groups.values():
        group.sort(key=lambda p: p.occurrences, reverse=True)
        kept.extend(group[:max_per_channel])
    return kept


def primitive_fidelity(reference: Primitive, candidate: Primitive) -> float:
    """0..1 similarity of two prototypes (1 = identical shape)."""
    distance = _prototype_distance(reference.prototype, candidate.prototype)
    if not np.isfinite(distance):
        return 0.0
    return float(max(0.0, 1.0 - distance / 2.0))


__all__ = [
    "CHANNEL_FACE",
    "CHANNEL_HEAD",
    "CHANNEL_LEFT_HAND",
    "CHANNEL_RIGHT_HAND",
    "HEAD_POSE_NAMES",
    "MotionEvent",
    "PROTOTYPE_LENGTH",
    "Primitive",
    "events_to_primitives",
    "extract_events",
    "merge_primitives",
    "primitive_fidelity",
    "resample_trajectory",
]
