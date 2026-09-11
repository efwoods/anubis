"""Tier 3: the scalar measurements — rates and resting postures.

Every measurement here is one number with a name, a unit in that name, and a
formula over named landmarks. A measurement also carries how many seconds of
motion it was computed from, because a habit read off three seconds of
footage is a guess and a habit read off ten minutes is a habit; the signature
layer folds these by seconds, and the prompt layer omits anything not yet
reliable.

These describe *where the body sits and how often it moves*; they do not and
cannot describe the shape of a movement through time. That is what the
primitives are for (:mod:`primitives`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.anubis.utils.motion.codec import (
    FACE_ENCODING_BASIS,
    FACE_ENCODING_DENSE,
    MotionWindow,
)
from src.anubis.utils.motion.landmarks import (
    BODY_JOINT_INDEX,
    BODY_VALUES_PER_JOINT,
    FACE_POINT_COUNT,
    FACE_VALUES_PER_POINT,
    HEAD_POSE_NAMES,
)
from src.anubis.utils.motion.normalize import normalize_body_frames

# Eye aspect ratio points (MediaPipe mesh), subject's own left and right.
_RIGHT_EYE = {"corners": (33, 133), "pairs": ((159, 145), (160, 144), (158, 153))}
_LEFT_EYE = {"corners": (362, 263), "pairs": ((386, 374), (385, 380), (387, 373))}
_MOUTH_CORNERS = (61, 291)
_INNER_LIPS = (13, 14)
_EYE_OUTER = (33, 263)
_LEFT_BROW_TOP, _LEFT_EYE_TOP = 334, 386
_RIGHT_BROW_TOP, _RIGHT_EYE_TOP = 105, 159
_LEFT_IRIS, _RIGHT_IRIS = (473, 474, 475, 476, 477), (468, 469, 470, 471, 472)

BLINK_EAR_THRESHOLD = 0.21
BLINK_BURST_WINDOW_SECONDS = 0.4
VISIBILITY_THRESHOLD = 0.5
STILL_SPEED_SHOULDERS_PER_SECOND = 0.15
GESTURE_SPEED_SHOULDERS_PER_SECOND = 0.6
EYE_CONTACT_YAW_DEGREES = 12.0
EYE_CONTACT_PITCH_DEGREES = 12.0

_HEAD_INDEX = {name: index for index, name in enumerate(HEAD_POSE_NAMES)}


@dataclass
class Measurement:
    """One measured value and how much motion it rests on."""

    value: float | str
    seconds: float
    samples: int = 1


def _rate_per_minute(count: int, seconds: float) -> float:
    return (count / seconds) * 60.0 if seconds > 0 else 0.0


def _onsets(mask: np.ndarray) -> np.ndarray:
    """Return the indices where a boolean series turns from False to True."""
    if mask.size == 0:
        return np.zeros(0, dtype=np.int64)
    edges = np.diff(mask.astype(np.int8), prepend=0)
    return np.flatnonzero(edges == 1)


def _dominant_period_seconds(series: np.ndarray, rate: float) -> float | None:
    """Return the period of the strongest oscillation in a series, or ``None`` when flat."""
    if series.size < 8 or rate <= 0:
        return None
    centred = series - series.mean()
    if float(np.abs(centred).max()) < 1e-6:
        return None
    spectrum = np.abs(np.fft.rfft(centred))
    frequencies = np.fft.rfftfreq(series.size, d=1.0 / rate)
    spectrum[0] = 0.0
    usable = frequencies > (1.0 / max(series.size / rate, 1e-6))
    if not np.any(usable):
        return None
    peak = int(np.argmax(np.where(usable, spectrum, 0.0)))
    if frequencies[peak] <= 0:
        return None
    return float(1.0 / frequencies[peak])


# ---------------------------------------------------------------------------
# Face
# ---------------------------------------------------------------------------


def eye_aspect_ratio(points: np.ndarray, eye: dict[str, Any]) -> float:
    """Return the eye aspect ratio (lid gap over eye width) for one eye."""
    c0, c1 = eye["corners"]
    width = float(np.linalg.norm(points[c0, :2] - points[c1, :2]))
    if width < 1e-6:
        return 0.0
    heights = [float(np.linalg.norm(points[a, :2] - points[b, :2])) for a, b in eye["pairs"]]
    return float(sum(heights) / (len(heights) * width))


def _face_frames(window: MotionWindow, basis: Any | None) -> np.ndarray | None:
    """Return dense ``[frames, 1434]`` mesh residuals, decoding coefficients if needed."""
    stream = window.streams.get("face")
    if stream is None:
        return None
    if window.face_encoding == FACE_ENCODING_DENSE:
        return stream.frames
    if window.face_encoding == FACE_ENCODING_BASIS and basis is not None:
        return basis.decode(stream.frames)
    return None


def measure_face(window: MotionWindow, basis: Any | None) -> dict[str, Measurement]:
    """Measure blink, gaze, mouth and brow scalars from the face stream."""
    frames = _face_frames(window, basis)
    stream = window.streams.get("face")
    if frames is None or stream is None or frames.shape[0] < 4:
        return {}
    rate = float(stream.sample_rate_hz)
    seconds = frames.shape[0] / rate
    points = frames.reshape(frames.shape[0], FACE_POINT_COUNT, FACE_VALUES_PER_POINT)
    ear = np.array(
        [
            (eye_aspect_ratio(frame, _LEFT_EYE) + eye_aspect_ratio(frame, _RIGHT_EYE)) / 2.0
            for frame in points
        ]
    )
    closed = ear < BLINK_EAR_THRESHOLD
    blink_starts = _onsets(closed)
    blink_count = int(blink_starts.size)
    result: dict[str, Measurement] = {
        "blink_rate_per_minute": Measurement(_rate_per_minute(blink_count, seconds), seconds),
    }
    if blink_count >= 2:
        gaps = np.diff(blink_starts) / rate
        result["blink_burst_ratio"] = Measurement(
            float(np.mean(gaps < BLINK_BURST_WINDOW_SECONDS)), seconds, blink_count
        )
    if blink_count:
        durations = []
        for start in blink_starts:
            end = start
            while end < closed.size and closed[end]:
                end += 1
            durations.append((end - start) / rate * 1000.0)
        result["blink_duration_ms"] = Measurement(float(np.median(durations)), seconds, blink_count)
    inter_ocular = np.linalg.norm(points[:, _EYE_OUTER[0], :2] - points[:, _EYE_OUTER[1], :2], axis=1)
    inter_ocular = np.where(inter_ocular > 1e-6, inter_ocular, 1.0)
    mouth_width = np.linalg.norm(points[:, _MOUTH_CORNERS[0], :2] - points[:, _MOUTH_CORNERS[1], :2], axis=1)
    result["smile_baseline"] = Measurement(float(np.median(mouth_width / inter_ocular)), seconds)
    jaw_open = np.linalg.norm(points[:, _INNER_LIPS[0], :2] - points[:, _INNER_LIPS[1], :2], axis=1) / inter_ocular
    speaking_mask = _speaking_mask(window, frames.shape[0], rate)
    if speaking_mask is not None and speaking_mask.any():
        result["jaw_open_ratio_speaking"] = Measurement(
            float(np.median(jaw_open[speaking_mask])), float(speaking_mask.sum() / rate)
        )
    brow_gap = (
        np.linalg.norm(points[:, _LEFT_BROW_TOP, :2] - points[:, _LEFT_EYE_TOP, :2], axis=1)
        + np.linalg.norm(points[:, _RIGHT_BROW_TOP, :2] - points[:, _RIGHT_EYE_TOP, :2], axis=1)
    ) / (2.0 * inter_ocular)
    result["brow_activity"] = Measurement(
        float(np.percentile(brow_gap, 95) - np.percentile(brow_gap, 5)), seconds
    )
    result["gaze_offset_ratio"] = Measurement(float(np.median(np.abs(_iris_offset(points)))), seconds)
    return result


def _iris_offset(points: np.ndarray) -> np.ndarray:
    """Return the horizontal iris position within the eye, −1 (own right) to +1 (own left), per frame."""
    left_iris = points[:, list(_LEFT_IRIS), 0].mean(axis=1)
    right_iris = points[:, list(_RIGHT_IRIS), 0].mean(axis=1)
    left_c0, left_c1 = _LEFT_EYE["corners"]
    right_c0, right_c1 = _RIGHT_EYE["corners"]

    def _offset(iris_x: np.ndarray, a: int, b: int) -> np.ndarray:
        lo = np.minimum(points[:, a, 0], points[:, b, 0])
        hi = np.maximum(points[:, a, 0], points[:, b, 0])
        span = np.where(hi - lo > 1e-6, hi - lo, 1.0)
        return ((iris_x - lo) / span) * 2.0 - 1.0

    return (_offset(left_iris, left_c0, left_c1) + _offset(right_iris, right_c0, right_c1)) / 2.0


def _speaking_mask(window: MotionWindow, frame_count: int, rate: float) -> np.ndarray | None:
    if not window.speech:
        return None
    mask = np.zeros(frame_count, dtype=bool)
    for segment in window.speech:
        if str(segment.get("kind") or "speaking") != "speaking":
            continue
        start = int(max(0.0, float(segment.get("start") or 0.0)) * rate)
        end = int(min(frame_count / rate, float(segment.get("end") or 0.0)) * rate)
        if end > start:
            mask[start:end] = True
    return mask


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------


def measure_head(window: MotionWindow) -> dict[str, Measurement]:
    """Measure resting pose, ranges, turn speed and nods from the head stream."""
    stream = window.streams.get("head_pose")
    if stream is None or stream.frames.shape[0] < 4:
        return {}
    rate = float(stream.sample_rate_hz)
    frames = stream.frames
    seconds = frames.shape[0] / rate
    yaw = frames[:, _HEAD_INDEX["yaw_degrees"]]
    pitch = frames[:, _HEAD_INDEX["pitch_degrees"]]
    roll = frames[:, _HEAD_INDEX["roll_degrees"]]
    result: dict[str, Measurement] = {
        "resting_head_roll_degrees": Measurement(float(np.median(roll)), seconds),
        "resting_head_pitch_degrees": Measurement(float(np.median(pitch)), seconds),
        "head_yaw_range_degrees": Measurement(
            float(np.percentile(yaw, 95) - np.percentile(yaw, 5)), seconds
        ),
        "head_pitch_range_degrees": Measurement(
            float(np.percentile(pitch, 95) - np.percentile(pitch, 5)), seconds
        ),
    }
    yaw_speed = np.abs(np.diff(yaw)) * rate
    moving = yaw_speed > 5.0
    if moving.any():
        result["head_turn_speed_degrees_per_second"] = Measurement(
            float(np.median(yaw_speed[moving])), seconds, int(moving.sum())
        )
    # Nods: pitch oscillations with a period between 0.3 s and 1.5 s.
    nod_count = _count_oscillations(pitch, rate, min_period=0.3, max_period=1.5, min_amplitude=2.0)
    result["nod_rate_per_minute"] = Measurement(_rate_per_minute(nod_count, seconds), seconds)
    contact = (np.abs(yaw) < EYE_CONTACT_YAW_DEGREES) & (np.abs(pitch) < EYE_CONTACT_PITCH_DEGREES)
    result["eye_contact_fraction"] = Measurement(float(np.mean(contact)), seconds)
    return result


def _count_oscillations(
    series: np.ndarray, rate: float, *, min_period: float, max_period: float, min_amplitude: float
) -> int:
    """Count peak-to-peak swings in a plausible period band with real amplitude."""
    if series.size < 4:
        return 0
    smoothed = np.convolve(series, np.ones(3) / 3.0, mode="same")
    peaks = [
        index
        for index in range(1, smoothed.size - 1)
        if smoothed[index] > smoothed[index - 1] and smoothed[index] >= smoothed[index + 1]
    ]
    count = 0
    for first, second in zip(peaks, peaks[1:]):
        period = (second - first) / rate
        trough = float(smoothed[first:second].min())
        amplitude = float(min(smoothed[first], smoothed[second]) - trough)
        if min_period <= period <= max_period and amplitude >= min_amplitude:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Body
# ---------------------------------------------------------------------------


def measure_body(window: MotionWindow) -> dict[str, Measurement]:
    """Measure posture, sway, stillness and gestures from the body stream."""
    stream = window.streams.get("body")
    if stream is None or stream.frames.shape[0] < 4:
        return {}
    rate = float(stream.sample_rate_hz)
    normalized, shoulder_width = normalize_body_frames(stream.frames)
    seen = shoulder_width > 1e-4
    if seen.sum() < 4:
        return {}
    joints = normalized.reshape(normalized.shape[0], -1, BODY_VALUES_PER_JOINT)[seen]
    raw = stream.frames.reshape(stream.frames.shape[0], -1, BODY_VALUES_PER_JOINT)[seen]
    seconds = joints.shape[0] / rate
    result: dict[str, Measurement] = {}

    left_shoulder = raw[:, BODY_JOINT_INDEX["left_shoulder"], :2]
    right_shoulder = raw[:, BODY_JOINT_INDEX["right_shoulder"], :2]
    shoulder_vector = left_shoulder - right_shoulder
    tilt = np.degrees(np.arctan2(shoulder_vector[:, 1], np.where(np.abs(shoulder_vector[:, 0]) > 1e-6, shoulder_vector[:, 0], 1e-6)))
    result["shoulder_tilt_degrees"] = Measurement(float(np.median(tilt)), seconds)

    nose_z = joints[:, BODY_JOINT_INDEX["nose"], 2]
    result["forward_lean_degrees"] = Measurement(
        float(np.degrees(np.arctan(np.median(-nose_z)))), seconds
    )

    mid_shoulder_x = (raw[:, BODY_JOINT_INDEX["left_shoulder"], 0] + raw[:, BODY_JOINT_INDEX["right_shoulder"], 0]) / 2.0
    sway_units = mid_shoulder_x / np.where(shoulder_width[seen] > 1e-4, shoulder_width[seen], 1.0)
    period = _dominant_period_seconds(sway_units, rate)
    if period is not None:
        result["torso_sway_period_seconds"] = Measurement(period, seconds)
    result["torso_sway_amplitude_shoulders"] = Measurement(
        float(np.percentile(sway_units, 95) - np.percentile(sway_units, 5)), seconds
    )

    velocities = np.linalg.norm(np.diff(joints[:, :, :3], axis=0), axis=2) * rate  # [frames-1, joints]
    total_speed = velocities.mean(axis=1)
    result["stillness_fraction"] = Measurement(
        float(np.mean(total_speed < STILL_SPEED_SHOULDERS_PER_SECOND)), seconds
    )

    hand_names = ("left_wrist", "right_wrist")
    gesture_counts: dict[str, int] = {}
    amplitudes: list[float] = []
    heights: list[float] = []
    energies: dict[str, float] = {}
    visible_any = np.zeros(joints.shape[0], dtype=bool)
    rest_positions: list[np.ndarray] = []
    shoulder_to_nose = np.abs(joints[:, BODY_JOINT_INDEX["nose"], 1])
    shoulder_to_nose = np.where(shoulder_to_nose > 1e-4, shoulder_to_nose, 1.0)
    for name in hand_names:
        index = BODY_JOINT_INDEX[name]
        visible = joints[:, index, 3] > VISIBILITY_THRESHOLD
        visible_any |= visible
        speed = np.linalg.norm(np.diff(joints[:, index, :3], axis=0), axis=1) * rate
        moving = np.concatenate([[False], speed > GESTURE_SPEED_SHOULDERS_PER_SECOND]) & visible
        gesture_counts[name] = int(_onsets(moving).size)
        energies[name] = float(np.sum(speed[visible[1:]] ** 2)) if visible.any() else 0.0
        if visible.any():
            distance = np.linalg.norm(joints[visible, index, :3], axis=1)
            amplitudes.append(float(np.percentile(distance, 95)))
            heights.extend((-joints[visible, index, 1] / shoulder_to_nose[visible]).tolist())
            still = visible & np.concatenate([[True], speed < STILL_SPEED_SHOULDERS_PER_SECOND])
            if still.any():
                rest_positions.append(np.median(joints[still, index, :2], axis=0))
    total_gestures = sum(gesture_counts.values())
    result["gesture_rate_per_minute"] = Measurement(_rate_per_minute(total_gestures, seconds), seconds)
    result["hands_visible_fraction"] = Measurement(float(np.mean(visible_any)), seconds)
    if amplitudes:
        result["gesture_amplitude_shoulders"] = Measurement(float(max(amplitudes)), seconds, total_gestures or 1)
    if heights:
        result["gesture_height_ratio"] = Measurement(float(np.median(heights)), seconds, total_gestures or 1)
    if any(energies.values()):
        dominant = max(energies, key=energies.get)  # type: ignore[arg-type]
        share = energies[dominant] / max(sum(energies.values()), 1e-9)
        result["dominant_gesture_hand"] = Measurement(
            "right" if dominant == "right_wrist" else "left", seconds, total_gestures or 1
        )
        result["dominant_gesture_hand_share"] = Measurement(float(share), seconds, total_gestures or 1)
    if rest_positions:
        rest = np.mean(np.stack(rest_positions), axis=0)
        # y is in shoulder widths below the mid-shoulder line (image y grows downward).
        if rest[1] > 1.2:
            zone = "lap"
        elif rest[1] > 0.4:
            zone = "chest"
        elif rest[1] > -0.4:
            zone = "shoulder height"
        else:
            zone = "face height"
        result["hands_rest_position"] = Measurement(zone, seconds)
    elif not visible_any.any():
        result["hands_rest_position"] = Measurement("out of frame", seconds)
    return result


def measure_window(window: MotionWindow, basis: Any | None = None) -> dict[str, Measurement]:
    """Return every scalar this window supports, keyed by measurement name."""
    result: dict[str, Measurement] = {}
    result.update(measure_body(window))
    result.update(measure_head(window))
    result.update(measure_face(window, basis))
    return result


def measurement_seconds(measurements: dict[str, Measurement]) -> float:
    """Return the most seconds any measurement in the set rests on."""
    return max((entry.seconds for entry in measurements.values()), default=0.0)


__all__ = [
    "Measurement",
    "eye_aspect_ratio",
    "measure_body",
    "measure_face",
    "measure_head",
    "measure_window",
    "measurement_seconds",
    "math",
]
