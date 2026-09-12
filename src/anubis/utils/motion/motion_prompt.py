"""Layer D: measurements and primitives become words.

No model writes this text. A scalar passes through a band table that maps a
number to a clause; a primitive is phrased from the shape of its prototype
(which way it goes, how fast it rises, how it settles) and its timing. The
result is deterministic and auditable: every sentence traces to a number.

Only reliable readings speak. ``render_motion_block`` drops any measurement
that has not accumulated ``min_seconds`` of motion or whose spread is too
wide for its mean, and any primitive seen fewer than ``min_occurrences``
times, so a habit read off a few seconds of footage never reaches a prompt.

The rendered block is grouped the way the video vendor's behavioural layer
wants it — HEAD, EYES, POSTURE, HANDS, FACE — and the same block feeds the
avatar's own ROLE section, the emotion stills, the idle loops and the
lip-sync generation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.anubis.utils.motion.landmarks import LandmarkSet
from src.anubis.utils.motion.primitives import (
    CHANNEL_FACE,
    CHANNEL_HEAD,
    CHANNEL_LEFT_HAND,
    CHANNEL_RIGHT_HAND,
    Primitive,
)
from src.anubis.utils.motion.signature import is_reliable, signature_value

# (lower bound inclusive, upper bound exclusive, template). ``{value}`` is the mean.
_BANDS: dict[str, tuple[tuple[float, float, str], ...]] = {
    "blink_rate_per_minute": (
        (0.0, 8.0, "blinks rarely, about {value:.0f} times a minute"),
        (8.0, 20.0, "blinks about {value:.0f} times a minute"),
        (20.0, 1e9, "blinks often, about {value:.0f} times a minute"),
    ),
    "eye_contact_fraction": (
        (0.0, 0.35, "looks away from the camera more often than not, holding eye contact about {percent:.0f} percent of the time"),
        (0.35, 0.7, "holds eye contact about {percent:.0f} percent of the time"),
        (0.7, 1.01, "holds steady eye contact, about {percent:.0f} percent of the time"),
    ),
    "head_yaw_range_degrees": (
        (0.0, 10.0, "keeps the head nearly still side to side"),
        (10.0, 30.0, "turns the head moderately, through about {value:.0f} degrees"),
        (30.0, 1e9, "turns the head freely, through about {value:.0f} degrees"),
    ),
    "head_turn_speed_degrees_per_second": (
        (0.0, 30.0, "turns the head slowly"),
        (30.0, 80.0, "turns the head at an unhurried pace"),
        (80.0, 1e9, "turns the head quickly"),
    ),
    "nod_rate_per_minute": (
        (0.0, 2.0, ""),
        (2.0, 8.0, "nods now and then, about {value:.0f} times a minute"),
        (8.0, 1e9, "nods often, about {value:.0f} times a minute"),
    ),
    "torso_sway_amplitude_shoulders": (
        (0.0, 0.08, "keeps the torso still"),
        (0.08, 0.25, "sways the torso slightly"),
        (0.25, 1e9, "sways the torso noticeably"),
    ),
    "stillness_fraction": (
        (0.0, 0.4, "is rarely still, almost always in some small motion"),
        (0.4, 0.75, "is still about {percent:.0f} percent of the time"),
        (0.75, 1.01, "is still most of the time, about {percent:.0f} percent"),
    ),
    "gesture_rate_per_minute": (
        (0.0, 2.0, "seldom gestures"),
        (2.0, 8.0, "gestures now and then, about {value:.0f} times a minute"),
        (8.0, 1e9, "gestures often, about {value:.0f} times a minute"),
    ),
    "gesture_amplitude_shoulders": (
        (0.0, 0.5, "keeps gestures small, close to the body"),
        (0.5, 1.0, "gestures about a shoulder-width out from the body"),
        (1.0, 1e9, "gestures broadly, well out from the body"),
    ),
    "brow_activity": (
        (0.0, 0.04, "keeps the brows still"),
        (0.04, 0.09, "lifts the brows a little on emphasis"),
        (0.09, 1e9, "lifts the brows noticeably on emphasis"),
    ),
}


def _percent(value: float) -> float:
    return float(value) * 100.0


def _band_clause(key: str, value: float) -> str:
    for lower, upper, template in _BANDS.get(key, ()):
        if lower <= value < upper:
            return template.format(value=value, percent=_percent(value))
    return ""


def _side(degrees: float) -> str:
    return "toward the person's own left" if degrees > 0 else "toward the person's own right"


def _reliable_value(signature: dict[str, Any], key: str, *, min_seconds: float, **kwargs: Any) -> float | str | None:
    entry = signature.get(key)
    if not is_reliable(entry, min_seconds=min_seconds, **kwargs):
        return None
    return signature_value(signature, key)


def _join(clauses: list[str]) -> str:
    parts = [clause for clause in clauses if clause]
    if not parts:
        return ""
    return "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# Primitive phrasing
# ---------------------------------------------------------------------------


def _shape_words(prototype: np.ndarray) -> tuple[str, str]:
    """Report where the peak sits in time, and how the movement resolves."""
    magnitude = np.linalg.norm(prototype, axis=1)
    if magnitude.size == 0 or float(magnitude.max()) < 1e-9:
        return "", ""
    peak = int(np.argmax(magnitude))
    fraction = peak / max(magnitude.size - 1, 1)
    if fraction < 0.35:
        rise = "reaches its peak quickly"
    elif fraction < 0.7:
        rise = "builds evenly to its peak"
    else:
        rise = "builds slowly to its peak"
    end_ratio = float(magnitude[-1] / magnitude.max())
    if end_ratio < 0.3:
        settle = "returns to where it started"
    elif end_ratio < 0.7:
        settle = "settles part of the way back"
    else:
        settle = "holds at its new position"
    return rise, settle


def _hand_direction(prototype: np.ndarray, channel: str) -> str:
    magnitude = np.linalg.norm(prototype, axis=1)
    peak = prototype[int(np.argmax(magnitude))] if magnitude.size else np.zeros(3)
    x, y, z = (float(value) for value in peak[:3])
    axis = int(np.argmax(np.abs([x, y, z])))
    if axis == 1:
        return "upward" if y < 0 else "downward"
    if axis == 2:
        return "forward" if z < 0 else "back"
    # x grows toward the subject's own left in the normalized body frame.
    own_left = x > 0
    if channel == CHANNEL_LEFT_HAND:
        return "outward" if own_left else "inward across the body"
    return "inward across the body" if own_left else "outward"


def _head_direction(prototype: np.ndarray) -> str:
    magnitude = np.linalg.norm(prototype, axis=1)
    peak = prototype[int(np.argmax(magnitude))] if magnitude.size else np.zeros(3)
    yaw, pitch, roll = (float(value) for value in peak[:3])
    axis = int(np.argmax(np.abs([yaw, pitch, roll])))
    if axis == 0:
        return f"a head turn {_side(yaw)}"
    if axis == 1:
        return "a dip of the head" if pitch < 0 else "a lift of the head"
    return f"a tilt of the head {_side(roll)}"


def _context_clause(primitive: Primitive) -> str:
    total = sum(primitive.context.values()) or 0
    if total < 3:
        return ""
    speaking = primitive.context.get("speaking", 0) / total
    if speaking >= 0.75:
        return "while speaking"
    if speaking <= 0.25:
        return "while listening"
    return ""


def describe_primitive(
    primitive: Primitive,
    *,
    basis: Any | None = None,
    landmark_set: LandmarkSet | None = None,
) -> str:
    """Return one clause for one recurring movement, with its timing."""
    duration = f"about {primitive.duration_mean:.1f} s" if primitive.duration_mean >= 0.15 else "very briefly"
    rise, settle = _shape_words(primitive.prototype)
    context = _context_clause(primitive)
    tail = f" {context}" if context else ""
    if primitive.channel in (CHANNEL_LEFT_HAND, CHANNEL_RIGHT_HAND):
        hand = "left" if primitive.channel == CHANNEL_LEFT_HAND else "right"
        direction = _hand_direction(primitive.prototype, primitive.channel)
        reach = f"reaching about {primitive.amplitude_mean:.1f} shoulder-widths"
        return f"a {hand}-hand movement {direction}{tail}, lasting {duration}, that {rise}, {reach}, and {settle}"
    if primitive.channel == CHANNEL_HEAD:
        return f"{_head_direction(primitive.prototype)} of about {primitive.amplitude_mean:.0f} degrees{tail}, lasting {duration}, that {rise} and {settle}"
    if primitive.channel == CHANNEL_FACE:
        region = "a movement across the face"
        if basis is not None and landmark_set is not None:
            from src.anubis.utils.motion.basis import describe_component

            magnitude = np.linalg.norm(primitive.prototype, axis=1)
            peak = primitive.prototype[int(np.argmax(magnitude))]
            try:
                component_vector = peak @ basis.components
                region = describe_component(component_vector, landmark_set)
            except Exception:  # noqa: BLE001 - phrasing must never fail a prompt
                region = "a movement across the face"
        return f"a brief expression moving {region}{tail}, lasting {duration}, that {rise} and {settle}"
    return ""


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------


def render_motion_block(
    signature: dict[str, Any],
    primitives: list[Primitive],
    *,
    min_seconds: float,
    min_occurrences: int = 3,
    basis: Any | None = None,
    landmark_set: LandmarkSet | None = None,
    max_primitives_per_channel: int = 2,
) -> str:
    """Render the HEAD / EYES / POSTURE / HANDS / FACE block for one emotion, or ``""``."""
    reliable = lambda key, **kw: _reliable_value(signature, key, min_seconds=min_seconds, **kw)  # noqa: E731
    lines: list[str] = []

    head: list[str] = []
    roll = reliable("resting_head_roll_degrees", absolute_tolerance=4.0)
    if isinstance(roll, float) and abs(roll) >= 3.0:
        head.append(f"rests with the head tilted about {abs(roll):.0f} degrees {_side(roll)}")
    yaw_range = reliable("head_yaw_range_degrees")
    if isinstance(yaw_range, float):
        head.append(_band_clause("head_yaw_range_degrees", yaw_range))
    turn_speed = reliable("head_turn_speed_degrees_per_second")
    if isinstance(turn_speed, float):
        head.append(_band_clause("head_turn_speed_degrees_per_second", turn_speed))
    nod = reliable("nod_rate_per_minute", absolute_tolerance=1.5)
    if isinstance(nod, float):
        head.append(_band_clause("nod_rate_per_minute", nod))
    head.extend(_primitive_clauses(primitives, CHANNEL_HEAD, min_occurrences, max_primitives_per_channel, basis, landmark_set))
    if any(head):
        lines.append("HEAD: " + _join(head))

    eyes: list[str] = []
    blink = reliable("blink_rate_per_minute")
    if isinstance(blink, float):
        clause = _band_clause("blink_rate_per_minute", blink)
        burst = reliable("blink_burst_ratio", absolute_tolerance=0.15)
        if isinstance(burst, float) and burst >= 0.4:
            clause += ", often two blinks close together"
        eyes.append(clause)
    contact = reliable("eye_contact_fraction", absolute_tolerance=0.12)
    if isinstance(contact, float):
        eyes.append(_band_clause("eye_contact_fraction", contact))
    if any(eyes):
        lines.append("EYES: " + _join(eyes))

    posture: list[str] = []
    lean = reliable("forward_lean_degrees", absolute_tolerance=5.0)
    if isinstance(lean, float) and abs(lean) >= 5.0:
        posture.append(
            f"leans forward about {lean:.0f} degrees" if lean > 0 else f"sits back about {abs(lean):.0f} degrees"
        )
    tilt = reliable("shoulder_tilt_degrees", absolute_tolerance=3.0)
    if isinstance(tilt, float) and abs(tilt) >= 3.0:
        posture.append(f"carries the shoulders about {abs(tilt):.0f} degrees off level, the own-left side {'lower' if tilt > 0 else 'higher'}")
    elif isinstance(tilt, float):
        posture.append("keeps the shoulders level")
    sway = reliable("torso_sway_amplitude_shoulders")
    if isinstance(sway, float):
        clause = _band_clause("torso_sway_amplitude_shoulders", sway)
        period = reliable("torso_sway_period_seconds")
        if isinstance(period, float) and sway >= 0.08 and 1.0 <= period <= 20.0:
            clause += f", about once every {period:.0f} seconds"
        posture.append(clause)
    still = reliable("stillness_fraction", absolute_tolerance=0.12)
    if isinstance(still, float):
        posture.append(_band_clause("stillness_fraction", still))
    if any(posture):
        lines.append("POSTURE: " + _join(posture))

    hands: list[str] = []
    gesture_rate = reliable("gesture_rate_per_minute")
    if isinstance(gesture_rate, float):
        clause = _band_clause("gesture_rate_per_minute", gesture_rate)
        dominant = reliable("dominant_gesture_hand")
        if isinstance(dominant, str) and gesture_rate >= 2.0:
            clause += f", leading with the {dominant} hand"
        hands.append(clause)
    amplitude = reliable("gesture_amplitude_shoulders")
    if isinstance(amplitude, float) and isinstance(gesture_rate, float) and gesture_rate >= 2.0:
        hands.append(_band_clause("gesture_amplitude_shoulders", amplitude))
    height = reliable("gesture_height_ratio", absolute_tolerance=0.2)
    if isinstance(height, float) and isinstance(gesture_rate, float) and gesture_rate >= 2.0:
        if height >= 0.8:
            hands.append("raises the hands to face height")
        elif height >= 0.3:
            hands.append("raises the hands to chest height")
        else:
            hands.append("keeps the hands low")
    rest = reliable("hands_rest_position")
    if isinstance(rest, str):
        hands.append(
            "the hands leave the frame between points" if rest == "out of frame" else f"the hands rest at {rest} between points"
        )
    hands.extend(_primitive_clauses(primitives, CHANNEL_LEFT_HAND, min_occurrences, max_primitives_per_channel, basis, landmark_set))
    hands.extend(_primitive_clauses(primitives, CHANNEL_RIGHT_HAND, min_occurrences, max_primitives_per_channel, basis, landmark_set))
    if any(hands):
        lines.append("HANDS: " + _join(hands))

    face: list[str] = []
    brow = reliable("brow_activity")
    if isinstance(brow, float):
        face.append(_band_clause("brow_activity", brow))
    smile = reliable("smile_baseline", absolute_tolerance=0.03)
    if isinstance(smile, float):
        if smile >= 0.62:
            face.append("a clear smile sits at rest")
        elif smile >= 0.55:
            face.append("a faint smile sits at rest")
    jaw = reliable("jaw_open_ratio_speaking", absolute_tolerance=0.02)
    if isinstance(jaw, float):
        if jaw >= 0.12:
            face.append("opens the mouth wide while speaking")
        elif jaw >= 0.06:
            face.append("opens the mouth moderately while speaking")
        else:
            face.append("speaks with the mouth barely open")
    face.extend(_primitive_clauses(primitives, CHANNEL_FACE, min_occurrences, max_primitives_per_channel, basis, landmark_set))
    if any(face):
        lines.append("FACE: " + _join(face))

    return "\n".join(lines)


def _primitive_clauses(
    primitives: list[Primitive],
    channel: str,
    min_occurrences: int,
    limit: int,
    basis: Any | None,
    landmark_set: LandmarkSet | None,
) -> list[str]:
    chosen = [p for p in primitives if p.channel == channel and p.occurrences >= min_occurrences]
    chosen.sort(key=lambda p: p.occurrences, reverse=True)
    return [describe_primitive(p, basis=basis, landmark_set=landmark_set) for p in chosen[:limit]]


def render_role_section(blocks_by_emotion: dict[str, str]) -> str:
    """Render the ``=== HOW YOU MOVE ===`` text: the neutral block, then what changes per emotion."""
    neutral = (blocks_by_emotion.get("neutral") or "").strip()
    parts: list[str] = []
    if neutral:
        parts.append(neutral)
    for emotion, block in blocks_by_emotion.items():
        text = (block or "").strip()
        if emotion == "neutral" or not text or text == neutral:
            continue
        parts.append(f"When feeling {emotion}:\n{text}")
    return "\n\n".join(parts)


def compose_video_prompt(cinematic_prompt: str, behavioural_block: str) -> str:
    """Layer the prompt a video vendor wants: a stable foundation line, then behaviour."""
    foundation = (cinematic_prompt or "").strip()
    behaviour = (behavioural_block or "").strip()
    if not behaviour:
        return foundation
    behaviour_line = (
        "The person moves the way this person really moves, described from "
        "measurements of that person:\n" + behaviour
    )
    return f"{foundation}\n{behaviour_line}" if foundation else behaviour_line


__all__ = [
    "compose_video_prompt",
    "describe_primitive",
    "render_motion_block",
    "render_role_section",
]
