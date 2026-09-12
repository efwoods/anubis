"""Tiers 2 and 3 and the renderer: measurements, primitives, folding, prompt text."""

from __future__ import annotations

import numpy as np

from src.anubis.utils.motion import (
    codec,
    measurements,
    motion_prompt,
    primitives,
    signature,
)
from unit_tests.motion_fixtures import body_stream as _body_stream
from unit_tests.motion_fixtures import head_stream as _head_stream
from unit_tests.motion_fixtures import sweeping_body as _sweeping_body

# --- measurements ------------------------------------------------------------


def test_resting_head_roll_and_eye_contact_are_measured():
    window = codec.MotionWindow(streams={"head_pose": _head_stream(300, roll=6.0)})
    result = measurements.measure_window(window)
    assert abs(result["resting_head_roll_degrees"].value - 6.0) < 1e-3
    assert result["eye_contact_fraction"].value == 1.0
    assert result["resting_head_roll_degrees"].seconds == 10.0


def test_a_still_body_reads_as_still_and_hands_at_rest():
    window = codec.MotionWindow(streams={"body": _body_stream(150)})
    result = measurements.measure_window(window)
    assert result["stillness_fraction"].value == 1.0
    assert result["gesture_rate_per_minute"].value == 0.0
    assert result["hands_rest_position"].value in ("lap", "chest")
    assert abs(result["shoulder_tilt_degrees"].value) < 1e-3


def test_gestures_are_counted_from_wrist_motion():
    window = codec.MotionWindow(streams={"body": _sweeping_body()})
    result = measurements.measure_window(window)
    assert result["gesture_rate_per_minute"].value > 10.0
    assert result["dominant_gesture_hand"].value == "right"
    assert result["gesture_amplitude_shoulders"].value > 0.5


def test_blinks_are_counted_from_the_mesh():
    rate, frames = 30.0, 600
    mesh = np.zeros((frames, 478, 3), dtype=np.float32) + 0.5
    # Open eyes: corners 0.1 apart, lids 0.03 apart -> EAR 0.3. Blink: lids meet.
    for corners, pairs in ((measurements._RIGHT_EYE["corners"], measurements._RIGHT_EYE["pairs"]), (measurements._LEFT_EYE["corners"], measurements._LEFT_EYE["pairs"])):
        mesh[:, corners[0], :2] = (0.40, 0.5)
        mesh[:, corners[1], :2] = (0.50, 0.5)
        for upper, lower in pairs:
            mesh[:, upper, :2] = (0.45, 0.485)
            mesh[:, lower, :2] = (0.45, 0.515)
    blink_frames = [int(second * rate) for second in (2, 5, 5.2, 9, 14)]
    for start in blink_frames:
        for eye in (measurements._RIGHT_EYE, measurements._LEFT_EYE):
            for upper, lower in eye["pairs"]:
                mesh[start : start + 4, upper, 1] = 0.5
                mesh[start : start + 4, lower, 1] = 0.5
    window = codec.MotionWindow(
        streams={"face": codec.StreamWindow(mesh.reshape(frames, -1), rate)},
        face_encoding=codec.FACE_ENCODING_DENSE,
    )
    result = measurements.measure_face(window, None)
    assert abs(result["blink_rate_per_minute"].value - 15.0) < 1e-6  # 5 blinks in 20 s
    assert result["blink_burst_ratio"].value > 0.2  # the 5.0 / 5.2 pair


# --- primitives --------------------------------------------------------------


def test_a_repeated_sweep_becomes_one_primitive_with_its_timing():
    window = codec.MotionWindow(streams={"body": _sweeping_body()}, speech=[{"start": 0, "end": 30, "kind": "speaking"}])
    events = primitives.extract_events(window)
    right = [event for event in events if event.channel == primitives.CHANNEL_RIGHT_HAND]
    assert 7 <= len(right) <= 10
    found = primitives.events_to_primitives(events, emotion="neutral", landmark_set_version="v")
    right_primitives = [p for p in found if p.channel == primitives.CHANNEL_RIGHT_HAND]
    assert len(right_primitives) >= 1
    top = max(right_primitives, key=lambda p: p.occurrences)
    assert 0.25 <= top.duration_mean <= 0.6
    assert top.amplitude_mean > 0.8
    assert top.prototype.shape == (primitives.PROTOTYPE_LENGTH, 3)
    assert top.context.get("speaking", 0) >= 3
    record = top.to_record()
    again = primitives.Primitive.from_record(record)
    assert np.allclose(again.prototype, top.prototype)


def test_merging_folds_a_repeat_into_the_same_primitive():
    window = codec.MotionWindow(streams={"body": _sweeping_body()})
    first = primitives.events_to_primitives(primitives.extract_events(window), emotion="neutral", landmark_set_version="v")
    merged = primitives.merge_primitives(first, first)
    assert len(merged) == len(first)
    assert sum(p.occurrences for p in merged) == 2 * sum(p.occurrences for p in first)


def test_resample_normalizes_start_and_amplitude():
    trajectory = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32) + 5.0
    resampled = primitives.resample_trajectory(trajectory, 8)
    assert resampled.shape == (8, 3)
    assert np.allclose(resampled[0], 0.0)
    assert abs(np.linalg.norm(resampled, axis=1).max() - 1.0) < 1e-6


# --- signature ---------------------------------------------------------------


def test_folding_weights_by_seconds_and_tracks_variance():
    folded = signature.fold_measurements({}, {"blink_rate_per_minute": measurements.Measurement(10.0, 10.0)})
    folded = signature.fold_measurements(folded, {"blink_rate_per_minute": measurements.Measurement(20.0, 30.0)})
    entry = folded["blink_rate_per_minute"]
    assert abs(entry["mean"] - 17.5) < 1e-6
    assert entry["seconds"] == 40.0 and entry["samples"] == 2
    assert entry["variance"] > 0.0
    categorical = signature.fold_measurements({}, {"dominant_gesture_hand": measurements.Measurement("right", 5.0)})
    categorical = signature.fold_measurements(categorical, {"dominant_gesture_hand": measurements.Measurement("left", 1.0)})
    assert signature.signature_value(categorical, "dominant_gesture_hand") == "right"


def test_unreliable_readings_are_recognized():
    assert not signature.is_reliable({"mean": 10, "variance": 0, "seconds": 5}, min_seconds=30)
    assert signature.is_reliable({"mean": 10, "variance": 1, "seconds": 60}, min_seconds=30)
    assert not signature.is_reliable({"mean": 10, "variance": 400, "seconds": 60}, min_seconds=30)
    assert signature.is_reliable({"mean": 0.0, "variance": 0.5, "seconds": 60}, min_seconds=30, absolute_tolerance=1.0)


def test_track_and_golden_pruning_respect_budgets():
    tracks = [{"track_id": str(i), "emotion": "neutral", "duration_seconds": 10, "created_at": f"2026-01-0{i}"} for i in range(1, 8)]
    doomed = signature.tracks_to_prune(tracks, retention_seconds=30)
    assert set(doomed) == {"1", "2", "3", "4"}
    golden = [
        {"segment_id": "a", "emotion": "neutral", "duration_seconds": 50, "identity_confidence": 0.95},
        {"segment_id": "b", "emotion": "neutral", "duration_seconds": 50, "identity_confidence": 0.6},
        {"segment_id": "c", "emotion": "joy", "duration_seconds": 50, "identity_confidence": 0.7},
    ]
    assert signature.golden_segments_to_prune(golden, budget_seconds=100) == ["b"]


# --- prompt ------------------------------------------------------------------


def test_render_block_speaks_only_reliable_habits_and_phrases_primitives():
    sig = signature.fold_measurements(
        {},
        {
            "resting_head_roll_degrees": measurements.Measurement(6.0, 60.0),
            "blink_rate_per_minute": measurements.Measurement(17.0, 60.0),
            "blink_burst_ratio": measurements.Measurement(0.5, 60.0),
            "eye_contact_fraction": measurements.Measurement(0.7, 60.0),
            "gesture_rate_per_minute": measurements.Measurement(9.0, 60.0),
            "dominant_gesture_hand": measurements.Measurement("right", 60.0),
            "gesture_amplitude_shoulders": measurements.Measurement(0.6, 60.0),
            "hands_rest_position": measurements.Measurement("out of frame", 60.0),
            "stillness_fraction": measurements.Measurement(0.5, 5.0),  # too few seconds
        },
    )
    window = codec.MotionWindow(streams={"body": _sweeping_body()})
    found = primitives.events_to_primitives(primitives.extract_events(window), emotion="neutral", landmark_set_version="v")
    block = motion_prompt.render_motion_block(sig, found, min_seconds=30.0, min_occurrences=3)
    assert "HEAD: rests with the head tilted about 6 degrees toward the person's own left" in block
    assert "blinks about 17 times a minute, often two blinks close together" in block
    assert "eye contact" in block and "70 percent of the time" in block
    assert "leading with the right hand" in block
    assert "the hands leave the frame between points" in block
    assert "right-hand movement" in block and "lasting about 0." in block
    assert "still" not in block.split("POSTURE:")[-1] if "POSTURE:" in block else True


def test_empty_signature_renders_nothing_and_role_section_groups_emotions():
    assert motion_prompt.render_motion_block({}, [], min_seconds=30.0) == ""
    section = motion_prompt.render_role_section({"neutral": "HEAD: a", "joy": "HEAD: b", "fear": "HEAD: a"})
    assert section.startswith("HEAD: a")
    assert "When feeling joy:\nHEAD: b" in section
    assert "When feeling fear" not in section


def test_compose_video_prompt_layers_foundation_then_behaviour():
    text = motion_prompt.compose_video_prompt("Medium close-up.", "HEAD: nods.")
    assert text.startswith("Medium close-up.\n")
    assert text.endswith("HEAD: nods.")
    assert motion_prompt.compose_video_prompt("Medium close-up.", "") == "Medium close-up."
