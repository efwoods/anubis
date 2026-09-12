"""Tier 0 and Tier 1: the landmark registry, the buffer codec, normalization, the basis."""

from __future__ import annotations

import numpy as np
import pytest

from src.anubis.utils.motion import basis as basis_module
from src.anubis.utils.motion import codec, landmarks, normalize
from unit_tests.motion_fixtures import random_body as _random_body
from unit_tests.motion_fixtures import synthetic_face as _synthetic_face

# --- landmarks ---------------------------------------------------------------


def test_default_landmark_set_declares_body_face_and_head_pose():
    landmark_set = landmarks.get_landmark_set(None)
    assert landmark_set.version == landmarks.DEFAULT_LANDMARK_SET_VERSION
    assert landmark_set.stream("body").values_per_frame == 33 * 4
    assert landmark_set.stream("face").values_per_frame == 478 * 3
    assert landmark_set.stream("head_pose").values_per_frame == 6
    assert "lips" in landmark_set.groups["face"]
    assert "left_shoulder" in landmarks.BODY_JOINT_NAMES


def test_unknown_landmark_set_is_an_error_not_a_guess():
    with pytest.raises(KeyError):
        landmarks.get_landmark_set("nothing_like_this")


def test_a_new_landmark_set_registers_without_touching_the_default():
    hands = landmarks.LandmarkSet(
        version="test_hands_v1",
        description="a decoder's limb set",
        streams=(landmarks.Stream("limb", ("shoulder", "elbow", "carpal"), 3),),
    )
    landmarks.register_landmark_set(hands)
    try:
        assert landmarks.get_landmark_set("test_hands_v1").stream("limb").values_per_frame == 9
        assert landmarks.get_landmark_set(None).version == landmarks.DEFAULT_LANDMARK_SET_VERSION
        with pytest.raises(ValueError):
            landmarks.register_landmark_set(hands)
    finally:
        landmarks.LANDMARK_SETS.pop("test_hands_v1", None)


# --- codec -------------------------------------------------------------------


def test_frames_round_trip_through_float16_within_precision():
    frames = _random_body(12)
    rebuilt = codec.decode_frames(codec.encode_frames(frames), frames.shape[1])
    assert rebuilt.shape == frames.shape
    assert np.allclose(rebuilt, frames, atol=2e-3)


def test_window_payload_round_trips_and_validates_widths():
    window = codec.MotionWindow(
        streams={
            "body": codec.StreamWindow(_random_body(15), 15.0),
            "head_pose": codec.StreamWindow(np.zeros((30, 6), dtype=np.float32), 30.0),
        },
        emotion="joy",
        speech=[{"start": 0.0, "end": 0.5, "kind": "speaking"}],
    )
    payload = codec.window_to_payload(window)
    back = codec.window_from_payload(payload)
    assert back.emotion == "joy"
    assert back.streams["body"].frame_count == 15
    assert back.streams["head_pose"].sample_rate_hz == 30.0
    assert back.speech[0]["kind"] == "speaking"

    payload["streams"]["body"]["values_per_frame"] = 7
    with pytest.raises(ValueError):
        codec.window_from_payload(payload)


def test_a_face_stream_must_declare_its_encoding():
    window = codec.MotionWindow(
        streams={"face": codec.StreamWindow(_synthetic_face(4), 30.0)},
        face_encoding=codec.FACE_ENCODING_NONE,
    )
    with pytest.raises(ValueError):
        codec.window_from_payload(codec.window_to_payload(window))


def test_a_basis_encoded_face_stream_carries_its_own_width():
    window = codec.MotionWindow(
        streams={"face": codec.StreamWindow(np.zeros((6, 48), dtype=np.float32), 30.0)},
        face_encoding=codec.FACE_ENCODING_BASIS,
        basis_id="abc",
    )
    back = codec.window_from_payload(codec.window_to_payload(window))
    assert back.streams["face"].frames.shape == (6, 48)
    assert back.basis_id == "abc"


# --- normalize ---------------------------------------------------------------


def test_body_normalization_is_invariant_to_camera_distance_and_position():
    frames = _random_body(10)
    joints = frames.reshape(10, 33, 4)
    moved = joints.copy()
    moved[:, :, :2] = moved[:, :, :2] * 2.5 + np.array([0.3, -0.1], dtype=np.float32)
    moved[:, :, 2] = moved[:, :, 2] * 2.5
    normalized_a, width_a = normalize.normalize_body_frames(frames)
    normalized_b, width_b = normalize.normalize_body_frames(moved.reshape(10, -1))
    assert np.allclose(width_b, width_a * 2.5, atol=1e-5)
    assert np.allclose(normalized_a, normalized_b, atol=1e-4)
    mid = (normalized_a.reshape(10, 33, 4)[:, 11, :3] + normalized_a.reshape(10, 33, 4)[:, 12, :3]) / 2
    assert np.allclose(mid, 0.0, atol=1e-6)


def test_face_canonicalization_removes_roll_and_scale():
    # No deformation: the eye points that fix the face's frame must not wander.
    face = _synthetic_face(1, amplitude=0.0)[0].reshape(478, 3)
    canonical = normalize.canonicalize_face(face)
    # Rotate the whole face by 20 degrees in the image plane and scale it.
    angle = np.deg2rad(20.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    turned = (face - face.mean(axis=0)) @ rotation.T * 1.7 + 0.5
    assert np.allclose(normalize.canonicalize_face(turned), canonical, atol=1e-3)
    pose = normalize.head_pose_geometric(turned)
    assert abs(abs(pose[2]) - 20.0) < 2.0  # roll recovered


def test_head_pose_from_matrix_reads_yaw():
    angle = np.deg2rad(30.0)
    matrix = np.eye(4)
    matrix[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    matrix[0, 3], matrix[1, 3] = 1.5, -2.0
    pose = normalize.head_pose_from_matrix(matrix)
    assert abs(pose[0] - 30.0) < 1e-3
    assert abs(pose[3] - 1.5) < 1e-6 and abs(pose[4] + 2.0) < 1e-6
    assert abs(pose[5] - 1.0) < 1e-6


# --- basis -------------------------------------------------------------------


def test_basis_recovers_known_deformations_and_reports_error():
    # Three known deformations over a fixed base: three components explain them.
    residuals = _synthetic_face(120, amplitude=0.03)
    fitted = basis_module.fit_basis(residuals, landmark_set_version="mediapipe_body33_face478_v1", component_count=3)
    assert fitted.component_count == 3
    assert sum(fitted.explained_variance_ratio) > 0.99
    coefficients = fitted.encode(residuals)
    assert coefficients.shape == (120, 3)
    rebuilt = fitted.decode(coefficients)
    assert np.sqrt(np.mean((rebuilt - residuals) ** 2)) < 5e-3
    assert fitted.reconstruction_error < 5e-3
    record = fitted.to_record()
    again = basis_module.MotionBasis.from_record({**record, "basis_id": "x"})
    assert np.allclose(again.components, fitted.components)
    assert again.basis_id == "x"


def test_describe_component_names_the_region_it_moves():
    landmark_set = landmarks.get_landmark_set(None)
    component = np.zeros((478, 3), dtype=np.float32)
    for index in landmark_set.groups["face"]["lips"]:
        component[index] = (0.0, -1.0, 0.0)  # lips upward (image y grows downward)
    text = basis_module.describe_component(component.reshape(-1), landmark_set)
    assert "lips" in text and "upward" in text


def test_basis_needs_enough_frames():
    with pytest.raises(ValueError):
        basis_module.fit_basis(np.zeros((3, 1434), dtype=np.float32), landmark_set_version="v")
