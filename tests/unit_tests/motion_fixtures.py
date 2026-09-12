"""Synthetic motion streams shared by the motion tests (no camera needed)."""

from __future__ import annotations

import numpy as np

from src.anubis.utils.motion import codec, landmarks


def body_stream(frames: int, rate: float = 15.0) -> codec.StreamWindow:
    joints = np.zeros((frames, 33, 4), dtype=np.float32)
    joints[:, :, 3] = 1.0
    joints[:, landmarks.BODY_JOINT_INDEX["left_shoulder"], :3] = (0.6, 0.5, 0.0)
    joints[:, landmarks.BODY_JOINT_INDEX["right_shoulder"], :3] = (0.4, 0.5, 0.0)
    joints[:, landmarks.BODY_JOINT_INDEX["nose"], :3] = (0.5, 0.3, -0.05)
    joints[:, landmarks.BODY_JOINT_INDEX["left_wrist"], :3] = (0.62, 0.9, 0.0)
    joints[:, landmarks.BODY_JOINT_INDEX["right_wrist"], :3] = (0.38, 0.9, 0.0)
    return codec.StreamWindow(joints.reshape(frames, -1), rate)


def head_stream(frames: int, rate: float = 30.0, *, roll: float = 6.0) -> codec.StreamWindow:
    head = np.zeros((frames, 6), dtype=np.float32)
    head[:, 2] = roll
    head[:, 5] = 1.0
    return codec.StreamWindow(head, rate)


def sweeping_body(rate: float = 15.0, seconds: float = 30.0, sweep_every: float = 3.0, sweep_seconds: float = 0.4) -> codec.StreamWindow:
    """A body stream whose right wrist sweeps outward every few seconds for 0.4 s."""
    frames = int(rate * seconds)
    stream = body_stream(frames, rate)
    joints = stream.frames.reshape(frames, 33, 4)
    index = landmarks.BODY_JOINT_INDEX["right_wrist"]
    rest = np.array([0.38, 0.9, 0.0], dtype=np.float32)
    for start_seconds in np.arange(1.0, seconds - 1.0, sweep_every):
        start = int(start_seconds * rate)
        length = int(sweep_seconds * rate)
        for step in range(length):
            phase = step / max(length - 1, 1)
            joints[start + step, index, :3] = rest + np.array(
                [-0.25 * np.sin(np.pi * phase), -0.1 * np.sin(np.pi * phase), 0.0], dtype=np.float32
            )
    stream.frames = joints.reshape(frames, -1)
    return stream


def random_body(frames: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    joints = rng.random((frames, 33, 4)).astype(np.float32)
    joints[:, :, 3] = 1.0
    joints[:, landmarks.BODY_JOINT_INDEX["left_shoulder"], :3] = (0.6, 0.5, 0.0)
    joints[:, landmarks.BODY_JOINT_INDEX["right_shoulder"], :3] = (0.4, 0.5, 0.0)
    return joints.reshape(frames, -1)


def synthetic_face(frames: int, *, seed: int = 0, amplitude: float = 0.02) -> np.ndarray:
    """A face mesh built from three known deformations over a fixed base."""
    rng = np.random.default_rng(seed)
    base = rng.random((landmarks.FACE_POINT_COUNT, 3)).astype(np.float32) * 0.2 + 0.4
    base[33] = (0.40, 0.45, 0.0)
    base[133] = (0.45, 0.45, 0.0)
    base[362] = (0.55, 0.45, 0.0)
    base[263] = (0.60, 0.45, 0.0)
    base[10] = (0.50, 0.30, 0.0)
    base[152] = (0.50, 0.70, 0.0)
    base[1] = (0.50, 0.52, -0.05)
    deformations = rng.standard_normal((3, landmarks.FACE_POINT_COUNT, 3)).astype(np.float32)
    weights = rng.standard_normal((frames, 3)).astype(np.float32) * amplitude
    mesh = base[None] + np.einsum("fk,kpc->fpc", weights, deformations)
    return mesh.reshape(frames, -1)
