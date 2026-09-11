"""Layer A: take the camera out of the coordinates.

Nothing downstream may measure the camera. Sitting closer must not read as a
bigger gesture, leaning must not read as a hand movement, and a head turned
away must not read as a changed expression. So every frame is split three
ways before anything is measured:

* **body** — every joint expressed relative to the mid-shoulder point and
  divided by the shoulder width, so distances are in *shoulder widths* and
  positions are *relative to the torso*;
* **head pose** — yaw, pitch and roll of the head, plus its translation and
  scale, taken from MediaPipe's facial transformation matrix when the caller
  has one and estimated from the mesh geometry when not;
* **face residual** — the mesh after the rigid head motion is removed: centred
  on the face, scaled by the inter-ocular distance, rotated back to a
  face-forward frame. What is left is expression and nothing else.

Image coordinates have ``y`` growing downward; that is kept as-is in the
stored data and accounted for only where a direction is put into words.
"""

from __future__ import annotations

import math

import numpy as np

from src.anubis.utils.motion.landmarks import (
    BODY_JOINT_INDEX,
    BODY_VALUES_PER_JOINT,
    FACE_POINT_COUNT,
    FACE_VALUES_PER_POINT,
)

# Mesh indices used to fix the face's own frame of reference.
_LEFT_EYE_OUTER = 263
_RIGHT_EYE_OUTER = 33
_LEFT_EYE_INNER = 362
_RIGHT_EYE_INNER = 133
_NOSE_TIP = 1
_CHIN = 152
_FOREHEAD = 10

_MIN_SCALE = 1e-4


def body_joints(frame: np.ndarray) -> np.ndarray:
    """Return a flat body frame viewed as ``[33, 4]``."""
    return np.asarray(frame, dtype=np.float32).reshape(-1, BODY_VALUES_PER_JOINT)


def normalize_body_frames(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Express every joint in shoulder widths relative to the mid-shoulder point.

    Returns the normalized frames (same shape as the input) and the shoulder
    width per frame in the original units, which callers use to reject frames
    where the shoulders were not actually seen.
    """
    array = np.asarray(frames, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    joints = array.reshape(array.shape[0], -1, BODY_VALUES_PER_JOINT)
    left = joints[:, BODY_JOINT_INDEX["left_shoulder"], :3]
    right = joints[:, BODY_JOINT_INDEX["right_shoulder"], :3]
    origin = (left + right) / 2.0
    width = np.linalg.norm((left - right)[:, :2], axis=1)
    safe_width = np.where(width > _MIN_SCALE, width, 1.0)
    normalized = joints.copy()
    normalized[:, :, :3] = (joints[:, :, :3] - origin[:, None, :]) / safe_width[:, None, None]
    return normalized.reshape(array.shape[0], -1), width


def face_points(frame: np.ndarray) -> np.ndarray:
    """Return a flat face frame viewed as ``[478, 3]``."""
    return np.asarray(frame, dtype=np.float32).reshape(FACE_POINT_COUNT, FACE_VALUES_PER_POINT)


def _rotation_to_euler_degrees(rotation: np.ndarray) -> tuple[float, float, float]:
    """Return yaw, pitch, roll (degrees) from a 3x3 rotation, Tait-Bryan y-x-z order."""
    matrix = np.asarray(rotation, dtype=np.float64)
    sy = -matrix[2, 0]
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    if abs(math.cos(pitch)) > 1e-6:
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])
        roll = math.atan2(matrix[2, 1], matrix[2, 2])
    else:
        yaw = math.atan2(-matrix[0, 1], matrix[1, 1])
        roll = 0.0
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def head_pose_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return yaw, pitch, roll, tx, ty, scale from a 4x4 facial transformation matrix."""
    array = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    rotation = array[:3, :3]
    scale = float(np.cbrt(abs(np.linalg.det(rotation)))) or 1.0
    yaw, pitch, roll = _rotation_to_euler_degrees(rotation / scale)
    return np.array(
        [yaw, pitch, roll, float(array[0, 3]), float(array[1, 3]), scale],
        dtype=np.float32,
    )


def _face_frame_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the centre, rotation and scale that put a face into its own forward frame."""
    left_eye = (points[_LEFT_EYE_OUTER] + points[_LEFT_EYE_INNER]) / 2.0
    right_eye = (points[_RIGHT_EYE_OUTER] + points[_RIGHT_EYE_INNER]) / 2.0
    centre = (left_eye + right_eye) / 2.0
    x_axis = left_eye - right_eye
    inter_ocular = float(np.linalg.norm(x_axis))
    if inter_ocular < _MIN_SCALE:
        return centre, np.eye(3, dtype=np.float64), 1.0
    x_axis = x_axis / inter_ocular
    down = points[_CHIN] - points[_FOREHEAD]
    down = down - np.dot(down, x_axis) * x_axis
    down_norm = float(np.linalg.norm(down))
    if down_norm < _MIN_SCALE:
        y_axis = np.array([0.0, 1.0, 0.0])
    else:
        y_axis = down / down_norm
    z_axis = np.cross(x_axis, y_axis)
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm < _MIN_SCALE:
        z_axis = np.array([0.0, 0.0, 1.0])
    else:
        z_axis = z_axis / z_norm
    rotation = np.stack([x_axis, y_axis, z_axis], axis=0)  # rows: face axes in image space
    return centre, rotation, inter_ocular


def head_pose_geometric(points: np.ndarray) -> np.ndarray:
    """Estimate yaw, pitch, roll, tx, ty, scale from the mesh alone.

    Used when a caller has no transformation matrix. Roll is the angle of the
    inter-ocular line; yaw and pitch come from how the nose tip sits between
    the eyes and between forehead and chin.
    """
    face = np.asarray(points, dtype=np.float64).reshape(FACE_POINT_COUNT, 3)
    centre, rotation, inter_ocular = _face_frame_axes(face)
    x_axis = rotation[0]
    roll = math.degrees(math.atan2(x_axis[1], x_axis[0]))
    nose = face[_NOSE_TIP] - centre
    local = rotation @ nose
    scale = inter_ocular if inter_ocular > _MIN_SCALE else 1.0
    yaw = math.degrees(math.atan2(local[0], scale * 0.9))
    vertical_span = float(np.linalg.norm(face[_CHIN] - face[_FOREHEAD])) or 1.0
    nose_fraction = float(np.dot(face[_NOSE_TIP] - face[_FOREHEAD], rotation[1])) / vertical_span
    pitch = math.degrees((nose_fraction - 0.55) * math.pi / 2.0)
    return np.array(
        [yaw, pitch, roll, float(centre[0]), float(centre[1]), float(scale)],
        dtype=np.float32,
    )


def canonicalize_face(points: np.ndarray) -> np.ndarray:
    """Remove rigid head motion from one mesh frame.

    The face is centred between the eyes, scaled to unit inter-ocular
    distance, and rotated so the eye line is the x axis and the
    forehead-to-chin line is the y axis. The result is the expression alone,
    as a flat ``[478 * 3]`` residual.
    """
    face = np.asarray(points, dtype=np.float64).reshape(FACE_POINT_COUNT, 3)
    centre, rotation, inter_ocular = _face_frame_axes(face)
    scale = inter_ocular if inter_ocular > _MIN_SCALE else 1.0
    local = ((face - centre) / scale) @ rotation.T
    return local.reshape(-1).astype(np.float32)


def canonicalize_face_frames(frames: np.ndarray) -> np.ndarray:
    """Apply :func:`canonicalize_face` to every frame of a ``[frames, 1434]`` stream."""
    array = np.asarray(frames, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return np.stack([canonicalize_face(frame) for frame in array], axis=0)


def head_pose_frames_geometric(frames: np.ndarray) -> np.ndarray:
    """Estimate the head pose for every frame of a face stream."""
    array = np.asarray(frames, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return np.stack([head_pose_geometric(frame) for frame in array], axis=0)
