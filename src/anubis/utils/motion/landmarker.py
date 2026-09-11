"""The MediaPipe landmarkers, for server-side wireframing of video.

The same two models the browser runs — ``PoseLandmarker`` and
``FaceLandmarker`` — so a track learned from an uploaded video is written
against the same named coordinates as one learned from the live camera. The
model files are fetched once into ``MOTION_MODEL_CACHE_DIR``. Everything in
this module is synchronous CPU work and is meant to run in a worker thread.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np

from src.anubis.utils.motion.landmarks import (
    BODY_JOINT_NAMES,
    BODY_VALUES_PER_JOINT,
    FACE_POINT_COUNT,
    FACE_VALUES_PER_POINT,
)
from src.anubis.utils.motion.normalize import head_pose_from_matrix, head_pose_geometric

logger = logging.getLogger(__name__)

_download_lock = threading.Lock()


def _ensure_model_file(url: str, cache_dir: str) -> str:
    """Return a local path for a model URL, downloading it once."""
    os.makedirs(cache_dir, exist_ok=True)
    filename = url.rsplit("/", 1)[-1] or "model.task"
    path = os.path.join(cache_dir, filename)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    with _download_lock:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
        import httpx

        logger.info("Fetching landmarker model %s", url)
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
        temporary = path + ".part"
        with open(temporary, "wb") as handle:
            handle.write(response.content)
        os.replace(temporary, path)
    return path


class FrameLandmarker:
    """Wraps both landmarkers over frames of one video, in timestamp order."""

    def __init__(self, context: Any) -> None:
        """Initialize."""
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        cache_dir = str(getattr(context, "motion_model_cache_dir", None) or "/tmp/anubis-motion-models")
        pose_path = _ensure_model_file(str(getattr(context, "motion_pose_model_url")), cache_dir)
        face_path = _ensure_model_file(str(getattr(context, "motion_face_model_url")), cache_dir)
        self._mp = mp
        self._vision = vision
        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=pose_path),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
            )
        )
        self._face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=face_path),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                output_facial_transformation_matrixes=True,
                output_face_blendshapes=False,
            )
        )

    def close(self) -> None:
        """Release the landmarker models."""
        for model in (self._pose, self._face):
            try:
                model.close()
            except Exception:  # noqa: BLE001
                pass

    def _image(self, rgb: np.ndarray) -> Any:
        return self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    def pose(self, rgb: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        """Return a flat ``[33 * 4]`` body frame (x, y, z, visibility), or ``None``."""
        result = self._pose.detect_for_video(self._image(rgb), timestamp_ms)
        if not result.pose_landmarks:
            return None
        landmarks = result.pose_landmarks[0]
        if len(landmarks) != len(BODY_JOINT_NAMES):
            return None
        frame = np.zeros((len(BODY_JOINT_NAMES), BODY_VALUES_PER_JOINT), dtype=np.float32)
        for index, landmark in enumerate(landmarks):
            frame[index] = (
                float(landmark.x),
                float(landmark.y),
                float(landmark.z),
                float(getattr(landmark, "visibility", 0.0) or 0.0),
            )
        return frame.reshape(-1)

    def face(self, rgb: np.ndarray, timestamp_ms: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Return a flat ``[478 * 3]`` mesh frame and a ``[6]`` head pose, or ``None``."""
        result = self._face.detect_for_video(self._image(rgb), timestamp_ms)
        if not result.face_landmarks:
            return None
        landmarks = result.face_landmarks[0]
        if len(landmarks) != FACE_POINT_COUNT:
            return None
        mesh = np.zeros((FACE_POINT_COUNT, FACE_VALUES_PER_POINT), dtype=np.float32)
        for index, landmark in enumerate(landmarks):
            mesh[index] = (float(landmark.x), float(landmark.y), float(landmark.z))
        matrices = getattr(result, "facial_transformation_matrixes", None) or []
        if matrices:
            pose = head_pose_from_matrix(np.asarray(matrices[0], dtype=np.float64))
        else:
            pose = head_pose_geometric(mesh)
        return mesh.reshape(-1), pose


__all__ = ["FrameLandmarker"]
