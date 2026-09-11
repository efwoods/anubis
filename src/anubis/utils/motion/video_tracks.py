"""Wireframing an uploaded video: frames → named coordinates → windows.

This is the identity pipeline's motion analysis, run for video exclusively.
Frames are pulled with the ``ffmpeg`` the audio path already uses, landmarked
with the same models the browser runs, normalized, cut into windows, and
handed to ``record_motion_window`` like any other source. One frame near the
start of every window is kept as a JPEG data URI so the identity gate can
compare it with the reference image before anything is attributed.

Everything here is synchronous CPU work; ``extract_motion_windows`` is meant
to run under ``asyncio.to_thread``.
"""

from __future__ import annotations

import base64
import io
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from src.anubis.utils.motion.codec import (
    FACE_ENCODING_DENSE,
    FACE_ENCODING_NONE,
    MotionWindow,
    StreamWindow,
)
from src.anubis.utils.motion.landmarks import DEFAULT_LANDMARK_SET_VERSION
from src.anubis.utils.motion.normalize import canonicalize_face
from src.anubis.utils.motion.repository import SOURCE_UPLOADED_VIDEO

logger = logging.getLogger(__name__)

_FRAME_WIDTH = 640


@dataclass
class ExtractedWindow:
    """A window plus the frame the identity gate compares against the reference."""

    window: MotionWindow
    sample_frame_data_uri: str | None
    start_seconds: float


@dataclass
class _Accumulator:
    body: list[np.ndarray] = field(default_factory=list)
    face: list[np.ndarray] = field(default_factory=list)
    head: list[np.ndarray] = field(default_factory=list)
    sample_frame: np.ndarray | None = None


def _ffmpeg_executable() -> str:
    from src.anubis.utils.utility import _ffmpeg_executable as resolve

    return resolve()


def _probe_dimensions(path: str) -> tuple[int, int]:
    """Width and height of the decoded frames at the working width."""
    from src.anubis.utils.utility import _ffmpeg_executable as resolve

    ffprobe = resolve().replace("ffmpeg", "ffprobe")
    try:
        output = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
        width, height = (int(value) for value in output.split(",")[:2])
    except Exception:  # noqa: BLE001 - fall back to a square guess
        width, height = _FRAME_WIDTH, _FRAME_WIDTH
    scale = _FRAME_WIDTH / max(width, 1)
    return _FRAME_WIDTH, max(2, int(round(height * scale / 2)) * 2)


def iter_frames(path: str, *, rate_hz: float, max_seconds: float) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(timestamp_seconds, rgb_frame)`` at ``rate_hz`` for up to ``max_seconds``."""
    width, height = _probe_dimensions(path)
    command = [
        _ffmpeg_executable(), "-v", "error", "-nostdin",
        "-i", path, "-t", f"{max_seconds:.3f}",
        "-vf", f"fps={rate_hz},scale={width}:{height}",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ]
    frame_bytes = width * height * 3
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_bytes * 4)
    index = 0
    try:
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(frame_bytes)
            if len(chunk) < frame_bytes:
                break
            frame = np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3)
            yield index / rate_hz, frame
            index += 1
    finally:
        try:
            process.stdout.close()  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
        process.wait(timeout=30)


def _frame_data_uri(rgb: np.ndarray) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _flush(accumulator: _Accumulator, *, face_rate: float, body_rate: float, start_seconds: float,
           landmark_set_version: str, emotion: str, source_document_name: str | None,
           speech: list[dict[str, Any]]) -> ExtractedWindow | None:
    streams: dict[str, StreamWindow] = {}
    if accumulator.body:
        streams["body"] = StreamWindow(frames=np.stack(accumulator.body), sample_rate_hz=body_rate)
    face_encoding = FACE_ENCODING_NONE
    if accumulator.face:
        streams["face"] = StreamWindow(frames=np.stack(accumulator.face), sample_rate_hz=face_rate)
        streams["head_pose"] = StreamWindow(frames=np.stack(accumulator.head), sample_rate_hz=face_rate)
        face_encoding = FACE_ENCODING_DENSE
    if not streams:
        return None
    window = MotionWindow(
        landmark_set_version=landmark_set_version,
        source=SOURCE_UPLOADED_VIDEO,
        emotion=emotion,
        streams=streams,
        face_encoding=face_encoding,
        speech=[
            {"start": float(seg["start"]) - start_seconds, "end": float(seg["end"]) - start_seconds, "kind": seg.get("kind", "speaking")}
            for seg in speech
            if float(seg.get("end", 0)) > start_seconds and float(seg.get("start", 0)) < start_seconds + 1e9
        ],
        source_document_name=source_document_name,
    )
    sample = _frame_data_uri(accumulator.sample_frame) if accumulator.sample_frame is not None else None
    return ExtractedWindow(window=window, sample_frame_data_uri=sample, start_seconds=start_seconds)


def extract_motion_windows(
    path: str,
    context: Any,
    *,
    emotion: str = "neutral",
    source_document_name: str | None = None,
    speech: list[dict[str, Any]] | None = None,
) -> list[ExtractedWindow]:
    """Landmark a video file into windows. Synchronous; run in a worker thread."""
    from src.anubis.utils.motion.landmarker import FrameLandmarker

    face_rate = float(getattr(context, "motion_face_sample_rate_hz", 30.0) or 30.0)
    body_rate = float(getattr(context, "motion_body_sample_rate_hz", 15.0) or 15.0)
    window_seconds = float(getattr(context, "motion_track_window_seconds", 10.0) or 10.0)
    max_seconds = float(getattr(context, "motion_analysis_max_video_seconds", 600.0) or 600.0)
    landmark_set_version = str(getattr(context, "motion_landmark_set_version", None) or DEFAULT_LANDMARK_SET_VERSION)
    body_every = max(1, int(round(face_rate / body_rate)))

    landmarker = FrameLandmarker(context)
    windows: list[ExtractedWindow] = []
    accumulator = _Accumulator()
    window_start = 0.0
    frame_index = 0
    try:
        for timestamp, rgb in iter_frames(path, rate_hz=face_rate, max_seconds=max_seconds):
            if timestamp - window_start >= window_seconds:
                flushed = _flush(
                    accumulator, face_rate=face_rate, body_rate=face_rate / body_every,
                    start_seconds=window_start, landmark_set_version=landmark_set_version,
                    emotion=emotion, source_document_name=source_document_name, speech=speech or [],
                )
                if flushed is not None:
                    windows.append(flushed)
                accumulator = _Accumulator()
                window_start = timestamp
            timestamp_ms = int(round(timestamp * 1000.0))
            face_result = landmarker.face(rgb, timestamp_ms)
            if face_result is not None:
                mesh, pose = face_result
                accumulator.face.append(canonicalize_face(mesh))
                accumulator.head.append(pose)
                if accumulator.sample_frame is None:
                    accumulator.sample_frame = rgb.copy()
            if frame_index % body_every == 0:
                body_frame = landmarker.pose(rgb, timestamp_ms)
                if body_frame is not None:
                    accumulator.body.append(body_frame)
                    if accumulator.sample_frame is None:
                        accumulator.sample_frame = rgb.copy()
            frame_index += 1
        flushed = _flush(
            accumulator, face_rate=face_rate, body_rate=face_rate / body_every,
            start_seconds=window_start, landmark_set_version=landmark_set_version,
            emotion=emotion, source_document_name=source_document_name, speech=speech or [],
        )
        if flushed is not None:
            windows.append(flushed)
    finally:
        landmarker.close()
    logger.info("Wireframed %s into %d motion windows", source_document_name or path, len(windows))
    return windows


__all__ = ["ExtractedWindow", "extract_motion_windows", "iter_frames"]
