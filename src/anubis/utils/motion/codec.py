"""Buffers: how a stream of frames becomes bytes and back.

Every stream on a track is stored frame-major as little-endian **float16**.
Two bytes per value is the same size as the int16 the design budgets for, and
float16 needs no per-channel scale table: normalized coordinates, angles in
degrees, basis coefficients and visibility scores all fit their own range with
about three significant digits, which is more precision than the landmark
models deliver.

A ``MotionWindow`` is the unit that moves between the browser (or a decoder,
or the video pipeline) and the store: a few seconds of every stream, each at
its own sample rate, tagged with the landmark set it was written against.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.anubis.utils.motion.landmarks import (
    BODY_JOINT_INDEX,
    BODY_JOINT_NAMES,
    BODY_VALUES_PER_JOINT,
    DEFAULT_LANDMARK_SET_VERSION,
    get_landmark_set,
)

SOURCE_NEURAL_DECODER = "neural_decoder"

# A decoder may send only the joints it has. Shoulder-width normalize still
# needs a torso frame, so missing joints sit at this rest pose with visibility
# 0 — a measurement frame, not claimed decoded data.
CANONICAL_REST_JOINT_XYZ: dict[str, tuple[float, float, float]] = {
    "left_shoulder": (0.6, 0.5, 0.0),
    "right_shoulder": (0.4, 0.5, 0.0),
    "nose": (0.5, 0.3, -0.05),
}

FACE_ENCODING_NONE = "none"
FACE_ENCODING_DENSE = "dense"
FACE_ENCODING_BASIS = "basis"

_DTYPE = np.dtype("<f2")


def encode_frames(frames: np.ndarray) -> bytes:
    """Serialize a ``[frames, values]`` array as little-endian float16 bytes."""
    array = np.ascontiguousarray(np.asarray(frames, dtype=np.float32))
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("A stream is a two-dimensional array of frames by values.")
    return array.astype(_DTYPE).tobytes()


def decode_frames(payload: bytes, values_per_frame: int) -> np.ndarray:
    """Rebuild a ``[frames, values]`` float32 array from float16 bytes."""
    if values_per_frame <= 0:
        raise ValueError("values_per_frame must be positive.")
    flat = np.frombuffer(payload, dtype=_DTYPE).astype(np.float32)
    if flat.size % values_per_frame:
        raise ValueError(
            f"Buffer of {flat.size} values is not a whole number of "
            f"{values_per_frame}-value frames."
        )
    return flat.reshape(-1, values_per_frame)


@dataclass
class StreamWindow:
    """One stream's frames for a window, with the rate they were sampled at."""

    frames: np.ndarray  # [frame_count, values_per_frame], float32
    sample_rate_hz: float

    @property
    def frame_count(self) -> int:
        """Return the number of frames in the stream."""
        return int(self.frames.shape[0])

    @property
    def duration_seconds(self) -> float:
        """Return the stream's length in seconds."""
        if self.sample_rate_hz <= 0:
            return 0.0
        return self.frame_count / float(self.sample_rate_hz)


@dataclass
class MotionWindow:
    """A few seconds of motion across every captured stream."""

    landmark_set_version: str = DEFAULT_LANDMARK_SET_VERSION
    source: str = "live_camera"
    emotion: str = "neutral"
    captured_at: str | None = None
    streams: dict[str, StreamWindow] = field(default_factory=dict)
    # ``none`` when no face stream, ``dense`` for raw mesh residuals,
    # ``basis`` when the face stream holds coefficients against ``basis_id``.
    face_encoding: str = FACE_ENCODING_NONE
    basis_id: str | None = None
    # Optional segments aligned to the window: [{"start", "end", "kind"}],
    # ``kind`` in ``speaking`` / ``silent``; used as primitive context.
    speech: list[dict[str, Any]] = field(default_factory=list)
    identity_confidence: float | None = None
    source_document_name: str | None = None

    @property
    def duration_seconds(self) -> float:
        """Return the stream's length in seconds."""
        return max(
            (stream.duration_seconds for stream in self.streams.values()), default=0.0
        )

    def byte_length(self) -> int:
        """Return the size of every buffer in bytes."""
        return sum(int(stream.frames.size) * 2 for stream in self.streams.values())


def expand_sparse_body_frames(
    sparse_frames: np.ndarray, present_joints: list[str]
) -> np.ndarray:
    """Expand a named-joint body buffer into the stored 33 × 4 layout.

    ``sparse_frames`` is ``[frames, 4 * len(present_joints)]``. Unknown names
    are an error. Present joints keep the values they arrived with. Missing
    joints stay at rest with visibility 0; canonical shoulders and nose are
    placed so shoulder-width normalize does not divide by zero.
    """
    if not present_joints:
        raise ValueError("present_joints must name at least one joint.")
    seen: set[str] = set()
    for name in present_joints:
        if name not in BODY_JOINT_INDEX:
            raise ValueError(f"Unknown body joint {name!r}.")
        if name in seen:
            raise ValueError(f"present_joints names {name!r} more than once.")
        seen.add(name)
    array = np.asarray(sparse_frames, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    expected_width = BODY_VALUES_PER_JOINT * len(present_joints)
    if array.shape[1] != expected_width:
        raise ValueError(
            f"A sparse body stream with {len(present_joints)} joints needs "
            f"{expected_width} values per frame, not {array.shape[1]}."
        )
    joints = np.zeros((array.shape[0], len(BODY_JOINT_NAMES), BODY_VALUES_PER_JOINT), dtype=np.float32)
    for name, xyz in CANONICAL_REST_JOINT_XYZ.items():
        joints[:, BODY_JOINT_INDEX[name], :3] = np.asarray(xyz, dtype=np.float32)
    for column, name in enumerate(present_joints):
        start = column * BODY_VALUES_PER_JOINT
        joints[:, BODY_JOINT_INDEX[name], :] = array[:, start : start + BODY_VALUES_PER_JOINT]
    return joints.reshape(array.shape[0], -1)


def window_to_payload(window: MotionWindow) -> dict[str, Any]:
    """Return the JSON shape a window travels as (buffers base64-encoded)."""
    return {
        "schema_version": 1,
        "landmark_set_version": window.landmark_set_version,
        "source": window.source,
        "emotion": window.emotion,
        "captured_at": window.captured_at,
        "face_encoding": window.face_encoding,
        "basis_id": window.basis_id,
        "speech": list(window.speech),
        "identity_confidence": window.identity_confidence,
        "source_document_name": window.source_document_name,
        "streams": {
            name: {
                "sample_rate_hz": float(stream.sample_rate_hz),
                "frame_count": stream.frame_count,
                "values_per_frame": int(stream.frames.shape[1]),
                "data_b64": base64.b64encode(encode_frames(stream.frames)).decode("ascii"),
            }
            for name, stream in window.streams.items()
        },
    }


def window_from_payload(payload: dict[str, Any]) -> MotionWindow:
    """Parse a window sent by a client; every buffer is checked against the set."""
    if not isinstance(payload, dict):
        raise ValueError("A motion window is a JSON object.")
    landmark_set = get_landmark_set(payload.get("landmark_set_version"))
    source = str(payload.get("source") or "live_camera")
    face_encoding = str(payload.get("face_encoding") or FACE_ENCODING_NONE)
    if face_encoding not in (FACE_ENCODING_NONE, FACE_ENCODING_DENSE, FACE_ENCODING_BASIS):
        raise ValueError(f"Unknown face encoding {face_encoding!r}.")
    streams: dict[str, StreamWindow] = {}
    raw_streams = payload.get("streams") or {}
    if not isinstance(raw_streams, dict) or not raw_streams:
        raise ValueError("A motion window needs at least one stream.")
    for name, raw in raw_streams.items():
        if not landmark_set.has_stream(name):
            raise ValueError(
                f"Landmark set {landmark_set.version!r} has no stream {name!r}."
            )
        if not isinstance(raw, dict):
            raise ValueError(f"Stream {name!r} is not an object.")
        declared_width = raw.get("values_per_frame")
        expected_width = landmark_set.stream(name).values_per_frame
        present_joints = raw.get("present_joints")
        if present_joints is not None and source != SOURCE_NEURAL_DECODER:
            raise ValueError("present_joints is only valid on a neural_decoder window.")
        sparse_neural_body = name == "body" and present_joints is not None
        if sparse_neural_body:
            if not isinstance(present_joints, list) or not present_joints:
                raise ValueError("present_joints must name at least one joint.")
            present_joint_names = [str(joint) for joint in present_joints]
            width = BODY_VALUES_PER_JOINT * len(present_joint_names)
            if declared_width is not None and int(declared_width) != width:
                raise ValueError(
                    f"Stream {name!r} declares {declared_width} values per frame; "
                    f"{len(present_joint_names)} present joints need {width}."
                )
        elif name == "face" and face_encoding == FACE_ENCODING_BASIS:
            # Coefficients: the width is the basis' component count, not the mesh.
            if not declared_width or int(declared_width) <= 0:
                raise ValueError("A basis-encoded face stream must declare its width.")
            width = int(declared_width)
        else:
            width = expected_width
            if declared_width is not None and int(declared_width) != expected_width:
                raise ValueError(
                    f"Stream {name!r} declares {declared_width} values per frame; "
                    f"the landmark set expects {expected_width}."
                )
        try:
            data = base64.b64decode(str(raw.get("data_b64") or ""), validate=True)
        except Exception as decode_error:  # noqa: BLE001
            raise ValueError(f"Stream {name!r} is not valid base64.") from decode_error
        frames = decode_frames(data, width)
        if sparse_neural_body:
            frames = expand_sparse_body_frames(frames, present_joint_names)
        declared_count = raw.get("frame_count")
        if declared_count is not None and int(declared_count) != frames.shape[0]:
            raise ValueError(
                f"Stream {name!r} declares {declared_count} frames but holds "
                f"{frames.shape[0]}."
            )
        rate = float(raw.get("sample_rate_hz") or 0.0)
        if rate <= 0:
            raise ValueError(f"Stream {name!r} needs a positive sample rate.")
        if not np.all(np.isfinite(frames)):
            raise ValueError(f"Stream {name!r} holds non-finite values.")
        streams[name] = StreamWindow(frames=frames, sample_rate_hz=rate)
    if "face" in streams and face_encoding == FACE_ENCODING_NONE:
        raise ValueError("A face stream must say whether it is dense or basis-encoded.")
    speech = payload.get("speech") or []
    if not isinstance(speech, list):
        speech = []
    confidence = payload.get("identity_confidence")
    return MotionWindow(
        landmark_set_version=landmark_set.version,
        source=source,
        emotion=str(payload.get("emotion") or "neutral"),
        captured_at=payload.get("captured_at"),
        streams=streams,
        face_encoding=face_encoding,
        basis_id=(str(payload["basis_id"]) if payload.get("basis_id") else None),
        speech=[segment for segment in speech if isinstance(segment, dict)],
        identity_confidence=(float(confidence) if confidence is not None else None),
        source_document_name=payload.get("source_document_name"),
    )
