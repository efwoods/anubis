"""Recording a window: the one path every source goes through.

A window arrives from the browser's ambient loop, from the identity
pipeline's video branch, from a fidelity check over a generated clip, or from
a decoder reading motor cortex. From here on the source is a label. The
window is normalized, its face is encoded against the avatar's basis (or
kept dense and added to the golden set until a basis can be fitted), its
scalars are folded into the signature, its events are merged into the
primitive dictionary, the byte budgets are enforced, and the profile text is
re-rendered. Basis fitting and merging are pure CPU on the request path's
own process, so callers run this under ``schedule_background`` or in a
worker thread, never inline with a reply.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.anubis.utils.motion.basis import MotionBasis, fit_basis
from src.anubis.utils.motion.codec import (
    FACE_ENCODING_BASIS,
    FACE_ENCODING_DENSE,
    FACE_ENCODING_NONE,
    MotionWindow,
    StreamWindow,
    decode_frames,
    encode_frames,
)
from src.anubis.utils.motion.landmarks import get_landmark_set
from src.anubis.utils.motion.measurements import measure_window, measurement_seconds
from src.anubis.utils.motion.motion_prompt import (
    render_motion_block,
    render_role_section,
)
from src.anubis.utils.motion.primitives import (
    Primitive,
    events_to_primitives,
    extract_events,
    merge_primitives,
    primitive_fidelity,
)
from src.anubis.utils.motion.signature import (
    fold_measurements,
    golden_segments_to_prune,
    signature_value,
    tracks_to_prune,
)

logger = logging.getLogger(__name__)


def _setting(context: Any, name: str, default: float) -> float:
    value = getattr(context, name, None)
    try:
        return float(value) if value is not None and str(value).strip() != "" else float(default)
    except (TypeError, ValueError):
        return float(default)


def _flag(context: Any, name: str, default: bool = True) -> bool:
    value = getattr(context, name, None)
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


async def load_current_basis(repository: Any, assistant_id: str) -> MotionBasis | None:
    """Load the avatar's current basis, or ``None``."""
    record = await repository.get_current_basis(assistant_id)
    return MotionBasis.from_record(record) if record else None


async def load_basis(repository: Any, basis_id: str | None) -> MotionBasis | None:
    """Load a basis by id, or ``None``."""
    if not basis_id:
        return None
    record = await repository.get_basis(basis_id)
    return MotionBasis.from_record(record) if record else None


def _stream_columns(prefix: str, stream: StreamWindow | None) -> dict[str, Any]:
    if stream is None:
        return {f"{prefix}_sample_rate_hz": None, f"{prefix}_frame_count": None, prefix: None}
    return {
        f"{prefix}_sample_rate_hz": float(stream.sample_rate_hz),
        f"{prefix}_frame_count": stream.frame_count,
        prefix: encode_frames(stream.frames),
    }


async def _maybe_fit_basis(
    repository: Any, context: Any, *, user_id: str, assistant_id: str, landmark_set_version: str
) -> MotionBasis | None:
    """Fit the first basis, or refit, once the golden set holds enough seconds."""
    segments = await repository.list_golden_segments(assistant_id)
    total_seconds = sum(float(segment.get("duration_seconds") or 0.0) for segment in segments)
    minimum = _setting(context, "motion_basis_min_seconds", 60.0)
    if total_seconds < minimum:
        return None
    current = await repository.get_current_basis(assistant_id)
    if current is not None:
        # Refit only when the golden set has grown well past what the basis was fitted on.
        fitted = float(current.get("fitted_seconds") or 0.0)
        if total_seconds < fitted * 1.5:
            return MotionBasis.from_record(current)
    landmark_set = get_landmark_set(landmark_set_version)
    width = landmark_set.stream("face").values_per_frame
    residuals: list[np.ndarray] = []
    for meta in segments:
        full = await repository.get_golden_segment(meta["segment_id"])
        if not full or not full.get("face"):
            continue
        residuals.append(decode_frames(full["face"], width))
    if not residuals:
        return None
    stacked = np.concatenate(residuals, axis=0)
    try:
        basis = fit_basis(
            stacked,
            landmark_set_version=landmark_set_version,
            component_count=int(_setting(context, "motion_basis_components", 48)),
            fitted_seconds=total_seconds,
        )
    except ValueError as fit_error:
        logger.info("Basis not fitted for %s yet: %s", assistant_id, fit_error)
        return None
    record = basis.to_record()
    record.update({"user_id": user_id, "assistant_id": assistant_id})
    basis.basis_id = await repository.add_basis(record)
    await repository.delete_unreferenced_bases(assistant_id)
    logger.info(
        "Fitted a %d-component motion basis for %s from %.0f s (reconstruction error %.4f)",
        basis.component_count, assistant_id, total_seconds, basis.reconstruction_error,
    )
    return basis


async def record_motion_window(
    repository: Any,
    context: Any,
    *,
    user_id: str,
    assistant_id: str,
    window: MotionWindow,
) -> dict[str, Any]:
    """Persist one window and fold it into everything derived. Returns a summary."""
    if not _flag(context, "motion_learning_enabled", True):
        return {"recorded": False, "reason": "disabled"}
    landmark_set = get_landmark_set(window.landmark_set_version)
    emotion = window.emotion or "neutral"
    basis = await load_current_basis(repository, assistant_id)
    face_stream = window.streams.get("face")
    face_encoding = window.face_encoding
    basis_id = window.basis_id

    # A window already encoded by the browser against a basis we hold is used
    # as-is; anything dense is encoded here when a basis exists, and kept in
    # the golden set while the basis is still being learned.
    if face_stream is not None and face_encoding == FACE_ENCODING_BASIS:
        basis = await load_basis(repository, basis_id) or basis
        if basis is None or basis.component_count != face_stream.frames.shape[1]:
            logger.info("Dropping a basis-encoded face stream with no matching basis for %s", assistant_id)
            window.streams.pop("face", None)
            face_stream = None
            face_encoding = FACE_ENCODING_NONE
            basis_id = None
    golden_added = False
    dense_face: np.ndarray | None = None
    if face_stream is not None and face_encoding == FACE_ENCODING_DENSE:
        dense_face = face_stream.frames
        budget = _setting(context, "motion_golden_seconds_per_avatar", 120.0)
        existing = await repository.list_golden_segments(assistant_id)
        held = sum(float(segment.get("duration_seconds") or 0.0) for segment in existing)
        if basis is None or held < budget:
            await repository.add_golden_segment(
                {
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "source": window.source,
                    "emotion": emotion,
                    "landmark_set_version": landmark_set.version,
                    "duration_seconds": window.duration_seconds,
                    "identity_confidence": window.identity_confidence,
                    "byte_length": window.byte_length(),
                    **_stream_columns("body", window.streams.get("body")),
                    **_stream_columns("head", window.streams.get("head_pose")),
                    **_stream_columns("face", face_stream),
                }
            )
            golden_added = True
        if basis is None:
            basis = await _maybe_fit_basis(
                repository, context, user_id=user_id, assistant_id=assistant_id,
                landmark_set_version=landmark_set.version,
            )
        if basis is not None:
            coefficients = basis.encode(dense_face)
            window.streams["face"] = StreamWindow(frames=coefficients, sample_rate_hz=face_stream.sample_rate_hz)
            window.face_encoding = FACE_ENCODING_BASIS
            window.basis_id = basis.basis_id
            face_encoding = FACE_ENCODING_BASIS
            basis_id = basis.basis_id
            face_stream = window.streams["face"]

    track_id = await repository.add_track(
        {
            "user_id": user_id,
            "assistant_id": assistant_id,
            "source": window.source,
            "source_document_name": window.source_document_name,
            "emotion": emotion,
            "landmark_set_version": landmark_set.version,
            "basis_id": basis_id,
            "face_encoding": face_encoding,
            "captured_at": window.captured_at,
            "duration_seconds": window.duration_seconds,
            "speech": window.speech,
            "identity_confidence": window.identity_confidence,
            "byte_length": window.byte_length(),
            **_stream_columns("body", window.streams.get("body")),
            **_stream_columns("head", window.streams.get("head_pose")),
            **_stream_columns("face", face_stream),
        }
    )

    # Scalars: measure with the dense face when we still have it, so the
    # bootstrap windows contribute eye and mouth readings too.
    measure_input = window
    if dense_face is not None and face_encoding == FACE_ENCODING_BASIS:
        measure_input = MotionWindow(
            landmark_set_version=window.landmark_set_version, source=window.source, emotion=emotion,
            captured_at=window.captured_at,
            streams={**window.streams, "face": StreamWindow(frames=dense_face, sample_rate_hz=face_stream.sample_rate_hz)},  # type: ignore[union-attr]
            face_encoding=FACE_ENCODING_DENSE, speech=window.speech,
        )
    measurements = measure_window(measure_input, basis)
    seconds = measurement_seconds(measurements) or window.duration_seconds
    existing_signature = await repository.get_signature(assistant_id, emotion) or {}
    folded = fold_measurements(existing_signature.get("signature") or {}, measurements)
    await repository.upsert_signature(
        {
            "assistant_id": assistant_id,
            "emotion": emotion,
            "user_id": user_id,
            "windows_observed": int(existing_signature.get("windows_observed") or 0) + 1,
            "seconds_observed": float(existing_signature.get("seconds_observed") or 0.0) + float(seconds),
            "signature": folded,
        }
    )

    # Primitives.
    events = extract_events(window, basis)
    new_primitives = events_to_primitives(events, emotion=emotion, landmark_set_version=landmark_set.version)
    stored = [Primitive.from_record(record) for record in await repository.list_primitives(assistant_id, emotion=emotion)]
    merged = merge_primitives(
        stored, new_primitives, max_per_channel=int(_setting(context, "motion_primitive_max_count", 8))
    )
    await repository.replace_primitives(assistant_id, emotion, [p.to_record() for p in merged], user_id=user_id)

    # Budgets.
    pruned_tracks = tracks_to_prune(
        await repository.list_tracks(assistant_id),
        retention_seconds=_setting(context, "motion_track_retention_seconds_per_avatar", 600.0),
    )
    if pruned_tracks:
        await repository.delete_tracks(pruned_tracks)
        await repository.delete_unreferenced_bases(assistant_id)
    pruned_golden = golden_segments_to_prune(
        await repository.list_golden_segments(assistant_id),
        budget_seconds=_setting(context, "motion_golden_seconds_per_avatar", 120.0),
    )
    if pruned_golden:
        await repository.delete_golden_segments(pruned_golden)

    profile = await render_profile(repository, context, user_id=user_id, assistant_id=assistant_id)
    return {
        "recorded": True,
        "track_id": track_id,
        "emotion": emotion,
        "seconds": float(seconds),
        "measurements": sorted(measurements.keys()),
        "events": len(events),
        "primitives": len(merged),
        "basis_id": basis.basis_id if basis else None,
        "golden_added": golden_added,
        "pruned_tracks": len(pruned_tracks),
        "pruned_golden": len(pruned_golden),
        "role_section_characters": len(profile.get("role_section") or ""),
    }


async def render_profile(
    repository: Any, context: Any, *, user_id: str, assistant_id: str
) -> dict[str, Any]:
    """Re-render every emotion's block and the ROLE section from what is stored."""
    basis = await load_current_basis(repository, assistant_id)
    landmark_set = get_landmark_set(basis.landmark_set_version) if basis else None
    min_seconds = _setting(context, "motion_signature_min_seconds", 30.0)
    min_occurrences = int(_setting(context, "motion_primitive_min_occurrences", 3))
    blocks: dict[str, str] = {}
    total_seconds = 0.0
    for record in await repository.list_signatures(assistant_id):
        emotion = str(record["emotion"])
        total_seconds += float(record.get("seconds_observed") or 0.0)
        primitives = [Primitive.from_record(p) for p in await repository.list_primitives(assistant_id, emotion=emotion)]
        block = render_motion_block(
            record.get("signature") or {},
            primitives,
            min_seconds=min_seconds,
            min_occurrences=min_occurrences,
            basis=basis,
            landmark_set=landmark_set,
        )
        if block:
            blocks[emotion] = block
    existing = await repository.get_profile(assistant_id) or {}
    profile = {
        "assistant_id": assistant_id,
        "user_id": user_id,
        "role_section": render_role_section(blocks),
        "blocks": blocks,
        "motion_fidelity": existing.get("motion_fidelity") or {},
        "seconds_observed": total_seconds,
    }
    await repository.upsert_profile(profile)
    return profile


async def record_fidelity(
    repository: Any,
    context: Any,
    *,
    user_id: str,
    assistant_id: str,
    emotion: str,
    generated_window: MotionWindow,
    asset_id: str | None = None,
) -> dict[str, Any]:
    """Score a generated clip's motion against the person's; store the score on the profile.

    The generated clip is measured with the same code as the person, then
    compared scalar by scalar (relative error) and primitive by primitive
    (prototype similarity). This is the actual-versus-predicted chart per
    named joint from the design references, as numbers.
    """
    basis = await load_current_basis(repository, assistant_id)
    generated = measure_window(generated_window, basis)
    reference = (await repository.get_signature(assistant_id, emotion) or {}).get("signature") or {}
    scalar_scores: dict[str, float] = {}
    for key, measurement in generated.items():
        actual = signature_value(reference, key)
        if actual is None:
            continue
        if isinstance(actual, str) or isinstance(measurement.value, str):
            scalar_scores[key] = 1.0 if str(actual) == str(measurement.value) else 0.0
            continue
        scale = max(abs(float(actual)), 1e-6)
        scalar_scores[key] = float(max(0.0, 1.0 - abs(float(measurement.value) - float(actual)) / scale))
    generated_events = extract_events(generated_window, basis)
    generated_primitives = events_to_primitives(
        generated_events, emotion=emotion, landmark_set_version=generated_window.landmark_set_version
    )
    reference_primitives = [Primitive.from_record(p) for p in await repository.list_primitives(assistant_id, emotion=emotion)]
    primitive_scores: dict[str, float] = {}
    for reference_primitive in reference_primitives:
        candidates = [p for p in generated_primitives if p.channel == reference_primitive.channel]
        if not candidates:
            primitive_scores[f"{reference_primitive.channel}:{reference_primitive.primitive_id[:8]}"] = 0.0
            continue
        primitive_scores[f"{reference_primitive.channel}:{reference_primitive.primitive_id[:8]}"] = max(
            primitive_fidelity(reference_primitive, candidate) for candidate in candidates
        )
    all_scores = list(scalar_scores.values()) + list(primitive_scores.values())
    overall = float(np.mean(all_scores)) if all_scores else 0.0
    profile = await repository.get_profile(assistant_id) or {
        "assistant_id": assistant_id, "user_id": user_id, "role_section": "", "blocks": {}, "seconds_observed": 0.0,
    }
    fidelity = dict(profile.get("motion_fidelity") or {})
    fidelity[emotion] = {
        "overall": overall,
        "scalars": scalar_scores,
        "primitives": primitive_scores,
        "asset_id": asset_id,
    }
    await repository.upsert_profile({**profile, "motion_fidelity": fidelity})
    return fidelity[emotion]


__all__ = [
    "load_basis",
    "load_current_basis",
    "record_fidelity",
    "record_motion_window",
    "render_profile",
]
