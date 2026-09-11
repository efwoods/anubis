"""Recording windows end to end against the in-memory repository, the identity gate, and the prompt seams."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.anubis.utils.media_generation import prompts as media_prompts
from src.anubis.utils.motion import codec, identity, landmarks, service
from src.anubis.utils.motion.repository import (
    SOURCE_GENERATED_CLIP,
    SOURCE_LIVE_CAMERA,
    SOURCE_NEURAL_DECODER,
    SOURCE_UPLOADED_VIDEO,
    InMemoryMotionRepository,
)
from unit_tests.motion_fixtures import body_stream as _body_stream
from unit_tests.motion_fixtures import head_stream as _head_stream
from unit_tests.motion_fixtures import sweeping_body as _sweeping_body
from unit_tests.motion_fixtures import synthetic_face as _synthetic_face


def _context(**overrides):
    base = {
        "motion_learning_enabled": "true",
        "motion_basis_min_seconds": 4.0,
        "motion_basis_components": 3,
        "motion_golden_seconds_per_avatar": 120.0,
        "motion_track_retention_seconds_per_avatar": 600.0,
        "motion_primitive_max_count": 8,
        "motion_primitive_min_occurrences": 3,
        "motion_signature_min_seconds": 20.0,
        "motion_prompt_enabled": "true",
        "lip_sync_prompt_enabled": "true",
        "lip_sync_cinematic_prompt": "Medium close-up.",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _window(*, with_face: bool, seconds: float = 30.0, emotion: str = "neutral") -> codec.MotionWindow:
    from src.anubis.utils.motion.normalize import canonicalize_face_frames

    streams = {"body": _sweeping_body(seconds=seconds), "head_pose": _head_stream(int(30 * seconds))}
    encoding = codec.FACE_ENCODING_NONE
    if with_face:
        streams["face"] = codec.StreamWindow(canonicalize_face_frames(_synthetic_face(int(30 * seconds), amplitude=0.03)), 30.0)
        encoding = codec.FACE_ENCODING_DENSE
    return codec.MotionWindow(streams=streams, face_encoding=encoding, emotion=emotion, source=SOURCE_LIVE_CAMERA)


@pytest.mark.asyncio
async def test_recording_windows_fits_a_basis_and_renders_a_profile():
    repository = InMemoryMotionRepository()
    context = _context()
    first = await service.record_motion_window(repository, context, user_id="u", assistant_id="a", window=_window(with_face=True, seconds=10.0))
    assert first["recorded"] and first["golden_added"]
    # 10 s of golden face is enough (min 4 s) so a basis exists and the track was encoded.
    assert first["basis_id"] is not None
    track = await repository.get_track(first["track_id"])
    assert track["face_encoding"] == codec.FACE_ENCODING_BASIS
    assert track["face_frame_count"] == 300
    second = await service.record_motion_window(repository, context, user_id="u", assistant_id="a", window=_window(with_face=True, seconds=30.0))
    assert second["recorded"]
    signature = await repository.get_signature("a", "neutral")
    assert signature["windows_observed"] == 2
    assert signature["seconds_observed"] >= 40.0
    profile = await repository.get_profile("a")
    assert "HEAD:" in profile["role_section"]
    assert "HANDS:" in profile["role_section"]
    assert "neutral" in profile["blocks"]


@pytest.mark.asyncio
async def test_disabled_learning_records_nothing():
    repository = InMemoryMotionRepository()
    result = await service.record_motion_window(repository, _context(motion_learning_enabled="false"), user_id="u", assistant_id="a", window=_window(with_face=False))
    assert result == {"recorded": False, "reason": "disabled"}
    assert await repository.list_tracks("a") == []


@pytest.mark.asyncio
async def test_retention_prunes_oldest_tracks():
    repository = InMemoryMotionRepository()
    context = _context(motion_track_retention_seconds_per_avatar=45.0)
    for _ in range(3):
        await service.record_motion_window(repository, context, user_id="u", assistant_id="a", window=_window(with_face=False, seconds=30.0))
    remaining = await repository.list_tracks("a")
    assert sum(t["duration_seconds"] for t in remaining) <= 45.0 + 1e-6


@pytest.mark.asyncio
async def test_fidelity_scores_a_generated_clip_against_the_person():
    repository = InMemoryMotionRepository()
    context = _context()
    await service.record_motion_window(repository, context, user_id="u", assistant_id="a", window=_window(with_face=False, seconds=30.0))
    same = _window(with_face=False, seconds=30.0)
    same.source = SOURCE_GENERATED_CLIP
    score = await service.record_fidelity(repository, context, user_id="u", assistant_id="a", emotion="neutral", generated_window=same, asset_id="clip")
    assert score["overall"] > 0.8
    still = codec.MotionWindow(streams={"body": _body_stream(450)}, source=SOURCE_GENERATED_CLIP)
    worse = await service.record_fidelity(repository, context, user_id="u", assistant_id="a", emotion="neutral", generated_window=still)
    assert worse["overall"] < score["overall"]
    assert (await repository.get_profile("a"))["motion_fidelity"]["neutral"]["overall"] == worse["overall"]


# --- identity ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_identity_policy_per_source(monkeypatch):
    async def _no_comparison(*_args, **_kwargs):
        return None

    monkeypatch.setattr(identity, "confirm_person_matches_reference", _no_comparison)
    identity.live_camera_identity_cache.forget("u", "a")
    common = dict(user_id="u", assistant_id="a", minimum_confidence=0.8, reverify_seconds=600.0, reference_image_data_uri="data:ref", frame_data_uri="data:frame")
    # A classifier outage fails closed.
    verdict = await identity.verify_identity(source=SOURCE_LIVE_CAMERA, is_personal_avatar=True, camera_facing="self", **common)
    assert not verdict.accepted
    # A world-facing camera never teaches.
    verdict = await identity.verify_identity(source=SOURCE_LIVE_CAMERA, is_personal_avatar=True, camera_facing="world", **common)
    assert not verdict.accepted and "not facing" in verdict.reason
    # A neural source is proved by authentication, on the personal avatar only.
    assert (await identity.verify_identity(source=SOURCE_NEURAL_DECODER, is_personal_avatar=True, camera_facing=None, **common)).accepted
    assert not (await identity.verify_identity(source=SOURCE_NEURAL_DECODER, is_personal_avatar=False, camera_facing=None, **common)).accepted
    # Platform output is never gated.
    assert (await identity.verify_identity(source=SOURCE_GENERATED_CLIP, is_personal_avatar=False, camera_facing=None, **common)).accepted


@pytest.mark.asyncio
async def test_identity_accepts_a_confident_match_and_caches_it(monkeypatch):
    async def _match(*_args, **_kwargs):
        return identity.MotionSubjectPresence(same_person=True, is_the_focus=True, confidence=0.93, reasoning="same jaw and brows")

    monkeypatch.setattr(identity, "confirm_person_matches_reference", _match)
    identity.live_camera_identity_cache.forget("u", "a")
    common = dict(user_id="u", assistant_id="a", minimum_confidence=0.8, reverify_seconds=600.0, reference_image_data_uri="data:ref", frame_data_uri="data:frame")
    verdict = await identity.verify_identity(source=SOURCE_LIVE_CAMERA, is_personal_avatar=True, camera_facing="self", **common)
    assert verdict.accepted and verdict.confidence == 0.93
    calls = []

    async def _count(*_args, **_kwargs):
        calls.append(1)
        return None

    monkeypatch.setattr(identity, "confirm_person_matches_reference", _count)
    again = await identity.verify_identity(source=SOURCE_LIVE_CAMERA, is_personal_avatar=True, camera_facing="self", **common)
    assert again.accepted and not calls  # cached, no second vision call
    # Uploaded video for a non-personal avatar still needs the match.
    below = identity.MotionSubjectPresence(same_person=True, is_the_focus=False, confidence=0.9, reasoning="in a crowd")
    assert not identity.presence_accepts(below, minimum_confidence=0.8)


# --- the generation prompt seams --------------------------------------------


def test_still_and_idle_prompts_take_the_motion_block():
    block = "HEAD: rests tilted.\nEYES: blinks 17 times a minute.\nPOSTURE: leans forward.\nHANDS: hands rest at chest."
    generic = media_prompts.idle_loop_prompt_for("joy")
    assert "occasional subtle fidget" in generic
    measured = media_prompts.idle_loop_prompt_for("joy", motion_prompt=block)
    assert "occasional subtle fidget" not in measured
    assert "blinks 17 times a minute" in measured
    still = media_prompts.still_prompt_for("joy", motion_prompt=block)
    assert "leans forward" in still and "blinks 17" not in still
    assert media_prompts.still_prompt_for("joy") == media_prompts.still_prompt_for("joy", motion_prompt="")
    assert "no person" in media_prompts.idle_loop_prompt_for("joy", "non_human", motion_prompt=block)


def test_lip_sync_prompt_is_layered_and_keyed_into_the_cache():
    from src.anubis.utils.media_generation import lip_sync

    context = _context()
    prompt = lip_sync.build_lip_sync_prompt(context, "HEAD: nods.")
    assert prompt.startswith("Medium close-up.")
    assert prompt.endswith("HEAD: nods.")
    assert lip_sync.build_lip_sync_prompt(_context(lip_sync_prompt_enabled="false"), "HEAD: nods.") is None
    assert lip_sync.text_digest("Hello there") != lip_sync.text_digest("Hello there", prompt)
    assert lip_sync.text_digest("Hello there", prompt) == lip_sync.text_digest("hello  THERE", prompt)


def test_landmark_set_version_is_recorded_on_tracks():
    window = _window(with_face=False, seconds=2.0)
    assert window.landmark_set_version == landmarks.DEFAULT_LANDMARK_SET_VERSION
    assert window.source in (SOURCE_LIVE_CAMERA, SOURCE_UPLOADED_VIDEO)
    assert np.isfinite(window.duration_seconds)
