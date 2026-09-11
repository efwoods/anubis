"""Who is being learned from — a policy per source, and it fails closed.

The platform must never learn a stranger's body as the avatar's own. Every
window is checked before it is recorded, and the check depends on where the
coordinates came from:

* ``live_camera`` — the person is proved by sight: the camera must face the
  person (``self``), the avatar must be the caller's own personal avatar, and
  a vision call must confirm the face in the frame is the face in the stored
  reference image. Verified once per capture session and again every
  ``MOTION_IDENTITY_REVERIFY_SECONDS``.
* ``uploaded_video`` — the identity pipeline runs for every avatar, so only
  the vision match against that avatar's reference image applies; in
  footage with several people, only the tracked person whose crop matches
  is kept.
* ``neural_decoder`` — there is no face. The person is proved by
  authentication: the caller's own account and personal avatar.
* ``generated_clip`` — the platform's own output, checked by nobody; it is
  never folded into the signature, only scored against it.

Unlike ``reference_subject.classify_reference_subject``, which falls back to
"person, low risk" when the classifier is down, nothing here falls open. A
classifier outage means the window is dropped.
"""

from __future__ import annotations

import logging
import time

from pydantic import BaseModel, Field

from src.anubis.utils.motion.repository import (
    SOURCE_GENERATED_CLIP,
    SOURCE_LIVE_CAMERA,
    SOURCE_NEURAL_DECODER,
    SOURCE_UPLOADED_VIDEO,
)

logger = logging.getLogger(__name__)


class MotionSubjectPresence(BaseModel):
    """Whether the person in the reference photograph is the person in this frame."""

    same_person: bool = Field(description="Whether the face in the second image is the face in the first image.")
    is_the_focus: bool = Field(description="Whether that person is the subject of the second image rather than incidental.")
    confidence: float = Field(ge=0.0, le=1.0, description="How sure, from 0.0 to 1.0, that both images show the same person.")
    reasoning: str = Field(description="One or two sentences explaining the judgement.")


async def confirm_person_matches_reference(
    reference_image_data_uri: str, frame_data_uri: str
) -> MotionSubjectPresence | None:
    """One vision call: is the person in ``frame_data_uri`` the person in the reference?

    Returns ``None`` when the comparison could not be made; callers treat
    ``None`` as a refusal.
    """
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from src.anubis.utils.model import init_image_description_model
        from src.anubis.utils.prompts.system_prompts import (
            MOTION_SUBJECT_IDENTITY_PROMPT,
        )

        model = init_image_description_model().with_structured_output(schema=MotionSubjectPresence)
        response = await model.ainvoke(
            [
                SystemMessage(content=MOTION_SUBJECT_IDENTITY_PROMPT),
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "The first image is the reference photograph. The second image is a frame from a camera or a video.",
                        },
                        {"type": "image_url", "image_url": {"url": reference_image_data_uri}},
                        {"type": "image_url", "image_url": {"url": frame_data_uri}},
                    ]
                ),
            ]
        )
        if isinstance(response, MotionSubjectPresence):
            return response
        return MotionSubjectPresence.model_validate(response)
    except Exception as identity_error:  # noqa: BLE001 - fail closed, never open
        logger.info("Could not compare a motion frame with the reference image: %s", identity_error)
        return None


def presence_accepts(presence: MotionSubjectPresence | None, *, minimum_confidence: float) -> bool:
    """Apply the acceptance rule: same person, the focus, and confident enough."""
    if presence is None:
        return False
    return bool(presence.same_person and presence.is_the_focus and presence.confidence >= minimum_confidence)


class IdentityVerdict(BaseModel):
    """IdentityVerdict."""
    accepted: bool
    confidence: float = 0.0
    reason: str = ""


class LiveCameraIdentityCache:
    """Remember a verified capture session so each window does not cost a vision call."""

    def __init__(self) -> None:
        """Initialize."""
        self._verified: dict[tuple[str, str], tuple[float, float]] = {}

    def get(self, user_id: str, assistant_id: str, *, max_age_seconds: float) -> float | None:
        """Return the cached confidence, or ``None`` when absent or stale."""
        entry = self._verified.get((user_id, assistant_id))
        if entry is None:
            return None
        verified_at, confidence = entry
        if time.monotonic() - verified_at > max_age_seconds:
            return None
        return confidence

    def set(self, user_id: str, assistant_id: str, confidence: float) -> None:
        """Remember a verification for this owner and avatar."""
        self._verified[(user_id, assistant_id)] = (time.monotonic(), confidence)

    def forget(self, user_id: str, assistant_id: str) -> None:
        """Drop any cached verification for this owner and avatar."""
        self._verified.pop((user_id, assistant_id), None)


live_camera_identity_cache = LiveCameraIdentityCache()


async def verify_identity(
    *,
    source: str,
    user_id: str,
    assistant_id: str,
    is_personal_avatar: bool,
    camera_facing: str | None,
    reference_image_data_uri: str | None,
    frame_data_uri: str | None,
    minimum_confidence: float,
    reverify_seconds: float,
) -> IdentityVerdict:
    """Apply the per-source policy. Anything not positively accepted is refused."""
    if source == SOURCE_GENERATED_CLIP:
        return IdentityVerdict(accepted=True, confidence=1.0, reason="platform output")
    if source == SOURCE_NEURAL_DECODER:
        if not is_personal_avatar:
            return IdentityVerdict(accepted=False, reason="a neural source may only drive the caller's own personal avatar")
        return IdentityVerdict(accepted=True, confidence=1.0, reason="authenticated owner")
    if source == SOURCE_LIVE_CAMERA:
        if not is_personal_avatar:
            return IdentityVerdict(accepted=False, reason="the live camera may only teach the caller's own personal avatar")
        if (camera_facing or "self") != "self":
            return IdentityVerdict(accepted=False, reason="the camera is not facing the person")
        cached = live_camera_identity_cache.get(user_id, assistant_id, max_age_seconds=reverify_seconds)
        if cached is not None:
            return IdentityVerdict(accepted=True, confidence=cached, reason="verified earlier this session")
    if source not in (SOURCE_LIVE_CAMERA, SOURCE_UPLOADED_VIDEO):
        return IdentityVerdict(accepted=False, reason=f"unknown source {source!r}")
    if not reference_image_data_uri:
        return IdentityVerdict(accepted=False, reason="the avatar has no reference image to match against")
    if not frame_data_uri:
        return IdentityVerdict(accepted=False, reason="no frame was supplied to match against the reference image")
    presence = await confirm_person_matches_reference(reference_image_data_uri, frame_data_uri)
    if not presence_accepts(presence, minimum_confidence=minimum_confidence):
        reason = presence.reasoning if presence else "the comparison could not be made"
        return IdentityVerdict(accepted=False, confidence=(presence.confidence if presence else 0.0), reason=reason)
    if source == SOURCE_LIVE_CAMERA:
        live_camera_identity_cache.set(user_id, assistant_id, presence.confidence)  # type: ignore[union-attr]
    return IdentityVerdict(accepted=True, confidence=presence.confidence, reason=presence.reasoning)  # type: ignore[union-attr]


__all__ = [
    "IdentityVerdict",
    "MotionSubjectPresence",
    "confirm_person_matches_reference",
    "live_camera_identity_cache",
    "presence_accepts",
    "verify_identity",
]
