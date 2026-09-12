"""How the person moves: named coordinates through time, learned and rendered.

A wireframe over the person — body joints and the face mesh — is recorded as
a timeline of *named* coordinates from whatever source has them (a webcam in
the browser, an uploaded video, a clip the platform generated, a decoder
reading motor cortex), normalized so the camera drops out, encoded against
the person's own expression basis, cut into recurring movements with their
prototype trajectories, folded into a converging signature, and rendered as
deterministic behavioural text that drives the avatar's words, its stills,
its idle loops and its lip-synced clips.

Nothing here touches the LangGraph store. See :mod:`repository`.
"""

from src.anubis.utils.motion.repository import (
    MOTION_SOURCES,
    SOURCE_GENERATED_CLIP,
    SOURCE_LIVE_CAMERA,
    SOURCE_NEURAL_DECODER,
    SOURCE_UPLOADED_VIDEO,
    InMemoryMotionRepository,
    PostgresMotionRepository,
    ensure_motion_tables,
    get_motion_repository,
    set_motion_repository,
)

__all__ = [
    "MOTION_SOURCES",
    "SOURCE_GENERATED_CLIP",
    "SOURCE_LIVE_CAMERA",
    "SOURCE_NEURAL_DECODER",
    "SOURCE_UPLOADED_VIDEO",
    "InMemoryMotionRepository",
    "PostgresMotionRepository",
    "ensure_motion_tables",
    "get_motion_repository",
    "set_motion_repository",
]
