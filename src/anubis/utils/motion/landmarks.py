"""The versioned landmark vocabulary every motion track is written against.

A motion track is a timeline of *named* coordinates. Where the coordinates
came from — a webcam in the browser, a video uploaded to the identity
pipeline, a clip the platform itself generated, or a decoder reading motor
cortex — is recorded on the track as its ``source`` and matters nowhere past
this module. What matters is the **landmark set**: which named points, in
which order, with how many values each, grouped into which anatomical
regions. Every stored row records the version of the set it was written
against, so a later, denser set (hand landmarks, iris points, a limb set from
a neural decoder) is a new registration here and a version bump on new rows,
never a rewrite of old ones.

Naming follows MediaPipe for the body (BlazePose's 33 joints) and the face
(the 468-point mesh plus the 10 iris points). Face regions are named from the
subject's own left and right, which is how MediaPipe labels the mesh.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MOTION_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Body: BlazePose's 33 named joints, in MediaPipe order.
# ---------------------------------------------------------------------------

BODY_JOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

BODY_JOINT_INDEX: dict[str, int] = {
    name: index for index, name in enumerate(BODY_JOINT_NAMES)
}

# Each body joint carries x, y, z and a visibility score.
BODY_VALUES_PER_JOINT = 4

BODY_GROUPS: dict[str, tuple[str, ...]] = {
    "head": (
        "nose",
        "left_eye_inner",
        "left_eye",
        "left_eye_outer",
        "right_eye_inner",
        "right_eye",
        "right_eye_outer",
        "left_ear",
        "right_ear",
        "mouth_left",
        "mouth_right",
    ),
    "shoulders": ("left_shoulder", "right_shoulder"),
    "left_arm": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_arm": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_hand": ("left_wrist", "left_pinky", "left_index", "left_thumb"),
    "right_hand": ("right_wrist", "right_pinky", "right_index", "right_thumb"),
    "torso": ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
    "left_leg": ("left_hip", "left_knee", "left_ankle"),
    "right_leg": ("right_hip", "right_knee", "right_ankle"),
}

# ---------------------------------------------------------------------------
# Face: the 468-point mesh plus 10 iris points, each x, y, z.
# ---------------------------------------------------------------------------

FACE_POINT_COUNT = 478
FACE_VALUES_PER_POINT = 3

# Canonical MediaPipe index sets, used to say which part of the face a basis
# component moves. "left" and "right" are the subject's own.
FACE_REGIONS: dict[str, tuple[int, ...]] = {
    "lips": (
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402,
        317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415,
        310, 311, 312, 13, 82, 81, 42, 183, 78,
    ),
    "jaw": (152, 148, 176, 149, 150, 136, 172, 377, 400, 378, 379, 365, 397, 199, 175, 18, 200),
    "left_eye": (263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398),
    "right_eye": (33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173),
    "left_brow": (276, 283, 282, 295, 285, 300, 293, 334, 296, 336),
    "right_brow": (46, 53, 52, 65, 55, 70, 63, 105, 66, 107),
    "brow_center": (9, 8, 168, 6, 107, 336, 55, 285),
    "nose": (168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 326, 327),
    "left_cheek": (425, 427, 411, 416, 376, 352, 280, 330, 266, 423, 426),
    "right_cheek": (205, 207, 187, 192, 147, 123, 50, 101, 36, 203, 206),
    "left_nasolabial": (429, 358, 279, 420, 456, 391, 393, 322),
    "right_nasolabial": (209, 129, 49, 198, 236, 165, 167, 92),
    "forehead": (10, 338, 297, 332, 109, 67, 103, 54, 151, 9, 8, 21, 251),
    "left_iris": (473, 474, 475, 476, 477),
    "right_iris": (468, 469, 470, 471, 472),
}

# The rigid head pose, from MediaPipe's facial transformation matrix.
HEAD_POSE_NAMES: tuple[str, ...] = (
    "yaw_degrees",
    "pitch_degrees",
    "roll_degrees",
    "translation_x",
    "translation_y",
    "scale",
)


@dataclass(frozen=True)
class Stream:
    """One channel group inside a landmark set: a name and its width per frame."""

    name: str
    point_names: tuple[str, ...]
    values_per_point: int

    @property
    def values_per_frame(self) -> int:
        """Return the stream's width in values per frame."""
        return len(self.point_names) * self.values_per_point


@dataclass(frozen=True)
class LandmarkSet:
    """A named, versioned vocabulary of coordinates.

    ``streams`` are captured at possibly different sample rates (the face at a
    higher rate than the body, because micro-expressions are faster than
    gestures), so each stream is a separate buffer on a track.
    """

    version: str
    description: str
    streams: tuple[Stream, ...]
    groups: dict[str, dict[str, tuple[int, ...]]] = field(default_factory=dict)

    def stream(self, name: str) -> Stream:
        """Return the named stream, or raise ``KeyError``."""
        for candidate in self.streams:
            if candidate.name == name:
                return candidate
        raise KeyError(f"Landmark set {self.version!r} has no stream {name!r}.")

    def has_stream(self, name: str) -> bool:
        """Report whether the set declares a stream by this name."""
        return any(candidate.name == name for candidate in self.streams)


def _face_point_names() -> tuple[str, ...]:
    return tuple(f"face_{index}" for index in range(FACE_POINT_COUNT))


MEDIAPIPE_BODY33_FACE478_V1 = LandmarkSet(
    version="mediapipe_body33_face478_v1",
    description=(
        "MediaPipe BlazePose 33 body joints (x, y, z, visibility) at a body rate, "
        "the 478-point face mesh (x, y, z) at a face rate, and the rigid head pose "
        "from the facial transformation matrix."
    ),
    streams=(
        Stream("body", BODY_JOINT_NAMES, BODY_VALUES_PER_JOINT),
        Stream("face", _face_point_names(), FACE_VALUES_PER_POINT),
        Stream("head_pose", HEAD_POSE_NAMES, 1),
    ),
    groups={
        "body": {
            group: tuple(BODY_JOINT_INDEX[name] for name in names)
            for group, names in BODY_GROUPS.items()
        },
        "face": dict(FACE_REGIONS),
    },
)

LANDMARK_SETS: dict[str, LandmarkSet] = {
    MEDIAPIPE_BODY33_FACE478_V1.version: MEDIAPIPE_BODY33_FACE478_V1,
}

DEFAULT_LANDMARK_SET_VERSION = MEDIAPIPE_BODY33_FACE478_V1.version


def register_landmark_set(landmark_set: LandmarkSet) -> None:
    """Add a landmark set (a denser mesh, hand points, a decoder's limb set)."""
    if landmark_set.version in LANDMARK_SETS:
        raise ValueError(f"Landmark set {landmark_set.version!r} is already registered.")
    LANDMARK_SETS[landmark_set.version] = landmark_set


def get_landmark_set(version: str | None) -> LandmarkSet:
    """Resolve a version string to its set; an unknown version is an error, never a guess."""
    key = str(version or "").strip() or DEFAULT_LANDMARK_SET_VERSION
    try:
        return LANDMARK_SETS[key]
    except KeyError as missing:
        raise KeyError(f"Unknown landmark set version {key!r}.") from missing


def face_region_of_point(point_index: int) -> str | None:
    """Name the region a face point belongs to, or ``None`` when unlisted."""
    for region, indices in FACE_REGIONS.items():
        if point_index in indices:
            return region
    return None
