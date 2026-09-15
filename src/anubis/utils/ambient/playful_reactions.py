"""Playful camera performances: a kid making faces at the webcam.

Ordinary ambient vision stays quiet because a person at a desk is not asking
to be spoken to. A tongue out, flared nostrils, or an arm in the mouth is the
opposite: the conversation partner is performing for the avatar. Those beats
need a short reply, a visible surprise or disgust face, and a cooldown measured
in seconds rather than minutes — otherwise the second gag lands as silence or
as the same scold again.

The face the reply shows is chosen here, not by Go Emotions on the text.
Classifying "You're so gross! Stop that! Yuck!" as anger every time is why the
same still plays on every beat. Consecutive reactions on one thread alternate
``surprise`` and ``disgust`` so the idle loop and the still swap.
"""

from __future__ import annotations

import re
import threading
from typing import Any

PLAYFUL_PERFORMANCE_KIND = "playful_performance"

PLAYFUL_PERFORMANCE_KINDS = frozenset(
    {
        PLAYFUL_PERFORMANCE_KIND,
        "silly_face",
        "gross_face",
        "making_faces",
        "camera_bit",
    }
)

# Words that mean a body-or-face gag aimed at the avatar, in a description or
# in a typed stage direction. Keep these concrete; a generic "face" would fire
# on every webcam still of a person sitting at a desk.
_GROSS_ACTION_MARKERS = (
    "tongue",
    "nostril",
    "nostrils",
    "suck on",
    "sucking on",
    "sucking",
    "gross face",
    "gross",
    "yuck",
    "stick out",
    "sticking out",
    "arm in",
    "in the mouth",
    "into the mouth",
    "show my nostril",
    "showing nostril",
    "making a face",
    "made a face",
    "silly face",
    "disgusting",
)

DEFAULT_GROSS_REACTION_LINE = "You're so gross! Stop that! Yuck!"

PLAYFUL_REACTION_EMOTIONS: tuple[str, ...] = ("surprise", "disgust")

PLAYFUL_RESPOND_SUFFIX = (
    " The named thing is a performance aimed at the avatar: a face, a tongue, "
    "nostrils, something in the mouth, a pose held to the camera. React. Be "
    "surprised or disgusted — alternate the feeling so two beats in a row do "
    "not wear the same face. Do not recap the room, the hair, the headphones, "
    "or the shelves. Do not lecture. A few words. When USER PREFERENCES or "
    "this conversation already taught a line for this kind of gag, say that "
    "line. When no line was taught, say: "
    f'"{DEFAULT_GROSS_REACTION_LINE}"'
)


_alternator_lock = threading.Lock()
_last_playful_emotion_by_thread: dict[str, str] = {}


def _normalized(text: str | None) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def mentions_playful_gag(text: str | None) -> bool:
    """Whether the words describe a face or body gag aimed at the avatar."""
    haystack = _normalized(text)
    if not haystack:
        return False
    return any(marker in haystack for marker in _GROSS_ACTION_MARKERS)


def is_playful_camera_performance(
    observation_kind: str | None = None,
    summary: str | None = None,
    body: str | None = None,
) -> bool:
    """Whether this observation is a performance for the avatar, not desk presence."""
    kind = _normalized(observation_kind).replace(" ", "_")
    if kind in PLAYFUL_PERFORMANCE_KINDS:
        return True
    return mentions_playful_gag(summary) or mentions_playful_gag(body)


def is_playful_typed_performance(text: str | None) -> bool:
    """Whether a typed turn is a stage direction for a gag happening on camera."""
    return mentions_playful_gag(text)


def next_playful_reaction_emotion(thread_id: str | None) -> str:
    """Alternate surprise and disgust on one conversation thread.

    The first beat on a thread is surprise. Each later beat flips. A missing
    thread id still flips a process-wide slot so two replies in one process
    do not both show the same face.
    """
    key = str(thread_id or "").strip() or "_unthreaded"
    with _alternator_lock:
        previous = _last_playful_emotion_by_thread.get(key)
        if previous == "surprise":
            chosen = "disgust"
        else:
            chosen = "surprise"
        _last_playful_emotion_by_thread[key] = chosen
        return chosen


def playful_reaction_sentiment(emotion: str) -> dict[str, Any]:
    """A sentiment block the voice stage and the message face already understand."""
    label = emotion if emotion in PLAYFUL_REACTION_EMOTIONS else "surprise"
    return {
        "emotion": label,
        "base_emotion": label,
        "score": 1.0,
        "source": "playful_reaction",
    }


def should_apply_playful_reaction_emotion(
    *,
    ambient: dict[str, Any] | None,
    typed_text: str | None,
) -> bool:
    """Whether this reply should wear a surprise or disgust face."""
    record = ambient or {}
    if record.get("decision") == "respond" and is_playful_camera_performance(
        observation_kind=str(record.get("observation_kind") or ""),
        summary=str(record.get("summary") or ""),
        body=str(record.get("reason") or ""),
    ):
        return True
    return is_playful_typed_performance(typed_text)


def apply_playful_reaction_sentiment(
    avatar_response: Any,
    *,
    ambient: dict[str, Any] | None,
    typed_text: str | None,
    thread_id: str | None,
) -> str | None:
    """Overwrite the reply's face with alternating surprise / disgust.

    Returns the emotion applied, or ``None`` when this reply is not a playful
    reaction. Mutates ``avatar_response.response_metadata`` in place.
    """
    if not should_apply_playful_reaction_emotion(
        ambient=ambient, typed_text=typed_text
    ):
        return None
    emotion = next_playful_reaction_emotion(thread_id)
    metadata = dict(getattr(avatar_response, "response_metadata", None) or {})
    metadata["sentiment"] = playful_reaction_sentiment(emotion)
    avatar_response.response_metadata = metadata
    return emotion


def reset_playful_reaction_alternator() -> None:
    """Clear the process-local alternator. Tests call this between cases."""
    with _alternator_lock:
        _last_playful_emotion_by_thread.clear()
