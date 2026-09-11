"""Store namespaces for the avatar's learned psychology.

Every record here belongs to the AVATAR rather than to the person conversing with
it, so the namespace is scoped ``(creator_id, assistant_id, kind)`` — the shape the
avatar-side namespaces already use (``identity``, ``quote``, ``document``,
``analysis``, ``style_profile``). That is deliberately NOT the learning shape
``(user_id, assistant_id, kind)``: what the target's love languages are does not
change depending on who is talking to the avatar.

Two of the three kinds hold a single scalar record under the key ``current``, read
with ``aget_through_cache`` on every turn:

``psychological_profile``  the consolidated profile, accumulated across uploads.
``current_emotion``        the avatar's live emotional state and its baseline.

The third holds embedded Documents, because it is searched rather than fetched:

``emotional_trigger``      one record per generalized trigger, so a live message can
                           be matched against the kinds of thing that move the
                           target. The store index embeds ``page_content`` only
                           (see ``langgraph.json``), so the generalized trigger
                           description must be part of the page content, never
                           metadata alone.
"""

from __future__ import annotations

PSYCHOLOGICAL_KIND_PROFILE = "psychological_profile"
PSYCHOLOGICAL_KIND_CURRENT_EMOTION = "current_emotion"
PSYCHOLOGICAL_KIND_EMOTIONAL_TRIGGER = "emotional_trigger"

# Both scalar records live under one well-known key per avatar: there is exactly
# one current profile and exactly one current emotional state.
CURRENT_RECORD_KEY = "current"


def psychological_profile_namespace(
    creator_id: str, assistant_id: str
) -> tuple[str, str, str]:
    """Where the consolidated psychological profile for one avatar lives."""
    return (creator_id, assistant_id, PSYCHOLOGICAL_KIND_PROFILE)


def current_emotion_namespace(
    creator_id: str, assistant_id: str
) -> tuple[str, str, str]:
    """Where the avatar's live emotional state and its baseline live."""
    return (creator_id, assistant_id, PSYCHOLOGICAL_KIND_CURRENT_EMOTION)


def emotional_trigger_namespace(
    creator_id: str, assistant_id: str
) -> tuple[str, str, str]:
    """Where the generalized emotional triggers searched at conversation time live."""
    return (creator_id, assistant_id, PSYCHOLOGICAL_KIND_EMOTIONAL_TRIGGER)


__all__ = [
    "CURRENT_RECORD_KEY",
    "PSYCHOLOGICAL_KIND_CURRENT_EMOTION",
    "PSYCHOLOGICAL_KIND_EMOTIONAL_TRIGGER",
    "PSYCHOLOGICAL_KIND_PROFILE",
    "current_emotion_namespace",
    "emotional_trigger_namespace",
    "psychological_profile_namespace",
]
