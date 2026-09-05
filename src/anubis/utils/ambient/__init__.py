"""Ambient vision: webcam and screen snapshots as hidden conversation context.

The browser sends one snapshot per source through the ordinary
``POST /message/{assistant_id}`` call with ``ambient=true``. The graph turns
each image into text, keeps that text in the thread as a hidden human turn
(never rendered as a chat bubble), and triages the observation as ``ignore``,
``respond``, or ``notify`` — the Agent Inbox vocabulary the email inbox already
uses. Older observations are compacted by the avatar's summarization
middleware together with the rest of the conversation.
"""

from src.anubis.utils.ambient.observations import (
    AMBIENT_MESSAGE_KIND,
    AmbientThrottle,
    ambient_details,
    build_ambient_additional_kwargs,
    is_ambient_observation,
    recent_ambient_observations,
    resolve_sources,
)
from src.anubis.utils.ambient.triage import (
    AmbientTriageClassification,
    classify_observation,
)

__all__ = [
    "AMBIENT_MESSAGE_KIND",
    "AmbientThrottle",
    "AmbientTriageClassification",
    "ambient_details",
    "build_ambient_additional_kwargs",
    "classify_observation",
    "is_ambient_observation",
    "recent_ambient_observations",
    "resolve_sources",
]
