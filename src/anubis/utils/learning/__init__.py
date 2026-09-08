"""Continuous learning and personalization for avatar conversations.

Everything an avatar learns about the person it is talking to — how often that
person engages, how the person feels right now and over the whole conversation,
which avatar messages the person rated well or poorly, the feedback the person
wrote, what the person said feels real, and the preferences the person dictated
or revealed — is stored per ``(user_id, assistant_id)`` in the LangGraph store
and read back by ``load_consciousness`` into dedicated system-prompt sections.

Modules:

* ``namespaces`` — the store namespaces every other module writes and reads.
* ``engagement`` — the per-user engagement record (counts, recency, frequency).
* ``sentiment`` — immediate message sentiment and conversation sentiment summaries.
* ``feedback`` — ratings, feedback messages, "what feels real", preferences, and
  the one-shot retrieval that fills the prompt sections.
* ``bulk_learning`` — the background sweep that aggregates ratings, finalizes
  conversation sentiment history, and infers preferences once an account has gone
  idle.
"""

from src.anubis.utils.learning.namespaces import (
    LEARNING_PENDING_NAMESPACE_ROOT,
    engagement_namespace,
    feedback_namespace,
    learning_pending_namespace,
    preference_namespace,
    rating_namespace,
    sentiment_namespace,
    what_feels_real_namespace,
)

__all__ = [
    "LEARNING_PENDING_NAMESPACE_ROOT",
    "engagement_namespace",
    "feedback_namespace",
    "learning_pending_namespace",
    "preference_namespace",
    "rating_namespace",
    "sentiment_namespace",
    "what_feels_real_namespace",
]
