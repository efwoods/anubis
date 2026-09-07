"""Store namespaces for continuous learning.

Every learning record is scoped to the conversing user and the avatar:
``(user_id, assistant_id, kind)`` — the same shape the episodic ``memory``
namespace uses (``identity_tools.py``). The kinds are deliberately separate
namespaces rather than one namespace with a metadata filter so that every
prompt-time lookup stays a plain ``store.asearch`` call that can run inside
the single ``asyncio.gather`` in ``load_consciousness``.

Embedded records (feedback messages, rated messages, sentiment history, what
feels real, preferences) are stored as ``{"document": Document.to_json()}`` so
the store index embeds ``document.kwargs.page_content`` and similarity
retrieval works. Scalar records (the engagement counters, the running
conversation sentiment) are stored as ``{"value": ...}`` like ``style_profile``.
"""

from __future__ import annotations

LEARNING_KIND_ENGAGEMENT = "engagement"
LEARNING_KIND_FEEDBACK = "feedback"
LEARNING_KIND_RATING_POSITIVE = "rating_positive"
LEARNING_KIND_RATING_NEGATIVE = "rating_negative"
LEARNING_KIND_SENTIMENT = "sentiment"
LEARNING_KIND_WHAT_FEELS_REAL = "what_feels_real"
LEARNING_KIND_PREFERENCE = "preference"

RATING_POSITIVE = "positive"
RATING_NEGATIVE = "negative"

# Root of the pending-sweep markers. One namespace per user —
# ``("learning_pending", user_id)`` — keyed by thread id, so the sweeper can
# enumerate every user with unprocessed conversations through
# ``store.alist_namespaces(prefix=("learning_pending",))`` without knowing any
# user identifiers in advance, and a marker survives a process restart.
LEARNING_PENDING_NAMESPACE_ROOT = "learning_pending"


def engagement_namespace(user_id: str, assistant_id: str) -> tuple[str, str, str]:
    return (user_id, assistant_id, LEARNING_KIND_ENGAGEMENT)


def feedback_namespace(user_id: str, assistant_id: str) -> tuple[str, str, str]:
    return (user_id, assistant_id, LEARNING_KIND_FEEDBACK)


def rating_namespace(
    user_id: str, assistant_id: str, rating: str
) -> tuple[str, str, str]:
    """The namespace holding avatar messages the user rated ``rating``.

    ``rating`` is ``"positive"`` or ``"negative"``; anything else raises so a
    typo can never silently create a third namespace.
    """
    if rating == RATING_POSITIVE:
        return (user_id, assistant_id, LEARNING_KIND_RATING_POSITIVE)
    if rating == RATING_NEGATIVE:
        return (user_id, assistant_id, LEARNING_KIND_RATING_NEGATIVE)
    raise ValueError(f"Unknown rating polarity: {rating!r}")


def sentiment_namespace(user_id: str, assistant_id: str) -> tuple[str, str, str]:
    return (user_id, assistant_id, LEARNING_KIND_SENTIMENT)


def what_feels_real_namespace(
    user_id: str, assistant_id: str
) -> tuple[str, str, str]:
    return (user_id, assistant_id, LEARNING_KIND_WHAT_FEELS_REAL)


def preference_namespace(user_id: str, assistant_id: str) -> tuple[str, str, str]:
    return (user_id, assistant_id, LEARNING_KIND_PREFERENCE)


def learning_pending_namespace(user_id: str) -> tuple[str, str]:
    return (LEARNING_PENDING_NAMESPACE_ROOT, user_id)
