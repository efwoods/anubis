"""Thumbs and written feedback on an avatar's replies, kept per user and avatar.

Every like, dislike, or note the conversation partner leaves on a reply is
written to the LangGraph store under ``(user_id, assistant_id,
"message_feedback")``, shaped like the identity documents so the store's
vector index embeds the text. The transcript route reads the records back and
attaches each one to its reply, so a rating survives a page reload, and the
avatar preferences route returns them so the browser can refresh its view
right after a thumb is pressed. The same records are what a later recall can
hand the avatar as precedent for which replies the conversation partner
wanted more or less of.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

MESSAGE_FEEDBACK_NAMESPACE_SUFFIX = "message_feedback"

FEEDBACK_TYPES = frozenset({"like", "dislike"})

# What the conversation partner said feels real (or off) about the reply. Kept
# on the same record as the thumb so one press restores the whole row.
FEELS_TYPES = frozenset({"feels_real", "feels_fake"})

# How much of the rated reply the embedded record quotes. Enough to recall
# "replies like this one" by similarity, short enough to keep the row small.
CONTENT_EXCERPT_CHARACTERS = 600


def message_feedback_namespace(user_id: str, assistant_id: str) -> tuple[str, str, str]:
    """Return the store namespace holding one user's feedback for one avatar."""
    return (str(user_id), str(assistant_id), MESSAGE_FEEDBACK_NAMESPACE_SUFFIX)


def message_feedback_key(message_id: str | None, request_id: str | None) -> str | None:
    """Return the store key for one rated reply: its stored message id, else its request id."""
    for candidate in (message_id, request_id):
        text = str(candidate or "").strip()
        if text:
            return text
    return None


def _item_value(item: Any) -> dict[str, Any]:
    value = getattr(item, "value", None)
    if value is None and isinstance(item, dict):
        value = item.get("value", item)
    return dict(value) if isinstance(value, dict) else {}


def _feedback_page_content(
    *,
    feedback_type: str | None,
    feels: str | None,
    comment: str | None,
    content_excerpt: str,
) -> str:
    if feedback_type in FEEDBACK_TYPES:
        verb = "liked" if feedback_type == "like" else "disliked"
        text = f"The conversation partner {verb} this reply"
    elif feels == "feels_real":
        text = "The conversation partner said this reply feels real"
    elif feels == "feels_fake":
        text = "The conversation partner said this reply feels fake or off"
    else:
        text = "The conversation partner reacted to this reply"
    if content_excerpt:
        text += f': "{content_excerpt}"'
    text += "."
    if feedback_type in FEEDBACK_TYPES and feels == "feels_real":
        text += " The conversation partner said this reply feels real."
    elif feedback_type in FEEDBACK_TYPES and feels == "feels_fake":
        text += " The conversation partner said this reply feels fake or off."
    if comment:
        text += f" Feedback from the conversation partner: {comment}"
    return text


def feedback_view(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the shape the browser keeps on a message.

    ``{type, feels, comment, rating_score, recorded_at}``: ``type`` is the thumb
    (``like`` / ``dislike`` / ``None``), ``feels`` is ``feels_real`` /
    ``feels_fake`` / ``None``.
    """
    if not record:
        return None
    feedback_type = record.get("feedback_type")
    feels = record.get("feels")
    if (
        feedback_type not in FEEDBACK_TYPES
        and feels not in FEELS_TYPES
        and not record.get("comment")
    ):
        return None
    return {
        "type": feedback_type if feedback_type in FEEDBACK_TYPES else None,
        "feels": feels if feels in FEELS_TYPES else None,
        "comment": record.get("comment"),
        "rating_score": record.get("rating_score"),
        "recorded_at": record.get("recorded_at"),
    }


async def record_message_feedback(
    store: Any,
    user_id: str,
    assistant_id: str,
    *,
    thread_id: str | None,
    message_id: str | None,
    request_id: str | None,
    feedback_type: str | None,
    comment: str | None = None,
    content: str | None = None,
    feels: str | None = None,
    rating_score: float | None = None,
) -> dict[str, Any] | None:
    """Record (or replace) the conversation partner's feedback on one reply.

    A thumb without a note keeps any note already recorded; a note without a
    new thumb keeps the recorded thumb; a feels-real mark keeps the thumb and
    a thumb keeps the feels-real mark. Returns the stored record, or ``None``
    when the store is unavailable, the reply cannot be identified, or nothing
    (no thumb, no feels-real mark, no note) was given.
    """
    if store is None or not user_id or not assistant_id:
        return None
    if feedback_type not in FEEDBACK_TYPES:
        feedback_type = None
    if feels not in FEELS_TYPES:
        feels = None
    if feedback_type is None and feels is None and not (comment or "").strip():
        return None
    key = message_feedback_key(message_id, request_id)
    if key is None:
        return None
    namespace = message_feedback_namespace(user_id, assistant_id)
    previous: dict[str, Any] = {}
    try:
        existing = await store.aget(namespace, key)
    except Exception:  # noqa: BLE001 - a miss and a store error read the same
        existing = None
    if existing is not None:
        previous = _item_value(existing)
    final_comment = (comment or "").strip() or previous.get("comment") or None
    content_excerpt = (content or "").strip()[:CONTENT_EXCERPT_CHARACTERS] or str(
        previous.get("content_excerpt") or ""
    )
    previous_type = previous.get("feedback_type")
    final_type = feedback_type or (
        previous_type if previous_type in FEEDBACK_TYPES else None
    )
    previous_feels = previous.get("feels")
    final_feels = feels or (previous_feels if previous_feels in FEELS_TYPES else None)
    final_rating_score = (
        rating_score if rating_score is not None else previous.get("rating_score")
    )
    page_content = _feedback_page_content(
        feedback_type=final_type,
        feels=final_feels,
        comment=final_comment,
        content_excerpt=content_excerpt,
    )
    document = Document(
        page_content=page_content,
        metadata={
            "user_id": user_id,
            "assistant_id": assistant_id,
            "thread_id": thread_id,
            "feedback_type": final_type,
            "feels": final_feels,
        },
    )
    value = {
        "document": document.to_json(),
        "thread_id": thread_id,
        "message_id": (message_id or "").strip() or previous.get("message_id"),
        "request_id": (request_id or "").strip() or previous.get("request_id"),
        "feedback_type": final_type,
        "feels": final_feels,
        "rating_score": final_rating_score,
        "comment": final_comment,
        "content_excerpt": content_excerpt,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    await store.aput(namespace, key=key, value=value)
    return value


async def list_message_feedback(
    store: Any,
    user_id: str,
    assistant_id: str,
    *,
    thread_id: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Every feedback record for one avatar, narrowed to one thread when given."""
    if store is None or not user_id or not assistant_id:
        return []
    namespace = message_feedback_namespace(user_id, assistant_id)
    search_filter = {"thread_id": thread_id} if thread_id else None
    try:
        items = await store.asearch(
            namespace, filter=search_filter, limit=max(1, int(limit))
        )
    except Exception:  # noqa: BLE001 - feedback must never fail a transcript
        logger.debug("Message feedback unavailable", exc_info=True)
        return []
    records: list[dict[str, Any]] = []
    for item in items or []:
        value = _item_value(item)
        if (
            value.get("feedback_type") in FEEDBACK_TYPES
            or value.get("feels") in FEELS_TYPES
            or value.get("comment")
        ):
            records.append(value)
    return records


def attach_message_feedback(
    messages: list[Any], records: list[dict[str, Any]]
) -> list[Any]:
    """Put each stored rating on the reply it belongs to.

    A reply is matched by its stored message id first, then by the request id
    the turn was streamed under. Messages that are not plain dicts are left
    untouched, as are messages nobody rated.
    """
    if not records:
        return messages
    by_message_id: dict[str, dict[str, Any]] = {}
    by_request_id: dict[str, dict[str, Any]] = {}
    for record in records:
        message_id = str(record.get("message_id") or "").strip()
        request_id = str(record.get("request_id") or "").strip()
        if message_id:
            by_message_id[message_id] = record
        if request_id:
            by_request_id[request_id] = record
    attached: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            attached.append(message)
            continue
        response_metadata = message.get("response_metadata") or {}
        request_id = (
            message.get("request_id")
            or (
                response_metadata.get("request_id")
                if isinstance(response_metadata, dict)
                else None
            )
            or ""
        )
        stored: dict[str, Any] | None = by_message_id.get(
            str(message.get("id") or "")
        ) or by_request_id.get(str(request_id))
        view = feedback_view(stored)
        if view is None:
            attached.append(message)
            continue
        attached.append({**message, "feedback": view})
    return attached
