"""Ratings, feedback messages, "what feels real", preferences, and prompt retrieval.

Every record here is an embedded Document in one of the learning namespaces
(``namespaces.py``) so the prompt-time retrieval is a similarity search against
the user's latest message — the same mechanism episodic memory uses. Each
Document carries ``metadata.fact`` (the atomic text) so the de-duplication gate
shared with ``learn_information_about_the_user`` applies unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from langchain_core.documents import Document

from src.anubis.utils.learning.engagement import (
    load_engagement_record,
    render_engagement_section,
)
from src.anubis.utils.learning.namespaces import (
    RATING_NEGATIVE,
    RATING_POSITIVE,
    feedback_namespace,
    preference_namespace,
    rating_namespace,
    sentiment_namespace,
    what_feels_real_namespace,
)
from src.anubis.utils.learning.sentiment import (
    load_current_conversation_sentiment,
    render_conversation_sentiment,
)

logger = logging.getLogger(__name__)

PREFERENCE_CATEGORIES = ("communication_style", "address", "topic", "format", "other")
PREFERENCE_SOURCES = ("dictated", "inferred", "rating_summary")
WHAT_FEELS_REAL_POLARITIES = ("feels_real", "feels_fake")

# Prompt-section retrieval is top-K by similarity to the user's latest message
# with NO score floor: these records are few, every one of them is something
# the user deliberately told the avatar, and a preference stated in unrelated
# words ("call me Sam") must still reach the prompt. The cap that keeps one
# voluble user from crowding out the identity sections is the env-configurable
# ``LEARNING_PROMPT_RETRIEVAL_LIMIT``.
_RATED_MESSAGE_CHARACTER_LIMIT = 800


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def build_learning_document(
    page_content: str,
    *,
    user_id: str,
    assistant_id: str,
    kind: str,
    fact: str,
    fact_context: str | None = None,
    document_id: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Document:
    metadata: dict[str, Any] = {
        "user_id": user_id,
        "assistant_id": assistant_id,
        "document_id": document_id or str(uuid.uuid4()),
        "kind": kind,
        "fact": fact,
        "fact_context": fact_context or "",
        "recorded_at": _utc_now_iso(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return Document(page_content=page_content, metadata=metadata)


async def _already_stored(store: Any, namespace: tuple, fact: str) -> bool:
    """The de-duplication gate: an equal atomic fact already in ``namespace``."""
    from src.anubis.utils.tools.identity.identity_tools import (
        _store_items_contain_fact,
    )

    try:
        items = await store.asearch(namespace, query=fact, limit=10)
    except Exception as search_error:  # noqa: BLE001 - never block a write on a search
        logger.debug("Learning de-duplication search failed: %s", search_error)
        return False
    return _store_items_contain_fact(items, fact)


async def _put_document(store: Any, namespace: tuple, document: Document) -> None:
    await store.aput(
        namespace,
        key=document.metadata["document_id"],
        value={"document": document.to_json()},
    )


# ── writes ──────────────────────────────────────────────────────────────────


async def store_feedback_message(
    store: Any,
    user_id: str,
    assistant_id: str,
    *,
    comment: str,
    thread_id: str | None = None,
    message_id: str | None = None,
    related_avatar_message: str | None = None,
) -> Document | None:
    """Store a feedback message the user wrote about the avatar (immediate)."""
    comment = (comment or "").strip()
    if not comment:
        return None
    namespace = feedback_namespace(user_id, assistant_id)
    if await _already_stored(store, namespace, comment):
        return None
    page_content = comment
    if related_avatar_message:
        page_content = (
            "The user gave this feedback about the avatar message below.\n"
            f"FEEDBACK: {comment}\n"
            f"AVATAR MESSAGE: {_truncate(related_avatar_message)}"
        )
    document = build_learning_document(
        page_content,
        user_id=user_id,
        assistant_id=assistant_id,
        kind="feedback_message",
        fact=comment,
        fact_context=related_avatar_message or "",
        extra_metadata={"thread_id": thread_id, "message_id": message_id},
    )
    await _put_document(store, namespace, document)
    return document


async def store_message_rating(
    store: Any,
    user_id: str,
    assistant_id: str,
    *,
    rating: str,
    thread_id: str,
    message_id: str,
    avatar_message_text: str,
    preceding_user_message_text: str | None = None,
    rating_score: float | None = None,
    request_id: str | None = None,
) -> Document:
    """Record that the user rated one avatar message positively or negatively.

    Keyed by ``message_id`` so re-rating replaces the earlier record, and the
    opposite-polarity record for the same message is removed so a message never
    counts on both sides.
    """
    if rating not in (RATING_POSITIVE, RATING_NEGATIVE):
        raise ValueError(f"Unknown rating polarity: {rating!r}")
    namespace = rating_namespace(user_id, assistant_id, rating)
    opposite = RATING_NEGATIVE if rating == RATING_POSITIVE else RATING_POSITIVE
    try:
        await store.adelete(rating_namespace(user_id, assistant_id, opposite), message_id)
    except Exception:  # noqa: BLE001 - a missing opposite record is the normal case
        pass

    page_content_lines = []
    if preceding_user_message_text:
        page_content_lines.append(f"USER: {_truncate(preceding_user_message_text)}")
    page_content_lines.append(
        f"AVATAR (rated {rating} by the user): {_truncate(avatar_message_text)}"
    )
    document = build_learning_document(
        "\n".join(page_content_lines),
        user_id=user_id,
        assistant_id=assistant_id,
        kind=f"rating_{rating}",
        fact=avatar_message_text,
        fact_context=preceding_user_message_text or "",
        document_id=message_id,
        extra_metadata={
            "rating": rating,
            "rating_score": rating_score,
            "thread_id": thread_id,
            "message_id": message_id,
            "request_id": request_id,
            "rated_at": _utc_now_iso(),
            "aggregated": False,
        },
    )
    await _put_document(store, namespace, document)
    return document


async def store_what_feels_real(
    store: Any,
    user_id: str,
    assistant_id: str,
    *,
    statement: str,
    statement_context: str | None,
    polarity: str,
    source: str = "dictated",
    thread_id: str | None = None,
    message_id: str | None = None,
) -> Document | None:
    """Record what the user said (or was inferred to find) feels real or fake."""
    statement = (statement or "").strip()
    if not statement:
        return None
    if polarity not in WHAT_FEELS_REAL_POLARITIES:
        raise ValueError(f"Unknown polarity: {polarity!r}")
    namespace = what_feels_real_namespace(user_id, assistant_id)
    if await _already_stored(store, namespace, statement):
        return None
    label = "feels real" if polarity == "feels_real" else "feels fake or off"
    page_content = f"To the user, this {label}: {statement}"
    if statement_context:
        page_content += f"\nContext: {statement_context}"
    document = build_learning_document(
        page_content,
        user_id=user_id,
        assistant_id=assistant_id,
        kind="what_feels_real",
        fact=statement,
        fact_context=statement_context or "",
        extra_metadata={
            "polarity": polarity,
            "source": source,
            "thread_id": thread_id,
            "message_id": message_id,
        },
    )
    await _put_document(store, namespace, document)
    return document


async def store_user_preference(
    store: Any,
    user_id: str,
    assistant_id: str,
    *,
    preference: str,
    preference_context: str | None,
    category: str = "other",
    source: str = "dictated",
) -> Document | None:
    """Record one personalization preference (dictated, inferred, or from ratings)."""
    preference = (preference or "").strip()
    if not preference:
        return None
    if category not in PREFERENCE_CATEGORIES:
        category = "other"
    if source not in PREFERENCE_SOURCES:
        source = "inferred"
    namespace = preference_namespace(user_id, assistant_id)
    if await _already_stored(store, namespace, preference):
        return None
    readable_category = category.replace("_", " ")
    page_content = f"User preference ({readable_category}): {preference}"
    if preference_context:
        page_content += f"\nContext: {preference_context}"
    document = build_learning_document(
        page_content,
        user_id=user_id,
        assistant_id=assistant_id,
        kind="user_preference",
        fact=preference,
        fact_context=preference_context or "",
        extra_metadata={"category": category, "source": source},
    )
    await _put_document(store, namespace, document)
    return document


def _truncate(text: str, limit: int = _RATED_MESSAGE_CHARACTER_LIMIT) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + " ..."


# ── reads ───────────────────────────────────────────────────────────────────


def _item_document(item: Any) -> Document | None:
    value = getattr(item, "value", None) or {}
    document_json = value.get("document") if isinstance(value, dict) else None
    kwargs = (document_json or {}).get("kwargs") or {}
    if not isinstance(kwargs, dict) or not kwargs.get("page_content"):
        return None
    return Document(
        page_content=kwargs.get("page_content", ""),
        metadata=dict(kwargs.get("metadata") or {}),
    )


async def _search_documents(
    store: Any,
    namespace: tuple,
    *,
    query: str | None,
    limit: int,
) -> list[Document]:
    """Top-``limit`` Documents of ``namespace`` ranked by similarity to ``query``."""
    try:
        items = await store.asearch(namespace, query=query, limit=limit)
    except Exception as search_error:  # noqa: BLE001 - prompt sections are best effort
        logger.debug("Learning retrieval failed for %s: %s", namespace, search_error)
        return []
    documents: list[Document] = []
    for item in items or []:
        document = _item_document(item)
        if document is not None:
            documents.append(document)
    return documents


async def list_thread_ratings(
    store: Any, user_id: str, assistant_id: str, thread_id: str
) -> list[dict[str, Any]]:
    """Every rating recorded on ``thread_id`` (for the client to restore state)."""
    ratings: list[dict[str, Any]] = []
    for polarity in (RATING_POSITIVE, RATING_NEGATIVE):
        documents = await _search_documents(
            store,
            rating_namespace(user_id, assistant_id, polarity),
            query=None,
            limit=500,
        )
        for document in documents:
            if document.metadata.get("thread_id") != thread_id:
                continue
            ratings.append(
                {
                    "message_id": document.metadata.get("message_id"),
                    "rating": polarity,
                    "rating_score": document.metadata.get("rating_score"),
                    "rated_at": document.metadata.get("rated_at"),
                }
            )
    return ratings


async def list_unaggregated_ratings(
    store: Any, user_id: str, assistant_id: str
) -> list[Document]:
    """Ratings the background sweep has not yet folded into a preference summary."""
    documents: list[Document] = []
    for polarity in (RATING_POSITIVE, RATING_NEGATIVE):
        for document in await _search_documents(
            store,
            rating_namespace(user_id, assistant_id, polarity),
            query=None,
            limit=500,
        ):
            if not document.metadata.get("aggregated"):
                documents.append(document)
    return documents


async def mark_ratings_aggregated(
    store: Any, user_id: str, assistant_id: str, documents: list[Document]
) -> None:
    for document in documents:
        polarity = document.metadata.get("rating")
        if polarity not in (RATING_POSITIVE, RATING_NEGATIVE):
            continue
        document.metadata["aggregated"] = True
        await _put_document(
            store, rating_namespace(user_id, assistant_id, polarity), document
        )


def _document_view(document: Document) -> dict[str, Any]:
    metadata = document.metadata or {}
    return {
        "id": metadata.get("document_id"),
        "text": metadata.get("fact") or document.page_content,
        "context": metadata.get("fact_context") or "",
        "category": metadata.get("category"),
        "polarity": metadata.get("polarity"),
        "source": metadata.get("source"),
        "recorded_at": metadata.get("recorded_at"),
    }


async def list_learned_records(
    store: Any, user_id: str, assistant_id: str, *, limit: int = 200
) -> dict[str, list[dict[str, Any]]]:
    """Everything learned about one person for one avatar, for a client to show.

    Returns ``preferences`` (dictated, inferred, and rating-summary
    preferences), ``feedback_messages``, and ``what_feels_real``.
    """
    if store is None or not user_id or not assistant_id:
        return {"preferences": [], "feedback_messages": [], "what_feels_real": []}
    preference_documents, feedback_documents, realness_documents = await asyncio.gather(
        _search_documents(
            store, preference_namespace(user_id, assistant_id), query=None, limit=limit
        ),
        _search_documents(
            store, feedback_namespace(user_id, assistant_id), query=None, limit=limit
        ),
        _search_documents(
            store, what_feels_real_namespace(user_id, assistant_id), query=None, limit=limit
        ),
    )

    def _newest_first(documents: list[Document]) -> list[dict[str, Any]]:
        views = [_document_view(document) for document in documents]
        views.sort(key=lambda view: str(view.get("recorded_at") or ""), reverse=True)
        return views

    return {
        "preferences": _newest_first(preference_documents),
        "feedback_messages": _newest_first(feedback_documents),
        "what_feels_real": _newest_first(realness_documents),
    }


@dataclass
class LearningSections:
    """The rendered text of every learning-driven system-prompt section."""

    user_engagement: str = ""
    user_feedback_messages: str = ""
    positively_rated_messages: str = ""
    negatively_rated_messages: str = ""
    current_conversation_sentiment: str = ""
    conversation_sentiment_history: str = ""
    what_feels_real: str = ""
    user_preferences: str = ""
    engagement_record: dict[str, Any] = field(default_factory=dict)
    what_feels_real_recorded: bool = False


def _join_documents(documents: list[Document]) -> str:
    return "\n\n".join(document.page_content.strip() for document in documents if document.page_content)


async def retrieve_learning_sections(
    store: Any,
    user_id: str,
    assistant_id: str,
    *,
    thread_id: str | None,
    query: str,
    limit: int = 10,
) -> LearningSections:
    """Fill every learning section for one turn in a single concurrent batch."""
    sections = LearningSections()
    if store is None or not user_id or not assistant_id:
        return sections

    async def _current_sentiment():
        if not thread_id:
            return None
        return await load_current_conversation_sentiment(
            store, user_id, assistant_id, thread_id
        )

    (
        engagement_record,
        feedback_documents,
        positive_documents,
        negative_documents,
        current_sentiment,
        history_documents,
        what_feels_real_documents,
        preference_documents,
    ) = await asyncio.gather(
        load_engagement_record(store, user_id, assistant_id),
        _search_documents(
            store, feedback_namespace(user_id, assistant_id), query=query, limit=limit
        ),
        _search_documents(
            store,
            rating_namespace(user_id, assistant_id, RATING_POSITIVE),
            query=query,
            limit=limit,
        ),
        _search_documents(
            store,
            rating_namespace(user_id, assistant_id, RATING_NEGATIVE),
            query=query,
            limit=limit,
        ),
        _current_sentiment(),
        _search_documents(
            store, sentiment_namespace(user_id, assistant_id), query=query, limit=limit
        ),
        _search_documents(
            store,
            what_feels_real_namespace(user_id, assistant_id),
            query=query,
            limit=limit,
        ),
        _search_documents(
            store, preference_namespace(user_id, assistant_id), query=query, limit=limit
        ),
    )

    # The sentiment namespace holds both the scalar running summaries and the
    # embedded history Documents; only the history Documents come back from a
    # similarity search with content, and the current thread's own history entry
    # (written by an earlier sweep of this same thread) is excluded because the
    # running summary already covers the current conversation.
    history_documents = [
        document
        for document in history_documents
        if document.metadata.get("kind") == "conversation_sentiment_history"
        and document.metadata.get("thread_id") != thread_id
    ]

    sections.engagement_record = engagement_record
    sections.user_engagement = render_engagement_section(engagement_record)
    sections.user_feedback_messages = _join_documents(feedback_documents)
    sections.positively_rated_messages = _join_documents(positive_documents)
    sections.negatively_rated_messages = _join_documents(negative_documents)
    sections.current_conversation_sentiment = render_conversation_sentiment(
        current_sentiment
    )
    sections.conversation_sentiment_history = _join_documents(history_documents)
    sections.what_feels_real = _join_documents(what_feels_real_documents)
    sections.user_preferences = _join_documents(preference_documents)
    sections.what_feels_real_recorded = bool(what_feels_real_documents)
    if not sections.what_feels_real_recorded:
        # A similarity search against an unrelated query can miss the few
        # records that exist; an unqualified listing settles whether any exist.
        any_records = await _search_documents(
            store,
            what_feels_real_namespace(user_id, assistant_id),
            query=None,
            limit=1,
        )
        sections.what_feels_real_recorded = bool(any_records)
    return sections


def should_ask_what_feels_real(
    sections: LearningSections, ask_after_messages: int
) -> bool:
    """Whether the avatar should naturally ask the user what feels real this turn."""
    if ask_after_messages <= 0 or sections.what_feels_real_recorded:
        return False
    message_count = int((sections.engagement_record or {}).get("message_count") or 0)
    return message_count >= ask_after_messages
