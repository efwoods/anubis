"""The background learning sweep.

Pressing like or dislike does not teach the avatar on the spot: ratings are
collected, and this sweep folds them — together with the whole conversation —
into durable learning once the account has gone quiet. Concretely, for every
conversation an account touched since the last sweep:

1. the conversation's sentiment summary is finalized into the sentiment history;
2. ratings not yet aggregated become "the user responds well to / poorly to"
   preference records;
3. preferences, communication style, and "what feels real" candidates are
   inferred from the user's own messages.

Pending work is recorded as store markers under
``("learning_pending", user_id)`` keyed by thread id (written by the
``observe_user`` node on every human turn), so a restart never loses a
conversation that still needs processing. ``run_learning_sweeper`` is the
lifespan task that wakes every ``LEARNING_SWEEP_INTERVAL_SECONDS`` and
processes every account whose newest message is older than
``LEARNING_IDLE_SECONDS``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, Literal

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from src.anubis.utils.learning.engagement import mark_sweep_complete
from src.anubis.utils.learning.feedback import (
    list_unaggregated_ratings,
    mark_ratings_aggregated,
    store_user_preference,
    store_what_feels_real,
)
from src.anubis.utils.learning.namespaces import (
    LEARNING_PENDING_NAMESPACE_ROOT,
    learning_pending_namespace,
)
from src.anubis.utils.learning.sentiment import (
    finalize_conversation_sentiment,
    invoke_structured,
    render_transcript,
    user_messages_text,
)

logger = logging.getLogger(__name__)


# ── pending markers ─────────────────────────────────────────────────────────


async def mark_thread_pending(
    store: Any,
    *,
    user_id: str,
    assistant_id: str,
    thread_id: str,
    creator_id: str | None = None,
    now: datetime | None = None,
) -> None:
    """Record that ``thread_id`` has new activity the next sweep must process."""
    if store is None or not user_id or not assistant_id or not thread_id:
        return
    await store.aput(
        learning_pending_namespace(user_id),
        key=thread_id,
        value={
            "value": {
                "user_id": user_id,
                "assistant_id": assistant_id,
                "thread_id": thread_id,
                "creator_id": creator_id,
                "last_message_at": (now or datetime.now(tz=UTC)).isoformat(),
            }
        },
    )


async def clear_thread_pending(store: Any, user_id: str, thread_id: str) -> None:
    try:
        await store.adelete(learning_pending_namespace(user_id), thread_id)
    except Exception as delete_error:  # noqa: BLE001 - a missing marker is fine
        logger.debug("Could not clear pending marker %s/%s: %s", user_id, thread_id, delete_error)


async def list_pending_accounts(store: Any) -> dict[str, list[dict[str, Any]]]:
    """Every pending marker, grouped by user id."""
    pending_by_user: dict[str, list[dict[str, Any]]] = {}
    namespaces = await store.alist_namespaces(
        prefix=(LEARNING_PENDING_NAMESPACE_ROOT,), limit=1000
    )
    for namespace in namespaces or []:
        if len(namespace) < 2:
            continue
        user_id = namespace[1]
        try:
            items = await store.asearch(tuple(namespace[:2]), limit=1000)
        except Exception as search_error:  # noqa: BLE001
            logger.debug("Could not list pending markers for %s: %s", user_id, search_error)
            continue
        for item in items or []:
            value = getattr(item, "value", None) or {}
            record = value.get("value") if isinstance(value, dict) else None
            if isinstance(record, dict) and record.get("thread_id"):
                pending_by_user.setdefault(user_id, []).append(dict(record))
    return pending_by_user


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def select_idle_accounts(
    pending_by_user: dict[str, list[dict[str, Any]]],
    *,
    now: datetime,
    idle_seconds: float,
) -> list[str]:
    """Users whose newest pending message is at least ``idle_seconds`` old."""
    idle_users: list[str] = []
    for user_id, records in pending_by_user.items():
        newest: datetime | None = None
        for record in records:
            stamp = _parse_timestamp(record.get("last_message_at"))
            if stamp is not None and (newest is None or stamp > newest):
                newest = stamp
        if newest is None or (now - newest).total_seconds() >= idle_seconds:
            idle_users.append(user_id)
    return idle_users


# ── structured inference ────────────────────────────────────────────────────


class RatingPreferenceSummary(BaseModel):
    """What the user's ratings of avatar messages reveal about their preferences."""

    positive_patterns: list[str] = Field(
        default_factory=list,
        description=(
            "Patterns the user responds well to, each phrased as a complete standalone "
            "preference, for example 'The user responds well to short, direct replies "
            "that answer the question first.'"
        ),
    )
    negative_patterns: list[str] = Field(
        default_factory=list,
        description=(
            "Patterns the user responds poorly to, each phrased as a complete standalone "
            "preference, for example 'The user dislikes replies that end with a question.'"
        ),
    )


RATING_PREFERENCE_SUMMARY_SYSTEM_PROMPT = """
<ROLE>
You are an expert at inferring what a person likes and dislikes from the messages that person rated.
</ROLE>

<INSTRUCTIONS>
The RATED_MESSAGES are avatar replies that the USER rated positively or negatively, with the user's own preceding message when available.
Infer the patterns the USER responds well to and the patterns the USER responds poorly to: tone, length, directness, humor, formality, structure, topics, how the USER is addressed.
Write each pattern as one complete standalone sentence about the USER's preference.
Only infer patterns that several ratings support or that one rating states unambiguously. Never invent a preference the ratings do not show.
</INSTRUCTIONS>
"""


class InferredPreference(BaseModel):
    preference: str = Field(description="One complete standalone preference of the user.")
    category: Literal["communication_style", "address", "topic", "format", "other"] = Field(
        description="The kind of preference."
    )
    evidence: str = Field(description="The user's words that support this preference.")


class InferredRealnessStatement(BaseModel):
    statement: str = Field(
        description="One complete standalone statement of what feels real or fake to the user."
    )
    polarity: Literal["feels_real", "feels_fake"]
    evidence: str = Field(description="The user's words that support this statement.")


class InferredUserPreferences(BaseModel):
    """Preferences, communication style, and realness signals inferred from the user's messages."""

    preferences: list[InferredPreference] = Field(default_factory=list)
    communication_style: str = Field(
        default="",
        description=(
            "One or two sentences describing how the USER communicates: length, "
            "formality, humor, punctuation, directness, and the style of reply the USER "
            "seems to want in return. Empty when the transcript gives no signal."
        ),
    )
    what_feels_real: list[InferredRealnessStatement] = Field(default_factory=list)


INFER_USER_PREFERENCES_SYSTEM_PROMPT = """
<ROLE>
You are an expert at learning a person's preferences and communication style from the messages that person wrote.
</ROLE>

<INSTRUCTIONS>
The USER_MESSAGES are every message the USER sent in one conversation with an avatar, oldest first. The TRANSCRIPT shows the same conversation with the avatar's replies for context.
Infer:
1. The USER's explicit and strongly implied preferences: how the USER wants to be addressed, what topics the USER cares about, what reply format the USER wants, and how the USER wants the avatar to communicate. Each preference must be one complete standalone sentence, and the evidence must be the USER's own words.
2. The USER's communication style in one or two sentences.
3. Statements the USER made about what feels real, authentic, or genuine about the avatar, and what feels fake, scripted, or off. Each must be one complete standalone sentence with the USER's own words as evidence.
Only infer what the USER's own messages support. Never infer from the avatar's replies. Leave every list empty when the messages give no signal.
</INSTRUCTIONS>
"""


def _rating_documents_text(documents: list[Document]) -> str:
    return "\n\n".join(document.page_content.strip() for document in documents if document.page_content)


async def aggregate_ratings(store: Any, user_id: str, assistant_id: str) -> int:
    """Fold un-aggregated ratings into preference records; returns how many were folded."""
    documents = await list_unaggregated_ratings(store, user_id, assistant_id)
    if not documents:
        return 0
    try:
        response = await invoke_structured(
            RatingPreferenceSummary,
            RATING_PREFERENCE_SUMMARY_SYSTEM_PROMPT,
            "<RATED_MESSAGES>\n" + _rating_documents_text(documents) + "\n</RATED_MESSAGES>",
        )
    except Exception as inference_error:  # noqa: BLE001 - leave ratings for the next sweep
        logger.warning("Rating aggregation failed for %s/%s: %s", user_id, assistant_id, inference_error)
        return 0
    if isinstance(response, tuple):
        response = response[0]
    summary = response if isinstance(response, RatingPreferenceSummary) else RatingPreferenceSummary.model_validate(response)
    for pattern in summary.positive_patterns:
        await store_user_preference(
            store,
            user_id,
            assistant_id,
            preference=pattern,
            preference_context="Learned from avatar messages the user rated positively.",
            category="communication_style",
            source="rating_summary",
        )
    for pattern in summary.negative_patterns:
        await store_user_preference(
            store,
            user_id,
            assistant_id,
            preference=pattern,
            preference_context="Learned from avatar messages the user rated negatively.",
            category="communication_style",
            source="rating_summary",
        )
    await mark_ratings_aggregated(store, user_id, assistant_id, documents)
    return len(documents)


async def infer_preferences_from_conversation(
    store: Any, user_id: str, assistant_id: str, thread_id: str, messages: list[Any]
) -> int:
    """Infer preferences, style, and realness signals from the user's messages."""
    user_texts = user_messages_text(messages)
    if not user_texts:
        return 0
    human_text = (
        "<USER_MESSAGES>\n"
        + "\n".join(f"- {text}" for text in user_texts)
        + "\n</USER_MESSAGES>\n\n<TRANSCRIPT>\n"
        + render_transcript(messages)
        + "\n</TRANSCRIPT>"
    )
    try:
        response = await invoke_structured(
            InferredUserPreferences, INFER_USER_PREFERENCES_SYSTEM_PROMPT, human_text
        )
    except Exception as inference_error:  # noqa: BLE001
        logger.warning("Preference inference failed for %s/%s: %s", user_id, assistant_id, inference_error)
        return 0
    if isinstance(response, tuple):
        response = response[0]
    inferred = (
        response
        if isinstance(response, InferredUserPreferences)
        else InferredUserPreferences.model_validate(response)
    )
    stored = 0
    for preference in inferred.preferences:
        document = await store_user_preference(
            store,
            user_id,
            assistant_id,
            preference=preference.preference,
            preference_context=preference.evidence,
            category=preference.category,
            source="inferred",
        )
        stored += document is not None
    if inferred.communication_style.strip():
        document = await store_user_preference(
            store,
            user_id,
            assistant_id,
            preference=inferred.communication_style.strip(),
            preference_context="The user's own communication style, inferred from the user's messages.",
            category="communication_style",
            source="inferred",
        )
        stored += document is not None
    for realness in inferred.what_feels_real:
        document = await store_what_feels_real(
            store,
            user_id,
            assistant_id,
            statement=realness.statement,
            statement_context=realness.evidence,
            polarity=realness.polarity,
            source="inferred",
            thread_id=thread_id,
        )
        stored += document is not None
    return stored


# ── the sweep ───────────────────────────────────────────────────────────────


async def load_thread_messages(graph: Any, thread_id: str) -> list[Any]:
    """The checkpointed messages of one thread through the in-process graph."""
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    values = getattr(snapshot, "values", None) or {}
    return list(values.get("messages") or [])


async def run_account_learning_sweep(
    store: Any,
    graph: Any,
    user_id: str,
    pending_records: list[dict[str, Any]],
) -> dict[str, int]:
    """Process every pending conversation for one user. Never raises."""
    counters = {"threads": 0, "ratings_aggregated": 0, "inferred_records": 0}
    swept_assistants: set[str] = set()
    for record in pending_records:
        thread_id = record.get("thread_id")
        assistant_id = record.get("assistant_id")
        if not thread_id or not assistant_id:
            continue
        try:
            messages = await load_thread_messages(graph, thread_id)
            if messages:
                await finalize_conversation_sentiment(
                    store, user_id, assistant_id, thread_id, messages
                )
                counters["inferred_records"] += await infer_preferences_from_conversation(
                    store, user_id, assistant_id, thread_id, messages
                )
            counters["threads"] += 1
            swept_assistants.add(assistant_id)
        except Exception as sweep_error:  # noqa: BLE001 - one bad thread must not stop the rest
            logger.warning(
                "Learning sweep failed for user %s thread %s: %s", user_id, thread_id, sweep_error
            )
            continue
        await clear_thread_pending(store, user_id, thread_id)

    for assistant_id in swept_assistants:
        try:
            counters["ratings_aggregated"] += await aggregate_ratings(
                store, user_id, assistant_id
            )
            await mark_sweep_complete(store, user_id, assistant_id)
        except Exception as aggregation_error:  # noqa: BLE001
            logger.warning(
                "Rating aggregation failed for user %s avatar %s: %s",
                user_id,
                assistant_id,
                aggregation_error,
            )
    return counters


async def run_learning_sweep_once(
    store: Any, graph: Any, *, idle_seconds: float, now: datetime | None = None
) -> dict[str, int]:
    """One pass over every idle account; returns aggregate counters."""
    now = now or datetime.now(tz=UTC)
    pending_by_user = await list_pending_accounts(store)
    idle_users = select_idle_accounts(pending_by_user, now=now, idle_seconds=idle_seconds)
    totals = {"accounts": 0, "threads": 0, "ratings_aggregated": 0, "inferred_records": 0}
    for user_id in idle_users:
        counters = await run_account_learning_sweep(
            store, graph, user_id, pending_by_user[user_id]
        )
        totals["accounts"] += 1
        for key in ("threads", "ratings_aggregated", "inferred_records"):
            totals[key] += counters.get(key, 0)
    return totals


async def run_learning_sweeper(app: Any) -> None:
    """Lifespan task: sweep idle accounts every ``learning_sweep_interval_seconds``."""
    context = app.state.context
    interval_seconds = float(getattr(context, "learning_sweep_interval_seconds", 300) or 300)
    idle_seconds = float(getattr(context, "learning_idle_seconds", 600) or 600)
    logger.info(
        "Learning sweeper started (interval %.0fs, idle window %.0fs)",
        interval_seconds,
        idle_seconds,
    )
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            totals = await run_learning_sweep_once(
                app.state.store, app.state.graph, idle_seconds=idle_seconds
            )
            if totals["accounts"]:
                logger.info("Learning sweep processed %s", totals)
        except asyncio.CancelledError:
            logger.info("Learning sweeper stopped")
            raise
        except Exception as sweep_error:  # noqa: BLE001 - the loop must survive
            logger.exception("Learning sweep failed: %s", sweep_error)
