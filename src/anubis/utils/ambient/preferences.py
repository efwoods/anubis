"""Owner preferences about ambient observations, learned from card decisions.

Every decision the conversation partner makes on a notification card
(dismiss, reply, a free-text note) is recorded in the LangGraph store under
``(user_id, assistant_id, "ambient_preference")``, shaped like the identity
documents so the store's vector index embeds the text. The triage node recalls
the closest preferences by similarity to the fresh observation and hands them
to the classifier as precedent — the same learning lever the email inbox uses.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

AMBIENT_PREFERENCE_NAMESPACE_SUFFIX = "ambient_preference"

# The per-card record beside the aggregated preference: which thumb and which
# note one particular observation received, so the card can show them again
# after a reload. Kept in its own namespace so similarity recall over the
# aggregated preferences is not crowded by one row per card.
AMBIENT_DECISION_NAMESPACE_SUFFIX = "ambient_decision"

RATING_DECISIONS = frozenset({"accept", "ignore"})

# What a card can record beyond a thumb and a note: the conversation partner
# let the avatar act on the avatar's offer, replied in person instead, left
# the notice alone, or rated what the avatar did after being allowed to act.
ACTION_TAKEN_AVATAR = "avatar_replied"
ACTION_TAKEN_OWNER = "owner_replied"
ACTIONS_TAKEN = frozenset({ACTION_TAKEN_AVATAR, ACTION_TAKEN_OWNER})

# The aggregated preference each outcome counts up, per observation kind.
OUTCOME_DECISIONS = frozenset(
    {"allowed_action", "replied_self", "left_alone", "liked_action", "disliked_action"}
)

DECISION_PHRASES: dict[str, str] = {
    "accept": "more notices like this",
    "ignore": "fewer notices like this",
    "response": "a note",
    "allowed_action": "let the avatar act on the offer",
    "replied_self": "replied in person",
    "left_alone": "left the notice alone",
    "liked_action": "liked what the avatar did when allowed to act",
    "disliked_action": "disliked what the avatar did when allowed to act",
}


def ambient_preference_namespace(
    user_id: str, assistant_id: str
) -> tuple[str, str, str]:
    """Return the store namespace holding one user's preferences for one avatar."""
    return (str(user_id), str(assistant_id), AMBIENT_PREFERENCE_NAMESPACE_SUFFIX)


def ambient_decision_namespace(user_id: str, assistant_id: str) -> tuple[str, str, str]:
    """Return the store namespace holding one user's per-card decisions for one avatar."""
    return (str(user_id), str(assistant_id), AMBIENT_DECISION_NAMESPACE_SUFFIX)


def _preference_key(observation_kind: str, decision: str) -> str:
    return f"{observation_kind}:{decision}"


def _preference_page_content(
    *, observation_kind: str, summary: str, decision: str, note: str | None
) -> str:
    phrase = DECISION_PHRASES.get(decision)
    outcome = f"{decision} ({phrase})" if phrase else decision
    text = f"{observation_kind}: {summary or 'a scene of this kind'} -> {outcome}."
    if note:
        text += f" Note from the conversation partner: {note}"
    return text


def _item_value(item: Any) -> dict[str, Any]:
    value = getattr(item, "value", None)
    if value is None and isinstance(item, dict):
        value = item.get("value", item)
    return dict(value) if isinstance(value, dict) else {}


async def record_ambient_preference(
    store: Any,
    user_id: str,
    assistant_id: str,
    *,
    observation_kind: str,
    summary: str,
    decision: str,
    note: str | None = None,
) -> dict[str, Any] | None:
    """Record (or count up) one decision the conversation partner made."""
    if store is None or not user_id or not assistant_id:
        return None
    namespace = ambient_preference_namespace(user_id, assistant_id)
    kind = (observation_kind or "other").strip().lower()[:40] or "other"
    key = _preference_key(kind, decision)
    count = 1
    previous_note = None
    try:
        existing = await store.aget(namespace, key)
    except Exception:  # noqa: BLE001 - a miss and a store error read the same
        existing = None
    if existing is not None:
        previous = _item_value(existing)
        count = int(previous.get("count") or 0) + 1
        previous_note = previous.get("note")
    final_note = (note or "").strip() or previous_note or None
    page_content = _preference_page_content(
        observation_kind=kind, summary=summary, decision=decision, note=final_note
    )
    document = Document(
        page_content=page_content,
        metadata={
            "user_id": user_id,
            "assistant_id": assistant_id,
            "observation_kind": kind,
            "decision": decision,
        },
    )
    value = {
        "document": document.to_json(),
        "observation_kind": kind,
        "decision": decision,
        "summary": (summary or "").strip()[:300],
        "note": final_note,
        "count": count,
        "last_decided_at": datetime.now(UTC).isoformat(),
    }
    await store.aput(namespace, key=key, value=value)
    return value


async def recall_ambient_preferences(
    store: Any,
    user_id: str,
    assistant_id: str,
    *,
    query: str,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Recall the preferences closest to ``query`` (notes included) as precedent."""
    if store is None or not user_id or not assistant_id:
        return []
    namespace = ambient_preference_namespace(user_id, assistant_id)
    try:
        items = await store.asearch(
            namespace,
            query=(query or "").strip()[:2000] or None,
            limit=max(1, int(limit)),
        )
    except Exception:  # noqa: BLE001 - preferences must never fail a turn
        logger.debug("Ambient preferences unavailable", exc_info=True)
        return []
    preferences: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items or []:
        value = _item_value(item)
        if not value.get("decision"):
            continue
        key = _preference_key(
            str(value.get("observation_kind") or "other"), str(value["decision"])
        )
        if key in seen:
            continue
        seen.add(key)
        preferences.append(
            {
                "observation_kind": value.get("observation_kind"),
                "decision": value.get("decision"),
                "summary": value.get("summary"),
                "note": value.get("note"),
                "count": value.get("count"),
            }
        )
    return preferences


async def record_ambient_decision(
    store: Any,
    user_id: str,
    assistant_id: str,
    *,
    observation_id: str | None,
    observation_kind: str,
    summary: str,
    decision: str,
    note: str | None = None,
    action_taken: str | None = None,
    rated_after_action: str | None = None,
) -> dict[str, Any] | None:
    """Record what one notification card received: thumb, note, action, outcome.

    ``accept`` and ``ignore`` set the card's rating and keep any note already
    on the card; ``response`` sets the note and keeps the rating;
    ``action_taken`` says whether the avatar or the conversation partner
    replied; ``rated_after_action`` is the thumb on the avatar's reply after
    the avatar was allowed to act; ``left_alone`` marks a card nobody chose
    anything on. Returns the stored record, or ``None`` when the card cannot
    be identified.
    """
    if store is None or not user_id or not assistant_id:
        return None
    card_id = str(observation_id or "").strip()
    if not card_id:
        return None
    namespace = ambient_decision_namespace(user_id, assistant_id)
    previous: dict[str, Any] = {}
    try:
        existing = await store.aget(namespace, card_id)
    except Exception:  # noqa: BLE001 - a miss and a store error read the same
        existing = None
    if existing is not None:
        previous = _item_value(existing)
    rating = decision if decision in RATING_DECISIONS else previous.get("rating")
    final_note = (note or "").strip() if decision == "response" else None
    final_note = final_note or previous.get("note") or None
    action_taken = (
        action_taken if action_taken in ACTIONS_TAKEN else previous.get("action_taken")
    )
    rated_after_action = (
        rated_after_action
        if rated_after_action in ("like", "dislike")
        else previous.get("rated_after_action")
    )
    # A card that was acted on, rated, or noted was not left alone, whatever
    # order the records arrived in.
    left_alone = bool(previous.get("left_alone")) or decision == "left_alone"
    if rating or final_note or action_taken or rated_after_action:
        left_alone = False
    kind = (observation_kind or "other").strip().lower()[:40] or "other"
    page_content = _preference_page_content(
        observation_kind=kind,
        summary=summary,
        decision=str(rating or decision),
        note=final_note,
    )
    document = Document(
        page_content=page_content,
        metadata={
            "user_id": user_id,
            "assistant_id": assistant_id,
            "observation_id": card_id,
            "observation_kind": kind,
        },
    )
    value = {
        "document": document.to_json(),
        "observation_id": card_id,
        "observation_kind": kind,
        "summary": (summary or "").strip()[:300] or previous.get("summary") or "",
        "rating": rating,
        "note": final_note,
        "action_taken": action_taken,
        "rated_after_action": rated_after_action,
        "left_alone": left_alone,
        "decided_at": datetime.now(UTC).isoformat(),
    }
    await store.aput(namespace, key=card_id, value=value)
    return value


def ambient_decision_view(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the shape the browser keeps per card: ``{observation_id, rating, note}``."""
    if not record or not record.get("observation_id"):
        return None
    return {
        "observation_id": record.get("observation_id"),
        "observation_kind": record.get("observation_kind"),
        "rating": record.get("rating"),
        "note": record.get("note"),
        "action_taken": record.get("action_taken"),
        "rated_after_action": record.get("rated_after_action"),
        "left_alone": bool(record.get("left_alone")),
        "decided_at": record.get("decided_at"),
    }


async def list_ambient_decisions(
    store: Any,
    user_id: str,
    assistant_id: str,
    *,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Every per-card decision for one avatar, in the browser's shape."""
    if store is None or not user_id or not assistant_id:
        return []
    namespace = ambient_decision_namespace(user_id, assistant_id)
    try:
        items = await store.asearch(namespace, limit=max(1, int(limit)))
    except Exception:  # noqa: BLE001 - preferences must never fail a request
        logger.debug("Ambient decisions unavailable", exc_info=True)
        return []
    decisions: list[dict[str, Any]] = []
    for item in items or []:
        view = ambient_decision_view(_item_value(item))
        if view is not None:
            decisions.append(view)
    return decisions
