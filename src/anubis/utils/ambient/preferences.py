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


def ambient_preference_namespace(
    user_id: str, assistant_id: str
) -> tuple[str, str, str]:
    """Return the store namespace holding one user's preferences for one avatar."""
    return (str(user_id), str(assistant_id), AMBIENT_PREFERENCE_NAMESPACE_SUFFIX)


def _preference_key(observation_kind: str, decision: str) -> str:
    return f"{observation_kind}:{decision}"


def _preference_page_content(
    *, observation_kind: str, summary: str, decision: str, note: str | None
) -> str:
    text = f"{observation_kind}: {summary or 'a scene of this kind'} -> {decision}."
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
