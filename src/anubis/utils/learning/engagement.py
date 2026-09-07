"""The per-user, per-avatar engagement record.

One scalar store record per ``(user_id, assistant_id)`` — key ``"engagement"``
under :func:`engagement_namespace` — counting how often and how recently the
person has engaged with the avatar. ``record_engagement`` updates the record on
every human turn (called from the ``observe_user`` graph node) and
``render_engagement_section`` turns the record into the prose that fills the
``=== USER ENGAGEMENT ===`` system-prompt section.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from src.anubis.utils.learning.namespaces import engagement_namespace

logger = logging.getLogger(__name__)

ENGAGEMENT_RECORD_KEY = "engagement"

# How many distinct conversation identifiers the record keeps. The count of
# conversations is tracked separately and is never capped; the identifier list
# exists only to tell a returning conversation from a new one.
_MAX_TRACKED_CONVERSATION_IDS = 500

# How many days of per-day message counts the record keeps for the frequency
# figure. Older days roll off so the record cannot grow without bound.
_DAILY_COUNT_WINDOW_DAYS = 30


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def empty_engagement_record() -> dict[str, Any]:
    return {
        "message_count": 0,
        "conversation_count": 0,
        "conversation_thread_ids": [],
        "first_engaged_at": None,
        "last_engaged_at": None,
        "daily_message_counts": {},
        "geo_visit_count": 0,
        "last_sweep_at": None,
        "messages_since_last_sweep": 0,
    }


def apply_engagement(
    record: dict[str, Any] | None, thread_id: str | None, now: datetime | None = None
) -> dict[str, Any]:
    """Return ``record`` updated for one more human message (pure, no store)."""
    now = now or _utc_now()
    updated = {**empty_engagement_record(), **(record or {})}

    updated["message_count"] = int(updated.get("message_count") or 0) + 1
    updated["messages_since_last_sweep"] = (
        int(updated.get("messages_since_last_sweep") or 0) + 1
    )
    if not updated.get("first_engaged_at"):
        updated["first_engaged_at"] = now.isoformat()
    updated["last_engaged_at"] = now.isoformat()

    tracked_ids = list(updated.get("conversation_thread_ids") or [])
    if thread_id and thread_id not in tracked_ids:
        tracked_ids.append(thread_id)
        updated["conversation_count"] = int(updated.get("conversation_count") or 0) + 1
        if len(tracked_ids) > _MAX_TRACKED_CONVERSATION_IDS:
            tracked_ids = tracked_ids[-_MAX_TRACKED_CONVERSATION_IDS:]
    updated["conversation_thread_ids"] = tracked_ids

    daily_counts = dict(updated.get("daily_message_counts") or {})
    today_key = now.date().isoformat()
    daily_counts[today_key] = int(daily_counts.get(today_key) or 0) + 1
    oldest_kept_day = (now - timedelta(days=_DAILY_COUNT_WINDOW_DAYS)).date()
    daily_counts = {
        day_key: count
        for day_key, count in daily_counts.items()
        if _parse_day(day_key) is not None and _parse_day(day_key) >= oldest_kept_day
    }
    updated["daily_message_counts"] = daily_counts
    return updated


def _parse_day(day_key: str):
    try:
        return datetime.fromisoformat(day_key).date()
    except ValueError:
        return None


def messages_per_day(record: dict[str, Any], now: datetime | None = None) -> float:
    """Average messages per day over the tracked window (at most 30 days)."""
    now = now or _utc_now()
    daily_counts = record.get("daily_message_counts") or {}
    total_in_window = sum(int(count or 0) for count in daily_counts.values())
    first_engaged_at = _parse_timestamp(record.get("first_engaged_at"))
    if first_engaged_at is None:
        return 0.0
    days_engaged = max(1, min(_DAILY_COUNT_WINDOW_DAYS, (now - first_engaged_at).days + 1))
    return round(total_in_window / days_engaged, 2)


def describe_elapsed(seconds: float) -> str:
    """Human phrasing for an elapsed duration ("just now", "3 hours ago", ...)."""
    seconds = max(0.0, seconds)
    if seconds < 90:
        return "just now"
    minutes = seconds / 60
    if minutes < 90:
        return f"{int(round(minutes))} minutes ago"
    hours = minutes / 60
    if hours < 36:
        return f"{int(round(hours))} hours ago"
    days = hours / 24
    if days < 14:
        return f"{int(round(days))} days ago"
    weeks = days / 7
    if weeks < 9:
        return f"{int(round(weeks))} weeks ago"
    months = days / 30
    return f"{int(round(months))} months ago"


def render_engagement_section(
    record: dict[str, Any] | None, now: datetime | None = None
) -> str:
    """Prose for the ``=== USER ENGAGEMENT ===`` prompt section (empty when unknown)."""
    if not record or not record.get("message_count"):
        return ""
    now = now or _utc_now()
    message_count = int(record.get("message_count") or 0)
    conversation_count = int(record.get("conversation_count") or 0)
    last_engaged_at = _parse_timestamp(record.get("last_engaged_at"))
    first_engaged_at = _parse_timestamp(record.get("first_engaged_at"))
    lines = [
        (
            f"The user has sent {message_count} message{'s' if message_count != 1 else ''} "
            f"across {conversation_count} conversation{'s' if conversation_count != 1 else ''} "
            "with you."
        )
    ]
    if first_engaged_at is not None:
        lines.append(
            "The user first engaged with you "
            f"{describe_elapsed((now - first_engaged_at).total_seconds())}."
        )
    if last_engaged_at is not None:
        lines.append(
            "The user's most recent engagement before this message was "
            f"{describe_elapsed((now - last_engaged_at).total_seconds())}."
        )
    frequency = messages_per_day(record, now)
    if frequency > 0:
        lines.append(
            f"The user's engagement frequency is about {frequency:g} messages per day."
        )
    geo_visit_count = int(record.get("geo_visit_count") or 0)
    if geo_visit_count:
        lines.append(
            f"The user has visited your real-world location {geo_visit_count} "
            f"time{'s' if geo_visit_count != 1 else ''}."
        )
    return "\n".join(lines)


async def load_engagement_record(
    store: Any, user_id: str, assistant_id: str
) -> dict[str, Any]:
    item = await store.aget(
        engagement_namespace(user_id, assistant_id), key=ENGAGEMENT_RECORD_KEY
    )
    value = getattr(item, "value", None) or {}
    record = value.get("value") if isinstance(value, dict) else None
    return dict(record) if isinstance(record, dict) else empty_engagement_record()


async def save_engagement_record(
    store: Any, user_id: str, assistant_id: str, record: dict[str, Any]
) -> None:
    await store.aput(
        engagement_namespace(user_id, assistant_id),
        key=ENGAGEMENT_RECORD_KEY,
        value={"value": record},
    )


async def record_engagement(
    store: Any,
    user_id: str,
    assistant_id: str,
    thread_id: str | None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Count one more human message and return the updated record."""
    record = await load_engagement_record(store, user_id, assistant_id)
    updated = apply_engagement(record, thread_id, now)
    await save_engagement_record(store, user_id, assistant_id, updated)
    return updated


async def record_geo_visit(
    store: Any, user_id: str, assistant_id: str, now: datetime | None = None
) -> dict[str, Any]:
    """Count one real-world visit to the avatar's geo-located place."""
    record = await load_engagement_record(store, user_id, assistant_id)
    record["geo_visit_count"] = int(record.get("geo_visit_count") or 0) + 1
    record["last_geo_visit_at"] = (now or _utc_now()).isoformat()
    await save_engagement_record(store, user_id, assistant_id, record)
    return record


async def mark_sweep_complete(
    store: Any, user_id: str, assistant_id: str, now: datetime | None = None
) -> None:
    record = await load_engagement_record(store, user_id, assistant_id)
    record["last_sweep_at"] = (now or _utc_now()).isoformat()
    record["messages_since_last_sweep"] = 0
    await save_engagement_record(store, user_id, assistant_id, record)
