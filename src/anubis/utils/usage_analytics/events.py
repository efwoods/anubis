"""Normalise the action events a consenting browser reports.

The browser records every action the person takes in the Neural Nexus web
application: a click, a key that submits, a form field that changed, a route
change, a request to the API, an upload, an error. Each arrives as a small
JSON object; this module turns that object into the row shape the
``usage_analytics_events`` table stores and refuses anything oversized or
malformed, so a hostile or buggy client cannot fill the table with garbage.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

EVENT_KINDS = frozenset(
    {
        "click",
        "input",
        "submit",
        "keyboard",
        "navigation",
        "api_request",
        "upload",
        "error",
        "visibility",
        "session",
        "custom",
    }
)

MAX_EVENT_NAME_CHARACTERS = 160
MAX_TARGET_CHARACTERS = 400
MAX_ROUTE_CHARACTERS = 400
MAX_DETAIL_BYTES = 4_000
DEFAULT_MAX_EVENTS_PER_REQUEST = 200


class UsageEventError(ValueError):
    """An event the browser sent cannot be stored."""


def _clip(value: Any, limit: int) -> str | None:
    """Return ``value`` as text clipped to ``limit`` characters, or ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:limit]


def _occurred_at(value: Any) -> datetime:
    """Parse the browser's timestamp; fall back to now when unparsable."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return datetime.now(UTC)
    if isinstance(value, str) and value.strip():
        text = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return datetime.now(UTC)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return datetime.now(UTC)


def _bounded_detail(detail: Any) -> dict[str, Any]:
    """Keep the free-form detail a small JSON object."""
    if not isinstance(detail, dict):
        return {}
    encoded = json.dumps(detail, default=str)
    if len(encoded.encode("utf-8")) <= MAX_DETAIL_BYTES:
        return detail
    trimmed: dict[str, Any] = {}
    for key, value in detail.items():
        candidate = {**trimmed, str(key): value}
        if len(json.dumps(candidate, default=str).encode("utf-8")) > MAX_DETAIL_BYTES:
            break
        trimmed = candidate
    trimmed["truncated"] = True
    return trimmed


def normalise_event(
    raw_event: Any,
    *,
    user_id: str,
    session_id: str | None,
    assistant_id: str | None = None,
    thread_id: str | None = None,
    route: str | None = None,
) -> dict[str, Any]:
    """Turn one browser event into a storable row.

    The event's own ``assistant_id``, ``thread_id`` and ``route`` win over the
    batch-level values, because a batch can straddle a route change.
    """
    if not isinstance(raw_event, dict):
        raise UsageEventError("Each event must be a JSON object.")
    kind = (
        str(raw_event.get("kind") or raw_event.get("event_kind") or "").strip().lower()
    )
    if kind not in EVENT_KINDS:
        raise UsageEventError(
            f"Unknown event kind {kind!r}; expected one of {sorted(EVENT_KINDS)}."
        )
    name = _clip(
        raw_event.get("name") or raw_event.get("event_name"), MAX_EVENT_NAME_CHARACTERS
    )
    if not name:
        raise UsageEventError("Each event needs a name.")
    return {
        "user_id": str(user_id),
        "session_id": _clip(session_id, MAX_EVENT_NAME_CHARACTERS),
        "assistant_id": _clip(
            raw_event.get("assistant_id") or assistant_id, MAX_EVENT_NAME_CHARACTERS
        ),
        "thread_id": _clip(
            raw_event.get("thread_id") or thread_id, MAX_EVENT_NAME_CHARACTERS
        ),
        "event_kind": kind,
        "event_name": name,
        "route": _clip(raw_event.get("route") or route, MAX_ROUTE_CHARACTERS),
        "target": _clip(raw_event.get("target"), MAX_TARGET_CHARACTERS),
        "detail": _bounded_detail(raw_event.get("detail")),
        "occurred_at": _occurred_at(raw_event.get("occurred_at")),
    }


def normalise_event_batch(
    raw_events: Any,
    *,
    user_id: str,
    session_id: str | None,
    assistant_id: str | None = None,
    thread_id: str | None = None,
    route: str | None = None,
    max_events: int = DEFAULT_MAX_EVENTS_PER_REQUEST,
) -> list[dict[str, Any]]:
    """Normalise every event of one batch; refuse an oversized batch outright."""
    if not isinstance(raw_events, list):
        raise UsageEventError("events must be a list.")
    if len(raw_events) > max_events:
        raise UsageEventError(f"At most {max_events} events per request.")
    return [
        normalise_event(
            raw_event,
            user_id=user_id,
            session_id=session_id,
            assistant_id=assistant_id,
            thread_id=thread_id,
            route=route,
        )
        for raw_event in raw_events
    ]


def summarise_recent_actions(events: list[dict[str, Any]], limit: int = 25) -> str:
    """Render recent events as the short list the screenshot describer reads."""
    lines: list[str] = []
    for event in events[-limit:]:
        occurred = event.get("occurred_at")
        stamp = (
            occurred.strftime("%H:%M:%S")
            if isinstance(occurred, datetime)
            else str(occurred or "")
        )
        parts = [event.get("event_kind") or "", event.get("event_name") or ""]
        target = event.get("target")
        if target:
            parts.append(f"on {target}")
        route = event.get("route")
        if route:
            parts.append(f"at {route}")
        lines.append(f"- {stamp} {' '.join(part for part in parts if part)}".rstrip())
    return "\n".join(lines)
