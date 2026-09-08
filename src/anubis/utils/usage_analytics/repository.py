"""Storage for opt-in usage analytics: consent, action events, page captures.

Three tables, all keyed by the account that consented:

* ``usage_analytics_consent`` — one row per user: whether the account has
  opted in, where the choice was made (signup, avatar settings, account
  settings), and when.
* ``usage_analytics_events`` — one row per action the browser reported.
* ``usage_analytics_screenshots`` — one row per capture of the web page: the
  describing model's text, a downscaled JPEG thumbnail, and the route.

The in-memory repository mirrors the Postgres one so the routes and the
describer can be tested without a database, in the same shape as the report
repository in ``src/anubis/utils/analytics/reports.py``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from src.anubis.utils.postgres_ddl import execute_ddl_script

logger = logging.getLogger(__name__)

CONSENT_TABLE_NAME = "usage_analytics_consent"
EVENTS_TABLE_NAME = "usage_analytics_events"
SCREENSHOTS_TABLE_NAME = "usage_analytics_screenshots"

CONSENT_SOURCES = frozenset({"signup", "avatar_settings", "account_settings", "api"})

_CREATE_TABLES_SQL = f"""
CREATE TABLE IF NOT EXISTS {CONSENT_TABLE_NAME} (
    user_id TEXT PRIMARY KEY,
    enabled BOOLEAN NOT NULL DEFAULT FALSE,
    source TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS {EVENTS_TABLE_NAME} (
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_id TEXT NOT NULL,
    session_id TEXT,
    assistant_id TEXT,
    thread_id TEXT,
    event_kind TEXT NOT NULL,
    event_name TEXT NOT NULL,
    route TEXT,
    target TEXT,
    detail JSONB NOT NULL DEFAULT '{{}}'::jsonb
);
CREATE INDEX IF NOT EXISTS usage_analytics_events_user_occurred_idx
    ON {EVENTS_TABLE_NAME} (user_id, occurred_at);
CREATE INDEX IF NOT EXISTS usage_analytics_events_kind_idx
    ON {EVENTS_TABLE_NAME} (event_kind, occurred_at);
CREATE TABLE IF NOT EXISTS {SCREENSHOTS_TABLE_NAME} (
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_id TEXT NOT NULL,
    session_id TEXT,
    assistant_id TEXT,
    thread_id TEXT,
    route TEXT,
    trigger TEXT,
    description TEXT,
    recent_actions TEXT,
    thumbnail BYTEA,
    thumbnail_mime TEXT,
    width INTEGER,
    height INTEGER,
    model_name TEXT,
    total_cost DOUBLE PRECISION NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending'
);
CREATE INDEX IF NOT EXISTS usage_analytics_screenshots_user_occurred_idx
    ON {SCREENSHOTS_TABLE_NAME} (user_id, occurred_at);
"""

_EVENT_COLUMNS = (
    "id",
    "created_at",
    "occurred_at",
    "user_id",
    "session_id",
    "assistant_id",
    "thread_id",
    "event_kind",
    "event_name",
    "route",
    "target",
    "detail",
)

_SCREENSHOT_COLUMNS = (
    "id",
    "created_at",
    "occurred_at",
    "user_id",
    "session_id",
    "assistant_id",
    "thread_id",
    "route",
    "trigger",
    "description",
    "recent_actions",
    "thumbnail_mime",
    "width",
    "height",
    "model_name",
    "total_cost",
    "status",
)


def _now() -> datetime:
    return datetime.now(UTC)


def _isoformat(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _as_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def normalise_consent_source(source: str | None) -> str:
    """Return a known consent source, defaulting to ``api``."""
    text = (source or "").strip().lower()
    return text if text in CONSENT_SOURCES else "api"


def consent_view(row: dict[str, Any] | None, user_id: str) -> dict[str, Any]:
    """Return the consent record as the browser reads the record."""
    if row is None:
        return {
            "user_id": user_id,
            "enabled": False,
            "source": None,
            "created_at": None,
            "updated_at": None,
            "recorded": False,
        }
    return {
        "user_id": row.get("user_id", user_id),
        "enabled": bool(row.get("enabled")),
        "source": row.get("source"),
        "created_at": _isoformat(row.get("created_at")),
        "updated_at": _isoformat(row.get("updated_at")),
        "recorded": True,
    }


def event_view(row: dict[str, Any]) -> dict[str, Any]:
    """Return one stored event as the summary route returns the event."""
    return {
        "id": str(row.get("id")),
        "occurred_at": _isoformat(row.get("occurred_at")),
        "created_at": _isoformat(row.get("created_at")),
        "session_id": row.get("session_id"),
        "assistant_id": row.get("assistant_id"),
        "thread_id": row.get("thread_id"),
        "event_kind": row.get("event_kind"),
        "event_name": row.get("event_name"),
        "route": row.get("route"),
        "target": row.get("target"),
        "detail": dict(row.get("detail") or {}),
    }


def screenshot_view(row: dict[str, Any]) -> dict[str, Any]:
    """Return one stored capture, without the thumbnail bytes."""
    return {
        "id": str(row.get("id")),
        "occurred_at": _isoformat(row.get("occurred_at")),
        "created_at": _isoformat(row.get("created_at")),
        "session_id": row.get("session_id"),
        "assistant_id": row.get("assistant_id"),
        "thread_id": row.get("thread_id"),
        "route": row.get("route"),
        "trigger": row.get("trigger"),
        "description": row.get("description"),
        "recent_actions": row.get("recent_actions"),
        "has_thumbnail": bool(row.get("thumbnail_mime")),
        "thumbnail_mime": row.get("thumbnail_mime"),
        "width": row.get("width"),
        "height": row.get("height"),
        "model_name": row.get("model_name"),
        "total_cost": float(row.get("total_cost") or 0.0),
        "status": row.get("status") or "pending",
    }


class InMemoryUsageAnalyticsRepository:
    """A dictionary-backed repository for tests and the store-less dev server."""

    def __init__(self) -> None:
        """Start empty."""
        self.consent: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.screenshots: dict[str, dict[str, Any]] = {}

    async def get_consent(self, user_id: str) -> dict[str, Any] | None:
        """Return the consent row of one user, or ``None``."""
        return self.consent.get(str(user_id))

    async def set_consent(
        self, user_id: str, enabled: bool, source: str | None = None
    ) -> dict[str, Any]:
        """Record whether one user opted in, and where the choice was made."""
        existing = self.consent.get(str(user_id))
        now = _now()
        row = {
            "user_id": str(user_id),
            "enabled": bool(enabled),
            "source": normalise_consent_source(source),
            "created_at": existing["created_at"] if existing else now,
            "updated_at": now,
        }
        self.consent[str(user_id)] = row
        return row

    async def is_enabled(self, user_id: str) -> bool:
        """Return whether one user has opted in."""
        row = await self.get_consent(user_id)
        return bool(row and row.get("enabled"))

    async def record_events(self, rows: list[dict[str, Any]]) -> int:
        """Store a batch of normalised events; return how many were stored."""
        for row in rows:
            stored = {
                **row,
                "id": str(row.get("id") or uuid4()),
                "created_at": _now(),
            }
            self.events.append(stored)
        return len(rows)

    async def list_events(
        self,
        user_id: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        session_id: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return one user's events, newest first, within the given bounds."""
        matches = [
            row
            for row in self.events
            if row["user_id"] == str(user_id)
            and (since is None or row["occurred_at"] >= since)
            and (until is None or row["occurred_at"] <= until)
            and (session_id is None or row.get("session_id") == session_id)
        ]
        matches.sort(key=lambda row: row["occurred_at"], reverse=True)
        return matches[offset : offset + limit]

    async def recent_events(
        self, user_id: str, session_id: str | None, limit: int = 25
    ) -> list[dict[str, Any]]:
        """Return the latest events of one session, oldest first, for the describer."""
        rows = await self.list_events(user_id, session_id=session_id, limit=limit)
        return list(reversed(rows))

    async def create_screenshot(self, row: dict[str, Any]) -> dict[str, Any]:
        """Insert one capture row; return the stored row with the id filled in."""
        screenshot_id = str(row.get("id") or uuid4())
        stored = {
            **row,
            "id": screenshot_id,
            "created_at": _now(),
            "status": row.get("status") or "pending",
            "total_cost": float(row.get("total_cost") or 0.0),
        }
        self.screenshots[screenshot_id] = stored
        return stored

    async def complete_screenshot(
        self,
        screenshot_id: str,
        *,
        description: str | None,
        model_name: str | None,
        total_cost: float,
        status: str,
    ) -> dict[str, Any] | None:
        """Record the describer's outcome on one capture row."""
        stored = self.screenshots.get(str(screenshot_id))
        if stored is None:
            return None
        stored.update(
            {
                "description": description,
                "model_name": model_name,
                "total_cost": float(total_cost or 0.0),
                "status": status,
            }
        )
        return stored

    async def list_screenshots(
        self,
        user_id: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        session_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return one user's captures, newest first, without thumbnail bytes."""
        matches = [
            row
            for row in self.screenshots.values()
            if row["user_id"] == str(user_id)
            and (since is None or row["occurred_at"] >= since)
            and (until is None or row["occurred_at"] <= until)
            and (session_id is None or row.get("session_id") == session_id)
        ]
        matches.sort(key=lambda row: row["occurred_at"], reverse=True)
        return matches[offset : offset + limit]

    async def get_screenshot(
        self, user_id: str, screenshot_id: str
    ) -> dict[str, Any] | None:
        """Return one capture of one user, or ``None``."""
        row = self.screenshots.get(str(screenshot_id))
        if row is None or row["user_id"] != str(user_id):
            return None
        return row

    async def get_thumbnail(
        self, user_id: str, screenshot_id: str
    ) -> tuple[bytes, str] | None:
        """Return the thumbnail bytes and mime of one capture, or ``None``."""
        row = await self.get_screenshot(user_id, screenshot_id)
        if row is None or not row.get("thumbnail"):
            return None
        return bytes(row["thumbnail"]), str(row.get("thumbnail_mime") or "image/jpeg")

    async def delete_user_data(self, user_id: str) -> dict[str, int]:
        """Delete every event and capture of one user; return the counts."""
        events_before = len(self.events)
        self.events = [row for row in self.events if row["user_id"] != str(user_id)]
        screenshot_ids = [
            key
            for key, row in self.screenshots.items()
            if row["user_id"] == str(user_id)
        ]
        for key in screenshot_ids:
            self.screenshots.pop(key, None)
        return {
            "events": events_before - len(self.events),
            "screenshots": len(screenshot_ids),
        }

    async def purge_older_than(self, days: int) -> dict[str, int]:
        """Delete events and captures older than ``days``; return the counts."""
        cutoff = _now() - timedelta(days=int(days))
        events_before = len(self.events)
        self.events = [row for row in self.events if row["occurred_at"] >= cutoff]
        stale = [
            key for key, row in self.screenshots.items() if row["occurred_at"] < cutoff
        ]
        for key in stale:
            self.screenshots.pop(key, None)
        return {"events": events_before - len(self.events), "screenshots": len(stale)}


class PostgresUsageAnalyticsRepository:
    """Repository over the application's psycopg connection pool."""

    def __init__(self, pool: Any) -> None:
        """Bind to the application's ``AsyncConnectionPool``."""
        self.pool = pool

    async def _fetchall(self, sql: str, params: tuple = ()) -> list[tuple]:
        """Run one query and return every row."""
        async with self.pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(sql, params)
                return await cursor.fetchall()

    async def _fetchone(self, sql: str, params: tuple = ()) -> tuple | None:
        """Run one query and return the first row."""
        async with self.pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(sql, params)
                return await cursor.fetchone()

    async def _execute(self, sql: str, params: tuple = ()) -> int:
        """Run one statement and return the affected row count."""
        async with self.pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(sql, params)
                return int(cursor.rowcount or 0)

    async def get_consent(self, user_id: str) -> dict[str, Any] | None:
        """Return the consent row of one user, or ``None``."""
        row = await self._fetchone(
            f"SELECT user_id, enabled, source, created_at, updated_at FROM {CONSENT_TABLE_NAME} WHERE user_id = %s;",
            (str(user_id),),
        )
        if row is None:
            return None
        return dict(
            zip(("user_id", "enabled", "source", "created_at", "updated_at"), row)
        )

    async def set_consent(
        self, user_id: str, enabled: bool, source: str | None = None
    ) -> dict[str, Any]:
        """Record whether one user opted in, and where the choice was made."""
        await self._execute(
            f"""
            INSERT INTO {CONSENT_TABLE_NAME} (user_id, enabled, source, created_at, updated_at)
            VALUES (%s, %s, %s, now(), now())
            ON CONFLICT (user_id) DO UPDATE
                SET enabled = EXCLUDED.enabled,
                    source = EXCLUDED.source,
                    updated_at = now();
            """,
            (str(user_id), bool(enabled), normalise_consent_source(source)),
        )
        stored = await self.get_consent(user_id)
        return stored or {
            "user_id": str(user_id),
            "enabled": bool(enabled),
            "source": normalise_consent_source(source),
        }

    async def is_enabled(self, user_id: str) -> bool:
        """Return whether one user has opted in."""
        row = await self.get_consent(user_id)
        return bool(row and row.get("enabled"))

    async def record_events(self, rows: list[dict[str, Any]]) -> int:
        """Store a batch of normalised events; return how many were stored."""
        if not rows:
            return 0
        from psycopg.types.json import Jsonb

        async with self.pool.connection() as connection:
            async with connection.cursor() as cursor:
                for row in rows:
                    await cursor.execute(
                        f"""
                        INSERT INTO {EVENTS_TABLE_NAME}
                            (id, occurred_at, user_id, session_id, assistant_id, thread_id,
                             event_kind, event_name, route, target, detail)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                        """,
                        (
                            str(row.get("id") or uuid4()),
                            row.get("occurred_at") or _now(),
                            str(row["user_id"]),
                            row.get("session_id"),
                            row.get("assistant_id"),
                            row.get("thread_id"),
                            row["event_kind"],
                            row["event_name"],
                            row.get("route"),
                            row.get("target"),
                            Jsonb(dict(row.get("detail") or {})),
                        ),
                    )
        return len(rows)

    @staticmethod
    def _event_row(row: tuple) -> dict[str, Any]:
        """Turn one fetched tuple into an event dictionary."""
        record = dict(zip(_EVENT_COLUMNS, row))
        record["id"] = str(record["id"])
        record["detail"] = dict(record.get("detail") or {})
        return record

    async def list_events(
        self,
        user_id: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        session_id: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return one user's events, newest first, within the given bounds."""
        clauses = ["user_id = %s"]
        params: list[Any] = [str(user_id)]
        if since is not None:
            clauses.append("occurred_at >= %s")
            params.append(since)
        if until is not None:
            clauses.append("occurred_at <= %s")
            params.append(until)
        if session_id:
            clauses.append("session_id = %s")
            params.append(session_id)
        params.extend([int(limit), int(offset)])
        rows = await self._fetchall(
            f"""
            SELECT {", ".join(_EVENT_COLUMNS)} FROM {EVENTS_TABLE_NAME}
            WHERE {" AND ".join(clauses)}
            ORDER BY occurred_at DESC
            LIMIT %s OFFSET %s;
            """,
            tuple(params),
        )
        return [self._event_row(row) for row in rows]

    async def recent_events(
        self, user_id: str, session_id: str | None, limit: int = 25
    ) -> list[dict[str, Any]]:
        """Return the latest events of one session, oldest first, for the describer."""
        rows = await self.list_events(
            user_id, session_id=session_id or None, limit=limit
        )
        return list(reversed(rows))

    async def create_screenshot(self, row: dict[str, Any]) -> dict[str, Any]:
        """Insert one capture row; return the stored row with the id filled in."""
        screenshot_id = str(row.get("id") or uuid4())
        await self._execute(
            f"""
            INSERT INTO {SCREENSHOTS_TABLE_NAME}
                (id, occurred_at, user_id, session_id, assistant_id, thread_id, route,
                 trigger, description, recent_actions, thumbnail, thumbnail_mime,
                 width, height, model_name, total_cost, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """,
            (
                screenshot_id,
                row.get("occurred_at") or _now(),
                str(row["user_id"]),
                row.get("session_id"),
                row.get("assistant_id"),
                row.get("thread_id"),
                row.get("route"),
                row.get("trigger"),
                row.get("description"),
                row.get("recent_actions"),
                row.get("thumbnail"),
                row.get("thumbnail_mime"),
                row.get("width"),
                row.get("height"),
                row.get("model_name"),
                float(row.get("total_cost") or 0.0),
                row.get("status") or "pending",
            ),
        )
        return {**row, "id": screenshot_id, "status": row.get("status") or "pending"}

    async def complete_screenshot(
        self,
        screenshot_id: str,
        *,
        description: str | None,
        model_name: str | None,
        total_cost: float,
        status: str,
    ) -> dict[str, Any] | None:
        """Record the describer's outcome on one capture row."""
        updated = await self._execute(
            f"""
            UPDATE {SCREENSHOTS_TABLE_NAME}
            SET description = %s, model_name = %s, total_cost = %s, status = %s
            WHERE id = %s;
            """,
            (
                description,
                model_name,
                float(total_cost or 0.0),
                status,
                str(screenshot_id),
            ),
        )
        if not updated:
            return None
        return {
            "id": str(screenshot_id),
            "description": description,
            "model_name": model_name,
            "total_cost": float(total_cost or 0.0),
            "status": status,
        }

    @staticmethod
    def _screenshot_row(row: tuple) -> dict[str, Any]:
        """Turn one fetched tuple into a capture dictionary."""
        record = dict(zip(_SCREENSHOT_COLUMNS, row))
        record["id"] = str(record["id"])
        return record

    async def list_screenshots(
        self,
        user_id: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        session_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return one user's captures, newest first, without thumbnail bytes."""
        clauses = ["user_id = %s"]
        params: list[Any] = [str(user_id)]
        if since is not None:
            clauses.append("occurred_at >= %s")
            params.append(since)
        if until is not None:
            clauses.append("occurred_at <= %s")
            params.append(until)
        if session_id:
            clauses.append("session_id = %s")
            params.append(session_id)
        params.extend([int(limit), int(offset)])
        rows = await self._fetchall(
            f"""
            SELECT {", ".join(_SCREENSHOT_COLUMNS)} FROM {SCREENSHOTS_TABLE_NAME}
            WHERE {" AND ".join(clauses)}
            ORDER BY occurred_at DESC
            LIMIT %s OFFSET %s;
            """,
            tuple(params),
        )
        return [self._screenshot_row(row) for row in rows]

    async def get_screenshot(
        self, user_id: str, screenshot_id: str
    ) -> dict[str, Any] | None:
        """Return one capture of one user, or ``None``."""
        row = await self._fetchone(
            f"""
            SELECT {", ".join(_SCREENSHOT_COLUMNS)} FROM {SCREENSHOTS_TABLE_NAME}
            WHERE user_id = %s AND id = %s;
            """,
            (str(user_id), str(screenshot_id)),
        )
        return None if row is None else self._screenshot_row(row)

    async def get_thumbnail(
        self, user_id: str, screenshot_id: str
    ) -> tuple[bytes, str] | None:
        """Return the thumbnail bytes and mime of one capture, or ``None``."""
        row = await self._fetchone(
            f"SELECT thumbnail, thumbnail_mime FROM {SCREENSHOTS_TABLE_NAME} WHERE user_id = %s AND id = %s;",
            (str(user_id), str(screenshot_id)),
        )
        if row is None or row[0] is None:
            return None
        return bytes(row[0]), str(row[1] or "image/jpeg")

    async def delete_user_data(self, user_id: str) -> dict[str, int]:
        """Delete every event and capture of one user; return the counts."""
        events = await self._execute(
            f"DELETE FROM {EVENTS_TABLE_NAME} WHERE user_id = %s;", (str(user_id),)
        )
        screenshots = await self._execute(
            f"DELETE FROM {SCREENSHOTS_TABLE_NAME} WHERE user_id = %s;", (str(user_id),)
        )
        return {"events": events, "screenshots": screenshots}

    async def purge_older_than(self, days: int) -> dict[str, int]:
        """Delete events and captures older than ``days``; return the counts."""
        events = await self._execute(
            f"DELETE FROM {EVENTS_TABLE_NAME} WHERE occurred_at < now() - (%s * INTERVAL '1 day');",
            (int(days),),
        )
        screenshots = await self._execute(
            f"DELETE FROM {SCREENSHOTS_TABLE_NAME} WHERE occurred_at < now() - (%s * INTERVAL '1 day');",
            (int(days),),
        )
        return {"events": events, "screenshots": screenshots}


_repository: Any | None = None


def set_usage_analytics_repository(repository: Any | None) -> None:
    """Publish the repository the routes and the describer use."""
    global _repository
    _repository = repository


def get_usage_analytics_repository() -> Any | None:
    """Return the published repository, or ``None`` before the lifespan ran."""
    return _repository


async def ensure_usage_analytics_tables(pool: Any) -> None:
    """Create the consent, events, and screenshots tables when absent."""
    await execute_ddl_script(pool, _CREATE_TABLES_SQL)
    logger.info("Usage analytics tables are ready.")
