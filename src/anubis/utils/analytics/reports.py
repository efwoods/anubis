"""Saved reports: one row per answer the avatar wrote about a period.

Whenever the avatar answers a question that covers a period (how much was
spent in August, what shipped since the last sprint digest) the answer is
saved here with the chart specifications made in that turn, so the owner can
search past reports in conversation and the scheduler can deliver a report
to the inbox after an unattended run. The ``search`` column is a generated
full-text vector over the title and the summary, so a search such as
"burn rate" finds every report that mentions burn rate without a separate
index maintenance step.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from src.anubis.utils.postgres_ddl import execute_ddl_script

logger = logging.getLogger(__name__)

REPORTS_TABLE_NAME = "avatar_reports"

REPORT_KINDS: tuple[str, ...] = (
    "platform_usage",
    "finance",
    "vendor_usage",
    "development",
    "website",
    "sprint_digest",
    "spend_digest",
    "custom",
)

_CREATE_TABLES_SQL = f"""
CREATE TABLE IF NOT EXISTS {REPORTS_TABLE_NAME} (
    report_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    assistant_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    summary_markdown TEXT NOT NULL DEFAULT '',
    charts JSONB NOT NULL DEFAULT '[]',
    sources JSONB NOT NULL DEFAULT '[]',
    period_start TIMESTAMPTZ,
    period_end TIMESTAMPTZ,
    thread_id TEXT,
    message_id TEXT,
    schedule_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    search TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(title, '') || ' ' || coalesce(summary_markdown, ''))
    ) STORED
);
CREATE INDEX IF NOT EXISTS avatar_reports_search_idx
    ON {REPORTS_TABLE_NAME} USING GIN (search);
CREATE INDEX IF NOT EXISTS avatar_reports_owner_created_idx
    ON {REPORTS_TABLE_NAME} (user_id, assistant_id, created_at DESC);
"""

_REPORT_COLUMNS = (
    "report_id, user_id, assistant_id, kind, title, summary_markdown, charts, "
    "sources, period_start, period_end, thread_id, message_id, schedule_id, "
    "created_at"
)
_REPORT_NAMES = [name.strip() for name in _REPORT_COLUMNS.split(",")]


def _now() -> datetime:
    """Return the current moment in UTC."""
    return datetime.now(UTC)


def _isoformat(value: Any) -> Any:
    """Return ``value`` as an ISO 8601 string when the value is a datetime."""
    return value.isoformat() if isinstance(value, datetime) else value


def _as_datetime(value: Any) -> datetime | None:
    """Parse an ISO string (or pass a datetime through); ``None`` when absent."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def normalise_report_kind(kind: str | None) -> str:
    """Return ``kind`` when the value is a known report kind, else ``custom``."""
    candidate = str(kind or "").strip().lower()
    return candidate if candidate in REPORT_KINDS else "custom"


def public_report_view(row: dict[str, Any]) -> dict[str, Any]:
    """Project a stored report as the chat tools and the browser see the report.

    Timestamps are ISO strings and the chart specifications are inline so the
    browser can redraw every chart interactively.
    """
    return {
        "report_id": str(row.get("report_id")),
        "assistant_id": row.get("assistant_id"),
        "kind": row.get("kind"),
        "title": row.get("title"),
        "summary_markdown": row.get("summary_markdown") or "",
        "charts": list(row.get("charts") or []),
        "sources": list(row.get("sources") or []),
        "period_start": _isoformat(row.get("period_start")),
        "period_end": _isoformat(row.get("period_end")),
        "thread_id": row.get("thread_id"),
        "message_id": row.get("message_id"),
        "schedule_id": row.get("schedule_id"),
        "created_at": _isoformat(row.get("created_at")),
    }


class ReportRepository(Protocol):
    """Storage every report repository implements."""

    async def create(self, row: dict[str, Any]) -> dict[str, Any]:
        """Insert one report; return the stored row."""

    async def list(
        self,
        user_id: str,
        *,
        assistant_id: str | None = None,
        query: str | None = None,
        kind: str | None = None,
        since: Any = None,
        until: Any = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return the owner's reports, newest first, optionally searched."""

    async def get(self, user_id: str, report_id: str) -> dict[str, Any] | None:
        """Return one report the owner holds."""

    async def delete(self, user_id: str, report_id: str) -> bool:
        """Delete one report; ``True`` when a row was removed."""

    async def latest_for_thread(
        self, user_id: str, thread_id: str
    ) -> dict[str, Any] | None:
        """Return the newest report saved from one conversation thread."""

    async def kinds(self, user_id: str, assistant_id: str) -> list[str]:
        """Return the distinct report kinds saved for one avatar."""


class InMemoryReportRepository:
    """Dictionary-backed twin for tests and the local dev server."""

    def __init__(self) -> None:
        """Start empty."""
        self.rows: dict[str, dict[str, Any]] = {}
        self.pool = None

    async def create(self, row: dict[str, Any]) -> dict[str, Any]:
        """Insert one report; return the stored row."""
        report_id = str(row.get("report_id") or uuid4())
        stored = {
            "charts": [],
            "sources": [],
            "summary_markdown": "",
            **row,
            "report_id": report_id,
            "kind": normalise_report_kind(row.get("kind")),
            "period_start": _as_datetime(row.get("period_start")),
            "period_end": _as_datetime(row.get("period_end")),
            "created_at": _now(),
        }
        self.rows[report_id] = stored
        return dict(stored)

    async def list(
        self,
        user_id: str,
        *,
        assistant_id: str | None = None,
        query: str | None = None,
        kind: str | None = None,
        since: Any = None,
        until: Any = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return the owner's reports, newest first, optionally searched."""
        needle = str(query or "").strip().lower()
        since_moment = _as_datetime(since)
        until_moment = _as_datetime(until)
        matching = []
        for row in self.rows.values():
            if row.get("user_id") != user_id:
                continue
            if assistant_id is not None and row.get("assistant_id") != assistant_id:
                continue
            if kind and row.get("kind") != kind:
                continue
            if since_moment is not None and row["created_at"] < since_moment:
                continue
            if until_moment is not None and row["created_at"] > until_moment:
                continue
            if needle:
                haystack = (
                    f"{row.get('title') or ''} {row.get('summary_markdown') or ''}"
                ).lower()
                if needle not in haystack:
                    continue
            matching.append(row)
        matching.sort(key=lambda row: row["created_at"], reverse=True)
        window = matching[int(offset) : int(offset) + int(limit)]
        return [dict(row) for row in window]

    async def get(self, user_id: str, report_id: str) -> dict[str, Any] | None:
        """Return one report the owner holds."""
        row = self.rows.get(str(report_id))
        if row is None or row.get("user_id") != user_id:
            return None
        return dict(row)

    async def delete(self, user_id: str, report_id: str) -> bool:
        """Delete one report; ``True`` when a row was removed."""
        row = self.rows.get(str(report_id))
        if row is None or row.get("user_id") != user_id:
            return False
        del self.rows[str(report_id)]
        return True

    async def latest_for_thread(
        self, user_id: str, thread_id: str
    ) -> dict[str, Any] | None:
        """Return the newest report saved from one conversation thread."""
        candidates = [
            row
            for row in self.rows.values()
            if row.get("user_id") == user_id and row.get("thread_id") == thread_id
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda row: row["created_at"], reverse=True)
        return dict(candidates[0])

    async def kinds(self, user_id: str, assistant_id: str) -> list[str]:
        """Return the distinct report kinds saved for one avatar."""
        return sorted(
            {
                str(row.get("kind"))
                for row in self.rows.values()
                if row.get("user_id") == user_id
                and row.get("assistant_id") == assistant_id
            }
        )


class PostgresReportRepository:
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

    @staticmethod
    def _report_row(row: tuple) -> dict[str, Any]:
        """Turn one fetched tuple into a report dictionary."""
        record = dict(zip(_REPORT_NAMES, row))
        record["report_id"] = str(record["report_id"])
        record["charts"] = list(record.get("charts") or [])
        record["sources"] = list(record.get("sources") or [])
        return record

    async def create(self, row: dict[str, Any]) -> dict[str, Any]:
        """Insert one report; return the stored row."""
        from psycopg.types.json import Jsonb

        report_id = str(row.get("report_id") or uuid4())
        await self._execute(
            f"""
            INSERT INTO {REPORTS_TABLE_NAME}
                (report_id, user_id, assistant_id, kind, title, summary_markdown,
                 charts, sources, period_start, period_end, thread_id, message_id,
                 schedule_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """,
            (
                report_id,
                row["user_id"],
                row["assistant_id"],
                normalise_report_kind(row.get("kind")),
                str(row.get("title") or "Untitled report"),
                str(row.get("summary_markdown") or ""),
                Jsonb(list(row.get("charts") or [])),
                Jsonb(list(row.get("sources") or [])),
                _as_datetime(row.get("period_start")),
                _as_datetime(row.get("period_end")),
                row.get("thread_id"),
                row.get("message_id"),
                row.get("schedule_id"),
            ),
        )
        stored = await self.get(row["user_id"], report_id)
        return stored or {**row, "report_id": report_id}

    async def list(
        self,
        user_id: str,
        *,
        assistant_id: str | None = None,
        query: str | None = None,
        kind: str | None = None,
        since: Any = None,
        until: Any = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return the owner's reports, newest first, optionally searched."""
        conditions = ["user_id = %s"]
        params: list[Any] = [user_id]
        if assistant_id is not None:
            conditions.append("assistant_id = %s")
            params.append(assistant_id)
        if kind:
            conditions.append("kind = %s")
            params.append(kind)
        if since is not None:
            conditions.append("created_at >= %s")
            params.append(_as_datetime(since))
        if until is not None:
            conditions.append("created_at <= %s")
            params.append(_as_datetime(until))
        if query and str(query).strip():
            conditions.append("search @@ plainto_tsquery('english', %s)")
            params.append(str(query).strip())
        params.extend([max(1, int(limit)), max(0, int(offset))])
        rows = await self._fetchall(
            f"SELECT {_REPORT_COLUMNS} FROM {REPORTS_TABLE_NAME} "
            f"WHERE {' AND '.join(conditions)} "
            "ORDER BY created_at DESC LIMIT %s OFFSET %s;",
            tuple(params),
        )
        return [self._report_row(row) for row in rows]

    async def get(self, user_id: str, report_id: str) -> dict[str, Any] | None:
        """Return one report the owner holds."""
        row = await self._fetchone(
            f"SELECT {_REPORT_COLUMNS} FROM {REPORTS_TABLE_NAME} "
            "WHERE user_id = %s AND report_id = %s;",
            (user_id, str(report_id)),
        )
        return self._report_row(row) if row else None

    async def delete(self, user_id: str, report_id: str) -> bool:
        """Delete one report; ``True`` when a row was removed."""
        removed = await self._execute(
            f"DELETE FROM {REPORTS_TABLE_NAME} WHERE user_id = %s AND report_id = %s;",
            (user_id, str(report_id)),
        )
        return removed > 0

    async def latest_for_thread(
        self, user_id: str, thread_id: str
    ) -> dict[str, Any] | None:
        """Return the newest report saved from one conversation thread."""
        row = await self._fetchone(
            f"SELECT {_REPORT_COLUMNS} FROM {REPORTS_TABLE_NAME} "
            "WHERE user_id = %s AND thread_id = %s ORDER BY created_at DESC LIMIT 1;",
            (user_id, thread_id),
        )
        return self._report_row(row) if row else None

    async def kinds(self, user_id: str, assistant_id: str) -> list[str]:
        """Return the distinct report kinds saved for one avatar."""
        rows = await self._fetchall(
            f"SELECT DISTINCT kind FROM {REPORTS_TABLE_NAME} "
            "WHERE user_id = %s AND assistant_id = %s ORDER BY kind;",
            (user_id, assistant_id),
        )
        return [str(row[0]) for row in rows]


_repository: Any | None = None


def set_report_repository(repository: Any | None) -> None:
    """Publish the process-wide report repository (or clear with ``None``)."""
    global _repository
    _repository = repository


def get_report_repository() -> Any | None:
    """Return the published report repository, or ``None``."""
    return _repository


async def ensure_reports_table(pool: Any) -> None:
    """Create the reports table and indexes if absent. Best-effort at boot."""
    try:
        await execute_ddl_script(pool, _CREATE_TABLES_SQL)
    except Exception as table_error:  # noqa: BLE001 - non-fatal at startup
        logger.error("Could not ensure the avatar_reports table exists: %s", table_error)
