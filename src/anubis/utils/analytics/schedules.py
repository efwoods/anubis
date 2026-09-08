"""Recurring reports: the owner's standing questions and the loop that asks them.

A schedule is a question the avatar asks on the owner's behalf on a cadence
(daily, weekly, or monthly), such as "summarise what was built since the last
sprint digest". The scheduler loop (``run_due_schedules_forever``) claims the
schedules whose ``next_run_at`` has passed, runs each question through the
graph via a callback the API supplies, reads the report the run saved, and
delivers that report to the agent inbox as a notification the owner can read
from the panel or in conversation.

Claiming advances ``next_run_at`` in the same statement that selects the row
(``FOR UPDATE SKIP LOCKED``), so two API processes sharing one database never
run the same schedule twice.
"""

from __future__ import annotations

import asyncio
import calendar
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Awaitable, Callable, Protocol
from uuid import uuid4
from zoneinfo import ZoneInfo

from src.anubis.utils.postgres_ddl import execute_ddl_script

logger = logging.getLogger(__name__)

SCHEDULES_TABLE_NAME = "report_schedules"

INTERVAL_DAILY = "daily"
INTERVAL_WEEKLY = "weekly"
INTERVAL_MONTHLY = "monthly"
SCHEDULE_INTERVALS: tuple[str, ...] = (INTERVAL_DAILY, INTERVAL_WEEKLY, INTERVAL_MONTHLY)

DELIVERY_INBOX = "inbox"

SCHEDULED_RUN_HOUR = 9

REPORT_SENDER_NAME = "Neural Nexus reports"
REPORT_SOURCE_KIND = "report"
INBOX_BODY_CHARACTER_LIMIT = 4000

SPRINT_DIGEST_TITLE = "Sprint digest"
SPRINT_DIGEST_QUESTION = (
    "Summarise what was built since the last sprint digest: commits, features, "
    "hours, what is in progress, and what is upcoming. Chart the work per day "
    "and save the report."
)
SPEND_DIGEST_TITLE = "Spend digest"
SPEND_DIGEST_QUESTION = (
    "Summarise last month's spend by vendor and by category, the current burn "
    "rate, the revenue for the month, and a forecast for next month. Chart the "
    "spend per vendor and the burn rate over time, and save the report."
)

_CREATE_TABLES_SQL = f"""
CREATE TABLE IF NOT EXISTS {SCHEDULES_TABLE_NAME} (
    schedule_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    assistant_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    question TEXT NOT NULL,
    interval TEXT NOT NULL,
    next_run_at TIMESTAMPTZ NOT NULL,
    last_run_at TIMESTAMPTZ,
    last_error TEXT,
    delivery TEXT NOT NULL DEFAULT 'inbox',
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS report_schedules_due_idx
    ON {SCHEDULES_TABLE_NAME} (enabled, next_run_at);
CREATE INDEX IF NOT EXISTS report_schedules_owner_idx
    ON {SCHEDULES_TABLE_NAME} (user_id, assistant_id);
"""

_SCHEDULE_COLUMNS = (
    "schedule_id, user_id, assistant_id, kind, title, question, interval, "
    "next_run_at, last_run_at, last_error, delivery, enabled, created_at"
)
_SCHEDULE_NAMES = [name.strip() for name in _SCHEDULE_COLUMNS.split(",")]

_CLAIM_DUE_SQL = f"""
UPDATE {SCHEDULES_TABLE_NAME}
SET next_run_at = CASE interval
        WHEN 'daily' THEN next_run_at + INTERVAL '1 day'
        WHEN 'weekly' THEN next_run_at + INTERVAL '7 days'
        ELSE next_run_at + INTERVAL '1 month'
    END
WHERE schedule_id IN (
    SELECT schedule_id FROM {SCHEDULES_TABLE_NAME}
    WHERE enabled AND next_run_at <= %s
    ORDER BY next_run_at ASC
    FOR UPDATE SKIP LOCKED
    LIMIT %s
)
RETURNING {_SCHEDULE_COLUMNS};
"""


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


def normalise_interval(interval: str | None) -> str:
    """Return one of the supported intervals, defaulting to weekly."""
    candidate = str(interval or "").strip().lower()
    return candidate if candidate in SCHEDULE_INTERVALS else INTERVAL_WEEKLY


def _add_months(moment: datetime, months: int) -> datetime:
    """Advance ``moment`` by whole months, clamping the day to the month's length."""
    month_index = moment.month - 1 + months
    year = moment.year + month_index // 12
    month = month_index % 12 + 1
    day = min(moment.day, calendar.monthrange(year, month)[1])
    return moment.replace(year=year, month=month, day=day)


def next_run_after(interval: str, from_time: datetime) -> datetime:
    """Return the run after ``from_time`` for ``interval``.

    Daily adds one day, weekly adds seven days, monthly adds one calendar
    month (the day is clamped when the next month is shorter).
    """
    normalised = normalise_interval(interval)
    if normalised == INTERVAL_DAILY:
        return from_time + timedelta(days=1)
    if normalised == INTERVAL_WEEKLY:
        return from_time + timedelta(days=7)
    return _add_months(from_time, 1)


def _zone(timezone_name: str | None) -> Any:
    """Return the owner's zone, falling back to UTC for an unknown name."""
    if not timezone_name:
        return UTC
    try:
        return ZoneInfo(str(timezone_name))
    except Exception:  # noqa: BLE001 - an unknown zone name falls back to UTC
        return UTC


def first_run_time(
    interval: str, now: datetime, timezone_name: str | None = None
) -> datetime:
    """Return the first run for a new schedule, in UTC.

    Weekly runs on the next Monday at 09:00 in the owner's zone, monthly on
    the first day of the next month at 09:00, daily tomorrow at 09:00. The
    result is always strictly after ``now``.
    """
    zone = _zone(timezone_name)
    local_now = (now if now.tzinfo else now.replace(tzinfo=UTC)).astimezone(zone)
    at_nine = local_now.replace(
        hour=SCHEDULED_RUN_HOUR, minute=0, second=0, microsecond=0
    )
    normalised = normalise_interval(interval)
    if normalised == INTERVAL_DAILY:
        candidate = at_nine + timedelta(days=1)
    elif normalised == INTERVAL_WEEKLY:
        days_until_monday = (7 - local_now.weekday()) % 7 or 7
        candidate = at_nine + timedelta(days=days_until_monday)
    else:
        candidate = _add_months(at_nine.replace(day=1), 1)
    while candidate <= local_now:
        candidate = next_run_after(normalised, candidate)
    return candidate.astimezone(UTC)


def public_schedule_view(row: dict[str, Any]) -> dict[str, Any]:
    """Project a schedule as the chat tools and the browser see the schedule."""
    return {
        "schedule_id": str(row.get("schedule_id")),
        "assistant_id": row.get("assistant_id"),
        "kind": row.get("kind"),
        "title": row.get("title"),
        "question": row.get("question"),
        "interval": row.get("interval"),
        "next_run_at": _isoformat(row.get("next_run_at")),
        "last_run_at": _isoformat(row.get("last_run_at")),
        "last_error": row.get("last_error"),
        "delivery": row.get("delivery") or DELIVERY_INBOX,
        "enabled": bool(row.get("enabled", True)),
        "created_at": _isoformat(row.get("created_at")),
    }


class ScheduleRepository(Protocol):
    """Storage every schedule repository implements."""

    async def create(self, row: dict[str, Any]) -> dict[str, Any]:
        """Insert one schedule; return the stored row."""

    async def list_for_avatar(
        self, user_id: str, assistant_id: str
    ) -> list[dict[str, Any]]:
        """Return every schedule (enabled or not) for one avatar."""

    async def disable(self, user_id: str, schedule_id: str) -> bool:
        """Turn one schedule off; ``True`` when a row changed."""

    async def claim_due(self, now: datetime, limit: int) -> list[dict[str, Any]]:
        """Claim due schedules, advancing each ``next_run_at``; return the claimed rows."""

    async def mark_ran(
        self, schedule_id: str, ran_at: datetime, error: str | None = None
    ) -> None:
        """Record when a schedule last ran and whether the run failed."""


class InMemoryScheduleRepository:
    """Dictionary-backed twin for tests and the local dev server."""

    def __init__(self) -> None:
        """Start empty."""
        self.rows: dict[str, dict[str, Any]] = {}
        self.pool = None

    async def create(self, row: dict[str, Any]) -> dict[str, Any]:
        """Insert one schedule; return the stored row."""
        schedule_id = str(row.get("schedule_id") or uuid4())
        next_run_at = _as_datetime(row.get("next_run_at")) or next_run_after(
            row.get("interval") or INTERVAL_WEEKLY, _now()
        )
        stored = {
            "delivery": DELIVERY_INBOX,
            "enabled": True,
            "last_run_at": None,
            "last_error": None,
            **row,
            "schedule_id": schedule_id,
            "interval": normalise_interval(row.get("interval")),
            "next_run_at": next_run_at,
            "created_at": _now(),
        }
        self.rows[schedule_id] = stored
        return dict(stored)

    async def list_for_avatar(
        self, user_id: str, assistant_id: str
    ) -> list[dict[str, Any]]:
        """Return every schedule (enabled or not) for one avatar."""
        rows = [
            row
            for row in self.rows.values()
            if row.get("user_id") == user_id and row.get("assistant_id") == assistant_id
        ]
        rows.sort(key=lambda row: row["created_at"])
        return [dict(row) for row in rows]

    async def disable(self, user_id: str, schedule_id: str) -> bool:
        """Turn one schedule off; ``True`` when a row changed."""
        row = self.rows.get(str(schedule_id))
        if row is None or row.get("user_id") != user_id or not row.get("enabled"):
            return False
        row["enabled"] = False
        return True

    async def claim_due(self, now: datetime, limit: int) -> list[dict[str, Any]]:
        """Claim due schedules, advancing each ``next_run_at``; return the claimed rows."""
        due = [
            row
            for row in self.rows.values()
            if row.get("enabled") and row["next_run_at"] <= now
        ]
        due.sort(key=lambda row: row["next_run_at"])
        claimed = []
        for row in due[: max(0, int(limit))]:
            row["next_run_at"] = next_run_after(row["interval"], row["next_run_at"])
            claimed.append(dict(row))
        return claimed

    async def mark_ran(
        self, schedule_id: str, ran_at: datetime, error: str | None = None
    ) -> None:
        """Record when a schedule last ran and whether the run failed."""
        row = self.rows.get(str(schedule_id))
        if row is None:
            return
        row["last_run_at"] = ran_at
        row["last_error"] = error


class PostgresScheduleRepository:
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
    def _schedule_row(row: tuple) -> dict[str, Any]:
        """Turn one fetched tuple into a schedule dictionary."""
        record = dict(zip(_SCHEDULE_NAMES, row))
        record["schedule_id"] = str(record["schedule_id"])
        record["enabled"] = bool(record.get("enabled"))
        return record

    async def create(self, row: dict[str, Any]) -> dict[str, Any]:
        """Insert one schedule; return the stored row."""
        schedule_id = str(row.get("schedule_id") or uuid4())
        interval = normalise_interval(row.get("interval"))
        next_run_at = _as_datetime(row.get("next_run_at")) or next_run_after(
            interval, _now()
        )
        await self._execute(
            f"""
            INSERT INTO {SCHEDULES_TABLE_NAME}
                (schedule_id, user_id, assistant_id, kind, title, question, interval,
                 next_run_at, delivery, enabled)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """,
            (
                schedule_id,
                row["user_id"],
                row["assistant_id"],
                str(row.get("kind") or "custom"),
                str(row.get("title") or "Scheduled report"),
                str(row["question"]),
                interval,
                next_run_at,
                str(row.get("delivery") or DELIVERY_INBOX),
                bool(row.get("enabled", True)),
            ),
        )
        fetched = await self._fetchone(
            f"SELECT {_SCHEDULE_COLUMNS} FROM {SCHEDULES_TABLE_NAME} WHERE schedule_id = %s;",
            (schedule_id,),
        )
        return self._schedule_row(fetched) if fetched else {**row, "schedule_id": schedule_id}

    async def list_for_avatar(
        self, user_id: str, assistant_id: str
    ) -> list[dict[str, Any]]:
        """Return every schedule (enabled or not) for one avatar."""
        rows = await self._fetchall(
            f"SELECT {_SCHEDULE_COLUMNS} FROM {SCHEDULES_TABLE_NAME} "
            "WHERE user_id = %s AND assistant_id = %s ORDER BY created_at ASC;",
            (user_id, assistant_id),
        )
        return [self._schedule_row(row) for row in rows]

    async def disable(self, user_id: str, schedule_id: str) -> bool:
        """Turn one schedule off; ``True`` when a row changed."""
        changed = await self._execute(
            f"UPDATE {SCHEDULES_TABLE_NAME} SET enabled = FALSE "
            "WHERE user_id = %s AND schedule_id = %s AND enabled;",
            (user_id, str(schedule_id)),
        )
        return changed > 0

    async def claim_due(self, now: datetime, limit: int) -> list[dict[str, Any]]:
        """Claim due schedules, advancing each ``next_run_at``; return the claimed rows."""
        rows = await self._fetchall(_CLAIM_DUE_SQL, (now, max(1, int(limit))))
        return [self._schedule_row(row) for row in rows]

    async def mark_ran(
        self, schedule_id: str, ran_at: datetime, error: str | None = None
    ) -> None:
        """Record when a schedule last ran and whether the run failed."""
        await self._execute(
            f"UPDATE {SCHEDULES_TABLE_NAME} SET last_run_at = %s, last_error = %s "
            "WHERE schedule_id = %s;",
            (ran_at, (error or None) and str(error)[:1000], str(schedule_id)),
        )


_repository: Any | None = None


def set_schedule_repository(repository: Any | None) -> None:
    """Publish the process-wide schedule repository (or clear with ``None``)."""
    global _repository
    _repository = repository


def get_schedule_repository() -> Any | None:
    """Return the published schedule repository, or ``None``."""
    return _repository


async def ensure_schedules_table(pool: Any) -> None:
    """Create the schedules table and indexes if absent. Best-effort at boot."""
    try:
        await execute_ddl_script(pool, _CREATE_TABLES_SQL)
    except Exception as table_error:  # noqa: BLE001 - non-fatal at startup
        logger.error(
            "Could not ensure the report_schedules table exists: %s", table_error
        )


async def seed_default_schedules(
    repository: Any,
    *,
    user_id: str,
    assistant_id: str,
    timezone_name: str | None = None,
) -> list[dict[str, Any]]:
    """Insert the weekly sprint digest and the monthly spend digest once per avatar.

    Returns the rows created; an avatar that already holds any schedule is
    left untouched so the owner's own edits are never overwritten.
    """
    if not user_id or not assistant_id:
        return []
    existing = await repository.list_for_avatar(user_id, assistant_id)
    if existing:
        return []
    now = _now()
    created = []
    for kind, title, question, interval in (
        ("sprint_digest", SPRINT_DIGEST_TITLE, SPRINT_DIGEST_QUESTION, INTERVAL_WEEKLY),
        ("spend_digest", SPEND_DIGEST_TITLE, SPEND_DIGEST_QUESTION, INTERVAL_MONTHLY),
    ):
        created.append(
            await repository.create(
                {
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "kind": kind,
                    "title": title,
                    "question": question,
                    "interval": interval,
                    "next_run_at": first_run_time(interval, now, timezone_name),
                    "delivery": DELIVERY_INBOX,
                    "enabled": True,
                }
            )
        )
    return created


RunQuestion = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


async def run_schedule_once(
    schedule_row: dict[str, Any],
    *,
    run_question: RunQuestion,
    report_repository: Any,
    inbox_repository: Any,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run one schedule's question and deliver the resulting report to the inbox.

    ``run_question`` receives the schedule row and returns
    ``{"thread_id": str, "reply_text": str}`` after running the question
    through the graph. The newest report saved from that thread becomes the
    inbox notification's subject and body; when no report was saved the
    reply text stands in.
    """
    from src.anubis.utils.inbox.repository import (
        DECISION_NOTIFY,
        STATE_PENDING_OWNER,
    )

    ran_at = now or _now()
    run_result = await run_question(schedule_row) or {}
    thread_id = str(run_result.get("thread_id") or "")
    reply_text = str(run_result.get("reply_text") or "")
    report = None
    if report_repository is not None and thread_id:
        report = await report_repository.latest_for_thread(
            schedule_row["user_id"], thread_id
        )
    report_id = str(report.get("report_id")) if report else None
    subject = (report or {}).get("title") or schedule_row.get("title") or "Report"
    body_text = (
        str((report or {}).get("summary_markdown") or "")[:INBOX_BODY_CHARACTER_LIMIT]
        or reply_text[:INBOX_BODY_CHARACTER_LIMIT]
    )
    item = None
    if inbox_repository is not None:
        item = await inbox_repository.create_item(
            {
                "user_id": schedule_row["user_id"],
                "assistant_id": schedule_row["assistant_id"],
                "source_kind": REPORT_SOURCE_KIND,
                "account_key": None,
                "external_id": report_id or thread_id or str(uuid4()),
                "external_thread_id": thread_id or None,
                "sender": REPORT_SENDER_NAME,
                "recipients": [],
                "subject": subject,
                "body_text": body_text,
                "received_at": ran_at,
                "message_kind": REPORT_SOURCE_KIND,
                "decision": DECISION_NOTIFY,
                "needs_owner_action": True,
                "reason": f"Scheduled report: {schedule_row.get('title') or subject}",
                "confidence": 1.0,
                "confidence_detail": {
                    "report_id": report_id,
                    "schedule_id": str(schedule_row.get("schedule_id")),
                    "thread_id": thread_id or None,
                },
                "state": STATE_PENDING_OWNER,
            }
        )
    return {
        "schedule_id": str(schedule_row.get("schedule_id")),
        "thread_id": thread_id or None,
        "report_id": report_id,
        "inbox_item_id": (item or {}).get("item_id"),
        "ran_at": ran_at.isoformat(),
    }


def _scheduler_enabled(context: Any) -> bool:
    """Read the scheduler's on/off flag from the context."""
    return str(
        getattr(context, "report_scheduler_enabled", None) or "true"
    ).strip().lower() in ("1", "true", "yes", "on")


async def _purge_old_tool_calls() -> None:
    """Delete tool-call rows older than ninety days when a pool is published."""
    from src.anubis.utils.analytics.tool_calls import (
        get_tool_call_pool,
        purge_tool_calls_older_than,
    )

    pool = get_tool_call_pool()
    if pool is None:
        return
    try:
        await purge_tool_calls_older_than(pool, 90)
    except Exception:  # noqa: BLE001 - housekeeping never stops the scheduler
        logger.debug("tool_calls purge failed", exc_info=True)


async def run_due_schedules_forever(
    context: Any, *, run_question: RunQuestion, claim_limit: int = 10
) -> None:
    """Run due schedules on the configured interval until cancelled.

    Shaped like the inbox poller: sleep first, claim what is due, run each
    claimed schedule under the configured timeout, and record any failure on
    the schedule row. ``run_question`` is the API's callback that runs one
    question through the graph on a fresh thread.
    """
    if not _scheduler_enabled(context):
        return
    poll_seconds = float(
        getattr(context, "report_scheduler_poll_seconds", None) or 60.0
    )
    run_timeout_seconds = float(
        getattr(context, "report_schedule_run_timeout_seconds", None) or 600.0
    )
    while True:
        try:
            await asyncio.sleep(poll_seconds)
            await _run_due_schedules_once(
                run_question=run_question,
                claim_limit=claim_limit,
                run_timeout_seconds=run_timeout_seconds,
            )
            await _purge_old_tool_calls()
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001
            logger.debug("Report scheduler iteration failed", exc_info=True)


async def _run_due_schedules_once(
    *, run_question: RunQuestion, claim_limit: int, run_timeout_seconds: float
) -> int:
    """Claim and run every due schedule once; return how many ran."""
    from src.anubis.utils.inbox.repository import get_inbox_repository

    schedule_repository = get_schedule_repository()
    if schedule_repository is None:
        return 0
    now = _now()
    due_rows = await schedule_repository.claim_due(now, claim_limit)
    ran = 0
    for schedule_row in due_rows:
        schedule_id = str(schedule_row.get("schedule_id"))
        try:
            await asyncio.wait_for(
                run_schedule_once(
                    schedule_row,
                    run_question=run_question,
                    report_repository=get_report_repository_lazily(),
                    inbox_repository=get_inbox_repository(),
                    now=now,
                ),
                timeout=run_timeout_seconds,
            )
            await schedule_repository.mark_ran(schedule_id, _now(), error=None)
            ran += 1
        except asyncio.CancelledError:
            raise
        except Exception as run_error:  # noqa: BLE001 - one failure never stops the rest
            logger.warning("Scheduled report %s failed: %s", schedule_id, run_error)
            await schedule_repository.mark_ran(
                schedule_id, _now(), error=str(run_error)[:1000]
            )
    return ran


def get_report_repository_lazily() -> Any | None:
    """Return the published report repository without importing at module scope."""
    from src.anubis.utils.analytics.reports import get_report_repository

    return get_report_repository()
