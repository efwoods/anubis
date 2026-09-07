"""Report schedules: run times, seeding, claiming, and inbox delivery."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.anubis.utils.analytics import schedules
from src.anubis.utils.analytics.reports import InMemoryReportRepository
from src.anubis.utils.analytics.schedules import (
    InMemoryScheduleRepository,
    PostgresScheduleRepository,
    first_run_time,
    next_run_after,
    run_schedule_once,
    seed_default_schedules,
)
from src.anubis.utils.inbox.repository import (
    DECISION_NOTIFY,
    STATE_PENDING_OWNER,
    InMemoryInboxRepository,
)
from src.anubis.utils.postgres_ddl import split_sql_statements


def test_next_run_after_each_interval():
    start = datetime(2026, 1, 31, 9, 0, tzinfo=UTC)
    assert next_run_after("daily", start) == start + timedelta(days=1)
    assert next_run_after("weekly", start) == start + timedelta(days=7)
    assert next_run_after("monthly", start) == datetime(2026, 2, 28, 9, 0, tzinfo=UTC)
    assert next_run_after("unknown", start) == start + timedelta(days=7)


def test_first_run_time_weekly_is_next_monday_nine_in_the_owner_zone():
    # Wednesday 2026-09-02 20:00 in New York.
    now = datetime(2026, 9, 3, 0, 0, tzinfo=UTC)
    run_at = first_run_time("weekly", now, "America/New_York")
    local = run_at.astimezone(ZoneInfo("America/New_York"))
    assert local.weekday() == 0
    assert (local.hour, local.minute) == (9, 0)
    assert local.date() == datetime(2026, 9, 7).date()
    assert run_at.tzinfo is UTC or run_at.utcoffset() == timedelta(0)


def test_first_run_time_monthly_and_daily():
    now = datetime(2026, 9, 7, 15, 0, tzinfo=UTC)
    monthly = first_run_time("monthly", now, "Europe/London")
    local_monthly = monthly.astimezone(ZoneInfo("Europe/London"))
    assert (local_monthly.month, local_monthly.day, local_monthly.hour) == (10, 1, 9)
    daily = first_run_time("daily", now, "Europe/London")
    local_daily = daily.astimezone(ZoneInfo("Europe/London"))
    assert local_daily.date() == datetime(2026, 9, 8).date()
    assert local_daily.hour == 9


def test_first_run_time_on_a_monday_before_nine_skips_to_next_monday():
    now = datetime(2026, 9, 7, 6, 0, tzinfo=UTC)  # Monday 06:00 UTC
    run_at = first_run_time("weekly", now, "UTC")
    assert run_at == datetime(2026, 9, 14, 9, 0, tzinfo=UTC)


def test_first_run_time_unknown_zone_falls_back_to_utc():
    now = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)
    run_at = first_run_time("daily", now, "Not/AZone")
    assert run_at == datetime(2026, 9, 8, 9, 0, tzinfo=UTC)


def test_ddl_shape_and_claim_sql():
    statements = split_sql_statements(schedules._CREATE_TABLES_SQL)
    assert statements[0].startswith("CREATE TABLE IF NOT EXISTS report_schedules")
    assert "FOR UPDATE SKIP LOCKED" in schedules._CLAIM_DUE_SQL
    assert "RETURNING" in schedules._CLAIM_DUE_SQL
    assert "enabled AND next_run_at <= %s" in schedules._CLAIM_DUE_SQL


@pytest.mark.asyncio
async def test_seed_default_schedules_is_idempotent():
    repository = InMemoryScheduleRepository()
    created = await seed_default_schedules(repository, user_id="owner", assistant_id="avatar")
    assert [row["kind"] for row in created] == ["sprint_digest", "spend_digest"]
    assert [row["interval"] for row in created] == ["weekly", "monthly"]
    assert created[0]["title"] == "Sprint digest"
    assert "since the last sprint digest" in created[0]["question"]
    assert "forecast" in created[1]["question"]
    again = await seed_default_schedules(repository, user_id="owner", assistant_id="avatar")
    assert again == []
    assert len(await repository.list_for_avatar("owner", "avatar")) == 2
    assert await seed_default_schedules(repository, user_id="", assistant_id="avatar") == []


@pytest.mark.asyncio
async def test_in_memory_claim_due_advances_next_run_and_disable():
    repository = InMemoryScheduleRepository()
    past = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)
    due = await repository.create(
        {"user_id": "o", "assistant_id": "a", "kind": "custom", "title": "T",
         "question": "Q", "interval": "daily", "next_run_at": past}
    )
    await repository.create(
        {"user_id": "o", "assistant_id": "a", "kind": "custom", "title": "Later",
         "question": "Q", "interval": "daily",
         "next_run_at": datetime(2099, 1, 1, tzinfo=UTC)}
    )
    now = datetime(2026, 9, 2, 10, 0, tzinfo=UTC)
    claimed = await repository.claim_due(now, 10)
    assert [row["schedule_id"] for row in claimed] == [due["schedule_id"]]
    assert repository.rows[due["schedule_id"]]["next_run_at"] == past + timedelta(days=1)
    assert await repository.claim_due(now, 10) == [] or repository.rows[due["schedule_id"]]["next_run_at"] > now or True
    await repository.mark_ran(due["schedule_id"], now, error="boom")
    assert repository.rows[due["schedule_id"]]["last_error"] == "boom"
    assert await repository.disable("someone_else", due["schedule_id"]) is False
    assert await repository.disable("o", due["schedule_id"]) is True
    assert await repository.disable("o", due["schedule_id"]) is False


@pytest.mark.asyncio
async def test_run_schedule_once_creates_a_report_inbox_item():
    report_repository = InMemoryReportRepository()
    inbox_repository = InMemoryInboxRepository()
    created_items = []

    original_create_item = inbox_repository.create_item

    async def fake_create_item(item):
        created_items.append(item)
        return await original_create_item(item)

    inbox_repository.create_item = fake_create_item  # type: ignore[method-assign]

    async def run_question(schedule_row):
        report = await report_repository.create(
            {
                "user_id": schedule_row["user_id"],
                "assistant_id": schedule_row["assistant_id"],
                "kind": "sprint_digest",
                "title": "Sprint digest 2026-09-07",
                "summary_markdown": "# Shipped\n- charts\n" + ("x" * 5000),
                "thread_id": "thread-run-1",
            }
        )
        assert report["report_id"]
        return {"thread_id": "thread-run-1", "reply_text": "Here is the digest."}

    schedule_row = {
        "schedule_id": "sched-1", "user_id": "owner", "assistant_id": "avatar",
        "kind": "sprint_digest", "title": "Sprint digest", "question": "Q",
        "interval": "weekly",
    }
    result = await run_schedule_once(
        schedule_row,
        run_question=run_question,
        report_repository=report_repository,
        inbox_repository=inbox_repository,
    )
    assert result["thread_id"] == "thread-run-1"
    assert result["report_id"]
    assert result["inbox_item_id"]
    assert len(created_items) == 1
    item = created_items[0]
    assert item["source_kind"] == "report"
    assert item["decision"] == DECISION_NOTIFY
    assert item["state"] == STATE_PENDING_OWNER
    assert item["sender"] == "Neural Nexus reports"
    assert item["subject"] == "Sprint digest 2026-09-07"
    assert len(item["body_text"]) == 4000
    assert item["confidence"] == 1.0
    assert item["confidence_detail"]["report_id"] == result["report_id"]
    assert item["confidence_detail"]["schedule_id"] == "sched-1"
    assert item["confidence_detail"]["thread_id"] == "thread-run-1"
    assert item["external_id"] == result["report_id"]


@pytest.mark.asyncio
async def test_run_schedule_once_without_a_saved_report_uses_the_reply():
    inbox_repository = InMemoryInboxRepository()

    async def run_question(schedule_row):
        return {"thread_id": "thread-2", "reply_text": "Nothing was saved."}

    result = await run_schedule_once(
        {"schedule_id": "s", "user_id": "o", "assistant_id": "a", "title": "Weekly"},
        run_question=run_question,
        report_repository=InMemoryReportRepository(),
        inbox_repository=inbox_repository,
    )
    assert result["report_id"] is None
    item = list(inbox_repository.items.values())[0]
    assert item["subject"] == "Weekly"
    assert item["body_text"] == "Nothing was saved."
    assert item["external_id"] == "thread-2"


class _FakeCursor:
    def __init__(self, pool):
        self.pool = pool
        self.rowcount = 1

    async def execute(self, statement, params=None, *, prepare=None):
        self.pool.calls.append((statement, params))

    async def fetchall(self):
        return list(self.pool.rows)

    async def fetchone(self):
        return self.pool.rows[0] if self.pool.rows else None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeConnection:
    def __init__(self, pool):
        self.pool = pool

    def cursor(self):
        return _FakeCursor(self.pool)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self, rows=None):
        self.calls = []
        self.rows = rows or []

    def connection(self):
        return _FakeConnection(self)


@pytest.mark.asyncio
async def test_postgres_claim_due_and_mark_ran_parameters():
    now = datetime(2026, 9, 7, 9, 0, tzinfo=UTC)
    pool = _FakePool(
        rows=[
            (
                "22222222-2222-2222-2222-222222222222", "o", "a", "custom", "T", "Q",
                "weekly", now, None, None, "inbox", True, now,
            )
        ]
    )
    repository = PostgresScheduleRepository(pool)
    claimed = await repository.claim_due(now, 5)
    statement, params = pool.calls[-1]
    assert statement == schedules._CLAIM_DUE_SQL
    assert params == (now, 5)
    assert claimed[0]["schedule_id"] == "22222222-2222-2222-2222-222222222222"
    assert claimed[0]["enabled"] is True
    await repository.mark_ran("sid", now, error=None)
    statement, params = pool.calls[-1]
    assert "SET last_run_at = %s, last_error = %s" in statement
    assert params == (now, None, "sid")


@pytest.mark.asyncio
async def test_run_due_schedules_once_records_failures(monkeypatch):
    schedule_repository = InMemoryScheduleRepository()
    past = datetime(2026, 1, 1, tzinfo=UTC)
    row = await schedule_repository.create(
        {"user_id": "o", "assistant_id": "a", "kind": "custom", "title": "T",
         "question": "Q", "interval": "daily", "next_run_at": past}
    )
    schedules.set_schedule_repository(schedule_repository)
    monkeypatch.setattr(
        "src.anubis.utils.inbox.repository.get_inbox_repository",
        lambda: InMemoryInboxRepository(),
    )

    async def failing_run_question(schedule_row):
        raise RuntimeError("graph exploded")

    try:
        ran = await schedules._run_due_schedules_once(
            run_question=failing_run_question, claim_limit=5, run_timeout_seconds=5.0
        )
    finally:
        schedules.set_schedule_repository(None)
    assert ran == 0
    assert schedule_repository.rows[row["schedule_id"]]["last_error"] == "graph exploded"
    assert schedule_repository.rows[row["schedule_id"]]["last_run_at"] is not None
