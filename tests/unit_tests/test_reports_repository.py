"""Saved reports: in-memory repository behaviour and the Postgres SQL shape."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.anubis.utils.analytics import reports
from src.anubis.utils.analytics.reports import (
    REPORT_KINDS,
    InMemoryReportRepository,
    PostgresReportRepository,
    get_report_repository,
    normalise_report_kind,
    public_report_view,
    set_report_repository,
)
from src.anubis.utils.postgres_ddl import split_sql_statements


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


def _row(**overrides):
    base = {
        "user_id": "owner",
        "assistant_id": "avatar",
        "kind": "finance",
        "title": "August spend",
        "summary_markdown": "Burn rate was 3k.",
        "charts": [{"chart_id": "abc"}],
        "thread_id": "thread-1",
    }
    base.update(overrides)
    return base


def test_ddl_creates_table_search_index_and_owner_index():
    statements = split_sql_statements(reports._CREATE_TABLES_SQL)
    assert len(statements) == 3
    assert "avatar_reports" in statements[0]
    assert "GENERATED ALWAYS AS" in statements[0]
    assert "USING GIN (search)" in statements[1]
    assert "(user_id, assistant_id, created_at DESC)" in statements[2]


def test_kind_normalisation():
    assert normalise_report_kind("finance") == "finance"
    assert normalise_report_kind("SPRINT_DIGEST") == "sprint_digest"
    assert normalise_report_kind("nonsense") == "custom"
    assert "custom" in REPORT_KINDS


@pytest.mark.asyncio
async def test_in_memory_create_list_get_delete_and_search():
    repository = InMemoryReportRepository()
    first = await repository.create(_row())
    await repository.create(_row(title="Sprint 3", kind="sprint_digest", summary_markdown="Shipped charts."))
    await repository.create(_row(user_id="someone_else", title="Burn rate elsewhere"))

    listed = await repository.list("owner")
    assert [report["title"] for report in listed] == ["Sprint 3", "August spend"]

    searched = await repository.list("owner", query="BURN RATE")
    assert [report["title"] for report in searched] == ["August spend"]

    by_kind = await repository.list("owner", kind="sprint_digest")
    assert [report["title"] for report in by_kind] == ["Sprint 3"]

    fetched = await repository.get("owner", first["report_id"])
    assert fetched["charts"] == [{"chart_id": "abc"}]
    assert await repository.get("someone_else", first["report_id"]) is None

    assert await repository.kinds("owner", "avatar") == ["finance", "sprint_digest"]
    assert await repository.delete("someone_else", first["report_id"]) is False
    assert await repository.delete("owner", first["report_id"]) is True
    assert await repository.get("owner", first["report_id"]) is None


@pytest.mark.asyncio
async def test_in_memory_latest_for_thread_and_period_filters():
    repository = InMemoryReportRepository()
    older = await repository.create(_row(title="older"))
    older_created = older["created_at"] - timedelta(days=2)
    repository.rows[older["report_id"]]["created_at"] = older_created
    newer = await repository.create(_row(title="newer"))

    latest = await repository.latest_for_thread("owner", "thread-1")
    assert latest["report_id"] == newer["report_id"]
    assert await repository.latest_for_thread("owner", "missing") is None

    since = datetime.now(UTC) - timedelta(days=1)
    recent = await repository.list("owner", since=since.isoformat())
    assert [report["title"] for report in recent] == ["newer"]
    assert len(await repository.list("owner", limit=1)) == 1
    assert [report["title"] for report in await repository.list("owner", offset=1)] == ["older"]


def test_public_report_view_uses_iso_timestamps_and_inline_charts():
    moment = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    view = public_report_view(
        {
            "report_id": "abc",
            "created_at": moment,
            "period_start": moment,
            "charts": [{"chart_id": "c1"}],
            "title": "T",
        }
    )
    assert view["created_at"] == moment.isoformat()
    assert view["period_start"] == moment.isoformat()
    assert view["period_end"] is None
    assert view["charts"] == [{"chart_id": "c1"}]
    assert view["summary_markdown"] == ""


@pytest.mark.asyncio
async def test_postgres_list_uses_full_text_search_and_parameters():
    pool = _FakePool()
    repository = PostgresReportRepository(pool)
    await repository.list("owner", assistant_id="avatar", query="burn rate", kind="finance", limit=5, offset=2)
    statement, params = pool.calls[-1]
    assert "search @@ plainto_tsquery('english', %s)" in statement
    assert "ORDER BY created_at DESC LIMIT %s OFFSET %s" in statement
    assert params == ("owner", "avatar", "finance", "burn rate", 5, 2)
    assert "burn rate" not in statement


@pytest.mark.asyncio
async def test_postgres_create_inserts_then_reads_back():
    moment = datetime(2026, 9, 1, tzinfo=UTC)
    pool = _FakePool(
        rows=[
            (
                "11111111-1111-1111-1111-111111111111", "owner", "avatar", "finance",
                "August spend", "Burn rate", [{"chart_id": "abc"}], [], None, None,
                "thread-1", None, None, moment,
            )
        ]
    )
    repository = PostgresReportRepository(pool)
    stored = await repository.create(_row(report_id="11111111-1111-1111-1111-111111111111"))
    insert_statement, insert_params = pool.calls[0]
    assert insert_statement.strip().startswith("INSERT INTO avatar_reports")
    assert insert_params[0] == "11111111-1111-1111-1111-111111111111"
    assert insert_params[3] == "finance"
    assert stored["title"] == "August spend"
    assert stored["charts"] == [{"chart_id": "abc"}]


@pytest.mark.asyncio
async def test_postgres_delete_and_latest_for_thread_are_scoped_to_the_owner():
    pool = _FakePool()
    repository = PostgresReportRepository(pool)
    assert await repository.delete("owner", "rid") is True
    statement, params = pool.calls[-1]
    assert "WHERE user_id = %s AND report_id = %s" in statement
    assert params == ("owner", "rid")
    assert await repository.latest_for_thread("owner", "thread-9") is None
    statement, params = pool.calls[-1]
    assert "thread_id = %s ORDER BY created_at DESC LIMIT 1" in statement
    assert params == ("owner", "thread-9")


def test_repository_publication_round_trip():
    repository = InMemoryReportRepository()
    set_report_repository(repository)
    try:
        assert get_report_repository() is repository
    finally:
        set_report_repository(None)
    assert get_report_repository() is None
