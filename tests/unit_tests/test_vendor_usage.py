"""Vendor usage: upserts and period queries over a fake pool."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from src.anubis.utils.analytics import vendor_usage
from src.anubis.utils.postgres_ddl import split_sql_statements


class _FakeCursor:
    def __init__(self, pool):
        self.pool = pool

    async def execute(self, statement, params=None, *, prepare=None):
        self.pool.calls.append((statement.strip(), params))

    async def fetchall(self):
        return list(self.pool.rows)

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


def test_ddl_has_a_composite_primary_key():
    statements = split_sql_statements(vendor_usage._CREATE_TABLES_SQL)
    assert "PRIMARY KEY (user_id, provider, day, metric)" in statements[0]


@pytest.mark.asyncio
async def test_record_rows_upserts_and_skips_incomplete_rows():
    pool = _FakePool()
    written = await vendor_usage.record_rows(
        pool,
        "owner",
        "openai",
        [
            {"day": "2026-09-01", "metric": "cost", "value": 12.5, "unit": "usd"},
            {"day": datetime(2026, 9, 2, tzinfo=UTC), "metric": "tokens", "value": 1000},
            {"metric": "cost", "value": 1},
            {"day": "2026-09-03", "value": 1},
        ],
        source="usage_page",
    )
    assert written == 2
    assert all("ON CONFLICT (user_id, provider, day, metric) DO UPDATE" in call[0] for call in pool.calls)
    assert pool.calls[0][1] == ("owner", "openai", date(2026, 9, 1), "cost", 12.5, "usd", "usage_page")
    assert pool.calls[1][1][2] == date(2026, 9, 2)


@pytest.mark.asyncio
async def test_usage_by_period_and_totals_parameterise_the_filters():
    pool = _FakePool(rows=[(date(2026, 9, 1), "openai", "cost", 12.5, "usd")])
    result = await vendor_usage.usage_by_period(
        pool, "owner", "openai", "2026-09-01", "2026-09-30", metric="cost"
    )
    statement, params = pool.calls[-1]
    assert params == ("owner", date(2026, 9, 1), date(2026, 9, 30), "openai", "openai", "cost", "cost")
    assert result["rows"] == [["2026-09-01", "openai", "cost", 12.5, "usd"]]
    assert result["period_start"] == "2026-09-01"

    pool.rows = [("openai", "cost", "usd", 40.0)]
    totals = await vendor_usage.usage_totals(pool, "owner", since=None, until=None)
    statement, params = pool.calls[-1]
    assert "GROUP BY provider, metric, unit" in statement
    assert params[0] == "owner"
    assert totals["columns"] == ["provider", "metric", "unit", "total"]
    assert totals["rows"] == [["openai", "cost", "usd", 40.0]]
