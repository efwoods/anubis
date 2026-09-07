"""The fire-and-forget tool-call recorder and the run-id timer."""

from __future__ import annotations

import asyncio
import time

import pytest

from src.anubis.utils.analytics import tool_calls
from src.anubis.utils.analytics.tool_calls import (
    ToolCallTimer,
    ensure_tool_calls_table,
    get_tool_call_pool,
    purge_tool_calls_older_than,
    record_tool_call,
    set_tool_call_pool,
)
from src.anubis.utils.postgres_ddl import split_sql_statements


class _FakeCursor:
    def __init__(self, pool):
        self.pool = pool
        self.rowcount = 3

    async def execute(self, statement, params=None, *, prepare=None):
        if self.pool.fail:
            raise RuntimeError("database down")
        self.pool.calls.append((" ".join(statement.split()), params, prepare))

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
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def connection(self):
        return _FakeConnection(self)


@pytest.fixture(autouse=True)
def _clear_pool():
    set_tool_call_pool(None)
    yield
    set_tool_call_pool(None)


def test_ddl_holds_table_and_two_indexes():
    statements = split_sql_statements(tool_calls._CREATE_TABLE_SQL)
    assert len(statements) == 3
    assert statements[0].startswith("CREATE TABLE IF NOT EXISTS tool_calls")


@pytest.mark.asyncio
async def test_ensure_table_runs_unprepared_statements():
    pool = _FakePool()
    await ensure_tool_calls_table(pool)
    assert len(pool.calls) == 3
    assert all(prepare is False for _, _, prepare in pool.calls)


@pytest.mark.asyncio
async def test_record_tool_call_inserts_in_the_background():
    pool = _FakePool()
    set_tool_call_pool(pool)
    assert get_tool_call_pool() is pool
    record_tool_call(
        user_id="owner", assistant_id="avatar", thread_id="thread",
        tool_name="make_chart", status="success", duration_ms=12.5,
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert len(pool.calls) == 1
    statement, params, _ = pool.calls[0]
    assert statement.startswith("INSERT INTO tool_calls")
    assert params[1:] == ("owner", "avatar", "thread", "make_chart", "success", 12.5)


@pytest.mark.asyncio
async def test_record_tool_call_is_silent_without_a_pool_or_name_or_on_failure():
    record_tool_call(user_id=None, assistant_id=None, thread_id=None, tool_name="x", status="success", duration_ms=1)
    failing = _FakePool(fail=True)
    set_tool_call_pool(failing)
    record_tool_call(user_id=None, assistant_id=None, thread_id=None, tool_name="", status="success", duration_ms=1)
    record_tool_call(user_id=None, assistant_id=None, thread_id=None, tool_name="y", status="error", duration_ms=1)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert failing.calls == []


def test_record_tool_call_without_a_running_loop_does_nothing():
    set_tool_call_pool(_FakePool())
    record_tool_call(user_id=None, assistant_id=None, thread_id=None, tool_name="z", status="success", duration_ms=1)


def test_timer_measures_by_run_id():
    timer = ToolCallTimer()
    timer.start(None)
    assert timer.in_flight() == 0
    timer.start("run-1")
    assert timer.in_flight() == 1
    time.sleep(0.01)
    elapsed = timer.finish("run-1")
    assert elapsed >= 5.0
    assert timer.in_flight() == 0
    assert timer.finish("run-1") == 0.0
    assert timer.finish(None) == 0.0


@pytest.mark.asyncio
async def test_purge_uses_a_day_parameter_and_returns_the_row_count():
    pool = _FakePool()
    removed = await purge_tool_calls_older_than(pool, 90)
    statement, params, _ = pool.calls[0]
    assert "INTERVAL '1 day'" in statement
    assert params == (90,)
    assert removed == 3
