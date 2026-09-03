"""The boot-time table scripts run one statement at a time, unprepared.

The application pool is opened with ``prepare_threshold: 0``, so psycopg
prepares every ``execute``; Postgres refuses a prepared statement holding more
than one command. Every multi-command ``CREATE TABLE`` script must therefore go
through ``execute_ddl_script``, which splits it and disables preparation.
"""

from __future__ import annotations

import pytest

from src.anubis.utils.connected_accounts import repository as connected_repository
from src.anubis.utils.inbox import repository as inbox_repository
from src.anubis.utils.media_assets import repository as media_repository
from src.anubis.utils.postgres_ddl import execute_ddl_script, split_sql_statements


class _FakeCursor:
    def __init__(self, calls):
        self.calls = calls

    async def execute(self, statement, params=None, *, prepare=None):
        self.calls.append((statement, prepare))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeConnection:
    def __init__(self, calls):
        self.calls = calls

    def cursor(self):
        return _FakeCursor(self.calls)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self):
        self.calls = []

    def connection(self):
        return _FakeConnection(self.calls)


SCRIPTS = {
    "connected_accounts": connected_repository._CREATE_CONNECTED_ACCOUNTS_TABLE_SQL,
    "media_assets": media_repository._CREATE_TABLES_SQL,
    "inbox": inbox_repository._CREATE_TABLES_SQL,
}


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_every_boot_script_holds_several_commands(name):
    statements = split_sql_statements(SCRIPTS[name])
    assert len(statements) > 1
    assert all(statement.upper().startswith("CREATE ") for statement in statements)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(SCRIPTS))
async def test_execute_ddl_script_sends_one_unprepared_statement_at_a_time(name):
    pool = _FakePool()
    await execute_ddl_script(pool, SCRIPTS[name])
    assert len(pool.calls) == len(split_sql_statements(SCRIPTS[name]))
    for statement, prepare in pool.calls:
        assert ";" not in statement
        assert prepare is False


@pytest.mark.asyncio
async def test_the_three_ensure_helpers_use_the_splitter():
    pool = _FakePool()
    await connected_repository.ensure_connected_accounts_table(pool)
    await media_repository.ensure_media_asset_tables(pool)
    await inbox_repository.ensure_inbox_tables(pool)
    expected = sum(len(split_sql_statements(script)) for script in SCRIPTS.values())
    assert len(pool.calls) == expected
    assert all(prepare is False for _statement, prepare in pool.calls)
