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


@pytest.mark.asyncio
async def test_api_metrics_boot_adds_both_cache_columns_unprepared():
    """Regression: the two cache-column ``ALTER TABLE`` commands once rode one
    prepared ``execute``, failed on every boot from 2026-09-22, and every
    ``api_metrics`` insert then failed on the missing ``cached_prompt_tokens``
    column."""
    from src.anubis.utils.billing import metering

    pool = _FakePool()
    await metering.ensure_api_metrics_table(pool)
    sent_statements = [statement for statement, _prepare in pool.calls]
    assert all(prepare is False for _statement, prepare in pool.calls)
    assert all(";" not in statement for statement in sent_statements)
    assert any("cached_prompt_tokens" in statement for statement in sent_statements)
    assert any("cache_write_tokens" in statement for statement in sent_statements)


def test_no_module_passes_a_multi_command_constant_straight_to_execute():
    """Any ``<cursor>.execute(<CONSTANT>)`` whose constant holds more than one
    command fails under ``prepare_threshold: 0``; such a constant must go
    through ``execute_ddl_script`` instead."""
    import ast
    import pathlib

    offending_calls = []
    for source_path in pathlib.Path("src").rglob("*.py"):
        module_tree = ast.parse(source_path.read_text())
        multi_command_constants = {}
        for node in ast.walk(module_tree):
            if not isinstance(node, ast.Assign):
                continue
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                sql_text = node.value.value
            elif isinstance(node.value, ast.JoinedStr):
                sql_text = "".join(
                    part.value
                    for part in node.value.values
                    if isinstance(part, ast.Constant)
                )
            else:
                continue
            if len(split_sql_statements(sql_text)) > 1:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        multi_command_constants[target.id] = node.lineno
        for node in ast.walk(module_tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "execute"
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in multi_command_constants
            ):
                offending_calls.append(
                    f"{source_path}:{node.lineno} executes {node.args[0].id}"
                )
    assert offending_calls == []
