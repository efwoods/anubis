"""Run multi-statement DDL through the app's psycopg pool.

The pool in ``src/api/webapp.py`` is opened with ``prepare_threshold: 0``, so
psycopg sends every ``execute`` as a server-side prepared statement, and
Postgres refuses a prepared statement holding more than one command
(``cannot insert multiple commands into a prepared statement``). The
``CREATE TABLE`` scripts of the connected-accounts, media-asset, and inbox
repositories hold several commands each, so they run through
``execute_ddl_script`` here: one command per ``execute`` with preparation off.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def split_sql_statements(sql_script: str) -> list[str]:
    """Split a DDL script on statement terminators, dropping blanks.

    The scripts this serves contain no string literals with semicolons, so a
    plain split is exact; anything more would be speculation.
    """
    return [
        statement.strip() for statement in sql_script.split(";") if statement.strip()
    ]


async def execute_ddl_script(pool: Any, sql_script: str) -> None:
    """Execute every statement of ``sql_script`` on one pooled connection.

    Raises whatever the driver raises; callers decide whether boot survives it.
    """
    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            for statement in split_sql_statements(sql_script):
                await cursor.execute(statement, prepare=False)
