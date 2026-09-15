"""A SELECT-only gate for the owner's custom analytics queries.

The personal avatar may ask Postgres a question the named metrics do not
cover. The statement still has to be safe against a production database:
one statement, reads only, known tables, a timeout, and a row cap. Writes,
comments that hide a second statement, and unknown relations are refused
before anything reaches the server.
"""

from __future__ import annotations

import re
from typing import Any

ALLOWED_TABLES: frozenset[str] = frozenset(
    {
        "api_metrics",
        "tool_calls",
        "vendor_usage_daily",
        "finance_transactions",
        "assistant",
        "thread",
        "reports",
        "report_schedules",
        "reference_forecasts",
    }
)

FORBIDDEN_VERBS: tuple[str, ...] = (
    "INSERT",
    "UPDATE",
    "DELETE",
    "ALTER",
    "DROP",
    "COPY",
    "GRANT",
    "REVOKE",
    "TRUNCATE",
    "CREATE",
    "COMMENT",
    "VACUUM",
    "LOCK",
    "NOTIFY",
    "LISTEN",
    "UNLISTEN",
    "DO",
    "CALL",
    "EXECUTE",
    "PREPARE",
    "DEALLOCATE",
    "SET",
    "RESET",
    "SHOW",
)

DEFAULT_ROW_CAP = 500
DEFAULT_STATEMENT_TIMEOUT_MS = 5000

_COMMENT_PATTERN = re.compile(r"(--.*?$)|(/\*.*?\*/)", re.MULTILINE | re.DOTALL)
_STATEMENT_SPLIT_PATTERN = re.compile(r";")
_IDENTIFIER_PATTERN = re.compile(
    r"\b(?:FROM|JOIN|INTO|UPDATE|TABLE)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
_CTE_NAME_PATTERN = re.compile(
    r"\b(?:WITH|,)\s+([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(",
    re.IGNORECASE,
)
_VERB_PATTERN = re.compile(
    r"\b(" + "|".join(FORBIDDEN_VERBS) + r")\b",
    re.IGNORECASE,
)


class AnalyticsSqlRefused(ValueError):
    """The statement did not pass the SELECT-only gate."""


def strip_sql_comments(sql: str) -> str:
    """Remove line and block comments so a write cannot hide behind them."""
    return _COMMENT_PATTERN.sub(" ", str(sql or ""))


def validate_analytics_sql(sql: str) -> str:
    """Return the cleaned statement, or raise ``AnalyticsSqlRefused``.

    Accepts a single ``SELECT`` or ``WITH`` statement that only names tables
    in :data:`ALLOWED_TABLES`.
    """
    cleaned = strip_sql_comments(sql).strip()
    if not cleaned:
        raise AnalyticsSqlRefused("The query is empty.")
    pieces = [piece.strip() for piece in _STATEMENT_SPLIT_PATTERN.split(cleaned) if piece.strip()]
    if len(pieces) != 1:
        raise AnalyticsSqlRefused("Only one statement is allowed.")
    statement = pieces[0]
    first_word = statement.split(None, 1)[0].upper()
    if first_word not in {"SELECT", "WITH"}:
        raise AnalyticsSqlRefused("Only SELECT or WITH queries are allowed.")
    if _VERB_PATTERN.search(statement):
        raise AnalyticsSqlRefused("The query contains a verb that is not allowed.")
    named_tables = {match.group(1).lower() for match in _IDENTIFIER_PATTERN.finditer(statement)}
    cte_names = {match.group(1).lower() for match in _CTE_NAME_PATTERN.finditer(statement)}
    unknown = sorted(named_tables - ALLOWED_TABLES - cte_names)
    if unknown:
        raise AnalyticsSqlRefused(
            "These tables are not allowed: " + ", ".join(unknown) + "."
        )
    if not named_tables:
        raise AnalyticsSqlRefused(
            "The query must read one of: " + ", ".join(sorted(ALLOWED_TABLES)) + "."
        )
    return statement


class _AlreadyOpenConnection:
    """Present an open connection as the pool ``connection()`` context."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    async def __aenter__(self) -> Any:
        return self._connection

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class SingleConnectionPool:
    """A pool stand-in that hands out one already-open connection."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def connection(self) -> _AlreadyOpenConnection:
        return _AlreadyOpenConnection(self._connection)


async def run_validated_analytics_sql(
    pool: Any,
    sql: str,
    *,
    row_cap: int = DEFAULT_ROW_CAP,
    statement_timeout_ms: int = DEFAULT_STATEMENT_TIMEOUT_MS,
) -> dict[str, Any]:
    """Run a gated SELECT and return ``{columns, rows, row_count, truncated}``."""
    statement = validate_analytics_sql(sql)
    limit = max(1, int(row_cap))
    timeout_ms = max(100, int(statement_timeout_ms))
    fetch_limit = limit + 1
    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(f"SET LOCAL statement_timeout = {timeout_ms}")
            await cursor.execute(statement)
            description = getattr(cursor, "description", None) or []
            columns = [
                str(getattr(column, "name", None) or column[0])
                for column in description
            ]
            fetched = await cursor.fetchmany(fetch_limit)
    truncated = len(fetched) > limit
    rows = fetched[:limit]
    return {
        "columns": columns,
        "rows": [list(row) for row in rows],
        "row_count": len(rows),
        "truncated": truncated,
    }


async def run_analytics_sql_with_preferred_pool(
    context: Any,
    pool: Any,
    sql: str,
) -> dict[str, Any]:
    """Use the read-only URI when set; otherwise the application pool."""
    readonly_uri = str(
        getattr(context, "analytics_readonly_postgres_uri", None) or ""
    ).strip()
    if not readonly_uri:
        return await run_validated_analytics_sql(pool, sql)
    import psycopg

    async with await psycopg.AsyncConnection.connect(readonly_uri) as connection:
        return await run_validated_analytics_sql(SingleConnectionPool(connection), sql)
