"""Vendor usage: one daily number per provider, metric, and owner.

The vendor connectors (LangSmith, OpenAI, Anthropic, and any other usage
source the owner signs in to) each read a usage page or API and produce rows
such as ``{"day": "2026-09-01", "metric": "cost", "value": 12.4, "unit": "usd"}``.
``record_rows`` upserts those rows so a re-read of the same day overwrites the
earlier number instead of double counting, and the two query functions answer
"what did we spend on vendors in a period" and "how much did we use".
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from typing import Any

from src.anubis.utils.postgres_ddl import execute_ddl_script

logger = logging.getLogger(__name__)

VENDOR_USAGE_TABLE_NAME = "vendor_usage_daily"

_CREATE_TABLES_SQL = f"""
CREATE TABLE IF NOT EXISTS {VENDOR_USAGE_TABLE_NAME} (
    user_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    day DATE NOT NULL,
    metric TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL DEFAULT 0,
    unit TEXT,
    source TEXT,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, provider, day, metric)
);
CREATE INDEX IF NOT EXISTS vendor_usage_daily_user_day_idx
    ON {VENDOR_USAGE_TABLE_NAME} (user_id, day);
"""

_UPSERT_SQL = f"""
INSERT INTO {VENDOR_USAGE_TABLE_NAME} (user_id, provider, day, metric, value, unit, source, recorded_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, now())
ON CONFLICT (user_id, provider, day, metric) DO UPDATE SET
    value = EXCLUDED.value,
    unit = EXCLUDED.unit,
    source = EXCLUDED.source,
    recorded_at = now();
"""


async def ensure_vendor_usage_table(pool: Any) -> None:
    """Create the vendor usage table if absent. Best-effort at boot."""
    try:
        await execute_ddl_script(pool, _CREATE_TABLES_SQL)
    except Exception as table_error:  # noqa: BLE001 - non-fatal at startup
        logger.error("Could not ensure the vendor_usage_daily table exists: %s", table_error)


def _as_date(value: Any) -> date:
    """Return ``value`` as a date (ISO strings and datetimes accepted)."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _period_dates(since: Any, until: Any) -> tuple[date, date]:
    """Return the period as dates, defaulting to the last thirty days."""
    end = _as_date(until) if until else datetime.now(UTC).date()
    start = _as_date(since) if since else end - timedelta(days=30)
    return start, end


def _plain(value: Any) -> Any:
    """Turn database values into JSON-friendly values."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    try:
        from decimal import Decimal

        if isinstance(value, Decimal):
            return float(value)
    except ImportError:  # pragma: no cover - decimal is standard
        pass
    return value


async def record_rows(
    pool: Any,
    user_id: str,
    provider: str,
    rows: list[dict[str, Any]],
    source: str | None = None,
) -> int:
    """Upsert daily usage rows for one provider; return how many were written.

    Each row holds ``day``, ``metric``, ``value``, and optionally ``unit``.
    Rows without a day or a metric are skipped.
    """
    written = 0
    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            for row in rows or []:
                day = row.get("day") or row.get("date")
                metric = str(row.get("metric") or "").strip()
                if not day or not metric:
                    continue
                await cursor.execute(
                    _UPSERT_SQL,
                    (
                        user_id,
                        str(provider),
                        _as_date(day),
                        metric,
                        float(row.get("value") or 0.0),
                        row.get("unit"),
                        source or row.get("source"),
                    ),
                )
                written += 1
    return written


async def usage_by_period(
    pool: Any,
    user_id: str,
    provider: str | None = None,
    since: Any = None,
    until: Any = None,
    metric: str | None = None,
) -> dict[str, Any]:
    """Return daily usage rows for one provider (or all) in the period."""
    start_date, end_date = _period_dates(since, until)
    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                f"""
                SELECT day, provider, metric, value, unit
                FROM {VENDOR_USAGE_TABLE_NAME}
                WHERE user_id = %s AND day >= %s AND day <= %s
                  AND (%s::text IS NULL OR provider = %s)
                  AND (%s::text IS NULL OR metric = %s)
                ORDER BY day, provider, metric;
                """,
                (user_id, start_date, end_date, provider, provider, metric, metric),
            )
            rows = await cursor.fetchall()
    return {
        "columns": ["day", "provider", "metric", "value", "unit"],
        "rows": [[_plain(value) for value in row] for row in rows],
        "period_start": start_date.isoformat(),
        "period_end": end_date.isoformat(),
    }


async def usage_totals(
    pool: Any, user_id: str, since: Any = None, until: Any = None
) -> dict[str, Any]:
    """Sum usage by provider, metric, and unit over the period."""
    start_date, end_date = _period_dates(since, until)
    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                f"""
                SELECT provider, metric, unit, ROUND(SUM(value)::numeric, 4) AS total
                FROM {VENDOR_USAGE_TABLE_NAME}
                WHERE user_id = %s AND day >= %s AND day <= %s
                GROUP BY provider, metric, unit
                ORDER BY provider, metric;
                """,
                (user_id, start_date, end_date),
            )
            rows = await cursor.fetchall()
    return {
        "columns": ["provider", "metric", "unit", "total"],
        "rows": [[_plain(value) for value in row] for row in rows],
        "period_start": start_date.isoformat(),
        "period_end": end_date.isoformat(),
    }
