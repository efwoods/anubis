"""Snapshots of the owner's reference forecast spreadsheet."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from src.anubis.utils.postgres_ddl import execute_ddl_script

logger = logging.getLogger(__name__)

REFERENCE_FORECASTS_TABLE_NAME = "reference_forecasts"

DEFAULT_REPORTING_SPREADSHEET_ID = "1_f5q4gJ3gU0ynwMGvZVNARGp_VYT5XE6hcXtokA-VPA"

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {REFERENCE_FORECASTS_TABLE_NAME} (
    id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    spreadsheet_id TEXT NOT NULL,
    sheet_title TEXT,
    metric TEXT NOT NULL,
    period TEXT,
    value DOUBLE PRECISION,
    unit TEXT,
    raw_label TEXT,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS reference_forecasts_user_metric_idx
    ON {REFERENCE_FORECASTS_TABLE_NAME} (user_id, metric, recorded_at DESC);
"""

_INSERT_SQL = f"""
INSERT INTO {REFERENCE_FORECASTS_TABLE_NAME}
    (id, user_id, spreadsheet_id, sheet_title, metric, period, value, unit, raw_label, recorded_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now());
"""

EXPECTED_BURN_METRIC_NAMES: frozenset[str] = frozenset(
    {
        "expected_burn",
        "expected_burn_usd",
        "forecast_burn",
        "projected_burn",
        "budget",
        "expected cost",
        "expected_cost",
    }
)


async def ensure_reference_forecasts_table(pool: Any) -> None:
    """Create the reference forecast table if absent."""
    try:
        await execute_ddl_script(pool, _CREATE_TABLE_SQL)
    except Exception as table_error:  # noqa: BLE001
        logger.error("Could not ensure the reference_forecasts table exists: %s", table_error)


def metric_name_from_header(header: str) -> str:
    """Turn a spreadsheet header into a stable metric name."""
    cleaned = "".join(
        character.lower() if character.isalnum() else "_"
        for character in str(header or "").strip()
    )
    return "_".join(part for part in cleaned.split("_") if part) or "value"


def rows_from_sheet_values(
    values: list[list[Any]], *, spreadsheet_id: str, sheet_title: str | None
) -> list[dict[str, Any]]:
    """Map a header row plus data rows into forecast snapshots."""
    if not values:
        return []
    headers = [metric_name_from_header(cell) for cell in values[0]]
    snapshots: list[dict[str, Any]] = []
    for row in values[1:]:
        period = str(row[0] or "").strip() if row else ""
        for index, header in enumerate(headers[1:], start=1):
            if index >= len(row):
                continue
            raw = row[index]
            number: float | None
            try:
                number = float(str(raw).replace(",", "").replace("$", "").strip())
            except (TypeError, ValueError):
                continue
            snapshots.append(
                {
                    "spreadsheet_id": spreadsheet_id,
                    "sheet_title": sheet_title,
                    "metric": header,
                    "period": period or None,
                    "value": number,
                    "unit": "usd",
                    "raw_label": str(values[0][index]) if index < len(values[0]) else header,
                }
            )
    return snapshots


async def record_sheet_rows(
    pool: Any,
    user_id: str,
    rows: list[dict[str, Any]],
) -> int:
    """Insert snapshot rows; return how many were written."""
    import uuid

    written = 0
    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            for row in rows or []:
                metric = str(row.get("metric") or "").strip()
                if not metric:
                    continue
                await cursor.execute(
                    _INSERT_SQL,
                    (
                        str(uuid.uuid4()),
                        user_id,
                        str(row.get("spreadsheet_id") or ""),
                        row.get("sheet_title"),
                        metric,
                        row.get("period"),
                        float(row.get("value") or 0.0),
                        row.get("unit") or "usd",
                        row.get("raw_label"),
                    ),
                )
                written += 1
    return written


async def latest_expected_burn(pool: Any, user_id: str) -> dict[str, Any] | None:
    """Return the newest expected-burn figure for the owner, if stored."""
    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                f"""
                SELECT metric, period, value, unit, recorded_at
                FROM {REFERENCE_FORECASTS_TABLE_NAME}
                WHERE user_id = %s
                ORDER BY recorded_at DESC
                LIMIT 50;
                """,
                (user_id,),
            )
            rows = await cursor.fetchall()
    for metric, period, value, unit, recorded_at in rows:
        if str(metric or "").strip().lower() in EXPECTED_BURN_METRIC_NAMES:
            recorded = recorded_at.isoformat() if isinstance(recorded_at, datetime) else recorded_at
            return {
                "metric": metric,
                "period": period,
                "expected_burn_usd": float(value or 0.0),
                "unit": unit,
                "recorded_at": recorded,
            }
    if rows:
        metric, period, value, unit, recorded_at = rows[0]
        recorded = recorded_at.isoformat() if isinstance(recorded_at, datetime) else recorded_at
        return {
            "metric": metric,
            "period": period,
            "expected_burn_usd": float(value or 0.0),
            "unit": unit,
            "recorded_at": recorded,
        }
    return None


async def fetch_published_sheet_values(spreadsheet_id: str) -> list[list[str]]:
    """Read a publicly published spreadsheet as rows (CSV export). Empty when private."""
    import csv
    import io

    import httpx

    url = (
        f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv"
    )
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        response = await client.get(url)
    if response.status_code >= 400 or "text/html" in (
        response.headers.get("content-type") or ""
    ):
        return []
    return list(csv.reader(io.StringIO(response.text)))


async def snapshot_published_reporting_sheet(
    pool: Any, user_id: str, spreadsheet_id: str | None = None
) -> dict[str, Any]:
    """Snapshot the reporting spreadsheet from its public CSV export when OAuth is unavailable."""
    chosen_id = spreadsheet_id or DEFAULT_REPORTING_SPREADSHEET_ID
    values = await fetch_published_sheet_values(chosen_id)
    snapshots = rows_from_sheet_values(
        values, spreadsheet_id=chosen_id, sheet_title="published"
    )
    stored = 0
    if pool is not None and snapshots:
        stored = await record_sheet_rows(pool, user_id, snapshots)
    return {
        "status": "ok" if values else "unavailable",
        "spreadsheet_id": chosen_id,
        "values": values[:200],
        "snapshots": snapshots[:100],
        "stored": stored,
    }
