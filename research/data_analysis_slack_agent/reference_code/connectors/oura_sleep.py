"""Reference connector: Oura v2 -> sandbox parquet.

Real implementation lives at src/anubis/utils/tools/health/sleep_tools.py.
"""

from __future__ import annotations

import io
from datetime import date, timedelta
from uuid import uuid4

import httpx
import pandas as pd
from langchain.tools import tool


OURA_BASE = "https://api.ouraring.com/v2/usercollection/sleep"


async def _fetch(token: str, start: date, end: date) -> list[dict]:
    headers = {"Authorization": f"Bearer {token}"}
    params = {"start_date": start.isoformat(), "end_date": end.isoformat()}
    async with httpx.AsyncClient(timeout=30.0) as client:
        items: list[dict] = []
        url = OURA_BASE
        while url:
            r = await client.get(url, headers=headers, params=params if url == OURA_BASE else None)
            r.raise_for_status()
            payload = r.json()
            items.extend(payload.get("data", []))
            url = payload.get("next_token")
            if url:
                url = f"{OURA_BASE}?next_token={url}"
        return items


def _normalize(items: list[dict]) -> pd.DataFrame:
    rows = []
    for it in items:
        rows.append(
            {
                "date": it.get("day"),
                "total_sleep_min": int((it.get("total_sleep_duration") or 0) / 60),
                "efficiency_pct": float(it.get("efficiency") or 0),
                "rem_min": int((it.get("rem_sleep_duration") or 0) / 60),
                "deep_min": int((it.get("deep_sleep_duration") or 0) / 60),
                "awake_min": int((it.get("awake_time") or 0) / 60),
                "hrv_ms": float(it.get("average_hrv") or 0),
                "score": int(it.get("score") or 0),
            }
        )
    return pd.DataFrame(rows)


def make_tool(backend):
    @tool(parse_docstring=True)
    async def ingest_oura_sleep(days: int = 7) -> str:
        """Pull Oura v2 sleep records and write parquet.

        Args:
            days: Lookback window in days, capped at 90.
        """
        from src.anubis.utils.context import GlobalContext

        context = GlobalContext()
        days = min(int(days), 90)
        end = date.today()
        start = end - timedelta(days=days)
        items = await _fetch(context.oura_api_token, start, end)
        df = _normalize(items)

        run_id = uuid4().hex[:8]
        path = f"/data/sleep/clean/{end.isoformat()}__{run_id}.parquet"
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        await backend.upload_files([(path, buf.getvalue())])
        return path

    return ingest_oura_sleep
