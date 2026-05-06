"""Reference connector: AWS Cost Explorer -> sandbox parquet.

Real implementation lives at src/anubis/utils/tools/aws/cost_tools.py. This
reference shows the exact shape expected by the agent: a @tool that calls the
API outside the sandbox, normalizes the rows, and uploads bytes via
backend.upload_files([(path, bytes)]).
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pandas as pd
from langchain.tools import tool


def _client(context):
    import boto3

    if context.aws_profile:
        session = boto3.Session(profile_name=context.aws_profile)
        return session.client("ce", region_name="us-east-1")
    return boto3.client(
        "ce",
        region_name="us-east-1",
        aws_access_key_id=context.aws_access_key_id,
        aws_secret_access_key=context.aws_secret_access_key,
    )


def _query(ce, start_date: str, end_date: str) -> pd.DataFrame:
    rows: list[dict] = []
    next_token = None
    while True:
        kwargs = dict(
            TimePeriod={"Start": start_date, "End": end_date},
            Granularity="DAILY",
            Metrics=["UnblendedCost", "UsageQuantity"],
            GroupBy=[
                {"Type": "DIMENSION", "Key": "SERVICE"},
                {"Type": "DIMENSION", "Key": "USAGE_TYPE"},
            ],
        )
        if next_token:
            kwargs["NextPageToken"] = next_token
        resp = ce.get_cost_and_usage(**kwargs)
        for day in resp["ResultsByTime"]:
            d = day["TimePeriod"]["Start"]
            for g in day.get("Groups", []):
                service, usage_type = g["Keys"]
                amt = g["Metrics"]["UnblendedCost"]
                qty = g["Metrics"]["UsageQuantity"]
                rows.append(
                    {
                        "date": d,
                        "service": service,
                        "usage_type": usage_type,
                        "cost_usd": float(amt["Amount"]),
                        "currency": amt["Unit"],
                        "usage_qty": float(qty["Amount"]),
                    }
                )
        next_token = resp.get("NextPageToken")
        if not next_token:
            break
    return pd.DataFrame(rows)


def make_tool(backend):
    """Bind to a deepagents backend so the parquet is uploaded into the sandbox FS."""

    @tool(parse_docstring=True)
    async def ingest_aws_cost(days: int = 30) -> str:
        """Pull AWS Cost Explorer data and write parquet to /data/aws/clean/<run>.parquet.

        Args:
            days: Lookback window in days, capped at 30 to bound CE call cost.
        """
        from src.anubis.utils.context import GlobalContext

        context = GlobalContext()
        days = min(int(days), 30)
        end = datetime.now(tz=timezone.utc).date()
        start = end - timedelta(days=days)
        ce = _client(context)
        df = _query(ce, start.isoformat(), end.isoformat())

        run_id = uuid4().hex[:8]
        path = f"/data/aws/clean/{end.isoformat()}__{run_id}.parquet"
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        await backend.upload_files([(path, buf.getvalue())])
        return path

    return ingest_aws_cost
