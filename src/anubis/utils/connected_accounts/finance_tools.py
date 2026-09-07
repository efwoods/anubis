"""Bank and card tools over the Finance (Plaid) connection.

The connection holds an encrypted Plaid access token per institution. The
analytics package keeps the transactions in ``finance_transactions`` and
syncs them from Plaid on demand; these tools answer the owner's money
questions from that table and name the institution and accounts involved.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from langchain.tools import tool

logger = logging.getLogger(__name__)

FINANCE_TOOL_NAMES: tuple[str, ...] = ("finance_accounts", "finance_transactions", "finance_spend_summary")


def _period(since: str | None, until: str | None) -> tuple[str, str]:
    now = datetime.now(UTC)
    end = str(until or now.date().isoformat())[:10]
    start = str(since or (now - timedelta(days=30)).date().isoformat())[:10]
    return start, end


def build_finance_tools(context: Any, accounts: list[dict[str, Any]], *, store: Any = None, pool: Any = None) -> list[Any]:
    """Build the finance tools for every connected institution."""
    banks = [record for record in accounts if record.get("kind") == "bank"]
    if not banks:
        return []

    def _select(connection: str | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if connection is None or not str(connection).strip():
            return banks[0], None
        wanted = str(connection).strip().lower()
        for record in banks:
            transport = record.get("transport") or {}
            if wanted in (str(record.get("display_label") or "").lower(), str(transport.get("institution_name") or "").lower()):
                return record, None
        return None, {"status": "unknown_connection", "error": f"No connected institution named {connection!r}. Connected: {[record.get('display_label') for record in banks]}."}

    async def _ensure_synced(record: dict[str, Any]) -> dict[str, Any] | None:
        if pool is None:
            return {"status": "unavailable", "error": "The transactions store is not available in this process."}
        try:
            from src.anubis.utils.analytics import finance
        except ImportError:
            return {"status": "unavailable", "error": "The finance module is not installed."}
        user_id = str(record.get("user_id") or "")
        try:
            cursor_row = await finance.read_sync_cursor(pool, user_id, record)
        except Exception:
            cursor_row = None
        minimum = int(getattr(context, "finance_sync_min_interval_minutes", None) or 360)
        if finance.should_sync(cursor_row, minimum):
            try:
                await finance.sync_transactions(context, pool, user_id, record)
            except Exception as sync_error:
                logger.info("Plaid sync failed for %s: %s", record.get("display_label"), sync_error)
                return {"status": "sync_failed", "error": f"The latest transactions could not be fetched: {sync_error}. Answering from stored transactions."}
        return None

    @tool
    async def finance_accounts(connection: str | None = None) -> dict[str, Any]:
        """List the connected institutions and their accounts (name, type, last digits)."""
        return {
            "status": "ok",
            "institutions": [
                {
                    "label": record.get("display_label"),
                    "institution": (record.get("transport") or {}).get("institution_name"),
                    "accounts": (record.get("transport") or {}).get("accounts") or [],
                    "environment": (record.get("transport") or {}).get("environment"),
                }
                for record in banks
            ],
        }

    @tool
    async def finance_transactions(since: str | None = None, until: str | None = None, category: str | None = None, connection: str | None = None, limit: int = 100) -> dict[str, Any]:
        """List bank and card transactions in a period (ISO dates; default the last 30 days).

        Use for "what did I pay for", "show my spending on <category>", or to
        find a vendor's charges. Amounts are positive for money out.
        """
        record, error = _select(connection)
        if error:
            return error
        warning = await _ensure_synced(record)
        if warning and warning.get("status") == "unavailable":
            return warning
        from src.anubis.utils.analytics import finance

        start, end = _period(since, until)
        rows = await finance.list_transactions(pool, str(record.get("user_id") or ""), start, end, category=category, limit=max(1, min(int(limit or 100), 500)))
        return {"status": "ok", "connection": record.get("display_label"), "since": start, "until": end, "transactions": rows, "warning": (warning or {}).get("error")}

    @tool
    async def finance_spend_summary(since: str | None = None, until: str | None = None, group_by: str = "category", connection: str | None = None) -> dict[str, Any]:
        """Summarise spending in a period grouped by category, merchant, month, or day.

        Use for "how much did I spend last month", burn rate (group_by month),
        advertising spend (category), and as the input to make_chart. Default
        the last 30 days.
        """
        record, error = _select(connection)
        if error:
            return error
        warning = await _ensure_synced(record)
        if warning and warning.get("status") == "unavailable":
            return warning
        from src.anubis.utils.analytics import finance

        start, end = _period(since, until)
        summary = await finance.spend_by_period(pool, str(record.get("user_id") or ""), start, end, group_by=str(group_by or "category"))
        advertising = await finance.advertising_spend(pool, str(record.get("user_id") or ""), start, end)
        return {"status": "ok", "connection": record.get("display_label"), "since": start, "until": end, "group_by": group_by, "summary": summary, "advertising_spend_usd": advertising, "warning": (warning or {}).get("error")}

    return [finance_accounts, finance_transactions, finance_spend_summary]
