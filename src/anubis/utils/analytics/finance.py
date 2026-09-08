"""Finance: bank and card transactions through Plaid, stored for spend questions.

The owner links a bank or card through the Finance connector (Plaid Link);
the connected-account record carries the encrypted Plaid access token and the
linked accounts. ``sync_transactions`` walks Plaid's ``/transactions/sync``
cursor for that item and upserts every transaction into
``finance_transactions``, so questions such as "how much did we spend on
advertising in August" answer from the stored rows rather than from a fresh
Plaid call each time. ``finance_sync_cursors`` remembers where each
connection's sync left off and when the last sync ran, so a question inside
the minimum interval reads stored rows only.

Plaid's sign convention: a positive ``amount`` is money leaving the account
(an outflow); negative amounts are inflows such as refunds and deposits.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from typing import Any

from src.anubis.utils.postgres_ddl import execute_ddl_script

logger = logging.getLogger(__name__)

TRANSACTIONS_TABLE_NAME = "finance_transactions"
CURSORS_TABLE_NAME = "finance_sync_cursors"

PLAID_PAGE_SIZE = 500
PLAID_SYNC_PATH = "/transactions/sync"
PLAID_ACCOUNTS_PATH = "/accounts/get"

SPEND_GROUPINGS: tuple[str, ...] = ("category", "month", "merchant", "day")

ADVERTISING_CATEGORY_TOKENS: tuple[str, ...] = (
    "advertising",
    "marketing",
    "ads",
)
ADVERTISING_MERCHANT_TOKENS: tuple[str, ...] = (
    "google ads",
    "facebook ads",
    "meta ads",
    "meta platforms",
    "linkedin",
    "twitter ads",
    "x ads",
    "tiktok ads",
    "reddit ads",
    "microsoft advertising",
    "bing ads",
)

_CREATE_TABLES_SQL = f"""
CREATE TABLE IF NOT EXISTS {TRANSACTIONS_TABLE_NAME} (
    transaction_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    connection_key TEXT NOT NULL,
    account_id TEXT,
    date DATE NOT NULL,
    amount DOUBLE PRECISION NOT NULL DEFAULT 0,
    name TEXT,
    merchant TEXT,
    category JSONB NOT NULL DEFAULT '[]',
    personal_finance_category TEXT,
    pending BOOLEAN NOT NULL DEFAULT FALSE,
    raw JSONB NOT NULL DEFAULT '{{}}',
    synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS finance_transactions_user_date_idx
    ON {TRANSACTIONS_TABLE_NAME} (user_id, date);
CREATE TABLE IF NOT EXISTS {CURSORS_TABLE_NAME} (
    user_id TEXT NOT NULL,
    connection_key TEXT NOT NULL,
    cursor TEXT,
    last_synced_at TIMESTAMPTZ,
    last_error TEXT,
    PRIMARY KEY (user_id, connection_key)
);
"""

_UPSERT_TRANSACTION_SQL = f"""
INSERT INTO {TRANSACTIONS_TABLE_NAME}
    (transaction_id, user_id, connection_key, account_id, date, amount, name,
     merchant, category, personal_finance_category, pending, raw, synced_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
ON CONFLICT (transaction_id) DO UPDATE SET
    account_id = EXCLUDED.account_id,
    date = EXCLUDED.date,
    amount = EXCLUDED.amount,
    name = EXCLUDED.name,
    merchant = EXCLUDED.merchant,
    category = EXCLUDED.category,
    personal_finance_category = EXCLUDED.personal_finance_category,
    pending = EXCLUDED.pending,
    raw = EXCLUDED.raw,
    synced_at = now();
"""

_DELETE_TRANSACTION_SQL = f"""
DELETE FROM {TRANSACTIONS_TABLE_NAME} WHERE user_id = %s AND transaction_id = %s;
"""

_SELECT_CURSOR_SQL = f"""
SELECT cursor, last_synced_at, last_error FROM {CURSORS_TABLE_NAME}
WHERE user_id = %s AND connection_key = %s;
"""

_UPSERT_CURSOR_SQL = f"""
INSERT INTO {CURSORS_TABLE_NAME} (user_id, connection_key, cursor, last_synced_at, last_error)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (user_id, connection_key) DO UPDATE SET
    cursor = COALESCE(EXCLUDED.cursor, {CURSORS_TABLE_NAME}.cursor),
    last_synced_at = EXCLUDED.last_synced_at,
    last_error = EXCLUDED.last_error;
"""


async def ensure_finance_tables(pool: Any) -> None:
    """Create the finance tables if absent. Best-effort at boot."""
    try:
        await execute_ddl_script(pool, _CREATE_TABLES_SQL)
    except Exception as table_error:  # noqa: BLE001 - non-fatal at startup
        logger.error("Could not ensure the finance tables exist: %s", table_error)


class PlaidClient:
    """A thin asynchronous client for Plaid's JSON API.

    Every request carries the client id and secret from the context; the
    environment (sandbox or production) selects the host.
    """

    def __init__(self, context: Any, http_client: Any = None) -> None:
        """Bind to the configured Plaid credentials and an optional httpx client."""
        environment = str(getattr(context, "plaid_environment", None) or "sandbox")
        self.base_url = f"https://{environment.strip().lower()}.plaid.com"
        self.client_id = str(getattr(context, "plaid_client_id", None) or "")
        self.secret = str(getattr(context, "plaid_secret", None) or "")
        self._http_client = http_client

    def is_configured(self) -> bool:
        """Whether both Plaid credentials are present."""
        return bool(self.client_id and self.secret)

    async def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST ``body`` (plus the credentials) to ``path`` and return the JSON reply."""
        import httpx

        payload = {"client_id": self.client_id, "secret": self.secret, **body}
        url = f"{self.base_url}{path}"
        if self._http_client is not None:
            response = await self._http_client.post(url, json=payload)
        else:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload)
        if response.status_code >= 400:
            detail = ""
            try:
                error_body = response.json()
                detail = str(
                    error_body.get("error_message") or error_body.get("error_code") or ""
                )
            except Exception:  # noqa: BLE001 - the body may not be JSON
                detail = response.text[:300]
            raise RuntimeError(f"Plaid {path} failed ({response.status_code}): {detail}")
        return response.json()


def _access_token_for(record: dict[str, Any], context: Any) -> str:
    """Decrypt the Plaid access token stored on a connected-account record."""
    from src.anubis.utils.secret_store import decrypt_secret

    encrypted = str(record.get("encrypted_secret") or "")
    if not encrypted:
        raise RuntimeError("The finance connection holds no Plaid access token.")
    plaintext = decrypt_secret(encrypted, context)
    # Some records keep a JSON bundle rather than a bare token.
    if plaintext.strip().startswith("{"):
        import json

        bundle = json.loads(plaintext)
        return str(bundle.get("access_token") or bundle.get("token") or "")
    return plaintext


def _connection_key_of(record: dict[str, Any]) -> str:
    """Return the key that identifies one connection across tables."""
    return str(record.get("connection_key") or record.get("account_key") or "")


def _transaction_date(transaction: dict[str, Any]) -> date:
    """Return the transaction's date (authorised date when the posted date is absent)."""
    raw = transaction.get("date") or transaction.get("authorized_date")
    if isinstance(raw, date):
        return raw
    return date.fromisoformat(str(raw)[:10])


def _category_list(transaction: dict[str, Any]) -> list[str]:
    """Return the legacy category hierarchy as a list of strings."""
    raw = transaction.get("category")
    if isinstance(raw, list):
        return [str(entry) for entry in raw]
    return []


def _personal_finance_category(transaction: dict[str, Any]) -> str | None:
    """Return Plaid's personal finance category (primary or detailed)."""
    raw = transaction.get("personal_finance_category")
    if isinstance(raw, dict):
        return str(raw.get("detailed") or raw.get("primary") or "") or None
    if raw:
        return str(raw)
    return None


async def _read_cursor(pool: Any, user_id: str, connection_key: str) -> dict[str, Any]:
    """Return the stored sync cursor row (empty when none)."""
    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(_SELECT_CURSOR_SQL, (user_id, connection_key))
            row = await cursor.fetchone()
    if not row:
        return {}
    return {"cursor": row[0], "last_synced_at": row[1], "last_error": row[2]}


async def _write_cursor(
    pool: Any,
    user_id: str,
    connection_key: str,
    cursor_value: str | None,
    error: str | None,
) -> None:
    """Record where the sync left off and whether the sync failed."""
    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                _UPSERT_CURSOR_SQL,
                (user_id, connection_key, cursor_value, datetime.now(UTC), error),
            )


async def sync_transactions(
    context: Any,
    pool: Any,
    user_id: str,
    record: dict[str, Any],
    *,
    http_client: Any = None,
) -> dict[str, Any]:
    """Pull every new, changed, and removed transaction for one connection.

    Loops ``/transactions/sync`` from the stored cursor until ``has_more`` is
    false, upserting added and modified rows and deleting removed ones, then
    stores the final cursor. Returns ``{added, modified, removed, cursor}``.
    """
    from psycopg.types.json import Jsonb

    connection_key = _connection_key_of(record)
    client = PlaidClient(context, http_client=http_client)
    if not client.is_configured():
        raise RuntimeError("Plaid is not configured (PLAID_CLIENT_ID / PLAID_SECRET).")
    access_token = _access_token_for(record, context)
    cursor_row = await _read_cursor(pool, user_id, connection_key)
    cursor_value = cursor_row.get("cursor") or ""
    added = modified = removed = 0
    try:
        has_more = True
        while has_more:
            body: dict[str, Any] = {
                "access_token": access_token,
                "count": PLAID_PAGE_SIZE,
            }
            if cursor_value:
                body["cursor"] = cursor_value
            page = await client.post(PLAID_SYNC_PATH, body)
            upserts = [
                *[(transaction, "added") for transaction in page.get("added") or []],
                *[(transaction, "modified") for transaction in page.get("modified") or []],
            ]
            async with pool.connection() as connection:
                async with connection.cursor() as database_cursor:
                    for transaction, change in upserts:
                        await database_cursor.execute(
                            _UPSERT_TRANSACTION_SQL,
                            (
                                str(transaction.get("transaction_id")),
                                user_id,
                                connection_key,
                                transaction.get("account_id"),
                                _transaction_date(transaction),
                                float(transaction.get("amount") or 0.0),
                                transaction.get("name"),
                                transaction.get("merchant_name"),
                                Jsonb(_category_list(transaction)),
                                _personal_finance_category(transaction),
                                bool(transaction.get("pending")),
                                Jsonb(dict(transaction)),
                            ),
                        )
                        if change == "added":
                            added += 1
                        else:
                            modified += 1
                    for removal in page.get("removed") or []:
                        transaction_id = (
                            removal.get("transaction_id")
                            if isinstance(removal, dict)
                            else removal
                        )
                        await database_cursor.execute(
                            _DELETE_TRANSACTION_SQL, (user_id, str(transaction_id))
                        )
                        removed += 1
            cursor_value = str(page.get("next_cursor") or cursor_value)
            has_more = bool(page.get("has_more"))
    except Exception as sync_error:
        await _write_cursor(
            pool, user_id, connection_key, cursor_value or None, str(sync_error)[:400]
        )
        raise
    await _write_cursor(pool, user_id, connection_key, cursor_value or None, None)
    return {
        "added": added,
        "modified": modified,
        "removed": removed,
        "cursor": cursor_value or None,
    }


def should_sync(cursor_row: dict[str, Any] | None, minimum_interval_minutes: int) -> bool:
    """Whether enough time has passed since the last sync to call Plaid again."""
    if not cursor_row:
        return True
    last_synced_at = cursor_row.get("last_synced_at")
    if last_synced_at is None:
        return True
    if isinstance(last_synced_at, str):
        last_synced_at = datetime.fromisoformat(last_synced_at.replace("Z", "+00:00"))
    if last_synced_at.tzinfo is None:
        last_synced_at = last_synced_at.replace(tzinfo=UTC)
    interval = timedelta(minutes=max(0, int(minimum_interval_minutes or 0)))
    return datetime.now(UTC) - last_synced_at >= interval


async def read_sync_cursor(
    pool: Any, user_id: str, record: dict[str, Any]
) -> dict[str, Any]:
    """Return the stored cursor row for one connection (empty when none)."""
    return await _read_cursor(pool, user_id, _connection_key_of(record))


def _period_dates(since: Any, until: Any) -> tuple[date, date]:
    """Return the period as dates, defaulting to the last thirty days."""
    end = until or datetime.now(UTC)
    start = since or (end - timedelta(days=30))
    end_date = end.date() if isinstance(end, datetime) else end
    start_date = start.date() if isinstance(start, datetime) else start
    return start_date, end_date


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


async def _fetchall(pool: Any, sql: str, params: tuple) -> list[tuple]:
    """Run one parameterised query on the pool and return every row."""
    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(sql, params)
            return await cursor.fetchall()


async def spend_by_period(
    pool: Any,
    user_id: str,
    since: Any = None,
    until: Any = None,
    group_by: str = "category",
) -> dict[str, Any]:
    """Sum outflows (amount > 0) by category, month, merchant, or day."""
    start_date, end_date = _period_dates(since, until)
    grouping = str(group_by or "category").strip().lower()
    if grouping not in SPEND_GROUPINGS:
        grouping = "category"
    bucket_sql = {
        "category": "COALESCE(personal_finance_category, category->>0, 'uncategorised')",
        "month": "to_char(date_trunc('month', date), 'YYYY-MM')",
        "merchant": "COALESCE(merchant, name, 'unknown')",
        "day": "date::text",
    }[grouping]
    rows = await _fetchall(
        pool,
        f"""
        SELECT {bucket_sql} AS bucket,
               ROUND(SUM(amount)::numeric, 2) AS spend_usd,
               COUNT(*) AS transactions
        FROM {TRANSACTIONS_TABLE_NAME}
        WHERE user_id = %s AND date >= %s AND date <= %s AND amount > 0 AND NOT pending
        GROUP BY 1
        ORDER BY {"1" if grouping in ("month", "day") else "spend_usd DESC"};
        """,
        (user_id, start_date, end_date),
    )
    total = sum(float(row[1] or 0.0) for row in rows)
    return {
        "columns": [grouping, "spend_usd", "transactions"],
        "rows": [[_plain(value) for value in row] for row in rows],
        "total_spend_usd": round(total, 2),
        "period_start": start_date.isoformat(),
        "period_end": end_date.isoformat(),
        "note": "Outflows only (positive Plaid amounts); pending transactions are excluded.",
    }


async def advertising_spend(
    pool: Any, user_id: str, since: Any = None, until: Any = None
) -> dict[str, Any]:
    """Sum advertising outflows by matching categories and known ad merchants."""
    start_date, end_date = _period_dates(since, until)
    category_patterns = [f"%{token}%" for token in ADVERTISING_CATEGORY_TOKENS]
    merchant_patterns = [f"%{token}%" for token in ADVERTISING_MERCHANT_TOKENS]
    rows = await _fetchall(
        pool,
        f"""
        SELECT COALESCE(merchant, name, 'unknown') AS merchant,
               ROUND(SUM(amount)::numeric, 2) AS spend_usd,
               COUNT(*) AS transactions
        FROM {TRANSACTIONS_TABLE_NAME}
        WHERE user_id = %s AND date >= %s AND date <= %s AND amount > 0 AND NOT pending
          AND (
              lower(COALESCE(personal_finance_category, '')) LIKE ANY(%s)
              OR lower(COALESCE(category::text, '')) LIKE ANY(%s)
              OR lower(COALESCE(merchant, name, '')) LIKE ANY(%s)
          )
        GROUP BY 1 ORDER BY spend_usd DESC;
        """,
        (
            user_id,
            start_date,
            end_date,
            category_patterns,
            category_patterns,
            merchant_patterns,
        ),
    )
    total = sum(float(row[1] or 0.0) for row in rows)
    return {
        "columns": ["merchant", "spend_usd", "transactions"],
        "rows": [[_plain(value) for value in row] for row in rows],
        "advertising_spend_usd": round(total, 2),
        "period_start": start_date.isoformat(),
        "period_end": end_date.isoformat(),
    }


def customer_acquisition_cost(advertising_usd: float, new_users: int) -> dict[str, Any]:
    """Divide advertising spend by new users; a zero denominator returns ``None``."""
    spend = float(advertising_usd or 0.0)
    users = int(new_users or 0)
    if users <= 0:
        return {
            "advertising_spend_usd": round(spend, 2),
            "new_users": users,
            "customer_acquisition_cost_usd": None,
            "note": "No new users in the period, so the cost per new user is undefined.",
        }
    return {
        "advertising_spend_usd": round(spend, 2),
        "new_users": users,
        "customer_acquisition_cost_usd": round(spend / users, 2),
    }


async def accounts_for_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the linked bank accounts a finance connection stored at link time."""
    transport = (record or {}).get("transport") or {}
    accounts = transport.get("accounts")
    if not isinstance(accounts, list):
        return []
    return [
        {
            "account_id": account.get("account_id") or account.get("id"),
            "name": account.get("name") or account.get("official_name"),
            "mask": account.get("mask"),
            "type": account.get("type"),
            "subtype": account.get("subtype"),
            "institution_name": transport.get("institution_name"),
        }
        for account in accounts
        if isinstance(account, dict)
    ]


async def list_transactions(
    pool: Any,
    user_id: str,
    since: Any,
    until: Any,
    *,
    category: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return the owner's transactions in a period, newest first, optionally by category."""
    start, end = _period_dates(since, until)
    sql = (
        "SELECT transaction_id, connection_key, account_id, date, amount, name, merchant, "
        "category, personal_finance_category, pending FROM finance_transactions "
        "WHERE user_id = %s AND date >= %s AND date <= %s"
    )
    params: list[Any] = [user_id, start, end]
    if category:
        sql += " AND (category::text ILIKE %s OR personal_finance_category ILIKE %s)"
        params.extend([f"%{category}%", f"%{category}%"])
    sql += " ORDER BY date DESC, amount DESC LIMIT %s"
    params.append(int(limit))
    rows = await _fetchall(pool, sql, tuple(params))
    columns = [
        "transaction_id", "connection_key", "account_id", "date", "amount", "name",
        "merchant", "category", "personal_finance_category", "pending",
    ]
    return [{column: _plain(value) for column, value in zip(columns, row)} for row in rows]
