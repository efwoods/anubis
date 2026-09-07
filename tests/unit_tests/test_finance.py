"""Finance: Plaid sync over a mocked transport, stored cursors, and spend maths."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest

from src.anubis.utils.analytics import finance
from src.anubis.utils.analytics.finance import (
    PlaidClient,
    accounts_for_record,
    customer_acquisition_cost,
    should_sync,
    sync_transactions,
)
from src.anubis.utils.postgres_ddl import split_sql_statements


class _FakeCursor:
    def __init__(self, pool):
        self.pool = pool
        self.rowcount = 1

    async def execute(self, statement, params=None, *, prepare=None):
        self.pool.calls.append((statement.strip(), params))

    async def fetchall(self):
        return list(self.pool.rows)

    async def fetchone(self):
        return self.pool.cursor_row

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeConnection:
    def __init__(self, pool):
        self.pool = pool

    def cursor(self):
        return _FakeCursor(self.pool)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self, rows=None, cursor_row=None):
        self.calls = []
        self.rows = rows or []
        self.cursor_row = cursor_row

    def connection(self):
        return _FakeConnection(self)


def _context():
    return SimpleNamespace(
        plaid_client_id="client", plaid_secret="secret", plaid_environment="sandbox"
    )


def test_ddl_creates_both_tables():
    statements = split_sql_statements(finance._CREATE_TABLES_SQL)
    assert any("finance_transactions" in statement for statement in statements)
    assert any("finance_sync_cursors" in statement for statement in statements)
    assert any("(user_id, date)" in statement for statement in statements)


def test_plaid_client_base_url_follows_the_environment():
    assert PlaidClient(_context()).base_url == "https://sandbox.plaid.com"
    assert PlaidClient(SimpleNamespace(plaid_environment="Production")).base_url == (
        "https://production.plaid.com"
    )
    assert PlaidClient(SimpleNamespace()).is_configured() is False


@pytest.mark.asyncio
async def test_sync_transactions_walks_two_pages_and_stores_the_cursor(monkeypatch):
    monkeypatch.setattr(
        "src.anubis.utils.secret_store.decrypt_secret",
        lambda ciphertext, context: "access-sandbox-token",
    )
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if body.get("cursor") in (None, ""):
            return httpx.Response(
                200,
                json={
                    "added": [
                        {"transaction_id": "t1", "account_id": "acc", "date": "2026-09-01",
                         "amount": 12.5, "name": "Google Ads", "merchant_name": "Google Ads",
                         "category": ["Advertising"], "pending": False,
                         "personal_finance_category": {"primary": "GENERAL_SERVICES",
                                                       "detailed": "GENERAL_SERVICES_ADVERTISING"}},
                    ],
                    "modified": [],
                    "removed": [],
                    "next_cursor": "cursor-page-1",
                    "has_more": True,
                },
            )
        return httpx.Response(
            200,
            json={
                "added": [
                    {"transaction_id": "t2", "account_id": "acc", "date": "2026-09-02",
                     "amount": -50.0, "name": "Refund", "pending": True},
                ],
                "modified": [
                    {"transaction_id": "t1", "account_id": "acc", "date": "2026-09-01",
                     "amount": 13.0, "name": "Google Ads"},
                ],
                "removed": [{"transaction_id": "t0"}],
                "next_cursor": "cursor-final",
                "has_more": False,
            },
        )

    transport = httpx.MockTransport(handler)
    pool = _FakePool(cursor_row=None)
    async with httpx.AsyncClient(transport=transport) as http_client:
        result = await sync_transactions(
            _context(),
            pool,
            "owner",
            {"connection_key": "plaid:item-1", "encrypted_secret": "ciphertext"},
            http_client=http_client,
        )
    assert result == {"added": 2, "modified": 1, "removed": 1, "cursor": "cursor-final"}
    assert [path for path, _ in requests] == ["/transactions/sync", "/transactions/sync"]
    assert requests[0][1]["client_id"] == "client"
    assert requests[0][1]["access_token"] == "access-sandbox-token"
    assert "cursor" not in requests[0][1]
    assert requests[1][1]["cursor"] == "cursor-page-1"

    upserts = [call for call in pool.calls if call[0].startswith("INSERT INTO finance_transactions")]
    deletes = [call for call in pool.calls if call[0].startswith("DELETE FROM finance_transactions")]
    cursor_writes = [call for call in pool.calls if call[0].startswith("INSERT INTO finance_sync_cursors")]
    assert len(upserts) == 3
    assert upserts[0][1][0] == "t1"
    assert upserts[0][1][9] == "GENERAL_SERVICES_ADVERTISING"
    assert deletes[0][1] == ("owner", "t0")
    assert cursor_writes[-1][1][:3] == ("owner", "plaid:item-1", "cursor-final")
    assert cursor_writes[-1][1][4] is None


@pytest.mark.asyncio
async def test_sync_transactions_records_the_error_on_failure(monkeypatch):
    monkeypatch.setattr(
        "src.anubis.utils.secret_store.decrypt_secret",
        lambda ciphertext, context: json.dumps({"access_token": "bundled"}),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error_code": "INVALID_ACCESS_TOKEN", "error_message": "bad token"})

    pool = _FakePool(cursor_row=("old-cursor", None, None))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(RuntimeError, match="bad token"):
            await sync_transactions(
                _context(), pool, "owner",
                {"account_key": "plaid:item-2", "encrypted_secret": "c"},
                http_client=http_client,
            )
    cursor_writes = [call for call in pool.calls if call[0].startswith("INSERT INTO finance_sync_cursors")]
    assert cursor_writes[-1][1][2] == "old-cursor"
    assert "bad token" in cursor_writes[-1][1][4]


def test_should_sync_respects_the_minimum_interval():
    assert should_sync(None, 360) is True
    assert should_sync({"last_synced_at": None}, 360) is True
    recent = datetime.now(UTC) - timedelta(minutes=5)
    assert should_sync({"last_synced_at": recent}, 360) is False
    assert should_sync({"last_synced_at": recent.isoformat()}, 1) is True
    old = datetime.now(UTC) - timedelta(hours=7)
    assert should_sync({"last_synced_at": old}, 360) is True


def test_customer_acquisition_cost_handles_zero_users():
    empty = customer_acquisition_cost(120.0, 0)
    assert empty["customer_acquisition_cost_usd"] is None
    assert "undefined" in empty["note"]
    assert customer_acquisition_cost(120.0, 4)["customer_acquisition_cost_usd"] == 30.0


@pytest.mark.asyncio
async def test_spend_by_period_and_advertising_use_parameters_only():
    pool = _FakePool(rows=[("Advertising", 40.0, 2), ("Software", 10.0, 1)])
    since = datetime(2026, 8, 1, tzinfo=UTC)
    until = datetime(2026, 8, 31, tzinfo=UTC)
    result = await finance.spend_by_period(pool, "owner", since, until, group_by="category")
    statement, params = pool.calls[-1]
    assert "amount > 0 AND NOT pending" in statement
    assert params[0] == "owner"
    assert result["total_spend_usd"] == 50.0
    assert result["columns"][0] == "category"

    bad_grouping = await finance.spend_by_period(pool, "owner", since, until, group_by="drop table")
    assert bad_grouping["columns"][0] == "category"

    pool.rows = [("Google Ads", 40.0, 2)]
    advertising = await finance.advertising_spend(pool, "owner", since, until)
    statement, params = pool.calls[-1]
    assert "LIKE ANY(%s)" in statement
    assert "%advertising%" in params[3]
    assert "%google ads%" in params[5]
    assert advertising["advertising_spend_usd"] == 40.0


@pytest.mark.asyncio
async def test_accounts_for_record_reads_the_transport():
    record = {
        "transport": {
            "institution_name": "Chase",
            "accounts": [{"account_id": "a1", "name": "Checking", "mask": "1234", "type": "depository"}],
        }
    }
    accounts = await accounts_for_record(record)
    assert accounts == [
        {"account_id": "a1", "name": "Checking", "mask": "1234", "type": "depository",
         "subtype": None, "institution_name": "Chase"}
    ]
    assert await accounts_for_record({}) == []
