"""Bank connections through Plaid Link: link token, page, exchange, record."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from src.anubis.utils import secret_store
from src.anubis.utils.connected_accounts import plaid_link
from src.anubis.utils.connected_accounts.pending_logins import (
    InMemoryPendingLoginRepository,
)
from src.anubis.utils.connected_accounts.providers import get_provider
from src.anubis.utils.connected_accounts.store import public_account_view


def _context(**overrides):
    values = dict(
        connected_account_encryption_key=secret_store.generate_encryption_key(),
        connect_oauth_state_secret="",
        connect_oauth_state_max_age_seconds=600,
        connect_oauth_http_timeout_seconds=5.0,
        plaid_client_id="plaid-id",
        plaid_secret="plaid-secret",
        plaid_environment="sandbox",
        plaid_products="transactions",
        plaid_country_codes="US",
        mcp_oauth_client_name="Neural Nexus",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _transport(seen):
    def _handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((request.url.path, body))
        assert request.url.host == "sandbox.plaid.com"
        assert body["client_id"] == "plaid-id" and body["secret"] == "plaid-secret"
        if request.url.path == "/link/token/create":
            return httpx.Response(200, json={"link_token": "link-sandbox-1"})
        if request.url.path == "/item/public_token/exchange":
            return httpx.Response(200, json={"access_token": "access-sandbox-1", "item_id": "item-1"})
        if request.url.path == "/accounts/get":
            return httpx.Response(200, json={"item": {"institution_id": "ins_1"}, "accounts": [{"account_id": "a1", "name": "Checking", "mask": "0000", "type": "depository", "subtype": "checking"}, {"account_id": "a2", "name": "Savings", "mask": "1111", "type": "depository", "subtype": "savings"}]})
        if request.url.path == "/institutions/get_by_id":
            return httpx.Response(200, json={"institution": {"name": "Bank of America"}})
        return httpx.Response(404)

    return httpx.MockTransport(_handle)


@pytest.mark.asyncio
async def test_start_creates_a_link_token_and_a_signed_page_address():
    seen = []
    repository = InMemoryPendingLoginRepository()
    async with httpx.AsyncClient(transport=_transport(seen)) as client:
        started = await plaid_link.start_plaid_link(
            _context(), user_id="u", assistant_id="a", provider=get_provider("plaid"),
            repository=repository, http_client=client,
        )
    assert started["link_url"].startswith("/connect_account/plaid/link?t=")
    create_body = next(body for path, body in seen if path == "/link/token/create")
    assert create_body["products"] == ["transactions"] and create_body["country_codes"] == ["US"]
    pending = await repository.peek(started["nonce"])
    assert pending["payload"]["link_token"] == "link-sandbox-1"


@pytest.mark.asyncio
async def test_exchange_stores_the_encrypted_access_token_and_the_accounts():
    context = _context()
    seen = []
    pending = {"user_id": "u", "assistant_id": "a", "provider": "plaid"}
    async with httpx.AsyncClient(transport=_transport(seen)) as client:
        record = await plaid_link.exchange_public_token(
            context, public_token="public-1", pending=pending, existing_records=[], http_client=client
        )
    assert record["account_key"] == "plaid:item-1"
    assert record["display_label"] == "Bank of America"
    assert record["transport"]["institution_name"] == "Bank of America"
    assert len(record["transport"]["accounts"]) == 2
    assert secret_store.decrypt_secret(record["encrypted_secret"], context) == "access-sandbox-1"
    view = public_account_view(record)
    assert view["account_count"] == 2 and view["institution_name"] == "Bank of America"
    assert "access-sandbox-1" not in json.dumps(view)


@pytest.mark.asyncio
async def test_missing_plaid_keys_answer_503():
    with pytest.raises(plaid_link.PlaidLinkError) as raised:
        await plaid_link.start_plaid_link(
            _context(plaid_secret=""), user_id="u", assistant_id="a", provider=get_provider("plaid"),
            repository=InMemoryPendingLoginRepository(),
        )
    assert raised.value.status_code == 503


def test_the_link_page_loads_plaid_and_never_holds_the_secret():
    html = plaid_link.render_link_page_html(
        link_token="link-sandbox-1", nonce="n", login_token="tok", allowed_origins=["http://localhost:5173"]
    )
    assert plaid_link.PLAID_LINK_SCRIPT_URL in html
    assert "link-sandbox-1" in html and "plaid-secret" not in html
    assert "/connect_account/plaid/exchange" in html
    assert "neural-nexus:login-result" in html
