"""Vendors reached with the key they issue, instead of a scraped session.

Anthropic, OpenAI, and LangSmith publish no OAuth for third-party applications
but each hands the owner a personal API key. These assert the two things that
make that trustworthy: the key is proved before anything is stored, and a read
that needs an administrator key says so by name rather than returning an empty
report that would read as "you spent nothing".
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from src.anubis.utils import secret_store
from src.anubis.utils.connected_accounts import get_provider
from src.anubis.utils.connected_accounts.connect_handlers import (
    ConnectRefused,
    ConnectRequest,
    connect_account,
)
from src.anubis.utils.connected_accounts.tool_factories import tool_names_for
from src.anubis.utils.connected_accounts.vendor_key_tools import (
    VendorKeyRejected,
    VendorUnreachable,
    build_vendor_key_tools,
    verify_api_key,
)

KEY = "sk-ant-secret-value-1234"


def _context():
    return SimpleNamespace(
        connected_account_encryption_key=secret_store.generate_encryption_key()
    )


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _record(context, provider="anthropic", key=KEY):
    return {
        "account_key": f"{provider}:1234",
        "provider": provider,
        "kind": "analytics",
        "credential_mechanism": "api_key",
        "account_address": f"{provider}:1234",
        "display_label": provider.title(),
        "encrypted_secret": secret_store.encrypt_secret(key, context),
    }


# -- proving the key -------------------------------------------------------


@pytest.mark.parametrize(
    "provider,header,value",
    [
        ("anthropic", "x-api-key", KEY),
        ("openai", "authorization", f"Bearer {KEY}"),
        ("langsmith", "x-api-key", KEY),
    ],
)
def test_each_vendor_is_asked_the_way_it_expects(provider, header, value):
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update({key.lower(): item for key, item in request.headers.items()})
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"data": []})

    async def run():
        async with _client(handler) as client:
            return await verify_api_key(provider, KEY, http_client=client)

    assert asyncio.run(run())["status"] == "ok"
    assert seen[header] == value
    if provider == "anthropic":
        # Anthropic refuses a request that does not name the API version.
        assert seen["anthropic-version"]


def test_a_refused_key_names_the_page_that_issues_one():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid"})

    async def run():
        async with _client(handler) as client:
            return await verify_api_key("anthropic", "wrong", http_client=client)

    with pytest.raises(VendorKeyRejected) as raised:
        asyncio.run(run())
    assert "console.anthropic.com" in str(raised.value)


def test_an_unreachable_vendor_is_not_reported_as_a_bad_key():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    async def run():
        async with _client(handler) as client:
            return await verify_api_key("openai", KEY, http_client=client)

    with pytest.raises(VendorUnreachable):
        asyncio.run(run())


# -- connecting ------------------------------------------------------------


def test_connecting_proves_the_key_and_stores_only_ciphertext(monkeypatch):
    context = _context()

    async def _verify(provider, key, **kwargs):
        assert key == KEY
        return {"status": "ok"}

    import src.anubis.utils.connected_accounts.vendor_key_tools as vendor_key_tools

    monkeypatch.setattr(vendor_key_tools, "verify_api_key", _verify)

    record = asyncio.run(
        connect_account(
            ConnectRequest(
                provider=get_provider("anthropic"),
                fields={"api_key": KEY},
                assistant_id="assistant-1",
                context=context,
            )
        )
    )

    assert record["provider"] == "anthropic"
    assert record["credential_mechanism"] == "api_key"
    assert KEY not in str(record)
    assert secret_store.decrypt_secret(record["encrypted_secret"], context) == KEY
    # The last four characters tell two keys apart without showing either.
    assert record["account_address"].endswith("1234")


def test_a_rejected_key_stores_nothing(monkeypatch):
    import src.anubis.utils.connected_accounts.vendor_key_tools as vendor_key_tools

    async def _reject(provider, key, **kwargs):
        raise VendorKeyRejected("That key was refused. Copy a current key from X.")

    monkeypatch.setattr(vendor_key_tools, "verify_api_key", _reject)

    with pytest.raises(ConnectRefused) as raised:
        asyncio.run(
            connect_account(
                ConnectRequest(
                    provider=get_provider("openai"),
                    fields={"api_key": "wrong"},
                    assistant_id="assistant-1",
                    context=_context(),
                )
            )
        )
    assert raised.value.status_code == 400
    assert "refused" in raised.value.detail


def test_connecting_without_a_key_says_where_to_get_one():
    with pytest.raises(ConnectRefused) as raised:
        asyncio.run(
            connect_account(
                ConnectRequest(
                    provider=get_provider("langsmith"),
                    fields={},
                    assistant_id="assistant-1",
                    context=_context(),
                )
            )
        )
    assert "smith.langchain.com" in raised.value.detail


def test_the_card_advertises_the_tools_before_anything_is_connected():
    assert tool_names_for(get_provider("anthropic")) == [
        "anthropic_usage",
        "anthropic_models",
    ]
    assert tool_names_for(get_provider("langsmith")) == [
        "langsmith_projects",
        "langsmith_runs",
    ]


# -- using -----------------------------------------------------------------


def _tools(context, provider="anthropic"):
    built = build_vendor_key_tools(context, [_record(context, provider)])
    return {tool.name: tool for tool in built}


def test_a_spend_read_that_needs_an_administrator_key_says_so(monkeypatch):
    """An empty report here would read as 'you spent nothing', which is worse."""
    context = _context()

    class _Response:
        status_code = 401
        text = "admin key required"

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None, params=None):
            return _Response()

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: _Client())

    result = asyncio.run(_tools(context)["anthropic_usage"].coroutine())

    assert result["status"] == "needs_admin_key"
    assert "administrator key" in result["error"]
    assert "admin-keys" in result["error"]


def test_a_reachable_read_returns_what_the_vendor_said(monkeypatch):
    context = _context()

    class _Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"data": [{"id": "claude-opus-5"}, {"id": "claude-sonnet-5"}]}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None, params=None):
            assert headers["x-api-key"] == KEY
            return _Response()

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: _Client())

    result = asyncio.run(_tools(context)["anthropic_models"].coroutine())

    assert result["status"] == "ok"
    assert result["models"] == ["claude-opus-5", "claude-sonnet-5"]


def test_an_unreachable_vendor_returns_a_sentence_rather_than_raising(monkeypatch):
    context = _context()

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None, params=None):
            raise RuntimeError("connection reset")

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: _Client())

    result = asyncio.run(_tools(context)["anthropic_models"].coroutine())

    assert result["status"] == "unreachable"
    assert "connection reset" in result["error"]
