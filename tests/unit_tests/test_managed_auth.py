"""Connecting an account through a certified client, without holding its token.

The vendor's own API is not exercised here — a mock transport stands in — so
what these assert is the part that is ours: that a server with no provider
configured says so plainly, that vendor states are translated rather than
leaked, that a connection is remembered by identifier and never by credential,
and that a drifted endpoint is reported by name instead of surfacing inside a
connect card.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from src.anubis.utils.connected_accounts.managed_auth import (
    STATE_CONNECTED,
    STATE_EXPIRED,
    STATE_FAILED,
    STATE_PENDING,
    ManagedAuthError,
    ManagedAuthNotConfigured,
    get_managed_auth_provider,
    managed_auth_available,
)
from src.anubis.utils.connected_accounts.managed_auth.composio import (
    ComposioProvider,
    toolkit_for_provider,
)

USER_ID = "user-1"


def _context(**overrides):
    settings = {
        "managed_auth_provider": "composio",
        "composio_api_key": "key-123",
        "composio_base_url": "https://vendor.example/api/v3",
        "composio_http_timeout_seconds": 5.0,
    }
    settings.update(overrides)
    return SimpleNamespace(**settings)


def _provider(handler, **overrides):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ComposioProvider(_context(**overrides), http_client=client), client


def _run(provider, client, coroutine_factory):
    async def run():
        try:
            return await coroutine_factory()
        finally:
            await client.aclose()

    return asyncio.run(run())


# -- configuration ---------------------------------------------------------


def test_a_server_with_no_provider_configured_says_so():
    """The ordinary state of a server that needs none. Not an error condition."""
    context = SimpleNamespace(managed_auth_provider=None)

    assert managed_auth_available(context) is False
    with pytest.raises(ManagedAuthNotConfigured) as raised:
        get_managed_auth_provider(context)
    assert "MANAGED_AUTH_PROVIDER" in str(raised.value)


def test_a_provider_named_but_not_implemented_is_refused_by_name():
    context = SimpleNamespace(managed_auth_provider="some-other-vendor")

    with pytest.raises(ManagedAuthNotConfigured) as raised:
        get_managed_auth_provider(context)
    assert "some-other-vendor" in str(raised.value)


def test_a_configured_provider_with_no_key_is_refused_before_any_call():
    context = _context(composio_api_key="")

    with pytest.raises(ManagedAuthNotConfigured) as raised:
        get_managed_auth_provider(context)
    assert "COMPOSIO_API_KEY" in str(raised.value)


def test_the_toolkit_slug_is_known_for_the_providers_that_need_this_path():
    assert toolkit_for_provider("gmail") == "gmail"
    assert toolkit_for_provider("google_calendar") == "googlecalendar"
    # A provider that connects with a password must not be routed here.
    assert toolkit_for_provider("email_account") == ""


# -- auth configuration ----------------------------------------------------


def test_an_existing_configuration_is_reused_and_remembered():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        return httpx.Response(200, json={"items": [{"id": "auth-1"}]})

    provider, client = _provider(handler)

    async def run():
        first = await provider.ensure_auth_config("gmail")
        second = await provider.ensure_auth_config("gmail")
        return first, second

    first, second = _run(provider, client, run)

    assert first == second == "auth-1"
    # Remembered, so a second connection does not re-ask the vendor.
    assert calls == ["GET /api/v3/auth_configs"]


def test_a_configuration_is_created_when_none_exists():
    created: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"items": []})
        created["body"] = request.content.decode("utf-8")
        return httpx.Response(201, json={"auth_config": {"id": "auth-new"}})

    provider, client = _provider(handler)

    identifier = _run(provider, client, lambda: provider.ensure_auth_config("gmail"))

    assert identifier == "auth-new"
    assert '"use_composio_managed_auth"' in str(created["body"])
    assert '"gmail"' in str(created["body"])


# -- connecting ------------------------------------------------------------


def test_starting_a_connection_returns_the_page_the_owner_signs_in_on():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/auth_configs"):
            return httpx.Response(200, json={"items": [{"id": "auth-1"}]})
        assert request.headers["x-api-key"] == "key-123"
        body = request.content.decode("utf-8")
        assert USER_ID in body
        assert "auth-1" in body
        return httpx.Response(
            200,
            json={"id": "conn-1", "redirect_url": "https://accounts.example/consent"},
        )

    provider, client = _provider(handler)

    connection = _run(
        provider,
        client,
        lambda: provider.start_connection(user_id=USER_ID, toolkit="gmail"),
    )

    assert connection.authorization_url == "https://accounts.example/consent"
    assert connection.connection_id == "conn-1"
    assert connection.state == STATE_PENDING
    # Nothing resembling a credential comes back, because none is issued to us.
    assert "token" not in str(connection).lower()


def test_a_vendor_that_returns_no_sign_in_address_is_an_error_not_an_empty_popup():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/auth_configs"):
            return httpx.Response(200, json={"items": [{"id": "auth-1"}]})
        return httpx.Response(200, json={"id": "conn-1"})

    provider, client = _provider(handler)

    with pytest.raises(ManagedAuthError):
        _run(
            provider,
            client,
            lambda: provider.start_connection(user_id=USER_ID, toolkit="gmail"),
        )


@pytest.mark.parametrize(
    "vendor_status,expected",
    [
        ("ACTIVE", STATE_CONNECTED),
        ("INITIATED", STATE_PENDING),
        ("EXPIRED", STATE_EXPIRED),
        ("FAILED", STATE_FAILED),
        ("INACTIVE", STATE_FAILED),
        # An unrecognised state must keep the card waiting rather than declare
        # a failure the owner would have to act on.
        ("SOMETHING_NEW", STATE_PENDING),
    ],
)
def test_vendor_states_are_translated_not_leaked(vendor_status, expected):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "conn-1",
                "status": vendor_status,
                "user_email": "evan@example.com",
            },
        )

    provider, client = _provider(handler)

    connection = _run(provider, client, lambda: provider.connection_state("conn-1"))

    assert connection.state == expected
    assert connection.details["vendor_status"] == vendor_status


def test_a_connected_account_reports_the_address_it_belongs_to():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "connected_account": {
                    "id": "conn-1",
                    "status": "ACTIVE",
                    "data": {"email": "evan@example.com"},
                    "toolkit": {"slug": "gmail"},
                }
            },
        )

    provider, client = _provider(handler)

    connection = _run(provider, client, lambda: provider.connection_state("conn-1"))

    assert connection.is_connected
    assert connection.account_identifier == "evan@example.com"
    assert connection.toolkit == "gmail"


# -- using -----------------------------------------------------------------


def test_a_tool_runs_as_the_owner_against_the_owners_account():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = request.content.decode("utf-8")
        return httpx.Response(200, json={"successful": True, "data": {"messages": []}})

    provider, client = _provider(handler)

    result = _run(
        provider,
        client,
        lambda: provider.execute_tool(
            tool_slug="GMAIL_FETCH_EMAILS",
            user_id=USER_ID,
            arguments={"max_results": 5},
        ),
    )

    assert captured["path"].endswith("/tools/execute/GMAIL_FETCH_EMAILS")
    assert USER_ID in str(captured["body"])
    assert result["successful"] is True


def test_tools_are_described_from_whichever_envelope_the_vendor_used():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "slug": "GMAIL_SEND_EMAIL",
                        "name": "Send email",
                        "description": "Send a message",
                        "input_parameters": {"type": "object"},
                    },
                    {"name": "no slug here"},
                ]
            },
        )

    provider, client = _provider(handler)

    tools = _run(provider, client, lambda: provider.list_tools("gmail"))

    assert [tool["slug"] for tool in tools] == ["GMAIL_SEND_EMAIL", "no slug here"]
    assert tools[0]["description"] == "Send a message"


# -- failures --------------------------------------------------------------


def test_a_rejected_api_key_says_the_key_was_rejected():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "nope"})

    provider, client = _provider(handler)

    with pytest.raises(ManagedAuthError) as raised:
        _run(provider, client, lambda: provider.ensure_auth_config("gmail"))
    assert "rejected the configured API key" in str(raised.value)


def test_an_unreachable_vendor_names_the_address_it_could_not_reach():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    provider, client = _provider(handler)

    with pytest.raises(ManagedAuthError) as raised:
        _run(provider, client, lambda: provider.ensure_auth_config("gmail"))
    assert "vendor.example" in str(raised.value)


def test_verification_names_the_endpoint_that_disagreed():
    """The check written for the moment the API key first arrives.

    Composio's reference spans two API versions that spell some paths
    differently, so a drifted path must be reported by name here rather than
    discovered later inside a connect card.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/tools"):
            return httpx.Response(404, text="no such path")
        return httpx.Response(200, json={"items": []})

    provider, client = _provider(handler)

    report = _run(provider, client, provider.verify_configuration)

    assert report["ok"] is False
    failed = [check for check in report["checks"] if not check["ok"]]
    assert [check["endpoint"] for check in failed] == ["tools"]
    assert "404" in failed[0]["error"]


def test_verification_passes_when_every_endpoint_answers():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": []})

    provider, client = _provider(handler)

    report = _run(provider, client, provider.verify_configuration)

    assert report["ok"] is True
    assert {check["endpoint"] for check in report["checks"]} == {
        "auth_configs",
        "tools",
        "connected_accounts",
    }
