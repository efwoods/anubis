"""Custom connectors that demand a login: probe, discovery, registration, exchange."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from src.anubis.utils import secret_store
from src.anubis.utils.connected_accounts import mcp_oauth
from src.anubis.utils.connected_accounts.pending_logins import (
    InMemoryPendingLoginRepository,
)
from src.anubis.utils.connected_accounts.providers import get_provider

SERVER = "https://mcp.example.com/mcp"


def _context():
    return SimpleNamespace(
        connected_account_encryption_key=secret_store.generate_encryption_key(),
        connect_oauth_state_secret="",
        connect_oauth_state_max_age_seconds=600,
        connect_oauth_http_timeout_seconds=5.0,
        connect_oauth_redirect_base_url="http://localhost:9600",
        mcp_oauth_client_name="Neural Nexus",
        mcp_connector_probe_timeout_seconds=2.0,
    )


def _transport(status_for_initialize=200, *, with_metadata=True, registration_status=201):
    seen = []

    def _handle(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, str(request.url)))
        path = request.url.path
        if path == "/mcp" and request.method == "POST":
            body = json.loads(request.content or b"{}")
            if body.get("method") == "initialize":
                headers = {}
                if status_for_initialize == 401 and with_metadata:
                    headers["WWW-Authenticate"] = (
                        'Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource"'
                    )
                return httpx.Response(status_for_initialize, headers=headers, json={"result": {}})
        if path.startswith("/.well-known/oauth-protected-resource"):
            if not with_metadata:
                return httpx.Response(404)
            return httpx.Response(
                200,
                json={
                    "resource": SERVER,
                    "authorization_servers": ["https://auth.example.com"],
                    "scopes_supported": ["mcp:tools"],
                },
            )
        if path.startswith("/.well-known/oauth-authorization-server"):
            return httpx.Response(
                200,
                json={
                    "issuer": "https://auth.example.com",
                    "authorization_endpoint": "https://auth.example.com/authorize",
                    "token_endpoint": "https://auth.example.com/token",
                    "registration_endpoint": "https://auth.example.com/register",
                },
            )
        if path == "/register":
            payload = json.loads(request.content)
            seen.append(("registration", payload))
            return httpx.Response(
                registration_status,
                json={"client_id": "dyn-client", "client_secret": "dyn-secret", "redirect_uris": payload["redirect_uris"]},
            )
        if path == "/token":
            form = dict(httpx.QueryParams(request.content.decode()))
            seen.append(("token", form))
            return httpx.Response(200, json={"access_token": "mcp-access", "refresh_token": "mcp-refresh", "expires_in": 3600})
        return httpx.Response(404)

    return httpx.MockTransport(_handle), seen


@pytest.mark.asyncio
async def test_an_open_server_needs_no_login():
    transport, _ = _transport(200)
    async with httpx.AsyncClient(transport=transport) as client:
        probe = await mcp_oauth.probe_authorization(SERVER, _context(), http_client=client)
    assert probe["status"] == mcp_oauth.AUTHORIZATION_OPEN


@pytest.mark.asyncio
async def test_a_401_with_metadata_needs_oauth_and_without_needs_a_token():
    transport, _ = _transport(401, with_metadata=True)
    async with httpx.AsyncClient(transport=transport) as client:
        probe = await mcp_oauth.probe_authorization(SERVER, _context(), http_client=client)
    assert probe["status"] == mcp_oauth.AUTHORIZATION_NEEDS_OAUTH
    assert probe["resource_metadata_url"].endswith("oauth-protected-resource")

    transport, _ = _transport(401, with_metadata=False)
    async with httpx.AsyncClient(transport=transport) as client:
        probe = await mcp_oauth.probe_authorization(SERVER, _context(), http_client=client)
    assert probe["status"] == mcp_oauth.AUTHORIZATION_NEEDS_TOKEN


@pytest.mark.asyncio
async def test_begin_registers_a_client_and_builds_a_pkce_authorization_url():
    transport, seen = _transport(401)
    repository = InMemoryPendingLoginRepository()
    async with httpx.AsyncClient(transport=transport) as client:
        started = await mcp_oauth.begin_mcp_login(
            _context(),
            user_id="u",
            assistant_id="a",
            provider=get_provider("custom_mcp"),
            server_url=SERVER,
            name="Notes",
            repository=repository,
            http_client=client,
        )
    registration = next(entry[1] for entry in seen if entry[0] == "registration")
    assert registration["client_name"] == "Neural Nexus"
    assert registration["redirect_uris"] == ["http://localhost:9600/connect_account/oauth/callback"]
    url = started["authorization_url"]
    assert url.startswith("https://auth.example.com/authorize?")
    assert "code_challenge=" in url and "code_challenge_method=S256" in url
    assert "resource=https%3A%2F%2Fmcp.example.com%2Fmcp" in url
    assert "client_id=dyn-client" in url
    pending = await repository.peek(started["nonce"])
    assert pending["mode"] == "mcp_oauth"
    assert pending["payload"]["client_id"] == "dyn-client"
    assert "dyn-secret" not in json.dumps(pending["payload"])


@pytest.mark.asyncio
async def test_complete_exchanges_the_code_and_lists_tools(monkeypatch):
    from src.anubis.utils.connected_accounts import mcp_server_tools

    async def _probe(server_url, bearer_token, timeout_seconds):
        assert bearer_token == "mcp-access"
        return [SimpleNamespace(name="search"), SimpleNamespace(name="fetch")]

    monkeypatch.setattr(mcp_server_tools, "probe_server_tools", _probe)
    context = _context()
    transport, seen = _transport(401)
    repository = InMemoryPendingLoginRepository()
    async with httpx.AsyncClient(transport=transport) as client:
        started = await mcp_oauth.begin_mcp_login(
            context, user_id="u", assistant_id="a", provider=get_provider("custom_mcp"),
            server_url=SERVER, name="Notes", repository=repository, http_client=client,
        )
        pending = await repository.consume(started["nonce"])
        record = await mcp_oauth.complete_mcp_login(
            context, code="code-1", pending=pending, existing_records=[], http_client=client
        )
    token_form = next(entry[1] for entry in seen if entry[0] == "token")
    assert token_form["code_verifier"] and token_form["client_secret"] == "dyn-secret"
    assert record["transport"]["auth_type"] == "oauth"
    assert record["transport"]["tool_names"] == ["fetch", "search"]
    assert "mcp-access" not in json.dumps({k: v for k, v in record.items() if k != "encrypted_secret"})

    async with httpx.AsyncClient(transport=transport) as client:
        bearer = await mcp_oauth.bearer_for_record(record, context, http_client=client)
    assert bearer == "mcp-access"


@pytest.mark.asyncio
async def test_registration_refusal_is_reported_plainly():
    transport, _ = _transport(401, registration_status=403)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(mcp_oauth.McpOAuthError) as raised:
            await mcp_oauth.begin_mcp_login(
                _context(), user_id="u", assistant_id="a", provider=get_provider("custom_mcp"),
                server_url=SERVER, name="Notes", repository=InMemoryPendingLoginRepository(),
                http_client=client,
            )
    assert "access token" in raised.value.detail
