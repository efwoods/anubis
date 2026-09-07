"""The popup routes: start, callback, the acknowledgement turn, the re-armed card."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.anubis.utils import secret_store
from src.anubis.utils.connected_accounts import oauth_flow
from src.anubis.utils.connected_accounts import repository as repository_module
from src.anubis.utils.connected_accounts.pending_logins import (
    InMemoryPendingLoginRepository,
    set_pending_login_repository,
)
from src.api import webapp as webapp_module

USER_ID = "auth0|owner"
ASSISTANT_ID = "assistant-1"


def _context(**overrides):
    values = dict(
        connected_account_encryption_key=secret_store.generate_encryption_key(),
        connect_oauth_state_secret="",
        connect_oauth_state_max_age_seconds=600,
        connect_oauth_http_timeout_seconds=5.0,
        connect_oauth_redirect_base_url="http://localhost:9600",
        connect_oauth_popup_target_origins="http://localhost:5173",
        google_oauth_client_id="google-client",
        google_oauth_client_secret="google-secret",
        max_connected_accounts_per_user=10,
        max_custom_mcp_connectors_per_user=10,
        mailbox_request_timeout_seconds=5.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _current_user():
    return {"API_KEY": "sk-test", "identities": [{"user_id": USER_ID}]}


def _json_request(payload):
    async def _json():
        return payload

    return SimpleNamespace(json=_json)


@pytest.fixture
def installed(monkeypatch):
    repository = repository_module.InMemoryConnectedAccountRepository()
    repository_module.set_repository(repository)
    pending = InMemoryPendingLoginRepository()
    set_pending_login_repository(pending)
    context = _context()
    monkeypatch.setattr(webapp_module, "get_client", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        webapp_module.app, "state", SimpleNamespace(context=context, store=None, graph=None)
    )

    async def _resolve(client, request, user, api_key):
        return {"assistant_id": ASSISTANT_ID}

    monkeypatch.setattr(webapp_module, "_resolve_personal_avatar_for_connection", _resolve)
    yield SimpleNamespace(repository=repository, pending=pending, context=context)
    repository_module.set_repository(None)
    set_pending_login_repository(None)


def _google_transport():
    def _handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == "oauth2.googleapis.com":
            return httpx.Response(200, json={"access_token": "access-1", "refresh_token": "r", "expires_in": 3600})
        if request.url.host == "openidconnect.googleapis.com":
            return httpx.Response(200, json={"email": "evan@example.com"})
        return httpx.Response(404)

    return httpx.MockTransport(_handle)


@pytest.mark.asyncio
async def test_start_then_callback_stores_only_ciphertext_and_posts_a_result(installed, monkeypatch):
    from src.anubis.utils.tools.email import imap_client

    monkeypatch.setattr(imap_client, "verify_credentials", lambda credentials: None)
    started = await webapp_module.connect_account_oauth_start(
        request=_json_request({"provider": "gmail"}), current_user=_current_user()
    )
    body = json.loads(started.body)
    assert body["authorization_url"].startswith("https://accounts.google.com/")
    state = dict(httpx.QueryParams(body["authorization_url"].split("?", 1)[1]))["state"]

    original_complete = oauth_flow.complete_oauth

    async def _complete(context, **kwargs):
        async with httpx.AsyncClient(transport=_google_transport()) as client:
            return await original_complete(context, http_client=client, **kwargs)

    monkeypatch.setattr(oauth_flow, "complete_oauth", _complete)
    response = await webapp_module.connect_account_oauth_callback(
        request=SimpleNamespace(), code="code-1", state=state
    )
    html = response.body.decode()
    assert '"ok": true' in html
    assert '"account_key": "gmail:evan@example.com"' in html
    assert "access-1" not in html
    stored = await installed.repository.list_for_user(USER_ID)
    assert len(stored) == 1
    assert stored[0]["credential_mechanism"] == "oauth"
    assert "access-1" not in json.dumps({k: v for k, v in stored[0].items() if k != "encrypted_secret"})

    # A replayed callback stores nothing more and reports the failure plainly.
    replay = await webapp_module.connect_account_oauth_callback(
        request=SimpleNamespace(), code="code-1", state=state
    )
    assert '"ok": false' in replay.body.decode()
    assert len(await installed.repository.list_for_user(USER_ID)) == 1


@pytest.mark.asyncio
async def test_a_tampered_state_stores_nothing(installed):
    response = await webapp_module.connect_account_oauth_callback(
        request=SimpleNamespace(), code="c", state="bad.state"
    )
    assert '"ok": false' in response.body.decode()
    assert await installed.repository.list_for_user(USER_ID) == []


@pytest.mark.asyncio
async def test_a_refused_consent_renders_a_plain_message(installed):
    started = await webapp_module.connect_account_oauth_start(
        request=_json_request({"provider": "gmail"}), current_user=_current_user()
    )
    state = dict(httpx.QueryParams(json.loads(started.body)["authorization_url"].split("?", 1)[1]))["state"]
    response = await webapp_module.connect_account_oauth_callback(
        request=SimpleNamespace(), state=state, error="access_denied"
    )
    assert "cancelled or refused" in response.body.decode()


@pytest.mark.asyncio
async def test_start_refuses_a_form_provider(installed):
    with pytest.raises(webapp_module.HTTPException) as raised:
        await webapp_module.connect_account_oauth_start(
            request=_json_request({"provider": "website"}), current_user=_current_user()
        )
    assert raised.value.status_code == 400


@pytest.mark.asyncio
async def test_the_generic_connect_route_answers_open_login_popup_for_gmail(installed):
    response = await webapp_module.connect_account_route(
        request=_json_request({"provider": "gmail"}), current_user=_current_user()
    )
    body = json.loads(response.body)
    assert body["connected"] is False
    assert body["action"] == "open_login_popup"
    assert body["login_endpoint"] == "/connect_account/oauth/start"
    assert body["card"]["login_mode"] == "oauth_popup"


@pytest.mark.asyncio
async def test_the_acknowledgement_turn_is_server_authored_and_hidden(installed):
    from src.anubis.utils.connected_accounts.providers import get_provider
    from src.anubis.utils.connected_accounts.store import build_account_record

    record = build_account_record(
        provider=get_provider("github"),
        account_address="evan",
        display_label="evan",
        encrypted_secret="cipher",
        assistant_id=ASSISTANT_ID,
    )
    await installed.repository.upsert(USER_ID, record)
    text, kwargs = await webapp_module._connection_acknowledgement_turn(
        USER_ID, "account:github:evan"
    )
    assert "GitHub" in text and "credential" in text
    assert kwargs["hidden"] is True
    assert kwargs["kind"] == "connection_acknowledgement"
    assert kwargs["connection"]["status"] == "connected"
    assert "cipher" not in json.dumps(kwargs)
    with pytest.raises(webapp_module.HTTPException) as raised:
        await webapp_module._connection_acknowledgement_turn(USER_ID, "account:gmail:nobody")
    assert raised.value.status_code == 404


@pytest.mark.asyncio
async def test_pending_interrupt_is_read_from_the_graph_state(installed, monkeypatch):
    class _Graph:
        async def aget_state(self, config):
            return SimpleNamespace(
                interrupts=[SimpleNamespace(id="i1", value={"kind": "connect_account", "provider": "gmail"})],
                tasks=[],
            )

    webapp_module.app.state.graph = _Graph()
    pending = await webapp_module._pending_interrupt_for_thread("thread-1")
    assert pending == {"thread_id": "thread-1", "interrupt": {"kind": "connect_account", "provider": "gmail"}}

    class _Idle:
        async def aget_state(self, config):
            return SimpleNamespace(interrupts=[], tasks=[])

    webapp_module.app.state.graph = _Idle()
    assert await webapp_module._pending_interrupt_for_thread("thread-1") is None


@pytest.mark.asyncio
async def test_an_unconfigured_vendor_falls_back_to_the_live_browser(installed, monkeypatch):
    from src.anubis.utils.connected_accounts import browser_login

    installed.context.github_oauth_client_id = ""
    installed.context.github_oauth_client_secret = ""
    opened = {}

    async def _fake_start_login(context, *, user_id, assistant_id, provider, site_url=None, name=None, reconnect_account_key=None):
        opened.update({"provider": provider.name, "site_url": site_url, "user_id": user_id})
        return {"login_id": "login-1", "nonce": "n", "view_url": "/connect_account/browser/login-1?t=tok", "expires_in": 600, "provider": provider.name}

    monkeypatch.setattr(browser_login, "start_login", _fake_start_login)
    response = await webapp_module.connect_account_oauth_start(
        request=_json_request({"provider": "github"}), current_user=_current_user()
    )
    body = json.loads(response.body)
    assert body["login_mode"] == "browser_session" and body["fallback"] == "browser_session"
    assert body["view_url"].startswith("/connect_account/browser/")
    assert opened == {"provider": "github", "site_url": "https://github.com/login", "user_id": USER_ID}


def test_a_live_sign_in_record_uses_session_tools_whatever_the_kind():
    from src.anubis.utils.connected_accounts.providers import get_provider
    from src.anubis.utils.connected_accounts.tool_factories import tool_names_for

    record = {"provider": "gmail", "kind": "mailbox", "credential_mechanism": "browser_session"}
    assert "open_connected_site" in tool_names_for(get_provider("gmail"), record)
    assert "search_mailbox" not in " ".join(tool_names_for(get_provider("gmail"), record))


@pytest.mark.asyncio
async def test_a_bank_without_plaid_keys_asks_for_the_banks_website(installed):
    installed.context.plaid_client_id = ""
    installed.context.plaid_secret = ""
    installed.context.plaid_environment = "sandbox"
    response = await webapp_module.connect_account_plaid_link_token(
        request=SimpleNamespace(headers={"content-type": "application/json"}, json=_json_request({"provider": "plaid"}).json),
        current_user=_current_user(),
    )
    body = json.loads(response.body)
    assert body["action"] == "needs_site_url"
    assert body["login_endpoint"] == "/connect_account/browser/start"
    assert [field["name"] for field in body["fields"]] == ["site_url", "name"]
