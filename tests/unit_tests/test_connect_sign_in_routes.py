"""The connect routes choose the owner's OWN browser, and fall back to the hosted one.

These exercise the route layer: which browser a connect card is sent to, that
the finish and cancel routes serve both, and that a machine going offline
between the check and the call is a reason to open the hosted window rather
than to fail the connection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.anubis.utils import secret_store
from src.anubis.utils.connected_accounts import desktop_login
from src.anubis.utils.connected_accounts import repository as repository_module
from src.anubis.utils.connected_accounts.browser_login import sign_login_token
from src.anubis.utils.connected_accounts.browser_sessions import BrowserSessionError
from src.anubis.utils.connected_accounts.pending_logins import (
    InMemoryPendingLoginRepository,
    set_pending_login_repository,
)
from src.api import webapp as webapp_module

USER_ID = "auth0|owner"
ASSISTANT_ID = "assistant-1"

STORAGE_STATE = {
    "cookies": [{"name": "SID", "value": "session", "domain": ".instagram.com", "path": "/"}],
    "origins": [],
}


def _context(**overrides):
    values = dict(
        connected_account_encryption_key=secret_store.generate_encryption_key(),
        connect_oauth_state_secret="",
        connect_oauth_state_max_age_seconds=600,
        connect_oauth_http_timeout_seconds=5.0,
        connect_oauth_redirect_base_url="http://localhost:9600",
        connect_oauth_popup_target_origins="http://localhost:5173",
        browser_session_login_ttl_seconds=600,
        browser_session_max_concurrent_logins=2,
        max_connected_accounts_per_user=10,
        max_custom_mcp_connectors_per_user=10,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _current_user():
    return {"API_KEY": "sk-test", "identities": [{"user_id": USER_ID}]}


def _json_request(payload, headers=None):
    async def _json():
        return payload

    return SimpleNamespace(
        json=_json,
        headers=headers or {},
        query_params={},
    )


def _device(device_id="device-1", label="Evan's desktop"):
    return SimpleNamespace(
        device_id=device_id,
        device_label=label,
        user_id=USER_ID,
        server_name="neuralnexus",
        allowed_roots=(),
        device_secret="secret",
        platform="linux",
    )


@pytest.fixture
def installed(monkeypatch):
    repository = repository_module.InMemoryConnectedAccountRepository()
    repository_module.set_repository(repository)
    set_pending_login_repository(InMemoryPendingLoginRepository())
    context = _context()
    monkeypatch.setattr(webapp_module, "get_client", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        webapp_module.app, "state", SimpleNamespace(context=context, store=None, graph=None)
    )

    async def _resolve(client, request, user, api_key):
        return {"assistant_id": ASSISTANT_ID}

    monkeypatch.setattr(webapp_module, "_resolve_personal_avatar_for_connection", _resolve)
    desktop_login._desktop_logins.clear()
    yield SimpleNamespace(repository=repository, context=context)
    desktop_login._desktop_logins.clear()
    repository_module.set_repository(None)
    set_pending_login_repository(None)


def _machine_online(monkeypatch, sessions=None):
    online = sessions if sessions is not None else [_device()]
    monkeypatch.setattr(desktop_login, "online_devices", lambda user_id: list(online))
    monkeypatch.setattr(
        "src.anubis.utils.tools.data_analysis.relay.get_session",
        lambda device_id: next((s for s in online if s.device_id == device_id), None),
    )

    async def call_device_tool(session, tool_name, tool_arguments):
        if tool_name == desktop_login.TOOL_OPEN_SIGN_IN_PAGE:
            return {"opened": True, "url": tool_arguments["url"]}
        return {
            "storage_state": STORAGE_STATE,
            "cookie_count": 1,
            "profile": {"identifier": "Google Chrome:Default"},
        }

    monkeypatch.setattr(desktop_login, "call_device_tool", call_device_tool)
    return online


def _hosted_browser_records(monkeypatch, opened):
    """Record any call to the hosted sign-in browser instead of launching one."""

    async def start_login(context, **kwargs):
        opened.append(kwargs)
        return {
            "login_id": "hosted-1",
            "nonce": "nonce-1",
            "view_url": "/connect_account/browser/hosted-1?t=token",
            "expires_in": 600,
            "provider": kwargs["provider"].name,
            "site_url": kwargs.get("site_url"),
        }

    monkeypatch.setattr(
        "src.anubis.utils.connected_accounts.browser_login.start_login", start_login
    )


# ---------------------------------------------------------------------------
# Which browser a connect card is sent to
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_machine_online_means_the_page_opens_in_the_owners_own_browser(
    installed, monkeypatch
):
    _machine_online(monkeypatch)
    opened_hosted: list = []
    _hosted_browser_records(monkeypatch, opened_hosted)
    response = await webapp_module.connect_account_browser_start(
        request=_json_request({"provider": "instagram"}), current_user=_current_user()
    )
    body = json.loads(response.body)
    assert body["login_mode"] == "desktop_browser"
    assert body["device_label"] == "Evan's desktop"
    assert body["login_token"]
    assert opened_hosted == []


@pytest.mark.asyncio
async def test_no_machine_online_falls_back_to_the_hosted_window(installed, monkeypatch):
    monkeypatch.setattr(desktop_login, "online_devices", lambda user_id: [])
    opened_hosted: list = []
    _hosted_browser_records(monkeypatch, opened_hosted)
    response = await webapp_module.connect_account_browser_start(
        request=_json_request({"provider": "instagram"}), current_user=_current_user()
    )
    body = json.loads(response.body)
    assert body["login_mode"] == "browser_session"
    assert body["view_url"].startswith("/connect_account/browser/")
    assert len(opened_hosted) == 1


@pytest.mark.asyncio
async def test_the_card_can_ask_for_the_hosted_window_outright(installed, monkeypatch):
    _machine_online(monkeypatch)
    opened_hosted: list = []
    _hosted_browser_records(monkeypatch, opened_hosted)
    response = await webapp_module.connect_account_browser_start(
        request=_json_request({"provider": "instagram", "use_hosted_browser": True}),
        current_user=_current_user(),
    )
    assert json.loads(response.body)["login_mode"] == "browser_session"
    assert len(opened_hosted) == 1


@pytest.mark.asyncio
async def test_a_machine_that_goes_offline_mid_start_still_connects_through_the_hosted_window(
    installed, monkeypatch
):
    _machine_online(monkeypatch)

    async def call_device_tool(session, tool_name, tool_arguments):
        raise BrowserSessionError(502, "device-1 could not be reached.")

    monkeypatch.setattr(desktop_login, "call_device_tool", call_device_tool)
    opened_hosted: list = []
    _hosted_browser_records(monkeypatch, opened_hosted)
    response = await webapp_module.connect_account_browser_start(
        request=_json_request({"provider": "instagram"}), current_user=_current_user()
    )
    assert json.loads(response.body)["login_mode"] == "browser_session"
    assert len(opened_hosted) == 1


@pytest.mark.asyncio
async def test_a_named_machine_is_the_one_the_tab_opens_on(installed, monkeypatch):
    laptop = _device("device-2", "Evan's laptop")
    _machine_online(monkeypatch, [_device(), laptop])
    response = await webapp_module.connect_account_browser_start(
        request=_json_request({"provider": "instagram", "device_id": "device-2"}),
        current_user=_current_user(),
    )
    assert json.loads(response.body)["device_label"] == "Evan's laptop"


@pytest.mark.asyncio
async def test_the_provider_catalog_says_whether_a_sign_in_would_open_in_your_own_browser(
    installed, monkeypatch
):
    _machine_online(monkeypatch)
    response = await webapp_module.connectable_providers(
        request=SimpleNamespace(), current_user=_current_user()
    )
    body = json.loads(response.body)
    assert body["sign_in_in_your_own_browser"] is True
    assert body["sign_in_device_label"] == "Evan's desktop"
    assert body["providers"]

    monkeypatch.setattr(desktop_login, "online_devices", lambda user_id: [])
    offline = json.loads(
        (
            await webapp_module.connectable_providers(
                request=SimpleNamespace(), current_user=_current_user()
            )
        ).body
    )
    assert offline["sign_in_in_your_own_browser"] is False
    assert offline["sign_in_device_label"] is None


# ---------------------------------------------------------------------------
# Finishing and cancelling, through either browser
# ---------------------------------------------------------------------------


async def _start_in_own_browser(monkeypatch, installed, provider="instagram"):
    _machine_online(monkeypatch)
    response = await webapp_module.connect_account_browser_start(
        request=_json_request({"provider": provider}), current_user=_current_user()
    )
    return json.loads(response.body)


@pytest.mark.asyncio
async def test_finishing_a_sign_in_from_the_owners_browser_stores_the_account(
    installed, monkeypatch
):
    started = await _start_in_own_browser(monkeypatch, installed)
    stored: list = []

    async def _store(user_id, record):
        stored.append((user_id, record))

    monkeypatch.setattr(webapp_module, "_store_connected_record_without_session", _store)
    monkeypatch.setattr(
        webapp_module, "_connected_account_records_without_session", lambda user_id: _empty()
    )
    monkeypatch.setattr(webapp_module, "_after_record_stored", lambda provider, record: None)
    response = await webapp_module.connect_account_browser_finish(
        request=_json_request({}, headers={"X-Login-Token": started["login_token"]}),
        login_id=started["login_id"],
    )
    body = json.loads(response.body)
    assert body["ok"] is True
    assert body["signed_in_on"] == "your own browser"
    assert body["device_label"] == "Evan's desktop"
    assert body["cookie_count"] == 1
    assert stored and stored[0][0] == USER_ID
    assert stored[0][1]["credential_mechanism"] == "browser_session"


async def _empty():
    return []


@pytest.mark.asyncio
async def test_a_failed_import_reaches_the_card_as_a_plain_message(installed, monkeypatch):
    started = await _start_in_own_browser(monkeypatch, installed)

    async def call_device_tool(session, tool_name, tool_arguments):
        if tool_name == desktop_login.TOOL_OPEN_SIGN_IN_PAGE:
            return {"opened": True, "url": tool_arguments["url"]}
        return "Sign in to instagram.com in your browser first."

    monkeypatch.setattr(desktop_login, "call_device_tool", call_device_tool)
    monkeypatch.setattr(
        webapp_module, "_connected_account_records_without_session", lambda user_id: _empty()
    )
    response = await webapp_module.connect_account_browser_finish(
        request=_json_request({}, headers={"X-Login-Token": started["login_token"]}),
        login_id=started["login_id"],
    )
    body = json.loads(response.body)
    assert body["ok"] is False
    assert "Sign in to instagram.com" in body["error"]


@pytest.mark.asyncio
async def test_a_token_for_another_login_is_refused(installed, monkeypatch):
    started = await _start_in_own_browser(monkeypatch, installed)
    other_token = sign_login_token(
        installed.context, login_id="someone-elses", user_id=USER_ID, nonce="n"
    )
    with pytest.raises(webapp_module.HTTPException) as refusal:
        await webapp_module.connect_account_browser_finish(
            request=_json_request({}, headers={"X-Login-Token": other_token}),
            login_id=started["login_id"],
        )
    assert refusal.value.status_code == 401


@pytest.mark.asyncio
async def test_cancelling_forgets_a_sign_in_made_in_the_owners_own_browser(
    installed, monkeypatch
):
    started = await _start_in_own_browser(monkeypatch, installed)
    response = await webapp_module.connect_account_browser_cancel(
        request=_json_request({}, headers={"X-Login-Token": started["login_token"]}),
        login_id=started["login_id"],
    )
    assert json.loads(response.body)["cancelled"] is True
    assert desktop_login.get_desktop_login(started["login_id"]) is None
