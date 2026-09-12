"""Signing in to a site in the OWNER'S OWN browser, through their desktop connector.

The daemon is faked here at the one seam that crosses the machine boundary —
``desktop_login.call_device_tool`` — so the whole flow (which machine is
chosen, what the card is told, what the stored record ends up holding, and
every refusal) is exercised without a relay socket, a browser, or a desktop.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from src.anubis.utils import secret_store
from src.anubis.utils.connected_accounts import desktop_login
from src.anubis.utils.connected_accounts.browser_sessions import (
    BrowserSessionError,
    decrypt_storage_state,
)
from src.anubis.utils.connected_accounts.providers import get_provider
from src.anubis.utils.connected_accounts.store import public_account_view


def _context(**overrides):
    values = dict(
        connected_account_encryption_key=secret_store.generate_encryption_key(),
        connect_oauth_state_secret="",
        connect_oauth_state_max_age_seconds=600,
        connect_oauth_redirect_base_url="http://localhost:9600",
        browser_session_login_ttl_seconds=600,
        browser_session_max_concurrent_logins=2,
        admin_user_id="auth0|admin",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _device(device_id="device-1", label="Evan's desktop", user_id="auth0|evan"):
    """A relay session as ``sessions_for_user`` hands one over."""
    return SimpleNamespace(
        device_id=device_id,
        device_label=label,
        user_id=user_id,
        server_name="neuralnexus",
        allowed_roots=(),
        device_secret="secret",
        platform="linux",
    )


STORAGE_STATE = {
    "cookies": [
        {"name": "SID", "value": "google-session", "domain": ".google.com", "path": "/"},
        {"name": "LSID", "value": "accounts-session", "domain": "accounts.google.com", "path": "/"},
    ],
    "origins": [],
}


@pytest.fixture(autouse=True)
def no_leftover_logins():
    """Each test starts and ends with no sign-in in progress."""
    desktop_login._desktop_logins.clear()
    yield
    desktop_login._desktop_logins.clear()


@pytest.fixture
def one_device_online(monkeypatch):
    """One machine of the owner's is online."""
    session = _device()
    monkeypatch.setattr(desktop_login, "online_devices", lambda user_id: [session])
    return session


def _fake_device_tools(monkeypatch, *, on_open=None, on_import=None, calls=None):
    """Answer the two daemon tools without a relay socket."""

    async def call_device_tool(session, tool_name, tool_arguments):
        if calls is not None:
            calls.append((session.device_id, tool_name, tool_arguments))
        if tool_name == desktop_login.TOOL_OPEN_SIGN_IN_PAGE:
            if callable(on_open):
                return on_open(tool_arguments)
            return {
                "opened": True,
                "url": tool_arguments["url"],
                "site_hostname": "accounts.google.com",
                "opened_with": "xdg-open",
            }
        if tool_name == desktop_login.TOOL_IMPORT_SIGN_IN_SESSION:
            if callable(on_import):
                return on_import(tool_arguments)
            return {
                "site_hostname": "accounts.google.com",
                "registrable_domain": "google.com",
                "storage_state": STORAGE_STATE,
                "cookie_count": 2,
                "profile": {"identifier": "Google Chrome:Default"},
            }
        raise AssertionError(f"unexpected tool {tool_name}")

    monkeypatch.setattr(desktop_login, "call_device_tool", call_device_tool)


def _start(context, provider_name="google_calendar", **overrides):
    arguments = dict(
        user_id="auth0|evan",
        assistant_id="assistant-1",
        provider=get_provider(provider_name),
        site_url="https://accounts.google.com/signin",
        name=None,
        reconnect_account_key=None,
    )
    arguments.update(overrides)
    return asyncio.run(desktop_login.start_desktop_login(context, **arguments))


# ---------------------------------------------------------------------------
# Which machine the page opens on
# ---------------------------------------------------------------------------


def test_no_machine_online_means_the_owners_browser_cannot_be_used(monkeypatch):
    monkeypatch.setattr(desktop_login, "online_devices", lambda user_id: [])
    assert desktop_login.desktop_sign_in_available("auth0|evan") is False
    assert desktop_login.choose_device("auth0|evan") is None


def test_one_machine_online_is_the_one_the_page_opens_on(one_device_online):
    assert desktop_login.desktop_sign_in_available("auth0|evan") is True
    assert desktop_login.choose_device("auth0|evan").device_id == "device-1"


def test_a_named_machine_is_honoured_and_an_unknown_one_is_not_substituted(monkeypatch):
    laptop = _device("device-2", "Evan's laptop")
    monkeypatch.setattr(
        desktop_login, "online_devices", lambda user_id: [_device(), laptop]
    )
    assert desktop_login.choose_device("auth0|evan", "device-2").device_label == "Evan's laptop"
    assert desktop_login.choose_device("auth0|evan", "device-missing") is None


# ---------------------------------------------------------------------------
# Starting the sign-in
# ---------------------------------------------------------------------------


def test_the_sign_in_page_is_opened_in_the_owners_browser_and_the_card_is_told_where(
    monkeypatch, one_device_online
):
    calls: list = []
    _fake_device_tools(monkeypatch, calls=calls)
    started = _start(_context())
    assert calls == [
        (
            "device-1",
            desktop_login.TOOL_OPEN_SIGN_IN_PAGE,
            {"url": "https://accounts.google.com/signin"},
        )
    ]
    assert started["login_mode"] == "desktop_browser"
    assert started["device_label"] == "Evan's desktop"
    assert started["site_hostname"] == "accounts.google.com"
    assert started["login_token"]
    assert "Evan's desktop" in started["instructions"]
    assert desktop_login.get_desktop_login(started["login_id"]) is not None


def test_the_token_the_card_gets_back_verifies_against_that_login(
    monkeypatch, one_device_online
):
    from src.anubis.utils.connected_accounts.browser_login import verify_login_token

    _fake_device_tools(monkeypatch)
    context = _context()
    started = _start(context)
    payload = verify_login_token(context, started["login_token"], started["login_id"])
    assert payload["user_id"] == "auth0|evan"
    assert payload["nonce"] == started["nonce"]
    with pytest.raises(BrowserSessionError):
        verify_login_token(context, started["login_token"], "another-login")


def test_a_site_address_is_required(monkeypatch, one_device_online):
    _fake_device_tools(monkeypatch)
    website = get_provider("website")
    with pytest.raises(BrowserSessionError) as refusal:
        _start(_context(), provider_name="website", site_url="")
    assert refusal.value.status_code == 400
    assert website is not None


def test_no_machine_online_refuses_so_the_caller_can_use_the_hosted_window(monkeypatch):
    monkeypatch.setattr(desktop_login, "online_devices", lambda user_id: [])
    _fake_device_tools(monkeypatch)
    with pytest.raises(BrowserSessionError) as refusal:
        _start(_context())
    assert refusal.value.status_code == 409
    assert "connector" in refusal.value.detail


def test_a_browser_that_refuses_to_open_is_reported_in_the_owners_terms(
    monkeypatch, one_device_online
):
    _fake_device_tools(
        monkeypatch, on_open=lambda arguments: "This machine has no desktop session."
    )
    with pytest.raises(BrowserSessionError) as refusal:
        _start(_context())
    assert refusal.value.status_code == 502
    assert "no desktop session" in refusal.value.detail
    assert not desktop_login._desktop_logins


def test_a_connector_too_old_for_the_sign_in_tools_says_to_update_it(
    monkeypatch, one_device_online
):
    async def call_mcp_filesystem_tool(connection, tool_name, tool_arguments):
        raise RuntimeError(
            f"The Model Context Protocol filesystem server does not expose a tool "
            f"named {tool_name!r}."
        )

    monkeypatch.setattr(
        "src.anubis.utils.tools.data_analysis.mcp_client.call_mcp_filesystem_tool",
        call_mcp_filesystem_tool,
    )
    monkeypatch.setattr(
        "src.anubis.utils.tools.data_analysis.relay.connection_from_session",
        lambda session: SimpleNamespace(url="http://127.0.0.1:8000"),
    )
    with pytest.raises(BrowserSessionError) as refusal:
        _start(_context())
    assert refusal.value.status_code == 409
    assert "too old" in refusal.value.detail


# ---------------------------------------------------------------------------
# Keeping the session the owner just made
# ---------------------------------------------------------------------------


def _finish(context, started, existing_records=None):
    return asyncio.run(
        desktop_login.finish_desktop_login(
            context,
            login_id=started["login_id"],
            user_id="auth0|evan",
            existing_records=existing_records or [],
        )
    )


def test_the_session_comes_back_from_the_owners_browser_and_is_stored_encrypted(
    monkeypatch, one_device_online
):
    _fake_device_tools(monkeypatch)
    monkeypatch.setattr(
        "src.anubis.utils.tools.data_analysis.relay.get_session",
        lambda device_id: one_device_online,
    )
    context = _context()
    started = _start(context)
    finished = _finish(context, started)
    record = finished["record"]
    assert finished["cookie_count"] == 2
    assert finished["device_label"] == "Evan's desktop"
    assert finished["browser_profile"] == "Google Chrome:Default"
    assert record["credential_mechanism"] == "browser_session"
    assert record["user_id"] == "auth0|evan"
    # The cookies are in the record, and they are not readable from it.
    assert "google-session" not in json.dumps(record)
    assert decrypt_storage_state(record, context) == STORAGE_STATE


def test_the_record_says_which_machine_and_which_browser_it_came_from(
    monkeypatch, one_device_online
):
    _fake_device_tools(monkeypatch)
    monkeypatch.setattr(
        "src.anubis.utils.tools.data_analysis.relay.get_session",
        lambda device_id: one_device_online,
    )
    context = _context()
    finished = _finish(context, _start(context))
    transport = finished["record"]["transport"]
    assert transport["captured_from"] == "desktop_browser"
    assert transport["device_label"] == "Evan's desktop"
    assert transport["browser_profile"] == "Google Chrome:Default"
    # The record is filed under the site the avatar will visit afterwards, not
    # under the sign-in page the owner passed through.
    assert transport["hostname"] == "calendar.google.com"
    assert transport["site_url"] == "https://accounts.google.com/signin"


def test_the_public_view_of_the_record_never_carries_the_session(
    monkeypatch, one_device_online
):
    _fake_device_tools(monkeypatch)
    monkeypatch.setattr(
        "src.anubis.utils.tools.data_analysis.relay.get_session",
        lambda device_id: one_device_online,
    )
    context = _context()
    finished = _finish(context, _start(context))
    assert "google-session" not in json.dumps(public_account_view(finished["record"]))


def test_two_sign_ins_to_one_site_become_two_records(monkeypatch, one_device_online):
    _fake_device_tools(monkeypatch)
    monkeypatch.setattr(
        "src.anubis.utils.tools.data_analysis.relay.get_session",
        lambda device_id: one_device_online,
    )
    context = _context()
    first = _finish(context, _start(context))["record"]
    second = _finish(context, _start(context), existing_records=[first])["record"]
    assert first["account_key"] != second["account_key"]
    assert first["display_label"] != second["display_label"]


def test_signing_in_again_refreshes_the_named_record_rather_than_adding_one(
    monkeypatch, one_device_online
):
    _fake_device_tools(monkeypatch)
    monkeypatch.setattr(
        "src.anubis.utils.tools.data_analysis.relay.get_session",
        lambda device_id: one_device_online,
    )
    context = _context()
    first = _finish(context, _start(context))["record"]
    again = _start(context, reconnect_account_key=first["account_key"])
    refreshed = _finish(context, again, existing_records=[first])["record"]
    assert refreshed["account_key"] == first["account_key"]
    assert refreshed["display_label"] == first["display_label"]


def test_a_browser_holding_no_session_for_the_site_says_to_finish_signing_in(
    monkeypatch, one_device_online
):
    _fake_device_tools(
        monkeypatch,
        on_import=lambda arguments: {
            "storage_state": {"cookies": [], "origins": []},
            "cookie_count": 0,
        },
    )
    monkeypatch.setattr(
        "src.anubis.utils.tools.data_analysis.relay.get_session",
        lambda device_id: one_device_online,
    )
    context = _context()
    started = _start(context)
    with pytest.raises(BrowserSessionError) as refusal:
        _finish(context, started)
    assert refusal.value.status_code == 400
    assert "Finish signing in" in refusal.value.detail


def test_a_site_never_signed_in_to_reaches_the_owner_as_the_daemons_own_words(
    monkeypatch, one_device_online
):
    _fake_device_tools(
        monkeypatch,
        on_import=lambda arguments: "Sign in to accounts.google.com in your browser first.",
    )
    monkeypatch.setattr(
        "src.anubis.utils.tools.data_analysis.relay.get_session",
        lambda device_id: one_device_online,
    )
    context = _context()
    started = _start(context)
    with pytest.raises(BrowserSessionError) as refusal:
        _finish(context, started)
    assert refusal.value.status_code == 400
    assert "Sign in to accounts.google.com" in refusal.value.detail


def test_a_machine_that_went_offline_mid_sign_in_says_so(monkeypatch, one_device_online):
    _fake_device_tools(monkeypatch)
    context = _context()
    started = _start(context)
    monkeypatch.setattr(
        "src.anubis.utils.tools.data_analysis.relay.get_session", lambda device_id: None
    )
    with pytest.raises(BrowserSessionError) as refusal:
        _finish(context, started)
    assert refusal.value.status_code == 409
    assert "went offline" in refusal.value.detail


def test_another_account_cannot_finish_someone_elses_sign_in(monkeypatch, one_device_online):
    _fake_device_tools(monkeypatch)
    context = _context()
    started = _start(context)
    with pytest.raises(BrowserSessionError) as refusal:
        asyncio.run(
            desktop_login.finish_desktop_login(
                context,
                login_id=started["login_id"],
                user_id="auth0|someone-else",
                existing_records=[],
            )
        )
    assert refusal.value.status_code == 404


def test_a_sign_in_cannot_be_finished_twice(monkeypatch, one_device_online):
    _fake_device_tools(monkeypatch)
    monkeypatch.setattr(
        "src.anubis.utils.tools.data_analysis.relay.get_session",
        lambda device_id: one_device_online,
    )
    context = _context()
    started = _start(context)
    _finish(context, started)
    with pytest.raises(BrowserSessionError) as refusal:
        _finish(context, started)
    assert refusal.value.status_code == 404


# ---------------------------------------------------------------------------
# Abandoning and expiry
# ---------------------------------------------------------------------------


def test_cancelling_forgets_the_sign_in_and_leaves_the_owners_tab_alone(
    monkeypatch, one_device_online
):
    _fake_device_tools(monkeypatch)
    started = _start(_context())
    assert desktop_login.cancel_desktop_login(started["login_id"], "auth0|evan") is True
    assert desktop_login.get_desktop_login(started["login_id"]) is None
    assert desktop_login.cancel_desktop_login(started["login_id"], "auth0|evan") is False


def test_another_account_cannot_cancel_a_sign_in(monkeypatch, one_device_online):
    _fake_device_tools(monkeypatch)
    started = _start(_context())
    assert desktop_login.cancel_desktop_login(started["login_id"], "auth0|other") is False
    assert desktop_login.get_desktop_login(started["login_id"]) is not None


def test_a_sign_in_nobody_finished_is_forgotten_when_it_expires(
    monkeypatch, one_device_online
):
    _fake_device_tools(monkeypatch)
    started = _start(_context())
    desktop_login.get_desktop_login(started["login_id"]).expires_at = 1.0
    assert desktop_login.reap_expired_desktop_logins() == 1
    assert desktop_login.get_desktop_login(started["login_id"]) is None
