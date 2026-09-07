"""Durable browser sessions and the live sign-in window."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.anubis.utils import secret_store
from src.anubis.utils.connected_accounts import browser_login, browser_sessions
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
        browser_session_max_open=6,
        browser_session_idle_seconds=900,
        browser_session_frame_interval_ms=250,
        admin_user_id="auth0|admin",
        browser_chromium_executable_path=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


# --------------------------------------------------------------------------
# Login-page detection and site matching
# --------------------------------------------------------------------------


def test_login_pages_are_recognised_by_address_or_by_form():
    assert browser_sessions.login_page_detected("https://x.test/login", "")
    assert browser_sessions.login_page_detected("https://accounts.google.com/v3/signin", "", site_hostname="x.test")
    assert browser_sessions.login_page_detected("https://x.test/", "<input type='password'>")
    assert not browser_sessions.login_page_detected(
        "https://x.test/dashboard", "<a href='/logout'>Log out</a><input type='password'>"
    )
    assert not browser_sessions.login_page_detected("https://x.test/home", "<h1>Welcome back</h1>")


def test_same_site_allows_subdomains_only():
    assert browser_sessions.same_site("https://api.x.test/a", "x.test")
    assert browser_sessions.same_site("https://x.test/a", "x.test")
    assert not browser_sessions.same_site("https://evil.test/x.test", "x.test")


def test_session_transport_is_encrypted_and_hidden_from_the_public_view():
    context = _context()
    transport = browser_sessions.build_session_transport(
        storage_state={"cookies": [{"name": "sid", "value": "secret-cookie"}], "origins": []},
        context=context,
        home_url="https://smith.langchain.com/",
        final_url="https://smith.langchain.com/projects",
        site_url="https://smith.langchain.com/",
        heuristic_signed_in=True,
        recipe_key="langsmith",
    )
    record = {
        "account_key": "langsmith:smith.langchain.com",
        "provider": "langsmith",
        "kind": "analytics",
        "credential_mechanism": "browser_session",
        "account_address": "smith.langchain.com",
        "display_label": "LangSmith",
        "status": "connected",
        "transport": transport,
    }
    view = public_account_view(record)
    assert "secret-cookie" not in json.dumps(view)
    assert view["session_saved_at"]
    assert browser_sessions.decrypt_storage_state(record, context)["cookies"][0]["value"] == "secret-cookie"
    assert browser_sessions.home_url_for(record, get_provider("langsmith")) == "https://smith.langchain.com/"


# --------------------------------------------------------------------------
# The live login with a fake browser
# --------------------------------------------------------------------------


class _FakePage:
    def __init__(self, url="https://smith.langchain.com/login"):
        self.url = url
        self.html = "<input type='password'>"
        self.viewport_size = {"width": 1280, "height": 800}
        self.mouse = SimpleNamespace(actions=[])
        self.keyboard = SimpleNamespace(actions=[])
        for name in ("move", "down", "up", "click", "wheel"):
            setattr(self.mouse, name, self._recorder(self.mouse.actions, name))
        for name in ("down", "up", "press", "insert_text"):
            setattr(self.keyboard, name, self._recorder(self.keyboard.actions, name))

    @staticmethod
    def _recorder(store, name):
        async def _record(*args, **kwargs):
            store.append((name, args, kwargs))

        return _record

    async def goto(self, url, **kwargs):
        self.url = url

    async def content(self):
        return self.html

    async def set_viewport_size(self, size):
        self.viewport_size = dict(size)


class _FakeContext:
    def __init__(self):
        self.page = _FakePage()
        self.closed = False

    async def new_page(self):
        return self.page

    async def storage_state(self):
        return {"cookies": [{"name": "sid", "value": "cookie-value"}], "origins": []}

    async def close(self):
        self.closed = True


@pytest.fixture
def fake_browser(monkeypatch):
    contexts = []

    async def _new_context(context, *, storage_state=None, user_agent=None):
        fake = _FakeContext()
        contexts.append(fake)
        return fake

    monkeypatch.setattr(browser_sessions, "new_context", _new_context)
    monkeypatch.setattr(browser_login, "new_context", _new_context)
    browser_login._live_logins.clear()
    yield contexts
    browser_login._live_logins.clear()


@pytest.mark.asyncio
async def test_start_and_finish_store_an_encrypted_session(fake_browser):
    context = _context()
    started = await browser_login.start_login(
        context, user_id="auth0|owner", assistant_id="a", provider=get_provider("langsmith")
    )
    assert started["view_url"].startswith(f"/connect_account/browser/{started['login_id']}?t=")
    token = started["view_url"].split("?t=", 1)[1]
    payload = browser_login.verify_login_token(context, token, started["login_id"])
    assert payload["user_id"] == "auth0|owner"
    with pytest.raises(browser_sessions.BrowserSessionError):
        browser_login.verify_login_token(context, token, "another-login")

    login = browser_login.get_live_login(started["login_id"])
    login.page.url = "https://smith.langchain.com/projects"
    login.page.html = "<a href='/logout'>Log out</a>"
    finished = await browser_login.finish_login(
        context, login_id=started["login_id"], user_id="auth0|owner", existing_records=[]
    )
    record = finished["record"]
    assert finished["heuristic_signed_in"] is True
    assert record["account_key"].startswith("langsmith:smith.langchain.com#")
    assert record["credential_mechanism"] == "browser_session"
    assert "cookie-value" not in json.dumps(public_account_view(record))
    assert browser_sessions.decrypt_storage_state(record, context)["cookies"][0]["value"] == "cookie-value"
    assert fake_browser[0].closed is True
    assert browser_login.get_live_login(started["login_id"]) is None
    with pytest.raises(browser_sessions.BrowserSessionError):
        await browser_login.finish_login(
            context, login_id=started["login_id"], user_id="auth0|owner", existing_records=[]
        )


@pytest.mark.asyncio
async def test_logins_are_capped_and_a_custom_site_needs_an_address(fake_browser):
    context = _context()
    await browser_login.start_login(context, user_id="u", assistant_id="a", provider=get_provider("openai"))
    await browser_login.start_login(context, user_id="u", assistant_id="a", provider=get_provider("anthropic"))
    with pytest.raises(browser_sessions.BrowserSessionError) as raised:
        await browser_login.start_login(context, user_id="u", assistant_id="a", provider=get_provider("langsmith"))
    assert raised.value.status_code == 409
    browser_login._live_logins.clear()
    with pytest.raises(browser_sessions.BrowserSessionError) as missing:
        await browser_login.start_login(context, user_id="u", assistant_id="a", provider=get_provider("custom_site"))
    assert missing.value.status_code == 400


@pytest.mark.asyncio
async def test_input_is_forwarded_and_never_logged(fake_browser, caplog):
    context = _context()
    started = await browser_login.start_login(context, user_id="u", assistant_id="a", provider=get_provider("openai"))
    login = browser_login.get_live_login(started["login_id"])
    viewport = {"width": 1280, "height": 800}
    caplog.set_level("DEBUG")
    await browser_login.dispatch_input(login, {"type": "mouse", "action": "click", "x": 10, "y": 20}, viewport)
    await browser_login.dispatch_input(login, {"type": "text", "text": "hunter2-password"}, viewport)
    await browser_login.dispatch_input(login, {"type": "key", "action": "press", "key": "Enter"}, viewport)
    await browser_login.dispatch_input(login, {"type": "navigate", "url": "https://evil.test/"}, viewport)
    assert ("click", (10.0, 20.0), {"button": "left"}) in login.page.mouse.actions
    assert ("insert_text", ("hunter2-password",), {}) in login.page.keyboard.actions
    assert login.page.url == "https://platform.openai.com/login"
    assert "hunter2-password" not in caplog.text


@pytest.mark.asyncio
async def test_keepalive_flags_a_lapsed_session_and_notifies(fake_browser, monkeypatch):
    context = _context()
    transport = browser_sessions.build_session_transport(
        storage_state={"cookies": [], "origins": []}, context=context,
        home_url="https://platform.openai.com/usage", final_url="https://platform.openai.com/usage",
        site_url="https://platform.openai.com/", heuristic_signed_in=True,
    )
    record = {
        "account_key": "openai:platform.openai.com", "provider": "openai", "kind": "analytics",
        "credential_mechanism": "browser_session", "account_address": "platform.openai.com",
        "display_label": "OpenAI", "status": "connected", "assistant_id": "a", "transport": transport,
    }
    flagged = []
    notified = []

    async def _mark(store, user_id, key):
        flagged.append(key)

    async def _notify(record_, user_id, reason):
        notified.append(reason)

    monkeypatch.setattr("src.anubis.utils.connected_accounts.store.mark_account_needs_reconnect", _mark)
    monkeypatch.setattr(browser_sessions, "_notify_reconnect", _notify)

    original_new_context = browser_sessions.new_context

    async def _new_context_redirecting(context_, *, storage_state=None, user_agent=None):
        fake = await original_new_context(context_, storage_state=storage_state, user_agent=user_agent)

        async def _goto(url, **kwargs):
            fake.page.url = "https://auth.openai.com/log-in"

        fake.page.goto = _goto
        return fake

    monkeypatch.setattr(browser_sessions, "new_context", _new_context_redirecting)
    result = await browser_sessions.keepalive_record(context, None, record, user_id="u")
    assert result["status"] == "needs_reconnect"
    assert flagged == ["openai:platform.openai.com"] and notified


@pytest.mark.asyncio
async def test_keepalive_refreshes_a_live_session(fake_browser, monkeypatch):
    context = _context()
    transport = browser_sessions.build_session_transport(
        storage_state={"cookies": [], "origins": []}, context=context,
        home_url="https://platform.openai.com/usage", final_url="https://platform.openai.com/usage",
        site_url="https://platform.openai.com/", heuristic_signed_in=True,
    )
    record = {
        "account_key": "openai:platform.openai.com", "provider": "openai", "kind": "analytics",
        "credential_mechanism": "browser_session", "account_address": "platform.openai.com",
        "display_label": "OpenAI", "status": "connected", "assistant_id": "a", "transport": transport,
    }
    saved = []

    async def _save(store, user_id, record_):
        saved.append(record_)

    monkeypatch.setattr("src.anubis.utils.connected_accounts.store.save_connected_account", _save)
    await browser_sessions.forget_session(record["account_key"])
    original_new_context = browser_sessions.new_context

    async def _signed_in_context(context_, *, storage_state=None, user_agent=None):
        fake = await original_new_context(context_, storage_state=storage_state, user_agent=user_agent)
        fake.page.html = "<a href='/logout'>Log out</a>"
        return fake

    monkeypatch.setattr(browser_sessions, "new_context", _signed_in_context)
    result = await browser_sessions.keepalive_record(context, None, record, user_id="u")
    assert result["status"] == "refreshed"
    assert saved and saved[0]["transport"]["browser_session"]["cookie_count"] == 1
    await browser_sessions.forget_session(record["account_key"])


@pytest.mark.asyncio
async def test_two_sign_ins_to_one_site_make_two_accounts_and_reconnect_keeps_one(fake_browser):
    context = _context()
    first = await browser_login.start_login(
        context, user_id="u", assistant_id="a", provider=get_provider("github"), name="work",
        site_url="https://github.com/login",
    )
    login = browser_login.get_live_login(first["login_id"])
    login.page.url = "https://github.com/notifications"
    login.page.html = "<a href='/logout'>Sign out</a>"
    first_record = (await browser_login.finish_login(context, login_id=first["login_id"], user_id="u", existing_records=[]))["record"]
    second = await browser_login.start_login(
        context, user_id="u", assistant_id="a", provider=get_provider("github"), name="personal",
        site_url="https://github.com/login",
    )
    login = browser_login.get_live_login(second["login_id"])
    login.page.url = "https://github.com/notifications"
    login.page.html = "<a href='/logout'>Sign out</a>"
    second_record = (await browser_login.finish_login(context, login_id=second["login_id"], user_id="u", existing_records=[first_record]))["record"]
    assert first_record["account_key"] != second_record["account_key"]
    assert first_record["display_label"] == "work" and second_record["display_label"] == "personal"
    assert first_record["credential_mechanism"] == "browser_session"

    again = await browser_login.start_login(
        context, user_id="u", assistant_id="a", provider=get_provider("github"),
        site_url="https://github.com/login", reconnect_account_key=first_record["account_key"],
    )
    login = browser_login.get_live_login(again["login_id"])
    login.page.url = "https://github.com/"
    login.page.html = "<a href='/logout'>Sign out</a>"
    refreshed = (await browser_login.finish_login(context, login_id=again["login_id"], user_id="u", existing_records=[first_record, second_record]))["record"]
    assert refreshed["account_key"] == first_record["account_key"]
    assert refreshed["display_label"] == "work"
