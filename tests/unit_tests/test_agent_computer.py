"""The agent computer keeps its Chromium context after I'm done or Skip."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.anubis.utils.connected_accounts.agent_computer import (
    COMPUTER_HANDOFF_INTERRUPT_KIND,
    VENDOR_DASHBOARD_QUEUE,
    AgentComputerSession,
    DashboardStep,
    advance_queue,
    build_handoff_card,
    finish_handoff,
    page_looks_like_login,
    reset_computers_for_tests,
    skip_handoff,
    start_computer,
)


class FakePage:
    def __init__(self, url: str = "https://cursor.com/dashboard/spending") -> None:
        self.url = url
        self.html = "<html><body>Sign out</body></html>"
        self.viewport_size = {"width": 1280, "height": 800}
        self.closed = False

    async def content(self) -> str:
        return self.html

    async def goto(self, url: str, **kwargs) -> None:
        self.url = url

    async def screenshot(self, **kwargs) -> bytes:
        return b"jpeg-bytes"

    async def set_viewport_size(self, size: dict) -> None:
        self.viewport_size = size


class FakeContext:
    def __init__(self, page: FakePage | None = None) -> None:
        self.page = page or FakePage()
        self.pages = [self.page]
        self.closed = False

    async def new_page(self) -> FakePage:
        return self.page

    async def storage_state(self) -> dict:
        return {"cookies": [{"name": "sid", "value": "1"}]}

    async def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_computers():
    reset_computers_for_tests()
    yield
    reset_computers_for_tests()


class FakeStore:
    def __init__(self) -> None:
        self.saved: list[dict] = []

    async def aput(self, namespace, key, value):
        self.saved.append(value)


@pytest.mark.asyncio
async def test_finish_does_not_close_the_browser_context(monkeypatch):
    page = FakePage()
    browser_context = FakeContext(page)
    session = AgentComputerSession(
        session_id="abc123def456",
        user_id="admin",
        assistant_id="avatar",
        browser_context=browser_context,
        page=page,
        nonce="nonce",
        current_provider="cursor",
        current_url=page.url,
        current_task="Sign in to Cursor",
        queue=list(VENDOR_DASHBOARD_QUEUE),
    )

    async def fake_persist(*arguments, **keyword_arguments):
        return {"status": "ok", "cookie_count": 1, "context_closed": False}

    async def fake_recipe(*arguments, **keyword_arguments):
        return {"status": "ok", "rows": []}

    monkeypatch.setattr(
        "src.anubis.utils.connected_accounts.agent_computer.persist_computer_session",
        fake_persist,
    )
    monkeypatch.setattr(
        "src.anubis.utils.connected_accounts.agent_computer.run_recipe_on_computer",
        fake_recipe,
    )
    finished = await finish_handoff(
        SimpleNamespace(), None, session, existing_records=[], pool=None
    )
    assert finished["status"] == "done"
    assert finished["context_closed"] is False
    assert browser_context.closed is False
    assert session.context_is_open is True


@pytest.mark.asyncio
async def test_skip_advances_the_queue_without_closing():
    page = FakePage()
    browser_context = FakeContext(page)
    session = AgentComputerSession(
        session_id="skip1",
        user_id="admin",
        assistant_id="avatar",
        browser_context=browser_context,
        page=page,
        nonce="nonce",
        queue=list(VENDOR_DASHBOARD_QUEUE),
        queue_index=0,
        current_provider="cursor",
        current_url=VENDOR_DASHBOARD_QUEUE[0].url,
    )
    skipped = await skip_handoff(session)
    assert skipped["status"] == "skipped"
    assert skipped["context_closed"] is False
    assert skipped["next_provider"] == "cursor"
    assert "usage" in skipped["next_url"]
    assert session.queue_index == 1
    assert browser_context.closed is False


@pytest.mark.asyncio
async def test_login_wall_is_detected_and_re_raises_action_needed():
    page = FakePage(url="https://cursor.com/login")
    page.html = '<html><body><input type="password" name="password"></body></html>'
    session = AgentComputerSession(
        session_id="login1",
        user_id="admin",
        assistant_id="avatar",
        browser_context=FakeContext(page),
        page=page,
        nonce="nonce",
        current_url=page.url,
        current_provider="cursor",
        queue=[VENDOR_DASHBOARD_QUEUE[0]],
    )
    assert await page_looks_like_login(session) is True

    async def fake_persist(*arguments, **keyword_arguments):
        return {"status": "ok", "cookie_count": 0, "context_closed": False}

    from src.anubis.utils.connected_accounts import agent_computer as computer_module

    original = computer_module.persist_computer_session
    computer_module.persist_computer_session = fake_persist
    try:
        finished = await finish_handoff(
            SimpleNamespace(), None, session, existing_records=[], pool=None
        )
    finally:
        computer_module.persist_computer_session = original
    assert finished["status"] == "login_required"
    assert finished["context_closed"] is False


def test_handoff_card_names_action_needed_and_waiting_providers():
    page = FakePage()
    session = AgentComputerSession(
        session_id="card1",
        user_id="admin",
        assistant_id="avatar",
        browser_context=FakeContext(page),
        page=page,
        nonce="nonce",
        queue=list(VENDOR_DASHBOARD_QUEUE),
        current_task=VENDOR_DASHBOARD_QUEUE[0].task,
        current_provider="cursor",
        current_url=VENDOR_DASHBOARD_QUEUE[0].url,
        preview_jpeg_b64="abc",
    )
    card = build_handoff_card(
        session, context=SimpleNamespace(connect_oauth_state_secret="secret-for-tests-32chars!!")
    )
    assert card["kind"] == COMPUTER_HANDOFF_INTERRUPT_KIND
    assert card["status"] == "action_needed"
    assert card["preview_frame"] == "abc"
    assert card["providers_waiting"][0] == "cursor"
    assert "claude_app" in card["providers_waiting"]
    assert "takeover" in card["actions"]


@pytest.mark.asyncio
async def test_start_computer_reuses_an_open_context():
    page = FakePage()
    browser_context = FakeContext(page)
    first = await start_computer(
        SimpleNamespace(),
        user_id="admin",
        assistant_id="avatar",
        browser_context=browser_context,
        page=page,
        queue=[
            DashboardStep(
                provider="cursor",
                url="https://cursor.com/dashboard/spending",
                recipe="spending",
                task="Sign in",
            )
        ],
    )
    second = await start_computer(
        SimpleNamespace(),
        user_id="admin",
        assistant_id="avatar",
        start_url="https://cursor.com/dashboard/usage",
        provider="cursor",
        task="Usage",
    )
    assert first.session_id == second.session_id
    assert first.browser_context is browser_context
    assert browser_context.closed is False
    assert page.url.endswith("/usage")


def test_advance_queue_walks_every_dashboard_then_finishes():
    session = AgentComputerSession(
        session_id="q",
        user_id="admin",
        assistant_id="a",
        browser_context=FakeContext(),
        page=FakePage(),
        nonce="n",
        queue=list(VENDOR_DASHBOARD_QUEUE),
    )
    providers = [session.queue[0].provider]
    while advance_queue(session) is not None:
        providers.append(session.queue[session.queue_index].provider)
    assert providers[0] == "cursor"
    assert "claude_app" in providers
    assert "openai" in providers
    assert "elevenlabs" in providers
    assert "xai" in providers
    assert advance_queue(session) is None
