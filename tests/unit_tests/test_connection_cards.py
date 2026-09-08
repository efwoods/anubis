"""Connect cards persist on the reply, and the tool carries a card on every result."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.anubis.utils import secret_store
from src.anubis.utils.connected_accounts import connection_cards, connection_tools
from src.anubis.utils.connected_accounts.providers import get_provider
from src.anubis.utils.connected_accounts.store import build_account_record


def _context():
    return SimpleNamespace(
        connected_account_encryption_key=secret_store.generate_encryption_key(),
        mcp_connector_probe_timeout_seconds=1.0,
        connect_oauth_http_timeout_seconds=1.0,
    )


def test_cards_are_lifted_from_connect_tool_messages_only():
    card = {"provider": "gmail", "status": "connected", "tool_count": 6}
    messages = [
        HumanMessage(content="connect my gmail"),
        AIMessage(content=""),
        ToolMessage(
            content=json.dumps({"status": "connected", "card": card}),
            name="connect_account",
            tool_call_id="1",
        ),
        ToolMessage(content=json.dumps({"card": {"x": 1}}), name="other_tool", tool_call_id="2"),
        ToolMessage(content="not json", name="connect_account", tool_call_id="3"),
    ]
    assert connection_cards.connection_records_from_messages(messages) == [card]


def test_the_acknowledgement_card_comes_from_the_last_human_turn():
    card = {"provider": "github", "status": "connected"}
    messages = [
        HumanMessage(content="hi"),
        HumanMessage(
            content="acknowledge",
            additional_kwargs={"hidden": True, "kind": "connection_acknowledgement", "connection": card},
        ),
    ]
    assert connection_cards.connection_acknowledgement_card(messages) == [card]
    assert connection_cards.connection_acknowledgement_card([HumanMessage(content="x")]) == []


def test_card_records_and_status_lines():
    provider = get_provider("gmail")
    record = build_account_record(
        provider=provider,
        account_address="evan@example.com",
        display_label="evan",
        encrypted_secret="cipher",
        assistant_id="a",
    )
    card = connection_cards.connection_card_record(provider, record, status="connected")
    assert card["tool_count"] == 6
    assert card["login_mode"] == "oauth_popup"
    assert "cipher" not in json.dumps(card)
    assert connection_cards.card_status_line(card) == "Gmail · Added · 6 tools · Connected as evan"
    assert connection_cards.card_status_line({"display_name": "X", "status": "pending_login"}) == "X · Waiting for sign-in"
    instruction = connection_cards.acknowledgement_instruction(card)
    assert "Gmail" in instruction and "evan" in instruction and "credential" in instruction


def _connect_tool(monkeypatch, resume_value, accounts=(), allow_interrupt=True):
    raised = []

    def _fake_interrupt(payload):
        raised.append(payload)
        return resume_value

    monkeypatch.setattr(connection_tools, "interrupt", _fake_interrupt)

    class _Store:
        async def asearch(self, *args, **kwargs):
            return [SimpleNamespace(value=record) for record in accounts]

        async def aput(self, *args, **kwargs):
            pass

        async def aget(self, *args, **kwargs):
            return None

    tools = connection_tools.build_connection_tools(
        _context(),
        store=_Store(),
        user_id="auth0|owner",
        assistant_id="a",
        connected_accounts=list(accounts),
        allow_interrupt=allow_interrupt,
    )
    return {tool.name: tool for tool in tools}, raised


def test_the_gmail_card_opens_the_google_popup(monkeypatch):
    from src.anubis.utils.connected_accounts import repository as repository_module

    repository_module.set_repository(None)
    tools, raised = _connect_tool(monkeypatch, {"type": "cancel"})
    result = asyncio.run(tools["connect_account"].coroutine(provider="gmail"))
    card = raised[0]
    assert card["login_mode"] == "oauth_popup"
    assert card["login_endpoint"] == "/connect_account/oauth/start"
    assert card["login_request"] == {"provider": "gmail"}
    assert card["uses_form"] is False
    assert result["status"] == "cancelled"
    assert result["card"]["status"] == "cancelled"


def test_an_unattended_run_never_pauses(monkeypatch):
    tools, raised = _connect_tool(monkeypatch, {"type": "apply"}, allow_interrupt=False)
    result = asyncio.run(tools["connect_account"].coroutine(provider="github"))
    assert raised == []
    assert result["status"] == "not_connected"


def test_a_custom_connector_with_an_open_server_connects_without_a_card(monkeypatch):
    from src.anubis.utils.connected_accounts import mcp_oauth, mcp_server_tools
    from src.anubis.utils.connected_accounts import repository as repository_module

    repository = repository_module.InMemoryConnectedAccountRepository()
    repository_module.set_repository(repository)
    try:
        async def _open(server_url, context, *, http_client=None):
            return {"status": mcp_oauth.AUTHORIZATION_OPEN, "www_authenticate": None, "resource_metadata_url": None}

        async def _probe(server_url, bearer_token, timeout_seconds):
            return [SimpleNamespace(name="search")]

        monkeypatch.setattr(mcp_oauth, "probe_authorization", _open)
        monkeypatch.setattr(mcp_server_tools, "probe_server_tools", _probe)
        tools, raised = _connect_tool(monkeypatch, {"type": "apply"})
        result = asyncio.run(
            tools["connect_account"].coroutine(
                provider="custom_mcp", server_url="https://mcp.example.com/mcp", name="Notes"
            )
        )
        assert raised == []
        assert result["status"] == "connected"
        assert result["card"]["status"] == "connected"
        assert result["available_tools"] == ["search"]
        stored = asyncio.run(repository.list_for_user("auth0|owner"))
        assert len(stored) == 1 and stored[0]["display_label"] == "Notes"
    finally:
        repository_module.set_repository(None)


def test_a_custom_connector_that_needs_oauth_raises_a_popup_card(monkeypatch):
    from src.anubis.utils.connected_accounts import mcp_oauth

    async def _needs(server_url, context, *, http_client=None):
        return {"status": mcp_oauth.AUTHORIZATION_NEEDS_OAUTH, "www_authenticate": "Bearer", "resource_metadata_url": "https://mcp.example.com/.well-known/oauth-protected-resource"}

    monkeypatch.setattr(mcp_oauth, "probe_authorization", _needs)
    tools, raised = _connect_tool(monkeypatch, {"type": "cancel"})
    asyncio.run(
        tools["connect_account"].coroutine(
            provider="custom_mcp", server_url="https://mcp.example.com/mcp", name="Notes"
        )
    )
    card = raised[0]
    assert card["login_mode"] == "oauth_popup"
    assert card["login_request"] == {"provider": "custom_mcp", "server_url": "https://mcp.example.com/mcp", "name": "Notes"}
    assert card["prefilled_fields"]["server_url"] == "https://mcp.example.com/mcp"


def test_stale_accounts_are_reported(monkeypatch):
    tools, _ = _connect_tool(monkeypatch, {"type": "apply"})
    stale = connection_tools.build_connection_tools(
        _context(),
        store=None,
        user_id="u",
        assistant_id="a",
        connected_accounts=[],
        stale_accounts=[{"provider": "gmail", "display_label": "evan", "account_address": "e@x.com"}],
    )
    report = {tool.name: tool for tool in stale}["list_connections_needing_sign_in"]
    result = asyncio.run(report.coroutine())
    assert result["count"] == 1 and result["accounts"][0]["provider"] == "gmail"
