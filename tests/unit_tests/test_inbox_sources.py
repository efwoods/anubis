"""Triage over every connected account: discovery and delivery through the account's tools."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from langchain.tools import tool
from langchain_core.messages import AIMessage

from src.anubis.utils.connected_accounts import repository as accounts_repository_module
from src.anubis.utils.inbox import poller, sources
from src.anubis.utils.inbox import repository as inbox_repository_module


class _FakeModel:
    """A tool-calling model that follows a script, then answers structured."""

    def __init__(self, script, structured_answer):
        self.script = list(script)
        self.structured_answer = structured_answer
        self.bound_tools = None
        self.transcripts = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def with_structured_output(self, schema):
        parent = self

        class _Structured:
            def with_config(self, **kwargs):
                return self

            async def ainvoke(self, messages):
                parent.transcripts.append(messages)
                return parent.structured_answer

        return _Structured()

    async def ainvoke(self, messages):
        if self.script:
            calls = self.script.pop(0)
            return AIMessage(content="", tool_calls=calls)
        return AIMessage(content="done")


def _record(**overrides):
    record = {
        "account_key": "slack:app.slack.com",
        "provider": "slack",
        "kind": "messaging",
        "credential_mechanism": "browser_session",
        "account_address": "app.slack.com",
        "display_label": "Slack",
        "status": "connected",
        "assistant_id": "avatar-1",
        "user_id": "owner-1",
        "transport": {"browser_session": {"storage_state_encrypted": "x"}, "site_url": "https://app.slack.com/"},
    }
    record.update(overrides)
    return record


def test_message_sources_exclude_devices_banks_and_websites():
    assert sources.is_message_source(_record())
    assert sources.is_message_source(_record(provider="github", kind="developer", credential_mechanism="oauth"))
    assert not sources.is_message_source(_record(kind="bank", credential_mechanism="plaid_link"))
    assert not sources.is_message_source(_record(kind="website", credential_mechanism="url_only"))
    assert not sources.is_message_source(_record(kind="data_source", credential_mechanism="device_pairing"))
    assert not sources.is_message_source(_record(status="needs_reconnect"))


@pytest.mark.asyncio
async def test_discovery_calls_the_account_tools_and_shapes_messages(monkeypatch):
    seen_calls = []

    @tool
    async def open_connected_site(connection: str | None = None, path: str = "/") -> dict:
        """Open a page."""
        seen_calls.append(("open", path))
        return {"status": "ok", "text": "Notifications: Ada mentioned you: lunch tomorrow?"}

    async def _fake_tools(context, records, store=None, **kwargs):
        return [open_connected_site]

    monkeypatch.setattr(
        "src.anubis.utils.connected_accounts.tool_factories.build_tools_for_accounts", _fake_tools
    )
    answer = sources.DiscoveredItems(
        items=[
            sources.DiscoveredItem(external_id="msg-1", sender="Ada", subject="lunch tomorrow?", body_text="lunch tomorrow?", url="https://app.slack.com/m/1"),
            sources.DiscoveredItem(external_id="", sender="Bob", body_text="hey"),
        ],
        checked=["/notifications"],
    )
    model = _FakeModel(
        script=[[{"name": "open_connected_site", "args": {"path": "/notifications"}, "id": "call-1"}]],
        structured_answer=answer,
    )
    messages, raw = await sources.discover_new_items(
        SimpleNamespace(), None, _record(), since=datetime(2026, 9, 1, tzinfo=UTC), limit=5, model=model
    )
    assert seen_calls == [("open", "/notifications")]
    assert raw.checked == ["/notifications"]
    assert messages[0]["message_id"] == "msg-1"
    assert messages[0]["provider"] == "slack" and messages[0]["url"] == "https://app.slack.com/m/1"
    assert messages[1]["message_id"].startswith("content:")
    # The transcript handed to the structured model holds the tool result.
    transcript_text = json.dumps([getattr(m, "content", "") for m in model.transcripts[0]])
    assert "Ada mentioned you" in transcript_text


@pytest.mark.asyncio
async def test_delivery_posts_through_the_account_tools(monkeypatch):
    posted = []

    @tool
    async def type_into_connected_field(connection: str | None = None, selector: str = "", text: str = "", submit: bool = False) -> dict:
        """Type text."""
        posted.append(text)
        return {"status": "ok"}

    async def _fake_tools(context, records, store=None, **kwargs):
        return [type_into_connected_field]

    monkeypatch.setattr(
        "src.anubis.utils.connected_accounts.tool_factories.build_tools_for_accounts", _fake_tools
    )
    model = _FakeModel(
        script=[[{"name": "type_into_connected_field", "args": {"selector": "#reply", "text": "See you at noon.", "submit": True}, "id": "c1"}]],
        structured_answer=sources.DeliveryResult(sent=True, detail="Posted in the thread."),
    )
    result = await sources.deliver_reply_via_account(
        SimpleNamespace(), None, _record(), message={"sender": "Ada", "subject": "lunch", "url": "https://app.slack.com/m/1"}, draft_text="See you at noon.", model=model
    )
    assert result.sent is True and posted == ["See you at noon."]


@pytest.mark.asyncio
async def test_the_poller_visits_every_non_mailbox_account_and_creates_items(monkeypatch):
    accounts = accounts_repository_module.InMemoryConnectedAccountRepository()
    accounts_repository_module.set_repository(accounts)
    inbox = inbox_repository_module.InMemoryInboxRepository()
    inbox_repository_module.set_inbox_repository(inbox)
    try:
        await accounts.upsert("owner-1", _record())
        await accounts.upsert("owner-1", _record(account_key="plaid:item", provider="plaid", kind="bank", credential_mechanism="plaid_link"))
        discovered = []

        async def _fake_discover(context, store, record, *, since, limit, max_steps, model=None):
            discovered.append(record["account_key"])
            return [
                {"message_id": "msg-1", "sender": "Ada", "subject": "lunch?", "body_text": "lunch?", "sent_at": "2026-09-07T10:00:00+00:00", "provider": "slack", "url": "https://app.slack.com/m/1"}
            ], sources.DiscoveredItems()

        ran = []

        async def _fake_run(context, *, user_id, assistant_id, account_key, message, assistant=None, source_kind="email"):
            ran.append((account_key, source_kind, message["message_id"]))
            return {"item_id": "i1"}

        monkeypatch.setattr(sources, "discover_new_items", _fake_discover)
        monkeypatch.setattr(poller, "run_inbox_for_message", _fake_run)
        context = SimpleNamespace(inbox_account_poll_enabled="true", inbox_account_poll_interval_seconds=1800.0, inbox_discovery_max_items=10, inbox_discovery_max_steps=8)
        result = await poller.poll_other_accounts(context)
        assert result == {"polled": 1, "new_items": 1}
        assert discovered == ["slack:app.slack.com"]
        assert ran == [("slack:app.slack.com", "slack", "msg-1")]
        state = await inbox.get_poll_state("slack:app.slack.com")
        assert state and state.get("last_polled_at")
        # Within the interval the account is skipped; an owner-requested poll is not.
        assert (await poller.poll_other_accounts(context))["polled"] == 0
        assert (await poller.poll_other_accounts(context, only_user_id="owner-1"))["polled"] == 1
        # Disabled by configuration.
        off = SimpleNamespace(inbox_account_poll_enabled="false")
        assert (await poller.poll_other_accounts(off)).get("disabled") is True
    finally:
        accounts_repository_module.set_repository(None)
        inbox_repository_module.set_inbox_repository(None)
