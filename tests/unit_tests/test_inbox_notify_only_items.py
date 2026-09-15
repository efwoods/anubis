"""Items this system wrote for the owner, rather than items that arrived from somebody.

A scheduled report and an account question both appear in the agent inbox, and
neither has anybody to reply TO. Two things follow, and both were wrong before:

- the panel must not offer "send a reply" beside them, because no such action can
  be carried out;
- resolving one must not invoke the triage graph. These items have no paused run
  behind them — nothing ever interrupted, so there is no checkpoint for
  ``Command(resume=...)`` to deliver a decision to — and invoking the graph would
  start a fresh run that triages a notification as though it were incoming mail.
"""

from __future__ import annotations

import pytest

from src.anubis.utils.inbox.repository import (
    ACTION_ACCEPT_BAN,
    ACTION_NOTIFY_OWNER,
    ACTION_REVOKE_BAN,
    NOTIFY_ONLY_SOURCE_KINDS,
    STATE_IGNORED,
    STATE_PENDING_OWNER,
    STATE_RESOLVED,
    available_actions_for,
)

_ACKNOWLEDGE_ONLY_SOURCE_KINDS = tuple(
    kind
    for kind in NOTIFY_ONLY_SOURCE_KINDS
    if kind not in ("moderation", "appeal")
)


@pytest.mark.parametrize("source_kind", _ACKNOWLEDGE_ONLY_SOURCE_KINDS)
def test_only_acknowledging_is_offered(source_kind: str) -> None:
    assert available_actions_for({"source_kind": source_kind}) == [ACTION_NOTIFY_OWNER]


def test_an_account_question_is_one_of_them() -> None:
    assert "account_discovery" in NOTIFY_ONLY_SOURCE_KINDS
    # The pre-existing report kind is fixed by the same change.
    assert "report" in NOTIFY_ONLY_SOURCE_KINDS
    assert "moderation" in NOTIFY_ONLY_SOURCE_KINDS
    assert "appeal" in NOTIFY_ONLY_SOURCE_KINDS
    assert "phone_call" in NOTIFY_ONLY_SOURCE_KINDS


def test_moderation_offers_revoke_and_accept_only() -> None:
    assert available_actions_for({"source_kind": "moderation"}) == [
        ACTION_REVOKE_BAN,
        ACTION_ACCEPT_BAN,
    ]
    assert available_actions_for({"source_kind": "appeal"}) == [
        ACTION_REVOKE_BAN,
        ACTION_ACCEPT_BAN,
    ]


def test_inbox_search_and_source_filters() -> None:
    from src.anubis.utils.inbox.repository import item_matches_inbox_filters

    item = {
        "source_kind": "moderation",
        "subject": "Ban: someone@example.com",
        "reason": "quoted sexual content",
        "body_text": "the quoted line",
        "sender": "someone@example.com",
    }
    assert item_matches_inbox_filters(item, source_kind="moderation", query="quoted")
    assert not item_matches_inbox_filters(item, source_kind="report")
    assert not item_matches_inbox_filters(item, query="mailbox")


def test_ordinary_items_keep_every_action_they_had() -> None:
    mailbox_actions = available_actions_for({"source_kind": "gmail"})
    assert "send_reply" in mailbox_actions
    group_actions = available_actions_for({"source_kind": "slack"})
    assert "post_reply" in group_actions
    assert "moderate" in group_actions


def test_actions_a_bot_reported_itself_still_win() -> None:
    # A group bot records what its own platform can do; that must not be
    # overridden by any of the fallbacks.
    assert available_actions_for(
        {"source_kind": "report", "available_actions": ["post_reply"]}
    ) == ["post_reply"]


class _Repository:
    def __init__(self, item: dict) -> None:
        self.item = dict(item)
        self.updates: list[dict] = []

    async def get_item(self, item_id: str):
        return dict(self.item) if item_id == self.item["item_id"] else None

    async def update_item(self, item_id: str, **fields):
        self.updates.append({"item_id": item_id, **fields})
        self.item.update(fields)
        return dict(self.item)


def _install(monkeypatch, repository, on_graph_call) -> None:
    import src.anubis.utils.inbox.poller as poller_module

    monkeypatch.setattr(poller_module, "get_inbox_repository", lambda: repository)
    monkeypatch.setattr(poller_module, "_graph", on_graph_call)


@pytest.mark.asyncio
async def test_resolving_an_account_question_never_touches_the_triage_graph(
    monkeypatch,
) -> None:
    from src.anubis.utils.inbox.poller import resume_inbox_item

    repository = _Repository(
        {
            "item_id": "item-1",
            "source_kind": "account_discovery",
            "state": STATE_PENDING_OWNER,
        }
    )

    def _explode():
        raise AssertionError("a server-authored item has no paused run to resume")

    _install(monkeypatch, repository, _explode)
    result = await resume_inbox_item(
        None, item_id="item-1", human_response={"action": "connected"}
    )
    assert result["state"] == STATE_RESOLVED
    assert repository.updates[0]["owner_decision"] == {"action": "connected"}


@pytest.mark.asyncio
async def test_the_owner_saying_no_ignores_rather_than_resolves(monkeypatch) -> None:
    from src.anubis.utils.inbox.poller import resume_inbox_item

    repository = _Repository(
        {"item_id": "item-2", "source_kind": "report", "state": STATE_PENDING_OWNER}
    )
    _install(monkeypatch, repository, lambda: pytest.fail("graph must not be built"))
    result = await resume_inbox_item(
        None, item_id="item-2", human_response={"action": "ignore"}
    )
    assert result["state"] == STATE_IGNORED


@pytest.mark.asyncio
async def test_an_item_that_is_already_answered_is_left_alone(monkeypatch) -> None:
    from src.anubis.utils.inbox.poller import resume_inbox_item

    repository = _Repository(
        {
            "item_id": "item-3",
            "source_kind": "account_discovery",
            "state": STATE_RESOLVED,
        }
    )
    _install(monkeypatch, repository, lambda: pytest.fail("graph must not be built"))
    result = await resume_inbox_item(None, item_id="item-3", human_response={})
    assert result["state"] == STATE_RESOLVED
    assert repository.updates == []


@pytest.mark.asyncio
async def test_revoking_a_moderation_item_lifts_the_ban_without_the_graph(
    monkeypatch,
) -> None:
    from src.anubis.utils.inbox.poller import resume_inbox_item

    repository = _Repository(
        {
            "item_id": "item-mod",
            "source_kind": "moderation",
            "state": STATE_PENDING_OWNER,
            "confidence_detail": {"ban_id": "ban-1"},
        }
    )
    lifted: list[str] = []

    async def fake_lift(pool, ban_id, appeal_note):
        lifted.append(ban_id)
        return {"ban_id": ban_id, "lifted_at": "2026-09-12T00:00:00+00:00"}

    monkeypatch.setattr(
        "src.anubis.utils.runtime_handles.get_postgres_pool", lambda: object()
    )
    monkeypatch.setattr("src.security.bans.lift_ban", fake_lift)
    _install(monkeypatch, repository, lambda: pytest.fail("graph must not be built"))
    result = await resume_inbox_item(
        None,
        item_id="item-mod",
        human_response={"type": "accept", "args": {"action": "revoke_ban"}},
    )
    assert result["state"] == STATE_RESOLVED
    assert lifted == ["ban-1"]


@pytest.mark.asyncio
async def test_accepting_a_moderation_item_leaves_the_ban_and_skips_the_graph(
    monkeypatch,
) -> None:
    from src.anubis.utils.inbox.poller import resume_inbox_item

    repository = _Repository(
        {
            "item_id": "item-mod-2",
            "source_kind": "moderation",
            "state": STATE_PENDING_OWNER,
            "confidence_detail": {"ban_id": "ban-2"},
        }
    )
    lifted: list[str] = []

    async def fake_lift(pool, ban_id, appeal_note):
        lifted.append(ban_id)
        return {"ban_id": ban_id}

    monkeypatch.setattr(
        "src.anubis.utils.runtime_handles.get_postgres_pool", lambda: object()
    )
    monkeypatch.setattr("src.security.bans.lift_ban", fake_lift)
    _install(monkeypatch, repository, lambda: pytest.fail("graph must not be built"))
    result = await resume_inbox_item(
        None,
        item_id="item-mod-2",
        human_response={"type": "accept", "args": {"action": "accept_ban"}},
    )
    assert result["state"] == STATE_RESOLVED
    assert lifted == []
