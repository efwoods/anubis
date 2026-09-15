"""Ban appeals arrive as mail to the Neural Nexus contact addresses."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.anubis.utils.inbox.appeals import (
    APPEAL_SOURCE_KIND,
    appeal_contact_phrase,
    appeal_inbox_addresses,
    message_is_ban_appeal,
)
from src.anubis.utils.inbox.repository import (
    InMemoryInboxRepository,
    STATE_PENDING_OWNER,
    STATE_RESOLVED,
)


def _admin_context() -> SimpleNamespace:
    return SimpleNamespace(
        admin_user_id="admin-1",
        admin_account_email="e.woods.business@icloud.com",
        ban_appeal_inbox_addresses=(
            "contact@neuralnexus.site,business@neuralnexus.site,"
            "support@neuralnexus.site"
        ),
    )


def test_the_three_neural_nexus_addresses_are_the_appeal_inbox() -> None:
    addresses = appeal_inbox_addresses(_admin_context())
    assert addresses == [
        "contact@neuralnexus.site",
        "business@neuralnexus.site",
        "support@neuralnexus.site",
    ]
    phrase = appeal_contact_phrase(_admin_context())
    assert "contact@neuralnexus.site" in phrase
    assert "business@neuralnexus.site" in phrase
    assert "support@neuralnexus.site" in phrase


def test_a_custom_contact_email_is_the_only_appeal_address() -> None:
    context = SimpleNamespace(ban_appeal_contact_email="appeals@example.com")
    assert appeal_inbox_addresses(context) == ["appeals@example.com"]
    assert appeal_contact_phrase(context) == "appeals@example.com"


@pytest.mark.parametrize(
    "recipients",
    [
        "contact@neuralnexus.site",
        "Support Desk <support@neuralnexus.site>",
        "business@neuralnexus.site, other@example.com",
    ],
)
def test_mail_to_an_appeal_address_on_the_admin_mailbox_is_an_appeal(
    recipients: str,
) -> None:
    assert message_is_ban_appeal(
        {"recipients": recipients, "sender": "banned@example.com"},
        user_id="admin-1",
        context=_admin_context(),
    )


def test_mail_to_an_appeal_address_on_someone_elses_mailbox_is_not() -> None:
    assert not message_is_ban_appeal(
        {"recipients": "contact@neuralnexus.site", "sender": "banned@example.com"},
        user_id="someone-else",
        context=_admin_context(),
    )


def test_ordinary_mail_to_the_admin_mailbox_is_not_an_appeal() -> None:
    assert not message_is_ban_appeal(
        {"recipients": "e.woods.business@icloud.com", "sender": "friend@example.com"},
        user_id="admin-1",
        context=_admin_context(),
    )


def test_a_forwarded_alias_is_still_an_appeal() -> None:
    assert message_is_ban_appeal(
        {
            "recipients": "e.woods.business@icloud.com",
            "original_to": "support@neuralnexus.site",
            "sender": "banned@example.com",
        },
        user_id="admin-1",
        context=_admin_context(),
    )


@pytest.mark.asyncio
async def test_an_appeal_is_recorded_without_running_mailbox_triage(
    monkeypatch,
) -> None:
    from src.anubis.utils.inbox.poller import run_inbox_for_message

    repository = InMemoryInboxRepository()
    monkeypatch.setattr(
        "src.anubis.utils.inbox.poller.get_inbox_repository", lambda: repository
    )
    monkeypatch.setattr(
        "src.anubis.utils.runtime_handles.get_postgres_pool", lambda: None
    )

    def _explode():
        raise AssertionError("an appeal must not run mailbox triage")

    monkeypatch.setattr(
        "src.anubis.utils.inbox.poller._graph", _explode
    )
    item = await run_inbox_for_message(
        _admin_context(),
        user_id="admin-1",
        assistant_id="admin-personal-avatar",
        account_key="mailbox:icloud",
        message={
            "sender": "Banned Person <banned@example.com>",
            "recipients": "contact@neuralnexus.site",
            "subject": "Please lift this ban",
            "body_text": "I did not mean to send that.",
            "rfc822_message_id": "<appeal-1@example.com>",
        },
    )
    assert item is not None
    assert item["source_kind"] == APPEAL_SOURCE_KIND
    assert item["state"] == STATE_PENDING_OWNER
    assert item["available_actions"] == ["revoke_ban", "accept_ban"]
    assert "I did not mean to send that." in item["body_text"]


@pytest.mark.asyncio
async def test_accepting_an_appeal_closes_the_item_without_the_graph(
    monkeypatch,
) -> None:
    from src.anubis.utils.inbox.poller import resume_inbox_item

    repository = InMemoryInboxRepository()
    stored = await repository.create_item(
        {
            "item_id": "appeal-item",
            "source_kind": APPEAL_SOURCE_KIND,
            "state": STATE_PENDING_OWNER,
            "confidence_detail": {"ban_id": "ban-9"},
        }
    )
    monkeypatch.setattr(
        "src.anubis.utils.inbox.poller.get_inbox_repository", lambda: repository
    )
    monkeypatch.setattr(
        "src.anubis.utils.inbox.poller._graph",
        lambda: pytest.fail("graph must not be built"),
    )
    lifted: list[str] = []

    async def fake_lift(pool, ban_id, appeal_note):
        lifted.append(ban_id)
        return {"ban_id": ban_id}

    monkeypatch.setattr("src.security.bans.lift_ban", fake_lift)
    result = await resume_inbox_item(
        None,
        item_id=stored["item_id"],
        human_response={"type": "accept", "args": {"action": "accept_ban"}},
    )
    assert result["state"] == STATE_RESOLVED
    assert lifted == []
