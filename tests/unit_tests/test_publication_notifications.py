"""Reading a platform's "you published something" email as a content event.

This is the transport that covers the platforms offering no webhook, and it is
the one most able to do damage if it is loose: a social network emails
constantly, and almost none of it is the owner publishing. The tests are
therefore weighted toward what must be refused.
"""

from __future__ import annotations

import pytest

from src.anubis.utils.connected_accounts import ownership
from src.anubis.utils.subscriptions import email_notifications as notifications


def _record(provider="twitter", handle="evan"):
    return {
        "account_key": f"{provider}:{handle}",
        "provider": provider,
        "kind": "social",
        "display_label": provider,
        "assistant_id": "avatar-1",
        "user_id": "owner-1",
        "transport": {},
        "ownership": ownership.build_ownership(
            state=ownership.OWNERSHIP_PROVEN,
            method=ownership.METHOD_OAUTH_IDENTITY,
            handle=handle,
        ),
    }


def test_a_platform_sender_is_recognised():
    assert notifications.provider_for_sender("info@x.com").name == "twitter"
    assert notifications.provider_for_sender("no-reply@linkedin.com").name == "linkedin"
    assert notifications.provider_for_sender("hi@tiktok.com").name == "tiktok"


def test_a_platform_subdomain_still_matches():
    """Platforms send from notify.linkedin.com while the registry names linkedin.com."""
    provider = notifications.provider_for_sender("jobs@e.linkedin.com")
    assert provider is not None and provider.name == "linkedin"


def test_an_unrelated_sender_costs_nothing():
    """The domain check runs before any model call, which is what keeps mail cheap."""
    assert notifications.provider_for_sender("friend@example.com") is None
    assert notifications.provider_for_sender("") is None
    assert notifications.provider_for_sender(None) is None


def test_a_lookalike_domain_does_not_match():
    """``notlinkedin.com`` must not be read as LinkedIn."""
    assert notifications.provider_for_sender("spam@notlinkedin.com") is None


def test_an_address_on_the_platform_carrying_the_handle_belongs_to_the_owner():
    assert notifications.url_belongs_to_account(
        "https://x.com/evan/status/123", _record()
    )


def test_another_persons_post_on_the_same_platform_is_refused():
    """A notice about somebody else's post is not identity material."""
    assert not notifications.url_belongs_to_account(
        "https://x.com/someone_else/status/123", _record()
    )


def test_an_address_on_a_different_site_is_refused():
    assert not notifications.url_belongs_to_account(
        "https://evil.example.com/evan", _record()
    )


@pytest.mark.asyncio
async def test_a_publication_notice_becomes_a_content_event(monkeypatch):
    events = []

    async def _fake_event(**kwargs):
        events.append(kwargs)
        return {"status": "ingested"}

    async def _fake_read(context, *, subject, body_text, sender):
        return notifications.PublicationNotice(
            is_publication_notice=True,
            content_url="https://x.com/evan/status/9",
            title="A post",
            published_at=None,
            reasoning="says your post is live",
        )

    async def _fake_account(store, *, user_id, personal_avatar_id, provider_name):
        return _record()

    monkeypatch.setattr(notifications, "read_publication_notice", _fake_read)
    monkeypatch.setattr(notifications, "_proven_account_for", _fake_account)
    monkeypatch.setattr(
        "src.anubis.utils.subscriptions.intake.record_content_event", _fake_event
    )

    result = await notifications.handle_mail_as_publication(
        object(),
        store=None,
        user_id="owner-1",
        personal_avatar_id="avatar-1",
        sender="info@x.com",
        subject="Your post is live",
        body_text="See it here",
        message_id="mail-1",
    )
    assert result == {"status": "ingested"}
    assert events[0]["url"] == "https://x.com/evan/status/9"
    assert events[0]["transport"] == "email_notification"


@pytest.mark.asyncio
async def test_ordinary_platform_mail_is_not_a_publication(monkeypatch):
    """Someone replied, someone followed you: constant, and never an ingest."""

    async def _fake_read(context, *, subject, body_text, sender):
        return notifications.PublicationNotice(
            is_publication_notice=False,
            content_url=None,
            title=None,
            published_at=None,
            reasoning="this is a follower notification",
        )

    monkeypatch.setattr(notifications, "read_publication_notice", _fake_read)
    result = await notifications.handle_mail_as_publication(
        object(),
        store=None,
        user_id="owner-1",
        personal_avatar_id="avatar-1",
        sender="info@x.com",
        subject="You have a new follower",
        body_text="Someone followed you",
        message_id="mail-2",
    )
    assert result is None


@pytest.mark.asyncio
async def test_a_notice_with_no_proven_account_is_refused(monkeypatch):
    """A platform can email anyone; only a proven account makes it identity."""

    async def _fake_read(context, *, subject, body_text, sender):
        return notifications.PublicationNotice(
            is_publication_notice=True,
            content_url="https://x.com/evan/status/9",
            title=None,
            published_at=None,
            reasoning="your post is live",
        )

    async def _no_account(store, *, user_id, personal_avatar_id, provider_name):
        return None

    monkeypatch.setattr(notifications, "read_publication_notice", _fake_read)
    monkeypatch.setattr(notifications, "_proven_account_for", _no_account)

    result = await notifications.handle_mail_as_publication(
        object(),
        store=None,
        user_id="owner-1",
        personal_avatar_id="avatar-1",
        sender="info@x.com",
        subject="Your post is live",
        body_text="See it",
        message_id="mail-3",
    )
    assert result is not None and result["status"] == "refused"


@pytest.mark.asyncio
async def test_a_notice_about_someone_elses_post_is_refused(monkeypatch):
    async def _fake_read(context, *, subject, body_text, sender):
        return notifications.PublicationNotice(
            is_publication_notice=True,
            content_url="https://x.com/stranger/status/9",
            title=None,
            published_at=None,
            reasoning="a post is live",
        )

    async def _fake_account(store, *, user_id, personal_avatar_id, provider_name):
        return _record()

    monkeypatch.setattr(notifications, "read_publication_notice", _fake_read)
    monkeypatch.setattr(notifications, "_proven_account_for", _fake_account)

    result = await notifications.handle_mail_as_publication(
        object(),
        store=None,
        user_id="owner-1",
        personal_avatar_id="avatar-1",
        sender="info@x.com",
        subject="A post is live",
        body_text="See it",
        message_id="mail-4",
    )
    assert result is not None and result["status"] == "refused"
    assert "does not belong" in result["detail"]
