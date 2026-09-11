"""The intake every content announcement passes through, and the signatures guarding it.

Two things are being protected here. The intake must never let an unproven
account's content reach identity, and must never ingest the same item twice —
a redelivery costs a full transcription. The webhook verifiers must refuse a
forged body, because the callback route cannot be authenticated and the
signature is all that stands between a stranger and the avatar's identity.
"""

from __future__ import annotations

import hashlib
import hmac

import pytest

from src.anubis.utils.connected_accounts import ownership
from src.anubis.utils.subscriptions import intake, payloads, repository, transports


@pytest.fixture(autouse=True)
def _fresh_repository():
    """Give every test its own in-memory repository."""
    store = repository.InMemorySubscriptionRepository()
    repository.set_subscription_repository(store)
    yield store
    repository.set_subscription_repository(None)


def _proven_record(**overrides):
    record = {
        "account_key": "youtube:evan",
        "provider": "youtube",
        "kind": "social",
        "display_label": "YouTube",
        "assistant_id": "avatar-1",
        "user_id": "owner-1",
        "status": "connected",
        "transport": {},
        "ownership": ownership.build_ownership(
            state=ownership.OWNERSHIP_PROVEN,
            method=ownership.METHOD_OAUTH_IDENTITY,
            handle="evan",
        ),
    }
    record.update(overrides)
    return record


@pytest.fixture
def _ingests(monkeypatch):
    """Capture what the intake would have handed to the media pipeline."""
    calls = []

    async def _fake_ingest(**kwargs):
        calls.append(kwargs)
        return {"status": "started", "job_id": f"job-{len(calls)}"}

    monkeypatch.setattr(intake, "ingest_content_url", _fake_ingest)
    return calls


def _connect_account(monkeypatch, record):
    async def _get(store, user_id, key):
        return record if record and key == record.get("account_key") else None

    import src.anubis.utils.connected_accounts.store as store_module

    monkeypatch.setattr(store_module, "get_connected_account", _get)


@pytest.mark.asyncio
async def test_a_proven_account_reaches_the_media_pipeline(monkeypatch, _ingests):
    _connect_account(monkeypatch, _proven_record())
    result = await intake.record_content_event(
        provider="youtube",
        connection_key="youtube:evan",
        personal_avatar_id="avatar-1",
        user_id="owner-1",
        external_item_id="video-1",
        url="https://www.youtube.com/watch?v=abc",
        transport="websub",
    )
    assert result["status"] == "ingested"
    assert len(_ingests) == 1
    assert _ingests[0]["url"] == "https://www.youtube.com/watch?v=abc"


@pytest.mark.asyncio
async def test_an_unproven_account_is_refused(monkeypatch, _ingests):
    """The prerequisite the whole feature rests on: no proof, no identity."""
    record = _proven_record(ownership=ownership.build_ownership(state="unproven"))
    _connect_account(monkeypatch, record)
    result = await intake.record_content_event(
        provider="youtube",
        connection_key="youtube:evan",
        personal_avatar_id="avatar-1",
        user_id="owner-1",
        external_item_id="video-1",
        url="https://www.youtube.com/watch?v=abc",
        transport="websub",
    )
    assert result["status"] == "refused"
    assert _ingests == []


@pytest.mark.asyncio
async def test_content_for_another_avatar_is_refused(monkeypatch, _ingests):
    _connect_account(monkeypatch, _proven_record())
    result = await intake.record_content_event(
        provider="youtube",
        connection_key="youtube:evan",
        personal_avatar_id="avatar-2",
        user_id="owner-1",
        external_item_id="video-1",
        url="https://www.youtube.com/watch?v=abc",
        transport="websub",
    )
    assert result["status"] == "refused"
    assert _ingests == []


@pytest.mark.asyncio
async def test_an_announcement_with_no_connection_is_refused(monkeypatch, _ingests):
    """A feed-shaped source with nothing behind it proves nothing."""
    _connect_account(monkeypatch, None)
    result = await intake.record_content_event(
        provider="podcast_feed",
        connection_key=None,
        personal_avatar_id="avatar-1",
        user_id="owner-1",
        external_item_id="episode-1",
        url="https://example.com/ep1",
        transport="websub",
    )
    assert result["status"] == "refused"
    assert _ingests == []


@pytest.mark.asyncio
async def test_a_redelivery_is_recognised_and_costs_nothing(monkeypatch, _ingests):
    """Every push transport retries; transcribing an hour of video twice is real money."""
    _connect_account(monkeypatch, _proven_record())
    arguments = dict(
        provider="youtube",
        connection_key="youtube:evan",
        personal_avatar_id="avatar-1",
        user_id="owner-1",
        external_item_id="video-1",
        url="https://www.youtube.com/watch?v=abc",
        transport="websub",
    )
    first = await intake.record_content_event(**arguments)
    second = await intake.record_content_event(**arguments)
    assert first["status"] == "ingested"
    assert second["status"] == "duplicate"
    assert len(_ingests) == 1


@pytest.mark.asyncio
async def test_an_announcement_without_an_address_is_refused(monkeypatch, _ingests):
    _connect_account(monkeypatch, _proven_record())
    result = await intake.record_content_event(
        provider="youtube",
        connection_key="youtube:evan",
        personal_avatar_id="avatar-1",
        user_id="owner-1",
        external_item_id="video-1",
        url=None,
        transport="websub",
    )
    assert result["status"] == "refused"
    assert _ingests == []


# -- signature verification -------------------------------------------------


def test_a_websub_delivery_with_a_good_signature_is_accepted():
    body = b"<feed><entry/></feed>"
    secret = "s3cret"
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert transports.verify_websub_signature(
        raw_body=body, header_value=f"sha256={digest}", secret=secret
    )


def test_a_forged_websub_body_is_refused():
    """The signature is the only thing standing between a stranger and identity."""
    secret = "s3cret"
    digest = hmac.new(secret.encode(), b"original", hashlib.sha256).hexdigest()
    assert not transports.verify_websub_signature(
        raw_body=b"tampered", header_value=f"sha256={digest}", secret=secret
    )


def test_a_websub_delivery_naming_an_unknown_algorithm_is_refused():
    """A caller must not be able to name a weak algorithm."""
    body = b"payload"
    assert not transports.verify_websub_signature(
        raw_body=body, header_value="md5=abc123", secret="s3cret"
    )


def test_a_websub_delivery_with_no_signature_is_refused():
    assert not transports.verify_websub_signature(
        raw_body=b"payload", header_value=None, secret="s3cret"
    )


def test_an_eventsub_delivery_signs_id_timestamp_and_body_together():
    secret = "twitch-secret"
    message_id = "msg-1"
    timestamp = "2026-09-10T12:00:00Z"
    body = b'{"event":{}}'
    digest = hmac.new(
        secret.encode(),
        message_id.encode() + timestamp.encode() + body,
        hashlib.sha256,
    ).hexdigest()
    assert transports.verify_eventsub_signature(
        raw_body=body,
        message_id=message_id,
        timestamp=timestamp,
        signature=f"sha256={digest}",
        secret=secret,
    )
    # The same body under a different message id must not verify.
    assert not transports.verify_eventsub_signature(
        raw_body=body,
        message_id="msg-2",
        timestamp=timestamp,
        signature=f"sha256={digest}",
        secret=secret,
    )


def test_a_meta_delivery_is_checked_against_the_application_secret():
    body = b'{"entry":[]}'
    secret = "app-secret"
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert transports.verify_meta_signature(
        raw_body=body, header_value=f"sha256={digest}", app_secret=secret
    )
    assert not transports.verify_meta_signature(
        raw_body=b"other", header_value=f"sha256={digest}", app_secret=secret
    )


def test_a_stale_delivery_is_treated_as_a_replay():
    """A signature stays valid forever, so freshness has to be checked separately."""
    assert transports.is_replay("2020-01-01T00:00:00Z") is True
    assert transports.is_replay(None) is True
    assert transports.is_replay("not a timestamp") is True


# -- payload parsing --------------------------------------------------------


def test_a_youtube_push_yields_the_video_id_as_the_stable_identifier():
    """YouTube redelivers the whole entry on every edit, so the id cannot include it."""
    body = b"""<?xml version="1.0"?>
    <feed xmlns:yt="http://www.youtube.com/xml/schemas/2015"
          xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>yt:video:XYZ</id>
        <yt:videoId>XYZ</yt:videoId>
        <title>A talk</title>
        <link rel="alternate" href="https://www.youtube.com/watch?v=XYZ"/>
        <published>2026-09-10T10:00:00+00:00</published>
      </entry>
    </feed>"""
    items = payloads.parse_websub_atom(body)
    assert len(items) == 1
    assert items[0]["external_item_id"] == "XYZ"
    assert items[0]["url"] == "https://www.youtube.com/watch?v=XYZ"
    assert items[0]["title"] == "A talk"


def test_a_podcast_push_yields_the_episode_guid():
    body = b"""<?xml version="1.0"?>
    <rss><channel>
      <item>
        <guid>episode-42</guid>
        <link>https://example.com/ep42</link>
        <title>Episode 42</title>
      </item>
    </channel></rss>"""
    items = payloads.parse_websub_atom(body)
    assert len(items) == 1
    assert items[0]["external_item_id"] == "episode-42"


def test_a_malformed_push_yields_nothing_rather_than_raising():
    assert payloads.parse_websub_atom(b"not xml at all") == []


def test_the_channel_is_read_out_of_a_youtube_push():
    body = b"<feed><yt:channelId>UC123</yt:channelId></feed>"
    assert payloads.websub_topic_of(body) == "UC123"


def test_a_twitch_event_names_the_channel_to_read():
    items = payloads.parse_eventsub(
        {
            "subscription": {"id": "sub-1"},
            "event": {
                "id": "event-1",
                "broadcaster_user_login": "evan",
                "started_at": "2026-09-10T10:00:00Z",
            },
        }
    )
    assert items[0]["external_item_id"] == "event-1"
    assert items[0]["url"] == "https://www.twitch.tv/evan"


def test_a_meta_change_yields_the_media_permalink():
    items = payloads.parse_meta_change(
        {
            "entry": [
                {
                    "id": "17841400000000000",
                    "changes": [
                        {
                            "field": "media",
                            "value": {
                                "id": "media-9",
                                "permalink": "https://instagram.com/p/abc",
                                "caption": "A photo",
                            },
                        }
                    ],
                }
            ]
        }
    )
    assert items[0]["external_item_id"] == "media-9"
    assert items[0]["url"] == "https://instagram.com/p/abc"
