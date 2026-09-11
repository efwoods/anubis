"""Registering content subscriptions with each platform, and verifying what it sends.

Three push protocols, one shape. Each platform wants a different handshake and
signs its deliveries differently, and every one of them will retry a failing
callback and then drop the subscription — so the two things that matter here
are getting the handshake right once and verifying the signature on every
delivery afterwards.

**Why signature verification is not optional.** The callback route cannot be
authenticated: the caller is YouTube, not the owner, and it carries no session.
The signature is therefore the *only* thing standing between an announcement
and the avatar's identity. Without it anyone who learned a callback address
could post a payload naming any video and have it ingested as the owner's own
words. Every verifier below compares against the RAW request body, before
parsing, because a payload re-serialized after parsing is no longer the bytes
the platform signed.

**Leases.** WebSub subscriptions expire — typically in five to ten days — and a
lapsed lease is silent: nothing fails, content simply stops arriving. The
renewal task is what keeps a subscription real, and it is a timer over stored
expiry times rather than a poll of the platforms.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urljoin, urlparse

from src.anubis.utils.connected_accounts.providers import (
    CONTENT_TRANSPORT_EMAIL_NOTIFICATION,
    CONTENT_TRANSPORT_EVENTSUB,
    CONTENT_TRANSPORT_META_GRAPH,
    CONTENT_TRANSPORT_NONE,
    CONTENT_TRANSPORT_WEBSUB,
    get_provider,
)
from src.anubis.utils.subscriptions.repository import (
    SUBSCRIPTION_ACTIVE,
    SUBSCRIPTION_FAILED,
    SUBSCRIPTION_PENDING,
    get_subscription_repository,
)

logger = logging.getLogger(__name__)

# Google's public hub, which YouTube's per-channel feeds are published through.
YOUTUBE_HUB_URL = "https://pubsubhubbub.appspot.com/subscribe"
YOUTUBE_TOPIC_TEMPLATE = (
    "https://www.youtube.com/xml/feeds/videos.xml?channel_id={channel_id}"
)
TWITCH_EVENTSUB_URL = "https://api.twitch.tv/helix/eventsub/subscriptions"


def callback_url_for(context: Any, provider_name: str) -> str | None:
    """Return the public address this provider's platform should call.

    A subscription is only as good as the address it hands out, and a private
    development address is worse than none: the platform accepts it, the
    handshake fails, and the owner is told they are subscribed to something that
    can never deliver. So a missing base address refuses the subscription
    outright rather than registering a callback nobody can reach.
    """
    base = str(getattr(context, "social_webhook_callback_base_url", "") or "").strip()
    if not base:
        return None
    if not base.startswith("https://"):
        # Every platform in use requires TLS on a callback, so an http:// base
        # would be rejected at the handshake with a message that says nothing
        # useful. Refuse here, where the reason can be stated.
        logger.warning("The social webhook callback base must be an https:// address.")
        return None
    return urljoin(base.rstrip("/") + "/", f"social_webhook/{provider_name}")


def new_subscription_secret() -> str:
    """Mint the shared secret a platform signs its deliveries with."""
    return secrets.token_hex(32)


def _lease_seconds(context: Any) -> int:
    try:
        return max(3600, int(getattr(context, "social_webhook_lease_seconds", 0) or 0))
    except (TypeError, ValueError):
        return 432000


async def subscribe_to_content(
    context: Any,
    *,
    record: dict[str, Any],
    personal_avatar_id: str,
    user_id: str,
    avatar_name: str | None = None,
    avatar_description: str | None = None,
) -> dict[str, Any]:
    """Subscribe to everything a proven account publishes from now on.

    Chooses the transport from the provider registry rather than from anything
    about how the account was connected, records the subscription, and performs
    the platform handshake where there is one. An account whose platform neither
    pushes nor notifies is reported as not subscribable — deliberately, rather
    than being swept on a timer.
    """
    provider = get_provider(str(record.get("provider") or ""))
    if provider is None:
        return {"status": "error", "detail": "Unknown provider."}
    transport = provider.content_transport
    if transport == CONTENT_TRANSPORT_NONE:
        return {
            "status": "not_subscribable",
            "detail": (
                f"{provider.display_name} offers no way to announce new content, "
                "so nothing can arrive on its own."
            ),
        }

    repository = get_subscription_repository()
    topic, topic_url = await _topic_for(context, record, provider)
    existing = await repository.find_subscription(
        connection_key=str(record.get("account_key") or ""), topic=topic
    )

    subscription = await repository.upsert_subscription(
        {
            "subscription_id": (existing or {}).get("subscription_id"),
            "user_id": user_id,
            "personal_avatar_id": personal_avatar_id,
            "connection_key": record.get("account_key"),
            "provider": provider.name,
            "transport": transport,
            "topic": topic,
            "topic_url": topic_url,
            "avatar_name": avatar_name,
            "avatar_description": avatar_description,
            "callback_url": callback_url_for(context, provider.name),
            "secret": (existing or {}).get("secret") or new_subscription_secret(),
            "status": SUBSCRIPTION_PENDING,
        }
    )

    # A notification transport needs no handshake with anyone: the platform
    # already emails the owner, and the mailbox watcher is what listens. It is
    # active the moment it is recorded.
    if transport == CONTENT_TRANSPORT_EMAIL_NOTIFICATION:
        await repository.set_subscription_status(
            subscription["subscription_id"],
            status=SUBSCRIPTION_ACTIVE,
            detail=(
                f"{provider.display_name} emails you when you publish; those "
                "notices are read from your connected mailbox."
            ),
        )
        return {
            "status": "subscribed",
            "transport": transport,
            "subscription_id": subscription["subscription_id"],
        }

    handshake = await _perform_handshake(
        context, provider, subscription, record=record, transport=transport
    )
    if handshake.get("status") == "error":
        await repository.set_subscription_status(
            subscription["subscription_id"],
            status=SUBSCRIPTION_FAILED,
            detail=str(handshake.get("detail") or "")[:500],
        )
    return {**handshake, "subscription_id": subscription["subscription_id"]}


async def _topic_for(
    context: Any, record: dict[str, Any], provider: Any
) -> tuple[str | None, str | None]:
    """Return the platform's identifier for what we are subscribing to."""
    transport = record.get("transport") or {}
    if provider.name == "youtube":
        channel_id = str(transport.get("youtube_channel_id") or "")
        if channel_id:
            return channel_id, YOUTUBE_TOPIC_TEMPLATE.format(channel_id=channel_id)
        return None, None
    if provider.name == "twitch":
        return str(transport.get("twitch_user_id") or "") or None, None
    if provider.name in {"instagram", "facebook"}:
        return str(transport.get("meta_object_id") or "") or None, None
    if provider.name in {"podcast_feed", "profile_url"}:
        feed_url = str(transport.get("feed_url") or transport.get("site_url") or "")
        if not feed_url:
            return None, None
        return feed_url, feed_url
    return None, None


async def _perform_handshake(
    context: Any,
    provider: Any,
    subscription: dict[str, Any],
    *,
    record: dict[str, Any],
    transport: str,
) -> dict[str, Any]:
    """Ask the platform to start calling our callback."""
    callback_url = subscription.get("callback_url")
    if not callback_url:
        return {
            "status": "error",
            "detail": (
                "No public callback address is configured, so the platform has "
                "nowhere to deliver. Set SOCIAL_WEBHOOK_CALLBACK_BASE_URL."
            ),
        }
    if not subscription.get("topic"):
        return {
            "status": "error",
            "detail": (
                f"The {provider.display_name} account does not yet name what to "
                "subscribe to."
            ),
        }
    if transport == CONTENT_TRANSPORT_WEBSUB:
        return await _websub_handshake(context, subscription)
    if transport == CONTENT_TRANSPORT_EVENTSUB:
        return await _eventsub_handshake(context, subscription, record=record)
    if transport == CONTENT_TRANSPORT_META_GRAPH:
        return await _meta_handshake(context, subscription, record=record)
    return {"status": "error", "detail": f"No handshake for {transport!r}."}


async def _websub_handshake(
    context: Any, subscription: dict[str, Any]
) -> dict[str, Any]:
    """Ask a WebSub hub to verify and then deliver.

    The hub answers this request immediately but subscribes asynchronously: it
    calls our callback back with a challenge, and only that second exchange
    makes the subscription real. So this leaves the row pending and the callback
    route is what marks it active.
    """
    import httpx

    topic_url = str(subscription.get("topic_url") or subscription.get("topic") or "")
    hub_url = (
        YOUTUBE_HUB_URL
        if "youtube.com" in topic_url
        else await _discover_hub(topic_url)
    )
    if not hub_url:
        return {
            "status": "not_subscribable",
            "detail": (
                "This feed advertises no hub, so it cannot push. Connect an "
                "account that notifies you instead."
            ),
        }
    form = {
        "hub.mode": "subscribe",
        "hub.topic": topic_url,
        "hub.callback": subscription.get("callback_url"),
        "hub.verify": "async",
        "hub.secret": subscription.get("secret"),
        "hub.lease_seconds": str(_lease_seconds(context)),
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(hub_url, data=form)
    except Exception as hub_error:  # noqa: BLE001 - reported, never raised
        return {"status": "error", "detail": f"The hub could not be reached: {hub_error}"}
    if response.status_code >= 300:
        return {
            "status": "error",
            "detail": f"The hub refused the subscription ({response.status_code}).",
        }
    return {"status": "pending", "transport": CONTENT_TRANSPORT_WEBSUB}


async def _discover_hub(feed_url: str) -> str | None:
    """Find the hub a feed publishes through, from the feed itself."""
    import re

    import httpx

    if not feed_url:
        return None
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(feed_url)
            if response.status_code >= 400:
                return None
            body = response.text
    except Exception:  # noqa: BLE001 - an unreachable feed has no hub
        return None
    match = re.search(
        r'<link[^>]+rel=["\']hub["\'][^>]+href=["\']([^"\']+)["\']'
        r'|<link[^>]+href=["\']([^"\']+)["\'][^>]+rel=["\']hub["\']',
        body,
        re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1) or match.group(2)


async def _eventsub_handshake(
    context: Any, subscription: dict[str, Any], *, record: dict[str, Any]
) -> dict[str, Any]:
    """Register a Twitch EventSub subscription for the broadcaster's streams."""
    import httpx

    client_id = str(getattr(context, "twitch_client_id", "") or "")
    app_token = str(getattr(context, "twitch_app_access_token", "") or "")
    if not (client_id and app_token):
        return {
            "status": "not_configured",
            "detail": (
                "Twitch push needs the application's own credentials "
                "(TWITCH_CLIENT_ID, TWITCH_APP_ACCESS_TOKEN). Until they are "
                "set, Twitch notices are read from your mailbox instead."
            ),
        }
    payload = {
        "type": "stream.online",
        "version": "1",
        "condition": {"broadcaster_user_id": subscription.get("topic")},
        "transport": {
            "method": "webhook",
            "callback": subscription.get("callback_url"),
            "secret": subscription.get("secret"),
        },
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                TWITCH_EVENTSUB_URL,
                json=payload,
                headers={
                    "Client-Id": client_id,
                    "Authorization": f"Bearer {app_token}",
                    "Content-Type": "application/json",
                },
            )
    except Exception as eventsub_error:  # noqa: BLE001
        return {"status": "error", "detail": f"Twitch refused: {eventsub_error}"}
    if response.status_code >= 300:
        return {
            "status": "error",
            "detail": f"Twitch refused the subscription ({response.status_code}).",
        }
    body = response.json() if response.content else {}
    external_id = ((body.get("data") or [{}])[0] or {}).get("id")
    return {
        "status": "pending",
        "transport": CONTENT_TRANSPORT_EVENTSUB,
        "external_id": external_id,
    }


async def _meta_handshake(
    context: Any, subscription: dict[str, Any], *, record: dict[str, Any]
) -> dict[str, Any]:
    """Record a Meta Graph subscription.

    Meta's webhooks are configured against the application in Meta's own
    console and require app review before they deliver for a real account, so
    there is no per-account call to make here. The row is what lets a delivery
    be matched to an avatar once the application is approved; until then the
    same content still arrives through the owner's email notifications.
    """
    if not str(getattr(context, "meta_webhook_verify_token", "") or ""):
        return {
            "status": "not_configured",
            "detail": (
                "Meta push needs the application's webhook verify token and app "
                "review. Until then, Instagram and Facebook notices are read "
                "from your mailbox."
            ),
        }
    return {"status": "pending", "transport": CONTENT_TRANSPORT_META_GRAPH}


def verify_websub_signature(
    *, raw_body: bytes, header_value: str | None, secret: str
) -> bool:
    """Check a WebSub delivery against the secret the hub was given.

    The header names its own algorithm (``sha1=...`` historically, ``sha256=...``
    on modern hubs), so the algorithm is read from the header rather than
    assumed — but only from a known set, because letting a caller name any
    algorithm lets it name a weak one.
    """
    if not (header_value and secret):
        return False
    _, _, digest = header_value.partition("=")
    algorithm_name = header_value.split("=", 1)[0].strip().lower()
    algorithms = {
        "sha1": hashlib.sha1,
        "sha256": hashlib.sha256,
        "sha384": hashlib.sha384,
        "sha512": hashlib.sha512,
    }
    algorithm = algorithms.get(algorithm_name)
    if algorithm is None or not digest:
        return False
    expected = hmac.new(secret.encode("utf-8"), raw_body, algorithm).hexdigest()
    return hmac.compare_digest(expected, digest.strip())


def verify_eventsub_signature(
    *,
    raw_body: bytes,
    message_id: str | None,
    timestamp: str | None,
    signature: str | None,
    secret: str,
) -> bool:
    """Check a Twitch delivery, which signs id + timestamp + body together."""
    if not (message_id and timestamp and signature and secret):
        return False
    message = message_id.encode("utf-8") + timestamp.encode("utf-8") + raw_body
    expected = (
        "sha256="
        + hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    )
    return hmac.compare_digest(expected, signature.strip())


def verify_meta_signature(
    *, raw_body: bytes, header_value: str | None, app_secret: str
) -> bool:
    """Check a Meta Graph delivery against the application secret."""
    if not (header_value and app_secret):
        return False
    expected = (
        "sha256="
        + hmac.new(app_secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    )
    return hmac.compare_digest(expected, header_value.strip())


def is_replay(timestamp: str | None, *, tolerance_seconds: int = 600) -> bool:
    """Whether a signed delivery is too old to accept.

    A signature stays valid forever, so without a freshness window a captured
    delivery could be replayed indefinitely to re-trigger an ingest.
    """
    if not timestamp:
        return True
    try:
        moment = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return True
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return abs((datetime.now(UTC) - moment).total_seconds()) > tolerance_seconds


def lease_expiry(context: Any, lease_seconds: Any = None) -> datetime:
    """Return when a lease just granted will run out."""
    try:
        seconds = int(lease_seconds) if lease_seconds else _lease_seconds(context)
    except (TypeError, ValueError):
        seconds = _lease_seconds(context)
    return datetime.now(UTC) + timedelta(seconds=max(3600, seconds))


def topic_hostname(topic_url: str) -> str:
    """Return the host a topic address belongs to, for matching a delivery."""
    try:
        return (urlparse(topic_url).hostname or "").lower()
    except Exception:  # noqa: BLE001
        return ""


async def renew_due_subscriptions(context: Any) -> dict[str, Any]:
    """Refresh every lease about to run out. Returns what it did.

    Renewal is a re-subscribe: WebSub has no separate renew verb, and asking
    again with the same topic and secret extends the lease. Failures are
    recorded per subscription and never stop the others, because one dead feed
    must not silence a working channel.
    """
    from datetime import timedelta

    repository = get_subscription_repository()
    margin = float(
        getattr(context, "social_subscription_renewal_margin_seconds", 0) or 86400.0
    )
    horizon = datetime.now(UTC) + timedelta(seconds=margin)
    due = await repository.list_due_for_renewal(before=horizon)
    renewed = 0
    failed = 0
    for subscription in due:
        transport = str(subscription.get("transport") or "")
        if transport != CONTENT_TRANSPORT_WEBSUB:
            # Only WebSub leases expire. Twitch subscriptions persist until
            # revoked, Meta's belong to the application, and a notification
            # transport has no lease at all.
            continue
        outcome = await _websub_handshake(context, subscription)
        if outcome.get("status") == "error":
            failed += 1
            await repository.set_subscription_status(
                subscription["subscription_id"],
                status=SUBSCRIPTION_FAILED,
                detail=str(outcome.get("detail") or "")[:500],
            )
        else:
            renewed += 1
            await repository.set_subscription_status(
                subscription["subscription_id"],
                status=SUBSCRIPTION_PENDING,
                detail="Renewal requested; awaiting the hub's confirmation.",
            )
    return {"due": len(due), "renewed": renewed, "failed": failed}


async def renew_subscriptions_forever(context: Any) -> None:
    """Keep leases alive for as long as the application runs."""
    import asyncio

    interval = float(
        getattr(context, "social_subscription_renewal_interval_seconds", 0) or 3600.0
    )
    while True:
        try:
            await asyncio.sleep(max(60.0, interval))
            report = await renew_due_subscriptions(context)
            if report.get("due"):
                logger.info("Subscription renewal: %s", report)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the loop outlives one bad pass
            logger.exception("A subscription renewal pass failed.")
