"""Turning each platform's delivery format into the one event shape the intake takes.

Every platform announces in its own vocabulary — YouTube sends Atom XML,
Twitch sends a JSON event envelope, Meta sends a nested change feed, and a
podcast hub sends RSS. All of them are saying the same small thing: something
was published, here is its address, here is an identifier stable enough to
recognise a redelivery by.

The identifier is the part that repays care. It is what the intake dedupes on,
so it has to be the platform's own id for the *content* — never a delivery id,
which is different on every retry, and never the URL alone, which can change
while the content stays the same.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def parse_websub_atom(raw_body: bytes) -> list[dict[str, Any]]:
    """Read a WebSub push (YouTube's Atom, or an RSS feed) into content items.

    YouTube redelivers the whole feed entry on every edit — a retitled video
    arrives again with the same ``yt:videoId`` — which is exactly why the video
    id is the identifier and the entry's ``updated`` stamp is not part of it.
    """
    try:
        from xml.etree import ElementTree
    except ImportError:  # pragma: no cover - stdlib always present
        return []

    try:
        root = ElementTree.fromstring(raw_body.decode("utf-8", errors="replace"))
    except Exception as parse_error:  # noqa: BLE001 - a malformed push is dropped
        logger.warning("Could not parse a WebSub payload: %s", parse_error)
        return []

    namespaces = {
        "atom": "http://www.w3.org/2005/Atom",
        "yt": "http://www.youtube.com/xml/schemas/2015",
    }
    items: list[dict[str, Any]] = []

    for entry in root.findall("atom:entry", namespaces):
        video_id = entry.findtext("yt:videoId", default="", namespaces=namespaces)
        entry_id = entry.findtext("atom:id", default="", namespaces=namespaces)
        title = entry.findtext("atom:title", default="", namespaces=namespaces)
        published = entry.findtext(
            "atom:published", default="", namespaces=namespaces
        )
        url = ""
        link = entry.find("atom:link", namespaces)
        if link is not None:
            url = link.attrib.get("href", "")
        if video_id and not url:
            url = f"https://www.youtube.com/watch?v={video_id}"
        external_id = video_id or entry_id or url
        if not external_id:
            continue
        items.append(
            {
                "external_item_id": external_id,
                "url": url or None,
                "title": title or None,
                "published_at": published or None,
            }
        )

    if items:
        return items

    # A plain RSS feed rather than Atom: the same content, a different shape.
    for item in root.iter("item"):
        guid = (item.findtext("guid") or "").strip()
        link = (item.findtext("link") or "").strip()
        title = (item.findtext("title") or "").strip()
        published = (item.findtext("pubDate") or "").strip()
        external_id = guid or link
        if not external_id:
            continue
        items.append(
            {
                "external_item_id": external_id,
                "url": link or None,
                "title": title or None,
                "published_at": published or None,
            }
        )
    return items


def parse_eventsub(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Read a Twitch EventSub notification into content items.

    A stream going live is an event about something that does not exist as a
    recording yet, so what is ingested is the channel address; the recording
    itself arrives later through the same channel page.
    """
    event = payload.get("event") or {}
    subscription = payload.get("subscription") or {}
    broadcaster = (
        event.get("broadcaster_user_login")
        or event.get("broadcaster_user_name")
        or ""
    )
    external_id = str(event.get("id") or subscription.get("id") or "")
    if not (external_id and broadcaster):
        return []
    return [
        {
            "external_item_id": external_id,
            "url": f"https://www.twitch.tv/{broadcaster}",
            "title": event.get("title") or None,
            "published_at": event.get("started_at") or None,
        }
    ]


def parse_meta_change(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Read a Meta Graph change notification into content items."""
    items: list[dict[str, Any]] = []
    for entry in payload.get("entry") or []:
        entry_id = str(entry.get("id") or "")
        for change in entry.get("changes") or []:
            value = change.get("value") or {}
            media_id = str(value.get("id") or value.get("media_id") or "")
            permalink = value.get("permalink") or value.get("permalink_url")
            external_id = media_id or f"{entry_id}:{change.get('field')}"
            if not external_id:
                continue
            items.append(
                {
                    "external_item_id": external_id,
                    "url": permalink,
                    "title": value.get("caption") or value.get("message") or None,
                    "published_at": value.get("created_time") or None,
                    "object_id": entry_id,
                }
            )
        # Instagram delivers some updates as messaging-shaped entries.
        for media in entry.get("media") or []:
            media_id = str(media.get("id") or "")
            if not media_id:
                continue
            items.append(
                {
                    "external_item_id": media_id,
                    "url": media.get("permalink"),
                    "title": media.get("caption") or None,
                    "published_at": media.get("timestamp") or None,
                    "object_id": entry_id,
                }
            )
    return items


def websub_topic_of(raw_body: bytes) -> str | None:
    """Return the feed a WebSub push belongs to, read from the payload itself.

    A hub does not repeat the topic in a header, so matching a delivery to a
    subscription means reading the feed's own self link. For YouTube the
    channel id is carried explicitly, which is the value the subscription is
    keyed on.
    """
    text = raw_body.decode("utf-8", errors="replace")
    channel = re.search(r"<yt:channelId>([^<]+)</yt:channelId>", text)
    if channel:
        return channel.group(1).strip()
    self_link = re.search(
        r'<link[^>]+rel=["\']self["\'][^>]+href=["\']([^"\']+)["\']', text
    )
    if self_link:
        return self_link.group(1).strip()
    return None
