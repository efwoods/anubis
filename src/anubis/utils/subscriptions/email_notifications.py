"""Recognising "you published something" in the owner's mail, and acting on it.

This is the transport that makes the feature whole. X, TikTok and LinkedIn
offer no webhook anyone can have, but every one of them emails the owner when
the owner publishes — and the owner's mailbox is already connected and already
being watched. So the notice the platform sends *to the person* becomes the
event that updates the person's avatar.

Two rules keep this from becoming a firehose, and both are refusals:

**The sender must be a platform we subscribed to.** Matched on the sending
domain against the provider registry, so a newsletter that merely mentions X is
not a publication event.

**The content must be the owner's own.** A social network emails constantly —
someone replied, someone followed you, here is what you missed. Almost none of
it is the owner publishing. So a message that passes the domain check still
goes to a model that has to find an actual published-by-you notice and the
address of the thing published; anything else is dropped. The address is then
checked against a proven-owned account before a single byte is ingested,
because a notification about somebody else's post is not identity material.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from src.anubis.utils.connected_accounts.providers import (
    KIND_SOCIAL,
    PROVIDER_REGISTRY,
    get_provider,
)

logger = logging.getLogger(__name__)


class PublicationNotice(BaseModel):
    """What a platform's email says the owner just published, if anything."""

    is_publication_notice: bool = Field(
        description=(
            "True only when this message tells the owner that THEY published "
            "something — their post is live, their video is published, their "
            "episode is out. Somebody else's activity, a reply, a follow, a "
            "digest, a promotion, or a security notice is false."
        )
    )
    content_url: str | None = Field(
        default=None,
        description=(
            "The address of the thing the owner published, taken verbatim from "
            "the message. Null when the message names none."
        ),
    )
    title: str | None = Field(
        default=None, description="What the published thing is called, if named."
    )
    published_at: str | None = Field(
        default=None, description="When it was published, if the message says."
    )
    reasoning: str = Field(
        description="The specific wording that decided this, quoted from the message."
    )


PUBLICATION_NOTICE_SYSTEM_PROMPT = """
# Role and Objective

You read one email a social platform sent to an account owner and decide
whether it is telling them that THEY have just published something.

# Why this is strict

A true answer causes the linked content to be fetched, transcribed, analyzed
and folded into a reconstruction of this person's identity, at real cost. A
message about anybody else's activity must never do that.

# Answer true only for

- "Your video is live", "your post was published", "your episode is out".
- A confirmation that content the owner created is now visible to others.

# Answer false for

- Someone replied to you, mentioned you, followed you, liked your post.
- A digest, a recommendation, "what you missed", trending content.
- Security notices, billing, policy updates, marketing.
- Anything about another person's content, even if the owner is mentioned.

# The address

Return `content_url` exactly as it appears in the message. Do not invent,
complete, or correct an address. If the message names no address for the
published thing, return null and answer the rest honestly.
"""


def sender_domain_of(address: str | None) -> str:
    """Return the domain a message came from, lowercased."""
    if not address:
        return ""
    match = re.search(r"[\w.+-]+@([\w.-]+)", address)
    if match:
        return match.group(1).strip().lower().rstrip(".")
    return address.strip().lower().rstrip(".")


def provider_for_sender(sender: str | None) -> Any | None:
    """Return the social provider that emails from this domain, if any.

    Matches a subdomain to the registered domain, because platforms send from
    addresses like ``notify.linkedin.com`` while the registry names
    ``linkedin.com``.
    """
    domain = sender_domain_of(sender)
    if not domain:
        return None
    for provider in PROVIDER_REGISTRY.values():
        if provider.kind != KIND_SOCIAL:
            continue
        for candidate in provider.notification_sender_domains:
            candidate = candidate.lower()
            if domain == candidate or domain.endswith("." + candidate):
                return provider
    return None


async def read_publication_notice(
    context: Any, *, subject: str, body_text: str, sender: str
) -> PublicationNotice | None:
    """Ask whether one message announces the owner's own publication."""
    from src.anubis.utils.model import init_model

    model = init_model(model_without_tools=True, response_format=PublicationNotice)
    prompt = (
        f"From: {sender}\nSubject: {subject}\n\n{(body_text or '')[:8000]}"
    )
    try:
        answer = await model.ainvoke(
            [
                {"role": "system", "content": PUBLICATION_NOTICE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
    except Exception as read_error:  # noqa: BLE001 - an unread notice is dropped
        logger.warning("Could not read a publication notice: %s", read_error)
        return None
    parsed = getattr(answer, "parsed", None) or answer
    if isinstance(parsed, PublicationNotice):
        return parsed
    if hasattr(parsed, "model_dump"):
        try:
            return PublicationNotice(**parsed.model_dump())
        except Exception:  # noqa: BLE001
            return None
    return None


def url_belongs_to_account(url: str, record: dict[str, Any]) -> bool:
    """Whether a published address plausibly belongs to this proven account.

    Host is checked against the provider's own site, and where the account has
    a known handle the address must carry it. Without this a notification from
    a platform the owner happens to use could carry any address on that
    platform — including another person's post — and have it ingested as the
    owner's own words.
    """
    provider = get_provider(str(record.get("provider") or ""))
    if provider is None:
        return False
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:  # noqa: BLE001
        return False
    if not host:
        return False

    provider_hosts = {
        (urlparse(address).hostname or "").lower()
        for address in (provider.home_url, provider.login_url)
        if address
    }
    provider_hosts.discard("")
    matched_host = any(
        host == candidate or host.endswith("." + candidate.removeprefix("www."))
        for candidate in provider_hosts
    )
    if not matched_host:
        return False

    from src.anubis.utils.connected_accounts.ownership import ownership_of

    handle = str(ownership_of(record).get("handle") or "").lower()
    if not handle:
        # Without a known handle the host match is all there is. That is weaker
        # than we would like, which is why an account with no readable handle
        # is not marked proven in the first place.
        return True
    return handle in url.lower()


async def handle_mail_as_publication(
    context: Any,
    *,
    store: Any,
    user_id: str,
    personal_avatar_id: str,
    sender: str,
    subject: str,
    body_text: str,
    message_id: str,
) -> dict[str, Any] | None:
    """Treat one email as a possible publication notice; ingest when it is one.

    Returns ``None`` when the message is not a publication notice at all, so
    the caller can carry on treating it as ordinary mail. The inbox's own
    triage is unaffected either way — a message can be both something the
    avatar learns from and something the owner needs to see.
    """
    provider = provider_for_sender(sender)
    if provider is None:
        return None

    notice = await read_publication_notice(
        context, subject=subject, body_text=body_text, sender=sender
    )
    if notice is None or not notice.is_publication_notice or not notice.content_url:
        return None

    record = await _proven_account_for(
        store,
        user_id=user_id,
        personal_avatar_id=personal_avatar_id,
        provider_name=provider.name,
    )
    if record is None:
        logger.info(
            "Dropped a %s publication notice: no proven account for this avatar.",
            provider.name,
        )
        return {
            "status": "refused",
            "detail": (
                f"A {provider.display_name} notice arrived, but no proven "
                f"{provider.display_name} account is connected to this avatar."
            ),
        }
    if not url_belongs_to_account(notice.content_url, record):
        logger.info(
            "Dropped a %s notice whose address is not the owner's.", provider.name
        )
        return {
            "status": "refused",
            "detail": "The announced address does not belong to your account.",
        }

    from src.anubis.utils.subscriptions.intake import record_content_event
    from src.anubis.utils.subscriptions.repository import (
        get_subscription_repository,
    )

    repository = get_subscription_repository()
    subscription = await repository.find_subscription(
        connection_key=str(record.get("account_key") or "")
    )
    return await record_content_event(
        provider=provider.name,
        connection_key=str(record.get("account_key") or ""),
        personal_avatar_id=personal_avatar_id,
        user_id=user_id,
        external_item_id=notice.content_url or message_id,
        url=notice.content_url,
        title=notice.title,
        published_at=notice.published_at,
        transport="email_notification",
        subscription_id=(subscription or {}).get("subscription_id"),
        avatar_name=(subscription or {}).get("avatar_name"),
        avatar_description=(subscription or {}).get("avatar_description"),
        store=store,
    )


async def _proven_account_for(
    store: Any, *, user_id: str, personal_avatar_id: str, provider_name: str
) -> dict[str, Any] | None:
    """Return this avatar's proven account on one platform, if it has one."""
    from src.anubis.utils.connected_accounts.ownership import (
        is_owned_by_personal_avatar,
    )
    from src.anubis.utils.connected_accounts.store import read_connected_accounts

    records = await read_connected_accounts(store, user_id)
    for record in records:
        if record.get("provider") != provider_name:
            continue
        if is_owned_by_personal_avatar(
            record, personal_avatar_id=personal_avatar_id
        ):
            return record
    return None
