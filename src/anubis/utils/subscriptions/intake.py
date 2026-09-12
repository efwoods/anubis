"""The one place a newly published piece of content becomes an identity update.

Every transport ends here. A YouTube WebSub push, a Twitch EventSub callback, a
Meta Graph delivery, and a LinkedIn notification read out of the owner's
mailbox all arrive as the same call, and the differences between them stop at
this door. That is the point of the design: the platforms differ wildly in how
they announce, and nothing downstream should have to know which one announced.

Four things happen to every event, in this order, and the order matters:

1. **Dedupe.** Every push transport in use redelivers, and the same new video
   can legitimately arrive twice — once as a webhook, once as the platform's
   email to the owner. Transcribing an hour of video twice is a real bill, so
   the uniqueness check comes before any work.
2. **Ownership.** The account the content came from must be a social account
   bound to this personal avatar whose ownership has been proven. This is the
   prerequisite the whole feature rests on: an avatar reconstructs one person,
   and content from an account nobody proved is theirs would teach it to speak
   as somebody else.
3. **Billing.** The owner's live record is resolved and the ingest goes through
   ``start_identity_media_job_from_chat``, the same path an interactive upload
   takes. Nothing here bypasses the tier gate, the allotment check, the rate
   limit, or the Stripe meter — a subscription that ingested for free would be
   an unbounded cost with no ceiling and no visibility.
4. **Ingest.** The existing media pipeline does the actual work: fetch,
   transcribe or describe, classify, analyze, index.

Nothing in this module raises. A transport calling it is usually answering a
platform that will retry and then unsubscribe a failing endpoint, so every
failure is recorded on the event row and reported back as a value.
"""

from __future__ import annotations

import logging
from typing import Any

from src.anubis.utils.connected_accounts.ownership import (
    is_owned_by_personal_avatar,
    refusal_reason,
)
from src.anubis.utils.subscriptions.repository import (
    EVENT_FAILED,
    EVENT_INGESTED,
    EVENT_INGESTING,
    EVENT_RECEIVED,
    EVENT_REFUSED,
    get_subscription_repository,
)

logger = logging.getLogger(__name__)


async def record_content_event(
    *,
    provider: str,
    connection_key: str | None,
    personal_avatar_id: str,
    user_id: str,
    external_item_id: str,
    url: str | None,
    title: str | None = None,
    published_at: Any = None,
    transport: str,
    subscription_id: str | None = None,
    avatar_name: str | None = None,
    avatar_description: str | None = None,
    store: Any = None,
) -> dict[str, Any]:
    """Accept one announcement of newly published content and ingest it.

    Returns a status dictionary rather than raising, and is safe to call twice
    with the same ``external_item_id`` — the second call reports ``duplicate``
    and does no work.
    """
    repository = get_subscription_repository()

    existing = await repository.find_event(
        personal_avatar_id=personal_avatar_id,
        provider=provider,
        external_item_id=external_item_id,
    )
    if existing:
        return {
            "status": "duplicate",
            "event_id": existing.get("event_id"),
            "detail": "This content was already received.",
        }

    event = await repository.create_event(
        {
            "subscription_id": subscription_id,
            "user_id": user_id,
            "personal_avatar_id": personal_avatar_id,
            "connection_key": connection_key,
            "provider": provider,
            "transport": transport,
            "external_item_id": external_item_id,
            "url": url,
            "title": title,
            "published_at": published_at,
            "state": EVENT_RECEIVED,
        }
    )
    event_id = str(event.get("event_id") or "")
    if subscription_id:
        await repository.touch_subscription(subscription_id)

    if not url:
        await repository.set_event_state(
            event_id, state=EVENT_REFUSED, detail="The announcement carried no address."
        )
        return {"status": "refused", "event_id": event_id, "detail": "No address."}

    refusal = await _refuse_unless_owned(
        connection_key=connection_key,
        personal_avatar_id=personal_avatar_id,
        user_id=user_id,
        store=store,
    )
    if refusal:
        await repository.set_event_state(
            event_id, state=EVENT_REFUSED, detail=refusal
        )
        logger.info("Refused content from %s: %s", provider, refusal)
        return {"status": "refused", "event_id": event_id, "detail": refusal}

    await repository.set_event_state(event_id, state=EVENT_INGESTING)
    outcome = await ingest_content_url(
        user_id=user_id,
        personal_avatar_id=personal_avatar_id,
        url=url,
        avatar_name=avatar_name,
        avatar_description=avatar_description,
    )
    if outcome.get("status") in {"started", "accepted", "ok"}:
        await repository.set_event_state(
            event_id,
            state=EVENT_INGESTED,
            media_job_id=str(outcome.get("job_id") or "") or None,
        )
        return {
            "status": "ingested",
            "event_id": event_id,
            "job_id": outcome.get("job_id"),
        }

    await repository.set_event_state(
        event_id,
        state=EVENT_FAILED if outcome.get("status") == "error" else EVENT_REFUSED,
        detail=str(outcome.get("detail") or outcome.get("error") or "")[:500],
    )
    return {
        "status": outcome.get("status") or "failed",
        "event_id": event_id,
        "detail": outcome.get("detail") or outcome.get("error"),
    }


async def _refuse_unless_owned(
    *,
    connection_key: str | None,
    personal_avatar_id: str,
    user_id: str,
    store: Any,
) -> str | None:
    """Return why this content may not build identity, or ``None`` when it may.

    A feed-shaped source carries no connection of its own, which is exactly the
    case the ownership rule exists for: without a connected, proven account
    behind it, nothing arrives.
    """
    if not connection_key:
        return (
            "This content is not attached to a connected account, so there is "
            "nothing proving it belongs to you."
        )
    from src.anubis.utils.connected_accounts.store import get_connected_account

    record = await get_connected_account(store, user_id, connection_key)
    if not record:
        return "The account this content came from is no longer connected."
    if is_owned_by_personal_avatar(record, personal_avatar_id=personal_avatar_id):
        return None
    return refusal_reason(record, personal_avatar_id=personal_avatar_id) or (
        "This account has not been proven to belong to you."
    )


async def ingest_content_url(
    *,
    user_id: str,
    personal_avatar_id: str,
    url: str,
    avatar_name: str | None = None,
    avatar_description: str | None = None,
) -> dict[str, Any]:
    """Hand one address to the ordinary media pipeline, fully metered.

    Deliberately a thin wrapper over the starter the in-chat identity tool uses
    rather than a second path into the media graph. Everything that makes an
    upload safe — the creator check, the tier capability, the token estimate,
    the allotment, the rate limit, the Stripe meter — lives behind that starter,
    and a background caller needs all of it more than an interactive one does,
    not less.
    """
    from src.anubis.utils.runtime_handles import get_identity_media_job_starter

    starter = get_identity_media_job_starter()
    if starter is None:
        return {
            "status": "error",
            "detail": "The media pipeline is not available in this process.",
        }

    from src.security.auth import get_user_by_identity_user_id

    # The owner's tier is read live rather than remembered, because an account
    # that downgraded must stop being billed at the old rate. A lookup that
    # fails refuses the ingest instead of running it unbilled.
    owner = await get_user_by_identity_user_id(user_id)
    if owner is None:
        return {
            "status": "error",
            "detail": (
                "The account that owns this avatar could not be resolved, so the "
                "ingest was not started rather than being run unbilled."
            ),
        }

    # The avatar's own name and description ride the subscription row. Reading
    # them back over HTTP is not open to a background caller — the LangGraph
    # client authenticates with the owner's plaintext API key, which nothing
    # unattended holds and nothing should store. The name matters because the
    # media pipeline uses it as the target's name when deciding which turns in
    # a recording are the avatar's own.
    assistant_ctx = {
        "name": avatar_name,
        "description": avatar_description,
        "assistant_id": personal_avatar_id,
        "metadata": {"user_id": user_id},
    }

    try:
        return await starter(
            user_id=user_id,
            assistant_id=personal_avatar_id,
            assistant_ctx=assistant_ctx,
            current_user=owner,
            attachments=[],
            urls=[url],
        )
    except Exception as ingest_error:  # noqa: BLE001 - a transport must not fail
        logger.exception("Subscription ingest failed for %s", url)
        return {"status": "error", "detail": str(ingest_error)}
