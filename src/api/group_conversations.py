"""The one endpoint every group-conversation bot talks to, and the owner's controls.

A Slack, Discord, or Twitch bot posts a batch of messages to
``POST /groups/{assistant_id}/events`` and receives one decision per message,
which the bot carries out itself. Everything else here is the owner's side:
correcting a decision after the fact, keeping the rules the avatar follows,
reading what is waiting, and seeing or leaving the rooms the avatar is in.

Pending decisions are ``inbox_items`` rows, so ``GET /inbox/items``,
``GET /inbox/count``, ``POST /inbox/items/{item_id}/decide``, the inbox panel
and its badge all light up for group conversations with no new user interface.
That reuse is the main reason this feature needs no migration.

Authorization lives in one place, ``resolve_avatar_for_group``: every platform
requires the caller to have created the avatar, and Twitch additionally
requires the avatar to be the caller's own personal avatar. Handlers import
from ``webapp`` lazily because ``webapp`` includes this router.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.groups.events import (
    DecisionCorrection,
    GroupEventsRequest,
)
from src.security.auth import get_current_user

logger = logging.getLogger(__name__)

group_conversations_route = APIRouter(tags=["group conversations"])

# Twitch speaks for the owner as a person on the owner's own channel, so only
# the owner's personal avatar may take part there. Discord and Slack accept any
# avatar the caller created.
PERSONAL_AVATAR_ONLY_PLATFORMS = ("twitch",)


def _enabled() -> bool:
    return str(getattr(GlobalContext(), "group_conversation_enabled", "true") or "").strip().lower() in (
        "true",
        "1",
        "yes",
    )


def _require_enabled() -> None:
    if not _enabled():
        raise HTTPException(status_code=404, detail="Group conversations are not enabled.")


def _store_or_503() -> Any:
    from src.api.webapp import app

    store = getattr(app.state, "store", None)
    if store is None:
        raise HTTPException(
            status_code=503, detail="The avatar store is not configured."
        )
    return store


async def resolve_avatar_for_group(
    assistant_id: str, current_user: dict, platform: str
) -> tuple[dict, str]:
    """Authorize one caller for one avatar on one platform.

    Returns ``(assistant, creator_user_id)``. The creator identifier is what
    every store namespace for this avatar is keyed under, so every handler here
    uses the returned value rather than reading the caller's identity again.
    """
    from src.anubis.utils.personal_avatar import is_personal_avatar
    from src.api.webapp import resolve_assistant_for_creator

    assistant, creator_user_id = await resolve_assistant_for_creator(
        assistant_id, current_user, "use this avatar in a group conversation"
    )
    if platform in PERSONAL_AVATAR_ONLY_PLATFORMS and not is_personal_avatar(assistant):
        raise HTTPException(
            status_code=403,
            detail=(
                f"Only your own personal avatar can take part on {platform}. "
                "Select the personal avatar and try again."
            ),
        )
    return assistant, creator_user_id


@group_conversations_route.post("/groups/{assistant_id}/events")
async def receive_group_events(
    assistant_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Decide a batch of room messages; return one decision per message.

    Body: ``{platform, channel_id, channel_name, owns_channel,
    available_actions, events: [{event_id, author_id, author_name, text,
    mentioned, posted_at, metadata}]}``.

    Each decision is ``{event_id, action, moderation_action, reply, reasoning,
    confidence, item_id}``. A decision that needs the owner comes back as
    ``notify`` with an ``item_id``: nothing is said in the room and nobody is
    moderated until the owner has answered.
    """
    from src.anubis.utils.billing.tiers import UsageMeter
    from src.anubis.utils.groups.precedent import record_channel
    from src.anubis.utils.groups.runner import triage_group_events
    from src.api.webapp import app, enforce_remaining_allotment

    _require_enabled()
    payload = await request.json()
    try:
        events_request = GroupEventsRequest(**(payload if isinstance(payload, dict) else {}))
    except Exception as validation_error:  # noqa: BLE001 - the bot needs to know what was wrong
        raise HTTPException(status_code=400, detail=str(validation_error)) from validation_error

    context = GlobalContext()
    ceiling = int(getattr(context, "group_max_events_per_request", None) or 100)
    if len(events_request.events) > ceiling:
        # Refused rather than truncated: a bot must never believe messages were
        # decided when the messages were silently dropped.
        raise HTTPException(
            status_code=400,
            detail=f"At most {ceiling} messages may be sent in one batch; {len(events_request.events)} were sent.",
        )
    if not events_request.events:
        return JSONResponse({"assistant_id": assistant_id, "decisions": []})

    assistant, creator_user_id = await resolve_avatar_for_group(
        assistant_id, current_user, events_request.platform
    )
    await enforce_remaining_allotment(
        app.state, current_user, UsageMeter.MESSAGES, assistant_id=assistant_id
    )
    store = _store_or_503()
    await record_channel(
        store,
        creator_user_id,
        assistant_id,
        platform=events_request.platform,
        channel_id=events_request.channel_id,
        channel_name=events_request.channel_name,
        owns_channel=events_request.owns_channel,
    )
    decisions = await triage_group_events(
        context,
        events_request,
        user_id=creator_user_id,
        assistant_id=assistant_id,
        assistant=assistant,
        concurrency=int(getattr(context, "group_conversation_concurrency", None) or 4),
        recent_window=int(getattr(context, "group_recent_events_for_triage", None) or 12),
    )
    return JSONResponse(
        {
            "assistant_id": assistant_id,
            "platform": events_request.platform,
            "channel_id": events_request.channel_id,
            "decisions": [decision.model_dump() for decision in decisions],
        }
    )


@group_conversations_route.post("/groups/{assistant_id}/decisions")
async def correct_group_decision(
    assistant_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Correct a decision the avatar already made; the correction becomes a rule."""
    from src.anubis.utils.groups.precedent import apply_decision_correction

    _require_enabled()
    payload = await request.json()
    try:
        correction = DecisionCorrection(**(payload if isinstance(payload, dict) else {}))
    except Exception as validation_error:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(validation_error)) from validation_error
    _, creator_user_id = await resolve_avatar_for_group(
        assistant_id, current_user, correction.platform
    )
    result = await apply_decision_correction(
        _store_or_503(), creator_user_id, assistant_id, correction
    )
    return JSONResponse(result)


@group_conversations_route.get("/groups/{assistant_id}/policy")
async def read_group_policy(
    assistant_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return the rules the avatar follows in rooms."""
    from src.anubis.utils.groups.precedent import list_policy_rules

    _require_enabled()
    _, creator_user_id = await resolve_avatar_for_group(assistant_id, current_user, "")
    return JSONResponse(
        {"rules": await list_policy_rules(_store_or_503(), creator_user_id, assistant_id)}
    )


@group_conversations_route.post("/groups/{assistant_id}/policy")
async def add_group_policy_rule(
    assistant_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Add one rule. Body: ``{rule, rule_context}``."""
    from src.anubis.utils.groups.precedent import store_policy_rule

    _require_enabled()
    payload = await request.json()
    payload = payload if isinstance(payload, dict) else {}
    rule = str(payload.get("rule") or "").strip()
    if not rule:
        raise HTTPException(status_code=400, detail="A rule is required.")
    _, creator_user_id = await resolve_avatar_for_group(assistant_id, current_user, "")
    document = await store_policy_rule(
        _store_or_503(),
        creator_user_id,
        assistant_id,
        rule=rule,
        rule_context=str(payload.get("rule_context") or ""),
        source="dictated",
    )
    return JSONResponse(
        {
            "status": "already_known" if document is None else "stored",
            "rule_id": None if document is None else document.metadata.get("rule_id"),
        }
    )


@group_conversations_route.delete("/groups/{assistant_id}/policy/{rule_id}")
async def delete_group_policy_rule(
    assistant_id: str,
    rule_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Forget one rule."""
    from src.anubis.utils.groups.precedent import delete_policy_rule

    _require_enabled()
    _, creator_user_id = await resolve_avatar_for_group(assistant_id, current_user, "")
    deleted = await delete_policy_rule(
        _store_or_503(), creator_user_id, assistant_id, rule_id
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="No such rule.")
    return JSONResponse({"status": "deleted", "rule_id": rule_id})


@group_conversations_route.get("/groups/{assistant_id}/notifications")
async def read_group_notifications(
    assistant_id: str,
    unread_only: bool = True,
    current_user: dict = Depends(get_current_user),
):
    """Return what is waiting for the owner from the rooms."""
    from src.anubis.utils.groups.precedent import list_notifications

    _require_enabled()
    _, creator_user_id = await resolve_avatar_for_group(assistant_id, current_user, "")
    notifications = await list_notifications(
        _store_or_503(), creator_user_id, assistant_id, unread_only=unread_only
    )
    return JSONResponse(
        {"waiting_count": len(notifications), "notifications": notifications}
    )


@group_conversations_route.post("/groups/{assistant_id}/notifications")
async def acknowledge_group_notifications(
    assistant_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Mark notifications seen. Body: ``{notification_ids: [...]}``."""
    from src.anubis.utils.groups.precedent import acknowledge_notifications

    _require_enabled()
    payload = await request.json()
    payload = payload if isinstance(payload, dict) else {}
    notification_ids = [
        str(entry) for entry in (payload.get("notification_ids") or []) if str(entry)
    ]
    if not notification_ids:
        raise HTTPException(status_code=400, detail="notification_ids is required.")
    _, creator_user_id = await resolve_avatar_for_group(assistant_id, current_user, "")
    acknowledged = await acknowledge_notifications(
        _store_or_503(), creator_user_id, assistant_id, notification_ids
    )
    return JSONResponse({"acknowledged": acknowledged})


@group_conversations_route.get("/groups/{assistant_id}/channels")
async def read_group_channels(
    assistant_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Which rooms this avatar takes part in."""
    from src.anubis.utils.groups.precedent import list_channels

    _require_enabled()
    _, creator_user_id = await resolve_avatar_for_group(assistant_id, current_user, "")
    return JSONResponse(
        {"channels": await list_channels(_store_or_503(), creator_user_id, assistant_id)}
    )


@group_conversations_route.post("/groups/{assistant_id}/channels")
async def join_group_channel(
    assistant_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Take part in one room. Body: ``{platform, channel_id, channel_name, owns_channel}``."""
    from src.anubis.utils.groups.precedent import record_channel

    _require_enabled()
    payload = await request.json()
    payload = payload if isinstance(payload, dict) else {}
    platform = str(payload.get("platform") or "").strip().lower()
    channel_id = str(payload.get("channel_id") or "").strip()
    if not platform or not channel_id:
        raise HTTPException(
            status_code=400, detail="platform and channel_id are required."
        )
    _, creator_user_id = await resolve_avatar_for_group(
        assistant_id, current_user, platform
    )
    channel = await record_channel(
        _store_or_503(),
        creator_user_id,
        assistant_id,
        platform=platform,
        channel_id=channel_id,
        channel_name=str(payload.get("channel_name") or ""),
        owns_channel=bool(payload.get("owns_channel")),
    )
    return JSONResponse({"status": "joined", "channel": channel})


@group_conversations_route.delete("/groups/{assistant_id}/channels")
async def leave_group_channel_route(
    assistant_id: str,
    platform: str,
    channel_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Stop taking part in one room."""
    from src.anubis.utils.groups.precedent import forget_channel

    _require_enabled()
    _, creator_user_id = await resolve_avatar_for_group(
        assistant_id, current_user, platform.strip().lower()
    )
    left = await forget_channel(
        _store_or_503(),
        creator_user_id,
        assistant_id,
        platform=platform.strip().lower(),
        channel_id=channel_id.strip(),
    )
    if not left:
        raise HTTPException(status_code=404, detail="This avatar is not in that room.")
    return JSONResponse({"status": "left", "channel_id": channel_id})


__all__ = ["group_conversations_route", "resolve_avatar_for_group"]
