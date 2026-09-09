"""Running the group-conversation graph: one run per message, one thread per message.

The twin of ``src/anubis/utils/inbox/poller.py``. The graph runs IN-PROCESS
with the application's durable checkpointer, on a thread whose id is the inbox
item's id, so a decision waiting on the owner survives a restart and is resumed
with a ``Command(resume=[HumanResponse])`` from wherever the owner answers.

A batch from a bot is triaged concurrently under a semaphore, and every message
is answered independently: one message that fails to classify becomes a
notification for the owner rather than failing the batch, because a bot must
always get one decision back for every message the bot sent.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from langgraph.types import Command

from src.anubis.utils.groups.events import (
    GroupDecision,
    GroupEvent,
    GroupEventsRequest,
)
from src.anubis.utils.inbox.repository import (
    STATE_PENDING_OWNER,
    get_inbox_repository,
)

logger = logging.getLogger(__name__)

_checkpointer: Any | None = None
_store: Any | None = None
_compiled_graph: Any | None = None


def set_group_runtime(checkpointer: Any, store: Any) -> None:
    """Publish the checkpointer and store the in-process graph runs with."""
    global _checkpointer, _store, _compiled_graph
    _checkpointer = checkpointer
    _store = store
    _compiled_graph = None


def _graph() -> Any:
    global _compiled_graph
    if _compiled_graph is None:
        from src.subgraphs.group_conversation.graph import build_group_graph

        _compiled_graph = build_group_graph(checkpointer=_checkpointer, store=_store)
    return _compiled_graph


def viewer_user_id(platform: str, author_id: str) -> str:
    """Build a synthetic, stable identity for one person in a room.

    The reply is generated under this identity rather than the owner's, so what
    the avatar learns about a stranger who spoke in a public room lands under
    that stranger, and none of the owner's private capabilities are reachable
    from a message the owner did not write.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"group-viewer:{platform}:{author_id}"))


def _run_config(
    item: dict[str, Any],
    *,
    assistant: dict[str, Any] | None,
    viewer_id: str,
) -> dict[str, Any]:
    metadata = dict((assistant or {}).get("metadata") or {})
    metadata.setdefault("user_id", item["user_id"])
    metadata.setdefault("is_personal_avatar_of_creator", True)
    return {
        "configurable": {
            "thread_id": item["item_id"],
            # The owner still owns the avatar and the billing; the VIEWER is who
            # the avatar is speaking to.
            "user_id": item["user_id"],
            "viewer_user_id": viewer_id,
            "assistant_id": item["assistant_id"],
            # A stranger's words in a public room must never move the owner's
            # engagement, sentiment or ban records.
            "skip_observation": True,
            "skip_content_moderation": True,
            "user_ctx": {"name": None, "description": None},
            "assistant_ctx": {
                "name": (assistant or {}).get("name"),
                "description": (assistant or {}).get("description"),
                "assistant_id": item["assistant_id"],
                "metadata": metadata,
            },
        }
    }


def _account_key(platform: str, channel_id: str) -> str:
    return f"{platform}:{channel_id}"


async def run_group_conversation_for_event(
    context: Any,
    *,
    user_id: str,
    assistant_id: str,
    assistant: dict[str, Any] | None,
    platform: str,
    channel_id: str,
    channel_name: str,
    owns_channel: bool,
    available_actions: list[str],
    event: GroupEvent,
    recent_events: list[GroupEvent],
) -> GroupDecision:
    """Record one message as an inbox item and run its triage to a decision."""
    repository = get_inbox_repository()
    if repository is None:
        raise RuntimeError("The inbox repository has not been published.")

    account_key = _account_key(platform, channel_id)
    existing = await repository.find_item_by_external_id(
        assistant_id=assistant_id,
        source_kind=platform,
        account_key=account_key,
        external_id=event.event_id,
    )
    if existing is not None:
        # A bot resending a batch after a network failure must get the decision
        # already made, not a second run and a second reply.
        return _decision_from_item(existing, event)

    speaker = event.author_name or event.author_id
    item = await repository.create_item(
        {
            "user_id": user_id,
            "assistant_id": assistant_id,
            "source_kind": platform,
            "account_key": account_key,
            "external_id": event.event_id,
            "external_thread_id": channel_id,
            "sender": f"{account_key}:{event.author_id}",
            "sender_domain": account_key,
            "recipients": [],
            "subject": f"{channel_name or channel_id} — {speaker}",
            "body_text": event.text,
            "received_at": event.posted_at,
            "state": STATE_PENDING_OWNER,
        }
    )
    viewer_id = viewer_user_id(platform, event.author_id)
    initial_state = {
        "item_id": item["item_id"],
        "user_id": user_id,
        "viewer_user_id": viewer_id,
        "assistant_id": assistant_id,
        "assistant_name": (assistant or {}).get("name") or "",
        "platform": platform,
        "channel_id": channel_id,
        "channel_name": channel_name,
        "owns_channel": owns_channel,
        "available_actions": list(available_actions or []),
        "event": event.model_dump(),
        "recent_events": [entry.model_dump() for entry in recent_events],
    }
    try:
        final_state = await _graph().ainvoke(
            initial_state,
            config=_run_config(item, assistant=assistant, viewer_id=viewer_id),
            context=context,
        )
    except Exception as run_error:  # noqa: BLE001 - the bot still needs one decision back
        logger.exception(
            "Group triage failed for item %s: %s", item["item_id"], run_error
        )
        await repository.update_item(
            item["item_id"], state="failed", reason=str(run_error)
        )
        return GroupDecision(
            event_id=event.event_id,
            author_id=event.author_id,
            action="ignore",
            reasoning=f"The triage run failed ({run_error}); nothing was done.",
            item_id=item["item_id"],
        )
    return _decision_from_state(
        final_state, await repository.get_item(item["item_id"]), event
    )


def _decision_from_state(
    final_state: dict[str, Any] | None,
    item: dict[str, Any] | None,
    event: GroupEvent,
) -> GroupDecision:
    """Turn the run's final state into the one decision the bot carries out.

    A run that paused on the owner returns ``notify``: nothing is said in the
    room and nobody is moderated until the owner has answered, whatever the
    avatar was proposing to do.
    """
    state = final_state or {}
    row = item or {}
    classification = state.get("classification") or {}
    waiting = str(row.get("state") or "") == STATE_PENDING_OWNER

    action = str(
        state.get("chosen_action")
        or classification.get("decision")
        or ("respond" if state.get("mentioned") else "ignore")
    )
    if action == "post_reply":
        action = "respond"
    if action == "notify_owner":
        action = "notify"
    if waiting:
        action = "notify"

    reply = str((state.get("draft") or {}).get("body") or "") or None
    moderation_action = str(state.get("moderation_action") or "none")
    if action != "moderate":
        moderation_action = "none"
    return GroupDecision(
        event_id=event.event_id,
        author_id=event.author_id,
        action=action if action in ("ignore", "respond", "notify", "moderate") else "notify",
        moderation_action=moderation_action,
        # Nothing is posted in the room while the owner still has to answer.
        reply=None if waiting or action != "respond" else reply,
        reasoning=str(classification.get("reason") or row.get("reason") or ""),
        confidence=float(state.get("confidence") or row.get("confidence") or 0.0),
        applied_rule=str(classification.get("applied_rule") or "") or None,
        item_id=str(row.get("item_id") or state.get("item_id") or "") or None,
        notification_id=None if not waiting else str(row.get("item_id") or ""),
    )


def _decision_from_item(item: dict[str, Any] | None, event: GroupEvent) -> GroupDecision:
    """Answer a resent message from the row, without deciding the message twice."""
    if item is None:
        return GroupDecision(
            event_id=event.event_id, author_id=event.author_id, action="ignore"
        )
    detail = item.get("confidence_detail") or {}
    waiting = str(item.get("state") or "") == STATE_PENDING_OWNER
    action = str(detail.get("action") or item.get("decision") or "ignore")
    if action == "post_reply":
        action = "respond"
    if waiting:
        action = "notify"
    moderation_action = str(detail.get("moderation_action") or "none")
    if action != "moderate":
        moderation_action = "none"
    return GroupDecision(
        event_id=str(item.get("external_id") or event.event_id),
        author_id=event.author_id,
        action=action if action in ("ignore", "respond", "notify", "moderate") else "notify",
        moderation_action=moderation_action,
        reply=None if waiting or action != "respond" else (item.get("draft") or None),
        reasoning=str(item.get("reason") or ""),
        confidence=float(item.get("confidence") or 0.0),
        item_id=str(item.get("item_id") or ""),
    )


async def triage_group_events(
    context: Any,
    request: GroupEventsRequest,
    *,
    user_id: str,
    assistant_id: str,
    assistant: dict[str, Any] | None = None,
    concurrency: int = 4,
    recent_window: int = 12,
) -> list[GroupDecision]:
    """Decide every message in one batch, concurrently, one decision each."""
    semaphore = asyncio.Semaphore(max(1, concurrency))
    events = list(request.events)

    async def _one(index: int, event: GroupEvent) -> GroupDecision:
        async with semaphore:
            # The rolling window is what was said before this message, so the
            # avatar reads the room rather than one line out of context.
            window_start = max(0, index - max(0, recent_window))
            try:
                return await run_group_conversation_for_event(
                    context,
                    user_id=user_id,
                    assistant_id=assistant_id,
                    assistant=assistant,
                    platform=request.platform,
                    channel_id=request.channel_id,
                    channel_name=request.channel_name,
                    owns_channel=request.owns_channel,
                    available_actions=request.available_actions,
                    event=event,
                    recent_events=events[window_start:index],
                )
            except Exception as event_error:  # noqa: BLE001 - one message never fails the batch
                logger.exception(
                    "Group triage failed for %s on %s: %s",
                    event.event_id,
                    request.platform,
                    event_error,
                )
                return GroupDecision(
                    event_id=event.event_id,
                    author_id=event.author_id,
                    action="ignore",
                    reasoning=f"This message could not be decided ({event_error}).",
                )

    return list(
        await asyncio.gather(
            *(_one(index, event) for index, event in enumerate(events))
        )
    )


async def resume_group_item(
    context: Any, *, item_id: str, human_response: dict[str, Any]
) -> dict[str, Any] | None:
    """Deliver the owner's decision to the paused run and let the run finish."""
    repository = get_inbox_repository()
    if repository is None:
        return None
    item = await repository.get_item(item_id)
    if item is None:
        return None
    if item.get("state") != STATE_PENDING_OWNER:
        return item
    viewer_id = viewer_user_id(
        str(item.get("source_kind") or ""),
        str(item.get("sender") or "").rsplit(":", 1)[-1],
    )
    config = _run_config(item, assistant=None, viewer_id=viewer_id)
    try:
        await _graph().ainvoke(
            Command(resume=[human_response]), config=config, context=context
        )
    except Exception as resume_error:  # noqa: BLE001 - the item records the failure
        logger.exception(
            "Resuming group item %s failed: %s", item_id, resume_error
        )
        await repository.update_item(item_id, state="failed", reason=str(resume_error))
    return await repository.get_item(item_id)


__all__ = [
    "resume_group_item",
    "run_group_conversation_for_event",
    "set_group_runtime",
    "triage_group_events",
    "viewer_user_id",
]
