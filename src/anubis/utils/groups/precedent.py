"""What the avatar has learned about taking part in a room, and where that lives.

Three owner-scoped store namespaces, ported from the ``z`` line's live-stream
moderation (``5557c15``) and kept in the same shape so the decisions recorded
there are still readable:

* ``(creator_id, assistant_id, "group_policy")`` — the owner's rules, whether
  dictated in conversation, set through the policy route, or learned from a
  correction.
* ``(creator_id, assistant_id, "group_decision")`` — every decision the avatar
  made, keyed by ``stable_event_key`` so a correction updates the record rather
  than adding a second one.
* ``(creator_id, assistant_id, "group_notification")`` — what is waiting for
  the owner.

Rules and decisions are written as ``Document`` values through
``Document.to_json()`` because the store's vector index is configured in
``langgraph.json`` to embed ``document.kwargs.page_content`` — the same
discipline ``src/anubis/utils/ambient/preferences.py`` follows. Anything
written another way is stored but never retrieved by similarity, which for
this feature would mean an owner's rule that silently stops being applied.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from langchain_core.documents import Document

from src.anubis.utils.groups.events import (
    GroupEvent,
    render_event,
    stable_event_key,
)

logger = logging.getLogger(__name__)


def group_policy_namespace(creator_id: str, assistant_id: str) -> tuple[str, str, str]:
    """Where the owner's rules for taking part in rooms live."""
    return (creator_id, assistant_id, "group_policy")


def group_decision_namespace(creator_id: str, assistant_id: str) -> tuple[str, str, str]:
    """Where every decision the avatar made about a room message lives."""
    return (creator_id, assistant_id, "group_decision")


def group_notification_namespace(
    creator_id: str, assistant_id: str
) -> tuple[str, str, str]:
    """Where what is waiting for the owner lives."""
    return (creator_id, assistant_id, "group_notification")


def group_channel_namespace(creator_id: str, assistant_id: str) -> tuple[str, str, str]:
    """Which rooms this avatar takes part in."""
    return (creator_id, assistant_id, "group_channel")


def group_follow_up_namespace(creator_id: str, assistant_id: str) -> tuple[str, str, str]:
    """Return where the avatar keeps what it will come back to."""
    return (creator_id, assistant_id, "group_follow_up")


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _item_document(item: Any) -> Document | None:
    value = getattr(item, "value", None) or {}
    document_json = value.get("document") if isinstance(value, dict) else None
    kwargs = (document_json or {}).get("kwargs") or {}
    if not isinstance(kwargs, dict) or not kwargs.get("page_content"):
        return None
    return Document(
        page_content=kwargs["page_content"],
        metadata=dict(kwargs.get("metadata") or {}),
    )


async def _search(
    store: Any, namespace: tuple, *, query: str | None, limit: int
) -> list[Document]:
    """Retrieve by similarity, and never let a retrieval failure stop a decision."""
    try:
        items = await store.asearch(namespace, query=query, limit=limit)
    except Exception as search_error:  # noqa: BLE001 - an empty recall still decides
        logger.debug("Group precedent retrieval failed for %s: %s", namespace, search_error)
        return []
    documents = [_item_document(item) for item in items or []]
    return [document for document in documents if document is not None]


async def _put(store: Any, namespace: tuple, key: str, document: Document) -> None:
    await store.aput(namespace, key=key, value={"document": document.to_json()})


# ── the owner's rules ───────────────────────────────────────────────────────


async def store_policy_rule(
    store: Any,
    creator_id: str,
    assistant_id: str,
    *,
    rule: str,
    rule_context: str = "",
    source: str = "dictated",
) -> Document | None:
    """Record one rule, unless the same rule is already known."""
    rule = (rule or "").strip()
    if not rule:
        return None
    namespace = group_policy_namespace(creator_id, assistant_id)
    for existing in await _search(store, namespace, query=rule, limit=10):
        if (existing.metadata.get("rule") or "").strip().casefold() == rule.casefold():
            return None
    rule_id = str(uuid.uuid4())
    page_content = f"GROUP CONVERSATION RULE: {rule}"
    if rule_context:
        page_content += f"\nApplies when: {rule_context}"
    document = Document(
        page_content=page_content,
        metadata={
            "document_id": rule_id,
            "rule_id": rule_id,
            "rule": rule,
            "rule_context": rule_context,
            "source": source,
            "recorded_at": _utc_now_iso(),
            "fact": rule,
        },
    )
    await _put(store, namespace, rule_id, document)
    return document


async def list_policy_rules(
    store: Any, creator_id: str, assistant_id: str, *, query: str | None = None, limit: int = 500
) -> list[dict[str, Any]]:
    """Every rule, or the rules most like one message when a query is given."""
    documents = await _search(
        store, group_policy_namespace(creator_id, assistant_id), query=query, limit=limit
    )
    return [
        {
            "rule_id": document.metadata.get("rule_id"),
            "rule": document.metadata.get("rule"),
            "rule_context": document.metadata.get("rule_context"),
            "source": document.metadata.get("source"),
            "recorded_at": document.metadata.get("recorded_at"),
        }
        for document in documents
    ]


async def delete_policy_rule(
    store: Any, creator_id: str, assistant_id: str, rule_id: str
) -> bool:
    """Forget one rule."""
    namespace = group_policy_namespace(creator_id, assistant_id)
    existing = await store.aget(namespace, key=rule_id)
    if existing is None:
        return False
    await store.adelete(namespace, rule_id)
    return True


# ── what the avatar decided ─────────────────────────────────────────────────


async def store_decision_record(
    store: Any,
    creator_id: str,
    assistant_id: str,
    *,
    platform: str,
    channel_id: str,
    channel_name: str,
    event: GroupEvent,
    action: str,
    moderation_action: str,
    reasoning: str,
    confidence: float,
    applied_rule: str = "",
    reply_text: str | None = None,
    owner_approved: bool = False,
    cold_direct_message: bool = False,
) -> Document:
    """Record one decision, so the next similar message is decided the same way."""
    key = stable_event_key(platform, channel_id, event.event_id)
    lines = [
        f"EVENT: {render_event(event, platform, channel_name)}",
        f"DECISION: {action}"
        + (f" ({moderation_action})" if action == "moderate" else "")
        + f" — {reasoning}",
    ]
    if reply_text:
        lines.append(f"REPLY: {reply_text[:600]}")
    document = Document(
        page_content="\n".join(lines),
        metadata={
            "document_id": key,
            "event_key": key,
            "event_id": event.event_id,
            "platform": platform,
            "channel_id": channel_id,
            "channel_name": channel_name,
            "author_id": event.author_id,
            "author_name": event.author_name,
            "text": event.text,
            "mentioned": event.mentioned,
            "action": action,
            "moderation_action": moderation_action,
            "reasoning": reasoning,
            "confidence": confidence,
            "applied_rule": applied_rule,
            "reply_text": reply_text,
            "decided_at": _utc_now_iso(),
            "owner_approved": owner_approved,
            # Whether this direct message went to somebody who had never spoken
            # to the avatar. Only such a decision, once the owner has allowed
            # it, is precedent for the next one.
            "cold_direct_message": cold_direct_message,
            "corrected_action": None,
            "corrected_moderation_action": None,
            "fact": event.text,
        },
    )
    await _put(store, group_decision_namespace(creator_id, assistant_id), key, document)
    return document


async def recall_decisions(
    store: Any, creator_id: str, assistant_id: str, *, query: str, limit: int
) -> list[dict[str, Any]]:
    """Past decisions most like this message, the owner's corrections included."""
    documents = await _search(
        store, group_decision_namespace(creator_id, assistant_id), query=query, limit=limit
    )
    return [
        {"page_content": document.page_content, **document.metadata}
        for document in documents
    ]


async def has_moderation_precedent(
    store: Any,
    creator_id: str,
    assistant_id: str,
    *,
    platform: str,
    channel_id: str,
    moderation_action: str,
    limit: int = 200,
) -> bool:
    """Has the owner ever allowed THIS action in THIS room before?

    The gate on the two irreversible actions. A high confidence score is not
    enough to time out or ban somebody: the owner must have accepted, or
    corrected the avatar into, that same action in that same room at least
    once. Anything else waits for the owner however sure the avatar is.
    """
    documents = await _search(
        store,
        group_decision_namespace(creator_id, assistant_id),
        query=None,
        limit=limit,
    )
    for document in documents:
        metadata = document.metadata
        if str(metadata.get("platform") or "") != platform:
            continue
        if str(metadata.get("channel_id") or "") != channel_id:
            continue
        # A correction is the owner's own word and outranks what the avatar did.
        corrected = str(metadata.get("corrected_moderation_action") or "")
        if corrected:
            if corrected == moderation_action:
                return True
            continue
        if str(metadata.get("action") or "") != "moderate":
            continue
        if str(metadata.get("moderation_action") or "") != moderation_action:
            continue
        # Only an action the owner actually saw counts as precedent.
        if metadata.get("owner_approved"):
            return True
    return False


async def has_direct_message_precedent(
    store: Any,
    creator_id: str,
    assistant_id: str,
    *,
    platform: str,
    channel_id: str,
    limit: int = 200,
) -> bool:
    """Has the owner ever allowed a direct message to a stranger from this room?

    The twin of ``has_moderation_precedent``, and for the same reason. A direct
    message to somebody who has never spoken to the avatar is not reversible in
    the way a public message is: it arrives in a private conversation, wearing
    the owner's name, unasked for. Platforms read it as spam and a person reads
    it as the owner having messaged them.

    So the avatar cannot decide on its own that a cold direct message is
    acceptable here. The owner has to have allowed one — and, exactly as with a
    ban, only a decision the owner themselves accepted or corrected the avatar
    into counts, so the avatar can never bootstrap its own permission from
    something it did unilaterally.
    """
    documents = await _search(
        store,
        group_decision_namespace(creator_id, assistant_id),
        query=None,
        limit=limit,
    )
    for document in documents:
        metadata = document.metadata
        if str(metadata.get("platform") or "") != platform:
            continue
        if str(metadata.get("channel_id") or "") != channel_id:
            continue
        if not metadata.get("owner_approved"):
            continue
        corrected = str(metadata.get("corrected_action") or "")
        action = corrected or str(metadata.get("action") or "")
        if action == "direct_message" and metadata.get("cold_direct_message"):
            return True
    return False


async def has_exchanged_with(
    store: Any,
    creator_id: str,
    assistant_id: str,
    *,
    platform: str,
    channel_id: str,
    author_id: str,
    limit: int = 200,
) -> bool:
    """Has this person ever actually spoken TO the avatar, or it to them?

    Talking in a room the avatar happens to be in is not the same as addressing
    it. Somebody who has never once spoken to the avatar receiving a private
    message from it is a cold approach however long they have been in the
    channel, so the looser reading — anybody who has ever typed here — is the
    wrong one and is not used.

    Counts as having exchanged: a message that mentioned the avatar, or any
    message the avatar itself answered, privately or in the room.
    """
    documents = await _search(
        store,
        group_decision_namespace(creator_id, assistant_id),
        query=None,
        limit=limit,
    )
    for document in documents:
        metadata = document.metadata
        if str(metadata.get("platform") or "") != platform:
            continue
        if str(metadata.get("channel_id") or "") != channel_id:
            continue
        if str(metadata.get("author_id") or "") != author_id:
            continue
        if metadata.get("mentioned"):
            return True
        if str(metadata.get("action") or "") in (
            "respond",
            "reply_in_thread",
            "direct_message",
        ):
            return True
    return False


async def record_follow_up(
    store: Any,
    creator_id: str,
    assistant_id: str,
    *,
    platform: str,
    channel_id: str,
    channel_name: str,
    event: GroupEvent,
    due_at: str,
    what: str,
) -> dict[str, Any]:
    """Remember something the avatar said it would come back to.

    Keyed by the event, so the same message cannot queue two follow-ups however
    many times a bot resends it.
    """
    key = stable_event_key(platform, channel_id, event.event_id)
    payload = {
        "follow_up_id": key,
        "platform": platform,
        "channel_id": channel_id,
        "channel_name": channel_name,
        "event": event.model_dump(),
        "due_at": due_at,
        "what": what,
        "recorded_at": _utc_now_iso(),
        "resolved": False,
    }
    await store.aput(
        group_follow_up_namespace(creator_id, assistant_id), key=key, value={"value": payload}
    )
    return payload


async def due_follow_ups(
    store: Any, creator_id: str, assistant_id: str, *, now: str | None = None
) -> list[dict[str, Any]]:
    """Everything the avatar meant to come back to whose time has arrived."""
    moment = now or _utc_now_iso()
    try:
        items = await store.asearch(
            group_follow_up_namespace(creator_id, assistant_id), limit=500
        )
    except Exception:  # noqa: BLE001 - nothing due is the right answer here
        return []
    due = []
    for item in items or []:
        value = getattr(item, "value", None) or {}
        payload = value.get("value") if isinstance(value, dict) else None
        if not isinstance(payload, dict) or payload.get("resolved"):
            continue
        if str(payload.get("due_at") or "") <= moment:
            due.append(payload)
    due.sort(key=lambda payload: payload.get("due_at") or "")
    return due


async def resolve_follow_up(
    store: Any, creator_id: str, assistant_id: str, follow_up_id: str
) -> bool:
    """Mark one as dealt with, so it fires once rather than every poll."""
    namespace = group_follow_up_namespace(creator_id, assistant_id)
    item = await store.aget(namespace, key=follow_up_id)
    value = getattr(item, "value", None) or {}
    payload = value.get("value") if isinstance(value, dict) else None
    if not isinstance(payload, dict):
        return False
    payload["resolved"] = True
    payload["resolved_at"] = _utc_now_iso()
    await store.aput(namespace, key=follow_up_id, value={"value": payload})
    return True


async def mark_decision_owner_approved(
    store: Any,
    creator_id: str,
    assistant_id: str,
    *,
    platform: str,
    channel_id: str,
    event_id: str,
    approved_action: str,
    approved_moderation_action: str,
) -> None:
    """Record that the owner themselves allowed this decision, which builds precedent."""
    namespace = group_decision_namespace(creator_id, assistant_id)
    key = stable_event_key(platform, channel_id, event_id)
    item = await store.aget(namespace, key=key)
    document = _item_document(item) if item is not None else None
    if document is None:
        return
    document.metadata["owner_approved"] = True
    document.metadata["action"] = approved_action
    document.metadata["moderation_action"] = approved_moderation_action
    document.page_content += f"\nTHE OWNER ALLOWED: {approved_action}" + (
        f" ({approved_moderation_action})" if approved_action == "moderate" else ""
    )
    await _put(store, namespace, key, document)


async def apply_decision_correction(
    store: Any, creator_id: str, assistant_id: str, correction: Any
) -> dict[str, Any]:
    """Learn from the owner: update the decision and write a rule from the correction."""
    namespace = group_decision_namespace(creator_id, assistant_id)
    key = stable_event_key(
        correction.platform, correction.channel_id, correction.event_id
    )
    item = await store.aget(namespace, key=key)
    document = _item_document(item) if item is not None else None
    event_text = document.metadata.get("text") if document else None

    corrected = correction.corrected_action
    if corrected == "moderate":
        corrected += f" ({correction.corrected_moderation_action})"
    if event_text:
        rule = f'When a message like "{event_text[:200]}" appears, {corrected} the message'
    else:
        rule = (
            f"For message {correction.event_id} on {correction.platform}, "
            f"{corrected} the message"
        )
    if document is not None and document.metadata.get("action"):
        previous = document.metadata["action"]
        if previous == "moderate":
            previous += f" ({document.metadata.get('moderation_action')})"
        rule += f" instead of {previous}"
    rule += "."
    if correction.note.strip():
        rule += f" The owner's reason: {correction.note.strip()}"
    rule_document = await store_policy_rule(
        store,
        creator_id,
        assistant_id,
        rule=rule,
        rule_context="Learned from an owner correction.",
        source="correction",
    )

    if document is not None:
        document.metadata["corrected_action"] = correction.corrected_action
        document.metadata["corrected_moderation_action"] = (
            correction.corrected_moderation_action
        )
        document.metadata["correction_note"] = correction.note
        # A correction the owner made is the owner allowing that action here.
        document.metadata["owner_approved"] = True
        document.page_content += (
            f"\nOWNER CORRECTION: {corrected} — "
            f"{correction.note.strip() or 'no reason given'}"
        )
        await _put(store, namespace, key, document)
    return {
        "event_id": correction.event_id,
        "decision_found": document is not None,
        # What the avatar actually did, so a caller can tell whether there are
        # words of its own still standing that the correction should fix.
        "previous_action": (
            str(document.metadata.get("action") or "") if document is not None else ""
        ),
        "previous_reply": (
            str(document.metadata.get("reply_text") or "") if document is not None else ""
        ),
        "learned_rule": rule_document.metadata["rule"] if rule_document else None,
        "rule_already_known": rule_document is None,
    }


# ── what is waiting for the owner ───────────────────────────────────────────


async def queue_notification(
    store: Any,
    creator_id: str,
    assistant_id: str,
    *,
    platform: str,
    channel_id: str,
    channel_name: str,
    event: GroupEvent,
    action: str,
    moderation_action: str,
    reasoning: str,
    item_id: str | None = None,
) -> dict[str, Any]:
    """Put one message in front of the owner.

    The notification is keyed by the event rather than by a fresh identifier,
    because the node that queues the notification re-runs from the top every
    time the owner resumes the paused graph. An unkeyed write would put the
    same message in front of the owner again on every resume.
    """
    notification_id = stable_event_key(platform, channel_id, event.event_id)
    payload = {
        "notification_id": notification_id,
        "platform": platform,
        "channel_id": channel_id,
        "channel_name": channel_name,
        "event": event.model_dump(),
        "action": action,
        "moderation_action": moderation_action,
        "reasoning": reasoning,
        "item_id": item_id,
        "queued_at": _utc_now_iso(),
        "acknowledged": False,
    }
    await store.aput(
        group_notification_namespace(creator_id, assistant_id),
        key=notification_id,
        value={"value": payload},
    )
    return payload


async def list_notifications(
    store: Any, creator_id: str, assistant_id: str, *, unread_only: bool = True
) -> list[dict[str, Any]]:
    """Return what the owner has not yet seen, oldest first."""
    try:
        items = await store.asearch(
            group_notification_namespace(creator_id, assistant_id), limit=500
        )
    except Exception:  # noqa: BLE001 - an empty queue is the right answer here
        return []
    notifications = []
    for item in items or []:
        value = getattr(item, "value", None) or {}
        payload = value.get("value") if isinstance(value, dict) else None
        if not isinstance(payload, dict):
            continue
        if unread_only and payload.get("acknowledged"):
            continue
        notifications.append(payload)
    notifications.sort(key=lambda payload: payload.get("queued_at") or "")
    return notifications


async def acknowledge_notifications(
    store: Any, creator_id: str, assistant_id: str, notification_ids: list[str]
) -> int:
    """Mark notifications seen; returns how many were found."""
    namespace = group_notification_namespace(creator_id, assistant_id)
    acknowledged = 0
    for notification_id in notification_ids:
        item = await store.aget(namespace, key=notification_id)
        value = getattr(item, "value", None) or {}
        payload = value.get("value") if isinstance(value, dict) else None
        if not isinstance(payload, dict):
            continue
        payload["acknowledged"] = True
        await store.aput(namespace, key=notification_id, value={"value": payload})
        acknowledged += 1
    return acknowledged


# ── which rooms the avatar is in ────────────────────────────────────────────


async def record_channel(
    store: Any,
    creator_id: str,
    assistant_id: str,
    *,
    platform: str,
    channel_id: str,
    channel_name: str = "",
    owns_channel: bool = False,
) -> dict[str, Any]:
    """Remember that this avatar takes part in this room."""
    key = f"{platform}:{channel_id}"
    payload = {
        "platform": platform,
        "channel_id": channel_id,
        "channel_name": channel_name,
        "owns_channel": owns_channel,
        "joined_at": _utc_now_iso(),
    }
    await store.aput(
        group_channel_namespace(creator_id, assistant_id), key=key, value={"value": payload}
    )
    return payload


async def list_channels(
    store: Any, creator_id: str, assistant_id: str
) -> list[dict[str, Any]]:
    """Every room this avatar takes part in."""
    try:
        items = await store.asearch(
            group_channel_namespace(creator_id, assistant_id), limit=500
        )
    except Exception:  # noqa: BLE001
        return []
    channels = []
    for item in items or []:
        value = getattr(item, "value", None) or {}
        payload = value.get("value") if isinstance(value, dict) else None
        if isinstance(payload, dict):
            channels.append(payload)
    channels.sort(key=lambda payload: (payload.get("platform") or "", payload.get("channel_name") or ""))
    return channels


async def forget_channel(
    store: Any, creator_id: str, assistant_id: str, *, platform: str, channel_id: str
) -> bool:
    """Leave one room."""
    namespace = group_channel_namespace(creator_id, assistant_id)
    key = f"{platform}:{channel_id}"
    existing = await store.aget(namespace, key=key)
    if existing is None:
        return False
    await store.adelete(namespace, key)
    return True


__all__ = [
    "acknowledge_notifications",
    "due_follow_ups",
    "group_follow_up_namespace",
    "has_direct_message_precedent",
    "has_exchanged_with",
    "record_follow_up",
    "resolve_follow_up",
    "apply_decision_correction",
    "delete_policy_rule",
    "forget_channel",
    "group_channel_namespace",
    "group_decision_namespace",
    "group_notification_namespace",
    "group_policy_namespace",
    "has_moderation_precedent",
    "list_channels",
    "list_notifications",
    "list_policy_rules",
    "mark_decision_owner_approved",
    "queue_notification",
    "recall_decisions",
    "record_channel",
    "store_decision_record",
    "store_policy_rule",
]
