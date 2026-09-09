"""The avatar taking part in a room: Slack, Discord, or Twitch.

One run per message, on a thread whose id is the inbox item's id, so a decision
waiting on the owner survives a restart and can be answered from the inbox
panel, from chat, or from the Agent Inbox app — exactly like the mailbox
triage this graph is modelled on.

    START → accept_event → recall_precedent → route_by_mention
      mentioned → draft_in_voice → post_reply → record_outcome → END
      passive   → classify
          ignore   → record_outcome → END
          respond  → draft_in_voice → score_confidence ┐
          moderate → score_confidence                  ├ high → act → record_outcome
          notify   → ──────────────────────────────────┴ low  → await_owner
                                                                  → apply_owner_decision
                                                                  → update_preferences
                                                                  → route_by_action
                                                                  → post_reply
                                                                  | apply_moderation
                                                                  | record_outcome → END

A direct @mention is always answered and never triaged: somebody spoke to the
avatar, so the avatar replies. Everything else is the passing stream of the
room, where the avatar decides for itself.

Two rules make this safe, and both are behaviour rather than configuration:

* **Moderation reaches the owner until precedent exists.** ``preference_prior``
  already caps a decision with no history below the threshold. On top of that,
  a timeout or a ban requires the owner to have allowed that same action in
  that same room before (``has_moderation_precedent``); confidence alone can
  never produce one.
* **The reply is written under a viewer's identity, never the owner's.** The
  run carries a synthetic per-viewer user id, ``skip_observation`` and
  ``skip_content_moderation``, so a stranger's words in a public room can never
  move the owner's learning records or ban the owner whose credential carries
  the request.

Nothing here talks to a platform. The bot that posted the batch receives the
decision and carries it out, which is what keeps one brain serving three
platforms.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.groups.events import GroupEvent, render_event
from src.anubis.utils.groups.precedent import (
    has_moderation_precedent,
    queue_notification,
    recall_decisions,
    store_decision_record,
    store_policy_rule,
)
from src.anubis.utils.inbox.repository import (
    ACTION_MODERATE,
    ACTION_NOTIFY_OWNER,
    ACTION_POST_REPLY,
    STATE_AUTO_SENT,
    STATE_IGNORED,
    STATE_PENDING_OWNER,
    STATE_RESOLVED,
    STATE_SENT,
    get_inbox_repository,
)

logger = logging.getLogger(__name__)

DECISION_IGNORE = "ignore"
DECISION_RESPOND = "respond"
DECISION_NOTIFY = "notify"
DECISION_MODERATE = "moderate"

# The two actions that cannot be undone by the owner afterwards, and so require
# the owner to have allowed that same action in that same room before.
IRREVERSIBLE_MODERATION_ACTIONS = ("timeout", "ban")


class GroupConversationState(TypedDict, total=False):
    """Everything one message's run carries between nodes."""

    item_id: str
    user_id: str
    viewer_user_id: str
    assistant_id: str
    assistant_name: str
    platform: str
    channel_id: str
    channel_name: str
    owns_channel: bool
    available_actions: list[str]
    event: dict[str, Any]
    recent_events: list[dict[str, Any]]
    speaker_id: str
    speaker_name: str
    mentioned: bool
    policy_rules: list[dict[str, Any]]
    past_decisions: list[dict[str, Any]]
    preferences: list[dict[str, Any]]
    classification: dict[str, Any]
    draft: dict[str, Any] | None
    draft_original: dict[str, Any] | None
    confidence: float
    confidence_detail: dict[str, Any]
    owner_decision: dict[str, Any] | None
    chosen_action: str | None
    moderation_action: str
    outcome: str
    error: str | None


def _repository() -> Any:
    repository = get_inbox_repository()
    if repository is None:
        raise RuntimeError("The inbox repository has not been published.")
    return repository


def _event_of(state: GroupConversationState) -> GroupEvent:
    return GroupEvent(**(state.get("event") or {}))


def _recent_events_of(state: GroupConversationState) -> list[GroupEvent]:
    return [GroupEvent(**entry) for entry in (state.get("recent_events") or [])]


def _room_key(state: GroupConversationState) -> str:
    return f"{state.get('platform')}:{state.get('channel_id')}"


def _speaker_key(state: GroupConversationState) -> str:
    """Build the preference key for this person in this room.

    ``inbox_preferences`` ranks a match on ``sender`` above one on
    ``sender_domain`` above one on ``message_kind``, which for a room reads as
    *this person here* → *anyone in this room* → *this kind of message
    anywhere*. An email domain and a chat room generalise the same way, which
    is why this feature needs no new table and no migration.
    """
    return f"{_room_key(state)}:{state.get('speaker_id')}"


# ── nodes ───────────────────────────────────────────────────────────────────


async def accept_event(
    state: GroupConversationState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> dict[str, Any]:
    """Normalize the message and note who spoke."""
    event = _event_of(state)
    return {
        "event": event.model_dump(),
        "speaker_id": event.author_id,
        "speaker_name": event.author_name or event.author_id,
        "mentioned": bool(event.mentioned),
        "moderation_action": "none",
        "error": None,
    }


async def recall_precedent(
    state: GroupConversationState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> dict[str, Any]:
    """Read the owner's rules, past decisions, and what the owner decided for this room."""
    from src.anubis.utils.groups.precedent import list_policy_rules

    store = getattr(runtime, "store", None)
    event = _event_of(state)
    limit = int(getattr(runtime.context, "group_precedent_recall_limit", None) or 8)
    policy_rules: list[dict[str, Any]] = []
    past_decisions: list[dict[str, Any]] = []
    if store is not None:
        policy_rules = await list_policy_rules(
            store, state["user_id"], state["assistant_id"], query=event.text, limit=limit
        )
        past_decisions = await recall_decisions(
            store,
            state["user_id"],
            state["assistant_id"],
            query=event.text,
            limit=limit,
        )
    preferences = await _repository().recall_preferences(
        assistant_id=state["assistant_id"],
        sender=_speaker_key(state),
        sender_domain=_room_key(state),
        message_kind=None,
    )
    return {
        "policy_rules": policy_rules,
        "past_decisions": past_decisions,
        "preferences": preferences,
    }


def route_by_mention(state: GroupConversationState) -> str:
    """Somebody spoke to the avatar: answer. Otherwise decide whether to take part."""
    return "draft_in_voice" if state.get("mentioned") else "classify"


async def classify(
    state: GroupConversationState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> dict[str, Any]:
    """Decide: ignore, respond, notify, or moderate."""
    from src.anubis.utils.groups.triage import classify_group_event

    classification = await classify_group_event(
        runtime.context,
        event=_event_of(state),
        platform=str(state.get("platform") or ""),
        channel_name=str(state.get("channel_name") or ""),
        owner_name=str(state.get("assistant_name") or ""),
        recent_events=_recent_events_of(state),
        policy_rules=state.get("policy_rules") or [],
        past_decisions=state.get("past_decisions") or [],
        available_actions=list(state.get("available_actions") or []),
        owns_channel=bool(state.get("owns_channel")),
    )
    await _repository().update_item(
        state["item_id"],
        message_kind=classification.message_kind,
        decision=classification.decision,
        needs_owner_action=classification.needs_owner_action,
        reason=classification.reason,
    )
    return {
        "classification": classification.model_dump(),
        "moderation_action": classification.moderation_action,
    }


def route_after_classify(state: GroupConversationState) -> str:
    """Route an ignore to the outcome, a reply to drafting, the rest to scoring or the owner."""
    decision = (state.get("classification") or {}).get("decision")
    if decision == DECISION_IGNORE:
        return "record_outcome"
    if decision == DECISION_RESPOND:
        return "draft_in_voice"
    if decision == DECISION_MODERATE:
        return "score_confidence"
    return "await_owner"


async def _voice_system_prompt(
    state: GroupConversationState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> str:
    """Build the avatar's consciousness prompt so the reply carries the owner's voice."""
    try:
        from langchain_core.messages import HumanMessage

        from src.anubis.utils.nodes import _build_consciousness_system_message_update

        pseudo_state = {
            "messages": [
                HumanMessage(content=str(_event_of(state).text or "")[:2000])
            ],
            "user_state": {
                # The viewer, never the owner: what the avatar learns about a
                # stranger in a public room lands under the stranger.
                "user_id": state.get("viewer_user_id") or state["user_id"],
                "user_name": state.get("speaker_name") or "",
                "user_description": "",
            },
            "assistant_state": {
                "assistant_id": state["assistant_id"],
                "assistant_name": state.get("assistant_name") or "",
                "assistant_description": "",
            },
        }
        update = await _build_consciousness_system_message_update(
            pseudo_state, config, runtime
        )
        system_messages = update.get("system_message") or []
        if system_messages:
            first = system_messages[0]
            return str(getattr(first, "content", first))
    except Exception:  # noqa: BLE001 - fall back to a plain voice instruction
        logger.debug("Consciousness prompt unavailable for the group reply", exc_info=True)
    return (
        f"You are {state.get('assistant_name') or 'the owner'}, speaking in your own voice "
        "in a group conversation."
    )


GROUP_REPLY_SUFFIX = """
<REPLY_TASK>
Write one short message to post in this room, as the owner would write the message: the owner's register, sentence length, and manner. Answer what was actually said. This is a public room, so say nothing private and promise nothing on the owner's behalf. Do not greet the room, do not introduce yourself, and never mention that an assistant or an avatar wrote the message.
</REPLY_TASK>
"""


async def draft_in_voice(
    state: GroupConversationState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> dict[str, Any]:
    """Write the reply under the avatar's consciousness prompt."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic import BaseModel, Field

    from src.anubis.utils.model import init_model

    class GroupReply(BaseModel):
        """One message to post in the room."""

        body: str = Field(description="The message to post, in the owner's voice.")
        summary: str = Field(default="", description="One sentence saying what the message does.")

    voice_prompt = await _voice_system_prompt(state, config, runtime)
    event = _event_of(state)
    model = init_model(model_without_tools=False, response_format=GroupReply)
    try:
        response = await model.ainvoke(
            input=[
                SystemMessage(content=(voice_prompt or "") + GROUP_REPLY_SUFFIX),
                HumanMessage(
                    content="<ROOM>\n"
                    + "\n".join(
                        render_event(
                            entry,
                            str(state.get("platform") or ""),
                            str(state.get("channel_name") or ""),
                        )
                        for entry in _recent_events_of(state)
                    )
                    + "\n</ROOM>\n\n<MESSAGE>\n"
                    + render_event(
                        event,
                        str(state.get("platform") or ""),
                        str(state.get("channel_name") or ""),
                    )
                    + "\n</MESSAGE>"
                ),
            ]
        )
    except Exception as draft_error:  # noqa: BLE001 - an undraftable message goes to the owner
        logger.warning("Group reply drafting failed for %s: %s", state.get("item_id"), draft_error)
        return {
            "draft": None,
            "classification": {
                **(state.get("classification") or {}),
                "decision": DECISION_NOTIFY,
                "reason": f"The reply could not be written ({draft_error}); the owner should decide.",
            },
        }
    draft = {
        "body": str(getattr(response, "body", "") or "").strip(),
        "summary": str(getattr(response, "summary", "") or "").strip(),
    }
    await _repository().update_item(state["item_id"], draft=draft["body"])
    return {"draft": draft, "draft_original": dict(draft)}


def route_after_draft(state: GroupConversationState) -> str:
    """Post a mention straight back; score a passive reply first."""
    if not (state.get("draft") or {}).get("body"):
        return "await_owner"
    return "post_reply" if state.get("mentioned") else "score_confidence"


async def score_confidence(
    state: GroupConversationState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> dict[str, Any]:
    """How sure is the avatar, given what the owner has decided in this room before?"""
    from src.anubis.utils.inbox.triage import (
        DraftReply,
        combine_confidence,
        judge_alignment,
        preference_prior,
    )

    classification = state.get("classification") or {}
    is_moderation = classification.get("decision") == DECISION_MODERATE
    threshold = float(
        getattr(runtime.context, "group_auto_moderate_confidence", None) or 0.97
        if is_moderation
        else getattr(runtime.context, "group_auto_respond_confidence", None) or 0.9
    )
    preferences = state.get("preferences") or []
    prior, prior_reason = preference_prior(
        preferences,
        auto_send_threshold=threshold,
        sender=_speaker_key(state),
        sender_domain=_room_key(state),
    )
    event = _event_of(state)
    if is_moderation:
        # There is no draft to judge for a moderation action, so the classifier's
        # own confidence stands in for the alignment half of the score.
        alignment_score = float(classification.get("confidence") or 0.0)
        alignment_reason = classification.get("reason") or ""
    else:
        draft = DraftReply(
            subject="",
            body=str((state.get("draft") or {}).get("body") or ""),
            summary=str((state.get("draft") or {}).get("summary") or ""),
        )
        alignment = await judge_alignment(
            runtime.context,
            message={
                "sender": state.get("speaker_name"),
                "subject": f"{state.get('platform')} · {state.get('channel_name')}",
                "body_text": event.text,
            },
            draft=draft,
            preferences=preferences,
        )
        alignment_score = alignment.alignment_score
        alignment_reason = alignment.reason
    confidence = combine_confidence(alignment_score, prior)
    detail = {
        "alignment_score": alignment_score,
        "alignment_reason": alignment_reason,
        "prior": prior,
        "prior_reason": prior_reason,
        "threshold": threshold,
    }
    await _repository().update_item(
        state["item_id"], confidence=confidence, confidence_detail=detail
    )
    return {"confidence": confidence, "confidence_detail": detail}


async def route_after_confidence(
    state: GroupConversationState, runtime: Runtime[GlobalContext] | None = None
) -> str:
    """Act only when sure, and never take an irreversible action without precedent."""
    classification = state.get("classification") or {}
    detail = state.get("confidence_detail") or {}
    threshold = float(detail.get("threshold") or 0.9)
    confident = float(state.get("confidence") or 0.0) >= threshold

    if classification.get("needs_owner_action"):
        return "await_owner"
    if classification.get("decision") == DECISION_MODERATE:
        moderation_action = str(state.get("moderation_action") or "none")
        if moderation_action in IRREVERSIBLE_MODERATION_ACTIONS:
            store = getattr(runtime, "store", None) if runtime is not None else None
            if store is None:
                return "await_owner"
            allowed_before = await has_moderation_precedent(
                store,
                state["user_id"],
                state["assistant_id"],
                platform=str(state.get("platform") or ""),
                channel_id=str(state.get("channel_id") or ""),
                moderation_action=moderation_action,
            )
            if not allowed_before:
                return "await_owner"
        return "apply_moderation" if confident else "await_owner"
    return "post_reply" if confident else "await_owner"


def _human_interrupt(state: GroupConversationState) -> dict[str, Any]:
    """Build the Agent Inbox ``HumanInterrupt`` for this message."""
    classification = state.get("classification") or {}
    event = _event_of(state)
    where = f"{state.get('platform')} · {state.get('channel_name') or state.get('channel_id')}"
    decision = classification.get("decision")
    action = ACTION_MODERATE if decision == DECISION_MODERATE else (
        ACTION_POST_REPLY if (state.get("draft") or {}).get("body") else ACTION_NOTIFY_OWNER
    )
    available = [ACTION_POST_REPLY, ACTION_NOTIFY_OWNER]
    if state.get("owns_channel") and state.get("available_actions"):
        available.append(ACTION_MODERATE)
    return {
        "action_request": {
            "action": action,
            "args": {
                "platform": state.get("platform"),
                "channel": state.get("channel_name") or state.get("channel_id"),
                "from": state.get("speaker_name"),
                "text": event.text,
                "body": (state.get("draft") or {}).get("body"),
                "moderation_action": state.get("moderation_action"),
                "available_actions": available,
                "available_moderation_actions": list(state.get("available_actions") or []),
                "summary": classification.get("reason"),
            },
        },
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": True,
            "allow_accept": True,
        },
        "description": (
            f'{state.get("speaker_name")} in {where}: "{event.text[:200]}" — '
            f"{classification.get('reason') or ''} "
            f"(confidence {float(state.get('confidence') or 0.0):.2f})"
        ),
    }


async def await_owner(
    state: GroupConversationState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> dict[str, Any]:
    """Pause for the owner, after mirroring the pending item and queueing the notice."""
    repository = _repository()
    await repository.update_item(state["item_id"], state=STATE_PENDING_OWNER)
    store = getattr(runtime, "store", None)
    if store is not None:
        classification = state.get("classification") or {}
        await queue_notification(
            store,
            state["user_id"],
            state["assistant_id"],
            platform=str(state.get("platform") or ""),
            channel_id=str(state.get("channel_id") or ""),
            channel_name=str(state.get("channel_name") or ""),
            event=_event_of(state),
            action=str(classification.get("decision") or DECISION_NOTIFY),
            moderation_action=str(state.get("moderation_action") or "none"),
            reasoning=str(classification.get("reason") or ""),
            item_id=str(state.get("item_id") or ""),
        )
    decision = interrupt(_human_interrupt(state))
    responses = decision if isinstance(decision, list) else [decision]
    first = next((entry for entry in responses if isinstance(entry, dict)), None) or {}
    return {"owner_decision": first}


async def apply_owner_decision(
    state: GroupConversationState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> dict[str, Any]:
    """Read what the owner chose: the action, the reply text, the moderation action."""
    decision = state.get("owner_decision") or {}
    decision_type = str(decision.get("type") or "ignore").lower()
    draft = dict(state.get("draft") or {})
    draft_original = dict(state.get("draft") or {})
    chosen_action = None
    moderation_action = str(state.get("moderation_action") or "none")

    if decision_type == "edit":
        args = decision.get("args") or {}
        if isinstance(args, dict):
            named_action = str(args.get("action") or "").strip().lower()
            if named_action:
                chosen_action = named_action
            edited = args.get("args") if isinstance(args.get("args"), dict) else args
            if isinstance(edited, dict):
                if edited.get("body"):
                    draft["body"] = edited["body"]
                if edited.get("moderation_action"):
                    moderation_action = str(edited["moderation_action"]).strip().lower()
    elif decision_type == "response":
        text = decision.get("args")
        if isinstance(text, str) and text.strip():
            draft = {"body": text.strip(), "summary": "The owner's own words."}
            chosen_action = ACTION_POST_REPLY
            decision_type = "edit"

    await _repository().update_item(
        state["item_id"],
        owner_decision={**decision, "type": decision_type},
        draft=draft.get("body"),
    )
    return {
        "owner_decision": {**decision, "type": decision_type},
        "draft": draft,
        "draft_original": draft_original,
        "chosen_action": chosen_action,
        "moderation_action": moderation_action,
    }


async def update_preferences(
    state: GroupConversationState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> dict[str, Any]:
    """Every owner decision teaches the next message in this room.

    This is the part of the feature that matters most: the same decision is
    written three ways — against this person in this room, against the room
    itself, and against the kind of message anywhere — so what the owner
    decides once is applied to the next similar message, whoever sends it.
    """
    decision = state.get("owner_decision") or {}
    classification = state.get("classification") or {}
    decision_type = str(decision.get("type") or "ignore").lower()
    message_kind = str(classification.get("message_kind") or "") or None
    repository = _repository()

    edit_summary = None
    if decision_type == "edit":
        edit_summary = "The owner changed what the avatar was going to do."
        chosen = str(state.get("chosen_action") or "")
        if chosen:
            edit_summary = f"The owner chose {chosen} instead."

    # Three rows, three levels of generality. The room-level row is keyed by
    # the room rather than by an empty sender, because the unique key spans
    # (assistant_id, sender, message_kind, decision): an empty sender there
    # would collide with the kind-only row and silently count up one row
    # instead of writing two.
    for sender, sender_domain in (
        (_speaker_key(state), _room_key(state)),
        (_room_key(state), _room_key(state)),
        ("", ""),
    ):
        if not message_kind and not sender:
            continue
        await repository.record_preference(
            user_id=state["user_id"],
            assistant_id=state["assistant_id"],
            sender=sender,
            sender_domain=sender_domain,
            message_kind=message_kind,
            decision=decision_type,
            edit_summary=edit_summary,
            example_subject=str(_event_of(state).text or "")[:200],
        )

    store = getattr(runtime, "store", None)
    if store is not None and decision_type in ("accept", "edit"):
        # The owner's own words become a rule the next similar message follows.
        # The precedent that a timeout or a ban requires is written by
        # record_outcome, which is where the decision record itself is stored.
        note = str((decision.get("args") or {}).get("note") or "") if isinstance(decision.get("args"), dict) else ""
        if note.strip():
            await store_policy_rule(
                store,
                state["user_id"],
                state["assistant_id"],
                rule=note.strip(),
                rule_context=f"Said by the owner about {_room_key(state)}.",
                source="correction",
            )
    return {}


def route_by_action(state: GroupConversationState) -> str:
    """Carry out whichever action the owner chose."""
    decision_type = str((state.get("owner_decision") or {}).get("type") or "ignore").lower()
    if decision_type == "ignore":
        return "record_outcome"
    action = str(state.get("chosen_action") or "").strip().lower()
    if not action:
        classification_decision = (state.get("classification") or {}).get("decision")
        if classification_decision == DECISION_MODERATE:
            action = ACTION_MODERATE
        elif (state.get("draft") or {}).get("body"):
            action = ACTION_POST_REPLY
        else:
            action = ACTION_NOTIFY_OWNER
    if action == ACTION_POST_REPLY:
        return "post_reply" if (state.get("draft") or {}).get("body") else "record_outcome"
    if action == ACTION_MODERATE:
        return "apply_moderation"
    return "record_outcome"


async def post_reply(
    state: GroupConversationState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> dict[str, Any]:
    """Hand the reply back for the bot to post in the room.

    Nothing is transmitted here. The bot that posted the batch receives the
    reply in the decision and posts the reply itself, which is what lets one
    brain serve Slack, Discord and Twitch without knowing any of them.
    """
    automatic = not state.get("owner_decision")
    return {"outcome": STATE_AUTO_SENT if automatic else STATE_SENT}


async def apply_moderation(
    state: GroupConversationState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> dict[str, Any]:
    """Hand the moderation action back for the bot to carry out."""
    moderation_action = str(state.get("moderation_action") or "none")
    if moderation_action == "none":
        return {"outcome": STATE_RESOLVED}
    logger.info(
        "Group moderation %s decided for %s in %s",
        moderation_action,
        state.get("speaker_name"),
        _room_key(state),
    )
    automatic = not state.get("owner_decision")
    return {"outcome": STATE_AUTO_SENT if automatic else STATE_SENT}


async def record_outcome(
    state: GroupConversationState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> dict[str, Any]:
    """Close the item and record the decision, so the next similar message can recall it."""
    from datetime import UTC, datetime

    classification = state.get("classification") or {}
    owner_decision = state.get("owner_decision")
    outcome = state.get("outcome")
    if not outcome:
        if classification.get("decision") == DECISION_IGNORE:
            outcome = STATE_IGNORED
        elif owner_decision and str(owner_decision.get("type") or "").lower() == "ignore":
            outcome = STATE_IGNORED
        else:
            outcome = STATE_RESOLVED
    await _repository().update_item(
        state["item_id"],
        state=outcome,
        resolved_at=datetime.now(UTC).isoformat(),
        # The row has to describe the decision on its own: a bot that resends a
        # batch after a network failure is answered from the row rather than
        # decided a second time.
        confidence_detail={
            **(state.get("confidence_detail") or {}),
            "action": str(
                state.get("chosen_action") or classification.get("decision") or ""
            ),
            "moderation_action": str(state.get("moderation_action") or "none"),
            "mentioned": bool(state.get("mentioned")),
        },
    )
    store = getattr(runtime, "store", None)
    if store is not None:
        await store_decision_record(
            store,
            state["user_id"],
            state["assistant_id"],
            platform=str(state.get("platform") or ""),
            channel_id=str(state.get("channel_id") or ""),
            channel_name=str(state.get("channel_name") or ""),
            event=_event_of(state),
            action=str(
                state.get("chosen_action")
                or classification.get("decision")
                or (DECISION_RESPOND if state.get("mentioned") else DECISION_IGNORE)
            ),
            moderation_action=str(state.get("moderation_action") or "none"),
            reasoning=str(classification.get("reason") or ""),
            confidence=float(state.get("confidence") or 0.0),
            applied_rule=str(classification.get("applied_rule") or ""),
            reply_text=(state.get("draft") or {}).get("body"),
            # An action the owner accepted or chose is the owner allowing that
            # action in this room, which is the precedent an irreversible
            # action needs the next time one is proposed here.
            owner_approved=str(
                (state.get("owner_decision") or {}).get("type") or ""
            ).lower()
            in ("accept", "edit"),
        )
    return {"outcome": outcome}


def build_group_workflow() -> StateGraph:
    """Assemble the graph (uncompiled)."""
    workflow = StateGraph(GroupConversationState, context_schema=GlobalContext)
    workflow.add_node("accept_event", accept_event)
    workflow.add_node("recall_precedent", recall_precedent)
    workflow.add_node("classify", classify)
    workflow.add_node("draft_in_voice", draft_in_voice)
    workflow.add_node("score_confidence", score_confidence)
    workflow.add_node("await_owner", await_owner)
    workflow.add_node("apply_owner_decision", apply_owner_decision)
    workflow.add_node("update_preferences", update_preferences)
    workflow.add_node("post_reply", post_reply)
    workflow.add_node("apply_moderation", apply_moderation)
    workflow.add_node("record_outcome", record_outcome)

    workflow.add_edge(START, "accept_event")
    workflow.add_edge("accept_event", "recall_precedent")
    workflow.add_conditional_edges(
        "recall_precedent",
        route_by_mention,
        {"draft_in_voice": "draft_in_voice", "classify": "classify"},
    )
    workflow.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "record_outcome": "record_outcome",
            "draft_in_voice": "draft_in_voice",
            "score_confidence": "score_confidence",
            "await_owner": "await_owner",
        },
    )
    workflow.add_conditional_edges(
        "draft_in_voice",
        route_after_draft,
        {
            "post_reply": "post_reply",
            "score_confidence": "score_confidence",
            "await_owner": "await_owner",
        },
    )
    workflow.add_conditional_edges(
        "score_confidence",
        route_after_confidence,
        {
            "post_reply": "post_reply",
            "apply_moderation": "apply_moderation",
            "await_owner": "await_owner",
        },
    )
    workflow.add_edge("await_owner", "apply_owner_decision")
    workflow.add_edge("apply_owner_decision", "update_preferences")
    workflow.add_conditional_edges(
        "update_preferences",
        route_by_action,
        {
            "post_reply": "post_reply",
            "apply_moderation": "apply_moderation",
            "record_outcome": "record_outcome",
        },
    )
    workflow.add_edge("post_reply", "record_outcome")
    workflow.add_edge("apply_moderation", "record_outcome")
    workflow.add_edge("record_outcome", END)
    return workflow


group_conversation_workflow = build_group_workflow()


def build_group_graph(checkpointer: Any = None, store: Any = None):
    """Compile with the application's durable checkpointer (for in-process runs)."""
    return group_conversation_workflow.compile(checkpointer=checkpointer, store=store)


# The export the LangGraph server registers (``langgraph.json``); the server
# supplies its own checkpointer.
group_conversation_graph = group_conversation_workflow.compile()
