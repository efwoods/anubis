"""The inbox triage graph.

One run per incoming message, on a thread whose id is the inbox item's id, so a
pending human decision can be resumed from the panel, from chat, or from the
LangChain Agent Inbox app — every one of them delivers the same
``HumanResponse`` to the same interrupt.

    START → accept_message → recall_preferences → classify
      ignore  → record_outcome → END
      notify  → await_owner (interrupt: notify_owner) → apply_owner_decision
                → update_preferences → record_outcome → END
      respond → draft_with_avatar → score_confidence → confidence_gate
                  high → send_reply → record_outcome → END
                  low  → await_owner (interrupt: send_reply) → apply_owner_decision
                         → update_preferences → send_reply → record_outcome → END

Interrupt payloads use the Agent Inbox ``HumanInterrupt`` schema exactly
(``action_request``, ``config``, ``description``) and resume values are
``HumanResponse`` lists (``[{"type": accept|edit|ignore|response, "args"}]``).

The owner's preferences drive the confidence score (see ``inbox/triage.py``)
and are UPDATED on both human-decision routes, so each decision moves the next
similar message toward an automatic reply or toward ignore. Nothing is sent
except through the confidence gate or an accepted/edited decision.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.inbox.repository import (
    DECISION_IGNORE,
    DECISION_RESPOND,
    STATE_AUTO_SENT,
    STATE_FAILED,
    STATE_IGNORED,
    STATE_PENDING_OWNER,
    STATE_RESOLVED,
    STATE_SENT,
    get_inbox_repository,
    sender_domain_of,
)

logger = logging.getLogger(__name__)

ACTION_NOTIFY_OWNER = "notify_owner"
ACTION_SEND_REPLY = "send_reply"


class InboxState(TypedDict, total=False):
    """Everything one triage run carries between nodes."""

    item_id: str
    user_id: str
    assistant_id: str
    assistant_name: str
    account_key: str | None
    message: dict[str, Any]
    preferences: list[dict[str, Any]]
    classification: dict[str, Any]
    draft: dict[str, Any] | None
    confidence: float
    confidence_detail: dict[str, Any]
    owner_decision: dict[str, Any] | None
    outcome: str
    error: str | None


def _repository() -> Any:
    repository = get_inbox_repository()
    if repository is None:
        raise RuntimeError("The inbox repository has not been published.")
    return repository


async def accept_message(
    state: InboxState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> dict[str, Any]:
    """Normalize the incoming message and make sure its item row exists."""
    message = dict(state.get("message") or {})
    message.setdefault("received_at", message.get("sent_at"))
    return {"message": message, "error": None}


async def recall_preferences(
    state: InboxState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> dict[str, Any]:
    """Read what the owner has decided for this sender / domain before."""
    message = state["message"]
    repository = _repository()
    preferences = await repository.recall_preferences(
        assistant_id=state["assistant_id"],
        sender=message.get("sender"),
        sender_domain=sender_domain_of(message.get("sender")),
        message_kind=None,
    )
    return {"preferences": preferences}


async def classify(
    state: InboxState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> dict[str, Any]:
    """Structured classification: ignore / notify / respond."""
    from src.anubis.utils.inbox.triage import classify_message

    classification = await classify_message(
        runtime.context,
        message=state["message"],
        preferences=state.get("preferences") or [],
    )
    # Re-read preferences now that the message kind is known, so the kind's
    # precedent joins the sender's.
    repository = _repository()
    preferences = await repository.recall_preferences(
        assistant_id=state["assistant_id"],
        sender=state["message"].get("sender"),
        sender_domain=sender_domain_of(state["message"].get("sender")),
        message_kind=classification.message_kind,
    )
    await repository.update_item(
        state["item_id"],
        message_kind=classification.message_kind,
        decision=classification.decision,
        needs_owner_action=classification.needs_owner_action,
        reason=classification.reason,
    )
    return {"classification": classification.model_dump(), "preferences": preferences}


def route_after_classify(state: InboxState) -> str:
    """Send ignores straight to the outcome, replies to drafting, the rest to the owner."""
    decision = (state.get("classification") or {}).get("decision")
    if decision == DECISION_IGNORE:
        return "record_outcome"
    if decision == DECISION_RESPOND:
        return "draft_with_avatar"
    return "await_owner"


async def _voice_system_prompt(
    state: InboxState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> str:
    """Build the avatar's consciousness prompt so the draft is in the owner's voice."""
    try:
        from langchain_core.messages import HumanMessage

        from src.anubis.utils.nodes import _build_consciousness_system_message_update

        pseudo_state = {
            "messages": [
                HumanMessage(
                    content=str(state["message"].get("body_text") or "")[:2000]
                )
            ],
            "user_state": {
                "user_id": state["user_id"],
                "user_name": "",
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
        logger.debug(
            "Consciousness prompt unavailable for the inbox draft", exc_info=True
        )
    return (
        f"You are {state.get('assistant_name') or 'the owner'}, writing a reply to your own "
        "email in your own voice."
    )


async def draft_with_avatar(
    state: InboxState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> dict[str, Any]:
    """Write the reply under the avatar's consciousness prompt."""
    from src.anubis.utils.inbox.triage import draft_reply

    voice_prompt = await _voice_system_prompt(state, config, runtime)
    draft = await draft_reply(
        runtime.context, message=state["message"], voice_system_prompt=voice_prompt
    )
    await _repository().update_item(state["item_id"], draft=draft.body)
    return {"draft": draft.model_dump()}


async def score_confidence(
    state: InboxState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> dict[str, Any]:
    """Preferences drive the score: an alignment judgement times a decision prior."""
    from src.anubis.utils.inbox.triage import (
        DraftReply,
        combine_confidence,
        judge_alignment,
        preference_prior,
    )

    threshold = float(
        getattr(runtime.context, "inbox_auto_send_confidence", None) or 0.9
    )
    draft = DraftReply(
        **(state.get("draft") or {"subject": "", "body": "", "summary": ""})
    )
    preferences = state.get("preferences") or []
    alignment = await judge_alignment(
        runtime.context, message=state["message"], draft=draft, preferences=preferences
    )
    prior, prior_reason = preference_prior(preferences, auto_send_threshold=threshold)
    confidence = combine_confidence(alignment.alignment_score, prior)
    detail = {
        "alignment_score": alignment.alignment_score,
        "alignment_reason": alignment.reason,
        "prior": prior,
        "prior_reason": prior_reason,
        "threshold": threshold,
    }
    await _repository().update_item(
        state["item_id"], confidence=confidence, confidence_detail=detail
    )
    return {"confidence": confidence, "confidence_detail": detail}


def route_after_confidence(
    state: InboxState, runtime: Runtime[GlobalContext] | None = None
) -> str:
    """Send above the threshold when no real-world action is needed; otherwise ask the owner."""
    detail = state.get("confidence_detail") or {}
    threshold = float(detail.get("threshold") or 0.9)
    needs_owner = bool((state.get("classification") or {}).get("needs_owner_action"))
    if not needs_owner and float(state.get("confidence") or 0.0) >= threshold:
        return "send_reply"
    return "await_owner"


def _human_interrupt(state: InboxState) -> dict[str, Any]:
    """Build the Agent Inbox ``HumanInterrupt`` for this item."""
    classification = state.get("classification") or {}
    message = state["message"]
    if classification.get("decision") == DECISION_RESPOND:
        draft = state.get("draft") or {}
        return {
            "action_request": {
                "action": ACTION_SEND_REPLY,
                "args": {
                    "to": message.get("sender"),
                    "subject": draft.get("subject"),
                    "body": draft.get("body"),
                    "in_reply_to": message.get("rfc822_message_id"),
                },
            },
            "config": {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": True,
                "allow_accept": True,
            },
            "description": (
                f'Reply to {message.get("sender")} about "{message.get("subject")}"? '
                f"Confidence {float(state.get('confidence') or 0.0):.2f}: "
                f"{(state.get('confidence_detail') or {}).get('alignment_reason') or classification.get('reason') or ''}"
            ),
        }
    return {
        "action_request": {
            "action": ACTION_NOTIFY_OWNER,
            "args": {
                "from": message.get("sender"),
                "subject": message.get("subject"),
                "summary": classification.get("reason"),
                "needs_owner_action": classification.get("needs_owner_action"),
            },
        },
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": False,
            "allow_accept": True,
        },
        "description": (
            f'{message.get("sender")} — "{message.get("subject")}": {classification.get("reason") or ""}'
        ),
    }


async def await_owner(
    state: InboxState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> dict[str, Any]:
    """Pause for the owner. The item is marked pending before the interrupt.

    Everything after ``interrupt`` runs again on resume, so the mirror write
    happens before it and the resume value is the only thing read after it.
    """
    repository = _repository()
    await repository.update_item(state["item_id"], state=STATE_PENDING_OWNER)
    payload = _human_interrupt(state)
    decision = interrupt(payload)
    responses = decision if isinstance(decision, list) else [decision]
    first = next((entry for entry in responses if isinstance(entry, dict)), None) or {}
    return {"owner_decision": first}


async def apply_owner_decision(
    state: InboxState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> dict[str, Any]:
    """Turn the ``HumanResponse`` into the draft to send (or not)."""
    decision = state.get("owner_decision") or {}
    decision_type = str(decision.get("type") or "ignore").lower()
    draft = dict(state.get("draft") or {})
    if decision_type == "edit":
        args = decision.get("args") or {}
        if isinstance(args, dict):
            edited = args.get("args") if isinstance(args.get("args"), dict) else args
            draft["subject"] = edited.get("subject") or draft.get("subject")
            draft["body"] = edited.get("body") or draft.get("body")
    elif decision_type == "response":
        # Free text from the owner: a notify item answered from chat, or a
        # reply the owner dictated instead of the draft.
        text = decision.get("args")
        if isinstance(text, str) and text.strip():
            draft = {
                "subject": draft.get("subject")
                or f"Re: {state['message'].get('subject') or ''}",
                "body": text.strip(),
                "summary": "The owner's own reply.",
            }
            decision_type = "edit"
    await _repository().update_item(
        state["item_id"],
        owner_decision={**decision, "type": decision_type},
        draft=draft.get("body"),
    )
    return {"owner_decision": {**decision, "type": decision_type}, "draft": draft}


async def update_preferences(
    state: InboxState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> dict[str, Any]:
    """Every owner decision teaches the next triage of a similar message."""
    decision = state.get("owner_decision") or {}
    classification = state.get("classification") or {}
    message = state["message"]
    decision_type = str(decision.get("type") or "ignore").lower()
    edit_summary = None
    if decision_type == "edit":
        edit_summary = "The owner edited the draft before sending."
    await _repository().record_preference(
        user_id=state["user_id"],
        assistant_id=state["assistant_id"],
        sender=message.get("sender"),
        sender_domain=sender_domain_of(message.get("sender")),
        message_kind=classification.get("message_kind"),
        decision=decision_type,
        edit_summary=edit_summary,
        example_subject=message.get("subject"),
    )
    return {}


def route_after_owner(state: InboxState) -> str:
    """Send when the owner accepted or edited a reply; otherwise just record the outcome."""
    decision_type = str(
        (state.get("owner_decision") or {}).get("type") or "ignore"
    ).lower()
    is_reply_item = (state.get("classification") or {}).get(
        "decision"
    ) == DECISION_RESPOND
    if is_reply_item and decision_type in ("accept", "edit"):
        return "send_reply"
    return "record_outcome"


async def send_reply(
    state: InboxState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> dict[str, Any]:
    """Transmit the draft through the connected mailbox."""
    from src.anubis.utils.inbox.delivery import send_email_reply

    draft = state.get("draft") or {}
    message = state["message"]
    try:
        await send_email_reply(
            runtime.context,
            user_id=state["user_id"],
            account_key=state.get("account_key"),
            to_address=str(message.get("sender") or ""),
            subject=str(draft.get("subject") or f"Re: {message.get('subject') or ''}"),
            body_text=str(draft.get("body") or ""),
            in_reply_to=message.get("rfc822_message_id"),
        )
    except Exception as send_error:  # noqa: BLE001 - recorded, never raised into the run
        logger.warning(
            "Inbox reply could not be sent for %s: %s", state["item_id"], send_error
        )
        return {"outcome": STATE_FAILED, "error": str(send_error)}
    automatic = not state.get("owner_decision")
    return {"outcome": STATE_AUTO_SENT if automatic else STATE_SENT}


async def record_outcome(
    state: InboxState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> dict[str, Any]:
    """Write the final state onto the item row."""
    from datetime import UTC, datetime

    outcome = state.get("outcome")
    classification = state.get("classification") or {}
    decision = state.get("owner_decision")
    if not outcome:
        if classification.get("decision") == DECISION_IGNORE:
            outcome = STATE_IGNORED
        elif decision and str(decision.get("type") or "").lower() == "ignore":
            outcome = STATE_IGNORED
        else:
            outcome = STATE_RESOLVED
    await _repository().update_item(
        state["item_id"],
        state=outcome,
        resolved_at=datetime.now(UTC).isoformat(),
        **(
            {
                "reason": f"{classification.get('reason') or ''} (send failed: {state.get('error')})"
            }
            if state.get("error")
            else {}
        ),
    )
    return {"outcome": outcome}


def build_inbox_workflow() -> StateGraph:
    """Assemble the graph (uncompiled)."""
    workflow = StateGraph(InboxState, context_schema=GlobalContext)
    workflow.add_node("accept_message", accept_message)
    workflow.add_node("recall_preferences", recall_preferences)
    workflow.add_node("classify", classify)
    workflow.add_node("draft_with_avatar", draft_with_avatar)
    workflow.add_node("score_confidence", score_confidence)
    workflow.add_node("await_owner", await_owner)
    workflow.add_node("apply_owner_decision", apply_owner_decision)
    workflow.add_node("update_preferences", update_preferences)
    workflow.add_node("send_reply", send_reply)
    workflow.add_node("record_outcome", record_outcome)

    workflow.add_edge(START, "accept_message")
    workflow.add_edge("accept_message", "recall_preferences")
    workflow.add_edge("recall_preferences", "classify")
    workflow.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "record_outcome": "record_outcome",
            "draft_with_avatar": "draft_with_avatar",
            "await_owner": "await_owner",
        },
    )
    workflow.add_edge("draft_with_avatar", "score_confidence")
    workflow.add_conditional_edges(
        "score_confidence",
        route_after_confidence,
        {"send_reply": "send_reply", "await_owner": "await_owner"},
    )
    workflow.add_edge("await_owner", "apply_owner_decision")
    workflow.add_edge("apply_owner_decision", "update_preferences")
    workflow.add_conditional_edges(
        "update_preferences",
        route_after_owner,
        {"send_reply": "send_reply", "record_outcome": "record_outcome"},
    )
    workflow.add_edge("send_reply", "record_outcome")
    workflow.add_edge("record_outcome", END)
    return workflow


inbox_workflow = build_inbox_workflow()


def build_inbox_graph(checkpointer: Any = None, store: Any = None):
    """Compile with the application's durable checkpointer (for in-process runs)."""
    return inbox_workflow.compile(checkpointer=checkpointer, store=store)


# The export the LangGraph server registers (``langgraph.json``); the server
# supplies its own checkpointer.
inbox_graph = inbox_workflow.compile()
