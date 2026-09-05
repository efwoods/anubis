"""The ``ambient_triage`` node of the outer message workflow and its routing.

Runs only when the last message is an ambient observation (after
``resolve_human_message_images`` has turned the images into text). Classifies
the observation, emits an ``ambient_decision`` stream event for the client,
and rewrites the hidden message in place with the decision recorded in
``additional_kwargs["ambient"]`` and, for ``respond`` / ``notify``, the
instruction the deep agent follows. ``ignore`` ends the run: the hidden
observation is persisted as context and no model reply is produced.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END
from langgraph.runtime import Runtime

from src.anubis.utils.ambient.observations import (
    DECISION_IGNORE,
    ambient_details,
    compose_observation_text,
    is_ambient_observation,
    make_hidden_human_message,
    message_text,
    recent_ambient_observations,
    recent_visible_messages,
    split_observation_text,
    strip_instruction,
)
from src.anubis.utils.ambient.preferences import recall_ambient_preferences
from src.anubis.utils.ambient.triage import classify_observation
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.state import GlobalState

logger = logging.getLogger(__name__)

AMBIENT_TRIAGE_NODE = "ambient_triage"
RECENT_VISIBLE_TURNS_FOR_TRIAGE = 6
EARLIER_OBSERVATIONS_FOR_TRIAGE = 5


def route_after_image_resolution(
    state: GlobalState,
) -> Literal["ambient_triage", "anubis"]:
    """Send an ambient observation to triage; every other turn to the avatar."""
    messages = state.get("messages") or []
    if messages and is_ambient_observation(messages[-1]):
        return AMBIENT_TRIAGE_NODE
    return "anubis"


def route_after_ambient_triage(state: GlobalState) -> str:
    """``ignore`` ends the run; ``respond`` and ``notify`` reach the avatar."""
    decision = str(state.get("route_decision") or DECISION_IGNORE).strip().lower()
    return END if decision == DECISION_IGNORE else "anubis"


def _writer():
    try:
        return get_stream_writer()
    except Exception:  # noqa: BLE001 - outside a run there is no stream
        return lambda _payload: None


async def ambient_triage(
    state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]
) -> dict[str, Any]:
    """Classify the last (ambient) message and record the decision on the message."""
    messages = list(state.get("messages") or [])
    if not messages or not is_ambient_observation(messages[-1]):
        return {"route_decision": DECISION_IGNORE}
    last = messages[-1]
    if not isinstance(last, HumanMessage) or not last.id:
        logger.warning(
            "ambient_triage: last message is not an addressable HumanMessage"
        )
        return {"route_decision": DECISION_IGNORE}

    ambient = ambient_details(last) or {}
    _header, body = split_observation_text(message_text(last))
    body = strip_instruction(body)
    context = (
        runtime.context
        if isinstance(runtime.context, GlobalContext)
        else GlobalContext()
    )
    user_id = (state.get("user_state") or {}).get("user_id")
    assistant_id = (state.get("assistant_state") or {}).get("assistant_id")
    assistant_name = (state.get("assistant_state") or {}).get("assistant_name") or ""

    preferences = await recall_ambient_preferences(
        runtime.store,
        user_id,
        assistant_id,
        query=body,
        limit=int(context.ambient_preference_recall_limit or 8),
    )
    previous_observations = recent_ambient_observations(
        messages[:-1], EARLIER_OBSERVATIONS_FOR_TRIAGE
    )
    recent_messages = recent_visible_messages(
        messages[:-1], RECENT_VISIBLE_TURNS_FOR_TRIAGE
    )

    try:
        classification = await classify_observation(
            context,
            assistant_name=assistant_name,
            observation_text=body,
            recent_messages=recent_messages,
            previous_observations=previous_observations,
            preferences=preferences,
            voice_mode=bool(ambient.get("voice_mode")),
        )
        decision_fields = {
            "decision": classification.decision,
            "summary": classification.summary,
            "reason": classification.reason,
            "observation_kind": classification.observation_kind,
            "salience": classification.salience,
            "needs_owner_action": classification.needs_owner_action,
        }
    except Exception:  # noqa: BLE001 - an undecidable observation is ignored
        logger.exception("Ambient triage failed; ignoring the observation")
        decision_fields = {
            "decision": DECISION_IGNORE,
            "summary": "",
            "reason": "The observation could not be classified.",
            "observation_kind": "other",
            "salience": 0.0,
            "needs_owner_action": False,
        }

    updated_ambient = {**ambient, **decision_fields}
    _writer()({"type": "ambient_decision", **updated_ambient})

    additional_kwargs = {
        **(last.additional_kwargs or {}),
        "hidden": True,
        "ambient": updated_ambient,
    }
    rewritten = make_hidden_human_message(
        compose_observation_text(updated_ambient, body),
        additional_kwargs,
        message_id=last.id,
    )
    return {
        "messages": [RemoveMessage(id=last.id), rewritten],
        "route_decision": updated_ambient["decision"],
    }
