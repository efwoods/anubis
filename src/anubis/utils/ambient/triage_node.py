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
    DECISION_RESPOND,
    ambient_details,
    ambient_speech_cooldown,
    compose_observation_text,
    is_ambient_observation,
    is_narration_observation,
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

#: The ``observation_kind`` recorded on a narrated scene, so the reply can be
#: told apart from an ordinary one wherever observations are read back.
NARRATION_OBSERVATION_KIND = "scene_narration"

#: What ``route_after_ambient_triage`` is handed for a narrated observation, so
#: the run ends at the description instead of paying for a rephrasing of it.
#: Kept apart from the decision the CLIENT is told (``respond``), which stays
#: what it always was so that nothing downstream has to learn a new word.
DECISION_NARRATE_ROUTE = "narrate"

#: What a narration turn records as its reason. The decision was not made by a
#: classifier, and a record that pretended otherwise would be a lie in the
#: thread the conversation partner can read.
NARRATION_REASON = (
    "Scene narration is switched on: the conversation partner asked to be told "
    "what is in view, so every observation is described to them."
)


def _narration_decision(body: str) -> dict[str, Any]:
    """Record the decision for an observation captured under scene narration.

    Narration answers a request the conversation partner already made, once,
    deliberately, and usually because they cannot see the scene themselves.
    There is nothing left to decide: classifying each frame would spend a model
    call and a store query to re-derive an answer that was settled when the
    switch was turned on, and would sometimes come back "ignore" — which here
    means leaving a person waiting in silence for a description they asked for.

    This is the one place the salience floors and the quiet period after the
    avatar last spoke do not apply, and that is deliberate rather than an
    oversight: those exist to keep the avatar from speaking with nothing to
    say, and a standing request is the opposite situation.
    """
    spoken = " ".join((body or "").split())
    return {
        "decision": DECISION_RESPOND,
        # The words the browser will read out, in full. Not truncated the way
        # an ordinary summary is: this IS the reading, and a description cut
        # off at 300 characters is a hazard left unsaid.
        "narration": spoken,
        "summary": spoken[:300],
        "reason": NARRATION_REASON,
        "observation_kind": NARRATION_OBSERVATION_KIND,
        "salience": 1.0,
        "needs_owner_action": False,
        "proposed_action": "none",
        "action_description": "",
    }


def route_after_image_resolution(
    state: GlobalState,
) -> Literal["ambient_triage", "anubis"]:
    """Send an ambient observation to triage; every other turn to the avatar."""
    messages = state.get("messages") or []
    if messages and is_ambient_observation(messages[-1]):
        return AMBIENT_TRIAGE_NODE
    return "anubis"


def route_after_ambient_triage(state: GlobalState) -> str:
    """``ignore`` ends the run; ``respond`` and ``notify`` reach the avatar.

    Scene narration ends the run too, and that is the single biggest thing
    making the mode usable. The description the vision pass just produced is
    ALREADY the sentence meant to be read out — second person, hazards first,
    under eighty words — so sending it through the avatar for rephrasing bought
    nothing and cost everything: a full deep-agent turn between the camera and
    the person's ears, which is what made readings arrive half a minute apart
    when the browser was asking every few seconds. It also billed a reply per
    reading and filled the thread with one avatar message every few seconds.

    The description is carried to the browser on the ``ambient_decision`` frame
    and spoken there, in the avatar's own voice, the moment it arrives. The
    observation itself stays on the thread as context, so the avatar can still
    talk about what it saw when asked.
    """
    decision = str(state.get("route_decision") or DECISION_IGNORE).strip().lower()
    if decision in (DECISION_IGNORE, DECISION_NARRATE_ROUTE):
        return END
    return "anubis"


def _writer():
    try:
        return get_stream_writer()
    except Exception:  # noqa: BLE001 - outside a run there is no stream
        return lambda _payload: None


def _gate_by_salience_and_cooldown(
    *,
    decision: str,
    salience: float,
    thread_id: str | None,
    context: GlobalContext,
) -> tuple[str, str | None]:
    """Decide whether a 'respond' or 'notify' is actually allowed to interrupt.

    The classifier judges one observation at a time and has no sense of its own
    rate, so two limits sit between the classifier's judgement and the
    conversation partner. A decision below the salience floor for that decision
    is not worth an interruption at all. A decision arriving inside the quiet
    period after the avatar last spoke is not worth breaking that quiet for,
    unless the observation is salient enough to override the quiet period.

    Returns the decision to act on, and the reason it was demoted when it was.
    """
    if decision == DECISION_IGNORE:
        return decision, None

    floor = (
        context.ambient_respond_salience_floor
        if decision == DECISION_RESPOND
        else context.ambient_notify_salience_floor
    )
    floor_value = float(floor if floor is not None else 0.0)
    if salience < floor_value:
        return (
            DECISION_IGNORE,
            f"salience {salience:.2f} is below the {decision} floor {floor_value:.2f}",
        )

    override = float(
        context.ambient_respond_cooldown_override_salience
        if context.ambient_respond_cooldown_override_salience is not None
        else 1.0
    )
    if salience >= override:
        return decision, None

    remaining = ambient_speech_cooldown.seconds_remaining(
        thread_id,
        float(
            context.ambient_respond_cooldown_seconds
            if context.ambient_respond_cooldown_seconds is not None
            else 0.0
        ),
    )
    if remaining is not None:
        return (
            DECISION_IGNORE,
            f"the avatar spoke recently; {remaining:.0f}s of quiet remain and "
            f"salience {salience:.2f} does not reach the override {override:.2f}",
        )
    return decision, None


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

    if is_narration_observation(ambient):
        # Narration bypasses the classifier entirely (see
        # ``_narration_decision``): no model call, no preference recall, no
        # salience gate, no cooldown. The conversation partner is waiting to
        # hear this one, and every stage skipped here is latency they spend
        # standing still.
        narrated = {**ambient, **_narration_decision(body)}
        _writer()({"type": "ambient_decision", **narrated})
        rewritten_narration = make_hidden_human_message(
            compose_observation_text(narrated, body),
            {
                **(last.additional_kwargs or {}),
                "hidden": bool((last.additional_kwargs or {}).get("hidden", True)),
                "ambient": narrated,
            },
            message_id=last.id,
        )
        return {
            "messages": [RemoveMessage(id=last.id), rewritten_narration],
            # Ends the run: the description is the reading, and it has already
            # been sent to the browser on the decision frame above.
            "route_decision": DECISION_NARRATE_ROUTE,
        }

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
            sources=[str(source) for source in (ambient.get("sources") or [])],
            camera_facing=ambient.get("camera_facing"),
        )
        decision_fields = {
            "decision": classification.decision,
            "summary": classification.summary,
            "reason": classification.reason,
            "observation_kind": classification.observation_kind,
            "salience": classification.salience,
            "needs_owner_action": classification.needs_owner_action,
            # What the avatar offers to do once allowed; the card shows a
            # button for this and the allowed action comes back as a turn.
            "proposed_action": classification.proposed_action,
            "action_description": classification.action_description,
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
            "proposed_action": "none",
            "action_description": "",
        }

    thread_id = (config.get("configurable") or {}).get("thread_id")
    classified_salience = decision_fields["salience"]
    gated_decision, demotion_reason = _gate_by_salience_and_cooldown(
        decision=str(decision_fields["decision"]),
        salience=(
            float(classified_salience)
            if isinstance(classified_salience, (int, float))
            else 0.0
        ),
        thread_id=str(thread_id) if thread_id else None,
        context=context,
    )
    if demotion_reason is not None:
        logger.info(
            "ambient_triage: demoting %s to %s (%s)",
            decision_fields["decision"],
            gated_decision,
            demotion_reason,
        )
        decision_fields["demoted_from"] = decision_fields["decision"]
        decision_fields["demotion_reason"] = demotion_reason
        decision_fields["decision"] = gated_decision
        # A demoted observation proposes nothing: the card that would have
        # carried the offer is never shown.
        decision_fields["proposed_action"] = "none"
        decision_fields["action_description"] = ""
    elif gated_decision != DECISION_IGNORE:
        # Only a decision that actually reaches the conversation partner starts
        # the quiet period, so an observation silenced for any other reason does
        # not begin a cooldown it never earned.
        ambient_speech_cooldown.mark_spoken(str(thread_id) if thread_id else None)

    updated_ambient = {**ambient, **decision_fields}
    _writer()({"type": "ambient_decision", **updated_ambient})

    # A spoken turn heard in the room stays visible (``hidden`` False); a
    # webcam / screen observation stays hidden.
    additional_kwargs = {
        **(last.additional_kwargs or {}),
        "hidden": bool((last.additional_kwargs or {}).get("hidden", True)),
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
