"""Triage of one ambient observation: ignore, respond, or notify.

One structured-output call, modelled on the email inbox's
``classify_message``. The classifier reads the fresh observation together with
the last few visible turns, the earlier observations in the thread, and the
owner's recorded preferences, and returns an ``AmbientTriageClassification``.
An undecidable or failed classification is ``ignore``: an observation the
avatar cannot judge must never turn into a stream of interruptions.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from src.anubis.utils.ambient.observations import (
    AMBIENT_DECISIONS,
    DECISION_IGNORE,
)

logger = logging.getLogger(__name__)


class AmbientTriageClassification(BaseModel):
    """Decide what the avatar does with one thing the avatar just noticed."""

    decision: str = Field(
        description=(
            "One of 'ignore' (ordinary activity, nothing to add), 'respond' (the "
            "avatar would naturally say something or do something now), or "
            "'notify' (the conversation partner should be told about this)."
        )
    )
    needs_owner_action: bool = Field(
        description=(
            "True when the conversation partner must act in the real world on "
            "what was seen (fix an error, answer a call, leave for an appointment)."
        )
    )
    observation_kind: str = Field(
        description=(
            "A short lowercase label for the kind of scene, used to remember the "
            "conversation partner's preference for that kind next time: for example "
            "'writing_code', 'video_call', 'error_dialog', 'person_absent', "
            "'reading_documentation'."
        )
    )
    summary: str = Field(
        description=(
            "One line, present tense, neutral third person, saying what is "
            "happening. Do not name a camera or a screenshot."
        )
    )
    salience: float = Field(
        description="How much this observation matters right now, from 0.0 to 1.0."
    )
    reason: str = Field(description="One or two sentences justifying the decision.")


AMBIENT_CLASSIFY_SYSTEM_PROMPT = """<TASK>
The assistant reviews one ambient observation on behalf of the avatar the conversation partner is talking with. An ambient observation is a written description of what the conversation partner's webcam shows and what is on the conversation partner's screen, captured automatically while the conversation continues. Decide whether the avatar should ignore the observation, respond to the conversation partner now, or notify the conversation partner about what was seen.
</TASK>
<RULES>
- Choose 'ignore' for ordinary activity that continues what was already seen: the conversation partner working, reading, typing, or sitting quietly, and any scene that changed little since the earlier observations. Ignore is the default; most observations are ignored.
- Choose 'respond' only when the avatar would naturally speak up as a present friend would: the conversation partner is visibly stuck on something the avatar can help with, something on the screen directly concerns the current conversation, the conversation partner is looking at the avatar as if waiting for the avatar, or a notable change happened that a friend in the room would remark on. When VOICE_MODE is true, choose 'respond' only when speaking aloud would not interrupt the conversation partner.
- Choose 'notify' when the conversation partner should see something or act in the real world and is not looking at the avatar: an error or alert on the screen, a message or a call that needs a reply, an appointment or a deadline visible on the screen, a safety concern, or anything consequential and ambiguous. Never choose 'notify' for the same situation twice in a row when an earlier observation already carried 'notify' for that situation.
- The conversation partner's recorded decisions are precedent. A note written by the conversation partner is a standing instruction and overrides every rule above.
- Name the observation kind with a short lowercase label so the same kind is recognized next time.
- Write the summary in one line, present tense, neutral third person, without naming a camera, a webcam, or a screenshot.
</RULES>
"""


def describe_ambient_preferences(preferences: list[dict[str, Any]]) -> str:
    """Render the owner's recorded decisions for the classifier."""
    if not preferences:
        return (
            "The conversation partner has recorded no decisions about ambient "
            "observations yet."
        )
    lines = []
    for preference in preferences:
        kind = preference.get("observation_kind") or "any kind"
        line = (
            f"- {kind}: the conversation partner chose "
            f"'{preference.get('decision')}' {int(preference.get('count') or 1)} time(s)"
        )
        if preference.get("summary"):
            line += f" for a scene like: {preference['summary']}"
        if preference.get("note"):
            line += f"; note from the conversation partner: {preference['note']}"
        lines.append(line)
    return "\n".join(lines)


def describe_earlier_observations(previous_observations: list[dict[str, Any]]) -> str:
    """Render earlier observations (oldest first) for the classifier."""
    if not previous_observations:
        return "No earlier observations in this conversation."
    lines = []
    for observation in previous_observations:
        lines.append(
            f"- {observation.get('captured_at') or 'earlier'} "
            f"[{observation.get('decision') or 'undecided'}] "
            f"{observation.get('summary') or (observation.get('text') or '')[:300]}"
        )
    return "\n".join(lines)


def build_classification_prompt(
    *,
    assistant_name: str,
    observation_text: str,
    recent_messages: list[str],
    previous_observations: list[dict[str, Any]],
    preferences: list[dict[str, Any]],
    voice_mode: bool,
) -> str:
    """Build the human turn handed to the classifier."""
    conversation = (
        "\n".join(recent_messages) if recent_messages else "No visible turns yet."
    )
    return (
        f"<AVATAR>\n{assistant_name or 'the avatar'}\n</AVATAR>\n\n"
        "<OWNER_PRECEDENT>\n"
        + describe_ambient_preferences(preferences)
        + "\n</OWNER_PRECEDENT>\n\n"
        "<RECENT_CONVERSATION>\n" + conversation + "\n</RECENT_CONVERSATION>\n\n"
        "<EARLIER_OBSERVATIONS>\n"
        + describe_earlier_observations(previous_observations)
        + "\n</EARLIER_OBSERVATIONS>\n\n"
        f"<VOICE_MODE>{'true' if voice_mode else 'false'}</VOICE_MODE>\n\n"
        "<OBSERVATION>\n" + (observation_text or "").strip()[:6000] + "\n</OBSERVATION>"
    )


def normalize_classification(response: Any) -> AmbientTriageClassification:
    """Coerce a structured-output response into a valid classification."""
    decision = str(getattr(response, "decision", "") or "").strip().lower()
    if decision not in AMBIENT_DECISIONS:
        decision = DECISION_IGNORE
    try:
        salience = float(getattr(response, "salience", 0.0) or 0.0)
    except (TypeError, ValueError):
        salience = 0.0
    salience = min(1.0, max(0.0, salience))
    return AmbientTriageClassification(
        decision=decision,
        needs_owner_action=bool(getattr(response, "needs_owner_action", False)),
        observation_kind=(
            str(getattr(response, "observation_kind", "") or "other")
            .strip()
            .lower()[:40]
            or "other"
        ),
        summary=str(getattr(response, "summary", "") or "").strip()[:300],
        salience=salience,
        reason=str(getattr(response, "reason", "") or "").strip(),
    )


async def classify_observation(
    context: Any,
    *,
    assistant_name: str,
    observation_text: str,
    recent_messages: list[str],
    previous_observations: list[dict[str, Any]],
    preferences: list[dict[str, Any]],
    voice_mode: bool,
) -> AmbientTriageClassification:
    """Classify one ambient observation with the owner's preferences as precedent."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from src.anubis.utils.model import init_model

    model = init_model(
        model_without_tools=False, response_format=AmbientTriageClassification
    )
    human = build_classification_prompt(
        assistant_name=assistant_name,
        observation_text=observation_text,
        recent_messages=recent_messages,
        previous_observations=previous_observations,
        preferences=preferences,
        voice_mode=voice_mode,
    )
    response = await model.ainvoke(
        input=[
            SystemMessage(content=AMBIENT_CLASSIFY_SYSTEM_PROMPT),
            HumanMessage(content=human),
        ]
    )
    return normalize_classification(response)
