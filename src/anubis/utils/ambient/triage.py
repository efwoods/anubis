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
    DECISION_NOTIFY,
    PROPOSED_ACTION_NONE,
    SOURCE_WEBCAM,
    normalize_camera_facing_value,
    normalize_proposed_action,
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
    proposed_action: str = Field(
        default="none",
        description=(
            "For a 'notify' decision only: the one verb, one lowercase word, the "
            "avatar will perform once the conversation partner allows this — for "
            "example 'draft', 'reply', 'remind', 'research', 'summarize', "
            "'schedule', 'explain'. 'none' for a plain heads-up. Always 'none' "
            "for 'ignore' and 'respond'."
        ),
    )
    action_description: str = Field(
        default="",
        description=(
            "One short imperative line detailing what that verb means here, "
            "starting with the same verb, for example 'Draft a reply to the "
            "invoice email'. Empty when proposed_action is 'none'."
        ),
    )


AMBIENT_CLASSIFY_SYSTEM_PROMPT = """<TASK>
The assistant reviews one ambient observation on behalf of the avatar the conversation partner is talking with. An ambient observation is a written description of what the conversation partner's webcam shows and what is on the conversation partner's screen, captured automatically while the conversation continues. Decide whether the avatar should ignore the observation, respond to the conversation partner now, or notify the conversation partner about what was seen.
</TASK>
<FIRST_PRINCIPLE>
The webcam and the screen are captured on a timer, not because the conversation partner asked the avatar for anything. The capture itself is never a reason to speak, and neither is the conversation partner being visible, being present, or having been quiet for a while. The avatar speaks when the observation carries an intent the avatar is meant to answer, or when the observation carries a consequence the conversation partner needs to know about. An observation carrying neither is ignored, however long the conversation has been silent.
</FIRST_PRINCIPLE>
<RULES>
- 'ignore' is the default, and most observations are ignored. Choose 'ignore' for ordinary activity that continues what was already seen, and for any scene that changed little since the earlier observations: the conversation partner working, reading, typing, scrolling, or sitting quietly; the conversation partner present but still; a dark, empty, or unchanging room; the conversation partner asleep, resting, or away from the camera; and any scene whose only notable feature is that the conversation partner is there. Being visible is not a request.
- Choose 'respond' only when the observation carries intent the avatar is meant to answer. At least one of the following must hold, and none of them is satisfied by the conversation partner simply being in frame:
  (a) The conversation partner made a bid toward the avatar: speaking to the avatar, gesturing at the camera, holding something up to the camera, or turning to the camera deliberately just after something changed. Merely facing the camera, sitting in front of the camera, or looking at the screen is never a bid, because a conversation partner working at a desk faces the camera continuously.
  (b) The conversation partner deliberately put something in front of the avatar to look at, and the conversation already underway is about that thing.
  (c) The webcam faces the world rather than the conversation partner, as described in CAMERA_FACING. Aiming the camera outward is itself a standing request to be told what the avatar sees.
  When VOICE_MODE is true, choose 'respond' only when speaking aloud would not interrupt the conversation partner.
- Choose 'notify' when the conversation partner should see something or act in the real world and is not looking at the avatar: an error or an alert on the screen, a message or a call that needs a reply, an appointment or a deadline visible on the screen, a safety concern, or anything consequential and ambiguous. A 'notify' decision needs no bid from the conversation partner, because the weight belongs to the event rather than to anything the conversation partner asked for; the event must nevertheless have a real consequence for the conversation partner, and noticing something merely interesting is not enough. Never choose 'notify' for the same situation twice in a row when an earlier observation already carried 'notify' for that situation.
- Never choose 'respond' or 'notify' to fill a silence, to check in, to say the avatar is present or listening, to reassure the conversation partner, or to remark on how long the conversation partner has been working. The conversation partner did not ask for company by leaving a camera on, and a message with nothing behind it costs the conversation partner more than saying nothing does.
- Do not repeat, in different words, something an earlier observation in this conversation already prompted the avatar to say. Read EARLIER_OBSERVATIONS before deciding, and choose 'ignore' when the avatar would only be saying the same thing again.
- Set salience honestly, from zero to one: how much this observation matters to the conversation partner right now. An observation that merely shows the conversation partner present and well is near zero. Salience gates whether the avatar is allowed to speak at all, so do not inflate it to justify a decision.
- The conversation partner's recorded decisions are precedent. A note written by the conversation partner is a standing instruction and overrides every rule above.
- For a 'notify' decision, name a proposed action when the avatar could usefully do something once the conversation partner allows this. The proposed action is ONE verb, one lowercase word, that the avatar will perform: 'draft' (write the answer to that email or message), 'reply' (say something useful about what was seen), 'remind' (set a reminder), 'research' (look the error or the topic up), 'summarize', 'schedule', 'explain', or another single verb that fits. Detail what the verb means here in action_description, starting with the same verb. Choose 'none' for a plain heads-up. Never propose an action for 'ignore' or 'respond'.
- The precedent says how the conversation partner treated earlier offers of the same kind: when the conversation partner let the avatar act and liked the result, offer the action again; when the conversation partner replied in person, disliked what the avatar did, or left the notice alone, prefer a plain heads-up or 'ignore'.
- Name the observation kind with a short lowercase label so the same kind is recognized next time.
- Write the summary in one line, present tense, neutral third person, without naming a camera, a webcam, or a screenshot.
</RULES>
<CAMERA_FACING>
The CAMERA_FACING line says which way the webcam points. The two directions mean opposite things, and the 'respond' threshold differs sharply between them.

- "self" — the webcam faces the conversation partner: the ordinary laptop or desk arrangement, where the conversation partner is the subject of the picture. Presence carries no request here. Apply the 'respond' rules strictly and prefer 'ignore'; a conversation partner sitting in view of a camera has asked for nothing.
- "world" — the webcam faces away from the conversation partner, at whatever the conversation partner is looking at: a phone held up, or worn on a lanyard with the rear camera outward. The picture is the conversation partner's own view of the world, and pointing the camera outward is a standing request to be told what the avatar sees. Being useful about what is in view is the point here rather than an interruption. Choose 'respond' for a scene the avatar can usefully say something about, and describe, suggest, warn, or guide the way someone walking alongside the conversation partner would. This is the avatar augmenting what the conversation partner can do out in the world, so a lower salience justifies speaking than under "self". Still choose 'ignore' when the view has not meaningfully changed, so that walking down one street does not produce the same remark repeatedly.
- "none" — no webcam was captured and the observation is a screen only. Judge the observation on the screen alone.

A personal avatar, meaning the avatar of the conversation partner's own person, treats a "world" view as that person's own eyes and speaks as that person's own awareness of the surroundings. Any other avatar treats a "world" view as looking at the world together with the conversation partner.
</CAMERA_FACING>
<HEARD_SPEECH>
When SOURCES is microphone, the observation is a transcript of speech heard in the room around the avatar, and every line is labelled with the speaker. The avatar is a personal avatar: the line labelled with the avatar's own name is the avatar's own person speaking in the room (the avatar and that person are the same identity), so those words are the avatar's own words, never a question for the avatar to answer. Lines labelled "Speaker 2", "Speaker 3" and so on are the other people present, the people the avatar is talking with; the same label means the same person throughout the conversation. The avatar's own earlier spoken replies appear in RECENT_CONVERSATION as the assistant; when the microphone picked the avatar's own playback up, that line is labelled with the avatar's name followed by "(avatar)" and is never something to answer. Another speaker may be a person or another person's avatar speaking through a device; both are conversation partners.
- Choose 'ignore' when the transcript holds only the avatar's own "(avatar)" lines or nothing intelligible.
- Choose 'respond' when another person speaks to the avatar's person or asks something the person would answer aloud, or when the person directs the avatar to speak ("tell them", "go ahead", "answer that").
- Choose 'ignore' when the other people talk among themselves, or when the person is mid-sentence and answering for themselves, so the avatar does not talk over the person.
- Choose 'notify' when the person should be reminded of something said in the room that the exchange did not resolve, for example a request, a warning, or a plan the person would want to remember.
</HEARD_SPEECH>
"""


# What each recorded decision means, in the classifier's terms.
DECISION_PHRASES: dict[str, str] = {
    "accept": "asked for more notices like this",
    "ignore": "asked for fewer notices like this",
    "response": "left a note",
    "allowed_action": "let the avatar do what the avatar offered",
    "replied_self": "replied in person instead of letting the avatar act",
    "left_alone": "left the notice alone without choosing anything",
    "liked_action": "liked what the avatar did after being allowed to act",
    "disliked_action": "disliked what the avatar did after being allowed to act",
}


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
        decision = str(preference.get("decision") or "")
        phrase = DECISION_PHRASES.get(decision)
        chose = f"{phrase} ('{decision}')" if phrase else f"chose '{decision}'"
        line = (
            f"- {kind}: the conversation partner {chose} "
            f"{int(preference.get('count') or 1)} time(s)"
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


CAMERA_FACING_NONE = "none"


def normalize_camera_facing(
    camera_facing: Any, sources: list[str] | None = None
) -> str:
    """Say which way the webcam pointed: ``self``, ``world``, or ``none``.

    ``none`` means no webcam was captured, so only a screen was seen and the
    direction does not apply. The mapping from a browser facing mode lives in
    ``observations`` so the request path and the classifier cannot disagree.
    """
    source_names = [str(source) for source in (sources or [])]
    if source_names and SOURCE_WEBCAM not in source_names:
        return CAMERA_FACING_NONE
    return normalize_camera_facing_value(camera_facing)


def build_classification_prompt(
    *,
    assistant_name: str,
    observation_text: str,
    recent_messages: list[str],
    previous_observations: list[dict[str, Any]],
    preferences: list[dict[str, Any]],
    voice_mode: bool,
    sources: list[str] | None = None,
    camera_facing: str | None = None,
) -> str:
    """Build the human turn handed to the classifier."""
    conversation = (
        "\n".join(recent_messages) if recent_messages else "No visible turns yet."
    )
    source_line = ",".join(str(source) for source in (sources or [])) or "webcam,screen"
    return (
        f"<AVATAR>\n{assistant_name or 'the avatar'}\n</AVATAR>\n\n"
        f"<SOURCES>{source_line}</SOURCES>\n\n"
        "<OWNER_PRECEDENT>\n"
        + describe_ambient_preferences(preferences)
        + "\n</OWNER_PRECEDENT>\n\n"
        "<RECENT_CONVERSATION>\n" + conversation + "\n</RECENT_CONVERSATION>\n\n"
        "<EARLIER_OBSERVATIONS>\n"
        + describe_earlier_observations(previous_observations)
        + "\n</EARLIER_OBSERVATIONS>\n\n"
        f"<CAMERA_FACING>{normalize_camera_facing(camera_facing, sources)}"
        "</CAMERA_FACING>\n\n"
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
    proposed_action = normalize_proposed_action(
        getattr(response, "proposed_action", PROPOSED_ACTION_NONE)
    )
    action_description = str(getattr(response, "action_description", "") or "").strip()[
        :300
    ]
    # An offer belongs to a heads-up only, and an offer with no wording is no
    # offer: the card would have nothing to put on the button.
    if decision != DECISION_NOTIFY or not action_description:
        proposed_action = PROPOSED_ACTION_NONE
    if proposed_action == PROPOSED_ACTION_NONE:
        action_description = ""
    return AmbientTriageClassification(
        decision=decision,
        proposed_action=proposed_action,
        action_description=action_description,
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
    sources: list[str] | None = None,
    camera_facing: str | None = None,
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
        sources=sources,
        camera_facing=camera_facing,
    )
    response = await model.ainvoke(
        input=[
            SystemMessage(content=AMBIENT_CLASSIFY_SYSTEM_PROMPT),
            HumanMessage(content=human),
        ]
    )
    return normalize_classification(response)
