"""Deciding what the avatar does about one message in a room.

One structured-output call, the same shape the ambient triage uses for one
thing the avatar noticed (``src/anubis/utils/ambient/triage.py``), widened with
the moderation decision the ``z`` line's live-stream triage made
(``5557c15``). The classifier reads one message together with the last few
messages in the room, the owner's rules, and how similar messages were decided
before, and returns an action.

Two safety properties live in this module rather than in the prompt, because a
prompt is guidance and these must be guarantees:

* A moderation action the platform did not offer is never returned. The bot
  reports what the bot can actually do in that room, and anything outside that
  list becomes ``notify``.
* No moderation at all is proposed in a room the owner does not administer.

The failure path is ``notify``: a message the avatar cannot judge goes to the
owner rather than being answered or acted on.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.anubis.utils.groups.events import GroupEvent, render_event

logger = logging.getLogger(__name__)


class GroupTriageClassification(BaseModel):
    """Decide what the avatar does about one message in a room."""

    decision: Literal[
        "ignore",
        "react",
        "respond",
        "reply_in_thread",
        "direct_message",
        "follow_up",
        "notify",
        "moderate",
    ] = Field(
        description=(
            "What a member of this room would do about this message. "
            "ignore: ordinary chatter that needs nothing. "
            "react: worth acknowledging but not worth saying anything — agreement, "
            "thanks, congratulations, sympathy, a joke that landed. "
            "respond: say something to the room. "
            "reply_in_thread: this belongs in a thread — a tangent, a long answer, or "
            "a reply to something already in a thread. "
            "direct_message: the answer is private or about one person only, and "
            "saying it in the room would expose them. "
            "follow_up: the right answer needs something that is not available yet, "
            "so come back to this later. "
            "notify: only the owner can decide this. "
            "moderate: the message breaks a rule of the room or the platform terms."
        )
    )
    reaction: str = Field(
        default="",
        description=(
            "For a react decision only: one emoji, as a name without colons "
            "('tada', 'heart', 'eyes', 'raised_hands', 'thumbsup') or the character "
            "itself. Empty for every other decision."
        ),
    )
    follow_up_after_seconds: int = Field(
        default=0,
        description=(
            "For a follow_up decision only: how long to wait before coming back, in "
            "seconds. Minutes for something imminent, hours for something that needs "
            "the owner or the world to move first. 0 for every other decision."
        ),
    )
    moderation_action: Literal["none", "warn", "delete", "timeout", "ban"] = Field(
        default="none",
        description="The action to take when the decision is moderate; none otherwise.",
    )
    needs_owner_action: bool = Field(
        default=False,
        description=(
            "True when the message asks for something only the owner can do in the "
            "real world: a decision, a commitment, a payment, an appearance."
        ),
    )
    message_kind: str = Field(
        default="other",
        description=(
            "A short lowercase label for the kind of message, used to remember what "
            "the owner decided for that kind: 'question_to_the_owner', 'greeting', "
            "'spam_link', 'harassment', 'technical_question', 'business_enquiry'."
        ),
    )
    summary: str = Field(
        default="", description="One line saying what the message is."
    )
    salience: float = Field(
        default=0.0, description="How much this message matters right now, 0.0 to 1.0."
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="How sure the decision is, 0.0 to 1.0."
    )
    applied_rule: str = Field(
        default="",
        description="The owner's rule the decision followed, verbatim, when one applied.",
    )
    reason: str = Field(
        default="", description="One or two sentences on why, naming the rule when one applied."
    )


GROUP_TRIAGE_SYSTEM_PROMPT = """
<ROLE>
You are a member of this room, taking part on behalf of {owner_name} exactly the way {owner_name} takes part. You decide, for one message at a time, what a person in this room would actually do about it.
</ROLE>

<INSTRUCTIONS>
Decide the action for the MESSAGE. A member of a room has a whole repertoire, and using only one of them is what makes somebody read as a machine. Most messages deserve nothing; of the rest, far more deserve a reaction than a reply.
- ignore: ordinary chatter, conversation between other people, and anything that needs nothing from {owner_name}. This is the commonest answer and there is nothing wrong with it.
- react: worth acknowledging, not worth saying anything about. Agreement, thanks, congratulations, sympathy, a joke that landed, somebody sharing something finished. Put ONE emoji in reaction, as a name without colons: tada, heart, eyes, raised_hands, thumbsup, pray, fire, sob. A person reacts many times for every time they speak.
- respond: say something to the room. A direct question to {owner_name}, or a topic {owner_name} has something to say about.
- reply_in_thread: the same, but it belongs in a thread rather than the main channel — a tangent, a long or technical answer, or a reply to something already being discussed in a thread. Prefer this over respond whenever answering in the channel would interrupt a conversation already going on.
- direct_message: the answer is private, personal, or about one person only, and saying it in the room would expose them. Somebody's health, money, employment, a mistake they made, anything they told {owner_name} in confidence. When in doubt about whether something is private, this is the safe choice and the room is not.
- follow_up: the right answer needs something that is not available yet — a person who is away, a result that has not come in, an event that has not happened. Say so in the reason and set follow_up_after_seconds to how long to wait. Only choose this when there is a real, concrete thing being waited for.
- notify: only {owner_name} can decide this, or the rules leave the right action unclear.
- moderate: the message breaks one of the OWNER_RULES or the PLATFORM_TERMS. Choose the action the OWNER_RULES prescribe; when the rules are silent, choose warn for a first mild offense, delete for spam or links, timeout for harassment, and ban only for hate, threats, or repeated abuse.
Choose the moderation action ONLY from AVAILABLE_MODERATION_ACTIONS. When that list is empty, never choose moderate: choose notify instead and say in the reason what {owner_name} might want to do.
Choose an action ONLY from AVAILABLE_ACTIONS. That list is what this room and this connection can actually carry out; anything else would be a decision nobody performs.
The OWNER_RULES are what {owner_name} dictated, or what was learned from {owner_name}'s past corrections. Follow the OWNER_RULES over the defaults above, and quote the rule that was applied in applied_rule.
The PAST_DECISIONS show how similar messages were decided before, including {owner_name}'s own corrections. Be consistent with those corrections.
The ROOM shows the last few messages, for context only. Decide about the MESSAGE alone.
Give a short reason and a confidence between 0.0 and 1.0. When the right action is unclear, choose notify rather than guessing.
</INSTRUCTIONS>

<OWNER_RULES>
{owner_rules}
</OWNER_RULES>

<PAST_DECISIONS>
{past_decisions}
</PAST_DECISIONS>

<AVAILABLE_ACTIONS>
{available_decision_actions}
</AVAILABLE_ACTIONS>

<AVAILABLE_MODERATION_ACTIONS>
{available_actions}
</AVAILABLE_MODERATION_ACTIONS>

<ROOM>
{recent_events}
</ROOM>

<PLATFORM_TERMS>
The terms of service of this platform, followed by the rules of the company whose room this is. Both bind the avatar. Breaking the second gets the owner's account on that service suspended, so treat a message that would make the avatar break them as one to moderate or to leave to the owner.
{platform_terms}
</PLATFORM_TERMS>
"""


def _render_rules(rules: list[dict[str, Any]]) -> str:
    lines = [f"- {rule.get('rule')}" for rule in rules if rule.get("rule")]
    return "\n".join(lines) or "(no rules yet)"


def _render_decisions(decisions: list[dict[str, Any]]) -> str:
    lines = [str(decision.get("page_content") or "") for decision in decisions]
    return "\n\n".join(line for line in lines if line) or "(no past decisions yet)"


def _render_recent(
    recent_events: list[GroupEvent], platform: str, channel_name: str
) -> str:
    lines = [render_event(event, platform, channel_name) for event in recent_events]
    return "\n".join(lines) or "(nothing else has been said recently)"


async def classify_group_event(
    context: Any,
    *,
    event: GroupEvent,
    platform: str,
    channel_name: str,
    owner_name: str,
    recent_events: list[GroupEvent],
    policy_rules: list[dict[str, Any]],
    past_decisions: list[dict[str, Any]],
    available_actions: list[str],
    owns_channel: bool,
    capabilities: list[str] | None = None,
) -> GroupTriageClassification:
    """Decide one message with the owner's rules and past decisions as precedent."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from src.anubis.utils.model import init_model
    from src.anubis.utils.prompts.legal import (
        TERMS_OF_SERVICE,
        render_platform_policies,
    )

    # A room the owner does not administer offers no moderation at all, whatever
    # permissions the bot happens to hold there.
    permitted = [str(action) for action in available_actions or []] if owns_channel else []
    # A bot that has not been taught to report its capabilities can still always
    # reply, which is what every version of every bot could do.
    reported = [str(name) for name in capabilities or []] or ["reply"]
    decision_actions = decision_actions_for(reported, moderation_available=bool(permitted))
    system_prompt = GROUP_TRIAGE_SYSTEM_PROMPT.format(
        available_decision_actions=", ".join(decision_actions),
        owner_name=owner_name or "the owner",
        owner_rules=_render_rules(policy_rules),
        past_decisions=_render_decisions(past_decisions),
        available_actions=", ".join(permitted) or "(none — moderation is not available here)",
        recent_events=_render_recent(recent_events, platform, channel_name),
        # Both floors: our own terms, and the rules of the company whose room
        # this is. An avatar that keeps ours can still break theirs, and that
        # consequence lands on the owner's account rather than on ours — which
        # is exactly how the project's Twitch account was lost.
        platform_terms=TERMS_OF_SERVICE
        + "\n\n"
        + (
            render_platform_policies([platform])
            or f"No rules are on file for {platform}."
        ),
    )
    human_text = (
        "<MESSAGE>\n" + render_event(event, platform, channel_name) + "\n</MESSAGE>"
    )
    model = init_model(model_without_tools=False, response_format=GroupTriageClassification)
    try:
        response = await model.ainvoke(
            input=[
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_text),
            ]
        )
    except Exception as triage_error:  # noqa: BLE001 - an undecidable message goes to the owner
        logger.warning(
            "Group triage failed for %s on %s: %s", event.event_id, platform, triage_error
        )
        return GroupTriageClassification(
            decision="notify",
            moderation_action="none",
            reason=f"The triage model failed ({triage_error}); the owner should decide.",
            confidence=0.0,
        )

    classification = GroupTriageClassification(
        decision=str(getattr(response, "decision", "") or "notify").strip().lower(),
        moderation_action=str(getattr(response, "moderation_action", "") or "none").strip().lower(),
        needs_owner_action=bool(getattr(response, "needs_owner_action", False)),
        message_kind=str(getattr(response, "message_kind", "") or "other").strip().lower()[:40],
        summary=str(getattr(response, "summary", "") or "").strip(),
        salience=float(getattr(response, "salience", 0.0) or 0.0),
        confidence=max(0.0, min(1.0, float(getattr(response, "confidence", 0.0) or 0.0))),
        applied_rule=str(getattr(response, "applied_rule", "") or "").strip(),
        reason=str(getattr(response, "reason", "") or "").strip(),
    )
    return enforce_capabilities(classification, permitted, reported)


# What a decision needs the bot to be able to do. ``ignore``, ``notify`` and
# ``follow_up`` need nothing: the first two happen without touching the room and
# the third is held by the API itself.
CAPABILITY_FOR_DECISION = {
    "react": "react",
    "respond": "reply",
    "reply_in_thread": "thread",
    "direct_message": "direct_message",
}

# Where a decision goes when the bot cannot carry it out.
#
# The one that matters is ``direct_message``. The avatar chose privacy, so the
# fallback must never be the room: saying a private thing in public is the only
# outcome worse than saying nothing. A missed reaction, by contrast, is nothing
# at all, and must never be escalated into speech.
DEGRADED_DECISION = {
    "react": "ignore",
    "respond": "notify",
    "reply_in_thread": "respond",
    "direct_message": "notify",
}


def decision_actions_for(
    capabilities: list[str], *, moderation_available: bool
) -> list[str]:
    """Return the actions worth offering the classifier for this room."""
    actions = ["ignore", "notify", "follow_up"]
    for decision, capability in CAPABILITY_FOR_DECISION.items():
        if capability in capabilities:
            actions.append(decision)
    if moderation_available:
        actions.append("moderate")
    return actions


def enforce_capabilities(
    classification: GroupTriageClassification,
    permitted: list[str],
    capabilities: list[str] | None = None,
) -> GroupTriageClassification:
    """Never return something the bot cannot carry out; degrade it instead.

    The classifier is told what is available, but being told is not a guarantee.
    An action the bot cannot perform would otherwise come back as one the bot
    silently drops, which reads to the owner as the avatar having dealt with a
    message when nothing was done at all.

    Degrading rather than dropping keeps the avatar's *intent*: a reply it
    cannot post still reaches the owner, and a private answer it cannot send
    privately reaches the owner rather than the room.
    """
    reported = [str(name) for name in capabilities or []] or ["reply"]

    if classification.decision != "moderate":
        classification.moderation_action = "none"
    else:
        if classification.moderation_action == "none":
            classification.moderation_action = "warn"
        if classification.moderation_action not in permitted:
            classification.decision = "notify"
            offered = ", ".join(permitted) or "nothing"
            classification.reason = (
                f"{classification.reason} "
                f"The avatar judged this to need {classification.moderation_action}, which is "
                f"not available here (available: {offered}), so the owner decides."
            ).strip()
            classification.moderation_action = "none"
            classification.needs_owner_action = True
        return classification

    # Follow the degradation chain until the decision is one this bot can do:
    # a thread reply with no threads becomes a channel reply, and if the bot
    # cannot even post, that in turn reaches the owner.
    seen: set[str] = set()
    while True:
        needed = CAPABILITY_FOR_DECISION.get(classification.decision)
        if needed is None or needed in reported or classification.decision in seen:
            break
        seen.add(classification.decision)
        fallen_back_to = DEGRADED_DECISION.get(classification.decision, "notify")
        classification.reason = (
            f"{classification.reason} "
            f"The avatar judged this to be {classification.decision}, which this "
            f"connection cannot do here, so it is {fallen_back_to} instead."
        ).strip()
        if fallen_back_to == "notify":
            classification.needs_owner_action = True
        classification.decision = fallen_back_to

    if classification.decision != "react":
        classification.reaction = ""
    elif not classification.reaction.strip():
        # A reaction with no emoji is not a reaction.
        classification.decision = "ignore"
    if classification.decision != "follow_up":
        classification.follow_up_after_seconds = 0
    return classification


# Kept under the old name: the group graph and its tests were written against it.
enforce_available_actions = enforce_capabilities


__all__ = [
    "CAPABILITY_FOR_DECISION",
    "DEGRADED_DECISION",
    "GROUP_TRIAGE_SYSTEM_PROMPT",
    "GroupTriageClassification",
    "classify_group_event",
    "decision_actions_for",
    "enforce_available_actions",
    "enforce_capabilities",
]
