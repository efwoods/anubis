"""Triage reasoning: classify, draft in the owner's voice, and score confidence.

Three structured-output calls, each a Pydantic model so the graph never parses
free text:

``TriageClassification``
    ``ignore`` / ``notify`` / ``respond``, whether the message names a
    real-world action the owner must take themselves, a short ``message_kind``
    (newsletter, receipt, meeting request, personal note, …) used as the
    preference key, and the reason.

``DraftReply``
    The reply, written under the avatar's own consciousness prompt so it
    carries the owner's voice, plus a one-line summary of what the reply does.

``PreferenceAlignment``
    Whether the draft aligns with how the owner has decided similar messages
    before — the first of the two signals that make up the confidence score.

The second signal is a prior from the owner's decision counts: accepts raise
it, edits and ignores lower it, and a sender the owner has never ruled on caps
the score below the auto-send threshold, so a stranger always reaches the owner.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from src.anubis.utils.inbox.repository import (
    DECISION_IGNORE,
    DECISION_NOTIFY,
    DECISION_RESPOND,
)

logger = logging.getLogger(__name__)


class TriageClassification(BaseModel):
    """Decide what the avatar should do with one incoming message on the owner's behalf."""

    decision: str = Field(
        description=(
            "One of 'ignore' (nothing to do: promotions, automated notices, newsletters), "
            "'notify' (the owner must see this or act in the real world: a deadline, a bill, "
            "a request only the owner can fulfil), or 'respond' (a reply from the owner is "
            "expected and the avatar can write it)."
        )
    )
    needs_owner_action: bool = Field(
        description=(
            "True when the message asks for something only the owner can do in the real "
            "world (attend, pay, sign, decide), regardless of whether a reply is also due."
        )
    )
    message_kind: str = Field(
        description=(
            "A short lowercase label for the kind of message, used to remember the owner's "
            "preferences: 'newsletter', 'receipt', 'meeting_request', 'personal_note', "
            "'support_request', 'security_alert', 'invoice', 'introduction', or similar."
        )
    )
    reason: str = Field(description="One sentence explaining the decision.")


class DraftReply(BaseModel):
    """A reply written as the owner, in the owner's voice."""

    subject: str = Field(description="The reply's subject line (usually 'Re: ...').")
    body: str = Field(
        description="The full reply text, ready to send, in the owner's voice."
    )
    summary: str = Field(description="One sentence saying what the reply does.")


class PreferenceAlignment(BaseModel):
    """Judge whether a drafted reply matches how the owner handles this kind of message."""

    aligned: bool = Field(
        description="True when the draft does what the owner's recorded decisions suggest."
    )
    alignment_score: float = Field(
        description=(
            "0.0 to 1.0: how confidently the draft matches the owner's recorded preferences "
            "for this sender and message kind. 0.5 when there is no relevant precedent."
        )
    )
    reason: str = Field(description="One sentence explaining the score.")


class OwnerEditLesson(BaseModel):
    """What the owner's rewrite of a drafted reply teaches about the owner's writing."""

    edit_summary: str = Field(
        description=(
            "One sentence naming what the owner changed, written so the same change can be "
            "recognized next time: 'the owner removed the sign-off', 'the owner shortened "
            "the reply to two sentences', 'the owner answered the question directly instead "
            "of thanking the sender first'."
        )
    )
    writing_lesson: str = Field(
        description=(
            "One reusable instruction, phrased as a standing preference about how the owner "
            "writes, that would make the next draft closer to what the owner sent: 'Write "
            "replies without a closing sign-off', 'Keep replies under three sentences'. "
            "Empty when the rewrite teaches nothing reusable, such as correcting a single "
            "fact that applies only to this message."
        )
    )
    changed_substantially: bool = Field(
        description=(
            "True when the owner rewrote the substance or the voice of the reply, rather "
            "than fixing a typo or adjusting a word or two."
        )
    )


def _describe_preferences(preferences: list[dict[str, Any]]) -> str:
    if not preferences:
        return "The owner has recorded no decisions for this sender or this kind of message."
    lines = []
    for preference in preferences:
        who = (
            preference.get("sender") or preference.get("sender_domain") or "any sender"
        )
        kind = preference.get("message_kind") or "any kind"
        lines.append(
            f"- {who} / {kind}: the owner chose '{preference.get('decision')}' "
            f"{int(preference.get('count') or 1)} time(s)"
            + (
                f"; note: {preference['edit_summary']}"
                if preference.get("edit_summary")
                else ""
            )
        )
    return "\n".join(lines)


def _describe_message(message: dict[str, Any]) -> str:
    return (
        f"From: {message.get('sender') or ''}\n"
        f"To: {', '.join(message.get('recipients') or []) if isinstance(message.get('recipients'), list) else message.get('recipients') or ''}\n"
        f"Subject: {message.get('subject') or ''}\n"
        f"Date: {message.get('received_at') or message.get('sent_at') or ''}\n\n"
        f"{str(message.get('body_text') or '')[:6000]}"
    )


CLASSIFY_SYSTEM_PROMPT = """<TASK>
The assistant triages one incoming message on behalf of the message's recipient, who is the owner of a personal avatar. Decide whether the avatar should ignore the message, notify the owner, or respond to the message as the owner.
</TASK>
<RULES>
- Choose 'ignore' for promotions, newsletters, automated notifications that need nothing, and social-media digests.
- Choose 'notify' when the owner must see the message or act in the real world: a bill, a deadline, a security alert, a request only the owner can fulfil, or anything ambiguous and consequential.
- Choose 'respond' when the message expects a reply from the owner and the avatar can write that reply from what the message says: a scheduling question, a simple request for information the owner would readily give, a friendly note.
- The owner's recorded decisions for this sender and this kind of message are precedent. Follow the precedent unless the message plainly differs from the earlier ones.
- Name the message kind with a short lowercase label so the same kind is recognized next time.
</RULES>
"""

DRAFT_SYSTEM_SUFFIX = """
<REPLY_TASK>
Write the reply to the incoming message below as the owner would write it: the owner's greeting, sentence length, register, and sign-off. Answer what the message actually asks. Do not invent commitments, dates, or facts the owner has not stated; when a detail is unknown, say the owner will confirm it. Never mention that an assistant or an avatar wrote the reply.
</REPLY_TASK>
"""

ALIGNMENT_SYSTEM_PROMPT = """<TASK>
The assistant judges whether a drafted reply matches how the owner of a personal avatar has decided similar incoming messages before. The owner's recorded decisions are precedent: 'respond' decisions with accepted drafts mean the owner is comfortable with the avatar replying to this sender and this kind of message; 'edit' notes show what the owner changes; 'ignore' or 'notify' decisions mean the owner does not want an automatic reply here.
</TASK>
<RULES>
- Score 0.9 or above only when the precedent clearly supports an automatic reply of this kind and the draft does what the owner's edits asked for.
- Score 0.5 when there is no relevant precedent.
- Score below 0.3 when the precedent says the owner handles this sender or kind personally.
</RULES>
"""

OWNER_EDIT_SYSTEM_PROMPT = """<TASK>
The assistant compares a reply drafted for the owner of a personal avatar with the version the owner actually sent after editing the draft. Name what the owner changed, and state the one standing preference about the owner's writing that would have produced the owner's version in the first place.
</TASK>
<RULES>
- Describe what the owner changed, not what the message was about.
- The writing lesson must be reusable on an unrelated message to an unrelated person. A correction that applies only to this one message — a date, a name, a single fact — teaches nothing reusable, so leave the writing lesson empty in that case.
- Report a substantial change only when the owner rewrote the substance or the voice of the reply. Fixing a typo, a word, or punctuation is not substantial.
- Write both fields in plain sentences about the owner. Do not address the owner, and do not mention drafts, avatars, or assistants.
</RULES>
"""


async def classify_message(
    context: Any, *, message: dict[str, Any], preferences: list[dict[str, Any]]
) -> TriageClassification:
    """Classify one incoming message with the owner's preferences as precedent."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from src.anubis.utils.model import init_model

    model = init_model(model_without_tools=False, response_format=TriageClassification)
    human = (
        "<OWNER_PRECEDENT>\n"
        + _describe_preferences(preferences)
        + "\n</OWNER_PRECEDENT>\n\n"
        "<MESSAGE>\n" + _describe_message(message) + "\n</MESSAGE>"
    )
    response = await model.ainvoke(
        input=[
            SystemMessage(content=CLASSIFY_SYSTEM_PROMPT),
            HumanMessage(content=human),
        ]
    )
    decision = str(getattr(response, "decision", "") or "").strip().lower()
    if decision not in (DECISION_IGNORE, DECISION_NOTIFY, DECISION_RESPOND):
        decision = DECISION_NOTIFY
    return TriageClassification(
        decision=decision,
        needs_owner_action=bool(getattr(response, "needs_owner_action", False)),
        message_kind=str(getattr(response, "message_kind", "") or "other")
        .strip()
        .lower()[:40],
        reason=str(getattr(response, "reason", "") or "").strip(),
    )


async def draft_reply(
    context: Any, *, message: dict[str, Any], voice_system_prompt: str
) -> DraftReply:
    """Write the reply under the avatar's consciousness prompt."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from src.anubis.utils.model import init_model

    model = init_model(model_without_tools=False, response_format=DraftReply)
    response = await model.ainvoke(
        input=[
            SystemMessage(content=(voice_system_prompt or "") + DRAFT_SYSTEM_SUFFIX),
            HumanMessage(
                content="<INCOMING_MESSAGE>\n"
                + _describe_message(message)
                + "\n</INCOMING_MESSAGE>"
            ),
        ]
    )
    subject = str(getattr(response, "subject", "") or "").strip()
    if not subject:
        original = str(message.get("subject") or "").strip()
        subject = (
            original
            if original.lower().startswith("re:")
            else f"Re: {original}".strip()
        )
    return DraftReply(
        subject=subject,
        body=str(getattr(response, "body", "") or "").strip(),
        summary=str(getattr(response, "summary", "") or "").strip(),
    )


async def judge_alignment(
    context: Any,
    *,
    message: dict[str, Any],
    draft: DraftReply,
    preferences: list[dict[str, Any]],
) -> PreferenceAlignment:
    """Judge the draft against the owner's precedent (the LLM half of the score)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from src.anubis.utils.model import init_model

    model = init_model(model_without_tools=False, response_format=PreferenceAlignment)
    human = (
        "<OWNER_PRECEDENT>\n"
        + _describe_preferences(preferences)
        + "\n</OWNER_PRECEDENT>\n\n"
        "<MESSAGE>\n" + _describe_message(message) + "\n</MESSAGE>\n\n"
        f"<DRAFT_REPLY>\nSubject: {draft.subject}\n\n{draft.body}\n</DRAFT_REPLY>"
    )
    response = await model.ainvoke(
        input=[
            SystemMessage(content=ALIGNMENT_SYSTEM_PROMPT),
            HumanMessage(content=human),
        ]
    )
    score = float(getattr(response, "alignment_score", 0.5) or 0.5)
    return PreferenceAlignment(
        aligned=bool(getattr(response, "aligned", False)),
        alignment_score=max(0.0, min(1.0, score)),
        reason=str(getattr(response, "reason", "") or "").strip(),
    )


async def summarize_owner_edit(
    context: Any, *, draft_body: str, final_body: str, message: dict[str, Any]
) -> OwnerEditLesson:
    """Read one owner rewrite for the lesson that generalizes past this message."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from src.anubis.utils.model import init_model

    model = init_model(model_without_tools=False, response_format=OwnerEditLesson)
    human = (
        "<INCOMING_MESSAGE>\n"
        + _describe_message(message)
        + "\n</INCOMING_MESSAGE>\n\n"
        "<DRAFTED_FOR_THE_OWNER>\n"
        + str(draft_body or "")[:6000]
        + "\n</DRAFTED_FOR_THE_OWNER>\n\n"
        "<WHAT_THE_OWNER_SENT>\n"
        + str(final_body or "")[:6000]
        + "\n</WHAT_THE_OWNER_SENT>"
    )
    response = await model.ainvoke(
        input=[
            SystemMessage(content=OWNER_EDIT_SYSTEM_PROMPT),
            HumanMessage(content=human),
        ]
    )
    return OwnerEditLesson(
        edit_summary=str(getattr(response, "edit_summary", "") or "").strip(),
        writing_lesson=str(getattr(response, "writing_lesson", "") or "").strip(),
        changed_substantially=bool(getattr(response, "changed_substantially", False)),
    )


CALENDAR_EVENT_SYSTEM_PROMPT = """<TASK>
The assistant reads one incoming message and states the appointment the message asks the recipient to attend, so the appointment can be booked on the recipient's calendar.
</TASK>
<RULES>
- Take the day and the time from the message. Write both as ISO 8601: '2026-09-14T15:00:00' for an appointment at a stated time, and '2026-09-14' for a whole day.
- Resolve a relative day such as 'Thursday' or 'tomorrow' against the date the message was sent, which is given as the message date.
- Leave the start empty when the message names no day at all. Never invent a day or a time that the message does not state.
- Leave the end empty when the message states no finish time.
- The summary is the appointment as the recipient would want to read the appointment on a calendar, naming the other person: 'Call with Dana Rios', 'Dentist'.
</RULES>
"""


class CalendarEventRequest(BaseModel):
    """The appointment an incoming message asks the owner to attend."""

    summary: str = Field(
        description="The appointment's title, as the owner would read the title on a calendar."
    )
    start: str = Field(
        description=(
            "ISO 8601 start: '2026-09-14T15:00:00' for a stated time, '2026-09-14' for a "
            "whole day. Empty when the message names no day."
        )
    )
    end: str = Field(
        description="ISO 8601 finish, or empty when the message states no finish time."
    )
    description: str = Field(
        description="One or two sentences of context from the message, or empty."
    )
    location: str = Field(
        description="Where the appointment happens, including a meeting link, or empty."
    )


async def extract_calendar_event(
    context: Any, *, message: dict[str, Any]
) -> CalendarEventRequest:
    """Read the appointment out of one message so the owner does not retype the appointment."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from src.anubis.utils.model import init_model

    model = init_model(model_without_tools=False, response_format=CalendarEventRequest)
    response = await model.ainvoke(
        input=[
            SystemMessage(content=CALENDAR_EVENT_SYSTEM_PROMPT),
            HumanMessage(
                content="<INCOMING_MESSAGE>\n"
                + _describe_message(message)
                + "\n</INCOMING_MESSAGE>"
            ),
        ]
    )
    return CalendarEventRequest(
        summary=str(getattr(response, "summary", "") or "").strip(),
        start=str(getattr(response, "start", "") or "").strip(),
        end=str(getattr(response, "end", "") or "").strip(),
        description=str(getattr(response, "description", "") or "").strip(),
        location=str(getattr(response, "location", "") or "").strip(),
    )


SPECIFICITY_WEIGHT_SENDER = 1.0
SPECIFICITY_WEIGHT_DOMAIN = 0.5
SPECIFICITY_WEIGHT_MESSAGE_KIND = 0.25


def preference_specificity_weight(
    preference: dict[str, Any],
    *,
    sender: str | None = None,
    sender_domain: str | None = None,
) -> float:
    """Weight one recorded decision by how closely the decision speaks to this message.

    A decision the owner made about this exact correspondent is the strongest
    evidence; one made about everyone in the same domain or the same chat room
    is weaker; one that matched only on the kind of message is weakest — both
    the coarse row that lets "the owner always ignores recruiter mail" reach a
    recruiter who has never written before, and a row about a different person
    entirely, which ``recall_preferences`` also returns for a matching kind.

    Without this weighting, a decision about somebody else would count for as
    much as the owner's explicit history with the person now writing.
    """
    recorded_sender = str(preference.get("sender") or "").strip().lower()
    recorded_domain = str(preference.get("sender_domain") or "").strip().lower()
    this_sender = str(sender or "").strip().lower()
    this_domain = str(sender_domain or "").strip().lower()
    if this_sender or this_domain:
        if recorded_sender and recorded_sender == this_sender:
            return SPECIFICITY_WEIGHT_SENDER
        if recorded_domain and recorded_domain == this_domain:
            return SPECIFICITY_WEIGHT_DOMAIN
        return SPECIFICITY_WEIGHT_MESSAGE_KIND
    # Without a sender to compare against, the row's own shape is all there is
    # to go on.
    if recorded_sender:
        return SPECIFICITY_WEIGHT_SENDER
    if recorded_domain:
        return SPECIFICITY_WEIGHT_DOMAIN
    return SPECIFICITY_WEIGHT_MESSAGE_KIND


def preference_prior(
    preferences: list[dict[str, Any]],
    *,
    auto_send_threshold: float,
    sender: str | None = None,
    sender_domain: str | None = None,
) -> tuple[float, str]:
    """Compute the count-based half of the confidence score.

    Accepted or auto-sent replies for this sender/kind raise the prior toward
    1.0; edits, ignores, and notifies pull it down. With no history at all the
    prior is capped just below the auto-send threshold, so a sender the owner
    has never ruled on always reaches the owner.

    Each recorded decision counts in proportion to how closely the decision
    speaks to this message (``preference_specificity_weight``), so a decision
    about this exact correspondent outweighs one about the whole domain or
    room, which in turn outweighs one that matched only on the kind of message.
    """
    if not preferences:
        return min(
            0.6, auto_send_threshold - 0.05
        ), "no owner decisions yet for this sender or kind"
    supportive = 0.0
    opposing = 0.0
    for preference in preferences:
        count = int(preference.get("count") or 1) * preference_specificity_weight(
            preference, sender=sender, sender_domain=sender_domain
        )
        decision = str(preference.get("decision") or "")
        if decision in ("accept", "auto_sent", DECISION_RESPOND):
            supportive += count
        elif decision in ("edit",):
            supportive += count * 0.5
            opposing += count * 0.5
        else:
            opposing += count
    total = supportive + opposing
    if total <= 0:
        return 0.5, "no weighted decisions"
    prior = supportive / total
    # One or two decisions are not yet a pattern.
    prior = prior * min(1.0, 0.6 + 0.2 * total)
    return max(
        0.0, min(1.0, prior)
    ), f"{supportive:.0f} supportive vs {opposing:.0f} opposing decision(s)"


def combine_confidence(alignment_score: float, prior: float) -> float:
    """``alignment × prior``: both signals must agree for an automatic send."""
    return max(0.0, min(1.0, float(alignment_score) * float(prior)))
