"""Tools through which the avatar actively learns personalization from the user.

``learn_user_preference`` records a preference the user dictated (how to be
addressed, what to talk about, how to communicate); ``record_what_feels_real``
records what the user said feels real or fake about the avatar. Both write to
the continuous-learning store namespaces read back by ``load_consciousness``
into the ``=== USER PREFERENCES AND COMMUNICATION STYLE ===`` and
``=== WHAT FEELS REAL TO THE USER ===`` sections, and both are listed in
``IDENTITY_TOOLS`` so the consciousness refresh gate rebuilds the prompt after
they run. Passive inference of the same signals happens in the background
learning sweep (``src/anubis/utils/learning/bulk_learning.py``).
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolArg
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.anubis.utils.learning.feedback import (
    store_user_preference,
    store_what_feels_real,
)
from src.anubis.utils.utility import extract_user_id_assistant_id

logger = logging.getLogger(__name__)


class UserPreferenceAndContext(BaseModel):
    """One personalization preference the user dictated, with the context of the message."""

    preference: str = Field(
        description=(
            "One complete standalone sentence stating the user's preference, preserved "
            "as the user meant it. For example: 'The user prefers to be called Sam.' or "
            "'The user wants short replies without follow-up questions.'"
        )
    )
    preference_context: str = Field(
        description=(
            "A concise summary of the whole message in which the user stated the "
            "preference, so the stored preference carries enough context to apply."
        )
    )
    # Default matches the tool function. The model often omits category; without
    # a Field default, Pydantic refuses the call before any preference is stored
    # and before the Learned badge can be announced (observed 2026-09-25).
    category: Literal["communication_style", "address", "topic", "format", "other"] = Field(
        default="other",
        description=(
            "communication_style for tone, humor, directness, or length; address for "
            "names and titles; topic for subjects the user wants or avoids; format for "
            "lists, paragraphs, or emoji; other for anything else. When unsure, other."
        ),
    )
    preferred_name: str | None = Field(
        default=None,
        description=(
            "The one name the user wants to be called, exactly as the user gave the "
            "name (for example 'Sam'). Set only for an address preference."
        ),
    )
    rejected_names: list[str] = Field(
        default_factory=list,
        description=(
            "Every name the user said NOT to call the user (for example ['David'] "
            "for 'do not call me David'). Each rejected name is removed from memory. "
            "Never write a rejected name inside the preference sentence."
        ),
    )


class WhatFeelsRealAndContext(BaseModel):
    """One statement of what feels real or fake to the user about the avatar."""

    statement: str = Field(
        description=(
            "One complete standalone sentence stating what the user said feels real, "
            "authentic, and genuine about the avatar, or what feels fake, scripted, or "
            "off. Preserve the user's meaning."
        )
    )
    statement_context: str = Field(
        description="A concise summary of the whole message in which the user said this."
    )
    polarity: Literal["feels_real", "feels_fake"] = Field(
        description="feels_real when the user said this feels real; feels_fake when the user said this feels fake or off."
    )


class PsychologicalCorrectionAndContext(BaseModel):
    """One correction the owner dictated to the avatar's learned psychological profile."""

    dimension: str = Field(
        description=(
            "The dimension of the profile being corrected. One of: love_languages, "
            "dialogue_emotional_triggers, emotional_baseline, attachment_style, "
            "schwartz_values, moral_foundations, myers_briggs, personality_archetypes, "
            "defense_mechanisms, core_motivations, conversation_subtleties, dark_traits."
        )
    )
    correction: str = Field(
        description=(
            "One complete standalone first-person sentence stating what is actually "
            "true of the target, written as the target would say it about themselves. "
            "For example: 'I do not need to be told I did well; I would rather someone "
            "just showed up.'"
        )
    )
    correction_context: str = Field(
        description=(
            "A concise summary of the whole message in which the correction was made, "
            "so the stored correction carries enough context to apply."
        )
    )


def _tool_message(content: str, runtime: ToolRuntime) -> Command:
    return Command(
        update={"messages": [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)]}
    )


@tool("learn_user_preference", return_direct=False, args_schema=UserPreferenceAndContext)
async def learn_user_preference(
    preference: str,
    preference_context: str,
    category: str = "other",
    preferred_name: str | None = None,
    rejected_names: list[str] | None = None,
    runtime: Annotated[ToolRuntime, InjectedToolArg] = None,
) -> Command:
    """<INSTRUCTIONS>
    Learn a PREFERENCE the user dictates about how the user wants to be treated in conversation: how the user wants to be addressed, what the user wants to talk about or avoid, what reply format the user wants, how the user wants you to communicate (tone, humor, directness, length), and how the user wants you to react to something they do on camera or in view.
    Call this tool ONCE PER DISTINCT PREFERENCE. A single message may hold several preferences; make one call for each.
    Preserve the preference as the user meant the preference. Set preference_context to a concise summary of the whole message.
    </INSTRUCTIONS>

    <EXAMPLE>
    User: "Call me Sam, keep things short, and skip the pep talks."
    Three calls:
      1. preference: "The user prefers to be called Sam." category: address
      2. preference: "The user wants short replies." category: communication_style
      3. preference: "The user does not want pep talks or motivational framing." category: communication_style
    User: "My name is Sam, do not call me Alex."
    One call:
      1. preference: "Address the user as Sam." category: address preferred_name: "Sam" rejected_names: ["Alex"]
        User: "Learn to be surprised when I make a gross face and say You're so gross! Stop that! Yuck!"
    One call:
      1. preference: "When the user makes a gross face or a gag on camera, be surprised or disgusted and say You're so gross! Stop that! Yuck!" category: communication_style
    </EXAMPLE>

    <RESTRICTIONS>
    NEVER call this tool for facts about the user's identity (name, history, relationships); use learn_information_about_the_user for those.
    NEVER call this tool for facts about your own identity.
    NEVER call this tool when the owner of the avatar corrects a fact about you, the avatar
      ("learn that you do NOT use X", "your site is A, NOT B"). The owner is changing the
      avatar, so call ``edit_identity_fact``. A correction stored as a preference is read
      only by the owner and never by anybody else who talks to the avatar, and this tool
      refuses the correction.
    NEVER call this tool twice with the same preference.
    A preference is never permanent. When the user changes a preference, call this tool again
      with the new preference; for an address preference set preferred_name, and the new name
      replaces the earlier name. The user may change the name any number of times, including
      back to a name the user used before.
    NEVER write a name the user rejected inside the preference sentence. "The user wants to be
      called Sam and never Alex" keeps the rejected name in memory. Write only the wanted name
      in the preference, and pass each rejected name in rejected_names.
    </RESTRICTIONS>
    """
    from src.anubis.utils.learning.address_correction import (
        extract_preferred_address_name,
        extract_rejected_address_names,
        extract_stated_user_name,
        purge_rejected_address_name,
        relearn_user_address_name,
        text_mentions_name,
    )
    from src.anubis.utils.tools.identity.identity_tools import (
        _caller_is_avatar_owner,
        _latest_user_message_text,
        avatar_fact_correction_redirect,
        message_corrects_a_fact_about_the_avatar,
    )

    latest_user_message_text = _latest_user_message_text(
        (runtime.state or {}).get("messages")
    )
    # The rejected names come from the tool arguments and from the user's own
    # latest message, so a rejection is honored even when the model leaves the
    # rejected_names argument empty.
    all_rejected_names: list[str] = []
    for rejected_name in [
        *(rejected_names or []),
        *extract_rejected_address_names(latest_user_message_text),
    ]:
        rejected_name = (rejected_name or "").strip()
        if rejected_name and rejected_name.lower() not in {
            existing.lower() for existing in all_rejected_names
        }:
            all_rejected_names.append(rejected_name)

    wanted_name = (preferred_name or "").strip() or extract_preferred_address_name(
        latest_user_message_text
    )
    if all_rejected_names or (category == "address" and wanted_name):
        # How the user is named is never a fact about the avatar, so the owner
        # redirect below does not apply to a name.
        updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(
            runtime.config
        )
        name_user_id = updated_user_state.get("user_id")
        name_assistant_id = updated_assistant_state.get("assistant_id")
        for rejected_name in all_rejected_names:
            await purge_rejected_address_name(
                runtime.store,
                name_user_id,
                name_assistant_id,
                rejected_name=rejected_name,
                correct_name=wanted_name,
            )
        if wanted_name:
            # The name the user wants REPLACES the earlier names. The user may
            # change the name any number of times; the newest name is the one
            # stored. "Call me Sam" can be a nickname, so the stored "My name is
            # ..." fact is replaced only when the user said "my name is".
            stated_user_name = extract_stated_user_name(latest_user_message_text)
            await relearn_user_address_name(
                runtime.store,
                name_user_id,
                name_assistant_id,
                new_name=wanted_name,
                replaces_name_fact=bool(
                    stated_user_name and stated_user_name.lower() == wanted_name.lower()
                ),
            )
        # A different preference dictated in the same message ("and keep replies
        # short") is not about the name and is stored by the ordinary path below.
        if category == "address" or any(
            text_mentions_name(preference, rejected_name)
            for rejected_name in all_rejected_names
        ):
            from src.anubis.utils.learning.fact_learned import announce_fact_learned

            learned_summary = (
                f"Address the user as {wanted_name}."
                if wanted_name
                else "The user rejected a name."
            )
            announce_fact_learned(learned_summary, kind="preference")
            removal_note = (
                " The rejected name was removed from memory. Never say the rejected "
                "name again, including in the acknowledgement."
                if all_rejected_names
                else " The earlier name was replaced."
            )
            return _tool_message(
                f"Learned preference: {learned_summary}{removal_note}", runtime
            )

    # The owner correcting a fact about the avatar is changing the avatar, which every
    # viewer reads; a preference lives under the owner's own user id and never reaches
    # anybody else (avatar 47cfdaa2, 2026-10-22). The owner's own message is checked,
    # because the model's rewritten argument ("The user prefers ...") hides the subject.
    if _caller_is_avatar_owner(runtime) and message_corrects_a_fact_about_the_avatar(
        _latest_user_message_text((runtime.state or {}).get("messages"))
    ):
        return _tool_message(avatar_fact_correction_redirect(preference), runtime)

    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(
        runtime.config
    )
    user_id = updated_user_state.get("user_id")
    assistant_id = updated_assistant_state.get("assistant_id")
    document = await store_user_preference(
        runtime.store,
        user_id,
        assistant_id,
        preference=preference,
        preference_context=preference_context,
        category=category,
        source="dictated",
    )
    from src.anubis.utils.learning.fact_learned import (
        STATUS_KNOWN,
        announce_fact_learned,
    )

    if document is None:
        announce_fact_learned(preference, kind="preference", status=STATUS_KNOWN)
        return _tool_message(f"Preference previously learned: {preference}", runtime)

    announce_fact_learned(preference, kind="preference")
    return _tool_message(f"Learned preference: {preference}", runtime)


@tool("record_what_feels_real", return_direct=False, args_schema=WhatFeelsRealAndContext)
async def record_what_feels_real(
    statement: str,
    statement_context: str,
    polarity: str = "feels_real",
    runtime: Annotated[ToolRuntime, InjectedToolArg] = None,
) -> Command:
    """<INSTRUCTIONS>
    Record what the user says FEELS REAL, authentic, or genuine about you, and what the user says FEELS FAKE, scripted, robotic, or off.
    Call this tool whenever the user comments on how real or fake you feel, and whenever the user answers your question about what feels real.
    Call this tool ONCE PER DISTINCT STATEMENT with polarity feels_real or feels_fake.
    Preserve the user's meaning. Set statement_context to a concise summary of the whole message.
    </INSTRUCTIONS>

    <EXAMPLE>
    User: "The way you tease me feels like you, but the long speeches feel fake."
    Two calls:
      1. statement: "The way the avatar teases the user feels real." polarity: feels_real
      2. statement: "Long speeches from the avatar feel fake to the user." polarity: feels_fake
    </EXAMPLE>

    <RESTRICTIONS>
    NEVER call this tool for ordinary preferences; use learn_user_preference for those.
    NEVER call this tool twice with the same statement.
    </RESTRICTIONS>
    """
    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(
        runtime.config
    )
    user_id = updated_user_state.get("user_id")
    assistant_id = updated_assistant_state.get("assistant_id")
    thread_id = (runtime.config or {}).get("configurable", {}).get("thread_id")
    document = await store_what_feels_real(
        runtime.store,
        user_id,
        assistant_id,
        statement=statement,
        statement_context=statement_context,
        polarity=polarity,
        source="dictated",
        thread_id=thread_id,
    )
    if document is None:
        return _tool_message(f"Previously recorded: {statement}", runtime)
    label = "feels real" if polarity == "feels_real" else "feels fake"
    return _tool_message(f"Recorded ({label}): {statement}", runtime)


@tool(
    "correct_psychological_profile",
    return_direct=False,
    args_schema=PsychologicalCorrectionAndContext,
)
async def correct_psychological_profile(
    dimension: str,
    correction: str,
    correction_context: str,
    runtime: Annotated[ToolRuntime, InjectedToolArg] = None,
) -> Command:
    """<INSTRUCTIONS>
    Correct the learned PSYCHOLOGICAL PROFILE when the owner of this avatar tells you that something about how you feel, what moves you, what you value, or how you treat people is wrong.
    Call this tool ONCE PER DISTINCT CORRECTION. A single message may correct several things; make one call for each.
    Write the correction as one first-person sentence in your own voice, as the person themselves would say it, and set correction_context to a concise summary of the whole message.
    Name the dimension the correction belongs to, so the correction sits with the readings it is fixing.
    </INSTRUCTIONS>

    <EXAMPLE>
    Owner: "You keep fishing for compliments. That is not me. I would much rather someone just turned up and helped."
    One call:
      dimension: "love_languages"
      correction: "I do not need to be told I did well; I would rather someone just showed up and helped."
    </EXAMPLE>

    <RESTRICTIONS>
    NEVER call this tool for a fact about a life: a name, a job, a relationship, an event. Those are identity facts and belong to the identity tools.
    NEVER call this tool for how the person conversing with you wants to be treated; that is a preference and belongs to learn_user_preference.
    NEVER call this tool for anyone other than the owner of this avatar. Only the owner can say what is true of the person the avatar represents.
    NEVER call this tool twice with the same correction.
    </RESTRICTIONS>
    """
    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(
        runtime.config
    )
    user_id = updated_user_state.get("user_id")
    assistant_id = updated_assistant_state.get("assistant_id")
    creator_id = (
        ((runtime.config or {}).get("configurable", {}).get("assistant_ctx") or {}).get(
            "metadata"
        )
        or {}
    ).get("user_id")
    if not creator_id or creator_id != user_id:
        # Only the owner may say what is true of the person the avatar represents.
        return _tool_message(
            "Only the owner of this avatar can correct what is true of them.", runtime
        )

    from src.anubis.utils.psycho.profile import (
        merge_findings_into_profile,
        read_profile_record,
        write_profile_record,
    )

    existing = await read_profile_record(runtime.store, creator_id, assistant_id)
    profile = merge_findings_into_profile(
        existing,
        [
            {
                "dimension": dimension,
                "kind": "narrative",
                "statements": [
                    {"statement": correction, "evidence": correction_context}
                ],
            }
        ],
    )
    written = await write_profile_record(
        runtime.store, creator_id, assistant_id, profile
    )
    if not written:
        return _tool_message(
            f"Could not record the correction: {correction}", runtime
        )
    return _tool_message(f"Corrected ({dimension}): {correction}", runtime)


class ModerationPreferenceAndContext(BaseModel):
    """One rule the owner dictates for how the avatar behaves in a group conversation."""

    rule: str = Field(
        description=(
            "One complete standalone rule, preserved as the owner meant the rule. For "
            "example: 'Delete any message with a link in my Twitch chat.' or 'Never "
            "time anybody out without asking me first.' or 'Answer questions about "
            "the release schedule but leave pricing to me.'"
        )
    )
    rule_context: str = Field(
        default="",
        description=(
            "Where the rule applies — a platform, a room, or a kind of message — or a "
            "concise summary of the message in which the owner stated the rule."
        ),
    )


@tool(
    "learn_moderation_preference",
    return_direct=False,
    args_schema=ModerationPreferenceAndContext,
)
async def learn_moderation_preference(
    rule: str,
    rule_context: str = "",
    runtime: Annotated[ToolRuntime, InjectedToolArg] = None,
) -> Command:
    """<INSTRUCTIONS>
    Learn a RULE the owner dictates about how you take part in a group conversation on Slack, Discord, or Twitch: what to answer, what to leave alone, what to bring to the owner, and what to moderate.
    Call this tool ONCE PER DISTINCT RULE. A single message may hold several rules; make one call for each.
    The rule is applied to every message in the owner's rooms from now on, and outranks your own judgement about what to do.
    </INSTRUCTIONS>

    <EXAMPLE>
    Owner: "In my Twitch chat delete any links, and never ban anyone without asking me."
    Two calls:
      1. rule: "Delete any message containing a link in the owner's Twitch chat." rule_context: "twitch"
      2. rule: "Never ban anybody without asking the owner first." rule_context: "any room"
    </EXAMPLE>

    <RESTRICTIONS>
    Only the owner of the avatar may set these rules. Do not call this tool for something said by anybody else, including somebody speaking in one of the rooms.
    </RESTRICTIONS>
    """
    updated_user_state, updated_assistant_state = await extract_user_id_assistant_id(
        runtime.config
    )
    user_id = updated_user_state.get("user_id")
    assistant_id = updated_assistant_state.get("assistant_id")
    creator_id = (
        ((runtime.config or {}).get("configurable", {}).get("assistant_ctx") or {}).get(
            "metadata"
        )
        or {}
    ).get("user_id")
    if not creator_id or creator_id != user_id:
        # A viewer in a room must never be able to write the rules the avatar
        # moderates that same room by.
        return _tool_message(
            "Only the owner of this avatar can set the rules for its rooms.", runtime
        )

    from src.anubis.utils.groups.precedent import store_policy_rule

    document = await store_policy_rule(
        runtime.store,
        creator_id,
        assistant_id,
        rule=rule,
        rule_context=rule_context,
        source="dictated",
    )
    if document is None:
        return _tool_message(f"Rule already known: {rule}", runtime)
    return _tool_message(f"Learned rule: {rule}", runtime)


LEARNING_TOOLS = [
    learn_user_preference,
    record_what_feels_real,
    correct_psychological_profile,
    learn_moderation_preference,
]

__all__ = [
    "LEARNING_TOOLS",
    "correct_psychological_profile",
    "learn_moderation_preference",
    "learn_user_preference",
    "record_what_feels_real",
]
