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
    category: Literal["communication_style", "address", "topic", "format", "other"] = Field(
        description=(
            "communication_style for tone, humor, directness, or length; address for "
            "names and titles; topic for subjects the user wants or avoids; format for "
            "lists, paragraphs, or emoji; other for anything else."
        )
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


def _tool_message(content: str, runtime: ToolRuntime) -> Command:
    return Command(
        update={"messages": [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)]}
    )


@tool("learn_user_preference", return_direct=False, args_schema=UserPreferenceAndContext)
async def learn_user_preference(
    preference: str,
    preference_context: str,
    category: str = "other",
    runtime: Annotated[ToolRuntime, InjectedToolArg] = None,
) -> Command:
    """<INSTRUCTIONS>
    Learn a PREFERENCE the user dictates about how the user wants to be treated in conversation: how the user wants to be addressed, what the user wants to talk about or avoid, what reply format the user wants, and how the user wants you to communicate (tone, humor, directness, length).
    Call this tool ONCE PER DISTINCT PREFERENCE. A single message may hold several preferences; make one call for each.
    Preserve the preference as the user meant the preference. Set preference_context to a concise summary of the whole message.
    </INSTRUCTIONS>

    <EXAMPLE>
    User: "Call me Sam, keep things short, and skip the pep talks."
    Three calls:
      1. preference: "The user prefers to be called Sam." category: address
      2. preference: "The user wants short replies." category: communication_style
      3. preference: "The user does not want pep talks or motivational framing." category: communication_style
    </EXAMPLE>

    <RESTRICTIONS>
    NEVER call this tool for facts about the user's identity (name, history, relationships); use learn_information_about_the_user for those.
    NEVER call this tool for facts about your own identity.
    NEVER call this tool twice with the same preference.
    </RESTRICTIONS>
    """
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
    if document is None:
        return _tool_message(f"Preference previously learned: {preference}", runtime)
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


LEARNING_TOOLS = [learn_user_preference, record_what_feels_real]

__all__ = [
    "LEARNING_TOOLS",
    "learn_user_preference",
    "record_what_feels_real",
]
