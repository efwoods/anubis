"""The standard set of conversation starters, generated once per avatar.

A new conversation opens with three short messages a visitor might type to
begin talking with the avatar. Those chips must be about this avatar — a
recruiter is asked about joining, a restaurant about ordering, a pastor about
prayer — and they must not depend on anything the visitor has said, because
nothing has been said yet.

Before this module the browser produced the chips by sending a hidden
``/message`` turn every time a new conversation opened and the browser held no
cached list. That is a full avatar inference per browser per avatar, paid to
produce three sentences that do not change between conversations. Here the
starters are produced once by a structured-output call on the classification
model, grounded in the avatar's name, description, and the identity facts the
avatar holds, and stored on the assistant record under
``metadata["conversation_starters"]``. Every browser reads the same list from
the avatar record and caches the list locally; nothing is generated on the
reply path.

The set is regenerated when deep research finishes, because that is the moment
the avatar learns who the avatar is: a set written from a bare name and a one
line description is a placeholder, and the set written from verified facts is
the one visitors should see. ``src/api/webapp.py`` owns the storage side (the
LangGraph assistants client and the research job); this module owns the prompt,
the schema, the normalization, and the record shape.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CONVERSATION_STARTERS_METADATA_KEY = "conversation_starters"
"""Assistant metadata key holding the stored record (see ``build_conversation_starters_record``)."""

CONVERSATION_STARTER_COUNT = 3
CONVERSATION_STARTER_MAX_CHARACTERS = 160
CONVERSATION_STARTER_IDENTITY_FACT_LIMIT = 60
"""How many identity facts the prompt is given; the whole identity is not needed for three lines."""

SOURCE_IDENTITY_ONLY = "identity"
"""The record was written from the avatar's name and description alone (no research had run)."""
SOURCE_DEEP_RESEARCH = "deep_research"
"""The record was written after deep research finished, from verified facts."""
SOURCE_OWNER_REQUEST = "owner_request"
"""The owner asked for a fresh set through the API."""


class ConversationStarters(BaseModel):
    """Exactly three opening messages a visitor would type to this avatar."""

    starters: list[str] = Field(
        description=(
            "Exactly three short messages, each one a natural first message a "
            "visitor would type to begin a conversation with this avatar. Each "
            "message is at most one sentence and at most 120 characters. Each "
            "message is grounded in who the avatar is and what the avatar does. "
            "No two messages ask the same thing."
        ),
        min_length=CONVERSATION_STARTER_COUNT,
        max_length=CONVERSATION_STARTER_COUNT,
    )


CONVERSATION_STARTERS_SYSTEM_PROMPT = """
<ROLE>
You write the three suggested opening messages a visitor sees before starting a conversation with an AI avatar of a specific person or organization.
</ROLE>

<INSTRUCTIONS>
Read the AVATAR block: the avatar's name, the avatar's description, the organization links the avatar represents, and the identity facts the avatar holds about the avatar's own life and work.
Write exactly three short messages a visitor would type to start talking with this avatar.
Each message must be grounded in who this avatar is and what this avatar does. Use the specific facts in the AVATAR block: the avatar's role, the avatar's organization, the avatar's work, the avatar's story.
A recruiter's visitors ask about joining or the requirements to join. A restaurant's visitors ask to see the menu or to place an order. A pastor's visitors ask for prayer or scripture. A loved one's visitors check in and ask for a story. A founder's visitors ask about that founder's foundation or company and how to support that work.
When the AVATAR block lists an organization link, exactly one of the three messages asks the avatar to share that organization's public website link.
Write each message as the visitor speaking to the avatar in the second person ("you", "your"), in the language of the avatar's description.
Each message is one sentence, at most 120 characters, with no numbering, no quotation marks, and no trailing explanation.
Do not write a message that would fit any avatar. "Hey, how are you?", "Tell me about yourself", "What should we talk about?", and "Tell me more" are wrong.
Do not mention the words "avatar", "AI", "assistant", "chatbot", or "model".
</INSTRUCTIONS>

<OUTPUT>
Return the three messages in the `starters` field. Return nothing else.
</OUTPUT>
"""

_GENERIC_STARTER_TEXTS = {
    "hey, how are you?",
    "hey how are you",
    "how are you?",
    "tell me about yourself",
    "tell me about yourself.",
    "what should we talk about?",
    "tell me more",
    "tell me more.",
    "hello",
    "hi",
    "hi!",
    "hello!",
}


def conversation_starters_enabled(context: Any) -> bool:
    """Whether the standard set is generated and stored at all (env ``CONVERSATION_STARTERS_ENABLED``)."""
    raw = getattr(context, "conversation_starters_enabled", "true")
    return str(raw if raw is not None else "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def normalize_conversation_starters(values: Any) -> list[str]:
    """Clean a candidate list into at most three distinct, specific, short starters.

    Strips surrounding quotation marks and list numbering the model sometimes
    adds despite the schema, drops empty and over-long entries, drops the
    greetings that fit any avatar, and keeps the first occurrence of each
    distinct message.
    """
    if not isinstance(values, (list, tuple)):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        text = value.strip()
        # "1. Message" / "- Message" / "\"Message\""
        while text and text[0] in "-*•":
            text = text[1:].strip()
        if len(text) > 2 and text[0].isdigit():
            head, _, tail = text.partition(".")
            if head.strip().isdigit() and tail.strip():
                text = tail.strip()
        text = text.strip("\"'“”‘’ ").strip()
        if not text or len(text) > CONVERSATION_STARTER_MAX_CHARACTERS:
            continue
        key = text.lower()
        if key in _GENERIC_STARTER_TEXTS or key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) == CONVERSATION_STARTER_COUNT:
            break
    return cleaned


def build_conversation_starters_record(
    starters: list[str],
    *,
    source: str,
    identity_fact_count: int = 0,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build the value stored under ``metadata["conversation_starters"]``.

    ``generated_at`` is what the browser keys the local cache on: a newer stamp
    on the avatar record replaces whatever list the browser cached.
    """
    stamp = generated_at or datetime.now(UTC)
    return {
        "starters": list(starters),
        "generated_at": stamp.isoformat(),
        "source": source,
        "identity_fact_count": int(identity_fact_count),
    }


def conversation_starters_record_of(assistant: Any) -> dict[str, Any] | None:
    """Read the stored record off an assistant, from metadata or a lifted top-level field.

    Returns ``None`` when the record is missing or malformed. The starters in
    the returned record are already normalized.
    """
    if not isinstance(assistant, dict):
        return None
    candidates: list[Any] = []
    metadata = assistant.get("metadata")
    if isinstance(metadata, dict):
        candidates.append(metadata.get(CONVERSATION_STARTERS_METADATA_KEY))
    candidates.append(assistant.get(CONVERSATION_STARTERS_METADATA_KEY))
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        starters = normalize_conversation_starters(candidate.get("starters"))
        if len(starters) < 2:
            continue
        record = dict(candidate)
        record["starters"] = starters
        return record
    return None


def conversation_starters_of(assistant: Any) -> list[str]:
    """Return the stored starters for an assistant, or an empty list."""
    record = conversation_starters_record_of(assistant)
    return list(record["starters"]) if record else []


async def invoke_structured(
    response_format: type[BaseModel], system_prompt: str, human_text: str
) -> Any:
    """One structured-output call on the classification model. Isolated so tests can replace this."""
    from src.anubis.utils.model import init_model

    model = init_model(response_format=response_format)
    response = await model.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_text)]
    )
    if isinstance(response, tuple):
        response = response[0]
    if isinstance(response, response_format):
        return response
    return response_format.model_validate(response)


def _identity_fact_texts(facts: list[dict[str, Any]] | list[str]) -> list[str]:
    texts: list[str] = []
    for fact in facts or []:
        if isinstance(fact, str):
            text = fact.strip()
        elif isinstance(fact, dict):
            text = str(fact.get("fact") or fact.get("text") or "").strip()
        else:
            continue
        if text:
            texts.append(text)
    return texts


def build_conversation_starters_human_text(
    *,
    name: str | None,
    description: str | None,
    organization_links: list[str] | None,
    identity_facts: list[str],
) -> str:
    """Render the AVATAR block the prompt reads."""
    lines = ["<AVATAR>"]
    lines.append(f"Name: {(name or '').strip() or 'Unknown'}")
    if description and description.strip():
        lines.append(f"Description: {description.strip()}")
    if organization_links:
        lines.append("Organization links:")
        lines.extend(f"- {href}" for href in organization_links)
    if identity_facts:
        lines.append("Identity facts the avatar holds:")
        lines.extend(f"- {fact}" for fact in identity_facts)
    else:
        lines.append(
            "Identity facts the avatar holds: none yet; ground the messages in "
            "the name and description."
        )
    lines.append("</AVATAR>")
    return "\n".join(lines)


async def generate_conversation_starters(
    store: Any,
    *,
    creator_id: str,
    assistant_id: str,
    name: str | None,
    description: str | None,
) -> tuple[list[str], int]:
    """Write a fresh set of starters for one avatar.

    Returns ``(starters, identity_fact_count)``. The starters list is empty when
    the model returned nothing usable; the caller decides whether to keep the
    previous record in that case. Never raises for a store failure — the set can
    still be written from the name and description.
    """
    from src.anubis.utils.organization_links import organization_links_from_text
    from src.anubis.utils.research.deep_research import load_existing_identity_facts

    identity_facts: list[str] = []
    if store is not None and creator_id:
        try:
            held = await load_existing_identity_facts(
                store,
                creator_id,
                assistant_id,
                limit=CONVERSATION_STARTER_IDENTITY_FACT_LIMIT,
            )
            identity_facts = _identity_fact_texts(held)
        except Exception:  # noqa: BLE001 - starters can still come from the description
            logger.debug(
                "Could not read identity facts for the conversation starters of %s",
                assistant_id,
                exc_info=True,
            )
    organization_links = organization_links_from_text(description)
    human_text = build_conversation_starters_human_text(
        name=name,
        description=description,
        organization_links=organization_links,
        identity_facts=identity_facts,
    )
    response = await invoke_structured(
        ConversationStarters, CONVERSATION_STARTERS_SYSTEM_PROMPT, human_text
    )
    raw_starters = getattr(response, "starters", None)
    if raw_starters is None and isinstance(response, dict):
        raw_starters = response.get("starters")
    return normalize_conversation_starters(raw_starters), len(identity_facts)
