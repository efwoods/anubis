"""User sentiment: the immediate message reading and the conversation summary.

Two signals feed the system prompt:

* **Immediate sentiment** — the Go Emotions classification of the user's
  latest message (the same classifier the avatar's replies are scored with,
  ``src/anubis/utils/emotion_classifier.py``). Rendered into the existing
  ``=== USER EMOTIONS ===`` section.
* **Conversation sentiment summary** — a short structured summary of how the
  whole current conversation feels, refreshed each turn and kept as a scalar
  store record ``current:{thread_id}``. When the background sweep finalizes a
  conversation, the summary is written as an embedded Document
  ``history:{thread_id}`` so past conversations are retrievable by similarity
  for the ``=== HISTORY OF SENTIMENT SUMMARIES ... ===`` section.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel, Field

from src.anubis.utils.learning.namespaces import sentiment_namespace

logger = logging.getLogger(__name__)

CURRENT_SENTIMENT_KEY_PREFIX = "current:"
SENTIMENT_HISTORY_KEY_PREFIX = "history:"

# How many trailing conversation messages the summarizer reads. Long threads are
# already summarized in the running record, which is passed back in as context.
_TRANSCRIPT_TAIL_MESSAGES = 40
_TRANSCRIPT_MESSAGE_CHARACTER_LIMIT = 1200


class ConversationSentimentSummary(BaseModel):
    """A short reading of how the user feels across the current conversation."""

    sentiment_summary: str = Field(
        description=(
            "Two to four sentences describing how the user has been feeling over the "
            "conversation so far, how that feeling has moved, and what the user seems "
            "to want emotionally from the avatar. Written about the user in the third "
            "person."
        )
    )
    dominant_emotions: list[str] = Field(
        description="The two to four emotions most present in the user's messages.",
        default_factory=list,
    )
    overall_polarity: Literal["positive", "neutral", "negative", "mixed"] = Field(
        description="The overall polarity of the user's sentiment across the conversation."
    )
    engagement_signal: Literal["rising", "steady", "falling", "unknown"] = Field(
        description=(
            "Whether the user's investment in the conversation is rising, steady, or "
            "falling, judged from message length, questions asked, and warmth."
        )
    )


CONVERSATION_SENTIMENT_SUMMARY_SYSTEM_PROMPT = """
<ROLE>
You are an expert at reading the emotional state of a person from a conversation transcript.
</ROLE>

<INSTRUCTIONS>
Read the TRANSCRIPT between a USER and an AVATAR. Summarize the sentiment of the USER only.
Describe how the USER has been feeling across the whole conversation, how that feeling has changed, and what the USER seems to want emotionally from the AVATAR.
Name the dominant emotions of the USER, the overall polarity of the USER's sentiment, and whether the USER's engagement is rising, steady, or falling.
When a PREVIOUS_SUMMARY is provided, update that summary with the newest messages rather than starting over.
Never describe the AVATAR's feelings. Never invent events that are not in the transcript.
</INSTRUCTIONS>
"""


def message_text(content: Any) -> str:
    """Flatten a message ``content`` (string or content blocks) into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts)
    return "" if content is None else str(content)


def render_transcript(messages: list[Any], tail: int = _TRANSCRIPT_TAIL_MESSAGES) -> str:
    """``USER: ... / AVATAR: ...`` lines for the trailing human and avatar messages."""
    lines: list[str] = []
    for message in list(messages or [])[-tail:]:
        speaker = _speaker_label(message)
        if speaker is None:
            continue
        text = message_text(getattr(message, "content", message)).strip()
        if not text:
            continue
        if len(text) > _TRANSCRIPT_MESSAGE_CHARACTER_LIMIT:
            text = text[:_TRANSCRIPT_MESSAGE_CHARACTER_LIMIT] + " ..."
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def _speaker_label(message: Any) -> str | None:
    if isinstance(message, HumanMessage):
        return "USER"
    if isinstance(message, AIMessage):
        # Tool-planning turns carry no user-visible reply.
        if getattr(message, "tool_calls", None):
            return None
        return "AVATAR"
    if isinstance(message, dict):
        message_type = message.get("type") or message.get("role")
        if message_type in ("human", "user"):
            return "USER"
        if message_type in ("ai", "assistant") and not message.get("tool_calls"):
            return "AVATAR"
    return None


def user_messages_text(messages: list[Any], tail: int = _TRANSCRIPT_TAIL_MESSAGES) -> list[str]:
    """The user's own messages only, oldest first, for preference inference."""
    texts: list[str] = []
    for message in list(messages or [])[-tail:]:
        if _speaker_label(message) != "USER":
            continue
        text = message_text(getattr(message, "content", message)).strip()
        if text:
            texts.append(text)
    return texts


async def classify_user_message_sentiment(text: str) -> dict[str, Any] | None:
    """Go Emotions reading of one user message, off the event loop."""
    from src.anubis.utils.emotion_classifier import classify_go_emotions

    if not text or not text.strip():
        return None
    return await asyncio.to_thread(classify_go_emotions, text)


def render_immediate_sentiment(sentiment: dict[str, Any] | None) -> str:
    """Prose for ``=== USER EMOTIONS ===`` from a Go Emotions reading."""
    if not sentiment:
        return ""
    emotion = sentiment.get("emotion")
    base_emotion = sentiment.get("base_emotion")
    score = sentiment.get("score")
    if not emotion:
        return ""
    confidence = ""
    if isinstance(score, (int, float)):
        confidence = f" (confidence {float(score):.2f})"
    if base_emotion and base_emotion != emotion:
        return (
            f"The user's most recent message reads as {emotion}, "
            f"a form of {base_emotion}{confidence}."
        )
    return f"The user's most recent message reads as {emotion}{confidence}."


async def invoke_structured(response_format: type[BaseModel], system_prompt: str, human_text: str):
    """Run one structured-output model call. Isolated so tests can replace this."""
    from src.anubis.utils.model import init_model

    model = init_model(response_format=response_format)
    return await model.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_text)]
    )


async def summarize_conversation_sentiment(
    messages: list[Any], previous_summary: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Structured sentiment summary of the user across ``messages`` (None on failure)."""
    transcript = render_transcript(messages)
    if not transcript.strip():
        return None
    human_text = ""
    if previous_summary:
        human_text += (
            "<PREVIOUS_SUMMARY>\n"
            + json.dumps(previous_summary, ensure_ascii=False)
            + "\n</PREVIOUS_SUMMARY>\n\n"
        )
    human_text += "<TRANSCRIPT>\n" + transcript + "\n</TRANSCRIPT>"
    try:
        response = await invoke_structured(
            ConversationSentimentSummary,
            CONVERSATION_SENTIMENT_SUMMARY_SYSTEM_PROMPT,
            human_text,
        )
    except Exception as summarization_error:  # noqa: BLE001 - best-effort signal
        logger.warning("Conversation sentiment summary failed: %s", summarization_error)
        return None
    if isinstance(response, tuple):
        response = response[0]
    if isinstance(response, BaseModel):
        summary = response.model_dump()
    elif isinstance(response, dict):
        summary = dict(response)
    else:
        return None
    summary["summarized_at"] = datetime.now(tz=UTC).isoformat()
    summary["message_count"] = len(messages or [])
    return summary


def render_conversation_sentiment(summary: dict[str, Any] | None) -> str:
    """Prose for the current-conversation sentiment prompt section."""
    if not summary or not summary.get("sentiment_summary"):
        return ""
    parts = [str(summary["sentiment_summary"]).strip()]
    dominant = [str(e) for e in (summary.get("dominant_emotions") or []) if e]
    if dominant:
        parts.append("Dominant emotions: " + ", ".join(dominant) + ".")
    polarity = summary.get("overall_polarity")
    if polarity:
        parts.append(f"Overall polarity: {polarity}.")
    engagement_signal = summary.get("engagement_signal")
    if engagement_signal and engagement_signal != "unknown":
        parts.append(f"Engagement is {engagement_signal}.")
    return " ".join(parts)


async def load_current_conversation_sentiment(
    store: Any, user_id: str, assistant_id: str, thread_id: str
) -> dict[str, Any] | None:
    item = await store.aget(
        sentiment_namespace(user_id, assistant_id),
        key=f"{CURRENT_SENTIMENT_KEY_PREFIX}{thread_id}",
    )
    value = getattr(item, "value", None) or {}
    summary = value.get("value") if isinstance(value, dict) else None
    return dict(summary) if isinstance(summary, dict) else None


async def save_current_conversation_sentiment(
    store: Any, user_id: str, assistant_id: str, thread_id: str, summary: dict[str, Any]
) -> None:
    await store.aput(
        sentiment_namespace(user_id, assistant_id),
        key=f"{CURRENT_SENTIMENT_KEY_PREFIX}{thread_id}",
        value={"value": summary},
    )


async def update_current_conversation_sentiment(
    store: Any, user_id: str, assistant_id: str, thread_id: str, messages: list[Any]
) -> dict[str, Any] | None:
    """Refresh and persist the running summary for ``thread_id``; returns the summary."""
    previous_summary = await load_current_conversation_sentiment(
        store, user_id, assistant_id, thread_id
    )
    summary = await summarize_conversation_sentiment(messages, previous_summary)
    if summary is None:
        return previous_summary
    await save_current_conversation_sentiment(
        store, user_id, assistant_id, thread_id, summary
    )
    return summary


def build_sentiment_history_document(
    summary: dict[str, Any], *, user_id: str, assistant_id: str, thread_id: str
) -> Document:
    rendered = render_conversation_sentiment(summary)
    metadata = {
        "user_id": user_id,
        "assistant_id": assistant_id,
        "thread_id": thread_id,
        "document_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"sentiment-history:{thread_id}")),
        "fact": rendered,
        "kind": "conversation_sentiment_history",
        "summarized_at": summary.get("summarized_at"),
        "overall_polarity": summary.get("overall_polarity"),
        "engagement_signal": summary.get("engagement_signal"),
        "dominant_emotions": list(summary.get("dominant_emotions") or []),
    }
    return Document(page_content=rendered, metadata=metadata)


async def finalize_conversation_sentiment(
    store: Any, user_id: str, assistant_id: str, thread_id: str, messages: list[Any]
) -> dict[str, Any] | None:
    """Write the conversation's final summary into the sentiment history."""
    summary = await update_current_conversation_sentiment(
        store, user_id, assistant_id, thread_id, messages
    )
    if not summary:
        return None
    document = build_sentiment_history_document(
        summary, user_id=user_id, assistant_id=assistant_id, thread_id=thread_id
    )
    await store.aput(
        sentiment_namespace(user_id, assistant_id),
        key=f"{SENTIMENT_HISTORY_KEY_PREFIX}{thread_id}",
        value={"document": document.to_json()},
    )
    return summary
