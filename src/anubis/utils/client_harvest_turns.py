"""Client harvest turns: hidden requests the browser sends through ``/message``.

The browser asks the avatar for things that are not conversation — a list of
follow-up suggestions for the chips above the composer, or a first-person
profile description — by sending a turn that begins with a
``[neural-nexus:...]`` marker. Those turns, and the replies they receive, are
machine traffic: a person never typed them and the avatar never "said" a JSON
list.

Earlier browser builds sent the follow-up harvest as an ordinary turn, so a
thread could carry the harvest and its JSON reply forever. A model that sees
that history answers "hey" with ``["Hi! ...", ...]`` — the browser hides the
list as leaked JSON and the person sees no reply at all. These helpers keep
such turns out of what the model reads and out of the transcript the browser
lists, without touching the stored thread.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

CLIENT_HARVEST_MARKER_PREFIX = "[neural-nexus:"
"""Prefix of every browser-sent hidden request (suggestions, description)."""

CLIENT_HARVEST_MESSAGE_KIND = "client_harvest"
"""``additional_kwargs["kind"]`` stamped on a harvest turn by ``/message``."""

SUGGESTION_LIST_MIN_ITEMS = 2
SUGGESTION_LIST_MAX_ITEMS = 6
SUGGESTION_LIST_MAX_ITEM_CHARACTERS = 160


def _text_of(message: Any) -> str:
    """Return the text of a message object or serialized dict, or ``""``."""
    content: Any
    if isinstance(message, BaseMessage):
        content = message.content
    elif isinstance(message, dict):
        content = message.get("content")
    else:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "\n".join(parts)
    return ""


def _is_human(message: Any) -> bool:
    if isinstance(message, HumanMessage):
        return True
    return isinstance(message, dict) and message.get("type") == "human"


def _is_ai(message: Any) -> bool:
    if isinstance(message, AIMessage):
        return True
    return isinstance(message, dict) and message.get("type") == "ai"


def is_client_harvest_marker_text(text: str | None) -> bool:
    """Whether a turn's text is a browser harvest request."""
    return str(text or "").lstrip().startswith(CLIENT_HARVEST_MARKER_PREFIX)


def is_client_harvest_turn(message: Any) -> bool:
    """Whether a message is a browser harvest request (a marked human turn)."""
    return _is_human(message) and is_client_harvest_marker_text(_text_of(message))


def is_suggestion_list_reply(message: Any) -> bool:
    """Whether an assistant message is a JSON list of short follow-up prompts.

    This mirrors the browser's own detection so a reply the browser would hide
    as leaked JSON is also kept away from the model.
    """
    if not _is_ai(message):
        return False
    text = _text_of(message).strip()
    if not (text.startswith("[") and text.endswith("]")):
        return False
    try:
        parsed = json.loads(text)
    except ValueError:
        return False
    if not isinstance(parsed, list):
        return False
    if not SUGGESTION_LIST_MIN_ITEMS <= len(parsed) <= SUGGESTION_LIST_MAX_ITEMS:
        return False
    return all(
        isinstance(item, str)
        and item.strip()
        and len(item.strip()) <= SUGGESTION_LIST_MAX_ITEM_CHARACTERS
        for item in parsed
    )


def without_stale_client_harvest_turns(messages: list[Any]) -> list[Any]:
    """Drop past harvest turns, their replies, and stray suggestion lists.

    The final message is kept even when the final message is a harvest turn:
    that is the live request the model is about to answer. Everything earlier
    that is a harvest request, the assistant reply directly after one, or an
    assistant reply that is itself a suggestion list, is left out.
    """
    if not messages:
        return list(messages)
    last_index = len(messages) - 1
    kept: list[Any] = []
    previous_was_harvest = False
    for index, message in enumerate(messages):
        if index == last_index and is_client_harvest_turn(message):
            kept.append(message)
            break
        if is_client_harvest_turn(message):
            previous_was_harvest = True
            continue
        if _is_ai(message) and (previous_was_harvest or is_suggestion_list_reply(message)):
            previous_was_harvest = False
            continue
        previous_was_harvest = False
        kept.append(message)
    return kept
