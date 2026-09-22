"""Tell the conversation when the avatar has just stored a fact.

The learn tools write a ToolMessage the model reads ("Learned: …") and
otherwise leave no mark on the reply the person sees. This module is the
visible half: a ``fact_learned`` stream frame so the browser can paint a
badge mid-turn, and a ``learned_facts`` list on the reply's
``response_metadata`` so the badge is still there after a reload.

Only facts announced this turn (``announce_fact_learned``, which runs only
after a successful new store) reach the badge. A ToolMessage that says the
fact was previously learned never announces, so the badge never re-shows an
already-known fact — even when older learn ToolMessages still sit in the
deep agent's message list.
"""

from __future__ import annotations

import contextvars
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage

logger = logging.getLogger(__name__)

FACT_LEARNED_EVENT = "fact_learned"

_LEARNED_IDENTITY = re.compile(r"^Learned:\s*(.+)$", re.DOTALL)
_LEARNED_PREFERENCE = re.compile(r"^Learned preference:\s*(.+)$", re.DOTALL)
_LEARNED_ABOUT_THE_USER = re.compile(r"^Learned about the user:\s*(.+)$", re.DOTALL)

_KIND_IDENTITY = "identity"
_KIND_PREFERENCE = "preference"
_KIND_MEMORY = "memory"
_KIND_USER = "user"
_KNOWN_KINDS = {_KIND_IDENTITY, _KIND_PREFERENCE, _KIND_MEMORY, _KIND_USER}

# ToolMessages the learn tools build themselves often omit ``name``. When
# LangGraph does set the name, it is the more reliable kind than the
# "Learned:" prefix, which identity and memory share. The user-fact prefix
# ``Learned about the user:`` is the reliable signal when ``name`` is absent.
_TOOL_NAME_KINDS = {
    "learn_user_preference": _KIND_PREFERENCE,
    "create_episodic_memory": _KIND_MEMORY,
    "update_self_identity_mem_from_user_txt": _KIND_IDENTITY,
    "learn_information_about_the_user": _KIND_USER,
}

_MAX_FACT_CHARS = 280

_turn_learned_facts: contextvars.ContextVar[list[dict[str, str]] | None] = (
    contextvars.ContextVar("turn_learned_facts", default=None)
)


class TurnLearnedFactsCollector:
    """Per-turn list of facts just stored, carried on a context variable.

    ``begin_turn`` installs a fresh list before the deep agent runs;
    ``announce_fact_learned`` appends only after a successful new store;
    ``attach_learned_facts_metadata`` reads the list onto the reply. Child
    tasks inherit the same list object, so tool calls see the turn buffer.
    """

    @staticmethod
    def begin_turn() -> None:
        """Start collecting facts learned during the current turn."""
        _turn_learned_facts.set([])

    @staticmethod
    def add(entry: dict[str, str]) -> None:
        """Record one fact stored during this turn (no-op outside a turn)."""
        facts = _turn_learned_facts.get()
        if facts is None:
            return
        key = entry["fact"].casefold()
        if any(existing["fact"].casefold() == key for existing in facts):
            return
        facts.append(dict(entry))

    @staticmethod
    def collect() -> list[dict[str, str]]:
        """Return the facts stored so far in this turn (a copy)."""
        facts = _turn_learned_facts.get()
        return [dict(entry) for entry in (facts or [])]

    @staticmethod
    def end_turn() -> None:
        """Stop collecting; later ``add`` calls are ignored."""
        _turn_learned_facts.set(None)

    @staticmethod
    def active() -> bool:
        """Whether a turn is currently collecting."""
        return _turn_learned_facts.get() is not None


def _trim_fact(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= _MAX_FACT_CHARS:
        return cleaned
    return cleaned[: _MAX_FACT_CHARS - 1].rstrip() + "…"


def _resolved_kind(kind: str | None, tool_name: str | None) -> str:
    if kind in _KNOWN_KINDS:
        return kind
    named = _TOOL_NAME_KINDS.get(str(tool_name or ""))
    if named:
        return named
    return _KIND_IDENTITY


def parse_learned_fact_from_tool_content(
    content: Any, *, kind: str | None = None, tool_name: str | None = None
) -> dict[str, str] | None:
    """Read one successful learn ToolMessage. Failures and duplicates return None."""
    text = content if isinstance(content, str) else str(content or "")
    stripped = text.strip()
    if not stripped or stripped.startswith("Not learned:"):
        return None
    # Duplicate / already-known results must never become a badge.
    if "previously learned" in stripped.casefold():
        return None
    preference = _LEARNED_PREFERENCE.match(stripped)
    if preference:
        fact = _trim_fact(preference.group(1))
        if not fact:
            return None
        return {
            "fact": fact,
            "kind": _resolved_kind(kind or _KIND_PREFERENCE, tool_name),
            "source": "conversation",
        }
    # Check before ``Learned:`` — ToolMessages built by hand often omit
    # ``name``, so the prefix is the reliable signal for a user fact.
    about_the_user = _LEARNED_ABOUT_THE_USER.match(stripped)
    if about_the_user:
        fact = _trim_fact(about_the_user.group(1))
        if not fact:
            return None
        return {
            "fact": fact,
            "kind": _resolved_kind(kind or _KIND_USER, tool_name),
            "source": "conversation",
        }
    identity = _LEARNED_IDENTITY.match(stripped)
    if identity:
        fact = _trim_fact(identity.group(1))
        if not fact:
            return None
        return {
            "fact": fact,
            "kind": _resolved_kind(kind, tool_name),
            "source": "conversation",
        }
    return None


def _messages_from_current_turn(messages: list[Any] | None) -> list[Any]:
    """Tool results after the latest human (typed or ambient) turn."""
    last_human_index: int | None = None
    for index, message in enumerate(messages or []):
        message_type = getattr(message, "type", None)
        if isinstance(message, HumanMessage) or message_type in {"human", "user"}:
            last_human_index = index
    if last_human_index is None:
        return list(messages or [])
    return list(messages or [])[last_human_index + 1 :]


def collect_learned_facts_from_messages(
    messages: list[Any] | None,
) -> list[dict[str, str]]:
    """Unique successful learn ToolMessages after the latest human, oldest first."""
    collected: list[dict[str, str]] = []
    seen: set[str] = set()
    for message in _messages_from_current_turn(messages):
        if not isinstance(message, ToolMessage):
            continue
        parsed = parse_learned_fact_from_tool_content(
            getattr(message, "content", ""),
            tool_name=getattr(message, "name", None),
        )
        if parsed is None:
            continue
        key = parsed["fact"].casefold()
        if key in seen:
            continue
        seen.add(key)
        collected.append(parsed)
    return collected


def announce_fact_learned(
    fact: str,
    *,
    kind: str = _KIND_IDENTITY,
    source: str = "conversation",
) -> dict[str, str] | None:
    """Emit a ``fact_learned`` frame. Returns the payload, or None when empty."""
    trimmed = _trim_fact(fact)
    if not trimmed:
        return None
    payload = {
        "type": FACT_LEARNED_EVENT,
        "fact": trimmed,
        "kind": kind if kind in _KNOWN_KINDS else _KIND_IDENTITY,
        "source": source or "conversation",
    }
    TurnLearnedFactsCollector.add(
        {
            "fact": payload["fact"],
            "kind": payload["kind"],
            "source": payload["source"],
        }
    )
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()(payload)
    except Exception:  # noqa: BLE001 — outside a run there is no stream
        logger.debug("fact_learned frame not streamed", exc_info=True)
    return payload


def attach_learned_facts_metadata(
    avatar_response: Any, messages: list[Any] | None
) -> None:
    """Copy this turn's newly stored facts onto the reply the person sees.

    When a turn collector is active, only facts ``announce_fact_learned``
    recorded (successful new stores) are attached — never older learn
    ToolMessages that may still appear in ``messages``. Outside a turn
    (unit tests), fall back to parsing ``messages``.
    """
    if TurnLearnedFactsCollector.active():
        facts = TurnLearnedFactsCollector.collect()
    else:
        facts = collect_learned_facts_from_messages(messages)
    if not facts:
        return
    metadata = dict(getattr(avatar_response, "response_metadata", None) or {})
    metadata["learned_facts"] = facts
    avatar_response.response_metadata = metadata
