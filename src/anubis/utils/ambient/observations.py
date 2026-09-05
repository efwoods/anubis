"""Pure helpers for ambient observations: sources, message tagging, throttling.

An ambient observation is a ``HumanMessage`` the conversation partner never
typed. The endpoint builds the message with the images attached, exactly like a
typed turn with attachments, and tags the message through ``additional_kwargs``:

``hidden``
    ``True`` — the API drops the message from transcript listings and the
    frontend never paints a bubble for the turn.
``kind``
    ``"ambient_observation"``.
``ambient``
    The observation record: ``observation_id``, ``sources`` (``webcam`` /
    ``screen`` / ``microphone``), ``captured_at``, ``voice_mode`` and, once the
    triage node has run, ``decision``, ``summary``, ``reason``,
    ``observation_kind``, ``salience`` and ``needs_owner_action``.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage

AMBIENT_MESSAGE_KIND = "ambient_observation"

DECISION_IGNORE = "ignore"
DECISION_RESPOND = "respond"
DECISION_NOTIFY = "notify"
AMBIENT_DECISIONS = (DECISION_IGNORE, DECISION_RESPOND, DECISION_NOTIFY)

SOURCE_WEBCAM = "webcam"
SOURCE_SCREEN = "screen"
SOURCE_MICROPHONE = "microphone"
KNOWN_SOURCES = (SOURCE_WEBCAM, SOURCE_SCREEN, SOURCE_MICROPHONE)

_SOURCE_BY_FILENAME_STEM = {
    "webcam": SOURCE_WEBCAM,
    "camera": SOURCE_WEBCAM,
    "screen": SOURCE_SCREEN,
    "screenshot": SOURCE_SCREEN,
    "display": SOURCE_SCREEN,
    "microphone": SOURCE_MICROPHONE,
    "mic": SOURCE_MICROPHONE,
    "audio": SOURCE_MICROPHONE,
}

OBSERVATION_HEADER_PREFIX = "[AMBIENT_OBSERVATION"

RESPOND_INSTRUCTION = (
    "The conversation partner did not type this: the assistant noticed this on "
    "the conversation partner's webcam or screen, and decided to speak up. "
    "React the way this avatar naturally would on noticing this — briefly, in "
    "the avatar's own voice — or use a tool when a tool helps. Do not read the "
    "description back, and do not mention a camera or a screenshot unless doing "
    "so is natural."
)

NOTIFY_INSTRUCTION = (
    "The conversation partner did not type this: the assistant noticed this on "
    "the conversation partner's webcam or screen, and decided the conversation "
    "partner should hear about this. Write one short heads-up message to the "
    "conversation partner saying what was noticed and what the assistant "
    "suggests. Do not take actions and do not call tools."
)


def resolve_sources(filenames: list[str], sources_form_value: str | None) -> list[str]:
    """Name the source of each attached file, aligned with ``filenames``.

    An explicit ``sources`` form value (a JSON list, or a comma-separated
    string) wins when the count matches the files. Otherwise the filename
    stem decides (``webcam.jpg`` → ``webcam``); anything else is ``image``.
    """
    explicit: list[str] = []
    raw = (sources_form_value or "").strip()
    if raw:
        parsed: Any = None
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = [part.strip() for part in raw.split(",") if part.strip()]
        if isinstance(parsed, list):
            explicit = [str(item).strip().lower() for item in parsed]
    resolved: list[str] = []
    for index, filename in enumerate(filenames):
        if index < len(explicit) and explicit[index]:
            resolved.append(explicit[index])
            continue
        stem = str(filename or "").rsplit("/", 1)[-1].split(".", 1)[0].strip().lower()
        resolved.append(_SOURCE_BY_FILENAME_STEM.get(stem, "image"))
    return resolved


def build_ambient_additional_kwargs(
    *,
    sources: list[str],
    captured_at: str | None,
    voice_mode: bool,
    image_filenames: list[str] | None = None,
    observation_id: str | None = None,
) -> dict[str, Any]:
    """Build the ``additional_kwargs`` of an ambient ``HumanMessage`` before triage."""
    additional_kwargs: dict[str, Any] = {
        "hidden": True,
        "kind": AMBIENT_MESSAGE_KIND,
        "ambient": {
            "observation_id": observation_id or str(uuid4()),
            "sources": list(sources),
            "captured_at": captured_at or "",
            "voice_mode": bool(voice_mode),
        },
    }
    if image_filenames:
        additional_kwargs["image_filenames"] = list(image_filenames)
    return additional_kwargs


def _additional_kwargs_of(message: Any) -> dict[str, Any]:
    if isinstance(message, BaseMessage):
        return dict(message.additional_kwargs or {})
    if isinstance(message, dict):
        kwargs = message.get("additional_kwargs")
        return dict(kwargs) if isinstance(kwargs, dict) else {}
    return {}


def is_hidden_message(message: Any) -> bool:
    """Whether a stored message is hidden from transcripts (any hidden kind)."""
    return bool(_additional_kwargs_of(message).get("hidden"))


def is_ambient_observation(message: Any) -> bool:
    """Whether a message (object or serialized dict) is an ambient observation."""
    return _additional_kwargs_of(message).get("kind") == AMBIENT_MESSAGE_KIND


def ambient_details(message: Any) -> dict[str, Any] | None:
    """Return the ``ambient`` record of an ambient observation, or ``None``."""
    if not is_ambient_observation(message):
        return None
    details = _additional_kwargs_of(message).get("ambient")
    return dict(details) if isinstance(details, dict) else {}


def message_text(message: Any) -> str:
    """Plain text of a message whose content may be a string or content blocks."""
    content = (
        message.content
        if isinstance(message, BaseMessage)
        else (message or {}).get("content")
        if isinstance(message, (BaseMessage, dict))
        else ""
    )
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content or "")


def observation_header(ambient: dict[str, Any]) -> str:
    """Render the first line of an ambient observation's text."""
    sources = ",".join(str(source) for source in (ambient.get("sources") or []))
    header = (
        f"{OBSERVATION_HEADER_PREFIX} id={ambient.get('observation_id') or ''}"
        f" captured_at={ambient.get('captured_at') or ''}"
        f" sources={sources}"
    )
    decision = ambient.get("decision")
    if decision:
        header += f" decision={decision}"
    return header + "]"


def split_observation_text(text: str) -> tuple[str | None, str]:
    """Split an observation's text into its header line (if any) and body."""
    stripped = (text or "").lstrip()
    if not stripped.startswith(OBSERVATION_HEADER_PREFIX):
        return None, text or ""
    first_line, _, rest = stripped.partition("\n")
    return first_line, rest.strip()


def strip_instruction(body: str) -> str:
    """Drop a previously appended respond/notify instruction from a body."""
    for instruction in (RESPOND_INSTRUCTION, NOTIFY_INSTRUCTION):
        marker = "\n\n" + instruction
        if body.endswith(marker):
            return body[: -len(marker)]
    return body


def compose_observation_text(ambient: dict[str, Any], body: str) -> str:
    """Header + body + the instruction matching the triage decision."""
    parts = [observation_header(ambient), body.strip()]
    decision = ambient.get("decision")
    if decision == DECISION_RESPOND:
        parts.append(RESPOND_INSTRUCTION)
    elif decision == DECISION_NOTIFY:
        parts.append(NOTIFY_INSTRUCTION)
    return "\n".join(part for part in parts[:2] if part) + (
        "\n\n" + parts[2] if len(parts) > 2 else ""
    )


def recent_ambient_observations(
    messages: list[Any], limit: int, *, exclude_message_id: str | None = None
) -> list[dict[str, Any]]:
    """Collect the most recent ambient observations, oldest first, as records."""
    found: list[dict[str, Any]] = []
    for message in reversed(list(messages or [])):
        if not is_ambient_observation(message):
            continue
        message_id = getattr(message, "id", None) or (
            message.get("id") if isinstance(message, dict) else None
        )
        if exclude_message_id and message_id == exclude_message_id:
            continue
        details = ambient_details(message) or {}
        _header, body = split_observation_text(message_text(message))
        found.append(
            {
                "observation_id": details.get("observation_id"),
                "captured_at": details.get("captured_at"),
                "decision": details.get("decision"),
                "summary": details.get("summary"),
                "observation_kind": details.get("observation_kind"),
                "text": strip_instruction(body),
            }
        )
        if len(found) >= max(1, int(limit)):
            break
    found.reverse()
    return found


def recent_visible_messages(messages: list[Any], limit: int) -> list[str]:
    """Collect the last ``limit`` visible human/assistant turns as ``role: text``."""
    lines: list[str] = []
    for message in reversed(list(messages or [])):
        if is_hidden_message(message):
            continue
        role = getattr(message, "type", None) or (
            message.get("type") if isinstance(message, dict) else None
        )
        if role not in ("human", "ai"):
            continue
        text = message_text(message).strip()
        if not text:
            continue
        who = "conversation partner" if role == "human" else "assistant"
        lines.append(f"{who}: {text[:300]}")
        if len(lines) >= max(1, int(limit)):
            break
    lines.reverse()
    return lines


def make_hidden_human_message(
    content: str, additional_kwargs: dict[str, Any], *, message_id: str | None = None
) -> HumanMessage:
    """Build a hidden ambient ``HumanMessage`` carrying the given tag."""
    return HumanMessage(
        id=message_id or str(uuid4()),
        content=content,
        additional_kwargs=dict(additional_kwargs),
    )


class AmbientThrottle:
    """Process-local minimum interval between ambient observations per thread.

    The browser paces itself with its own interval; this is the API's floor so a
    misconfigured or hostile client cannot flood a thread with vision calls.
    """

    def __init__(self) -> None:
        """Start with no thread seen."""
        self._last_seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def check_and_mark(
        self,
        thread_id: str | None,
        min_interval_seconds: float,
        *,
        now: float | None = None,
    ) -> float | None:
        """Return seconds still to wait, or ``None`` when the observation may proceed."""
        if not thread_id or min_interval_seconds <= 0:
            return None
        moment = now if now is not None else time.monotonic()
        with self._lock:
            self._evict(moment, min_interval_seconds)
            previous = self._last_seen.get(thread_id)
            if previous is not None:
                elapsed = moment - previous
                if elapsed < min_interval_seconds:
                    return round(min_interval_seconds - elapsed, 3)
            self._last_seen[thread_id] = moment
        return None

    def _evict(self, now: float, min_interval_seconds: float) -> None:
        stale_after = max(min_interval_seconds * 10, 600.0)
        for thread_id in [
            key for key, seen in self._last_seen.items() if now - seen > stale_after
        ]:
            self._last_seen.pop(thread_id, None)


ambient_throttle = AmbientThrottle()
