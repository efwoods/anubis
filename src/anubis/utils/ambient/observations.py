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
import re
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

# What the avatar offers to do about a ``notify`` observation once the
# conversation partner allows it: one verb the avatar will perform (``draft``,
# ``reply``, ``remind``, ``research``, ``summarize``, ``schedule`` ...). The
# card puts the verb on the button and the wording beside it. ``none`` is a
# plain heads-up.
PROPOSED_ACTION_NONE = "none"
PROPOSED_ACTION_MAX_LETTERS = 20
_PROPOSED_ACTION_LETTERS = re.compile(r"[^a-z]")

# The hidden turn that carries an allowed action back to the avatar. The
# decision recorded on the avatar's reply is ``act`` so the browser can tell a
# reply the conversation partner asked for from a heads-up card.
AMBIENT_ACTION_MESSAGE_KIND = "ambient_action"
DECISION_ACT = "act"
ACTION_HEADER_PREFIX = "[AMBIENT_ACTION"
OFFER_LINE_PREFIX = "[AMBIENT_OFFER]"

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

# Spoken turns heard in the room (a live-voice utterance labelled by speaker).
RESPOND_INSTRUCTION_SPEECH = (
    "The assistant heard this spoken aloud in the room; the lines are labelled "
    "by speaker. The line labelled with the assistant's own name is the "
    "assistant's own person speaking (the same identity as the assistant), and "
    "the other labelled speakers are the people the assistant is talking with. "
    "The assistant decided to speak aloud. Answer the other person the way the "
    "assistant's own person would, in the first person and in that person's own "
    "voice, continuing what the person said rather than answering the person. "
    "Address another speaker by name only when the name is a real name, never as "
    "'Speaker 2'. Do not repeat the transcript and do not say that the words "
    "were transcribed."
)

NOTIFY_INSTRUCTION_SPEECH = (
    "The assistant heard this spoken aloud in the room; the lines are labelled "
    "by speaker, and the line labelled with the assistant's own name is the "
    "assistant's own person speaking. The assistant decided that person should "
    "be reminded of something said. Write one short heads-up saying what was "
    "heard and what the assistant suggests. Do not take actions and do not call "
    "tools."
)

# Appended to a notify instruction when the triage named something the
# assistant could do once allowed. The offer line under the header names it.
OFFER_SUFFIX = (
    " The line beginning with [AMBIENT_OFFER] names what the assistant could do "
    "next if the conversation partner allows this. End the heads-up with one "
    "short clause offering exactly that, and wait: the conversation partner "
    "will choose."
)
NOTIFY_INSTRUCTION_WITH_OFFER = NOTIFY_INSTRUCTION + OFFER_SUFFIX
NOTIFY_INSTRUCTION_SPEECH_WITH_OFFER = NOTIFY_INSTRUCTION_SPEECH + OFFER_SUFFIX

ALL_INSTRUCTIONS = (
    RESPOND_INSTRUCTION,
    NOTIFY_INSTRUCTION,
    RESPOND_INSTRUCTION_SPEECH,
    NOTIFY_INSTRUCTION_SPEECH,
    NOTIFY_INSTRUCTION_WITH_OFFER,
    NOTIFY_INSTRUCTION_SPEECH_WITH_OFFER,
)

ACTION_INSTRUCTION = (
    "The conversation partner did not type this. The assistant earlier noticed "
    "the scene described above and offered to do the named action; the "
    "conversation partner has now allowed the action. Do the action now, in "
    "the avatar's own voice: speak directly to the conversation partner, and "
    "use a tool when the action needs one. Do not ask for permission again, do "
    "not read the scene back, and do not mention a camera or a screenshot "
    "unless doing so is natural."
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
    hidden: bool = True,
) -> dict[str, Any]:
    """Build the ``additional_kwargs`` of an ambient ``HumanMessage`` before triage.

    ``hidden=False`` keeps the turn visible in transcripts: a spoken turn heard
    in the room is shown to the owner as what was heard even though the avatar
    triages the turn like an observation.
    """
    additional_kwargs: dict[str, Any] = {
        "hidden": bool(hidden),
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


def is_ambient_action(message: Any) -> bool:
    """Whether a message is the hidden turn carrying an allowed action."""
    return _additional_kwargs_of(message).get("kind") == AMBIENT_ACTION_MESSAGE_KIND


def ambient_details(message: Any) -> dict[str, Any] | None:
    """Return the ``ambient`` record of an observation or an allowed action, or ``None``."""
    if not is_ambient_observation(message) and not is_ambient_action(message):
        return None
    details = _additional_kwargs_of(message).get("ambient")
    return dict(details) if isinstance(details, dict) else {}


def normalize_proposed_action(value: Any) -> str:
    """Coerce a proposed action to one lowercase verb, or ``none``.

    The first word is kept and everything but letters is dropped, so
    "Draft a reply" becomes ``draft`` and "Reply!" becomes ``reply``.
    """
    first_word = str(value or "").strip().split(" ", 1)[0] if value else ""
    action = _PROPOSED_ACTION_LETTERS.sub("", first_word.lower())[
        :PROPOSED_ACTION_MAX_LETTERS
    ]
    return action or PROPOSED_ACTION_NONE


def proposed_offer(ambient: dict[str, Any]) -> tuple[str, str] | None:
    """Return the ``(action, description)`` a notify observation offers, or ``None``."""
    if not ambient or ambient.get("decision") != DECISION_NOTIFY:
        return None
    action = normalize_proposed_action(ambient.get("proposed_action"))
    description = str(ambient.get("action_description") or "").strip()
    if action == PROPOSED_ACTION_NONE or not description:
        return None
    return action, description


def build_ambient_action_additional_kwargs(
    *,
    observation_id: str,
    observation_kind: str,
    action: str,
    action_description: str,
    summary: str,
) -> dict[str, Any]:
    """Build the ``additional_kwargs`` of the hidden turn carrying an allowed action.

    The turn is hidden like an observation but is not one: the triage node
    does not run on it, and the avatar's reply is stamped with this record
    (``decision`` ``act``) so a thumb on that reply is learned against the
    observation it answered.
    """
    return {
        "hidden": True,
        "kind": AMBIENT_ACTION_MESSAGE_KIND,
        "ambient": {
            "observation_id": str(observation_id),
            "observation_kind": (observation_kind or "other").strip().lower()[:40]
            or "other",
            "decision": DECISION_ACT,
            "action": normalize_proposed_action(action),
            "action_description": (action_description or "").strip()[:300],
            "summary": (summary or "").strip()[:300],
        },
    }


def compose_ambient_action_text(ambient: dict[str, Any]) -> str:
    """Return the text of the hidden turn that asks the avatar to carry out an allowed action."""
    header = (
        f"{ACTION_HEADER_PREFIX} id={ambient.get('observation_id') or ''}"
        f" kind={ambient.get('observation_kind') or 'other'}"
        f" action={ambient.get('action') or 'reply'}]"
    )
    scene = str(ambient.get("summary") or "").strip()
    description = str(ambient.get("action_description") or "").strip()
    lines = [header]
    if scene:
        lines.append(f"Scene noticed earlier: {scene}")
    lines.append(
        f"Allowed action: {description or 'reply to the conversation partner'}"
    )
    return "\n".join(lines) + "\n\n" + ACTION_INSTRUCTION


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
    if proposed_offer(ambient) is not None:
        header += f" proposed_action={normalize_proposed_action(ambient.get('proposed_action'))}"
    return header + "]"


def split_observation_text(text: str) -> tuple[str | None, str]:
    """Split an observation's text into its header line (if any) and body."""
    stripped = (text or "").lstrip()
    if not stripped.startswith(OBSERVATION_HEADER_PREFIX):
        return None, text or ""
    first_line, _, rest = stripped.partition("\n")
    return first_line, rest.strip()


def is_speech_observation(ambient: dict[str, Any]) -> bool:
    """Whether an observation was heard (microphone only), not seen."""
    sources = [str(source) for source in (ambient.get("sources") or [])]
    return bool(sources) and all(source == SOURCE_MICROPHONE for source in sources)


def strip_instruction(body: str) -> str:
    """Drop a previously appended respond/notify instruction from a body."""
    for instruction in ALL_INSTRUCTIONS:
        marker = "\n\n" + instruction
        if body.endswith(marker):
            body = body[: -len(marker)]
            break
    return strip_offer_line(body)


def strip_offer_line(body: str) -> str:
    """Drop the ``[AMBIENT_OFFER]`` line a notify observation was rewritten with."""
    stripped = (body or "").lstrip()
    if not stripped.startswith(OFFER_LINE_PREFIX):
        return body
    _offer, _newline, rest = stripped.partition("\n")
    return rest.strip()


def compose_observation_text(ambient: dict[str, Any], body: str) -> str:
    """Header + body + the instruction matching the triage decision."""
    offer = proposed_offer(ambient)
    lead = observation_header(ambient)
    if offer is not None:
        lead += f"\n{OFFER_LINE_PREFIX} {offer[1]}"
    parts = [lead, strip_offer_line(body).strip()]
    decision = ambient.get("decision")
    heard = is_speech_observation(ambient)
    if decision == DECISION_RESPOND:
        parts.append(RESPOND_INSTRUCTION_SPEECH if heard else RESPOND_INSTRUCTION)
    elif decision == DECISION_NOTIFY and offer is not None:
        parts.append(
            NOTIFY_INSTRUCTION_SPEECH_WITH_OFFER
            if heard
            else NOTIFY_INSTRUCTION_WITH_OFFER
        )
    elif decision == DECISION_NOTIFY:
        parts.append(NOTIFY_INSTRUCTION_SPEECH if heard else NOTIFY_INSTRUCTION)
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
                "proposed_action": details.get("proposed_action"),
                "action_description": details.get("action_description"),
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
