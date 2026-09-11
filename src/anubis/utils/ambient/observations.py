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

CAMERA_FACING_SELF = "self"
CAMERA_FACING_WORLD = "world"

# What a browser reports for a camera track, mapped to the two directions that
# matter: "user" is the front camera pointed at the person, "environment" the
# rear camera pointed at whatever the person is looking at.
_CAMERA_FACING_BY_TRACK_SETTING: dict[str, str] = {
    "user": CAMERA_FACING_SELF,
    "front": CAMERA_FACING_SELF,
    "self": CAMERA_FACING_SELF,
    "environment": CAMERA_FACING_WORLD,
    "rear": CAMERA_FACING_WORLD,
    "back": CAMERA_FACING_WORLD,
    "world": CAMERA_FACING_WORLD,
}


def normalize_camera_facing_value(camera_facing: Any) -> str:
    """Reduce a browser facing mode to ``self`` or ``world``.

    An absent or unrecognized value reads as ``self``, the conservative
    direction: a camera pointed at the conversation partner carries no request,
    so the avatar stays quiet unless something else justifies speaking.
    """
    key = str(camera_facing or "").strip().lower()
    return _CAMERA_FACING_BY_TRACK_SETTING.get(key, CAMERA_FACING_SELF)


OBSERVATION_HEADER_PREFIX = "[AMBIENT_OBSERVATION"

# Why the triage chose to speak, handed to the avatar as the thing to react
# to. The system prompt promises the avatar this line exists on a respond
# turn, so a respond turn must always carry it.
REASON_LINE_PREFIX = "[AMBIENT_REASON]"

RESPOND_INSTRUCTION = (
    "The conversation partner did not type this: the assistant noticed this on "
    "the conversation partner's webcam or screen, and decided to speak up. The "
    "line beginning with [AMBIENT_REASON] names the one specific thing that "
    "justified speaking. Speak about that thing, in the avatar's own voice and "
    "briefly, or use a tool when a tool helps. Do not recite the description "
    "back word for word, and do not mention a camera or a screenshot unless "
    "doing so is natural. Never pad the reply with presence, reassurance, or a "
    "check-in: if the named thing warrants only a few words, say only those "
    "few words."
)

NOTIFY_INSTRUCTION = (
    "The conversation partner did not type this: the assistant noticed this on "
    "the conversation partner's webcam or screen, and decided the conversation "
    "partner should hear about this. Write one short heads-up message to the "
    "conversation partner saying what was noticed and what the assistant "
    "suggests. Do not take actions and do not call tools."
)

# Scene narration (the accessibility mode): the conversation partner has asked
# to be told continuously what is in view, so an observation is not a scene the
# assistant happened to notice — it is the answer to a question already asked.
# Nothing is classified on these turns and nothing is weighed for salience:
# every observation is spoken, because the conversation partner is relying on
# the assistant for what they cannot see for themselves.
NARRATE_INSTRUCTION = (
    "The conversation partner has switched on scene narration: they have asked "
    "the assistant to tell them what is in view, continuously, and they may not "
    "be able to see the scene themselves. This description is what the camera "
    "is pointed at right now. Say what is there, out loud, to the conversation "
    "partner. Lead with anything that bears on their safety or their next step "
    "— an obstacle, a step or kerb, a vehicle, a door, a person approaching, a "
    "sign or a screen they would want read to them — and then the rest of the "
    "scene in the order it matters. Place things from the conversation "
    "partner's point of view: to your left, ahead of you, on the far side. "
    "Read short visible text out exactly as written. Keep it to one or two "
    "sentences unless something genuinely needs more; this is spoken aloud and "
    "another description follows in seconds. Say only what is in the "
    "description: never invent a detail, and when something is unclear say so "
    "in a few words rather than guessing. Do not greet, do not check in, do "
    "not say the assistant is looking or watching, and do not mention a "
    "camera, an image or a frame — just say what is there."
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
    NARRATE_INSTRUCTION,
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
    camera_facing: str | None = None,
    narrate: bool = False,
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
            # Which way the webcam pointed: the triage classifier reads a
            # world-facing camera as a standing request to be told what is in
            # view, and a self-facing camera as carrying no request at all.
            "camera_facing": normalize_camera_facing_value(camera_facing),
            # Scene narration is on: the conversation partner asked to be told
            # what is in view and is waiting to hear this one. There is nothing
            # to classify — the request was made once, for every observation —
            # so this bypasses triage entirely rather than arguing with it.
            "narrate": bool(narrate),
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
    if is_narration_observation(ambient):
        # Marked on the header so that, read back later in the thread, a
        # description spoken to somebody who could not see is never mistaken
        # for a scene the assistant chose to bring up on its own.
        header += " narration=on"
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


def is_narration_observation(ambient: dict[str, Any] | None) -> bool:
    """Whether this observation was captured with scene narration switched on.

    A narration observation is the answer to a standing request: the
    conversation partner asked to be told what is in view and every capture
    since is part of that answer. It is never classified, never weighed for
    salience, and never silenced by the quiet period after the avatar last
    spoke — silence is the one thing it must not produce.
    """
    return bool((ambient or {}).get("narrate"))


def is_speech_observation(ambient: dict[str, Any]) -> bool:
    """Whether an observation was heard (microphone only), not seen."""
    sources = [str(source) for source in (ambient.get("sources") or [])]
    return bool(sources) and all(source == SOURCE_MICROPHONE for source in sources)


def strip_instruction(body: str) -> str:
    """Drop a previously appended respond/notify instruction from a body."""
    for instruction in ALL_INSTRUCTIONS:
        marker = "\n\n" + instruction
        position = body.find(marker)
        if position != -1:
            # Truncate rather than trimming a suffix: a respond instruction is
            # followed by the [AMBIENT_REASON] line, so the instruction is not
            # always the last thing in the body.
            body = body[:position]
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
    if decision == DECISION_RESPOND and is_narration_observation(ambient):
        # No instruction at all: a narrated observation is not answered by the
        # avatar. The description IS the reading and the browser speaks it
        # directly, so an instruction here would be a line of dead text stored
        # on the thread for nobody. ``NARRATE_INSTRUCTION`` is kept in
        # ``ALL_INSTRUCTIONS`` so threads written before this still strip clean.
        pass
    elif decision == DECISION_RESPOND:
        parts.append(RESPOND_INSTRUCTION_SPEECH if heard else RESPOND_INSTRUCTION)
    elif decision == DECISION_NOTIFY and offer is not None:
        parts.append(
            NOTIFY_INSTRUCTION_SPEECH_WITH_OFFER
            if heard
            else NOTIFY_INSTRUCTION_WITH_OFFER
        )
    elif decision == DECISION_NOTIFY:
        parts.append(NOTIFY_INSTRUCTION_SPEECH if heard else NOTIFY_INSTRUCTION)
    if (
        decision == DECISION_RESPOND
        and len(parts) > 2
        and not is_narration_observation(ambient)
    ):
        # Narration carries no reason line: the reason is the standing request,
        # and the system prompt promises that line names the one thing that
        # justified breaking silence. Nothing was broken here.
        reason = str(ambient.get("reason") or "").strip().replace("\n", " ")
        if reason:
            parts[2] += f"\n{REASON_LINE_PREFIX} {reason}"
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


class AmbientSpeechCooldown:
    """Process-local quiet period after the avatar speaks about what it noticed.

    ``AmbientThrottle`` bounds how often a thread may be *looked at*. This
    bounds how often the avatar may *say something* about what was seen, which
    is a different and much longer interval: a person sharing a webcam consents
    to being looked at, not to being spoken to every time the classifier finds
    something remarkable. Without this, the classifier's per-observation
    judgement is the only thing standing between a long share and a stream of
    interruptions, and independent draws on a quiet scene eventually produce
    one.
    """

    def __init__(self) -> None:
        """Start with no thread having spoken."""
        self._last_spoken: dict[str, float] = {}
        self._lock = threading.Lock()

    def seconds_remaining(
        self,
        thread_id: str | None,
        cooldown_seconds: float,
        *,
        now: float | None = None,
    ) -> float | None:
        """Return seconds still to wait, or ``None`` when the avatar may speak.

        This only reads. ``mark_spoken`` records the moment, so that an
        observation demoted to ``ignore`` for any other reason does not start a
        cooldown it never earned.
        """
        if not thread_id or cooldown_seconds <= 0:
            return None
        moment = now if now is not None else time.monotonic()
        with self._lock:
            self._evict(moment, cooldown_seconds)
            previous = self._last_spoken.get(thread_id)
            if previous is None:
                return None
            elapsed = moment - previous
            if elapsed >= cooldown_seconds:
                return None
            return round(cooldown_seconds - elapsed, 3)

    def mark_spoken(self, thread_id: str | None, *, now: float | None = None) -> None:
        """Record that the avatar has just spoken about an observation."""
        if not thread_id:
            return
        moment = now if now is not None else time.monotonic()
        with self._lock:
            self._last_spoken[thread_id] = moment

    def _evict(self, now: float, cooldown_seconds: float) -> None:
        stale_after = max(cooldown_seconds * 10, 600.0)
        for thread_id in [
            key for key, seen in self._last_spoken.items() if now - seen > stale_after
        ]:
            self._last_spoken.pop(thread_id, None)


ambient_throttle = AmbientThrottle()
ambient_speech_cooldown = AmbientSpeechCooldown()


# --- What is being shared at this exact moment ------------------------------
#
# Ambient observations pile up in the thread and never expire. An observation
# of a screen the conversation partner stopped sharing an hour ago reads
# exactly like a description of what is on that screen right now, and the
# avatar, asked what is on the screen, will happily narrate the stale one. The
# browser reports which sources are live on every turn; these helpers turn that
# report into a section of the system prompt that separates the present from
# the record.

#: The freshest description of each source still counts as the present for this
#: long. Past this, a live source's newest observation is named with its age so
#: the avatar reaches for ``look_now`` instead of reading the description back.
LIVE_OBSERVATION_FRESH_SECONDS = 90.0


def newest_observation_age_seconds(
    messages: list[Any], source: str, *, now: float | None = None
) -> float | None:
    """How long ago the newest observation of one source was captured.

    :param messages: The thread's messages.
    :param source: ``webcam`` / ``screen`` / ``microphone``.
    :param now: Unix seconds to measure against; the clock by default.
    :returns: Seconds since the newest observation of that source, or ``None``
        when the conversation holds no observation of that source, or when the
        one it holds carries no readable ``captured_at``.
    """
    from datetime import datetime, timezone

    reference = float(now) if now is not None else datetime.now(timezone.utc).timestamp()
    for message in reversed(list(messages or [])):
        if not is_ambient_observation(message):
            continue
        details = ambient_details(message) or {}
        if source not in [str(name) for name in (details.get("sources") or [])]:
            continue
        stamp = str(details.get("captured_at") or "").strip()
        if not stamp:
            return None
        try:
            parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, reference - parsed.timestamp())
    return None


def describe_age(seconds: float | None) -> str:
    """Say an age in words the way a person would say the age out loud."""
    if seconds is None:
        return "at an unknown time"
    if seconds < 60:
        return "less than a minute ago"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"about {minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = int(minutes // 60)
    return f"about {hours} hour{'s' if hours != 1 else ''} ago"


def build_live_shares_block(
    live_sources: list[str] | None,
    messages: list[Any],
    *,
    can_look_now: bool,
    may_control_shares: bool = False,
    peekable_sources: list[str] | None = None,
    scene_narration_on: bool = False,
    now: float | None = None,
) -> str:
    """Build the ``<LIVE_SHARES>`` section of the system prompt.

    The section is added only to a conversation that holds at least one scene
    observation or has something shared right now; a conversation that has
    never involved a camera keeps the prompt the conversation already had.

    :param live_sources: What the browser reported live on this turn.
    :param messages: The thread's messages, read for the age of the newest
        observation of each source.
    :param can_look_now: Whether the ``look_now`` tool is attached this turn.
    :param may_control_shares: Whether the owner allowed this avatar, in this
        browser, to look on its own and to switch a share off.
    :param peekable_sources: What the browser can open for a single look right
        now — the camera when its peek permission is granted, the desktop when
        the owner granted a desktop peek this browser is still holding.
    :param scene_narration_on: Whether the browser reported scene narration
        switched on this turn — the camera is being described to the
        conversation partner continuously because they asked, and usually
        because they cannot see the scene themselves.
    :param now: Unix seconds to measure ages against; the clock by default.
    :returns: The section, or ``""`` when the conversation needs no section.
    """
    live = [
        source
        for source in (SOURCE_WEBCAM, SOURCE_SCREEN)
        if source in [str(name) for name in (live_sources or [])]
    ]
    peekable = [
        source
        for source in (SOURCE_WEBCAM, SOURCE_SCREEN)
        if source in [str(name) for name in (peekable_sources or [])]
        and source not in live
    ]
    seen = [
        source
        for source in (SOURCE_WEBCAM, SOURCE_SCREEN)
        if newest_observation_age_seconds(messages, source, now=now) is not None
        or any(
            is_ambient_observation(message)
            and source
            in [str(name) for name in (ambient_details(message) or {}).get("sources") or []]
            for message in (messages or [])
        )
    ]
    if (
        not live
        and not seen
        and not may_control_shares
        and not peekable
        and not scene_narration_on
    ):
        return ""

    lines: list[str] = []
    if scene_narration_on:
        # Said first, because it changes how every other line is read: the
        # observations arriving are not scenes the assistant happened to
        # notice, they are answers to a standing request from somebody who
        # may have nothing but the assistant's voice to go on.
        lines.append(
            "Scene narration is ON. The conversation partner asked to be told "
            "what is in front of them, continuously, and may not be able to see "
            "the scene themselves; the camera is pointed outward at the world "
            "and every observation of it is described to them aloud. Treat "
            "each observation turn as that description: say what is there, "
            "lead with anything that bears on their safety or their next step, "
            "place things from their point of view, read visible text exactly, "
            "keep it short, and never invent what the description does not "
            "say. When they speak, answer them; they may be asking about "
            "something just described. When they ask for the describing to "
            "stop, call set_scene_narration with enabled=false."
        )
    if live:
        lines.append(
            "Being shared at this moment: "
            + " and ".join(live)
            + "."
        )
    else:
        lines.append("Nothing is being shared at this moment.")
    # The camera and the desktop are two different views of two different
    # things, and the avatar answers the wrong question when it treats them as
    # one "what can I see". Name what each one is, every turn the section is
    # built, so the choice between them is never a guess.
    lines.append(
        "The camera and the desktop are separate views and are never "
        "interchangeable. The camera shows the conversation partner themselves "
        "and the room they are in. The desktop shows what is on their screen — "
        "the application, page, code or error in front of them. Answer a "
        "question about one only from that one, and say which of the two is "
        "being described whenever both are in play."
    )
    if peekable:
        lines.append(
            "Not being shared, but open to a single look right now: "
            + " and ".join(peekable)
            + ". The conversation partner allowed this avatar to open "
            + ("it" if len(peekable) == 1 else "them")
            + " for one look and close "
            + ("it" if len(peekable) == 1 else "them")
            + " again. That is a glance taken when the answer needs it, never "
            "a standing watch."
        )

    stopped = [source for source in seen if source not in live]
    for source in stopped:
        age = newest_observation_age_seconds(messages, source, now=now)
        lines.append(
            f"The {source} is NOT being shared any more. This conversation still "
            f"holds descriptions of that {source}, the newest captured "
            f"{describe_age(age)}. Those describe what the {source} used to "
            "show, not what the "
            f"{source} shows now."
        )

    for source in live:
        age = newest_observation_age_seconds(messages, source, now=now)
        if age is not None and age > LIVE_OBSERVATION_FRESH_SECONDS:
            lines.append(
                f"The newest description of the {source} was captured "
                f"{describe_age(age)}, so that description may no longer match "
                f"what the {source} shows."
            )

    if stopped or not live:
        lines.append(
            "Never describe a source that is not being shared as though the "
            "assistant can see that source now. Say plainly that the "
            "conversation partner is not sharing it."
        )

    # The marks the model-facing copy of the conversation carries. Without
    # this the marks are unexplained text in the middle of an observation.
    lines.append(
        "Every webcam / screen observation in this conversation is marked. "
        f"[{CURRENT_VIEW_MARKER} ...] is what that source shows now. "
        f"[{EARLIER_VIEW_MARKER} ...] is what that source showed at the time "
        "named in the mark and is history — the share may have ended, or a "
        "later look may have replaced it. Describe what is in view now only "
        f"from a [{CURRENT_VIEW_MARKER}] observation or from a look taken this "
        f"turn. A [{EARLIER_VIEW_MARKER}] observation may be referred to as "
        "something seen earlier, never as something in view now."
    )

    if can_look_now:
        reachable = live + peekable
        if reachable:
            lines.append(
                "When the conversation partner asks what the assistant sees, or "
                "when what is on "
                + " or ".join(reachable)
                + " at this moment decides the answer, call look_now for a fresh "
                "look rather than answering from a description already in this "
                "conversation. Name the source the question is about — webcam "
                "for the camera, screen for the desktop — instead of asking for "
                "everything by default."
            )
        else:
            lines.append(
                "When the conversation partner asks what the assistant sees, "
                "call look_now rather than answering from a description already "
                "in this conversation. The tool will confirm that nothing is "
                "being shared, which is the answer to give."
            )
        if may_control_shares and SOURCE_SCREEN not in reachable:
            lines.append(
                "The desktop is not being shared and cannot be opened by this "
                "avatar; asking look_now for it puts a button in front of the "
                "conversation partner to press."
            )
    if may_control_shares or peekable:
        lines.append(
            "A camera or a desktop is opened only for the look that needs it "
            "and is closed straight after — never left running, and never "
            "opened when nothing about the current scene bears on the answer. "
            "Use stop_sharing to switch one off when the conversation partner "
            "asks or when it has plainly served its purpose."
        )

    return "\n<LIVE_SHARES>\n" + "\n".join(lines) + "\n</LIVE_SHARES>\n"


# --- Marking each observation as the current view or an earlier one ---------
#
# The LIVE_SHARES section says which sources are shared at this moment, but the
# observations themselves sit in the thread as flat present-tense descriptions:
# "screen: a terminal showing three repositories". Read one of those in the
# middle of a conversation and nothing in it says whether it is what the screen
# shows now or what the screen showed twenty minutes ago, before the share
# ended. So each observation is marked, in the copy handed to the model only,
# with which of the two it is.

#: Appended to an observation that is the newest, still-live look at its sources.
CURRENT_VIEW_MARKER = "CURRENT VIEW"

#: Appended to every other observation, with the reason it is no longer current.
EARLIER_VIEW_MARKER = "EARLIER VIEW"


def _scene_sources_of(ambient: dict[str, Any]) -> list[str]:
    """The webcam / screen sources of one observation; a heard turn has none."""
    return [
        str(source)
        for source in (ambient.get("sources") or [])
        if str(source) in (SOURCE_WEBCAM, SOURCE_SCREEN)
    ]


def describe_view_currency(
    ambient: dict[str, Any],
    *,
    live_sources: list[str],
    is_newest_for_every_source: bool,
    age_seconds: float | None,
) -> str:
    """Say whether one observation is the current view of its sources, and why.

    :param ambient: The observation's ``ambient`` record.
    :param live_sources: What is being shared at this moment.
    :param is_newest_for_every_source: Whether this observation is the most
        recent one covering each of its own sources.
    :param age_seconds: How long ago the observation was captured.
    :returns: The bracketed marker to put after the observation's header.
    """
    sources = _scene_sources_of(ambient)
    if not sources:
        return ""
    live = [source for source in sources if source in (live_sources or [])]
    stopped = [source for source in sources if source not in (live_sources or [])]
    age = describe_age(age_seconds)

    if stopped:
        which = " and ".join(stopped)
        was = "are" if len(stopped) > 1 else "is"
        reason = (
            f"the {which} {was} NOT being shared any more, so this describes what "
            f"the {which} showed {age}, not what the {which} shows now"
        )
        if live:
            reason += f" (the {' and '.join(live)} is still being shared)"
        return f"[{EARLIER_VIEW_MARKER} — captured {age}; {reason}]"

    if not is_newest_for_every_source:
        return (
            f"[{EARLIER_VIEW_MARKER} — captured {age}; a later look at the "
            f"{' and '.join(sources)} came after this one]"
        )

    if age_seconds is not None and age_seconds > LIVE_OBSERVATION_FRESH_SECONDS:
        return (
            f"[{EARLIER_VIEW_MARKER} — captured {age}; the "
            f"{' and '.join(sources)} is still being shared, but this is old "
            "enough that the scene may have changed since]"
        )

    return f"[{CURRENT_VIEW_MARKER} — captured {age} and still being shared]"


def mark_view_currency(
    messages: list[Any], live_sources: list[str] | None, *, now: float | None = None
) -> list[Any]:
    """Return the messages with every scene observation marked current or earlier.

    The returned list is for the model only — the marks are never written back
    to the thread, because whether an observation is current is true of the
    moment it is read, not of the observation.

    :param messages: The thread's messages.
    :param live_sources: What the browser reported live on this turn.
    :param now: Unix seconds to measure ages against; the clock by default.
    :returns: A new list; messages that are not scene observations are the very
        same objects, unchanged.
    """
    from datetime import datetime, timezone

    reference = float(now) if now is not None else datetime.now(timezone.utc).timestamp()
    live = [str(source) for source in (live_sources or [])]

    # Which observation is the newest for each source, walking newest first.
    newest_seen: set[str] = set()
    is_newest: dict[int, bool] = {}
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not is_ambient_observation(message):
            continue
        sources = _scene_sources_of(ambient_details(message) or {})
        if not sources:
            continue
        is_newest[index] = all(source not in newest_seen for source in sources)
        newest_seen.update(sources)

    marked: list[Any] = []
    for index, message in enumerate(messages or []):
        if index not in is_newest:
            marked.append(message)
            continue
        ambient = ambient_details(message) or {}
        text = message_text(message)
        header, body = split_observation_text(text)
        if not header:
            marked.append(message)
            continue
        marker = describe_view_currency(
            ambient,
            live_sources=live,
            is_newest_for_every_source=is_newest[index],
            age_seconds=_age_of_stamp(ambient.get("captured_at"), reference),
        )
        if not marker:
            marked.append(message)
            continue
        rebuilt = "\n".join(part for part in (f"{header} {marker}", body) if part)
        try:
            marked.append(message.model_copy(update={"content": rebuilt}))
        except AttributeError:
            # A plain dict message (tests, a client that sent one) is copied by
            # hand rather than losing its mark.
            copied = dict(message)
            copied["content"] = rebuilt
            marked.append(copied)
    return marked


def _age_of_stamp(stamp: Any, reference: float) -> float | None:
    """Seconds between an ISO ``captured_at`` and ``reference``, or ``None``."""
    from datetime import datetime, timezone

    text = str(stamp or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, reference - parsed.timestamp())
