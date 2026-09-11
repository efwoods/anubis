"""The avatar's live emotional state: what the avatar is feeling right now.

The system prompt has always had a ``=== YOUR EMOTIONS ===`` slot and it has always
been blank, because nothing ever wrote it (the code that would have is commented
out in ``src/anubis/utils/nodes.py``). This module fills it.

The state is an eight-emotion Plutchik vector with two parts:

``baseline_wheel``  the target's emotional temperament, read from uploaded media by
                    the ``emotional_baseline`` dimension. This is who the person is
                    when nothing in particular is happening.
``wheel``           where the avatar is right now, which starts at the baseline and
                    is moved by the conversation.

Three things move it, and none of them costs a model call, which is what lets the
refresh live inside the existing ``observe_user`` branch:

* the sentiment of the message just received, already classified by Go Emotions in
  that same node;
* the sentiment of the avatar's own last reply, already classified in ``think``;
* any stored emotional trigger the incoming message matches, retrieved by the same
  similarity search the rest of the prompt already runs.

Between turns the state decays back toward the baseline on a half-life, so a
conversation that made the avatar angry an hour ago does not leave it angry
forever, and a fresh conversation starts from the person's own temperament.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import Any, Iterable, Mapping

from src.anubis.utils.psycho.namespaces import (
    CURRENT_RECORD_KEY,
    current_emotion_namespace,
)

logger = logging.getLogger(__name__)

# The eight primary emotions of the Plutchik wheel, in wheel order.
PLUTCHIK_PRIMARY_EMOTIONS = (
    "joy",
    "trust",
    "fear",
    "surprise",
    "sadness",
    "disgust",
    "anger",
    "anticipation",
)

# Go Emotions reports seven base labels; six of them are Plutchik primaries under a
# different name and one ("neutral") moves nothing. Trust and anticipation have no
# Go Emotions equivalent, so they are shaped by the baseline and by triggers only.
GO_EMOTIONS_BASE_TO_PLUTCHIK = {
    "joy": "joy",
    "anger": "anger",
    "sadness": "sadness",
    "fear": "fear",
    "surprise": "surprise",
    "disgust": "disgust",
}

# How far one signal can move the state. Small on purpose: a mood is the sum of a
# conversation, not a reaction to its latest sentence.
NUDGE_FROM_INCOMING_MESSAGE = 0.15
NUDGE_FROM_OWN_REPLY = 0.20
NUDGE_FROM_MATCHED_TRIGGER = 0.30
# Below this an emotion is not worth naming in the prompt.
MIN_RENDERED_EMOTION = 0.2

# Similarity below which a retrieved trigger is not treated as a match.
#
# Calibrated against the embedding model the store actually uses
# (``huggingface:microsoft/harrier-oss-v1-270m``) rather than guessed: on one
# avatar's stored triggers, genuine matches scored 0.586 and above while
# unrelated messages topped out at 0.538. This sits inside that gap. It is a
# narrow gap, so the value is deliberately a named constant to be re-measured
# whenever the embedding model in ``langgraph.json`` changes — a threshold
# carried over from a different model either admits everything, which makes the
# avatar react emotionally to the weather, or admits nothing, which makes the
# whole mechanism silently inert.
DEFAULT_TRIGGER_MATCH_MINIMUM_SCORE = 0.56


def _now() -> datetime:
    return datetime.now(tz=UTC)


def neutral_wheel() -> dict[str, float]:
    """An emotional wheel with every primary emotion at zero."""
    return {emotion: 0.0 for emotion in PLUTCHIK_PRIMARY_EMOTIONS}


def normalize_wheel(wheel: Mapping[str, Any] | None) -> dict[str, float]:
    """Coerce any stored or model-produced wheel into the eight known emotions."""
    normalized = neutral_wheel()
    for emotion, value in (wheel or {}).items():
        key = str(emotion).strip().lower()
        if key not in normalized:
            continue
        try:
            normalized[key] = min(max(float(value), 0.0), 1.0)
        except (TypeError, ValueError):
            continue
    return normalized


def empty_current_emotion_record() -> dict[str, Any]:
    return {
        "wheel": neutral_wheel(),
        "baseline_wheel": neutral_wheel(),
        "rendered": "",
        "updated_at": _now().isoformat(),
    }


def _decay_toward_baseline(
    wheel: Mapping[str, float],
    baseline: Mapping[str, float],
    elapsed_hours: float,
    half_life_hours: float,
) -> dict[str, float]:
    """Move the state a fraction of the way back to the baseline.

    One half-life returns half the remaining distance, which is the behaviour a
    mood actually has: it fades fastest right after the thing that caused it.
    """
    if half_life_hours <= 0 or elapsed_hours <= 0:
        return dict(wheel)
    retained = math.pow(0.5, elapsed_hours / half_life_hours)
    decayed: dict[str, float] = {}
    for emotion in PLUTCHIK_PRIMARY_EMOTIONS:
        base = float(baseline.get(emotion, 0.0))
        current = float(wheel.get(emotion, 0.0))
        decayed[emotion] = base + (current - base) * retained
    return decayed


def _nudge(wheel: dict[str, float], emotion: str | None, amount: float) -> None:
    """Raise one emotion toward one by ``amount`` of the distance remaining."""
    if not emotion:
        return
    key = GO_EMOTIONS_BASE_TO_PLUTCHIK.get(str(emotion).strip().lower())
    if key is None:
        key = str(emotion).strip().lower()
    if key not in wheel:
        return
    current = wheel[key]
    wheel[key] = min(1.0, current + (1.0 - current) * max(0.0, min(amount, 1.0)))


def render_current_emotion(record: Mapping[str, Any] | None) -> str:
    """Prose for ``=== YOUR EMOTIONS ===``, written in the avatar's own voice."""
    wheel = normalize_wheel((record or {}).get("wheel"))
    named = sorted(
        (
            (emotion, score)
            for emotion, score in wheel.items()
            if score >= MIN_RENDERED_EMOTION
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:3]
    if not named:
        return ""
    if len(named) == 1:
        feelings = named[0][0]
    else:
        feelings = (
            ", ".join(emotion for emotion, _ in named[:-1]) + f" and {named[-1][0]}"
        )
    strongest, strongest_score = named[0]
    intensity = (
        "strongly"
        if strongest_score >= 0.66
        else "mildly"
        if strongest_score < 0.4
        else ""
    )
    opening = f"Right now I am feeling {feelings}".strip()
    if intensity:
        opening = f"Right now I am feeling {intensity} {feelings}"
    return (
        f"{opening}. Let this colour how I say things — my warmth, my patience, how "
        f"much I volunteer — without ever announcing it or naming the feeling unless "
        f"the person asks how I am."
    )


def build_current_emotion_record(
    wheel: Mapping[str, float], baseline_wheel: Mapping[str, float]
) -> dict[str, Any]:
    record = {
        "wheel": normalize_wheel(wheel),
        "baseline_wheel": normalize_wheel(baseline_wheel),
        "updated_at": _now().isoformat(),
    }
    record["rendered"] = render_current_emotion(record)
    return record


def advance_current_emotion(
    record: Mapping[str, Any] | None,
    *,
    incoming_message_base_emotion: str | None = None,
    own_reply_base_emotion: str | None = None,
    matched_trigger_emotions: Iterable[str] = (),
    half_life_hours: float = 6.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Decay the state toward the baseline, then apply this turn's signals.

    Pure arithmetic over signals the graph has already computed, so this can run
    inside a node that must not add latency.
    """
    record = dict(record or empty_current_emotion_record())
    baseline = normalize_wheel(record.get("baseline_wheel"))
    wheel = normalize_wheel(record.get("wheel"))

    now = now or _now()
    elapsed_hours = 0.0
    updated_at = record.get("updated_at")
    if updated_at:
        try:
            previous = datetime.fromisoformat(str(updated_at))
            if previous.tzinfo is None:
                previous = previous.replace(tzinfo=UTC)
            elapsed_hours = max(0.0, (now - previous).total_seconds() / 3600.0)
        except (TypeError, ValueError):
            elapsed_hours = 0.0
    wheel = _decay_toward_baseline(wheel, baseline, elapsed_hours, half_life_hours)

    _nudge(wheel, incoming_message_base_emotion, NUDGE_FROM_INCOMING_MESSAGE)
    _nudge(wheel, own_reply_base_emotion, NUDGE_FROM_OWN_REPLY)
    for trigger_emotion in matched_trigger_emotions:
        _nudge(wheel, trigger_emotion, NUDGE_FROM_MATCHED_TRIGGER)

    advanced = {
        "wheel": wheel,
        "baseline_wheel": baseline,
        "updated_at": now.isoformat(),
    }
    advanced["rendered"] = render_current_emotion(advanced)
    return advanced


async def read_current_emotion_record(
    store: Any, creator_id: str, assistant_id: str
) -> dict[str, Any] | None:
    """Fetch the emotional state through the store cache. Best effort."""
    if store is None or not creator_id or not assistant_id:
        return None
    try:
        from src.anubis.utils.store_cache import aget_through_cache

        item = await aget_through_cache(
            store,
            current_emotion_namespace(creator_id, assistant_id),
            CURRENT_RECORD_KEY,
        )
    except Exception as read_error:  # noqa: BLE001 - never cost a turn its prompt
        logger.warning("Could not read the current emotional state: %s", read_error)
        return None
    value = getattr(item, "value", None)
    return value if isinstance(value, dict) else None


async def read_current_emotion_text(
    store: Any, creator_id: str, assistant_id: str
) -> str:
    """The rendered emotional state for the system prompt, or an empty string."""
    record = await read_current_emotion_record(store, creator_id, assistant_id)
    if not record:
        return ""
    return record.get("rendered") or render_current_emotion(record)


async def write_current_emotion_record(
    store: Any, creator_id: str, assistant_id: str, record: Mapping[str, Any]
) -> bool:
    """Persist the emotional state and drop the cached copy."""
    if store is None or not creator_id or not assistant_id:
        return False
    namespace = current_emotion_namespace(creator_id, assistant_id)
    try:
        await store.aput(namespace, key=CURRENT_RECORD_KEY, value=dict(record))
    except Exception as write_error:  # noqa: BLE001 - best effort, never fail a turn
        logger.warning("Could not write the current emotional state: %s", write_error)
        return False
    try:
        from src.anubis.utils.store_cache import invalidate_store_cache_entry

        invalidate_store_cache_entry(namespace, CURRENT_RECORD_KEY)
    except Exception as cache_error:  # noqa: BLE001 - a stale read expires on its own
        logger.warning(
            "Could not invalidate the cached emotional state: %s", cache_error
        )
    return True


__all__ = [
    "GO_EMOTIONS_BASE_TO_PLUTCHIK",
    "PLUTCHIK_PRIMARY_EMOTIONS",
    "advance_current_emotion",
    "build_current_emotion_record",
    "empty_current_emotion_record",
    "neutral_wheel",
    "normalize_wheel",
    "read_current_emotion_record",
    "read_current_emotion_text",
    "render_current_emotion",
    "write_current_emotion_record",
]


async def matched_trigger_emotions(
    store: Any,
    creator_id: str,
    assistant_id: str,
    message_text: str,
    *,
    limit: int = 3,
    minimum_score: float = DEFAULT_TRIGGER_MATCH_MINIMUM_SCORE,
) -> list[str]:
    """Emotions of the stored triggers this message resembles.

    The triggers were written with a GENERALIZED description of what kind of thing
    set the target off, precisely so an unseen message can be matched against them.
    The store index embeds page content, and the trigger description leads the page
    content, so an ordinary similarity search is the match.

    A score floor is applied because every similarity search returns its nearest
    neighbours whether or not they are close: without the floor, an innocuous
    message would always "match" the least-unrelated trigger on file and the
    avatar would react to nothing.
    """
    if store is None or not creator_id or not assistant_id:
        return []
    text = (message_text or "").strip()
    if not text:
        return []
    try:
        items = await store.asearch(
            current_emotion_namespace(creator_id, assistant_id)[:2]
            + ("emotional_trigger",),
            query=text,
            limit=limit,
        )
    except Exception as search_error:  # noqa: BLE001 - best effort, never fail a turn
        logger.warning("Could not search emotional triggers: %s", search_error)
        return []

    emotions: list[str] = []
    for item in items or []:
        score = getattr(item, "score", None)
        if score is not None:
            try:
                if float(score) < minimum_score:
                    continue
            except (TypeError, ValueError):
                pass
        value = getattr(item, "value", None) or {}
        metadata = ((value.get("document") or {}).get("kwargs") or {}).get(
            "metadata"
        ) or {}
        emotion = str(metadata.get("emotion") or "").strip().lower()
        if emotion:
            emotions.append(emotion)
    return emotions


async def refresh_current_emotion(
    store: Any,
    creator_id: str,
    assistant_id: str,
    *,
    incoming_message_text: str = "",
    incoming_message_base_emotion: str | None = None,
    own_reply_base_emotion: str | None = None,
    half_life_hours: float = 6.0,
) -> dict[str, Any] | None:
    """Move the avatar's emotional state by one turn and persist it.

    Called from ``observe_user``, which already runs in parallel with the reply's
    other preparation and already holds the Go Emotions reading of the incoming
    message — so this costs one store read, one similarity search and one write,
    and no model call at all.
    """
    record = await read_current_emotion_record(store, creator_id, assistant_id)
    if record is None:
        # Nothing has been read about this avatar's temperament yet. Starting a
        # state from a blank baseline would make the avatar drift on conversation
        # alone with nothing to return to, so wait for an upload to seed it.
        return None
    triggers = await matched_trigger_emotions(
        store, creator_id, assistant_id, incoming_message_text
    )
    advanced = advance_current_emotion(
        record,
        incoming_message_base_emotion=incoming_message_base_emotion,
        own_reply_base_emotion=own_reply_base_emotion,
        matched_trigger_emotions=triggers,
        half_life_hours=half_life_hours,
    )
    await write_current_emotion_record(store, creator_id, assistant_id, advanced)
    return advanced


__all__ += ["matched_trigger_emotions", "refresh_current_emotion"]
