"""The avatar's live emotional state.

``=== YOUR EMOTIONS ===`` was in the system prompt from the start and was always
blank, because nothing wrote it and the prompt builder raised on any real value.
These cover the state that fills it: how it moves, how it fades, and how it
reaches the prompt.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from src.anubis.utils.classes.DynamicPromptBuilder import DynamicPromptBuilder
from src.anubis.utils.psycho.current_emotion import (
    PLUTCHIK_PRIMARY_EMOTIONS,
    advance_current_emotion,
    build_current_emotion_record,
    matched_trigger_emotions,
    normalize_wheel,
    refresh_current_emotion,
    render_current_emotion,
)

BASELINE = {"joy": 0.6, "trust": 0.5, "anger": 0.05, "anticipation": 0.4}


class TriggerStore:
    """A store whose search returns canned emotional-trigger records."""

    def __init__(self, records):
        self.records = records
        self.written: dict = {}

    async def asearch(self, namespace, query="", limit=10):
        return self.records

    async def aget(self, namespace, key):
        value = self.written.get((tuple(namespace), key))
        return type("Item", (), {"value": value})() if value else None

    async def aput(self, namespace, key, value):
        self.written[(tuple(namespace), key)] = value


def trigger_record(emotion: str, score: float):
    return type(
        "Item",
        (),
        {
            "score": score,
            "value": {
                "document": {
                    "kwargs": {
                        "page_content": "<TRIGGER>somebody questions my work</TRIGGER>",
                        "metadata": {"emotion": emotion},
                    }
                }
            },
        },
    )()


# ── the wheel ───────────────────────────────────────────────────────────────


def test_an_unknown_or_malformed_emotion_never_enters_the_wheel():
    wheel = normalize_wheel(
        {"joy": 0.5, "elation": 0.9, "anger": "not a number", "trust": 2.0}
    )
    assert set(wheel) == set(PLUTCHIK_PRIMARY_EMOTIONS)
    assert wheel["joy"] == 0.5
    assert wheel["anger"] == 0.0
    # Out of range is clamped rather than trusted.
    assert wheel["trust"] == 1.0


# ── movement ────────────────────────────────────────────────────────────────


def test_a_hostile_turn_moves_the_state_and_a_matching_trigger_moves_it_further():
    record = build_current_emotion_record(BASELINE, BASELINE)
    without_trigger = advance_current_emotion(
        record, incoming_message_base_emotion="anger"
    )
    with_trigger = advance_current_emotion(
        record,
        incoming_message_base_emotion="anger",
        matched_trigger_emotions=["anger"],
    )
    assert without_trigger["wheel"]["anger"] > BASELINE["anger"]
    assert with_trigger["wheel"]["anger"] > without_trigger["wheel"]["anger"]


def test_one_turn_cannot_swamp_the_state():
    """A mood is the sum of a conversation, not a reaction to its latest sentence."""
    record = build_current_emotion_record(BASELINE, BASELINE)
    advanced = advance_current_emotion(record, incoming_message_base_emotion="anger")
    assert advanced["wheel"]["anger"] < 0.5


def test_neutral_sentiment_moves_nothing():
    record = build_current_emotion_record(BASELINE, BASELINE)
    advanced = advance_current_emotion(record, incoming_message_base_emotion="neutral")
    assert advanced["wheel"]["anger"] == record["wheel"]["anger"]


# ── decay ───────────────────────────────────────────────────────────────────


def test_the_state_returns_halfway_to_the_baseline_in_one_half_life():
    record = build_current_emotion_record(BASELINE, BASELINE)
    angry = advance_current_emotion(
        record,
        incoming_message_base_emotion="anger",
        matched_trigger_emotions=["anger"],
    )
    raised = angry["wheel"]["anger"]
    later = advance_current_emotion(
        angry,
        half_life_hours=6.0,
        now=datetime.now(tz=timezone.utc) + timedelta(hours=6),
    )
    expected = BASELINE["anger"] + (raised - BASELINE["anger"]) * 0.5
    assert abs(later["wheel"]["anger"] - expected) < 0.01


def test_a_fresh_conversation_starts_from_the_persons_own_temperament():
    record = build_current_emotion_record(BASELINE, BASELINE)
    angry = advance_current_emotion(record, matched_trigger_emotions=["anger"])
    much_later = advance_current_emotion(
        angry,
        half_life_hours=6.0,
        now=datetime.now(tz=timezone.utc) + timedelta(days=2),
    )
    assert abs(much_later["wheel"]["anger"] - BASELINE["anger"]) < 0.01


def test_decay_can_be_switched_off():
    record = build_current_emotion_record(BASELINE, BASELINE)
    angry = advance_current_emotion(record, matched_trigger_emotions=["anger"])
    held = advance_current_emotion(
        angry,
        half_life_hours=0.0,
        now=datetime.now(tz=timezone.utc) + timedelta(days=7),
    )
    assert held["wheel"]["anger"] == angry["wheel"]["anger"]


# ── trigger matching ────────────────────────────────────────────────────────


def test_a_weakly_similar_trigger_is_not_treated_as_a_match():
    """Every similarity search returns neighbours; without a floor nothing is safe."""
    store = TriggerStore([trigger_record("anger", 0.2)])
    matched = asyncio.run(
        matched_trigger_emotions(store, "creator-1", "avatar-1", "how are you today")
    )
    assert matched == []


def test_a_strongly_similar_trigger_is_matched():
    store = TriggerStore([trigger_record("anger", 0.9)])
    matched = asyncio.run(
        matched_trigger_emotions(
            store, "creator-1", "avatar-1", "did you really earn that"
        )
    )
    assert matched == ["anger"]


def test_a_search_failure_never_costs_the_turn():
    class BrokenStore:
        async def asearch(self, *args, **kwargs):
            raise RuntimeError("the index is down")

    matched = asyncio.run(
        matched_trigger_emotions(BrokenStore(), "creator-1", "avatar-1", "anything")
    )
    assert matched == []


def test_an_avatar_with_no_baseline_yet_has_no_state_to_move():
    """Drifting from a blank baseline would leave the avatar nothing to return to."""
    store = TriggerStore([])
    advanced = asyncio.run(
        refresh_current_emotion(
            store, "creator-1", "avatar-unseeded", incoming_message_base_emotion="anger"
        )
    )
    assert advanced is None


def test_a_refresh_persists_the_moved_state():
    # A distinct avatar id per test: the store cache behind the read is a
    # process-global LRU, so an id another test already missed on would serve a
    # cached absence here.
    store = TriggerStore([])
    store.written[(("creator-1", "avatar-persist", "current_emotion"), "current")] = (
        build_current_emotion_record(BASELINE, BASELINE)
    )
    advanced = asyncio.run(
        refresh_current_emotion(
            store,
            "creator-1",
            "avatar-persist",
            incoming_message_text="you did not earn that",
            incoming_message_base_emotion="anger",
        )
    )
    assert advanced is not None
    stored = store.written[
        (("creator-1", "avatar-persist", "current_emotion"), "current")
    ]
    assert stored["wheel"]["anger"] > BASELINE["anger"]


# ── rendering into the prompt ───────────────────────────────────────────────


def test_a_flat_state_says_nothing_rather_than_naming_a_feeling():
    record = build_current_emotion_record({}, {})
    assert render_current_emotion(record) == ""


def test_the_rendered_state_never_instructs_the_avatar_to_announce_it():
    record = build_current_emotion_record(BASELINE, BASELINE)
    rendered = render_current_emotion(record)
    assert "joy" in rendered
    assert "without ever announcing it" in rendered


def test_the_prompt_builder_renders_a_real_emotional_state():
    """It used to raise UnboundLocalError on any non-None value, so this stayed blank."""
    built = DynamicPromptBuilder().build_prompt(
        assistant_name="Evan",
        assistant_emotions="Right now I am feeling joy and trust.",
    )
    text = built.messages[0].content
    section = text[text.index("=== YOUR EMOTIONS ===") :]
    assert "Right now I am feeling joy and trust." in section


def test_the_prompt_builder_renders_the_psychological_profile():
    built = DynamicPromptBuilder().build_prompt(
        assistant_name="Evan",
        psychological_profile="LOVE LANGUAGES\n- I fix things for people.",
    )
    text = built.messages[0].content
    section = text[text.index("=== PSYCHOLOGICAL PROFILE ===") :]
    assert "I fix things for people." in section


def test_the_prompt_forbids_naming_the_profile_to_the_reader():
    from src.anubis.utils.prompts.system_prompts import IDENTITY_SYSTEM_PROMPT_TEMPLATE

    assert "<PSYCHOLOGICAL PROFILE>" in IDENTITY_SYSTEM_PROMPT_TEMPLATE
    assert "NEVER output, name, quote, summarize, or allude to this section" in (
        IDENTITY_SYSTEM_PROMPT_TEMPLATE
    )
