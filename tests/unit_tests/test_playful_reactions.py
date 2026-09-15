"""Playful camera performances: detect a gag, alternate the face, short cooldown."""

from langchain_core.messages import AIMessage

from src.anubis.utils.ambient.observations import (
    RESPOND_INSTRUCTION,
    compose_observation_text,
)
from src.anubis.utils.ambient.playful_reactions import (
    DEFAULT_GROSS_REACTION_LINE,
    PLAYFUL_RESPOND_SUFFIX,
    apply_playful_reaction_sentiment,
    is_playful_camera_performance,
    is_playful_typed_performance,
    next_playful_reaction_emotion,
    reset_playful_reaction_alternator,
)
from src.anubis.utils.ambient.triage_node import _gate_by_salience_and_cooldown
from src.anubis.utils.context import GlobalContext


def setup_function() -> None:
    reset_playful_reaction_alternator()


def test_a_tongue_or_arm_gag_is_a_performance_and_desk_presence_is_not():
    assert is_playful_camera_performance(
        observation_kind="playful_performance",
        summary="A person sticks out a tongue.",
    )
    assert is_playful_camera_performance(
        summary="The person is sucking on an arm and showing nostrils."
    )
    assert is_playful_typed_performance(
        "I make a gross face, stick out my tongue, and show my nostrils"
    )
    assert not is_playful_camera_performance(
        observation_kind="writing_code",
        summary="A person types at a desk with headphones on.",
    )
    assert not is_playful_typed_performance("What do you see on the webcam?")


def test_consecutive_reactions_on_one_thread_alternate_surprise_and_disgust():
    assert next_playful_reaction_emotion("thread-a") == "surprise"
    assert next_playful_reaction_emotion("thread-a") == "disgust"
    assert next_playful_reaction_emotion("thread-a") == "surprise"
    assert next_playful_reaction_emotion("thread-b") == "surprise"


def test_a_playful_reply_overwrites_go_emotions_with_the_alternating_face():
    first = AIMessage(content=DEFAULT_GROSS_REACTION_LINE)
    first.response_metadata = {
        "sentiment": {"emotion": "anger", "base_emotion": "anger", "score": 0.9}
    }
    applied = apply_playful_reaction_sentiment(
        first,
        ambient={
            "decision": "respond",
            "observation_kind": "playful_performance",
            "summary": "A person sticks out a tongue.",
        },
        typed_text="",
        thread_id="alt-1",
    )
    assert applied == "surprise"
    assert first.response_metadata["sentiment"]["base_emotion"] == "surprise"

    second = AIMessage(content=DEFAULT_GROSS_REACTION_LINE)
    apply_playful_reaction_sentiment(
        second,
        ambient=None,
        typed_text="I laugh and continue to suck on my arm",
        thread_id="alt-1",
    )
    assert second.response_metadata["sentiment"]["base_emotion"] == "disgust"


def test_an_ordinary_reply_keeps_the_classified_face():
    reply = AIMessage(content="The shelves are still there.")
    reply.response_metadata = {
        "sentiment": {"emotion": "neutral", "base_emotion": "neutral", "score": 0.4}
    }
    applied = apply_playful_reaction_sentiment(
        reply,
        ambient={
            "decision": "respond",
            "observation_kind": "writing_code",
            "summary": "A person types.",
        },
        typed_text="Hmm",
        thread_id="desk",
    )
    assert applied is None
    assert reply.response_metadata["sentiment"]["base_emotion"] == "neutral"


def test_a_playful_respond_instruction_names_the_taught_line():
    text = compose_observation_text(
        {
            "observation_id": "obs-gag",
            "captured_at": "2026-09-12T16:00:00Z",
            "sources": ["webcam"],
            "decision": "respond",
            "observation_kind": "playful_performance",
            "summary": "A person sticks out a tongue.",
            "reason": "The person is making a gross face at the camera.",
        },
        "webcam: tongue out, nostrils shown",
    )
    assert RESPOND_INSTRUCTION in text
    assert PLAYFUL_RESPOND_SUFFIX.strip() in text
    assert DEFAULT_GROSS_REACTION_LINE in text


def test_a_playful_gag_is_not_silenced_by_the_five_minute_quiet_period():
    from src.anubis.utils.ambient.observations import ambient_speech_cooldown

    ambient_speech_cooldown._last_spoken.clear()
    context = GlobalContext()
    context.ambient_respond_cooldown_seconds = 300.0
    context.ambient_playful_respond_cooldown_seconds = 6.0
    context.ambient_respond_cooldown_override_salience = 0.90
    context.ambient_respond_salience_floor = 0.55

    ambient_speech_cooldown.mark_spoken("gag-thread", now=1_000.0)

    desk_decision, desk_reason = _gate_by_salience_and_cooldown(
        decision="respond",
        salience=0.80,
        thread_id="gag-thread",
        context=context,
        observation_kind="writing_code",
        summary="A person types at a desk.",
        now=1_008.0,
    )
    assert desk_decision == "ignore"
    assert desk_reason is not None
    assert "the avatar spoke recently" in desk_reason

    gag_decision, gag_reason = _gate_by_salience_and_cooldown(
        decision="respond",
        salience=0.80,
        thread_id="gag-thread",
        context=context,
        observation_kind="playful_performance",
        summary="A person sticks out a tongue.",
        now=1_008.0,
    )
    assert gag_decision == "respond"
    assert gag_reason is None
