"""A spoken turn shows what the room said, not the avatar's internal framing.

Observations are written with an ``[AMBIENT_OBSERVATION id=... ]`` first line so
the avatar can tell a scene it noticed from a turn the person typed. A spoken
turn is deliberately left visible — the owner should see what was heard — but on
2026-09-07 a reload showed the bare header attributed to the owner:

    You
    [AMBIENT_OBSERVATION id=... sources=microphone decision=ignore]
    (nothing intelligible was heard)

Two things were wrong: a turn where nothing intelligible was heard has nothing
to show and should not be visible at all, and a turn that IS shown must not
carry the model's framing into the transcript.
"""

from __future__ import annotations

from src.anubis.utils.ambient.observations import (
    build_ambient_additional_kwargs,
    is_hidden_message,
    observation_header,
)
from src.api.webapp import _message_without_observation_header

HEARD = "Speaker 1: are you coming?"


def _observation_message(body: str) -> dict:
    ambient = {
        "observation_id": "obs-1",
        "sources": ["microphone"],
        "captured_at": "2026-09-07T04:23:28+00:00",
        "voice_mode": True,
    }
    kwargs = build_ambient_additional_kwargs(
        sources=["microphone"], captured_at=ambient["captured_at"], voice_mode=True
    )
    return {
        "type": "human",
        "content": f"{observation_header(ambient)}\n{body}",
        "additional_kwargs": kwargs,
    }


def test_the_internal_header_never_reaches_the_transcript():
    shown = _message_without_observation_header(_observation_message(HEARD))
    assert shown["content"] == HEARD
    assert "AMBIENT_OBSERVATION" not in shown["content"]
    # Everything else about the message is untouched.
    assert shown["additional_kwargs"]["kind"] == "ambient_observation"


def test_a_turn_the_person_typed_is_returned_unchanged():
    typed = {"type": "human", "content": "hey kiddo", "additional_kwargs": {}}
    assert _message_without_observation_header(typed) is typed


def test_a_message_without_a_header_is_left_alone():
    message = _observation_message(HEARD)
    message["content"] = HEARD
    assert _message_without_observation_header(message)["content"] == HEARD


def test_a_spoken_turn_is_shown_and_an_empty_one_is_hidden():
    """``hidden`` is what decides whether the reader ever sees the turn."""
    heard = build_ambient_additional_kwargs(
        sources=["microphone"], captured_at="", voice_mode=True, hidden=False
    )
    assert is_hidden_message({"additional_kwargs": heard}) is False

    nothing_heard = build_ambient_additional_kwargs(
        sources=["microphone"], captured_at="", voice_mode=True, hidden=True
    )
    assert is_hidden_message({"additional_kwargs": nothing_heard}) is True
