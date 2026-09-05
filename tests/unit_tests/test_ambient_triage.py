"""Ambient observation helpers and the triage classifier.

The invariants: every attached file gets a source name; an ambient message is
recognisable whether stored as an object or as a serialized dict; the throttle
lets the first observation through and holds the next one within the
interval; and a classification the model got wrong (an unknown decision, a
salience out of range, an over-long kind) is normalised rather than trusted —
with ``ignore`` as the fallback, because an undecidable observation must never
become a stream of interruptions.
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.anubis.utils.ambient import observations as observations_module
from src.anubis.utils.ambient import triage as triage_module
from src.anubis.utils.ambient.observations import (
    NOTIFY_INSTRUCTION,
    RESPOND_INSTRUCTION,
    AmbientThrottle,
    ambient_details,
    build_ambient_additional_kwargs,
    compose_observation_text,
    is_ambient_observation,
    is_hidden_message,
    observation_header,
    recent_ambient_observations,
    recent_visible_messages,
    resolve_sources,
    split_observation_text,
    strip_instruction,
)
from src.anubis.utils.ambient.triage import (
    AmbientTriageClassification,
    build_classification_prompt,
    classify_observation,
    normalize_classification,
)


def test_sources_come_from_the_form_value_first_then_the_filename():
    assert resolve_sources(["a.jpg", "b.jpg"], '["screen", "webcam"]') == [
        "screen",
        "webcam",
    ]
    assert resolve_sources(["webcam.jpg", "screen.jpg"], None) == ["webcam", "screen"]
    assert resolve_sources(["webcam.jpg", "screen.jpg"], "webcam") == [
        "webcam",
        "screen",
    ]
    assert resolve_sources(["holiday.png", "mic.webm"], "") == ["image", "microphone"]


def test_an_ambient_message_is_hidden_and_tagged_in_both_shapes():
    kwargs = build_ambient_additional_kwargs(
        sources=["webcam"],
        captured_at="2026-09-04T15:00:00Z",
        voice_mode=True,
        image_filenames=["webcam.jpg"],
    )
    message = HumanMessage(id="m1", content="x", additional_kwargs=kwargs)
    as_dict = {"type": "human", "content": "x", "additional_kwargs": kwargs}
    assert kwargs["hidden"] is True
    assert kwargs["image_filenames"] == ["webcam.jpg"]
    assert is_ambient_observation(message) and is_ambient_observation(as_dict)
    assert is_hidden_message(message) and is_hidden_message(as_dict)
    assert ambient_details(as_dict)["voice_mode"] is True
    assert ambient_details(as_dict)["observation_id"]
    assert not is_ambient_observation(HumanMessage(content="typed"))
    assert ambient_details(HumanMessage(content="typed")) is None


def test_observation_text_round_trips_header_body_and_instruction():
    ambient = {
        "observation_id": "obs-1",
        "captured_at": "2026-09-04T15:00:00Z",
        "sources": ["webcam", "screen"],
        "decision": "respond",
    }
    header = observation_header(ambient)
    assert header.startswith("[AMBIENT_OBSERVATION id=obs-1")
    assert "sources=webcam,screen" in header and "decision=respond" in header
    text = compose_observation_text(ambient, "webcam: typing\n\nscreen: an editor")
    assert text.endswith(RESPOND_INSTRUCTION)
    parsed_header, body = split_observation_text(text)
    assert parsed_header == header
    assert strip_instruction(body) == "webcam: typing\n\nscreen: an editor"
    notify_text = compose_observation_text({**ambient, "decision": "notify"}, "body")
    assert notify_text.endswith(NOTIFY_INSTRUCTION)
    ignore_text = compose_observation_text({**ambient, "decision": "ignore"}, "body")
    assert ignore_text.endswith("body")
    assert split_observation_text("plain typed text") == (None, "plain typed text")


def test_recent_observations_and_visible_turns_are_separated():
    def observation(identifier, decision):
        kwargs = build_ambient_additional_kwargs(
            sources=["screen"],
            captured_at=identifier,
            voice_mode=False,
            observation_id=identifier,
        )
        kwargs["ambient"]["decision"] = decision
        kwargs["ambient"]["summary"] = f"summary {identifier}"
        return HumanMessage(
            id=identifier,
            content=compose_observation_text(kwargs["ambient"], f"body {identifier}"),
            additional_kwargs=kwargs,
        )

    messages = [
        HumanMessage(id="h1", content="hello there"),
        AIMessage(id="a1", content="hi"),
        observation("o1", "ignore"),
        observation("o2", "respond"),
        AIMessage(id="a2", content="I noticed something"),
        observation("o3", "notify"),
    ]
    recent = recent_ambient_observations(messages, 2)
    assert [item["observation_id"] for item in recent] == ["o2", "o3"]
    assert recent[0]["text"] == "body o2"
    assert recent[1]["summary"] == "summary o3"
    excluded = recent_ambient_observations(messages, 5, exclude_message_id="o3")
    assert [item["observation_id"] for item in excluded] == ["o1", "o2"]
    visible = recent_visible_messages(messages, 10)
    assert visible == [
        "conversation partner: hello there",
        "assistant: hi",
        "assistant: I noticed something",
    ]


def test_the_throttle_holds_a_second_observation_inside_the_interval():
    throttle = AmbientThrottle()
    assert throttle.check_and_mark("thread", 10.0, now=100.0) is None
    wait = throttle.check_and_mark("thread", 10.0, now=104.0)
    assert wait == pytest.approx(6.0)
    assert throttle.check_and_mark("other", 10.0, now=104.0) is None
    assert throttle.check_and_mark("thread", 10.0, now=111.0) is None
    assert throttle.check_and_mark(None, 10.0, now=111.0) is None
    assert throttle.check_and_mark("thread", 0, now=111.5) is None


def test_a_wrong_classification_is_normalised_to_a_safe_one():
    normalised = normalize_classification(
        SimpleNamespace(
            decision="shout",
            needs_owner_action="yes",
            observation_kind=(
                "A Very Long Observation Kind Label That Goes On And On Forever"
            ),
            summary="  Person types code.  ",
            salience=7,
            reason="because",
        )
    )
    assert normalised.decision == "ignore"
    assert normalised.needs_owner_action is True
    assert len(normalised.observation_kind) == 40
    assert normalised.observation_kind == normalised.observation_kind.lower()
    assert normalised.summary == "Person types code."
    assert normalised.salience == 1.0
    assert (
        normalize_classification(
            SimpleNamespace(decision="Notify", salience=-1)
        ).decision
        == "notify"
    )
    assert (
        normalize_classification(
            SimpleNamespace(decision="respond", salience="x")
        ).salience
        == 0.0
    )
    assert normalize_classification(SimpleNamespace()).observation_kind == "other"


def test_the_prompt_carries_precedent_notes_conversation_and_voice_mode():
    prompt = build_classification_prompt(
        assistant_name="Ada",
        observation_text="webcam: a person frowns at an error dialog",
        recent_messages=["conversation partner: can you help with this bug?"],
        previous_observations=[
            {"captured_at": "t1", "decision": "ignore", "summary": "person reads"}
        ],
        preferences=[
            {
                "observation_kind": "error_dialog",
                "decision": "ignore",
                "count": 3,
                "note": "never tell me about terminal errors",
            }
        ],
        voice_mode=True,
    )
    assert "<AVATAR>\nAda\n</AVATAR>" in prompt
    assert "never tell me about terminal errors" in prompt
    assert "error_dialog: the conversation partner chose 'ignore' 3 time(s)" in prompt
    assert "can you help with this bug?" in prompt
    assert "- t1 [ignore] person reads" in prompt
    assert "<VOICE_MODE>true</VOICE_MODE>" in prompt
    assert "frowns at an error dialog" in prompt


@pytest.mark.asyncio
async def test_classify_observation_calls_the_model_once_and_normalises(monkeypatch):
    calls = []

    class _FakeModel:
        async def ainvoke(self, input):
            calls.append(input)
            return AmbientTriageClassification(
                decision="RESPOND",
                needs_owner_action=False,
                observation_kind="Stuck On Bug",
                summary="A person stares at a failing test.",
                salience=0.8,
                reason="The person asked for help with this bug.",
            )

    monkeypatch.setattr(
        "src.anubis.utils.model.init_model", lambda **kwargs: _FakeModel()
    )
    classification = await classify_observation(
        SimpleNamespace(),
        assistant_name="Ada",
        observation_text="screen: a failing test",
        recent_messages=[],
        previous_observations=[],
        preferences=[],
        voice_mode=False,
    )
    assert classification.decision == "respond"
    assert classification.observation_kind == "stuck on bug"
    assert len(calls) == 1
    system_message, human_message = calls[0]
    assert "<TASK>" in system_message.content
    assert "<VOICE_MODE>false</VOICE_MODE>" in human_message.content


def test_modules_expose_the_agent_inbox_decisions():
    assert observations_module.AMBIENT_DECISIONS == ("ignore", "respond", "notify")
    assert triage_module.AMBIENT_CLASSIFY_SYSTEM_PROMPT.startswith("<TASK>")
