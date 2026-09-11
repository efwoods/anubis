"""Routing of an ambient observation through the outer message workflow.

The outer workflow is rebuilt here with the real ``resolve_human_message_images``
and ``ambient_triage`` nodes and a stand-in for the avatar (``anubis``) that
appends a reply, so no vision model, classifier, or store is needed.

Invariants: a typed turn with an attachment never enters triage; an ambient
turn keeps its message id and its hidden tag after the images are described;
``ignore`` ends the run with the observation persisted and the avatar untouched;
``respond`` and ``notify`` reach the avatar with the matching instruction; the
client hears the decision through the ``ambient_decision`` stream event and
every vision call through ``image_description_usage``.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

import src.anubis.utils.ambient.triage_node as triage_node_module
import src.anubis.utils.nodes as nodes_module
from src.anubis.utils.ambient.observations import (
    NOTIFY_INSTRUCTION,
    NOTIFY_INSTRUCTION_WITH_OFFER,
    RESPOND_INSTRUCTION,
    build_ambient_additional_kwargs,
)
from src.anubis.utils.ambient.triage import AmbientTriageClassification
from src.anubis.utils.ambient.triage_node import (
    AMBIENT_TRIAGE_NODE,
    ambient_triage,
    route_after_ambient_triage,
    route_after_image_resolution,
)
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.state import GlobalState

IMAGE_BLOCK = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}


class _FakeDescriber:
    prompts: list = []

    def __init__(self, system_prompt=None):
        type(self).prompts.append(system_prompt)

    async def describe(self, image_data, filename):
        return {
            "description": f"described {filename}",
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "total_cost": 0.0001,
            "model_name": "vision-test",
            "latency_ms": 5.0,
        }


@pytest.fixture
def workflow(monkeypatch):
    # The speech cooldown is a process-local singleton, so one test's spoken
    # observation would otherwise silence the next test's.
    from src.anubis.utils.ambient.observations import ambient_speech_cooldown

    ambient_speech_cooldown._last_spoken.clear()
    monkeypatch.setattr(nodes_module, "ImageDescriptionClass", _FakeDescriber)
    _FakeDescriber.prompts = []
    decisions = {"next": "ignore", "calls": [], "salience": 0.95}

    async def fake_classify(context, **kwargs):
        decisions["calls"].append(kwargs)
        return AmbientTriageClassification(
            decision=decisions["next"],
            needs_owner_action=decisions["next"] == "notify",
            observation_kind="writing_code",
            summary="A person writes code.",
            salience=decisions.get("salience", 0.95),
            reason="test",
            proposed_action=decisions.get("proposed_action", "none"),
            action_description=decisions.get("action_description", ""),
        )

    monkeypatch.setattr(triage_node_module, "classify_observation", fake_classify)
    avatar_runs = []

    async def fake_anubis(state):
        avatar_runs.append(state["messages"][-1])
        return {"messages": [AIMessage(content="avatar reply")]}

    builder = StateGraph(GlobalState, context_schema=GlobalContext)
    builder.add_node(
        "resolve_human_message_images", nodes_module.resolve_human_message_images
    )
    builder.add_node(AMBIENT_TRIAGE_NODE, ambient_triage)
    builder.add_node("anubis", fake_anubis)
    builder.add_edge(START, "resolve_human_message_images")
    builder.add_conditional_edges(
        "resolve_human_message_images",
        route_after_image_resolution,
        {AMBIENT_TRIAGE_NODE: AMBIENT_TRIAGE_NODE, "anubis": "anubis"},
    )
    builder.add_conditional_edges(
        AMBIENT_TRIAGE_NODE, route_after_ambient_triage, {END: END, "anubis": "anubis"}
    )
    builder.add_edge("anubis", END)
    app = builder.compile(checkpointer=MemorySaver())
    return app, decisions, avatar_runs


def _ambient_turn(message_id="obs-message", voice_mode=False, camera_facing=None):
    kwargs = build_ambient_additional_kwargs(
        sources=["webcam", "screen"],
        captured_at="2026-09-04T15:00:00Z",
        voice_mode=voice_mode,
        image_filenames=["webcam.jpg", "screen.jpg"],
        observation_id="obs-1",
        camera_facing=camera_facing,
    )
    return HumanMessage(
        id=message_id,
        content=[{"type": "text", "text": ""}, IMAGE_BLOCK, IMAGE_BLOCK],
        additional_kwargs=kwargs,
    )


def _input(message):
    return {
        "messages": [message],
        "user_state": {"user_id": "u1"},
        "assistant_state": {"assistant_id": "a1", "assistant_name": "Ada"},
    }


async def _run(app, message, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    events = []
    async for mode, payload in app.astream(
        _input(message),
        config,
        context=GlobalContext(),
        stream_mode=["custom", "updates"],
    ):
        events.append((mode, payload))
    state = await app.aget_state(config)
    custom = [payload for mode, payload in events if mode == "custom"]
    return state.values["messages"], custom


@pytest.mark.asyncio
async def test_an_ignored_observation_is_persisted_hidden_and_ends_the_run(workflow):
    app, decisions, avatar_runs = workflow
    decisions["next"] = "ignore"
    messages, custom = await _run(app, _ambient_turn(), "ignore-thread")

    assert avatar_runs == []
    assert len(messages) == 1
    stored = messages[0]
    assert isinstance(stored, HumanMessage)
    assert stored.id == "obs-message"
    assert stored.additional_kwargs["hidden"] is True
    assert stored.additional_kwargs["ambient"]["decision"] == "ignore"
    assert stored.additional_kwargs["ambient"]["summary"] == "A person writes code."
    assert "image_filenames" not in stored.additional_kwargs
    assert stored.content.startswith("[AMBIENT_OBSERVATION id=obs-1")
    assert "decision=ignore]" in stored.content.splitlines()[0]
    assert "webcam: described webcam.jpg" in stored.content
    assert "screen: described screen.jpg" in stored.content
    assert not stored.content.endswith(RESPOND_INSTRUCTION)
    assert (
        _FakeDescriber.prompts
        and "describe_ambient_image_spec" in _FakeDescriber.prompts[0]
    )

    kinds = [payload["type"] for payload in custom]
    assert kinds.count("image_description_usage") == 2
    assert kinds[-1] == "ambient_decision"
    decision = custom[-1]
    assert decision["decision"] == "ignore" and decision["observation_id"] == "obs-1"


@pytest.mark.asyncio
async def test_a_respond_decision_reaches_the_avatar_with_the_instruction(workflow):
    app, decisions, avatar_runs = workflow
    decisions["next"] = "respond"
    messages, _custom = await _run(
        app, _ambient_turn(voice_mode=True), "respond-thread"
    )

    assert len(avatar_runs) == 1
    # The instruction is followed by the reason the triage chose to speak,
    # which is the specific thing the avatar is told to react to.
    assert RESPOND_INSTRUCTION in avatar_runs[0].content
    assert avatar_runs[0].content.endswith("[AMBIENT_REASON] test")
    assert messages[0].additional_kwargs["ambient"]["decision"] == "respond"
    assert messages[0].additional_kwargs["ambient"]["voice_mode"] is True
    assert (
        isinstance(messages[-1], AIMessage) and messages[-1].content == "avatar reply"
    )
    assert decisions["calls"][0]["voice_mode"] is True
    assert decisions["calls"][0]["assistant_name"] == "Ada"


@pytest.mark.asyncio
async def test_a_notify_decision_asks_for_a_heads_up(workflow):
    app, decisions, avatar_runs = workflow
    decisions["next"] = "notify"
    messages, _custom = await _run(app, _ambient_turn(), "notify-thread")
    assert avatar_runs[0].content.endswith(NOTIFY_INSTRUCTION)
    assert messages[0].additional_kwargs["ambient"]["needs_owner_action"] is True


@pytest.mark.asyncio
async def test_a_notify_with_an_offer_ends_the_heads_up_by_offering(workflow):
    app, decisions, avatar_runs = workflow
    decisions["next"] = "notify"
    decisions["proposed_action"] = "research"
    decisions["action_description"] = "Research the error on the screen"
    messages, custom = await _run(app, _ambient_turn(), "offer-thread")
    assert avatar_runs[0].content.endswith(NOTIFY_INSTRUCTION_WITH_OFFER)
    assert "[AMBIENT_OFFER] Research the error on the screen" in avatar_runs[0].content
    ambient = messages[0].additional_kwargs["ambient"]
    assert ambient["proposed_action"] == "research"
    assert ambient["action_description"] == "Research the error on the screen"
    decision_events = [
        event for event in custom if event.get("type") == "ambient_decision"
    ]
    assert decision_events[0]["proposed_action"] == "research"


@pytest.mark.asyncio
async def test_a_typed_turn_with_an_attachment_skips_triage(workflow):
    app, decisions, avatar_runs = workflow
    typed = HumanMessage(
        id="typed",
        content=[{"type": "text", "text": "what is this?"}, IMAGE_BLOCK],
        additional_kwargs={"image_filenames": ["photo.jpg"]},
    )
    messages, custom = await _run(app, typed, "typed-thread")
    assert decisions["calls"] == []
    assert len(avatar_runs) == 1
    assert "Image descriptions:" in messages[0].content
    assert "[photo.jpg]" in messages[0].content
    assert messages[0].id == "typed"
    assert "hidden" not in messages[0].additional_kwargs
    assert [payload["type"] for payload in custom] == ["image_description_usage"]
    assert _FakeDescriber.prompts[-1] is None


@pytest.mark.asyncio
async def test_earlier_observations_are_handed_to_the_classifier(workflow):
    app, decisions, _avatar_runs = workflow
    decisions["next"] = "ignore"
    await _run(app, _ambient_turn("first"), "history-thread")
    config = {"configurable": {"thread_id": "history-thread"}}
    async for _ in app.astream(
        {"messages": [_ambient_turn("second")]}, config, context=GlobalContext()
    ):
        pass
    second_call = decisions["calls"][-1]
    assert [
        item["observation_id"] for item in second_call["previous_observations"]
    ] == ["obs-1"]
    state = await app.aget_state(config)
    assert [message.id for message in state.values["messages"]] == ["first", "second"]


@pytest.mark.asyncio
async def test_a_low_salience_respond_is_demoted_and_the_avatar_stays_quiet(workflow):
    """A remark not worth making does not reach the conversation partner.

    This is the case that produced a run of unprompted messages on a quiet
    webcam: the classifier judges one observation at a time, so a low-stakes
    'respond' repeated often enough becomes a stream of interruptions.
    """
    app, decisions, avatar_runs = workflow
    decisions["next"] = "respond"
    decisions["salience"] = 0.10
    messages, custom = await _run(app, _ambient_turn(), "low-salience-thread")

    assert avatar_runs == []
    ambient = messages[0].additional_kwargs["ambient"]
    assert ambient["decision"] == "ignore"
    assert ambient["demoted_from"] == "respond"
    assert "below the respond floor" in ambient["demotion_reason"]
    # The observation is still kept as context even though nothing was said.
    assert messages[0].content.startswith("[AMBIENT_OBSERVATION")
    assert not any(isinstance(message, AIMessage) for message in messages)
    decision_frames = [
        frame for frame in custom if frame.get("type") == "ambient_decision"
    ]
    assert decision_frames and decision_frames[-1]["decision"] == "ignore"


@pytest.mark.asyncio
async def test_a_notify_needs_more_salience_than_a_respond(workflow):
    """A card interrupts more than a reply, so the notify floor sits higher."""
    app, decisions, avatar_runs = workflow
    decisions["next"] = "notify"
    # Above the respond floor of 0.55, below the notify floor of 0.70.
    decisions["salience"] = 0.60
    messages, _custom = await _run(app, _ambient_turn(), "notify-floor-thread")

    assert avatar_runs == []
    ambient = messages[0].additional_kwargs["ambient"]
    assert ambient["decision"] == "ignore"
    assert ambient["demoted_from"] == "notify"
    # A demoted observation offers nothing, because no card is ever shown.
    assert ambient["proposed_action"] == "none"


@pytest.mark.asyncio
async def test_the_avatar_does_not_speak_twice_inside_the_quiet_period(workflow):
    """Having just spoken, the avatar stays quiet about the next observation."""
    app, decisions, avatar_runs = workflow
    decisions["next"] = "respond"
    # Above every floor, but below the cooldown override of 0.90.
    decisions["salience"] = 0.80

    first_messages, _first = await _run(
        app, _ambient_turn(message_id="obs-first"), "cooldown-thread"
    )
    assert len(avatar_runs) == 1
    assert first_messages[0].additional_kwargs["ambient"]["decision"] == "respond"

    second_messages, _second = await _run(
        app, _ambient_turn(message_id="obs-second"), "cooldown-thread"
    )
    # Still one avatar run: the second observation was noticed silently.
    assert len(avatar_runs) == 1
    second_ambient = second_messages[-1].additional_kwargs["ambient"]
    assert second_ambient["decision"] == "ignore"
    assert "the avatar spoke recently" in second_ambient["demotion_reason"]


@pytest.mark.asyncio
async def test_a_salient_enough_observation_overrides_the_quiet_period(workflow):
    """Something urgent is not silenced by a cooldown an ordinary remark began."""
    app, decisions, avatar_runs = workflow
    decisions["next"] = "respond"
    decisions["salience"] = 0.80
    await _run(app, _ambient_turn(message_id="obs-first"), "override-thread")
    assert len(avatar_runs) == 1

    decisions["salience"] = 0.95
    messages, _custom = await _run(
        app, _ambient_turn(message_id="obs-second"), "override-thread"
    )
    assert len(avatar_runs) == 2
    observations = [
        message for message in messages if "ambient" in message.additional_kwargs
    ]
    assert observations[-1].additional_kwargs["ambient"]["decision"] == "respond"


@pytest.mark.asyncio
async def test_the_classifier_is_told_which_way_the_camera_points(workflow):
    """A rear camera reads as the person's own view of the world."""
    app, decisions, _avatar_runs = workflow
    decisions["next"] = "ignore"
    await _run(
        app,
        _ambient_turn(camera_facing="environment"),
        "facing-thread",
    )
    assert decisions["calls"][0]["camera_facing"] == "world"
