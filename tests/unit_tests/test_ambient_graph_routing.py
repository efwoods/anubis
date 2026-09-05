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
    monkeypatch.setattr(nodes_module, "ImageDescriptionClass", _FakeDescriber)
    _FakeDescriber.prompts = []
    decisions = {"next": "ignore", "calls": []}

    async def fake_classify(context, **kwargs):
        decisions["calls"].append(kwargs)
        return AmbientTriageClassification(
            decision=decisions["next"],
            needs_owner_action=decisions["next"] == "notify",
            observation_kind="writing_code",
            summary="A person writes code.",
            salience=0.4,
            reason="test",
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


def _ambient_turn(message_id="obs-message", voice_mode=False):
    kwargs = build_ambient_additional_kwargs(
        sources=["webcam", "screen"],
        captured_at="2026-09-04T15:00:00Z",
        voice_mode=voice_mode,
        image_filenames=["webcam.jpg", "screen.jpg"],
        observation_id="obs-1",
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
    assert avatar_runs[0].content.endswith(RESPOND_INSTRUCTION)
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
