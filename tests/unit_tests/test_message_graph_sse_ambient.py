"""``message_graph_sse`` on an ambient turn: the decision frame and the meters.

A fake graph plays the stream a triaged observation produces. For an ignored
observation there is no reply: the client gets ``ambient_decision`` and then
``done`` with empty content and the decision repeated, the image descriptions
are metered, and the messaging meter is left alone. For a reply the frames are
the ordinary ones with the decision in front.
"""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.api import webapp as webapp_module

DECISION = {
    "type": "ambient_decision",
    "observation_id": "obs-1",
    "decision": "ignore",
    "summary": "A person writes code.",
    "reason": "routine",
}
IMAGE_USAGE = {
    "type": "image_description_usage",
    "source": "webcam",
    "input_tokens": 100,
    "output_tokens": 20,
    "total_tokens": 120,
    "total_cost": 0.0001,
    "latency_ms": 4.0,
    "model_name": "vision-test",
}


class _FakeGraph:
    def __init__(self, events):
        self._events = events

    async def astream(self, input, config, context, stream_mode, subgraphs):
        for event in self._events:
            yield event

    async def aget_state(self, config):
        return SimpleNamespace(next=(), tasks=(), interrupts=())


@pytest.fixture
def harness(monkeypatch):
    updates = []
    image_meterings = []
    message_meterings = []

    class _Threads:
        async def update(self, thread_id, metadata):
            updates.append((thread_id, metadata))

    monkeypatch.setattr(
        webapp_module,
        "get_client",
        lambda **kwargs: SimpleNamespace(threads=_Threads()),
    )

    async def fake_image_meter(app_state, current_user, payload, **kwargs):
        image_meterings.append((payload, kwargs))

    async def fake_message_meter(**kwargs):
        message_meterings.append(kwargs)
        return {"tokens": 1}

    monkeypatch.setattr(
        webapp_module, "_meter_image_description_usage", fake_image_meter
    )
    monkeypatch.setattr(webapp_module, "_meter_message_usage", fake_message_meter)
    return updates, image_meterings, message_meterings


async def _frames(events):
    generator = webapp_module.message_graph_sse(
        _FakeGraph(events),
        HumanMessage(content="[AMBIENT_OBSERVATION id=obs-1]"),
        {"configurable": {"thread_id": "t1"}},
        SimpleNamespace(),
        thread_id="t1",
        user_id="u1",
        assistant_id="a1",
        conversation_title_value="t1",
        start_time_ns=0,
        request_id="r1",
        langgraph_client_headers={},
        app_state=SimpleNamespace(),
        current_user={"API_KEY": "k", "identities": [{"user_id": "u1"}]},
        include_usage_metrics=False,
    )
    frames = []
    async for chunk in generator:
        if chunk.startswith("data: "):
            frames.append(json.loads(chunk[len("data: ") :].strip()))
    return frames


@pytest.mark.asyncio
async def test_an_ignored_observation_streams_the_decision_then_done(harness):
    updates, image_meterings, message_meterings = harness
    frames = await _frames(
        [
            ((), "custom", IMAGE_USAGE),
            ((), "custom", {**IMAGE_USAGE, "source": "screen"}),
            ((), "custom", DECISION),
        ]
    )
    assert [frame["type"] for frame in frames] == ["ambient_decision", "done"]
    assert frames[0]["decision"] == "ignore"
    done = frames[1]
    assert done["content"] == ""
    assert done["ambient"]["decision"] == "ignore"
    assert done["ambient"]["observation_id"] == "obs-1"
    assert "type" not in done["ambient"]
    assert [payload["source"] for payload, _ in image_meterings] == ["webcam", "screen"]
    assert image_meterings[0][1]["thread_id"] == "t1"
    assert message_meterings == []
    assert updates and updates[0][0] == "t1"


@pytest.mark.asyncio
async def test_a_reply_to_an_observation_streams_tokens_after_the_decision(harness):
    _updates, _image_meterings, message_meterings = harness
    reply = AIMessage(
        content="I noticed you are stuck",
        response_metadata={"ambient": {"decision": "respond"}},
    )
    frames = await _frames(
        [
            ((), "custom", {**DECISION, "decision": "respond"}),
            ((), "custom", {"type": "assistant_token", "text": "I noticed"}),
            ((), "updates", {"think": {"messages": [reply]}}),
        ]
    )
    assert [frame["type"] for frame in frames] == [
        "ambient_decision",
        "assistant_token",
        "done",
    ]
    assert frames[-1]["content"] == "I noticed you are stuck"
    assert frames[-1]["ambient"]["decision"] == "respond"
    assert frames[-1]["response_metadata"]["ambient"]["decision"] == "respond"
    assert len(message_meterings) == 1
