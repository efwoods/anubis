"""Frames the browser acts on must actually reach the browser.

``look_tools`` talks to the browser through custom stream frames rather than
through a pause: ``share_stop`` switches a camera or a screen capture off, and
``share_request`` puts the "let it look at my screen" button in front of the
person. Neither is answered, so neither takes an interrupt — which also means
neither has a reply path that would fail loudly if the frame were lost.

That is exactly how they WERE lost. ``message_graph_sse`` dispatches custom
frames by type and, for a while, knew only about ``assistant_token`` and its
siblings; anything else fell through and was dropped in silence. The
``stop_sharing`` tool and the screen button were inert end to end, with the
tool cheerfully reporting success to the model. Nothing in the look tests
caught it, because they assert on what the tool RETURNS and the return value
was right — it was the side effect that went nowhere.

So these tests are deliberately about the seam and nothing else: a frame put on
the stream by a tool comes out of the SSE generator unchanged. The guard covers
every module that speaks to the browser this way, not only the look tools —
accessibility's ``scene_narration`` rides the same seam and would be lost to
the same gap.
"""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.anubis.utils.tools.vision.accessibility_tools import SCENE_NARRATION_EVENT
from src.anubis.utils.tools.vision.look_tools import (
    SHARE_REQUEST_EVENT,
    SHARE_STOP_EVENT,
)
from src.api import webapp as webapp_module


class _FakeGraph:
    def __init__(self, events):
        self._events = events

    async def astream(self, input, config, context, stream_mode, subgraphs):
        for event in self._events:
            yield event

    async def aget_state(self, config):
        return SimpleNamespace(next=(), tasks=(), interrupts=())


@pytest.fixture
def _quiet_meters(monkeypatch):
    class _Threads:
        async def update(self, thread_id, metadata):
            return None

    monkeypatch.setattr(
        webapp_module,
        "get_client",
        lambda **kwargs: SimpleNamespace(threads=_Threads()),
    )

    async def fake_message_meter(**kwargs):
        return {"tokens": 1}

    monkeypatch.setattr(webapp_module, "_meter_message_usage", fake_message_meter)


async def _frames(events):
    generator = webapp_module.message_graph_sse(
        _FakeGraph(events),
        HumanMessage(content="what is on my screen?"),
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
async def test_the_button_offer_reaches_the_browser(_quiet_meters):
    offer = {
        "type": SHARE_REQUEST_EVENT,
        "sources": ["screen"],
        "reason": "to read the error",
    }
    frames = await _frames(
        [
            ((), "custom", offer),
            ((), "updates", {"anubis": {"messages": [AIMessage(content="Press it")]}}),
        ]
    )
    assert offer in frames, "the offer never reached the person, so no button appeared"
    assert [frame["type"] for frame in frames] == [
        "turn_started",
        SHARE_REQUEST_EVENT,
        "done",
    ]


@pytest.mark.asyncio
async def test_switching_a_share_off_reaches_the_browser(_quiet_meters):
    stop = {"type": SHARE_STOP_EVENT, "sources": ["webcam", "screen"]}
    frames = await _frames(
        [
            ((), "custom", stop),
            ((), "updates", {"anubis": {"messages": [AIMessage(content="Done")]}}),
        ]
    )
    assert stop in frames, "stop_sharing told the model it worked and did nothing"


@pytest.mark.asyncio
async def test_the_frame_is_forwarded_exactly_as_the_tool_wrote_it(_quiet_meters):
    # The browser matches on these fields; a frame that arrives reshaped is as
    # useless as one that does not arrive.
    offer = {
        "type": SHARE_REQUEST_EVENT,
        "sources": ["screen"],
        "reason": "",
    }
    frames = await _frames([((), "custom", offer)])
    forwarded = [frame for frame in frames if frame["type"] == SHARE_REQUEST_EVENT]
    assert forwarded == [offer]


def test_every_browser_directed_frame_is_on_the_forwarding_list():
    # The list and the tools that write to it live in different modules, so
    # nothing but this test notices when a new browser-directed frame is added
    # to one and not the other — which is the shape of the original bug, and
    # the shape it would take again for any tool that speaks to the browser
    # through a frame instead of a pause. Every such constant belongs here.
    assert SHARE_STOP_EVENT in webapp_module.BROWSER_DIRECTED_FRAMES
    assert SHARE_REQUEST_EVENT in webapp_module.BROWSER_DIRECTED_FRAMES
    assert SCENE_NARRATION_EVENT in webapp_module.BROWSER_DIRECTED_FRAMES


@pytest.mark.asyncio
async def test_a_scene_narration_frame_reaches_the_browser(_quiet_meters):
    # Accessibility rides the same seam: the avatar turning scene narration on
    # is a frame the browser acts on, and it would be dropped by exactly the
    # same gap that swallowed the share frames.
    switch = {"type": SCENE_NARRATION_EVENT, "state": "on"}
    frames = await _frames([((), "custom", switch)])
    assert switch in frames
