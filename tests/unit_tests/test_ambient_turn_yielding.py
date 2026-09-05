"""Ambient observations yield to the person's own turns on a thread.

A typed or spoken turn arriving while a webcam / screen observation is still
being processed on the same thread stops that observation and waits for the
run behind the observation to wind down before streaming, so the two never
write the thread's checkpoint at the same time. An observation arriving while
any turn is running on the thread is refused with 409 and a ``Retry-After``.
An observation stopped before the graph triaged the observation is removed
from the thread instead of being kept as context.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from langchain_core.messages import HumanMessage, RemoveMessage

from src.api import webapp as webapp_module
from src.api.message_stops import (
    AMBIENT_BUSY_RETRY_AFTER_SECONDS,
    ActiveMessageTurnRegistry,
    discard_ambient_observation,
    yield_in_flight_ambient_observations,
)


def _register(registry, request_id, thread_id, *, ambient=False):
    return registry.register(
        request_id=request_id,
        thread_id=thread_id,
        assistant_id="a1",
        user_id="u1",
        ambient=ambient,
    )


@pytest.mark.asyncio
async def test_a_typed_turn_stops_the_observations_on_its_thread_and_waits():
    registry = ActiveMessageTurnRegistry()
    observation = _register(registry, "obs-1", "t1", ambient=True)
    other_thread_observation = _register(registry, "obs-2", "t2", ambient=True)
    typed_elsewhere = _register(registry, "typed-2", "t1")

    async def wind_down():
        await asyncio.sleep(0.02)
        observation.mark_finished()

    asyncio.get_running_loop().create_task(wind_down())
    stopped = await yield_in_flight_ambient_observations(registry, "t1")

    assert stopped == ["obs-1"]
    assert observation.stop_requested.is_set()
    assert observation.finished.is_set()
    assert not other_thread_observation.stop_requested.is_set()
    assert not typed_elsewhere.stop_requested.is_set()


@pytest.mark.asyncio
async def test_yielding_gives_up_after_the_timeout():
    registry = ActiveMessageTurnRegistry()
    observation = _register(registry, "obs-1", "t1", ambient=True)

    stopped = await yield_in_flight_ambient_observations(
        registry, "t1", timeout_seconds=0.02
    )

    assert stopped == ["obs-1"]
    assert observation.stop_requested.is_set()
    assert not observation.finished.is_set()


@pytest.mark.asyncio
async def test_a_thread_with_no_observation_has_nothing_to_yield():
    registry = ActiveMessageTurnRegistry()
    _register(registry, "typed-1", "t1")

    assert await yield_in_flight_ambient_observations(registry, "t1") == []
    assert await yield_in_flight_ambient_observations(registry, None) == []
    assert await yield_in_flight_ambient_observations(None, "t1") == []


def test_an_observation_on_a_busy_thread_is_refused_with_409():
    registry = ActiveMessageTurnRegistry()
    _register(registry, "typed-1", "t1")

    with pytest.raises(HTTPException) as refusal:
        webapp_module.refuse_ambient_observation_on_busy_thread(registry, "t1")

    assert refusal.value.status_code == 409
    assert refusal.value.headers["Retry-After"] == str(AMBIENT_BUSY_RETRY_AFTER_SECONDS)


def test_an_observation_on_an_idle_thread_is_not_refused():
    registry = ActiveMessageTurnRegistry()
    _register(registry, "typed-1", "t1")

    webapp_module.refuse_ambient_observation_on_busy_thread(registry, "t2")
    webapp_module.refuse_ambient_observation_on_busy_thread(registry, None)
    webapp_module.refuse_ambient_observation_on_busy_thread(None, "t1")


@pytest.mark.asyncio
async def test_discarding_an_untriaged_observation_removes_the_message():
    updates = []

    class _Graph:
        async def aupdate_state(self, config, values, as_node=None):
            updates.append((config, values, as_node))

    removed = await discard_ambient_observation(
        _Graph(), {"configurable": {"thread_id": "t1"}}, "obs-message"
    )

    assert removed is True
    (_config, values, as_node) = updates[0]
    assert as_node == "anubis"
    assert isinstance(values["messages"][0], RemoveMessage)
    assert values["messages"][0].id == "obs-message"
    assert await discard_ambient_observation(_Graph(), {}, None) is False


class _RecordingGraph:
    """Streams one token after a delay; remembers when the run started."""

    def __init__(self, delay=0.0, tokens=("hi",)):
        self._delay = delay
        self._tokens = tokens
        self.started_at = None
        self.state_updates = []

    async def astream(self, input, config, context, stream_mode, subgraphs):
        self.started_at = asyncio.get_running_loop().time()
        for token in self._tokens:
            await asyncio.sleep(self._delay)
            yield ((), "custom", {"type": "assistant_token", "text": token})

    async def aupdate_state(self, config, values, as_node=None):
        self.state_updates.append((values, as_node))

    async def aget_state(self, config):
        return SimpleNamespace(next=(), tasks=(), interrupts=())


@pytest.fixture
def quiet_api(monkeypatch):
    class _Threads:
        async def update(self, thread_id, metadata):
            return None

    monkeypatch.setattr(
        webapp_module,
        "get_client",
        lambda **kwargs: SimpleNamespace(threads=_Threads()),
    )

    async def fake_message_meter(**kwargs):
        return None

    monkeypatch.setattr(webapp_module, "_meter_message_usage", fake_message_meter)


async def _collect(generator, on_frame=None):
    frames = []
    async for chunk in generator:
        if not chunk.startswith("data: "):
            continue
        frame = json.loads(chunk[len("data: ") :].strip())
        frames.append(frame)
        if on_frame is not None:
            await on_frame(frame)
    return frames


def _stream(graph, registry, *, human_message, request_id, ambient):
    return webapp_module.message_graph_sse(
        graph,
        human_message,
        {"configurable": {"thread_id": "t1"}},
        SimpleNamespace(),
        thread_id="t1",
        user_id="u1",
        assistant_id="a1",
        conversation_title_value="t1",
        start_time_ns=0,
        request_id=request_id,
        langgraph_client_headers={},
        app_state=SimpleNamespace(active_message_turns=registry),
        current_user={"API_KEY": "k", "identities": [{"user_id": "u1"}]},
        include_usage_metrics=False,
        ambient=ambient,
    )


@pytest.mark.asyncio
async def test_a_typed_turn_waits_for_the_observation_before_streaming(quiet_api):
    registry = ActiveMessageTurnRegistry()
    observation = _register(registry, "obs-1", "t1", ambient=True)
    graph = _RecordingGraph()
    loop = asyncio.get_running_loop()
    finished_at = None

    async def wind_down():
        nonlocal finished_at
        await observation.stop_requested.wait()
        await asyncio.sleep(0.02)
        finished_at = loop.time()
        observation.mark_finished()

    loop.create_task(wind_down())
    frames = await _collect(
        _stream(
            graph,
            registry,
            human_message=HumanMessage(content="what is on my screen?"),
            request_id="typed-1",
            ambient=False,
        )
    )

    assert observation.stop_requested.is_set()
    assert finished_at is not None and graph.started_at >= finished_at
    assert frames[-1]["type"] == "done" and frames[-1]["content"] == "hi"
    assert registry.find(request_id="typed-1") is None


@pytest.mark.asyncio
async def test_a_stopped_untriaged_observation_is_discarded(quiet_api):
    registry = ActiveMessageTurnRegistry()
    graph = _RecordingGraph(delay=5.0)
    human_message = HumanMessage(
        id="obs-message",
        content="[AMBIENT_OBSERVATION id=obs-1]",
        additional_kwargs={"hidden": True, "kind": "ambient_observation"},
    )

    async def stop_after_first_frame(frame):
        if frame["type"] == "turn_started":
            registry.find(request_id="obs-1").request_stop()

    frames = await _collect(
        _stream(
            graph,
            registry,
            human_message=human_message,
            request_id="obs-1",
            ambient=True,
        ),
        on_frame=stop_after_first_frame,
    )

    assert frames[-1]["type"] == "done" and frames[-1]["stopped"] is True
    removals = [
        values["messages"][0]
        for values, _as_node in graph.state_updates
        if isinstance(values["messages"][0], RemoveMessage)
    ]
    assert [removal.id for removal in removals] == ["obs-message"]


@pytest.mark.asyncio
async def test_a_finished_turn_is_marked_finished_for_waiters(quiet_api):
    registry = ActiveMessageTurnRegistry()
    seen = {}

    async def remember_turn(frame):
        if frame["type"] == "turn_started":
            seen["turn"] = registry.find(request_id="typed-1")

    await _collect(
        _stream(
            _RecordingGraph(),
            registry,
            human_message=HumanMessage(content="hello"),
            request_id="typed-1",
            ambient=False,
        ),
        on_frame=remember_turn,
    )

    assert seen["turn"].finished.is_set()
