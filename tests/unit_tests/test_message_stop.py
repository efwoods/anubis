"""Stopping a reply mid-stream: the stop route, the pump, and the finalizers.

A fake graph plays tokens slowly. Stopping through the registry (what
``POST /message/{assistant_id}/stop`` does) must cancel the graph run, record
the partial reply on the thread, meter the estimated usage, and end the stream
with a ``done`` frame flagged ``stopped``. A client that disconnects instead
must leave the same record behind through the background finalizer.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.api import webapp as webapp_module
from src.api.message_stops import (
    STOP_REQUESTED,
    STREAM_ENDED,
    ActiveMessageTurnRegistry,
    GraphStreamPump,
    build_stopped_reply_metadata,
    estimate_completion_tokens,
)


class _SlowTokenGraph:
    """Streams one token per ``delay`` seconds and records state updates."""

    def __init__(self, tokens, delay=0.01):
        self._tokens = tokens
        self._delay = delay
        self.state_updates = []
        self.stream_closed = False
        self.tokens_emitted = 0

    async def astream(self, input, config, context, stream_mode, subgraphs):
        try:
            for token in self._tokens:
                await asyncio.sleep(self._delay)
                self.tokens_emitted += 1
                yield ((), "custom", {"type": "assistant_token", "text": token})
        finally:
            self.stream_closed = True

    async def aupdate_state(self, config, values, as_node=None):
        self.state_updates.append((config, values, as_node))

    async def aget_state(self, config):
        return SimpleNamespace(next=(), tasks=(), interrupts=())


@pytest.fixture
def harness(monkeypatch):
    thread_updates = []
    message_meterings = []

    class _Threads:
        async def update(self, thread_id, metadata):
            thread_updates.append((thread_id, metadata))

    monkeypatch.setattr(
        webapp_module,
        "get_client",
        lambda **kwargs: SimpleNamespace(threads=_Threads()),
    )

    async def fake_message_meter(**kwargs):
        message_meterings.append(kwargs)
        token_usage = (kwargs["response_metadata"] or {}).get("token_usage") or {}
        return {"tokens": token_usage.get("total_tokens", 0)}

    monkeypatch.setattr(webapp_module, "_meter_message_usage", fake_message_meter)
    return thread_updates, message_meterings


def _app_state():
    return SimpleNamespace(
        active_message_turns=ActiveMessageTurnRegistry(),
        context=SimpleNamespace(model="test-model"),
    )


def _stream(graph, app_state, request_id="r1", estimated_prompt_tokens=250):
    return webapp_module.message_graph_sse(
        graph,
        SimpleNamespace(content="hello"),
        {"configurable": {"thread_id": "t1"}},
        SimpleNamespace(),
        thread_id="t1",
        user_id="u1",
        assistant_id="a1",
        conversation_title_value="t1",
        start_time_ns=0,
        request_id=request_id,
        langgraph_client_headers={},
        app_state=app_state,
        current_user={"API_KEY": "k", "identities": [{"user_id": "u1"}]},
        estimated_request_tokens=SimpleNamespace(
            input_tokens=estimated_prompt_tokens,
            total_tokens=estimated_prompt_tokens + 100,
        ),
        include_usage_metrics=True,
    )


def _parse(chunk):
    return json.loads(chunk[len("data: ") :].strip())


async def _settle(condition, attempts=50):
    """Give background finalizers a moment to run."""
    for _ in range(attempts):
        if condition():
            return
        await asyncio.sleep(0.01)
    assert condition()


@pytest.mark.asyncio
async def test_the_first_frame_names_the_turn_and_the_turn_is_registered_while_streaming(
    harness,
):
    app_state = _app_state()
    graph = _SlowTokenGraph(["Hi", " there"])
    generator = _stream(graph, app_state)

    first = _parse(await generator.__anext__())
    assert first == {"type": "turn_started", "request_id": "r1", "thread_id": "t1"}
    # The turn is stoppable from the moment the client learns its id.
    assert app_state.active_message_turns.active_request_ids() == ["r1"]
    frames = [first]
    async for chunk in generator:
        if chunk.startswith("data: "):
            frames.append(_parse(chunk))
    assert [frame["type"] for frame in frames][-1] == "done"
    assert frames[-1]["content"] == "Hi there"
    assert "stopped" not in frames[-1]
    assert len(app_state.active_message_turns) == 0
    assert graph.state_updates == []


@pytest.mark.asyncio
async def test_stopping_through_the_registry_ends_the_stream_with_a_stopped_done_frame(
    harness,
):
    thread_updates, message_meterings = harness
    app_state = _app_state()
    graph = _SlowTokenGraph(["The ", "quick ", "brown ", "fox ", "jumps"])
    frames = []
    async for chunk in _stream(graph, app_state):
        if not chunk.startswith("data: "):
            continue
        frame = _parse(chunk)
        frames.append(frame)
        if frame["type"] == "assistant_token" and len(frames) == 3:
            # Two tokens have arrived; the person presses Stop.
            turn = app_state.active_message_turns.find(request_id="r1")
            assert turn is not None and turn.user_id == "u1"
            turn.request_stop()

    types = [frame["type"] for frame in frames]
    assert types[0] == "turn_started"
    assert types[-1] == "done"
    assert types.count("assistant_token") <= 3
    done = frames[-1]
    assert done["stopped"] is True
    assert done["stopped_by"] == "user"
    partial = "".join(
        frame["text"] for frame in frames if frame["type"] == "assistant_token"
    )
    assert done["content"] == partial
    assert done["response_metadata"]["stopped_by_user"] is True

    # The graph run was cancelled, not run to completion.
    assert graph.stream_closed is True
    assert graph.tokens_emitted < 5

    # The partial reply was recorded on the thread as the avatar's turn.
    assert len(graph.state_updates) == 1
    _config, values, as_node = graph.state_updates[0]
    assert as_node == "anubis"
    recorded = values["messages"][0]
    assert recorded.content == partial
    assert recorded.response_metadata["stopped"] is True
    assert recorded.id == "stopped-r1"

    # The frame went out without waiting on metering (the Stripe call can take
    # seconds); the estimated usage is metered right after, in the background.
    assert "usage" not in done
    await _settle(lambda: len(message_meterings) == 1)
    token_usage = message_meterings[0]["response_metadata"]["token_usage"]
    assert token_usage["prompt_tokens"] == 250
    assert token_usage["completion_tokens"] == estimate_completion_tokens(partial)
    assert token_usage["estimated"] is True
    assert thread_updates and thread_updates[0][0] == "t1"
    assert len(app_state.active_message_turns) == 0


@pytest.mark.asyncio
async def test_a_stop_before_any_token_records_nothing_on_the_thread(harness):
    _thread_updates, message_meterings = harness
    app_state = _app_state()
    graph = _SlowTokenGraph(["never"], delay=1.0)
    frames = []
    async for chunk in _stream(graph, app_state):
        if not chunk.startswith("data: "):
            continue
        frame = _parse(chunk)
        frames.append(frame)
        if frame["type"] == "usage_estimate":
            # The registry entry exists by now; stop before the first token.
            app_state.active_message_turns.find(thread_id="t1").request_stop()

    assert [frame["type"] for frame in frames] == [
        "turn_started",
        "usage_estimate",
        "done",
    ]
    assert frames[-1]["stopped"] is True
    assert frames[-1]["content"] == ""
    assert graph.state_updates == []
    # The prompt still reached the model, so the estimate is metered.
    await _settle(lambda: len(message_meterings) == 1)
    assert message_meterings[0]["response_metadata"]["token_usage"] == {
        "prompt_tokens": 250,
        "completion_tokens": 0,
        "total_tokens": 250,
        "estimated": True,
    }


@pytest.mark.asyncio
async def test_a_disconnected_client_still_gets_the_partial_reply_recorded(harness):
    thread_updates, message_meterings = harness
    app_state = _app_state()
    graph = _SlowTokenGraph(["a", "b", "c", "d"])
    generator = _stream(graph, app_state)

    async def consume_two_tokens():
        seen = 0
        async for chunk in generator:
            if chunk.startswith("data: ") and _parse(chunk)["type"] == "assistant_token":
                seen += 1
                if seen == 2:
                    # Starlette cancels the response task when the socket closes.
                    raise asyncio.CancelledError()

    consumer = asyncio.create_task(consume_two_tokens())
    with pytest.raises(asyncio.CancelledError):
        await consumer
    # The generator was abandoned mid-yield; closing it is what Starlette /
    # garbage collection would do next.
    await generator.aclose()

    # Let the background finalizer run.
    await _settle(lambda: bool(graph.state_updates and message_meterings))

    assert graph.stream_closed is True
    assert len(graph.state_updates) == 1
    _config, values, as_node = graph.state_updates[0]
    assert as_node == "anubis"
    assert values["messages"][0].content == "ab"
    assert values["messages"][0].response_metadata["stopped_by"] == "disconnect"
    assert message_meterings[0]["response_metadata"]["stopped_by"] == "disconnect"
    assert thread_updates and thread_updates[0][0] == "t1"
    assert len(app_state.active_message_turns) == 0


@pytest.mark.asyncio
async def test_the_pump_hands_items_through_then_reports_the_end():
    async def source():
        yield 1
        yield 2

    pump = GraphStreamPump(source())
    assert await pump.next_item() == 1
    assert await pump.next_item() == 2
    assert await pump.next_item() is STREAM_ENDED
    await pump.aclose()


@pytest.mark.asyncio
async def test_the_pump_wakes_a_waiting_consumer_on_stop():
    async def source():
        await asyncio.sleep(10)
        yield "late"

    pump = GraphStreamPump(source())
    waiter = asyncio.create_task(pump.next_item())
    await asyncio.sleep(0)
    pump.request_stop()
    assert await asyncio.wait_for(waiter, timeout=1) is STOP_REQUESTED
    await pump.aclose()
    assert pump.done


@pytest.mark.asyncio
async def test_the_pump_reraises_a_graph_error_to_the_consumer():
    async def source():
        yield 1
        raise RuntimeError("model unavailable")

    pump = GraphStreamPump(source())
    assert await pump.next_item() == 1
    with pytest.raises(RuntimeError, match="model unavailable"):
        await pump.next_item()
    await pump.aclose()


def test_the_registry_finds_the_newest_turn_on_a_thread():
    registry = ActiveMessageTurnRegistry()
    older = registry.register(
        request_id="r1", thread_id="t1", assistant_id="a1", user_id="u1"
    )
    older.started_at_ns -= 1_000
    newer = registry.register(
        request_id="r2", thread_id="t1", assistant_id="a1", user_id="u1"
    )
    assert registry.find(request_id="r1") is older
    assert registry.find(thread_id="t1") is newer
    assert registry.find(request_id="missing", thread_id="t1") is newer
    assert registry.find(request_id="missing") is None
    registry.unregister("r2")
    assert registry.find(thread_id="t1") is older


@pytest.mark.asyncio
async def test_a_stop_requested_before_the_pump_exists_is_honoured_on_attach():
    registry = ActiveMessageTurnRegistry()
    turn = registry.register(
        request_id="r1", thread_id="t1", assistant_id="a1", user_id="u1"
    )
    turn.request_stop()
    woke = []
    turn.attach_wake(lambda: woke.append(True))
    assert woke == [True]
    # A second request is a no-op.
    turn.request_stop()
    assert woke == [True]


def test_stopped_reply_metadata_estimates_usage():
    metadata = build_stopped_reply_metadata(
        "twelve chars",
        estimated_prompt_tokens=40,
        model_name="m",
        stopped_by="user",
    )
    assert metadata["token_usage"] == {
        "prompt_tokens": 40,
        "completion_tokens": 3,
        "total_tokens": 43,
        "estimated": True,
    }
    assert metadata["model_name"] == "m"
    assert metadata["stopped_by_user"] is True
    assert estimate_completion_tokens("") == 0
    assert estimate_completion_tokens("a") == 1


class _StopRouteRequest:
    def __init__(self, registry):
        self.app = SimpleNamespace(state=SimpleNamespace(active_message_turns=registry))


def _user(user_id):
    return {"API_KEY": "k", "identities": [{"user_id": user_id}]}


@pytest.mark.asyncio
async def test_the_stop_route_stops_only_the_callers_own_turn():
    registry = ActiveMessageTurnRegistry()
    turn = registry.register(
        request_id="r1", thread_id="t1", assistant_id="a1", user_id="u1"
    )
    request = _StopRouteRequest(registry)

    with pytest.raises(HTTPException) as unknown:
        await webapp_module.stop_avatar_message(
            request, "a1", request_id="nope", thread_id=None, current_user=_user("u1")
        )
    assert unknown.value.status_code == 404

    with pytest.raises(HTTPException) as someone_else:
        await webapp_module.stop_avatar_message(
            request, "a1", request_id="r1", thread_id=None, current_user=_user("u2")
        )
    assert someone_else.value.status_code == 404

    with pytest.raises(HTTPException) as wrong_avatar:
        await webapp_module.stop_avatar_message(
            request, "a2", request_id="r1", thread_id=None, current_user=_user("u1")
        )
    assert wrong_avatar.value.status_code == 404
    assert not turn.stop_requested.is_set()

    with pytest.raises(HTTPException) as no_ids:
        await webapp_module.stop_avatar_message(
            request, "a1", request_id=None, thread_id=None, current_user=_user("u1")
        )
    assert no_ids.value.status_code == 422

    response = await webapp_module.stop_avatar_message(
        request, "a1", request_id=None, thread_id="t1", current_user=_user("u1")
    )
    assert response.status_code == 200
    assert json.loads(response.body) == {
        "status": "stopping",
        "request_id": "r1",
        "thread_id": "t1",
    }
    assert turn.stop_requested.is_set()
