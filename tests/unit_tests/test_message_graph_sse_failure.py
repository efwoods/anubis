"""``message_graph_sse`` when the run fails after the stream has begun.

The response is already a 200 with frames on the wire, so the failure cannot be
a status code. The stream ends with an ``error`` frame that says why, in a
shape the client reports like an HTTP failure, and the generator returns rather
than raising — raising cut the connection with no terminating chunk, which the
browser reported as a bare "stream ended unexpectedly".
"""

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from langchain_core.messages import HumanMessage

from src.api import webapp as webapp_module


class _FailingGraph:
    def __init__(self, events, error):
        self._events = events
        self._error = error

    async def astream(self, input, config, context, stream_mode, subgraphs):
        for event in self._events:
            yield event
        raise self._error

    async def aget_state(self, config):
        return SimpleNamespace(next=(), tasks=(), interrupts=())


async def _frames(events, error):
    generator = webapp_module.message_graph_sse(
        _FailingGraph(events, error),
        HumanMessage(content="hello"),
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
async def test_a_vendor_out_of_credit_ends_with_a_503_error_frame():
    frames = await _frames(
        [((), "custom", {"type": "assistant_token", "text": "Hel"})],
        RuntimeError(
            "Error code: 429 - insufficient_quota: You exceeded your current quota"
        ),
    )
    assert [frame["type"] for frame in frames] == [
        "turn_started",
        "assistant_token",
        "error",
    ]
    error = frames[-1]
    assert error["status"] == 503
    assert error["code"] == "model_provider_credit_exhausted"
    assert "on our side" in error["message"]
    assert error["request_id"] == "r1"
    assert error["thread_id"] == "t1"


@pytest.mark.asyncio
async def test_a_402_raised_inside_the_run_keeps_its_status_and_sentence():
    frames = await _frames(
        [],
        HTTPException(status_code=402, detail="The monthly allotment is spent."),
    )
    assert [frame["type"] for frame in frames] == ["turn_started", "error"]
    assert frames[-1]["status"] == 402
    assert frames[-1]["message"] == "The monthly allotment is spent."


@pytest.mark.asyncio
async def test_any_other_failure_is_reported_without_its_internals():
    frames = await _frames([], ValueError("secret internal detail"))
    error = frames[-1]
    assert error["type"] == "error"
    assert error["status"] == 500
    assert "secret" not in error["message"]


def test_structured_http_detail_keeps_code_and_sentence():
    frame = webapp_module._stream_error_frame(
        HTTPException(
            status_code=402,
            detail={"error": "allotment_exhausted", "detail": "Spent."},
        ),
        request_id="r1",
        thread_id=None,
    )
    assert frame["code"] == "allotment_exhausted"
    assert frame["message"] == "Spent."
    assert frame["thread_id"] is None
