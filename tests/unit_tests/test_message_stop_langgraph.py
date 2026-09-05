"""The stop path against a real compiled LangGraph graph.

The fakes in ``test_message_stop.py`` pin down the stream contract; this file
pins down the two LangGraph assumptions the contract rests on:

* closing the pump cancels the graph run — the node that is streaming tokens
  is interrupted rather than left running to completion in the background;
* ``aupdate_state(..., as_node=<node>)`` on a run that was cancelled mid-node
  records the partial reply after the human turn and leaves nothing pending,
  so the next human message starts a fresh run.
"""

import asyncio
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph, add_messages

from src.api.message_stops import (
    STOP_REQUESTED,
    GraphStreamPump,
    persist_stopped_reply,
)


class _State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def _build_graph(observed: dict):
    async def slow_reply(state: _State):
        writer = get_stream_writer()
        try:
            for token in ["The ", "quick ", "brown ", "fox ", "jumps ", "over"]:
                await asyncio.sleep(0.02)
                writer({"type": "assistant_token", "text": token})
                observed["tokens"] = observed.get("tokens", 0) + 1
            observed["completed"] = True
            return {"messages": [AIMessage(content="The quick brown fox jumps over")]}
        except asyncio.CancelledError:
            observed["cancelled"] = True
            raise

    workflow = StateGraph(_State)
    workflow.add_node("anubis", slow_reply)
    workflow.add_edge(START, "anubis")
    workflow.add_edge("anubis", END)
    return workflow.compile(checkpointer=InMemorySaver())


@pytest.mark.asyncio
async def test_closing_the_pump_cancels_the_running_node_and_the_partial_reply_is_recorded():
    observed: dict = {}
    graph = _build_graph(observed)
    config = {"configurable": {"thread_id": "t1"}}

    pump = GraphStreamPump(
        graph.astream(
            {"messages": [HumanMessage(content="Tell me a story")]},
            config=config,
            stream_mode=["custom", "updates"],
            subgraphs=True,
        )
    )
    streamed: list[str] = []
    while True:
        item = await pump.next_item()
        if item is STOP_REQUESTED:
            break
        _namespace, mode, payload = item
        if mode == "custom" and payload.get("type") == "assistant_token":
            streamed.append(payload["text"])
            if len(streamed) == 2:
                pump.request_stop()
    await pump.aclose()

    # The node was interrupted, not run to the end behind the client's back.
    for _ in range(50):
        if observed.get("cancelled") or observed.get("completed"):
            break
        await asyncio.sleep(0.01)
    assert observed.get("cancelled") is True
    assert observed.get("completed") is None
    assert observed["tokens"] < 6

    # Nothing but the human turn is on the thread yet, and the node is pending.
    before = await graph.aget_state(config)
    assert [type(message).__name__ for message in before.values["messages"]] == [
        "HumanMessage"
    ]
    assert before.next == ("anubis",)

    partial = "".join(streamed)
    recorded = await persist_stopped_reply(
        graph,
        config,
        partial,
        request_id="r1",
        response_metadata={"stopped": True, "stopped_by": "user"},
    )
    assert recorded is True

    after = await graph.aget_state(config)
    assert [type(message).__name__ for message in after.values["messages"]] == [
        "HumanMessage",
        "AIMessage",
    ]
    assert after.values["messages"][-1].content == partial
    assert after.values["messages"][-1].response_metadata["stopped"] is True
    assert after.next == ()

    # The next human message starts a fresh run that sees the truncated reply
    # as ordinary history.
    observed.clear()
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Go on")]}, config=config
    )
    assert observed.get("completed") is True
    assert [message.content for message in result["messages"]] == [
        "Tell me a story",
        partial,
        "Go on",
        "The quick brown fox jumps over",
    ]
