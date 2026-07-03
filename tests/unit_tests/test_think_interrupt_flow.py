"""Protect the refactored ``think`` node's durable interrupt/resume forwarding.

Drives the real ``think`` node through both the normal path and the
human-in-the-loop interrupt → resume path, substituting a fake deep agent (a real
checkpointed LangGraph that conditionally ``interrupt``s) so no live model is needed.
This guards the two-checkpointer forwarding the spike validated, now wired into
production ``think``.
"""

from typing import Annotated

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

import src.anubis.graph as graph_mod
from src.anubis.utils.state import GlobalState


class _FakeAgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    system_message: list
    user_identity_documents: list
    assistant_identity_documents: list
    recalled_memory_documents: list
    user_state: dict
    assistant_state: dict
    internal_thoughts: list


def _build_fake_agent(checkpointer):
    """A minimal deep-agent stand-in: replies, and interrupts when asked to correct."""

    def act(state):
        content = getattr(state["messages"][-1], "content", "")
        if isinstance(content, str) and "correct me" in content:
            decision = interrupt({"kind": "fact_correction", "preview": "x"})
            dtype = decision.get("type") if isinstance(decision, dict) else "approve"
            return {"messages": [AIMessage(content=f"corrected:{dtype}")]}
        return {"messages": [AIMessage(content="normal reply")]}

    g = StateGraph(_FakeAgentState)
    g.add_node("act", act)
    g.add_edge(START, "act")
    g.add_edge("act", END)
    return g.compile(checkpointer=checkpointer)


@pytest.fixture
def think_app(monkeypatch):
    shared_checkpointer = MemorySaver()
    monkeypatch.setattr(
        graph_mod, "get_deep_agent_checkpointer", lambda: shared_checkpointer
    )
    monkeypatch.setattr(
        graph_mod,
        "build_avatar_deep_agent",
        lambda context, *, checkpointer=None, store=None: _build_fake_agent(
            checkpointer
        ),
    )
    # Avoid loading the Go Emotions classifier in a unit test.
    monkeypatch.setattr(graph_mod, "_attach_go_emotions_metadata", lambda m: None)

    outer = StateGraph(GlobalState)
    outer.add_node("think", graph_mod.think)
    outer.add_edge(START, "think")
    outer.add_edge("think", END)
    return outer.compile(checkpointer=MemorySaver())


def _input(text: str) -> dict:
    return {
        "messages": [HumanMessage(content=text)],
        "user_state": {"user_id": "u"},
        "assistant_state": {"assistant_id": "a"},
    }


def _interrupts(app, config):
    snap = app.get_state(config)
    return [i for t in snap.tasks for i in t.interrupts] or list(
        getattr(snap, "interrupts", []) or []
    )


@pytest.mark.asyncio
async def test_think_normal_turn_streams_reply(think_app):
    config = {"configurable": {"thread_id": "normal-1"}}
    result = await think_app.ainvoke(_input("hello there"), config)
    assert result["messages"][-1].content == "normal reply"
    assert _interrupts(think_app, config) == []


@pytest.mark.asyncio
async def test_think_interrupt_then_resume_approve(think_app):
    config = {"configurable": {"thread_id": "correct-1"}}

    # Pass 1: the deep agent interrupts; ``think`` surfaces it through the outer graph.
    await think_app.ainvoke(_input("please correct me"), config)
    pending = _interrupts(think_app, config)
    assert pending, "expected an outer interrupt to be pending"
    assert pending[0].value.get("kind") == "fact_correction"

    # Resume with approve → the decision forwards into the deep agent and it finishes.
    resumed = await think_app.ainvoke(Command(resume={"type": "approve"}), config)
    assert resumed["messages"][-1].content == "corrected:approve"
    assert _interrupts(think_app, config) == []


@pytest.mark.asyncio
async def test_think_interrupt_then_resume_edit(think_app):
    config = {"configurable": {"thread_id": "correct-2"}}
    await think_app.ainvoke(_input("correct me now"), config)
    assert _interrupts(think_app, config)
    resumed = await think_app.ainvoke(Command(resume={"type": "edit"}), config)
    assert resumed["messages"][-1].content == "corrected:edit"
