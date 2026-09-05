"""A reply to an ambient observation carries the observation's triage record.

Drives the real ``think`` node with a fake deep agent (as the interrupt-flow
tests do) and checks that ``response_metadata["ambient"]`` is set on the reply
to a hidden ambient turn and absent on the reply to a typed turn — the field
the client reads to render a ``notify`` reply as a notification card.
"""

from typing import Annotated

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from typing_extensions import TypedDict

import src.anubis.graph as graph_mod
from src.anubis.utils.ambient.observations import build_ambient_additional_kwargs
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
    conversation_summary_event: dict | None
    conversation_summary_session_id: str | None


def _build_fake_agent(checkpointer):
    def act(state):
        return {"messages": [AIMessage(content="heads up")]}

    builder = StateGraph(_FakeAgentState)
    builder.add_node("act", act)
    builder.add_edge(START, "act")
    builder.add_edge("act", END)
    return builder.compile(checkpointer=checkpointer)


@pytest.fixture
def think_app(monkeypatch):
    shared_checkpointer = MemorySaver()
    monkeypatch.setattr(
        graph_mod, "get_deep_agent_checkpointer", lambda: shared_checkpointer
    )

    def fake_builder(
        context, *, checkpointer=None, store=None, extra_tools=None, backend=None
    ):
        return _build_fake_agent(checkpointer)

    monkeypatch.setattr(graph_mod, "build_avatar_deep_agent", fake_builder)
    monkeypatch.setattr(graph_mod, "_attach_go_emotions_metadata", lambda m: None)
    outer = StateGraph(GlobalState)
    outer.add_node("think", graph_mod.think)
    outer.add_edge(START, "think")
    outer.add_edge("think", END)
    return outer.compile(checkpointer=MemorySaver())


def _input(message):
    return {
        "messages": [message],
        "user_state": {"user_id": "u"},
        "assistant_state": {"assistant_id": "a"},
    }


@pytest.mark.asyncio
async def test_the_reply_to_an_ambient_turn_is_tagged(think_app):
    kwargs = build_ambient_additional_kwargs(
        sources=["screen"], captured_at="t", voice_mode=False, observation_id="obs-9"
    )
    kwargs["ambient"].update(
        {"decision": "notify", "summary": "An error dialog is open."}
    )
    message = HumanMessage(
        id="obs",
        content="[AMBIENT_OBSERVATION id=obs-9]\nscreen: error",
        additional_kwargs=kwargs,
    )
    result = await think_app.ainvoke(
        _input(message), {"configurable": {"thread_id": "t1"}}
    )
    reply = result["messages"][-1]
    assert reply.content == "heads up"
    assert reply.response_metadata["ambient"]["decision"] == "notify"
    assert reply.response_metadata["ambient"]["observation_id"] == "obs-9"


@pytest.mark.asyncio
async def test_the_reply_to_a_typed_turn_is_not_tagged(think_app):
    result = await think_app.ainvoke(
        _input(HumanMessage(content="hello")), {"configurable": {"thread_id": "t2"}}
    )
    assert "ambient" not in (result["messages"][-1].response_metadata or {})
