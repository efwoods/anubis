"""Protect the ``mcp_discovery`` node's discover → consent → persist flow.

Drives the real ``mcp_discovery`` node through a minimal checkpointed outer
graph (same pattern as ``test_think_interrupt_flow``), substituting a fake
``discover_announced_server`` so no live Model Context Protocol server is
needed. Covers: the consent interrupt is raised with the announced server;
approving persists the single per-user connection bound to the answering
avatar; declining records a per-avatar marker; and non-owned avatars are
never offered a connection.
"""

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

import src.anubis.graph as graph_mod
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.state import GlobalState
from src.anubis.utils.tools.data_analysis import (
    McpConnection,
    bound_connection_for,
    is_declined,
)

_ANNOUNCED = McpConnection(
    url="http://localhost:8000/mcp",
    transport="streamable_http",
    server_name="Ubuntu-OS-Filesystem",
    allowed_roots=("/home/user/data",),
)


@pytest.fixture
def discovery_app(monkeypatch):
    """A one-node graph running the real ``mcp_discovery`` with a fake discovery."""

    async def _fake_discover(discovery_url, timeout_seconds):
        return _ANNOUNCED

    monkeypatch.setattr(graph_mod, "discover_announced_server", _fake_discover)

    store = InMemoryStore()
    outer = StateGraph(GlobalState)
    outer.add_node("mcp_discovery", graph_mod.mcp_discovery)
    outer.add_edge(START, "mcp_discovery")
    outer.add_edge("mcp_discovery", END)
    app = outer.compile(checkpointer=MemorySaver(), store=store)
    return app, store


def _input(user_id="u", assistant_id="a") -> dict:
    return {
        "messages": [HumanMessage(content="hello")],
        "user_state": {"user_id": user_id},
        "assistant_state": {"assistant_id": assistant_id},
    }


def _owned_config(thread_id, user_id="u", assistant_id="a") -> dict:
    # The avatar's metadata.user_id == the conversing user_id ⇒ the user owns it.
    return {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            "assistant_id": assistant_id,
            "assistant_ctx": {"metadata": {"user_id": user_id}},
        }
    }


def _interrupts(app, config):
    snap = app.get_state(config)
    return [i for t in snap.tasks for i in t.interrupts] or list(
        getattr(snap, "interrupts", []) or []
    )


@pytest.mark.asyncio
async def test_discovery_offers_and_approve_persists_bound_connection(discovery_app):
    app, store = discovery_app
    config = _owned_config("connect-1")

    # Pass 1: an announced-but-unconnected server triggers the consent interrupt.
    await app.ainvoke(_input(), config, context=GlobalContext())
    pending = _interrupts(app, config)
    assert pending, "expected a consent interrupt"
    assert pending[0].value.get("kind") == "mcp_connect_consent"
    assert pending[0].value["server"]["url"] == _ANNOUNCED.url

    # Resume with apply → the single per-user connection is saved, bound to "a".
    await app.ainvoke(Command(resume={"type": "apply"}), config, context=GlobalContext())
    assert _interrupts(app, config) == []
    bound = await bound_connection_for(store, "u", "a")
    assert bound is not None and bound.url == _ANNOUNCED.url
    # The user's other avatar shares the record but is not the bound avatar.
    assert await bound_connection_for(store, "u", "other") is None


@pytest.mark.asyncio
async def test_discovery_decline_marks_only_this_avatar(discovery_app):
    app, store = discovery_app
    config = _owned_config("decline-1")

    await app.ainvoke(_input(), config, context=GlobalContext())
    assert _interrupts(app, config)

    await app.ainvoke(Command(resume={"type": "cancel"}), config, context=GlobalContext())
    assert _interrupts(app, config) == []
    # No connection saved; this avatar is marked declined, another is not.
    assert await bound_connection_for(store, "u", "a") is None
    assert await is_declined(store, "u", "a") is True
    assert await is_declined(store, "u", "other") is False


@pytest.mark.asyncio
async def test_discovery_skips_non_owned_avatar(discovery_app):
    app, _store = discovery_app
    # Avatar metadata owner differs from the conversing user ⇒ not owned ⇒ no offer.
    config = {
        "configurable": {
            "thread_id": "notowned-1",
            "user_id": "u",
            "assistant_id": "a",
            "assistant_ctx": {"metadata": {"user_id": "someone_else"}},
        }
    }
    await app.ainvoke(_input(), config, context=GlobalContext())
    assert _interrupts(app, config) == []
