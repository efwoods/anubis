"""Protect the ``mcp_auto_adopt`` node's resolve → bind flow.

Drives the real ``mcp_auto_adopt`` node through a minimal checkpointed outer
graph (same pattern as ``test_think_interrupt_flow``), substituting a fake
``resolve_available_connections`` so no live Model Context Protocol server is
needed.

Adoption is automatic and raises NO interrupt: a daemon registers using the
user's own API key, so the credential already proves the machine belongs to the
account, and four machines would otherwise mean four approval prompts. Covers:
every reachable machine is bound to the answering avatar; several machines are
adopted in one pass; adoption is idempotent across turns; a machine the user
explicitly disconnected stays suppressed while the user's other machines are
still adopted; and avatars that are not the user's own personal avatar are never
adopted onto.
"""

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore

import src.anubis.graph as graph_mod
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.state import GlobalState
from src.anubis.utils.tools.data_analysis import (
    McpConnection,
    bound_connections_for,
    is_declined,
    mark_declined,
)

_UBUNTU = McpConnection(
    url="http://localhost:8000/mcp/relay/d-ubuntu",
    transport="streamable_http",
    server_name="Ubuntu-OS-Filesystem",
    allowed_roots=("/home/user/data",),
    device_id="d-ubuntu",
    device_label="Ubuntu",
    platform="ubuntu",
)

_MACOS = McpConnection(
    url="http://localhost:8000/mcp/relay/d-macos",
    transport="streamable_http",
    server_name="macOS-Filesystem",
    allowed_roots=("/Users/evan/data",),
    device_id="d-macos",
    device_label="macOS",
    platform="macos",
)


def _adopt_app(monkeypatch, available):
    """A one-node graph running the real ``mcp_auto_adopt`` with a fake resolver."""

    async def _fake_resolve(store, user_id, context, *, ignore_failure_backoff=False):
        return list(available)

    monkeypatch.setattr(graph_mod, "resolve_available_connections", _fake_resolve)

    store = InMemoryStore()
    outer = StateGraph(GlobalState)
    outer.add_node("mcp_auto_adopt", graph_mod.mcp_auto_adopt)
    outer.add_edge(START, "mcp_auto_adopt")
    outer.add_edge("mcp_auto_adopt", END)
    app = outer.compile(checkpointer=MemorySaver(), store=store)
    return app, store


@pytest.fixture
def single_device_app(monkeypatch):
    return _adopt_app(monkeypatch, [_UBUNTU])


@pytest.fixture
def two_device_app(monkeypatch):
    return _adopt_app(monkeypatch, [_UBUNTU, _MACOS])


def _input(user_id="u", assistant_id="a") -> dict:
    return {
        "messages": [HumanMessage(content="hello")],
        "user_state": {"user_id": user_id},
        "assistant_state": {"assistant_id": assistant_id},
    }


def _owned_config(thread_id, user_id="u", assistant_id="a") -> dict:
    # Owner match AND the personal-avatar flag ⇒ the MCP capability is adopted.
    return {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            "assistant_id": assistant_id,
            "assistant_ctx": {
                "metadata": {
                    "user_id": user_id,
                    "is_personal_avatar_of_creator": True,
                }
            },
        }
    }


def _interrupts(app, config):
    snap = app.get_state(config)
    return [i for t in snap.tasks for i in t.interrupts] or list(
        getattr(snap, "interrupts", []) or []
    )


@pytest.mark.asyncio
async def test_reachable_device_is_adopted_without_any_interrupt(single_device_app):
    app, store = single_device_app
    config = _owned_config("adopt-1")

    await app.ainvoke(_input(), config, context=GlobalContext())

    # The consent interrupt is gone: registration with the user's own API key is
    # the authorization, so the turn completes in one pass.
    assert _interrupts(app, config) == []
    bound = await bound_connections_for(store, "u", "a")
    assert [connection.device_id for connection in bound] == ["d-ubuntu"]
    assert bound[0].device_label == "Ubuntu"
    # The user's other avatar shares the namespace but is not the bound avatar.
    assert await bound_connections_for(store, "u", "other") == []


@pytest.mark.asyncio
async def test_every_reachable_device_is_adopted_in_one_pass(two_device_app):
    app, store = two_device_app
    config = _owned_config("adopt-many-1")

    await app.ainvoke(_input(), config, context=GlobalContext())

    bound = await bound_connections_for(store, "u", "a")
    assert {connection.device_id for connection in bound} == {"d-ubuntu", "d-macos"}
    assert {connection.device_label for connection in bound} == {"Ubuntu", "macOS"}


@pytest.mark.asyncio
async def test_adoption_is_idempotent_across_turns(two_device_app):
    app, store = two_device_app

    await app.ainvoke(_input(), _owned_config("adopt-idem-1"), context=GlobalContext())
    first = await bound_connections_for(store, "u", "a")
    first_connected_at = sorted(
        (await store.asearch(("u", "mcp_connection"), limit=10)),
        key=lambda item: item.key,
    )[0].value["connected_at"]

    await app.ainvoke(_input(), _owned_config("adopt-idem-2"), context=GlobalContext())
    second = await bound_connections_for(store, "u", "a")
    second_connected_at = sorted(
        (await store.asearch(("u", "mcp_connection"), limit=10)),
        key=lambda item: item.key,
    )[0].value["connected_at"]

    assert len(second) == len(first) == 2
    # Already-bound devices are skipped rather than re-saved, so the adoption
    # timestamp is not churned on every conversation turn.
    assert second_connected_at == first_connected_at


@pytest.mark.asyncio
async def test_suppressed_device_is_not_readopted_but_others_are(two_device_app):
    app, store = two_device_app

    # The user explicitly disconnected the Mac; without a per-device suppression
    # marker, the next turn would silently bind the Mac again and the disconnect
    # would look broken.
    await mark_declined(store, "u", "a", _MACOS)

    await app.ainvoke(_input(), _owned_config("suppress-1"), context=GlobalContext())

    bound = await bound_connections_for(store, "u", "a")
    assert [connection.device_id for connection in bound] == ["d-ubuntu"]
    assert await is_declined(store, "u", "a", "d-macos") is True
    # Suppression is per avatar AND per device: neither the Ubuntu machine nor
    # the user's other avatars are affected.
    assert await is_declined(store, "u", "a", "d-ubuntu") is False
    assert await is_declined(store, "u", "other", "d-macos") is False


@pytest.mark.asyncio
async def test_adoption_skips_non_owned_avatar(two_device_app):
    app, store = two_device_app
    # Avatar metadata owner differs from the conversing user ⇒ not owned ⇒ never
    # adopted, so a visitor cannot reach the owner's machines.
    config = {
        "configurable": {
            "thread_id": "notowned-1",
            "user_id": "u",
            "assistant_id": "a",
            "assistant_ctx": {
                "metadata": {
                    "user_id": "someone_else",
                    "is_personal_avatar_of_creator": True,
                }
            },
        }
    }
    await app.ainvoke(_input(), config, context=GlobalContext())
    assert _interrupts(app, config) == []
    assert await bound_connections_for(store, "u", "a") == []


@pytest.mark.asyncio
async def test_adoption_skips_owned_non_personal_avatar(two_device_app):
    app, store = two_device_app
    # The user owns the avatar, but it is NOT flagged as their personal avatar ⇒
    # the desktop MCP capability is exclusive to the personal avatar ⇒ no adoption.
    config = {
        "configurable": {
            "thread_id": "notpersonal-1",
            "user_id": "u",
            "assistant_id": "a",
            "assistant_ctx": {"metadata": {"user_id": "u"}},
        }
    }
    await app.ainvoke(_input(), config, context=GlobalContext())
    assert _interrupts(app, config) == []
    assert await bound_connections_for(store, "u", "a") == []


@pytest.mark.asyncio
async def test_no_reachable_device_leaves_the_turn_untouched(monkeypatch):
    app, store = _adopt_app(monkeypatch, [])
    config = _owned_config("none-1")

    await app.ainvoke(_input(), config, context=GlobalContext())

    assert _interrupts(app, config) == []
    assert await bound_connections_for(store, "u", "a") == []
