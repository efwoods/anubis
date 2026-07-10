"""Unit tests for the MCP relay bridge, registration resolver, and Bearer auth.

Covers the API side of the local-daemon relay (``anubis-mcp-server-ubuntu``):

- the in-process relay registry correlates a ``proxy`` request with its
  ``proxy_response`` and decodes the body, fails cleanly when the device is
  offline, and times out when no response arrives;
- ``McpConnection`` round-trips the per-device secret through the store;
- ``build_mcp_client`` attaches ``Authorization: Bearer <device_secret>`` only
  when a secret is present;
- ``resolve_available_connection`` prefers a live relay registration, treats an
  offline relay registration as unavailable, and falls back to SSE discovery
  only when no registration exists.
"""

import asyncio
import json

import pytest
from langgraph.store.memory import InMemoryStore

import src.anubis.utils.tools.data_analysis.discovery as discovery
import src.anubis.utils.tools.data_analysis.relay as relay
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.tools.data_analysis import McpConnection
from src.anubis.utils.tools.data_analysis.backend import mcp_registration_namespace
from src.anubis.utils.tools.data_analysis.discovery import (
    REGISTRATION_KEY,
    bound_connection_for,
    resolve_available_connection,
    save_user_connection,
)


class _FakeWebSocket:
    """Records frames the API sends; the test drives responses back by hand."""

    def __init__(self):
        self.sent: list[dict] = []

    async def send_text(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


@pytest.fixture(autouse=True)
def _clear_registry():
    """Reset the module-level relay registry around every test."""
    relay._sessions_by_device.clear()
    relay._device_id_by_user.clear()
    yield
    relay._sessions_by_device.clear()
    relay._device_id_by_user.clear()


def _register(websocket, *, device_id="d1", user_id="u1", secret="mcp_dev_x"):
    return relay.register_session(
        device_id=device_id,
        user_id=user_id,
        device_secret=secret,
        server_name="Ubuntu-OS-Filesystem",
        allowed_roots=("/data",),
        websocket=websocket,
    )


def test_registry_presence_indexes_by_device_and_user():
    websocket = _FakeWebSocket()
    _register(websocket)
    assert relay.is_online("d1") is True
    assert relay.is_online("nope") is False
    assert relay.session_for_user("u1").device_id == "d1"

    relay.drop_session("d1", websocket)
    assert relay.is_online("d1") is False
    assert relay.session_for_user("u1") is None


def test_drop_session_ignores_superseded_socket():
    first = _FakeWebSocket()
    _register(first)
    second = _FakeWebSocket()
    _register(second)  # reconnect replaces the session
    # A late disconnect of the OLD socket must not evict the live reconnect.
    relay.drop_session("d1", first)
    assert relay.is_online("d1") is True
    assert relay.get_session("d1").websocket is second


def test_proxy_request_correlates_and_decodes():
    async def run():
        websocket = _FakeWebSocket()
        _register(websocket)

        task = asyncio.create_task(
            relay.proxy_request(
                "d1",
                method="POST",
                path=relay.LOCAL_MCP_PATH,
                headers={"content-type": "application/json"},
                body=b'{"hello":"world"}',
                timeout_seconds=5.0,
            )
        )
        # Let proxy_request send its frame, then answer with the matching id.
        await asyncio.sleep(0)
        frame = websocket.sent[0]
        assert frame["type"] == relay.FRAME_PROXY
        assert frame["path"] == relay.LOCAL_MCP_PATH
        relay.handle_incoming(
            "d1",
            {
                "type": relay.FRAME_PROXY_RESPONSE,
                "request_id": frame["request_id"],
                "status_code": 200,
                "headers": {"content-type": "application/json"},
                "body": '{"ok":true}',
                "body_encoding": "text",
            },
        )
        status, headers, body = await task
        assert status == 200
        assert headers["content-type"] == "application/json"
        assert body == b'{"ok":true}'

    asyncio.run(run())


def test_proxy_request_offline_raises():
    async def run():
        with pytest.raises(RuntimeError):
            await relay.proxy_request(
                "absent",
                method="POST",
                path="/mcp",
                headers={},
                body=b"",
                timeout_seconds=1.0,
            )

    asyncio.run(run())


def test_proxy_request_times_out_without_response():
    async def run():
        _register(_FakeWebSocket())
        with pytest.raises(TimeoutError):
            await relay.proxy_request(
                "d1",
                method="POST",
                path="/mcp",
                headers={},
                body=b"",
                timeout_seconds=0.05,
            )
        # The pending future is cleaned up, not leaked.
        assert relay.get_session("d1").pending_responses == {}

    asyncio.run(run())


def test_connection_round_trips_device_secret():
    connection = McpConnection(
        url="https://api.neuralnexus.site/mcp/relay/d1",
        transport="streamable_http",
        server_name="Ubuntu-OS-Filesystem",
        allowed_roots=("/data",),
        device_secret="mcp_dev_secret",
    )
    value = connection.to_store_value(assistant_id="a1")
    assert value["device_secret"] == "mcp_dev_secret"
    rebuilt = McpConnection.from_mapping(value)
    assert rebuilt.device_secret == "mcp_dev_secret"
    assert rebuilt.url == connection.url


def test_build_mcp_client_attaches_bearer_only_with_secret():
    client_mod = pytest.importorskip("langchain_mcp_adapters.client")
    from src.anubis.utils.tools.data_analysis.mcp_client import build_mcp_client

    captured: dict = {}

    class _FakeClient:
        def __init__(self, config):
            captured["config"] = config

    original = client_mod.MultiServerMCPClient
    client_mod.MultiServerMCPClient = _FakeClient
    try:
        with_secret = McpConnection(
            url="https://api.neuralnexus.site/mcp/relay/d1",
            transport="streamable_http",
            server_name="Ubuntu-OS-Filesystem",
            device_secret="mcp_dev_secret",
        )
        build_mcp_client(with_secret)
        server_config = captured["config"]["Ubuntu-OS-Filesystem"]
        assert server_config["headers"] == {
            "Authorization": "Bearer mcp_dev_secret"
        }

        without_secret = McpConnection(
            url="http://localhost:8000/mcp",
            transport="streamable_http",
            server_name="Ubuntu-OS-Filesystem",
        )
        build_mcp_client(without_secret)
        assert "headers" not in captured["config"]["Ubuntu-OS-Filesystem"]
    finally:
        client_mod.MultiServerMCPClient = original


def _put_registration(store, user_id, record):
    async def run():
        await store.aput(
            mcp_registration_namespace(user_id), REGISTRATION_KEY, record
        )

    asyncio.run(run())


def test_resolver_returns_relay_connection_when_online():
    store = InMemoryStore()
    _put_registration(
        store,
        "u1",
        {
            "status": "pending_consent",
            "connection_mode": "relay",
            "server_name": "Ubuntu-OS-Filesystem",
            "device_id": "d1",
            "device_secret": "mcp_dev_secret",
            "mcp_url": "https://api.neuralnexus.site/mcp/relay/d1",
            "allowed_roots": ["/data"],
            "last_seen_at": "2026-07-09T00:00:00+00:00",
        },
    )
    _register(_FakeWebSocket(), device_id="d1", user_id="u1", secret="mcp_dev_secret")

    connection = asyncio.run(
        resolve_available_connection(store, "u1", GlobalContext())
    )
    assert connection is not None
    # Live session wins: loopback bridge URL for THIS process, not the stale
    # production URL left in the registration record.
    assert connection.url == relay.bridge_url_for_device("d1")
    assert connection.device_secret == "mcp_dev_secret"


def test_resolver_prefers_live_session_over_stale_registration_device():
    """A new daemon under a different device_id must still be discoverable.

    Heartbeats used to refresh only ``last_seen_at`` on an older registration,
    so ``is_online(old_device_id)`` was false while the live socket belonged to
    the new device. Live ``session_for_user`` is the authoritative source.
    """
    store = InMemoryStore()
    _put_registration(
        store,
        "u1",
        {
            "status": "pending_consent",
            "connection_mode": "relay",
            "server_name": "Ubuntu-OS-Filesystem",
            "device_id": "stale-device",
            "device_secret": "old_secret",
            "mcp_url": "https://api.neuralnexus.site/mcp/relay/stale-device",
            "allowed_roots": ["/old"],
            "last_seen_at": "2026-07-09T00:00:00+00:00",
        },
    )
    _register(
        _FakeWebSocket(),
        device_id="live-device",
        user_id="u1",
        secret="new_secret",
    )

    connection = asyncio.run(
        resolve_available_connection(store, "u1", GlobalContext())
    )
    assert connection is not None
    assert connection.url == relay.bridge_url_for_device("live-device")
    assert connection.device_secret == "new_secret"
    assert connection.allowed_roots == ("/data",)


def test_bound_connection_refreshes_stale_url_from_live_relay():
    """Consent saved a host-only URL; live relay must replace it each turn."""

    async def run():
        store = InMemoryStore()
        stale = McpConnection(
            url="http://host.docker.internal:8000/mcp",
            transport="streamable_http",
            server_name="Ubuntu-OS-Filesystem",
            allowed_roots=("/old",),
            device_secret=None,
        )
        await save_user_connection(
            store, "u1", connection=stale, assistant_id="a1"
        )
        _register(
            _FakeWebSocket(),
            device_id="live-device",
            user_id="u1",
            secret="mcp_dev_live",
        )

        bound = await bound_connection_for(store, "u1", "a1")
        assert bound is not None
        assert bound.url == relay.bridge_url_for_device("live-device")
        assert bound.device_secret == "mcp_dev_live"
        # Store is rewritten so a brief relay blip still has a usable URL.
        saved = await store.aget(("u1", "mcp_connection"), "connection")
        assert saved is not None
        assert saved.value["url"] == bound.url
        assert saved.value["device_secret"] == "mcp_dev_live"

    asyncio.run(run())


def test_resolver_skips_relay_registration_when_offline():
    store = InMemoryStore()
    _put_registration(
        store,
        "u1",
        {
            "connection_mode": "relay",
            "device_id": "d1",
            "device_secret": "mcp_dev_secret",
            "mcp_url": "https://api.neuralnexus.site/mcp/relay/d1",
            "last_seen_at": "2026-07-09T00:00:00+00:00",
        },
    )
    # No live socket registered → the relay is offline → nothing offered, and we
    # do NOT fall through to the unrelated SSE endpoint.
    connection = asyncio.run(
        resolve_available_connection(store, "u1", GlobalContext())
    )
    assert connection is None


def test_resolver_falls_back_to_sse_without_registration(monkeypatch):
    store = InMemoryStore()
    sentinel = McpConnection(
        url="http://localhost:8000/mcp",
        transport="streamable_http",
        server_name="Ubuntu-OS-Filesystem",
    )

    async def _fake_discover(url, timeout, *, ignore_failure_backoff=False):
        return sentinel

    monkeypatch.setattr(discovery, "discover_announced_server", _fake_discover)
    connection = asyncio.run(
        resolve_available_connection(store, "u1", GlobalContext())
    )
    assert connection is sentinel
