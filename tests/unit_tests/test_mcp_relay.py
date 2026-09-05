"""Unit tests for the MCP relay bridge, registration resolver, and Bearer auth.

Covers the API side of the local-daemon relay (``anubis-mcp-server-ubuntu-desktop``
and its macOS / mobile / Windows siblings):

- the in-process relay registry holds MANY devices per user simultaneously,
  correlates a ``proxy`` request with its ``proxy_response`` and decodes the
  body, fails cleanly when a device is offline, and times out when no response
  arrives;
- dropping one device's session leaves the user's other devices connected — the
  regression that motivated device-keying, since one daemon's shutdown used to
  remove the record every daemon shared;
- ``McpConnection`` round-trips the per-device secret and identity through the
  store;
- ``build_mcp_client`` attaches ``Authorization: Bearer <device_secret>`` only
  when a secret is present;
- ``resolve_available_connections`` returns every reachable device, treats an
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
    bound_connections_for,
    resolve_available_connections,
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
    relay._device_ids_by_user.clear()
    yield
    relay._sessions_by_device.clear()
    relay._device_ids_by_user.clear()


def _register(
    websocket,
    *,
    device_id="d1",
    user_id="u1",
    secret="mcp_dev_x",
    device_label="Ubuntu",
    allowed_roots=("/data",),
):
    return relay.register_session(
        device_id=device_id,
        user_id=user_id,
        device_secret=secret,
        server_name="Ubuntu-OS-Filesystem",
        allowed_roots=allowed_roots,
        websocket=websocket,
        device_label=device_label,
    )


def test_registry_presence_indexes_by_device_and_user():
    websocket = _FakeWebSocket()
    _register(websocket)
    assert relay.is_online("d1") is True
    assert relay.is_online("nope") is False
    assert [s.device_id for s in relay.sessions_for_user("u1")] == ["d1"]
    relay.drop_session("d1", websocket)
    assert relay.sessions_for_user("u1") == []


def test_registry_holds_several_devices_for_one_user():
    """Four machines connected at once must all stay addressable."""
    sockets = {}
    for device_id, label in (
        ("d-ubuntu", "Ubuntu"),
        ("d-macos", "macOS"),
        ("d-mobile", "iPhone"),
        ("d-windows", "Windows"),
    ):
        sockets[device_id] = _FakeWebSocket()
        _register(
            sockets[device_id],
            device_id=device_id,
            user_id="u1",
            device_label=label,
        )

    sessions = relay.sessions_for_user("u1")
    assert len(sessions) == 4
    # Ordered by label so repeated questions produce a stable device order.
    assert [s.device_label for s in sessions] == [
        "Ubuntu",
        "Windows",
        "iPhone",
        "macOS",
    ]
    assert all(relay.is_online(device_id) for device_id in sockets)


def test_dropping_one_device_leaves_the_others_connected():
    """The production incident: one daemon stopping must not unplug the rest."""
    ubuntu_socket = _FakeWebSocket()
    macos_socket = _FakeWebSocket()
    _register(ubuntu_socket, device_id="d-ubuntu", user_id="u1", device_label="Ubuntu")
    _register(macos_socket, device_id="d-macos", user_id="u1", device_label="macOS")

    relay.drop_session("d-ubuntu", ubuntu_socket)

    assert relay.is_online("d-ubuntu") is False
    assert relay.is_online("d-macos") is True
    assert [s.device_id for s in relay.sessions_for_user("u1")] == ["d-macos"]


def test_sessions_are_isolated_between_users():
    _register(_FakeWebSocket(), device_id="d1", user_id="u1")
    _register(_FakeWebSocket(), device_id="d2", user_id="u2")
    assert [s.device_id for s in relay.sessions_for_user("u1")] == ["d1"]
    assert [s.device_id for s in relay.sessions_for_user("u2")] == ["d2"]


def test_drop_session_ignores_superseded_socket():
    first = _FakeWebSocket()
    _register(first, device_id="d1")
    second = _FakeWebSocket()
    _register(second, device_id="d1")
    # The late disconnect of the replaced socket must not evict the live one.
    relay.drop_session("d1", first)
    assert relay.is_online("d1") is True
    relay.drop_session("d1", second)
    assert relay.is_online("d1") is False


def test_proxy_request_correlates_and_decodes():
    async def run():
        websocket = _FakeWebSocket()
        _register(websocket, device_id="d1")

        async def respond_when_framed():
            for _ in range(100):
                if websocket.sent:
                    break
                await asyncio.sleep(0.01)
            request_id = websocket.sent[0]["request_id"]
            relay.handle_incoming(
                "d1",
                {
                    "type": relay.FRAME_PROXY_RESPONSE,
                    "request_id": request_id,
                    "status_code": 200,
                    "headers": {"content-type": "application/json"},
                    "body": '{"ok": true}',
                    "body_encoding": "text",
                },
            )

        responder = asyncio.create_task(respond_when_framed())
        status, headers, body = await relay.proxy_request(
            "d1",
            method="POST",
            path="/mcp",
            headers={"content-type": "application/json"},
            body=b'{"jsonrpc": "2.0"}',
            timeout_seconds=5.0,
        )
        await responder
        assert status == 200
        assert headers["content-type"] == "application/json"
        assert json.loads(body) == {"ok": True}

    asyncio.run(run())


def test_proxy_request_offline_raises():
    async def run():
        with pytest.raises(RuntimeError):
            await relay.proxy_request(
                "missing",
                method="POST",
                path="/mcp",
                headers={},
                body=b"",
                timeout_seconds=1.0,
            )

    asyncio.run(run())


def test_proxy_request_times_out_without_response():
    async def run():
        _register(_FakeWebSocket(), device_id="d1")
        with pytest.raises(TimeoutError):
            await relay.proxy_request(
                "d1",
                method="POST",
                path="/mcp",
                headers={},
                body=b"",
                timeout_seconds=0.05,
            )

    asyncio.run(run())


def test_connection_round_trips_device_identity_and_secret():
    connection = McpConnection(
        url="http://127.0.0.1:8000/mcp/relay/d1",
        transport="streamable_http",
        server_name="Ubuntu-OS-Filesystem",
        allowed_roots=("/data",),
        device_secret="mcp_dev_secret",
        device_id="d1",
        device_label="Ubuntu",
        platform="ubuntu",
    )
    stored = connection.to_store_value(assistant_id="a1")
    assert stored["device_secret"] == "mcp_dev_secret"
    assert stored["device_id"] == "d1"
    assert stored["device_label"] == "Ubuntu"
    assert stored["platform"] == "ubuntu"

    rebuilt = McpConnection.from_mapping(stored)
    assert rebuilt.device_secret == "mcp_dev_secret"
    assert rebuilt.device_id == "d1"
    assert rebuilt.device_label == "Ubuntu"
    assert rebuilt.platform == "ubuntu"


def test_connection_from_session_carries_device_identity():
    session = _register(_FakeWebSocket(), device_id="d-macos", device_label="macOS")
    connection = relay.connection_from_session(session)
    assert connection.device_id == "d-macos"
    assert connection.device_label == "macOS"
    assert connection.url == relay.bridge_url_for_device("d-macos")


def test_build_mcp_client_attaches_bearer_only_with_secret():
    # ``build_mcp_client`` imports ``MultiServerMCPClient`` lazily INSIDE the
    # function (the repository cold-start rule), so the fake must replace the
    # attribute on the SOURCE module the import reads from — patching our own
    # module would be shadowed by the function-local import.
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
        assert server_config["headers"] == {"Authorization": "Bearer mcp_dev_secret"}

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
            mcp_registration_namespace(user_id), record["device_id"], record
        )

    asyncio.run(run())


def _relay_registration(device_id, *, secret="mcp_dev_secret", label="Ubuntu"):
    return {
        "status": "pending_consent",
        "connection_mode": "relay",
        "server_name": "Ubuntu-OS-Filesystem",
        "device_id": device_id,
        "device_label": label,
        "platform": "ubuntu",
        "device_secret": secret,
        "mcp_url": f"https://api.neuralnexus.site/mcp/relay/{device_id}",
        "allowed_roots": ["/data"],
        "last_seen_at": "2026-07-09T00:00:00+00:00",
    }


def test_resolver_returns_relay_connection_when_online():
    store = InMemoryStore()
    _put_registration(store, "u1", _relay_registration("d1"))
    _register(_FakeWebSocket(), device_id="d1", user_id="u1", secret="mcp_dev_secret")

    connections = asyncio.run(
        resolve_available_connections(store, "u1", GlobalContext())
    )
    assert len(connections) == 1
    # Live session wins: loopback bridge URL for THIS process, not the stale
    # production URL left in the registration record.
    assert connections[0].url == relay.bridge_url_for_device("d1")
    assert connections[0].device_secret == "mcp_dev_secret"


def test_resolver_returns_every_online_device():
    """Two machines online at once must both be resolved, not just the first."""
    store = InMemoryStore()
    _put_registration(store, "u1", _relay_registration("d-ubuntu", label="Ubuntu"))
    _put_registration(store, "u1", _relay_registration("d-macos", label="macOS"))
    _register(
        _FakeWebSocket(), device_id="d-ubuntu", user_id="u1", device_label="Ubuntu"
    )
    _register(_FakeWebSocket(), device_id="d-macos", user_id="u1", device_label="macOS")

    connections = asyncio.run(
        resolve_available_connections(store, "u1", GlobalContext())
    )
    assert {connection.device_id for connection in connections} == {
        "d-ubuntu",
        "d-macos",
    }


def test_resolver_returns_only_the_online_half():
    """One machine asleep must not hide the machine that is awake."""
    store = InMemoryStore()
    _put_registration(store, "u1", _relay_registration("d-ubuntu", label="Ubuntu"))
    _put_registration(store, "u1", _relay_registration("d-macos", label="macOS"))
    _register(_FakeWebSocket(), device_id="d-macos", user_id="u1", device_label="macOS")

    connections = asyncio.run(
        resolve_available_connections(store, "u1", GlobalContext())
    )
    assert [connection.device_id for connection in connections] == ["d-macos"]


def test_bound_connection_refreshes_stale_url_from_live_relay():
    """Adoption saved a host-only URL; a live relay must replace it each turn."""

    async def run():
        store = InMemoryStore()
        stale = McpConnection(
            url="http://host.docker.internal:8000/mcp",
            transport="streamable_http",
            server_name="Ubuntu-OS-Filesystem",
            allowed_roots=("/old",),
            device_secret=None,
            device_id="live-device",
            device_label="Ubuntu",
        )
        await save_user_connection(store, "u1", connection=stale, assistant_id="a1")
        _register(
            _FakeWebSocket(),
            device_id="live-device",
            user_id="u1",
            secret="mcp_dev_live",
        )

        bound = await bound_connections_for(store, "u1", "a1")
        assert len(bound) == 1
        assert bound[0].url == relay.bridge_url_for_device("live-device")
        assert bound[0].device_secret == "mcp_dev_live"
        # Store is rewritten so a brief relay blip still has a usable URL.
        saved = await store.aget(("u1", "mcp_connection"), "live-device")
        assert saved is not None
        assert saved.value["url"] == bound[0].url
        assert saved.value["device_secret"] == "mcp_dev_live"

    asyncio.run(run())


def test_resolver_skips_relay_registration_when_offline():
    store = InMemoryStore()
    _put_registration(store, "u1", _relay_registration("d1"))
    # No live socket registered → the relay is offline → nothing offered, and we
    # do NOT fall through to the unrelated SSE endpoint.
    connections = asyncio.run(
        resolve_available_connections(store, "u1", GlobalContext())
    )
    assert connections == []


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
    connections = asyncio.run(
        resolve_available_connections(store, "u1", GlobalContext())
    )
    assert connections == [sentinel]


def test_legacy_singleton_registration_is_migrated_to_a_device_key():
    """Installations written before device-keying must keep working untouched."""

    async def run():
        store = InMemoryStore()
        namespace = mcp_registration_namespace("u1")
        await store.aput(
            namespace, discovery.LEGACY_REGISTRATION_KEY, _relay_registration("d-old")
        )

        records = await discovery.read_user_registrations(store, "u1")
        assert [record["device_id"] for record in records] == ["d-old"]
        # Re-keyed under the device, and the legacy key is gone so it can never
        # shadow a real per-device record again.
        assert await store.aget(namespace, "d-old") is not None
        assert await store.aget(namespace, discovery.LEGACY_REGISTRATION_KEY) is None

    asyncio.run(run())


def test_legacy_singleton_connection_recovers_device_id_from_relay_url():
    """A legacy connection record stored no device id; the URL still carries one."""

    async def run():
        store = InMemoryStore()
        namespace = ("u1", "mcp_connection")
        await store.aput(
            namespace,
            discovery.LEGACY_CONNECTION_KEY,
            {
                "status": "connected",
                "url": "https://api.neuralnexus.site/mcp/relay/d-legacy",
                "transport": "streamable_http",
                "server_name": "Ubuntu-OS-Filesystem",
                "allowed_roots": ["/data"],
                "assistant_id": "a1",
            },
        )

        records = await discovery.read_user_connections(store, "u1")
        assert [record["device_id"] for record in records] == ["d-legacy"]
        assert await store.aget(namespace, "d-legacy") is not None
        assert await store.aget(namespace, discovery.LEGACY_CONNECTION_KEY) is None

    asyncio.run(run())


def test_bound_relay_record_without_live_socket_is_marked_offline():
    """A relay-mode record with no socket in THIS process must not be dialed.

    Development and production share one Postgres store, so a record written by
    the other process carries the other process's loopback bridge address.
    Such a device is offline for this process: the connection is still returned
    (so the prompt can name the machine as offline) but with ``online=False`` so
    the think node withholds the tools instead of dialing a dead port.
    """

    async def run():
        store = InMemoryStore()
        other_process_bridge = McpConnection(
            url="http://127.0.0.1:8000/mcp/relay/prod-only-device",
            transport="streamable_http",
            server_name="Ubuntu-OS-Filesystem",
            allowed_roots=("/data",),
            device_secret="mcp_dev_prod",
            device_id="prod-only-device",
            device_label="linux-pc",
        )
        live = McpConnection(
            url="http://127.0.0.1:9600/mcp/relay/live-device",
            transport="streamable_http",
            server_name="Ubuntu-OS-Filesystem-dev",
            allowed_roots=("/data",),
            device_secret="mcp_dev_live",
            device_id="live-device",
            device_label="linux-pc-dev",
        )
        # A tunnel/local record has no relay socket by design; reachability is
        # only learned by dialing, so the record stays ``online``.
        tunnel = McpConnection(
            url="https://laptop.example.com/mcp",
            transport="streamable_http",
            server_name="macOS-Filesystem",
            allowed_roots=("/Users/me",),
            device_secret="mcp_dev_tunnel",
            device_id="tunnel-device",
            device_label="macOS",
        )
        for connection in (other_process_bridge, live, tunnel):
            await save_user_connection(
                store, "u1", connection=connection, assistant_id="a1"
            )
        _register(
            _FakeWebSocket(),
            device_id="live-device",
            user_id="u1",
            secret="mcp_dev_live",
        )

        bound = {
            connection.device_id: connection
            for connection in await bound_connections_for(store, "u1", "a1")
        }
        assert set(bound) == {"prod-only-device", "live-device", "tunnel-device"}
        assert bound["prod-only-device"].online is False
        assert bound["live-device"].online is True
        assert bound["tunnel-device"].online is True
        # The offline marker is derived, never written back to the store.
        saved = await store.aget(("u1", "mcp_connection"), "prod-only-device")
        assert saved is not None and "online" not in saved.value

    asyncio.run(run())
