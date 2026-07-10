"""MCP server discovery and per-user connection persistence.

The avatar does not read a hard-coded Model Context Protocol server URL. It
*discovers* a server: the installed filesystem server announces itself over
Server-Sent Events (``GET /discovery``), and this module subscribes to that
channel, reads the announced connection details, and — once the user consents
in chat — saves the connection so the personal avatar reuses it for every
future query.

Two persistence facts, both in the cross-thread LangGraph store:

- **The connection** is a *singleton per user*, bound to a *single avatar*
  (namespace ``(user_id, "mcp_connection")``). Only the bound avatar receives
  the data-analysis capability; the user's other avatars do not.
- **A decline** is recorded *per avatar* (namespace
  ``(user_id, assistant_id, "mcp_connection_declined")``) so declining on one
  avatar never suppresses the offer on the user's other avatars.

The heavy ``httpx``/``httpx_sse`` imports are deferred into the discovery
function per the repository cold-start rule.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from src.anubis.utils.tools.data_analysis.backend import (
    mcp_connection_declined_namespace,
    mcp_connection_namespace,
    mcp_registration_namespace,
)

logger = logging.getLogger(__name__)

# Fixed key names inside the per-user / per-avatar namespaces (each namespace
# holds exactly one record, so the key is a constant). ``CONNECTION_KEY`` is
# public so the API's disconnect endpoint (which uses the langgraph_sdk
# ``StoreClient`` HTTP API rather than the in-process ``BaseStore``) deletes
# the same key this module writes.
CONNECTION_KEY = "connection"
_DECLINED_KEY = "declined"
REGISTRATION_KEY = "registration"

# Timestamp of the most recent failed/empty discovery attempt, keyed by
# discovery URL, so an absent server is not re-dialed on every conversation
# turn. Mirrors the failure-backoff cache in ``mcp_client``.
_discovery_last_failure_monotonic: dict[str, float] = {}
_DISCOVERY_FAILURE_RETRY_SECONDS = 30.0


@dataclass(frozen=True)
class McpConnection:
    """A resolved connection to a Model Context Protocol filesystem server.

    Carries exactly what building the tool client needs. Constructed either
    from a discovery announcement or from a saved store record.
    """

    url: str
    transport: str
    server_name: str
    allowed_roots: tuple[str, ...] = ()
    # Per-device secret the API presents to the MCP server as
    # ``Authorization: Bearer <device_secret>`` on every tool call. Present for
    # relay/tunnel installs (the daemon generates it locally and registers it);
    # ``None`` for the co-located SSE-discovery dev path, which is unauthenticated.
    device_secret: str | None = None

    def to_store_value(self, *, assistant_id: str) -> dict[str, Any]:
        """Serialize as the per-user connection record bound to one avatar."""
        return {
            "status": "connected",
            "url": self.url,
            "transport": self.transport,
            "server_name": self.server_name,
            "allowed_roots": list(self.allowed_roots),
            "device_secret": self.device_secret,
            "assistant_id": assistant_id,
            "connected_at": datetime.now(UTC).isoformat(),
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> McpConnection:
        """Rebuild a connection from a store record or announcement payload."""
        return cls(
            url=value["url"],
            transport=value.get("transport", "streamable_http"),
            server_name=value.get("server_name", "Ubuntu-OS-Filesystem"),
            allowed_roots=tuple(value.get("allowed_roots", []) or []),
            device_secret=value.get("device_secret"),
        )


async def discover_announced_server(
    discovery_url: str,
    timeout_seconds: float,
    *,
    ignore_failure_backoff: bool = False,
) -> McpConnection | None:
    """Read one ``announce`` event from a server's discovery SSE channel.

    Returns the announced :class:`McpConnection`, or ``None`` when no server
    answers within ``timeout_seconds`` (server not installed / not running) —
    the caller treats ``None`` as "nothing to offer this turn". A failure is
    remembered briefly so an absent server is not re-dialed every turn;
    pass ``ignore_failure_backoff=True`` for an explicit user "connect"
    request, which should always dial (the user may have just started the
    server).
    """
    last_failure = _discovery_last_failure_monotonic.get(discovery_url)
    if (
        not ignore_failure_backoff
        and last_failure is not None
        and (time.monotonic() - last_failure) < _DISCOVERY_FAILURE_RETRY_SECONDS
    ):
        return None

    import httpx
    from httpx_sse import aconnect_sse

    async def _read_announcement() -> McpConnection | None:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            async with aconnect_sse(client, "GET", discovery_url) as event_source:
                async for server_sent_event in event_source.aiter_sse():
                    if server_sent_event.event == "announce":
                        payload = json.loads(server_sent_event.data)
                        return McpConnection.from_mapping(payload)
        return None

    try:
        # A single overall deadline so a reachable-but-silent server cannot
        # stall the conversation turn.
        connection = await asyncio.wait_for(
            _read_announcement(), timeout=timeout_seconds
        )
    except Exception:
        _discovery_last_failure_monotonic[discovery_url] = time.monotonic()
        logger.info(
            "No Model Context Protocol server announced at %s; "
            "no connection offered this turn.",
            discovery_url,
            exc_info=True,
        )
        return None

    if connection is None:
        _discovery_last_failure_monotonic[discovery_url] = time.monotonic()
        return None

    _discovery_last_failure_monotonic.pop(discovery_url, None)
    logger.info(
        "Discovered Model Context Protocol server %r at %s (tools url %s)",
        connection.server_name,
        discovery_url,
        connection.url,
    )
    return connection


async def read_user_registration(store: Any, user_id: str) -> dict[str, Any] | None:
    """Return the user's pending MCP daemon registration record, or ``None``.

    Written by the daemon's ``POST /mcp/register`` call (see ``webapp.py``);
    read by :func:`resolve_available_connection` to decide whether a server is
    reachable this turn without dialing the co-located SSE endpoint.
    """
    if store is None:
        return None
    item = await store.aget(mcp_registration_namespace(user_id), REGISTRATION_KEY)
    return None if item is None else (item.value or None)


def _connection_from_registration(record: dict[str, Any]) -> McpConnection:
    """Build a client :class:`McpConnection` from a stored registration record.

    Every connection mode (relay / tunnel / local) is driven as a
    ``streamable_http`` client: in relay mode ``mcp_url`` is this API's own
    ``/mcp/relay/<device_id>`` bridge, in tunnel/local it is the directly
    reachable server URL. The device secret rides along as the Bearer credential.
    """
    return McpConnection(
        url=record["mcp_url"],
        transport="streamable_http",
        server_name=record.get("server_name", "Ubuntu-OS-Filesystem"),
        allowed_roots=tuple(record.get("allowed_roots", []) or []),
        device_secret=record.get("device_secret"),
    )


def _registration_is_fresh(record: dict[str, Any], stale_seconds: float) -> bool:
    """Whether a registration's heartbeat is recent enough to trust as online.

    Used for the tunnel/local modes, which have no live socket to prove
    presence; the daemon heartbeats every ~30s via ``POST /mcp/heartbeat``.
    """
    last_seen = record.get("last_seen_at")
    if not last_seen:
        return False
    try:
        last_seen_at = datetime.fromisoformat(last_seen)
    except ValueError:
        return False
    if last_seen_at.tzinfo is None:
        last_seen_at = last_seen_at.replace(tzinfo=UTC)
    return (datetime.now(UTC) - last_seen_at).total_seconds() <= stale_seconds


async def resolve_available_connection(
    store: Any,
    user_id: str,
    context: Any,
    *,
    ignore_failure_backoff: bool = False,
) -> McpConnection | None:
    """Find a currently-reachable MCP server for a user, or ``None``.

    Resolution order, replacing the old single global SSE dial so that remote
    (relay/tunnel) installs work, while co-located dev still functions:

    1. **Pending registration** (the daemon pushed its presence to this API).
       For ``relay`` mode the daemon holds a live outbound socket, so presence
       is confirmed by the in-process relay registry; for ``tunnel``/``local``
       modes presence is inferred from a fresh heartbeat. Either way the stored
       ``mcp_url`` (relay bridge, or a directly reachable URL) becomes the
       connection.
    2. **SSE fallback** — only when no registration exists — dials the
       co-located discovery endpoint (``discover_announced_server``) for the
       same-machine development flow.
    """
    record = await read_user_registration(store, user_id)
    if record is not None:
        connection_mode = record.get("connection_mode", "relay")
        device_id = record.get("device_id")
        if connection_mode == "relay":
            from src.anubis.utils.tools.data_analysis import relay

            if relay.is_online(device_id):
                return _connection_from_registration(record)
            # Registered but the outbound socket is down: the server is not
            # reachable this turn. Do not fall through to the unrelated SSE
            # endpoint — report nothing available.
            return None
        stale_seconds = float(
            getattr(context, "data_analysis_registration_stale_seconds", 120.0)
        )
        if _registration_is_fresh(record, stale_seconds):
            return _connection_from_registration(record)
        return None

    return await discover_announced_server(
        context.data_analysis_mcp_discovery_url,
        float(context.data_analysis_discovery_timeout_seconds),
        ignore_failure_backoff=ignore_failure_backoff,
    )


async def read_user_connection(store: Any, user_id: str) -> dict[str, Any] | None:
    """Return the user's single saved connection record, or ``None``."""
    if store is None:
        return None
    item = await store.aget(mcp_connection_namespace(user_id), CONNECTION_KEY)
    return None if item is None else (item.value or None)


async def save_user_connection(
    store: Any,
    user_id: str,
    *,
    connection: McpConnection,
    assistant_id: str,
) -> None:
    """Save the user's single connection, bound to exactly one avatar.

    Overwrites any prior record: a user has at most one connection, and
    connecting a different avatar re-binds that single record.
    """
    await store.aput(
        mcp_connection_namespace(user_id),
        key=CONNECTION_KEY,
        value=connection.to_store_value(assistant_id=assistant_id),
    )


async def clear_user_connection(store: Any, user_id: str) -> bool:
    """Delete the user's saved connection (disconnect). Report whether one existed."""
    if store is None:
        return False
    existing = await store.aget(mcp_connection_namespace(user_id), CONNECTION_KEY)
    if existing is None:
        return False
    await store.adelete(mcp_connection_namespace(user_id), CONNECTION_KEY)
    return True


async def mark_declined(
    store: Any, user_id: str, assistant_id: str, connection: McpConnection
) -> None:
    """Record that this avatar declined the offer, so it stops being asked."""
    await store.aput(
        mcp_connection_declined_namespace(user_id, assistant_id),
        key=_DECLINED_KEY,
        value={
            "declined_at": datetime.now(UTC).isoformat(),
            "server_url": connection.url,
        },
    )


async def is_declined(store: Any, user_id: str, assistant_id: str) -> bool:
    """Report whether this avatar previously declined the connection offer."""
    if store is None:
        return False
    item = await store.aget(
        mcp_connection_declined_namespace(user_id, assistant_id), _DECLINED_KEY
    )
    return item is not None


async def clear_declined(store: Any, user_id: str, assistant_id: str) -> None:
    """Remove the decline marker so automatic offers resume for this avatar.

    Called when the user explicitly connects: a past decline only means
    "stop offering automatically" — an explicit connect always wins.
    """
    if store is None:
        return
    await store.adelete(
        mcp_connection_declined_namespace(user_id, assistant_id), _DECLINED_KEY
    )


async def bound_connection_for(
    store: Any, user_id: str, assistant_id: str
) -> McpConnection | None:
    """Return the connection iff it is established AND bound to this avatar.

    This is the sole gate for the data-analysis capability: no environment
    switch, no per-avatar enable flag — only a saved, consented connection
    whose bound avatar matches the avatar currently answering.
    """
    record = await read_user_connection(store, user_id)
    if record is None or record.get("status") != "connected":
        return None
    if record.get("assistant_id") != assistant_id:
        return None
    return McpConnection.from_mapping(record)
