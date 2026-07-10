"""In-process registry + HTTP-over-WebSocket bridge for relay-mode MCP servers.

The local Neural Nexus MCP daemon (``anubis-mcp-server-ubuntu``) exposes no
inbound port. In its default ``relay`` connection mode the daemon opens ONE
outbound WebSocket to this API (path ``/mcp/relay``) and tunnels HTTP over that
single socket:

- the API sends a ``proxy`` frame carrying one HTTP request
  (``method`` / ``path`` / ``headers`` / ``body``);
- the daemon replays that request against its own ``127.0.0.1`` MCP server and
  returns a ``proxy_response`` frame (``status_code`` / ``headers`` / ``body``).

This module is the API side of that tunnel. It holds a process-local registry
mapping each connected ``device_id`` to the live :class:`RelaySession` (the
WebSocket, the owning ``user_id``, the device secret, and the announced server
metadata), plus a ``user_id -> device_id`` index so a conversation turn can
find the caller's own device. :func:`proxy_request` frames one HTTP call, awaits
the correlated ``proxy_response`` through an ``asyncio.Future``, and returns the
reconstructed ``(status, headers, body)``.

Single-process assumption: ``langgraph.json`` mounts this FastAPI application
(``webapp.py:app``) and the graph (``graph.py:graph``) in one server process, so
the ``/mcp/relay`` WebSocket handler and the graph nodes that call
:func:`proxy_request` share this module-level registry — the same assumption the
sibling ``mcp_client`` / ``discovery`` module-level caches already rely on. A
multi-replica deployment would need a shared broker plus sticky routing of a
device's socket to the replica serving its turns (documented limitation, not
implemented here).

The frame ``type`` names below MUST match the daemon's ``src/daemon/relay.py``
in ``anubis-mcp-server-ubuntu``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Relay WebSocket frame type names — fixed contract with the local daemon.
FRAME_REGISTER = "register"
FRAME_REGISTERED = "registered"
FRAME_PROXY = "proxy"
FRAME_PROXY_RESPONSE = "proxy_response"
FRAME_PING = "ping"
FRAME_PONG = "pong"

# The daemon forwards a ``proxy`` frame to ``local_mcp_url + path``; the local
# FastMCP streamable-HTTP endpoint is served at ``/mcp`` (the daemon's default,
# and ``MCP_PATH`` in its settings). The register frame carries ``local_mcp_url``
# (host only) but not the path, so the bridge always targets this path.
LOCAL_MCP_PATH = "/mcp"


@dataclass
class RelaySession:
    """One live outbound relay socket from a user's local MCP daemon."""

    device_id: str
    user_id: str
    device_secret: str
    server_name: str
    allowed_roots: tuple[str, ...]
    websocket: Any
    last_seen_monotonic: float = field(default_factory=time.monotonic)
    # Correlation table: ``request_id -> Future`` awaiting a ``proxy_response``.
    pending_responses: dict[str, asyncio.Future] = field(default_factory=dict)


# Module-level registry (see module docstring for the single-process rationale).
_sessions_by_device: dict[str, RelaySession] = {}
_device_id_by_user: dict[str, str] = {}


def register_session(
    *,
    device_id: str,
    user_id: str,
    device_secret: str,
    server_name: str,
    allowed_roots: tuple[str, ...],
    websocket: Any,
) -> RelaySession:
    """Record a freshly connected daemon socket, replacing any prior one.

    A user runs at most one active MCP device per socket; a reconnect (new
    socket, same ``device_id``) supersedes the old session, whose pending
    requests are failed so their callers degrade rather than hang.
    """
    existing = _sessions_by_device.get(device_id)
    if existing is not None and existing.websocket is not websocket:
        _fail_pending(existing, RuntimeError("relay session superseded by reconnect"))

    session = RelaySession(
        device_id=device_id,
        user_id=user_id,
        device_secret=device_secret,
        server_name=server_name,
        allowed_roots=allowed_roots,
        websocket=websocket,
    )
    _sessions_by_device[device_id] = session
    _device_id_by_user[user_id] = device_id
    logger.info(
        "Relay session registered: device=%s user=%s server=%r roots=%d",
        device_id,
        user_id,
        server_name,
        len(allowed_roots),
    )
    return session


def drop_session(device_id: str, websocket: Any) -> None:
    """Remove a session on disconnect — but only if the socket still matches.

    The socket guard prevents a late disconnect of a superseded socket from
    evicting the live reconnect that already replaced the session.
    """
    session = _sessions_by_device.get(device_id)
    if session is None or session.websocket is not websocket:
        return
    _fail_pending(session, RuntimeError("relay session closed"))
    del _sessions_by_device[device_id]
    if _device_id_by_user.get(session.user_id) == device_id:
        del _device_id_by_user[session.user_id]
    logger.info("Relay session dropped: device=%s user=%s", device_id, session.user_id)


def _fail_pending(session: RelaySession, error: Exception) -> None:
    for future in session.pending_responses.values():
        if not future.done():
            future.set_exception(error)
    session.pending_responses.clear()


def get_session(device_id: str) -> RelaySession | None:
    """Return the live session for a device, or ``None`` when it is offline."""
    return _sessions_by_device.get(device_id)


def session_for_user(user_id: str) -> RelaySession | None:
    """Return the live session for a user's registered device, or ``None``."""
    device_id = _device_id_by_user.get(user_id)
    return None if device_id is None else _sessions_by_device.get(device_id)


def is_online(device_id: str | None) -> bool:
    """Whether a device currently holds a live relay socket."""
    return bool(device_id) and device_id in _sessions_by_device


def bridge_url_for_device(device_id: str) -> str:
    """Loopback URL for this process's ``/mcp/relay/{device_id}`` bridge.

    The avatar graph and the FastAPI relay bridge share one process, so tool
    calls must hit this container's own listen port — not a host-only address
    like ``host.docker.internal`` and not a stale production URL left in the
    store from an earlier daemon registration.
    """
    port = os.environ.get("PORT") or "8000"
    return f"http://127.0.0.1:{port}/mcp/relay/{device_id}"


def connection_from_session(session: RelaySession):
    """Build an :class:`~discovery.McpConnection` from a live relay session.

    Imported lazily to avoid a circular import with ``discovery``.
    """
    from src.anubis.utils.tools.data_analysis.discovery import McpConnection

    return McpConnection(
        url=bridge_url_for_device(session.device_id),
        transport="streamable_http",
        server_name=session.server_name,
        allowed_roots=session.allowed_roots,
        device_secret=session.device_secret,
    )


def handle_incoming(device_id: str, message: dict[str, Any]) -> None:
    """Route a frame the daemon sent us (called by the WebSocket read loop).

    Only ``proxy_response`` (fulfil the awaiting future) and ``pong`` (liveness)
    are meaningful here; anything else is ignored. The ``register`` frame is
    consumed by the endpoint before this loop starts.
    """
    session = _sessions_by_device.get(device_id)
    if session is None:
        return
    session.last_seen_monotonic = time.monotonic()

    message_type = message.get("type")
    if message_type == FRAME_PROXY_RESPONSE:
        request_id = message.get("request_id")
        future = session.pending_responses.pop(request_id, None)
        if future is not None and not future.done():
            future.set_result(message)
    elif message_type == FRAME_PONG:
        return


def _needs_base64(content: bytes) -> bool:
    """Whether raw bytes must be base64-framed (mirrors the daemon's check)."""
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


async def proxy_request(
    device_id: str,
    *,
    method: str,
    path: str,
    headers: dict[str, str],
    body: bytes,
    timeout_seconds: float,
) -> tuple[int, dict[str, str], bytes]:
    """Tunnel one HTTP request to the device and await its response.

    Frames the request as a ``proxy`` message, sends it over the device's
    socket, and waits (up to ``timeout_seconds``) for the matching
    ``proxy_response``. Returns ``(status_code, headers, body_bytes)``.

    Raises:
        RuntimeError: the device is offline.
        TimeoutError: no ``proxy_response`` arrived before the deadline.
    """
    session = _sessions_by_device.get(device_id)
    if session is None:
        raise RuntimeError(f"MCP relay device {device_id} is offline.")

    request_id = uuid.uuid4().hex
    if body:
        body_encoding = "base64" if _needs_base64(body) else "text"
        body_out = (
            base64.b64encode(body).decode("ascii")
            if body_encoding == "base64"
            else body.decode("utf-8", errors="replace")
        )
    else:
        body_encoding = "text"
        body_out = ""

    frame = {
        "type": FRAME_PROXY,
        "request_id": request_id,
        "method": method,
        "path": path,
        "headers": headers,
        "body": body_out,
        "body_encoding": body_encoding,
    }

    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    session.pending_responses[request_id] = future
    try:
        await session.websocket.send_text(json.dumps(frame))
        response = await asyncio.wait_for(future, timeout=timeout_seconds)
    finally:
        session.pending_responses.pop(request_id, None)

    status_code = int(response.get("status_code", 502))
    response_headers = {
        str(key): str(value) for key, value in (response.get("headers") or {}).items()
    }
    raw_body = response.get("body") or ""
    if response.get("body_encoding") == "base64":
        response_body = base64.b64decode(raw_body)
    else:
        response_body = raw_body.encode("utf-8")
    return status_code, response_headers, response_body
