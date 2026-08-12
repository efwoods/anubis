"""Model Context Protocol device discovery and per-device connection persistence.

The avatar does not read a hard-coded Model Context Protocol server URL. It
*discovers* servers: each Neural Nexus daemon announces itself — by opening an
outbound relay socket, by pushing a registration, or (co-located development
only) over a Server-Sent-Events channel — and this module persists the resulting
connections so the personal avatar reuses them on every future turn.

Three persistence facts, all in the cross-thread LangGraph store:

- **Connections** live in namespace ``(user_id, "mcp_connection")``, keyed by
  ``device_id`` — one record per machine. Each record names the single avatar
  the device is bound to; only that avatar receives the data-analysis
  capability.
- **Registrations** live in namespace ``(user_id, "mcp_registration")``, also
  keyed by ``device_id``.
- **Auto-adopt suppression** lives in namespace
  ``(user_id, assistant_id, "mcp_connection_declined")``, keyed by ``device_id``,
  so disconnecting the phone from one avatar affects neither the desktop nor the
  user's other avatars.

Every namespace previously held a single record under a constant key. That
singleton is why a second machine silently replaced the first, and why a
development daemon deleted the production registration in a shared store. The
constant keys survive only as :data:`LEGACY_CONNECTION_KEY` and
:data:`LEGACY_REGISTRATION_KEY`, which :func:`read_user_connections` and
:func:`read_user_registrations` migrate to device keys on first read so existing
installations need no manual step.

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
from src.anubis.utils.tools.data_analysis.devices import (
    UNKNOWN_DEVICE_LABEL,
    UNKNOWN_PLATFORM,
)

logger = logging.getLogger(__name__)

# The constant keys these namespaces used while each held exactly one record.
# Retained ONLY so a record written before device-keying is migrated on read;
# nothing writes these keys any more.
LEGACY_CONNECTION_KEY = "connection"
LEGACY_REGISTRATION_KEY = "registration"

# Backwards-compatible aliases. ``webapp.py`` and the daemon-facing endpoints
# imported these names while the records were singletons; keeping the aliases
# avoids a flag-day rename across modules that no longer use a constant key.
CONNECTION_KEY = LEGACY_CONNECTION_KEY
REGISTRATION_KEY = LEGACY_REGISTRATION_KEY

# Upper bound on records read from one namespace in a single search. A user with
# more devices than this has hit ``data_analysis_max_devices_per_user`` many
# times over; the bound only protects the store call from an unbounded scan.
_DEVICE_SEARCH_LIMIT = 100

# Timestamp of the most recent failed/empty discovery attempt, keyed by
# discovery URL, so an absent server is not re-dialed on every conversation
# turn. Mirrors the failure-backoff cache in ``mcp_client``.
_discovery_last_failure_monotonic: dict[str, float] = {}
_DISCOVERY_FAILURE_RETRY_SECONDS = 30.0


@dataclass(frozen=True)
class McpConnection:
    """A resolved connection to one device's Model Context Protocol server.

    Carries exactly what building the tool client needs, plus the device
    identity the conversation needs in order to attribute a result to a machine
    ("your Ubuntu desktop") rather than to an opaque device token. Constructed
    either from a live relay session, a stored record, or a discovery
    announcement.
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
    # Device identity. ``device_id`` is the store key for this connection, so a
    # connection without one cannot be saved; the SSE development path supplies
    # a synthetic identifier (see :func:`discover_announced_server`).
    device_id: str = ""
    device_label: str = UNKNOWN_DEVICE_LABEL
    platform: str = UNKNOWN_PLATFORM

    def to_store_value(self, *, assistant_id: str) -> dict[str, Any]:
        """Serialize as this device's connection record, bound to one avatar."""
        return {
            "status": "connected",
            "url": self.url,
            "transport": self.transport,
            "server_name": self.server_name,
            "allowed_roots": list(self.allowed_roots),
            "device_secret": self.device_secret,
            "device_id": self.device_id,
            "device_label": self.device_label,
            "platform": self.platform,
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
            device_id=value.get("device_id") or "",
            device_label=value.get("device_label") or UNKNOWN_DEVICE_LABEL,
            platform=value.get("platform") or UNKNOWN_PLATFORM,
        )


def device_id_from_relay_url(url: str) -> str:
    """Recover a device identifier from a ``/mcp/relay/<device_id>`` bridge URL.

    Used only when migrating a legacy singleton connection record, which was
    written before ``device_id`` was stored alongside the connection. The relay
    bridge URL always ends in the device identifier, so the identifier is
    recoverable without contacting the daemon.

    Returns an empty string for a non-relay URL (the co-located development
    path), whose record the caller keys synthetically instead.
    """
    marker = "/mcp/relay/"
    if marker not in url:
        return ""
    return url.rsplit(marker, 1)[-1].strip("/")


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

    This is the co-located development path, which predates device identity: the
    announcement carries no ``device_id``, so one is synthesized from the
    announced URL. That keeps the record device-keyed like every other record
    while remaining stable across turns for the same announced server.
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
    connection = _ensure_device_identity(connection)
    logger.info(
        "Discovered Model Context Protocol server %r at %s (tools url %s)",
        connection.server_name,
        discovery_url,
        connection.url,
    )
    return connection


def _ensure_device_identity(connection: McpConnection) -> McpConnection:
    """Fill in a device identifier and label for an announcement that lacks both.

    Announcements from the co-located Server-Sent-Events development path carry
    no device identity. A store record must still be device-keyed, so derive a
    stable identifier from the announced URL — stable meaning the same announced
    server always maps to the same key, rather than accumulating one record per
    conversation turn.
    """
    import hashlib

    from src.anubis.utils.tools.data_analysis.devices import derive_device_identity

    device_id = connection.device_id or device_id_from_relay_url(connection.url)
    if not device_id:
        device_id = (
            "announced-" + hashlib.sha1(connection.url.encode("utf-8")).hexdigest()[:12]
        )

    device_label, platform = derive_device_identity(
        {
            "device_label": (
                connection.device_label
                if connection.device_label != UNKNOWN_DEVICE_LABEL
                else ""
            ),
            "platform": (
                connection.platform if connection.platform != UNKNOWN_PLATFORM else ""
            ),
            "server_name": connection.server_name,
        }
    )
    return McpConnection(
        url=connection.url,
        transport=connection.transport,
        server_name=connection.server_name,
        allowed_roots=connection.allowed_roots,
        device_secret=connection.device_secret,
        device_id=device_id,
        device_label=device_label,
        platform=platform,
    )


async def _search_namespace(store: Any, namespace: tuple[str, ...]) -> list[Any]:
    """List every item in a namespace, tolerating a store that has none."""
    if store is None:
        return []
    try:
        return list(await store.asearch(namespace, limit=_DEVICE_SEARCH_LIMIT))
    except Exception:
        logger.warning("Could not search store namespace %s", namespace, exc_info=True)
        return []


async def _read_device_keyed_records(
    store: Any,
    namespace: tuple[str, ...],
    legacy_key: str,
) -> list[dict[str, Any]]:
    """Return every record in a device-keyed namespace, migrating legacy items.

    A record written before device-keying sits under ``legacy_key``. Migration
    re-writes that record under its own ``device_id`` and deletes the legacy
    key, so the singleton disappears the first time any code path reads the
    namespace and no operator step is needed. A legacy record whose value has no
    ``device_id`` recovers one from its relay bridge URL.
    """
    records: list[dict[str, Any]] = []
    for item in await _search_namespace(store, namespace):
        value = item.value or {}
        if item.key != legacy_key:
            records.append(value)
            continue

        device_id = value.get("device_id") or device_id_from_relay_url(
            str(value.get("url") or value.get("mcp_url") or "")
        )
        if not device_id:
            # Nothing identifies this device, so re-keying is impossible. Drop
            # the record rather than leave a singleton that would keep shadowing
            # real per-device records; the daemon re-registers within one
            # heartbeat interval.
            logger.info(
                "Discarding unidentifiable legacy record in namespace %s", namespace
            )
            await store.adelete(namespace, legacy_key)
            continue

        value["device_id"] = device_id
        await store.aput(namespace, key=device_id, value=value)
        await store.adelete(namespace, legacy_key)
        logger.info(
            "Migrated legacy singleton record in namespace %s to device key %s",
            namespace,
            device_id,
        )
        records.append(value)
    return records


async def read_user_registrations(store: Any, user_id: str) -> list[dict[str, Any]]:
    """Return every daemon registration record for a user, one per device.

    Written by each daemon's ``POST /mcp/register`` call (see ``webapp.py``);
    read by :func:`resolve_available_connections` to decide which devices are
    reachable this turn without dialing the co-located SSE endpoint.
    """
    return await _read_device_keyed_records(
        store, mcp_registration_namespace(user_id), LEGACY_REGISTRATION_KEY
    )


async def read_user_connections(store: Any, user_id: str) -> list[dict[str, Any]]:
    """Return every saved connection record for a user, one per device."""
    return await _read_device_keyed_records(
        store, mcp_connection_namespace(user_id), LEGACY_CONNECTION_KEY
    )


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
        device_id=record.get("device_id") or "",
        device_label=record.get("device_label") or UNKNOWN_DEVICE_LABEL,
        platform=record.get("platform") or UNKNOWN_PLATFORM,
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


async def resolve_available_connections(
    store: Any,
    user_id: str,
    context: Any,
    *,
    ignore_failure_backoff: bool = False,
) -> list[McpConnection]:
    """Find every currently-reachable device for a user.

    Resolution order, applied across all of the user's devices rather than
    stopping at the first hit:

    1. **Live relay sockets** (authoritative). The in-process registry wins over
       a stale store registration — for example when a daemon reconnects under a
       new ``device_id`` while heartbeats only refreshed ``last_seen_at`` on an
       older record.
    2. **Registrations for devices with no live socket.** In ``relay`` mode a
       device without a live socket is simply offline this turn, because the
       socket is the only path to that machine. In ``tunnel``/``local`` mode
       presence is inferred from a fresh heartbeat and the stored ``mcp_url``
       becomes the connection.
    3. **SSE fallback** — only when the user has no registrations and no live
       sockets at all — dials the co-located discovery endpoint for the
       same-machine development flow.

    Returns an empty list when nothing is reachable.
    """
    from src.anubis.utils.tools.data_analysis import relay

    connections: list[McpConnection] = []
    seen_device_ids: set[str] = set()

    for session in relay.sessions_for_user(user_id):
        connections.append(relay.connection_from_session(session))
        seen_device_ids.add(session.device_id)

    records = await read_user_registrations(store, user_id)
    for record in records:
        device_id = record.get("device_id") or ""
        if device_id and device_id in seen_device_ids:
            continue
        connection_mode = record.get("connection_mode", "relay")
        if connection_mode == "relay":
            # Registered but the outbound socket is down: this machine is not
            # reachable this turn. Every other device is still considered.
            continue
        stale_seconds = float(
            getattr(context, "data_analysis_registration_stale_seconds", 120.0)
        )
        if _registration_is_fresh(record, stale_seconds):
            connections.append(_connection_from_registration(record))
            if device_id:
                seen_device_ids.add(device_id)

    if connections or records:
        return connections

    announced = await discover_announced_server(
        context.data_analysis_mcp_discovery_url,
        float(context.data_analysis_discovery_timeout_seconds),
        ignore_failure_backoff=ignore_failure_backoff,
    )
    return [announced] if announced is not None else []


async def save_user_connection(
    store: Any,
    user_id: str,
    *,
    connection: McpConnection,
    assistant_id: str,
) -> None:
    """Save one device's connection, bound to exactly one avatar.

    Keyed by the connection's ``device_id``, so saving a second machine adds a
    record rather than replacing the first. Re-saving the same device overwrites
    only that device's record, which is how the stale-URL self-heal in
    :func:`bound_connections_for` refreshes a relay bridge address.
    """
    if not connection.device_id:
        raise ValueError(
            "Cannot save a Model Context Protocol connection without a device "
            "identifier; the record key is the device identifier."
        )
    await store.aput(
        mcp_connection_namespace(user_id),
        key=connection.device_id,
        value=connection.to_store_value(assistant_id=assistant_id),
    )


async def clear_user_connection(
    store: Any, user_id: str, device_id: str | None = None
) -> list[str]:
    """Delete saved connections (disconnect). Report which devices were removed.

    Args:
        store: The cross-thread store.
        user_id: Owner of the connections.
        device_id: Remove only this device's connection. When omitted, every
            connection the user has is removed — the "disconnect everything"
            request.

    Returns:
        The device identifiers whose connections were deleted, so the caller can
        name the affected machines back to the user.
    """
    if store is None:
        return []
    namespace = mcp_connection_namespace(user_id)
    records = await read_user_connections(store, user_id)
    removed: list[str] = []
    for record in records:
        record_device_id = record.get("device_id") or ""
        if device_id is not None and record_device_id != device_id:
            continue
        if not record_device_id:
            continue
        await store.adelete(namespace, record_device_id)
        removed.append(record_device_id)
    return removed


async def mark_declined(
    store: Any, user_id: str, assistant_id: str, connection: McpConnection
) -> None:
    """Suppress automatic adoption of one device on one avatar.

    Written when the user explicitly disconnects a machine. Without this marker
    ``mcp_auto_adopt`` would re-bind the device on the next conversation turn,
    making an explicit disconnect look broken.
    """
    if store is None or not connection.device_id:
        return
    await store.aput(
        mcp_connection_declined_namespace(user_id, assistant_id),
        key=connection.device_id,
        value={
            "declined_at": datetime.now(UTC).isoformat(),
            "device_id": connection.device_id,
            "device_label": connection.device_label,
        },
    )


async def suppressed_device_ids(
    store: Any, user_id: str, assistant_id: str
) -> set[str]:
    """Device identifiers this avatar must not adopt automatically."""
    items = await _search_namespace(
        store, mcp_connection_declined_namespace(user_id, assistant_id)
    )
    return {item.key for item in items}


async def is_declined(
    store: Any, user_id: str, assistant_id: str, device_id: str
) -> bool:
    """Report whether one device is suppressed on this avatar."""
    if store is None:
        return False
    item = await store.aget(
        mcp_connection_declined_namespace(user_id, assistant_id), device_id
    )
    return item is not None


async def clear_declined(
    store: Any, user_id: str, assistant_id: str, device_id: str | None = None
) -> None:
    """Remove suppression so automatic adoption resumes for this avatar.

    Called when the user explicitly connects: a past disconnect only means "stop
    adopting this device automatically" — an explicit connect always wins.
    Clears one device when ``device_id`` is given, otherwise every suppressed
    device on this avatar.
    """
    if store is None:
        return
    namespace = mcp_connection_declined_namespace(user_id, assistant_id)
    if device_id is not None:
        await store.adelete(namespace, device_id)
        return
    for item in await _search_namespace(store, namespace):
        await store.adelete(namespace, item.key)


async def bound_connections_for(
    store: Any, user_id: str, assistant_id: str
) -> list[McpConnection]:
    """Return every established connection bound to this avatar.

    This is the sole gate for the data-analysis capability: no environment
    switch, no per-avatar enable flag — only saved, adopted connections whose
    bound avatar matches the avatar currently answering.

    When a live relay socket exists for a device, that device's connection is
    rebuilt from the session (loopback bridge URL + current device secret) so a
    stale saved URL — for example ``host.docker.internal:8000`` left by an older
    local-mode adoption — cannot keep data analysis permanently broken. The
    refreshed record is written back only when something actually changed, so a
    healthy turn performs no extra store writes.
    """
    records = await read_user_connections(store, user_id)
    if not records:
        return []

    from src.anubis.utils.tools.data_analysis import relay

    connections: list[McpConnection] = []
    for record in records:
        if record.get("status") != "connected":
            continue
        if record.get("assistant_id") != assistant_id:
            continue

        device_id = record.get("device_id") or ""
        live_session = relay.get_session(device_id) if device_id else None
        if live_session is None:
            connections.append(McpConnection.from_mapping(record))
            continue

        connection = relay.connection_from_session(live_session)
        if (
            record.get("url") != connection.url
            or record.get("device_secret") != connection.device_secret
            or list(record.get("allowed_roots") or []) != list(connection.allowed_roots)
            or record.get("device_label") != connection.device_label
        ):
            await save_user_connection(
                store, user_id, connection=connection, assistant_id=assistant_id
            )
        connections.append(connection)

    return connections
