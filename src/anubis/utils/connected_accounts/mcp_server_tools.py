"""Tools from the owner's own Model Context Protocol servers ("custom connectors").

A custom connector is a record whose transport details hold a server URL the
owner supplied, plus the optional bearer token (encrypted) the server requires.
At turn time this module builds a ``MultiServerMCPClient`` per enabled
connector, fetches the server's tools, and hands them to the deep agent with
the connector's name prefixed onto every tool name, so two servers that both
expose a ``search`` tool do not collide and the model can tell which connector
a tool belongs to.

Modelled on ``data_analysis/mcp_client.py`` — including its two protections:
tool lists are cached per server so schemas are not re-fetched every turn, and
an unreachable server is remembered for a short while so the avatar does not
re-dial a dead address on every message. A connector that cannot be reached
contributes no tools for the turn; the conversation continues without it.

The same probe is used at connect time (``connect_handlers.py``) so a URL is
proven to answer before it is stored, mirroring how a mailbox password is proved
by a real login.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)


class McpServerUnreachableError(RuntimeError):
    """The server did not answer, or answered with something other than tools."""


# Per-server tool cache and failure memory, keyed by (url, has_token).
_tools_cache: dict[tuple[str, bool], list[Any]] = {}
_last_failure_monotonic: dict[tuple[str, bool], float] = {}
_FAILURE_RETRY_SECONDS = 30.0

# Tool names are constrained by the model providers to letters, digits,
# underscores, and hyphens, at most 64 characters. The prefix is derived from
# the connector's display label and squeezed into that alphabet.
_TOOL_NAME_ALPHABET = re.compile(r"[^a-zA-Z0-9_-]+")
_MAX_TOOL_NAME_LENGTH = 64


def connector_slug(display_label: str) -> str:
    """Derive the tool-name prefix for a connector from its label."""
    slug = _TOOL_NAME_ALPHABET.sub("_", str(display_label or "").strip()).strip("_")
    return (slug or "connector").lower()[:24]


def infer_transport(server_url: str) -> str:
    """Pick the Model Context Protocol transport from the URL the owner typed.

    Servers that speak the older SSE transport conventionally end their URL in
    ``/sse``; everything else is assumed to be Streamable HTTP, which is the
    current default and what FastMCP serves at ``/mcp``.
    """
    return (
        "sse"
        if str(server_url).rstrip("/").lower().endswith("/sse")
        else "streamable_http"
    )


def build_client(server_url: str, bearer_token: str | None, name: str = "connector"):
    """Construct a ``MultiServerMCPClient`` for one custom connector."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    server_config: dict[str, Any] = {
        "transport": infer_transport(server_url),
        "url": server_url,
    }
    if bearer_token:
        server_config["headers"] = {"Authorization": f"Bearer {bearer_token}"}
    return MultiServerMCPClient({name: server_config})


async def probe_server_tools(
    server_url: str, bearer_token: str | None, timeout_seconds: float
) -> list[Any]:
    """List a server's tools within a timeout, raising when it cannot be done.

    Used at connect time to prove the address before storing it, and at turn
    time (through the cache) to build the connector's tools.
    """
    try:
        client = build_client(server_url, bearer_token)
        return await asyncio.wait_for(client.get_tools(), timeout=timeout_seconds)
    except TimeoutError as timeout_error:
        raise McpServerUnreachableError(
            f"The server at {server_url} did not answer within "
            f"{timeout_seconds:.0f} seconds."
        ) from timeout_error
    except Exception as probe_error:
        raise McpServerUnreachableError(
            f"The server at {server_url} could not be reached or did not list "
            f"its tools: {probe_error}"
        ) from probe_error


def _prefixed(tool: Any, slug: str) -> Any:
    """Return the tool renamed with the connector prefix, within the length cap."""
    base_name = str(getattr(tool, "name", "") or "tool")
    candidate = f"{slug}__{base_name}"
    if len(candidate) > _MAX_TOOL_NAME_LENGTH:
        candidate = candidate[:_MAX_TOOL_NAME_LENGTH]
    try:
        tool.name = candidate
    except Exception:
        logger.debug("Could not rename tool %s for connector %s", base_name, slug)
    return tool


async def _tools_for_record(record: dict[str, Any], context: Any) -> list[Any]:
    from src.anubis.utils.secret_store import decrypt_secret

    transport = record.get("transport") or {}
    server_url = str(transport.get("server_url") or "")
    if not server_url:
        return []
    bearer_token: str | None = None
    if record.get("encrypted_secret"):
        try:
            bearer_token = decrypt_secret(record["encrypted_secret"], context)
        except Exception:
            logger.info(
                "Custom connector %s has an unreadable token; connecting without it",
                record.get("display_label"),
            )

    key = (server_url, bool(bearer_token))
    cached = _tools_cache.get(key)
    if cached is not None:
        return cached
    last_failure = _last_failure_monotonic.get(key)
    if (
        last_failure is not None
        and (time.monotonic() - last_failure) < _FAILURE_RETRY_SECONDS
    ):
        return []

    timeout_seconds = float(
        getattr(context, "mcp_connector_probe_timeout_seconds", None) or 20.0
    )
    try:
        tools = await probe_server_tools(server_url, bearer_token, timeout_seconds)
    except McpServerUnreachableError as unreachable_error:
        _last_failure_monotonic[key] = time.monotonic()
        logger.warning(
            "Custom connector %s unreachable; its tools are absent this turn: %s",
            record.get("display_label"),
            unreachable_error,
        )
        return []

    slug = connector_slug(str(record.get("display_label") or ""))
    renamed = [_prefixed(tool, slug) for tool in tools]
    _tools_cache[key] = renamed
    _last_failure_monotonic.pop(key, None)
    logger.info(
        "Loaded %d tools from custom connector %s",
        len(renamed),
        record.get("display_label"),
    )
    return renamed


async def build_mcp_server_tools(
    context: Any, accounts: list[dict[str, Any]]
) -> list[Any]:
    """Build the tools of every connected custom Model Context Protocol server.

    Args:
        context: The ``GlobalContext`` carrying the encryption key and the probe
            timeout.
        accounts: Connected-account records already filtered by
            ``bound_accounts_for``; only ``mcp_server`` records are used.

    Returns:
        The tools to append to the deep agent's tool list, possibly empty.
    """
    server_records = [
        record for record in accounts if record.get("kind") == "mcp_server"
    ]
    if not server_records:
        return []
    results = await asyncio.gather(
        *(_tools_for_record(record, context) for record in server_records),
        return_exceptions=True,
    )
    tools: list[Any] = []
    for record, result in zip(server_records, results):
        if isinstance(result, BaseException):
            logger.warning(
                "Custom connector %s failed to build tools: %s",
                record.get("display_label"),
                result,
            )
            continue
        tools.extend(result)
    return tools


def forget_cached_tools(server_url: str) -> None:
    """Drop the cached tool list for a server (after a disconnect or reconnect)."""
    for key in [key for key in _tools_cache if key[0] == server_url]:
        _tools_cache.pop(key, None)
    for key in [key for key in _last_failure_monotonic if key[0] == server_url]:
        _last_failure_monotonic.pop(key, None)
