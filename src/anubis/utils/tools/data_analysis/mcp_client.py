"""Model Context Protocol client access for the data-analysis capability.

The avatar reaches the host filesystem exclusively through a Model Context
Protocol filesystem server. Which server it talks to is not read from a
hard-coded URL — it comes from the per-user :class:`McpConnection` that was
discovered and saved (see ``discovery.py``). This module owns:

- building the ``MultiServerMCPClient`` for a given connection;
- a module-level cache of the fetched tool list keyed by the connection, so
  the tool schemas are not re-fetched on every conversation turn
  (``MultiServerMCPClient`` is stateless — each tool invocation opens a fresh
  session — so caching the tool objects is safe);
- graceful degradation: when the server behind a saved connection is
  unreachable, the capability is simply absent for the turn and the avatar
  keeps working without analysis tools.

The heavy ``langchain_mcp_adapters`` import is deferred into functions per the
repository cold-start rule.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from src.anubis.utils.tools.data_analysis.discovery import McpConnection

logger = logging.getLogger(__name__)

# Cache of fetched Model Context Protocol tool lists, keyed by
# (server_name, transport, url) — i.e. per saved connection. Populated on
# first successful fetch.
_mcp_tools_cache: dict[tuple[str, str, str], list[Any]] = {}

# Timestamp of the most recent failed fetch per cache key. Used to avoid
# re-dialing an unreachable server on every conversation turn.
_mcp_last_failure_monotonic: dict[tuple[str, str, str], float] = {}

# Seconds to wait after a failed fetch before trying the server again.
_FAILURE_RETRY_SECONDS = 30.0


def _cache_key(connection: McpConnection) -> tuple[str, str, str]:
    return (connection.server_name, connection.transport, connection.url)


def build_mcp_client(connection: McpConnection):
    """Construct a ``MultiServerMCPClient`` for a saved connection.

    Construction is cheap (no network activity happens until a tool call),
    so a fresh client per use is fine.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    return MultiServerMCPClient(
        {
            connection.server_name: {
                "transport": connection.transport,
                "url": connection.url,
            }
        }
    )


async def get_mcp_filesystem_tools(connection: McpConnection) -> list[Any]:
    """Fetch (and cache) the filesystem server's tool list for a connection.

    Returns an empty list when the server is unreachable — the caller treats
    an empty list as "capability unavailable this turn" and the avatar
    continues without analysis tools.
    """
    key = _cache_key(connection)

    cached_tools = _mcp_tools_cache.get(key)
    if cached_tools is not None:
        return cached_tools

    last_failure = _mcp_last_failure_monotonic.get(key)
    if (
        last_failure is not None
        and (time.monotonic() - last_failure) < _FAILURE_RETRY_SECONDS
    ):
        return []

    try:
        client = build_mcp_client(connection)
        tools = await client.get_tools()
    except Exception:
        _mcp_last_failure_monotonic[key] = time.monotonic()
        logger.warning(
            "Model Context Protocol filesystem server unreachable at %s; "
            "data-analysis capability disabled for this turn.",
            connection.url,
            exc_info=True,
        )
        return []

    _mcp_tools_cache[key] = tools
    _mcp_last_failure_monotonic.pop(key, None)
    logger.info(
        "Loaded %d Model Context Protocol filesystem tools from %s",
        len(tools),
        connection.url,
    )
    return tools


def parse_mcp_result(result: Any) -> Any:
    """Normalize LangChain Model-Context-Protocol tool output to Python values.

    Tool results arrive as a list of content blocks (``[{"type": "text",
    "text": ...}]``); unwrap the first text block and JSON-decode when
    possible. Ported from ``src/mcp/src/client/client.py``.
    """
    if (
        isinstance(result, list)
        and result
        and isinstance(result[0], dict)
        and "text" in result[0]
    ):
        text = result[0]["text"]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return result


async def call_mcp_filesystem_tool(
    connection: McpConnection, tool_name: str, tool_args: dict[str, Any]
) -> Any:
    """Invoke one named tool on the connection's server and normalize the result.

    Raises:
        RuntimeError: when the server is unreachable or does not expose the
            requested tool name.
    """
    tools = await get_mcp_filesystem_tools(connection)
    if not tools:
        raise RuntimeError(
            "The Model Context Protocol filesystem server at "
            f"{connection.url} is unreachable."
        )
    matching_tool = next((t for t in tools if t.name == tool_name), None)
    if matching_tool is None:
        raise RuntimeError(
            f"The Model Context Protocol filesystem server does not expose a "
            f"tool named {tool_name!r}."
        )
    return parse_mcp_result(await matching_tool.ainvoke(tool_args))
