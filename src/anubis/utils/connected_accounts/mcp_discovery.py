"""Find out whether a site offers a Model Context Protocol server, from its address.

The owner types a site — ``linear.app``, ``sentry.io``, their own company's
domain — and the avatar should reach it the way the site itself intends, if the
site intends anything at all. That is what this module answers: given an
address, is there an MCP server behind it, and where.

Why this is the first thing tried for an arbitrary site, ahead of every other
mechanism: an MCP server that wants authentication advertises it through
RFC 9728, and the client registers itself through RFC 7591 **dynamic client
registration**. No console visit, no application to create, no key to paste, no
credential typed into Neural Nexus — and it is the vendor's own supported
route rather than something worked around. `mcp_oauth.py` already implements
that half; this module is the part that finds the server in the first place.

Discovery is deliberately layered, because the standard is still settling:

1. ``/.well-known/mcp.json`` — the SEP-1649 / SEP-2127 proposal, the convention
   with the most weight behind it and the one most likely to be what a site
   publishes deliberately.
2. ``/.well-known/mcp-server`` — the competing IETF draft. Cheap to also ask.
3. Conventional addresses — ``mcp.<host>``, ``<host>/mcp``, ``<host>/sse``.
   Most vendors that run a server today publish nothing at all and simply sit
   at one of these.
4. A small table of servers whose address cannot be guessed from the domain.

**A published document is a claim, not a connection.** Every candidate this
module returns has been proven by opening an MCP session and listing its tools,
because a stale ``.well-known`` file pointing at a dead host is worse than no
document: it produces a connector that appears to exist and answers nothing.
The one exception is a server that demands authorization first — a 401 with a
challenge is itself proof the server is real, and the sign-in comes next.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)

WELL_KNOWN_PATHS: tuple[str, ...] = (
    "/.well-known/mcp.json",
    "/.well-known/mcp-server",
)

# Tried in order. A host that runs a server usually sits at one of these.
CONVENTIONAL_PATHS: tuple[str, ...] = ("/mcp", "/sse", "/api/mcp")
CONVENTIONAL_SUBDOMAIN = "mcp"

# A host name, and nothing that merely parses as one. ``urlparse`` accepts a
# netloc containing spaces, so a sentence typed into the address field would
# otherwise become five requests to a host that cannot exist.
HOSTNAME_PATTERN = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$")

DISCOVERY_TIMEOUT_SECONDS = 6.0
PROBE_TIMEOUT_SECONDS = 12.0

# Servers whose address a person could not guess from the domain they know.
# Only entries that save a real guess belong here; anything reachable by the
# conventions above must NOT be listed, or this table becomes a second source
# of truth that quietly goes stale.
KNOWN_SERVERS: dict[str, str] = {
    "github.com": "https://api.githubcopilot.com/mcp/",
    "huggingface.co": "https://huggingface.co/mcp",
}


@dataclass
class DiscoveredMcpServer:
    """One Model Context Protocol server found behind a site."""

    server_url: str
    site_url: str = ""
    name: str = ""
    description: str = ""
    source: str = ""
    needs_authorization: bool = False
    tool_names: list[str] = field(default_factory=list)

    def as_public_dict(self) -> dict[str, Any]:
        """Return the shape a card or a tool result carries."""
        return {
            "server_url": self.server_url,
            "site_url": self.site_url,
            "name": self.name,
            "description": self.description,
            "needs_authorization": self.needs_authorization,
            "tool_names": list(self.tool_names),
            "found_by": self.source,
        }


def normalize_site(site: str) -> tuple[str, str]:
    """Return ``(origin, host)`` for whatever the owner typed.

    People type ``linear.app``, ``https://linear.app/``, and
    ``https://linear.app/team/x`` and mean the same site.
    """
    text = str(site or "").strip()
    if not text:
        return "", ""
    if "://" not in text:
        text = f"https://{text}"
    parsed = urlparse(text)
    host = (parsed.hostname or "").lower()
    if not host or not HOSTNAME_PATTERN.match(host):
        return "", ""
    netloc = host if parsed.port is None else f"{host}:{parsed.port}"
    return urlunparse((parsed.scheme or "https", netloc, "", "", "", "")), host


def _server_url_from_document(document: Any, origin: str) -> tuple[str, str, str]:
    """Read ``(url, name, description)`` out of a discovery document.

    The two drafts disagree about shape and neither is final, so every spelling
    seen in the wild is accepted rather than one guessed correctly.
    """
    if not isinstance(document, dict):
        return "", "", ""
    candidates: list[dict[str, Any]] = []
    for key in ("servers", "mcpServers", "mcp_servers"):
        value = document.get(key)
        if isinstance(value, list):
            candidates.extend(item for item in value if isinstance(item, dict))
        elif isinstance(value, dict):
            candidates.extend(
                {"name": name, **item}
                for name, item in value.items()
                if isinstance(item, dict)
            )
    candidates.append(document)

    for candidate in candidates:
        for key in ("url", "endpoint", "server_url", "serverUrl", "uri", "href"):
            raw = candidate.get(key)
            if not isinstance(raw, str) or not raw.strip():
                continue
            address = raw.strip()
            if address.startswith("/"):
                address = f"{origin}{address}"
            if not address.startswith(("http://", "https://")):
                continue
            return (
                address,
                str(candidate.get("name") or document.get("name") or ""),
                str(
                    candidate.get("description")
                    or document.get("description")
                    or ""
                ),
            )
    return "", "", ""


async def _fetch_document(http_client: Any, url: str) -> Any:
    try:
        response = await http_client.get(
            url,
            timeout=DISCOVERY_TIMEOUT_SECONDS,
            headers={"accept": "application/json"},
        )
    except Exception:  # noqa: BLE001 - an address that does not answer is a miss
        return None
    if response.status_code != 200:
        return None
    try:
        return response.json()
    except Exception:  # noqa: BLE001 - a document that is not JSON is not one
        return None


async def _verify(
    server_url: str, context: Any, *, source: str, site_url: str, name: str = "", description: str = ""
) -> DiscoveredMcpServer | None:
    """Prove one candidate by opening a session, or by an authorization demand."""
    from src.anubis.utils.connected_accounts.mcp_oauth import (
        AUTHORIZATION_OPEN,
        probe_authorization,
    )
    from src.anubis.utils.connected_accounts.mcp_server_tools import (
        McpServerUnreachableError,
        probe_server_tools,
    )

    try:
        authorization = await probe_authorization(server_url, context)
    except Exception:  # noqa: BLE001 - treated as "nothing here"
        return None
    status = str(authorization.get("status") or "")
    if status == "unreachable":
        return None
    if status != AUTHORIZATION_OPEN:
        # A server that demands authorization has proved it exists; listing its
        # tools is exactly what it is refusing to do until the owner signs in.
        return DiscoveredMcpServer(
            server_url=server_url,
            site_url=site_url,
            name=name,
            description=description,
            source=source,
            needs_authorization=True,
        )

    try:
        tools = await probe_server_tools(server_url, None, PROBE_TIMEOUT_SECONDS)
    except McpServerUnreachableError:
        return None
    except Exception:  # noqa: BLE001 - an unusable server is not a discovery
        return None
    if not tools:
        return None
    return DiscoveredMcpServer(
        server_url=server_url,
        site_url=site_url,
        name=name,
        description=description,
        source=source,
        needs_authorization=False,
        tool_names=[str(getattr(tool, "name", "")) for tool in tools],
    )


async def discover_mcp_server(
    site: str, context: Any, *, http_client: Any = None
) -> DiscoveredMcpServer | None:
    """Find a Model Context Protocol server behind ``site``, or return ``None``.

    Never raises for a site that simply has none: not having one is the common
    case and the caller's next move is to say so, not to show a failure.
    """
    import httpx

    origin, host = normalize_site(site)
    if not origin:
        return None

    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(follow_redirects=True)
    try:
        known = KNOWN_SERVERS.get(host)
        if known:
            found = await _verify(
                known, context, source="known_server", site_url=origin
            )
            if found:
                return found

        for path in WELL_KNOWN_PATHS:
            document = await _fetch_document(client, f"{origin}{path}")
            if document is None:
                continue
            server_url, name, description = _server_url_from_document(document, origin)
            if not server_url:
                continue
            found = await _verify(
                server_url,
                context,
                source=f"well_known:{path}",
                site_url=origin,
                name=name,
                description=description,
            )
            if found:
                return found

        candidates = [f"{origin}{path}" for path in CONVENTIONAL_PATHS]
        if not host.startswith(f"{CONVENTIONAL_SUBDOMAIN}."):
            scheme = urlparse(origin).scheme or "https"
            candidates.insert(0, f"{scheme}://{CONVENTIONAL_SUBDOMAIN}.{host}/mcp")
            candidates.insert(1, f"{scheme}://{CONVENTIONAL_SUBDOMAIN}.{host}")
        for candidate in candidates:
            found = await _verify(
                candidate, context, source="conventional_address", site_url=origin
            )
            if found:
                return found
    finally:
        if owns_client:
            await client.aclose()

    return None


async def discover_many(
    sites: list[str], context: Any, *, concurrency: int = 4
) -> dict[str, DiscoveredMcpServer | None]:
    """Look several sites up at once, for a catalogue or a bulk import."""
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(site: str):
        async with semaphore:
            return site, await discover_mcp_server(site, context)

    results = await asyncio.gather(*(one(site) for site in sites))
    return dict(results)
