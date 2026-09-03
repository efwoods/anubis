"""Which tools a connected account contributes, keyed by the account's kind.

This is the second half of the "one row per provider" promise in
``providers.py``. A provider row says what an account IS; this table says what
an account of that kind lets the avatar DO. The ``think`` node walks the
connected accounts, groups them by kind, and calls one factory per kind, so a
provider whose kind already appears here needs no code beyond its row, and a
new kind is one factory module plus one entry in :data:`TOOL_FACTORIES`.

Every factory has the same shape — ``factory(context, accounts) -> list[tool]``
— and returns one flat tool set for ALL the accounts of that kind, disambiguated
by an ``account_label`` argument. That is what keeps the tool list (and the
prompt describing it) from growing with the number of accounts.

The tool-name lookups exist for the connect card, which reports how many tools
connecting an account adds. They are lazy imports so this module stays cheap to
import from the registry and the endpoints.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from src.anubis.utils.connected_accounts.providers import (
    KIND_MAILBOX,
    KIND_MCP_SERVER,
)

ToolFactory = Callable[[Any, list[dict[str, Any]]], Awaitable[list[Any]] | list[Any]]


def _mailbox_factory(context: Any, accounts: list[dict[str, Any]]) -> list[Any]:
    from src.anubis.utils.tools.email.mailbox_tools import build_mailbox_tools

    return build_mailbox_tools(context, accounts)


async def _mcp_server_factory(
    context: Any, accounts: list[dict[str, Any]]
) -> list[Any]:
    from src.anubis.utils.connected_accounts.mcp_server_tools import (
        build_mcp_server_tools,
    )

    return await build_mcp_server_tools(context, accounts)


TOOL_FACTORIES: dict[str, ToolFactory] = {
    KIND_MAILBOX: _mailbox_factory,
    KIND_MCP_SERVER: _mcp_server_factory,
}


def tool_names_for(provider: Any, record: dict[str, Any] | None = None) -> list[str]:
    """Return the tool names an account of this provider contributes.

    For kinds with a fixed tool surface (a mailbox) the names come from the
    factory module's declared tuple. For a custom Model Context Protocol server
    the names are whatever the probe found, stored on the record's transport
    details, because every such server exposes its own tools.
    """
    kind = getattr(provider, "kind", None)
    if kind == KIND_MAILBOX:
        from src.anubis.utils.tools.email.mailbox_tools import MAILBOX_TOOL_NAMES

        return list(MAILBOX_TOOL_NAMES)
    if kind == KIND_MCP_SERVER:
        transport = (record or {}).get("transport") or {}
        return [str(name) for name in transport.get("tool_names") or []]
    return []


async def build_tools_for_accounts(
    context: Any, accounts: list[dict[str, Any]]
) -> list[Any]:
    """Build every connected account's tools, one factory call per kind.

    Accounts of a kind with no factory contribute nothing (a coming-soon social
    account, for instance) rather than raising — the account exists in the
    catalog before its tools do.
    """
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for record in accounts:
        by_kind.setdefault(str(record.get("kind") or ""), []).append(record)

    tools: list[Any] = []
    for kind, kind_accounts in by_kind.items():
        factory = TOOL_FACTORIES.get(kind)
        if factory is None:
            continue
        produced = factory(context, kind_accounts)
        if hasattr(produced, "__await__"):
            produced = await produced
        tools.extend(produced or [])
    return tools
