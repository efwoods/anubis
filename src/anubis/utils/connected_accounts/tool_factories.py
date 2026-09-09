"""Which tools a connected account contributes, keyed by the account's kind.

This is the second half of the "one row per provider" promise in
``providers.py``. A provider row says what an account IS; this table says what
an account of that kind lets the avatar DO. The ``think`` node walks the
connected accounts, groups them by kind, and calls one factory per kind, so a
provider whose kind already appears here needs no code beyond its row, and a
new kind is one factory module plus one entry in :data:`TOOL_FACTORIES`.

Every factory has the same shape — ``factory(context, accounts, **runtime)``
— and returns one flat tool set for ALL the accounts of that kind,
disambiguated by an ``account_label`` / ``connection`` argument. ``runtime``
carries the cross-thread ``store`` (needed to persist a refreshed token or a
refreshed browser session) and the analysis ``bundle`` when the turn has one.

The tool-name lookups exist for the connect card, which reports how many tools
connecting an account adds. They are lazy imports so this module stays cheap to
import from the registry and the endpoints.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from src.anubis.utils.connected_accounts.providers import (
    KIND_ANALYTICS,
    KIND_BANK,
    KIND_CALENDAR,
    KIND_CRYPTO,
    KIND_DEVELOPER,
    KIND_HOSTING,
    KIND_MAILBOX,
    KIND_MCP_SERVER,
    KIND_MESSAGING,
    KIND_SOCIAL,
    KIND_WEBSITE,
    MECHANISM_BROWSER_SESSION,
    MECHANISM_OAUTH,
    MECHANISM_PASSWORD,
)

logger = logging.getLogger(__name__)

ToolFactory = Callable[..., Awaitable[list[Any]] | list[Any]]


def _mailbox_factory(context: Any, accounts: list[dict[str, Any]], **runtime: Any) -> list[Any]:
    from src.anubis.utils.tools.email.mailbox_tools import build_mailbox_tools

    return build_mailbox_tools(context, accounts, store=runtime.get("store"))


async def _mcp_server_factory(
    context: Any, accounts: list[dict[str, Any]], **runtime: Any
) -> list[Any]:
    from src.anubis.utils.connected_accounts.mcp_server_tools import (
        build_mcp_server_tools,
    )

    return await build_mcp_server_tools(context, accounts, store=runtime.get("store"))


def _browser_session_factory(
    context: Any, accounts: list[dict[str, Any]], **runtime: Any
) -> list[Any]:
    from src.anubis.utils.connected_accounts.browser_session_tools import (
        build_browser_session_tools,
    )

    return build_browser_session_tools(
        context, accounts, store=runtime.get("store"), bundle=runtime.get("bundle")
    )


def _website_factory(
    context: Any, accounts: list[dict[str, Any]], **runtime: Any
) -> list[Any]:
    from src.anubis.utils.connected_accounts.website_tools import build_website_tools

    return build_website_tools(
        context,
        accounts,
        store=runtime.get("store"),
        bundle=runtime.get("bundle"),
        all_accounts=runtime.get("all_accounts") or accounts,
    )


def _bank_factory(context: Any, accounts: list[dict[str, Any]], **runtime: Any) -> list[Any]:
    from src.anubis.utils.connected_accounts.finance_tools import build_finance_tools

    return build_finance_tools(context, accounts, store=runtime.get("store"), pool=runtime.get("pool"))


def _oauth_vendor_factory(
    context: Any, accounts: list[dict[str, Any]], **runtime: Any
) -> list[Any]:
    """Tools for accounts that talk to a vendor API with an OAuth token.

    GitHub, X, Vercel, Google Calendar, Google Analytics, and YouTube each get
    their client module; accounts of the same kind that signed in through a
    browser session go to the browser-session factory instead.
    """
    from src.anubis.utils.connected_accounts.vendor_api_tools import (
        build_vendor_api_tools,
    )

    oauth_accounts = [
        record for record in accounts if record.get("credential_mechanism") == MECHANISM_OAUTH
    ]
    session_accounts = [
        record
        for record in accounts
        if record.get("credential_mechanism") == MECHANISM_BROWSER_SESSION
    ]
    tools: list[Any] = []
    if oauth_accounts:
        tools.extend(
            build_vendor_api_tools(
                context, oauth_accounts, store=runtime.get("store"), pool=runtime.get("pool")
            )
        )
    if session_accounts:
        tools.extend(_browser_session_factory(context, session_accounts, **runtime))
    return tools


def _calendar_factory(
    context: Any, accounts: list[dict[str, Any]], **runtime: Any
) -> list[Any]:
    """Calendar tools, whichever way the account was connected.

    A calendar connected with an address and a password speaks CalDAV; one
    connected through a vendor sign-in speaks that vendor's API. Both present
    calendar tools, so the split lives here rather than in the prompt.
    """
    from src.anubis.utils.connected_accounts.caldav_tools import build_caldav_tools

    password_accounts = [
        record
        for record in accounts
        if record.get("credential_mechanism") == MECHANISM_PASSWORD
    ]
    other_accounts = [record for record in accounts if record not in password_accounts]
    tools: list[Any] = []
    if password_accounts:
        tools.extend(
            build_caldav_tools(context, password_accounts, store=runtime.get("store"))
        )
    if other_accounts:
        tools.extend(_oauth_vendor_factory(context, other_accounts, **runtime))
    return tools


TOOL_FACTORIES: dict[str, ToolFactory] = {
    KIND_MAILBOX: _mailbox_factory,
    KIND_MCP_SERVER: _mcp_server_factory,
    KIND_ANALYTICS: _oauth_vendor_factory,
    KIND_WEBSITE: _website_factory,
    KIND_BANK: _bank_factory,
    KIND_DEVELOPER: _oauth_vendor_factory,
    KIND_SOCIAL: _oauth_vendor_factory,
    KIND_CALENDAR: _calendar_factory,
    KIND_HOSTING: _oauth_vendor_factory,
    KIND_CRYPTO: _oauth_vendor_factory,
    KIND_MESSAGING: _oauth_vendor_factory,
}

# Tool names per provider for kinds with a fixed surface, so the connect card
# can say how many tools connecting the account adds without building them.
_BROWSER_SESSION_TOOL_NAMES: tuple[str, ...] = (
    "open_connected_site",
    "read_connected_page",
    "fetch_connected_json",
    "find_on_connected_site",
    "click_connected_element",
    "type_into_connected_field",
    "run_provider_recipe",
)
_WEBSITE_TOOL_NAMES: tuple[str, ...] = (
    "crawl_website",
    "website_audit",
    "website_traffic",
)
_BANK_TOOL_NAMES: tuple[str, ...] = (
    "finance_accounts",
    "finance_transactions",
    "finance_spend_summary",
)
_VENDOR_API_TOOL_NAMES: dict[str, tuple[str, ...]] = {
    "github": ("github_activity", "github_issues", "github_pull_requests"),
    "x": ("x_recent_posts", "x_post_reply"),
    "vercel": ("vercel_deployments", "vercel_usage"),
    "google_calendar": (
        "list_calendars",
        "calendar_events",
        "create_calendar_event",
        "update_calendar_event",
        "delete_calendar_event",
        "find_free_time",
    ),
    "google_analytics": ("analytics_traffic_report",),
    "youtube": ("youtube_channel_stats",),
    "coinbase": ("coinbase_accounts", "coinbase_transactions"),
}

def tool_names_for(provider: Any, record: dict[str, Any] | None = None) -> list[str]:
    """Return the tool names an account of this provider contributes.

    For kinds with a fixed tool surface (a mailbox) the names come from the
    factory module's declared tuple. For a custom Model Context Protocol server
    the names are whatever the probe found, stored on the record's transport
    details, because every such server exposes its own tools.
    """
    kind = getattr(provider, "kind", None)
    name = str(getattr(provider, "name", "") or "")
    mechanism = str(getattr(provider, "credential_mechanism", "") or "")
    if record and record.get("credential_mechanism") == MECHANISM_BROWSER_SESSION:
        return list(_BROWSER_SESSION_TOOL_NAMES)
    if kind == KIND_MAILBOX:
        from src.anubis.utils.tools.email.mailbox_tools import MAILBOX_TOOL_NAMES

        return list(MAILBOX_TOOL_NAMES)
    if kind == KIND_MCP_SERVER:
        transport = (record or {}).get("transport") or {}
        return [str(entry) for entry in transport.get("tool_names") or []]
    if kind == KIND_BANK:
        return list(_BANK_TOOL_NAMES)
    if kind == KIND_WEBSITE:
        return list(_WEBSITE_TOOL_NAMES)
    if kind == KIND_CALENDAR and mechanism == MECHANISM_PASSWORD:
        from src.anubis.utils.connected_accounts.caldav_tools import CALDAV_TOOL_NAMES

        return list(CALDAV_TOOL_NAMES)
    if name in _VENDOR_API_TOOL_NAMES and mechanism == MECHANISM_OAUTH:
        return list(_VENDOR_API_TOOL_NAMES[name])
    if mechanism == MECHANISM_BROWSER_SESSION:
        return list(_BROWSER_SESSION_TOOL_NAMES)
    return []


async def build_tools_for_accounts(
    context: Any,
    accounts: list[dict[str, Any]],
    *,
    store: Any = None,
    pool: Any = None,
    bundle: Any = None,
) -> list[Any]:
    """Build every connected account's tools, one factory call per kind.

    Accounts of a kind with no factory contribute nothing rather than raising —
    the account exists in the catalog before its tools do. A factory whose
    module is not installed yet (a phase still being built) is skipped with a
    log line for the same reason.
    """
    by_kind: dict[str, list[dict[str, Any]]] = {}
    session_records: list[dict[str, Any]] = []
    for record in accounts:
        # A record made by a live sign-in is used through the signed-in
        # session whatever kind the provider row declares (a Gmail mailbox
        # signed in on Google's page, a GitHub account signed in on GitHub).
        if record.get("credential_mechanism") == MECHANISM_BROWSER_SESSION:
            session_records.append(record)
            continue
        by_kind.setdefault(str(record.get("kind") or ""), []).append(record)

    tools: list[Any] = []
    if session_records:
        try:
            tools.extend(
                _browser_session_factory(
                    context, session_records, store=store, pool=pool, bundle=bundle, all_accounts=accounts
                )
            )
        except Exception:
            logger.exception("Browser-session tool factory failed; skipping")
    for kind, kind_accounts in by_kind.items():
        factory = TOOL_FACTORIES.get(kind)
        if factory is None:
            continue
        try:
            produced = factory(
                context,
                kind_accounts,
                store=store,
                pool=pool,
                bundle=bundle,
                all_accounts=accounts,
            )
            if hasattr(produced, "__await__"):
                produced = await produced
        except ImportError as missing_module:
            logger.info("No tool module yet for kind %s: %s", kind, missing_module)
            continue
        except Exception:
            logger.exception("Tool factory for kind %s failed; skipping", kind)
            continue
        tools.extend(produced or [])
    return tools
