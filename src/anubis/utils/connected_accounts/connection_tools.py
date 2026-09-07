"""The tool that asks the owner to connect an external account, mid-conversation.

Connecting an account used to require leaving the conversation for a settings
screen. This module lets the avatar raise the connect card in the chat itself:
the tool pauses the run with a LangGraph ``interrupt`` whose value describes the
card, the client renders that description, the owner signs in, and the run
resumes and continues the same turn with the newly connected account's tools
attached.

THE CREDENTIAL NEVER PASSES THROUGH THIS TOOL
    The obvious design — collect the address and password on the card and hand
    them back as the interrupt's resume value — must never be built. A resume
    value is written into the LangGraph checkpointer, so the password would come
    to rest in PostgreSQL in plaintext, inside thread state that
    ``GET /conversations/{thread_id}/messages`` reads back to the client.

    The card's button opens the vendor's OWN sign-in page in a popup (Google,
    GitHub, X, Plaid Link, or a live browser on any other site); the popup
    returns to the API, which proves the login, encrypts what the vendor
    handed back, and stores it. Only then does the client resume this run, and
    the resume value carries nothing but ``{"type": "apply"}``. This tool then
    re-reads storage to learn what was actually connected, so its answer is
    grounded in stored state rather than in anything the client claimed. Any
    credential-looking key that arrives in a resume value is ignored, never
    stored, and never echoed into the reply.

WHAT CONNECTS WITHOUT A CARD
    A custom Model Context Protocol server whose address the owner gave in
    conversation is probed and stored directly when the server needs no login;
    a website is fetched and stored directly. Both still return a ``card`` so
    the transcript shows "Added · N tools".

The card is described from the provider registry rather than from strings held
in the client, so a new provider ships its own card by adding a registry row.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.tools import tool
from langgraph.types import interrupt

from src.anubis.utils.connected_accounts.connection_cards import (
    CARD_STATUS_CANCELLED,
    CARD_STATUS_CONNECTED,
    CARD_STATUS_FAILED,
    CARD_STATUS_NOT_CONNECTED,
    CARD_STATUS_PENDING_LOGIN,
    connection_card_record,
)

logger = logging.getLogger(__name__)

# The interrupt discriminator the client switches on when choosing how to render
# a paused run. ``edit_identity_fact`` raises ``"fact_correction"``; this raises
# ``"connect_account"``. A client that does not recognize the kind must show the
# run as paused rather than guessing at a form.
CONNECT_ACCOUNT_INTERRUPT_KIND = "connect_account"

# Kept for callers that imported the old constant; the endpoint a card posts to
# now comes from the provider row (``connect_endpoint``).
CONNECT_MAILBOX_ENDPOINT = "/connect_mailbox"

LOGIN_RESULT_MESSAGE_TYPE = "neural-nexus:login-result"


def _describe_fields(provider: Any) -> list[dict[str, Any]]:
    """Render a provider's connect fields as plain data for the client."""
    return [
        {
            "name": field_spec.name,
            "label": field_spec.label,
            "input_type": field_spec.input_type,
            "placeholder": field_spec.placeholder,
            "help_text": field_spec.help_text,
            "required": bool(getattr(field_spec, "required", True)),
        }
        for field_spec in provider.connect_fields
    ]


def _connected_views(
    accounts: list[dict[str, Any]], provider_name: str
) -> list[dict[str, Any]]:
    """Public views of the accounts already connected for one provider.

    Shown on the card so an owner who already connected an address sees that
    rather than being asked for the same credential twice. These go through
    ``public_account_view``, so no ciphertext reaches the client.
    """
    from src.anubis.utils.connected_accounts.store import public_account_view

    return [
        public_account_view(record)
        for record in accounts
        if str(record.get("provider") or "").lower() == provider_name
    ]


def build_connect_card(
    provider: Any,
    connected_accounts: list[dict[str, Any]] | None = None,
    *,
    login_request: dict[str, Any] | None = None,
    prefilled_fields: dict[str, Any] | None = None,
    login_mode: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """Describe the sign-in card for one provider.

    The single source of the card's shape. The conversational path raises this
    as an ``interrupt`` value and the settings path reads the same description
    from ``GET /connectable_providers``, so the two surfaces cannot drift into
    showing different labels, different fields, or different help text for the
    same provider.

    Args:
        provider: The ``ConnectedAccountProvider`` to describe.
        connected_accounts: Records already bound to the avatar, used to show
            which addresses are connected already.
        login_request: The body the card posts to the login endpoint to open
            the popup (for a custom connector, the server URL and name).
        prefilled_fields: Values the form starts with (a server URL the owner
            already typed).
        login_mode: Override of the provider's mode (a custom connector that
            turned out to need a static token shows the form).
        message: One line the card shows beneath the description.

    Returns:
        The card description, carrying no credential and no ciphertext.
    """
    from src.anubis.utils.connected_accounts.providers import (
        LOGIN_ENDPOINTS_BY_MODE,
    )
    from src.anubis.utils.connected_accounts.tool_factories import tool_names_for

    tool_names = tool_names_for(provider)
    resolved_mode = login_mode or provider.login_mode
    resolved_login_endpoint = LOGIN_ENDPOINTS_BY_MODE.get(resolved_mode)
    resolved_login_request = dict(login_request or {"provider": provider.name})
    resolved_login_request.setdefault("provider", provider.name)
    return {
        "kind": CONNECT_ACCOUNT_INTERRUPT_KIND,
        "provider": provider.name,
        "display_name": provider.display_name,
        "card_description": provider.card_description,
        "summary": provider.summary,
        "category": provider.category,
        "featured": bool(provider.featured),
        "availability": provider.availability,
        "icon_key": provider.icon_key,
        "tool_count": len(tool_names),
        "tool_names": tool_names,
        "credential_mechanism": provider.credential_mechanism,
        "credential_help_url": provider.credential_help_url,
        "connect_endpoint": provider.connect_endpoint,
        "uses_form": resolved_mode == "form",
        "login_mode": resolved_mode,
        "uses_popup": resolved_login_endpoint is not None,
        "login_endpoint": resolved_login_endpoint,
        "login_request": resolved_login_request,
        "result_message_type": LOGIN_RESULT_MESSAGE_TYPE,
        "site_url": provider.login_url
        if provider.login_url and provider.login_url.startswith("http")
        else resolved_login_request.get("site_url"),
        "prefilled_fields": dict(prefilled_fields or {}),
        "message": message,
        "pairing_instructions": provider.pairing_instructions or None,
        "install_url": provider.install_url,
        "device_bound": bool(getattr(provider, "device_bound", False)),
        "fields": _describe_fields(provider),
        "already_connected": _connected_views(
            list(connected_accounts or []), provider.name
        ),
        "actions": ["apply", "cancel"],
    }


def _card_for(provider: Any, record: dict[str, Any] | None, status: str, error: str | None = None) -> dict[str, Any]:
    return connection_card_record(provider, record, status=status, error=error)


def build_connection_tools(
    context: Any,
    *,
    store: Any,
    user_id: str,
    assistant_id: str,
    connected_accounts: list[dict[str, Any]],
    stale_accounts: list[dict[str, Any]] | None = None,
    allow_interrupt: bool = True,
) -> list[Any]:
    """Build the per-turn account-connection tool set.

    Unlike the account tool factories this returns a tool even when nothing is
    connected — an owner with no accounts is exactly the owner who needs to ask
    for one, and a tool set that appears only after connecting would leave no
    way to connect.

    Args:
        context: The ``GlobalContext`` for this turn.
        store: The cross-thread store, re-read after the owner signs in so the
            reply reflects stored state rather than a client's claim.
        user_id: The authenticated owner whose accounts are being connected.
        assistant_id: The answering avatar, which the connected account binds to.
        connected_accounts: Records already bound to this avatar, used to show
            what is connected on the card.
        stale_accounts: Records bound to this avatar that need a new sign-in;
            the tool names them so the avatar can re-raise the card.
        allow_interrupt: ``False`` for a scheduled (unattended) run: the tool
            then reports ``not_connected`` instead of pausing a run nobody is
            watching.

    Returns:
        The tools to append to this turn's tool list.
    """
    from src.anubis.utils.connected_accounts.providers import (
        LOGIN_MODE_NONE,
        catalog_providers,
        get_provider,
    )

    stale_by_provider = {
        str(record.get("provider") or ""): record for record in (stale_accounts or [])
    }

    async def _direct_connect(
        resolved_provider: Any, fields: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, Any | None]:
        """Prove and store a connection that needs no popup; return (record, needs_login)."""
        from src.anubis.utils.connected_accounts.connect_handlers import (
            ConnectNeedsLogin,
            ConnectRefused,
            ConnectRequest,
        )
        from src.anubis.utils.connected_accounts.connect_handlers import (
            connect_account as run_connect_handler,
        )
        from src.anubis.utils.connected_accounts.store import (
            read_connected_accounts,
            save_connected_account,
        )

        existing = await read_connected_accounts(store, user_id)
        try:
            record = await run_connect_handler(
                ConnectRequest(
                    provider=resolved_provider,
                    fields=fields,
                    assistant_id=assistant_id,
                    context=context,
                    existing_records=existing,
                )
            )
        except ConnectNeedsLogin as needs_login:
            return None, needs_login
        except ConnectRefused:
            raise
        record["user_id"] = user_id
        await save_connected_account(store, user_id, record)
        if resolved_provider.kind == "mcp_server":
            from src.anubis.utils.connected_accounts.mcp_server_tools import (
                forget_cached_tools,
            )

            forget_cached_tools((record.get("transport") or {}).get("server_url") or "")
        return record, None

    @tool
    async def connect_account(
        provider: str = "gmail",
        server_url: str | None = None,
        name: str | None = None,
        site_url: str | None = None,
    ) -> dict[str, Any]:
        """Offer the owner a connection to one of their accounts, in this chat.

        Call this tool when the owner asks to connect, link, or add an account,
        or asks for something that needs an account that is not connected yet:
        read or send email (gmail), work in repositories (github), report on
        spending or burn rate (plaid), read LangSmith / OpenAI / Anthropic usage
        (langsmith, openai, anthropic), audit or crawl a website (website),
        post on X (x), read a calendar (google_calendar), or use the tools of a
        Model Context Protocol server (custom_mcp) or any other site
        (custom_site). Say in one sentence why the connection helps, then call
        this tool: a card appears in the conversation and the owner signs in on
        the vendor's own page in a window. Never ask the owner to type a
        password, token, or key into the chat; if the owner pastes one anyway,
        do not use the value, tell the owner to rotate that secret, and call
        this tool so the owner signs in properly.

        The run pauses while the owner completes the card. When the run resumes
        this tool reports which accounts are connected, and the account's tools
        become available in the same turn, so a request that prompted the
        connection can be carried out immediately afterwards. Then name two or
        three concrete things that can be done with the account now.

        Args:
            provider: Which provider to connect (see the catalog names above).
            server_url: For custom_mcp, the server's address the owner gave.
            name: For custom_mcp or custom_site, how to refer to the connector.
            site_url: For website or custom_site, the site's address.
        """
        provider_name = str(provider or "gmail").strip().lower()
        if provider_name in ("twitter", "x.com"):
            provider_name = "x"
        if provider_name in ("bank", "finance", "bank_account"):
            provider_name = "plaid"
        resolved_provider = get_provider(provider_name)
        if resolved_provider is None:
            supported = [entry.name for entry in catalog_providers()]
            return {
                "status": "unsupported_provider",
                "error": (
                    f"No provider named {provider!r} can be connected. "
                    f"Providers in the catalog: {supported}."
                ),
            }
        if not resolved_provider.is_available:
            return {
                "status": "coming_soon",
                "provider": resolved_provider.name,
                "message": resolved_provider.coming_soon_message(),
                "card": _card_for(resolved_provider, None, CARD_STATUS_NOT_CONNECTED),
            }
        if resolved_provider.login_mode == LOGIN_MODE_NONE:
            # A machine connects itself when the owner runs the daemon; there is
            # no card to complete, so the instructions are the whole answer.
            return {
                "status": "instructions",
                "provider": resolved_provider.name,
                "message": resolved_provider.pairing_instructions,
                "install_url": resolved_provider.install_url,
                "card": _card_for(resolved_provider, None, CARD_STATUS_NOT_CONNECTED),
            }

        from src.anubis.utils.connected_accounts.connect_handlers import ConnectRefused

        login_request: dict[str, Any] = {"provider": resolved_provider.name}
        prefilled: dict[str, Any] = {}
        card_message: str | None = None
        card_login_mode: str | None = None

        # Custom connectors and websites connect directly when no login is
        # needed. The card appears only when the server asks for one.
        direct_fields: dict[str, Any] | None = None
        if resolved_provider.name == "custom_mcp" and (server_url or "").strip():
            direct_fields = {"server_url": server_url, "name": name or ""}
        elif resolved_provider.credential_mechanism == "url_only":
            candidate = (site_url or server_url or "").strip()
            if not candidate:
                return {
                    "status": "needs_site_url",
                    "provider": resolved_provider.name,
                    "message": "Ask the owner which website address to connect, then call again with site_url.",
                }
            direct_fields = {"site_url": candidate, "name": name or ""}
        if direct_fields is not None:
            try:
                record, needs_login = await _direct_connect(resolved_provider, direct_fields)
            except ConnectRefused as refused:
                return {
                    "status": "failed",
                    "provider": resolved_provider.name,
                    "error": refused.detail,
                    "card": _card_for(resolved_provider, None, CARD_STATUS_FAILED, refused.detail),
                }
            if record is not None:
                from src.anubis.utils.connected_accounts.tool_factories import (
                    tool_names_for,
                )

                return {
                    "status": "connected",
                    "provider": resolved_provider.name,
                    "accounts": _connected_views([record], resolved_provider.name),
                    "available_tools": tool_names_for(resolved_provider, record),
                    "message": (
                        f"{record.get('display_label')} is connected. Its tools are "
                        "available now, in this turn. Carry on with what the owner asked."
                    ),
                    "card": _card_for(resolved_provider, record, CARD_STATUS_CONNECTED),
                }
            login_request = dict(needs_login.login_request)
            prefilled = {key: value for key, value in login_request.items() if key != "provider"}
            card_message = needs_login.message
            card_login_mode = needs_login.login_mode
        elif resolved_provider.name == "custom_site":
            candidate = (site_url or server_url or "").strip()
            if not candidate:
                return {
                    "status": "needs_site_url",
                    "provider": resolved_provider.name,
                    "message": "Ask the owner for the site's sign-in page address and a name, then call again.",
                }
            login_request.update({"site_url": candidate, "name": name or ""})
            prefilled = {"site_url": candidate, "name": name or ""}

        if not allow_interrupt:
            return {
                "status": "not_connected",
                "provider": resolved_provider.name,
                "message": (
                    f"No {resolved_provider.display_name} account is connected, and "
                    "this run is unattended, so no sign-in card can be shown. Say "
                    "which connection is missing."
                ),
                "card": _card_for(resolved_provider, None, CARD_STATUS_NOT_CONNECTED),
            }

        card = build_connect_card(
            resolved_provider,
            connected_accounts,
            login_request=login_request,
            prefilled_fields=prefilled,
            login_mode=card_login_mode,
            message=card_message,
        )

        # Everything after this line runs twice: once when the run pauses, and
        # again from the top when the client resumes it. Nothing above may have
        # side effects that must not repeat: the direct connect is idempotent
        # (re-saving the same record), and a login request carries no secret.
        decision = interrupt(card)

        # Only the decision type is read. A resume value carrying an address, a
        # password, or a token is ignored on purpose — see the module docstring.
        decision = decision if isinstance(decision, dict) else {}
        decision_type = str(decision.get("type") or "apply").strip().lower()

        if decision_type in ("cancel", "reject"):
            return {
                "status": "cancelled",
                "provider": resolved_provider.name,
                "message": (
                    f"The owner closed the {resolved_provider.display_name} "
                    "connect card without connecting. Nothing was connected and "
                    "no credential was stored."
                ),
                "card": _card_for(resolved_provider, None, CARD_STATUS_CANCELLED),
            }

        from src.anubis.utils.connected_accounts.store import bound_accounts_for

        try:
            refreshed_accounts = await bound_accounts_for(store, user_id, assistant_id)
        except Exception:
            logger.exception(
                "Could not re-read connected accounts after a connect card "
                "resumed for provider %s",
                resolved_provider.name,
            )
            return {
                "status": "error",
                "provider": resolved_provider.name,
                "error": (
                    "The connected accounts could not be read back after "
                    "sign-in. Ask the owner to try again."
                ),
                "card": _card_for(resolved_provider, None, CARD_STATUS_FAILED),
            }

        from src.anubis.utils.connected_accounts.tool_factories import tool_names_for

        connected_now = _connected_views(refreshed_accounts, resolved_provider.name)
        if not connected_now:
            # The card was dismissed after sign-in failed, or the sign-in never
            # completed. Reporting success here would have the avatar claim an
            # account it cannot reach, so say plainly that nothing was connected.
            return {
                "status": "not_connected",
                "provider": resolved_provider.name,
                "message": (
                    f"No {resolved_provider.display_name} account is connected. "
                    "The sign-in was not completed. Offer to try again."
                ),
                "card": _card_for(resolved_provider, None, CARD_STATUS_PENDING_LOGIN),
            }

        newest_record = next(
            (
                record
                for record in refreshed_accounts
                if record.get("account_key") == connected_now[-1].get("account_key")
            ),
            None,
        )
        available_tools = tool_names_for(resolved_provider, newest_record)
        return {
            "status": "connected",
            "provider": resolved_provider.name,
            "accounts": connected_now,
            "available_tools": available_tools,
            "message": (
                f"{resolved_provider.display_name} is connected. Its tools are "
                "available now, in this turn. Carry on with what the owner asked, "
                "then name two or three concrete things you can do with the account."
            ),
            "card": _card_for(resolved_provider, newest_record, CARD_STATUS_CONNECTED),
        }

    @tool
    async def list_connections_needing_sign_in() -> dict[str, Any]:
        """Report the owner's connected accounts whose sign-in has lapsed.

        Call this tool when a connected account's tool answers
        ``needs_reconnect``, or at the start of a conversation when the
        connector status block lists accounts needing sign-in. Then offer to
        sign in again with connect_account for that provider.
        """
        return {
            "count": len(stale_by_provider),
            "accounts": [
                {
                    "provider": provider_name,
                    "display_label": record.get("display_label"),
                    "account_address": record.get("account_address"),
                }
                for provider_name, record in stale_by_provider.items()
            ],
        }

    # The former name stays callable for one release so an in-flight prompt or
    # a cached tool call keeps working.
    connect_mailbox_account = tool("connect_mailbox_account")(connect_account.coroutine)
    connect_mailbox_account.description = (
        "Alias of connect_account. Prefer connect_account. "
        + connect_account.description
    )

    return [connect_account, connect_mailbox_account, list_connections_needing_sign_in]
