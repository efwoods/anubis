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

    Instead the card posts the credential straight to the provider's
    ``connect_endpoint`` (``POST /connect_account``), which proves it against
    the real server, encrypts it, and stores it. Only then does the client
    resume this run, and the resume value carries nothing but
    ``{"type": "apply"}``. This tool then re-reads storage to learn what was
    actually connected, so its answer is grounded in stored state rather than in
    anything the client claimed. Any credential-looking key that arrives in a
    resume value is ignored, never stored, and never echoed into the reply.

The card is described from the provider registry rather than from strings held
in the client, so a new provider ships its own card by adding a registry row.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.tools import tool
from langgraph.types import interrupt

logger = logging.getLogger(__name__)

# The interrupt discriminator the client switches on when choosing how to render
# a paused run. ``edit_identity_fact`` raises ``"fact_correction"``; this raises
# ``"connect_account"``. A client that does not recognize the kind must show the
# run as paused rather than guessing at a form.
CONNECT_ACCOUNT_INTERRUPT_KIND = "connect_account"

# Kept for callers that imported the old constant; the endpoint a card posts to
# now comes from the provider row (``connect_endpoint``).
CONNECT_MAILBOX_ENDPOINT = "/connect_mailbox"


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
    provider: Any, connected_accounts: list[dict[str, Any]] | None = None
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

    Returns:
        The card description, carrying no credential and no ciphertext.
    """
    from src.anubis.utils.connected_accounts.tool_factories import tool_names_for

    tool_names = tool_names_for(provider)
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
        "uses_form": bool(provider.uses_form),
        "pairing_instructions": provider.pairing_instructions or None,
        "install_url": provider.install_url,
        "fields": _describe_fields(provider),
        "already_connected": _connected_views(
            list(connected_accounts or []), provider.name
        ),
        "actions": ["apply", "cancel"],
    }


def build_connection_tools(
    context: Any,
    *,
    store: Any,
    user_id: str,
    assistant_id: str,
    connected_accounts: list[dict[str, Any]],
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

    Returns:
        The tools to append to this turn's tool list.
    """
    from src.anubis.utils.connected_accounts.providers import (
        catalog_providers,
        get_provider,
    )

    @tool
    async def connect_account(provider: str = "gmail") -> dict[str, Any]:
        """Ask the owner to connect one of their accounts, in this chat.

        Call this tool when the owner asks to connect, link, or add an account
        (an email account, a custom connector, a machine), or asks the assistant
        to do something that needs an account that is not connected yet — read
        email, send a message, use a tool from one of their servers. Calling this
        tool presents the owner with a connect card in the conversation; the
        owner completes it there. Do not ask the owner to type an address, a
        password, or a token into the chat, and never repeat one the owner sends.

        The run pauses while the owner completes the card. When the run resumes
        this tool reports which accounts are connected, and the account's tools
        become available in the same turn, so a request that prompted the
        connection can be carried out immediately afterwards.

        Args:
            provider: Which provider to connect. Defaults to "gmail". Other
                values: "custom_mcp" for the owner's own server, "desktop_mcp"
                for one of the owner's machines, or any provider named in the
                catalog.
        """
        provider_name = str(provider or "gmail").strip().lower()
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
            }
        if not resolved_provider.uses_form:
            # A machine connects itself when the owner runs the daemon; there is
            # no card to complete, so the instructions are the whole answer.
            return {
                "status": "instructions",
                "provider": resolved_provider.name,
                "message": resolved_provider.pairing_instructions,
                "install_url": resolved_provider.install_url,
            }

        card = build_connect_card(resolved_provider, connected_accounts)

        # Everything after this line runs twice: once when the run pauses, and
        # again from the top when the client resumes it. Nothing above may have
        # side effects, and nothing below may assume in-memory state survived.
        decision = interrupt(card)

        # Only the decision type is read. A resume value carrying an
        # `email_address`, `app_password`, or `bearer_token` is ignored on
        # purpose — see the module docstring. The credential's only path into
        # this system is the connect endpoint, which verifies and encrypts it.
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
            }

        from src.anubis.utils.connected_accounts.tool_factories import tool_names_for

        connected_now = _connected_views(refreshed_accounts, resolved_provider.name)
        if not connected_now:
            # The card was dismissed after sign-in failed, or the sign-in never
            # completed. Reporting success here would have the avatar claim an
            # account it cannot reach, so say plainly that nothing was connected.
            hint = (
                " Note that a Gmail account needs a 16-character app password "
                "rather than the account password."
                if resolved_provider.is_mailbox
                else ""
            )
            return {
                "status": "not_connected",
                "provider": resolved_provider.name,
                "message": (
                    f"No {resolved_provider.display_name} account is connected. "
                    "The sign-in was not completed. Offer to try again." + hint
                ),
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
                "available now, in this turn. Carry on with what the owner asked."
            ),
        }

    # The former name stays callable for one release so an in-flight prompt or
    # a cached tool call keeps working.
    connect_mailbox_account = tool("connect_mailbox_account")(connect_account.coroutine)
    connect_mailbox_account.description = (
        "Alias of connect_account. Prefer connect_account. "
        + connect_account.description
    )

    return [connect_account, connect_mailbox_account]
