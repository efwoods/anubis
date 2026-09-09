"""Connecting an account through a provider whose OAuth client is already certified.

Some companies will not let an application read their users' data on an
ordinary OAuth client. Reading a Gmail message needs a Google *restricted*
scope, which costs an annual third-party security assessment; Microsoft and
Yahoo have made comparable moves. Every other mail and calendar provider is
reached directly with an address and a password (``mail_autoconfig``,
``caldav_client``), so this path exists for the short list that refuses one.

The way through is a provider that has already passed those assessments and
lets an application connect a user's account on its certified client. The
trade is explicit and was accepted by the owner: the consent screen the user
sees carries that provider's name rather than Neural Nexus. In exchange the
account connects today, for anyone, with no cap, no assessment, and no fee.

What this package guarantees regardless of which vendor is configured:

* **No provider token is ever stored here.** The connection record keeps an
  identifier for the connection and nothing else; the vendor holds and refreshes
  the token. A database read cannot yield a Google credential because there is
  none to yield.
* **The vendor is replaceable.** Everything the rest of the codebase touches is
  the small protocol below, so swapping the provider — or moving to Neural
  Nexus's own certified client later — is one module, not a rewrite.
* **The user signs in with the same address and password they already know**, on
  the provider's own page. Nothing is typed into Neural Nexus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

# Connection states, normalised across vendors so callers never branch on a
# vendor's own spelling.
STATE_PENDING = "pending"
STATE_CONNECTED = "connected"
STATE_FAILED = "failed"
STATE_EXPIRED = "expired"


class ManagedAuthError(Exception):
    """The managed-auth provider refused or could not be reached."""


class ManagedAuthNotConfigured(ManagedAuthError):
    """No managed-auth provider is configured on this server.

    Raised rather than returned so a route answers 503 with a sentence naming
    the missing setting, instead of a card that opens an empty popup.
    """


@dataclass
class ManagedConnection:
    """One in-flight or finished connection at the provider."""

    connection_id: str
    authorization_url: str = ""
    state: str = STATE_PENDING
    account_identifier: str = ""
    display_label: str = ""
    toolkit: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_connected(self) -> bool:
        """Whether the account is live and usable."""
        return self.state == STATE_CONNECTED


class ManagedAuthProvider(Protocol):
    """What the rest of the codebase needs from a managed-auth vendor."""

    name: str

    async def ensure_auth_config(self, toolkit: str) -> str:
        """Return the vendor's configuration identifier for one toolkit."""
        ...

    async def start_connection(
        self, *, user_id: str, toolkit: str, callback_url: str = ""
    ) -> ManagedConnection:
        """Begin a sign-in and return the URL the popup should open."""
        ...

    async def connection_state(self, connection_id: str) -> ManagedConnection:
        """Report where one connection has got to."""
        ...

    async def list_tools(self, toolkit: str) -> list[dict[str, Any]]:
        """Describe the tools a connected account of this toolkit provides."""
        ...

    async def execute_tool(
        self, *, tool_slug: str, user_id: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Run one tool as the given user against their connected account."""
        ...

    async def disconnect(self, connection_id: str) -> None:
        """Forget one connected account at the vendor."""
        ...

    async def verify_configuration(self) -> dict[str, Any]:
        """Check the credentials and endpoints without changing anything."""
        ...


def get_managed_auth_provider(context: Any) -> ManagedAuthProvider:
    """Build the configured managed-auth provider.

    Raises :class:`ManagedAuthNotConfigured` when no provider is set up, which
    is the ordinary state of a server that has no need of one.
    """
    name = str(getattr(context, "managed_auth_provider", None) or "").strip().lower()
    if not name:
        raise ManagedAuthNotConfigured(
            "No managed-auth provider is configured. Set MANAGED_AUTH_PROVIDER "
            "and the provider's API key to connect accounts that refuse an "
            "account password."
        )
    if name == "composio":
        from src.anubis.utils.connected_accounts.managed_auth.composio import (
            ComposioProvider,
        )

        return ComposioProvider(context)
    raise ManagedAuthNotConfigured(
        f"MANAGED_AUTH_PROVIDER is set to {name!r}, which this server does not "
        "know how to use."
    )


def managed_auth_available(context: Any) -> bool:
    """Whether an account needing a certified client could be connected now."""
    try:
        get_managed_auth_provider(context)
    except ManagedAuthNotConfigured:
        return False
    return True
