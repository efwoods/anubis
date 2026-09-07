"""External accounts connected to a user's personal avatar.

Modules, split along the line between "what a provider is", "what the user has
connected", and "how a connection is made and used":

- :mod:`providers` — the static registry. One row per supported provider,
  carrying its account ``kind`` (a security boundary; see that module), its
  credential mechanism, and its catalog presentation. THE extension point.
- :mod:`store` — the facade every caller uses for per-user records, one per
  connected account, bound to a single avatar.
- :mod:`repository` — where those records live: the ``connected_accounts``
  Postgres table, with an in-memory twin for tests.
- :mod:`connect_handlers` — one proving-and-describing function per credential
  mechanism; ``POST /connect_account`` dispatches through it.
- :mod:`tool_factories` — which tools an account of each kind contributes.
- :mod:`listing` — the unified row shape accounts and devices share.
- :mod:`connection_tools` — the in-chat connect card, and direct connects.
- :mod:`connection_cards` — the persisted card record a reply carries.
- :mod:`oauth_state`, :mod:`oauth_providers`, :mod:`oauth_flow`,
  :mod:`pending_logins` — popup sign-in through a vendor's own OAuth page
  (Google, GitHub, X, Vercel), signed state, PKCE, refresh.
- :mod:`mcp_oauth` — OAuth for custom Model Context Protocol servers
  (discovery, dynamic client registration).
- :mod:`plaid_link` — banks through Plaid Link.
- :mod:`browser_sessions`, :mod:`browser_login` — the live browser the owner
  signs in through for sites with no OAuth, and the durable sessions kept.
- :mod:`browser_session_tools`, :mod:`recipes` — tools inside those sessions
  and vendor usage recipes; :mod:`website_tools` — crawl and audit websites;
  :mod:`vendor_api_tools`, :mod:`finance_tools` — API tools per vendor and bank.
- :mod:`mailbox_credentials` — mailbox credentials for any record mechanism.
- :mod:`mcp_server_tools` — tools from the owner's own Model Context Protocol servers.

Nothing here talks to a provider's servers at import time. The mailbox client
lives in ``src/anubis/utils/tools/email/imap_client.py`` and the tools the
model calls live beside it in ``mailbox_tools.py``, so this package stays
importable without pulling in a mail stack.
"""

from src.anubis.utils.connected_accounts.providers import (
    GMAIL_PROVIDER,
    PROVIDER_REGISTRY,
    ConnectedAccountProvider,
    ConnectFieldSpec,
    catalog_providers,
    get_provider,
    mailbox_providers,
    social_providers,
)
from src.anubis.utils.connected_accounts.store import (
    STATUS_CONNECTED,
    STATUS_NEEDS_RECONNECT,
    account_key,
    bound_accounts_for,
    build_account_record,
    clear_connected_account,
    connected_account_namespace,
    deduplicate_label,
    derive_display_label,
    get_connected_account,
    mark_account_needs_reconnect,
    public_account_view,
    read_connected_accounts,
    save_connected_account,
)

__all__ = [
    "GMAIL_PROVIDER",
    "PROVIDER_REGISTRY",
    "STATUS_CONNECTED",
    "STATUS_NEEDS_RECONNECT",
    "ConnectFieldSpec",
    "ConnectedAccountProvider",
    "account_key",
    "bound_accounts_for",
    "build_account_record",
    "catalog_providers",
    "clear_connected_account",
    "connected_account_namespace",
    "deduplicate_label",
    "derive_display_label",
    "get_connected_account",
    "get_provider",
    "mailbox_providers",
    "mark_account_needs_reconnect",
    "public_account_view",
    "read_connected_accounts",
    "save_connected_account",
    "social_providers",
]
