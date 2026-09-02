"""External accounts connected to a user's personal avatar.

Two modules, split along the line between "what a provider is" and "what the
user has connected":

- :mod:`providers` — the static registry. One row per supported provider,
  carrying its account ``kind`` (a security boundary; see that module) and the
  credential mechanism its connect flow uses.
- :mod:`store` — the per-user records, one per connected account, held in the
  cross-thread store and bound to a single avatar.

Nothing here talks to a provider's servers. The mailbox client lives in
``src/anubis/utils/tools/email/imap_client.py`` and the tools the model calls
live beside it in ``mailbox_tools.py``, so this package stays importable without
pulling in a mail stack.
"""

from src.anubis.utils.connected_accounts.providers import (
    GMAIL_PROVIDER,
    PROVIDER_REGISTRY,
    ConnectedAccountProvider,
    ConnectFieldSpec,
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
    "clear_connected_account",
    "connected_account_namespace",
    "deduplicate_label",
    "derive_display_label",
    "get_provider",
    "mailbox_providers",
    "mark_account_needs_reconnect",
    "public_account_view",
    "read_connected_accounts",
    "save_connected_account",
    "social_providers",
]
