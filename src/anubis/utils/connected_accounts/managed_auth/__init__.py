"""Connecting accounts through a provider whose OAuth client is already certified.

See :mod:`base` for why this path exists and what it guarantees.
"""

from src.anubis.utils.connected_accounts.managed_auth.base import (
    STATE_CONNECTED,
    STATE_EXPIRED,
    STATE_FAILED,
    STATE_PENDING,
    ManagedAuthError,
    ManagedAuthNotConfigured,
    ManagedAuthProvider,
    ManagedConnection,
    get_managed_auth_provider,
    managed_auth_available,
)

__all__ = [
    "STATE_CONNECTED",
    "STATE_EXPIRED",
    "STATE_FAILED",
    "STATE_PENDING",
    "ManagedAuthError",
    "ManagedAuthNotConfigured",
    "ManagedAuthProvider",
    "ManagedConnection",
    "get_managed_auth_provider",
    "managed_auth_available",
]
