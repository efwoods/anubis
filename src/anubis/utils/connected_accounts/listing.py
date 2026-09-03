"""One listing shape for everything the personal avatar is connected to.

Accounts (a mailbox, a custom server) and devices (a machine running the Neural
Nexus daemon) are stored differently — accounts in ``connected_accounts``,
devices in the Model Context Protocol registration and connection namespaces —
because their lifecycles differ: a device registers itself and is adopted, an
account is connected by the owner. The owner should not have to know that.
``GET /list_connections`` returns both through the projections here, and the
"+" menu and the settings screen render one row type.

Connection keys are prefixed so a single toggle endpoint can tell which store a
key belongs to: ``account:{provider}:{address}`` and ``device:{device_id}``.
"""

from __future__ import annotations

from typing import Any

ACCOUNT_KEY_PREFIX = "account:"
DEVICE_KEY_PREFIX = "device:"

# Platform values the daemons report, mapped to the frontend's icon keys.
_PLATFORM_ICON_KEYS = {
    "ubuntu": "ubuntu",
    "linux": "ubuntu",
    "macos": "apple",
    "darwin": "apple",
    "ios": "ios",
    "windows": "windows",
}


def split_connection_key(connection_key: str) -> tuple[str, str]:
    """Return ``("account" | "device", identifier)`` for a prefixed key.

    Raises ``ValueError`` for anything else, so a route can answer 400 rather
    than guessing which store a bare key meant.
    """
    key = str(connection_key or "").strip()
    if key.startswith(ACCOUNT_KEY_PREFIX):
        return "account", key[len(ACCOUNT_KEY_PREFIX) :]
    if key.startswith(DEVICE_KEY_PREFIX):
        return "device", key[len(DEVICE_KEY_PREFIX) :]
    raise ValueError(
        "A connection key must start with 'account:' or 'device:'; "
        f"got {connection_key!r}."
    )


def account_connection_view(record: dict[str, Any]) -> dict[str, Any]:
    """Project one connected-account record into the unified listing row.

    Built on ``public_account_view`` so the whitelist that keeps ciphertext and
    server URLs out of responses applies here too; the provider registry adds
    the presentation fields the record does not carry (icon, category, display
    name), and the tool factories supply the tool count.
    """
    from src.anubis.utils.connected_accounts.providers import get_provider
    from src.anubis.utils.connected_accounts.store import (
        STATUS_CONNECTED,
        public_account_view,
    )
    from src.anubis.utils.connected_accounts.tool_factories import tool_names_for

    view = public_account_view(record)
    provider = get_provider(str(view.get("provider") or ""))
    tool_names = tool_names_for(provider, record) if provider is not None else []
    sub_label = view.get("account_address") or ""
    if provider is not None and provider.kind == "mcp_server":
        # The owner typed the server URL; the listing shows the tool count
        # rather than reading the URL back (see public_account_view).
        sub_label = f"{len(tool_names)} tools"
    return {
        "connection_key": f"{ACCOUNT_KEY_PREFIX}{view.get('account_key')}",
        "source": "account",
        "provider": view.get("provider"),
        "provider_display_name": provider.display_name
        if provider
        else view.get("provider"),
        "category": provider.category if provider else "custom",
        "kind": view.get("kind"),
        "display_label": view.get("display_label"),
        "sub_label": sub_label,
        "status": view.get("status"),
        "connected": view.get("status") == STATUS_CONNECTED,
        "online": None,
        "icon_key": provider.icon_key if provider else "custom",
        "tool_count": len(tool_names),
        "tool_names": tool_names,
        "connected_at": view.get("connected_at"),
        "assistant_id": view.get("assistant_id"),
        "disconnect_endpoint": "/disconnect_account",
    }


def device_connection_view(device: dict[str, Any]) -> dict[str, Any]:
    """Project one device row (as ``GET /list_mcp_connections`` shapes it)."""
    platform = str(device.get("platform") or "").lower()
    label = device.get("device_label") or device.get("server_name") or "Machine"
    online = device.get("online")
    return {
        "connection_key": f"{DEVICE_KEY_PREFIX}{device.get('device_id')}",
        "source": "device",
        "provider": "desktop_mcp",
        "provider_display_name": "Your machines",
        "category": "device",
        "kind": "data_source",
        "display_label": label,
        "sub_label": (platform or "machine")
        + (" · online" if online else " · offline"),
        "status": "connected" if device.get("connected") else "registered",
        "connected": bool(device.get("connected")),
        "online": online,
        "icon_key": _PLATFORM_ICON_KEYS.get(platform, "mcp"),
        "tool_count": 0,
        "tool_names": [],
        "connected_at": device.get("connected_at"),
        "assistant_id": device.get("bound_assistant_id"),
        "device_id": device.get("device_id"),
        "disconnect_endpoint": "/disconnect_mcp",
    }
