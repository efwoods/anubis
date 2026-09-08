"""Helpers shared by the connected-account tests.

Gmail signs in through Google's popup now, so the app-password path (still
honoured for records connected before the popup landed) is exercised by
registering the unregistered legacy row under the ``gmail`` name for one test.
Likewise no catalog row is "coming soon" any more, so tests of that refusal
register a throwaway row.
"""

from __future__ import annotations

from dataclasses import replace

from src.anubis.utils.connected_accounts import providers as providers_module


def use_legacy_gmail(monkeypatch) -> None:
    """Register the app-password Gmail row for the duration of one test."""
    monkeypatch.setitem(
        providers_module.PROVIDER_REGISTRY,
        "gmail",
        providers_module.GMAIL_APP_PASSWORD_PROVIDER,
    )


def register_coming_soon_provider(monkeypatch, name: str = "slack"):
    """Register (or re-register) ``name`` as a coming-soon row for one test."""
    base = providers_module.PROVIDER_REGISTRY.get(name) or providers_module.SLACK_PROVIDER
    row = replace(
        base,
        name=name,
        availability=providers_module.AVAILABILITY_COMING_SOON,
    )
    monkeypatch.setitem(providers_module.PROVIDER_REGISTRY, name, row)
    return row


def open_authorization(monkeypatch) -> None:
    """Make every custom-connector probe report an open (no-login) server."""
    from src.anubis.utils.connected_accounts import mcp_oauth

    async def _open(server_url, context, *, http_client=None):
        return {
            "status": mcp_oauth.AUTHORIZATION_OPEN,
            "www_authenticate": None,
            "resource_metadata_url": None,
        }

    monkeypatch.setattr(mcp_oauth, "probe_authorization", _open)
