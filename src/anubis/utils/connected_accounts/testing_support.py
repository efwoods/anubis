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
    """Register the pre-popup Gmail row for the duration of one test."""
    monkeypatch.setitem(
        providers_module.PROVIDER_REGISTRY,
        "gmail",
        providers_module.GMAIL_APP_PASSWORD_PROVIDER,
    )


def use_password_mailbox(monkeypatch) -> str:
    """Give ``email_account`` concrete servers for the duration of one test.

    The real row names no servers on purpose — they are discovered from the
    address. A test of the password path is not a test of discovery, and it must
    not reach the network to run, so this pins the row to a host that answers
    nowhere and, importantly, is not one of the providers that have withdrawn
    password authentication. Returns the provider name to use in a request body.
    """
    monkeypatch.setitem(
        providers_module.PROVIDER_REGISTRY,
        "email_account",
        replace(
            providers_module.EMAIL_ACCOUNT_PROVIDER,
            imap_host="imap.example.com",
            imap_port=993,
            smtp_host="smtp.example.com",
            smtp_port=587,
            discovers_servers=False,
        ),
    )
    return "email_account"


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
