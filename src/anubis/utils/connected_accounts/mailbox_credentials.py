"""Live mailbox credentials from a stored record, whatever its mechanism.

A mailbox record connected before Google sign-in landed holds an encrypted
app password; one connected through the popup holds an encrypted OAuth token
bundle. Every mail operation — the tools, the inbox poller, the reply sender,
the writing-sample import — asks this one function for credentials so none of
them has to know which kind of record it holds. OAuth records present a fresh
access token through SASL XOAUTH2 (``imap_client.py``); an expired or revoked
token raises ``OAuthReconnectRequired`` and the record is flagged for the owner.
"""

from __future__ import annotations

from typing import Any


async def mailbox_credentials_for(
    record: dict[str, Any],
    context: Any,
    *,
    store: Any = None,
    user_id: str | None = None,
) -> Any:
    """Build ``MailboxCredentials`` for one connected mailbox record.

    Raises ``SecretDecryptionError`` / ``SecretEncryptionNotConfiguredError``
    for an unreadable app password and ``OAuthReconnectRequired`` for a token
    the vendor no longer honours.
    """
    from src.anubis.utils.tools.email.imap_client import (
        AUTH_MECHANISM_PASSWORD,
        AUTH_MECHANISM_XOAUTH2,
        MailboxCredentials,
    )

    timeout_seconds = float(
        getattr(context, "mailbox_request_timeout_seconds", None) or 30.0
    )
    common = {
        "account_address": record["account_address"],
        "imap_host": record["imap_host"],
        "imap_port": int(record.get("imap_port") or 993),
        "smtp_host": record.get("smtp_host"),
        "smtp_port": int(record.get("smtp_port") or 587),
        "drafts_mailbox": record.get("drafts_mailbox") or "Drafts",
        "timeout_seconds": timeout_seconds,
    }
    mechanism = str(record.get("credential_mechanism") or "app_password")
    if mechanism == "oauth":
        from src.anubis.utils.connected_accounts.oauth_flow import (
            get_fresh_access_token,
        )

        owner = user_id or record.get("user_id") or ""
        access_token = await get_fresh_access_token(context, store, str(owner), record)
        return MailboxCredentials(
            password="",
            auth_mechanism=AUTH_MECHANISM_XOAUTH2,
            access_token=access_token,
            **common,
        )

    from src.anubis.utils.secret_store import decrypt_secret

    return MailboxCredentials(
        password=decrypt_secret(record["encrypted_secret"], context),
        auth_mechanism=AUTH_MECHANISM_PASSWORD,
        **common,
    )
