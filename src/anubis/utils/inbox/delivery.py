"""Sending an inbox reply through the connected account it arrived on."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def send_email_reply(
    context: Any,
    *,
    user_id: str,
    account_key: str | None,
    to_address: str,
    subject: str,
    body_text: str,
    in_reply_to: str | None = None,
) -> dict[str, Any]:
    """Transmit a reply from the mailbox the message arrived on.

    The mailbox record is read through the connected-account facade and its
    credential decrypted only for the duration of the SMTP session.
    """
    from src.anubis.utils.connected_accounts import get_connected_account
    from src.anubis.utils.secret_store import decrypt_secret
    from src.anubis.utils.tools.email.imap_client import (
        MailboxCredentials,
        send_message,
    )

    if not account_key:
        raise RuntimeError("The item has no connected account to reply from.")
    record = await get_connected_account(None, user_id, account_key)
    if not record or record.get("kind") != "mailbox":
        raise RuntimeError(f"No connected mailbox {account_key!r} to reply from.")
    if not record.get("send_supported", True):
        raise RuntimeError("This mailbox provider does not support sending.")
    credentials = MailboxCredentials(
        account_address=record["account_address"],
        password=decrypt_secret(record["encrypted_secret"], context),
        imap_host=record["imap_host"],
        imap_port=int(record.get("imap_port") or 993),
        smtp_host=record.get("smtp_host"),
        smtp_port=int(record.get("smtp_port") or 587),
        drafts_mailbox=record.get("drafts_mailbox") or "Drafts",
        timeout_seconds=float(
            getattr(context, "mailbox_request_timeout_seconds", None) or 30.0
        ),
    )
    return await asyncio.to_thread(
        send_message,
        credentials,
        to_address=to_address,
        subject=subject,
        body_text=body_text,
        in_reply_to=in_reply_to,
    )
