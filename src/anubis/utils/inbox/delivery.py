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
    from src.anubis.utils.tools.email.imap_client import (
        send_message,
    )

    if not account_key:
        raise RuntimeError("The item has no connected account to reply from.")
    record = await get_connected_account(None, user_id, account_key)
    if not record or record.get("kind") != "mailbox":
        raise RuntimeError(f"No connected mailbox {account_key!r} to reply from.")
    if not record.get("send_supported", True):
        raise RuntimeError("This mailbox provider does not support sending.")
    from src.anubis.utils.connected_accounts.mailbox_credentials import (
        mailbox_credentials_for,
    )

    credentials = await mailbox_credentials_for(
        record, context, store=None, user_id=user_id
    )
    return await asyncio.to_thread(
        send_message,
        credentials,
        to_address=to_address,
        subject=subject,
        body_text=body_text,
        in_reply_to=in_reply_to,
    )
