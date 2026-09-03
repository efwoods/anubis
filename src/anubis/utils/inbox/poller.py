"""Feeding the inbox graph: poll connected mailboxes, run one triage per message.

The graph runs IN-PROCESS with the application's durable checkpointer (the
same ``AsyncPostgresSaver`` the chat uses), on a thread whose id is the inbox
item's id. A pending human decision therefore survives a restart and is resumed
by ``resume_inbox_item`` with a ``Command(resume=[HumanResponse])`` — from the
panel, from chat, or from the Agent Inbox app.

``poll_connected_mailboxes`` is what the lifespan task and ``POST /inbox/poll``
both call: every connected mailbox is opened read-only, unseen mail newer than
the remembered UID is fetched, each message not yet recorded becomes an item,
and its triage run starts. Failures on one mailbox never stop the others.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from langgraph.types import Command

from src.anubis.utils.inbox.repository import (
    STATE_PENDING_OWNER,
    get_inbox_repository,
    sender_domain_of,
)

logger = logging.getLogger(__name__)

_checkpointer: Any | None = None
_store: Any | None = None
_compiled_graph: Any | None = None


def set_inbox_runtime(checkpointer: Any, store: Any) -> None:
    """Publish the checkpointer and store the in-process graph runs with."""
    global _checkpointer, _store, _compiled_graph
    _checkpointer = checkpointer
    _store = store
    _compiled_graph = None


def _graph() -> Any:
    global _compiled_graph
    if _compiled_graph is None:
        from src.subgraphs.inbox.graph import build_inbox_graph

        _compiled_graph = build_inbox_graph(checkpointer=_checkpointer, store=_store)
    return _compiled_graph


def _run_config(
    item: dict[str, Any], assistant: dict[str, Any] | None
) -> dict[str, Any]:
    metadata = dict((assistant or {}).get("metadata") or {})
    metadata.setdefault("user_id", item["user_id"])
    metadata.setdefault("is_personal_avatar_of_creator", True)
    return {
        "configurable": {
            "thread_id": item["item_id"],
            "user_id": item["user_id"],
            "assistant_id": item["assistant_id"],
            "user_ctx": {"name": None, "description": None},
            "assistant_ctx": {
                "name": (assistant or {}).get("name"),
                "description": (assistant or {}).get("description"),
                "assistant_id": item["assistant_id"],
                "metadata": metadata,
            },
        }
    }


async def run_inbox_for_message(
    context: Any,
    *,
    user_id: str,
    assistant_id: str,
    account_key: str | None,
    message: dict[str, Any],
    assistant: dict[str, Any] | None = None,
    source_kind: str = "email",
) -> dict[str, Any] | None:
    """Record one incoming message as an item and run its triage.

    Returns the item row after the run (or after it paused on the owner), or
    ``None`` when the message was already recorded.
    """
    repository = get_inbox_repository()
    if repository is None:
        return None
    external_id = str(
        message.get("rfc822_message_id") or message.get("message_id") or ""
    )
    if external_id:
        existing = await repository.find_item_by_external_id(
            assistant_id=assistant_id,
            source_kind=source_kind,
            account_key=account_key,
            external_id=external_id,
        )
        if existing is not None:
            return None
    recipients = message.get("recipients")
    if isinstance(recipients, str):
        recipients = [part.strip() for part in recipients.split(",") if part.strip()]
    item = await repository.create_item(
        {
            "user_id": user_id,
            "assistant_id": assistant_id,
            "source_kind": source_kind,
            "account_key": account_key,
            "external_id": external_id or None,
            "external_thread_id": message.get("thread_id"),
            "sender": message.get("sender"),
            "sender_domain": sender_domain_of(message.get("sender")),
            "recipients": list(recipients or []),
            "subject": message.get("subject"),
            "body_text": message.get("body_text"),
            "received_at": message.get("sent_at"),
            "state": STATE_PENDING_OWNER,
        }
    )
    initial_state = {
        "item_id": item["item_id"],
        "user_id": user_id,
        "assistant_id": assistant_id,
        "assistant_name": (assistant or {}).get("name") or "",
        "account_key": account_key,
        "message": {**message, "recipients": list(recipients or [])},
    }
    try:
        await _graph().ainvoke(
            initial_state, config=_run_config(item, assistant), context=context
        )
    except Exception as run_error:  # noqa: BLE001 - the item records the failure
        logger.exception(
            "Inbox triage failed for item %s: %s", item["item_id"], run_error
        )
        await repository.update_item(
            item["item_id"], state="failed", reason=str(run_error)
        )
    return await repository.get_item(item["item_id"])


async def resume_inbox_item(
    context: Any, *, item_id: str, human_response: dict[str, Any]
) -> dict[str, Any] | None:
    """Deliver the owner's decision to the paused run and let it finish."""
    repository = get_inbox_repository()
    if repository is None:
        return None
    item = await repository.get_item(item_id)
    if item is None:
        return None
    if item.get("state") != STATE_PENDING_OWNER:
        return item
    config = _run_config(item, None)
    try:
        await _graph().ainvoke(
            Command(resume=[human_response]), config=config, context=context
        )
    except Exception as resume_error:  # noqa: BLE001
        logger.exception("Inbox resume failed for item %s: %s", item_id, resume_error)
        await repository.update_item(item_id, state="failed", reason=str(resume_error))
    return await repository.get_item(item_id)


async def poll_connected_mailboxes(
    context: Any, *, only_user_id: str | None = None
) -> dict[str, Any]:
    """Fetch unseen mail from every connected mailbox and triage each message."""
    from src.anubis.utils.connected_accounts.repository import (
        get_repository as accounts_repository,
    )
    from src.anubis.utils.secret_store import decrypt_secret
    from src.anubis.utils.tools.email.imap_client import (
        MailboxCredentials,
        fetch_unseen_messages,
    )

    repository = get_inbox_repository()
    accounts = accounts_repository()
    if repository is None or accounts is None:
        return {"polled": 0, "new_items": 0}
    mailboxes = [
        record
        for record in await accounts.list_by_kind("mailbox", "connected")
        if only_user_id is None
        or record.get("assistant_id")
        and _owner_of(record) == only_user_id
    ]
    fetch_limit = int(getattr(context, "inbox_fetch_max_messages", None) or 20)
    new_items = 0
    for record in mailboxes:
        account_key = record.get("account_key")
        user_id = _owner_of(record)
        assistant_id = record.get("assistant_id")
        if not (account_key and user_id and assistant_id):
            continue
        poll_state = await repository.get_poll_state(account_key) or {}
        after_uid = poll_state.get("last_seen_uid")
        try:
            credentials = MailboxCredentials(
                account_address=record["account_address"],
                password=decrypt_secret(record["encrypted_secret"], context),
                imap_host=record["imap_host"],
                imap_port=int(record.get("imap_port") or 993),
                smtp_host=record.get("smtp_host"),
                smtp_port=int(record.get("smtp_port") or 587),
                timeout_seconds=float(
                    getattr(context, "mailbox_request_timeout_seconds", None) or 30.0
                ),
            )
            messages = await asyncio.to_thread(
                fetch_unseen_messages,
                credentials,
                after_uid=after_uid,
                limit=fetch_limit,
            )
        except Exception as fetch_error:  # noqa: BLE001 - one mailbox must not stop the rest
            logger.warning("Inbox poll failed for %s: %s", account_key, fetch_error)
            await repository.set_poll_state(
                account_key=account_key,
                user_id=user_id,
                assistant_id=assistant_id,
                last_seen_uid=after_uid,
                last_error=str(fetch_error)[:400],
            )
            continue
        highest_uid = after_uid
        for message in messages:
            try:
                uid_value = int(message.get("uid") or message.get("message_id") or 0)
                highest_uid = max(int(highest_uid or 0), uid_value)
            except (TypeError, ValueError):
                pass
            result = await run_inbox_for_message(
                context,
                user_id=user_id,
                assistant_id=assistant_id,
                account_key=account_key,
                message=message,
            )
            if result is not None:
                new_items += 1
        await repository.set_poll_state(
            account_key=account_key,
            user_id=user_id,
            assistant_id=assistant_id,
            last_seen_uid=highest_uid,
            last_error=None,
        )
    return {
        "polled": len(mailboxes),
        "new_items": new_items,
        "at": datetime.now(UTC).isoformat(),
    }


def _owner_of(record: dict[str, Any]) -> str | None:
    """Return the connected account's owner.

    The record carries the personal avatar it is bound to; the owner id is
    stored beside it by the repository (``user_id`` on the table) and mirrored
    into the record for the poller by ``list_by_kind``.
    """
    return record.get("user_id") or record.get("owner_user_id")


async def poll_forever(context: Any) -> None:
    """Poll on the configured interval until cancelled (the lifespan task)."""
    interval = float(getattr(context, "inbox_poll_interval_seconds", None) or 300.0)
    enabled = str(
        getattr(context, "inbox_poll_enabled", None) or "true"
    ).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if not enabled:
        return
    while True:
        try:
            await asyncio.sleep(interval)
            await poll_connected_mailboxes(context)
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001
            logger.debug("Inbox poll iteration failed", exc_info=True)
