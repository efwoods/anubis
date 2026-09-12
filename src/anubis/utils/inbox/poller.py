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
    NOTIFY_ONLY_SOURCE_KINDS,
    STATE_IGNORED,
    STATE_PENDING_OWNER,
    STATE_RESOLVED,
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


async def _learn_from_publication_notice(
    context: Any,
    *,
    user_id: str,
    assistant_id: str,
    message: dict[str, Any],
    external_id: str,
) -> None:
    """Treat a message as a possible "you published something" notice.

    Cheap to skip and never fatal: the sender's domain is checked against the
    provider registry first, so the great majority of mail costs one dictionary
    lookup and nothing else. Only mail from a platform the owner connected
    reaches a model.
    """
    sender = str(message.get("sender") or "")
    if not sender:
        return
    try:
        from src.anubis.utils.subscriptions.email_notifications import (
            handle_mail_as_publication,
            provider_for_sender,
        )

        if provider_for_sender(sender) is None:
            return
        outcome = await handle_mail_as_publication(
            context,
            store=_store,
            user_id=user_id,
            personal_avatar_id=assistant_id,
            sender=sender,
            subject=str(message.get("subject") or ""),
            body_text=str(message.get("body_text") or ""),
            message_id=external_id,
        )
    except Exception:  # noqa: BLE001 - triage must run whatever happens here
        logger.exception("Could not read a message as a publication notice.")
        return
    if outcome:
        logger.info(
            "A publication notice from %s resolved as %s.",
            sender,
            outcome.get("status"),
        )


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
    # A platform's "your video is live" notice is both something the owner may
    # want to see and an announcement that the avatar's person published
    # something. Both readings are honoured: this runs alongside triage rather
    # than instead of it, so the inbox behaves exactly as it did while the
    # avatar also learns from what the notice points at.
    await _learn_from_publication_notice(
        context,
        user_id=user_id,
        assistant_id=assistant_id,
        message=message,
        external_id=external_id,
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

    # An item this system wrote for the owner has no paused triage run behind
    # it: nothing ever interrupted, so there is no checkpoint for ``Command(resume=...)``
    # to deliver a decision to, and invoking the graph would start a fresh run
    # that triages a notification as though it were incoming mail. Record the
    # owner's decision and close the item here instead.
    if str(item.get("source_kind") or "") in NOTIFY_ONLY_SOURCE_KINDS:
        decision = str((human_response or {}).get("action") or "").strip().lower()
        await repository.update_item(
            item_id,
            state=STATE_IGNORED if decision == "ignore" else STATE_RESOLVED,
            owner_decision=human_response,
            resolved_at=datetime.now(UTC),
        )
        return await repository.get_item(item_id)

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
    """Poll every connected account (mailboxes directly, others through their tools)."""
    mail_result = await _poll_mailboxes(context, only_user_id=only_user_id)
    account_result = await poll_other_accounts(context, only_user_id=only_user_id)
    return {
        "polled": int(mail_result.get("polled") or 0) + int(account_result.get("polled") or 0),
        "new_items": int(mail_result.get("new_items") or 0) + int(account_result.get("new_items") or 0),
        "mailboxes": mail_result,
        "accounts": account_result,
        "at": datetime.now(UTC).isoformat(),
    }


async def poll_other_accounts(
    context: Any, *, only_user_id: str | None = None
) -> dict[str, Any]:
    """Find new items on every non-mailbox connected account and triage each one.

    Discovery runs through the account's own tools (see ``sources.py``), so a
    newly connected kind of account is polled with no code of its own. Each
    account is visited at most once per ``INBOX_ACCOUNT_POLL_INTERVAL_SECONDS``
    because a discovery pass costs a model call and, for a signed-in site, a
    browser visit. A poll asked for by the owner (``only_user_id`` given) skips
    that spacing.
    """
    from src.anubis.utils.connected_accounts.repository import (
        get_repository as accounts_repository,
    )
    from src.anubis.utils.inbox.sources import discover_new_items, is_message_source

    repository = get_inbox_repository()
    accounts = accounts_repository()
    if repository is None or accounts is None:
        return {"polled": 0, "new_items": 0}
    enabled = str(getattr(context, "inbox_account_poll_enabled", None) or "true").strip().lower()
    if enabled not in ("1", "true", "yes", "on"):
        return {"polled": 0, "new_items": 0, "disabled": True}
    interval = float(getattr(context, "inbox_account_poll_interval_seconds", None) or 1800.0)
    limit = int(getattr(context, "inbox_discovery_max_items", None) or 10)
    max_steps = int(getattr(context, "inbox_discovery_max_steps", None) or 8)
    records: list[dict[str, Any]] = list(await accounts.list_all_connected())
    now = datetime.now(UTC)
    polled = 0
    new_items = 0
    for record in records:
        if not is_message_source(record) or record.get("kind") == "mailbox":
            continue
        account_key = record.get("account_key")
        user_id = _owner_of(record)
        assistant_id = record.get("assistant_id")
        if not (account_key and user_id and assistant_id):
            continue
        if only_user_id is not None and user_id != only_user_id:
            continue
        poll_state = await repository.get_poll_state(account_key) or {}
        last_polled = _as_datetime(poll_state.get("last_polled_at"))
        if last_polled and only_user_id is None and (now - last_polled).total_seconds() < interval:
            continue
        polled += 1
        try:
            messages, _answer = await discover_new_items(
                context, _store, record, since=last_polled, limit=limit, max_steps=max_steps
            )
        except Exception as discovery_error:  # noqa: BLE001 - one account must not stop the rest
            logger.warning("Inbox discovery failed for %s: %s", account_key, discovery_error)
            await repository.set_poll_state(
                account_key=account_key,
                user_id=user_id,
                assistant_id=assistant_id,
                last_seen_uid=poll_state.get("last_seen_uid"),
                last_error=str(discovery_error)[:400],
            )
            continue
        for message in messages:
            result = await run_inbox_for_message(
                context,
                user_id=user_id,
                assistant_id=assistant_id,
                account_key=account_key,
                message=message,
                source_kind=str(record.get("provider") or "account"),
            )
            if result is not None:
                new_items += 1
        await repository.set_poll_state(
            account_key=account_key,
            user_id=user_id,
            assistant_id=assistant_id,
            last_seen_uid=poll_state.get("last_seen_uid"),
            last_error=None,
        )
    return {"polled": polled, "new_items": new_items}


def _as_datetime(value: Any) -> datetime | None:
    """Return a timezone-aware datetime from a stored timestamp, or ``None``."""
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError:
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


async def _poll_mailboxes(
    context: Any, *, only_user_id: str | None = None
) -> dict[str, Any]:
    """Fetch unseen mail from every connected mailbox and triage each message."""
    from src.anubis.utils.connected_accounts.repository import (
        get_repository as accounts_repository,
    )
    from src.anubis.utils.tools.email.imap_client import (
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
            from src.anubis.utils.connected_accounts.mailbox_credentials import (
                mailbox_credentials_for,
            )

            credentials = await mailbox_credentials_for(
                record, context, store=_store, user_id=user_id
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


def inbox_store() -> Any:
    """Return the store the inbox runtime was published with.

    The IDLE watchers need it to decrypt a mailbox credential, and they must not
    reach into this module's private name to get it.
    """
    return _store


async def poll_now_for_user(context: Any, user_id: str) -> dict[str, Any]:
    """Fetch and triage one owner's mail immediately, skipping interval spacing.

    Called by an IDLE watcher the moment its server reports an arrival, so the
    fetch, the triage, and the publication-notice check all stay on the single
    path that reads a mailbox — IDLE decides only *when* that path runs, never
    what it does.
    """
    return await _poll_mailboxes(context, only_user_id=user_id)
