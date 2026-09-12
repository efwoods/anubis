"""Watching connected mailboxes with IMAP IDLE, so new mail wakes the avatar at once.

A thirty-minute poll was the difference between an avatar that answers a
message and one that answers it eventually. IDLE closes that gap: the
connection is held open and the server speaks first, so a message that arrives
now is triaged now.

**One task per mailbox, supervised.** Each watcher owns one connection and one
mailbox, and a watcher that dies is restarted with a backoff rather than taken
as a reason to stop watching the others. A mailbox whose server refuses IDLE
degrades on its own into a slow loop — ``wait_for_new_mail`` returns False, the
watcher fetches anyway, and the effect is the old polling behaviour for that
one account instead of a failure.

**Why this lives beside the poller rather than replacing it.** The interval
poll is still the safety net: IDLE tells you something arrived, not what you
missed while the process was restarting. The poll catches up; IDLE keeps up.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# How long to wait after a watcher fails before opening its connection again.
# Grows on repeated failure so a mailbox that is rejecting credentials is not
# reconnected in a tight loop.
_BACKOFF_START_SECONDS = 30.0
_BACKOFF_CEILING_SECONDS = 900.0


class MailboxIdleWatchers:
    """Supervises one IDLE watcher per connected mailbox."""

    def __init__(self, context: Any) -> None:
        """Hold the configuration every watcher reads."""
        self.context = context
        self._tasks: dict[str, asyncio.Task] = {}
        self._supervisor: asyncio.Task | None = None
        self._stopping = False

    @property
    def enabled(self) -> bool:
        """Whether IDLE watching is switched on for this deployment."""
        return str(
            getattr(self.context, "imap_idle_enabled", "true") or "true"
        ).strip().lower() in {"true", "1", "yes"}

    def start(self) -> None:
        """Begin supervising; safe to call once per process."""
        if not self.enabled or self._supervisor is not None:
            return
        self._supervisor = asyncio.create_task(self._supervise_forever())

    async def stop(self) -> None:
        """Cancel every watcher and the supervisor, for shutdown."""
        self._stopping = True
        if self._supervisor is not None:
            self._supervisor.cancel()
            self._supervisor = None
        for task in list(self._tasks.values()):
            task.cancel()
        self._tasks.clear()

    async def _supervise_forever(self) -> None:
        """Keep one watcher running per connected mailbox as accounts change."""
        while not self._stopping:
            try:
                await self._reconcile()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - supervision outlives one bad pass
                logger.exception("Could not reconcile the mailbox watchers.")
            await asyncio.sleep(120.0)

    async def _reconcile(self) -> None:
        """Start watchers for new mailboxes; drop those no longer connected."""
        from src.anubis.utils.connected_accounts.repository import (
            get_repository as accounts_repository,
        )

        accounts = accounts_repository()
        if accounts is None:
            return
        records = await accounts.list_by_kind("mailbox", "connected")
        live_keys: set[str] = set()
        for record in records:
            account_key = str(record.get("account_key") or "")
            if not account_key:
                continue
            live_keys.add(account_key)
            task = self._tasks.get(account_key)
            if task is None or task.done():
                self._tasks[account_key] = asyncio.create_task(
                    self._watch_mailbox(record)
                )
        for account_key in list(self._tasks):
            if account_key not in live_keys:
                self._tasks.pop(account_key).cancel()

    async def _watch_mailbox(self, record: dict[str, Any]) -> None:
        """Hold IDLE on one mailbox, fetching whenever the server speaks."""
        from src.anubis.utils.connected_accounts.mailbox_credentials import (
            mailbox_credentials_for,
        )
        from src.anubis.utils.inbox.poller import poll_now_for_user
        from src.anubis.utils.tools.email.imap_client import wait_for_new_mail

        account_key = str(record.get("account_key") or "")
        user_id = str(record.get("user_id") or record.get("owner_user_id") or "")
        if not (account_key and user_id):
            return
        backoff = _BACKOFF_START_SECONDS
        refresh_seconds = float(
            getattr(self.context, "imap_idle_refresh_seconds", 0) or 1500.0
        )

        while not self._stopping:
            try:
                from src.anubis.utils.inbox.poller import inbox_store

                credentials = await mailbox_credentials_for(
                    record, self.context, store=inbox_store(), user_id=user_id
                )
                arrived = await asyncio.to_thread(
                    wait_for_new_mail,
                    credentials,
                    timeout_seconds=refresh_seconds,
                )
                backoff = _BACKOFF_START_SECONDS
                if arrived:
                    # The fetch, the triage and the publication check all live
                    # in the poller's per-user pass, so IDLE only decides WHEN
                    # that runs. One code path reads a mailbox, whichever woke
                    # it up.
                    await poll_now_for_user(self.context, user_id)
            except asyncio.CancelledError:
                raise
            except Exception as watch_error:  # noqa: BLE001 - one mailbox only
                logger.warning(
                    "The IDLE watcher for %s failed; retrying in %.0fs: %s",
                    account_key,
                    backoff,
                    watch_error,
                )
                await asyncio.sleep(backoff)
                backoff = min(_BACKOFF_CEILING_SECONDS, backoff * 2)
