"""Storage for the agent inbox: items, the owner's learned preferences, poll cursors.

``inbox_items``
    One row per incoming message the triage graph has seen. The row is the
    panel's source of truth — what the badge counts, what the panel lists, what
    the chat tools report — and mirrors the graph's pending ``HumanInterrupt``
    so the owner can act from the panel or from chat and both resume the same
    graph thread (``thread_id`` = ``item_id``).

``inbox_preferences``
    What the owner decided for a sender / message kind, counted. Read before
    classification as few-shot precedent and by the confidence score, written
    by every owner decision, so each decision moves the next similar message
    toward auto-response or toward ignore.

``inbox_poll_state``
    The highest UID seen per connected mailbox, so a poll fetches only new mail.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

# Item states.
STATE_PENDING_OWNER = (
    "pending_owner"  # waiting on the owner (notify, or a low-confidence draft)
)
STATE_AUTO_SENT = "auto_sent"  # sent above the confidence threshold
STATE_SENT = "sent"  # sent after the owner accepted or edited
STATE_IGNORED = "ignored"
STATE_RESOLVED = "resolved"  # a notification the owner acknowledged
STATE_FAILED = "failed"

OPEN_STATES = (STATE_PENDING_OWNER,)

# Triage decisions.
DECISION_IGNORE = "ignore"
DECISION_NOTIFY = "notify"
DECISION_RESPOND = "respond"

_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS inbox_items (
    item_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    assistant_id TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    account_key TEXT,
    external_id TEXT,
    external_thread_id TEXT,
    sender TEXT,
    sender_domain TEXT,
    recipients JSONB NOT NULL DEFAULT '[]',
    subject TEXT,
    body_text TEXT,
    received_at TIMESTAMPTZ,
    message_kind TEXT,
    decision TEXT,
    needs_owner_action BOOLEAN NOT NULL DEFAULT FALSE,
    reason TEXT,
    draft TEXT,
    confidence DOUBLE PRECISION,
    confidence_detail JSONB NOT NULL DEFAULT '{}',
    state TEXT NOT NULL,
    owner_decision JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at TIMESTAMPTZ,
    UNIQUE (assistant_id, source_kind, account_key, external_id)
);
CREATE INDEX IF NOT EXISTS inbox_items_open_idx ON inbox_items (assistant_id, state);

CREATE TABLE IF NOT EXISTS inbox_preferences (
    preference_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    assistant_id TEXT NOT NULL,
    sender TEXT,
    sender_domain TEXT,
    message_kind TEXT,
    decision TEXT NOT NULL,
    edit_summary TEXT,
    example_subject TEXT,
    count INTEGER NOT NULL DEFAULT 1,
    last_decided_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (assistant_id, sender, message_kind, decision)
);
CREATE INDEX IF NOT EXISTS inbox_preferences_lookup_idx
    ON inbox_preferences (assistant_id, sender_domain, message_kind);

CREATE TABLE IF NOT EXISTS inbox_poll_state (
    account_key TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    assistant_id TEXT NOT NULL,
    last_seen_uid BIGINT,
    last_polled_at TIMESTAMPTZ,
    last_error TEXT
);
"""

_ITEM_COLUMNS = (
    "item_id, user_id, assistant_id, source_kind, account_key, external_id, "
    "external_thread_id, sender, sender_domain, recipients, subject, body_text, "
    "received_at, message_kind, decision, needs_owner_action, reason, draft, "
    "confidence, confidence_detail, state, owner_decision, created_at, updated_at, "
    "resolved_at"
)
_ITEM_NAMES = [name.strip() for name in _ITEM_COLUMNS.split(",")]

_PREFERENCE_COLUMNS = (
    "preference_id, user_id, assistant_id, sender, sender_domain, message_kind, "
    "decision, edit_summary, example_subject, count, last_decided_at"
)
_PREFERENCE_NAMES = [name.strip() for name in _PREFERENCE_COLUMNS.split(",")]


def _now() -> datetime:
    return datetime.now(UTC)


def _isoformat(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


def sender_domain_of(sender: str | None) -> str:
    """Return the domain part of an address, lower-cased ("" when there is none)."""
    address = str(sender or "")
    if "<" in address and ">" in address:
        address = address.split("<", 1)[1].split(">", 1)[0]
    return address.rsplit("@", 1)[-1].strip().lower() if "@" in address else ""


def public_item_view(item: dict[str, Any]) -> dict[str, Any]:
    """Project an item as the panel and the chat tools see it (a snippet, not the body)."""
    body = str(item.get("body_text") or "")
    return {
        "item_id": item.get("item_id"),
        "assistant_id": item.get("assistant_id"),
        "source_kind": item.get("source_kind"),
        "account_key": item.get("account_key"),
        "sender": item.get("sender"),
        "subject": item.get("subject"),
        "snippet": body[:400],
        "received_at": _isoformat(item.get("received_at")),
        "message_kind": item.get("message_kind"),
        "decision": item.get("decision"),
        "needs_owner_action": bool(item.get("needs_owner_action")),
        "reason": item.get("reason"),
        "draft": item.get("draft"),
        "confidence": item.get("confidence"),
        "confidence_detail": item.get("confidence_detail") or {},
        "state": item.get("state"),
        "owner_decision": item.get("owner_decision"),
        "created_at": _isoformat(item.get("created_at")),
        "updated_at": _isoformat(item.get("updated_at")),
        "resolved_at": _isoformat(item.get("resolved_at")),
    }


class InMemoryInboxRepository:
    """Dictionary-backed twin for tests and the local dev server."""

    def __init__(self) -> None:
        """Start empty."""
        self.items: dict[str, dict[str, Any]] = {}
        self.preferences: dict[str, dict[str, Any]] = {}
        self.poll_state: dict[str, dict[str, Any]] = {}
        self.pool = None

    # -- items ---------------------------------------------------------------

    async def find_item_by_external_id(
        self,
        *,
        assistant_id: str,
        source_kind: str,
        account_key: str | None,
        external_id: str,
    ) -> dict[str, Any] | None:
        """Return the item already recorded for this incoming message, if any."""
        for item in self.items.values():
            if (
                item["assistant_id"] == assistant_id
                and item["source_kind"] == source_kind
                and (item.get("account_key") or None) == (account_key or None)
                and item.get("external_id") == external_id
            ):
                return dict(item)
        return None

    async def create_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Insert an item; return the stored row."""
        item_id = str(item.get("item_id") or uuid4())
        now = _now().isoformat()
        stored = {
            "recipients": [],
            "confidence_detail": {},
            "needs_owner_action": False,
            **item,
            "item_id": item_id,
            "created_at": now,
            "updated_at": now,
        }
        stored.setdefault("sender_domain", sender_domain_of(stored.get("sender")))
        self.items[item_id] = stored
        return dict(stored)

    async def update_item(self, item_id: str, **fields: Any) -> dict[str, Any] | None:
        """Merge fields into an item; return the updated row."""
        item = self.items.get(item_id)
        if item is None:
            return None
        item.update(fields)
        item["updated_at"] = _now().isoformat()
        return dict(item)

    async def get_item(self, item_id: str) -> dict[str, Any] | None:
        """Return one item."""
        item = self.items.get(item_id)
        return dict(item) if item else None

    async def list_items(
        self,
        *,
        assistant_id: str,
        states: tuple[str, ...] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return an avatar's items, newest first."""
        rows = [
            item
            for item in self.items.values()
            if item["assistant_id"] == assistant_id
            and (states is None or item["state"] in states)
        ]
        rows.sort(key=lambda item: item["created_at"], reverse=True)
        return [dict(row) for row in rows[:limit]]

    async def count_open(self, assistant_id: str) -> int:
        """How many items await the owner."""
        return sum(
            1
            for item in self.items.values()
            if item["assistant_id"] == assistant_id and item["state"] in OPEN_STATES
        )

    # -- preferences ---------------------------------------------------------

    async def recall_preferences(
        self,
        *,
        assistant_id: str,
        sender: str | None,
        sender_domain: str | None,
        message_kind: str | None,
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        """Preferences for this sender, then its domain, then the message kind."""
        rows = [
            p for p in self.preferences.values() if p["assistant_id"] == assistant_id
        ]

        def rank(preference: dict[str, Any]) -> int:
            if sender and preference.get("sender") == sender:
                return 0
            if sender_domain and preference.get("sender_domain") == sender_domain:
                return 1
            if message_kind and preference.get("message_kind") == message_kind:
                return 2
            return 9

        ranked = [p for p in rows if rank(p) < 9]
        ranked.sort(key=lambda p: (rank(p), -int(p.get("count") or 0)))
        return [dict(p) for p in ranked[:limit]]

    async def record_preference(
        self,
        *,
        user_id: str,
        assistant_id: str,
        sender: str | None,
        sender_domain: str | None,
        message_kind: str | None,
        decision: str,
        edit_summary: str | None = None,
        example_subject: str | None = None,
    ) -> dict[str, Any]:
        """Upsert one preference row (counted)."""
        key = (assistant_id, sender or "", message_kind or "", decision)
        for preference in self.preferences.values():
            if (
                preference["assistant_id"],
                preference.get("sender") or "",
                preference.get("message_kind") or "",
                preference["decision"],
            ) == key:
                preference["count"] = int(preference.get("count") or 0) + 1
                preference["last_decided_at"] = _now().isoformat()
                if edit_summary:
                    preference["edit_summary"] = edit_summary
                if example_subject:
                    preference["example_subject"] = example_subject
                return dict(preference)
        preference_id = str(uuid4())
        stored = {
            "preference_id": preference_id,
            "user_id": user_id,
            "assistant_id": assistant_id,
            "sender": sender,
            "sender_domain": sender_domain,
            "message_kind": message_kind,
            "decision": decision,
            "edit_summary": edit_summary,
            "example_subject": example_subject,
            "count": 1,
            "last_decided_at": _now().isoformat(),
        }
        self.preferences[preference_id] = stored
        return dict(stored)

    # -- poll state ----------------------------------------------------------

    async def get_poll_state(self, account_key: str) -> dict[str, Any] | None:
        """Return the poll cursor for a mailbox."""
        state = self.poll_state.get(account_key)
        return dict(state) if state else None

    async def set_poll_state(
        self,
        *,
        account_key: str,
        user_id: str,
        assistant_id: str,
        last_seen_uid: int | None,
        last_error: str | None = None,
    ) -> None:
        """Record the poll cursor for a mailbox."""
        self.poll_state[account_key] = {
            "account_key": account_key,
            "user_id": user_id,
            "assistant_id": assistant_id,
            "last_seen_uid": last_seen_uid,
            "last_polled_at": _now().isoformat(),
            "last_error": last_error,
        }


class PostgresInboxRepository:
    """Repository over the application's psycopg connection pool."""

    def __init__(self, pool: Any) -> None:
        """Bind to the application's ``AsyncConnectionPool``."""
        self.pool = pool

    async def _fetchall(self, sql: str, params: tuple = ()) -> list[tuple]:
        async with self.pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(sql, params)
                return await cursor.fetchall()

    async def _fetchone(self, sql: str, params: tuple = ()) -> tuple | None:
        async with self.pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(sql, params)
                return await cursor.fetchone()

    async def _execute(self, sql: str, params: tuple = ()) -> int:
        async with self.pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(sql, params)
                return int(cursor.rowcount or 0)

    @staticmethod
    def _item_row(row: tuple) -> dict[str, Any]:
        record = dict(zip(_ITEM_NAMES, row))
        record["item_id"] = str(record["item_id"])
        for key in ("received_at", "created_at", "updated_at", "resolved_at"):
            record[key] = _isoformat(record.get(key))
        return record

    async def find_item_by_external_id(
        self,
        *,
        assistant_id: str,
        source_kind: str,
        account_key: str | None,
        external_id: str,
    ) -> dict[str, Any] | None:
        """Return the item already recorded for this incoming message, if any."""
        row = await self._fetchone(
            f"SELECT {_ITEM_COLUMNS} FROM inbox_items WHERE assistant_id = %s AND "
            "source_kind = %s AND account_key IS NOT DISTINCT FROM %s AND external_id = %s;",
            (assistant_id, source_kind, account_key, external_id),
        )
        return self._item_row(row) if row else None

    async def create_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Insert an item; return the stored row."""
        from psycopg.types.json import Jsonb

        item_id = str(item.get("item_id") or uuid4())
        await self._execute(
            """
            INSERT INTO inbox_items
                (item_id, user_id, assistant_id, source_kind, account_key, external_id,
                 external_thread_id, sender, sender_domain, recipients, subject, body_text,
                 received_at, message_kind, decision, needs_owner_action, reason, draft,
                 confidence, confidence_detail, state, owner_decision)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s);
            """,
            (
                item_id,
                item["user_id"],
                item["assistant_id"],
                item.get("source_kind") or "email",
                item.get("account_key"),
                item.get("external_id"),
                item.get("external_thread_id"),
                item.get("sender"),
                item.get("sender_domain") or sender_domain_of(item.get("sender")),
                Jsonb(list(item.get("recipients") or [])),
                item.get("subject"),
                item.get("body_text"),
                item.get("received_at"),
                item.get("message_kind"),
                item.get("decision"),
                bool(item.get("needs_owner_action")),
                item.get("reason"),
                item.get("draft"),
                item.get("confidence"),
                Jsonb(item.get("confidence_detail") or {}),
                item.get("state") or STATE_PENDING_OWNER,
                Jsonb(item["owner_decision"])
                if item.get("owner_decision") is not None
                else None,
            ),
        )
        return await self.get_item(item_id)

    async def update_item(self, item_id: str, **fields: Any) -> dict[str, Any] | None:
        """Merge fields into an item; return the updated row."""
        from psycopg.types.json import Jsonb

        if not fields:
            return await self.get_item(item_id)
        assignments = ["updated_at = now()"]
        params: list[Any] = []
        for key, value in fields.items():
            if key not in _ITEM_NAMES or key in ("item_id", "created_at", "updated_at"):
                continue
            assignments.append(f"{key} = %s")
            params.append(
                Jsonb(value)
                if key in ("recipients", "confidence_detail", "owner_decision")
                and value is not None
                else value
            )
        params.append(item_id)
        await self._execute(
            f"UPDATE inbox_items SET {', '.join(assignments)} WHERE item_id = %s;",
            tuple(params),
        )
        return await self.get_item(item_id)

    async def get_item(self, item_id: str) -> dict[str, Any] | None:
        """Return one item."""
        row = await self._fetchone(
            f"SELECT {_ITEM_COLUMNS} FROM inbox_items WHERE item_id = %s;", (item_id,)
        )
        return self._item_row(row) if row else None

    async def list_items(
        self,
        *,
        assistant_id: str,
        states: tuple[str, ...] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return an avatar's items, newest first."""
        if states:
            rows = await self._fetchall(
                f"SELECT {_ITEM_COLUMNS} FROM inbox_items WHERE assistant_id = %s AND "
                "state = ANY(%s) ORDER BY created_at DESC LIMIT %s;",
                (assistant_id, list(states), int(limit)),
            )
        else:
            rows = await self._fetchall(
                f"SELECT {_ITEM_COLUMNS} FROM inbox_items WHERE assistant_id = %s "
                "ORDER BY created_at DESC LIMIT %s;",
                (assistant_id, int(limit)),
            )
        return [self._item_row(row) for row in rows]

    async def count_open(self, assistant_id: str) -> int:
        """How many items await the owner."""
        row = await self._fetchone(
            "SELECT COUNT(*) FROM inbox_items WHERE assistant_id = %s AND state = ANY(%s);",
            (assistant_id, list(OPEN_STATES)),
        )
        return int(row[0] if row else 0)

    async def recall_preferences(
        self,
        *,
        assistant_id: str,
        sender: str | None,
        sender_domain: str | None,
        message_kind: str | None,
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        """Preferences for this sender, then its domain, then the message kind."""
        rows = await self._fetchall(
            f"""
            SELECT {_PREFERENCE_COLUMNS},
                   CASE WHEN sender = %s THEN 0
                        WHEN sender_domain = %s THEN 1
                        WHEN message_kind = %s THEN 2
                        ELSE 9 END AS rank
            FROM inbox_preferences
            WHERE assistant_id = %s AND (sender = %s OR sender_domain = %s OR message_kind = %s)
            ORDER BY rank ASC, count DESC
            LIMIT %s;
            """,
            (
                sender,
                sender_domain,
                message_kind,
                assistant_id,
                sender,
                sender_domain,
                message_kind,
                int(limit),
            ),
        )
        preferences = []
        for row in rows:
            record = dict(zip(_PREFERENCE_NAMES, row[: len(_PREFERENCE_NAMES)]))
            record["preference_id"] = str(record["preference_id"])
            record["last_decided_at"] = _isoformat(record.get("last_decided_at"))
            preferences.append(record)
        return preferences

    async def record_preference(
        self,
        *,
        user_id: str,
        assistant_id: str,
        sender: str | None,
        sender_domain: str | None,
        message_kind: str | None,
        decision: str,
        edit_summary: str | None = None,
        example_subject: str | None = None,
    ) -> dict[str, Any]:
        """Upsert one preference row (counted)."""
        await self._execute(
            """
            INSERT INTO inbox_preferences
                (preference_id, user_id, assistant_id, sender, sender_domain, message_kind,
                 decision, edit_summary, example_subject, count, last_decided_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 1, now())
            ON CONFLICT (assistant_id, sender, message_kind, decision) DO UPDATE SET
                count = inbox_preferences.count + 1,
                edit_summary = COALESCE(EXCLUDED.edit_summary, inbox_preferences.edit_summary),
                example_subject = COALESCE(EXCLUDED.example_subject, inbox_preferences.example_subject),
                last_decided_at = now();
            """,
            (
                str(uuid4()),
                user_id,
                assistant_id,
                sender or "",
                sender_domain or "",
                message_kind or "",
                decision,
                edit_summary,
                example_subject,
            ),
        )
        rows = await self.recall_preferences(
            assistant_id=assistant_id,
            sender=sender or "",
            sender_domain=sender_domain,
            message_kind=message_kind,
            limit=1,
        )
        return rows[0] if rows else {}

    async def get_poll_state(self, account_key: str) -> dict[str, Any] | None:
        """Return the poll cursor for a mailbox."""
        row = await self._fetchone(
            "SELECT account_key, user_id, assistant_id, last_seen_uid, last_polled_at, last_error "
            "FROM inbox_poll_state WHERE account_key = %s;",
            (account_key,),
        )
        if not row:
            return None
        record = dict(
            zip(
                [
                    "account_key",
                    "user_id",
                    "assistant_id",
                    "last_seen_uid",
                    "last_polled_at",
                    "last_error",
                ],
                row,
            )
        )
        record["last_polled_at"] = _isoformat(record.get("last_polled_at"))
        return record

    async def set_poll_state(
        self,
        *,
        account_key: str,
        user_id: str,
        assistant_id: str,
        last_seen_uid: int | None,
        last_error: str | None = None,
    ) -> None:
        """Record the poll cursor for a mailbox."""
        await self._execute(
            """
            INSERT INTO inbox_poll_state (account_key, user_id, assistant_id, last_seen_uid, last_polled_at, last_error)
            VALUES (%s, %s, %s, %s, now(), %s)
            ON CONFLICT (account_key) DO UPDATE SET
                user_id = EXCLUDED.user_id,
                assistant_id = EXCLUDED.assistant_id,
                last_seen_uid = COALESCE(EXCLUDED.last_seen_uid, inbox_poll_state.last_seen_uid),
                last_polled_at = now(),
                last_error = EXCLUDED.last_error;
            """,
            (account_key, user_id, assistant_id, last_seen_uid, last_error),
        )


_repository: Any | None = None


def set_inbox_repository(repository: Any | None) -> None:
    """Publish the process-wide repository (or clear it with ``None``)."""
    global _repository
    _repository = repository


def get_inbox_repository() -> Any | None:
    """Return the published repository, or ``None``."""
    return _repository


async def ensure_inbox_tables(pool: Any) -> None:
    """Create the inbox tables if they do not exist. Best-effort at boot."""
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(_CREATE_TABLES_SQL)
    except Exception as table_error:  # noqa: BLE001 - non-fatal at startup
        logger.error("Could not ensure the inbox tables exist: %s", table_error)
