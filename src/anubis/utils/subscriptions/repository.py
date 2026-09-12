"""Storage for content subscriptions and the events they deliver.

``content_subscriptions``
    One row per thing the avatar is subscribed to — a YouTube channel, a Twitch
    broadcaster, a podcast feed, or a connected mailbox. The row records which
    transport carries the announcement, because the transport is a property of
    the platform rather than of the subscription: YouTube pushes to our
    callback, LinkedIn emails the owner, and both end up as rows here that the
    rest of the system reads identically.

``content_events``
    One row per announcement received, whatever brought it. This table is what
    makes the pipeline safe to run against retrying platforms: every push
    transport in use redelivers, WebSub redelivers aggressively, and the same
    new video legitimately arrives both as a webhook and as an email
    notification. ``external_item_id`` is unique per subscription, so a
    redelivery is recognised and dropped rather than ingested a second time at
    full transcription cost.

The shape follows ``inbox/repository.py`` deliberately — Postgres class, an
in-memory twin for tests and ``langgraph dev``, and a module-level published
instance — so a reader who knows one knows the other.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from src.anubis.utils.postgres_ddl import execute_ddl_script

logger = logging.getLogger(__name__)

# Subscription states.
SUBSCRIPTION_PENDING = "pending"  # asked the platform, awaiting its handshake
SUBSCRIPTION_ACTIVE = "active"
SUBSCRIPTION_EXPIRED = "expired"  # a lease ran out before it could be renewed
SUBSCRIPTION_FAILED = "failed"
SUBSCRIPTION_DISABLED = "disabled"  # the owner turned it off

# Event states.
EVENT_RECEIVED = "received"
EVENT_INGESTING = "ingesting"
EVENT_INGESTED = "ingested"
EVENT_REFUSED = "refused"  # not owned, not relevant, or over budget
EVENT_FAILED = "failed"

_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS content_subscriptions (
    subscription_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    personal_avatar_id TEXT NOT NULL,
    connection_key TEXT NOT NULL,
    provider TEXT NOT NULL,
    transport TEXT NOT NULL,
    topic TEXT,
    topic_url TEXT,
    avatar_name TEXT,
    avatar_description TEXT,
    callback_url TEXT,
    external_id TEXT,
    secret TEXT,
    status TEXT NOT NULL,
    detail TEXT,
    expires_at TIMESTAMPTZ,
    last_event_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, subscription_id)
);
CREATE INDEX IF NOT EXISTS content_subscriptions_avatar_idx
    ON content_subscriptions (personal_avatar_id);
CREATE INDEX IF NOT EXISTS content_subscriptions_connection_idx
    ON content_subscriptions (connection_key);
CREATE INDEX IF NOT EXISTS content_subscriptions_renewal_idx
    ON content_subscriptions (status, expires_at);

CREATE TABLE IF NOT EXISTS content_events (
    event_id TEXT PRIMARY KEY,
    subscription_id TEXT,
    user_id TEXT NOT NULL,
    personal_avatar_id TEXT NOT NULL,
    connection_key TEXT,
    provider TEXT NOT NULL,
    transport TEXT NOT NULL,
    external_item_id TEXT NOT NULL,
    url TEXT,
    title TEXT,
    published_at TIMESTAMPTZ,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    state TEXT NOT NULL,
    detail TEXT,
    media_job_id TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS content_events_dedupe_idx
    ON content_events (personal_avatar_id, provider, external_item_id);
CREATE INDEX IF NOT EXISTS content_events_avatar_idx
    ON content_events (personal_avatar_id, received_at DESC);
"""

_SUBSCRIPTION_COLUMNS = (
    "subscription_id, user_id, personal_avatar_id, connection_key, provider, "
    "transport, topic, topic_url, avatar_name, avatar_description, callback_url, "
    "external_id, secret, status, detail, expires_at, last_event_at, created_at, "
    "updated_at"
)
_SUBSCRIPTION_NAMES = [name.strip() for name in _SUBSCRIPTION_COLUMNS.split(",")]

_EVENT_COLUMNS = (
    "event_id, subscription_id, user_id, personal_avatar_id, connection_key, "
    "provider, transport, external_item_id, url, title, published_at, "
    "received_at, state, detail, media_job_id"
)
_EVENT_NAMES = [name.strip() for name in _EVENT_COLUMNS.split(",")]

_TIMESTAMP_FIELDS = ("expires_at", "last_event_at", "created_at", "updated_at")
_EVENT_TIMESTAMP_FIELDS = ("published_at", "received_at")


def _now() -> datetime:
    return datetime.now(UTC)


def _isoformat(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


class InMemorySubscriptionRepository:
    """Dictionary-backed twin for tests and the local dev server."""

    def __init__(self) -> None:
        """Start empty."""
        self.subscriptions: dict[str, dict[str, Any]] = {}
        self.events: dict[str, dict[str, Any]] = {}
        self.pool = None

    # -- subscriptions -------------------------------------------------------

    async def upsert_subscription(self, subscription: dict[str, Any]) -> dict[str, Any]:
        """Insert or replace one subscription; return the stored row."""
        subscription_id = str(subscription.get("subscription_id") or uuid4())
        now = _now().isoformat()
        existing = self.subscriptions.get(subscription_id) or {}
        stored = {
            **existing,
            **{key: value for key, value in subscription.items() if value is not None},
            "subscription_id": subscription_id,
            "created_at": existing.get("created_at") or now,
            "updated_at": now,
        }
        self.subscriptions[subscription_id] = stored
        return dict(stored)

    async def get_subscription(self, subscription_id: str) -> dict[str, Any] | None:
        """Return one subscription by id."""
        found = self.subscriptions.get(str(subscription_id))
        return dict(found) if found else None

    async def find_subscription(
        self, *, connection_key: str, topic: str | None = None
    ) -> dict[str, Any] | None:
        """Return the subscription for a connection (and topic), if any."""
        for subscription in self.subscriptions.values():
            if subscription.get("connection_key") != connection_key:
                continue
            if topic is not None and subscription.get("topic") != topic:
                continue
            return dict(subscription)
        return None

    async def list_for_avatar(self, personal_avatar_id: str) -> list[dict[str, Any]]:
        """Return every subscription belonging to one personal avatar."""
        return [
            dict(subscription)
            for subscription in self.subscriptions.values()
            if subscription.get("personal_avatar_id") == personal_avatar_id
        ]

    async def list_due_for_renewal(
        self, *, before: datetime
    ) -> list[dict[str, Any]]:
        """Return active leases expiring before ``before``."""
        due: list[dict[str, Any]] = []
        for subscription in self.subscriptions.values():
            if subscription.get("status") != SUBSCRIPTION_ACTIVE:
                continue
            expires_at = subscription.get("expires_at")
            if not expires_at:
                continue
            moment = (
                datetime.fromisoformat(expires_at)
                if isinstance(expires_at, str)
                else expires_at
            )
            if moment <= before:
                due.append(dict(subscription))
        return due

    async def find_by_callback(
        self, *, provider: str, topic: str
    ) -> list[dict[str, Any]]:
        """Return subscriptions a delivered callback could belong to."""
        return [
            dict(subscription)
            for subscription in self.subscriptions.values()
            if subscription.get("provider") == provider
            and subscription.get("topic") == topic
        ]

    async def set_subscription_status(
        self,
        subscription_id: str,
        *,
        status: str,
        detail: str | None = None,
        expires_at: datetime | None = None,
        external_id: str | None = None,
    ) -> None:
        """Record the outcome of a handshake, a renewal, or a failure."""
        subscription = self.subscriptions.get(str(subscription_id))
        if not subscription:
            return
        subscription["status"] = status
        subscription["updated_at"] = _now().isoformat()
        if detail is not None:
            subscription["detail"] = detail
        if expires_at is not None:
            subscription["expires_at"] = expires_at.isoformat()
        if external_id is not None:
            subscription["external_id"] = external_id

    async def touch_subscription(self, subscription_id: str) -> None:
        """Stamp that an event arrived, which is how liveness is shown."""
        subscription = self.subscriptions.get(str(subscription_id))
        if subscription:
            subscription["last_event_at"] = _now().isoformat()

    async def delete_subscription(self, subscription_id: str) -> bool:
        """Remove one subscription."""
        return self.subscriptions.pop(str(subscription_id), None) is not None

    async def delete_for_connection(self, connection_key: str) -> int:
        """Remove every subscription of a disconnected account."""
        doomed = [
            key
            for key, subscription in self.subscriptions.items()
            if subscription.get("connection_key") == connection_key
        ]
        for key in doomed:
            self.subscriptions.pop(key, None)
        return len(doomed)

    # -- events --------------------------------------------------------------

    async def find_event(
        self, *, personal_avatar_id: str, provider: str, external_item_id: str
    ) -> dict[str, Any] | None:
        """Return the event already recorded for this announcement, if any."""
        for event in self.events.values():
            if (
                event.get("personal_avatar_id") == personal_avatar_id
                and event.get("provider") == provider
                and event.get("external_item_id") == external_item_id
            ):
                return dict(event)
        return None

    async def create_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Insert an event; return the stored row."""
        event_id = str(event.get("event_id") or uuid4())
        stored = {
            **event,
            "event_id": event_id,
            "received_at": event.get("received_at") or _now().isoformat(),
        }
        self.events[event_id] = stored
        return dict(stored)

    async def set_event_state(
        self,
        event_id: str,
        *,
        state: str,
        detail: str | None = None,
        media_job_id: str | None = None,
    ) -> None:
        """Move an event through received → ingesting → ingested / refused."""
        event = self.events.get(str(event_id))
        if not event:
            return
        event["state"] = state
        if detail is not None:
            event["detail"] = detail
        if media_job_id is not None:
            event["media_job_id"] = media_job_id

    async def list_events_for_avatar(
        self, personal_avatar_id: str, *, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Return the most recent events for one avatar, newest first."""
        rows = [
            dict(event)
            for event in self.events.values()
            if event.get("personal_avatar_id") == personal_avatar_id
        ]
        rows.sort(key=lambda event: str(event.get("received_at") or ""), reverse=True)
        return rows[: int(limit)]


class PostgresSubscriptionRepository:
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
    def _subscription_row(row: tuple) -> dict[str, Any]:
        record = dict(zip(_SUBSCRIPTION_NAMES, row))
        for key in _TIMESTAMP_FIELDS:
            record[key] = _isoformat(record.get(key))
        return record

    @staticmethod
    def _event_row(row: tuple) -> dict[str, Any]:
        record = dict(zip(_EVENT_NAMES, row))
        for key in _EVENT_TIMESTAMP_FIELDS:
            record[key] = _isoformat(record.get(key))
        return record

    async def upsert_subscription(self, subscription: dict[str, Any]) -> dict[str, Any]:
        """Insert or replace one subscription; return the stored row."""
        subscription_id = str(subscription.get("subscription_id") or uuid4())
        await self._execute(
            """
            INSERT INTO content_subscriptions
                (subscription_id, user_id, personal_avatar_id, connection_key,
                 provider, transport, topic, topic_url, avatar_name,
                 avatar_description, callback_url, external_id, secret, status,
                 detail, expires_at, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, now(), now())
            ON CONFLICT (user_id, subscription_id) DO UPDATE SET
                personal_avatar_id = EXCLUDED.personal_avatar_id,
                connection_key = EXCLUDED.connection_key,
                provider = EXCLUDED.provider,
                transport = EXCLUDED.transport,
                topic = EXCLUDED.topic,
                topic_url = EXCLUDED.topic_url,
                avatar_name = EXCLUDED.avatar_name,
                avatar_description = EXCLUDED.avatar_description,
                callback_url = EXCLUDED.callback_url,
                external_id = EXCLUDED.external_id,
                secret = EXCLUDED.secret,
                status = EXCLUDED.status,
                detail = EXCLUDED.detail,
                expires_at = EXCLUDED.expires_at,
                updated_at = now()
            """,
            (
                subscription_id,
                subscription.get("user_id"),
                subscription.get("personal_avatar_id"),
                subscription.get("connection_key"),
                subscription.get("provider"),
                subscription.get("transport"),
                subscription.get("topic"),
                subscription.get("topic_url"),
                subscription.get("avatar_name"),
                subscription.get("avatar_description"),
                subscription.get("callback_url"),
                subscription.get("external_id"),
                subscription.get("secret"),
                subscription.get("status") or SUBSCRIPTION_PENDING,
                subscription.get("detail"),
                subscription.get("expires_at"),
            ),
        )
        stored = await self.get_subscription(subscription_id)
        return stored or dict(subscription, subscription_id=subscription_id)

    async def get_subscription(self, subscription_id: str) -> dict[str, Any] | None:
        """Return one subscription by id."""
        row = await self._fetchone(
            f"SELECT {_SUBSCRIPTION_COLUMNS} FROM content_subscriptions "
            "WHERE subscription_id = %s",
            (str(subscription_id),),
        )
        return self._subscription_row(row) if row else None

    async def find_subscription(
        self, *, connection_key: str, topic: str | None = None
    ) -> dict[str, Any] | None:
        """Return the subscription for a connection (and topic), if any."""
        if topic is None:
            row = await self._fetchone(
                f"SELECT {_SUBSCRIPTION_COLUMNS} FROM content_subscriptions "
                "WHERE connection_key = %s LIMIT 1",
                (connection_key,),
            )
        else:
            row = await self._fetchone(
                f"SELECT {_SUBSCRIPTION_COLUMNS} FROM content_subscriptions "
                "WHERE connection_key = %s AND topic = %s LIMIT 1",
                (connection_key, topic),
            )
        return self._subscription_row(row) if row else None

    async def list_for_avatar(self, personal_avatar_id: str) -> list[dict[str, Any]]:
        """Return every subscription belonging to one personal avatar."""
        rows = await self._fetchall(
            f"SELECT {_SUBSCRIPTION_COLUMNS} FROM content_subscriptions "
            "WHERE personal_avatar_id = %s ORDER BY created_at",
            (personal_avatar_id,),
        )
        return [self._subscription_row(row) for row in rows]

    async def list_due_for_renewal(self, *, before: datetime) -> list[dict[str, Any]]:
        """Return active leases expiring before ``before``."""
        rows = await self._fetchall(
            f"SELECT {_SUBSCRIPTION_COLUMNS} FROM content_subscriptions "
            "WHERE status = %s AND expires_at IS NOT NULL AND expires_at <= %s",
            (SUBSCRIPTION_ACTIVE, before),
        )
        return [self._subscription_row(row) for row in rows]

    async def find_by_callback(
        self, *, provider: str, topic: str
    ) -> list[dict[str, Any]]:
        """Return subscriptions a delivered callback could belong to."""
        rows = await self._fetchall(
            f"SELECT {_SUBSCRIPTION_COLUMNS} FROM content_subscriptions "
            "WHERE provider = %s AND topic = %s",
            (provider, topic),
        )
        return [self._subscription_row(row) for row in rows]

    async def set_subscription_status(
        self,
        subscription_id: str,
        *,
        status: str,
        detail: str | None = None,
        expires_at: datetime | None = None,
        external_id: str | None = None,
    ) -> None:
        """Record the outcome of a handshake, a renewal, or a failure."""
        await self._execute(
            """
            UPDATE content_subscriptions SET
                status = %s,
                detail = COALESCE(%s, detail),
                expires_at = COALESCE(%s, expires_at),
                external_id = COALESCE(%s, external_id),
                updated_at = now()
            WHERE subscription_id = %s
            """,
            (status, detail, expires_at, external_id, str(subscription_id)),
        )

    async def touch_subscription(self, subscription_id: str) -> None:
        """Stamp that an event arrived, which is how liveness is shown."""
        await self._execute(
            "UPDATE content_subscriptions SET last_event_at = now(), "
            "updated_at = now() WHERE subscription_id = %s",
            (str(subscription_id),),
        )

    async def delete_subscription(self, subscription_id: str) -> bool:
        """Remove one subscription."""
        return (
            await self._execute(
                "DELETE FROM content_subscriptions WHERE subscription_id = %s",
                (str(subscription_id),),
            )
            > 0
        )

    async def delete_for_connection(self, connection_key: str) -> int:
        """Remove every subscription of a disconnected account."""
        return await self._execute(
            "DELETE FROM content_subscriptions WHERE connection_key = %s",
            (connection_key,),
        )

    async def find_event(
        self, *, personal_avatar_id: str, provider: str, external_item_id: str
    ) -> dict[str, Any] | None:
        """Return the event already recorded for this announcement, if any."""
        row = await self._fetchone(
            f"SELECT {_EVENT_COLUMNS} FROM content_events "
            "WHERE personal_avatar_id = %s AND provider = %s "
            "AND external_item_id = %s",
            (personal_avatar_id, provider, external_item_id),
        )
        return self._event_row(row) if row else None

    async def create_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Insert an event; return the stored row.

        ``ON CONFLICT DO NOTHING`` makes a redelivery that races another worker
        a no-op rather than an error: two API processes can receive the same
        WebSub push at once, and the unique index is what decides which one
        wins.
        """
        event_id = str(event.get("event_id") or uuid4())
        await self._execute(
            """
            INSERT INTO content_events
                (event_id, subscription_id, user_id, personal_avatar_id,
                 connection_key, provider, transport, external_item_id, url,
                 title, published_at, received_at, state, detail, media_job_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now(), %s, %s, %s)
            ON CONFLICT (personal_avatar_id, provider, external_item_id)
            DO NOTHING
            """,
            (
                event_id,
                event.get("subscription_id"),
                event.get("user_id"),
                event.get("personal_avatar_id"),
                event.get("connection_key"),
                event.get("provider"),
                event.get("transport"),
                event.get("external_item_id"),
                event.get("url"),
                event.get("title"),
                event.get("published_at"),
                event.get("state") or EVENT_RECEIVED,
                event.get("detail"),
                event.get("media_job_id"),
            ),
        )
        stored = await self.find_event(
            personal_avatar_id=str(event.get("personal_avatar_id") or ""),
            provider=str(event.get("provider") or ""),
            external_item_id=str(event.get("external_item_id") or ""),
        )
        return stored or dict(event, event_id=event_id)

    async def set_event_state(
        self,
        event_id: str,
        *,
        state: str,
        detail: str | None = None,
        media_job_id: str | None = None,
    ) -> None:
        """Move an event through received → ingesting → ingested / refused."""
        await self._execute(
            """
            UPDATE content_events SET
                state = %s,
                detail = COALESCE(%s, detail),
                media_job_id = COALESCE(%s, media_job_id)
            WHERE event_id = %s
            """,
            (state, detail, media_job_id, str(event_id)),
        )

    async def list_events_for_avatar(
        self, personal_avatar_id: str, *, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Return the most recent events for one avatar, newest first."""
        rows = await self._fetchall(
            f"SELECT {_EVENT_COLUMNS} FROM content_events "
            "WHERE personal_avatar_id = %s ORDER BY received_at DESC LIMIT %s",
            (personal_avatar_id, int(limit)),
        )
        return [self._event_row(row) for row in rows]


_repository: Any | None = None


def set_subscription_repository(repository: Any | None) -> None:
    """Publish the repository the whole process uses."""
    global _repository
    _repository = repository


def get_subscription_repository() -> Any:
    """Return the published repository, falling back to the in-memory twin.

    The fallback is what lets ``langgraph dev`` and unit tests exercise every
    path without a database, exactly as the inbox repository does.
    """
    global _repository
    if _repository is None:
        _repository = InMemorySubscriptionRepository()
    return _repository


async def ensure_subscription_tables(pool: Any) -> None:
    """Create the subscription tables if they do not exist. Best-effort at boot."""
    try:
        await execute_ddl_script(pool, _CREATE_TABLES_SQL)
    except Exception as table_error:  # noqa: BLE001 - non-fatal at startup
        logger.error(
            "Could not ensure the content subscription tables exist: %s", table_error
        )
