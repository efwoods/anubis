"""Durable storage for connected-account records: one Postgres table per deployment.

Connected accounts began life in the LangGraph cross-thread store under the
namespace ``(user_id, "connected_account")``. They move to a dedicated table for
one reason the store cannot satisfy: the owner intends to remove Auth0, and the
records that survive that removal must be keyed by the platform's own notion of
a user and a personal avatar, in a table the platform owns and can migrate,
rather than inside a JSON blob addressed by an identity provider's namespace
convention. The store also indexes every value it holds through the vector
index configuration, which is wasted work for a credential record.

The table keeps the whole record as ``JSONB`` beside a handful of indexed
columns. That keeps ``build_account_record`` / ``public_account_view`` (the two
functions every caller already uses) as the canonical record shape, so a new
provider that adds a transport field needs no schema change — the same reason
the Model Context Protocol connection records were kept as documents.

Two implementations share one interface:

- :class:`PostgresConnectedAccountRepository` — production, over the psycopg
  pool the FastAPI lifespan owns.
- :class:`InMemoryConnectedAccountRepository` — tests and ``langgraph dev``.

The active repository is published process-wide with :func:`set_repository`,
following ``runtime_handles.set_deep_agent_checkpointer``: the FastAPI lifespan
owns the pool and publishes the repository; graph nodes and tools read it back
without importing the web application (which would be a circular import).
When nothing has been published, the ``store.py`` facade falls back to the
legacy store namespace so existing installs keep working until the lifespan
runs the one-time migration.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Protocol

from src.anubis.utils.postgres_ddl import execute_ddl_script

logger = logging.getLogger(__name__)

CONNECTED_ACCOUNTS_TABLE_NAME = "connected_accounts"

_CREATE_CONNECTED_ACCOUNTS_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {CONNECTED_ACCOUNTS_TABLE_NAME} (
    connection_key TEXT NOT NULL,
    user_id TEXT NOT NULL,
    personal_avatar_id TEXT,
    provider TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    record JSONB NOT NULL,
    connected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, connection_key)
);
CREATE INDEX IF NOT EXISTS connected_accounts_user_avatar_idx
    ON {CONNECTED_ACCOUNTS_TABLE_NAME} (user_id, personal_avatar_id);
"""

_UPSERT_CONNECTED_ACCOUNT_SQL = f"""
INSERT INTO {CONNECTED_ACCOUNTS_TABLE_NAME}
    (connection_key, user_id, personal_avatar_id, provider, kind, status, record,
     connected_at, updated_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
ON CONFLICT (user_id, connection_key) DO UPDATE SET
    personal_avatar_id = EXCLUDED.personal_avatar_id,
    provider = EXCLUDED.provider,
    kind = EXCLUDED.kind,
    status = EXCLUDED.status,
    record = EXCLUDED.record,
    updated_at = now();
"""

_INSERT_IF_ABSENT_SQL = f"""
INSERT INTO {CONNECTED_ACCOUNTS_TABLE_NAME}
    (connection_key, user_id, personal_avatar_id, provider, kind, status, record,
     connected_at, updated_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
ON CONFLICT (user_id, connection_key) DO NOTHING;
"""

_SELECT_FOR_USER_SQL = f"""
SELECT record FROM {CONNECTED_ACCOUNTS_TABLE_NAME}
WHERE user_id = %s
ORDER BY connected_at ASC;
"""

_SELECT_BY_KIND_SQL = f"""
SELECT record, user_id FROM {CONNECTED_ACCOUNTS_TABLE_NAME}
WHERE kind = %s AND status = %s
ORDER BY connected_at ASC;
"""

_SELECT_ONE_SQL = f"""
SELECT record FROM {CONNECTED_ACCOUNTS_TABLE_NAME}
WHERE user_id = %s AND connection_key = %s;
"""

_DELETE_ONE_SQL = f"""
DELETE FROM {CONNECTED_ACCOUNTS_TABLE_NAME}
WHERE user_id = %s AND connection_key = %s;
"""

_DELETE_FOR_AVATAR_SQL = f"""
DELETE FROM {CONNECTED_ACCOUNTS_TABLE_NAME}
WHERE personal_avatar_id = %s;
"""

_DELETE_FOR_USER_SQL = f"""
DELETE FROM {CONNECTED_ACCOUNTS_TABLE_NAME}
WHERE user_id = %s;
"""

# The legacy store rows to migrate. The LangGraph Postgres store joins a
# namespace with "." into the ``prefix`` column, so the connected-account
# namespace ``(user_id, "connected_account")`` is the prefix
# ``"{user_id}.connected_account"``.
_LEGACY_NAMESPACE_SUFFIX = ".connected_account"
_SELECT_LEGACY_STORE_ROWS_SQL = """
SELECT prefix, key, value FROM store
WHERE prefix LIKE %s;
"""


class ConnectedAccountRepository(Protocol):
    """The operations the facade and the endpoints need from durable storage."""

    async def list_for_user(self, user_id: str) -> list[dict[str, Any]]:
        """Return every record a user has connected."""
        ...

    async def get(self, user_id: str, connection_key: str) -> dict[str, Any] | None:
        """Return one record by key, or ``None``."""
        ...

    async def upsert(self, user_id: str, record: dict[str, Any]) -> None:
        """Insert or replace one record, keyed by its account key."""
        ...

    async def delete(self, user_id: str, connection_key: str) -> bool:
        """Delete one record; report whether anything was removed."""
        ...


def _record_key(record: dict[str, Any]) -> str:
    key = str(record.get("account_key") or "").strip()
    if not key:
        raise ValueError(
            "Cannot save a connected account without an account key; the record "
            "key is the account key."
        )
    return key


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return datetime.now(UTC)


class InMemoryConnectedAccountRepository:
    """Dictionary-backed repository for tests and the local dev server."""

    def __init__(self) -> None:
        """Start empty; ``records`` maps user id to key to record."""
        self.records: dict[str, dict[str, dict[str, Any]]] = {}

    async def list_for_user(self, user_id: str) -> list[dict[str, Any]]:
        """Return every record a user has connected."""
        return [dict(record) for record in self.records.get(user_id, {}).values()]

    async def get(self, user_id: str, connection_key: str) -> dict[str, Any] | None:
        """Return one record by key, or ``None``."""
        record = self.records.get(user_id, {}).get(connection_key)
        return dict(record) if record is not None else None

    async def upsert(self, user_id: str, record: dict[str, Any]) -> None:
        """Insert or replace one record, keyed by its account key."""
        key = _record_key(record)
        self.records.setdefault(user_id, {})[key] = dict(record)

    async def delete(self, user_id: str, connection_key: str) -> bool:
        """Delete one record; report whether anything was removed."""
        return self.records.get(user_id, {}).pop(connection_key, None) is not None

    async def list_by_kind(
        self, kind: str, status: str = "connected"
    ) -> list[dict[str, Any]]:
        """Return every user's records of one kind in one status (for pollers)."""
        return [
            {**record, "user_id": record.get("user_id") or owner}
            for owner, records in self.records.items()
            for record in records.values()
            if record.get("kind") == kind and record.get("status") == status
        ]


class PostgresConnectedAccountRepository:
    """Repository over the application's psycopg connection pool."""

    def __init__(self, pool: Any) -> None:
        """Bind to the application's ``AsyncConnectionPool``."""
        self._pool = pool

    @staticmethod
    def _columns(record: dict[str, Any]) -> tuple[Any, ...]:
        from psycopg.types.json import Jsonb

        return (
            _record_key(record),
            record.get("assistant_id"),
            str(record.get("provider") or ""),
            str(record.get("kind") or ""),
            str(record.get("status") or ""),
            Jsonb(record),
            _parse_timestamp(record.get("connected_at")),
        )

    async def list_for_user(self, user_id: str) -> list[dict[str, Any]]:
        """Return every record a user has connected, oldest first."""
        async with self._pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(_SELECT_FOR_USER_SQL, (user_id,))
                rows = await cursor.fetchall()
        return [row[0] for row in rows if isinstance(row[0], dict)]

    async def get(self, user_id: str, connection_key: str) -> dict[str, Any] | None:
        """Return one record by key, or ``None``."""
        async with self._pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(_SELECT_ONE_SQL, (user_id, connection_key))
                row = await cursor.fetchone()
        if row is None or not isinstance(row[0], dict):
            return None
        return row[0]

    async def upsert(self, user_id: str, record: dict[str, Any]) -> None:
        """Insert or replace one record, keyed by its account key."""
        key, avatar_id, provider, kind, status, payload, connected_at = self._columns(
            record
        )
        async with self._pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    _UPSERT_CONNECTED_ACCOUNT_SQL,
                    (
                        key,
                        user_id,
                        avatar_id,
                        provider,
                        kind,
                        status,
                        payload,
                        connected_at,
                    ),
                )

    async def delete(self, user_id: str, connection_key: str) -> bool:
        """Delete one record; report whether anything was removed."""
        async with self._pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(_DELETE_ONE_SQL, (user_id, connection_key))
                return bool(cursor.rowcount and cursor.rowcount > 0)

    async def list_by_kind(
        self, kind: str, status: str = "connected"
    ) -> list[dict[str, Any]]:
        """Return every user's records of one kind in one status (for pollers)."""
        async with self._pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(_SELECT_BY_KIND_SQL, (kind, status))
                rows = await cursor.fetchall()
        records = []
        for row in rows:
            if isinstance(row[0], dict):
                record = dict(row[0])
                # The owner is a table column, not a record field; the poller
                # needs it to run the triage as that owner.
                record.setdefault("user_id", row[1])
                records.append(record)
        return records

    async def delete_for_avatar(self, assistant_id: str) -> int:
        """Remove every account bound to one avatar (avatar deletion)."""
        async with self._pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(_DELETE_FOR_AVATAR_SQL, (assistant_id,))
                return int(cursor.rowcount or 0)

    async def delete_for_user(self, user_id: str) -> int:
        """Remove every account a user connected (account deletion)."""
        async with self._pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(_DELETE_FOR_USER_SQL, (user_id,))
                return int(cursor.rowcount or 0)


_repository: ConnectedAccountRepository | None = None


def set_repository(repository: ConnectedAccountRepository | None) -> None:
    """Publish the process-wide repository (or clear it with ``None``)."""
    global _repository
    _repository = repository


def get_repository() -> ConnectedAccountRepository | None:
    """Return the published repository, or ``None`` when nothing was published."""
    return _repository


async def ensure_connected_accounts_table(pool: Any) -> None:
    """Create the ``connected_accounts`` table if it does not yet exist.

    Called once from the FastAPI lifespan startup, beside
    ``ensure_api_metrics_table``. Best-effort: a failure here must not prevent
    the app from serving, so it logs and returns rather than raising — the
    facade then falls back to the legacy store namespace for this process.
    """
    try:
        await execute_ddl_script(pool, _CREATE_CONNECTED_ACCOUNTS_TABLE_SQL)
    except Exception as table_error:  # noqa: BLE001 - non-fatal at startup
        logger.error(
            "Could not ensure %s table exists: %s",
            CONNECTED_ACCOUNTS_TABLE_NAME,
            table_error,
        )


async def migrate_store_connected_accounts_to_table(pool: Any) -> int:
    """Copy legacy store records into the table, once, without overwriting.

    Reads every ``(user_id, "connected_account")`` namespace straight from the
    LangGraph store table and inserts each record if the table does not already
    hold that key. ``ON CONFLICT DO NOTHING`` is what makes this safe to run on
    every boot: a record the owner has since rotated or disconnected in the
    table is never clobbered by a stale store copy. The store rows are left in
    place; the read path prefers the table from the moment the repository is
    published, so they are inert.

    Returns the number of rows inserted. Best-effort: a failure is logged and
    the boot continues, because the table is already authoritative for anything
    connected after this code shipped.
    """
    inserted = 0
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    _SELECT_LEGACY_STORE_ROWS_SQL, (f"%{_LEGACY_NAMESPACE_SUFFIX}",)
                )
                rows = await cursor.fetchall()
                for prefix, _key, value in rows:
                    if not isinstance(value, dict) or not str(prefix).endswith(
                        _LEGACY_NAMESPACE_SUFFIX
                    ):
                        continue
                    user_id = str(prefix)[: -len(_LEGACY_NAMESPACE_SUFFIX)]
                    try:
                        columns = PostgresConnectedAccountRepository._columns(value)
                    except ValueError:
                        continue
                    key, avatar_id, provider, kind, status, payload, connected_at = (
                        columns
                    )
                    await cursor.execute(
                        _INSERT_IF_ABSENT_SQL,
                        (
                            key,
                            user_id,
                            avatar_id,
                            provider,
                            kind,
                            status,
                            payload,
                            connected_at,
                        ),
                    )
                    inserted += int(cursor.rowcount or 0)
    except Exception as migration_error:  # noqa: BLE001 - non-fatal at startup
        logger.error(
            "Could not migrate legacy connected-account store records: %s",
            migration_error,
        )
        return inserted
    if inserted:
        logger.info(
            "Migrated %d connected-account record(s) from the store to %s",
            inserted,
            CONNECTED_ACCOUNTS_TABLE_NAME,
        )
    return inserted
