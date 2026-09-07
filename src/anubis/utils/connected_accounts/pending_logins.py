"""Server-side state for a login that is in flight in a popup window.

A popup login is two requests seconds or minutes apart: the start, which the
signed-in owner makes from the app, and the completion, which the vendor's
redirect (or the popup page itself) makes with no session of its own. What
the completion needs that must NOT travel through the browser lives here: the
PKCE verifier, a Model Context Protocol server's freshly registered client
secret, the server URL and name the owner typed. The signed ``state`` token
carries only the nonce that keys this row.

Rows are single-use. ``consume_pending`` marks the row completed in the same
statement that reads it, so two callbacks racing with the same code (a double
click, a replayed redirect) cannot both store an account. Expired rows are
purged on the next start.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

logger = logging.getLogger(__name__)

PENDING_LOGINS_TABLE = "pending_logins"

PENDING_LOGINS_DDL = f"""
CREATE TABLE IF NOT EXISTS {PENDING_LOGINS_TABLE} (
    nonce TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    assistant_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    mode TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    status TEXT NOT NULL DEFAULT 'started',
    result JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS pending_logins_expiry_idx
    ON {PENDING_LOGINS_TABLE} (expires_at);
"""

MODE_OAUTH = "oauth"
MODE_MCP_OAUTH = "mcp_oauth"
MODE_PLAID = "plaid"
MODE_BROWSER = "browser"

STATUS_STARTED = "started"
STATUS_COMPLETED = "completed"


class PendingLoginRepository(Protocol):
    """Where in-flight popup logins are kept."""

    async def create(self, row: dict[str, Any]) -> None:
        """Insert one started login."""

    async def consume(self, nonce: str) -> dict[str, Any] | None:
        """Atomically mark a started, unexpired login completed and return it."""

    async def peek(self, nonce: str) -> dict[str, Any] | None:
        """Return the row without changing it (status pages)."""

    async def purge_expired(self) -> int:
        """Delete expired rows; return how many were removed."""


def _now() -> datetime:
    return datetime.now(UTC)


def build_pending_row(
    *,
    nonce: str,
    user_id: str,
    assistant_id: str,
    provider: str,
    mode: str,
    payload: dict[str, Any] | None,
    max_age_seconds: int,
) -> dict[str, Any]:
    """Assemble the row for one started login."""
    created = _now()
    return {
        "nonce": nonce,
        "user_id": user_id,
        "assistant_id": assistant_id,
        "provider": provider,
        "mode": mode,
        "payload": dict(payload or {}),
        "status": STATUS_STARTED,
        "result": None,
        "created_at": created,
        "expires_at": created + timedelta(seconds=int(max_age_seconds)),
    }


class InMemoryPendingLoginRepository:
    """Dictionary-backed twin of the Postgres repository, for tests and dev."""

    def __init__(self) -> None:
        """Start with no logins in flight."""
        self._rows: dict[str, dict[str, Any]] = {}

    async def create(self, row: dict[str, Any]) -> None:
        """Insert one started login."""
        self._rows[row["nonce"]] = dict(row)

    async def consume(self, nonce: str) -> dict[str, Any] | None:
        """Mark a started, unexpired login completed and return the row."""
        row = self._rows.get(nonce)
        if row is None or row.get("status") != STATUS_STARTED:
            return None
        if row["expires_at"] < _now():
            return None
        row["status"] = STATUS_COMPLETED
        return dict(row)

    async def peek(self, nonce: str) -> dict[str, Any] | None:
        """Return the row without changing the row."""
        row = self._rows.get(nonce)
        return dict(row) if row else None

    async def purge_expired(self) -> int:
        """Delete expired rows; return how many were removed."""
        expired = [key for key, row in self._rows.items() if row["expires_at"] < _now()]
        for key in expired:
            self._rows.pop(key, None)
        return len(expired)


class PostgresPendingLoginRepository:
    """The ``pending_logins`` table behind a psycopg connection pool."""

    def __init__(self, pool: Any) -> None:
        """Bind the repository to the application's connection pool."""
        self._pool = pool

    async def create(self, row: dict[str, Any]) -> None:
        """Insert one started login."""
        async with self._pool.connection() as connection:
            await connection.execute(
                f"INSERT INTO {PENDING_LOGINS_TABLE} "
                "(nonce, user_id, assistant_id, provider, mode, payload, status, "
                "created_at, expires_at) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)",
                (
                    row["nonce"],
                    row["user_id"],
                    row["assistant_id"],
                    row["provider"],
                    row["mode"],
                    json.dumps(row.get("payload") or {}, default=str),
                    row.get("status") or STATUS_STARTED,
                    row["created_at"],
                    row["expires_at"],
                ),
            )

    async def consume(self, nonce: str) -> dict[str, Any] | None:
        """Mark a started, unexpired login completed and return the row (one statement)."""
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                f"UPDATE {PENDING_LOGINS_TABLE} SET status = %s "
                "WHERE nonce = %s AND status = %s AND expires_at > now() "
                "RETURNING nonce, user_id, assistant_id, provider, mode, payload, "
                "status, result, created_at, expires_at",
                (STATUS_COMPLETED, nonce, STATUS_STARTED),
            )
            row = await cursor.fetchone()
        return _row_to_dict(row)

    async def peek(self, nonce: str) -> dict[str, Any] | None:
        """Return the row without changing the row."""
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT nonce, user_id, assistant_id, provider, mode, payload, status, "
                f"result, created_at, expires_at FROM {PENDING_LOGINS_TABLE} "
                "WHERE nonce = %s",
                (nonce,),
            )
            row = await cursor.fetchone()
        return _row_to_dict(row)

    async def purge_expired(self) -> int:
        """Delete expired rows; return how many were removed."""
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                f"DELETE FROM {PENDING_LOGINS_TABLE} WHERE expires_at < now()"
            )
            return int(getattr(cursor, "rowcount", 0) or 0)


def _row_to_dict(row: Any) -> dict[str, Any] | None:
    if row is None:
        return None
    if isinstance(row, dict):
        data = dict(row)
    else:
        keys = (
            "nonce",
            "user_id",
            "assistant_id",
            "provider",
            "mode",
            "payload",
            "status",
            "result",
            "created_at",
            "expires_at",
        )
        data = dict(zip(keys, row))
    payload = data.get("payload")
    if isinstance(payload, str):
        try:
            data["payload"] = json.loads(payload)
        except Exception:
            data["payload"] = {}
    return data


_repository: PendingLoginRepository | None = None


def set_pending_login_repository(repository: PendingLoginRepository | None) -> None:
    """Publish the process-wide repository (the lifespan does this)."""
    global _repository
    _repository = repository


def get_pending_login_repository() -> PendingLoginRepository:
    """Return the published repository, or an in-memory one for this process.

    Falling back keeps ``langgraph dev`` and the unit tests working without a
    database; the in-memory rows are still single-use within the process.
    """
    global _repository
    if _repository is None:
        _repository = InMemoryPendingLoginRepository()
    return _repository


async def ensure_pending_logins_table(pool: Any) -> None:
    """Create the table on boot."""
    from src.anubis.utils.postgres_ddl import execute_ddl_script

    await execute_ddl_script(pool, PENDING_LOGINS_DDL)
