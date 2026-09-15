"""Persist phone calls and a durable place-lookup cache.

The in-memory twin is what unit tests use. The FastAPI lifespan publishes the
Postgres repository the same way the inbox does.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

PHONE_CALL_STATES = (
    "requested",
    "confirming",
    "ringing_owner",
    "listening",
    "dialing_destination",
    "in_progress",
    "ended",
    "failed",
    "refused",
)

CREATE_PHONE_CALLS_SQL = """
CREATE TABLE IF NOT EXISTS phone_calls (
    call_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    assistant_id TEXT NOT NULL,
    thread_id TEXT,
    direction TEXT NOT NULL,
    state TEXT NOT NULL,
    owner_mobile_e164 TEXT,
    destination_e164 TEXT,
    destination_name TEXT,
    room_name TEXT,
    livekit_room_sid TEXT,
    brief JSONB NOT NULL DEFAULT '{}'::jsonb,
    transcript TEXT,
    result JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

CREATE_PHONE_CALLS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS phone_calls_user_assistant_idx
    ON phone_calls (user_id, assistant_id, created_at DESC);
"""

CREATE_PLACE_LOOKUP_CACHE_SQL = """
CREATE TABLE IF NOT EXISTS place_lookup_cache (
    cache_key TEXT PRIMARY KEY,
    place_name TEXT NOT NULL,
    city TEXT NOT NULL,
    payload JSONB NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_PHONE_CALL_NAMES = (
    "call_id",
    "user_id",
    "assistant_id",
    "thread_id",
    "direction",
    "state",
    "owner_mobile_e164",
    "destination_e164",
    "destination_name",
    "room_name",
    "livekit_room_sid",
    "brief",
    "transcript",
    "result",
    "created_at",
    "updated_at",
)

_PHONE_CALL_COLUMNS = ", ".join(_PHONE_CALL_NAMES)

_repository: "PhoneCallRepository | None" = None


def get_phone_call_repository() -> "PhoneCallRepository | None":
    """Return the process-wide phone-call repository, or None."""
    return _repository


def set_phone_call_repository(repository: "PhoneCallRepository | None") -> None:
    """Publish the process-wide phone-call repository."""
    global _repository
    _repository = repository


async def ensure_phone_tables(pool: Any) -> None:
    """Create the phone_calls and place_lookup_cache tables if missing."""
    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(CREATE_PHONE_CALLS_SQL)
            await cursor.execute(CREATE_PHONE_CALLS_INDEX_SQL)
            await cursor.execute(CREATE_PLACE_LOOKUP_CACHE_SQL)


def _isoformat(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat()
    return str(value)


def _now() -> datetime:
    return datetime.now(UTC)


class PhoneCallRepository:
    """The operations every phone-call store must implement."""

    async def create_call(self, record: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    async def get_call(self, call_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    async def update_call(self, call_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        raise NotImplementedError

    async def list_for_assistant(
        self, user_id: str, assistant_id: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def read_place_cache(self, cache_key: str) -> dict[str, Any] | None:
        raise NotImplementedError

    async def write_place_cache(
        self, cache_key: str, place_name: str, city: str, payload: dict[str, Any], ttl_seconds: int
    ) -> None:
        raise NotImplementedError


class InMemoryPhoneCallRepository(PhoneCallRepository):
    """Process-local store for tests."""

    def __init__(self) -> None:
        self.calls: dict[str, dict[str, Any]] = {}
        self.places: dict[str, dict[str, Any]] = {}

    async def create_call(self, record: dict[str, Any]) -> dict[str, Any]:
        call_id = str(record.get("call_id") or uuid4())
        now = _now().isoformat()
        stored = {
            "call_id": call_id,
            "user_id": record["user_id"],
            "assistant_id": record["assistant_id"],
            "thread_id": record.get("thread_id"),
            "direction": record.get("direction") or "outbound",
            "state": record.get("state") or "requested",
            "owner_mobile_e164": record.get("owner_mobile_e164"),
            "destination_e164": record.get("destination_e164"),
            "destination_name": record.get("destination_name"),
            "room_name": record.get("room_name") or f"phone-{call_id}",
            "livekit_room_sid": record.get("livekit_room_sid"),
            "brief": dict(record.get("brief") or {}),
            "transcript": record.get("transcript"),
            "result": dict(record.get("result") or {}),
            "created_at": now,
            "updated_at": now,
        }
        self.calls[call_id] = stored
        return dict(stored)

    async def get_call(self, call_id: str) -> dict[str, Any] | None:
        stored = self.calls.get(str(call_id))
        return dict(stored) if stored else None

    async def update_call(self, call_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        stored = self.calls.get(str(call_id))
        if stored is None:
            return None
        allowed = {
            "state",
            "owner_mobile_e164",
            "destination_e164",
            "destination_name",
            "room_name",
            "livekit_room_sid",
            "brief",
            "transcript",
            "result",
        }
        for key, value in updates.items():
            if key in allowed:
                stored[key] = value
        stored["updated_at"] = _now().isoformat()
        return dict(stored)

    async def list_for_assistant(
        self, user_id: str, assistant_id: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        matching = [
            dict(record)
            for record in self.calls.values()
            if record["user_id"] == user_id and record["assistant_id"] == assistant_id
        ]
        matching.sort(key=lambda record: record["created_at"], reverse=True)
        return matching[:limit]

    async def read_place_cache(self, cache_key: str) -> dict[str, Any] | None:
        entry = self.places.get(cache_key)
        if entry is None:
            return None
        expires_at = datetime.fromisoformat(entry["expires_at"])
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if _now() > expires_at:
            self.places.pop(cache_key, None)
            return None
        return dict(entry["payload"])

    async def write_place_cache(
        self, cache_key: str, place_name: str, city: str, payload: dict[str, Any], ttl_seconds: int
    ) -> None:
        self.places[cache_key] = {
            "place_name": place_name,
            "city": city,
            "payload": dict(payload),
            "expires_at": (_now() + timedelta(seconds=max(1, int(ttl_seconds)))).isoformat(),
        }


class PostgresPhoneCallRepository(PhoneCallRepository):
    """Postgres-backed phone-call store."""

    def __init__(self, pool: Any) -> None:
        self.pool = pool

    async def _fetchone(self, sql: str, params: tuple = ()) -> tuple | None:
        async with self.pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(sql, params)
                return await cursor.fetchone()

    async def _fetchall(self, sql: str, params: tuple = ()) -> list[tuple]:
        async with self.pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(sql, params)
                return list(await cursor.fetchall())

    async def _execute(self, sql: str, params: tuple = ()) -> int:
        async with self.pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(sql, params)
                return int(cursor.rowcount or 0)

    @staticmethod
    def _row(row: tuple) -> dict[str, Any]:
        record = dict(zip(_PHONE_CALL_NAMES, row))
        record["created_at"] = _isoformat(record.get("created_at"))
        record["updated_at"] = _isoformat(record.get("updated_at"))
        record["brief"] = dict(record.get("brief") or {})
        record["result"] = dict(record.get("result") or {})
        return record

    async def create_call(self, record: dict[str, Any]) -> dict[str, Any]:
        from psycopg.types.json import Jsonb

        call_id = str(record.get("call_id") or uuid4())
        await self._execute(
            """
            INSERT INTO phone_calls
                (call_id, user_id, assistant_id, thread_id, direction, state,
                 owner_mobile_e164, destination_e164, destination_name, room_name,
                 livekit_room_sid, brief, transcript, result)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """,
            (
                call_id,
                record["user_id"],
                record["assistant_id"],
                record.get("thread_id"),
                record.get("direction") or "outbound",
                record.get("state") or "requested",
                record.get("owner_mobile_e164"),
                record.get("destination_e164"),
                record.get("destination_name"),
                record.get("room_name") or f"phone-{call_id}",
                record.get("livekit_room_sid"),
                Jsonb(dict(record.get("brief") or {})),
                record.get("transcript"),
                Jsonb(dict(record.get("result") or {})),
            ),
        )
        stored = await self.get_call(call_id)
        if stored is None:
            raise RuntimeError(f"phone call {call_id} was inserted but could not be read back")
        return stored

    async def get_call(self, call_id: str) -> dict[str, Any] | None:
        row = await self._fetchone(
            f"SELECT {_PHONE_CALL_COLUMNS} FROM phone_calls WHERE call_id = %s;",
            (str(call_id),),
        )
        return self._row(row) if row else None

    async def update_call(self, call_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        from psycopg.types.json import Jsonb

        assignments: list[str] = []
        params: list[Any] = []
        mapping = {
            "state": "state",
            "owner_mobile_e164": "owner_mobile_e164",
            "destination_e164": "destination_e164",
            "destination_name": "destination_name",
            "room_name": "room_name",
            "livekit_room_sid": "livekit_room_sid",
            "brief": "brief",
            "transcript": "transcript",
            "result": "result",
        }
        for key, column in mapping.items():
            if key not in updates:
                continue
            value = updates[key]
            if key in {"brief", "result"}:
                value = Jsonb(dict(value or {}))
            assignments.append(f"{column} = %s")
            params.append(value)
        if not assignments:
            return await self.get_call(call_id)
        assignments.append("updated_at = NOW()")
        params.append(str(call_id))
        await self._execute(
            f"UPDATE phone_calls SET {', '.join(assignments)} WHERE call_id = %s;",
            tuple(params),
        )
        return await self.get_call(call_id)

    async def list_for_assistant(
        self, user_id: str, assistant_id: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        rows = await self._fetchall(
            f"SELECT {_PHONE_CALL_COLUMNS} FROM phone_calls "
            "WHERE user_id = %s AND assistant_id = %s "
            "ORDER BY created_at DESC LIMIT %s;",
            (user_id, assistant_id, int(limit)),
        )
        return [self._row(row) for row in rows]

    async def read_place_cache(self, cache_key: str) -> dict[str, Any] | None:
        row = await self._fetchone(
            "SELECT payload, expires_at FROM place_lookup_cache WHERE cache_key = %s;",
            (cache_key,),
        )
        if row is None:
            return None
        payload, expires_at = row
        if isinstance(expires_at, datetime):
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            if _now() > expires_at:
                await self._execute(
                    "DELETE FROM place_lookup_cache WHERE cache_key = %s;",
                    (cache_key,),
                )
                return None
        return dict(payload or {})

    async def write_place_cache(
        self, cache_key: str, place_name: str, city: str, payload: dict[str, Any], ttl_seconds: int
    ) -> None:
        from psycopg.types.json import Jsonb

        expires_at = _now() + timedelta(seconds=max(1, int(ttl_seconds)))
        await self._execute(
            """
            INSERT INTO place_lookup_cache (cache_key, place_name, city, payload, expires_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (cache_key) DO UPDATE SET
                place_name = EXCLUDED.place_name,
                city = EXCLUDED.city,
                payload = EXCLUDED.payload,
                expires_at = EXCLUDED.expires_at;
            """,
            (cache_key, place_name, city, Jsonb(dict(payload)), expires_at),
        )
