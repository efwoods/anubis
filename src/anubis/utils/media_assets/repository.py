"""Postgres storage for generated avatar media and the voice-clone corpus.

Four tables, created on boot with ``CREATE TABLE IF NOT EXISTS`` like
``api_metrics``:

``avatar_emotion_media``
    One row per (avatar, emotion, asset kind): the seven emotion stills, the
    seven idle loops, and cached lip-sync clips (keyed by emotion plus a digest
    of the spoken text). Bytes are stored inline as ``BYTEA``; a signed vendor
    URL expires within hours, so the bytes are copied here the moment a
    generation completes and the vendor identifiers are kept for traceability.

``avatar_voice_clips``
    The target-only speech clips collected for cloning: what the settings
    recorder captures and what every later audio/video upload of the personal
    avatar contributes after the diarizer has isolated the owner's turns.

``avatar_voice``
    One row per avatar: the instant clone id, the professional clone id and
    its training state, and the running total of collected seconds.

``avatar_media_jobs``
    Durable background work (emotion generation, clone creation, professional
    clone training) that must survive a process restart — the in-process media
    job registry forgets a job after thirty minutes, and a professional clone
    trains for hours.

Two implementations, one interface, the same publish pattern as the
connected-account repository.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

ASSET_KIND_STILL = "still"
ASSET_KIND_IDLE_LOOP = "idle_loop"
ASSET_KIND_LIP_SYNC = "lip_sync"

VOICE_STATE_NOT_STARTED = "not_started"
VOICE_STATE_COLLECTING = "collecting"
VOICE_STATE_AWAITING_VERIFICATION = "awaiting_verification"
VOICE_STATE_TRAINING = "training"
VOICE_STATE_FINE_TUNED = "fine_tuned"
VOICE_STATE_FAILED = "failed"

JOB_STATE_PENDING = "pending"
JOB_STATE_RUNNING = "running"
JOB_STATE_COMPLETED = "completed"
JOB_STATE_FAILED = "failed"

_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS avatar_emotion_media (
    asset_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    assistant_id TEXT NOT NULL,
    emotion TEXT NOT NULL,
    asset_kind TEXT NOT NULL,
    variant_key TEXT NOT NULL DEFAULT '',
    mime_type TEXT NOT NULL,
    bytes BYTEA NOT NULL,
    byte_length BIGINT NOT NULL,
    duration_seconds REAL,
    width INTEGER,
    height INTEGER,
    vendor TEXT,
    vendor_request_id TEXT,
    elevenlabs_asset_id TEXT,
    prompt TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (assistant_id, emotion, asset_kind, variant_key)
);
CREATE INDEX IF NOT EXISTS avatar_emotion_media_assistant_idx
    ON avatar_emotion_media (assistant_id);

CREATE TABLE IF NOT EXISTS avatar_voice_clips (
    clip_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    assistant_id TEXT NOT NULL,
    source TEXT NOT NULL,
    source_document_name TEXT,
    mime_type TEXT NOT NULL,
    bytes BYTEA NOT NULL,
    byte_length BIGINT NOT NULL,
    duration_seconds REAL NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS avatar_voice_clips_assistant_idx
    ON avatar_voice_clips (assistant_id);

CREATE TABLE IF NOT EXISTS avatar_voice (
    assistant_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    instant_voice_id TEXT,
    instant_voice_seconds REAL NOT NULL DEFAULT 0,
    professional_voice_id TEXT,
    professional_state TEXT NOT NULL DEFAULT 'not_started',
    collected_seconds REAL NOT NULL DEFAULT 0,
    verification_requested_at TIMESTAMPTZ,
    training_started_at TIMESTAMPTZ,
    detail JSONB NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS avatar_media_jobs (
    job_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    assistant_id TEXT NOT NULL,
    job_kind TEXT NOT NULL,
    state TEXT NOT NULL,
    vendor_reference TEXT,
    detail JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS avatar_media_jobs_assistant_idx
    ON avatar_media_jobs (assistant_id, job_kind);
"""


def _now() -> datetime:
    return datetime.now(UTC)


class InMemoryMediaAssetRepository:
    """Dictionary-backed twin of the Postgres repository, for tests and dev."""

    def __init__(self) -> None:
        """Start empty."""
        self.assets: dict[str, dict[str, Any]] = {}
        self.clips: dict[str, dict[str, Any]] = {}
        self.voices: dict[str, dict[str, Any]] = {}
        self.jobs: dict[str, dict[str, Any]] = {}
        self.pool = None

    # -- emotion media -------------------------------------------------------

    async def upsert_emotion_asset(self, asset: dict[str, Any]) -> str:
        """Insert or replace one asset; return its id."""
        key = (
            asset["assistant_id"],
            asset["emotion"],
            asset["asset_kind"],
            asset.get("variant_key") or "",
        )
        for existing_id, existing in list(self.assets.items()):
            if (
                existing["assistant_id"],
                existing["emotion"],
                existing["asset_kind"],
                existing.get("variant_key") or "",
            ) == key:
                del self.assets[existing_id]
        asset_id = str(asset.get("asset_id") or uuid4())
        stored = {**asset, "asset_id": asset_id, "created_at": _now().isoformat()}
        stored.setdefault("variant_key", "")
        stored["byte_length"] = len(stored.get("bytes") or b"")
        self.assets[asset_id] = stored
        return asset_id

    async def list_emotion_assets(
        self, assistant_id: str, include_bytes: bool = False
    ) -> list[dict[str, Any]]:
        """Return every asset for an avatar, without bytes unless asked."""
        rows = [
            asset
            for asset in self.assets.values()
            if asset["assistant_id"] == assistant_id
        ]
        if include_bytes:
            return [dict(row) for row in rows]
        return [{k: v for k, v in row.items() if k != "bytes"} for row in rows]

    async def get_emotion_asset(self, asset_id: str) -> dict[str, Any] | None:
        """Return one asset with its bytes, or ``None``."""
        asset = self.assets.get(asset_id)
        return dict(asset) if asset else None

    async def delete_emotion_assets_for_avatar(
        self, assistant_id: str, asset_kinds: tuple[str, ...] | None = None
    ) -> int:
        """Delete an avatar's assets (optionally only certain kinds)."""
        removed = 0
        for asset_id, asset in list(self.assets.items()):
            if asset["assistant_id"] != assistant_id:
                continue
            if asset_kinds and asset["asset_kind"] not in asset_kinds:
                continue
            del self.assets[asset_id]
            removed += 1
        return removed

    # -- voice clips ---------------------------------------------------------

    async def add_voice_clip(self, clip: dict[str, Any]) -> str:
        """Store one target-only speech clip; return its id."""
        clip_id = str(uuid4())
        stored = {**clip, "clip_id": clip_id, "created_at": _now().isoformat()}
        stored["byte_length"] = len(stored.get("bytes") or b"")
        self.clips[clip_id] = stored
        return clip_id

    async def list_voice_clips(
        self, assistant_id: str, include_bytes: bool = False
    ) -> list[dict[str, Any]]:
        """Return an avatar's clips, oldest first."""
        rows = sorted(
            (
                clip
                for clip in self.clips.values()
                if clip["assistant_id"] == assistant_id
            ),
            key=lambda clip: clip["created_at"],
        )
        if include_bytes:
            return [dict(row) for row in rows]
        return [{k: v for k, v in row.items() if k != "bytes"} for row in rows]

    async def total_voice_seconds(self, assistant_id: str) -> float:
        """Sum of clip durations for an avatar."""
        return float(
            sum(
                float(clip.get("duration_seconds") or 0.0)
                for clip in self.clips.values()
                if clip["assistant_id"] == assistant_id
            )
        )

    async def delete_voice_clips_for_avatar(self, assistant_id: str) -> int:
        """Delete an avatar's clips."""
        before = len(self.clips)
        self.clips = {
            clip_id: clip
            for clip_id, clip in self.clips.items()
            if clip["assistant_id"] != assistant_id
        }
        return before - len(self.clips)

    # -- voice record --------------------------------------------------------

    async def get_voice(self, assistant_id: str) -> dict[str, Any] | None:
        """Return the avatar's voice record, or ``None``."""
        record = self.voices.get(assistant_id)
        return dict(record) if record else None

    async def upsert_voice(self, record: dict[str, Any]) -> None:
        """Insert or replace the avatar's voice record."""
        stored = dict(self.voices.get(record["assistant_id"], {}))
        stored.update(record)
        stored["updated_at"] = _now().isoformat()
        stored.setdefault("professional_state", VOICE_STATE_NOT_STARTED)
        stored.setdefault("collected_seconds", 0.0)
        stored.setdefault("detail", {})
        self.voices[record["assistant_id"]] = stored

    async def delete_voice_for_avatar(self, assistant_id: str) -> int:
        """Delete the avatar's voice record."""
        return 1 if self.voices.pop(assistant_id, None) else 0

    # -- durable jobs --------------------------------------------------------

    async def create_job(
        self,
        *,
        user_id: str,
        assistant_id: str,
        job_kind: str,
        detail: dict[str, Any] | None = None,
        vendor_reference: str | None = None,
        state: str = JOB_STATE_PENDING,
    ) -> str:
        """Create a durable job row; return its id."""
        job_id = str(uuid4())
        now = _now().isoformat()
        self.jobs[job_id] = {
            "job_id": job_id,
            "user_id": user_id,
            "assistant_id": assistant_id,
            "job_kind": job_kind,
            "state": state,
            "vendor_reference": vendor_reference,
            "detail": dict(detail or {}),
            "created_at": now,
            "updated_at": now,
        }
        return job_id

    async def update_job(
        self,
        job_id: str,
        *,
        state: str | None = None,
        detail: dict[str, Any] | None = None,
        vendor_reference: str | None = None,
    ) -> None:
        """Update a job's state, detail, or vendor reference."""
        job = self.jobs.get(job_id)
        if job is None:
            return
        if state is not None:
            job["state"] = state
        if detail is not None:
            job["detail"] = {**job.get("detail", {}), **detail}
        if vendor_reference is not None:
            job["vendor_reference"] = vendor_reference
        job["updated_at"] = _now().isoformat()

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Return one job, or ``None``."""
        job = self.jobs.get(job_id)
        return dict(job) if job else None

    async def list_jobs(
        self,
        *,
        assistant_id: str | None = None,
        job_kind: str | None = None,
        states: tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        """Return jobs matching the filters, newest first."""
        rows = [
            job
            for job in self.jobs.values()
            if (assistant_id is None or job["assistant_id"] == assistant_id)
            and (job_kind is None or job["job_kind"] == job_kind)
            and (states is None or job["state"] in states)
        ]
        return sorted(rows, key=lambda job: job["created_at"], reverse=True)


class PostgresMediaAssetRepository:
    """Repository over the application's psycopg connection pool."""

    def __init__(self, pool: Any) -> None:
        """Bind to the application's ``AsyncConnectionPool``."""
        self.pool = pool

    async def _execute(self, sql: str, params: tuple = ()) -> Any:
        async with self.pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(sql, params)
                return cursor.rowcount

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

    # -- emotion media -------------------------------------------------------

    _ASSET_COLUMNS = (
        "asset_id, user_id, assistant_id, emotion, asset_kind, variant_key, mime_type, "
        "byte_length, duration_seconds, width, height, vendor, vendor_request_id, "
        "elevenlabs_asset_id, prompt, created_at"
    )

    @classmethod
    def _asset_row(cls, row: tuple, include_bytes: bool = False) -> dict[str, Any]:
        names = [name.strip() for name in cls._ASSET_COLUMNS.split(",")]
        record = dict(zip(names, row))
        record["asset_id"] = str(record["asset_id"])
        if isinstance(record.get("created_at"), datetime):
            record["created_at"] = record["created_at"].isoformat()
        if include_bytes:
            record["bytes"] = bytes(row[len(names)])
        return record

    async def upsert_emotion_asset(self, asset: dict[str, Any]) -> str:
        """Insert or replace one asset; return its id."""
        asset_id = str(asset.get("asset_id") or uuid4())
        payload = asset.get("bytes") or b""
        await self._execute(
            f"""
            INSERT INTO avatar_emotion_media
                ({self._ASSET_COLUMNS.replace(", created_at", "")}, bytes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (assistant_id, emotion, asset_kind, variant_key) DO UPDATE SET
                asset_id = EXCLUDED.asset_id,
                mime_type = EXCLUDED.mime_type,
                byte_length = EXCLUDED.byte_length,
                duration_seconds = EXCLUDED.duration_seconds,
                width = EXCLUDED.width,
                height = EXCLUDED.height,
                vendor = EXCLUDED.vendor,
                vendor_request_id = EXCLUDED.vendor_request_id,
                elevenlabs_asset_id = EXCLUDED.elevenlabs_asset_id,
                prompt = EXCLUDED.prompt,
                bytes = EXCLUDED.bytes,
                created_at = now();
            """,
            (
                asset_id,
                asset["user_id"],
                asset["assistant_id"],
                asset["emotion"],
                asset["asset_kind"],
                asset.get("variant_key") or "",
                asset["mime_type"],
                len(payload),
                asset.get("duration_seconds"),
                asset.get("width"),
                asset.get("height"),
                asset.get("vendor"),
                asset.get("vendor_request_id"),
                asset.get("elevenlabs_asset_id"),
                asset.get("prompt"),
                payload,
            ),
        )
        return asset_id

    async def list_emotion_assets(
        self, assistant_id: str, include_bytes: bool = False
    ) -> list[dict[str, Any]]:
        """Return every asset for an avatar, without bytes unless asked."""
        columns = self._ASSET_COLUMNS + (", bytes" if include_bytes else "")
        rows = await self._fetchall(
            f"SELECT {columns} FROM avatar_emotion_media WHERE assistant_id = %s "
            "ORDER BY asset_kind, emotion;",
            (assistant_id,),
        )
        return [self._asset_row(row, include_bytes) for row in rows]

    async def get_emotion_asset(self, asset_id: str) -> dict[str, Any] | None:
        """Return one asset with its bytes, or ``None``."""
        row = await self._fetchone(
            f"SELECT {self._ASSET_COLUMNS}, bytes FROM avatar_emotion_media "
            "WHERE asset_id = %s;",
            (asset_id,),
        )
        return self._asset_row(row, include_bytes=True) if row else None

    async def delete_emotion_assets_for_avatar(
        self, assistant_id: str, asset_kinds: tuple[str, ...] | None = None
    ) -> int:
        """Delete an avatar's assets (optionally only certain kinds)."""
        if asset_kinds:
            return int(
                await self._execute(
                    "DELETE FROM avatar_emotion_media WHERE assistant_id = %s "
                    "AND asset_kind = ANY(%s);",
                    (assistant_id, list(asset_kinds)),
                )
                or 0
            )
        return int(
            await self._execute(
                "DELETE FROM avatar_emotion_media WHERE assistant_id = %s;",
                (assistant_id,),
            )
            or 0
        )

    # -- voice clips ---------------------------------------------------------

    async def add_voice_clip(self, clip: dict[str, Any]) -> str:
        """Store one target-only speech clip; return its id."""
        clip_id = str(uuid4())
        payload = clip.get("bytes") or b""
        await self._execute(
            """
            INSERT INTO avatar_voice_clips
                (clip_id, user_id, assistant_id, source, source_document_name,
                 mime_type, bytes, byte_length, duration_seconds)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """,
            (
                clip_id,
                clip["user_id"],
                clip["assistant_id"],
                clip.get("source") or "upload",
                clip.get("source_document_name"),
                clip.get("mime_type") or "audio/mpeg",
                payload,
                len(payload),
                float(clip.get("duration_seconds") or 0.0),
            ),
        )
        return clip_id

    async def list_voice_clips(
        self, assistant_id: str, include_bytes: bool = False
    ) -> list[dict[str, Any]]:
        """Return an avatar's clips, oldest first."""
        columns = (
            "clip_id, user_id, assistant_id, source, source_document_name, mime_type, "
            "byte_length, duration_seconds, created_at"
            + (", bytes" if include_bytes else "")
        )
        rows = await self._fetchall(
            f"SELECT {columns} FROM avatar_voice_clips WHERE assistant_id = %s "
            "ORDER BY created_at ASC;",
            (assistant_id,),
        )
        names = [name.strip() for name in columns.split(",")]
        clips = []
        for row in rows:
            record = dict(zip(names, row))
            record["clip_id"] = str(record["clip_id"])
            if isinstance(record.get("created_at"), datetime):
                record["created_at"] = record["created_at"].isoformat()
            if include_bytes:
                record["bytes"] = bytes(record["bytes"])
            clips.append(record)
        return clips

    async def total_voice_seconds(self, assistant_id: str) -> float:
        """Sum of clip durations for an avatar."""
        row = await self._fetchone(
            "SELECT COALESCE(SUM(duration_seconds), 0) FROM avatar_voice_clips "
            "WHERE assistant_id = %s;",
            (assistant_id,),
        )
        return float(row[0] if row else 0.0)

    async def delete_voice_clips_for_avatar(self, assistant_id: str) -> int:
        """Delete an avatar's clips."""
        return int(
            await self._execute(
                "DELETE FROM avatar_voice_clips WHERE assistant_id = %s;",
                (assistant_id,),
            )
            or 0
        )

    # -- voice record --------------------------------------------------------

    _VOICE_COLUMNS = (
        "assistant_id, user_id, instant_voice_id, instant_voice_seconds, "
        "professional_voice_id, professional_state, collected_seconds, "
        "verification_requested_at, training_started_at, detail, updated_at"
    )

    async def get_voice(self, assistant_id: str) -> dict[str, Any] | None:
        """Return the avatar's voice record, or ``None``."""
        row = await self._fetchone(
            f"SELECT {self._VOICE_COLUMNS} FROM avatar_voice WHERE assistant_id = %s;",
            (assistant_id,),
        )
        if not row:
            return None
        names = [name.strip() for name in self._VOICE_COLUMNS.split(",")]
        record = dict(zip(names, row))
        for key in ("verification_requested_at", "training_started_at", "updated_at"):
            if isinstance(record.get(key), datetime):
                record[key] = record[key].isoformat()
        return record

    async def upsert_voice(self, record: dict[str, Any]) -> None:
        """Insert or replace the avatar's voice record (merging over existing)."""
        from psycopg.types.json import Jsonb

        existing = await self.get_voice(record["assistant_id"]) or {}
        merged = {**existing, **record}
        await self._execute(
            """
            INSERT INTO avatar_voice
                (assistant_id, user_id, instant_voice_id, instant_voice_seconds,
                 professional_voice_id, professional_state, collected_seconds,
                 verification_requested_at, training_started_at, detail, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (assistant_id) DO UPDATE SET
                user_id = EXCLUDED.user_id,
                instant_voice_id = EXCLUDED.instant_voice_id,
                instant_voice_seconds = EXCLUDED.instant_voice_seconds,
                professional_voice_id = EXCLUDED.professional_voice_id,
                professional_state = EXCLUDED.professional_state,
                collected_seconds = EXCLUDED.collected_seconds,
                verification_requested_at = EXCLUDED.verification_requested_at,
                training_started_at = EXCLUDED.training_started_at,
                detail = EXCLUDED.detail,
                updated_at = now();
            """,
            (
                merged["assistant_id"],
                merged.get("user_id"),
                merged.get("instant_voice_id"),
                float(merged.get("instant_voice_seconds") or 0.0),
                merged.get("professional_voice_id"),
                merged.get("professional_state") or VOICE_STATE_NOT_STARTED,
                float(merged.get("collected_seconds") or 0.0),
                merged.get("verification_requested_at"),
                merged.get("training_started_at"),
                Jsonb(merged.get("detail") or {}),
            ),
        )

    async def delete_voice_for_avatar(self, assistant_id: str) -> int:
        """Delete the avatar's voice record."""
        return int(
            await self._execute(
                "DELETE FROM avatar_voice WHERE assistant_id = %s;", (assistant_id,)
            )
            or 0
        )

    # -- durable jobs --------------------------------------------------------

    _JOB_COLUMNS = (
        "job_id, user_id, assistant_id, job_kind, state, vendor_reference, detail, "
        "created_at, updated_at"
    )

    @classmethod
    def _job_row(cls, row: tuple) -> dict[str, Any]:
        names = [name.strip() for name in cls._JOB_COLUMNS.split(",")]
        record = dict(zip(names, row))
        record["job_id"] = str(record["job_id"])
        for key in ("created_at", "updated_at"):
            if isinstance(record.get(key), datetime):
                record[key] = record[key].isoformat()
        return record

    async def create_job(
        self,
        *,
        user_id: str,
        assistant_id: str,
        job_kind: str,
        detail: dict[str, Any] | None = None,
        vendor_reference: str | None = None,
        state: str = JOB_STATE_PENDING,
    ) -> str:
        """Create a durable job row; return its id."""
        from psycopg.types.json import Jsonb

        job_id = str(uuid4())
        await self._execute(
            """
            INSERT INTO avatar_media_jobs
                (job_id, user_id, assistant_id, job_kind, state, vendor_reference, detail)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
            """,
            (
                job_id,
                user_id,
                assistant_id,
                job_kind,
                state,
                vendor_reference,
                Jsonb(detail or {}),
            ),
        )
        return job_id

    async def update_job(
        self,
        job_id: str,
        *,
        state: str | None = None,
        detail: dict[str, Any] | None = None,
        vendor_reference: str | None = None,
    ) -> None:
        """Update a job's state, detail (merged), or vendor reference."""
        from psycopg.types.json import Jsonb

        assignments = ["updated_at = now()"]
        params: list[Any] = []
        if state is not None:
            assignments.append("state = %s")
            params.append(state)
        if detail is not None:
            assignments.append("detail = detail || %s")
            params.append(Jsonb(detail))
        if vendor_reference is not None:
            assignments.append("vendor_reference = %s")
            params.append(vendor_reference)
        params.append(job_id)
        await self._execute(
            f"UPDATE avatar_media_jobs SET {', '.join(assignments)} WHERE job_id = %s;",
            tuple(params),
        )

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Return one job, or ``None``."""
        row = await self._fetchone(
            f"SELECT {self._JOB_COLUMNS} FROM avatar_media_jobs WHERE job_id = %s;",
            (job_id,),
        )
        return self._job_row(row) if row else None

    async def list_jobs(
        self,
        *,
        assistant_id: str | None = None,
        job_kind: str | None = None,
        states: tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        """Return jobs matching the filters, newest first."""
        clauses = []
        params: list[Any] = []
        if assistant_id is not None:
            clauses.append("assistant_id = %s")
            params.append(assistant_id)
        if job_kind is not None:
            clauses.append("job_kind = %s")
            params.append(job_kind)
        if states:
            clauses.append("state = ANY(%s)")
            params.append(list(states))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = await self._fetchall(
            f"SELECT {self._JOB_COLUMNS} FROM avatar_media_jobs{where} "
            "ORDER BY created_at DESC;",
            tuple(params),
        )
        return [self._job_row(row) for row in rows]


_repository: Any | None = None


def set_media_asset_repository(repository: Any | None) -> None:
    """Publish the process-wide repository (or clear it with ``None``)."""
    global _repository
    _repository = repository


def get_media_asset_repository() -> Any | None:
    """Return the published repository, or ``None``."""
    return _repository


async def ensure_media_asset_tables(pool: Any) -> None:
    """Create the media tables if they do not exist. Best-effort at boot."""
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(_CREATE_TABLES_SQL)
    except Exception as table_error:  # noqa: BLE001 - non-fatal at startup
        logger.error("Could not ensure the avatar media tables exist: %s", table_error)


async def delete_media_for_avatar(repository: Any, assistant_id: str) -> dict[str, int]:
    """Remove every generated asset, clip, and voice record of one avatar."""
    if repository is None:
        return {}
    counts = {}
    try:
        counts["emotion_media"] = await repository.delete_emotion_assets_for_avatar(
            assistant_id
        )
        counts["voice_clips"] = await repository.delete_voice_clips_for_avatar(
            assistant_id
        )
        counts["voice"] = await repository.delete_voice_for_avatar(assistant_id)
    except Exception:  # noqa: BLE001
        logger.debug(
            "Could not delete media for avatar %s", assistant_id, exc_info=True
        )
    return counts
