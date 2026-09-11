"""Where motion lives: six Postgres tables, created on boot, plus an in-memory twin.

Nothing motion-related goes in the LangGraph store: the store embeds every
value at 640 dimensions, and a coordinate timeline is neither text nor
searchable. These tables mirror ``avatar_emotion_media``'s home and are
created by the same boot path.

* ``avatar_motion_tracks`` — the compact timeline: body joints, head pose and
  face coefficients (or, before a basis exists, the dense residual), float16,
  tagged with the landmark set and basis they were written against.
* ``avatar_motion_golden`` — raw dense mesh for a curated, bounded set of
  segments: the ground truth that survives any refit or a denser set.
* ``avatar_motion_basis`` — fitted bases; the current one is marked, older
  ones stay while a track still references them.
* ``avatar_motion_primitives`` — the prototype trajectories.
* ``avatar_motion_signature`` — the scalar running means, per emotion.
* ``avatar_motion_profile`` — the rendered text per emotion, read every turn.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from src.anubis.utils.postgres_ddl import execute_ddl_script

logger = logging.getLogger(__name__)

SOURCE_LIVE_CAMERA = "live_camera"
SOURCE_UPLOADED_VIDEO = "uploaded_video"
SOURCE_GENERATED_CLIP = "generated_clip"
SOURCE_NEURAL_DECODER = "neural_decoder"
MOTION_SOURCES = (
    SOURCE_LIVE_CAMERA,
    SOURCE_UPLOADED_VIDEO,
    SOURCE_GENERATED_CLIP,
    SOURCE_NEURAL_DECODER,
)

_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS avatar_motion_tracks (
    track_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    assistant_id TEXT NOT NULL,
    source TEXT NOT NULL,
    source_document_name TEXT,
    emotion TEXT NOT NULL DEFAULT 'neutral',
    landmark_set_version TEXT NOT NULL,
    basis_id UUID,
    face_encoding TEXT NOT NULL DEFAULT 'none',
    captured_at TIMESTAMPTZ,
    duration_seconds REAL NOT NULL,
    body_sample_rate_hz REAL,
    body_frame_count INTEGER,
    body BYTEA,
    head_sample_rate_hz REAL,
    head_frame_count INTEGER,
    head_pose BYTEA,
    face_sample_rate_hz REAL,
    face_frame_count INTEGER,
    face BYTEA,
    speech JSONB NOT NULL DEFAULT '[]',
    identity_confidence REAL,
    byte_length BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS avatar_motion_tracks_assistant_idx
    ON avatar_motion_tracks (assistant_id, emotion, created_at);

CREATE TABLE IF NOT EXISTS avatar_motion_golden (
    segment_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    assistant_id TEXT NOT NULL,
    source TEXT NOT NULL,
    emotion TEXT NOT NULL DEFAULT 'neutral',
    landmark_set_version TEXT NOT NULL,
    duration_seconds REAL NOT NULL,
    body_sample_rate_hz REAL,
    body_frame_count INTEGER,
    body BYTEA,
    head_sample_rate_hz REAL,
    head_frame_count INTEGER,
    head_pose BYTEA,
    face_sample_rate_hz REAL,
    face_frame_count INTEGER,
    face BYTEA,
    identity_confidence REAL,
    byte_length BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS avatar_motion_golden_assistant_idx
    ON avatar_motion_golden (assistant_id);

CREATE TABLE IF NOT EXISTS avatar_motion_basis (
    basis_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    assistant_id TEXT NOT NULL,
    landmark_set_version TEXT NOT NULL,
    component_count INTEGER NOT NULL,
    mean BYTEA NOT NULL,
    components BYTEA NOT NULL,
    explained_variance JSONB NOT NULL DEFAULT '[]',
    reconstruction_error REAL NOT NULL DEFAULT 0,
    fitted_frames INTEGER NOT NULL DEFAULT 0,
    fitted_seconds REAL NOT NULL DEFAULT 0,
    is_current BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS avatar_motion_basis_assistant_idx
    ON avatar_motion_basis (assistant_id, is_current);

CREATE TABLE IF NOT EXISTS avatar_motion_primitives (
    primitive_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    assistant_id TEXT NOT NULL,
    emotion TEXT NOT NULL DEFAULT 'neutral',
    channel TEXT NOT NULL,
    landmark_set_version TEXT NOT NULL,
    prototype BYTEA NOT NULL,
    prototype_length INTEGER NOT NULL,
    dimension INTEGER NOT NULL,
    occurrences INTEGER NOT NULL DEFAULT 0,
    duration_mean REAL NOT NULL DEFAULT 0,
    duration_std REAL NOT NULL DEFAULT 0,
    amplitude_mean REAL NOT NULL DEFAULT 0,
    amplitude_std REAL NOT NULL DEFAULT 0,
    context JSONB NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS avatar_motion_primitives_assistant_idx
    ON avatar_motion_primitives (assistant_id, emotion, channel);

CREATE TABLE IF NOT EXISTS avatar_motion_signature (
    assistant_id TEXT NOT NULL,
    emotion TEXT NOT NULL,
    user_id TEXT NOT NULL,
    windows_observed INTEGER NOT NULL DEFAULT 0,
    seconds_observed REAL NOT NULL DEFAULT 0,
    signature JSONB NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (assistant_id, emotion)
);

CREATE TABLE IF NOT EXISTS avatar_motion_profile (
    assistant_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    role_section TEXT NOT NULL DEFAULT '',
    blocks JSONB NOT NULL DEFAULT '{}',
    motion_fidelity JSONB NOT NULL DEFAULT '{}',
    seconds_observed REAL NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


_TRACK_META_COLUMNS = (
    "track_id, user_id, assistant_id, source, source_document_name, emotion, "
    "landmark_set_version, basis_id, face_encoding, captured_at, duration_seconds, "
    "body_sample_rate_hz, body_frame_count, head_sample_rate_hz, head_frame_count, "
    "face_sample_rate_hz, face_frame_count, speech, identity_confidence, byte_length, created_at"
)
_TRACK_BYTES_COLUMNS = "body, head_pose, face"
_GOLDEN_META_COLUMNS = (
    "segment_id, user_id, assistant_id, source, emotion, landmark_set_version, duration_seconds, "
    "body_sample_rate_hz, body_frame_count, head_sample_rate_hz, head_frame_count, "
    "face_sample_rate_hz, face_frame_count, identity_confidence, byte_length, created_at"
)
_BASIS_COLUMNS = (
    "basis_id, user_id, assistant_id, landmark_set_version, component_count, mean, components, "
    "explained_variance, reconstruction_error, fitted_frames, fitted_seconds, is_current, created_at"
)
_PRIMITIVE_COLUMNS = (
    "primitive_id, user_id, assistant_id, emotion, channel, landmark_set_version, prototype, "
    "prototype_length, dimension, occurrences, duration_mean, duration_std, amplitude_mean, "
    "amplitude_std, context, updated_at"
)


def _names(columns: str) -> list[str]:
    return [name.strip() for name in columns.split(",")]


class InMemoryMotionRepository:
    """Dictionary-backed twin of the Postgres repository, for tests."""

    def __init__(self) -> None:
        """Initialize."""
        self.tracks: dict[str, dict[str, Any]] = {}
        self.golden: dict[str, dict[str, Any]] = {}
        self.bases: dict[str, dict[str, Any]] = {}
        self.primitives: dict[str, dict[str, Any]] = {}
        self.signatures: dict[tuple[str, str], dict[str, Any]] = {}
        self.profiles: dict[str, dict[str, Any]] = {}

    # -- tracks ---------------------------------------------------------------

    async def add_track(self, track: dict[str, Any]) -> str:
        """Insert one track row; return its id."""
        track_id = str(track.get("track_id") or uuid4())
        record = {**track, "track_id": track_id, "created_at": _now().isoformat()}
        self.tracks[track_id] = record
        return track_id

    async def list_tracks(self, assistant_id: str, *, source: str | None = None) -> list[dict[str, Any]]:
        """List track metadata (no buffers) for an avatar, newest first."""
        rows = [
            {key: value for key, value in track.items() if key not in ("body", "head_pose", "face")}
            for track in self.tracks.values()
            if track["assistant_id"] == assistant_id and (source is None or track.get("source") == source)
        ]
        return sorted(rows, key=lambda row: row["created_at"], reverse=True)

    async def get_track(self, track_id: str) -> dict[str, Any] | None:
        """Return one track with its buffers, or ``None``."""
        return self.tracks.get(track_id)

    async def delete_tracks(self, track_ids: list[str]) -> int:
        """Delete the named tracks; return how many were deleted."""
        count = 0
        for track_id in track_ids:
            if self.tracks.pop(track_id, None) is not None:
                count += 1
        return count

    # -- golden ---------------------------------------------------------------

    async def add_golden_segment(self, segment: dict[str, Any]) -> str:
        """Insert one raw golden segment; return its id."""
        segment_id = str(segment.get("segment_id") or uuid4())
        self.golden[segment_id] = {**segment, "segment_id": segment_id, "created_at": _now().isoformat()}
        return segment_id

    async def list_golden_segments(self, assistant_id: str) -> list[dict[str, Any]]:
        """List golden segment metadata (no buffers) for an avatar."""
        return [
            {key: value for key, value in segment.items() if key not in ("body", "head_pose", "face")}
            for segment in self.golden.values()
            if segment["assistant_id"] == assistant_id
        ]

    async def get_golden_segment(self, segment_id: str) -> dict[str, Any] | None:
        """Return one golden segment with its buffers, or ``None``."""
        return self.golden.get(segment_id)

    async def delete_golden_segments(self, segment_ids: list[str]) -> int:
        """Delete the named golden segments; return how many were deleted."""
        return sum(1 for segment_id in segment_ids if self.golden.pop(segment_id, None) is not None)

    # -- basis ----------------------------------------------------------------

    async def add_basis(self, basis: dict[str, Any]) -> str:
        """Insert a basis as the avatar's current one; return its id."""
        basis_id = str(basis.get("basis_id") or uuid4())
        for other in self.bases.values():
            if other["assistant_id"] == basis["assistant_id"]:
                other["is_current"] = False
        self.bases[basis_id] = {**basis, "basis_id": basis_id, "is_current": True, "created_at": _now().isoformat()}
        return basis_id

    async def get_current_basis(self, assistant_id: str) -> dict[str, Any] | None:
        """Return the avatar's current basis record, or ``None``."""
        for basis in self.bases.values():
            if basis["assistant_id"] == assistant_id and basis.get("is_current"):
                return basis
        return None

    async def get_basis(self, basis_id: str) -> dict[str, Any] | None:
        """Return one basis record by id, or ``None``."""
        return self.bases.get(basis_id)

    async def delete_unreferenced_bases(self, assistant_id: str) -> int:
        """Delete superseded bases no track references; return the count."""
        referenced = {str(track.get("basis_id")) for track in self.tracks.values() if track.get("basis_id")}
        doomed = [
            basis_id
            for basis_id, basis in self.bases.items()
            if basis["assistant_id"] == assistant_id and not basis.get("is_current") and basis_id not in referenced
        ]
        for basis_id in doomed:
            del self.bases[basis_id]
        return len(doomed)

    # -- primitives -----------------------------------------------------------

    async def replace_primitives(self, assistant_id: str, emotion: str, records: list[dict[str, Any]], *, user_id: str) -> None:
        """Replace the avatar's primitives for one emotion."""
        for primitive_id, primitive in list(self.primitives.items()):
            if primitive["assistant_id"] == assistant_id and primitive["emotion"] == emotion:
                del self.primitives[primitive_id]
        for record in records:
            primitive_id = str(record.get("primitive_id") or uuid4())
            self.primitives[primitive_id] = {
                **record,
                "primitive_id": primitive_id,
                "assistant_id": assistant_id,
                "user_id": user_id,
                "emotion": emotion,
                "updated_at": _now().isoformat(),
            }

    async def list_primitives(self, assistant_id: str, *, emotion: str | None = None) -> list[dict[str, Any]]:
        """List primitive records for an avatar, most frequent first."""
        return [
            primitive
            for primitive in self.primitives.values()
            if primitive["assistant_id"] == assistant_id and (emotion is None or primitive["emotion"] == emotion)
        ]

    # -- signature ------------------------------------------------------------

    async def get_signature(self, assistant_id: str, emotion: str) -> dict[str, Any] | None:
        """Return the signature row for one emotion, or ``None``."""
        return self.signatures.get((assistant_id, emotion))

    async def list_signatures(self, assistant_id: str) -> list[dict[str, Any]]:
        """List every emotion's signature row for an avatar."""
        return [record for (aid, _), record in self.signatures.items() if aid == assistant_id]

    async def upsert_signature(self, record: dict[str, Any]) -> None:
        """Insert or update one emotion's signature row."""
        key = (record["assistant_id"], record["emotion"])
        self.signatures[key] = {**record, "updated_at": _now().isoformat()}

    # -- profile --------------------------------------------------------------

    async def get_profile(self, assistant_id: str) -> dict[str, Any] | None:
        """Return the rendered profile row, or ``None``."""
        return self.profiles.get(assistant_id)

    async def upsert_profile(self, record: dict[str, Any]) -> None:
        """Insert or update the rendered profile row."""
        self.profiles[record["assistant_id"]] = {**record, "updated_at": _now().isoformat()}

    # -- everything -----------------------------------------------------------

    async def delete_all_for_avatar(self, assistant_id: str) -> dict[str, int]:
        """Delete everything motion-related for an avatar; return counts per table."""
        counts = {"tracks": 0, "golden": 0, "bases": 0, "primitives": 0, "signatures": 0, "profiles": 0}
        for store, name in (
            (self.tracks, "tracks"),
            (self.golden, "golden"),
            (self.bases, "bases"),
            (self.primitives, "primitives"),
        ):
            for key in [k for k, v in store.items() if v["assistant_id"] == assistant_id]:
                del store[key]
                counts[name] += 1
        for key in [k for k in self.signatures if k[0] == assistant_id]:
            del self.signatures[key]
            counts["signatures"] += 1
        if self.profiles.pop(assistant_id, None) is not None:
            counts["profiles"] += 1
        return counts


class PostgresMotionRepository:
    """Repository over the application's psycopg connection pool."""

    def __init__(self, pool: Any) -> None:
        """Initialize."""
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

    @staticmethod
    def _row(columns: str, row: tuple, *, bytes_columns: str | None = None) -> dict[str, Any]:
        names = _names(columns)
        record = {name: _iso(value) for name, value in zip(names, row)}
        for key in ("track_id", "segment_id", "basis_id", "primitive_id"):
            if record.get(key) is not None:
                record[key] = str(record[key])
        if bytes_columns:
            for offset, name in enumerate(_names(bytes_columns)):
                value = row[len(names) + offset]
                record[name] = bytes(value) if value is not None else None
        return record

    # -- tracks ---------------------------------------------------------------

    async def add_track(self, track: dict[str, Any]) -> str:
        """Insert one track row; return its id."""
        import json

        track_id = str(track.get("track_id") or uuid4())
        await self._execute(
            f"""
            INSERT INTO avatar_motion_tracks ({_TRACK_META_COLUMNS.replace(', created_at', '')}, {_TRACK_BYTES_COLUMNS})
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s)
            """,
            (
                track_id,
                track["user_id"],
                track["assistant_id"],
                track["source"],
                track.get("source_document_name"),
                track.get("emotion") or "neutral",
                track["landmark_set_version"],
                track.get("basis_id"),
                track.get("face_encoding") or "none",
                track.get("captured_at"),
                float(track.get("duration_seconds") or 0.0),
                track.get("body_sample_rate_hz"),
                track.get("body_frame_count"),
                track.get("head_sample_rate_hz"),
                track.get("head_frame_count"),
                track.get("face_sample_rate_hz"),
                track.get("face_frame_count"),
                json.dumps(track.get("speech") or []),
                track.get("identity_confidence"),
                int(track.get("byte_length") or 0),
                track.get("body"),
                track.get("head_pose"),
                track.get("face"),
            ),
        )
        return track_id

    async def list_tracks(self, assistant_id: str, *, source: str | None = None) -> list[dict[str, Any]]:
        """List track metadata (no buffers) for an avatar, newest first."""
        sql = f"SELECT {_TRACK_META_COLUMNS} FROM avatar_motion_tracks WHERE assistant_id = %s"
        params: tuple = (assistant_id,)
        if source is not None:
            sql += " AND source = %s"
            params = (assistant_id, source)
        sql += " ORDER BY created_at DESC"
        return [self._row(_TRACK_META_COLUMNS, row) for row in await self._fetchall(sql, params)]

    async def get_track(self, track_id: str) -> dict[str, Any] | None:
        """Return one track with its buffers, or ``None``."""
        row = await self._fetchone(
            f"SELECT {_TRACK_META_COLUMNS}, {_TRACK_BYTES_COLUMNS} FROM avatar_motion_tracks WHERE track_id = %s",
            (track_id,),
        )
        return self._row(_TRACK_META_COLUMNS, row, bytes_columns=_TRACK_BYTES_COLUMNS) if row else None

    async def delete_tracks(self, track_ids: list[str]) -> int:
        """Delete the named tracks; return how many were deleted."""
        if not track_ids:
            return 0
        return int(
            await self._execute(
                "DELETE FROM avatar_motion_tracks WHERE track_id = ANY(%s::uuid[])", (list(track_ids),)
            )
        )

    # -- golden ---------------------------------------------------------------

    async def add_golden_segment(self, segment: dict[str, Any]) -> str:
        """Insert one raw golden segment; return its id."""
        segment_id = str(segment.get("segment_id") or uuid4())
        await self._execute(
            f"""
            INSERT INTO avatar_motion_golden ({_GOLDEN_META_COLUMNS.replace(', created_at', '')}, body, head_pose, face)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                segment_id,
                segment["user_id"],
                segment["assistant_id"],
                segment["source"],
                segment.get("emotion") or "neutral",
                segment["landmark_set_version"],
                float(segment.get("duration_seconds") or 0.0),
                segment.get("body_sample_rate_hz"),
                segment.get("body_frame_count"),
                segment.get("head_sample_rate_hz"),
                segment.get("head_frame_count"),
                segment.get("face_sample_rate_hz"),
                segment.get("face_frame_count"),
                segment.get("identity_confidence"),
                int(segment.get("byte_length") or 0),
                segment.get("body"),
                segment.get("head_pose"),
                segment.get("face"),
            ),
        )
        return segment_id

    async def list_golden_segments(self, assistant_id: str) -> list[dict[str, Any]]:
        """List golden segment metadata (no buffers) for an avatar."""
        rows = await self._fetchall(
            f"SELECT {_GOLDEN_META_COLUMNS} FROM avatar_motion_golden WHERE assistant_id = %s ORDER BY created_at DESC",
            (assistant_id,),
        )
        return [self._row(_GOLDEN_META_COLUMNS, row) for row in rows]

    async def get_golden_segment(self, segment_id: str) -> dict[str, Any] | None:
        """Return one golden segment with its buffers, or ``None``."""
        row = await self._fetchone(
            f"SELECT {_GOLDEN_META_COLUMNS}, body, head_pose, face FROM avatar_motion_golden WHERE segment_id = %s",
            (segment_id,),
        )
        return self._row(_GOLDEN_META_COLUMNS, row, bytes_columns="body, head_pose, face") if row else None

    async def delete_golden_segments(self, segment_ids: list[str]) -> int:
        """Delete the named golden segments; return how many were deleted."""
        if not segment_ids:
            return 0
        return int(
            await self._execute(
                "DELETE FROM avatar_motion_golden WHERE segment_id = ANY(%s::uuid[])", (list(segment_ids),)
            )
        )

    # -- basis ----------------------------------------------------------------

    async def add_basis(self, basis: dict[str, Any]) -> str:
        """Insert a basis as the avatar's current one; return its id."""
        import json

        basis_id = str(basis.get("basis_id") or uuid4())
        await self._execute(
            "UPDATE avatar_motion_basis SET is_current = FALSE WHERE assistant_id = %s", (basis["assistant_id"],)
        )
        await self._execute(
            f"""
            INSERT INTO avatar_motion_basis ({_BASIS_COLUMNS.replace(', created_at', '')})
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, TRUE)
            """,
            (
                basis_id,
                basis["user_id"],
                basis["assistant_id"],
                basis["landmark_set_version"],
                int(basis["component_count"]),
                basis["mean"],
                basis["components"],
                json.dumps(basis.get("explained_variance") or []),
                float(basis.get("reconstruction_error") or 0.0),
                int(basis.get("fitted_frames") or 0),
                float(basis.get("fitted_seconds") or 0.0),
            ),
        )
        return basis_id

    def _basis_row(self, row: tuple) -> dict[str, Any]:
        record = self._row(_BASIS_COLUMNS, row)
        record["mean"] = bytes(record["mean"])
        record["components"] = bytes(record["components"])
        return record

    async def get_current_basis(self, assistant_id: str) -> dict[str, Any] | None:
        """Return the avatar's current basis record, or ``None``."""
        row = await self._fetchone(
            f"SELECT {_BASIS_COLUMNS} FROM avatar_motion_basis WHERE assistant_id = %s AND is_current ORDER BY created_at DESC LIMIT 1",
            (assistant_id,),
        )
        return self._basis_row(row) if row else None

    async def get_basis(self, basis_id: str) -> dict[str, Any] | None:
        """Return one basis record by id, or ``None``."""
        row = await self._fetchone(f"SELECT {_BASIS_COLUMNS} FROM avatar_motion_basis WHERE basis_id = %s", (basis_id,))
        return self._basis_row(row) if row else None

    async def delete_unreferenced_bases(self, assistant_id: str) -> int:
        """Delete superseded bases no track references; return the count."""
        return int(
            await self._execute(
                """
                DELETE FROM avatar_motion_basis b
                WHERE b.assistant_id = %s AND NOT b.is_current
                  AND NOT EXISTS (SELECT 1 FROM avatar_motion_tracks t WHERE t.basis_id = b.basis_id)
                """,
                (assistant_id,),
            )
        )

    # -- primitives -----------------------------------------------------------

    async def replace_primitives(self, assistant_id: str, emotion: str, records: list[dict[str, Any]], *, user_id: str) -> None:
        """Replace the avatar's primitives for one emotion."""
        import json

        async with self.pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "DELETE FROM avatar_motion_primitives WHERE assistant_id = %s AND emotion = %s",
                    (assistant_id, emotion),
                )
                for record in records:
                    await cursor.execute(
                        f"""
                        INSERT INTO avatar_motion_primitives ({_PRIMITIVE_COLUMNS.replace(', updated_at', '')})
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        """,
                        (
                            str(record.get("primitive_id") or uuid4()),
                            user_id,
                            assistant_id,
                            emotion,
                            record["channel"],
                            record.get("landmark_set_version") or "",
                            record["prototype"],
                            int(record["prototype_length"]),
                            int(record["dimension"]),
                            int(record.get("occurrences") or 0),
                            float(record.get("duration_mean") or 0.0),
                            float(record.get("duration_std") or 0.0),
                            float(record.get("amplitude_mean") or 0.0),
                            float(record.get("amplitude_std") or 0.0),
                            json.dumps(record.get("context") or {}),
                        ),
                    )

    async def list_primitives(self, assistant_id: str, *, emotion: str | None = None) -> list[dict[str, Any]]:
        """List primitive records for an avatar, most frequent first."""
        sql = f"SELECT {_PRIMITIVE_COLUMNS} FROM avatar_motion_primitives WHERE assistant_id = %s"
        params: tuple = (assistant_id,)
        if emotion is not None:
            sql += " AND emotion = %s"
            params = (assistant_id, emotion)
        rows = await self._fetchall(sql + " ORDER BY occurrences DESC", params)
        records = []
        for row in rows:
            record = self._row(_PRIMITIVE_COLUMNS, row)
            record["prototype"] = bytes(record["prototype"])
            records.append(record)
        return records

    # -- signature ------------------------------------------------------------

    _SIGNATURE_COLUMNS = "assistant_id, emotion, user_id, windows_observed, seconds_observed, signature, updated_at"

    async def get_signature(self, assistant_id: str, emotion: str) -> dict[str, Any] | None:
        """Return the signature row for one emotion, or ``None``."""
        row = await self._fetchone(
            f"SELECT {self._SIGNATURE_COLUMNS} FROM avatar_motion_signature WHERE assistant_id = %s AND emotion = %s",
            (assistant_id, emotion),
        )
        return self._row(self._SIGNATURE_COLUMNS, row) if row else None

    async def list_signatures(self, assistant_id: str) -> list[dict[str, Any]]:
        """List every emotion's signature row for an avatar."""
        rows = await self._fetchall(
            f"SELECT {self._SIGNATURE_COLUMNS} FROM avatar_motion_signature WHERE assistant_id = %s", (assistant_id,)
        )
        return [self._row(self._SIGNATURE_COLUMNS, row) for row in rows]

    async def upsert_signature(self, record: dict[str, Any]) -> None:
        """Insert or update one emotion's signature row."""
        import json

        await self._execute(
            """
            INSERT INTO avatar_motion_signature (assistant_id, emotion, user_id, windows_observed, seconds_observed, signature, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, now())
            ON CONFLICT (assistant_id, emotion) DO UPDATE SET
                windows_observed = EXCLUDED.windows_observed,
                seconds_observed = EXCLUDED.seconds_observed,
                signature = EXCLUDED.signature,
                updated_at = now()
            """,
            (
                record["assistant_id"],
                record["emotion"],
                record["user_id"],
                int(record.get("windows_observed") or 0),
                float(record.get("seconds_observed") or 0.0),
                json.dumps(record.get("signature") or {}),
            ),
        )

    # -- profile --------------------------------------------------------------

    _PROFILE_COLUMNS = "assistant_id, user_id, role_section, blocks, motion_fidelity, seconds_observed, updated_at"

    async def get_profile(self, assistant_id: str) -> dict[str, Any] | None:
        """Return the rendered profile row, or ``None``."""
        row = await self._fetchone(
            f"SELECT {self._PROFILE_COLUMNS} FROM avatar_motion_profile WHERE assistant_id = %s", (assistant_id,)
        )
        return self._row(self._PROFILE_COLUMNS, row) if row else None

    async def upsert_profile(self, record: dict[str, Any]) -> None:
        """Insert or update the rendered profile row."""
        import json

        await self._execute(
            """
            INSERT INTO avatar_motion_profile (assistant_id, user_id, role_section, blocks, motion_fidelity, seconds_observed, updated_at)
            VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, now())
            ON CONFLICT (assistant_id) DO UPDATE SET
                role_section = EXCLUDED.role_section,
                blocks = EXCLUDED.blocks,
                motion_fidelity = EXCLUDED.motion_fidelity,
                seconds_observed = EXCLUDED.seconds_observed,
                updated_at = now()
            """,
            (
                record["assistant_id"],
                record["user_id"],
                record.get("role_section") or "",
                json.dumps(record.get("blocks") or {}),
                json.dumps(record.get("motion_fidelity") or {}),
                float(record.get("seconds_observed") or 0.0),
            ),
        )

    # -- everything -----------------------------------------------------------

    async def delete_all_for_avatar(self, assistant_id: str) -> dict[str, int]:
        """Delete everything motion-related for an avatar; return counts per table."""
        counts: dict[str, int] = {}
        for table, name in (
            ("avatar_motion_tracks", "tracks"),
            ("avatar_motion_golden", "golden"),
            ("avatar_motion_basis", "bases"),
            ("avatar_motion_primitives", "primitives"),
            ("avatar_motion_signature", "signatures"),
            ("avatar_motion_profile", "profiles"),
        ):
            counts[name] = int(await self._execute(f"DELETE FROM {table} WHERE assistant_id = %s", (assistant_id,)))
        return counts


_repository: Any | None = None


def set_motion_repository(repository: Any | None) -> None:
    """Publish the process-wide repository (or clear it with ``None``)."""
    global _repository
    _repository = repository


def get_motion_repository() -> Any | None:
    """Return the published repository, or ``None``."""
    return _repository


async def ensure_motion_tables(pool: Any) -> None:
    """Create the motion tables if they do not exist. Best-effort at boot."""
    try:
        await execute_ddl_script(pool, _CREATE_TABLES_SQL)
    except Exception as table_error:  # noqa: BLE001 - non-fatal at startup
        logger.error("Could not ensure the avatar motion tables exist: %s", table_error)
