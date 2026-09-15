"""Storage for the signed-in account's age verification.

One row per account: the date of birth the person submitted and when the
verification was recorded. Listings consult ``is_verified``; the HTTP
response never includes the date of birth.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from typing import Any

from src.anubis.utils.age_verification.policy import (
    DEFAULT_AGE_VERIFICATION_MINIMUM_YEARS,
    as_date,
    verification_is_current,
)
from src.anubis.utils.postgres_ddl import execute_ddl_script

logger = logging.getLogger(__name__)

AGE_VERIFICATION_TABLE_NAME = "age_verification"

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {AGE_VERIFICATION_TABLE_NAME} (
    user_id TEXT PRIMARY KEY,
    date_of_birth DATE NOT NULL,
    source TEXT,
    verified_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def _now() -> datetime:
    return datetime.now(UTC)


def _isoformat(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def verification_public_view(
    row: dict[str, Any] | None,
    *,
    minimum_years: int = DEFAULT_AGE_VERIFICATION_MINIMUM_YEARS,
    on_date: date | None = None,
) -> dict[str, Any]:
    """Return what the browser may read: verified or not, never the date of birth."""
    verified = verification_is_current(
        row, minimum_years=minimum_years, on_date=on_date
    )
    return {
        "verified": verified,
        "verified_at": _isoformat(row.get("verified_at")) if row and verified else None,
        "minimum_years": int(minimum_years),
    }


class InMemoryAgeVerificationRepository:
    """A dictionary-backed repository for tests and the store-less dev server."""

    def __init__(self) -> None:
        """Start empty."""
        self.rows: dict[str, dict[str, Any]] = {}

    async def get_verification(self, user_id: str) -> dict[str, Any] | None:
        """Return the verification row of one account, or None."""
        stored = self.rows.get(str(user_id))
        return dict(stored) if stored else None

    async def set_verification(
        self,
        user_id: str,
        date_of_birth: date,
        source: str | None = None,
    ) -> dict[str, Any]:
        """Record the date of birth that verified this account."""
        row = {
            "user_id": str(user_id),
            "date_of_birth": date_of_birth,
            "source": (source or "account_settings").strip() or "account_settings",
            "verified_at": _now(),
        }
        self.rows[str(user_id)] = row
        return dict(row)

    async def is_verified(
        self,
        user_id: str,
        *,
        minimum_years: int = DEFAULT_AGE_VERIFICATION_MINIMUM_YEARS,
        on_date: date | None = None,
    ) -> bool:
        """Return whether this account currently meets the minimum age."""
        return verification_is_current(
            await self.get_verification(user_id),
            minimum_years=minimum_years,
            on_date=on_date,
        )


class PostgresAgeVerificationRepository:
    """Age verification stored in the application Postgres."""

    def __init__(self, pool: Any) -> None:
        """Bind to the application connection pool."""
        self.pool = pool

    async def _fetchone(self, sql: str, params: tuple = ()) -> tuple | None:
        """Run one query and return the first row."""
        async with self.pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(sql, params)
                return await cursor.fetchone()

    async def _execute(self, sql: str, params: tuple = ()) -> None:
        """Run one statement."""
        async with self.pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(sql, params)

    async def get_verification(self, user_id: str) -> dict[str, Any] | None:
        """Return the verification row of one account, or None."""
        row = await self._fetchone(
            f"""
            SELECT user_id, date_of_birth, source, verified_at
            FROM {AGE_VERIFICATION_TABLE_NAME}
            WHERE user_id = %s;
            """,
            (str(user_id),),
        )
        if row is None:
            return None
        return {
            "user_id": row[0],
            "date_of_birth": as_date(row[1]),
            "source": row[2],
            "verified_at": row[3],
        }

    async def set_verification(
        self,
        user_id: str,
        date_of_birth: date,
        source: str | None = None,
    ) -> dict[str, Any]:
        """Record the date of birth that verified this account."""
        recorded_source = (source or "account_settings").strip() or "account_settings"
        await self._execute(
            f"""
            INSERT INTO {AGE_VERIFICATION_TABLE_NAME}
                (user_id, date_of_birth, source, verified_at)
            VALUES (%s, %s, %s, now())
            ON CONFLICT (user_id) DO UPDATE
                SET date_of_birth = EXCLUDED.date_of_birth,
                    source = EXCLUDED.source,
                    verified_at = now();
            """,
            (str(user_id), date_of_birth, recorded_source),
        )
        stored = await self.get_verification(user_id)
        return stored or {
            "user_id": str(user_id),
            "date_of_birth": date_of_birth,
            "source": recorded_source,
            "verified_at": _now(),
        }

    async def is_verified(
        self,
        user_id: str,
        *,
        minimum_years: int = DEFAULT_AGE_VERIFICATION_MINIMUM_YEARS,
        on_date: date | None = None,
    ) -> bool:
        """Return whether this account currently meets the minimum age."""
        return verification_is_current(
            await self.get_verification(user_id),
            minimum_years=minimum_years,
            on_date=on_date,
        )


_repository: Any | None = None


def set_age_verification_repository(repository: Any | None) -> None:
    """Publish the repository the listings and the routes use."""
    global _repository
    _repository = repository


def get_age_verification_repository() -> Any | None:
    """Return the published repository, or None before the lifespan ran."""
    return _repository


async def ensure_age_verification_table(pool: Any) -> None:
    """Create the age verification table when absent."""
    await execute_ddl_script(pool, _CREATE_TABLE_SQL)
    logger.info("Age verification table is ready.")
