"""The avatars an account has taken off the avatar-selection carousel.

Hiding an avatar from the carousel is an account setting, so the hidden
avatars follow the account into every browser the account signs in on. Only
the hidden assistant ids are stored, one row per hidden avatar, in Postgres
rather than the LangGraph store, so nothing here is embedded.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

CREATE_HIDDEN_CAROUSEL_AVATARS_SQL = """
CREATE TABLE IF NOT EXISTS user_hidden_carousel_avatars (
    user_id TEXT NOT NULL,
    assistant_id TEXT NOT NULL,
    hidden_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, assistant_id)
)
"""

# One account hides a handful of avatars; the cap only stops an oversized
# request body from writing an unbounded number of rows.
MAXIMUM_HIDDEN_CAROUSEL_AVATARS = 1000


async def ensure_hidden_carousel_avatar_table(pool: Any) -> None:
    """Create the user_hidden_carousel_avatars table if missing. Best-effort at boot."""
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(CREATE_HIDDEN_CAROUSEL_AVATARS_SQL)
    except Exception as table_error:  # noqa: BLE001 - non-fatal at startup
        logger.error(
            "Could not ensure the user_hidden_carousel_avatars table exists: %s",
            table_error,
        )


def normalize_hidden_assistant_ids(assistant_ids: Any) -> list[str]:
    """Return the distinct, non-empty assistant ids in request order, capped."""
    if not isinstance(assistant_ids, list):
        raise ValueError("assistant_ids must be a list of assistant ids.")
    normalized_assistant_ids: list[str] = []
    for assistant_id in assistant_ids:
        trimmed_assistant_id = str(assistant_id or "").strip()
        if (
            trimmed_assistant_id
            and trimmed_assistant_id not in normalized_assistant_ids
        ):
            normalized_assistant_ids.append(trimmed_assistant_id)
    if len(normalized_assistant_ids) > MAXIMUM_HIDDEN_CAROUSEL_AVATARS:
        raise ValueError(
            f"At most {MAXIMUM_HIDDEN_CAROUSEL_AVATARS} avatars can be hidden."
        )
    return normalized_assistant_ids


async def list_hidden_carousel_avatar_ids(pool: Any, user_id: str) -> list[str]:
    """Return the assistant ids the account has hidden, oldest hide first."""
    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "SELECT assistant_id FROM user_hidden_carousel_avatars "
                "WHERE user_id = %s ORDER BY hidden_at, assistant_id",
                (user_id,),
            )
            rows = await cursor.fetchall()
    return [str(row[0]) for row in rows]


async def replace_hidden_carousel_avatar_ids(
    pool: Any, user_id: str, assistant_ids: list[str]
) -> list[str]:
    """Make the account's hidden avatars exactly ``assistant_ids``.

    Avatars that stay hidden keep their original ``hidden_at``.
    """
    async with pool.connection() as connection:
        async with connection.transaction():
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "DELETE FROM user_hidden_carousel_avatars "
                    "WHERE user_id = %s AND NOT (assistant_id = ANY(%s))",
                    (user_id, assistant_ids),
                )
                for assistant_id in assistant_ids:
                    await cursor.execute(
                        "INSERT INTO user_hidden_carousel_avatars "
                        "(user_id, assistant_id) VALUES (%s, %s) "
                        "ON CONFLICT (user_id, assistant_id) DO NOTHING",
                        (user_id, assistant_id),
                    )
    return await list_hidden_carousel_avatar_ids(pool, user_id)
