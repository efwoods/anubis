"""Shared primitives for deleting an avatar and everything keyed to that avatar.

Why this module exists: ``/delete_user`` (``src/security/auth.py``) and
``/delete_avatar`` (``src/api/webapp.py``) each implemented avatar removal
independently, and the two implementations diverged. ``/delete_user`` deleted
the LangGraph assistant record but purged store rows keyed only on the
departing user, so every row keyed on the assistant alone — the
``(assistant_id, "creator_id")`` marker written by ``create_avatar``, the
style and ground-truth artifacts, and the identity and memory rows belonging to
other people who had chatted with that avatar — was left behind with no owner
able to reach the rows again. Routing both endpoints through
``purge_avatar_data`` keeps a single definition of "delete everything for this
avatar".

Three deletion mechanisms that look like they should work do not, and this
module compensates for all three:

1. ``assistants.search`` in the LangGraph SDK defaults to ``limit=10``. Callers
   that omit a limit silently enumerate only the first ten avatars, so an
   account with more avatars keeps the remainder forever once the Auth0 user is
   gone. ``search_all_avatars_for_user`` pages until the results run out.
2. ``assistants.delete(delete_threads=True)`` removes threads whose
   ``metadata.assistant_id`` matches. Anubis writes that identifier one level
   deeper, as ``metadata.thread_metadata.assistant_id`` (see every
   ``threads.update`` call in ``src/api/webapp.py``), so the SDK filter matches
   nothing and threads plus their checkpoints survive.
   ``delete_threads_for_assistant`` deletes on the nested path that Anubis
   actually writes.
3. No foreign key points from ``thread``, ``checkpoints``, ``checkpoint_blobs``,
   ``checkpoint_writes`` or ``store`` back to ``assistant``, so the database
   cascades nothing when an assistant row is removed. Every child row is
   deleted explicitly here.

Store namespaces are tuples that the LangGraph Postgres store joins with dots
into the ``store.prefix`` column. The same avatar appears in several positions
depending on the namespace shape — ``(user_id, assistant_id, kind)``,
``(assistant_id, user_id, "identity")``, ``(assistant_id, "creator_id")`` — so
membership is tested per dot-separated segment rather than with positional
``LIKE`` patterns. The segment comparison is also whitespace-tolerant, because
some ``creator_id`` rows in existing databases were written with a trailing
space in the assistant identifier and no positional pattern can match those.
"""

import logging
from typing import Any

from src.anubis.utils.store_cache import invalidate_store_cache_for_assistant

logger = logging.getLogger(__name__)

# Page size for paginated assistant searches. The LangGraph SDK caps nothing on
# its side; this is only how many avatars are fetched per round trip.
AVATAR_SEARCH_PAGE_SIZE = 100

# Matches a store row when any dot-separated segment of its namespace prefix
# equals the supplied identifier. Subsumes the four positional patterns the old
# code used (prefix = id, "id.%", "%.id.%", "%.id") and additionally matches
# segments carrying stray surrounding whitespace.
SQL_DELETE_STORE_ROWS_BY_NAMESPACE_SEGMENT = """
DELETE FROM store
 WHERE EXISTS (
     SELECT 1
       FROM unnest(string_to_array(prefix, '.')) AS namespace_segment
      WHERE trim(namespace_segment) = %s
 );
"""

# store_vectors carries ON DELETE CASCADE from store(prefix, key), so deleting
# the store rows above already removed the matching embeddings. No separate
# statement against store_vectors is needed or wanted.

SQL_SELECT_THREAD_IDS_FOR_ASSISTANT = """
SELECT thread_id
  FROM thread
 WHERE metadata->'thread_metadata'->>'assistant_id' = %s;
"""

SQL_DELETE_THREADS_FOR_ASSISTANT = """
DELETE FROM thread
 WHERE metadata->'thread_metadata'->>'assistant_id' = %s;
"""

# Checkpoint tables key on thread_id with no foreign key to thread, so each one
# is cleared explicitly for the threads being removed.
SQL_DELETE_CHECKPOINT_TABLES_FOR_THREADS = (
    "DELETE FROM checkpoint_writes WHERE thread_id = ANY(%s);",
    "DELETE FROM checkpoint_blobs WHERE thread_id = ANY(%s);",
    "DELETE FROM checkpoints WHERE thread_id = ANY(%s);",
    "DELETE FROM run WHERE thread_id = ANY(%s);",
)

# Ownership read that does NOT go through the LangGraph SDK. Needed because the
# SDK authenticates every call with the caller's API key, and the
# ``@auth.authenticate`` handler rejects an account whose email is unverified —
# so an unverified account deleting itself cannot enumerate avatars the way a
# verified account does. ``create_avatar`` writes the owner into
# ``assistant.metadata.user_id`` (the raw Auth0 identity id, no "auth0|"
# prefix), which is the same value the SDK search filters on.
SQL_SELECT_ASSISTANT_IDS_FOR_USER = """
SELECT assistant_id
  FROM assistant
 WHERE metadata->>'user_id' = %s;
"""

SQL_DELETE_API_METRICS_FOR_ASSISTANT = (
    "DELETE FROM api_metrics WHERE assistant_id = %s;"
)

SQL_DELETE_API_METRICS_FOR_USER = "DELETE FROM api_metrics WHERE user_id = %s;"


async def search_all_avatars_for_user(
    langgraph_sdk_client: Any,
    user_id: str,
    headers: dict | None = None,
) -> list[dict]:
    """Return every avatar owned by the user, paging past the SDK's default limit.

    ``assistants.search`` defaults to ``limit=10``. Omitting an explicit limit
    is what allowed accounts with more than ten avatars to survive account
    deletion, so every ownership enumeration must page. No ``graph_id`` filter
    is applied: an avatar created on any graph still belongs to the user and
    must be found.
    """
    all_avatars: list[dict] = []
    offset = 0
    while True:
        page = await langgraph_sdk_client.assistants.search(
            metadata={"user_id": user_id},
            limit=AVATAR_SEARCH_PAGE_SIZE,
            offset=offset,
            headers=headers,
        )
        if not page:
            break
        all_avatars.extend(page)
        if len(page) < AVATAR_SEARCH_PAGE_SIZE:
            break
        offset += AVATAR_SEARCH_PAGE_SIZE
    return all_avatars


async def select_assistant_ids_for_user(pool: Any, user_id: str) -> list[str]:
    """Return the avatar identifiers owned by the user, read straight from Postgres.

    The SDK-based ``search_all_avatars_for_user`` is the normal enumeration and
    stays the one used for verified accounts, because the SDK is also what
    deletes an avatar. This function exists for the one caller that cannot use
    the SDK at all — account deletion for an unverified email — where the only
    question is whether any avatar exists, not how to remove one.
    """
    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(SQL_SELECT_ASSISTANT_IDS_FOR_USER, (user_id,))
            return [str(row[0]) for row in await cursor.fetchall()]


async def delete_store_rows_for_assistant(pool: Any, assistant_id: str) -> int:
    """Delete every store row whose namespace mentions the avatar. Returns the row count."""
    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                SQL_DELETE_STORE_ROWS_BY_NAMESPACE_SEGMENT, (assistant_id,)
            )
            return cursor.rowcount


async def delete_store_rows_for_user(pool: Any, user_id: str) -> int:
    """Delete every store row whose namespace mentions the user. Returns the row count.

    Covers the departing user's rows on avatars they do not own — episodic
    memories, quotes and the ``(assistant_id, user_id, "identity")`` facts that
    other people's avatars learned about this user — which no avatar-scoped
    purge would reach.
    """
    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(SQL_DELETE_STORE_ROWS_BY_NAMESPACE_SEGMENT, (user_id,))
            return cursor.rowcount


async def delete_threads_for_assistant(pool: Any, assistant_id: str) -> int:
    """Delete the avatar's threads and every checkpoint row belonging to them.

    Runs in one explicit transaction so a partial failure cannot leave
    checkpoints behind pointing at threads that no longer exist. Children are
    removed before the threads themselves.
    """
    async with pool.connection() as connection:
        async with connection.transaction():
            async with connection.cursor() as cursor:
                await cursor.execute(
                    SQL_SELECT_THREAD_IDS_FOR_ASSISTANT, (assistant_id,)
                )
                thread_identifiers = [row[0] for row in await cursor.fetchall()]
                if not thread_identifiers:
                    return 0
                for statement in SQL_DELETE_CHECKPOINT_TABLES_FOR_THREADS:
                    await cursor.execute(statement, (thread_identifiers,))
                await cursor.execute(SQL_DELETE_THREADS_FOR_ASSISTANT, (assistant_id,))
                return len(thread_identifiers)


async def delete_api_metrics_for_assistant(pool: Any, assistant_id: str) -> int:
    """Delete billing and usage metric rows recorded against the avatar."""
    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(SQL_DELETE_API_METRICS_FOR_ASSISTANT, (assistant_id,))
            return cursor.rowcount


async def delete_api_metrics_for_user(pool: Any, user_id: str) -> int:
    """Delete billing and usage metric rows recorded against the user."""
    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(SQL_DELETE_API_METRICS_FOR_USER, (user_id,))
            return cursor.rowcount


async def purge_avatar_data(
    pool: Any,
    langgraph_sdk_client: Any,
    assistant_id: str,
    headers: dict | None = None,
) -> dict:
    """Delete an avatar and every row keyed to that avatar. Returns per-table counts.

    The LangGraph assistant record is deleted **last**, on purpose. Deleting the
    assistant first — which is what the previous ``/delete_user`` did — means a
    failure in any later step leaves data behind that no avatar-scoped query can
    ever find again, because the assistant identifier is no longer discoverable
    through any listing endpoint. Deleting the record last keeps the avatar
    visible and the purge retryable until every dependent row is gone.
    """
    deleted_thread_count = await delete_threads_for_assistant(pool, assistant_id)
    deleted_store_row_count = await delete_store_rows_for_assistant(pool, assistant_id)
    deleted_metric_row_count = await delete_api_metrics_for_assistant(
        pool, assistant_id
    )

    # The raw SQL above bypassed the store client, so the process-local
    # read-through cache still holds this avatar's reference image and style
    # profile and would keep serving both for up to STORE_CACHE_MAX_AGE_SECONDS.
    invalidate_store_cache_for_assistant(assistant_id)

    # delete_threads=True is retained only as a secondary sweep for any thread
    # created directly through the SDK with a top-level metadata.assistant_id.
    # Threads created by this application are already gone, deleted above on the
    # nested metadata path that the SDK filter cannot see.
    await langgraph_sdk_client.assistants.delete(
        assistant_id=assistant_id, delete_threads=True, headers=headers
    )

    counts = {
        "threads": deleted_thread_count,
        "store_rows": deleted_store_row_count,
        "api_metric_rows": deleted_metric_row_count,
    }
    logger.info("Purged avatar %s: %s", assistant_id, counts)
    return counts
