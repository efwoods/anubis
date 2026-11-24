"""Read the owner's conversations with one avatar straight out of the checkpoints.

Screen observations live as hidden messages on whichever thread was open when
the snapshot was taken, so a day's screen activity is spread across several
conversations. This module finds the threads touched in a period and reads
each thread's messages through the in-process graph — the same read
``load_thread_messages`` gives the learning sweeper. Nothing is copied.

Threads are found through the LangGraph API's own ``thread`` table, filtered on
the ``thread_metadata`` the message endpoint stamps on every thread (the same
filter ``GET /conversations`` passes to ``threads.search``) and on
``updated_at``, because a thread last touched before the period started cannot
hold an observation captured inside the period.

The in-process graph is published by the API lifespan. Under ``langgraph dev``
or a unit test without the lifespan, nothing is published and the screen
source reports that the screen history is unavailable.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

_activity_graph: Any | None = None

THREAD_READ_CONCURRENCY = 4
"""Threads read at once. Each read is one checkpoint load on the shared pool."""


def register_activity_graph(graph: Any) -> None:
    """Publish the compiled outer graph whose checkpointer holds every thread."""
    global _activity_graph
    _activity_graph = graph


def get_activity_graph() -> Any | None:
    """Return the published graph, or ``None`` when the lifespan has not run."""
    return _activity_graph


THREADS_TOUCHED_SINCE_SQL = """
SELECT thread_id::text
FROM thread
WHERE metadata @> %(metadata_filter)s::jsonb
  AND GREATEST(updated_at, state_updated_at) >= %(period_start)s
  AND created_at < %(period_end)s
ORDER BY GREATEST(updated_at, state_updated_at) DESC
LIMIT %(maximum_threads)s
"""


async def threads_touched_in_period(
    pool: Any,
    user_id: str,
    assistant_id: str,
    period_start: datetime,
    period_end: datetime,
    maximum_threads: int,
) -> list[str]:
    """Return the ids of the owner's threads with this avatar that changed in the period."""
    import json

    if pool is None or not user_id or not assistant_id:
        return []
    metadata_filter = json.dumps(
        {"thread_metadata": {"user_id": user_id, "assistant_id": assistant_id}}
    )
    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                THREADS_TOUCHED_SINCE_SQL,
                {
                    "metadata_filter": metadata_filter,
                    "period_start": period_start,
                    "period_end": period_end,
                    "maximum_threads": int(maximum_threads),
                },
            )
            rows = await cursor.fetchall()
    thread_ids: list[str] = []
    for row in rows:
        thread_id = row[0] if isinstance(row, (tuple, list)) else row.get("thread_id")
        if thread_id:
            thread_ids.append(str(thread_id))
    return thread_ids


async def read_thread_messages(
    graph: Any, thread_ids: list[str]
) -> dict[str, list[Any]]:
    """Load every listed thread's checkpointed messages; an unreadable thread is skipped."""
    from src.anubis.utils.learning.bulk_learning import load_thread_messages

    semaphore = asyncio.Semaphore(THREAD_READ_CONCURRENCY)
    messages_by_thread: dict[str, list[Any]] = {}

    async def read_one(thread_id: str) -> None:
        async with semaphore:
            try:
                messages_by_thread[thread_id] = await load_thread_messages(
                    graph, thread_id
                )
            except Exception:  # noqa: BLE001 - one unreadable thread never fails the review
                logger.debug("Could not read thread %s", thread_id, exc_info=True)

    await asyncio.gather(*(read_one(thread_id) for thread_id in thread_ids))
    return messages_by_thread
