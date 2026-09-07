"""Record every tool call the avatar makes, so feature usage can be measured.

``api_metrics`` (``src/anubis/utils/billing/metering.py``) captures one row per
billed model inference, which says how much a conversation cost but not which
capabilities the avatar reached for. The ``tool_calls`` table fills that gap:
one row per tool invocation with the tool name, the outcome, and the wall-clock
duration, keyed by user, avatar, and thread. The platform-metrics queries
(``platform_metrics.feature_usage_per_avatar``) read this table to answer
"which features does each personal avatar use most and least".

Recording is fire-and-forget: ``record_tool_call`` schedules the insert on the
running event loop and swallows every error, because a metrics write must never
slow down or break a user's turn. The graph pairs ``ToolCallTimer.start`` with
``on_tool_start`` and ``ToolCallTimer.finish`` with ``on_tool_end`` to measure
the duration of each call by the LangChain run id.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from uuid import uuid4

from src.anubis.utils.postgres_ddl import execute_ddl_script

logger = logging.getLogger(__name__)

TOOL_CALLS_TABLE_NAME = "tool_calls"

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TOOL_CALLS_TABLE_NAME} (
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_id TEXT,
    assistant_id TEXT,
    thread_id TEXT,
    tool_name TEXT NOT NULL,
    status TEXT NOT NULL,
    duration_ms DOUBLE PRECISION NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS tool_calls_user_created_idx
    ON {TOOL_CALLS_TABLE_NAME} (user_id, created_at);
CREATE INDEX IF NOT EXISTS tool_calls_assistant_created_idx
    ON {TOOL_CALLS_TABLE_NAME} (assistant_id, created_at);
"""

_INSERT_SQL = f"""
INSERT INTO {TOOL_CALLS_TABLE_NAME}
    (id, user_id, assistant_id, thread_id, tool_name, status, duration_ms)
VALUES (%s, %s, %s, %s, %s, %s, %s);
"""

_PURGE_SQL = f"""
DELETE FROM {TOOL_CALLS_TABLE_NAME}
WHERE created_at < now() - (%s * INTERVAL '1 day');
"""

STATUS_SUCCESS = "success"
STATUS_ERROR = "error"

_pool: Any | None = None
# Strong references to in-flight insert tasks; the event loop only keeps weak
# references, so an unreferenced task can be garbage-collected mid-flight.
_pending_tasks: set[asyncio.Task[Any]] = set()


def set_tool_call_pool(pool: Any | None) -> None:
    """Publish the psycopg pool the recorder inserts through (``None`` disables it)."""
    global _pool
    _pool = pool


def get_tool_call_pool() -> Any | None:
    """Return the published pool, or ``None`` when recording is disabled."""
    return _pool


async def ensure_tool_calls_table(pool: Any) -> None:
    """Create the ``tool_calls`` table and indexes if absent. Best-effort at boot."""
    try:
        await execute_ddl_script(pool, _CREATE_TABLE_SQL)
    except Exception as table_error:  # noqa: BLE001 - non-fatal at startup
        logger.error("Could not ensure the tool_calls table exists: %s", table_error)


async def _insert_tool_call(
    pool: Any,
    *,
    user_id: str | None,
    assistant_id: str | None,
    thread_id: str | None,
    tool_name: str,
    status: str,
    duration_ms: float,
) -> None:
    """Insert one row; every failure is logged at debug level and dropped."""
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    _INSERT_SQL,
                    (
                        str(uuid4()),
                        user_id,
                        assistant_id,
                        thread_id,
                        tool_name,
                        status,
                        float(duration_ms),
                    ),
                )
    except Exception:  # noqa: BLE001 - metrics never break a turn
        logger.debug("tool_calls insert failed", exc_info=True)


def record_tool_call(
    *,
    user_id: str | None,
    assistant_id: str | None,
    thread_id: str | None,
    tool_name: str,
    status: str,
    duration_ms: float,
) -> None:
    """Schedule a fire-and-forget insert of one tool call.

    Returns immediately. When no pool is published, no event loop is running,
    or the tool name is empty, nothing is recorded and no error is raised.
    """
    pool = _pool
    if pool is None or not tool_name:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    try:
        task = loop.create_task(
            _insert_tool_call(
                pool,
                user_id=user_id,
                assistant_id=assistant_id,
                thread_id=thread_id,
                tool_name=str(tool_name),
                status=str(status or STATUS_SUCCESS),
                duration_ms=float(duration_ms or 0.0),
            )
        )
    except Exception:  # noqa: BLE001 - scheduling must never raise
        logger.debug("tool_calls task scheduling failed", exc_info=True)
        return
    _pending_tasks.add(task)
    task.add_done_callback(_pending_tasks.discard)


class ToolCallTimer:
    """Measure the wall-clock duration of tool calls keyed by their run id.

    The graph calls ``start(run_id)`` on ``on_tool_start`` and
    ``finish(run_id)`` on ``on_tool_end``; ``finish`` returns the elapsed
    milliseconds (``0.0`` for an unknown run id) and forgets the run.
    """

    def __init__(self) -> None:
        """Start with no runs in flight."""
        self._started_at: dict[str, float] = {}

    def start(self, run_id: str | None) -> None:
        """Remember when the run identified by ``run_id`` began."""
        if run_id is None:
            return
        self._started_at[str(run_id)] = time.perf_counter()

    def finish(self, run_id: str | None) -> float:
        """Return the elapsed milliseconds for ``run_id`` and forget the run."""
        if run_id is None:
            return 0.0
        started_at = self._started_at.pop(str(run_id), None)
        if started_at is None:
            return 0.0
        return (time.perf_counter() - started_at) * 1000.0

    def in_flight(self) -> int:
        """How many runs have started and not finished."""
        return len(self._started_at)


async def purge_tool_calls_older_than(pool: Any, days: int) -> int:
    """Delete rows older than ``days`` days; return how many were removed."""
    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(_PURGE_SQL, (int(days),))
            return int(cursor.rowcount or 0)
