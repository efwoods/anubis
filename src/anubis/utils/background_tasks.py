"""Work a graph node hands off so the reply never waits for the work.

A graph node that awaits a slow side effect (a store write, a model call that
summarizes the conversation) puts that side effect on the reply's critical
path: the outer workflow joins its branches before the avatar runs, so the
avatar starts only once the slowest branch finishes. ``schedule_detached``
runs such a side effect on its own asyncio task instead, and the node returns
at once.

The detached task runs in a fresh ``contextvars.Context``. A task created with
``asyncio.create_task`` otherwise copies the creating task's context, which
inside a graph node carries the run's configuration, stream writer and
callback handlers; the run is usually over before the detached work finishes,
and a model call made under a finished run's callbacks would try to report
into a run that has already ended.

Tasks are held in a module-level set until they finish, because the event
loop keeps only a weak reference to a task and an unreferenced task can be
garbage-collected mid-flight. A process restart loses in-flight tasks; every
caller must be written so that a lost task is acceptable (the learning sweep
re-runs the conversation summary once an account goes idle, for example).
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from typing import Any, Coroutine

logger = logging.getLogger(__name__)

_detached_tasks: set[asyncio.Task[Any]] = set()


def _forget_detached_task(task: asyncio.Task[Any]) -> None:
    """Drop the finished task from the held set and log a failure the task raised."""
    _detached_tasks.discard(task)
    if task.cancelled():
        return
    task_exception = task.exception()
    if task_exception is not None:
        logger.warning(
            "Detached task %s failed: %s", task.get_name(), task_exception
        )


def schedule_detached(
    coroutine: Coroutine[Any, Any, Any], task_name: str
) -> asyncio.Task[Any]:
    """Run ``coroutine`` on its own task, outside the caller's context, and return the task.

    The caller does not await the returned task. A failure inside the task is
    logged and never propagates to the caller.
    """
    task = asyncio.create_task(
        coroutine, name=task_name, context=contextvars.Context()
    )
    _detached_tasks.add(task)
    task.add_done_callback(_forget_detached_task)
    return task


async def drain_detached_tasks() -> None:
    """Wait until every detached task has finished (tests and graceful shutdown)."""
    while _detached_tasks:
        await asyncio.gather(*list(_detached_tasks), return_exceptions=True)
