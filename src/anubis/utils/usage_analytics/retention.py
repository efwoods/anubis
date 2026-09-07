"""Forget old usage analytics rows on a schedule.

Consent covers a bounded window: ``USAGE_ANALYTICS_RETENTION_DAYS`` (default
90). ``purge_forever`` runs as a lifespan task, deleting events and captures
older than that window once per ``interval_seconds`` and never raising.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_RETENTION_DAYS = 90
DEFAULT_PURGE_INTERVAL_SECONDS = 6 * 60 * 60


async def purge_once(repository: Any, retention_days: int) -> dict[str, int]:
    """Delete rows older than ``retention_days``; a zero or negative window keeps everything."""
    if repository is None or int(retention_days) <= 0:
        return {"events": 0, "screenshots": 0}
    removed = await repository.purge_older_than(int(retention_days))
    if any(removed.values()):
        logger.info("Usage analytics purge removed %s", removed)
    return removed


async def purge_forever(
    repository: Any,
    retention_days: int,
    *,
    interval_seconds: float = DEFAULT_PURGE_INTERVAL_SECONDS,
) -> None:
    """Run ``purge_once`` on a fixed interval until cancelled."""
    while True:
        try:
            await purge_once(repository, retention_days)
        except asyncio.CancelledError:
            raise
        except Exception as purge_error:  # noqa: BLE001 - the loop must survive
            logger.warning("Usage analytics purge failed: %s", purge_error)
        await asyncio.sleep(max(60.0, float(interval_seconds)))
