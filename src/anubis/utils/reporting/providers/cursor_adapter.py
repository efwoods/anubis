"""Cursor spending/usage adapter."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


async def fetch_cursor_usage(
    context: Any,
    since: datetime | None = None,
    until: datetime | None = None,
) -> dict[str, Any]:
    """Fetch Cursor spending/usage.

    Returns:
        {
            "total_cost_usd": float,
            "period_start": str,
            "period_end": str,
        }
    """
    # Cursor usage would be tracked via Cursor's own billing/usage APIs
    # This is a placeholder
    
    since = since or (datetime.now(UTC) - timedelta(days=30))
    until = until or datetime.now(UTC)
    
    try:
        logger.info("Cursor usage fetch not fully implemented; returning stub.")
        
        return {
            "total_cost_usd": 0.0,
            "period_start": since.isoformat(),
            "period_end": until.isoformat(),
            "note": "Cursor usage API integration pending",
        }
    except Exception as usage_error:  # noqa: BLE001
        logger.error("Failed to fetch Cursor usage: %s", usage_error)
        return {
            "total_cost_usd": 0.0,
            "error": str(usage_error),
        }
