"""xAI (Grok) billing adapter."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


async def fetch_xai_usage(
    context: Any,
    since: datetime | None = None,
    until: datetime | None = None,
) -> dict[str, Any]:
    """Fetch usage from xAI billing.

    Returns:
        {
            "total_cost_usd": float,
            "images_generated": int,
            "videos_generated": int,
            "period_start": str,
            "period_end": str,
        }
    """
    api_key = getattr(context, "xai_api_key", None)
    if not api_key:
        logger.warning("xAI API key not configured.")
        return {
            "total_cost_usd": 0.0,
            "images_generated": 0,
            "videos_generated": 0,
            "error": "no_api_key",
        }
    
    since = since or (datetime.now(UTC) - timedelta(days=30))
    until = until or datetime.now(UTC)
    
    try:
        # xAI billing API
        # This is a placeholder implementation
        logger.info("xAI usage fetch not fully implemented; returning stub.")
        
        return {
            "total_cost_usd": 0.0,
            "images_generated": 0,
            "videos_generated": 0,
            "period_start": since.isoformat(),
            "period_end": until.isoformat(),
            "note": "xAI billing API integration pending",
        }
    except Exception as usage_error:  # noqa: BLE001
        logger.error("Failed to fetch xAI usage: %s", usage_error)
        return {
            "total_cost_usd": 0.0,
            "images_generated": 0,
            "videos_generated": 0,
            "error": str(usage_error),
        }
