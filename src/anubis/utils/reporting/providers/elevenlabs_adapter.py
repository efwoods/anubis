"""ElevenLabs usage adapter."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


async def fetch_elevenlabs_usage(
    context: Any,
    since: datetime | None = None,
    until: datetime | None = None,
) -> dict[str, Any]:
    """Fetch usage from ElevenLabs.

    Returns:
        {
            "total_cost_usd": float,
            "characters_used": int,
            "period_start": str,
            "period_end": str,
        }
    """
    api_key = getattr(context, "elevenlabs_api_key", None)
    if not api_key:
        logger.warning("ElevenLabs API key not configured.")
        return {
            "total_cost_usd": 0.0,
            "characters_used": 0,
            "error": "no_api_key",
        }
    
    since = since or (datetime.now(UTC) - timedelta(days=30))
    until = until or datetime.now(UTC)
    
    try:
        # ElevenLabs usage API
        # This is a placeholder implementation
        logger.info("ElevenLabs usage fetch not fully implemented; returning stub.")
        
        return {
            "total_cost_usd": 0.0,
            "characters_used": 0,
            "period_start": since.isoformat(),
            "period_end": until.isoformat(),
            "note": "ElevenLabs usage API integration pending",
        }
    except Exception as usage_error:  # noqa: BLE001
        logger.error("Failed to fetch ElevenLabs usage: %s", usage_error)
        return {
            "total_cost_usd": 0.0,
            "characters_used": 0,
            "error": str(usage_error),
        }
