"""OpenAI organization usage adapter."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


async def fetch_openai_usage(
    context: Any,
    since: datetime | None = None,
    until: datetime | None = None,
) -> dict[str, Any]:
    """Fetch usage from OpenAI organization.

    Returns:
        {
            "total_cost_usd": float,
            "total_tokens": int,
            "by_model": {...},
            "period_start": str,
            "period_end": str,
        }
    """
    api_key = getattr(context, "openai_api_key", None)
    if not api_key:
        logger.warning("OpenAI API key not configured.")
        return {
            "total_cost_usd": 0.0,
            "total_tokens": 0,
            "by_model": {},
            "error": "no_api_key",
        }
    
    since = since or (datetime.now(UTC) - timedelta(days=30))
    until = until or datetime.now(UTC)
    
    try:
        # OpenAI usage API endpoint
        # https://platform.openai.com/docs/api-reference/usage
        import openai  # noqa: PLC0415
        
        client = openai.AsyncOpenAI(api_key=api_key)
        
        # Note: OpenAI's usage API returns daily aggregates
        # This is a placeholder implementation
        logger.info("OpenAI usage fetch not fully implemented; returning stub.")
        
        return {
            "total_cost_usd": 0.0,
            "total_tokens": 0,
            "by_model": {},
            "period_start": since.isoformat(),
            "period_end": until.isoformat(),
            "note": "OpenAI usage API integration pending",
        }
    except Exception as usage_error:  # noqa: BLE001
        logger.error("Failed to fetch OpenAI usage: %s", usage_error)
        return {
            "total_cost_usd": 0.0,
            "total_tokens": 0,
            "by_model": {},
            "error": str(usage_error),
        }
