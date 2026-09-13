"""Cost metrics computation from api_metrics and billing data.

Computes cost per avatar, average cost per message, average cost per conversation,
and cost per new user from real telemetry stored in the api_metrics table.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


async def compute_cost_per_avatar(
    pool: Any,
    since: datetime | None = None,
    until: datetime | None = None,
) -> dict[str, Any]:
    """Compute cost per avatar from api_metrics.

    Returns:
        {
            "total_avatars": int,
            "total_cost_usd": float,
            "cost_per_avatar_usd": float,
            "by_avatar": [
                {"assistant_id": str, "assistant_name": str, "cost_usd": float, "tokens": int},
                ...
            ]
        }
    """
    if pool is None:
        return {
            "total_avatars": 0,
            "total_cost_usd": 0.0,
            "cost_per_avatar_usd": 0.0,
            "by_avatar": [],
        }
    
    since = since or (datetime.now(UTC) - timedelta(days=30))
    until = until or datetime.now(UTC)
    
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                # Cost per avatar
                await cursor.execute(
                    """
                    SELECT
                        metrics.assistant_id,
                        assistant.name AS assistant_name,
                        ROUND(SUM(metrics.cost_usd)::numeric, 4) AS cost_usd,
                        SUM(metrics.total_tokens) AS tokens
                    FROM api_metrics AS metrics
                    LEFT JOIN assistant ON assistant.assistant_id::text = metrics.assistant_id
                    WHERE metrics.created_at >= %s AND metrics.created_at < %s
                      AND metrics.assistant_id IS NOT NULL
                    GROUP BY metrics.assistant_id, assistant.name
                    ORDER BY cost_usd DESC;
                    """,
                    (since, until),
                )
                rows = await cursor.fetchall()
                
                by_avatar = [
                    {
                        "assistant_id": row[0],
                        "assistant_name": row[1] or "Unknown",
                        "cost_usd": float(row[2] or 0.0),
                        "tokens": int(row[3] or 0),
                    }
                    for row in rows
                ]
                
                total_avatars = len(by_avatar)
                total_cost = sum(item["cost_usd"] for item in by_avatar)
                avg_cost = total_cost / total_avatars if total_avatars > 0 else 0.0
                
                return {
                    "total_avatars": total_avatars,
                    "total_cost_usd": round(total_cost, 4),
                    "cost_per_avatar_usd": round(avg_cost, 4),
                    "by_avatar": by_avatar,
                }
    except Exception as cost_error:  # noqa: BLE001
        logger.error("Failed to compute cost per avatar: %s", cost_error)
        return {
            "total_avatars": 0,
            "total_cost_usd": 0.0,
            "cost_per_avatar_usd": 0.0,
            "by_avatar": [],
        }


async def compute_average_cost_per_message(
    pool: Any,
    since: datetime | None = None,
    until: datetime | None = None,
) -> dict[str, Any]:
    """Compute average cost per message from api_metrics.

    Returns:
        {
            "total_messages": int,
            "total_cost_usd": float,
            "avg_cost_per_message_usd": float,
        }
    """
    if pool is None:
        return {
            "total_messages": 0,
            "total_cost_usd": 0.0,
            "avg_cost_per_message_usd": 0.0,
        }
    
    since = since or (datetime.now(UTC) - timedelta(days=30))
    until = until or datetime.now(UTC)
    
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT
                        COUNT(*) AS messages,
                        ROUND(SUM(cost_usd)::numeric, 4) AS total_cost_usd
                    FROM api_metrics
                    WHERE inference_type IN ('message', 'adapter_inference')
                      AND created_at >= %s AND created_at < %s;
                    """,
                    (since, until),
                )
                row = await cursor.fetchone()
                
                messages = int(row[0] or 0)
                total_cost = float(row[1] or 0.0)
                avg_cost = total_cost / messages if messages > 0 else 0.0
                
                return {
                    "total_messages": messages,
                    "total_cost_usd": total_cost,
                    "avg_cost_per_message_usd": round(avg_cost, 4),
                }
    except Exception as cost_error:  # noqa: BLE001
        logger.error("Failed to compute average cost per message: %s", cost_error)
        return {
            "total_messages": 0,
            "total_cost_usd": 0.0,
            "avg_cost_per_message_usd": 0.0,
        }


async def compute_average_cost_per_conversation(
    pool: Any,
    since: datetime | None = None,
    until: datetime | None = None,
) -> dict[str, Any]:
    """Compute average cost per conversation from api_metrics.

    Returns:
        {
            "total_conversations": int,
            "total_cost_usd": float,
            "avg_cost_per_conversation_usd": float,
        }
    """
    if pool is None:
        return {
            "total_conversations": 0,
            "total_cost_usd": 0.0,
            "avg_cost_per_conversation_usd": 0.0,
        }
    
    since = since or (datetime.now(UTC) - timedelta(days=30))
    until = until or datetime.now(UTC)
    
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT
                        COUNT(DISTINCT thread_id) AS conversations,
                        ROUND(SUM(cost_usd)::numeric, 4) AS total_cost_usd
                    FROM api_metrics
                    WHERE inference_type IN ('message', 'adapter_inference')
                      AND created_at >= %s AND created_at < %s
                      AND thread_id IS NOT NULL;
                    """,
                    (since, until),
                )
                row = await cursor.fetchone()
                
                conversations = int(row[0] or 0)
                total_cost = float(row[1] or 0.0)
                avg_cost = total_cost / conversations if conversations > 0 else 0.0
                
                return {
                    "total_conversations": conversations,
                    "total_cost_usd": total_cost,
                    "avg_cost_per_conversation_usd": round(avg_cost, 4),
                }
    except Exception as cost_error:  # noqa: BLE001
        logger.error("Failed to compute average cost per conversation: %s", cost_error)
        return {
            "total_conversations": 0,
            "total_cost_usd": 0.0,
            "avg_cost_per_conversation_usd": 0.0,
        }


async def compute_cost_per_new_user(
    pool: Any,
    since: datetime | None = None,
    until: datetime | None = None,
) -> dict[str, Any]:
    """Compute cost per new user from api_metrics.

    Assumes each new user creates one personal avatar on signup.

    Returns:
        {
            "new_users": int,
            "total_cost_usd": float,
            "cost_per_new_user_usd": float,
        }
    """
    if pool is None:
        return {
            "new_users": 0,
            "total_cost_usd": 0.0,
            "cost_per_new_user_usd": 0.0,
        }
    
    since = since or (datetime.now(UTC) - timedelta(days=30))
    until = until or datetime.now(UTC)
    
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                # Find users who first appeared in the period
                await cursor.execute(
                    """
                    WITH first_seen AS (
                        SELECT user_id, MIN(created_at) AS first_seen_at
                        FROM api_metrics
                        WHERE user_id IS NOT NULL
                        GROUP BY user_id
                    )
                    SELECT COUNT(*) AS new_users
                    FROM first_seen
                    WHERE first_seen_at >= %s AND first_seen_at < %s;
                    """,
                    (since, until),
                )
                row = await cursor.fetchone()
                new_users = int(row[0] or 0)
                
                # Get cost for those new users (all their activity)
                await cursor.execute(
                    """
                    WITH first_seen AS (
                        SELECT user_id, MIN(created_at) AS first_seen_at
                        FROM api_metrics
                        WHERE user_id IS NOT NULL
                        GROUP BY user_id
                    ),
                    new_user_ids AS (
                        SELECT user_id
                        FROM first_seen
                        WHERE first_seen_at >= %s AND first_seen_at < %s
                    )
                    SELECT ROUND(SUM(cost_usd)::numeric, 4) AS total_cost_usd
                    FROM api_metrics
                    WHERE user_id IN (SELECT user_id FROM new_user_ids);
                    """,
                    (since, until),
                )
                row = await cursor.fetchone()
                total_cost = float(row[0] or 0.0)
                
                avg_cost = total_cost / new_users if new_users > 0 else 0.0
                
                return {
                    "new_users": new_users,
                    "total_cost_usd": total_cost,
                    "cost_per_new_user_usd": round(avg_cost, 4),
                }
    except Exception as cost_error:  # noqa: BLE001
        logger.error("Failed to compute cost per new user: %s", cost_error)
        return {
            "new_users": 0,
            "total_cost_usd": 0.0,
            "cost_per_new_user_usd": 0.0,
        }


async def compute_all_cost_metrics(
    pool: Any,
    since: datetime | None = None,
    until: datetime | None = None,
) -> dict[str, Any]:
    """Compute all cost metrics at once.

    Returns a comprehensive metrics dictionary suitable for reporting.
    """
    import asyncio  # noqa: PLC0415
    
    cost_per_avatar, cost_per_message, cost_per_conversation, cost_per_new_user = await asyncio.gather(
        compute_cost_per_avatar(pool, since, until),
        compute_average_cost_per_message(pool, since, until),
        compute_average_cost_per_conversation(pool, since, until),
        compute_cost_per_new_user(pool, since, until),
    )
    
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "period_start": (since or (datetime.now(UTC) - timedelta(days=30))).isoformat(),
        "period_end": (until or datetime.now(UTC)).isoformat(),
        "cost_per_avatar": cost_per_avatar["cost_per_avatar_usd"],
        "avg_cost_per_message": cost_per_message["avg_cost_per_message_usd"],
        "avg_cost_per_conversation": cost_per_conversation["avg_cost_per_conversation_usd"],
        "cost_per_new_user": cost_per_new_user["cost_per_new_user_usd"],
        "total_spend_usd": cost_per_avatar["total_cost_usd"],
        "total_tokens": sum(item["tokens"] for item in cost_per_avatar["by_avatar"]),
        "details": {
            "avatars": cost_per_avatar,
            "messages": cost_per_message,
            "conversations": cost_per_conversation,
            "new_users": cost_per_new_user,
        },
    }
