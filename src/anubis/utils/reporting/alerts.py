"""Cost/usage alerts to agent inbox.

Monitors spend thresholds and anomalies, sending notifications to the
personal avatar's owner via the inbox system.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


async def check_cost_thresholds(
    context: Any,
    pool: Any,
    user_id: str,
    assistant_id: str,
) -> list[dict[str, Any]]:
    """Check if any cost thresholds have been exceeded.

    Returns list of alerts to send to inbox.
    """
    alerts = []
    
    # Get configured thresholds
    daily_threshold = getattr(context, "cost_alert_daily_threshold_usd", None)
    monthly_threshold = getattr(context, "cost_alert_monthly_threshold_usd", None)
    
    if not daily_threshold and not monthly_threshold:
        return alerts
    
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                # Check daily spend
                if daily_threshold:
                    today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
                    await cursor.execute(
                        """
                        SELECT ROUND(SUM(cost_usd)::numeric, 4) AS daily_cost
                        FROM api_metrics
                        WHERE user_id = %s AND created_at >= %s;
                        """,
                        (user_id, today),
                    )
                    row = await cursor.fetchone()
                    daily_cost = float(row[0] or 0.0)
                    
                    if daily_cost >= daily_threshold:
                        alerts.append({
                            "type": "daily_threshold_exceeded",
                            "severity": "high",
                            "subject": "Daily cost threshold exceeded",
                            "body": f"Daily spend ${daily_cost:.2f} exceeded threshold ${daily_threshold:.2f}",
                            "cost_usd": daily_cost,
                            "threshold_usd": daily_threshold,
                        })
                
                # Check monthly spend
                if monthly_threshold:
                    month_start = datetime.now(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                    await cursor.execute(
                        """
                        SELECT ROUND(SUM(cost_usd)::numeric, 4) AS monthly_cost
                        FROM api_metrics
                        WHERE user_id = %s AND created_at >= %s;
                        """,
                        (user_id, month_start),
                    )
                    row = await cursor.fetchone()
                    monthly_cost = float(row[0] or 0.0)
                    
                    if monthly_cost >= monthly_threshold:
                        alerts.append({
                            "type": "monthly_threshold_exceeded",
                            "severity": "high",
                            "subject": "Monthly cost threshold exceeded",
                            "body": f"Monthly spend ${monthly_cost:.2f} exceeded threshold ${monthly_threshold:.2f}",
                            "cost_usd": monthly_cost,
                            "threshold_usd": monthly_threshold,
                        })
    
    except Exception as threshold_error:  # noqa: BLE001
        logger.error("Failed to check cost thresholds: %s", threshold_error)
    
    return alerts


async def detect_cost_anomalies(
    pool: Any,
    user_id: str,
    lookback_days: int = 7,
) -> list[dict[str, Any]]:
    """Detect unusual spending patterns.

    Returns list of anomaly alerts.
    """
    alerts = []
    
    if pool is None:
        return alerts
    
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                # Compare today's spend to average of last N days
                today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
                lookback_start = today - timedelta(days=lookback_days)
                
                await cursor.execute(
                    """
                    WITH daily_costs AS (
                        SELECT
                            date_trunc('day', created_at)::date AS day,
                            SUM(cost_usd) AS daily_cost
                        FROM api_metrics
                        WHERE user_id = %s
                          AND created_at >= %s
                        GROUP BY day
                    ),
                    stats AS (
                        SELECT
                            AVG(daily_cost) AS avg_daily_cost,
                            STDDEV(daily_cost) AS stddev_daily_cost
                        FROM daily_costs
                        WHERE day < %s
                    ),
                    today_cost AS (
                        SELECT daily_cost
                        FROM daily_costs
                        WHERE day = %s
                    )
                    SELECT
                        today_cost.daily_cost,
                        stats.avg_daily_cost,
                        stats.stddev_daily_cost
                    FROM today_cost, stats;
                    """,
                    (user_id, lookback_start, today, today),
                )
                row = await cursor.fetchone()
                
                if row and row[0] is not None and row[1] is not None:
                    today_cost = float(row[0])
                    avg_cost = float(row[1])
                    stddev_cost = float(row[2] or 0.0)
                    
                    # Alert if today's cost is > 2 standard deviations above average
                    if stddev_cost > 0 and today_cost > avg_cost + (2 * stddev_cost):
                        alerts.append({
                            "type": "cost_anomaly",
                            "severity": "medium",
                            "subject": "Unusual spending detected",
                            "body": f"Today's spend ${today_cost:.2f} is significantly higher than "
                                   f"average ${avg_cost:.2f} (±${stddev_cost:.2f})",
                            "today_cost_usd": today_cost,
                            "average_cost_usd": avg_cost,
                            "stddev_usd": stddev_cost,
                        })
    
    except Exception as anomaly_error:  # noqa: BLE001
        logger.error("Failed to detect cost anomalies: %s", anomaly_error)
    
    return alerts


async def send_cost_alert_to_inbox(
    context: Any,
    pool: Any,
    store: Any,
    user_id: str,
    assistant_id: str,
    alert: dict[str, Any],
) -> bool:
    """Send a cost alert to the user's agent inbox.

    Uses the existing inbox delivery system.
    """
    try:
        from src.anubis.utils.inbox.delivery import deliver_to_inbox  # noqa: PLC0415
        
        # Construct inbox item
        item = {
            "source": "cost_reporting",
            "kind": "alert",
            "subject": alert.get("subject", "Cost Alert"),
            "body": alert.get("body", ""),
            "severity": alert.get("severity", "medium"),
            "metadata": {
                **alert,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        }
        
        # Store in inbox namespace
        namespace = (user_id, assistant_id, "inbox")
        key = f"cost_alert_{datetime.now(UTC).timestamp()}"
        
        await store.aput(namespace, key, item)
        
        logger.info("Sent cost alert to inbox: %s", alert.get("subject"))
        return True
    
    except Exception as inbox_error:  # noqa: BLE001
        logger.error("Failed to send alert to inbox: %s", inbox_error)
        return False


async def run_cost_monitoring(
    context: Any,
    pool: Any,
    store: Any,
    user_id: str,
    assistant_id: str,
) -> dict[str, Any]:
    """Run cost monitoring checks and send alerts if needed.

    Returns summary of alerts sent.
    """
    import asyncio  # noqa: PLC0415
    
    # Check thresholds and anomalies
    threshold_alerts, anomaly_alerts = await asyncio.gather(
        check_cost_thresholds(context, pool, user_id, assistant_id),
        detect_cost_anomalies(pool, user_id),
    )
    
    all_alerts = threshold_alerts + anomaly_alerts
    
    # Send each alert to inbox
    sent_count = 0
    for alert in all_alerts:
        success = await send_cost_alert_to_inbox(
            context, pool, store, user_id, assistant_id, alert
        )
        if success:
            sent_count += 1
    
    return {
        "alerts_detected": len(all_alerts),
        "alerts_sent": sent_count,
        "alerts": all_alerts,
    }
