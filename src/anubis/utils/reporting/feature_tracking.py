"""Feature cost tracking: correlate git commits with development costs.

Tracks costs of features built by correlating git commits with AI usage
(Claude Code, Grok), timing, and predicts expectations for planned work.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


def get_git_commits_since(
    repo_path: str,
    since: datetime | None = None,
    until: datetime | None = None,
    branch: str = "HEAD",
) -> list[dict[str, Any]]:
    """Get git commits in a time range.

    Returns:
        List of commits with {hash, author, date, message}
    """
    since = since or (datetime.now(UTC) - timedelta(days=7))
    until = until or datetime.now(UTC)
    
    try:
        # Git log format: %H|%an|%aI|%s (hash|author|date|subject)
        result = subprocess.run(
            [
                "git",
                "-C",
                repo_path,
                "log",
                f"--since={since.isoformat()}",
                f"--until={until.isoformat()}",
                "--pretty=format:%H|%an|%aI|%s",
                branch,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        
        if result.returncode != 0:
            logger.error("Git log failed: %s", result.stderr)
            return []
        
        commits = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|", 3)
            if len(parts) == 4:
                commits.append({
                    "hash": parts[0],
                    "author": parts[1],
                    "date": parts[2],
                    "message": parts[3],
                })
        
        return commits
    except Exception as git_error:  # noqa: BLE001
        logger.error("Failed to get git commits: %s", git_error)
        return []


async def compute_feature_cost(
    pool: Any,
    commit_hash: str | None = None,
    commit_message_pattern: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> dict[str, Any]:
    """Compute cost of a feature by correlating commits with api_metrics.

    Args:
        pool: Database connection pool
        commit_hash: Specific commit hash to analyze
        commit_message_pattern: Pattern to match in commit messages
        since: Start of time window
        until: End of time window

    Returns:
        {
            "commits": [...],
            "total_cost_usd": float,
            "total_tokens": int,
            "duration_seconds": float,
            "cost_per_hour": float,
        }
    """
    if pool is None:
        return {
            "commits": [],
            "total_cost_usd": 0.0,
            "total_tokens": 0,
            "duration_seconds": 0.0,
            "cost_per_hour": 0.0,
        }
    
    since = since or (datetime.now(UTC) - timedelta(days=7))
    until = until or datetime.now(UTC)
    
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                # For now, aggregate all costs in the time window
                # Future: correlate with actual commit timestamps
                await cursor.execute(
                    """
                    SELECT
                        ROUND(SUM(cost_usd)::numeric, 4) AS total_cost_usd,
                        SUM(total_tokens) AS total_tokens,
                        EXTRACT(EPOCH FROM (MAX(created_at) - MIN(created_at))) AS duration_seconds
                    FROM api_metrics
                    WHERE created_at >= %s AND created_at < %s;
                    """,
                    (since, until),
                )
                row = await cursor.fetchone()
                
                total_cost = float(row[0] or 0.0)
                total_tokens = int(row[1] or 0)
                duration_seconds = float(row[2] or 0.0)
                
                cost_per_hour = (
                    (total_cost / (duration_seconds / 3600.0))
                    if duration_seconds > 0
                    else 0.0
                )
                
                return {
                    "commits": [],  # Would correlate with git data
                    "total_cost_usd": total_cost,
                    "total_tokens": total_tokens,
                    "duration_seconds": duration_seconds,
                    "cost_per_hour": round(cost_per_hour, 4),
                    "period_start": since.isoformat(),
                    "period_end": until.isoformat(),
                }
    except Exception as cost_error:  # noqa: BLE001
        logger.error("Failed to compute feature cost: %s", cost_error)
        return {
            "commits": [],
            "total_cost_usd": 0.0,
            "total_tokens": 0,
            "duration_seconds": 0.0,
            "cost_per_hour": 0.0,
        }


async def predict_feature_cost(
    pool: Any,
    estimated_hours: float,
    similar_features: list[str] | None = None,
) -> dict[str, Any]:
    """Predict cost of planned feature based on historical data.

    Args:
        pool: Database connection pool
        estimated_hours: Estimated development time
        similar_features: List of similar feature names/patterns for better estimation

    Returns:
        {
            "predicted_cost_usd": float,
            "predicted_tokens": int,
            "confidence": str,
            "basis": str,
        }
    """
    if pool is None:
        return {
            "predicted_cost_usd": 0.0,
            "predicted_tokens": 0,
            "confidence": "low",
            "basis": "no_data",
        }
    
    try:
        # Compute average cost per hour from recent history
        lookback = datetime.now(UTC) - timedelta(days=30)
        
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT
                        ROUND(AVG(cost_usd)::numeric, 4) AS avg_cost_per_inference,
                        AVG(total_tokens) AS avg_tokens_per_inference,
                        COUNT(*) AS sample_size
                    FROM api_metrics
                    WHERE created_at >= %s
                      AND inference_type IN ('message', 'adapter_inference');
                    """,
                    (lookback,),
                )
                row = await cursor.fetchone()
                
                avg_cost_per_inference = float(row[0] or 0.0)
                avg_tokens_per_inference = float(row[1] or 0.0)
                sample_size = int(row[2] or 0)
                
                # Rough heuristic: ~100 inferences per development hour
                # (This is a placeholder; would be calibrated from actual data)
                inferences_per_hour = 100
                
                predicted_cost = avg_cost_per_inference * inferences_per_hour * estimated_hours
                predicted_tokens = int(avg_tokens_per_inference * inferences_per_hour * estimated_hours)
                
                confidence = "high" if sample_size > 1000 else "medium" if sample_size > 100 else "low"
                
                return {
                    "predicted_cost_usd": round(predicted_cost, 4),
                    "predicted_tokens": predicted_tokens,
                    "estimated_hours": estimated_hours,
                    "confidence": confidence,
                    "basis": f"{sample_size} historical inferences",
                    "assumptions": {
                        "inferences_per_hour": inferences_per_hour,
                        "avg_cost_per_inference": avg_cost_per_inference,
                    },
                }
    except Exception as predict_error:  # noqa: BLE001
        logger.error("Failed to predict feature cost: %s", predict_error)
        return {
            "predicted_cost_usd": 0.0,
            "predicted_tokens": 0,
            "confidence": "low",
            "basis": "error",
        }


async def track_feature_progress(
    pool: Any,
    feature_name: str,
    repo_path: str = "/workspace",
) -> dict[str, Any]:
    """Track ongoing feature development costs in real-time.

    Returns current spend and commits for an in-progress feature.
    """
    # Get commits from last 7 days as proxy for current feature
    commits = get_git_commits_since(repo_path, since=datetime.now(UTC) - timedelta(days=7))
    
    # Get costs from same period
    feature_cost = await compute_feature_cost(
        pool,
        since=datetime.now(UTC) - timedelta(days=7),
    )
    
    return {
        "feature_name": feature_name,
        "commits_count": len(commits),
        "recent_commits": commits[:5],  # Last 5 commits
        "current_cost_usd": feature_cost["total_cost_usd"],
        "current_tokens": feature_cost["total_tokens"],
        "duration_hours": feature_cost["duration_seconds"] / 3600.0,
    }
