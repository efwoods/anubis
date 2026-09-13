"""Personal avatar tools for cost/usage reporting to Google Sheets.

Allows personal avatars to read current cost metrics, update the reporting
spreadsheet, track feature costs, and manage cost alerts.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from langchain.tools import tool
from langchain_core.tools import InjectedToolArg
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GetCostMetricsInput(BaseModel):
    """Input for get_cost_metrics tool."""

    period_days: int = Field(
        default=30,
        description="Number of days to look back for metrics (default 30).",
    )


@tool("get_cost_metrics", return_direct=False, args_schema=GetCostMetricsInput)
async def get_cost_metrics(
    period_days: int = 30,
    runtime: Annotated[Any, InjectedToolArg] = None,  # noqa: UP007
) -> dict[str, Any]:
    """Get current cost and usage metrics for the avatar and platform.

    Returns cost per avatar, average cost per message, average cost per
    conversation, and cost per new user based on real telemetry.
    """
    from datetime import UTC, datetime, timedelta  # noqa: PLC0415

    from src.anubis.utils.reporting.cost_metrics import compute_all_cost_metrics  # noqa: PLC0415

    if runtime is None:
        return {"error": "Runtime not available"}

    context = getattr(runtime, "context", None)
    pool = getattr(runtime, "pool", None)

    if pool is None:
        return {"error": "Database pool not available"}

    since = datetime.now(UTC) - timedelta(days=period_days)
    until = datetime.now(UTC)

    metrics = await compute_all_cost_metrics(pool, since, until)

    return {
        "success": True,
        "metrics": metrics,
        "note": "All metrics computed from real api_metrics telemetry",
    }


class UpdateCostReportInput(BaseModel):
    """Input for update_cost_report tool."""

    force_update: bool = Field(
        default=False,
        description="Force update even if recently updated.",
    )


@tool("update_cost_report", return_direct=False, args_schema=UpdateCostReportInput)
async def update_cost_report(
    force_update: bool = False,
    runtime: Annotated[Any, InjectedToolArg] = None,  # noqa: UP007
) -> dict[str, Any]:
    """Update the Google Sheet cost report with current metrics.

    Fetches current metrics and writes them to the configured Google Sheet.
    """
    from datetime import UTC, datetime, timedelta  # noqa: PLC0415

    from src.anubis.utils.reporting.cost_metrics import compute_all_cost_metrics  # noqa: PLC0415
    from src.anubis.utils.reporting.google_sheets import update_cost_report  # noqa: PLC0415

    if runtime is None:
        return {"error": "Runtime not available"}

    context = getattr(runtime, "context", None)
    pool = getattr(runtime, "pool", None)

    if pool is None:
        return {"error": "Database pool not available"}

    # Get current metrics
    since = datetime.now(UTC) - timedelta(days=30)
    until = datetime.now(UTC)
    metrics = await compute_all_cost_metrics(pool, since, until)

    # Update sheet
    success = await update_cost_report(context, metrics)

    return {
        "success": success,
        "updated_at": datetime.now(UTC).isoformat(),
        "metrics": metrics,
        "note": "Report updated in configured Google Sheet"
        if success
        else "Failed to update sheet (check credentials)",
    }


class TrackFeatureCostInput(BaseModel):
    """Input for track_feature_cost tool."""

    feature_name: str = Field(
        description="Name or description of the feature being tracked.",
    )
    period_days: int = Field(
        default=7,
        description="Number of days to look back (default 7).",
    )


@tool("track_feature_cost", return_direct=False, args_schema=TrackFeatureCostInput)
async def track_feature_cost(
    feature_name: str,
    period_days: int = 7,
    runtime: Annotated[Any, InjectedToolArg] = None,  # noqa: UP007
) -> dict[str, Any]:
    """Track development cost of a feature by correlating git commits and AI usage.

    Returns current spend, commits, and duration for ongoing feature work.
    """
    from datetime import UTC, datetime, timedelta  # noqa: PLC0415

    from src.anubis.utils.reporting.feature_tracking import track_feature_progress  # noqa: PLC0415

    if runtime is None:
        return {"error": "Runtime not available"}

    context = getattr(runtime, "context", None)
    pool = getattr(runtime, "pool", None)

    if pool is None:
        return {"error": "Database pool not available"}

    tracking = await track_feature_progress(pool, feature_name)

    return {
        "success": True,
        "feature_tracking": tracking,
        "note": f"Tracking costs for '{feature_name}' over last {period_days} days",
    }


class PredictFeatureCostInput(BaseModel):
    """Input for predict_feature_cost tool."""

    estimated_hours: float = Field(
        description="Estimated development time in hours.",
    )
    similar_features: list[str] | None = Field(
        default=None,
        description="Names of similar features for better estimation.",
    )


@tool("predict_feature_cost", return_direct=False, args_schema=PredictFeatureCostInput)
async def predict_feature_cost(
    estimated_hours: float,
    similar_features: list[str] | None = None,
    runtime: Annotated[Any, InjectedToolArg] = None,  # noqa: UP007
) -> dict[str, Any]:
    """Predict cost of planned feature based on historical development data.

    Returns predicted cost, tokens, and confidence level.
    """
    from src.anubis.utils.reporting.feature_tracking import predict_feature_cost  # noqa: PLC0415

    if runtime is None:
        return {"error": "Runtime not available"}

    pool = getattr(runtime, "pool", None)

    if pool is None:
        return {"error": "Database pool not available"}

    prediction = await predict_feature_cost(pool, estimated_hours, similar_features)

    return {
        "success": True,
        "prediction": prediction,
        "note": "Prediction based on recent historical data",
    }


class CheckCostAlertsInput(BaseModel):
    """Input for check_cost_alerts tool."""

    send_to_inbox: bool = Field(
        default=True,
        description="Whether to send alerts to inbox (default True).",
    )


@tool("check_cost_alerts", return_direct=False, args_schema=CheckCostAlertsInput)
async def check_cost_alerts(
    send_to_inbox: bool = True,
    runtime: Annotated[Any, InjectedToolArg] = None,  # noqa: UP007
) -> dict[str, Any]:
    """Check for cost threshold breaches and anomalies.

    Optionally sends alerts to the owner's agent inbox.
    """
    from src.anubis.utils.reporting.alerts import run_cost_monitoring  # noqa: PLC0415

    if runtime is None:
        return {"error": "Runtime not available"}

    context = getattr(runtime, "context", None)
    pool = getattr(runtime, "pool", None)
    store = getattr(runtime, "store", None)
    config = getattr(runtime, "config", {})

    user_id = config.get("user_id")
    assistant_id = config.get("assistant_id")

    if not user_id or not assistant_id:
        return {"error": "User ID or assistant ID not available"}

    if send_to_inbox:
        monitoring = await run_cost_monitoring(
            context, pool, store, user_id, assistant_id
        )
    else:
        from src.anubis.utils.reporting.alerts import (  # noqa: PLC0415
            check_cost_thresholds,
            detect_cost_anomalies,
        )

        import asyncio  # noqa: PLC0415

        threshold_alerts, anomaly_alerts = await asyncio.gather(
            check_cost_thresholds(context, pool, user_id, assistant_id),
            detect_cost_anomalies(pool, user_id),
        )
        monitoring = {
            "alerts_detected": len(threshold_alerts) + len(anomaly_alerts),
            "alerts_sent": 0,
            "alerts": threshold_alerts + anomaly_alerts,
        }

    return {
        "success": True,
        "monitoring": monitoring,
        "note": "Cost monitoring complete; alerts sent to inbox if configured"
        if send_to_inbox
        else "Cost monitoring complete; alerts not sent",
    }


# Export all tools
REPORTING_TOOLS = [
    get_cost_metrics,
    update_cost_report,
    track_feature_cost,
    predict_feature_cost,
    check_cost_alerts,
]
