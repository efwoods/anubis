"""Project next-period burn from product spend, growth, and subscriptions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from src.anubis.utils.analytics.forecast import forecast_series
from src.anubis.utils.analytics.platform_metrics import (
    LEDGER_FULLY_LOADED,
    LEDGER_PRODUCT,
    cost_per_new_user,
    first_seen_users_per_week,
    product_spend_by_day,
    unit_economics,
)


async def _daily_vendor_and_bank(
    pool: Any, user_id: str, since: datetime, until: datetime
) -> list[float]:
    """Return a daily fully-loaded add-on series aligned to calendar days."""
    from src.anubis.utils.analytics.finance import spend_by_period as finance_spend
    from src.anubis.utils.analytics.vendor_usage import usage_by_period

    vendor = await usage_by_period(pool, user_id, since=since, until=until)
    bank = await finance_spend(pool, user_id, since, until, group_by="day")
    by_day: dict[str, float] = {}
    for row in vendor.get("rows") or []:
        day = str(row[0] or "")[:10]
        metric = str(row[2] or "").strip().lower()
        unit = str(row[4] or "").strip().lower()
        value = float(row[3] or 0.0)
        if metric in ("cost", "subscription", "amount_seen") or unit == "usd":
            by_day[day] = by_day.get(day, 0.0) + value
    for row in bank.get("rows") or []:
        day = str(row[0] or "")[:10]
        by_day[day] = by_day.get(day, 0.0) + float(row[1] or 0.0)
    start = since.date()
    end = until.date()
    days: list[float] = []
    cursor = start
    while cursor <= end:
        days.append(round(by_day.get(cursor.isoformat(), 0.0), 4))
        cursor = cursor + timedelta(days=1)
    return days


async def _subscription_run_rate_usd(
    pool: Any, user_id: str, since: datetime, until: datetime
) -> float:
    """Mean daily subscription cost, scaled to a 30-day month."""
    from src.anubis.utils.analytics.vendor_usage import usage_by_period

    rows = await usage_by_period(
        pool, user_id, since=since, until=until, metric="subscription"
    )
    total = sum(float(row[3] or 0.0) for row in rows.get("rows") or [])
    return round(total, 4)


async def _sheet_expected_burn(pool: Any, user_id: str) -> dict[str, Any] | None:
    """Return the latest Google Sheet expected-burn snapshot, if any."""
    from src.anubis.utils.analytics.reference_forecasts import latest_expected_burn

    try:
        return await latest_expected_burn(pool, user_id)
    except Exception:
        return None


async def project_burn(
    pool: Any,
    user_id: str,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    horizon: int = 30,
    ledger: str = LEDGER_FULLY_LOADED,
) -> dict[str, Any]:
    """Forecast burn from product spend, new-user growth, and subscriptions."""
    end = until or datetime.now(UTC)
    start = since or end - timedelta(days=30)
    product_days = await product_spend_by_day(pool, start, end)
    add_on_days = (
        await _daily_vendor_and_bank(pool, user_id, start, end)
        if ledger == LEDGER_FULLY_LOADED
        else [0.0] * len(product_days)
    )
    if len(add_on_days) < len(product_days):
        add_on_days = add_on_days + [0.0] * (len(product_days) - len(add_on_days))
    observed = [
        float(product) + float(add_on)
        for product, add_on in zip(product_days, add_on_days[: len(product_days)])
    ]
    spend_forecast = forecast_series(observed, max(1, int(horizon)), season_length=7)
    if "error" in spend_forecast:
        return {"status": "error", "message": spend_forecast["error"]}
    growth = await first_seen_users_per_week(pool, start, end)
    new_users_series = [float(row[1] or 0.0) for row in growth.get("rows") or []]
    users_forecast = (
        forecast_series(new_users_series, max(1, int(horizon) // 7 or 1))
        if len(new_users_series) >= 3
        else {"error": "not enough new-user weeks"}
    )
    unit = await cost_per_new_user(pool, start, end)
    unit_row = (unit.get("rows") or [[0, 0.0, None]])[0]
    cost_of_one_new_user = unit_row[2]
    projected_new_users = 0.0
    if "point" in users_forecast:
        projected_new_users = float(sum(users_forecast["point"]))
    growth_cost = (
        projected_new_users * float(cost_of_one_new_user)
        if cost_of_one_new_user is not None
        else 0.0
    )
    subscriptions = await _subscription_run_rate_usd(pool, user_id, start, end)
    economics = await unit_economics(
        pool, start, end, ledger=ledger, user_id=user_id
    )
    sheet = await _sheet_expected_burn(pool, user_id)
    point = [round(value + (subscriptions / max(1, int(horizon))), 4) for value in spend_forecast["point"]]
    comparison = None
    if sheet and sheet.get("expected_burn_usd") is not None:
        projected_total = round(sum(point) + growth_cost, 2)
        comparison = {
            "sheet_expected_burn_usd": sheet["expected_burn_usd"],
            "projected_total_usd": projected_total,
            "delta_usd": round(projected_total - float(sheet["expected_burn_usd"]), 2),
            "sheet_period": sheet.get("period"),
        }
    return {
        "status": "ok",
        "ledger": ledger if ledger in (LEDGER_PRODUCT, LEDGER_FULLY_LOADED) else LEDGER_FULLY_LOADED,
        "method": spend_forecast["method"],
        "horizon": int(horizon),
        "observed_days": len(observed),
        "point": point,
        "lower": spend_forecast["lower"],
        "upper": spend_forecast["upper"],
        "slope_per_step": spend_forecast["slope_per_step"],
        "projected_new_users": round(projected_new_users, 2),
        "cost_per_new_user": cost_of_one_new_user,
        "growth_cost_usd": round(growth_cost, 4),
        "subscription_run_rate_usd": subscriptions,
        "unit_economics": economics,
        "sheet_comparison": comparison,
    }
