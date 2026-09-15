"""Vendor recipes: where a signed-in session finds usage and cost figures.

A dashboard has no public API for a third party, but the browser the owner
signed in with can read what the owner can read. A recipe names the page (or
the dashboard's own JSON endpoint) that holds a vendor's usage figures and
turns the answer into daily rows ``{day, metric, value, unit}`` that
``vendor_usage_daily`` stores. Recipes are best effort — a vendor may move a
page — so every parser returns what could be read and the tool reports the
rest plainly. Exact figures always remain available through the vendor's
official API key when the owner adds one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Callable

RECIPE_KIND_JSON = "json"
RECIPE_KIND_DOM = "dom"


@dataclass(frozen=True)
class Recipe:
    """One page or endpoint of a vendor that yields usage rows."""

    name: str
    kind: str
    url_template: str
    parser: Callable[[Any, dict[str, Any]], list[dict[str, Any]]]
    description: str = ""
    method: str = "GET"
    headers: dict[str, str] = field(default_factory=dict)


def _period_bounds(period: str) -> tuple[datetime, datetime]:
    """Return ``(start, end)`` for ``7d`` / ``30d`` / ``90d`` style periods."""
    now = datetime.now(UTC)
    match = re.match(r"^(\d+)([dwm])$", str(period or "30d").strip().lower())
    days = 30
    if match:
        count = int(match.group(1))
        unit = match.group(2)
        days = count if unit == "d" else count * 7 if unit == "w" else count * 30
    return now - timedelta(days=days), now


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rows_from_daily_buckets(
    buckets: list[dict[str, Any]], *, day_key: str, metrics: dict[str, tuple[str, str]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bucket in buckets or []:
        if not isinstance(bucket, dict):
            continue
        day_value = bucket.get(day_key)
        if isinstance(day_value, (int, float)):
            day = datetime.fromtimestamp(float(day_value), tz=UTC).date().isoformat()
        else:
            day = str(day_value or "")[:10]
        if not day:
            continue
        for source_key, (metric, unit) in metrics.items():
            number = _number(bucket.get(source_key))
            if number is not None:
                rows.append({"day": day, "metric": metric, "value": number, "unit": unit})
    return rows


def parse_openai_costs(document: Any, variables: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse the OpenAI platform's cost buckets (dashboard JSON)."""
    rows: list[dict[str, Any]] = []
    data = document.get("data") if isinstance(document, dict) else None
    for bucket in data or []:
        if not isinstance(bucket, dict):
            continue
        start = bucket.get("start_time")
        day = (
            datetime.fromtimestamp(float(start), tz=UTC).date().isoformat()
            if isinstance(start, (int, float))
            else str(start or "")[:10]
        )
        total = 0.0
        for result in bucket.get("results") or []:
            amount = (result or {}).get("amount") or {}
            number = _number(amount.get("value"))
            if number is not None:
                total += number
        if day:
            rows.append({"day": day, "metric": "cost", "value": round(total, 6), "unit": "usd"})
    return rows


def parse_anthropic_usage(document: Any, variables: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse the Anthropic console's usage report buckets."""
    data = document.get("data") if isinstance(document, dict) else None
    rows: list[dict[str, Any]] = []
    for bucket in data or []:
        if not isinstance(bucket, dict):
            continue
        day = str(bucket.get("starting_at") or bucket.get("date") or "")[:10]
        if not day:
            continue
        cost = _number(bucket.get("cost_usd") or bucket.get("amount"))
        if cost is not None:
            rows.append({"day": day, "metric": "cost", "value": cost, "unit": "usd"})
        for token_key in ("input_tokens", "output_tokens", "uncached_input_tokens"):
            number = _number(bucket.get(token_key))
            if number is not None:
                rows.append({"day": day, "metric": token_key, "value": number, "unit": "tokens"})
    return rows


def parse_langsmith_usage(document: Any, variables: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse LangSmith's organization usage (traces per day, cost when present)."""
    if isinstance(document, list):
        buckets = document
    elif isinstance(document, dict):
        buckets = document.get("data") or document.get("usage") or []
    else:
        buckets = []
    return _rows_from_daily_buckets(
        buckets,
        day_key="date",
        metrics={
            "trace_count": ("traces", "count"),
            "traces": ("traces", "count"),
            "cost": ("cost", "usd"),
            "total_cost": ("cost", "usd"),
        },
    )


def parse_text_numbers(text: Any, variables: dict[str, Any]) -> list[dict[str, Any]]:
    """Fallback: dollar amounts found on a page, dated today."""
    body = str(text or "")
    rows: list[dict[str, Any]] = []
    today = datetime.now(UTC).date().isoformat()
    for match in re.finditer(r"\$\s?([0-9][0-9,]*(?:\.[0-9]{1,2})?)", body):
        number = _number(match.group(1).replace(",", ""))
        if number is not None:
            rows.append({"day": today, "metric": "amount_seen", "value": number, "unit": "usd"})
    return rows[:20]


def parse_elevenlabs_usage(document: Any, variables: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse ElevenLabs subscription or character-usage documents."""
    rows: list[dict[str, Any]] = []
    today = datetime.now(UTC).date().isoformat()
    payload = document if isinstance(document, dict) else {}
    character_count = _number(
        payload.get("character_count")
        or payload.get("characters_used")
        or (payload.get("subscription") or {}).get("character_count")
    )
    character_limit = _number(
        payload.get("character_limit")
        or (payload.get("subscription") or {}).get("character_limit")
    )
    if character_count is not None:
        rows.append({"day": today, "metric": "characters", "value": character_count, "unit": "count"})
    if character_limit is not None:
        rows.append({"day": today, "metric": "character_limit", "value": character_limit, "unit": "count"})
    cost = _number(payload.get("cost_usd") or payload.get("amount"))
    if cost is not None:
        rows.append({"day": today, "metric": "cost", "value": cost, "unit": "usd"})
    return rows


def parse_xai_billing(document: Any, variables: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse an xAI billing JSON document into daily cost rows."""
    payload = document if isinstance(document, dict) else {}
    buckets = payload.get("data") or payload.get("usage") or payload.get("items") or []
    if isinstance(buckets, list) and buckets:
        return _rows_from_daily_buckets(
            buckets,
            day_key="date",
            metrics={"cost": ("cost", "usd"), "amount": ("cost", "usd"), "total": ("cost", "usd")},
        )
    return parse_text_numbers(document, variables)


def parse_cursor_spending(document: Any, variables: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse Cursor spending or usage JSON into subscription and usage rows."""
    payload = document if isinstance(document, dict) else {}
    today = datetime.now(UTC).date().isoformat()
    rows: list[dict[str, Any]] = []
    subscription = _number(
        payload.get("subscription")
        or payload.get("plan_amount")
        or payload.get("monthly_plan_usd")
    )
    if subscription is not None:
        rows.append({"day": today, "metric": "subscription", "value": subscription, "unit": "usd"})
    usage = _number(payload.get("on_demand_spend") or payload.get("usage_usd") or payload.get("overage"))
    if usage is not None:
        rows.append({"day": today, "metric": "cost", "value": usage, "unit": "usd"})
    included = _number(payload.get("included_tokens") or payload.get("included"))
    if included is not None:
        rows.append({"day": today, "metric": "included_tokens", "value": included, "unit": "tokens"})
    used = _number(payload.get("used_tokens") or payload.get("tokens"))
    if used is not None:
        rows.append({"day": today, "metric": "usage", "value": used, "unit": "tokens"})
    if rows:
        return rows
    return parse_text_numbers(document, variables)


def parse_claude_app_usage(document: Any, variables: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse Claude.ai settings usage into subscription and token rows."""
    payload = document if isinstance(document, dict) else {}
    today = datetime.now(UTC).date().isoformat()
    rows: list[dict[str, Any]] = []
    subscription = _number(payload.get("plan_cost") or payload.get("subscription"))
    if subscription is not None:
        rows.append({"day": today, "metric": "subscription", "value": subscription, "unit": "usd"})
    tokens = _number(payload.get("tokens") or payload.get("used_tokens"))
    if tokens is not None:
        rows.append({"day": today, "metric": "usage", "value": tokens, "unit": "tokens"})
    if rows:
        return rows
    return parse_text_numbers(document, variables)


RECIPES: dict[str, dict[str, Recipe]] = {
    "openai": {
        "costs": Recipe(
            name="costs",
            kind=RECIPE_KIND_JSON,
            url_template=(
                "https://api.openai.com/v1/organization/costs?start_time={start_epoch}"
                "&end_time={end_epoch}&bucket_width=1d&limit=180"
            ),
            parser=parse_openai_costs,
            description="Daily cost in USD for the organization (session-authenticated).",
        ),
        "usage_page": Recipe(
            name="usage_page",
            kind=RECIPE_KIND_DOM,
            url_template=(
                "https://platform.openai.com/settings/organization/usage"
                "?usage_section=spend-categories"
            ),
            parser=parse_text_numbers,
            description="Amounts shown on the spend-categories usage page (fallback).",
        ),
    },
    "anthropic": {
        "usage": Recipe(
            name="usage",
            kind=RECIPE_KIND_JSON,
            url_template=(
                "https://api.anthropic.com/v1/organizations/cost_report?starting_at={start_iso}"
                "&ending_at={end_iso}&bucket_width=1d"
            ),
            parser=parse_anthropic_usage,
            description="Daily cost in USD for the organization (session-authenticated).",
        ),
        "usage_page": Recipe(
            name="usage_page",
            kind=RECIPE_KIND_DOM,
            url_template="https://console.anthropic.com/settings/usage",
            parser=parse_text_numbers,
            description="Amounts shown on the usage page (fallback).",
        ),
    },
    "langsmith": {
        "usage": Recipe(
            name="usage",
            kind=RECIPE_KIND_JSON,
            url_template=(
                "https://api.smith.langchain.com/api/v1/orgs/current/usage?start_time={start_iso}"
                "&end_time={end_iso}"
            ),
            parser=parse_langsmith_usage,
            description="Traces per day and cost for the organization.",
        ),
        "usage_page": Recipe(
            name="usage_page",
            kind=RECIPE_KIND_DOM,
            url_template="https://smith.langchain.com/settings/usage",
            parser=parse_text_numbers,
            description="Amounts shown on the usage page (fallback).",
        ),
    },
    "elevenlabs": {
        "subscription": Recipe(
            name="subscription",
            kind=RECIPE_KIND_JSON,
            url_template="https://api.elevenlabs.io/v1/user/subscription",
            parser=parse_elevenlabs_usage,
            description="Character usage and plan limits for the workspace.",
        ),
        "usage_page": Recipe(
            name="usage_page",
            kind=RECIPE_KIND_DOM,
            url_template="https://elevenlabs.io/app/developers/analytics/usage",
            parser=parse_text_numbers,
            description="Amounts shown on the ElevenLabs analytics page.",
        ),
    },
    "xai": {
        "billing": Recipe(
            name="billing",
            kind=RECIPE_KIND_JSON,
            url_template="https://api.x.ai/v1/usage",
            parser=parse_xai_billing,
            description="Usage and billing for the xAI team, when the key can reach it.",
        ),
        "usage_page": Recipe(
            name="usage_page",
            kind=RECIPE_KIND_DOM,
            url_template="https://console.x.ai/team/1db9c97a-09ce-4be7-bca1-f0fb9e59ec18/settings/billing",
            parser=parse_text_numbers,
            description="Amounts shown on the xAI billing page.",
        ),
    },
    "cursor": {
        "spending": Recipe(
            name="spending",
            kind=RECIPE_KIND_DOM,
            url_template="https://cursor.com/dashboard/spending",
            parser=parse_text_numbers,
            description="Monthly plan and overage shown on the Cursor spending page.",
        ),
        "usage": Recipe(
            name="usage",
            kind=RECIPE_KIND_DOM,
            url_template="https://cursor.com/dashboard/usage",
            parser=parse_text_numbers,
            description="Included and used tokens shown on the Cursor usage page.",
        ),
    },
    "claude_app": {
        "usage_page": Recipe(
            name="usage_page",
            kind=RECIPE_KIND_DOM,
            url_template="https://claude.ai/settings/usage",
            parser=parse_claude_app_usage,
            description="Claude.ai subscription usage (not the Anthropic API console).",
        ),
    },
}


def recipes_for(recipe_key: str | None) -> dict[str, Recipe]:
    """Return the recipes of one vendor (empty for unknown keys)."""
    return dict(RECIPES.get(str(recipe_key or "").lower(), {}))


def render_url(recipe: Recipe, period: str) -> str:
    """Fill a recipe's URL template for the requested period."""
    start, end = _period_bounds(period)
    return recipe.url_template.format(
        start_epoch=int(start.timestamp()),
        end_epoch=int(end.timestamp()),
        start_iso=start.date().isoformat(),
        end_iso=end.date().isoformat(),
    )
