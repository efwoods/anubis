"""Platform metrics for the platform administrator.

Every function here answers one owner question ("how often do users send
messages", "which avatars do users speak to most", "what do users dislike")
from the application's own tables: ``api_metrics`` (one row per billed model
inference), ``tool_calls`` (one row per tool the avatar used),
``connected_accounts``, the LangGraph ``thread`` and ``assistant`` tables,
and the LangGraph ``store`` (message feedback). Every query is parameterised;
no caller-supplied text is interpolated into SQL.

Admin traffic is excluded from ``api_metrics`` by design (the metering layer
skips the administrator), so every count here describes real users only. Each
result carries that note so the avatar can say so.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

ADMIN_TRAFFIC_NOTE = "Admin traffic is excluded from api_metrics by design."
DEFAULT_PERIOD_DAYS = 30
MESSAGE_INFERENCE_TYPES: tuple[str, ...] = ("message", "adapter_inference")

SPEND_GROUPINGS: tuple[str, ...] = ("day", "model", "inference_type")

FEEDBACK_LIKE = "like"
FEEDBACK_DISLIKE = "dislike"
FEEDBACK_NAMESPACE_SUFFIX = "message_feedback"

MONTHS_PER_YEAR = 12
CENTS_PER_DOLLAR = 100.0


def default_period(
    since: datetime | None, until: datetime | None
) -> tuple[datetime, datetime]:
    """Return ``(since, until)`` defaulting to the last thirty days ending now."""
    end = until or datetime.now(UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    start = since or end - timedelta(days=DEFAULT_PERIOD_DAYS)
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    return start, end


def _result(columns: list[str], rows: list[tuple]) -> dict[str, Any]:
    """Shape one query result as ``{columns, rows, note}`` with plain values."""
    return {
        "columns": list(columns),
        "rows": [[_plain(value) for value in row] for row in rows],
        "note": ADMIN_TRAFFIC_NOTE,
    }


def _plain(value: Any) -> Any:
    """Turn database values into JSON-friendly values."""
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    try:
        from decimal import Decimal

        if isinstance(value, Decimal):
            return float(value)
    except ImportError:  # pragma: no cover - decimal is standard
        pass
    return value


async def _fetchall(pool: Any, sql: str, params: tuple) -> list[tuple]:
    """Run one parameterised query on the pool and return every row."""
    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(sql, params)
            return await cursor.fetchall()


async def messages_per_day(
    pool: Any, since: datetime | None = None, until: datetime | None = None
) -> dict[str, Any]:
    """Count user messages per day (owner question: how much traffic is there)."""
    start, end = default_period(since, until)
    rows = await _fetchall(
        pool,
        """
        SELECT date_trunc('day', created_at)::date AS day, COUNT(*) AS messages
        FROM api_metrics
        WHERE inference_type = ANY(%s) AND created_at >= %s AND created_at < %s
        GROUP BY 1 ORDER BY 1;
        """,
        (list(MESSAGE_INFERENCE_TYPES), start, end),
    )
    return _result(["day", "messages"], rows)


async def messages_per_user_per_day(
    pool: Any, since: datetime | None = None, until: datetime | None = None
) -> dict[str, Any]:
    """Average messages each active user sends per day (owner question: how often users message)."""
    start, end = default_period(since, until)
    rows = await _fetchall(
        pool,
        """
        SELECT day,
               COUNT(DISTINCT user_id) AS active_users,
               SUM(messages) AS messages,
               ROUND(AVG(messages)::numeric, 3) AS messages_per_user
        FROM (
            SELECT date_trunc('day', created_at)::date AS day, user_id, COUNT(*) AS messages
            FROM api_metrics
            WHERE inference_type = ANY(%s) AND created_at >= %s AND created_at < %s
              AND user_id IS NOT NULL
            GROUP BY 1, 2
        ) AS per_user
        GROUP BY day ORDER BY day;
        """,
        (list(MESSAGE_INFERENCE_TYPES), start, end),
    )
    return _result(["day", "active_users", "messages", "messages_per_user"], rows)


async def average_conversation_length(
    pool: Any, since: datetime | None = None, until: datetime | None = None
) -> dict[str, Any]:
    """Average and median conversation length in turns and in wall-clock minutes.

    Turns come from ``api_metrics`` (one row per message inference per thread);
    wall clock is the ``thread`` table's ``updated_at - created_at``.
    """
    start, end = default_period(since, until)
    rows = await _fetchall(
        pool,
        """
        WITH turns AS (
            SELECT thread_id, COUNT(*) AS turn_count
            FROM api_metrics
            WHERE inference_type = ANY(%s) AND created_at >= %s AND created_at < %s
              AND thread_id IS NOT NULL
            GROUP BY thread_id
        ),
        wall_clock AS (
            SELECT thread_id::text AS thread_id,
                   EXTRACT(EPOCH FROM (updated_at - created_at)) / 60.0 AS minutes
            FROM thread
            WHERE created_at >= %s AND created_at < %s
        )
        SELECT (SELECT COUNT(*) FROM turns) AS conversations,
               (SELECT ROUND(AVG(turn_count)::numeric, 2) FROM turns) AS average_turns,
               (SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY turn_count) FROM turns)
                   AS median_turns,
               (SELECT ROUND(AVG(minutes)::numeric, 2) FROM wall_clock) AS average_minutes,
               (SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY minutes) FROM wall_clock)
                   AS median_minutes;
        """,
        (list(MESSAGE_INFERENCE_TYPES), start, end, start, end),
    )
    return _result(
        ["conversations", "average_turns", "median_turns", "average_minutes", "median_minutes"],
        rows,
    )


async def avatars_by_conversation_count(
    pool: Any,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Rank avatars by distinct conversations (owner question: which avatars users speak to most)."""
    start, end = default_period(since, until)
    rows = await _fetchall(
        pool,
        """
        SELECT metrics.assistant_id,
               assistant.name AS assistant_name,
               COUNT(DISTINCT metrics.thread_id) AS conversations,
               COUNT(DISTINCT metrics.user_id) AS users,
               COUNT(*) AS messages
        FROM api_metrics AS metrics
        LEFT JOIN assistant ON assistant.assistant_id::text = metrics.assistant_id
        WHERE metrics.inference_type = ANY(%s)
          AND metrics.created_at >= %s AND metrics.created_at < %s
          AND metrics.assistant_id IS NOT NULL
        GROUP BY metrics.assistant_id, assistant.name
        ORDER BY conversations DESC, messages DESC
        LIMIT %s;
        """,
        (list(MESSAGE_INFERENCE_TYPES), start, end, max(1, int(limit))),
    )
    return _result(
        ["assistant_id", "assistant_name", "conversations", "users", "messages"], rows
    )


async def feature_usage_per_avatar(
    pool: Any,
    since: datetime | None = None,
    until: datetime | None = None,
    assistant_id: str | None = None,
) -> dict[str, Any]:
    """Count feature use per avatar (owner question: which features each avatar uses most and least).

    Three sources are unioned: tool calls by tool name, model inferences by
    inference type, and connected accounts by provider, each keyed by the
    avatar that used the feature.
    """
    start, end = default_period(since, until)
    rows = await _fetchall(
        pool,
        """
        SELECT assistant_id, source, feature, SUM(uses) AS uses
        FROM (
            SELECT assistant_id, 'tool' AS source, tool_name AS feature, COUNT(*) AS uses
            FROM tool_calls
            WHERE created_at >= %s AND created_at < %s AND assistant_id IS NOT NULL
            GROUP BY assistant_id, tool_name
            UNION ALL
            SELECT assistant_id, 'inference' AS source, inference_type AS feature, COUNT(*) AS uses
            FROM api_metrics
            WHERE created_at >= %s AND created_at < %s AND assistant_id IS NOT NULL
            GROUP BY assistant_id, inference_type
            UNION ALL
            SELECT personal_avatar_id AS assistant_id, 'connection' AS source,
                   provider AS feature, COUNT(*) AS uses
            FROM connected_accounts
            WHERE personal_avatar_id IS NOT NULL
            GROUP BY personal_avatar_id, provider
        ) AS usage
        WHERE (%s::text IS NULL OR assistant_id = %s)
        GROUP BY assistant_id, source, feature
        ORDER BY assistant_id, uses DESC;
        """,
        (start, end, start, end, assistant_id, assistant_id),
    )
    return _result(["assistant_id", "source", "feature", "uses"], rows)


async def first_seen_users_per_week(
    pool: Any, since: datetime | None = None, until: datetime | None = None
) -> dict[str, Any]:
    """Count new users by the week they first appeared (owner question: how fast the user base grows)."""
    start, end = default_period(since, until)
    rows = await _fetchall(
        pool,
        """
        SELECT date_trunc('week', first_seen)::date AS week, COUNT(*) AS new_users
        FROM (
            SELECT user_id, MIN(created_at) AS first_seen
            FROM api_metrics
            WHERE user_id IS NOT NULL
            GROUP BY user_id
        ) AS first_appearances
        WHERE first_seen >= %s AND first_seen < %s
        GROUP BY 1 ORDER BY 1;
        """,
        (start, end),
    )
    return _result(["week", "new_users"], rows)


async def active_users(
    pool: Any, since: datetime | None = None, until: datetime | None = None
) -> dict[str, Any]:
    """Count distinct users who sent a message in the period (owner question: how many active users)."""
    start, end = default_period(since, until)
    rows = await _fetchall(
        pool,
        """
        SELECT COUNT(DISTINCT user_id) AS active_users,
               COUNT(DISTINCT thread_id) AS conversations,
               COUNT(*) AS messages
        FROM api_metrics
        WHERE inference_type = ANY(%s) AND created_at >= %s AND created_at < %s
          AND user_id IS NOT NULL;
        """,
        (list(MESSAGE_INFERENCE_TYPES), start, end),
    )
    return _result(["active_users", "conversations", "messages"], rows)


LEDGER_PRODUCT = "product"
LEDGER_FULLY_LOADED = "fully_loaded"
UNIT_ECONOMICS_LEDGERS: tuple[str, ...] = (LEDGER_PRODUCT, LEDGER_FULLY_LOADED)
DEVELOPMENT_OVERHEAD_PROVIDERS: frozenset[str] = frozenset(
    {"cursor", "claude_app", "claude_code"}
)


async def cost_per_avatar(
    pool: Any, since: datetime | None = None, until: datetime | None = None
) -> dict[str, Any]:
    """Sum product spend per avatar (owner question: what does each avatar cost)."""
    start, end = default_period(since, until)
    rows = await _fetchall(
        pool,
        """
        SELECT metrics.assistant_id,
               assistant.name AS assistant_name,
               ROUND(SUM(metrics.cost_usd)::numeric, 4) AS cost_usd,
               COUNT(*) FILTER (WHERE metrics.inference_type = ANY(%s)) AS messages,
               COUNT(DISTINCT metrics.thread_id) AS conversations
        FROM api_metrics AS metrics
        LEFT JOIN assistant ON assistant.assistant_id::text = metrics.assistant_id
        WHERE metrics.created_at >= %s AND metrics.created_at < %s
          AND metrics.assistant_id IS NOT NULL
        GROUP BY metrics.assistant_id, assistant.name
        ORDER BY cost_usd DESC;
        """,
        (list(MESSAGE_INFERENCE_TYPES), start, end),
    )
    total_cost = sum(float(row[2] or 0.0) for row in rows)
    avatars = len(rows)
    result = _result(
        ["assistant_id", "assistant_name", "cost_usd", "messages", "conversations"],
        rows,
    )
    result["total_cost_usd"] = round(total_cost, 4)
    result["avatars"] = avatars
    result["cost_per_avatar"] = (
        round(total_cost / avatars, 4) if avatars else None
    )
    return result


async def average_cost_per_message(
    pool: Any, since: datetime | None = None, until: datetime | None = None
) -> dict[str, Any]:
    """Average product cost of one billed message turn."""
    start, end = default_period(since, until)
    rows = await _fetchall(
        pool,
        """
        SELECT COUNT(*) AS messages,
               ROUND(COALESCE(SUM(cost_usd), 0)::numeric, 4) AS total_cost_usd,
               CASE WHEN COUNT(*) = 0 THEN NULL
                    ELSE ROUND((SUM(cost_usd) / COUNT(*))::numeric, 6)
               END AS average_cost_per_message
        FROM api_metrics
        WHERE inference_type = ANY(%s) AND created_at >= %s AND created_at < %s;
        """,
        (list(MESSAGE_INFERENCE_TYPES), start, end),
    )
    return _result(
        ["messages", "total_cost_usd", "average_cost_per_message"], rows
    )


async def average_cost_per_conversation(
    pool: Any, since: datetime | None = None, until: datetime | None = None
) -> dict[str, Any]:
    """Average product cost of one conversation (distinct thread)."""
    start, end = default_period(since, until)
    rows = await _fetchall(
        pool,
        """
        SELECT COUNT(DISTINCT thread_id) AS conversations,
               ROUND(COALESCE(SUM(cost_usd), 0)::numeric, 4) AS total_cost_usd,
               CASE WHEN COUNT(DISTINCT thread_id) = 0 THEN NULL
                    ELSE ROUND((SUM(cost_usd) / COUNT(DISTINCT thread_id))::numeric, 6)
               END AS average_cost_per_conversation
        FROM api_metrics
        WHERE inference_type = ANY(%s) AND created_at >= %s AND created_at < %s
          AND thread_id IS NOT NULL;
        """,
        (list(MESSAGE_INFERENCE_TYPES), start, end),
    )
    return _result(
        ["conversations", "total_cost_usd", "average_cost_per_conversation"], rows
    )


async def cost_per_new_user(
    pool: Any, since: datetime | None = None, until: datetime | None = None
) -> dict[str, Any]:
    """Product COGS of users whose first billed row falls in the period.

    Each new user creates one personal avatar. This is not advertising CAC;
    finance ``cac`` remains advertising spend divided by new users.
    """
    start, end = default_period(since, until)
    rows = await _fetchall(
        pool,
        """
        WITH first_seen AS (
            SELECT user_id, MIN(created_at) AS first_seen
            FROM api_metrics
            WHERE user_id IS NOT NULL
            GROUP BY user_id
        ),
        cohort AS (
            SELECT user_id
            FROM first_seen
            WHERE first_seen >= %s AND first_seen < %s
        )
        SELECT (SELECT COUNT(*) FROM cohort) AS new_users,
               ROUND(COALESCE(SUM(metrics.cost_usd), 0)::numeric, 4) AS cohort_cost_usd,
               CASE WHEN (SELECT COUNT(*) FROM cohort) = 0 THEN NULL
                    ELSE ROUND((SUM(metrics.cost_usd) / (SELECT COUNT(*) FROM cohort))::numeric, 4)
               END AS cost_per_new_user
        FROM api_metrics AS metrics
        WHERE metrics.user_id IN (SELECT user_id FROM cohort)
          AND metrics.created_at >= %s AND metrics.created_at < %s;
        """,
        (start, end, start, end),
    )
    result = _result(["new_users", "cohort_cost_usd", "cost_per_new_user"], rows)
    result["note"] = (
        f"{ADMIN_TRAFFIC_NOTE} This is product cost of onboarding, not advertising CAC."
    )
    return result


async def product_spend_by_day(
    pool: Any, since: datetime | None = None, until: datetime | None = None
) -> list[float]:
    """Return daily product spend oldest-first, for forecasts."""
    spend = await spend_by_period(pool, since, until, group_by="day")
    return [float(row[1] or 0.0) for row in spend.get("rows") or []]


async def _vendor_cost_usd(
    pool: Any, user_id: str, since: datetime, until: datetime
) -> dict[str, Any]:
    """Sum vendor cost and subscription rows for the owner in the period."""
    from src.anubis.utils.analytics.vendor_usage import usage_totals

    totals = await usage_totals(pool, user_id, since, until)
    vendor_cost = 0.0
    subscription_cost = 0.0
    development_overhead = 0.0
    for row in totals.get("rows") or []:
        provider = str(row[0] or "")
        metric = str(row[1] or "").strip().lower()
        unit = str(row[2] or "").strip().lower()
        value = float(row[3] or 0.0)
        if unit not in ("", "usd") and metric not in ("cost", "subscription"):
            continue
        if metric == "subscription":
            subscription_cost += value
        elif metric in ("cost", "amount_seen"):
            vendor_cost += value
        else:
            continue
        if provider in DEVELOPMENT_OVERHEAD_PROVIDERS:
            development_overhead += value
    return {
        "vendor_cost_usd": round(vendor_cost, 4),
        "subscription_cost_usd": round(subscription_cost, 4),
        "development_overhead_usd": round(development_overhead, 4),
        "vendor_totals": totals,
    }


async def _bank_outflow_usd(
    pool: Any, user_id: str, since: datetime, until: datetime
) -> float:
    """Sum Plaid outflows for the owner in the period."""
    from src.anubis.utils.analytics.finance import spend_by_period as finance_spend

    spend = await finance_spend(pool, user_id, since, until, group_by="day")
    return float(spend.get("total_spend_usd") or 0.0)


async def unit_economics(
    pool: Any,
    since: datetime | None = None,
    until: datetime | None = None,
    *,
    ledger: str = LEDGER_PRODUCT,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Return the four unit-cost numbers, optionally with a fully-loaded ledger.

    ``ledger=product`` is avatar serving cost from ``api_metrics``.
    ``ledger=fully_loaded`` adds the owner's vendor invoices and bank outflows.
    Cursor and Claude sit on the development-overhead line, not in product COGS.
    """
    start, end = default_period(since, until)
    chosen_ledger = str(ledger or LEDGER_PRODUCT).strip().lower()
    if chosen_ledger not in UNIT_ECONOMICS_LEDGERS:
        chosen_ledger = LEDGER_PRODUCT
    per_avatar = await cost_per_avatar(pool, start, end)
    per_message = await average_cost_per_message(pool, start, end)
    per_conversation = await average_cost_per_conversation(pool, start, end)
    per_new_user = await cost_per_new_user(pool, start, end)
    message_row = (per_message.get("rows") or [[0, 0.0, None]])[0]
    conversation_row = (per_conversation.get("rows") or [[0, 0.0, None]])[0]
    new_user_row = (per_new_user.get("rows") or [[0, 0.0, None]])[0]
    product_cost = float(per_avatar.get("total_cost_usd") or 0.0)
    payload: dict[str, Any] = {
        "ledger": LEDGER_PRODUCT,
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "product_cost_usd": product_cost,
        "avatars": per_avatar.get("avatars") or 0,
        "cost_per_avatar": per_avatar.get("cost_per_avatar"),
        "messages": message_row[0],
        "average_cost_per_message": message_row[2],
        "conversations": conversation_row[0],
        "average_cost_per_conversation": conversation_row[2],
        "new_users": new_user_row[0],
        "cost_per_new_user": new_user_row[2],
        "per_avatar": per_avatar,
        "note": ADMIN_TRAFFIC_NOTE,
    }
    if chosen_ledger != LEDGER_FULLY_LOADED:
        return payload
    vendor = (
        await _vendor_cost_usd(pool, str(user_id or ""), start, end)
        if user_id
        else {
            "vendor_cost_usd": 0.0,
            "subscription_cost_usd": 0.0,
            "development_overhead_usd": 0.0,
            "vendor_totals": {"rows": []},
        }
    )
    bank_outflow = (
        await _bank_outflow_usd(pool, str(user_id), start, end) if user_id else 0.0
    )
    fully_loaded = (
        product_cost
        + float(vendor["vendor_cost_usd"])
        + float(vendor["subscription_cost_usd"])
        + bank_outflow
    )
    avatars = int(payload["avatars"] or 0)
    messages = int(payload["messages"] or 0)
    conversations = int(payload["conversations"] or 0)
    new_users = int(payload["new_users"] or 0)
    payload.update(
        {
            "ledger": LEDGER_FULLY_LOADED,
            "vendor_cost_usd": vendor["vendor_cost_usd"],
            "subscription_cost_usd": vendor["subscription_cost_usd"],
            "development_overhead_usd": vendor["development_overhead_usd"],
            "bank_outflow_usd": round(bank_outflow, 4),
            "fully_loaded_cost_usd": round(fully_loaded, 4),
            "fully_loaded_cost_per_avatar": (
                round(fully_loaded / avatars, 4) if avatars else None
            ),
            "fully_loaded_cost_per_message": (
                round(fully_loaded / messages, 6) if messages else None
            ),
            "fully_loaded_cost_per_conversation": (
                round(fully_loaded / conversations, 6) if conversations else None
            ),
            "fully_loaded_cost_per_new_user": (
                round(fully_loaded / new_users, 4) if new_users else None
            ),
            "note": (
                f"{ADMIN_TRAFFIC_NOTE} Fully-loaded adds vendor invoices and bank "
                "outflows. Cursor and Claude sit on development_overhead_usd, not "
                "in product COGS."
            ),
        }
    )
    return payload


async def spend_by_period(
    pool: Any,
    since: datetime | None = None,
    until: datetime | None = None,
    group_by: str = "day",
) -> dict[str, Any]:
    """Sum model spend from ``api_metrics`` by day, model, or inference type."""
    start, end = default_period(since, until)
    grouping = str(group_by or "day").strip().lower()
    if grouping not in SPEND_GROUPINGS:
        grouping = "day"
    if grouping == "day":
        sql = """
        SELECT date_trunc('day', created_at)::date AS bucket,
               ROUND(SUM(cost_usd)::numeric, 4) AS cost_usd,
               SUM(total_tokens) AS total_tokens, COUNT(*) AS inferences
        FROM api_metrics
        WHERE created_at >= %s AND created_at < %s
        GROUP BY 1 ORDER BY 1;
        """
    elif grouping == "model":
        sql = """
        SELECT COALESCE(model_name, 'unknown') AS bucket,
               ROUND(SUM(cost_usd)::numeric, 4) AS cost_usd,
               SUM(total_tokens) AS total_tokens, COUNT(*) AS inferences
        FROM api_metrics
        WHERE created_at >= %s AND created_at < %s
        GROUP BY 1 ORDER BY cost_usd DESC;
        """
    else:
        sql = """
        SELECT inference_type AS bucket,
               ROUND(SUM(cost_usd)::numeric, 4) AS cost_usd,
               SUM(total_tokens) AS total_tokens, COUNT(*) AS inferences
        FROM api_metrics
        WHERE created_at >= %s AND created_at < %s
        GROUP BY 1 ORDER BY cost_usd DESC;
        """
    rows = await _fetchall(pool, sql, (start, end))
    return _result([grouping, "cost_usd", "total_tokens", "inferences"], rows)


def _summarise_feedback(values: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate feedback values into likes, dislikes, excerpts, and comments."""
    likes = 0
    dislikes = 0
    liked_excerpts: list[str] = []
    disliked_excerpts: list[str] = []
    comments: list[dict[str, Any]] = []
    for value in values:
        feedback_type = str(value.get("feedback_type") or "").strip().lower()
        excerpt = str(value.get("content_excerpt") or "").strip()
        comment = str(value.get("comment") or "").strip()
        if feedback_type == FEEDBACK_LIKE:
            likes += 1
            if excerpt:
                liked_excerpts.append(excerpt[:280])
        elif feedback_type == FEEDBACK_DISLIKE:
            dislikes += 1
            if excerpt:
                disliked_excerpts.append(excerpt[:280])
        if comment:
            comments.append(
                {
                    "feedback_type": feedback_type or None,
                    "comment": comment[:1000],
                    "excerpt": excerpt[:280] or None,
                    "recorded_at": value.get("recorded_at"),
                    "assistant_id": value.get("assistant_id"),
                }
            )
    comments.sort(key=lambda entry: str(entry.get("recorded_at") or ""), reverse=True)
    return {
        "likes": likes,
        "dislikes": dislikes,
        "top_liked_excerpts": liked_excerpts[:10],
        "top_disliked_excerpts": disliked_excerpts[:10],
        "comments": comments[:50],
        "note": "Feedback comments are the users' own words: feature requests and complaints live here.",
    }


async def feedback_summary(
    store: Any, *, user_id: str, assistant_id: str, limit: int = 500
) -> dict[str, Any]:
    """Summarise one user's feedback on one avatar from the LangGraph store."""
    items = await store.asearch(
        (user_id, assistant_id, FEEDBACK_NAMESPACE_SUFFIX), limit=max(1, int(limit))
    )
    values = []
    for item in items or []:
        value = getattr(item, "value", None)
        if isinstance(value, dict):
            values.append({**value, "assistant_id": assistant_id})
    return _summarise_feedback(values)


async def feedback_summary_all(pool: Any, limit: int = 500) -> dict[str, Any]:
    """Summarise every user's message feedback platform-wide (owner question: what users love and hate).

    The LangGraph ``store`` table keys rows by a dotted namespace prefix; the
    feedback namespace is ``<user_id>.<assistant_id>.message_feedback``.
    """
    rows = await _fetchall(
        pool,
        """
        SELECT prefix, value
        FROM store
        WHERE split_part(prefix, '.', 3) = %s
        ORDER BY value->>'recorded_at' DESC NULLS LAST
        LIMIT %s;
        """,
        (FEEDBACK_NAMESPACE_SUFFIX, max(1, int(limit))),
    )
    values = []
    for prefix, value in rows:
        if isinstance(value, dict):
            parts = str(prefix or "").split(".")
            values.append(
                {**value, "assistant_id": parts[1] if len(parts) > 1 else None}
            )
    summary = _summarise_feedback(values)
    summary["note"] = f"{summary['note']} {ADMIN_TRAFFIC_NOTE}"
    return summary


def _monthly_amount_usd(item: dict[str, Any]) -> float:
    """Return one subscription item's recurring amount normalised to a month."""
    price = item.get("price") or {}
    unit_amount = price.get("unit_amount")
    if unit_amount is None:
        unit_amount_decimal = price.get("unit_amount_decimal")
        unit_amount = float(unit_amount_decimal) if unit_amount_decimal else 0.0
    quantity = int(item.get("quantity") or 1)
    recurring = price.get("recurring") or {}
    interval = str(recurring.get("interval") or "month")
    interval_count = int(recurring.get("interval_count") or 1)
    amount = float(unit_amount) * quantity / CENTS_PER_DOLLAR
    if interval == "year":
        return amount / (MONTHS_PER_YEAR * interval_count)
    if interval == "week":
        return amount * 52.0 / MONTHS_PER_YEAR / interval_count
    if interval == "day":
        return amount * 365.0 / MONTHS_PER_YEAR / interval_count
    return amount / interval_count


def _subscription_as_dict(subscription: Any) -> dict[str, Any]:
    """Return a Stripe object (or a plain dictionary) as a dictionary."""
    if isinstance(subscription, dict):
        return subscription
    to_dict = getattr(subscription, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return dict(subscription)


async def revenue_estimate(
    context: Any, app_state: Any, *, stripe_module: Any = None
) -> dict[str, Any]:
    """Estimate monthly recurring revenue from active Stripe subscriptions.

    Price ids are mapped to tiers through the provisioned billing
    configuration so the result says how many subscriptions each tier holds.
    """
    from src.anubis.utils.billing.config import current_stripe_billing_config

    if stripe_module is None:
        import stripe as stripe_module  # noqa: PLC0415 - lazy heavy import

    api_key = getattr(context, "stripe_secret_key", None) or getattr(
        context, "stripe_api_key", None
    )
    if api_key and not getattr(stripe_module, "api_key", None):
        stripe_module.api_key = api_key

    price_to_tier: dict[str, str] = {}
    try:
        billing_config = current_stripe_billing_config(app_state)
    except Exception:  # noqa: BLE001 - a missing config still allows a raw estimate
        billing_config = None
    for tier, identifiers in ((billing_config.tiers if billing_config else {}) or {}).items():
        tier_name = getattr(tier, "value", str(tier))
        base_price_id = getattr(identifiers, "base_price_id", None)
        if base_price_id:
            price_to_tier[str(base_price_id)] = tier_name
        for price_id in (getattr(identifiers, "metered_price_ids", None) or {}).values():
            price_to_tier[str(price_id)] = tier_name

    active_subscriptions = 0
    monthly_recurring_revenue = 0.0
    by_tier: dict[str, dict[str, float | int]] = {}
    listing = stripe_module.Subscription.list(status="active", limit=100)
    for raw_subscription in listing.auto_paging_iter():
        subscription = _subscription_as_dict(raw_subscription)
        active_subscriptions += 1
        items = ((subscription.get("items") or {}).get("data")) or []
        subscription_tier = "unknown"
        subscription_amount = 0.0
        for raw_item in items:
            item = _subscription_as_dict(raw_item)
            price_id = str(((item.get("price") or {}).get("id")) or "")
            if price_id in price_to_tier and subscription_tier == "unknown":
                subscription_tier = price_to_tier[price_id]
            subscription_amount += _monthly_amount_usd(item)
        monthly_recurring_revenue += subscription_amount
        tier_entry = by_tier.setdefault(
            subscription_tier, {"subscriptions": 0, "monthly_recurring_revenue_usd": 0.0}
        )
        tier_entry["subscriptions"] = int(tier_entry["subscriptions"]) + 1
        tier_entry["monthly_recurring_revenue_usd"] = round(
            float(tier_entry["monthly_recurring_revenue_usd"]) + subscription_amount, 2
        )
    return {
        "active_subscriptions": active_subscriptions,
        "monthly_recurring_revenue_usd": round(monthly_recurring_revenue, 2),
        "by_tier": by_tier,
        "note": "Recurring amounts only; metered usage charges are billed in arrears and excluded.",
    }


def is_platform_admin(
    context: Any, connected_accounts: list[dict[str, Any]] | None, user_id: str | None
) -> bool:
    """Return whether platform metrics may be shown to this owner.

    True only for the configured platform administrator (``ADMIN_USER_ID``).
    Platform-wide numbers describe every user, so no connected account can
    grant them; the ``connected_accounts`` argument is kept for the call
    sites and is not consulted.
    """
    admin_user_id = str(getattr(context, "admin_user_id", None) or "").strip()
    return bool(admin_user_id and user_id and str(user_id).strip() == admin_user_id)
