"""Platform metrics: SQL shape, parameters, the admin gate, feedback, and revenue."""

from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest

from src.anubis.utils.analytics import platform_metrics
from src.anubis.utils.analytics.platform_metrics import (
    ADMIN_TRAFFIC_NOTE,
    feedback_summary,
    feedback_summary_all,
    is_platform_admin,
    revenue_estimate,
)
from src.anubis.utils.billing.config import (
    BILLING_CONFIG_STATE_ATTRIBUTE,
    StripeBillingConfig,
    TierStripeIdentifiers,
)
from src.anubis.utils.billing.tiers import SubscriptionTier


class _FakeCursor:
    def __init__(self, pool):
        self.pool = pool

    async def execute(self, statement, params=None, *, prepare=None):
        self.pool.calls.append((" ".join(statement.split()), params))

    async def fetchall(self):
        return list(self.pool.rows)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeConnection:
    def __init__(self, pool):
        self.pool = pool

    def cursor(self):
        return _FakeCursor(self.pool)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self, rows=None):
        self.calls = []
        self.rows = rows or []

    def connection(self):
        return _FakeConnection(self)


SINCE = datetime(2026, 8, 1, tzinfo=UTC)
UNTIL = datetime(2026, 9, 1, tzinfo=UTC)


@pytest.mark.asyncio
async def test_messages_per_day_shape_and_parameters():
    pool = _FakePool(rows=[(date(2026, 8, 1), 12), (date(2026, 8, 2), 7)])
    result = await platform_metrics.messages_per_day(pool, SINCE, UNTIL)
    statement, params = pool.calls[-1]
    assert "FROM api_metrics" in statement
    assert "inference_type = ANY(%s)" in statement
    assert params == (["message", "adapter_inference"], SINCE, UNTIL)
    assert result["columns"] == ["day", "messages"]
    assert result["rows"] == [["2026-08-01", 12], ["2026-08-02", 7]]
    assert result["note"] == ADMIN_TRAFFIC_NOTE


@pytest.mark.asyncio
async def test_messages_per_user_per_day_averages_per_user():
    pool = _FakePool(rows=[(date(2026, 8, 1), 3, 12, 4.0)])
    result = await platform_metrics.messages_per_user_per_day(pool, SINCE, UNTIL)
    statement, _ = pool.calls[-1]
    assert "COUNT(DISTINCT user_id) AS active_users" in statement
    assert "AVG(messages)" in statement
    assert result["columns"] == ["day", "active_users", "messages", "messages_per_user"]


@pytest.mark.asyncio
async def test_average_conversation_length_uses_thread_wall_clock_and_percentiles():
    pool = _FakePool(rows=[(10, 4.5, 4.0, 12.25, 9.0)])
    result = await platform_metrics.average_conversation_length(pool, SINCE, UNTIL)
    statement, params = pool.calls[-1]
    assert "FROM thread" in statement
    assert "updated_at - created_at" in statement
    assert "percentile_cont(0.5)" in statement
    assert params == (["message", "adapter_inference"], SINCE, UNTIL, SINCE, UNTIL)
    assert result["rows"] == [[10, 4.5, 4.0, 12.25, 9.0]]


@pytest.mark.asyncio
async def test_avatars_by_conversation_count_joins_assistant_name():
    pool = _FakePool(rows=[("avatar-1", "Pastor", 9, 4, 30)])
    result = await platform_metrics.avatars_by_conversation_count(pool, SINCE, UNTIL, limit=3)
    statement, params = pool.calls[-1]
    assert "LEFT JOIN assistant ON assistant.assistant_id::text = metrics.assistant_id" in statement
    assert params[-1] == 3
    assert result["rows"][0][1] == "Pastor"


@pytest.mark.asyncio
async def test_feature_usage_unions_tools_inferences_and_connections():
    pool = _FakePool(rows=[("avatar-1", "tool", "make_chart", 5)])
    result = await platform_metrics.feature_usage_per_avatar(pool, SINCE, UNTIL, assistant_id="avatar-1")
    statement, params = pool.calls[-1]
    assert statement.count("UNION ALL") == 2
    assert "FROM tool_calls" in statement
    assert "FROM connected_accounts" in statement
    assert params == (SINCE, UNTIL, SINCE, UNTIL, "avatar-1", "avatar-1")
    assert result["columns"] == ["assistant_id", "source", "feature", "uses"]


@pytest.mark.asyncio
async def test_growth_active_users_and_spend_groupings():
    pool = _FakePool(rows=[(date(2026, 8, 3), 4)])
    growth = await platform_metrics.first_seen_users_per_week(pool, SINCE, UNTIL)
    assert "MIN(created_at) AS first_seen" in pool.calls[-1][0]
    assert growth["columns"] == ["week", "new_users"]

    pool.rows = [(5, 8, 40)]
    active = await platform_metrics.active_users(pool, SINCE, UNTIL)
    assert active["rows"] == [[5, 8, 40]]

    pool.rows = [("gpt-5", 1.5, 1000, 3)]
    by_model = await platform_metrics.spend_by_period(pool, SINCE, UNTIL, group_by="model")
    assert "COALESCE(model_name, 'unknown')" in pool.calls[-1][0]
    assert by_model["columns"][0] == "model"
    by_default = await platform_metrics.spend_by_period(pool, SINCE, UNTIL, group_by="; drop")
    assert by_default["columns"][0] == "day"


def test_is_platform_admin_requires_both_the_connection_and_the_admin_id():
    context = SimpleNamespace(admin_user_id="admin")
    assert is_platform_admin(context, [{"provider": "neural_nexus"}], "admin") is True
    assert is_platform_admin(context, [{"provider": "neural_nexus"}], "someone") is False
    assert is_platform_admin(context, [{"provider": "gmail"}], "admin") is False
    assert is_platform_admin(context, [], "admin") is False
    assert is_platform_admin(SimpleNamespace(admin_user_id=None), [{"provider": "neural_nexus"}], "admin") is False


@pytest.mark.asyncio
async def test_feedback_summary_from_the_store():
    class _Item:
        def __init__(self, value):
            self.value = value

    class _Store:
        async def asearch(self, namespace, *, limit=10, **kwargs):
            assert namespace == ("owner", "avatar", "message_feedback")
            return [
                _Item({"feedback_type": "like", "content_excerpt": "Great prayer", "comment": None, "recorded_at": "2026-09-01"}),
                _Item({"feedback_type": "dislike", "content_excerpt": "Too long", "comment": "Please add a summary", "recorded_at": "2026-09-02"}),
                _Item({"feedback_type": "like", "content_excerpt": "", "comment": "More charts please", "recorded_at": "2026-09-03"}),
            ]

    summary = await feedback_summary(_Store(), user_id="owner", assistant_id="avatar")
    assert summary["likes"] == 2
    assert summary["dislikes"] == 1
    assert summary["top_liked_excerpts"] == ["Great prayer"]
    assert summary["top_disliked_excerpts"] == ["Too long"]
    assert [entry["comment"] for entry in summary["comments"]] == ["More charts please", "Please add a summary"]


@pytest.mark.asyncio
async def test_feedback_summary_all_reads_the_store_table_by_prefix():
    pool = _FakePool(
        rows=[
            ("u1.a1.message_feedback", {"feedback_type": "like", "content_excerpt": "Nice", "recorded_at": "2026-09-01"}),
            ("u2.a2.message_feedback", {"feedback_type": "dislike", "comment": "Slow", "recorded_at": "2026-09-02"}),
        ]
    )
    summary = await feedback_summary_all(pool, limit=50)
    statement, params = pool.calls[-1]
    assert "FROM store" in statement
    assert "split_part(prefix, '.', 3) = %s" in statement
    assert params == ("message_feedback", 50)
    assert summary["likes"] == 1 and summary["dislikes"] == 1
    assert summary["comments"][0]["assistant_id"] == "a2"
    assert ADMIN_TRAFFIC_NOTE in summary["note"]


class _FakeListing:
    def __init__(self, subscriptions):
        self.subscriptions = subscriptions

    def auto_paging_iter(self):
        return iter(self.subscriptions)


class _FakeSubscriptionResource:
    calls = []

    @classmethod
    def list(cls, **kwargs):
        cls.calls.append(kwargs)
        return _FakeListing(
            [
                {"id": "sub_1", "items": {"data": [
                    {"quantity": 1, "price": {"id": "price_pro", "unit_amount": 2000, "recurring": {"interval": "month"}}},
                    {"quantity": 1, "price": {"id": "price_pro_meter", "unit_amount": 0, "recurring": {"interval": "month"}}},
                ]}},
                {"id": "sub_2", "items": {"data": [
                    {"quantity": 1, "price": {"id": "price_premium", "unit_amount": 120000, "recurring": {"interval": "year"}}},
                ]}},
                {"id": "sub_3", "items": {"data": [
                    {"quantity": 2, "price": {"id": "price_other", "unit_amount": 500, "recurring": {"interval": "month"}}},
                ]}},
            ]
        )


@pytest.mark.asyncio
async def test_revenue_estimate_maps_prices_to_tiers_with_a_fake_stripe_module():
    billing_config = StripeBillingConfig(
        meter_ids={},
        tiers={
            SubscriptionTier.PRO: TierStripeIdentifiers(base_price_id="price_pro", metered_price_ids={}),
            SubscriptionTier.PREMIUM: TierStripeIdentifiers(base_price_id="price_premium", metered_price_ids={}),
        },
    )
    app_state = SimpleNamespace(context=None, **{BILLING_CONFIG_STATE_ATTRIBUTE: billing_config})
    stripe_module = SimpleNamespace(Subscription=_FakeSubscriptionResource, api_key=None)
    context = SimpleNamespace(stripe_secret_key="sk_test")
    result = await revenue_estimate(context, app_state, stripe_module=stripe_module)
    assert _FakeSubscriptionResource.calls[-1] == {"status": "active", "limit": 100}
    assert stripe_module.api_key == "sk_test"
    assert result["active_subscriptions"] == 3
    assert result["monthly_recurring_revenue_usd"] == 20.0 + 100.0 + 10.0
    assert result["by_tier"]["pro"] == {"subscriptions": 1, "monthly_recurring_revenue_usd": 20.0}
    assert result["by_tier"]["premium"]["monthly_recurring_revenue_usd"] == 100.0
    assert result["by_tier"]["unknown"]["subscriptions"] == 1
