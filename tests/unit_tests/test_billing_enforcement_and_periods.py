"""Unit tests for allotment enforcement, usage periods, rate limits, and tier changes.

Covers the pure decision logic added for full-tier metering enforcement: the
exhausted-allotment block matrix across tiers and pay-per-use settings, the
configurable usage-period computation (calendar month, anchored month with
day-of-month clamping, fixed-length windows), the token rate-limit Retry-After
math, the upgrade/downgrade tier-change planner, explicit pay-per-use flag
resolution, the payment-method presence check, and the adapter-training
metering helper's Stripe payload and api_metrics row.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.anubis.utils.billing.gating import (
    TrialContext,
    customer_has_payment_method,
    exhausted_allotment_block_reason,
    plan_tier_change,
    resolve_effective_monthly_allotment,
    resolve_pay_per_use_enabled,
    resolve_usage_period_anchor,
)
from src.anubis.utils.billing.metering import (
    GLOBAL_USAGE_PERIOD_ANCHOR,
    report_adapter_training_usage,
    resolve_usage_period_end,
    resolve_usage_period_start,
    token_rate_limit_retry_after_seconds,
)
from src.anubis.utils.billing.tiers import (
    TIER_DEFINITIONS,
    SubscriptionTier,
    UsageMeter,
    tier_allotment_for_meter,
)

# ---------------------------------------------------------------------------
# Exhausted-allotment block matrix: tiers x pay-per-use x usage position
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tier", list(SubscriptionTier))
@pytest.mark.parametrize("pay_per_use_enabled", [True, False])
def test_usage_under_allotment_is_always_allowed(tier, pay_per_use_enabled):
    allotment = tier_allotment_for_meter(tier, UsageMeter.MESSAGING_TOKENS)
    assert allotment is not None
    reason = exhausted_allotment_block_reason(
        tier,
        UsageMeter.MESSAGING_TOKENS,
        allotment.monthly_allotment - 1,
        pay_per_use_enabled,
    )
    assert reason is None


@pytest.mark.parametrize("tier", list(SubscriptionTier))
def test_usage_at_allotment_is_allowed_only_with_pay_per_use(tier):
    allotment = tier_allotment_for_meter(tier, UsageMeter.MESSAGING_TOKENS)
    assert allotment is not None
    allowed_reason = exhausted_allotment_block_reason(
        tier, UsageMeter.MESSAGING_TOKENS, allotment.monthly_allotment, True
    )
    assert allowed_reason is None
    blocked_reason = exhausted_allotment_block_reason(
        tier, UsageMeter.MESSAGING_TOKENS, allotment.monthly_allotment, False
    )
    assert blocked_reason is not None
    assert f"{allotment.monthly_allotment:,}" in blocked_reason


def test_block_reason_guides_free_users_to_subscribe_and_paid_users_to_pay_per_use():
    free_allotment = tier_allotment_for_meter(
        SubscriptionTier.FREE, UsageMeter.MESSAGING_TOKENS
    )
    free_reason = exhausted_allotment_block_reason(
        SubscriptionTier.FREE,
        UsageMeter.MESSAGING_TOKENS,
        free_allotment.monthly_allotment,
        False,
    )
    assert "Subscribe" in free_reason

    pro_allotment = tier_allotment_for_meter(
        SubscriptionTier.PRO, UsageMeter.MESSAGING_TOKENS
    )
    pro_reason = exhausted_allotment_block_reason(
        SubscriptionTier.PRO,
        UsageMeter.MESSAGING_TOKENS,
        pro_allotment.monthly_allotment,
        False,
    )
    assert "/set_pay_per_use" in pro_reason


def test_missing_meter_dimension_is_not_decided_by_the_allotment_gate():
    # The capability gate is the authority for meters a tier lacks entirely.
    assert (
        exhausted_allotment_block_reason(
            SubscriptionTier.FREE,
            UsageMeter.DOCUMENT_UPLOAD_TOKENS,
            10_000_000,
            False,
        )
        is None
    )


# ---------------------------------------------------------------------------
# Usage-period computation
# ---------------------------------------------------------------------------


def test_default_period_is_the_first_of_the_current_utc_month():
    now = datetime(2026, 7, 13, 15, 30, tzinfo=timezone.utc)
    assert resolve_usage_period_start(now, 0) == datetime(
        2026, 7, 1, tzinfo=timezone.utc
    )


def test_anchored_month_uses_the_anchor_day_of_month():
    anchor = datetime(2026, 3, 10, 8, 0, tzinfo=timezone.utc)
    now = datetime(2026, 7, 13, tzinfo=timezone.utc)
    assert resolve_usage_period_start(now, 0, anchor) == datetime(
        2026, 7, 10, 8, 0, tzinfo=timezone.utc
    )
    # Before this month's boundary the period started last month.
    earlier_now = datetime(2026, 7, 5, tzinfo=timezone.utc)
    assert resolve_usage_period_start(earlier_now, 0, anchor) == datetime(
        2026, 6, 10, 8, 0, tzinfo=timezone.utc
    )


def test_anchored_month_clamps_day_31_to_short_months():
    anchor = datetime(2026, 1, 31, 12, 0, tzinfo=timezone.utc)
    now = datetime(2026, 3, 1, tzinfo=timezone.utc)
    # February 2026 has 28 days, so the boundary clamps to February 28.
    assert resolve_usage_period_start(now, 0, anchor) == datetime(
        2026, 2, 28, 12, 0, tzinfo=timezone.utc
    )


def test_first_period_never_starts_before_the_anchor_itself():
    anchor = datetime(2026, 7, 10, tzinfo=timezone.utc)
    now = datetime(2026, 7, 12, tzinfo=timezone.utc)
    assert resolve_usage_period_start(now, 0, anchor) == anchor


def test_fixed_length_windows_are_deterministic_from_the_global_anchor():
    now = datetime(2026, 7, 13, tzinfo=timezone.utc)
    period_start = resolve_usage_period_start(now, 7)
    assert (period_start - GLOBAL_USAGE_PERIOD_ANCHOR) % timedelta(days=7) == timedelta(0)
    assert period_start <= now < period_start + timedelta(days=7)


def test_fixed_length_windows_count_from_a_personal_anchor():
    anchor = datetime(2026, 7, 1, tzinfo=timezone.utc)
    now = datetime(2026, 7, 16, tzinfo=timezone.utc)
    assert resolve_usage_period_start(now, 7, anchor) == datetime(
        2026, 7, 15, tzinfo=timezone.utc
    )


def test_period_end_is_exclusive_and_matches_the_period_shape():
    month_start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    assert resolve_usage_period_end(month_start, 0) == datetime(
        2026, 8, 1, tzinfo=timezone.utc
    )
    week_start = datetime(2026, 7, 15, tzinfo=timezone.utc)
    assert resolve_usage_period_end(week_start, 7) == datetime(
        2026, 7, 22, tzinfo=timezone.utc
    )


@pytest.mark.parametrize(
    "raw_anchor, expected",
    [
        (None, None),
        (123, None),
        ("not-a-date", None),
        (
            "2026-07-05T10:00:00+00:00",
            datetime(2026, 7, 5, 10, 0, tzinfo=timezone.utc),
        ),
        ("2026-07-05T10:00:00Z", datetime(2026, 7, 5, 10, 0, tzinfo=timezone.utc)),
    ],
)
def test_resolve_usage_period_anchor_parses_defensively(raw_anchor, expected):
    user = {"app_metadata": {"usage_period_anchor": raw_anchor}}
    assert resolve_usage_period_anchor(user) == expected


# ---------------------------------------------------------------------------
# Token rate-limit Retry-After math
# ---------------------------------------------------------------------------


def test_rate_limit_is_disabled_when_the_cap_is_zero():
    assert token_rate_limit_retry_after_seconds(999_999, 0, 60, None) is None


def test_usage_under_the_cap_is_allowed():
    assert token_rate_limit_retry_after_seconds(99, 100, 60, None) is None


def test_usage_at_the_cap_without_a_timestamp_waits_a_full_window():
    assert token_rate_limit_retry_after_seconds(100, 100, 60, None) == 60


def test_retry_after_counts_down_as_the_oldest_row_ages():
    now = datetime(2026, 7, 13, 12, 0, 0, tzinfo=timezone.utc)
    oldest_usage_at = now - timedelta(seconds=45)
    retry_after = token_rate_limit_retry_after_seconds(
        150, 100, 60, oldest_usage_at, now=now
    )
    # The oldest row exits the 60-second window 15 seconds from now.
    assert retry_after == 16


def test_retry_after_is_clamped_between_one_second_and_the_window():
    now = datetime(2026, 7, 13, 12, 0, 0, tzinfo=timezone.utc)
    already_expired = now - timedelta(seconds=600)
    assert (
        token_rate_limit_retry_after_seconds(150, 100, 60, already_expired, now=now)
        == 1
    )
    skewed_future = now + timedelta(seconds=600)
    assert (
        token_rate_limit_retry_after_seconds(150, 100, 60, skewed_future, now=now)
        == 60
    )


# ---------------------------------------------------------------------------
# Tier-change planner: cleared on upgrade, retained on downgrade
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "current_tier, target_tier",
    [
        (SubscriptionTier.FREE, SubscriptionTier.PRO),
        (SubscriptionTier.FREE, SubscriptionTier.PREMIUM),
        (SubscriptionTier.PRO, SubscriptionTier.PREMIUM),
    ],
)
def test_upgrades_swap_immediately_and_reset_the_usage_anchor(
    current_tier, target_tier
):
    plan = plan_tier_change(current_tier, target_tier)
    assert plan.direction == "upgrade"
    assert plan.swap_items_immediately
    assert not plan.schedule_change_at_period_end
    assert plan.reset_usage_period_anchor


@pytest.mark.parametrize(
    "current_tier, target_tier",
    [
        (SubscriptionTier.PREMIUM, SubscriptionTier.PRO),
        (SubscriptionTier.PREMIUM, SubscriptionTier.FREE),
        (SubscriptionTier.PRO, SubscriptionTier.FREE),
    ],
)
def test_downgrades_defer_to_the_period_end_and_keep_the_usage_anchor(
    current_tier, target_tier
):
    plan = plan_tier_change(current_tier, target_tier)
    assert plan.direction == "downgrade"
    assert not plan.swap_items_immediately
    assert plan.schedule_change_at_period_end
    assert not plan.reset_usage_period_anchor


# ---------------------------------------------------------------------------
# Pay-per-use flag resolution and payment-method presence
# ---------------------------------------------------------------------------


def test_explicit_pay_per_use_flag_overrides_the_inferred_status():
    active_but_disabled = {
        "email": "person@example.com",
        "app_metadata": {
            "pay_per_use_enabled": False,
            "subscription_status": {"status": "active"},
        },
    }
    assert resolve_pay_per_use_enabled(active_but_disabled) is False

    trialing_but_enabled = {
        "email": "person@example.com",
        "app_metadata": {
            "pay_per_use_enabled": True,
            "subscription_status": {"status": "trialing"},
        },
    }
    assert resolve_pay_per_use_enabled(trialing_but_enabled) is True


def test_pay_per_use_is_inferred_from_active_status_and_denied_while_trialing():
    active_user = {
        "email": "person@example.com",
        "app_metadata": {"subscription_status": {"status": "active"}},
    }
    assert resolve_pay_per_use_enabled(active_user) is True
    trialing_user = {
        "email": "person@example.com",
        "app_metadata": {"subscription_status": {"status": "trialing"}},
    }
    assert resolve_pay_per_use_enabled(trialing_user) is False


@pytest.mark.parametrize(
    "customer_document, payment_methods, expected",
    [
        (None, None, False),
        ({}, [], False),
        ({"invoice_settings": {"default_payment_method": "pm_1"}}, [], True),
        ({"default_source": "card_1"}, [], True),
        ({}, [{"id": "pm_2"}], True),
    ],
)
def test_customer_has_payment_method(customer_document, payment_methods, expected):
    assert customer_has_payment_method(customer_document, payment_methods) is expected


# ---------------------------------------------------------------------------
# Adapter-training metering helper
# ---------------------------------------------------------------------------


class _RecordingMeterEvent:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)


class _FakeStripeClient:
    def __init__(self):
        class _Billing:
            pass

        self.billing = _Billing()
        self.billing.MeterEvent = _RecordingMeterEvent()


class _RecordingCursor:
    def __init__(self, executed):
        self._executed = executed

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exception_info):
        return False

    async def execute(self, sql, parameters=None):
        self._executed.append((sql, parameters))


class _RecordingConnection:
    def __init__(self, executed):
        self._executed = executed

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exception_info):
        return False

    def cursor(self):
        return _RecordingCursor(self._executed)


class _RecordingPool:
    def __init__(self):
        self.executed = []

    def connection(self):
        return _RecordingConnection(self.executed)


@pytest.mark.asyncio
async def test_report_adapter_training_usage_reports_meter_and_persists_row():
    stripe_client = _FakeStripeClient()
    pool = _RecordingPool()
    accepted = await report_adapter_training_usage(
        stripe_client,
        pool,
        stripe_customer_id="cus_123",
        metering_user_id="user-1",
        trained_adapter_count=2,
        assistant_id="assistant-1",
        idempotency_identifier="job-1:adapter_training_units",
    )
    assert accepted is True
    (meter_call,) = stripe_client.billing.MeterEvent.calls
    assert meter_call["event_name"] == UsageMeter.ADAPTER_TRAINING_UNITS.value
    assert meter_call["payload"] == {"stripe_customer_id": "cus_123", "value": "2"}
    assert meter_call["identifier"] == "job-1:adapter_training_units"
    (insert_call,) = pool.executed
    insert_parameters = insert_call[1]
    assert "adapter_training" in insert_parameters
    assert UsageMeter.ADAPTER_TRAINING_UNITS.value in insert_parameters
    # The unit count rides in total_tokens so period-usage sums stay uniform.
    assert 2 in insert_parameters


@pytest.mark.asyncio
async def test_report_adapter_training_usage_no_ops_on_zero_count():
    stripe_client = _FakeStripeClient()
    pool = _RecordingPool()
    accepted = await report_adapter_training_usage(
        stripe_client,
        pool,
        stripe_customer_id="cus_123",
        metering_user_id="user-1",
        trained_adapter_count=0,
    )
    assert accepted is False
    assert stripe_client.billing.MeterEvent.calls == []
    assert pool.executed == []


# ---------------------------------------------------------------------------
# Tier catalog sanity for the free pay-per-use vehicle
# ---------------------------------------------------------------------------


def test_free_tier_has_an_overage_rate_for_the_pay_per_use_vehicle():
    # The free $0 subscription exists solely to bill overage past the free
    # allotment, so the free messaging allotment must carry an overage rate.
    free_messaging = TIER_DEFINITIONS[SubscriptionTier.FREE].meter_allotments[
        UsageMeter.MESSAGING_TOKENS
    ]
    assert (
        free_messaging.overage_price_per_million is not None
        or free_messaging.overage_price_per_unit_usd is not None
    )


# ---------------------------------------------------------------------------
# Trial allotment floor: a mid-trial tier change keeps the higher trial
# allotment until trial_end (resolve_effective_monthly_allotment).
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 7, 20, tzinfo=timezone.utc)
_TRIAL_ACTIVE = _NOW + timedelta(days=5)
_TRIAL_LAPSED = _NOW - timedelta(days=1)


def test_effective_allotment_without_a_trial_is_the_plain_tier_allotment():
    effective = resolve_effective_monthly_allotment(
        SubscriptionTier.PRO, UsageMeter.MESSAGING_TOKENS, None, now=_NOW
    )
    assert effective == tier_allotment_for_meter(
        SubscriptionTier.PRO, UsageMeter.MESSAGING_TOKENS
    )


def test_trialing_premium_then_downgrade_to_pro_keeps_the_premium_floor():
    # Downgraded to pro (5M messaging) mid-trial, but the premium trial (20M)
    # is still running, so the premium allotment governs until trial_end.
    trial_context = TrialContext(
        trial_tier=SubscriptionTier.PREMIUM, trial_end=_TRIAL_ACTIVE
    )
    effective = resolve_effective_monthly_allotment(
        SubscriptionTier.PRO, UsageMeter.MESSAGING_TOKENS, trial_context, now=_NOW
    )
    assert effective.monthly_allotment == 20_000_000


def test_effective_allotment_drops_to_the_current_tier_after_trial_end():
    trial_context = TrialContext(
        trial_tier=SubscriptionTier.PREMIUM, trial_end=_TRIAL_LAPSED
    )
    effective = resolve_effective_monthly_allotment(
        SubscriptionTier.PRO, UsageMeter.MESSAGING_TOKENS, trial_context, now=_NOW
    )
    assert effective.monthly_allotment == 5_000_000


def test_trial_only_meter_stays_granted_during_the_window():
    # Trialing pro (grants document uploads) then downgraded to free (no upload
    # meter): the upload allotment stays granted until the trial ends.
    trial_context = TrialContext(
        trial_tier=SubscriptionTier.PRO, trial_end=_TRIAL_ACTIVE
    )
    effective = resolve_effective_monthly_allotment(
        SubscriptionTier.FREE,
        UsageMeter.DOCUMENT_UPLOAD_TOKENS,
        trial_context,
        now=_NOW,
    )
    assert effective is not None
    assert effective.monthly_allotment == 10_000_000


def test_trial_floor_flows_through_the_block_reason_via_override():
    # Usage sits above the plain pro allotment (5M) but under the premium trial
    # floor (20M); with the override the request is allowed, without it blocked.
    trial_context = TrialContext(
        trial_tier=SubscriptionTier.PREMIUM, trial_end=_TRIAL_ACTIVE
    )
    override = resolve_effective_monthly_allotment(
        SubscriptionTier.PRO, UsageMeter.MESSAGING_TOKENS, trial_context, now=_NOW
    )
    usage_between = 6_000_000
    assert (
        exhausted_allotment_block_reason(
            SubscriptionTier.PRO,
            UsageMeter.MESSAGING_TOKENS,
            usage_between,
            pay_per_use_enabled=False,
            allotment_override=override,
        )
        is None
    )
    assert (
        exhausted_allotment_block_reason(
            SubscriptionTier.PRO,
            UsageMeter.MESSAGING_TOKENS,
            usage_between,
            pay_per_use_enabled=False,
        )
        is not None
    )
