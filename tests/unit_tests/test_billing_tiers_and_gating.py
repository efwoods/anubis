"""Unit tests for the Stripe metering billing package.

Covers the tier catalog invariants (capability inheritance, four independent
meters, per-tier allotments), defensive tier coercion, billing-config JSON
parsing, anonymous-user resolution and metering-identifier resolution, the
Stripe meter-event helper's no-op and failure paths, the upload token estimate,
and the graduated-price decimal math used by the provisioning script.
"""

import json
from datetime import UTC, datetime, timedelta

import pytest

from src.anubis.utils.billing.config import load_stripe_billing_config
from src.anubis.utils.billing.gating import (
    SubscribeAction,
    customer_has_payment_method,
    exhausted_allotment_block_reason,
    is_anonymous_user,
    plan_subscribe_action,
    plan_tier_change,
    resolve_effective_monthly_allotment,
    resolve_metering_user_id,
    resolve_pay_per_use_enabled,
    resolve_stripe_customer_id,
    resolve_tier,
    resolve_trial_context,
    resolve_use_adapter_inference,
    user_has_capability,
)
from src.anubis.utils.billing.metering import (
    billable_tokens_from_metadata,
    report_meter_event,
)
from src.anubis.utils.billing.tiers import (
    TIER_DEFINITIONS,
    MeterAllotment,
    SubscriptionTier,
    TierCapability,
    UsageMeter,
    tier_allotment_for_meter,
    tier_from_value,
    tier_has_capability,
)

# ---------------------------------------------------------------------------
# Tier catalog invariants
# ---------------------------------------------------------------------------


def test_every_tier_is_defined_with_a_messaging_allotment():
    for tier in SubscriptionTier:
        definition = TIER_DEFINITIONS[tier]
        assert definition.tier == tier
        assert UsageMeter.MESSAGING_TOKENS in definition.meter_allotments


def test_capabilities_inherit_upward_across_tiers():
    assert tier_has_capability(SubscriptionTier.FREE, TierCapability.MESSAGE)
    assert not tier_has_capability(SubscriptionTier.FREE, TierCapability.UPLOAD)
    assert tier_has_capability(SubscriptionTier.PRO, TierCapability.MESSAGE)
    assert tier_has_capability(SubscriptionTier.PRO, TierCapability.UPLOAD)
    assert not tier_has_capability(
        SubscriptionTier.PRO, TierCapability.TRAIN_ADAPTER
    )
    for capability in TierCapability:
        assert tier_has_capability(SubscriptionTier.PREMIUM, capability)


def test_meter_dimensions_grow_with_tier():
    free_meters = set(TIER_DEFINITIONS[SubscriptionTier.FREE].meter_allotments)
    pro_meters = set(TIER_DEFINITIONS[SubscriptionTier.PRO].meter_allotments)
    premium_meters = set(
        TIER_DEFINITIONS[SubscriptionTier.PREMIUM].meter_allotments
    )
    assert free_meters == {UsageMeter.MESSAGING_TOKENS}
    assert pro_meters == free_meters | {UsageMeter.DOCUMENT_UPLOAD_TOKENS}
    assert premium_meters == set(UsageMeter)


def test_premium_allotments_exceed_pro_allotments():
    pro = TIER_DEFINITIONS[SubscriptionTier.PRO].meter_allotments
    premium = TIER_DEFINITIONS[SubscriptionTier.PREMIUM].meter_allotments
    for meter in pro:
        assert (
            premium[meter].monthly_allotment > pro[meter].monthly_allotment
        ), f"premium allotment for {meter.value} should exceed pro's"


def test_tier_allotment_for_meter_is_none_for_missing_dimension():
    assert (
        tier_allotment_for_meter(
            SubscriptionTier.FREE, UsageMeter.DOCUMENT_UPLOAD_TOKENS
        )
        is None
    )
    assert (
        tier_allotment_for_meter(
            SubscriptionTier.PREMIUM, UsageMeter.ADAPTER_TRAINING_UNITS
        )
        is not None
    )


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        (None, SubscriptionTier.FREE),
        ("", SubscriptionTier.FREE),
        ("nonsense", SubscriptionTier.FREE),
        ("PRO", SubscriptionTier.PRO),
        ("  premium  ", SubscriptionTier.PREMIUM),
        ("free", SubscriptionTier.FREE),
    ],
)
def test_tier_from_value_coerces_defensively(raw_value, expected):
    assert tier_from_value(raw_value) == expected


# ---------------------------------------------------------------------------
# Graduated-price decimal math (drives the provisioning script)
# ---------------------------------------------------------------------------


def test_per_million_overage_converts_to_cents_per_unit():
    allotment = MeterAllotment(
        meter=UsageMeter.MESSAGING_TOKENS,
        monthly_allotment=1_000_000,
        overage_price_per_million=2.00,
    )
    # $2 per million tokens = 200 cents per million = 0.0002 cents per token.
    assert float(allotment.stripe_unit_amount_decimal()) == pytest.approx(0.0002)


def test_per_unit_overage_converts_to_cents():
    allotment = MeterAllotment(
        meter=UsageMeter.ADAPTER_TRAINING_UNITS,
        monthly_allotment=5,
        overage_price_per_unit_usd=5.00,
    )
    assert float(allotment.stripe_unit_amount_decimal()) == pytest.approx(500.0)


def test_allotment_without_any_rate_raises():
    allotment = MeterAllotment(
        meter=UsageMeter.MESSAGING_TOKENS, monthly_allotment=1
    )
    with pytest.raises(ValueError):
        allotment.stripe_unit_amount_decimal()


# ---------------------------------------------------------------------------
# Billing-config JSON parsing
# ---------------------------------------------------------------------------


def _valid_config_document() -> dict:
    return {
        "meters": {meter.value: f"mtr_{meter.value}" for meter in UsageMeter},
        "tiers": {
            tier.value: {
                "base_price": f"price_{tier.value}_base",
                "metered_prices": {
                    meter.value: f"price_{tier.value}_{meter.value}"
                    for meter in TIER_DEFINITIONS[tier].meter_allotments
                },
            }
            for tier in SubscriptionTier
        },
    }


def test_load_config_round_trips_ids_and_orders_price_ids():
    config = load_stripe_billing_config(json.dumps(_valid_config_document()))
    assert config is not None
    premium = config.identifiers_for_tier(SubscriptionTier.PREMIUM)
    price_ids = premium.all_price_ids()
    assert price_ids[0] == "price_premium_base"
    assert len(price_ids) == 1 + len(
        TIER_DEFINITIONS[SubscriptionTier.PREMIUM].meter_allotments
    )
    assert config.meter_ids[UsageMeter.MESSAGING_TOKENS] == "mtr_messaging_tokens"


def test_load_config_returns_none_when_blank():
    assert load_stripe_billing_config(None) is None
    assert load_stripe_billing_config("   ") is None


def test_load_config_rejects_malformed_json_and_missing_base_price():
    with pytest.raises(ValueError):
        load_stripe_billing_config("{not json")
    broken = _valid_config_document()
    del broken["tiers"]["pro"]["base_price"]
    with pytest.raises(ValueError):
        load_stripe_billing_config(json.dumps(broken))


def test_load_config_ignores_metered_price_the_tier_does_not_grant():
    document = _valid_config_document()
    document["tiers"]["free"]["metered_prices"]["adapter_training_units"] = (
        "price_should_be_ignored"
    )
    config = load_stripe_billing_config(json.dumps(document))
    free_identifiers = config.identifiers_for_tier(SubscriptionTier.FREE)
    assert UsageMeter.ADAPTER_TRAINING_UNITS not in free_identifiers.metered_price_ids


def test_identifiers_for_unconfigured_tier_raises_key_error():
    document = _valid_config_document()
    del document["tiers"]["premium"]
    config = load_stripe_billing_config(json.dumps(document))
    with pytest.raises(KeyError):
        config.identifiers_for_tier(SubscriptionTier.PREMIUM)


# ---------------------------------------------------------------------------
# Anonymous / tier / customer / metering-id resolution
# ---------------------------------------------------------------------------


def _authenticated_user(tier: str = "pro") -> dict:
    return {
        "user_id": "auth0|abc123",
        "email": "person@example.com",
        "app_metadata": {
            "stripe_customer_id": "cus_123",
            "subscription_status": {"status": "active", "tier": tier},
        },
        "identities": [{"user_id": "abc123"}],
    }


def _anonymous_user() -> dict:
    return {
        "id": "supabase-ephemeral-id",
        "is_anonymous": True,
        "app_metadata": {"api_key": "hashed"},
        "identities": [{"user_id": "hashed-ip-value"}],
    }


def test_anonymous_user_detection():
    assert is_anonymous_user(None)
    assert is_anonymous_user(_anonymous_user())
    assert not is_anonymous_user(_authenticated_user())


def test_anonymous_user_is_hard_pinned_to_free_even_with_paid_metadata():
    anonymous = _anonymous_user()
    anonymous["app_metadata"]["tier"] = "premium"
    assert resolve_tier(anonymous) == SubscriptionTier.FREE
    assert not user_has_capability(anonymous, TierCapability.UPLOAD)


def test_authenticated_tier_resolution_and_capability():
    assert resolve_tier(_authenticated_user("premium")) == SubscriptionTier.PREMIUM
    assert user_has_capability(
        _authenticated_user("premium"), TierCapability.TRAIN_ADAPTER
    )
    assert resolve_tier(_authenticated_user("garbage")) == SubscriptionTier.FREE


def test_stripe_customer_id_resolution_tolerates_legacy_keys():
    assert resolve_stripe_customer_id(_authenticated_user()) == "cus_123"
    legacy = {
        "email": "person@example.com",
        "app_metadata": {"customer_dict": {"id": "cus_legacy_dict"}},
    }
    assert resolve_stripe_customer_id(legacy) == "cus_legacy_dict"
    older = {
        "email": "person@example.com",
        "app_metadata": {"customer": {"id": "cus_older"}},
    }
    assert resolve_stripe_customer_id(older) == "cus_older"
    assert resolve_stripe_customer_id(_anonymous_user()) is None


def test_metering_user_id_prefers_auth0_then_hashed_ip():
    assert resolve_metering_user_id(_authenticated_user()) == "auth0|abc123"
    assert resolve_metering_user_id(_anonymous_user()) == "hashed-ip-value"
    assert resolve_metering_user_id(None) is None


@pytest.mark.parametrize(
    "tier, adapter_requested, expected",
    [
        ("premium", True, True),
        ("premium", False, False),
        ("pro", True, False),
        ("free", True, False),
    ],
)
def test_resolve_use_adapter_inference_requires_premium_and_request(
    tier, adapter_requested, expected
):
    assert (
        resolve_use_adapter_inference(
            _authenticated_user(tier), adapter_requested
        )
        is expected
    )


def test_resolve_use_adapter_inference_falls_back_for_anonymous():
    assert resolve_use_adapter_inference(_anonymous_user(), True) is False


# ---------------------------------------------------------------------------
# Meter-event reporting helper
# ---------------------------------------------------------------------------


class _RecordingMeterEvent:
    """Fake ``stripe.billing.MeterEvent`` capturing async create calls."""

    def __init__(self, raise_error: bool = False):
        self.calls: list[dict] = []
        self._raise_error = raise_error

    async def create_async(self, **kwargs):
        if self._raise_error:
            raise RuntimeError("stripe unavailable")
        self.calls.append(kwargs)


class _FakeStripeClient:
    def __init__(self, raise_error: bool = False):
        class _Billing:
            pass

        self.billing = _Billing()
        self.billing.MeterEvent = _RecordingMeterEvent(raise_error=raise_error)


@pytest.mark.asyncio
async def test_report_meter_event_sends_payload_keyed_on_customer():
    stripe_client = _FakeStripeClient()
    accepted = await report_meter_event(
        stripe_client,
        UsageMeter.MESSAGING_TOKENS,
        "cus_123",
        1234,
        idempotency_identifier="request-1:messaging_tokens",
    )
    assert accepted is True
    (call,) = stripe_client.billing.MeterEvent.calls
    assert call["event_name"] == "messaging_tokens"
    assert call["payload"] == {"stripe_customer_id": "cus_123", "value": "1234"}
    assert call["identifier"] == "request-1:messaging_tokens"


@pytest.mark.asyncio
async def test_report_meter_event_no_ops_without_customer_or_value():
    stripe_client = _FakeStripeClient()
    assert not await report_meter_event(
        stripe_client, UsageMeter.MESSAGING_TOKENS, None, 100
    )
    assert not await report_meter_event(
        stripe_client, UsageMeter.MESSAGING_TOKENS, "cus_123", 0
    )
    assert stripe_client.billing.MeterEvent.calls == []


@pytest.mark.asyncio
async def test_report_meter_event_swallows_stripe_errors():
    stripe_client = _FakeStripeClient(raise_error=True)
    accepted = await report_meter_event(
        stripe_client, UsageMeter.MESSAGING_TOKENS, "cus_123", 100
    )
    assert accepted is False


# ---------------------------------------------------------------------------
# Token extraction and upload estimation
# ---------------------------------------------------------------------------


def test_billable_tokens_prefers_total_then_sums_parts():
    assert (
        billable_tokens_from_metadata({"token_usage": {"total_tokens": 42}}) == 42
    )
    assert (
        billable_tokens_from_metadata(
            {"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        )
        == 15
    )
    assert billable_tokens_from_metadata(None) == 0
    assert billable_tokens_from_metadata({}) == 0


# Upload token estimation moved to src/anubis/utils/billing/estimation.py and
# is covered by tests/unit_tests/test_token_estimation.py.


# ---------------------------------------------------------------------------
# Tier-change planning (upgrade/downgrade timing + usage-window rules)
# ---------------------------------------------------------------------------


class TestPlanTierChange:
    def test_every_upgrade_is_immediate_and_resets_usage(self) -> None:
        for current, target in [
            (SubscriptionTier.FREE, SubscriptionTier.PRO),
            (SubscriptionTier.FREE, SubscriptionTier.PREMIUM),
            (SubscriptionTier.PRO, SubscriptionTier.PREMIUM),
        ]:
            plan = plan_tier_change(current, target)
            assert plan.direction == "upgrade"
            assert plan.swap_items_immediately
            assert not plan.schedule_change_at_period_end
            assert plan.reset_usage_period_anchor

    def test_every_downgrade_is_scheduled_and_retains_usage(self) -> None:
        for current, target in [
            (SubscriptionTier.PRO, SubscriptionTier.FREE),
            (SubscriptionTier.PREMIUM, SubscriptionTier.FREE),
            (SubscriptionTier.PREMIUM, SubscriptionTier.PRO),
        ]:
            plan = plan_tier_change(current, target)
            assert plan.direction == "downgrade"
            assert not plan.swap_items_immediately
            assert plan.schedule_change_at_period_end
            assert not plan.reset_usage_period_anchor

    def test_trialing_upgrade_never_resets_the_usage_window(self) -> None:
        # One shared usage counter across a tier change during the trial.
        plan = plan_tier_change(
            SubscriptionTier.PRO, SubscriptionTier.PREMIUM, currently_trialing=True
        )
        assert plan.direction == "upgrade"
        assert plan.swap_items_immediately
        assert not plan.reset_usage_period_anchor

    def test_trialing_downgrade_stays_scheduled_without_reset(self) -> None:
        plan = plan_tier_change(
            SubscriptionTier.PRO, SubscriptionTier.FREE, currently_trialing=True
        )
        assert plan.direction == "downgrade"
        assert plan.schedule_change_at_period_end
        assert not plan.reset_usage_period_anchor


# ---------------------------------------------------------------------------
# POST /subscribe action planning (the single subscription entry point)
# ---------------------------------------------------------------------------


class TestPlanSubscribeAction:
    def test_no_subscription_starts_checkout(self) -> None:
        for status in (None, "canceled", "incomplete", "incomplete_expired"):
            assert (
                plan_subscribe_action(
                    status,
                    SubscriptionTier.FREE,
                    SubscriptionTier.PRO,
                    cancel_at_period_end=False,
                    has_pending_downgrade_schedule=False,
                )
                is SubscribeAction.START_CHECKOUT
            )

    def test_live_subscription_on_a_different_tier_changes_tier(self) -> None:
        for status in ("active", "trialing", "past_due"):
            assert (
                plan_subscribe_action(
                    status,
                    SubscriptionTier.PRO,
                    SubscriptionTier.PREMIUM,
                    cancel_at_period_end=False,
                    has_pending_downgrade_schedule=False,
                )
                is SubscribeAction.CHANGE_TIER
            )

    def test_same_tier_with_nothing_pending_is_a_no_op(self) -> None:
        assert (
            plan_subscribe_action(
                "active",
                SubscriptionTier.PRO,
                SubscriptionTier.PRO,
                cancel_at_period_end=False,
                has_pending_downgrade_schedule=False,
            )
            is SubscribeAction.NO_CHANGE_REQUIRED
        )

    def test_pending_cancellation_is_reactivated_automatically(self) -> None:
        assert (
            plan_subscribe_action(
                "active",
                SubscriptionTier.PRO,
                SubscriptionTier.PRO,
                cancel_at_period_end=True,
                has_pending_downgrade_schedule=False,
            )
            is SubscribeAction.REACTIVATE
        )

    def test_pending_downgrade_schedule_is_reactivated_automatically(self) -> None:
        assert (
            plan_subscribe_action(
                "active",
                SubscriptionTier.PREMIUM,
                SubscriptionTier.PREMIUM,
                cancel_at_period_end=False,
                has_pending_downgrade_schedule=True,
            )
            is SubscribeAction.REACTIVATE
        )

    def test_pending_change_with_a_different_tier_reactivates_then_changes(
        self,
    ) -> None:
        assert (
            plan_subscribe_action(
                "trialing",
                SubscriptionTier.PRO,
                SubscriptionTier.PREMIUM,
                cancel_at_period_end=True,
                has_pending_downgrade_schedule=False,
            )
            is SubscribeAction.REACTIVATE_AND_CHANGE_TIER
        )


# ---------------------------------------------------------------------------
# Trial context + trial-aware allotments (tier changes during a free trial)
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC)
_TRIAL_END_EPOCH = int((_NOW + timedelta(days=10)).timestamp())


def _pro_trial_user() -> dict:
    return {
        "user_id": "auth0|someone",
        "email": "someone@example.com",
        "app_metadata": {
            "trial_context": {"tier": "pro", "trial_end": _TRIAL_END_EPOCH}
        },
    }


class TestResolveTrialContext:
    def test_parses_the_written_shape(self) -> None:
        trial_context = resolve_trial_context(_pro_trial_user())
        assert trial_context is not None
        assert trial_context.trial_tier is SubscriptionTier.PRO
        assert int(trial_context.trial_end.timestamp()) == _TRIAL_END_EPOCH

    def test_missing_or_corrupt_context_is_none(self) -> None:
        assert resolve_trial_context(None) is None
        assert resolve_trial_context({"app_metadata": {}}) is None
        assert (
            resolve_trial_context(
                {"app_metadata": {"trial_context": {"tier": "pro"}}}
            )
            is None
        )
        assert (
            resolve_trial_context(
                {
                    "app_metadata": {
                        "trial_context": {"tier": "pro", "trial_end": "soon"}
                    }
                }
            )
            is None
        )


class TestResolveEffectiveMonthlyAllotment:
    def test_without_a_trial_the_tier_allotment_governs(self) -> None:
        assert resolve_effective_monthly_allotment(
            SubscriptionTier.FREE, UsageMeter.MESSAGING_TOKENS, None
        ) == tier_allotment_for_meter(
            SubscriptionTier.FREE, UsageMeter.MESSAGING_TOKENS
        )

    def test_trial_to_free_keeps_the_pro_allotments_inside_the_window(self) -> None:
        # Downgrade during the trial: the pro trial's messaging allotment and
        # the document-upload meter (which the free tier lacks entirely) stay
        # granted until the trial ends.
        trial_context = resolve_trial_context(_pro_trial_user())
        assert resolve_effective_monthly_allotment(
            SubscriptionTier.FREE,
            UsageMeter.MESSAGING_TOKENS,
            trial_context,
            now=_NOW,
        ) == tier_allotment_for_meter(
            SubscriptionTier.PRO, UsageMeter.MESSAGING_TOKENS
        )
        assert resolve_effective_monthly_allotment(
            SubscriptionTier.FREE,
            UsageMeter.DOCUMENT_UPLOAD_TOKENS,
            trial_context,
            now=_NOW,
        ) == tier_allotment_for_meter(
            SubscriptionTier.PRO, UsageMeter.DOCUMENT_UPLOAD_TOKENS
        )

    def test_trial_to_premium_takes_the_larger_allotment_per_meter(self) -> None:
        # Upgrade during the trial: premium's larger messaging allotment wins;
        # the trial allotment is only a floor.
        trial_context = resolve_trial_context(_pro_trial_user())
        assert resolve_effective_monthly_allotment(
            SubscriptionTier.PREMIUM,
            UsageMeter.MESSAGING_TOKENS,
            trial_context,
            now=_NOW,
        ) == tier_allotment_for_meter(
            SubscriptionTier.PREMIUM, UsageMeter.MESSAGING_TOKENS
        )

    def test_after_the_trial_window_plain_tier_allotments_apply(self) -> None:
        trial_context = resolve_trial_context(_pro_trial_user())
        after_trial = _NOW + timedelta(days=30)
        assert (
            resolve_effective_monthly_allotment(
                SubscriptionTier.FREE,
                UsageMeter.DOCUMENT_UPLOAD_TOKENS,
                trial_context,
                now=after_trial,
            )
            is None
        )
        assert resolve_effective_monthly_allotment(
            SubscriptionTier.FREE,
            UsageMeter.MESSAGING_TOKENS,
            trial_context,
            now=after_trial,
        ) == tier_allotment_for_meter(
            SubscriptionTier.FREE, UsageMeter.MESSAGING_TOKENS
        )

    def test_block_reason_honors_the_trial_allotment_override(self) -> None:
        # A free-tier user inside the trial window is judged against the pro
        # allotment: usage past the free allotment but under the pro allotment
        # is allowed.
        trial_context = resolve_trial_context(_pro_trial_user())
        trial_allotment = resolve_effective_monthly_allotment(
            SubscriptionTier.FREE,
            UsageMeter.MESSAGING_TOKENS,
            trial_context,
            now=_NOW,
        )
        free_allotment = tier_allotment_for_meter(
            SubscriptionTier.FREE, UsageMeter.MESSAGING_TOKENS
        )
        usage_past_free_under_pro = free_allotment.monthly_allotment + 1
        assert (
            exhausted_allotment_block_reason(
                SubscriptionTier.FREE,
                UsageMeter.MESSAGING_TOKENS,
                usage_past_free_under_pro,
                False,
                allotment_override=trial_allotment,
            )
            is None
        )
        assert (
            exhausted_allotment_block_reason(
                SubscriptionTier.FREE,
                UsageMeter.MESSAGING_TOKENS,
                trial_allotment.monthly_allotment,
                False,
                allotment_override=trial_allotment,
            )
            is not None
        )


# ---------------------------------------------------------------------------
# Pay-per-use resolution + payment-method detection
# ---------------------------------------------------------------------------


class TestResolvePayPerUseEnabled:
    def test_explicit_flag_wins_in_both_directions(self) -> None:
        enabled_user = {
            "user_id": "auth0|x",
            "email": "x@example.com",
            "app_metadata": {
                "pay_per_use_enabled": True,
                "stripe_customer_id": "cus_x",
                "subscription_status": {"status": "trialing", "tier": "pro"},
            },
        }
        assert resolve_pay_per_use_enabled(enabled_user)
        enabled_user["app_metadata"]["pay_per_use_enabled"] = False
        enabled_user["app_metadata"]["subscription_status"]["status"] = "active"
        assert not resolve_pay_per_use_enabled(enabled_user)

    def test_active_status_infers_enabled_but_trialing_does_not(self) -> None:
        user = {
            "user_id": "auth0|y",
            "email": "y@example.com",
            "app_metadata": {
                "stripe_customer_id": "cus_y",
                "subscription_status": {"status": "active", "tier": "pro"},
            },
        }
        assert resolve_pay_per_use_enabled(user)
        user["app_metadata"]["subscription_status"]["status"] = "trialing"
        assert not resolve_pay_per_use_enabled(user)

    def test_anonymous_users_never_get_pay_per_use(self) -> None:
        anonymous_user = {
            "is_anonymous": True,
            "identities": [{"user_id": "hashed-ip"}],
            "app_metadata": {
                "pay_per_use_enabled": True,
                "stripe_customer_id": "cus_anon",
            },
        }
        assert not resolve_pay_per_use_enabled(anonymous_user)


class TestCustomerHasPaymentMethod:
    def test_default_payment_method_counts(self) -> None:
        assert customer_has_payment_method(
            {"invoice_settings": {"default_payment_method": "pm_1"}}
        )

    def test_legacy_default_source_counts(self) -> None:
        assert customer_has_payment_method({"default_source": "card_1"})

    def test_payment_method_list_fallback_counts(self) -> None:
        assert customer_has_payment_method({}, [{"id": "pm_2"}])

    def test_no_payment_method_anywhere_is_false(self) -> None:
        assert not customer_has_payment_method({}, [])
        assert not customer_has_payment_method(None)
