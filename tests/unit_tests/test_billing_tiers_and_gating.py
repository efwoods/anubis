"""Unit tests for the Stripe metering billing package.

Covers the tier catalog invariants (capability inheritance, four independent
meters, per-tier allotments), defensive tier coercion, billing-config JSON
parsing, anonymous-user resolution and metering-identifier resolution, the
Stripe meter-event helper's no-op and failure paths, the upload token estimate,
and the graduated-price decimal math used by the provisioning script.
"""

import json

import pytest

from src.anubis.utils.billing.config import load_stripe_billing_config
from src.anubis.utils.billing.gating import (
    is_anonymous_user,
    resolve_metering_user_id,
    resolve_stripe_customer_id,
    resolve_tier,
    resolve_use_adapter_inference,
    user_has_capability,
)
from src.anubis.utils.billing.metering import (
    billable_tokens_from_metadata,
    estimate_upload_token_units,
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


def test_estimate_upload_token_units_scales_with_inputs():
    assert estimate_upload_token_units() == 0
    text_only = estimate_upload_token_units(text_character_count=4_000)
    assert text_only == int(4_000 * 0.25 * 3.0)
    with_media = estimate_upload_token_units(
        audio_seconds=60, text_character_count=4_000, image_count=2, url_count=1
    )
    assert with_media > text_only
