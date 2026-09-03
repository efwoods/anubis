# src/anubis/utils/billing/tiers.py

"""Subscription-tier and usage-meter definitions — the single source of truth.

This module is intentionally dependency-light (standard library only) so it can
be imported by the Stripe provisioning script, the FastAPI gating layer, and the
metering helpers without pulling in heavy SDKs or triggering cold-start penalties.

Design (see `_METERING_FEATURE.md` and `research/04_token_workload_cost_model.md`):

* Three tiers — ``free``, ``pro``, ``premium`` — each with a flat monthly base fee.
* Four separately-budgeted usage meters so one dimension can never cannibalize
  another's allotment: messaging tokens, document-upload tokens, adapter-training
  units, and adapter-inference tokens.
* Each tier grants a monthly allotment per meter (Stripe graduated-price tier 1,
  included at zero cost) and charges pay-per-use overage past the allotment
  (graduated tier 2). The allotment resets every Stripe billing period (monthly).
* Anonymous users are ALWAYS the ``free`` tier and can never subscribe; that rule
  is enforced in ``src/security/auth.py``, not here, but the free-tier allotments
  below are what anonymous usage meters against.

The numeric allotments and overage rates below are deliberate, editable defaults
grounded in the May-2026 cost model in ``research/04_token_workload_cost_model.md``
(gpt-4o-mini blended cost ≈ $0.375 per 1M tokens; ≈ $0.27 per avatar per month at
ten messages per day). Overage rates mark the raw model cost up to cover pro-rated
infrastructure. Adjust the numbers here and re-run ``scripts/provision_stripe_billing.py``
to change pricing; nothing else in the codebase hardcodes these values.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Dict, FrozenSet

# Stripe accepts at most twelve decimal places in ``unit_amount_decimal``. Every
# rate is quantized to that precision before being sent, so a rate that cannot be
# represented exactly is rounded once, deliberately, here — rather than arriving
# at Stripe as a float artifact that is rejected or silently re-rounded.
STRIPE_DECIMAL_PLACES = 12
_STRIPE_DECIMAL_QUANTUM = Decimal(1).scaleb(-STRIPE_DECIMAL_PLACES)


def _format_stripe_decimal(amount: Decimal) -> str:
    """Render ``amount`` as a plain decimal string Stripe accepts.

    Quantizing to twelve places and then normalizing strips the trailing zeros the
    quantum introduces, and ``format(..., "f")`` renders without exponent notation —
    ``Decimal.normalize`` turns 0.0002 into ``2E-4``, which Stripe rejects.
    """
    quantized = amount.quantize(_STRIPE_DECIMAL_QUANTUM)
    return format(quantized.normalize(), "f")


class SubscriptionTier(StrEnum):
    """The three subscription tiers a customer can hold."""

    FREE = "free"
    PRO = "pro"
    PREMIUM = "premium"


class UsageMeter(StrEnum):
    """The four separately-budgeted usage dimensions.

    The value of each member is the Stripe Billing Meter ``event_name`` that
    usage events are reported against, and the key used everywhere in the
    application to identify the dimension.
    """

    MESSAGING_TOKENS = "messaging_tokens"
    DOCUMENT_UPLOAD_TOKENS = "document_upload_tokens"
    ADAPTER_TRAINING_UNITS = "adapter_training_units"
    ADAPTER_INFERENCE_TOKENS = "adapter_inference_tokens"
    # Voice and video responses. Speech is billed by the vendor per character
    # (one credit per character), lip-sync video per second; both meters bill
    # the same units so the allotment maps onto vendor cost directly.
    SPEECH_CHARACTERS = "speech_characters"
    VIDEO_GENERATION_SECONDS = "video_generation_seconds"


class TierCapability(StrEnum):
    """Capabilities a tier unlocks. Higher tiers inherit lower-tier capabilities."""

    MESSAGE = "message"  # send and receive messages (all tiers)
    UPLOAD = "upload"  # update avatar identity with media (pro and premium)
    TRAIN_ADAPTER = "train_adapter"  # train fine-tuning adapters (premium only)
    AUDIO_RESPONSES = "audio_responses"  # cloned-voice speech (pro and premium)
    VIDEO_RESPONSES = "video_responses"  # lip-synced video replies (premium only)


# Ordered list of every meter's ``event_name``. Reused by the provisioning script
# and the metering helpers so meter identity stays consistent in one place.
METER_EVENT_NAMES: tuple[str, ...] = tuple(meter.value for meter in UsageMeter)


@dataclass(frozen=True)
class MeterAllotment:
    """One meter's monthly budget and overage rate for a single tier.

    ``monthly_allotment`` is the number of included units per billing period
    (the Stripe graduated tier-1 ``up_to`` value). ``overage_price_per_million``
    is what the customer is charged, in whole US dollars, per one million units
    consumed past the allotment; for the adapter-training meter the unit is a
    single trained adapter rather than a token, and the rate is expressed with
    ``overage_price_per_unit_usd`` instead.
    """

    meter: UsageMeter
    monthly_allotment: int
    overage_price_per_million: float | None = None
    overage_price_per_unit_usd: float | None = None

    def stripe_unit_amount_decimal(self) -> str:
        """Return the Stripe graduated tier-2 ``unit_amount_decimal`` in cents.

        Stripe prices are denominated in the currency's minor unit (cents for USD)
        and ``unit_amount_decimal`` permits sub-cent precision, which per-token
        pricing requires. A rate of ``overage_price_per_million`` dollars per one
        million units equals ``overage_price_per_million / 1_000_000`` dollars per
        unit, i.e. ``overage_price_per_million / 10_000`` cents per unit.

        The arithmetic runs in ``Decimal``, not binary floating point, because the
        result is compared against what Stripe echoes back to decide whether a
        provisioned price still matches this catalog. ``str(1.10 / 10_000.0)``
        yields ``"0.00011000000000000002"`` — twenty decimal places where Stripe
        permits twelve — which would be rejected on the way out and, if it were
        not, would never compare equal on the way back, so every provisioning run
        would believe the price had drifted and mint a replacement.
        """
        if self.overage_price_per_unit_usd is not None:
            return _format_stripe_decimal(
                Decimal(str(self.overage_price_per_unit_usd)) * Decimal(100)
            )
        if self.overage_price_per_million is not None:
            return _format_stripe_decimal(
                Decimal(str(self.overage_price_per_million)) / Decimal(10_000)
            )
        raise ValueError(
            f"Meter allotment for {self.meter.value} has no overage rate configured."
        )


@dataclass(frozen=True)
class TierDefinition:
    """A single subscription tier: base fee, capabilities, and per-meter budgets."""

    tier: SubscriptionTier
    display_name: str
    monthly_base_fee_usd: float
    capabilities: FrozenSet[TierCapability]
    meter_allotments: Dict[UsageMeter, MeterAllotment]
    # Number of free-trial days offered on this tier before the base fee applies.
    # Zero means no trial (the free tier is always zero; pro carries the 30-day
    # trial that the current live Stripe payment link already offers).
    trial_period_days: int = 0

    def stripe_base_unit_amount_cents(self) -> int:
        """Return the flat monthly base fee as an integer number of US cents.

        Converted through ``Decimal`` for the same reason as the overage rates:
        ``19.99 * 100`` is 1998.9999999999998 in binary floating point, and the
        cent value is compared against what Stripe reports to detect a price edit.
        """
        return int(
            (Decimal(str(self.monthly_base_fee_usd)) * Decimal(100)).quantize(
                Decimal(1)
            )
        )


# ---------------------------------------------------------------------------
# The concrete tier catalog. EDIT THESE NUMBERS to change pricing, then re-run
# scripts/provision_stripe_billing.py to recreate the Stripe objects.
# ---------------------------------------------------------------------------

TIER_DEFINITIONS: Dict[SubscriptionTier, TierDefinition] = {
    SubscriptionTier.FREE: TierDefinition(
        tier=SubscriptionTier.FREE,
        display_name="Neural Nexus Free Tier",
        monthly_base_fee_usd=0.0,
        capabilities=frozenset({TierCapability.MESSAGE}),
        trial_period_days=0,
        meter_allotments={
            UsageMeter.MESSAGING_TOKENS: MeterAllotment(
                meter=UsageMeter.MESSAGING_TOKENS,
                monthly_allotment=2_000_000,  # ≈ a light month of conversation
                overage_price_per_million=2.00,
            ),
        },
    ),
    SubscriptionTier.PRO: TierDefinition(
        tier=SubscriptionTier.PRO,
        display_name="Neural Nexus Pro Tier",
        monthly_base_fee_usd=20.0,
        capabilities=frozenset(
            {
                TierCapability.MESSAGE,
                TierCapability.UPLOAD,
                TierCapability.AUDIO_RESPONSES,
            }
        ),
        trial_period_days=30,
        meter_allotments={
            UsageMeter.MESSAGING_TOKENS: MeterAllotment(
                meter=UsageMeter.MESSAGING_TOKENS,
                monthly_allotment=5_000_000,
                overage_price_per_million=1.50,
            ),
            UsageMeter.DOCUMENT_UPLOAD_TOKENS: MeterAllotment(
                meter=UsageMeter.DOCUMENT_UPLOAD_TOKENS,
                monthly_allotment=10_000_000,
                overage_price_per_million=3.00,
            ),
            # Speech: vendor cost is ~$0.05 per 1,000 characters (Flash); the
            # allotment covers roughly two hundred short spoken replies and the
            # overage is priced at twice vendor cost.
            UsageMeter.SPEECH_CHARACTERS: MeterAllotment(
                meter=UsageMeter.SPEECH_CHARACTERS,
                monthly_allotment=50_000,
                overage_price_per_million=100.00,
            ),
        },
    ),
    SubscriptionTier.PREMIUM: TierDefinition(
        tier=SubscriptionTier.PREMIUM,
        display_name="Neural Nexus Premium Tier",
        monthly_base_fee_usd=50.0,
        capabilities=frozenset(
            {
                TierCapability.MESSAGE,
                TierCapability.UPLOAD,
                TierCapability.TRAIN_ADAPTER,
                TierCapability.AUDIO_RESPONSES,
                TierCapability.VIDEO_RESPONSES,
            }
        ),
        trial_period_days=0,
        meter_allotments={
            UsageMeter.MESSAGING_TOKENS: MeterAllotment(
                meter=UsageMeter.MESSAGING_TOKENS,
                monthly_allotment=20_000_000,
                overage_price_per_million=1.25,
            ),
            UsageMeter.DOCUMENT_UPLOAD_TOKENS: MeterAllotment(
                meter=UsageMeter.DOCUMENT_UPLOAD_TOKENS,
                monthly_allotment=40_000_000,
                overage_price_per_million=2.50,
            ),
            UsageMeter.ADAPTER_INFERENCE_TOKENS: MeterAllotment(
                meter=UsageMeter.ADAPTER_INFERENCE_TOKENS,
                monthly_allotment=10_000_000,
                overage_price_per_million=4.00,
            ),
            UsageMeter.ADAPTER_TRAINING_UNITS: MeterAllotment(
                meter=UsageMeter.ADAPTER_TRAINING_UNITS,
                monthly_allotment=5,  # five trained adapters included per month
                overage_price_per_unit_usd=5.00,
            ),
            UsageMeter.SPEECH_CHARACTERS: MeterAllotment(
                meter=UsageMeter.SPEECH_CHARACTERS,
                monthly_allotment=200_000,
                overage_price_per_million=100.00,
            ),
            # Lip-sync video: vendor cost is ~$0.14 per second; a few minutes
            # are included with the base fee and overage is priced at twice
            # vendor cost so the subscription covers the service.
            UsageMeter.VIDEO_GENERATION_SECONDS: MeterAllotment(
                meter=UsageMeter.VIDEO_GENERATION_SECONDS,
                monthly_allotment=180,
                overage_price_per_unit_usd=0.28,
            ),
        },
    ),
}


def tier_from_value(value: str | None) -> SubscriptionTier:
    """Coerce a stored tier string into a ``SubscriptionTier``, defaulting to free.

    Any missing, unknown, or malformed value resolves to the free tier so that a
    corrupt ``app_metadata`` record can never accidentally grant paid capabilities.
    """
    if not value:
        return SubscriptionTier.FREE
    try:
        return SubscriptionTier(str(value).strip().lower())
    except ValueError:
        return SubscriptionTier.FREE


def tier_has_capability(tier: SubscriptionTier, capability: TierCapability) -> bool:
    """Return whether ``tier`` unlocks ``capability``."""
    return capability in TIER_DEFINITIONS[tier].capabilities


def tier_allotment_for_meter(
    tier: SubscriptionTier, meter: UsageMeter
) -> MeterAllotment | None:
    """Return the tier's allotment for ``meter``, or ``None`` if the tier lacks it.

    A ``None`` result means the tier does not include that usage dimension at all
    (for example the free tier has no document-upload allotment), which the gating
    layer treats as "capability not available to this tier".
    """
    return TIER_DEFINITIONS[tier].meter_allotments.get(meter)
