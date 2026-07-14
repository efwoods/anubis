# src/anubis/utils/billing/config.py

"""Parse the Stripe object identifiers emitted by the provisioning script.

``scripts/provision_stripe_billing.py`` creates the four Billing Meters, three
tier products, and every flat/metered price, then prints a single JSON document
that the operator pastes into the ``STRIPE_BILLING_CONFIG_JSON`` environment
variable (surfaced as ``GlobalContext.stripe_billing_config_json``). This module
turns that JSON string back into typed, validated lookups that the FastAPI layer
uses to build Checkout line items, switch tiers, and report meter events.

Keeping the identifiers in one JSON blob (rather than fifteen separate env vars)
means switching between Stripe test mode and live mode is a single value swap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

from src.anubis.utils.billing.tiers import (
    TIER_DEFINITIONS,
    SubscriptionTier,
    UsageMeter,
)


@dataclass(frozen=True)
class TierStripeIdentifiers:
    """The Stripe price identifiers that make up one tier's subscription."""

    base_price_id: str
    metered_price_ids: Dict[UsageMeter, str]

    def all_price_ids(self) -> List[str]:
        """Every price id for this tier: the flat base first, then metered prices.

        This is the ordered set of subscription items used when creating a
        Checkout Session or replacing items during a tier change.
        """
        ordered = [self.base_price_id]
        for meter in UsageMeter:
            price_id = self.metered_price_ids.get(meter)
            if price_id is not None:
                ordered.append(price_id)
        return ordered


@dataclass(frozen=True)
class StripeBillingConfig:
    """Fully-resolved Stripe identifiers for meters and all three tiers."""

    meter_ids: Dict[UsageMeter, str]
    tiers: Dict[SubscriptionTier, TierStripeIdentifiers]
    # Billing-portal configuration created by the provisioning script; optional so
    # config JSON emitted before the portal existed still loads.
    portal_configuration_id: str | None = None

    def identifiers_for_tier(
        self, tier: SubscriptionTier
    ) -> TierStripeIdentifiers:
        """Return the Stripe identifiers for ``tier`` or raise if unconfigured."""
        identifiers = self.tiers.get(tier)
        if identifiers is None:
            raise KeyError(
                f"Stripe billing config has no prices for tier '{tier.value}'. "
                "Re-run scripts/provision_stripe_billing.py and refresh "
                "STRIPE_BILLING_CONFIG_JSON."
            )
        return identifiers


def load_stripe_billing_config(
    stripe_billing_config_json: str | None,
) -> StripeBillingConfig | None:
    """Parse ``stripe_billing_config_json`` into a ``StripeBillingConfig``.

    Returns ``None`` when the value is absent or blank so that callers can degrade
    gracefully (for example, treat every user as free tier when billing has not
    been provisioned yet) rather than crash at import time. Malformed JSON or a
    document missing required keys raises ``ValueError`` so a misconfiguration is
    caught loudly at startup rather than silently mis-billing customers.

    Expected JSON shape::

        {
          "meters": {"messaging_tokens": "mtr_...", ...},
          "tiers": {
            "free":    {"base_price": "price_...",
                        "metered_prices": {"messaging_tokens": "price_..."}},
            "pro":     {"base_price": "price_...", "metered_prices": {...}},
            "premium": {"base_price": "price_...", "metered_prices": {...}}
          },
          "portal_configuration": "bpc_..."   // optional
        }
    """
    if not stripe_billing_config_json or not stripe_billing_config_json.strip():
        return None

    try:
        document = json.loads(stripe_billing_config_json)
    except json.JSONDecodeError as decode_error:
        raise ValueError(
            f"STRIPE_BILLING_CONFIG_JSON is not valid JSON: {decode_error}"
        ) from decode_error

    raw_meters = document.get("meters", {})
    meter_ids: Dict[UsageMeter, str] = {}
    for meter in UsageMeter:
        meter_id = raw_meters.get(meter.value)
        if meter_id:
            meter_ids[meter] = meter_id

    raw_tiers = document.get("tiers", {})
    tiers: Dict[SubscriptionTier, TierStripeIdentifiers] = {}
    for tier in SubscriptionTier:
        raw_tier = raw_tiers.get(tier.value)
        if not raw_tier:
            continue
        base_price_id = raw_tier.get("base_price")
        if not base_price_id:
            raise ValueError(
                f"STRIPE_BILLING_CONFIG_JSON tier '{tier.value}' is missing 'base_price'."
            )
        raw_metered = raw_tier.get("metered_prices", {})
        metered_price_ids: Dict[UsageMeter, str] = {}
        # Only accept metered prices for meters the tier definition actually grants,
        # so a stale config cannot silently bill a meter the tier should not have.
        for meter in TIER_DEFINITIONS[tier].meter_allotments:
            price_id = raw_metered.get(meter.value)
            if price_id:
                metered_price_ids[meter] = price_id
        tiers[tier] = TierStripeIdentifiers(
            base_price_id=base_price_id,
            metered_price_ids=metered_price_ids,
        )

    raw_portal_configuration = document.get("portal_configuration")
    portal_configuration_id = (
        str(raw_portal_configuration) if raw_portal_configuration else None
    )

    return StripeBillingConfig(
        meter_ids=meter_ids,
        tiers=tiers,
        portal_configuration_id=portal_configuration_id,
    )
