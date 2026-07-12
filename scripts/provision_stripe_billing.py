#!/usr/bin/env python
# scripts/provision_stripe_billing.py

"""Idempotently create the Stripe meters, products, and prices for all three tiers.

This script turns the tier catalog in ``src/anubis/utils/billing/tiers.py`` into
concrete Stripe objects and prints the JSON to paste into
``STRIPE_BILLING_CONFIG_JSON``. It is the single, reviewable, repeatable way to
build the billing objects — run it against a TEST-mode key first, verify, then
re-run against the live key.

What it creates:

* Four Billing Meters (one per usage dimension), aggregating ``sum`` of the
  ``value`` field, keyed to the customer via ``stripe_customer_id``.
* One product per tier (free / pro / premium).
* One licensed flat monthly base price per tier (the subscription fee).
* One graduated metered price per (tier, meter) pair: tier 1 is the included
  monthly allotment at zero cost, tier 2 is pay-per-use overage.

Idempotency:

* Meters are matched by ``event_name`` (Stripe forbids two active meters sharing
  one), and reused when present.
* Products are matched by the ``neural_nexus_tier`` metadata key.
* Prices are matched by ``lookup_key``. Because Stripe prices are immutable once
  used, bump ``PRICE_LOOKUP_KEY_VERSION`` to force a fresh set of prices after
  changing any amount or allotment in ``tiers.py`` (archive the old ones by hand).

Usage:

    # test mode (default guard: refuses a live key unless --allow-live is passed)
    STRIPE_SECRET_KEY=sk_test_... python scripts/provision_stripe_billing.py

    # live mode (explicit opt-in)
    STRIPE_SECRET_KEY=sk_live_... python scripts/provision_stripe_billing.py --allow-live
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import stripe

# Allow running as a plain script from the repo root without installing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.anubis.utils.billing.tiers import (  # noqa: E402
    TIER_DEFINITIONS,
    MeterAllotment,
    SubscriptionTier,
    TierDefinition,
    UsageMeter,
)

# Bump this to force creation of a new immutable price set after editing amounts.
PRICE_LOOKUP_KEY_VERSION = "v1"
PRODUCT_TIER_METADATA_KEY = "neural_nexus_tier"


def _base_price_lookup_key(tier: SubscriptionTier) -> str:
    return f"nn_{tier.value}_base_{PRICE_LOOKUP_KEY_VERSION}"


def _metered_price_lookup_key(tier: SubscriptionTier, meter: UsageMeter) -> str:
    return f"nn_{tier.value}_{meter.value}_{PRICE_LOOKUP_KEY_VERSION}"


def find_or_create_meter(meter: UsageMeter) -> str:
    """Return the id of the Billing Meter for ``meter``, creating it if absent."""
    # StripeObject is not a dict subclass in stripe-python 15, so compare via
    # plain-dict conversion rather than ``.get`` attribute access.
    for existing in stripe.billing.Meter.list(status="active", limit=100).auto_paging_iter():
        existing_meter = existing.to_dict()
        if existing_meter.get("event_name") == meter.value:
            print(f"  meter '{meter.value}' exists -> {existing_meter['id']}")
            return existing_meter["id"]

    created = stripe.billing.Meter.create(
        display_name=meter.value.replace("_", " ").title(),
        event_name=meter.value,
        default_aggregation={"formula": "sum"},
        customer_mapping={"type": "by_id", "event_payload_key": "stripe_customer_id"},
        value_settings={"event_payload_key": "value"},
    )
    print(f"  meter '{meter.value}' CREATED -> {created['id']}")
    return created["id"]


def find_or_create_product(definition: TierDefinition) -> str:
    """Return the id of the product for a tier, creating it if absent."""
    for existing in stripe.Product.list(active=True, limit=100).auto_paging_iter():
        existing_product = existing.to_dict()
        existing_tier = (existing_product.get("metadata") or {}).get(
            PRODUCT_TIER_METADATA_KEY
        )
        if existing_tier == definition.tier.value:
            print(
                f"  product '{definition.tier.value}' exists -> {existing_product['id']}"
            )
            return existing_product["id"]

    created = stripe.Product.create(
        name=definition.display_name,
        metadata={PRODUCT_TIER_METADATA_KEY: definition.tier.value},
    )
    print(f"  product '{definition.tier.value}' CREATED -> {created['id']}")
    return created["id"]


def _find_price_by_lookup_key(lookup_key: str) -> str | None:
    prices = stripe.Price.list(lookup_keys=[lookup_key], active=True, limit=1).to_dict()
    if prices.get("data"):
        return prices["data"][0]["id"]
    return None


def find_or_create_base_price(
    definition: TierDefinition, product_id: str
) -> str:
    """Return the licensed flat monthly base price id for a tier."""
    lookup_key = _base_price_lookup_key(definition.tier)
    existing = _find_price_by_lookup_key(lookup_key)
    if existing:
        print(f"    base price '{lookup_key}' exists -> {existing}")
        return existing

    created = stripe.Price.create(
        product=product_id,
        currency="usd",
        unit_amount=definition.stripe_base_unit_amount_cents(),
        recurring={"interval": "month"},
        lookup_key=lookup_key,
        nickname=f"{definition.display_name} — base",
    )
    print(f"    base price '{lookup_key}' CREATED -> {created['id']}")
    return created["id"]


def find_or_create_metered_price(
    definition: TierDefinition,
    product_id: str,
    meter_id: str,
    allotment: MeterAllotment,
) -> str:
    """Return the graduated metered price id for one (tier, meter) pair."""
    lookup_key = _metered_price_lookup_key(definition.tier, allotment.meter)
    existing = _find_price_by_lookup_key(lookup_key)
    if existing:
        print(f"    metered price '{lookup_key}' exists -> {existing}")
        return existing

    created = stripe.Price.create(
        product=product_id,
        currency="usd",
        recurring={
            "interval": "month",
            "usage_type": "metered",
            "meter": meter_id,
        },
        billing_scheme="tiered",
        tiers_mode="graduated",
        tiers=[
            # Tier 1: the included monthly allotment, at zero cost.
            {"up_to": allotment.monthly_allotment, "unit_amount_decimal": "0"},
            # Tier 2: pay-per-use overage past the allotment.
            {"up_to": "inf", "unit_amount_decimal": allotment.stripe_unit_amount_decimal()},
        ],
        lookup_key=lookup_key,
        nickname=f"{definition.display_name} — {allotment.meter.value}",
    )
    print(f"    metered price '{lookup_key}' CREATED -> {created['id']}")
    return created["id"]


def provision() -> Dict[str, Any]:
    """Create/reuse every object and return the billing-config JSON document."""
    print("Provisioning Billing Meters:")
    meter_ids: Dict[UsageMeter, str] = {
        meter: find_or_create_meter(meter) for meter in UsageMeter
    }

    tiers_config: Dict[str, Any] = {}
    for tier, definition in TIER_DEFINITIONS.items():
        print(f"Provisioning tier '{tier.value}':")
        product_id = find_or_create_product(definition)
        base_price_id = find_or_create_base_price(definition, product_id)

        metered_price_ids: Dict[str, str] = {}
        for meter, allotment in definition.meter_allotments.items():
            price_id = find_or_create_metered_price(
                definition, product_id, meter_ids[meter], allotment
            )
            metered_price_ids[meter.value] = price_id

        tiers_config[tier.value] = {
            "product": product_id,
            "base_price": base_price_id,
            "metered_prices": metered_price_ids,
        }

    return {
        "meters": {meter.value: meter_id for meter, meter_id in meter_ids.items()},
        "tiers": tiers_config,
    }


def main() -> None:
    """Parse arguments, guard test/live mode, provision objects, and print config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-live",
        action="store_true",
        help="Permit running against a live (sk_live_) key. Off by default.",
    )
    args = parser.parse_args()

    secret_key = os.environ.get("STRIPE_SECRET_KEY")
    if not secret_key:
        raise SystemExit("STRIPE_SECRET_KEY is not set.")

    is_live_key = secret_key.startswith("sk_live_")
    if is_live_key and not args.allow_live:
        raise SystemExit(
            "Refusing to provision against a LIVE key without --allow-live. "
            "Provision and verify in test mode first (use an sk_test_ key)."
        )

    stripe.api_key = secret_key
    mode = "LIVE" if is_live_key else "TEST"
    print(f"=== Provisioning Stripe billing objects in {mode} mode ===")

    config_document = provision()

    print("\n=== DONE. Set STRIPE_BILLING_CONFIG_JSON to the following single line: ===\n")
    print(json.dumps(config_document, separators=(",", ":")))


if __name__ == "__main__":
    main()
