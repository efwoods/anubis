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
* One SELF-DESCRIBING base product per tier (free / pro / premium) whose name
  and description enumerate the tier's included monthly allotments — Stripe
  Checkout displays product names and descriptions on every line item, so the
  customer sees exactly what each line is.
* One SELF-DESCRIBING product per (tier, meter) pair — e.g. "Neural Nexus Pro
  — Messaging Tokens (5,000,000 included/month)" with the overage rate in the
  description — so metered line items in Checkout are no longer three
  identical "Neural Nexus Pro Tier" rows.
* One licensed flat monthly base price per tier (the subscription fee), under
  the tier's base product.
* One graduated metered price per (tier, meter) pair, under that pair's
  product: tier 1 is the included monthly allotment at zero cost, tier 2 is
  pay-per-use overage.
* One billing-portal configuration (invoices, payment methods, billing
  information, at-period-end cancellation; no plan switching — metered prices
  require tier changes to go through POST /subscribe).

Idempotency:

* Meters are matched by ``event_name`` (Stripe forbids two active meters sharing
  one), and reused when present.
* Products are matched by the metadata triple ``neural_nexus_tier`` +
  ``neural_nexus_product_role`` (+ ``neural_nexus_meter`` for metered
  products) at the current ``neural_nexus_catalog_version``; on reuse the
  name and description are refreshed so re-runs keep the customer-facing copy
  in sync with ``tiers.py``.
* Prices are matched by ``lookup_key``. Because Stripe prices are immutable once
  used, bump ``PRICE_LOOKUP_KEY_VERSION`` to force a fresh set of prices after
  changing any amount or allotment in ``tiers.py`` (archive the old ones by hand).

Migration from the v1 catalog (one product per tier, all prices attached to
that one product): re-run this script (test mode first), paste the printed
JSON into ``STRIPE_BILLING_CONFIG_JSON``, and restart the API. Existing
subscriptions keep their v1 prices until their next tier change (the tier
change swaps subscription items to the current price ids); archive the v1
prices and products by hand once no live subscription references them.

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
PRICE_LOOKUP_KEY_VERSION = "v2"
PRODUCT_TIER_METADATA_KEY = "neural_nexus_tier"
PRODUCT_ROLE_METADATA_KEY = "neural_nexus_product_role"
PRODUCT_METER_METADATA_KEY = "neural_nexus_meter"
PRODUCT_CATALOG_VERSION_METADATA_KEY = "neural_nexus_catalog_version"
PRODUCT_ROLE_BASE = "base"
PRODUCT_ROLE_METERED = "metered"
PORTAL_CONFIGURATION_METADATA_KEY = "neural_nexus_portal"
PORTAL_CONFIGURATION_METADATA_VALUE = f"portal_{PRICE_LOOKUP_KEY_VERSION}"

# Customer-facing names for each usage meter, shown on Checkout line items.
METER_DISPLAY_NAMES: Dict[UsageMeter, str] = {
    UsageMeter.MESSAGING_TOKENS: "Messaging Tokens",
    UsageMeter.DOCUMENT_UPLOAD_TOKENS: "Document Upload Tokens",
    UsageMeter.ADAPTER_INFERENCE_TOKENS: "Adapter Inference Tokens",
    UsageMeter.ADAPTER_TRAINING_UNITS: "Adapter Training",
}


def _base_price_lookup_key(tier: SubscriptionTier) -> str:
    return f"nn_{tier.value}_base_{PRICE_LOOKUP_KEY_VERSION}"


def _metered_price_lookup_key(tier: SubscriptionTier, meter: UsageMeter) -> str:
    return f"nn_{tier.value}_{meter.value}_{PRICE_LOOKUP_KEY_VERSION}"


def _allotment_unit_noun(meter: UsageMeter) -> str:
    """Return the plural unit noun the customer reads for one meter."""
    if meter is UsageMeter.ADAPTER_TRAINING_UNITS:
        return "trained adapters"
    return "tokens"


def _allotment_overage_sentence(allotment: MeterAllotment) -> str:
    """Return the customer-facing sentence describing the overage rate."""
    if allotment.overage_price_per_unit_usd is not None:
        return (
            f"additional usage is billed at "
            f"${allotment.overage_price_per_unit_usd:,.2f} per "
            f"{_allotment_unit_noun(allotment.meter).rstrip('s')} "
            "when pay-per-use is enabled."
        )
    if allotment.overage_price_per_million is not None:
        return (
            f"additional usage is billed at "
            f"${allotment.overage_price_per_million:,.2f} per 1,000,000 "
            f"{_allotment_unit_noun(allotment.meter)} "
            "when pay-per-use is enabled."
        )
    return "no overage rate applies."


def _allotment_summary(allotment: MeterAllotment) -> str:
    """Return the short 'name (allotment included/month)' summary for one meter."""
    return (
        f"{METER_DISPLAY_NAMES[allotment.meter]} "
        f"({allotment.monthly_allotment:,} included/month)"
    )


def _allotment_description(allotment: MeterAllotment) -> str:
    """Return the full customer-facing description for one (tier, meter) product."""
    return (
        f"{allotment.monthly_allotment:,} "
        f"{_allotment_unit_noun(allotment.meter)} included each month; "
        f"{_allotment_overage_sentence(allotment)}"
    )


def _tier_base_description(definition: TierDefinition) -> str:
    """Return the base product's description enumerating every included allotment."""
    included_summaries = [
        f"{allotment.monthly_allotment:,} "
        + (
            _allotment_unit_noun(meter)
            if meter is UsageMeter.ADAPTER_TRAINING_UNITS
            else METER_DISPLAY_NAMES[meter].lower()
        )
        for meter, allotment in definition.meter_allotments.items()
    ]
    if len(included_summaries) > 1:
        included_text = (
            ", ".join(included_summaries[:-1]) + f" and {included_summaries[-1]}"
        )
    else:
        included_text = included_summaries[0]
    base_fee_text = (
        f"${definition.monthly_base_fee_usd:,.2f} per month"
        if definition.monthly_base_fee_usd > 0
        else "no monthly fee"
    )
    trial_text = (
        f" Includes a {definition.trial_period_days}-day free trial."
        if definition.trial_period_days > 0
        else ""
    )
    return (
        f"{definition.display_name} subscription ({base_fee_text}): "
        f"includes {included_text} per month.{trial_text}"
    )


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


def _find_product_by_metadata(
    tier: SubscriptionTier, role: str, meter: UsageMeter | None
) -> str | None:
    """Return the id of the active product matching the metadata triple, if any.

    Matching is scoped to the current catalog version so v1's single
    product-per-tier is never mistaken for a v2 base or metered product.
    """
    for existing in stripe.Product.list(active=True, limit=100).auto_paging_iter():
        existing_product = existing.to_dict()
        product_metadata = existing_product.get("metadata") or {}
        if (
            product_metadata.get(PRODUCT_TIER_METADATA_KEY) == tier.value
            and product_metadata.get(PRODUCT_ROLE_METADATA_KEY) == role
            and product_metadata.get(PRODUCT_CATALOG_VERSION_METADATA_KEY)
            == PRICE_LOOKUP_KEY_VERSION
            and product_metadata.get(PRODUCT_METER_METADATA_KEY)
            == (meter.value if meter else None)
        ):
            return existing_product["id"]
    return None


def find_or_create_base_product(definition: TierDefinition) -> str:
    """Return the id of a tier's self-describing BASE product, creating it if absent.

    The base product carries the flat subscription fee. Checkout shows the
    product name and description on the line item, so both enumerate what the
    tier includes. On reuse, the name and description are refreshed so re-runs
    keep the customer-facing copy in sync with ``tiers.py``.
    """
    product_name = f"{definition.display_name} — Base Subscription"
    product_description = _tier_base_description(definition)
    existing_id = _find_product_by_metadata(
        definition.tier, PRODUCT_ROLE_BASE, meter=None
    )
    if existing_id:
        stripe.Product.modify(
            existing_id, name=product_name, description=product_description
        )
        print(f"  base product '{definition.tier.value}' exists -> {existing_id}")
        return existing_id

    created = stripe.Product.create(
        name=product_name,
        description=product_description,
        metadata={
            PRODUCT_TIER_METADATA_KEY: definition.tier.value,
            PRODUCT_ROLE_METADATA_KEY: PRODUCT_ROLE_BASE,
            PRODUCT_CATALOG_VERSION_METADATA_KEY: PRICE_LOOKUP_KEY_VERSION,
        },
    )
    print(f"  base product '{definition.tier.value}' CREATED -> {created['id']}")
    return created["id"]


def find_or_create_meter_product(
    definition: TierDefinition, allotment: MeterAllotment
) -> str:
    """Return the id of one (tier, meter) self-describing product, creating it if absent.

    Every metered price gets a dedicated product whose name says which meter
    the line item bills and how much is included — the fix for Checkout
    rendering three indistinguishable "Neural Nexus Pro Tier" rows. On reuse,
    the name and description are refreshed to match ``tiers.py``.
    """
    product_name = f"{definition.display_name} — {_allotment_summary(allotment)}"
    product_description = _allotment_description(allotment)
    existing_id = _find_product_by_metadata(
        definition.tier, PRODUCT_ROLE_METERED, meter=allotment.meter
    )
    if existing_id:
        stripe.Product.modify(
            existing_id, name=product_name, description=product_description
        )
        print(
            f"  meter product '{definition.tier.value}/{allotment.meter.value}' "
            f"exists -> {existing_id}"
        )
        return existing_id

    created = stripe.Product.create(
        name=product_name,
        description=product_description,
        metadata={
            PRODUCT_TIER_METADATA_KEY: definition.tier.value,
            PRODUCT_ROLE_METADATA_KEY: PRODUCT_ROLE_METERED,
            PRODUCT_METER_METADATA_KEY: allotment.meter.value,
            PRODUCT_CATALOG_VERSION_METADATA_KEY: PRICE_LOOKUP_KEY_VERSION,
        },
    )
    print(
        f"  meter product '{definition.tier.value}/{allotment.meter.value}' "
        f"CREATED -> {created['id']}"
    )
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


def find_or_create_billing_portal_configuration() -> str:
    """Return the id of the Neural Nexus billing-portal configuration.

    The portal configuration defines what the Stripe-hosted customer portal
    exposes: invoice history, payment-method updates, and billing-information
    updates, plus at-period-end cancellation. Plan switching stays DISABLED here
    — the hosted portal cannot switch plans that contain metered prices, so tier
    changes go through the API's POST /subscribe. Matched by a
    metadata tag so re-runs reuse the existing configuration.
    """
    for existing in stripe.billing_portal.Configuration.list(
        active=True, limit=100
    ).auto_paging_iter():
        existing_configuration = existing.to_dict()
        tagged_value = (existing_configuration.get("metadata") or {}).get(
            PORTAL_CONFIGURATION_METADATA_KEY
        )
        if tagged_value == PORTAL_CONFIGURATION_METADATA_VALUE:
            print(
                f"  portal configuration exists -> {existing_configuration['id']}"
            )
            return existing_configuration["id"]

    created = stripe.billing_portal.Configuration.create(
        business_profile={
            "headline": "Neural Nexus — manage your subscription",
        },
        features={
            "invoice_history": {"enabled": True},
            "payment_method_update": {"enabled": True},
            "customer_update": {
                "enabled": True,
                "allowed_updates": ["email", "address", "phone", "name"],
            },
            "subscription_cancel": {
                "enabled": True,
                "mode": "at_period_end",
            },
        },
        metadata={
            PORTAL_CONFIGURATION_METADATA_KEY: PORTAL_CONFIGURATION_METADATA_VALUE
        },
    )
    print(f"  portal configuration CREATED -> {created['id']}")
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
        base_product_id = find_or_create_base_product(definition)
        base_price_id = find_or_create_base_price(definition, base_product_id)

        metered_price_ids: Dict[str, str] = {}
        for meter, allotment in definition.meter_allotments.items():
            # Each metered price lives under a dedicated self-describing
            # (tier, meter) product so Checkout line items name the meter and
            # the included allotment instead of repeating the tier name.
            meter_product_id = find_or_create_meter_product(definition, allotment)
            price_id = find_or_create_metered_price(
                definition, meter_product_id, meter_ids[meter], allotment
            )
            metered_price_ids[meter.value] = price_id

        tiers_config[tier.value] = {
            "product": base_product_id,
            "base_price": base_price_id,
            "metered_prices": metered_price_ids,
        }

    print("Provisioning billing-portal configuration:")
    portal_configuration_id = find_or_create_billing_portal_configuration()

    return {
        "meters": {meter.value: meter_id for meter, meter_id in meter_ids.items()},
        "tiers": tiers_config,
        "portal_configuration": portal_configuration_id,
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
    print(
        "\nMigration notes (v1 -> v2 catalog):\n"
        "  1. Paste the JSON above into STRIPE_BILLING_CONFIG_JSON and restart the API.\n"
        "  2. Existing subscriptions keep their v1 prices until their next tier\n"
        "     change; new checkouts and tier changes use the v2 prices above.\n"
        "  3. Archive the v1 prices and the old one-product-per-tier products by\n"
        "     hand once no live subscription references them.\n"
    )


if __name__ == "__main__":
    main()
