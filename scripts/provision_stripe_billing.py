#!/usr/bin/env python
# scripts/provision_stripe_billing.py

"""Create — and MUTATE — the Stripe meters, products, and prices for all three tiers.

This script turns the tier catalog in ``src/anubis/utils/billing/tiers.py`` into
concrete Stripe objects. Editing a number there (a base fee, a monthly allotment,
an overage rate) and re-running this script is how pricing changes: the run
notices the live price no longer matches, replaces it, and moves existing
subscribers onto the replacement, so the customer portal, this API, and the
Stripe invoice all quote the same figure.

Which Stripe account it touches is decided by the environment file, not by an
ambient shell variable: ``.env.dev`` (test, ``sk_test_``) by default, ``.env``
(live, ``sk_live_``) with ``--live``. A key that does not match the file it came
from is a hard error.

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

How a price is mutated:

* Meters are matched by ``event_name`` (Stripe forbids two active meters sharing
  one), and reused when present.
* Products are matched by the metadata triple ``neural_nexus_tier`` +
  ``neural_nexus_product_role`` (+ ``neural_nexus_meter`` for metered
  products) at the current ``neural_nexus_catalog_version``; on reuse the
  name and description are refreshed so re-runs keep the customer-facing copy
  in sync with ``tiers.py``.
* Prices are matched by ``lookup_key`` and then COMPARED against ``tiers.py``.
  A Stripe price is immutable in amount and graduated tiers, so a changed number
  means a new price: the replacement is created with ``transfer_lookup_key=True``
  (which atomically moves the lookup key off the old price) and the old price is
  then archived. The lookup key — not the price id — is the stable name for "the
  current pro base price", which is what makes a re-run find and compare the
  right object.
* Live subscriptions still holding a replaced price are migrated onto the
  replacement, because the API gates allotments on ``tiers.py`` while Stripe
  bills whatever price the subscription item holds; leaving those apart charges
  a customer something different from what the portal shows them.

``PRICE_LOOKUP_KEY_VERSION`` is the catalog GENERATION. It is not how a price is
changed — bumping it renames every lookup key and re-tags every product. Edit
``tiers.py`` instead.

Usage:

    # test environment: reads .env.dev, requires an sk_test_ key
    python scripts/provision_stripe_billing.py

    # live environment: reads .env, requires an sk_live_ key
    python scripts/provision_stripe_billing.py --live
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Mapping

import stripe
from dotenv import dotenv_values

# Allow running as a plain script from the repo root without installing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.anubis.utils.billing.tiers import (  # noqa: E402
    TIER_DEFINITIONS,
    MeterAllotment,
    SubscriptionTier,
    TierDefinition,
    UsageMeter,
)

# The catalog generation, embedded in every price lookup key. See the module
# docstring for why bumping this is NOT how a price is changed.
PRICE_LOOKUP_KEY_VERSION = "v2"
PRODUCT_TIER_METADATA_KEY = "neural_nexus_tier"
PRODUCT_ROLE_METADATA_KEY = "neural_nexus_product_role"
PRODUCT_METER_METADATA_KEY = "neural_nexus_meter"
PRODUCT_CATALOG_VERSION_METADATA_KEY = "neural_nexus_catalog_version"
PRODUCT_ROLE_BASE = "base"
PRODUCT_ROLE_METERED = "metered"
PORTAL_CONFIGURATION_METADATA_KEY = "neural_nexus_portal"
PORTAL_CONFIGURATION_METADATA_VALUE = f"portal_{PRICE_LOOKUP_KEY_VERSION}"

SUBSCRIPTION_TIER_METADATA_KEY = "neural_nexus_tier"

# Statuses in which a subscription is still billing and therefore still needs to
# hold current prices. Mirrors LIVE_SUBSCRIPTION_STATUSES in the billing package.
LIVE_SUBSCRIPTION_STATUSES = ("active", "trialing", "past_due", "unpaid")

# Which environment file supplies the Stripe key for each mode.
TEST_ENVIRONMENT_FILE = ".env.dev"
LIVE_ENVIRONMENT_FILE = ".env"

# Customer-facing names for each usage meter, shown on Checkout line items.
METER_DISPLAY_NAMES: Dict[UsageMeter, str] = {
    UsageMeter.MESSAGING_TOKENS: "Messaging Tokens",
    UsageMeter.DOCUMENT_UPLOAD_TOKENS: "Document Upload Tokens",
    UsageMeter.ADAPTER_INFERENCE_TOKENS: "Adapter Inference Tokens",
    UsageMeter.ADAPTER_TRAINING_UNITS: "Adapter Training",
    UsageMeter.SPEECH_CHARACTERS: "Speech Characters",
    UsageMeter.VIDEO_GENERATION_SECONDS: "Video Generation Seconds",
}


def resolve_stripe_secret_key(use_live: bool) -> tuple[str, str]:
    """Return the Stripe secret key for the selected environment, and its source.

    The environment FILE decides which Stripe account is touched — ``.env.dev``
    for test, ``.env`` for live — because that is the separation the rest of the
    project already uses. The file is parsed with ``dotenv_values`` rather than
    loaded into the process: ``load_dotenv(override=True)`` would also overwrite
    variables docker-compose supplies through ``environment:``, including the
    empty ``STRIPE_BILLING_CONFIG_FILE=`` line these files carry.

    Falling back to the process environment covers a container that was handed a
    key without a mounted env file. Either way the key must match the environment
    it claims to be: an ``sk_live_`` key exported in a shell is exactly how a run
    intended for test would otherwise rewrite live prices.
    """
    environment_file = LIVE_ENVIRONMENT_FILE if use_live else TEST_ENVIRONMENT_FILE
    expected_prefix = "sk_live_" if use_live else "sk_test_"
    environment_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", environment_file)
    )

    file_values: Mapping[str, str | None] = (
        dotenv_values(environment_path) if os.path.isfile(environment_path) else {}
    )
    secret_key = (file_values.get("STRIPE_SECRET_KEY") or "").strip()
    source = environment_file
    if not secret_key:
        secret_key = (os.environ.get("STRIPE_SECRET_KEY") or "").strip()
        source = "the process environment"

    if not secret_key:
        raise SystemExit(
            f"No STRIPE_SECRET_KEY found in {environment_file} or the process "
            "environment."
        )
    if not secret_key.startswith(expected_prefix):
        actual_prefix = secret_key.split("_")[0:2]
        raise SystemExit(
            f"Refusing to run: the key from {source} starts with "
            f"'{'_'.join(actual_prefix)}_' but the "
            f"{'live' if use_live else 'test'} environment requires "
            f"'{expected_prefix}'. Check {environment_file}"
            + (
                ", or unset STRIPE_SECRET_KEY in your shell."
                if source == "the process environment"
                else "."
            )
        )
    return secret_key, source


def _base_price_lookup_key(tier: SubscriptionTier) -> str:
    return f"nn_{tier.value}_base_{PRICE_LOOKUP_KEY_VERSION}"


def _metered_price_lookup_key(tier: SubscriptionTier, meter: UsageMeter) -> str:
    return f"nn_{tier.value}_{meter.value}_{PRICE_LOOKUP_KEY_VERSION}"


def _allotment_unit_noun(meter: UsageMeter) -> str:
    """Return the plural unit noun the customer reads for one meter."""
    if meter is UsageMeter.ADAPTER_TRAINING_UNITS:
        return "trained adapters"
    if meter is UsageMeter.SPEECH_CHARACTERS:
        return "spoken characters"
    if meter is UsageMeter.VIDEO_GENERATION_SECONDS:
        return "seconds of video"
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
    keep the customer-facing copy in sync with ``tiers.py`` — which is what makes
    an edited price show up in the copy as well as the amount.
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
    rendering three indistinguishable "Neural Nexus Pro Tier" rows. On reuse, the
    name and description are refreshed to match ``tiers.py``.
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


def _find_active_price_by_lookup_key(
    lookup_key: str, expand_tiers: bool = False
) -> dict | None:
    """Return the active price holding ``lookup_key``, or None.

    ``expand_tiers`` is required for metered prices: without the graduated tiers
    there is no way to tell whether the live price still carries the allotment
    and overage rate ``tiers.py`` calls for, and an uncomparable price would be
    reused forever.
    """
    prices = stripe.Price.list(
        lookup_keys=[lookup_key],
        active=True,
        limit=1,
        **({"expand": ["data.tiers"]} if expand_tiers else {}),
    ).to_dict()
    data = prices.get("data") or []
    return data[0] if data else None


def _as_decimal(value: Any) -> Decimal | None:
    """Coerce a Stripe money field to ``Decimal``, or None when unusable.

    Stripe returns ``unit_amount_decimal`` as a ``Decimal`` through the SDK but as
    a string over raw HTTP, and test fixtures naturally use ints and floats. All
    of those have to compare equal to the same rate, and going through ``str``
    first keeps a float from contributing its own representation error.
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _cents_text(cents: int | None) -> str:
    return "unset" if cents is None else f"${cents / 100:,.2f}"


def base_price_differences(
    existing_price: Mapping[str, Any], definition: TierDefinition, product_id: str
) -> List[str]:
    """Return why a live flat price no longer matches ``tiers.py`` (empty when it does)."""
    differences: List[str] = []
    desired_amount = definition.stripe_base_unit_amount_cents()
    if existing_price.get("unit_amount") != desired_amount:
        differences.append(
            f"base fee {_cents_text(existing_price.get('unit_amount'))} -> "
            f"{_cents_text(desired_amount)}"
        )
    if existing_price.get("currency") != "usd":
        differences.append(f"currency {existing_price.get('currency')!r} -> 'usd'")
    if existing_price.get("product") != product_id:
        differences.append(
            f"product {existing_price.get('product')!r} -> {product_id!r}"
        )
    recurring = existing_price.get("recurring") or {}
    if recurring.get("interval") != "month":
        differences.append(f"interval {recurring.get('interval')!r} -> 'month'")
    return differences


def metered_price_differences(
    existing_price: Mapping[str, Any],
    allotment: MeterAllotment,
    product_id: str,
    meter_id: str,
) -> List[str]:
    """Return why a live metered price no longer matches ``tiers.py``.

    Money is compared as ``Decimal`` so that ``"0.00015"``, ``Decimal("0.000150")``
    and ``0.00015`` are one value — otherwise a re-run would "detect" a change
    that is only a difference of representation and mint a new price every time.
    """
    differences: List[str] = []
    if existing_price.get("product") != product_id:
        differences.append(
            f"product {existing_price.get('product')!r} -> {product_id!r}"
        )
    recurring = existing_price.get("recurring") or {}
    if recurring.get("meter") != meter_id:
        differences.append(f"meter {recurring.get('meter')!r} -> {meter_id!r}")
    if recurring.get("interval") != "month":
        differences.append(f"interval {recurring.get('interval')!r} -> 'month'")
    if existing_price.get("billing_scheme") != "tiered":
        differences.append(
            f"billing_scheme {existing_price.get('billing_scheme')!r} -> 'tiered'"
        )
    if existing_price.get("tiers_mode") != "graduated":
        differences.append(
            f"tiers_mode {existing_price.get('tiers_mode')!r} -> 'graduated'"
        )

    graduated_tiers = existing_price.get("tiers")
    if graduated_tiers is None:
        differences.append("graduated tiers were not expanded on the existing price")
        return differences
    if len(graduated_tiers) != 2:
        differences.append(f"{len(graduated_tiers)} graduated tiers -> 2")
        return differences

    included_tier, overage_tier = graduated_tiers
    if included_tier.get("up_to") != allotment.monthly_allotment:
        differences.append(
            f"allotment {included_tier.get('up_to')} -> "
            f"{allotment.monthly_allotment:,}"
        )
    desired_rate = Decimal(allotment.stripe_unit_amount_decimal())
    existing_rate = _as_decimal(
        overage_tier.get("unit_amount_decimal")
        if overage_tier.get("unit_amount_decimal") is not None
        else overage_tier.get("unit_amount")
    )
    if existing_rate is None or existing_rate != desired_rate:
        differences.append(
            f"overage rate {overage_tier.get('unit_amount_decimal')} -> "
            f"{allotment.stripe_unit_amount_decimal()} cents/unit"
        )
    return differences


def _replace_price(
    existing_price_id: str,
    lookup_key: str,
    differences: List[str],
    create_parameters: Dict[str, Any],
) -> str:
    """Create the replacement for a drifted price and archive the old one.

    The order is not interchangeable. ``transfer_lookup_key=True`` moves the
    lookup key off the existing price as part of the create; archiving first
    would strand the key on an inactive price and the next run would not find it.
    Archiving does not disturb customers — Stripe keeps billing and renewing
    subscriptions that reference an archived price — and ``migrate_live_subscriptions``
    moves them off it afterwards.
    """
    print(f"    price '{lookup_key}' CHANGED: {'; '.join(differences)}")
    replacement = stripe.Price.create(
        transfer_lookup_key=True, **create_parameters
    )
    stripe.Price.modify(existing_price_id, active=False)
    print(
        f"    price '{lookup_key}' REPLACED {existing_price_id} -> "
        f"{replacement['id']} (old price archived)"
    )
    return replacement["id"]


def find_or_create_base_price(
    definition: TierDefinition,
    product_id: str,
    superseded_price_tiers: Dict[str, str],
) -> str:
    """Return the licensed flat monthly base price id for a tier, replacing it on change."""
    lookup_key = _base_price_lookup_key(definition.tier)
    create_parameters: Dict[str, Any] = {
        "product": product_id,
        "currency": "usd",
        "unit_amount": definition.stripe_base_unit_amount_cents(),
        "recurring": {"interval": "month"},
        "lookup_key": lookup_key,
        "nickname": f"{definition.display_name} — base",
    }

    existing = _find_active_price_by_lookup_key(lookup_key)
    if existing is None:
        created = stripe.Price.create(**create_parameters)
        print(f"    base price '{lookup_key}' CREATED -> {created['id']}")
        return created["id"]

    differences = base_price_differences(existing, definition, product_id)
    if not differences:
        print(f"    base price '{lookup_key}' exists -> {existing['id']}")
        return existing["id"]

    replacement_id = _replace_price(
        existing["id"], lookup_key, differences, create_parameters
    )
    superseded_price_tiers[existing["id"]] = definition.tier.value
    return replacement_id


def find_or_create_metered_price(
    definition: TierDefinition,
    product_id: str,
    meter_id: str,
    allotment: MeterAllotment,
    superseded_price_tiers: Dict[str, str],
) -> str:
    """Return the graduated metered price id for one (tier, meter), replacing it on change."""
    lookup_key = _metered_price_lookup_key(definition.tier, allotment.meter)
    create_parameters: Dict[str, Any] = {
        "product": product_id,
        "currency": "usd",
        "recurring": {
            "interval": "month",
            "usage_type": "metered",
            "meter": meter_id,
        },
        "billing_scheme": "tiered",
        "tiers_mode": "graduated",
        "tiers": [
            # Tier 1: the included monthly allotment, at zero cost.
            {"up_to": allotment.monthly_allotment, "unit_amount_decimal": "0"},
            # Tier 2: pay-per-use overage past the allotment.
            {"up_to": "inf", "unit_amount_decimal": allotment.stripe_unit_amount_decimal()},
        ],
        "lookup_key": lookup_key,
        "nickname": f"{definition.display_name} — {allotment.meter.value}",
    }

    existing = _find_active_price_by_lookup_key(lookup_key, expand_tiers=True)
    if existing is None:
        created = stripe.Price.create(**create_parameters)
        print(f"    metered price '{lookup_key}' CREATED -> {created['id']}")
        return created["id"]

    differences = metered_price_differences(existing, allotment, product_id, meter_id)
    if not differences:
        print(f"    metered price '{lookup_key}' exists -> {existing['id']}")
        return existing["id"]

    replacement_id = _replace_price(
        existing["id"], lookup_key, differences, create_parameters
    )
    superseded_price_tiers[existing["id"]] = definition.tier.value
    return replacement_id


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


def migrate_live_subscriptions(
    superseded_price_tiers: Dict[str, str],
    price_ids_by_tier: Dict[str, List[str]],
) -> int:
    """Move live subscriptions off the prices this run replaced. Returns the count.

    Without this a price change only reaches new customers: existing subscribers
    keep billing the archived price while the API grants them the allotment from
    the edited ``tiers.py`` and the portal displays the edited figure — the
    customer is quoted one thing and charged another.

    Proration is deliberately ``none``. The customer did not ask for this change,
    so an operator's price edit must not produce a mid-period charge or credit.
    Usage already metered this period is unaffected: meter events key to the
    CUSTOMER (``stripe_customer_id``), not to the subscription item, so the
    replacement item continues to bill against the same accumulated total.

    Subscriptions managed by a subscription schedule are reported and skipped —
    Stripe refuses item changes on them, and releasing the schedule to force one
    through would silently discard a pending downgrade the customer requested.
    """
    if not superseded_price_tiers:
        return 0

    migrated_count = 0
    for subscription_object in stripe.Subscription.list(
        status="all", limit=100
    ).auto_paging_iter():
        subscription = subscription_object.to_dict()
        if subscription.get("status") not in LIVE_SUBSCRIPTION_STATUSES:
            continue

        existing_items = (subscription.get("items", {}) or {}).get("data", [])
        held_superseded_tiers = [
            superseded_price_tiers[price_id]
            for price_id in (
                (item.get("price") or {}).get("id") for item in existing_items
            )
            if price_id in superseded_price_tiers
        ]
        if not held_superseded_tiers:
            continue

        subscription_id = str(subscription["id"])
        tier = held_superseded_tiers[0]
        if subscription.get("schedule"):
            print(
                f"  subscription {subscription_id} ({tier}) holds a replaced price "
                "but is managed by a subscription schedule; NOT migrated. Let the "
                "pending change land, then re-run."
            )
            continue

        # Same item-replacement shape the API and the customer portal use for a
        # tier change: delete every current item, then add the tier's current
        # price set with quantity only on the licensed base price.
        target_price_ids = price_ids_by_tier[tier]
        replacement_items: List[Dict[str, Any]] = [
            {"id": item["id"], "deleted": True} for item in existing_items
        ]
        replacement_items.append({"price": target_price_ids[0], "quantity": 1})
        replacement_items.extend(
            {"price": price_id} for price_id in target_price_ids[1:]
        )

        stripe.Subscription.modify(
            subscription_id,
            items=replacement_items,
            proration_behavior="none",
            metadata={SUBSCRIPTION_TIER_METADATA_KEY: tier},
        )
        migrated_count += 1
        print(f"  subscription {subscription_id} MIGRATED to current {tier} prices")

    return migrated_count


def provision() -> Dict[str, Any]:
    """Create/reuse/replace every object and return the billing-config JSON document."""
    print("Provisioning Billing Meters:")
    meter_ids: Dict[UsageMeter, str] = {
        meter: find_or_create_meter(meter) for meter in UsageMeter
    }

    # Old price id -> the tier it priced, for every price replaced in this run.
    superseded_price_tiers: Dict[str, str] = {}
    tiers_config: Dict[str, Any] = {}
    price_ids_by_tier: Dict[str, List[str]] = {}

    for tier, definition in TIER_DEFINITIONS.items():
        print(f"Provisioning tier '{tier.value}':")
        base_product_id = find_or_create_base_product(definition)
        base_price_id = find_or_create_base_price(
            definition, base_product_id, superseded_price_tiers
        )

        metered_price_ids: Dict[str, str] = {}
        for meter, allotment in definition.meter_allotments.items():
            # Each metered price lives under a dedicated self-describing
            # (tier, meter) product so Checkout line items name the meter and
            # the included allotment instead of repeating the tier name.
            meter_product_id = find_or_create_meter_product(definition, allotment)
            price_id = find_or_create_metered_price(
                definition,
                meter_product_id,
                meter_ids[meter],
                allotment,
                superseded_price_tiers,
            )
            metered_price_ids[meter.value] = price_id

        tiers_config[tier.value] = {
            "product": base_product_id,
            "base_price": base_price_id,
            "metered_prices": metered_price_ids,
        }
        price_ids_by_tier[tier.value] = [base_price_id] + [
            metered_price_ids[meter.value]
            for meter in UsageMeter
            if meter.value in metered_price_ids
        ]

    if superseded_price_tiers:
        print("Migrating live subscriptions onto the replaced prices:")
        migrated_count = migrate_live_subscriptions(
            superseded_price_tiers, price_ids_by_tier
        )
        print(
            f"  {len(superseded_price_tiers)} price(s) replaced, "
            f"{migrated_count} subscription(s) migrated"
        )

    print("Provisioning billing-portal configuration:")
    portal_configuration_id = find_or_create_billing_portal_configuration()

    return {
        "meters": {meter.value: meter_id for meter, meter_id in meter_ids.items()},
        "tiers": tiers_config,
        "portal_configuration": portal_configuration_id,
    }


def main() -> None:
    """Parse arguments, select the environment, provision objects, and print config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            f"Provision the LIVE environment: read the Stripe key from "
            f"{LIVE_ENVIRONMENT_FILE} (sk_live_). Without this the key comes from "
            f"{TEST_ENVIRONMENT_FILE} (sk_test_)."
        ),
    )
    arguments = parser.parse_args()

    secret_key, key_source = resolve_stripe_secret_key(arguments.live)
    stripe.api_key = secret_key
    mode = "LIVE" if arguments.live else "TEST"
    print(f"=== Provisioning Stripe billing objects in {mode} mode ({key_source}) ===")

    config_document = provision()

    config_json = json.dumps(config_document, separators=(",", ":"))

    # File handoff (mirrors the stripe-cli webhook-secret handoff): write the
    # config JSON to STRIPE_BILLING_CONFIG_FILE so the API can read it without
    # STRIPE_BILLING_CONFIG_JSON being pasted into the env. Under docker-compose
    # this path is the shared /run/stripe volume the API also mounts; the compose
    # stripe-provision service runs this script into that volume on `up`.
    # Best-effort: a write failure (e.g. host run where /run/stripe is absent)
    # must not fail provisioning — the printed JSON below is always the fallback.
    config_file = os.environ.get(
        "STRIPE_BILLING_CONFIG_FILE", "/run/stripe/billing_config.json"
    )
    try:
        os.makedirs(os.path.dirname(config_file) or ".", exist_ok=True)
        # Atomic-ish replace so the API never reads a half-written config.
        temporary_path = f"{config_file}.tmp"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            handle.write(config_json)
        os.replace(temporary_path, config_file)
        print(f"\n=== Wrote billing config to {config_file} ===")
    except OSError as write_error:
        print(
            f"\n=== Could not write billing config to {config_file}: {write_error} ===\n"
            "    Falling back to the printed JSON below."
        )

    print("\n=== DONE. Set STRIPE_BILLING_CONFIG_JSON to the following single line: ===\n")
    print(config_json)


if __name__ == "__main__":
    main()
