# scripts/e2e_billing/harness.py

"""Shared helpers for the end-to-end billing scenarios (Stripe TEST mode).

Every scenario script in this directory drives real Stripe objects in TEST
mode — customers pinned to Stripe Test Clocks, per-tier subscriptions built
from the provisioned price ids, Billing Meter events, and invoice inspection —
plus (optionally) the running API for gating assertions.

Requirements:

* ``STRIPE_SECRET_KEY`` — an ``sk_test_`` key (live keys are refused).
* ``STRIPE_BILLING_CONFIG_JSON`` — the JSON printed by
  ``scripts/provision_stripe_billing.py`` (same value the API uses).
* Optional, for API-side assertions: ``E2E_API_BASE_URL`` (for example
  ``http://localhost:8124``) and ``E2E_API_KEY`` (a signed-up user's API key).
  Scenarios that need the API SKIP those assertions when these are unset.
* The trial-path scenario needs the Stripe webhook forwarder running
  (``stripe listen --forward-to <api>/stripe/webhook``) so subscription
  lifecycle events reach the API.

Known test-clock limitations (asserted around, not against):

* The local ``api_metrics`` usage window uses real wall-clock time — a test
  clock cannot advance the local window, so period-reset assertions are
  Stripe-side only.
* Billing Meter events for a test-clock customer must carry a ``timestamp``
  inside the clock's current window; Stripe rejects events too far from the
  meter's notion of now.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from typing import Any

import stripe

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from src.anubis.utils.billing.config import (  # noqa: E402
    StripeBillingConfig,
    load_stripe_billing_config,
)
from src.anubis.utils.billing.tiers import (  # noqa: E402
    TIER_DEFINITIONS,
    SubscriptionTier,
    UsageMeter,
)

TEST_CLOCK_READY_TIMEOUT_SECONDS = 120.0


class ScenarioReporter:
    """Collect and print PASS/FAIL/SKIP lines; exit non-zero on any failure."""

    def __init__(self, scenario_name: str) -> None:
        """Print the scenario banner and start with an empty failure list."""
        self.scenario_name = scenario_name
        self.failures: list[str] = []
        print(f"\n=== {scenario_name} ===")

    def check(self, description: str, condition: bool, detail: str = "") -> None:
        """Record one assertion: PASS when the condition holds, FAIL otherwise."""
        if condition:
            print(f"  PASS  {description}")
        else:
            print(f"  FAIL  {description}" + (f" — {detail}" if detail else ""))
            self.failures.append(description)

    def skip(self, description: str, reason: str) -> None:
        """Report an assertion that could not run (missing environment)."""
        print(f"  SKIP  {description} — {reason}")

    def finish(self) -> int:
        """Print the scenario verdict; return the process exit code."""
        if self.failures:
            print(f"=== {self.scenario_name}: {len(self.failures)} FAILURE(S) ===")
            return 1
        print(f"=== {self.scenario_name}: all checks passed ===")
        return 0


def configure_stripe_test_mode() -> None:
    """Point the stripe module at the TEST key; refuse live keys outright."""
    secret_key = os.environ.get("STRIPE_SECRET_KEY", "")
    if not secret_key:
        raise SystemExit("STRIPE_SECRET_KEY is not set.")
    if not secret_key.startswith("sk_test_"):
        raise SystemExit(
            "The end-to-end billing scenarios run ONLY against a TEST key "
            "(sk_test_...)."
        )
    stripe.api_key = secret_key


def load_billing_config() -> StripeBillingConfig:
    """Parse STRIPE_BILLING_CONFIG_JSON (same document the API consumes)."""
    billing_config = load_stripe_billing_config(
        os.environ.get("STRIPE_BILLING_CONFIG_JSON")
    )
    if billing_config is None:
        raise SystemExit(
            "STRIPE_BILLING_CONFIG_JSON is not set. Run "
            "scripts/provision_stripe_billing.py and export the printed JSON."
        )
    return billing_config


def api_base_url() -> str | None:
    """Return the running API's base URL for API-side assertions, if configured."""
    return os.environ.get("E2E_API_BASE_URL") or None


def api_key() -> str | None:
    """Return the API key used for API-side assertions, if configured."""
    return os.environ.get("E2E_API_KEY") or None


def create_test_clock(label: str) -> dict:
    """Create a frozen test clock starting at real now."""
    return stripe.test_helpers.TestClock.create(
        frozen_time=int(time.time()), name=f"e2e-billing-{label}"
    ).to_dict()


def advance_test_clock(test_clock_id: str, to_epoch_seconds: int) -> None:
    """Advance a test clock and poll until every simulated object settles."""
    stripe.test_helpers.TestClock.advance(
        test_clock_id, frozen_time=to_epoch_seconds
    )
    poll_deadline = time.monotonic() + TEST_CLOCK_READY_TIMEOUT_SECONDS
    while time.monotonic() < poll_deadline:
        clock = stripe.test_helpers.TestClock.retrieve(test_clock_id).to_dict()
        if clock.get("status") == "ready":
            return
        time.sleep(2.0)
    raise TimeoutError(f"Test clock {test_clock_id} did not settle in time.")


def delete_test_clock(test_clock_id: str) -> None:
    """Best-effort cleanup (deleting the clock deletes its simulated objects)."""
    try:
        stripe.test_helpers.TestClock.delete(test_clock_id)
    except Exception as cleanup_error:  # noqa: BLE001 - cleanup only
        print(f"  (cleanup) could not delete test clock: {cleanup_error}")


def create_customer_on_clock(test_clock_id: str, label: str) -> dict:
    """Create a throwaway test customer pinned to the given test clock."""
    return stripe.Customer.create(
        email=f"e2e-{label}-{uuid.uuid4().hex[:8]}@example.com",
        test_clock=test_clock_id,
        metadata={"e2e_billing": "true"},
    ).to_dict()


def attach_test_payment_method(customer_id: str) -> None:
    """Attach the always-succeeding test card and make the card the default."""
    payment_method = stripe.PaymentMethod.attach("pm_card_visa", customer=customer_id)
    stripe.Customer.modify(
        customer_id,
        invoice_settings={"default_payment_method": payment_method["id"]},
    )


def subscription_items_for_tier(
    billing_config: StripeBillingConfig, tier: SubscriptionTier
) -> list[dict]:
    """Build one tier's subscription items: base price (quantity 1) + metered."""
    identifiers = billing_config.identifiers_for_tier(tier)
    items: list[dict] = [{"price": identifiers.base_price_id, "quantity": 1}]
    for metered_price_id in identifiers.metered_price_ids.values():
        items.append({"price": metered_price_id})
    return items


def create_subscription_for_tier(
    billing_config: StripeBillingConfig,
    customer_id: str,
    tier: SubscriptionTier,
    trial_period_days: int | None = None,
) -> dict:
    """Create a per-tier subscription exactly the way the API does."""
    create_kwargs: dict[str, Any] = {
        "customer": customer_id,
        "items": subscription_items_for_tier(billing_config, tier),
        "metadata": {"neural_nexus_tier": tier.value, "e2e_billing": "true"},
    }
    if trial_period_days:
        create_kwargs["trial_period_days"] = trial_period_days
        create_kwargs["trial_settings"] = {
            "end_behavior": {"missing_payment_method": "cancel"}
        }
    return stripe.Subscription.create(**create_kwargs).to_dict()


def emit_meter_event(
    meter: UsageMeter,
    customer_id: str,
    value: int,
    timestamp_epoch_seconds: int | None = None,
) -> dict:
    """Report usage for one customer, mirroring report_meter_event's payload."""
    event_kwargs: dict[str, Any] = {
        "event_name": meter.value,
        "payload": {"stripe_customer_id": customer_id, "value": str(value)},
        "identifier": f"e2e-{uuid.uuid4().hex}",
    }
    if timestamp_epoch_seconds is not None:
        event_kwargs["timestamp"] = timestamp_epoch_seconds
    return stripe.billing.MeterEvent.create(**event_kwargs).to_dict()


def latest_invoice_lines(customer_id: str) -> list[dict]:
    """Return the newest invoice's line items for assertion."""
    invoices = stripe.Invoice.list(customer=customer_id, limit=1).to_dict().get(
        "data", []
    )
    if not invoices:
        return []
    return (invoices[0].get("lines") or {}).get("data", [])


def invoice_amounts_by_description(customer_id: str) -> dict[str, int]:
    """Map every line description on the newest invoice to the amount in cents."""
    return {
        (line.get("description") or ""): int(line.get("amount") or 0)
        for line in latest_invoice_lines(customer_id)
    }


def subscription_period_bounds(subscription: dict) -> tuple[int | None, int | None]:
    """Items-first period bounds (flexible billing mode), mirroring the API."""
    items = (subscription.get("items") or {}).get("data") or []
    first_item = items[0] if items and isinstance(items[0], dict) else {}
    period_start = first_item.get("current_period_start") or subscription.get(
        "current_period_start"
    )
    period_end = first_item.get("current_period_end") or subscription.get(
        "current_period_end"
    )
    return (
        int(period_start) if period_start else None,
        int(period_end) if period_end else None,
    )


def print_config_summary() -> None:
    """One-line sanity output so a mis-pointed environment is obvious."""
    print(
        "Stripe key:",
        (os.environ.get("STRIPE_SECRET_KEY") or "")[:11] + "…",
        "| API:",
        api_base_url() or "(not configured — API assertions skipped)",
    )


def tier_overage_cents_per_token(tier: SubscriptionTier, meter: UsageMeter) -> float:
    """Return the expected Stripe overage price (cents per unit) for one (tier, meter)."""
    allotment = TIER_DEFINITIONS[tier].meter_allotments[meter]
    return float(allotment.stripe_unit_amount_decimal())


def dump(value: Any) -> str:
    """Pretty-print any Stripe object/dict for FAIL diagnostics."""
    return json.dumps(value, indent=2, default=str)
