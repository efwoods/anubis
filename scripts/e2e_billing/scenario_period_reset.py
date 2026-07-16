#!/usr/bin/env python
# scripts/e2e_billing/scenario_period_reset.py

"""Usage resets at the period boundary (Stripe-side, via a test clock).

Advancing the clock past ``current_period_end`` must open a fresh billing
period: the subscription's period bounds move forward and a new invoice is
issued for the new period, so metered usage from the previous period never
carries into the next invoice.

Known limitation, deliberately NOT asserted here: the API's local
``api_metrics`` usage window keys on real wall-clock time, which a Stripe test
clock cannot advance — the local reset is covered by the pure period-math unit
tests (``tests/unit_tests/test_usage_period_math.py``) instead.
"""

from __future__ import annotations

import sys

import stripe
from harness import (
    ScenarioReporter,
    advance_test_clock,
    configure_stripe_test_mode,
    create_customer_on_clock,
    create_subscription_for_tier,
    create_test_clock,
    delete_test_clock,
    emit_meter_event,
    load_billing_config,
    print_config_summary,
    subscription_period_bounds,
)

from src.anubis.utils.billing.tiers import SubscriptionTier, UsageMeter


def run() -> int:
    """Advance past the period end and assert a fresh Stripe billing period."""
    configure_stripe_test_mode()
    billing_config = load_billing_config()
    print_config_summary()
    reporter = ScenarioReporter("usage period reset at the boundary")

    test_clock = create_test_clock("period-reset")
    try:
        customer = create_customer_on_clock(test_clock["id"], "period-reset")
        subscription = create_subscription_for_tier(
            billing_config, customer["id"], SubscriptionTier.PRO
        )
        first_period_start, first_period_end = subscription_period_bounds(
            subscription
        )
        reporter.check(
            "subscription has period bounds",
            bool(first_period_start and first_period_end),
        )

        # Some in-period usage, safely under the pro allotment.
        emit_meter_event(
            UsageMeter.MESSAGING_TOKENS,
            customer["id"],
            100_000,
            timestamp_epoch_seconds=int(test_clock["frozen_time"]) + 60,
        )

        advance_test_clock(test_clock["id"], int(first_period_end) + 3600)

        refreshed_subscription = stripe.Subscription.retrieve(
            subscription["id"]
        ).to_dict()
        second_period_start, second_period_end = subscription_period_bounds(
            refreshed_subscription
        )
        reporter.check(
            "a fresh period opened at the boundary",
            bool(
                second_period_start
                and second_period_end
                and second_period_start >= first_period_end
                and second_period_end > first_period_end
            ),
            f"first=({first_period_start},{first_period_end}) "
            f"second=({second_period_start},{second_period_end})",
        )

        invoices = (
            stripe.Invoice.list(customer=customer["id"], limit=10)
            .to_dict()
            .get("data", [])
        )
        reporter.check(
            "the boundary produced a new invoice (usage settled per period)",
            len(invoices) >= 2,
            f"{len(invoices)} invoice(s)",
        )
    finally:
        delete_test_clock(test_clock["id"])

    return reporter.finish()


if __name__ == "__main__":
    sys.exit(run())
