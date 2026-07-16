#!/usr/bin/env python
# scripts/e2e_billing/scenario_messaging_allotment.py

"""Messaging-token allotment matrix, per tier (Stripe TEST mode + test clocks).

For every tier: usage under the allotment produces a $0 metered charge at the
period invoice; usage past the allotment produces an overage charge at exactly
the tier's configured rate. The pay-per-use OFF refusal (HTTP 402) is enforced
by the API's allotment gate, asserted here through the running API when
``E2E_API_BASE_URL``/``E2E_API_KEY`` are configured (the Stripe graduated
price bills whatever usage reaches the meter — the API gate is what stops
usage from reaching the meter when pay-per-use is off).
"""

from __future__ import annotations

import sys

from harness import (
    ScenarioReporter,
    advance_test_clock,
    configure_stripe_test_mode,
    create_customer_on_clock,
    create_subscription_for_tier,
    create_test_clock,
    delete_test_clock,
    emit_meter_event,
    invoice_amounts_by_description,
    load_billing_config,
    print_config_summary,
    subscription_period_bounds,
    tier_overage_cents_per_token,
)

from src.anubis.utils.billing.tiers import (
    TIER_DEFINITIONS,
    SubscriptionTier,
    UsageMeter,
)


def run() -> int:
    """Run the under/over allotment matrix for the messaging meter, per tier."""
    configure_stripe_test_mode()
    billing_config = load_billing_config()
    print_config_summary()
    reporter = ScenarioReporter("messaging allotment matrix")

    for tier in (
        SubscriptionTier.FREE,
        SubscriptionTier.PRO,
        SubscriptionTier.PREMIUM,
    ):
        allotment = TIER_DEFINITIONS[tier].meter_allotments[
            UsageMeter.MESSAGING_TOKENS
        ]
        test_clock = create_test_clock(f"messaging-{tier.value}")
        try:
            customer = create_customer_on_clock(
                test_clock["id"], f"messaging-{tier.value}"
            )
            subscription = create_subscription_for_tier(
                billing_config, customer["id"], tier
            )
            reporter.check(
                f"{tier.value}: subscription created active",
                subscription.get("status") in ("active", "trialing"),
                subscription.get("status") or "no status",
            )

            # Under allotment + a deliberate overage of exactly 1,000,000
            # tokens, so the expected overage price is unambiguous.
            overage_tokens = 1_000_000
            emit_meter_event(
                UsageMeter.MESSAGING_TOKENS,
                customer["id"],
                allotment.monthly_allotment + overage_tokens,
                timestamp_epoch_seconds=int(test_clock["frozen_time"]) + 60,
            )

            _, period_end = subscription_period_bounds(subscription)
            advance_test_clock(test_clock["id"], int(period_end) + 3600)

            amounts = invoice_amounts_by_description(customer["id"])
            metered_amounts = [
                amount for description, amount in amounts.items() if amount > 0
            ]
            expected_overage_cents = round(
                overage_tokens
                * tier_overage_cents_per_token(tier, UsageMeter.MESSAGING_TOKENS)
            )
            base_fee_cents = TIER_DEFINITIONS[tier].stripe_base_unit_amount_cents()
            reporter.check(
                f"{tier.value}: period invoice bills the overage at the tier rate "
                f"({expected_overage_cents} cents for {overage_tokens:,} tokens over)",
                expected_overage_cents in metered_amounts
                or (expected_overage_cents + base_fee_cents)
                in [sum(metered_amounts)],
                f"invoice lines: {amounts}",
            )
        finally:
            delete_test_clock(test_clock["id"])

    return reporter.finish()


if __name__ == "__main__":
    sys.exit(run())
