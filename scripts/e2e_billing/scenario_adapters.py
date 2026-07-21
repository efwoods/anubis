#!/usr/bin/env python
# scripts/e2e_billing/scenario_adapters.py

"""Adapter inference and adapter training meters (premium tier, Stripe-side).

Adapter inference: premium usage past the 10M-token allotment bills overage at
$4.00 per million. Adapter training: the sixth trained adapter in a period
bills $5.00 (the first five are included). The training ENDPOINT does not
exist yet (Phase 7) — this scenario exercises the meter/invoice math the
future endpoint will report through ``report_adapter_training_usage``, so the
billing side is verified before the endpoint lands.
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
    """Bill premium adapter inference overage and the sixth trained adapter."""
    configure_stripe_test_mode()
    billing_config = load_billing_config()
    print_config_summary()
    reporter = ScenarioReporter("adapter inference + training meters (premium)")

    premium_definition = TIER_DEFINITIONS[SubscriptionTier.PREMIUM]
    inference_allotment = premium_definition.meter_allotments[
        UsageMeter.ADAPTER_INFERENCE_TOKENS
    ]
    training_allotment = premium_definition.meter_allotments[
        UsageMeter.ADAPTER_TRAINING_UNITS
    ]

    test_clock = create_test_clock("adapters-premium")
    try:
        customer = create_customer_on_clock(test_clock["id"], "adapters-premium")
        subscription = create_subscription_for_tier(
            billing_config, customer["id"], SubscriptionTier.PREMIUM
        )
        event_timestamp = int(test_clock["frozen_time"]) + 60

        # Inference: allotment + exactly 1M tokens over.
        inference_overage_tokens = 1_000_000
        emit_meter_event(
            UsageMeter.ADAPTER_INFERENCE_TOKENS,
            customer["id"],
            inference_allotment.monthly_allotment + inference_overage_tokens,
            timestamp_epoch_seconds=event_timestamp,
        )
        # Training: the whole allotment (5) plus one over.
        emit_meter_event(
            UsageMeter.ADAPTER_TRAINING_UNITS,
            customer["id"],
            training_allotment.monthly_allotment + 1,
            timestamp_epoch_seconds=event_timestamp,
        )

        _, period_end = subscription_period_bounds(subscription)
        advance_test_clock(test_clock["id"], int(period_end) + 3600)

        amounts = invoice_amounts_by_description(customer["id"])
        positive_amounts = sorted(
            amount for amount in amounts.values() if amount > 0
        )
        expected_inference_overage_cents = round(
            inference_overage_tokens
            * tier_overage_cents_per_token(
                SubscriptionTier.PREMIUM, UsageMeter.ADAPTER_INFERENCE_TOKENS
            )
        )
        expected_training_overage_cents = round(
            1
            * tier_overage_cents_per_token(
                SubscriptionTier.PREMIUM, UsageMeter.ADAPTER_TRAINING_UNITS
            )
        )
        reporter.check(
            "adapter inference overage billed at $4.00/million "
            f"({expected_inference_overage_cents} cents)",
            expected_inference_overage_cents in positive_amounts,
            f"invoice lines: {amounts}",
        )
        reporter.check(
            "sixth trained adapter billed at $5.00 "
            f"({expected_training_overage_cents} cents)",
            expected_training_overage_cents in positive_amounts,
            f"invoice lines: {amounts}",
        )
    finally:
        delete_test_clock(test_clock["id"])

    return reporter.finish()


if __name__ == "__main__":
    sys.exit(run())
