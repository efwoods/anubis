#!/usr/bin/env python
# scripts/e2e_billing/scenario_tier_changes.py

"""Tier-change execution semantics (Stripe-side, mirroring POST /subscribe).

* Upgrade (pro -> premium): items swap immediately to the premium prices.
* Downgrade (premium -> pro): a Subscription Schedule holds the premium items
  until the period boundary; the second phase carries the pro prices ("unused
  allotment continues on downgrade"). Advancing past the boundary leaves the
  subscription running on pro items.

The local usage-window rules (upgrade resets the anchor, downgrade and every
trialing change retain the window) are pure logic covered by
``tests/unit_tests/test_billing_tiers_and_gating.py``.
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
    load_billing_config,
    print_config_summary,
    subscription_items_for_tier,
    subscription_period_bounds,
)

from src.anubis.utils.billing.tiers import SubscriptionTier


def _price_ids_on_subscription(subscription: dict) -> set[str]:
    return {
        ((item.get("price") or {}).get("id") or "")
        for item in (subscription.get("items") or {}).get("data", [])
    }


def run() -> int:
    """Exercise upgrade-immediately and downgrade-at-boundary on Stripe."""
    configure_stripe_test_mode()
    billing_config = load_billing_config()
    print_config_summary()
    reporter = ScenarioReporter("tier-change execution (upgrade + downgrade)")

    pro_price_ids = {
        item["price"]
        for item in subscription_items_for_tier(billing_config, SubscriptionTier.PRO)
    }
    premium_price_ids = {
        item["price"]
        for item in subscription_items_for_tier(
            billing_config, SubscriptionTier.PREMIUM
        )
    }

    # ---- Upgrade: pro -> premium, immediate item swap --------------------
    upgrade_clock = create_test_clock("upgrade")
    try:
        customer = create_customer_on_clock(upgrade_clock["id"], "upgrade")
        subscription = create_subscription_for_tier(
            billing_config, customer["id"], SubscriptionTier.PRO
        )
        items_payload: list[dict] = [
            {"id": item["id"], "deleted": True}
            for item in subscription["items"]["data"]
        ] + [{"price": price_id} for price_id in sorted(premium_price_ids)]
        upgraded = stripe.Subscription.modify(
            subscription["id"],
            items=items_payload,
            proration_behavior="always_invoice",
        ).to_dict()
        reporter.check(
            "upgrade swaps every item to premium prices immediately",
            _price_ids_on_subscription(upgraded) == premium_price_ids,
            f"items now: {_price_ids_on_subscription(upgraded)}",
        )
    finally:
        delete_test_clock(upgrade_clock["id"])

    # ---- Downgrade: premium -> pro via schedule at the boundary ----------
    downgrade_clock = create_test_clock("downgrade")
    try:
        customer = create_customer_on_clock(downgrade_clock["id"], "downgrade")
        subscription = create_subscription_for_tier(
            billing_config, customer["id"], SubscriptionTier.PREMIUM
        )
        schedule = stripe.SubscriptionSchedule.create(
            from_subscription=subscription["id"]
        ).to_dict()
        current_phase = (schedule.get("phases") or [{}])[0]
        _, current_period_end = subscription_period_bounds(subscription)
        stripe.SubscriptionSchedule.modify(
            schedule["id"],
            end_behavior="release",
            phases=[
                {
                    "items": [
                        {"price": (item.get("price") or {}).get("id")}
                        for item in subscription["items"]["data"]
                    ],
                    "start_date": current_phase.get("start_date"),
                    "end_date": current_period_end,
                },
                {
                    "items": [
                        {"price": price_id} for price_id in sorted(pro_price_ids)
                    ],
                    "iterations": 1,
                },
            ],
        )
        before_boundary = stripe.Subscription.retrieve(subscription["id"]).to_dict()
        reporter.check(
            "downgrade keeps premium items until the boundary",
            _price_ids_on_subscription(before_boundary) == premium_price_ids,
            f"items now: {_price_ids_on_subscription(before_boundary)}",
        )

        advance_test_clock(downgrade_clock["id"], int(current_period_end) + 3600)
        after_boundary = stripe.Subscription.retrieve(subscription["id"]).to_dict()
        reporter.check(
            "the boundary switches the subscription to pro items",
            _price_ids_on_subscription(after_boundary) == pro_price_ids,
            f"items now: {_price_ids_on_subscription(after_boundary)}",
        )
        reporter.check(
            "the subscription stays running after the schedule releases",
            after_boundary.get("status") in ("active", "trialing"),
            after_boundary.get("status") or "no status",
        )
    finally:
        delete_test_clock(downgrade_clock["id"])

    return reporter.finish()


if __name__ == "__main__":
    sys.exit(run())
