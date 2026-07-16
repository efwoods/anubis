#!/usr/bin/env python
# scripts/e2e_billing/scenario_trial_paths.py

"""Pro free-trial expiry paths (Stripe-side, via test clocks).

* Trial WITHOUT a payment method: ``trial_settings.end_behavior.
  missing_payment_method = "cancel"`` cancels the subscription at trial end.
  In production the ``customer.subscription.deleted`` webhook then auto-creates
  the $0 free-tier subscription (asserted here only when the API + the
  ``stripe listen`` forwarder are running, because the webhook is the actor).
* Trial WITH a payment method: at trial end the subscription converts to a
  paying pro subscription (status ``active``).
"""

from __future__ import annotations

import sys
import time

import stripe
from harness import (
    ScenarioReporter,
    advance_test_clock,
    api_base_url,
    attach_test_payment_method,
    configure_stripe_test_mode,
    create_customer_on_clock,
    create_subscription_for_tier,
    create_test_clock,
    delete_test_clock,
    load_billing_config,
    print_config_summary,
)

from src.anubis.utils.billing.tiers import TIER_DEFINITIONS, SubscriptionTier

TRIAL_DAYS = TIER_DEFINITIONS[SubscriptionTier.PRO].trial_period_days
SECONDS_PER_DAY = 86_400


def run() -> int:
    """Expire a card-less trial (cancel) and a carded trial (convert to active)."""
    configure_stripe_test_mode()
    billing_config = load_billing_config()
    print_config_summary()
    reporter = ScenarioReporter("pro free-trial expiry paths")

    # ---- Trial without a payment method: cancels at trial end ------------
    cardless_clock = create_test_clock("trial-cardless")
    try:
        customer = create_customer_on_clock(cardless_clock["id"], "trial-cardless")
        subscription = create_subscription_for_tier(
            billing_config,
            customer["id"],
            SubscriptionTier.PRO,
            trial_period_days=TRIAL_DAYS,
        )
        reporter.check(
            "card-less subscription starts trialing",
            subscription.get("status") == "trialing",
            subscription.get("status") or "no status",
        )
        trial_end = int(subscription["trial_end"])
        advance_test_clock(cardless_clock["id"], trial_end + 3600)
        expired = stripe.Subscription.retrieve(subscription["id"]).to_dict()
        reporter.check(
            "trial without payment info cancels at trial end "
            "(missing_payment_method=cancel)",
            expired.get("status") == "canceled",
            expired.get("status") or "no status",
        )
        if api_base_url():
            # The deleted-webhook auto-creates the free-tier subscription;
            # give the forwarder a moment, then look for a fresh subscription.
            time.sleep(5.0)
            subscriptions_after = (
                stripe.Subscription.list(
                    customer=customer["id"], status="active", limit=3
                )
                .to_dict()
                .get("data", [])
            )
            reporter.check(
                "webhook auto-created the free-tier subscription after the "
                "card-less trial lapsed",
                any(
                    (s.get("metadata") or {}).get("neural_nexus_tier") == "free"
                    for s in subscriptions_after
                ),
                f"{len(subscriptions_after)} active subscription(s)",
            )
        else:
            reporter.skip(
                "webhook free-tier auto-enrollment",
                "E2E_API_BASE_URL not set (needs the API + stripe listen); "
                "note the webhook must also resolve an auth0_user_id, so this "
                "assertion needs a customer created through real signup",
            )
    finally:
        delete_test_clock(cardless_clock["id"])

    # ---- Trial with a payment method: converts to paying pro -------------
    carded_clock = create_test_clock("trial-carded")
    try:
        customer = create_customer_on_clock(carded_clock["id"], "trial-carded")
        attach_test_payment_method(customer["id"])
        subscription = create_subscription_for_tier(
            billing_config,
            customer["id"],
            SubscriptionTier.PRO,
            trial_period_days=TRIAL_DAYS,
        )
        reporter.check(
            "carded subscription starts trialing",
            subscription.get("status") == "trialing",
            subscription.get("status") or "no status",
        )
        trial_end = int(subscription["trial_end"])
        # Land shortly after the trial ends; the first paid invoice is charged
        # to the attached test card.
        advance_test_clock(carded_clock["id"], trial_end + 3600)
        converted = stripe.Subscription.retrieve(subscription["id"]).to_dict()
        reporter.check(
            "trial with payment info converts to an active pro subscription",
            converted.get("status") == "active",
            converted.get("status") or "no status",
        )
    finally:
        delete_test_clock(carded_clock["id"])

    return reporter.finish()


if __name__ == "__main__":
    sys.exit(run())
