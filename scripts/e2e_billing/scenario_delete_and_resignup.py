#!/usr/bin/env python
# scripts/e2e_billing/scenario_delete_and_resignup.py

"""Delete-and-re-signup subscription lifecycle (Stripe-side, via test clocks).

Drives the exact Stripe primitives ``delete_user`` and
``ensure_initial_subscription_after_verification`` use, without Auth0:

* **Delete**: every live subscription is set to ``cancel_at_period_end`` (a
  departed user is never billed for another period), and the customer keeps
  ``metadata.neural_nexus_trial_used``.
* **Re-signup within the same pay period**: the still-live subscription is
  adopted and ``clear_pending_cancellation`` (the shared billing helper the
  adoption path calls) clears the pending cancellation — same subscription id,
  no new invoice, no second charge.
* **Re-signup after the trial window**: the trialing subscription canceled at
  trial end, and ``resolve_checkout_trial_period_days`` refuses a second trial
  for the customer whose ``neural_nexus_trial_used`` flag is set.
* **Re-signup after the paid period lapsed**: the subscription canceled at the
  period boundary with exactly one paid invoice — the returning user enrolls
  free and re-selects a tier through Checkout like any new user.
"""

from __future__ import annotations

import sys

import stripe
from harness import (
    ScenarioReporter,
    advance_test_clock,
    attach_test_payment_method,
    configure_stripe_test_mode,
    create_customer_on_clock,
    create_subscription_for_tier,
    create_test_clock,
    delete_test_clock,
    load_billing_config,
    print_config_summary,
    subscription_period_bounds,
)

from src.anubis.utils.billing.subscription_lifecycle import (  # noqa: E402
    clear_pending_cancellation,
    resolve_checkout_trial_period_days,
)
from src.anubis.utils.billing.tiers import (  # noqa: E402
    TIER_DEFINITIONS,
    SubscriptionTier,
)

TRIAL_DAYS = TIER_DEFINITIONS[SubscriptionTier.PRO].trial_period_days
SECONDS_PER_DAY = 86_400


def _mark_deleted(customer_id: str, subscription_id: str) -> None:
    """Apply delete_user's Stripe writes: tag the customer, pend the cancel."""
    stripe.Customer.modify(
        customer_id,
        metadata={
            "deleted_auth0_user_id": "auth0|e2e-departed",
            "account_deleted_at": "e2e",
        },
    )
    stripe.Subscription.modify(subscription_id, cancel_at_period_end=True)


def _invoice_count(customer_id: str) -> int:
    """Count the customer's invoices (double-charge detector)."""
    return len(
        stripe.Invoice.list(customer=customer_id, limit=10)
        .to_dict()
        .get("data", [])
    )


def _charged_invoice_count(customer_id: str) -> int:
    """Count invoices that actually collected money.

    A subscription canceling at the period boundary still emits a closing
    ``subscription_cycle`` invoice that settles metered usage; with no usage
    the closing invoice totals zero and collects nothing, so only invoices
    with ``amount_paid > 0`` indicate a charge.
    """
    return sum(
        1
        for invoice in stripe.Invoice.list(customer=customer_id, limit=10)
        .to_dict()
        .get("data", [])
        if int(invoice.get("amount_paid") or 0) > 0
    )


def run() -> int:
    """Exercise the three delete-and-re-signup timing cases."""
    configure_stripe_test_mode()
    billing_config = load_billing_config()
    print_config_summary()
    reporter = ScenarioReporter("delete-and-re-signup lifecycle")

    # ---- Re-signup within the same paid period: adopt + reinstate --------
    same_period_clock = create_test_clock("resignup-same-period")
    try:
        customer = create_customer_on_clock(
            same_period_clock["id"], "resignup-same-period"
        )
        attach_test_payment_method(customer["id"])
        subscription = create_subscription_for_tier(
            billing_config, customer["id"], SubscriptionTier.PRO
        )
        invoices_before_deletion = _invoice_count(customer["id"])
        _mark_deleted(customer["id"], subscription["id"])
        pending = stripe.Subscription.retrieve(subscription["id"]).to_dict()
        reporter.check(
            "deletion leaves the paid subscription live but canceling at "
            "period end",
            pending.get("status") == "active"
            and pending.get("cancel_at_period_end") is True,
            f"status={pending.get('status')} "
            f"cancel_at_period_end={pending.get('cancel_at_period_end')}",
        )

        # Re-signup within the period: the adoption path clears the pending
        # cancellation on the SAME subscription.
        clear_pending_cancellation(stripe, pending)
        adopted = stripe.Subscription.retrieve(subscription["id"]).to_dict()
        reporter.check(
            "re-signup adoption reinstates the same subscription "
            "(cancel_at_period_end cleared)",
            adopted.get("id") == subscription["id"]
            and adopted.get("cancel_at_period_end") is False,
            f"cancel_at_period_end={adopted.get('cancel_at_period_end')}",
        )
        reporter.check(
            "adoption produced no new invoice (no double charge for the "
            "already-paid period)",
            _invoice_count(customer["id"]) == invoices_before_deletion,
            f"{_invoice_count(customer['id'])} invoice(s) vs "
            f"{invoices_before_deletion} before deletion",
        )
        period_start, _ = subscription_period_bounds(adopted)
        reporter.check(
            "adopted subscription exposes the original period start for the "
            "usage-period anchor rebuild",
            bool(period_start),
            "no current_period_start on the subscription",
        )
    finally:
        delete_test_clock(same_period_clock["id"])

    # ---- Delete during trial, return after trial_end: no second trial ----
    lapsed_trial_clock = create_test_clock("resignup-after-trial")
    try:
        customer = create_customer_on_clock(
            lapsed_trial_clock["id"], "resignup-after-trial"
        )
        subscription = create_subscription_for_tier(
            billing_config,
            customer["id"],
            SubscriptionTier.PRO,
            trial_period_days=TRIAL_DAYS,
        )
        # The trial grant stamps the one-trial-ever flag on the customer.
        stripe.Customer.modify(
            customer["id"], metadata={"neural_nexus_trial_used": "true"}
        )
        _mark_deleted(customer["id"], subscription["id"])
        trial_end = int(subscription["trial_end"])
        advance_test_clock(lapsed_trial_clock["id"], trial_end + 3600)
        lapsed = stripe.Subscription.retrieve(subscription["id"]).to_dict()
        reporter.check(
            "the deleted user's card-less trial cancels at trial end "
            "(trial window surpassed)",
            lapsed.get("status") == "canceled",
            lapsed.get("status") or "no status",
        )
        reporter.check(
            "a re-signup Checkout after the trial window grants NO second "
            "trial (neural_nexus_trial_used)",
            resolve_checkout_trial_period_days(
                stripe, customer["id"], TRIAL_DAYS
            )
            == 0,
            "resolve_checkout_trial_period_days returned a trial",
        )
    finally:
        delete_test_clock(lapsed_trial_clock["id"])

    # ---- Delete on a paid plan, return after the period lapsed -----------
    lapsed_period_clock = create_test_clock("resignup-after-period")
    try:
        customer = create_customer_on_clock(
            lapsed_period_clock["id"], "resignup-after-period"
        )
        attach_test_payment_method(customer["id"])
        subscription = create_subscription_for_tier(
            billing_config, customer["id"], SubscriptionTier.PRO
        )
        _mark_deleted(customer["id"], subscription["id"])
        _, period_end = subscription_period_bounds(
            stripe.Subscription.retrieve(subscription["id"]).to_dict()
        )
        advance_test_clock(lapsed_period_clock["id"], int(period_end) + 3600)
        ended = stripe.Subscription.retrieve(subscription["id"]).to_dict()
        reporter.check(
            "the deleted user's paid subscription cancels at the period "
            "boundary (never billed for another period)",
            ended.get("status") == "canceled",
            ended.get("status") or "no status",
        )
        reporter.check(
            "exactly one invoice collected money after the period lapsed "
            "(no charge for a second period; the closing cycle invoice "
            "totals zero)",
            _charged_invoice_count(customer["id"]) == 1,
            f"{_charged_invoice_count(customer['id'])} charged invoice(s)",
        )
    finally:
        delete_test_clock(lapsed_period_clock["id"])

    return reporter.finish()


if __name__ == "__main__":
    sys.exit(run())
