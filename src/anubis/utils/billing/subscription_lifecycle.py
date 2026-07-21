# src/anubis/utils/billing/subscription_lifecycle.py

"""Stripe subscription-lifecycle helpers shared by the API and auth layers.

Both ``src/api/webapp.py`` (tier changes, reactivation, webhook sync) and
``src/security/auth.py`` (delete-and-re-signup adoption) need to read a
subscription's billing-period bounds and to undo a pending period-end
cancellation. ``webapp.py`` imports from ``auth.py`` and never the reverse, so
these helpers live in the billing package — which imports neither — to stay
importable from both sides without a cycle. Every helper takes the configured
``stripe_client`` as a parameter; nothing here touches FastAPI app state.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

LIVE_SUBSCRIPTION_STATUSES = ("active", "trialing", "past_due")


def subscription_period_bounds(
    subscription: dict,
) -> tuple[int | None, int | None]:
    """Return ``(current_period_start, current_period_end)`` epoch seconds.

    Newer Stripe API versions (flexible billing mode) place the period bounds on
    each subscription item rather than the subscription top level, so this reads
    items-first with a top-level fallback.
    """
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


def release_pending_subscription_schedule(
    stripe_client, subscription: dict
) -> None:
    """Release the subscription's pending schedule (a scheduled downgrade), if any.

    A subscription attached to a schedule cannot have its items or cancellation
    state modified directly, so every mutation path (tier change, cancel,
    reactivate) releases the schedule first. Releasing keeps the subscription
    running on its current items — the pending change is simply abandoned.
    """
    schedule_id = subscription.get("schedule")
    if not schedule_id:
        return
    if isinstance(schedule_id, dict):
        schedule_id = schedule_id.get("id")
    try:
        stripe_client.SubscriptionSchedule.release(schedule_id)
    except Exception as release_error:  # noqa: BLE001 - surfaced by the follow-up modify
        logger.error(
            "Could not release subscription schedule %s: %s",
            schedule_id,
            release_error,
        )


def resolve_checkout_trial_period_days(
    stripe_client, customer_id: str | None, tier_trial_period_days: int
) -> int:
    """Return the trial days a Checkout session may grant, enforcing one trial ever.

    The Stripe customer record survives account deletion and carries
    ``metadata.neural_nexus_trial_used`` from the moment a trial is granted, so
    a returning user who re-selects a paid tier through Checkout pays from day
    one instead of harvesting a second trial. A caller without a customer id
    has no history to consult and keeps the tier's trial. Fail-safe: an
    unreadable customer record forfeits the trial (returns ``0``) rather than
    risking a duplicate grant.
    """
    if tier_trial_period_days <= 0:
        return 0
    if not customer_id:
        return tier_trial_period_days
    try:
        customer = stripe_client.Customer.retrieve(customer_id).to_dict()
    except Exception as customer_retrieve_error:  # noqa: BLE001 - fail safe
        logger.error(
            "Could not read trial history for customer %s; withholding the "
            "checkout trial: %s",
            customer_id,
            customer_retrieve_error,
        )
        return 0
    if (customer.get("metadata") or {}).get("neural_nexus_trial_used"):
        return 0
    return tier_trial_period_days


def clear_pending_cancellation(stripe_client, subscription: dict) -> None:
    """Undo a pending period-end cancellation or scheduled downgrade.

    Releases any pending subscription schedule, then clears
    ``cancel_at_period_end`` so the subscription keeps renewing. Used by the
    ``POST /subscribe`` reactivation path and by delete-and-re-signup adoption
    (``ensure_initial_subscription_after_verification``), where ``delete_user``
    previously set the cancellation flag. Stripe errors propagate — each caller
    decides between best-effort logging and an HTTP failure.
    """
    release_pending_subscription_schedule(stripe_client, subscription)
    stripe_client.Subscription.modify(
        subscription.get("id"), cancel_at_period_end=False
    )
