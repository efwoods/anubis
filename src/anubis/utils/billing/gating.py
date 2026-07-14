# src/anubis/utils/billing/gating.py

"""Resolve a request's subscription tier and Stripe customer, and gate capability.

These are pure helpers (no FastAPI, no Stripe network calls) that read the user
dictionary produced by the auth dependencies and answer three questions the
request path needs before doing billable work:

1. Is this an anonymous user? Anonymous users are ALWAYS the free tier and can
   never subscribe, so tier resolution short-circuits for them.
2. What tier is this user, resolved defensively (any corrupt value ⇒ free)?
3. What is the user's Stripe customer id, if any (anonymous users have none)?

The FastAPI dependency that turns a capability failure into an HTTP 402/403 lives
in ``src/api/webapp.py``; keeping the logic here makes it unit-testable and reused
by both the message path and the webhook/tier-sync path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, UTC
from typing import Any, Mapping

from src.anubis.utils.billing.tiers import (
    SubscriptionTier,
    TierCapability,
    UsageMeter,
    tier_allotment_for_meter,
    tier_from_value,
    tier_has_capability,
)


def is_anonymous_user(user: Mapping[str, Any] | None) -> bool:
    """Return whether ``user`` is an ephemeral anonymous (Supabase) sign-in.

    Anonymous users are created via ``sign_in_anonymously`` and carry
    ``is_anonymous: true``; they have no Auth0 account, no email, and no Stripe
    customer. Absent that flag, a user with neither an email nor a Stripe customer
    id is also treated as anonymous so a paid capability can never leak to one.
    """
    if not user:
        return True
    if user.get("is_anonymous") is True:
        return True
    app_metadata = user.get("app_metadata") or {}
    has_email = bool(user.get("email"))
    has_customer = bool(resolve_stripe_customer_id(user))
    has_subscription = bool(app_metadata.get("subscription_status"))
    return not (has_email or has_customer or has_subscription)


def resolve_stripe_customer_id(
    user: Mapping[str, Any] | None,
) -> str | None:
    """Return the user's Stripe customer id from ``app_metadata``, if present.

    Tolerates the historically inconsistent locations the id has been written to
    (``stripe_customer_id``, ``customer_dict.id``, ``customer.id``) so existing
    records keep working while new signups use the canonical ``stripe_customer_id``.
    Returns ``None`` for anonymous users, which makes Stripe meter reporting a no-op.
    """
    if not user:
        return None
    app_metadata = user.get("app_metadata") or {}
    canonical = app_metadata.get("stripe_customer_id")
    if canonical:
        return str(canonical)
    customer_dict = app_metadata.get("customer_dict") or {}
    if customer_dict.get("id"):
        return str(customer_dict["id"])
    legacy_customer = app_metadata.get("customer") or {}
    if isinstance(legacy_customer, Mapping) and legacy_customer.get("id"):
        return str(legacy_customer["id"])
    return None


def resolve_metering_user_id(user: Mapping[str, Any] | None) -> str | None:
    """Return a stable identifier for attributing usage to this user.

    Auth0 users carry a top-level ``user_id``. Anonymous users are a fresh
    Supabase sign-in per request, but the auth layer stamps a stable hashed-IP
    identifier into ``identities[0].user_id`` — that is the only durable handle
    for tracking an anonymous visitor's month-to-date usage, so free-tier
    allotment gating and ``api_metrics`` rows key on the same value.
    """
    if not user:
        return None
    top_level_user_id = user.get("user_id")
    if top_level_user_id:
        return str(top_level_user_id)
    identities = user.get("identities") or []
    if identities and isinstance(identities[0], Mapping):
        identity_user_id = identities[0].get("user_id")
        if identity_user_id:
            return str(identity_user_id)
    fallback_id = user.get("id")
    return str(fallback_id) if fallback_id else None


def resolve_tier(user: Mapping[str, Any] | None) -> SubscriptionTier:
    """Resolve the user's subscription tier, hard-pinning anonymous users to free.

    For authenticated users the tier is read from
    ``app_metadata.subscription_status.tier`` (kept in sync by the Stripe webhook),
    falling back to a top-level ``app_metadata.tier`` and finally to free. Any
    unknown or malformed value coerces to free via ``tier_from_value``.
    """
    if is_anonymous_user(user):
        return SubscriptionTier.FREE
    app_metadata = (user or {}).get("app_metadata") or {}
    subscription_status = app_metadata.get("subscription_status") or {}
    stored_tier = subscription_status.get("tier") or app_metadata.get("tier")
    return tier_from_value(stored_tier)


def user_has_capability(
    user: Mapping[str, Any] | None, capability: TierCapability
) -> bool:
    """Return whether the user's resolved tier unlocks ``capability``."""
    return tier_has_capability(resolve_tier(user), capability)


def resolve_pay_per_use_enabled(user: Mapping[str, Any] | None) -> bool:
    """Return whether this user may bill overage past a meter's monthly allotment.

    The decision follows the expected-behavior matrix in
    ``_METERING_FEATURE_TESTING.md``: a user *without* a payment method is
    hard-limited at the allotment, while a user *with* a payment method bills
    pay-per-use overage until the period resets — and premium users may
    explicitly disable pay-per-use to cap their own spend.

    Resolution order:

    1. An explicit ``app_metadata.pay_per_use_enabled`` boolean (written by the
       ``/set_pay_per_use`` endpoint, which verifies a payment method exists
       before allowing ``true``) always wins.
    2. Absent an explicit flag, pay-per-use is inferred from the cached
       subscription status: ``"active"`` means Stripe has successfully invoiced
       the customer (or completed a checkout that collected a payment method),
       so overage is billable. ``"trialing"`` deliberately does NOT infer
       pay-per-use — a trial started without a payment method must be limited
       at the allotment, matching "free trial ending without payment ⇒ free".
    3. Anonymous users can never bill overage.
    """
    if is_anonymous_user(user):
        return False
    app_metadata = (user or {}).get("app_metadata") or {}
    explicit_flag = app_metadata.get("pay_per_use_enabled")
    if isinstance(explicit_flag, bool):
        return explicit_flag
    subscription_status = app_metadata.get("subscription_status") or {}
    return subscription_status.get("status") == "active"


def exhausted_allotment_block_reason(
    tier: SubscriptionTier,
    meter: UsageMeter,
    month_to_date_usage: int,
    pay_per_use_enabled: bool,
) -> str | None:
    """Return a human-readable refusal reason when usage must be blocked, else ``None``.

    Pure decision logic shared by every metered endpoint (messages, uploads):

    * A tier without an allotment for ``meter`` is not decided here — the
      capability gate (``enforce_tier_capability``) is the authority for
      dimensions a tier lacks entirely.
    * Usage under the monthly allotment is always allowed.
    * Usage at or past the allotment is allowed only when pay-per-use is
      enabled (which requires a payment method on file), because only then can
      the Stripe graduated metered price actually bill the overage. Otherwise
      the request is refused until the period resets, the user adds a payment
      method and enables pay-per-use, or the user upgrades tiers.
    """
    allotment = tier_allotment_for_meter(tier, meter)
    if allotment is None:
        return None
    if month_to_date_usage < allotment.monthly_allotment:
        return None
    if pay_per_use_enabled:
        return None
    meter_display_name = meter.value.replace("_", " ")
    if tier == SubscriptionTier.FREE:
        return (
            f"Your free-tier monthly allotment of "
            f"{allotment.monthly_allotment:,} {meter_display_name} is exhausted. "
            "Subscribe to a paid tier, or add a payment method and enable "
            "pay-per-use, to continue this month."
        )
    return (
        f"Your {tier.value}-tier monthly allotment of "
        f"{allotment.monthly_allotment:,} {meter_display_name} is exhausted. "
        "Add a payment method and enable pay-per-use (POST /set_pay_per_use) "
        "to bill overage, or wait for the monthly reset."
    )


def resolve_usage_period_anchor(user: Mapping[str, Any] | None) -> datetime | None:
    """Return the user's personal usage-period anchor, if one has been written.

    ``app_metadata.usage_period_anchor`` is an ISO-8601 UTC timestamp written at
    tier upgrade (and at first checkout), marking the instant the local usage
    window restarted so the new tier begins with a fresh allotment. Parsed
    defensively: any missing or malformed value yields ``None`` and the period
    falls back to the environment-configured default.
    """
    if not user:
        return None
    app_metadata = user.get("app_metadata") or {}
    raw_anchor = app_metadata.get("usage_period_anchor")
    if not raw_anchor or not isinstance(raw_anchor, str):
        return None
    try:
        anchor = datetime.fromisoformat(raw_anchor.replace("Z", "+00:00"))
    except ValueError:
        return None
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=UTC)
    return anchor.astimezone(UTC)


_TIER_ORDER: dict[SubscriptionTier, int] = {
    SubscriptionTier.FREE: 0,
    SubscriptionTier.PRO: 1,
    SubscriptionTier.PREMIUM: 2,
}


@dataclass(frozen=True)
class TierChangePlan:
    """How one tier change must be executed, per the retained/cleared usage rules.

    * Upgrades take effect immediately (the user is paying for more right now):
      subscription items are swapped at once and the local usage window restarts
      (``reset_usage_period_anchor``) so the new tier starts with a fresh
      allotment — "usage cleared on upgrade".
    * Downgrades take effect at the period end via a Stripe Subscription
      Schedule: the user already paid for the higher tier through the period, so
      both Stripe billing and local allotment gating keep the higher tier until
      the boundary — "unused allotment continues on downgrade".
    """

    direction: str  # "upgrade" | "downgrade"
    swap_items_immediately: bool
    schedule_change_at_period_end: bool
    reset_usage_period_anchor: bool


def plan_tier_change(
    current_tier: SubscriptionTier, target_tier: SubscriptionTier
) -> TierChangePlan:
    """Return the execution plan for moving from ``current_tier`` to ``target_tier``.

    Same-tier "changes" are the caller's responsibility to reject before
    planning; this function only orders the tiers (free < pro < premium).
    """
    if _TIER_ORDER[target_tier] > _TIER_ORDER[current_tier]:
        return TierChangePlan(
            direction="upgrade",
            swap_items_immediately=True,
            schedule_change_at_period_end=False,
            reset_usage_period_anchor=True,
        )
    return TierChangePlan(
        direction="downgrade",
        swap_items_immediately=False,
        schedule_change_at_period_end=True,
        reset_usage_period_anchor=False,
    )


def customer_has_payment_method(
    customer_document: Mapping[str, Any] | None,
    payment_methods: list[Mapping[str, Any]] | None = None,
) -> bool:
    """Return whether a retrieved Stripe customer has any payment method on file.

    Checks the customer's ``invoice_settings.default_payment_method`` and legacy
    ``default_source`` first, then falls back to a non-empty payment-method list
    (from ``PaymentMethod.list``). Pure logic so the pay-per-use endpoint's
    payment-method requirement is unit-testable without Stripe.
    """
    customer_document = customer_document or {}
    invoice_settings = customer_document.get("invoice_settings") or {}
    if invoice_settings.get("default_payment_method"):
        return True
    if customer_document.get("default_source"):
        return True
    return bool(payment_methods)


def resolve_use_adapter_inference(
    user: Mapping[str, Any] | None, adapter_requested: bool
) -> bool:
    """Return whether this turn should use adapter inference and billing.

    Adapter inference is a Premium-only capability. When the client passes
    ``adapter=True`` but the user is not Premium (including anonymous and free
    tiers), this returns ``False`` so the request falls back to standard
    inference and ``messaging_tokens`` metering without raising an error.
    """
    if not adapter_requested:
        return False
    return resolve_tier(user) == SubscriptionTier.PREMIUM
