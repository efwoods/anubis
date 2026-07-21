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
from enum import StrEnum
from typing import Any, Mapping

from src.anubis.utils.billing.tiers import (
    MeterAllotment,
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
    estimated_request_tokens: int = 0,
    allotment_override: MeterAllotment | None = None,
) -> str | None:
    """Return a human-readable refusal reason when usage must be blocked, else ``None``.

    Pure decision logic shared by every metered endpoint (messages, uploads):

    * A tier without an allotment for ``meter`` is not decided here — the
      capability gate (``enforce_tier_capability``) is the authority for
      dimensions a tier lacks entirely.
    * Usage plus the estimated cost of THIS request under the monthly
      allotment is allowed. ``estimated_request_tokens`` is the pre-call
      estimate of the request's INPUT tokens (the measured system prompt,
      user text, image patches, analysis passes), so a request whose input
      alone would cross the allotment is refused before anything is spent
      (with an estimate of zero this reduces to the plain usage check).
    * Output tokens are deliberately NOT part of the gate: total input may
      not exceed the remaining allotment, but a request whose input fits may
      overshoot the allotment through its OUTPUT exactly once per period —
      ``month_to_date_usage`` records actual TOTAL tokens (input plus
      output), so the request that crossed the allotment is honored and the
      next request is blocked here.
    * Usage at or past the allotment is allowed only when pay-per-use is
      enabled (which requires a payment method on file), because only then can
      the Stripe graduated metered price actually bill the overage. Otherwise
      the request is refused until the period resets, the user adds a payment
      method and enables pay-per-use, or the user upgrades tiers.
    * ``allotment_override`` carries a trial-aware allotment
      (``resolve_effective_monthly_allotment``) when the caller has one, so a
      user inside a free-trial window keeps the trial allotment after
      changing tiers; without an override the tier's plain allotment governs.
    """
    allotment = (
        allotment_override
        if allotment_override is not None
        else tier_allotment_for_meter(tier, meter)
    )
    if allotment is None:
        return None
    estimated_request_tokens = max(0, estimated_request_tokens)
    if month_to_date_usage + estimated_request_tokens < allotment.monthly_allotment:
        return None
    if pay_per_use_enabled:
        return None
    meter_display_name = meter.value.replace("_", " ")
    # When usage alone is still under the allotment, the ESTIMATE is what
    # crosses the line — say so, or a user with visible remaining budget will
    # not understand why a large request was refused.
    estimate_prefix = ""
    if month_to_date_usage < allotment.monthly_allotment and estimated_request_tokens:
        estimate_prefix = (
            f"This request's input is estimated at {estimated_request_tokens:,} "
            f"{meter_display_name}, which would exceed your remaining allotment. "
        )
    if tier == SubscriptionTier.FREE:
        return (
            f"{estimate_prefix}"
            f"Your free-tier monthly allotment of "
            f"{allotment.monthly_allotment:,} {meter_display_name} is exhausted. "
            "Subscribe to a paid tier, or add a payment method and enable "
            "pay-per-use, to continue this month."
        )
    return (
        f"{estimate_prefix}"
        f"Your {tier.value}-tier monthly allotment of "
        f"{allotment.monthly_allotment:,} {meter_display_name} is exhausted. "
        "Add a payment method and enable pay-per-use (POST /set_pay_per_use) "
        "to bill overage, or wait for the monthly reset."
    )


def is_admin_metering_bypass(
    user: Mapping[str, Any] | None, admin_user_id: str | None
) -> bool:
    """Return whether this requester is the admin testing account.

    The admin testing account (``GlobalContext.admin_user_id``, environment
    variable ``ADMIN_USER_ID``) skips BOTH enforcement (402/429) and metering
    writes (Stripe meter events, ``api_metrics`` rows) so testing never
    pollutes real usage. A missing or empty ``admin_user_id`` never bypasses,
    and every other user — anonymous and unsubscribed included — stays
    metered. Keyed on ``resolve_metering_user_id`` because that is the same
    identity that enforcement and usage attribution key on.
    """
    if not admin_user_id:
        return False
    return resolve_metering_user_id(user) == str(admin_user_id)


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
    current_tier: SubscriptionTier,
    target_tier: SubscriptionTier,
    currently_trialing: bool = False,
) -> TierChangePlan:
    """Return the execution plan for moving from ``current_tier`` to ``target_tier``.

    Same-tier "changes" are the caller's responsibility to reject before
    planning; this function only orders the tiers (free < pro < premium).

    ``currently_trialing`` changes the usage-window rule: a tier change during
    an active free trial NEVER resets the usage window — the trial allotment
    and the new tier share one usage counter (usage up to the trial allotment
    stays free within the trial window; usage past the allotment follows the
    new tier's rules). Direction timing is unchanged: upgrades swap items
    immediately, downgrades schedule at the boundary (the trial end, when
    trialing).
    """
    if _TIER_ORDER[target_tier] > _TIER_ORDER[current_tier]:
        return TierChangePlan(
            direction="upgrade",
            swap_items_immediately=True,
            schedule_change_at_period_end=False,
            reset_usage_period_anchor=not currently_trialing,
        )
    return TierChangePlan(
        direction="downgrade",
        swap_items_immediately=False,
        schedule_change_at_period_end=True,
        reset_usage_period_anchor=False,
    )


class SubscribeAction(StrEnum):
    """What POST /subscribe must do for the caller's current subscription state."""

    START_CHECKOUT = "start_checkout"
    CHANGE_TIER = "change_tier"
    REACTIVATE = "reactivate"
    REACTIVATE_AND_CHANGE_TIER = "reactivate_and_change_tier"
    NO_CHANGE_REQUIRED = "no_change_required"


_LIVE_SUBSCRIPTION_STATUSES = ("active", "trialing", "past_due")


def plan_subscribe_action(
    current_status: str | None,
    current_tier: SubscriptionTier,
    requested_tier: SubscriptionTier,
    cancel_at_period_end: bool,
    has_pending_downgrade_schedule: bool,
) -> SubscribeAction:
    """Decide what POST /subscribe does — the single subscription entry point.

    Pure decision logic (unit-testable without Stripe):

    * No live subscription (``current_status`` outside active/trialing/
      past_due, including ``None`` and fully canceled) → start a Checkout
      session for the requested tier.
    * Live subscription with a pending period-end cancellation or a scheduled
      downgrade → reactivate (undo the pending change); when the requested
      tier also differs from the current tier, follow the reactivation with a
      tier change in the same call.
    * Live subscription on a different tier → change tier (the logic formerly
      exposed as POST /change_subscription_tier).
    * Live subscription already on the requested tier with nothing pending →
      nothing to do (the endpoint answers 200 with an explanatory message,
      not an error).
    """
    if current_status not in _LIVE_SUBSCRIPTION_STATUSES:
        return SubscribeAction.START_CHECKOUT
    has_pending_change = cancel_at_period_end or has_pending_downgrade_schedule
    if has_pending_change:
        if requested_tier == current_tier:
            return SubscribeAction.REACTIVATE
        return SubscribeAction.REACTIVATE_AND_CHANGE_TIER
    if requested_tier == current_tier:
        return SubscribeAction.NO_CHANGE_REQUIRED
    return SubscribeAction.CHANGE_TIER


def subscription_has_pending_downgrade_schedule(
    subscription: Mapping[str, Any] | None,
) -> bool:
    """Return whether a Stripe subscription has a pending schedule attached.

    Stripe may return ``schedule`` as a string id, an expanded dict, or omit /
    null it when none is attached. Any truthy value means a pending change
    (typically a scheduled downgrade) that POST /subscribe should reactivate.
    """
    if not subscription:
        return False
    schedule = subscription.get("schedule")
    if not schedule:
        return False
    if isinstance(schedule, dict):
        return bool(schedule.get("id"))
    return True


@dataclass(frozen=True)
class TrialContext:
    """An account's free-trial grant, written when the trial subscription is created.

    ``trial_tier`` is the tier whose allotments the trial granted (pro for the
    signup trial); ``trial_end`` is when the trial window closes. The context
    survives tier changes during the trial — the whole point is that changing
    tiers must not forfeit (or double) the trial allotment.
    """

    trial_tier: SubscriptionTier
    trial_end: datetime


def resolve_trial_context(user: Mapping[str, Any] | None) -> TrialContext | None:
    """Parse ``app_metadata.trial_context`` defensively; ``None`` when absent/corrupt.

    The shape written at trial creation is
    ``{"tier": "pro", "trial_end": <epoch seconds>}``. Anonymous users never
    have a trial context.
    """
    if not user:
        return None
    app_metadata = user.get("app_metadata") or {}
    raw_trial_context = app_metadata.get("trial_context")
    if not isinstance(raw_trial_context, Mapping):
        return None
    raw_trial_end = raw_trial_context.get("trial_end")
    try:
        trial_end = datetime.fromtimestamp(int(raw_trial_end), tz=UTC)
    except (TypeError, ValueError, OSError):
        return None
    return TrialContext(
        trial_tier=tier_from_value(raw_trial_context.get("tier")),
        trial_end=trial_end,
    )


def resolve_effective_monthly_allotment(
    tier: SubscriptionTier,
    meter: UsageMeter,
    trial_context: TrialContext | None,
    now: datetime | None = None,
) -> MeterAllotment | None:
    """Return the allotment governing ``meter`` for a user, trial-aware.

    Outside a trial window this is exactly ``tier_allotment_for_meter``.
    Inside the trial window (``now < trial_end``), the user keeps the trial
    tier's allotment alongside the current tier's — whichever is larger per
    meter — because changing tiers during a trial retains the trial allotment
    on one shared usage counter:

    * pro-trial → premium: premium's larger allotments win where premium
      grants more; the trial allotment still guarantees the floor.
    * pro-trial → free: the pro trial allotments (messaging AND document
      uploads) stay granted until the trial ends, then plain free-tier
      allotments apply.
    """
    current_allotment = tier_allotment_for_meter(tier, meter)
    if trial_context is None:
        return current_allotment
    now = now or datetime.now(UTC)
    if now >= trial_context.trial_end:
        return current_allotment
    trial_allotment = tier_allotment_for_meter(trial_context.trial_tier, meter)
    if trial_allotment is None:
        return current_allotment
    if current_allotment is None:
        return trial_allotment
    if trial_allotment.monthly_allotment > current_allotment.monthly_allotment:
        return trial_allotment
    return current_allotment


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
