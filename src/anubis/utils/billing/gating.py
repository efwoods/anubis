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

from typing import Any, Mapping

from src.anubis.utils.billing.tiers import (
    SubscriptionTier,
    TierCapability,
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
