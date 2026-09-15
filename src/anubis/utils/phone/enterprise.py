"""Per-user DID numbers are enterprise-only.

Standard and Pro verify the mobile they already have. A request for a
private Neural Nexus number is refused unless the subscription tier is
premium (the enterprise-grade tier today). This slice never buys a number.
"""

from __future__ import annotations

from typing import Any

from src.anubis.utils.billing.tiers import SubscriptionTier, tier_from_value


class PrivateNumberRefused(Exception):
    """The account is not allowed a private inbound number."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = 403


def wants_private_number(fields: dict[str, Any] | None) -> bool:
    """Whether the connect body asked for a dedicated inbound number."""
    if not fields:
        return False
    raw = fields.get("want_private_number")
    if raw is True or str(raw).strip().lower() in {"1", "true", "yes", "enterprise"}:
        return True
    return bool(str(fields.get("platform_number_e164") or "").strip())


def refuse_private_number_unless_enterprise(subscription_tier: str | None) -> None:
    """Refuse a private DID unless the tier is premium.

    Does not provision anything. The later enterprise path may store a number;
    this function only keeps every other tier off that path.
    """
    tier = tier_from_value(subscription_tier)
    if tier != SubscriptionTier.PREMIUM:
        raise PrivateNumberRefused(
            "A private Neural Nexus phone number is an enterprise feature. "
            "Use the mobile you already have: verify that number and call the "
            "shared platform number from it."
        )
