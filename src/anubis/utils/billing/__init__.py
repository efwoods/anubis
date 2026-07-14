# src/anubis/utils/billing/__init__.py

"""Metering and subscription-tier billing for Neural Nexus.

This package is the single source of truth for the three subscription tiers
(free, pro, premium), the four separately-budgeted, per-user, monthly-resetting
usage meters, and the helpers that report usage to Stripe Billing Meters and
enforce per-tier capability and allotment gating.
"""

from src.anubis.utils.billing.config import (
    StripeBillingConfig,
    TierStripeIdentifiers,
    load_stripe_billing_config,
)
from src.anubis.utils.billing.gating import (
    TierChangePlan,
    customer_has_payment_method,
    exhausted_allotment_block_reason,
    is_anonymous_user,
    plan_tier_change,
    resolve_metering_user_id,
    resolve_pay_per_use_enabled,
    resolve_stripe_customer_id,
    resolve_tier,
    resolve_usage_period_anchor,
    resolve_use_adapter_inference,
    user_has_capability,
)
from src.anubis.utils.billing.metering import (
    billable_tokens_from_metadata,
    ensure_api_metrics_table,
    estimate_upload_token_units,
    fetch_rolling_window_usage,
    fetch_usage_by_meter_since,
    fetch_usage_since,
    persist_api_metrics_row,
    report_adapter_training_usage,
    report_meter_event,
    resolve_usage_period_end,
    resolve_usage_period_start,
    token_rate_limit_retry_after_seconds,
)
from src.anubis.utils.billing.tiers import (
    METER_EVENT_NAMES,
    TIER_DEFINITIONS,
    MeterAllotment,
    SubscriptionTier,
    TierCapability,
    TierDefinition,
    UsageMeter,
    tier_allotment_for_meter,
    tier_from_value,
    tier_has_capability,
)

__all__ = [
    "SubscriptionTier",
    "UsageMeter",
    "TierCapability",
    "TierDefinition",
    "MeterAllotment",
    "TIER_DEFINITIONS",
    "METER_EVENT_NAMES",
    "tier_from_value",
    "tier_has_capability",
    "tier_allotment_for_meter",
    "StripeBillingConfig",
    "TierStripeIdentifiers",
    "load_stripe_billing_config",
    "is_anonymous_user",
    "customer_has_payment_method",
    "exhausted_allotment_block_reason",
    "plan_tier_change",
    "TierChangePlan",
    "resolve_metering_user_id",
    "resolve_pay_per_use_enabled",
    "resolve_stripe_customer_id",
    "resolve_tier",
    "resolve_usage_period_anchor",
    "resolve_use_adapter_inference",
    "user_has_capability",
    "report_meter_event",
    "report_adapter_training_usage",
    "billable_tokens_from_metadata",
    "estimate_upload_token_units",
    "fetch_usage_since",
    "fetch_usage_by_meter_since",
    "fetch_rolling_window_usage",
    "resolve_usage_period_start",
    "resolve_usage_period_end",
    "token_rate_limit_retry_after_seconds",
    "ensure_api_metrics_table",
    "persist_api_metrics_row",
]
