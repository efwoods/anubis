# src/anubis/utils/billing/config.py

"""Parse the Stripe object identifiers emitted by the provisioning script.

``scripts/provision_stripe_billing.py`` creates the four Billing Meters, three
tier products, and every flat/metered price, then prints a single JSON document
that the operator pastes into the ``STRIPE_BILLING_CONFIG_JSON`` environment
variable (surfaced as ``GlobalContext.stripe_billing_config_json``). This module
turns that JSON string back into typed, validated lookups that the FastAPI layer
uses to build Checkout line items, switch tiers, and report meter events.

Keeping the identifiers in one JSON blob (rather than fifteen separate env vars)
means switching between Stripe test mode and live mode is a single value swap.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List

from src.anubis.utils.billing.tiers import (
    TIER_DEFINITIONS,
    SubscriptionTier,
    UsageMeter,
)

logger = logging.getLogger(__name__)

# Attribute names used to cache the parsed config and the modification time of the
# file it came from, both stored on the FastAPI ``app.state``.
BILLING_CONFIG_STATE_ATTRIBUTE = "stripe_billing_config"
BILLING_CONFIG_MTIME_STATE_ATTRIBUTE = "stripe_billing_config_file_mtime"


@dataclass(frozen=True)
class TierStripeIdentifiers:
    """The Stripe price identifiers that make up one tier's subscription."""

    base_price_id: str
    metered_price_ids: Dict[UsageMeter, str]

    def all_price_ids(self) -> List[str]:
        """Every price id for this tier: the flat base first, then metered prices.

        This is the ordered set of subscription items used when creating a
        Checkout Session or replacing items during a tier change.
        """
        ordered = [self.base_price_id]
        for meter in UsageMeter:
            price_id = self.metered_price_ids.get(meter)
            if price_id is not None:
                ordered.append(price_id)
        return ordered


@dataclass(frozen=True)
class StripeBillingConfig:
    """Fully-resolved Stripe identifiers for meters and all three tiers."""

    meter_ids: Dict[UsageMeter, str]
    tiers: Dict[SubscriptionTier, TierStripeIdentifiers]
    # Billing-portal configuration created by the provisioning script; optional so
    # config JSON emitted before the portal existed still loads.
    portal_configuration_id: str | None = None

    def identifiers_for_tier(
        self, tier: SubscriptionTier
    ) -> TierStripeIdentifiers:
        """Return the Stripe identifiers for ``tier`` or raise if unconfigured."""
        identifiers = self.tiers.get(tier)
        if identifiers is None:
            raise KeyError(
                f"Stripe billing config has no prices for tier '{tier.value}'. "
                "Re-run scripts/provision_stripe_billing.py and refresh "
                "STRIPE_BILLING_CONFIG_JSON."
            )
        return identifiers


def load_stripe_billing_config(
    stripe_billing_config_json: str | None,
) -> StripeBillingConfig | None:
    """Parse ``stripe_billing_config_json`` into a ``StripeBillingConfig``.

    Returns ``None`` when the value is absent or blank so that callers can degrade
    gracefully (for example, treat every user as free tier when billing has not
    been provisioned yet) rather than crash at import time. Malformed JSON or a
    document missing required keys raises ``ValueError`` so a misconfiguration is
    caught loudly at startup rather than silently mis-billing customers.

    Expected JSON shape::

        {
          "meters": {"messaging_tokens": "mtr_...", ...},
          "tiers": {
            "free":    {"base_price": "price_...",
                        "metered_prices": {"messaging_tokens": "price_..."}},
            "pro":     {"base_price": "price_...", "metered_prices": {...}},
            "premium": {"base_price": "price_...", "metered_prices": {...}}
          },
          "portal_configuration": "bpc_..."   // optional
        }
    """
    if not stripe_billing_config_json or not stripe_billing_config_json.strip():
        return None

    try:
        document = json.loads(stripe_billing_config_json)
    except json.JSONDecodeError as decode_error:
        raise ValueError(
            f"STRIPE_BILLING_CONFIG_JSON is not valid JSON: {decode_error}"
        ) from decode_error

    raw_meters = document.get("meters", {})
    meter_ids: Dict[UsageMeter, str] = {}
    for meter in UsageMeter:
        meter_id = raw_meters.get(meter.value)
        if meter_id:
            meter_ids[meter] = meter_id

    raw_tiers = document.get("tiers", {})
    tiers: Dict[SubscriptionTier, TierStripeIdentifiers] = {}
    for tier in SubscriptionTier:
        raw_tier = raw_tiers.get(tier.value)
        if not raw_tier:
            continue
        base_price_id = raw_tier.get("base_price")
        if not base_price_id:
            raise ValueError(
                f"STRIPE_BILLING_CONFIG_JSON tier '{tier.value}' is missing 'base_price'."
            )
        raw_metered = raw_tier.get("metered_prices", {})
        metered_price_ids: Dict[UsageMeter, str] = {}
        # Only accept metered prices for meters the tier definition actually grants,
        # so a stale config cannot silently bill a meter the tier should not have.
        for meter in TIER_DEFINITIONS[tier].meter_allotments:
            price_id = raw_metered.get(meter.value)
            if price_id:
                metered_price_ids[meter] = price_id
        tiers[tier] = TierStripeIdentifiers(
            base_price_id=base_price_id,
            metered_price_ids=metered_price_ids,
        )

    raw_portal_configuration = document.get("portal_configuration")
    portal_configuration_id = (
        str(raw_portal_configuration) if raw_portal_configuration else None
    )

    return StripeBillingConfig(
        meter_ids=meter_ids,
        tiers=tiers,
        portal_configuration_id=portal_configuration_id,
    )


def resolve_stripe_billing_config_json(context) -> str | None:
    """Return the billing-config JSON from the env value, or the shared file.

    Mirrors ``_resolve_stripe_webhook_secret`` in webapp.py: prod (or any
    operator who prefers it) sets ``STRIPE_BILLING_CONFIG_JSON`` directly; local
    docker-compose leaves that empty and the ``stripe-provision`` service writes
    the JSON emitted by ``scripts/provision_stripe_billing.py`` to
    ``STRIPE_BILLING_CONFIG_FILE`` (``GlobalContext.stripe_billing_config_file``),
    so no JSON is ever pasted into the env or hand-edited on a reprovision.

    Env takes precedence over the file so a deliberate override always wins; the
    file is read only when the env value is absent or blank. Returns ``None`` when
    neither source yields a value so ``load_stripe_billing_config`` degrades to
    the free tier rather than crashing.
    """
    explicit = getattr(context, "stripe_billing_config_json", None)
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    config_file = getattr(context, "stripe_billing_config_file", None)
    if not config_file:
        return None
    try:
        with open(config_file, encoding="utf-8") as handle:
            value = handle.read().strip()
        return value or None
    except OSError:
        return None


def _config_file_modified_time(context: Any) -> float | None:
    """Return the mtime of the provisioned config file, or None when unusable."""
    config_file = getattr(context, "stripe_billing_config_file", None)
    if not config_file:
        return None
    try:
        return os.path.getmtime(config_file)
    except OSError:
        return None


def initialize_stripe_billing_config(app_state: Any) -> StripeBillingConfig | None:
    """Load the billing config into ``app_state`` at startup and return it.

    Records the config file's modification time alongside the parsed config so the
    first request after startup does not re-read a file that has not changed, and
    logs the env-shadows-file conflict that makes a reprovision look like a no-op.
    A malformed config degrades to ``None`` (every user treated as free tier)
    rather than preventing the API from starting.
    """
    context = getattr(app_state, "context", None)
    try:
        loaded_config = load_stripe_billing_config(
            resolve_stripe_billing_config_json(context)
        )
    except ValueError as billing_config_error:
        logger.error("Invalid STRIPE_BILLING_CONFIG_JSON: %s", billing_config_error)
        loaded_config = None

    setattr(app_state, BILLING_CONFIG_STATE_ATTRIBUTE, loaded_config)
    setattr(
        app_state,
        BILLING_CONFIG_MTIME_STATE_ATTRIBUTE,
        _config_file_modified_time(context),
    )

    source_conflict = billing_config_source_conflict(context)
    if source_conflict:
        logger.warning(source_conflict)
    return loaded_config


def current_stripe_billing_config(app_state: Any) -> StripeBillingConfig | None:
    """Return the billing config, re-reading the provisioned file when it changes.

    Price ids change whenever an operator edits ``tiers.py`` and re-runs
    ``scripts/provision_stripe_billing.py``, which rewrites
    ``STRIPE_BILLING_CONFIG_FILE``. Loading that file only once at startup meant
    every price edit needed an API restart to take effect, and until the restart
    the API would build Checkout sessions from price ids that had been archived.
    Watching the file's modification time closes that window: the first request
    after a reprovision picks the new ids up.

    Only the FILE is watched. An explicit ``STRIPE_BILLING_CONFIG_JSON`` in the
    environment cannot change without restarting the process anyway, and it takes
    precedence, so when it is set this function simply returns the cached value.

    A reload that fails to read or parse keeps the last good config rather than
    dropping to ``None``: a half-written or corrupt file must not silently demote
    every paying customer to the free tier.
    """
    cached_config = getattr(app_state, BILLING_CONFIG_STATE_ATTRIBUTE, None)
    context = getattr(app_state, "context", None)
    if context is None:
        return cached_config

    explicit = getattr(context, "stripe_billing_config_json", None)
    if explicit and str(explicit).strip():
        return cached_config

    modified_time = _config_file_modified_time(context)
    if modified_time is None:
        return cached_config
    if modified_time == getattr(app_state, BILLING_CONFIG_MTIME_STATE_ATTRIBUTE, None):
        return cached_config

    try:
        reloaded_config = load_stripe_billing_config(
            resolve_stripe_billing_config_json(context)
        )
    except ValueError as reload_error:
        logger.error(
            "Reprovisioned Stripe billing config is invalid; keeping the previously "
            "loaded price ids: %s",
            reload_error,
        )
        # Remember the mtime anyway so a broken file is not re-parsed per request.
        setattr(app_state, BILLING_CONFIG_MTIME_STATE_ATTRIBUTE, modified_time)
        return cached_config

    setattr(app_state, BILLING_CONFIG_MTIME_STATE_ATTRIBUTE, modified_time)
    if reloaded_config is None:
        return cached_config
    setattr(app_state, BILLING_CONFIG_STATE_ATTRIBUTE, reloaded_config)
    if cached_config is not None and reloaded_config != cached_config:
        logger.info(
            "Stripe billing config reloaded after a reprovision; tier price ids are "
            "now %s",
            {
                tier.value: identifiers.base_price_id
                for tier, identifiers in reloaded_config.tiers.items()
            },
        )
    return reloaded_config


def billing_config_source_conflict(context: Any) -> str | None:
    """Return a warning when the env config shadows a different provisioned file.

    ``STRIPE_BILLING_CONFIG_JSON`` deliberately wins over the file so an operator
    can force specific price ids. The failure that costs hours is the accidental
    version: a stale JSON blob left in ``.env``/``.env.dev`` silently overriding the
    file the provisioning script just wrote, so a price edit appears to do nothing.
    Returns ``None`` when there is no conflict.
    """
    explicit = getattr(context, "stripe_billing_config_json", None)
    if not explicit or not str(explicit).strip():
        return None
    config_file = getattr(context, "stripe_billing_config_file", None)
    if not config_file:
        return None
    try:
        with open(config_file, encoding="utf-8") as handle:
            file_value = handle.read().strip()
    except OSError:
        return None
    if not file_value or file_value == str(explicit).strip():
        return None
    return (
        f"STRIPE_BILLING_CONFIG_JSON is set in the environment and DIFFERS from the "
        f"provisioned file {config_file}. The environment value wins, so anything "
        f"written by scripts/provision_stripe_billing.py — including edited prices — "
        f"is being ignored. Clear STRIPE_BILLING_CONFIG_JSON to follow the file."
    )
