"""One Stripe customer per Stripe mode (live / test) for every signed-in account.

Dev (``.env.dev``, a ``sk_test_`` key) and prod (``anubis/.env``, a
``sk_live_`` key) share one Auth0 tenant, and Auth0 ``app_metadata`` held one
``stripe_customer_id`` for both. A customer created in test mode by dev was
then sent to Stripe by prod with the live key, and every meter event and usage
read failed: ``No such customer: 'cus_Ux1G8zxlKL1GFV'; a similar object exists
in test mode, but a live mode key was used to make this request`` (prod log,
2026-09-28, once per avatar reply).

``app_metadata.stripe_customer_ids`` now maps each Stripe mode to that mode's
customer. ``reconcile_stripe_customer_for_current_mode`` runs for every signed-in
request: when the map has no customer for the configured key's mode, the
function checks the legacy ``stripe_customer_id`` against Stripe, and reuses the
legacy customer only when the legacy customer exists in the current mode;
otherwise the function reuses a current-mode customer with the same email or
creates one. The legacy ``stripe_customer_id`` is never overwritten, so the
other mode keeps working. The in-memory ``app_metadata.stripe_customer_id`` of
the request's user is pointed at the current-mode customer, so every existing
reader of the customer id (``resolve_stripe_customer_id`` and 20 call sites)
receives the right customer without change.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)

STRIPE_MODE_LIVE = "live"
STRIPE_MODE_TEST = "test"

STRIPE_CUSTOMER_IDS_METADATA_KEY = "stripe_customer_ids"

_current_stripe_mode_cache: dict[str, str | None] = {}
_reconciled_customer_ids: dict[tuple[str, str], str] = {}
_reconcile_locks: dict[str, asyncio.Lock] = {}


def stripe_mode_of_secret_key(stripe_secret_key: str | None) -> str | None:
    """Return ``live`` / ``test`` from a Stripe secret or restricted key prefix."""
    secret_key_text = (stripe_secret_key or "").strip().strip('"')
    for key_prefix in ("sk_", "rk_"):
        if secret_key_text.startswith(f"{key_prefix}live_"):
            return STRIPE_MODE_LIVE
        if secret_key_text.startswith(f"{key_prefix}test_"):
            return STRIPE_MODE_TEST
    return None


def current_stripe_mode() -> str | None:
    """Stripe mode of the configured ``STRIPE_SECRET_KEY``, read once per process."""
    if "mode" not in _current_stripe_mode_cache:
        from src.anubis.utils.context import GlobalContext

        _current_stripe_mode_cache["mode"] = stripe_mode_of_secret_key(
            GlobalContext().stripe_secret_key
        )
    return _current_stripe_mode_cache["mode"]


def other_stripe_mode(stripe_mode: str) -> str:
    """Return the Stripe mode opposite to ``stripe_mode``."""
    return STRIPE_MODE_TEST if stripe_mode == STRIPE_MODE_LIVE else STRIPE_MODE_LIVE


def customer_id_for_current_mode(app_metadata: Mapping[str, Any] | None) -> str | None:
    """Return the current-mode customer from ``stripe_customer_ids``, if recorded."""
    stripe_mode = current_stripe_mode()
    if not stripe_mode or not app_metadata:
        return None
    customer_ids_by_mode = app_metadata.get(STRIPE_CUSTOMER_IDS_METADATA_KEY) or {}
    if not isinstance(customer_ids_by_mode, Mapping):
        return None
    customer_id = customer_ids_by_mode.get(stripe_mode)
    return str(customer_id) if customer_id else None


def _legacy_customer_id(app_metadata: Mapping[str, Any]) -> str | None:
    canonical_customer_id = app_metadata.get("stripe_customer_id")
    if canonical_customer_id:
        return str(canonical_customer_id)
    customer_dictionary = app_metadata.get("customer_dict") or {}
    if isinstance(customer_dictionary, Mapping) and customer_dictionary.get("id"):
        return str(customer_dictionary["id"])
    legacy_customer = app_metadata.get("customer") or {}
    if isinstance(legacy_customer, Mapping) and legacy_customer.get("id"):
        return str(legacy_customer["id"])
    return None


def _customer_exists_in_current_mode(stripe_client: Any, customer_id: str) -> bool:
    """Report whether the configured key can read the customer and the customer is not deleted."""
    try:
        customer = stripe_client.Customer.retrieve(customer_id)
    except Exception as retrieve_error:  # noqa: BLE001 - any refusal means "not usable here"
        error_code = getattr(retrieve_error, "code", None)
        if error_code != "resource_missing":
            # A network or authentication failure says nothing about the
            # customer; raise so the caller retries on a later request instead
            # of creating a duplicate customer.
            raise
        return False
    customer_dictionary = customer.to_dict() if hasattr(customer, "to_dict") else dict(customer)
    return not customer_dictionary.get("deleted")


def _find_or_create_current_mode_customer(
    stripe_client: Any, user: Mapping[str, Any], stripe_mode: str
) -> str:
    email = user.get("email")
    if email:
        existing_customers = (
            stripe_client.Customer.list(email=email, limit=1).to_dict().get("data", [])
        )
        if existing_customers:
            return str(existing_customers[0]["id"])
    customer = stripe_client.Customer.create(
        email=email or None,
        name=user.get("name") or None,
        metadata={
            "auth0_user_id": str(user.get("user_id") or ""),
            "neural_nexus_stripe_mode": stripe_mode,
        },
    )
    return str(customer["id"])


def _point_request_user_at_customer(
    app_metadata: dict, customer_ids_by_mode: dict, customer_id: str
) -> None:
    app_metadata[STRIPE_CUSTOMER_IDS_METADATA_KEY] = customer_ids_by_mode
    app_metadata["stripe_customer_id"] = customer_id
    subscription_status = app_metadata.get("subscription_status")
    if isinstance(subscription_status, dict) and subscription_status.get("customer_id"):
        subscription_status["customer_id"] = customer_id


async def reconcile_stripe_customer_for_current_mode(request: Any, user: dict) -> str | None:
    """Make ``user`` carry the customer that exists in the configured Stripe mode.

    Returns the current-mode customer id, or ``None`` when no Stripe key is
    configured, the user is anonymous, or Stripe could not be reached (the next
    request retries). Never raises.
    """
    stripe_mode = current_stripe_mode()
    auth0_user_id = user.get("user_id")
    stripe_client = getattr(getattr(request, "app", None), "state", None)
    stripe_client = getattr(stripe_client, "stripe", None)
    if not stripe_mode or not auth0_user_id or stripe_client is None:
        return None
    if user.get("is_anonymous") is True:
        return None
    app_metadata = user.setdefault("app_metadata", {})
    customer_ids_by_mode = dict(app_metadata.get(STRIPE_CUSTOMER_IDS_METADATA_KEY) or {})

    recorded_customer_id = customer_ids_by_mode.get(stripe_mode) or _reconciled_customer_ids.get(
        (str(auth0_user_id), stripe_mode)
    )
    if recorded_customer_id:
        customer_ids_by_mode[stripe_mode] = recorded_customer_id
        _point_request_user_at_customer(app_metadata, customer_ids_by_mode, recorded_customer_id)
        return recorded_customer_id

    lock = _reconcile_locks.setdefault(str(auth0_user_id), asyncio.Lock())
    async with lock:
        cached_customer_id = _reconciled_customer_ids.get((str(auth0_user_id), stripe_mode))
        if cached_customer_id:
            customer_ids_by_mode[stripe_mode] = cached_customer_id
            _point_request_user_at_customer(app_metadata, customer_ids_by_mode, cached_customer_id)
            return cached_customer_id
        legacy_customer_id = _legacy_customer_id(app_metadata)
        try:
            if legacy_customer_id and await asyncio.to_thread(
                _customer_exists_in_current_mode, stripe_client, legacy_customer_id
            ):
                current_mode_customer_id = legacy_customer_id
            else:
                current_mode_customer_id = await asyncio.to_thread(
                    _find_or_create_current_mode_customer, stripe_client, user, stripe_mode
                )
                if legacy_customer_id and not customer_ids_by_mode.get(
                    other_stripe_mode(stripe_mode)
                ):
                    # The legacy customer does not exist in this mode, so the
                    # legacy customer belongs to the other mode.
                    customer_ids_by_mode[other_stripe_mode(stripe_mode)] = legacy_customer_id
        except Exception as stripe_error:  # noqa: BLE001 - retried on the next request
            logger.error(
                "Could not resolve a %s-mode Stripe customer for %s: %s",
                stripe_mode,
                auth0_user_id,
                stripe_error,
            )
            return None

        customer_ids_by_mode[stripe_mode] = current_mode_customer_id
        _reconciled_customer_ids[(str(auth0_user_id), stripe_mode)] = current_mode_customer_id
        try:
            from src.security.auth import update_user_app_metadata_fields

            await update_user_app_metadata_fields(
                request,
                str(auth0_user_id),
                {STRIPE_CUSTOMER_IDS_METADATA_KEY: dict(customer_ids_by_mode)},
            )
        except Exception as auth0_error:  # noqa: BLE001 - the process cache still holds the id
            logger.error(
                "Resolved %s-mode Stripe customer %s for %s but could not store the "
                "customer in Auth0 app_metadata: %s",
                stripe_mode,
                current_mode_customer_id,
                auth0_user_id,
                auth0_error,
            )
        logger.info(
            "Stripe %s-mode customer for %s is %s (legacy customer %s)",
            stripe_mode,
            auth0_user_id,
            current_mode_customer_id,
            legacy_customer_id,
        )
        _point_request_user_at_customer(app_metadata, customer_ids_by_mode, current_mode_customer_id)
        return current_mode_customer_id
