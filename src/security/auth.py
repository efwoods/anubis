from urllib.parse import quote
from langgraph_sdk import Auth
from supabase import create_async_client
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Header,
    HTTPException,
    Request,
    Security,
)
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from pydantic import BaseModel

from typing import Optional

import os

from dotenv import load_dotenv

import httpx
from functools import lru_cache
from jose import jwt, JWTError
from fastapi.security import APIKeyHeader

from cachetools import TTLCache
import asyncio

import logging
import stripe
import time
import json

logger = logging.getLogger(__name__)

load_dotenv()

auth = Auth()

security_route = APIRouter()

# Every authenticated dependency accepts EITHER credential: the long-lived
# ``API-KEY`` header (what integrations and scripts hold) or the ``refresh_token``
# issued by POST /login sent as ``Authorization: Bearer`` (what a browser session
# holds, since /signup shows its API key exactly once and a browser has nowhere
# safe to keep it). Both schemes are declared auto_error=False so that a request
# carrying only one of them is not rejected by the scheme that is missing; the
# dependencies below decide what a total absence of credentials means, which
# differs between them (401 for the authenticated ones, an anonymous identity for
# the public ones).
optional_api_key_scheme = APIKeyHeader(name="API-KEY", auto_error=False)

optional_bearer_scheme = HTTPBearer(auto_error=False)

ALGORITHMS = ["RS256"]

DOMAIN = os.getenv("AUTH0_DOMAIN")
CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
AUDIENCE = os.getenv("AUTH0_AUDIENCE")
CONNECTION = os.getenv("AUTH0_CONNECTION", "Username-Password-Authentication")

BASE_AUTH_URL = f"https://{DOMAIN}"

import hashlib, secrets
from copy import deepcopy
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

_api_key_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)
_anonymous_supabase_user_cache: TTLCache = TTLCache(maxsize=1000, ttl=86400)
# Refresh-token sessions, keyed on the hash of the refresh token. Resolving one
# costs a token exchange against Auth0 plus a Management API read, so the result
# is cached exactly like ``_api_key_cache`` — same size, same TTL — and for the
# same reason: without it every single request would pay two upstream round trips.
_refresh_token_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)
_cache_lock = asyncio.Lock()


def generate_api_key() -> str:
    """Generates a secure, persistent API key."""
    return f"sk-{secrets.token_urlsafe(32)}"


def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


# The development client ip. Pinned so a local API, a local customer portal, and
# the anonymous Stripe customer they share all resolve to one identity.
DEVELOPMENT_MODE_CLIENT_IP = "172.18.0.1"


def resolve_request_hashed_ip(request: Request) -> str:
    """Return the hashed client ip that identifies an anonymous visitor.

    The single place this hash is derived. Anonymous metering and anonymous
    customer lookup must agree on it exactly: a second copy that drifted would
    silently split one visitor into two billing identities, or merge two
    visitors into one.
    """
    if getattr(request.app.state.context, "dev", None) == "TRUE":
        return _hash_key(DEVELOPMENT_MODE_CLIENT_IP)
    return _hash_key(request.headers.get("x-forwarded-for"))


async def update_assistant_config(
    hashed_api_key: str,
    provider_encoded_user_id: str,
    assistant_config: dict,
    request: Request,
):
    try:
        payload = {"app_metadata": {"assistant_config": assistant_config}}

        headers = await _mgmt_headers(request)
        response = await request.app.state.httpx_client.patch(
            f"{BASE_AUTH_URL}/api/v2/users/{provider_encoded_user_id}",
            json=payload,
            headers=headers,
        )

        response.raise_for_status()
        async with _cache_lock:
            del _api_key_cache[hashed_api_key]
        return response
    except Exception as e:
        raise HTTPException(
            detail="Error updating assistant configuration: {e}",
            status_code=response.status_code,
        )


from typing import Dict, Any


async def retry_async_httpx_request(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0,
    max_retries: int = 5,
    base_delay: float = 1.0,
) -> httpx.Response:
    """
    Async retry wrapper for httpx requests.
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(max_retries):
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json,
                    data=data,
                )

                if response.status_code in {429, 500, 502, 503, 504}:
                    raise httpx.HTTPStatusError(
                        f"Retryable HTTP error: {response.status_code}",
                        request=response.request,
                        response=response,
                    )

                response.raise_for_status()
                return response

            except (
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.ReadError,
                httpx.RemoteProtocolError,
                httpx.HTTPStatusError,
            ) as e:
                is_last_attempt = attempt == max_retries - 1

                if isinstance(e, httpx.HTTPStatusError):
                    status_code = e.response.status_code
                    if status_code not in {429, 500, 502, 503, 504}:
                        logger.exception("Non-retryable HTTP error")
                        raise

                if is_last_attempt:
                    logger.exception("Max retries exceeded")
                    raise

                delay = base_delay * (2**attempt)

                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {delay:.2f}s"
                )

                await asyncio.sleep(delay)

    raise RuntimeError("Unexpected retry failure")


# ── Management API token (cached) ──────────────────────────────────────────
_mgmt_token_cache: dict = {"token": None, "expires": 0}
import time


async def _get_mgmt_token(request: Request) -> str:
    """Get a Management API token using client credentials."""
    now = time.monotonic()
    if _mgmt_token_cache["token"] and now < _mgmt_token_cache["expires"]:
        return _mgmt_token_cache["token"]
    json = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "audience": f"{BASE_AUTH_URL}/api/v2/",
    }

    result = await retry_async_httpx_request(
        "POST", url=f"{BASE_AUTH_URL}/oauth/token", json=json
    )
    result.raise_for_status()
    data = result.json()
    _mgmt_token_cache["token"] = data["access_token"]
    _mgmt_token_cache["expires"] = now + data["expires_in"] - 60
    return _mgmt_token_cache["token"]


async def _mgmt_headers(request: Request) -> dict:
    access_token = await _get_mgmt_token(request)
    return {"Authorization": f"Bearer {access_token}"}


async def _provision_stripe_customer_and_default_tier(
    request: Request,
    user_id: Optional[str],
    email: Optional[str],
    name: Optional[str] = None,
) -> Optional[str]:
    """Create or REUSE a Stripe customer for a signup and pin the account to free tier.

    An existing customer with the same email is reused rather than duplicated:
    account deletion keeps the Stripe customer (with the
    ``neural_nexus_trial_used`` metadata flag), so a delete-and-re-signup with
    the same email reattaches to the original customer and can never harvest a
    second free trial. Establishes the single canonical
    ``app_metadata.stripe_customer_id`` (replacing the historically
    inconsistent ``customer_dict``/``customer`` keys) and records a default
    free-tier ``subscription_status`` so gating has a tier to read before the
    user ever subscribes. Best-effort: a Stripe or Auth0 failure logs and
    returns ``None`` rather than blocking signup, because the Stripe webhook
    and ``check_subscription_status`` reconcile the record later.
    """
    stripe_client = request.app.state.stripe
    customer = None
    try:
        if email:
            existing_customers = (
                stripe_client.Customer.list(email=email, limit=1)
                .to_dict()
                .get("data", [])
            )
            if existing_customers:
                customer = existing_customers[0]
                # Reattach: refresh the Auth0 linkage on the reused customer
                # and clear the delete-time markers (an empty string deletes a
                # Stripe metadata key) — the account is live again. The
                # neural_nexus_trial_used flag is deliberately never touched:
                # that flag is the trial-abuse system of record and must
                # survive every delete-and-re-signup cycle.
                stripe_client.Customer.modify(
                    customer["id"],
                    metadata={
                        "auth0_user_id": user_id or "",
                        "deleted_auth0_user_id": "",
                        "account_deleted_at": "",
                    },
                )
        if customer is None:
            # A new account always gets its OWN Stripe customer. An anonymous
            # visitor's customer is never reused here: anonymous and account
            # usage are separate allotments, reported separately, so the two
            # identities must stay separate billing records.
            customer = stripe_client.Customer.create(
                email=email,
                name=name or None,
                metadata={"auth0_user_id": user_id or ""},
            )
    except Exception as customer_error:
        logger.error(
            "Could not create/reuse Stripe customer for %s: %s",
            email,
            customer_error,
        )
        return None

    customer_id = customer["id"]
    app_metadata_update = {
        "stripe_customer_id": customer_id,
        "subscription_status": {
            "status": None,
            "tier": "free",
            "subscription_id": None,
            "customer_id": customer_id,
            "email": email,
        },
    }
    if not user_id:
        return customer_id
    try:
        headers = await _mgmt_headers(request)
        provider_encoded_user_id = quote(user_id, safe="")
        patch_response = await retry_async_httpx_request(
            method="PATCH",
            url=f"{BASE_AUTH_URL}/api/v2/users/{provider_encoded_user_id}",
            headers=headers,
            json={"app_metadata": app_metadata_update},
        )
        patch_response.raise_for_status()
    except Exception as patch_error:
        logger.error(
            "Created Stripe customer %s but could not persist it to Auth0 app_metadata "
            "for %s: %s",
            customer_id,
            user_id,
            patch_error,
        )
    return customer_id


def _subscription_items_for_tier(billing_config, tier) -> list[dict]:
    """Build server-side Subscription.create items: base price + metered prices.

    Mirrors the Checkout line items (``_checkout_line_items_for_tier`` in
    webapp.py): the licensed base price carries ``quantity=1``; metered prices
    are reported via meter events and carry no quantity.
    """
    identifiers = billing_config.identifiers_for_tier(tier)
    items: list[dict] = [{"price": identifiers.base_price_id, "quantity": 1}]
    for metered_price_id in identifiers.metered_price_ids.values():
        items.append({"price": metered_price_id})
    return items


def create_free_tier_subscription(
    stripe_client,
    billing_config,
    customer_id: str,
    auth0_user_id: str = "",
    extra_metadata: Optional[dict] = None,
) -> Optional[dict]:
    """Create the $0 free-tier subscription server-side (no Checkout, no card).

    The free-tier subscription is the billing vehicle for metering free usage
    (every meter event lands on a real subscription for cost analysis) and for
    pay-per-use overage once the user adds a card. The $0 base invoice
    finalizes without a payment method. Used by: the post-verification
    enrollment when a trial cannot be granted (trial already used, or prior
    subscription history) and anonymous per-hashed-IP metering customers.
    After a trial lapses without a card the ``customer.subscription.deleted``
    webhook only pins ``subscription_status`` to free/canceled in Auth0; the
    free-tier subscription is created on the next enrollment path, not by the
    webhook.
    Best-effort: returns ``None`` on failure (the account still gates as free
    from the default ``subscription_status``).
    """
    from src.anubis.utils.billing.tiers import SubscriptionTier

    metadata = {"neural_nexus_tier": SubscriptionTier.FREE.value}
    if auth0_user_id:
        metadata["auth0_user_id"] = auth0_user_id
    metadata.update(extra_metadata or {})
    try:
        return stripe_client.Subscription.create(
            customer=customer_id,
            items=_subscription_items_for_tier(
                billing_config, SubscriptionTier.FREE
            ),
            metadata=metadata,
        ).to_dict()
    except Exception as subscription_error:  # noqa: BLE001 - best-effort
        logger.error(
            "Could not create free-tier subscription for customer %s: %s",
            customer_id,
            subscription_error,
        )
        return None


async def ensure_initial_subscription_after_verification(
    request: Request, user: dict
) -> None:
    """Auto-enroll a newly VERIFIED account: pro free trial, or free tier.

    Called the first time an email-verified user is seen without a
    subscription (the API-key cache keeps this off the hot path; the
    ``initial_subscription_provisioned`` marker keeps it off Stripe on cache
    misses). The enrollment rules:

    * A verified account that has never used a trial receives the PRO tier
      with the 30-day free trial, created server-side without Checkout and
      without a card. ``trial_settings.end_behavior.missing_payment_method``
      is ``cancel`` (Stripe forbids "pause" on subscriptions with metered
      prices); when a trial lapses without a card the
      ``customer.subscription.deleted`` webhook pins the account to
      free/canceled, and the free-tier billing vehicle is created here on the
      next enrollment.
    * Trial abuse is denied by the Stripe customer record, which survives
      account deletion: a customer with ``metadata.neural_nexus_trial_used``
      (or any prior subscription) enrolls straight into the FREE tier. A
      returning user regains a paid tier only by choosing one through
      Checkout (``POST /subscribe``), like any new user — no paid
      subscription is ever auto-created for them.
    * A customer with a LIVE subscription (delete-and-re-signup within the
      same pay period) adopts that subscription: the pending period-end
      cancellation written by ``delete_user`` is cleared so the subscription
      keeps renewing, a free trial is retained until the end of the original
      trial period (never restarted, never extended), and no second charge is
      made for the already-paid period.

    Best-effort and idempotent: any Stripe failure logs and returns without
    writing the marker, so the next cache-miss request retries. Anonymous
    users never reach this function (no email verification for them).
    """
    from src.anubis.utils.billing.config import current_stripe_billing_config
    from src.anubis.utils.billing.gating import resolve_stripe_customer_id
    from src.anubis.utils.billing.subscription_lifecycle import (
        clear_pending_cancellation,
        subscription_period_bounds,
    )
    from src.anubis.utils.billing.tiers import TIER_DEFINITIONS, SubscriptionTier

    app_metadata = user.get("app_metadata") or {}
    if app_metadata.get("initial_subscription_provisioned"):
        return

    billing_config = current_stripe_billing_config(request.app.state)
    if billing_config is None:
        # Billing objects not provisioned (degraded mode) — retry on a later
        # request once configuration exists; do not write the marker.
        return

    email = user.get("email")
    auth0_user_id = user.get("user_id")
    if not email or not auth0_user_id:
        return

    stripe_client = request.app.state.stripe

    cached_subscription_status = app_metadata.get("subscription_status") or {}
    if cached_subscription_status.get("subscription_id"):
        # A subscription already exists (checkout or webhook beat this path);
        # just write the marker so this check never calls Stripe again.
        await update_user_app_metadata_fields(
            request, auth0_user_id, {"initial_subscription_provisioned": True}
        )
        user.setdefault("app_metadata", {})[
            "initial_subscription_provisioned"
        ] = True
        return

    try:
        customer_id = resolve_stripe_customer_id(user)
        if not customer_id:
            customer_id = await _provision_stripe_customer_and_default_tier(
                request=request,
                user_id=auth0_user_id,
                email=email,
                name=user.get("name"),
            )
        if not customer_id:
            return

        customer = stripe_client.Customer.retrieve(customer_id).to_dict()
        trial_already_used = bool(
            (customer.get("metadata") or {}).get("neural_nexus_trial_used")
        )
        # limit=10 (matching delete_user): limit=1 returns only the newest
        # subscription, which can hide a live subscription behind a newer
        # canceled/incomplete record.
        prior_subscriptions = (
            stripe_client.Subscription.list(
                customer=customer_id, status="all", limit=10
            )
            .to_dict()
            .get("data", [])
        )
        # A $0 subscription that exists only to meter an anonymous visitor is
        # not subscription history and must never deny an account its free
        # trial. An account's own customer should never carry one — anonymous
        # visitors keep their own separate customer — so this filter is a guard
        # against the two identities ever being merged onto one customer, which
        # would otherwise silently downgrade a new signup to the free tier. The
        # real trial-abuse guard remains metadata.neural_nexus_trial_used.
        prior_subscriptions = [
            subscription
            for subscription in prior_subscriptions
            if not (subscription.get("metadata") or {}).get("anonymous_hashed_ip")
        ]

        live_prior_subscription = next(
            (
                subscription
                for subscription in prior_subscriptions
                if subscription.get("status") in ("active", "trialing", "past_due")
            ),
            None,
        )

        subscription_status_update: dict
        app_metadata_update: dict = {
            "stripe_customer_id": customer_id,
            "initial_subscription_provisioned": True,
        }

        if live_prior_subscription is not None:
            # Delete-and-re-signup with the same email while the original
            # subscription (typically the trial) is still running: adopt the
            # running subscription — the free trial is retained until the end
            # of the original trial period, and the user is never charged a
            # second time for the already-paid period.
            subscription = live_prior_subscription
            if subscription.get("cancel_at_period_end") or subscription.get(
                "schedule"
            ):
                # delete_user set cancel_at_period_end (or a downgrade schedule
                # is pending); a reinstated subscription must keep renewing.
                # Best-effort: on failure the subscription still ends at the
                # period boundary and the account then gates as free.
                try:
                    clear_pending_cancellation(stripe_client, subscription)
                    subscription["cancel_at_period_end"] = False
                except Exception as reactivation_error:  # noqa: BLE001
                    logger.error(
                        "Could not clear the pending cancellation on adopted "
                        "subscription %s for user %s: %s",
                        subscription.get("id"),
                        auth0_user_id,
                        reactivation_error,
                    )
            subscription_status_update = {
                "status": subscription.get("status"),
                "tier": _tier_from_subscription(stripe_client, subscription),
                "subscription_id": subscription.get("id"),
                "customer_id": customer_id,
                "email": email,
            }
            if subscription.get("status") == "trialing" and subscription.get(
                "trial_end"
            ):
                app_metadata_update["trial_context"] = {
                    "tier": subscription_status_update["tier"],
                    "trial_end": int(subscription["trial_end"]),
                }
            # The previous account's usage_period_anchor died with the deleted
            # Auth0 record; without an anchor the local usage window falls back
            # to the calendar month, misaligned with the Stripe billing period.
            # Rebuild the anchor from the adopted subscription's real period
            # start so the local window stays aligned with the ongoing Stripe
            # period. Usage counted within that window survives the re-signup:
            # fetch_usage_since / fetch_usage_by_meter_since aggregate on the
            # durable stripe_customer_id (kept by delete_user) when present,
            # not on the freshly minted Auth0 user id, so spent tokens and any
            # free-trial usage carry over instead of restarting for the new
            # account.
            current_period_start, _ = subscription_period_bounds(subscription)
            if current_period_start:
                app_metadata_update["usage_period_anchor"] = datetime.fromtimestamp(
                    current_period_start, tz=timezone.utc
                ).isoformat()
        elif trial_already_used or prior_subscriptions:
            # Trial denied (used before, or some prior subscription history):
            # enroll straight into the free tier.
            subscription = create_free_tier_subscription(
                stripe_client, billing_config, customer_id, auth0_user_id
            )
            if subscription is None:
                return
            subscription_status_update = {
                "status": subscription.get("status"),
                "tier": SubscriptionTier.FREE.value,
                "subscription_id": subscription.get("id"),
                "customer_id": customer_id,
                "email": email,
            }
        else:
            # First verified account on this customer: grant the PRO tier
            # with the 30-day free trial, no card required.
            pro_definition = TIER_DEFINITIONS[SubscriptionTier.PRO]
            subscription = stripe_client.Subscription.create(
                customer=customer_id,
                items=_subscription_items_for_tier(
                    billing_config, SubscriptionTier.PRO
                ),
                trial_period_days=pro_definition.trial_period_days,
                # Stripe forbids "pause" with metered prices; "cancel" plus
                # the deleted-webhook's free/canceled status pin achieves
                # the trial-to-free product outcome.
                trial_settings={
                    "end_behavior": {"missing_payment_method": "cancel"}
                },
                metadata={
                    "auth0_user_id": auth0_user_id,
                    "neural_nexus_tier": SubscriptionTier.PRO.value,
                },
            ).to_dict()
            stripe_client.Customer.modify(
                customer_id, metadata={"neural_nexus_trial_used": "true"}
            )
            subscription_status_update = {
                "status": subscription.get("status"),
                "tier": SubscriptionTier.PRO.value,
                "subscription_id": subscription.get("id"),
                "customer_id": customer_id,
                "email": email,
            }
            app_metadata_update["usage_period_anchor"] = datetime.now(
                timezone.utc
            ).isoformat()
            if subscription.get("trial_end"):
                app_metadata_update["trial_context"] = {
                    "tier": SubscriptionTier.PRO.value,
                    "trial_end": int(subscription["trial_end"]),
                }
    except Exception as provisioning_error:  # noqa: BLE001 - best-effort
        logger.error(
            "Could not auto-enroll verified user %s into an initial "
            "subscription: %s",
            auth0_user_id,
            provisioning_error,
        )
        return

    app_metadata_update["subscription_status"] = subscription_status_update
    await update_user_app_metadata_fields(request, auth0_user_id, app_metadata_update)
    user.setdefault("app_metadata", {}).update(app_metadata_update)


# utility functions
async def _run_post_signup_side_effects(
    request: Request,
    created_user_id: Optional[str],
    email: str,
    name: Optional[str],
) -> None:
    """Best-effort work that must NOT block delivery of the one-time API key.

    Both steps are recoverable after the fact, so they run AFTER the signup
    response has been returned (scheduled as a FastAPI background task):

    * Stripe customer + default free-tier provisioning is reconciled anyway by
      the Stripe webhook and by ``ensure_initial_subscription_after_verification``
      on the first authenticated request, so a delay here changes nothing a
      caller can observe.
    * The verification email can always be re-requested via
      ``/resend_verification_email``.

    Keeping them on the request's critical path was the root cause of signup
    (and delete) returning a client-side "network error": the sequential Auth0
    Management API calls, the blocking Stripe SDK calls, and up to ~3.5 s of
    verification-email job polling could push the response past the client/proxy
    timeout, so the one-time API key was generated server-side but never
    delivered. Moving the work here lets ``signup_user`` return the key the
    instant the Auth0 user exists.
    """
    try:
        # Provision a Stripe customer and default free tier for the new account so
        # metering and tier gating have a canonical stripe_customer_id to work with.
        await _provision_stripe_customer_and_default_tier(
            request=request,
            user_id=created_user_id,
            email=email,
            name=name if name else None,
        )
    except Exception as provisioning_error:  # noqa: BLE001 - best-effort
        logger.error(
            "Post-signup Stripe provisioning failed for %s: %s",
            email,
            provisioning_error,
        )

    # Send the verification email explicitly via the jobs endpoint. Users
    # created through the Management API POST /api/v2/users do NOT reliably
    # receive an email from the `verify_email` create flag (verified against
    # the live tenant: the flag left the user Unverified with no email, while
    # POST /api/v2/jobs/verification-email returns 201 and sends).
    try:
        await send_verification_email(created_user_id, request=request)
    except Exception as verification_error:  # noqa: BLE001 - best-effort
        logger.error(
            "Could not send the verification email to %s during signup: %s",
            email,
            verification_error,
        )


async def signup_user(
    email: str,
    password: str,
    request: Request,
    name: Optional[str] = None,
    background_tasks: Optional[BackgroundTasks] = None,
) -> dict:
    api_key = generate_api_key()
    api_key_hash = _hash_key(api_key)

    payload = {
        "email": email,
        "password": password,
        "connection": CONNECTION,
        # The verification email is sent EXPLICITLY (see
        # _run_post_signup_side_effects) via the Management API
        # verification-email job so the send is deterministic and observable;
        # verify_email=False suppresses Auth0's implicit creation-time behavior
        # (which depends on tenant template/provider settings and was observed
        # not firing). The api_key hash is written into app_metadata at creation
        # time, so the account is fully usable the moment this call succeeds —
        # which is why the one-time key can be returned immediately below.
        "verify_email": False,
        "app_metadata": {
            "api_key": api_key_hash,
        },
    }

    # Only send a real name: SignupRequest.name defaults to None, and Auth0
    # rejects an explicit null name with a payload-validation 400.
    if name:
        payload["name"] = name

    # Create the Auth0 user. This is the ONLY step on the critical path: it
    # persists the api_key hash, so once it succeeds the returned key is valid.
    try:
        headers = await _mgmt_headers(request)
        response = await request.app.state.httpx_client.post(
            f"{BASE_AUTH_URL}/api/v2/users",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
    except HTTPException:
        raise
    except httpx.HTTPStatusError as create_error:
        # Auth0 returned a 4xx/5xx. Surface an actionable message instead of the
        # previous handler, which assumed ``e.response`` always existed and
        # crashed with AttributeError on timeouts.
        status_code = create_error.response.status_code
        try:
            error_body = create_error.response.json()
        except Exception:  # noqa: BLE001 - non-JSON error body
            error_body = create_error.response.text
        if status_code == 409:
            raise HTTPException(
                status_code=409,
                detail=(
                    "An account with this email already exists. If this is your "
                    "account, sign in or mint a new key via /rotate_api_key "
                    "(email + password) — the one-time key from any earlier "
                    "signup attempt cannot be shown again."
                ),
            )
        if status_code == 400:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid signup request. The password requires a lower case "
                    "and upper case character, at least 8 characters, and a "
                    f"special character: {error_body}"
                ),
            )
        raise HTTPException(
            status_code=status_code,
            detail=f"Error signing up user: {error_body}",
        )
    except Exception as unexpected_error:  # noqa: BLE001
        # Timeout / connection error reaching Auth0: no response object exists.
        raise HTTPException(
            status_code=502,
            detail=f"Error signing up user: {unexpected_error}",
        )

    created_user = response.json()
    created_user_id = created_user.get("user_id") or created_user.get("_id")

    # Defer Stripe provisioning + verification email so a slow upstream can
    # never cost the caller the one-time API key. When invoked outside a request
    # (background_tasks is None) the work runs inline so behavior is preserved.
    if background_tasks is not None:
        background_tasks.add_task(
            _run_post_signup_side_effects,
            request=request,
            created_user_id=created_user_id,
            email=email,
            name=name if name else None,
        )
        verification_message = (
            "A verification email is being sent. If it does not arrive shortly, "
            "call /resend_verification_email."
        )
    else:
        await _run_post_signup_side_effects(
            request=request,
            created_user_id=created_user_id,
            email=email,
            name=name if name else None,
        )
        verification_message = (
            "A verification email has been sent. If it does not arrive shortly, "
            "call /resend_verification_email."
        )

    return {
        "api_key": api_key,
        "message": "Save this key. This key is shown only once and used for every api request.",
        "verification": verification_message,
    }


async def logout_user(refresh_token: str, request: Request) -> None:
    response = await request.app.state.httpx_client.post(
        f"{BASE_AUTH_URL}/oauth/revoke",
        json={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "token": refresh_token,
        },
    )
    return response


async def login_user(email: str, password: str, request: Request) -> dict:
    """
    Authenticates a user and returns access/id/refresh tokens.
    Requires Resource Owner Password Grant to be enabled.
    """
    try:
        response = await request.app.state.httpx_client.post(
            f"{BASE_AUTH_URL}/oauth/token",
            json={
                "grant_type": "password",
                "username": email,
                "password": password,
                "audience": AUDIENCE,
                "scope": "openid profile email offline_access",
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
            },
        )
        return response  # access_token, id_token, refresh_token, expires_in
    except Exception as e:
        raise HTTPException(
            detail="Error logging in user: {e}", status_code=response.status_code
        )


async def set_login_status(user_id: str, logged_in: bool, request: Request) -> None:
    """
    Set the server-controlled `logged_in` flag in the Auth0 user's app_metadata.

    app_metadata is read-only to the user (user_metadata is user-writable), so
    it is the correct place for a session flag the user must not be able to
    spoof. Auth0 merges app_metadata by key on PATCH, so the sibling `api_key`
    is preserved. Any cached copy of the user is dropped so the next auth lookup
    (and /verify_login_status) reflects the new status instead of a stale one.

    `user_id` is the full Auth0 subject, e.g. "auth0|6a5e59310832afadd626e583".
    """
    encoded_user_id = quote(user_id, safe="")
    headers = await _mgmt_headers(request)
    response = await request.app.state.httpx_client.patch(
        f"{BASE_AUTH_URL}/api/v2/users/{encoded_user_id}",
        json={"app_metadata": {"logged_in": logged_in}},
        headers=headers,
    )
    response.raise_for_status()

    # identities[0]["user_id"] is the bare id (no "auth0|" prefix), which is what
    # the cached user objects are keyed on — mirror rotate_api_key's invalidation.
    bare_user_id = user_id.split("|")[-1]
    async with _cache_lock:
        stale = [
            key
            for key, cached_user in _api_key_cache.items()
            if cached_user["identities"][0]["user_id"] == bare_user_id
        ]
        for key in stale:
            del _api_key_cache[key]

        # Refresh-token sessions must go the same way, and on /logout they MUST:
        # the refresh token is revoked at Auth0 moments later, and a surviving
        # cache entry would keep authenticating that dead token until the entry
        # expired on its own.
        stale_sessions = [
            key
            for key, cached_session in _refresh_token_cache.items()
            if cached_session["user"]["identities"][0]["user_id"] == bare_user_id
        ]
        for key in stale_sessions:
            del _refresh_token_cache[key]


async def get_user(user_id: str, request: Request) -> dict:
    response = await request.app.state.httpx_client.get(
        f"{BASE_AUTH_URL}/api/v2/users/{user_id}",
        headers=await _mgmt_headers(request=request),
    )
    response.raise_for_status()
    return response.json()


async def send_verification_email(user_id: str, request: Request) -> dict:
    """Send the Auth0 verification email and report the DELIVERED/FAILED outcome.

    The Management API verification-email endpoint is asynchronous: the
    creation call only returns a job in ``pending`` status, which says nothing
    about delivery. This helper polls the job briefly so a tenant-side send
    failure (for example: no custom email provider configured — Auth0's
    built-in provider silently fails for any address that is not a tenant
    administrator) surfaces to the caller instead of masquerading as success.
    """
    headers = await _mgmt_headers(request=request)
    response = await request.app.state.httpx_client.post(
        f"{BASE_AUTH_URL}/api/v2/jobs/verification-email",
        json={"user_id": user_id},
        headers=headers,
    )
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.json())
    job = response.json()
    job_id = job.get("id")

    job_status = job.get("status", "pending")
    for poll_delay_seconds in (0.5, 1.0, 2.0):
        if job_status != "pending" or not job_id:
            break
        await asyncio.sleep(poll_delay_seconds)
        poll_response = await request.app.state.httpx_client.get(
            f"{BASE_AUTH_URL}/api/v2/jobs/{job_id}",
            headers=headers,
        )
        if poll_response.status_code >= 400:
            break
        job = poll_response.json()
        job_status = job.get("status", job_status)

    if job_status == "failed":
        raise HTTPException(
            status_code=502,
            detail=(
                "Auth0 accepted the verification email but the send FAILED. "
                "This is a tenant configuration problem — most commonly no "
                "custom email provider is configured (Auth0's built-in "
                "provider only delivers to tenant administrators). Configure "
                "one under Auth0 Dashboard → Branding → Email Provider."
            ),
        )
    if job_status == "completed":
        return {"message": "Verification email sent.", "status": job_status}
    return {
        "message": (
            "Verification email accepted by Auth0 and still sending; "
            "if nothing arrives, call /resend_verification_email."
        ),
        "status": job_status,
    }


# Token Verification


@lru_cache(maxsize=1)
async def _get_jwks(request: Request) -> dict:
    resp = await request.app.state.httpx_client.get(
        f"https://{DOMAIN}/.well-known/jwks.json"
    )
    resp.raise_for_status()
    return resp.json()


async def verify_token(token: str, request: Request) -> dict:
    """Decodes and validates an Auth0 JWT. Returns the payload."""
    jwks = await _get_jwks(request)
    unverified_header = jwt.get_unverified_header(token)

    rsa_key = next(
        (
            {
                "kty": key["kty"],
                "kid": key["kid"],
                "use": key["use"],
                "n": key["n"],
                "e": key["e"],
            }
            for key in jwks["keys"]
            if key["kid"] == unverified_header["kid"]
        ),
        None,
    )
    if not rsa_key:
        raise JWTError("Unable to find matching key")

    return jwt.decode(
        token,
        rsa_key,
        algorithms=ALGORITHMS,
        audience=AUDIENCE,
        issuer=f"https://{DOMAIN}/",
    )


# ── Schemas ────────────────────────────────────────────────────────────────
class SignupRequest(BaseModel):
    email: str
    password: str
    name: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


class LogoutRequest(BaseModel):
    # Optional: a session that authenticates WITH its refresh token already sent
    # it in the Authorization header, and /logout falls back to that. Requiring it
    # here would reject exactly the callers that have nothing to put in the body.
    refresh_token: str | None = None


class UserDataCache(BaseModel):
    pass
    # pass asdf


class UserDataReturn(UserDataCache):
    pass
    # api_key:


# ── Dependency: require valid token ────────────────────────────────────────


async def get_user_with_api_key(
    api_key: str, request: Request, require_verified_email: bool = True
) -> dict | None:
    """Resolve an Auth0 account from an API key (cached for verified users).

    ``require_verified_email=False`` exists for the endpoints an unverified
    account must still reach: ``/resend_verification_email`` (request another
    verification email), ``/logout``, and ``/delete_user`` (abandon the
    account entirely). Every other caller keeps the default and receives 401
    until the email is verified. Only verified users are cached or
    auto-enrolled into an initial subscription.
    """
    cache_key = _hash_key(api_key)

    async with _cache_lock:
        if cache_key in _api_key_cache:
            return _api_key_cache[cache_key]

    headers = await _mgmt_headers(request)
    try:
        result = await request.app.state.httpx_client.get(
            f"{BASE_AUTH_URL}/api/v2/users",
            params={"q": f'app_metadata.api_key:"{cache_key}"', "search_engine": "v3"},
            headers=headers,
        )
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.TimeoutException, httpx.TransportError) as exc:
        # The identity provider is unreachable (DNS/egress/outage, or the event
        # loop was starved). Surface a clean 503 instead of a generic 500 so the
        # client can retry rather than treating it as a request error.
        logger.warning("Auth lookup to %s failed (transport): %s", BASE_AUTH_URL, exc)
        raise HTTPException(
            status_code=503, detail="Authentication service temporarily unreachable."
        ) from exc
    result.raise_for_status()
    users = result.json()
    if not users:
        return None
    user = users[0]

    if user["email_verified"] != True:
        if require_verified_email:
            raise HTTPException(
                detail="Email is not yet verified. Please verify email to continue.",
                status_code=401,
            )
        # Unverified caller path (only /resend_verification_email): return the
        # account without caching (the cache must only ever hold verified
        # users) and without subscription auto-enrollment (enrollment is a
        # post-verification step).
        user.update({"API_KEY": api_key})
        return user

    # First time a VERIFIED account is seen without a subscription: auto-enroll
    # into the pro free trial (or straight into the free tier when a trial was
    # already used on this Stripe customer). Non-fatal and idempotent — a
    # failure here must never block authentication.
    try:
        await ensure_initial_subscription_after_verification(request, user)
    except Exception as enrollment_error:  # noqa: BLE001 - non-fatal
        logger.error(
            "Initial subscription enrollment failed for %s: %s",
            user.get("user_id"),
            enrollment_error,
        )

    # if user['app_metadata']['logged_in'] != True:
    #     raise HTTPException(detail="User is not logged in. Please log in to continue.")

    async with _cache_lock:
        _api_key_cache[cache_key] = user

    user.update({"API_KEY": api_key})

    # Guarantee the verified account owns exactly one personal avatar.
    #
    # This runs AFTER the cache write above, and that ordering is required, not
    # incidental. Provisioning creates a LangGraph assistant through the software
    # development kit, which is an HTTP call back into this very API; the
    # LangGraph server authenticates that call through the ``authenticate``
    # handler below, which calls THIS function again with the same key. Were
    # provisioning invoked before the cache write, the nested call would miss the
    # cache and re-enter provisioning without bound. With the entry already
    # cached, the nested authentication returns from cache and performs no side
    # effects. (``ensure_initial_subscription_after_verification`` above needs no
    # such care: that function only reaches Stripe and Auth0, never back into
    # this API.) ``ensure_personal_avatar_for_user`` additionally holds a
    # re-entrancy guard for the case where the cache entry is evicted mid-flight.
    #
    # Non-fatal and idempotent, like the subscription enrollment: the marker is
    # written only on success, so a failure simply retries on the next cache miss
    # and never costs the caller their authentication.
    try:
        from src.anubis.utils.personal_avatar import ensure_personal_avatar_for_user

        await ensure_personal_avatar_for_user(request, user, api_key)
    except Exception as provisioning_error:  # noqa: BLE001 - non-fatal
        logger.error(
            "Personal avatar provisioning failed for %s: %s",
            user.get("user_id"),
            provisioning_error,
        )

    return user


async def _seed_ephemeral_api_key_for_session(user: dict, ephemeral_api_key: str) -> str:
    """Make ``ephemeral_api_key`` resolve to ``user`` for the rest of this process.

    A refresh-token session has no API key, but fourteen call sites in
    ``src/api/webapp.py`` hand ``current_user["API_KEY"]`` to the LangGraph
    software development kit as an ``API-KEY`` header, and the ``authenticate``
    handler at the bottom of this module resolves that header back to an identity.
    Auth0 stores only the *hash* of the real key, so the real key cannot be
    recovered to satisfy them.

    Seeding ``_api_key_cache`` with the already-resolved user closes that gap
    without touching a single call site: ``get_user_with_api_key`` checks the cache
    before it queries Auth0, so the ephemeral key resolves to this user and is
    never looked up upstream (it does not exist upstream — no account carries its
    hash). The key is a normal ``generate_api_key()`` value, so it is
    indistinguishable in transit and useless once the entry expires.

    This depends on the LangGraph server and this FastAPI application sharing one
    process, which ``langgraph.json`` guarantees by mounting this app as
    ``http.app``. Were the API ever run with multiple uvicorn workers or replicas,
    a key minted in one worker would not resolve in another and graph-backed
    endpoints would 401 for refresh-token sessions; the mapping would then have to
    move to shared storage.

    Only ever call this for a VERIFIED account. The cache is what
    ``get_user_with_api_key`` consults before it applies the verified-email gate,
    so seeding an unverified user here would let that user's key pass a check the
    API-key path fails — see the unverified branch of the caller.
    """
    async with _cache_lock:
        _api_key_cache[_hash_key(ephemeral_api_key)] = user
    user["API_KEY"] = ephemeral_api_key
    return ephemeral_api_key


async def get_user_with_refresh_token(
    refresh_token: str, request: Request, require_verified_email: bool = True
) -> dict | None:
    """Resolve an Auth0 account from the refresh token POST /login returned.

    An Auth0 refresh token is opaque — it carries no claims and cannot be verified
    locally — so the only way to learn who it belongs to is to spend it at the
    token endpoint and read ``sub`` from the resulting ``id_token``. That is two
    upstream round trips (exchange, then Management API read), which is why the
    resolved session is cached under the token's hash for the cache TTL.

    ``require_verified_email`` mirrors ``get_user_with_api_key`` exactly, including
    the message, so a caller sees the same rejection whichever credential it sent.

    Returns None when the token is not a credential this tenant issued (revoked by
    /logout, expired, or simply wrong), which the dependencies turn into a 401.
    """
    cache_key = _hash_key(refresh_token)

    async with _cache_lock:
        cached_session = _refresh_token_cache.get(cache_key)

    if cached_session is not None:
        # The two caches expire independently, so the API-KEY entry may already be
        # gone while this session is still valid. Re-seeding is idempotent and
        # keeps ONE ephemeral key per session rather than minting a fresh key per
        # request, which would churn ``_api_key_cache`` and evict real accounts.
        await _seed_ephemeral_api_key_for_session(
            cached_session["user"], cached_session["ephemeral_api_key"]
        )
        return cached_session["user"]

    try:
        token_response = await request.app.state.httpx_client.post(
            f"{BASE_AUTH_URL}/oauth/token",
            json={
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "refresh_token": refresh_token,
            },
        )
    except (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.TimeoutException,
        httpx.TransportError,
    ) as exc:
        # Same treatment as the API-key path: an unreachable identity provider is
        # a 503 the client can retry, not a request error.
        logger.warning("Token exchange with %s failed (transport): %s", BASE_AUTH_URL, exc)
        raise HTTPException(
            status_code=503, detail="Authentication service temporarily unreachable."
        ) from exc

    if token_response.status_code != 200:
        # Auth0 answers a revoked or unknown refresh token with 401/403
        # invalid_grant. That is an authentication failure, not a server fault.
        return None

    id_token = token_response.json().get("id_token")
    if not id_token:
        # Only possible if the session was issued without the openid scope, which
        # /login always requests. Treat as unauthenticated rather than crashing.
        logger.warning("Refresh token exchange returned no id_token.")
        return None

    # Unverified claims are read here for the same reason /login reads them: the
    # token was just handed to us directly by Auth0 over TLS, so its signature adds
    # nothing that the transport has not already established.
    claims = jwt.get_unverified_claims(id_token)
    user = await get_user(claims["sub"], request=request)
    if not user:
        return None

    if user.get("email_verified") is not True:
        if require_verified_email:
            raise HTTPException(
                detail="Email is not yet verified. Please verify email to continue.",
                status_code=401,
            )
        # Unverified callers are never cached and never auto-enrolled, matching
        # get_user_with_api_key. The session still carries an API_KEY value
        # because /delete_user reads one, but it is deliberately NOT seeded into
        # the cache: it must fail to resolve, exactly as the real key of an
        # unverified account does when the nested LangGraph call re-applies the
        # verified-email gate. Failing closed keeps both credentials equally
        # permissive rather than making this one a way around the gate.
        user["API_KEY"] = generate_api_key()
        return user

    try:
        await ensure_initial_subscription_after_verification(request, user)
    except Exception as enrollment_error:  # noqa: BLE001 - non-fatal
        logger.error(
            "Initial subscription enrollment failed for %s: %s",
            user.get("user_id"),
            enrollment_error,
        )

    ephemeral_api_key = generate_api_key()
    async with _cache_lock:
        _refresh_token_cache[cache_key] = {
            "user": user,
            "ephemeral_api_key": ephemeral_api_key,
        }
    await _seed_ephemeral_api_key_for_session(user, ephemeral_api_key)

    # Provisioning runs AFTER both cache writes for the re-entrancy reason spelled
    # out in get_user_with_api_key: it calls back into this API through the
    # LangGraph software development kit, and that nested call authenticates with
    # the ephemeral key seeded above, which must already be resolvable.
    try:
        from src.anubis.utils.personal_avatar import ensure_personal_avatar_for_user

        await ensure_personal_avatar_for_user(request, user, ephemeral_api_key)
    except Exception as provisioning_error:  # noqa: BLE001 - non-fatal
        logger.error(
            "Personal avatar provisioning failed for %s: %s",
            user.get("user_id"),
            provisioning_error,
        )

    return user


async def _resolve_authenticated_user(
    request: Request,
    api_key: str | None,
    bearer_credentials: HTTPAuthorizationCredentials | None,
    require_verified_email: bool = True,
) -> dict | None:
    """Resolve whichever credential the caller sent, API key taking precedence.

    The precedence matters only when a client sends both, which no client does; the
    API key is tried first because it is the cheaper lookup.
    """
    if api_key:
        return await get_user_with_api_key(
            api_key, request, require_verified_email=require_verified_email
        )
    if bearer_credentials is not None:
        return await get_user_with_refresh_token(
            bearer_credentials.credentials,
            request,
            require_verified_email=require_verified_email,
        )
    return None


def bearer_credentials_from_request(
    request: Request,
) -> HTTPAuthorizationCredentials | None:
    """Parse the bearer credential out of a raw request.

    For the callers that invoke ``get_current_user`` as a plain function rather
    than through dependency injection — the metrics middleware does this to
    pre-authenticate catch-all routes. Such a caller MUST pass this, because an
    omitted argument leaves the parameter holding FastAPI's ``Depends`` sentinel
    rather than None, and the sentinel is not a credential.
    """
    authorization_header = request.headers.get("Authorization") or ""
    scheme, _, credentials = authorization_header.partition(" ")
    if scheme.lower() != "bearer" or not credentials:
        return None
    return HTTPAuthorizationCredentials(scheme=scheme, credentials=credentials)


def _invalid_credential_error(
    bearer_credentials: HTTPAuthorizationCredentials | None,
) -> HTTPException:
    """Describe the rejection in terms of the credential the caller actually sent."""
    if bearer_credentials is not None:
        return HTTPException(
            status_code=401,
            detail="Session is no longer valid. Please log in again.",
        )
    return HTTPException(status_code=401, detail="Invalid API key")


async def get_current_user(
    request: Request,
    api_key: str | None = Depends(optional_api_key_scheme),
    bearer_credentials: HTTPAuthorizationCredentials | None = Depends(
        optional_bearer_scheme
    ),
) -> dict:
    """
    This dependency validates the JWT and returns the payload.
    The 'sub' field in the payload is the Auth0 user_id.
    """
    logger.info("breakpoint")
    if not api_key and bearer_credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Please send API-KEY in request, or Authorization: Bearer <refresh_token>.",
        )

    user = await _resolve_authenticated_user(request, api_key, bearer_credentials)
    if not user:
        raise _invalid_credential_error(bearer_credentials)

    return user


async def get_current_user_allow_unverified(
    request: Request,
    api_key: str | None = Depends(optional_api_key_scheme),
    bearer_credentials: HTTPAuthorizationCredentials | None = Depends(
        optional_bearer_scheme
    ),
) -> dict:
    """
    Like `get_current_user`, but does NOT reject unverified users.

    Authenticate by API key or refresh token WITHOUT requiring a verified email.

    Used by the endpoints that an account which has not verified the signup
    email must still be able to reach: ``/resend_verification_email``
    (request another verification email), ``/verify_login_status`` (how the sign-up
    screen watches for the verification to land), ``/logout``, and ``/delete_user``
    (remove the unverified account and everything the account created).
    Every other endpoint uses ``get_current_user``.
    """
    if not api_key and bearer_credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Please send API-KEY in request, or Authorization: Bearer <refresh_token>.",
        )

    user = await _resolve_authenticated_user(
        request, api_key, bearer_credentials, require_verified_email=False
    )
    if not user:
        raise _invalid_credential_error(bearer_credentials)

    return user


from supabase import create_async_client
from langgraph_sdk import get_client


def _synthetic_supabase_anonymous_user(hashed_ip: str) -> dict[str, Any]:
    """Local fallback matching Supabase ``sign_in_anonymously()`` user shape."""
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    stable_id = str(uuid5(NAMESPACE_URL, f"anonymous:{hashed_ip}"))
    return {
        "id": stable_id,
        "app_metadata": {},
        "user_metadata": {},
        "aud": "authenticated",
        "confirmation_sent_at": None,
        "recovery_sent_at": None,
        "email_change_sent_at": None,
        "new_email": None,
        "new_phone": None,
        "invited_at": None,
        "action_link": None,
        "email": "",
        "phone": "",
        "created_at": now,
        "confirmed_at": None,
        "email_confirmed_at": None,
        "phone_confirmed_at": None,
        "last_sign_in_at": now,
        "role": "authenticated",
        "updated_at": now,
        "identities": [],
        "is_anonymous": True,
        "is_sso_user": False,
        "factors": None,
        "deleted_at": None,
        "banned_until": None,
    }


async def _anonymous_supabase_base_user(
    context: Any, hashed_ip: str
) -> dict[str, Any]:
    """Return cached Supabase anonymous user metadata, signing in at most once per IP/day."""
    async with _cache_lock:
        cached = _anonymous_supabase_user_cache.get(hashed_ip)
        if cached is not None:
            return deepcopy(cached)

    user: dict[str, Any] | None = None
    if context.supabase_url and context.supabase_key:
        try:
            supabase_client = await create_async_client(
                supabase_key=context.supabase_key, supabase_url=context.supabase_url
            )
            auth_response = await supabase_client.auth.sign_in_anonymously()
            user = json.loads(auth_response.user.model_dump_json())
        except Exception as exc:
            logger.warning(
                "Supabase anonymous sign-in failed for %s; using synthetic user: %s",
                hashed_ip[:8],
                exc,
            )

    if user is None:
        user = _synthetic_supabase_anonymous_user(hashed_ip)

    async with _cache_lock:
        _anonymous_supabase_user_cache[hashed_ip] = deepcopy(user)
    return deepcopy(user)


async def get_anonymous_user_with_anonymous_api_key(
    request: Request, assistant_id: str
) -> dict | None:

    # Derived by resolve_request_hashed_ip so every anonymous code path agrees
    # on who this visitor is.
    #   VPN_SIMULATED    2a1201bb6c0061be63fc4ce58a048136fa91d3afea9e21f62ae7988a20cc09f1
    #   NO_VPN_SIMULATED 72aefc13eebd36bf5ec1cbfa1f2e930117a62e07f600dc618c18725f3d52be15
    hashed_ip = resolve_request_hashed_ip(request)

    # async with _cache_lock:
    #     if cache_key in _api_key_cache:
    #         return _api_key_cache[cache_key]

    # is_banned = False
    # pool = request.app.state.pool
    # async with pool.connection() as conn:
    #     async with conn.cursor() as cur:
    #         await cur.execute(
    #             "SELECT 1 FROM user_schema.banned_users WHERE banned_user_id = %s LIMIT 1;",
    #             (hashed_ip,)
    #         )
    #         result = await cur.fetchone()
    #         if result:
    #             is_banned = True

    # if is_banned:
    #     raise HTTPException(status_code=401, detail="You have violated the terms of service. Please contact contact@neuralnexus.site to request appeal.")
    # Handle banned user (e.g., raise HTTPException)

    context = request.app.state.context

    user = await _anonymous_supabase_base_user(context, hashed_ip)
    user["identities"] = [{"user_id": hashed_ip}]

    if assistant_id != "":
        try:
            langgraph_client_headers = {"API-KEY": context.anonymous_api_key}
            langgraph_client = get_client(headers=langgraph_client_headers)
            assistant = await langgraph_client.assistants.get(assistant_id=assistant_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error selecting avatar.")

        public_assistant = assistant.get("metadata", {}).get("is_public", False)
        if not public_assistant:
            raise HTTPException(
                status_code=401,
                detail="Please select a public avatar or use your API key from signup.",
            )

        app_metadata = {
            "api_key": _hash_key(context.anonymous_api_key),
            "assistant_config": {
                "configurable": {
                    "assistant_id": assistant_id,
                    "user_id": hashed_ip,
                    "user_ctx": {"name": "Anonymous", "description": None},
                    "assistant_ctx": {
                        "name": assistant.get("name", None),
                        "description": assistant.get("description", None),
                        "metadata": assistant.get("metadata", {}),
                    },
                }
            },
        }
        user["app_metadata"] = app_metadata
        user["API_KEY"] = context.anonymous_api_key
        user["API-KEY"] = context.anonymous_api_key

    # Anonymous users use free-tier metering only (never a trial): lazily
    # attach the per-hashed-ip Stripe customer (with a $0 free-tier
    # subscription) so meter events for anonymous usage become visible in
    # Stripe cost analysis. Fail-open — a missing customer only means the
    # meter report stays a no-op, exactly the pre-existing behavior. Tier
    # gating is unaffected: is_anonymous_user still pins the tier to free.
    from src.security.anonymous_billing import (
        resolve_or_create_anonymous_billing_record,
    )

    anonymous_billing_record = await resolve_or_create_anonymous_billing_record(
        request, hashed_ip
    )
    if anonymous_billing_record is not None:
        anonymous_app_metadata = user.setdefault("app_metadata", {})
        anonymous_app_metadata[
            "stripe_customer_id"
        ] = anonymous_billing_record.stripe_customer_id
        # The free-tier subscription's billing cycle — NOT the calendar month —
        # is the window Stripe aggregates this visitor's meter events over, and
        # the customer portal reads that same aggregation. Caching the cycle in
        # the same two fields the Stripe webhook writes for authenticated users
        # (``current_period_start`` / ``current_period_end``) makes
        # ``resolve_usage_period_start_for_user`` resolve the identical window
        # for anonymous visitors, so usage-to-date and the 402 boundary agree
        # with what the portal displays. This does not confer any paid
        # capability: ``is_anonymous_user`` short-circuits on the
        # ``is_anonymous`` flag Supabase sets, which pins the tier to free and
        # pay-per-use to false regardless of the cached status.
        anonymous_subscription_status: dict[str, Any] = {
            "status": anonymous_billing_record.subscription_status,
            "subscription_id": anonymous_billing_record.subscription_id,
            "customer_id": anonymous_billing_record.stripe_customer_id,
            "email": None,
            "tier": "free",
        }
        if anonymous_billing_record.current_period_start:
            anonymous_subscription_status[
                "current_period_start"
            ] = anonymous_billing_record.current_period_start
        if anonymous_billing_record.current_period_end:
            anonymous_subscription_status[
                "current_period_end"
            ] = anonymous_billing_record.current_period_end
        anonymous_app_metadata["subscription_status"] = anonymous_subscription_status

    return user


async def get_current_user_or_anonymous_user(
    request: Request,
    assistant_id: str = "",
    api_key: str | None = Depends(optional_api_key_scheme),
    bearer_credentials: HTTPAuthorizationCredentials | None = Depends(
        optional_bearer_scheme
    ),
) -> dict:
    """
    This dependency validates the JWT and returns the payload.
    The 'sub' field in the payload is the Auth0 user_id.
    """
    logger.info("breakpoint")
    if not api_key and bearer_credentials is None:
        # create anonymous user
        user = await get_anonymous_user_with_anonymous_api_key(
            request=request, assistant_id=assistant_id
        )
    else:
        user = await _resolve_authenticated_user(request, api_key, bearer_credentials)

    if not user:
        # create anonymous user
        if not api_key and bearer_credentials is None:
            raise HTTPException(
                status_code=500, detail="Error creating anonymous user."
            )
        else:
            raise _invalid_credential_error(bearer_credentials)

    return user


async def get_current_user_or_anonymous_user_id(
    request: Request,
    api_key: str | None = Depends(optional_api_key_scheme),
    bearer_credentials: HTTPAuthorizationCredentials | None = Depends(
        optional_bearer_scheme
    ),
) -> dict:
    """
    This dependency validates the JWT and returns the payload.
    The 'sub' field in the payload is the Auth0 user_id.
    """
    logger.info("breakpoint")
    if not api_key and bearer_credentials is None:
        # create anonymous user
        user = await get_anonymous_user_with_anonymous_api_key(
            request=request, assistant_id=""
        )
    else:
        user = await _resolve_authenticated_user(request, api_key, bearer_credentials)

    if not user:
        # create anonymous user
        if not api_key and bearer_credentials is None:
            raise HTTPException(
                status_code=500, detail="Error creating anonymous user."
            )
        else:
            raise _invalid_credential_error(bearer_credentials)

    return user


# ── Routes ─────────────────────────────────────────────────────────────────
@security_route.post("/signup")
async def signup(
    body: SignupRequest, request: Request, background_tasks: BackgroundTasks
):
    user = await signup_user(
        body.email,
        body.password,
        name=body.name,
        request=request,
        background_tasks=background_tasks,
    )
    return user


@security_route.get("/get_current_user_id")
async def get_current_user_id(
    current_user: dict = Depends(get_current_user_or_anonymous_user_id),
):
    return current_user["identities"][0]["user_id"]


@security_route.get("/resend_verification_email")
async def resend_verification_email(
    request: Request,
    current_user: dict = Depends(get_current_user_allow_unverified),
):
    # Depends on get_current_user_allow_unverified: the callers of this
    # endpoint are exactly the accounts get_current_user rejects with
    # "Email is not yet verified".
    # Auth0 Management-API user objects carry ``user_id`` (JWT payloads carry
    # ``sub``); the request INSTANCE must be forwarded, not the Request class.
    return await send_verification_email(
        current_user.get("user_id") or current_user.get("sub"), request=request
    )


@security_route.post("/rotate_api_key")
async def rotate_api_key(request: Request, email: str, password: str):

    result = await login_user(email=email, password=password, request=request)
    id_token = result.json().get("id_token", None)
    if id_token:
        current_user = jwt.get_unverified_claims(id_token)
    else:
        raise HTTPException(detail="Invalid Credentials.", status_code=401)

    new_key = generate_api_key()
    new_key_hash = _hash_key(new_key)

    headers = await _mgmt_headers(request)
    user_id = current_user["sub"].split("|")[1]
    encoded_user_id = quote(current_user["sub"], safe="")
    try:
        response = await request.app.state.httpx_client.patch(
            f"{BASE_AUTH_URL}/api/v2/users/{encoded_user_id}",
            json={"app_metadata": {"api_key": new_key_hash}},
            headers=headers,
        )
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(
            detail=f"Error patching the new api_key: {e}",
            status_code=response.status_code,
        )

    async with _cache_lock:
        stale = [
            k
            for k, v in _api_key_cache.items()
            if v["identities"][0]["user_id"] == user_id
        ]
        for k in stale:
            del _api_key_cache[k]

    return {
        "api_key": new_key,
        "message": "Save this key. This key is shown only once and used on every api request.",
    }


@security_route.post("/forgot_password")
async def forgot_password(
    email: str, request: Request, current_user=Depends(get_current_user)
):
    try:
        headers = await _mgmt_headers(request=request)
        result = await request.app.state.httpx_client.post(
            f"{BASE_AUTH_URL}/dbconnections/change_password",
            json={
                "client_id": CLIENT_ID,
                "email": email,
                "connection": CONNECTION,  # e.g. "Username-Password-Authentication"
            },
            headers=headers,
        )

        result.raise_for_status()
        if result.status_code != 200:
            raise HTTPException(status_code=result.status_code, detail="Error: {e}")

        # Always return the same message — don't reveal if email exists
        return {"message": "If that email exists, a password reset link has been sent."}
    except Exception as e:
        raise HTTPException(status_code=result.status_code, detail="Error: {e}")


@security_route.delete("/delete_user")
async def delete_user(
    request: Request,
    current_user: dict = Depends(get_current_user_allow_unverified),
):
    # Deliberately authenticated WITHOUT requiring a verified email: a user who
    # signed up but never verified the signup email still owns an Auth0
    # account, an API key, and any avatar/store rows created before
    # verification, and must be able to remove that account. Requiring
    # verification first would strand exactly the accounts most likely to be
    # abandoned. The API key still proves ownership, so a caller can only ever
    # delete themselves.
    try:
        api_key_hash = current_user["app_metadata"]["api_key"]
        encoded_user_id = quote(current_user["user_id"], safe="")
        headers = await _mgmt_headers(request)

        # The Stripe customer is deliberately KEPT (never deleted): the
        # customer record carries the trial-usage history
        # (metadata.neural_nexus_trial_used) that prevents a delete-and-
        # re-signup from harvesting a fresh free trial. Any live subscription
        # is set to cancel at period end so a departed user is never billed
        # for another period. Signing up again with the same email reattaches
        # this customer: within the same pay period the subscription is
        # adopted and the pending cancellation is cleared
        # (ensure_initial_subscription_after_verification), so the
        # subscription — including a free trial in progress — is reinstated
        # without a second charge; after the period lapses the returning user
        # enrolls free and re-selects a paid tier through Checkout like any
        # new user. Both writes are best-effort — a Stripe failure must not
        # block account deletion.
        from src.anubis.utils.billing.gating import resolve_stripe_customer_id

        customer_id = resolve_stripe_customer_id(current_user) or ""

        if customer_id:
            stripe_client = request.app.state.stripe
            try:
                stripe_client.Customer.modify(
                    customer_id,
                    metadata={
                        "deleted_auth0_user_id": current_user.get("user_id", ""),
                        "account_deleted_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                live_subscriptions = stripe_client.Subscription.list(
                    customer=customer_id, status="all", limit=10
                ).to_dict()
                for subscription in live_subscriptions.get("data", []):
                    if subscription.get("status") in (
                        "active",
                        "trialing",
                        "past_due",
                    ) and not subscription.get("cancel_at_period_end"):
                        stripe_client.Subscription.modify(
                            subscription["id"], cancel_at_period_end=True
                        )
            except Exception as stripe_error:  # noqa: BLE001 - best-effort
                logger.error(
                    "Could not tag/cancel Stripe customer %s during user "
                    "deletion: %s",
                    customer_id,
                    stripe_error,
                )

        # retrieve all avatar ids created by the user:
        from langgraph_sdk import get_client

        from src.anubis.utils.avatar_deletion import (
            delete_api_metrics_for_user,
            delete_store_rows_for_user,
            purge_avatar_data,
            search_all_avatars_for_user,
            select_assistant_ids_for_user,
        )

        token = current_user["API_KEY"]
        headers = {"API-KEY": f"{token}"}
        langgraph_sdk_client = get_client(headers=headers)

        pool = request.app.state.pool
        user_id = current_user["identities"][0].get("user_id")

        # The user's own store rows and usage metrics are removed BEFORE the
        # avatars. Deleting avatars first means a later failure strands rows
        # whose avatar is already gone and therefore no longer discoverable
        # through any listing endpoint — which is how orphaned avatar data
        # accumulated in the first place.
        try:
            await delete_store_rows_for_user(pool, user_id)
            await delete_api_metrics_for_user(pool, user_id)
        except Exception as user_row_error:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Error deleting store rows and usage metrics during "
                    f"delete user: {user_row_error}"
                ),
            ) from user_row_error

        # The avatar sweep runs through the LangGraph SDK, and every SDK call
        # re-enters this application's own ``@auth.authenticate`` handler with
        # the caller's API key. That handler requires a verified email, so an
        # unverified account sweeping avatars raises AuthenticationError
        # ("Email is not yet verified") from inside the endpoint and the whole
        # deletion fails with a 500.
        #
        # An unverified account cannot own an avatar in the first place: every
        # avatar-creating endpoint depends on get_current_user, which rejects an
        # unverified email, and the only endpoints reachable without
        # verification are /resend_verification_email, /logout,
        # /verify_login_status and this one. So the sweep is skipped rather than
        # attempted — but the assumption is checked against Postgres instead of
        # trusted, because silently skipping a sweep that did have work to do is
        # exactly how orphaned avatar data accumulated before.
        if not current_user.get("email_verified"):
            orphan_assistant_ids = await select_assistant_ids_for_user(pool, user_id)
            if orphan_assistant_ids:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "This account has not verified its email yet but owns "
                        "avatars, which should not be possible. The account was "
                        "kept so the avatars stay reachable. Verify the email "
                        "and delete the account again. Avatars: "
                        f"{', '.join(orphan_assistant_ids)}"
                    ),
                )
            avatars: list[dict] = []
        else:
            # Paged, and deliberately not filtered by graph_id: assistants.search
            # defaults to limit=10, so an account with more than ten avatars used to
            # keep the remainder forever once the Auth0 user was deleted and the
            # owning user_id could never be presented again.
            avatars = await search_all_avatars_for_user(
                langgraph_sdk_client, user_id, headers=headers
            )

        # Every avatar is attempted even when one fails, so a single bad avatar
        # cannot strand the rest. The Auth0 user is only deleted when all of the
        # avatars are gone; leaving the account alive keeps the survivors
        # reachable for a retry.
        failed_assistant_ids: list[str] = []
        for avatar in avatars:
            assistant_id = avatar.get("assistant_id", "")
            try:
                await purge_avatar_data(
                    pool=pool,
                    langgraph_sdk_client=langgraph_sdk_client,
                    assistant_id=assistant_id,
                    headers=headers,
                )
            except Exception as avatar_error:  # noqa: BLE001 - continue the sweep
                failed_assistant_ids.append(assistant_id)
                logger.error(
                    "Failed to purge avatar %s during deletion of user %s: %s",
                    assistant_id,
                    user_id,
                    avatar_error,
                )

        if failed_assistant_ids:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Could not delete every avatar for this user; the account "
                    "was kept so the remaining avatars stay reachable. Failed "
                    f"avatars: {', '.join(failed_assistant_ids)}"
                ),
            )

        # # Delete the login information of the user
        headers = await _mgmt_headers(request)
        response = await retry_async_httpx_request(
            method="DELETE",
            url=f"{BASE_AUTH_URL}/api/v2/users/{encoded_user_id}",
            headers=headers,
        )
        # response = await
        # auth0_client = request.app.state.httpx_client
        # response = await auth0_client.delete(url=f"{BASE_AUTH_URL}/api/v2/users/{encoded_user_id}")
        #  headers=headers
        response.raise_for_status()
        if response.status_code == 204:
            # pop, not del: the entry may already have aged out of the TTL cache,
            # and a KeyError here would report a completed deletion as a failure.
            _api_key_cache.pop(api_key_hash, None)
            return {"message": "User deleted"}
        else:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Error deleting user: Auth0 returned status "
                    f"{response.status_code}"
                ),
            )
    except HTTPException:
        # Already carries a specific status code and message — re-raise rather
        # than flattening every failure into an indistinguishable 500.
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting user: {e}")


@security_route.post("/login")
async def login(body: LoginRequest, request: Request):
    """
    Authenticate a user with email + password against Auth0 (Resource Owner
    Password Grant) and return the token set: access_token, id_token,
    refresh_token, expires_in. The custom dashboard stores these client-side;
    the refresh_token is what /logout later revokes.

    Requires the Auth0 tenant/application to have the Resource Owner Password
    Grant enabled and a Default Directory set to the database connection.

    A successful sign-in is also the point at which a verified account is
    auto-enrolled into its initial subscription (see
    ``ensure_initial_subscription_after_verification``). Sign-in is the ONLY
    enrollment trigger the customer portal reaches: the portal authenticates
    with email + password and never holds an API key, so the API-key path in
    ``get_user_with_api_key`` never runs for a portal-only user, and its signup
    endpoint deliberately does not sign the user in — every portal user must log
    in after verifying their email. Without this call such an account would keep
    a free-tier Stripe customer with no subscription at all and would never
    receive the pro free trial.

    Known gap: an account that signs in BEFORE verifying, verifies afterward,
    and never signs in again stays unenrolled until its next sign-in, because
    the portal session outlives verification and the portal has no other way to
    reach this API as that user.

    Personal avatar provisioning is deliberately NOT performed here. Creating an
    avatar requires the LangGraph software development kit authenticated as the
    user, which needs the account's API key; this route only ever holds the email
    and password, and Auth0 stores only the key's hash. Provisioning therefore
    happens on the API-key path in ``get_user_with_api_key``, which every account
    reaches on its first API request after verification.
    """
    response = await login_user(body.email, body.password, request=request)
    # httpx does not raise on 4xx, so an invalid credential comes back as a
    # non-200 response we must surface explicitly (e.g. Auth0's 403
    # "invalid_grant" / "unauthorized").
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json())
    data = response.json()
    # The id_token carries the Auth0 user id in `sub`; mark the user logged in.
    claims = jwt.get_unverified_claims(data["id_token"])
    await set_login_status(claims["sub"], logged_in=True, request=request)

    # Auto-enroll a verified account into its initial subscription. The token
    # set carries neither email_verified nor app_metadata, so the full Auth0
    # record is fetched here. Awaited rather than backgrounded: the enrollment
    # short-circuits on the app_metadata.initial_subscription_provisioned marker
    # before making any Stripe call, so the multi-call cost is paid exactly once
    # per account, and awaiting it means the Stripe customer exists before the
    # customer portal looks it up by email immediately after this response
    # (that lookup returns 403 when no customer exists). Non-fatal: a failure
    # here must never cost the caller their sign-in, and the next sign-in
    # retries because the marker is only written on success.
    try:
        auth0_user = await get_user(claims["sub"], request=request)
        if auth0_user.get("email_verified") is True:
            await ensure_initial_subscription_after_verification(request, auth0_user)
    except Exception as enrollment_error:  # noqa: BLE001 - non-fatal
        logger.error(
            "Initial subscription enrollment failed at login for %s: %s",
            claims.get("sub"),
            enrollment_error,
        )

    return data


@security_route.post("/logout")
async def logout(
    request: Request,
    body: LogoutRequest | None = None,
    current_user: dict = Depends(get_current_user_allow_unverified),
    bearer_credentials: HTTPAuthorizationCredentials | None = Depends(
        optional_bearer_scheme
    ),
):
    """
    Revoke the refresh token at Auth0 so the session can no longer be renewed,
    and clear the user's `logged_in` app_metadata flag.

    The refresh token is read from the request body first and from the bearer
    credential second. A browser session authenticates WITH its refresh token, so
    it carries the token in the Authorization header and has no reason to repeat
    it in the body; an API-KEY client holds the token separately and sends it in
    the body. Both are optional because the caller is already identified by the
    credential that authenticated this request: when neither carries a token the
    `logged_in` flag is still cleared, which ends the session for this API, but
    nothing is revoked at Auth0 and the response says so rather than pretending.
    """
    refresh_token = (body.refresh_token if body else None) or (
        bearer_credentials.credentials if bearer_credentials else None
    )

    if refresh_token:
        response = await logout_user(refresh_token, request=request)
        # Auth0 POST /oauth/revoke returns 200 with an empty body on success.
        if response.status_code >= 400:
            raise HTTPException(
                status_code=response.status_code, detail=response.json()
            )

    # Drops the cached session as well as the flag, so a revoked token stops
    # authenticating immediately rather than at the end of its cache TTL.
    await set_login_status(current_user["user_id"], logged_in=False, request=request)

    if not refresh_token:
        return {
            "message": (
                "Logged out of this API. No refresh token was supplied, so the "
                "token itself was not revoked at the identity provider."
            )
        }
    return {"message": "Logged out successfully"}


@security_route.get("/verify_login_status")
async def verify_login_status(
    request: Request,
    current_user: dict = Depends(get_current_user_allow_unverified),
):
    """
    Report whether the user currently has an active login session, i.e. the
    `logged_in` app_metadata flag set by /login and cleared by /logout. Because
    /login and /logout invalidate the auth cache, this reads a fresh value from
    Auth0 rather than a stale cached copy.

    `email_verified` is reported alongside it because this is the one endpoint an
    account can reach BEFORE verifying its email, which makes it the only way a
    client can watch for the verification to land. It is never stale for an
    unverified caller: neither credential path caches an unverified account, so
    every call re-reads the account from Auth0 and the flag flips the moment the
    user follows the link in the email.
    """
    logged_in = current_user.get("app_metadata", {}).get("logged_in", False)
    return {
        "logged_in": bool(logged_in),
        "email_verified": current_user.get("email_verified") is True,
    }


import stripe
from datetime import datetime, timezone

from dataclasses import dataclass
from typing import Literal


@dataclass
class SubscriptionStatus:
    status: str = None
    subscription_id: str = None
    customer_id: str = None
    email: str = None
    last_updated: str = None
    # The subscription tier (free / pro / premium). Defaults to free so that a
    # missing or unsubscribed record grants only free-tier capabilities.
    tier: str = "free"

    def to_dict(self):
        return {
            "status": self.status,
            "subscription_id": self.subscription_id,
            "customer_id": self.customer_id,
            "email": self.email,
            "tier": self.tier,
        }

    def update(
        self,
        field: Literal[
            "status", "subscription_id", "customer_id", "email", "last_updated", "tier"
        ],
        value,
    ):
        match field:
            case "status":
                self.status = value
            case "subscription_id":
                self.subscription_id = value
            case "customer_id":
                self.customer_id = value
            case "email":
                self.email = value
            case "last_updated":
                self.last_updated = value
            case "tier":
                self.tier = value
        return self.to_dict()


def _tier_from_subscription(stripe_client, subscription: dict) -> str:
    """Resolve the tier of a Stripe subscription from its items' product metadata.

    The provisioning script tags each tier product with a ``neural_nexus_tier``
    metadata key. This reads that key off the first subscription item's product,
    defaulting to ``free`` when the subscription is inactive or unrecognized so a
    lookup can never accidentally grant a paid tier.
    """
    status = subscription.get("status")
    if status not in ("active", "trialing", "past_due"):
        return "free"
    try:
        items = subscription.get("items", {}).get("data", [])
        for item in items:
            product = item.get("price", {}).get("product")
            if not product:
                continue
            if isinstance(product, dict):
                product_metadata = product.get("metadata", {})
            else:
                product_metadata = (
                    stripe_client.Product.retrieve(product).to_dict().get("metadata", {})
                )
            tier = product_metadata.get("neural_nexus_tier")
            if tier:
                return tier
    except Exception as tier_error:
        logger.error("Could not resolve tier from subscription: %s", tier_error)
    return "free"


async def _evict_api_key_cache_for_user(auth0_user_id: str) -> None:
    """Drop every cached API-key entry for one Auth0 user.

    The five-minute ``_api_key_cache`` TTL would otherwise keep serving a stale
    tier/billing snapshot after Auth0 has been updated (for example by a Stripe
    webhook tier change), so any write to a user's app_metadata must evict that
    user's cache entries. The cache is keyed by hashed API key, and each cached
    value indexes by the provider-prefixed ``user_id`` and by the bare
    ``identities[0].user_id`` (the id without the provider prefix), so both
    forms are matched.
    """
    if not auth0_user_id:
        return
    bare_user_id = auth0_user_id.split("|")[-1]
    async with _cache_lock:
        stale_keys = [
            cache_key
            for cache_key, cached_user in _api_key_cache.items()
            if cached_user.get("user_id") == auth0_user_id
            or (
                (cached_user.get("identities") or [{}])[0].get("user_id")
                == bare_user_id
            )
        ]
        for cache_key in stale_keys:
            del _api_key_cache[cache_key]


async def update_user_subscription_status(
    request: Request, auth0_user_id: str, subscription_status: dict
) -> bool:
    """Write a resolved ``subscription_status`` (incl. tier) into Auth0 app_metadata.

    Called by the Stripe webhook to keep the cached tier/status in sync in real time.
    Auth0 merges ``app_metadata`` at the top level, so patching just the
    ``subscription_status`` key replaces that record without disturbing other
    metadata. After a successful patch the user's ``_api_key_cache`` entries are
    evicted so the next request reads the new tier immediately instead of after
    the five-minute TTL — otherwise a webhook-driven tier change (for example
    ``customer.subscription.updated`` or ``invoice.payment_failed``) would be
    served stale until the cache expired or the process restarted. Best-effort:
    logs and returns ``False`` on failure.
    """
    if not auth0_user_id:
        return False
    try:
        headers = await _mgmt_headers(request)
        provider_encoded_user_id = quote(auth0_user_id, safe="")
        response = await retry_async_httpx_request(
            method="PATCH",
            url=f"{BASE_AUTH_URL}/api/v2/users/{provider_encoded_user_id}",
            headers=headers,
            json={"app_metadata": {"subscription_status": subscription_status}},
        )
        response.raise_for_status()
        await _evict_api_key_cache_for_user(auth0_user_id)
        return True
    except Exception as sync_error:
        logger.error(
            "Could not sync subscription status to Auth0 for %s: %s",
            auth0_user_id,
            sync_error,
        )
        return False


async def update_user_app_metadata_fields(
    request: Request, auth0_user_id: str, fields: dict
) -> bool:
    """Patch top-level ``app_metadata`` keys for one Auth0 user and drop stale cache.

    Auth0 merges ``app_metadata`` at the top level, so patching only the supplied
    keys (for example ``pay_per_use_enabled`` or ``usage_period_anchor``) leaves
    every other key untouched. After a successful patch, every cached API-key
    entry for this user is evicted so the five-minute TTL cache cannot serve a
    stale billing flag to the next request. Best-effort: logs and returns
    ``False`` on failure.
    """
    if not auth0_user_id or not fields:
        return False
    try:
        headers = await _mgmt_headers(request)
        provider_encoded_user_id = quote(auth0_user_id, safe="")
        response = await retry_async_httpx_request(
            method="PATCH",
            url=f"{BASE_AUTH_URL}/api/v2/users/{provider_encoded_user_id}",
            headers=headers,
            json={"app_metadata": fields},
        )
        response.raise_for_status()
    except Exception as patch_error:
        logger.error(
            "Could not patch app_metadata fields %s for %s: %s",
            list(fields),
            auth0_user_id,
            patch_error,
        )
        return False

    await _evict_api_key_cache_for_user(auth0_user_id)
    return True


async def check_subscription_status(request: Request, current_user: dict) -> dict:
    stripe_client = request.app.state.stripe
    subscription_status = current_user["app_metadata"].get("subscription_status", None)
    email = current_user.get("email")

    # Anonymous / email-less users are always the free tier and never have a Stripe
    # subscription to look up, so short-circuit before touching Stripe.
    if not email:
        return SubscriptionStatus().to_dict()

    if not subscription_status or not subscription_status.get("subscription_id"):
        # Identify the customer server-side by email rather than scanning every
        # customer/subscription in the account.
        customers = stripe_client.Customer.list(email=email, limit=1).to_dict()["data"]
        if not customers:
            return SubscriptionStatus().to_dict()
        customer_id = customers[0]["id"]

        subscriptions = stripe_client.Subscription.list(
            customer=customer_id, status="all", limit=1
        ).to_dict()["data"]
        customer_subscription_status = SubscriptionStatus(customer_id=customer_id, email=email)
        if subscriptions:
            subscription = subscriptions[0]
            customer_subscription_status.update("subscription_id", subscription["id"])
            customer_subscription_status.update("status", subscription["status"])
            customer_subscription_status.update(
                "tier", _tier_from_subscription(stripe_client, subscription)
            )
            customer_subscription_status.update(
                "last_updated", datetime.now(tz=timezone.utc).isoformat()
            )
            # Cache the resolved status back into app_metadata under the correct key.
            current_user["app_metadata"][
                "subscription_status"
            ] = customer_subscription_status.to_dict()
            headers = await _mgmt_headers(request)
            payload = {"app_metadata": current_user["app_metadata"]}
            provider_encoded_user_id = quote(current_user["user_id"], safe="")
            try:
                response = await retry_async_httpx_request(
                    method="PATCH",
                    url=f"{BASE_AUTH_URL}/api/v2/users/{provider_encoded_user_id}",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
            except Exception as e:
                raise HTTPException(
                    detail="Error checking subscription status.", status_code=500
                )
    else:
        customer_subscription_status = SubscriptionStatus(
            email=subscription_status.get("email"),
            subscription_id=subscription_status.get("subscription_id"),
            customer_id=subscription_status.get("customer_id"),
            tier=subscription_status.get("tier", "free"),
        )

        try:
            subscription = stripe.Subscription.retrieve(
                id=subscription_status["subscription_id"]
            ).to_dict()
            customer_subscription_status.update(
                "status", subscription.get("status", None)
            )
            customer_subscription_status.update(
                "tier", _tier_from_subscription(stripe_client, subscription)
            )
            customer_subscription_status.update(
                "last_updated", datetime.now(tz=timezone.utc).isoformat()
            )
        except Exception as e:
            customer_subscription_status.update("status", subscription_status.get("status"))

    return customer_subscription_status.to_dict()


@auth.authenticate
async def authenticate(request: Request, authorization: str) -> dict:
    """
    This dependency validates the JWT and returns the payload.
    The 'sub' field in the payload is the Auth0 user_id.
    """
    logger.info("breakpoint")
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Please send API-KEY as 'Authorization': 'API-KEY' in request header.",
        )

    user = await get_user_with_api_key(authorization, request)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return {
        "identity": user["identities"][0]["user_id"],
        "metadata": {"user_id": user["identities"][0]["user_id"]},
    }

