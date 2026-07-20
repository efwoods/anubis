from urllib.parse import quote
from langgraph_sdk import Auth
from supabase import create_async_client
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Security
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

security = HTTPBearer()

api_key_scheme = APIKeyHeader(name="API-KEY")

anonymous_api_key_scheme = APIKeyHeader(name="API-KEY", auto_error=False)

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
_cache_lock = asyncio.Lock()


def generate_api_key() -> str:
    """Generates a secure, persistent API key."""
    return f"sk-{secrets.token_urlsafe(32)}"


def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


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
    from src.anubis.utils.billing.gating import resolve_stripe_customer_id
    from src.anubis.utils.billing.subscription_lifecycle import (
        clear_pending_cancellation,
        subscription_period_bounds,
    )
    from src.anubis.utils.billing.tiers import TIER_DEFINITIONS, SubscriptionTier

    app_metadata = user.get("app_metadata") or {}
    if app_metadata.get("initial_subscription_provisioned"):
        return

    billing_config = getattr(request.app.state, "stripe_billing_config", None)
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
            # start. Known limitation: api_metrics usage rows key on the Auth0
            # user id, so locally counted usage restarts for the new account
            # even though Stripe billing continues the same period.
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
async def signup_user(
    email: str, password: str, request: Request, name: Optional[str] = None
) -> dict:
    try:
        api_key = generate_api_key()
        api_key_hash = _hash_key(api_key)

        payload = {
            "email": email,
            "password": password,
            "connection": CONNECTION,
            # The verification email is sent EXPLICITLY below via the
            # Management API verification-email job so the send is
            # deterministic and observable; verify_email=False suppresses
            # Auth0's implicit creation-time behavior (which depends on
            # tenant template/provider settings and was observed not firing).
            "verify_email": False,
            "app_metadata": {
                "api_key": api_key_hash,
            },
        }

        # Only send a real name: SignupRequest.name defaults to None, and
        # Auth0 rejects an explicit null name with a payload-validation 400.
        if name:
            payload["name"] = name

        headers = await _mgmt_headers(request)
        response = await request.app.state.httpx_client.post(
            f"{BASE_AUTH_URL}/api/v2/users",
            json=payload,
            headers=headers,
        )

        response.raise_for_status()

        # Provision a Stripe customer and default free tier for the new account so
        # metering and tier gating have a canonical stripe_customer_id to work with.
        created_user = response.json()
        created_user_id = created_user.get("user_id") or created_user.get("_id")
        await _provision_stripe_customer_and_default_tier(
            request=request,
            user_id=created_user_id,
            email=email,
            name=name if name else None,
        )

        # Best-effort: a failed send must not lose the one-time API key —
        # the user can request another email via /resend_verification_email
        # (which accepts an unverified account).
        try:
            verification_result = await send_verification_email(
                created_user_id, request=request
            )
            verification_message = verification_result.get(
                "message", "A verification email has been sent."
            )
        except HTTPException as verification_error:
            logger.error(
                "Could not send the verification email to %s during signup: %s",
                email,
                verification_error.detail,
            )
            verification_message = (
                f"The verification email could not be sent: "
                f"{verification_error.detail}"
            )
        except Exception as verification_error:  # noqa: BLE001 - best-effort
            logger.error(
                "Could not send the verification email to %s during signup: %s",
                email,
                verification_error,
            )
            verification_message = (
                "The verification email could not be sent; request another "
                "one via /resend_verification_email."
            )

        result = {
            "api_key": api_key,
            "message": "Save this key. This key is shown only once and used for every api request.",
            "verification": verification_message,
        }

        return result
    except Exception as e:
        if e.response.status_code == 400:
            raise HTTPException(
                detail=f"Invalid Password. Password Requires a lower case and upper case character as well as at least 8 characters and a special character: {response.json().get('mesage', response.json())}",
                status_code=response.status_code,
            )
        else:
            raise HTTPException(
                detail=f"Error signing up user: {response.json()}",
                status_code=response.status_code,
            )


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
    refresh_token: str


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

    ``require_verified_email=False`` exists solely for
    ``/resend_verification_email``: an unverified account must be able to
    authenticate far enough to request another verification email. Every
    other caller keeps the default and receives 401 until the email is
    verified. Only verified users are cached or auto-enrolled into an
    initial subscription.
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
    return user


async def get_current_user(
    request: Request, api_key: str | None = Depends(api_key_scheme)
) -> dict:
    """
    This dependency validates the JWT and returns the payload.
    The 'sub' field in the payload is the Auth0 user_id.
    """
    logger.info("breakpoint")
    if not api_key:
        raise HTTPException(status_code=401, detail="Please send API-KEY in request.")

    user = await get_user_with_api_key(api_key, request)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return user


async def get_current_user_allow_unverified(
    request: Request, api_key: str | None = Depends(api_key_scheme)
) -> dict:
    """Authenticate by API key WITHOUT requiring a verified email.

    Used only by ``/resend_verification_email`` — an account that has not
    verified the signup email yet must still be able to request another
    verification email; every other endpoint uses ``get_current_user``.
    """
    if not api_key:
        raise HTTPException(status_code=401, detail="Please send API-KEY in request.")

    user = await get_user_with_api_key(
        api_key, request, require_verified_email=False
    )
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")

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

    logger.info(f"test breakpoint")
    # cache_key = _hash_key(request.headers.get('x-forwarded-for'))
    if request.app.state.context.dev == "TRUE":
        hashed_ip = _hash_key("172.18.0.1")
        # hashed_ip = '2a1201bb6c0061be63fc4ce58a048136fa91d3afea9e21f62ae7988a20cc09f1' # VPN_SIMULATED
        # hashed_ip = '72aefc13eebd36bf5ec1cbfa1f2e930117a62e07f600dc618c18725f3d52be15' # NO_VPN_SIMULATED
    else:
        hashed_ip = _hash_key(request.headers.get("x-forwarded-for"))
        # hashed_ip = '2a1201bb6c0061be63fc4ce58a048136fa91d3afea9e21f62ae7988a20cc09f1' # VPN_SIMULATED
        # hashed_ip = '72aefc13eebd36bf5ec1cbfa1f2e930117a62e07f600dc618c18725f3d52be15' # NO_VPN_SIMULA

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
        resolve_or_create_anonymous_stripe_customer,
    )

    anonymous_stripe_customer_id = await resolve_or_create_anonymous_stripe_customer(
        request, hashed_ip
    )
    if anonymous_stripe_customer_id:
        user.setdefault("app_metadata", {})[
            "stripe_customer_id"
        ] = anonymous_stripe_customer_id

    return user


async def get_current_user_or_anonymous_user(
    request: Request,
    assistant_id: str = "",
    api_key: str | None = Depends(anonymous_api_key_scheme),
) -> dict:
    """
    This dependency validates the JWT and returns the payload.
    The 'sub' field in the payload is the Auth0 user_id.
    """
    logger.info("breakpoint")
    if not api_key:
        # create anonymous user
        user = await get_anonymous_user_with_anonymous_api_key(
            request=request, assistant_id=assistant_id
        )
    else:
        user = await get_user_with_api_key(api_key, request)

    if not user:
        # create anonymous user
        if not api_key:
            raise HTTPException(
                status_code=500, detail="Error creating anonymous user."
            )
        else:
            raise HTTPException(status_code=401, detail="Invalid API key")

    return user


async def get_current_user_or_anonymous_user_id(
    request: Request, api_key: str | None = Depends(anonymous_api_key_scheme)
) -> dict:
    """
    This dependency validates the JWT and returns the payload.
    The 'sub' field in the payload is the Auth0 user_id.
    """
    logger.info("breakpoint")
    if not api_key:
        # create anonymous user
        user = await get_anonymous_user_with_anonymous_api_key(
            request=request, assistant_id=""
        )
    else:
        user = await get_user_with_api_key(api_key, request)

    if not user:
        # create anonymous user
        if not api_key:
            raise HTTPException(
                status_code=500, detail="Error creating anonymous user."
            )
        else:
            raise HTTPException(status_code=401, detail="Invalid API key")

    return user


# ── Routes ─────────────────────────────────────────────────────────────────
@security_route.post("/signup")
async def signup(body: SignupRequest, request: Request):
    user = await signup_user(body.email, body.password, name=body.name, request=request)
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
    # TODO: RATE LIMIT API CALL
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
async def delete_user(request: Request, current_user: dict = Depends(get_current_user)):
    # Optional: ensure users can only delete themselves unless admin
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

        token = current_user["API_KEY"]
        headers = {"API-KEY": f"{token}"}
        langgraph_sdk_client = get_client(headers=headers)

        metadata = {"user_id": current_user["identities"][0]["user_id"]}

        avatars = await langgraph_sdk_client.assistants.search(
            graph_id="Anubis", metadata=metadata
        )

        for avatar in avatars:
            assistant_id = avatar.get("assistant_id", "")
            try:
                delete_result = await langgraph_sdk_client.assistants.delete(
                    assistant_id=assistant_id, delete_threads=True, headers=headers
                )
            except Exception as e:
                raise HTTPException(
                    detail="Error deleting avatar for user.", status_code=500
                )

        # Delete all entries in the store and store vectors for the created avatars
        pool = request.app.state.pool
        user_id = current_user["identities"][0].get("user_id")
        SQL_STORE_DELETE_QUERY = """DELETE FROM store WHERE prefix = %s OR prefix LIKE %s or prefix LIKE %s or prefix LIKE %s;"""
        SQL_STORE_VECTOR_DELETE_QUERY = """DELETE FROM store WHERE prefix = %s OR prefix LIKE %s or prefix LIKE %s or prefix LIKE %s;"""
        params = (
            user_id,
            f"{user_id}.%",
            f"%.{user_id}.%",
            f"%.{user_id}",
        )
        try:
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(SQL_STORE_DELETE_QUERY, params)
                    await cur.execute(SQL_STORE_VECTOR_DELETE_QUERY, params)
        except Exception as e:
            raise HTTPException(
                detail="Error deleting items from store and store vectors during delete user."
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
            del _api_key_cache[api_key_hash]
            return {"message": "User deleted"}
        else:
            raise HTTPException(status_code=500, detail=f"Error deleting user: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting user: {e}")


# @security_route.post("/login")
# async def login(body: LoginRequest, request: Request):
#     try:
#         # returns: access_token, refresh_token, id_token, expires_in
#         response = await login_user(body.email, body.password, request=request)
#         response.raise_for_status()
#         logger.warning(f"response.status_code: {response.status_code}")
#         if response.status_code == 200:
#             data = response.json()
#             logger.warning(f"DATA: {data}")
#             user_info = jwt.get_unverified_claims(data.get('id_token'))
#             logger.warning(f"DATA: {user_info}")
#             logger.warning("XXXXXXXXXXXXXXXXXXXXX UPDATE USER LOGIN")
#             payload = {
#                 "app_metadata": {
#                     "logged_in": True
#                 }
#             }

#             logger.warning('update login status breakpoint')
#             # Note: user_id must be URL encoded (e.g., auth0|123 -> auth0%7C123)
#             encoded_id = quote(user_info['sub'], safe="")
#             headers = await _mgmt_headers(request)
#             await request.app.state.httpx_client.patch(
#                 f"{BASE_AUTH_URL}/api/v2/users/{encoded_id}",
#                 json=payload,
#                 headers=headers,
#             )
#             return data
#         else:
#             raise HTTPException(status_code=response.status_code, detail=response.json())
#     except Exception as e:
#         raise HTTPException(status_code=response.status_code, detail=response.json())

# @security_route.get("/get_user_profile")
# async def get_user_profile(request: Request, current_user: dict = Depends(get_current_user)):
# You don't need to pass user_id in the URL or body;
# it is extracted from the token you're wearing!
# user_id = current_user["user_id"]
# return {"user_id": user_id}

# @security_route.post("/logout")
# async def logout(body: LogoutRequest, request:Request, current_user: dict = Depends(get_current_user)):

#     response = await logout_user(body.refresh_token, request=request)
#     try:

#         response.raise_for_status()
#         if response.status_code == 200:
#             logger.warning("XXXXXXXXXXXXXXXXXXXXX UPDATE USER LOGIN")
#             payload = {
#                 "user_metadata": {
#                     "logged_in": False
#                 }
#             }

#             logger.warning('update login status breakpoint')
#             # Note: user_id must be URL encoded (e.g., auth0|123 -> auth0%7C123)
#             encoded_id = quote(current_user['user_id'], safe="")
#             headers = await _mgmt_headers(request)
#             await request.app.state.httpx_client.patch(
#                 f"{BASE_AUTH_URL}/api/v2/users/{encoded_id}",
#                 json=payload,
#                 headers=headers,
#             )
#         return {"message": "Logged out successfully"}
#     except Exception as e:
#         raise HTTPException(detail = response.json(), status_code=response.status_code)


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


# Token Authentication
# @auth.authenticate
# async def authenticate(authorization: str | None, request: Request) -> Auth.types.MinimalUserDict:
#     """LangGraph calls this on every request to verify the token."""
#     if not authorization:
#         raise Auth.exceptions.HTTPException(status_code=401, detail="No authorization header")

#     scheme, _, token = authorization.partition(" ")
#     if scheme.lower() != "bearer":
#         raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid auth scheme")

#     try:
#         payload = await verify_token(token, request=request)
#     except Exception as e:
#         raise Auth.exceptions.HTTPException(status_code=401, detail=str(e))

#     # Must return a dict with at least "identity"
#     return {
#         "identity": payload["sub"],          # Auth0 user ID e.g. "auth0|abc123"
#         "email":    payload.get("email"),
#         "permissions": payload.get("permissions", []),
#         "metadata": {"user_id": payload["sub"]}
#     }
