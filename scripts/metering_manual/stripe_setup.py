#!/usr/bin/env python3
"""Stripe + Auth0 state mutations for the metering walkthrough (test mode only).

curl + SQL cover the metering meters and the app-side toggles; the paid-tier and
free-trial-lifecycle situations additionally need real Stripe subscription state
that has no clean curl. This script provides those mutations against the test
customer, reading STRIPE_SECRET_KEY and STRIPE_BILLING_CONFIG_JSON from .env.dev.

Creating / canceling a subscription emits customer.subscription.created /
.updated / .deleted, which the running stripe-cli container forwards to the dev
webhook, so the account's tier syncs to Auth0 within a few seconds — poll
`nn_tier` (env.sh) or GET /verify_subscription_status until it flips.

TWO SEPARATE PIECES OF STATE DECIDE WHAT A METERING TEST SEES, and only one of
them lives in Stripe:

  * The Stripe subscription decides ``status`` and ``tier``. GET
    /verify_subscription_status re-reads the subscription from Stripe on every
    call (check_subscription_status), so tier/status are correct even when no
    webhook was delivered.
  * ``app_metadata.trial_context`` in Auth0 — ``{"tier", "trial_end"}`` — decides
    which ALLOTMENTS apply. While now < trial_end, resolve_effective_monthly_
    allotment grants the trial tier's allotment as a floor over the current
    tier's, and /verify_subscription_status lists any meter the trial tier grants
    even when the current tier does not. NOTHING in the product writes or clears
    this key except the post-verification signup provisioning, so canceling a
    subscription in Stripe leaves a live trial_context behind — which is exactly
    why a canceled account can report ``tier: free`` while showing the pro
    trial's 5M messaging / 10M document-upload allotments. Testing the plain free
    tier therefore requires clearing (or expiring) trial_context, not just
    canceling the subscription.

A third thing bites anyone mutating Auth0 from outside the API: the API keeps a
five-minute in-process TTL cache of each user document (``_api_key_cache``). Its
own writes evict the cache; a PATCH from this script cannot. Every command here
that touches app_metadata therefore ends by calling POST /set_pay_per_use, whose
metadata write evicts the cache, so the next /verify_subscription_status reads
the value this script just wrote instead of serving up to five minutes of stale
billing state.

Subcommands (customer id via --customer or NN_CUSTOMER):
    show                     subscriptions, default payment method, and the Auth0
                             trial_context / cached subscription_status
    attach-card              attach test card pm_card_visa and make it the default
    detach-cards             remove all cards (to test the "no card" 402 path)
    create --tier free|pro|premium [--trial-days N]
                             cancel live subs, then create an ACTIVE (or trialing)
                             subscription with the tier's base + metered prices
    cancel                   cancel every live subscription immediately
    end-trial                end an active trial now (bills, flips trialing->active)
    trial show|clear|expire|grant --tier pro|premium [--days N]
                             read/remove/back-date/write app_metadata.trial_context
    scenario <name>          drive the account into one complete, named billing
                             state (Stripe subscription AND trial_context), then
                             poll /verify_subscription_status until the API agrees:
                               free               active free tier, no trial
                               free-expired-trial free tier, trial_context in the past
                               canceled           no live subscription, no trial
                               canceled-in-trial  canceled while the trial still runs
                               premium-trial      premium, status trialing
                               premium-active     premium, status active, no trial

Safety: refuses to run unless the Stripe key is a test key (sk_test_...).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote

import httpx
import stripe

ENV_DEV = Path(__file__).resolve().parents[2] / ".env.dev"


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw = line.split("=", 1)
        values[key.strip()] = raw.strip().strip("'").strip('"')
    return values


def tier_price_ids(billing_config: dict, tier: str) -> list[str]:
    """Return the tier's base price followed by its graduated metered prices.

    Read straight from the billing config, mirroring ``_subscription_items_for_tier``
    in auth.py so a scenario subscribes exactly what Checkout and the app's own
    server-side enrollment subscribe.

    Listing prices by product instead does NOT work here, and silently: the
    provisioning script puts every metered price on its OWN product (one product
    per meter per tier — prod_... for "Pro messaging tokens" and so on), so
    ``Price.list(product=<tier product>)`` returns just the licensed base price
    and yields a subscription with no metered items at all.
    """
    tier_config = billing_config["tiers"][tier]
    base_price_id = tier_config.get("base_price")
    if not base_price_id:
        raise SystemExit(f"billing config has no base_price for tier {tier}")
    price_ids = [base_price_id]
    price_ids.extend((tier_config.get("metered_prices") or {}).values())
    return price_ids


def default_payment_method(customer: str) -> str | None:
    methods = stripe.PaymentMethod.list(customer=customer, type="card").data
    return methods[0].id if methods else None


def cancel_live(customer: str) -> None:
    for sub in stripe.Subscription.list(customer=customer, status="all", limit=100).auto_paging_iter():
        if sub.status in ("active", "trialing", "past_due", "incomplete"):
            stripe.Subscription.cancel(sub.id)
            print(f"  canceled {sub.id} ({sub.status})")


# ---------------------------------------------------------------------------
# Auth0 side — app_metadata.trial_context and the API's user-document cache
# ---------------------------------------------------------------------------


def auth0_management_headers(env: dict[str, str]) -> dict[str, str]:
    """Return Authorization headers for the Auth0 Management API.

    Uses the same client-credentials application the API itself uses for
    ``_mgmt_headers``, so the read:users / update:users scopes are already
    granted and no extra tenant configuration is needed.
    """
    domain = env["AUTH0_DOMAIN"]
    token_response = httpx.post(
        f"https://{domain}/oauth/token",
        json={
            "grant_type": "client_credentials",
            "client_id": env["AUTH0_CLIENT_ID"],
            "client_secret": env["AUTH0_CLIENT_SECRET"],
            "audience": f"https://{domain}/api/v2/",
        },
        timeout=30,
    )
    token_response.raise_for_status()
    return {"Authorization": f"Bearer {token_response.json()['access_token']}"}


def resolve_auth0_user_id(env: dict[str, str], customer: str) -> str:
    """Return the Auth0 user id for the test account.

    Prefers the NN_USER_ID exported by env.sh; otherwise looks the user up by the
    Stripe customer's email address, so the script still works in a shell where
    only NN_CUSTOMER was set.
    """
    explicit_user_id = os.environ.get("NN_USER_ID")
    if explicit_user_id:
        return explicit_user_id
    email = stripe.Customer.retrieve(customer).to_dict().get("email")
    if not email:
        raise SystemExit(
            f"customer {customer} has no email; export NN_USER_ID to identify the Auth0 user"
        )
    headers = auth0_management_headers(env)
    matches = httpx.get(
        f"https://{env['AUTH0_DOMAIN']}/api/v2/users-by-email",
        params={"email": email},
        headers=headers,
        timeout=30,
    ).json()
    if not matches:
        raise SystemExit(f"no Auth0 user found for email {email}")
    return matches[0]["user_id"]


def read_app_metadata(env: dict[str, str], auth0_user_id: str) -> dict:
    headers = auth0_management_headers(env)
    user_document = httpx.get(
        f"https://{env['AUTH0_DOMAIN']}/api/v2/users/{quote(auth0_user_id, safe='')}",
        headers=headers,
        timeout=30,
    ).json()
    return user_document.get("app_metadata") or {}


def patch_app_metadata(env: dict[str, str], auth0_user_id: str, fields: dict) -> None:
    """Merge ``fields`` into the user's app_metadata, then refresh the API cache.

    Auth0 merges app_metadata at the top level, so only the supplied keys change;
    passing ``None`` as a value deletes that key, which is how trial_context is
    cleared. The API's five-minute user-document cache is invisible to this PATCH,
    so ``refresh_api_user_cache`` is called afterwards — without that step the
    next /verify_subscription_status can keep reporting the previous trial state.
    """
    headers = auth0_management_headers(env)
    response = httpx.patch(
        f"https://{env['AUTH0_DOMAIN']}/api/v2/users/{quote(auth0_user_id, safe='')}",
        headers=headers,
        json={"app_metadata": fields},
        timeout=30,
    )
    response.raise_for_status()
    refresh_api_user_cache()


def refresh_api_user_cache() -> None:
    """Evict the API's cached copy of this account's Auth0 user document.

    ``_api_key_cache`` is an in-process TTLCache with a five-minute lifetime that
    only the API's own metadata writes evict. POST /set_pay_per_use performs such
    a write (``pay_per_use_enabled: false``, the neutral state every scenario
    starts from), so calling it forces the next request to re-read app_metadata
    from Auth0. Needs NN_API and NN_API_KEY; prints guidance when they are absent
    rather than failing, since the state itself is already written.
    """
    api = os.environ.get("NN_API")
    api_key = os.environ.get("NN_API_KEY")
    if not api or not api_key:
        print(
            "  note: NN_API / NN_API_KEY not exported — the API may serve up to "
            "5 minutes of cached billing metadata; source env.sh to avoid the wait"
        )
        return
    try:
        httpx.post(
            f"{api}/set_pay_per_use",
            params={"enabled": "false"},
            headers={"API-KEY": api_key},
            timeout=30,
        )
    except Exception as refresh_error:  # noqa: BLE001 - best effort
        print(f"  note: could not refresh the API user cache: {refresh_error}")


def poll_subscription_status(
    expected_status: str | None = None,
    expected_tier: str | None = None,
    attempts: int = 15,
) -> dict | None:
    """Poll GET /verify_subscription_status until the API reports the target state.

    The webhook that syncs Stripe into Auth0 is asynchronous, so a scenario is only
    proven once the API itself reports the expected status/tier. Returns the last
    payload seen (or None when the endpoint is unreachable).
    """
    api = os.environ.get("NN_API")
    api_key = os.environ.get("NN_API_KEY")
    if not api or not api_key:
        print("  (skipping verification — export NN_API and NN_API_KEY)")
        return None
    payload = None
    for _ in range(attempts):
        try:
            response = httpx.get(
                f"{api}/verify_subscription_status",
                headers={"API-KEY": api_key},
                timeout=30,
            )
        except Exception:  # noqa: BLE001 - the API may still be starting
            time.sleep(2)
            continue
        if response.status_code == 200:
            payload = response.json()
            status_matches = (
                expected_status is None or payload.get("status") == expected_status
            )
            tier_matches = expected_tier is None or payload.get("tier") == expected_tier
            if status_matches and tier_matches:
                break
        time.sleep(2)
    if payload is not None:
        print(json.dumps(payload, indent=2))
    return payload


def cmd_show(customer: str, env, _cfg) -> None:
    # stripe-python 15 returns StripeObject, whose attribute access is key lookup —
    # so a bare `.get(...)` resolves as the missing FIELD "get" and raises
    # AttributeError. Convert to a plain dict before using mapping methods.
    cust = stripe.Customer.retrieve(customer).to_dict()
    print("customer:", customer, "| email:", cust.get("email"))
    print("default_pm:", (cust.get("invoice_settings") or {}).get("default_payment_method"))
    print("cards:", [pm.id for pm in stripe.PaymentMethod.list(customer=customer, type="card").data])
    for raw_subscription in stripe.Subscription.list(customer=customer, status="all", limit=20).data:
        sub = raw_subscription.to_dict()
        print(
            f"  sub {sub['id']}: status={sub['status']} "
            f"cancel_at_period_end={sub.get('cancel_at_period_end')} "
            f"schedule={sub.get('schedule')} "
            f"tier_meta={(sub.get('metadata') or {}).get('neural_nexus_tier')}"
        )
    auth0_user_id = resolve_auth0_user_id(env, customer)
    app_metadata = read_app_metadata(env, auth0_user_id)
    print("auth0 user:", auth0_user_id)
    print("  trial_context      :", json.dumps(app_metadata.get("trial_context")))
    print("  subscription_status:", json.dumps(app_metadata.get("subscription_status")))
    print("  usage_period_anchor:", app_metadata.get("usage_period_anchor"))
    print("  pay_per_use_enabled:", app_metadata.get("pay_per_use_enabled"))
    trial_context = app_metadata.get("trial_context") or {}
    trial_end = trial_context.get("trial_end")
    if trial_end:
        remaining_seconds = int(trial_end) - int(time.time())
        state = (
            f"LIVE — {remaining_seconds // 3600}h left, allotments floored at "
            f"'{trial_context.get('tier')}'"
            if remaining_seconds > 0
            else f"expired {abs(remaining_seconds) // 3600}h ago — no allotment floor"
        )
        print("  trial window       :", state)


def cmd_attach_card(customer: str, _env, _cfg) -> None:
    pm = stripe.PaymentMethod.create(type="card", card={"token": "tok_visa"})
    stripe.PaymentMethod.attach(pm.id, customer=customer)
    stripe.Customer.modify(customer, invoice_settings={"default_payment_method": pm.id})
    print(f"attached + defaulted {pm.id} on {customer}")


def cmd_detach_cards(customer: str, _env, _cfg) -> None:
    for pm in stripe.PaymentMethod.list(customer=customer, type="card").data:
        stripe.PaymentMethod.detach(pm.id)
        print(f"detached {pm.id}")


def cmd_create(customer: str, env, cfg, tier: str, trial_days: int) -> dict:
    # The free tier's base price is $0, so its subscription finalizes without a
    # payment method; only a paid tier billing immediately needs a card.
    if tier != "free" and trial_days <= 0 and not default_payment_method(customer):
        raise SystemExit("no card on file — run `attach-card` first (or pass --trial-days)")
    print(f"canceling any live subscriptions on {customer} ...")
    cancel_live(customer)
    prices = tier_price_ids(cfg, tier)
    params: dict = {
        "customer": customer,
        "items": [{"price": pid} for pid in prices],
        "metadata": {"neural_nexus_tier": tier},
    }
    pm = default_payment_method(customer)
    if pm:
        params["default_payment_method"] = pm
    if trial_days > 0:
        params["trial_period_days"] = trial_days
    sub = stripe.Subscription.create(**params)
    print(f"created {sub.id}: status={sub.status} tier={tier} items={len(prices)}")
    print("  -> poll `nn_tier` until tier syncs via the webhook")
    return sub.to_dict()


def cmd_cancel(customer: str, _env, _cfg) -> None:
    cancel_live(customer)
    print("done")


# ---------------------------------------------------------------------------
# trial_context — the allotment floor that outlives the Stripe subscription
# ---------------------------------------------------------------------------


def cmd_trial(customer: str, env, _cfg, action: str, tier: str | None, days: int) -> None:
    auth0_user_id = resolve_auth0_user_id(env, customer)
    if action == "show":
        app_metadata = read_app_metadata(env, auth0_user_id)
        print(json.dumps(app_metadata.get("trial_context"), indent=2))
        return
    if action == "clear":
        # Auth0 deletes an app_metadata key when the patched value is null. This is
        # what makes an account gate as the PLAIN free tier: without it a canceled
        # account keeps the trial tier's allotments (and the trial tier's extra
        # meters) until the original trial_end passes.
        patch_app_metadata(env, auth0_user_id, {"trial_context": None})
        print(f"cleared trial_context on {auth0_user_id}")
        return
    if action == "expire":
        app_metadata = read_app_metadata(env, auth0_user_id)
        existing = app_metadata.get("trial_context") or {}
        expired_tier = tier or existing.get("tier") or "pro"
        # One day in the past: the same shape a real lapsed trial leaves behind, so
        # the expiry comparison in resolve_effective_monthly_allotment is exercised
        # rather than bypassed by a missing key.
        trial_end = int(time.time()) - 86_400
        patch_app_metadata(
            env, auth0_user_id, {"trial_context": {"tier": expired_tier, "trial_end": trial_end}}
        )
        print(f"back-dated trial_context to {trial_end} (tier={expired_tier}) on {auth0_user_id}")
        return
    if action == "grant":
        if not tier:
            raise SystemExit("trial grant needs --tier pro|premium")
        trial_end = int(time.time()) + days * 86_400
        patch_app_metadata(
            env, auth0_user_id, {"trial_context": {"tier": tier, "trial_end": trial_end}}
        )
        print(f"granted a {days}-day {tier} trial_context (ends {trial_end}) on {auth0_user_id}")
        return
    raise SystemExit(f"unknown trial action '{action}'")


# ---------------------------------------------------------------------------
# Whole named billing states
# ---------------------------------------------------------------------------

SCENARIOS = (
    "free",
    "free-expired-trial",
    "canceled",
    "canceled-in-trial",
    "premium-trial",
    "premium-active",
)


def cmd_scenario(customer: str, env, cfg, scenario: str, trial_days: int) -> None:
    """Drive the account into one complete, named billing state and verify it.

    Each scenario sets BOTH halves of the state — the Stripe subscription (which
    decides status and tier) and app_metadata.trial_context (which decides the
    allotment floor) — because setting only one half is what produces the
    confusing hybrids, such as a canceled account still advertising the pro
    trial's document-upload allotment.
    """
    auth0_user_id = resolve_auth0_user_id(env, customer)
    print(f"scenario '{scenario}' on {customer} / {auth0_user_id}")

    if scenario == "free":
        cancel_live(customer)
        cmd_create(customer, env, cfg, "free", 0)
        patch_app_metadata(env, auth0_user_id, {"trial_context": None})
        print("  trial_context cleared -> plain free tier: messaging only, no uploads")
        poll_subscription_status(expected_status="active", expected_tier="free")

    elif scenario == "free-expired-trial":
        cancel_live(customer)
        cmd_create(customer, env, cfg, "free", 0)
        cmd_trial(customer, env, cfg, "expire", "pro", 0)
        print("  trial_context back-dated -> the lapsed-trial free tier")
        poll_subscription_status(expected_status="active", expected_tier="free")

    elif scenario == "canceled":
        cancel_live(customer)
        patch_app_metadata(env, auth0_user_id, {"trial_context": None})
        print("  no live subscription, no trial -> canceled/free")
        poll_subscription_status(expected_status="canceled", expected_tier="free")

    elif scenario == "canceled-in-trial":
        # Create the trial first so trial_end is a real Stripe timestamp, then
        # cancel: the subscription is gone but the trial grant is not, which is the
        # state a user reaches by canceling partway through the signup trial.
        cancel_live(customer)
        subscription = cmd_create(customer, env, cfg, "pro", trial_days or 30)
        trial_end = int(subscription.get("trial_end") or (time.time() + (trial_days or 30) * 86_400))
        patch_app_metadata(
            env, auth0_user_id, {"trial_context": {"tier": "pro", "trial_end": trial_end}}
        )
        cancel_live(customer)
        refresh_api_user_cache()
        print("  canceled with a LIVE pro trial_context -> free tier at pro allotments")
        poll_subscription_status(expected_status="canceled", expected_tier="free")

    elif scenario == "premium-trial":
        cancel_live(customer)
        subscription = cmd_create(customer, env, cfg, "premium", trial_days or 14)
        trial_end = int(subscription.get("trial_end") or (time.time() + (trial_days or 14) * 86_400))
        patch_app_metadata(
            env, auth0_user_id, {"trial_context": {"tier": "premium", "trial_end": trial_end}}
        )
        print("  premium subscription in its free-trial window")
        poll_subscription_status(expected_status="trialing", expected_tier="premium")

    elif scenario == "premium-active":
        if not default_payment_method(customer):
            cmd_attach_card(customer, env, cfg)
        cancel_live(customer)
        cmd_create(customer, env, cfg, "premium", 0)
        patch_app_metadata(env, auth0_user_id, {"trial_context": None})
        print("  premium billing immediately, no trial")
        poll_subscription_status(expected_status="active", expected_tier="premium")

    else:
        raise SystemExit(f"unknown scenario '{scenario}'")


def cmd_end_trial(customer: str, _env, _cfg) -> None:
    ended = False
    for sub in stripe.Subscription.list(customer=customer, status="trialing", limit=10).data:
        stripe.Subscription.modify(sub.id, trial_end="now")
        print(f"ended trial on {sub.id} -> bills now, flips to active")
        ended = True
    if not ended:
        print("no trialing subscription found")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=[
        "show", "attach-card", "detach-cards", "create", "cancel", "end-trial",
        "trial", "scenario",
    ])
    parser.add_argument(
        "argument",
        nargs="?",
        help="trial: show|clear|expire|grant   scenario: " + "|".join(SCENARIOS),
    )
    parser.add_argument("--customer", default=os.environ.get("NN_CUSTOMER"))
    parser.add_argument("--tier", choices=["free", "pro", "premium"])
    parser.add_argument("--trial-days", type=int, default=0)
    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="trial grant: length of the granted trial_context window in days",
    )
    args = parser.parse_args()

    env = load_env(ENV_DEV)
    stripe.api_key = env["STRIPE_SECRET_KEY"]
    if not stripe.api_key.startswith("sk_test_"):
        raise SystemExit("refusing to run: STRIPE_SECRET_KEY is not a test key")
    cfg = json.loads(env["STRIPE_BILLING_CONFIG_JSON"])

    customer = args.customer
    if not customer:
        raise SystemExit("set NN_CUSTOMER or pass --customer cus_...")

    if args.command == "show":
        cmd_show(customer, env, cfg)
    elif args.command == "attach-card":
        cmd_attach_card(customer, env, cfg)
    elif args.command == "detach-cards":
        cmd_detach_cards(customer, env, cfg)
    elif args.command == "create":
        if not args.tier:
            raise SystemExit("create needs --tier pro|premium")
        cmd_create(customer, env, cfg, args.tier, args.trial_days)
    elif args.command == "cancel":
        cmd_cancel(customer, env, cfg)
    elif args.command == "end-trial":
        cmd_end_trial(customer, env, cfg)
    elif args.command == "trial":
        cmd_trial(customer, env, cfg, args.argument or "show", args.tier, args.days)
    elif args.command == "scenario":
        if args.argument not in SCENARIOS:
            raise SystemExit("scenario needs one of: " + ", ".join(SCENARIOS))
        cmd_scenario(customer, env, cfg, args.argument, args.trial_days)
    return 0


if __name__ == "__main__":
    sys.exit(main())
