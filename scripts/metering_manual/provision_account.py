#!/usr/bin/env python3
"""Create a fresh, verified, provisioned test account for metering validation.

Runs the full "cold start" a new user goes through, minus the un-sendable
verification email (the dev Auth0 tenant has no email provider), so you end up
with a disposable account sitting on the pro free trial with an avatar to
message:

  1. POST /signup                     -> creates the Auth0 user, returns the raw API key
  2. Auth0 Management API              -> PATCH email_verified:true (the email never sends)
  3. GET /verify_subscription_status   -> first authenticated call fires provisioning
                                          (pro tier, status=trialing) and yields the
                                          Stripe customer id
  4. POST /create_avatar               -> an avatar owned by the account to message

It then prints the four `export` lines to paste before `source env.sh`.

Config is read from .env.dev (AUTH0_DOMAIN / AUTH0_CLIENT_ID / AUTH0_CLIENT_SECRET
/ PORT). The management client-credentials app is the same one the API itself
uses for `_mgmt_headers`, so it already has update:users / read:users.

Usage:
    python scripts/metering_manual/provision_account.py \
        --email metering+$(date +%s)@example.com --name "Metering Test"
    # password defaults to a policy-satisfying value; override with --password
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

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


def mgmt_token(domain: str, client_id: str, client_secret: str) -> str:
    resp = httpx.post(
        f"https://{domain}/oauth/token",
        json={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "audience": f"https://{domain}/api/v2/",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", default="MeterTest!2026xY")
    parser.add_argument("--name", default="Metering Test")
    parser.add_argument("--avatar-name", default="Metering Test Avatar")
    args = parser.parse_args()

    env = load_env(ENV_DEV)
    api = f"http://localhost:{env.get('PORT', '8123')}"
    domain = env["AUTH0_DOMAIN"]
    client_id = env["AUTH0_CLIENT_ID"]
    client_secret = env["AUTH0_CLIENT_SECRET"]

    admin_user_id = env.get("ADMIN_USER_ID", "")

    # 1. signup -----------------------------------------------------------------
    print(f"[1/4] POST /signup  {args.email}")
    signup = httpx.post(
        f"{api}/signup",
        json={"email": args.email, "password": args.password, "name": args.name},
        timeout=60,
    )
    if signup.status_code >= 400:
        print(f"  signup failed: HTTP {signup.status_code} {signup.text[:400]}")
        return 1
    signup_body = signup.json()
    api_key = signup_body.get("api_key")
    if not api_key:
        print(f"  no api_key in signup response: {json.dumps(signup_body)[:400]}")
        return 1
    print("  ok — api_key received")

    # 2. verify email via the Management API ------------------------------------
    print("[2/4] Auth0 Management API — mark email_verified")
    token = mgmt_token(domain, client_id, client_secret)
    headers = {"Authorization": f"Bearer {token}"}
    lookup = httpx.get(
        f"https://{domain}/api/v2/users-by-email",
        params={"email": args.email},
        headers=headers,
        timeout=30,
    )
    lookup.raise_for_status()
    matches = lookup.json()
    if not matches:
        print("  could not find the just-created user by email")
        return 1
    user_id = matches[0]["user_id"]
    from urllib.parse import quote

    patch = httpx.patch(
        f"https://{domain}/api/v2/users/{quote(user_id, safe='')}",
        headers=headers,
        json={"email_verified": True},
        timeout=30,
    )
    patch.raise_for_status()
    print(f"  ok — {user_id} email_verified=true")

    if admin_user_id and user_id == admin_user_id:
        print(
            "  WARNING: this account IS ADMIN_USER_ID — enforcement is bypassed for it. "
            "Use a different email."
        )

    # 3. first authenticated call → provisioning (pro trial) --------------------
    print("[3/4] GET /verify_subscription_status — trigger provisioning")
    customer_id = None
    tier = status = None
    for attempt in range(10):
        vs = httpx.get(
            f"{api}/verify_subscription_status",
            headers={"API-KEY": api_key},
            timeout=60,
        )
        if vs.status_code == 200:
            data = vs.json()
            tier, status = data.get("tier"), data.get("status")
            customer_id = data.get("customer_id")
            if customer_id and tier:
                break
        time.sleep(2)
    print(f"  tier={tier} status={status} customer={customer_id}")

    # 4. create an avatar to message -------------------------------------------
    print("[4/4] POST /create_avatar")
    avatar = httpx.post(
        f"{api}/create_avatar",
        headers={"API-KEY": api_key},
        params={"name": args.avatar_name, "description": "Metering validation avatar"},
        timeout=60,
    )
    avatar_id = None
    if avatar.status_code < 400:
        body = avatar.json()
        avatar_id = body.get("assistant_id") or body.get("id")
    print(f"  avatar_id={avatar_id}  (HTTP {avatar.status_code})")

    print("\n===== paste these before `source scripts/metering_manual/env.sh` =====")
    print(f'export NN_API="{api}"')
    print(f'export NN_API_KEY="{api_key}"')
    print(f'export NN_AVATAR_ID="{avatar_id}"')
    print(f'export NN_USER_ID="{user_id}"')
    print(f'export NN_CUSTOMER="{customer_id}"')
    print("======================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
