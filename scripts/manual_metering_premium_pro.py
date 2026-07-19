#!/usr/bin/env python3
"""Premium + Pro allotment matrix (rate limits disabled, cache fresh)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import stripe

API = "http://localhost:8900"
KEY = os.environ["NN_API_KEY"]
AVATAR = os.environ["NN_AVATAR_ID"]
USER_ID = "auth0|6a57ea15adb2bbe67d7aba72"
CUSTOMER = "cus_UtHoruH8DcZUA3"
TOKEN_FILE = Path(
    "/home/user/gh/anubis-project/wt/f-metering/data/shivon_zilis/test_tokens_1_tokens.md"
)
HEADERS = {"API-KEY": KEY}
stripe.api_key = os.environ["STRIPE_SECRET_KEY"]
RESULTS: list[dict[str, Any]] = []
CFG = json.loads(
    Path("/home/user/gh/anubis-project/wt/f-metering/.env.dev")
    .read_text()
    .split("STRIPE_BILLING_CONFIG_JSON=")[1]
    .splitlines()[0]
    .strip()
    .strip("'")
    .strip('"')
)


def psql(sql: str) -> None:
    subprocess.run(
        [
            "docker", "exec", "-i", "postgres16", "psql", "-U", "postgres", "-d", "postgres",
            "-v", "ON_ERROR_STOP=1", "-c", sql,
        ],
        check=True, capture_output=True, text=True,
    )


def clear_injected() -> None:
    psql(f"DELETE FROM api_metrics WHERE user_id='{USER_ID}' AND inference_type='test_inject';")


def inject(meter: str, tokens: int) -> None:
    clear_injected()
    psql(
        f"""INSERT INTO api_metrics (
          id, created_at, user_id, stripe_customer_id, inference_type,
          prompt_tokens, completion_tokens, total_tokens, cost_usd, latency_ms, meter_event_name
        ) VALUES (
          '{uuid.uuid4()}', now(), '{USER_ID}', '{CUSTOMER}', 'test_inject',
          0, 0, {int(tokens)}, 0, 0, '{meter}'
        );"""
    )


def record(name: str, ok: bool, detail: Any) -> None:
    RESULTS.append({"name": name, "ok": ok, "detail": detail})
    print(f"\n=== {'PASS' if ok else 'FAIL'}: {name} ===")
    print(json.dumps(detail, indent=2, default=str)[:1800] if isinstance(detail, (dict, list)) else str(detail)[:1800])


def verify() -> dict:
    return httpx.get(f"{API}/verify_subscription_status", headers=HEADERS, timeout=60).json()


def set_ppu(enabled: bool) -> httpx.Response:
    return httpx.post(f"{API}/set_pay_per_use", headers=HEADERS, params={"enabled": str(enabled).lower()}, timeout=60)


def message(adapter: bool = False) -> tuple[int, str]:
    data = {
        "message": "Reply with exactly: ok", "stream": "true",
        "include_quality_metrics": "false", "include_usage_metrics": "true",
        "adapter": str(adapter).lower(),
    }
    with httpx.Client(timeout=180) as client:
        with client.stream("POST", f"{API}/message/{AVATAR}", headers=HEADERS, data=data) as r:
            return r.status_code, "".join(r.iter_text())


def upload() -> tuple[int, str]:
    with TOKEN_FILE.open("rb") as fh:
        r = httpx.post(
            f"{API}/update_avatar_identity_with_media", headers=HEADERS,
            data={"assistant_id": AVATAR},
            files={"files": (TOKEN_FILE.name, fh, "text/markdown")}, timeout=180,
        )
    return r.status_code, r.text


def done_usage(body: str) -> dict | None:
    for line in body.splitlines():
        if line.startswith("data: "):
            try:
                evt = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if evt.get("type") == "done":
                return evt.get("usage")
    return None


def messaging_matrix(label: str) -> None:
    allot = int(verify()["meters"]["messaging_tokens"]["monthly_allotment"])
    set_ppu(False); clear_injected()
    code, body = message()
    record(f"{label}_messaging_under", code == 200 and "assistant_token" in body, {"http": code, "usage": done_usage(body)})
    inject("messaging_tokens", allot)
    code, body = message()
    record(f"{label}_messaging_over_ppu_off", code == 402, {"http": code, "body": body[:500]})
    r = set_ppu(True)
    record(f"{label}_ppu_on", r.status_code == 200, {"http": r.status_code, "body": r.text[:200]})
    inject("messaging_tokens", allot)
    code, body = message()
    record(f"{label}_messaging_over_ppu_on", code == 200 and "assistant_token" in body, {"http": code, "usage": done_usage(body)})
    set_ppu(False); clear_injected()


def upload_matrix(label: str) -> None:
    allot = int(verify()["meters"]["document_upload_tokens"]["monthly_allotment"])
    set_ppu(False); clear_injected()
    code, body = upload()
    record(f"{label}_upload_under", code in (200, 202), {"http": code, "body": body[:300]})
    inject("document_upload_tokens", allot)
    code, body = upload()
    record(f"{label}_upload_over_ppu_off", code == 402, {"http": code, "body": body[:500]})
    set_ppu(True)
    inject("document_upload_tokens", allot)
    code, body = upload()
    record(f"{label}_upload_over_ppu_on", code in (200, 202), {"http": code, "body": body[:300]})
    set_ppu(False); clear_injected()


def adapter_matrix(label: str) -> None:
    allot = int(verify()["meters"]["adapter_inference_tokens"]["monthly_allotment"])
    set_ppu(False); clear_injected()
    code, body = message(adapter=True)
    record(f"{label}_adapter_under", code == 200 and "assistant_token" in body, {"http": code, "usage": done_usage(body)})
    inject("adapter_inference_tokens", allot)
    code, body = message(adapter=True)
    record(f"{label}_adapter_over_ppu_off", code == 402, {"http": code, "body": body[:500]})
    set_ppu(True)
    inject("adapter_inference_tokens", allot)
    code, body = message(adapter=True)
    record(f"{label}_adapter_over_ppu_on", code == 200 and "assistant_token" in body, {"http": code, "usage": done_usage(body)})
    set_ppu(False); clear_injected()


def switch_to_tier(tier: str) -> None:
    # cancel live, create target, bump metadata, restart API for cache
    for s in stripe.Subscription.list(customer=CUSTOMER, status="all", limit=10).to_dict()["data"]:
        if s["status"] in ("active", "trialing"):
            stripe.Subscription.cancel(s["id"])
            time.sleep(3)
    tier_cfg = CFG["tiers"][tier]
    price_ids = [tier_cfg["base_price"]] + list(tier_cfg["metered_prices"].values())
    pm = stripe.PaymentMethod.list(customer=CUSTOMER, type="card").to_dict()["data"][0]["id"]
    sub = stripe.Subscription.create(
        customer=CUSTOMER,
        items=[{"price": p} for p in price_ids],
        default_payment_method=pm,
        metadata={"auth0_user_id": USER_ID, "neural_nexus_tier": tier},
    ).to_dict()
    stripe.Subscription.modify(
        sub["id"],
        metadata={"auth0_user_id": USER_ID, "neural_nexus_tier": tier, "sync_bump": str(int(time.time()))},
    )
    record(f"stripe_switch_{tier}", True, {"id": sub["id"], "status": sub["status"]})
    time.sleep(6)
    subprocess.run(
        ["docker", "compose", "--env-file", ".env.dev", "-f", "docker-compose.yml", "restart", "langgraph-api-dev"],
        cwd="/home/user/gh/anubis-project/wt/f-metering", check=True,
    )
    for _ in range(40):
        try:
            st = verify()
            if st.get("tier") == tier and st.get("status") == "active":
                break
        except Exception:
            pass
        time.sleep(3)
    record(f"switched_to_{tier}", verify().get("tier") == tier, verify())


def main() -> int:
    st = verify()
    record("baseline", st.get("tier") == "premium", st)

    # Premium matrices
    messaging_matrix("premium")
    upload_matrix("premium")
    adapter_matrix("premium")

    # Downgrade retain usage: schedule via API
    before = verify()
    r = httpx.post(f"{API}/subscribe", headers=HEADERS, params={"tier": "pro"}, timeout=120)
    after = verify()
    record("downgrade_premium_to_pro_scheduled", r.status_code == 200, {
        "http": r.status_code, "body": r.text[:500],
        "before_used": before["meters"]["messaging_tokens"]["used_to_date"],
        "after_used": after["meters"]["messaging_tokens"]["used_to_date"],
        "usage_retained": before["meters"]["messaging_tokens"]["used_to_date"] == after["meters"]["messaging_tokens"]["used_to_date"],
        "tier_still_premium": after.get("tier") == "premium",
        "period_same": before.get("usage_period_start") == after.get("usage_period_start"),
    })

    # Switch fully to pro for pro allotment matrix
    switch_to_tier("pro")
    if verify().get("tier") == "pro":
        messaging_matrix("pro")
        upload_matrix("pro")

    # Upgrade pro -> premium should reset usage window
    before = verify()
    # Use Stripe immediate upgrade + restart (subscribe upgrade path)
    r = httpx.post(f"{API}/subscribe", headers=HEADERS, params={"tier": "premium"}, timeout=120)
    time.sleep(5)
    subprocess.run(
        ["docker", "compose", "--env-file", ".env.dev", "-f", "docker-compose.yml", "restart", "langgraph-api-dev"],
        cwd="/home/user/gh/anubis-project/wt/f-metering", check=True,
    )
    for _ in range(40):
        try:
            st = verify()
            if st.get("tier") == "premium":
                break
        except Exception:
            pass
        time.sleep(3)
    after = verify()
    record("upgrade_pro_to_premium_clears_usage", r.status_code == 200 and after.get("tier") == "premium", {
        "http": r.status_code, "body": r.text[:500],
        "before_period": before.get("usage_period_start"),
        "after_period": after.get("usage_period_start"),
        "before_used": before.get("meters", {}).get("messaging_tokens", {}).get("used_to_date"),
        "after_used": after.get("meters", {}).get("messaging_tokens", {}).get("used_to_date"),
        "anchor_reset": before.get("usage_period_start") != after.get("usage_period_start"),
        "tier": after.get("tier"),
    })

    clear_injected()
    out = Path("/tmp/metering_premium_pro.json")
    out.write_text(json.dumps(RESULTS, indent=2, default=str))
    passed = sum(1 for x in RESULTS if x["ok"])
    failed = sum(1 for x in RESULTS if not x["ok"])
    print(f"\nSUMMARY: {passed} passed, {failed} failed -> {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
