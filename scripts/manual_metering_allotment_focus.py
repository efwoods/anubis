#!/usr/bin/env python3
"""Focused allotment over/under tests with rate limits disabled."""

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

API = os.environ.get("NN_API", "http://localhost:8900")
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


def psql(sql: str) -> None:
    subprocess.run(
        [
            "docker", "exec", "-i", "postgres16", "psql", "-U", "postgres", "-d", "postgres",
            "-v", "ON_ERROR_STOP=1", "-c", sql,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def clear_injected() -> None:
    psql(
        f"DELETE FROM api_metrics WHERE user_id = '{USER_ID}' AND inference_type = 'test_inject';"
    )


def inject(meter: str, tokens: int) -> None:
    clear_injected()
    psql(
        f"""
INSERT INTO api_metrics (
  id, created_at, user_id, stripe_customer_id, inference_type,
  prompt_tokens, completion_tokens, total_tokens, cost_usd, latency_ms, meter_event_name
) VALUES (
  '{uuid.uuid4()}', now(), '{USER_ID}', '{CUSTOMER}', 'test_inject',
  0, 0, {int(tokens)}, 0, 0, '{meter}'
);
"""
    )


def record(name: str, ok: bool, detail: Any) -> None:
    RESULTS.append({"name": name, "ok": ok, "detail": detail})
    print(f"\n=== {'PASS' if ok else 'FAIL'}: {name} ===")
    print(json.dumps(detail, indent=2, default=str)[:2000] if isinstance(detail, (dict, list)) else str(detail)[:2000])


def verify() -> dict:
    r = httpx.get(f"{API}/verify_subscription_status", headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


def set_ppu(enabled: bool) -> httpx.Response:
    return httpx.post(
        f"{API}/set_pay_per_use", headers=HEADERS,
        params={"enabled": str(enabled).lower()}, timeout=60,
    )


def subscribe(tier: str) -> httpx.Response:
    return httpx.post(f"{API}/subscribe", headers=HEADERS, params={"tier": tier}, timeout=120)


def message(adapter: bool = False) -> tuple[int, str]:
    data = {
        "message": "Reply with exactly: ok",
        "stream": "true",
        "include_quality_metrics": "false",
        "include_usage_metrics": "true",
        "adapter": str(adapter).lower(),
    }
    with httpx.Client(timeout=180) as client:
        with client.stream("POST", f"{API}/message/{AVATAR}", headers=HEADERS, data=data) as r:
            return r.status_code, "".join(r.iter_text())


def upload() -> tuple[int, str]:
    with TOKEN_FILE.open("rb") as fh:
        r = httpx.post(
            f"{API}/update_avatar_identity_with_media",
            headers=HEADERS,
            data={"assistant_id": AVATAR},
            files={"files": (TOKEN_FILE.name, fh, "text/markdown")},
            timeout=180,
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
    st = verify()
    meters = st.get("meters") or {}
    if "messaging_tokens" not in meters:
        record(f"{label}_no_messaging_meter", False, st)
        return
    allot = int(meters["messaging_tokens"]["monthly_allotment"])
    set_ppu(False)
    clear_injected()
    code, body = message()
    record(f"{label}_messaging_under", code == 200 and "assistant_token" in body, {
        "http": code, "usage": done_usage(body), "status": verify()
    })
    inject("messaging_tokens", allot)
    code, body = message()
    record(f"{label}_messaging_over_ppu_off", code == 402, {"http": code, "body": body[:600]})
    r = set_ppu(True)
    record(f"{label}_ppu_on", r.status_code == 200, {"http": r.status_code, "body": r.text[:200]})
    inject("messaging_tokens", allot)
    code, body = message()
    record(f"{label}_messaging_over_ppu_on", code == 200 and "assistant_token" in body, {
        "http": code, "usage": done_usage(body), "snippet": body[:400]
    })
    set_ppu(False)
    clear_injected()


def upload_matrix(label: str) -> None:
    st = verify()
    meters = st.get("meters") or {}
    if "document_upload_tokens" not in meters:
        # free tier: upload should be capability-blocked
        code, body = upload()
        record(f"{label}_upload_capability_blocked", code in (402, 403), {"http": code, "body": body[:500]})
        return
    allot = int(meters["document_upload_tokens"]["monthly_allotment"])
    set_ppu(False)
    clear_injected()
    code, body = upload()
    record(f"{label}_upload_under", code in (200, 202), {"http": code, "body": body[:400]})
    inject("document_upload_tokens", allot)
    code, body = upload()
    record(f"{label}_upload_over_ppu_off", code == 402, {"http": code, "body": body[:600]})
    set_ppu(True)
    inject("document_upload_tokens", allot)
    code, body = upload()
    record(f"{label}_upload_over_ppu_on", code in (200, 202), {"http": code, "body": body[:400]})
    set_ppu(False)
    clear_injected()


def adapter_matrix(label: str) -> None:
    st = verify()
    meters = st.get("meters") or {}
    if "adapter_inference_tokens" not in meters:
        record(f"{label}_adapter_n/a", True, "tier lacks adapter inference")
        return
    allot = int(meters["adapter_inference_tokens"]["monthly_allotment"])
    set_ppu(False)
    clear_injected()
    code, body = message(adapter=True)
    record(f"{label}_adapter_under", code == 200 and "assistant_token" in body, {
        "http": code, "usage": done_usage(body)
    })
    inject("adapter_inference_tokens", allot)
    code, body = message(adapter=True)
    record(f"{label}_adapter_over_ppu_off", code == 402, {"http": code, "body": body[:600]})
    set_ppu(True)
    inject("adapter_inference_tokens", allot)
    code, body = message(adapter=True)
    record(f"{label}_adapter_over_ppu_on", code == 200 and "assistant_token" in body, {
        "http": code, "usage": done_usage(body), "snippet": body[:400]
    })
    set_ppu(False)
    clear_injected()


def ensure_tier(tier: str) -> None:
    st = verify()
    if st.get("tier") == tier and st.get("status") in ("active", "trialing"):
        record(f"already_on_{tier}", True, st)
        return
    r = subscribe(tier)
    body = r.text
    record(f"subscribe_{tier}", r.status_code == 200, {"http": r.status_code, "body": body[:800]})
    if r.status_code == 200:
        try:
            data = r.json()
        except Exception:
            data = {}
        if data.get("action") == "start_checkout" and data.get("url"):
            # Create subscription directly with test card instead of browser checkout
            create_paid_subscription(tier)
    # wait for webhook sync
    for _ in range(15):
        time.sleep(2)
        st = verify()
        if st.get("tier") == tier and st.get("status") in ("active", "trialing"):
            break
    record(f"ensure_tier_{tier}_result", verify().get("tier") == tier, verify())


def create_paid_subscription(tier: str) -> None:
    """Create an active subscription for ``tier`` using the attached test card."""
    # Load price ids from env billing config
    import ast
    raw = None
    # Prefer reading from running container env via a tiny probe — use .env.dev JSON
    cfg_path = Path("/home/user/gh/anubis-project/wt/f-metering/.env.dev")
    text = cfg_path.read_text()
    for line in text.splitlines():
        if line.startswith("STRIPE_BILLING_CONFIG_JSON="):
            raw = line.split("=", 1)[1].strip().strip("'").strip('"')
            break
    if not raw:
        raise RuntimeError("no billing config")
    cfg = json.loads(raw)
    tier_cfg = cfg["tiers"][tier]
    price_ids = [tier_cfg["base_price"]] + list(tier_cfg.get("metered_prices", {}).values())
    # Cancel any leftover live subs
    for sub in stripe.Subscription.list(customer=CUSTOMER, status="all", limit=10).auto_paging_iter():
        if sub.status in ("active", "trialing", "past_due"):
            stripe.Subscription.cancel(sub.id)
    sub = stripe.Subscription.create(
        customer=CUSTOMER,
        items=[{"price": pid} for pid in price_ids],
        default_payment_method=stripe.PaymentMethod.list(customer=CUSTOMER, type="card").data[0].id,
        metadata={"auth0_user_id": USER_ID, "neural_nexus_tier": tier},
    )
    record(f"stripe_create_{tier}_sub", True, {"id": sub.id, "status": sub.status})
    # Give webhook a moment; also patch Auth0 via verify by forcing a status check
    time.sleep(5)


def main() -> int:
    st = verify()
    record("baseline", True, {"tier": st.get("tier"), "status": st.get("status"), "ppu": st.get("pay_per_use_enabled")})

    # FREE
    if st.get("tier") != "free":
        # cancel live sub if any
        for sub in stripe.Subscription.list(customer=CUSTOMER, status="all", limit=5).auto_paging_iter():
            if sub.status in ("active", "trialing"):
                stripe.Subscription.cancel(sub.id)
        time.sleep(5)
    # Ensure Auth0 shows free
    for _ in range(10):
        st = verify()
        if st.get("tier") == "free":
            break
        time.sleep(2)
    record("free_status", verify().get("tier") == "free", verify())
    messaging_matrix("free")
    upload_matrix("free")

    # PRO (paid / active — trial already used)
    create_paid_subscription("pro")
    # Force Auth0 sync via webhook wait; if still free, manually not available — try subscribe
    for _ in range(20):
        st = verify()
        if st.get("tier") == "pro":
            break
        time.sleep(2)
    if verify().get("tier") != "pro":
        r = subscribe("pro")
        record("subscribe_pro_fallback", r.status_code == 200, r.text[:500])
        time.sleep(5)
    record("pro_status", verify().get("tier") == "pro", verify())
    if verify().get("tier") == "pro":
        messaging_matrix("pro")
        upload_matrix("pro")

    # PREMIUM upgrade (should clear usage period)
    before = verify()
    r = subscribe("premium")
    after = verify()
    record("upgrade_pro_to_premium", r.status_code == 200, {
        "http": r.status_code, "body": r.text[:600],
        "before_period": before.get("usage_period_start"),
        "after_period": after.get("usage_period_start"),
        "before_used": before.get("meters", {}).get("messaging_tokens", {}).get("used_to_date"),
        "after_used": after.get("meters", {}).get("messaging_tokens", {}).get("used_to_date"),
        "anchor_reset": before.get("usage_period_start") != after.get("usage_period_start"),
        "tier": after.get("tier"),
    })
    if after.get("tier") != "premium":
        create_paid_subscription("premium")
        time.sleep(5)
    record("premium_status", verify().get("tier") == "premium", verify())
    if verify().get("tier") == "premium":
        messaging_matrix("premium")
        upload_matrix("premium")
        adapter_matrix("premium")

    clear_injected()
    out = Path("/tmp/metering_allotment_focus.json")
    out.write_text(json.dumps(RESULTS, indent=2, default=str))
    passed = sum(1 for x in RESULTS if x["ok"])
    failed = sum(1 for x in RESULTS if not x["ok"])
    print(f"\nSUMMARY: {passed} passed, {failed} failed -> {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
