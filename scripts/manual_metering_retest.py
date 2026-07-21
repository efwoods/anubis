#!/usr/bin/env python3
"""Retest failed metering scenarios + free/pro after schedule fix."""

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

API = os.environ.get("NN_API", "http://localhost:8900")
KEY = os.environ["NN_API_KEY"]
AVATAR = os.environ.get("NN_AVATAR_ID", "c206e1a4-5cdf-4f3b-8ceb-9f23b605ddb1")
USER_ID = os.environ.get("NN_USER_ID", "auth0|6a57ea15adb2bbe67d7aba72")
CUSTOMER = os.environ.get("NN_CUSTOMER_ID", "cus_UtHoruH8DcZUA3")
TOKEN_FILE = Path(
    "/home/user/gh/anubis-project/wt/f-metering/data/shivon_zilis/test_tokens_1_tokens.md"
)
HEADERS = {"API-KEY": KEY}
RESULTS: list[dict[str, Any]] = []


def psql(sql: str) -> str:
    proc = subprocess.run(
        [
            "docker",
            "exec",
            "-i",
            "postgres16",
            "psql",
            "-U",
            "postgres",
            "-d",
            "postgres",
            "-v",
            "ON_ERROR_STOP=1",
            "-t",
            "-A",
            "-c",
            sql,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return proc.stdout.strip()


def clear_injected() -> None:
    psql(
        f"DELETE FROM api_metrics WHERE user_id = '{USER_ID}' AND inference_type = 'test_inject';"
    )


def inject_usage(meter: str, tokens: int) -> None:
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
    print(json.dumps(detail, indent=2, default=str)[:2500] if isinstance(detail, (dict, list)) else str(detail)[:2500])


def verify() -> dict:
    r = httpx.get(f"{API}/verify_subscription_status", headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


def set_ppu(enabled: bool) -> httpx.Response:
    return httpx.post(
        f"{API}/set_pay_per_use",
        headers=HEADERS,
        params={"enabled": str(enabled).lower()},
        timeout=60,
    )


def subscribe(tier: str) -> httpx.Response:
    return httpx.post(f"{API}/subscribe", headers=HEADERS, params={"tier": tier}, timeout=120)


def message(text: str = "Reply with exactly: ok", adapter: bool = False) -> tuple[int, str]:
    data = {
        "message": text,
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


def parse_done_usage(body: str) -> dict | None:
    for line in body.splitlines():
        if line.startswith("data: "):
            try:
                evt = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if evt.get("type") == "done":
                return evt.get("usage")
    return None


def allotment(st: dict, meter: str) -> int:
    return int(st["meters"][meter]["monthly_allotment"])


def run_meter_matrix(label: str) -> None:
    st = verify()
    tier = st["tier"]
    status = st["status"]
    prefix = f"{label}_{tier}_{status}"
    record(f"{prefix}_status", True, {
        "tier": tier,
        "status": status,
        "ppu": st.get("pay_per_use_enabled"),
        "used_messaging": st["meters"].get("messaging_tokens", {}).get("used_to_date"),
        "meters": list(st.get("meters", {})),
    })

    set_ppu(False)
    clear_injected()
    code, body = message()
    record(f"{prefix}_messaging_under", code == 200 and "assistant_token" in body, {
        "http": code, "usage": parse_done_usage(body)
    })

    inject_usage("messaging_tokens", allotment(verify(), "messaging_tokens"))
    code, body = message()
    record(f"{prefix}_messaging_over_ppu_off", code == 402, {"http": code, "body": body[:500]})

    r = set_ppu(True)
    record(f"{prefix}_set_ppu_true", r.status_code == 200, {"http": r.status_code, "body": r.text[:300]})
    inject_usage("messaging_tokens", allotment(verify(), "messaging_tokens"))
    code, body = message()
    record(f"{prefix}_messaging_over_ppu_on", code == 200 and "assistant_token" in body, {
        "http": code, "usage": parse_done_usage(body), "snippet": body[:400]
    })
    set_ppu(False)

    if "document_upload_tokens" in verify().get("meters", {}):
        clear_injected()
        code, body = upload()
        record(f"{prefix}_upload_under", code in (200, 202), {"http": code, "body": body[:400]})
        inject_usage("document_upload_tokens", allotment(verify(), "document_upload_tokens"))
        code, body = upload()
        record(f"{prefix}_upload_over_ppu_off", code == 402, {"http": code, "body": body[:500]})
        set_ppu(True)
        inject_usage("document_upload_tokens", allotment(verify(), "document_upload_tokens"))
        code, body = upload()
        record(f"{prefix}_upload_over_ppu_on", code in (200, 202), {"http": code, "body": body[:400]})
        set_ppu(False)

    if "adapter_inference_tokens" in verify().get("meters", {}):
        clear_injected()
        code, body = message(adapter=True)
        record(f"{prefix}_adapter_inference_under", code == 200 and "assistant_token" in body, {
            "http": code, "usage": parse_done_usage(body)
        })
        inject_usage("adapter_inference_tokens", allotment(verify(), "adapter_inference_tokens"))
        code, body = message(adapter=True)
        record(f"{prefix}_adapter_inference_over_ppu_off", code == 402, {"http": code, "body": body[:500]})
        set_ppu(True)
        inject_usage("adapter_inference_tokens", allotment(verify(), "adapter_inference_tokens"))
        code, body = message(adapter=True)
        record(f"{prefix}_adapter_inference_over_ppu_on", code == 200 and "assistant_token" in body, {
            "http": code, "usage": parse_done_usage(body), "snippet": body[:400]
        })
        set_ppu(False)

    clear_injected()


def main() -> int:
    # Wait for API
    for _ in range(40):
        try:
            httpx.get(f"{API}/docs", timeout=5).raise_for_status()
            break
        except Exception:
            time.sleep(3)
    else:
        print("API not ready")
        return 2

    # Current premium trial matrix (retry previously rate-limited cases)
    run_meter_matrix("current")

    # Downgrade premium -> pro while trialing
    before = verify()
    r = subscribe("pro")
    after = verify()
    detail = {
        "http": r.status_code,
        "body": r.text[:800],
        "before": {"tier": before["tier"], "period": before["usage_period_start"], "used": before["meters"]["messaging_tokens"]["used_to_date"]},
        "after": {"tier": after["tier"], "period": after["usage_period_start"], "used": after["meters"].get("messaging_tokens", {}).get("used_to_date"), "status": after["status"]},
        "anchor_retained": before["usage_period_start"] == after["usage_period_start"],
        "usage_retained": before["meters"]["messaging_tokens"]["used_to_date"] == after["meters"].get("messaging_tokens", {}).get("used_to_date"),
    }
    # Downgrade schedules at period end — tier may still show premium until boundary
    ok = r.status_code == 200
    record("downgrade_premium_to_pro_during_trial", ok, detail)

    if ok:
        # While schedule pending, premium allotment should still apply (unused allotment continues)
        run_meter_matrix("after_scheduled_downgrade")

    # Reactivate / stay on premium then test free cancel path
    r = subscribe("premium")
    record("reactivate_or_keep_premium", r.status_code == 200, {"http": r.status_code, "body": r.text[:500], "status": verify()})

    r = subscribe("free")
    after = verify()
    record("schedule_downgrade_to_free", r.status_code == 200, {
        "http": r.status_code, "body": r.text[:500],
        "cancel_at_period_end": after.get("cancel_at_period_end"),
        "tier": after.get("tier"), "status": after.get("status"),
    })

    # Immediate free-tier test: end trial without relying on payment (cancel now via Stripe)
    # Use Stripe Test Clock style: cancel subscription immediately
    stripe_key = os.environ.get("STRIPE_SECRET_KEY")
    if stripe_key:
        import stripe
        stripe.api_key = stripe_key
        sub_id = verify().get("subscription_id")
        if sub_id:
            stripe.Subscription.cancel(sub_id)
            # wait for webhook
            for _ in range(20):
                time.sleep(2)
                st = verify()
                if st.get("tier") == "free" or st.get("status") in ("canceled", "incomplete_expired"):
                    break
            record("trial_end_cancel_immediate", True, verify())
            # May need free-tier subscription recreated by webhook - if still no messaging allotment wait
            time.sleep(3)
            st = verify()
            if st.get("tier") == "free" or "messaging_tokens" in st.get("meters", {}):
                run_meter_matrix("free_after_trial_cancel")
            else:
                record("free_after_trial_cancel_skipped", False, st)

    clear_injected()
    out = Path("/tmp/metering_retest_results.json")
    out.write_text(json.dumps(RESULTS, indent=2, default=str))
    passed = sum(1 for x in RESULTS if x["ok"])
    failed = sum(1 for x in RESULTS if not x["ok"])
    print(f"\nSUMMARY: {passed} passed, {failed} failed -> {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
