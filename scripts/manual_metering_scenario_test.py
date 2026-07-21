#!/usr/bin/env python3
"""Manual metering scenario runner for TEST_SITUATIONS.md.

Uses injected api_metrics rows to force over-allotment without burning tokens.
Does not commit secrets; API key is passed via env.
"""

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
    os.environ.get(
        "NN_TEST_TOKEN_FILE",
        "/home/user/gh/anubis-project/wt/f-metering/data/shivon_zilis/test_tokens_1_tokens.md",
    )
)

HEADERS = {"API-KEY": KEY}
RESULTS: list[dict[str, Any]] = []


def psql(sql: str) -> str:
    proc = subprocess.run(
        ["docker", "exec", "-i", "postgres16", "psql", "-U", "postgres", "-d", "postgres", "-v", "ON_ERROR_STOP=1", "-t", "-A", "-c", sql],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"psql failed: {proc.stderr}\nSQL: {sql}")
    return proc.stdout.strip()


def clear_injected() -> None:
    psql(
        f"DELETE FROM api_metrics WHERE user_id = '{USER_ID}' AND inference_type = 'test_inject';"
    )


def inject_usage(meter: str, tokens: int) -> None:
    """Inject usage inside the current usage period (not merely '10 minutes ago').

    After a period-anchor reset, ``now() - 10 minutes`` can fall *before*
    ``usage_period_start`` and be ignored by allotment gating. Stamp the row
    at ``now()`` instead; rate limits must be disabled (0) during this harness.
    """
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


def subscribe(tier: str, pay_per_use: bool | None = None) -> httpx.Response:
    params: dict[str, Any] = {"tier": tier}
    if pay_per_use is not None:
        params["pay_per_use"] = str(pay_per_use).lower()
    return httpx.post(f"{API}/subscribe", headers=HEADERS, params=params, timeout=120)


def message(text: str = "Reply with exactly: ok", adapter: bool = False) -> tuple[int, str]:
    data = {
        "message": text,
        "stream": "true",
        "include_quality_metrics": "false",
        "include_usage_metrics": "true",
        "adapter": str(adapter).lower(),
    }
    with httpx.Client(timeout=180) as client:
        with client.stream(
            "POST",
            f"{API}/message/{AVATAR}",
            headers=HEADERS,
            data=data,
        ) as r:
            body = "".join(r.iter_text())
            return r.status_code, body


def upload() -> tuple[int, str]:
    if not TOKEN_FILE.exists():
        return 0, f"missing file {TOKEN_FILE}"
    with TOKEN_FILE.open("rb") as fh:
        r = httpx.post(
            f"{API}/update_avatar_identity_with_media",
            headers=HEADERS,
            data={"assistant_id": AVATAR},
            files={"files": (TOKEN_FILE.name, fh, "text/markdown")},
            timeout=180,
        )
    return r.status_code, r.text


def record(name: str, ok: bool, detail: Any) -> None:
    RESULTS.append({"name": name, "ok": ok, "detail": detail})
    flag = "PASS" if ok else "FAIL"
    print(f"\n=== {flag}: {name} ===")
    if isinstance(detail, (dict, list)):
        print(json.dumps(detail, indent=2, default=str)[:2000])
    else:
        print(str(detail)[:2000])


def parse_sse_usage(body: str) -> dict | None:
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            evt = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        if evt.get("type") == "done" and isinstance(evt.get("usage"), dict):
            return evt["usage"]
        if evt.get("type") == "usage_estimate" and isinstance(evt.get("usage"), dict):
            # keep looking for done; stash estimate
            estimate = evt["usage"]
    return locals().get("estimate")


def expect_message_ok(name: str, **kwargs) -> None:
    status, body = message(**kwargs)
    usage = parse_sse_usage(body)
    ok = status == 200 and ("assistant_token" in body or '"type": "done"' in body)
    record(name, ok, {"http": status, "usage": usage, "snippet": body[:500]})


def expect_message_402(name: str, **kwargs) -> None:
    status, body = message(**kwargs)
    ok = status == 402
    record(name, ok, {"http": status, "body": body[:800]})


def expect_upload_ok(name: str) -> None:
    status, body = upload()
    ok = status in (200, 202)
    record(name, ok, {"http": status, "body": body[:800]})


def expect_upload_402(name: str) -> None:
    status, body = upload()
    ok = status == 402
    record(name, ok, {"http": status, "body": body[:800]})


def allotment(tier_status: dict, meter: str) -> int:
    return int(tier_status["meters"][meter]["monthly_allotment"])


def main() -> int:
    clear_injected()
    st = verify()
    record(
        "baseline_status",
        True,
        {
            "status": st.get("status"),
            "tier": st.get("tier"),
            "pay_per_use_enabled": st.get("pay_per_use_enabled"),
            "meters": {
                k: {"allotment": v["monthly_allotment"], "used": v["used_to_date"]}
                for k, v in st.get("meters", {}).items()
            },
        },
    )

    # Ensure PPU off to start
    r = set_ppu(False)
    record("set_ppu_false", r.status_code == 200, {"http": r.status_code, "body": r.text[:300]})

    tier = st.get("tier")
    status = st.get("status")

    # ---------- Current tier (premium + trialing): messaging under ----------
    clear_injected()
    expect_message_ok(f"{tier}_trial_messaging_under_allotment")

    # ---------- messaging over, PPU off ----------
    inject_usage("messaging_tokens", allotment(verify(), "messaging_tokens"))
    expect_message_402(f"{tier}_trial_messaging_over_ppu_off")

    # ---------- messaging over, PPU on ----------
    r = set_ppu(True)
    record("set_ppu_true", r.status_code == 200, {"http": r.status_code, "body": r.text[:400]})
    if r.status_code == 200:
        inject_usage("messaging_tokens", allotment(verify(), "messaging_tokens"))
        expect_message_ok(f"{tier}_trial_messaging_over_ppu_on")
    else:
        record(f"{tier}_trial_messaging_over_ppu_on", False, "could not enable PPU")

    # ---------- document upload under ----------
    r = set_ppu(False)
    clear_injected()
    if "document_upload_tokens" in verify().get("meters", {}):
        expect_upload_ok(f"{tier}_trial_upload_under_allotment")
        inject_usage("document_upload_tokens", allotment(verify(), "document_upload_tokens"))
        expect_upload_402(f"{tier}_trial_upload_over_ppu_off")
        r = set_ppu(True)
        if r.status_code == 200:
            inject_usage(
                "document_upload_tokens",
                allotment(verify(), "document_upload_tokens"),
            )
            expect_upload_ok(f"{tier}_trial_upload_over_ppu_on")
        set_ppu(False)

    # ---------- adapter inference (premium) ----------
    if "adapter_inference_tokens" in verify().get("meters", {}):
        clear_injected()
        set_ppu(False)
        expect_message_ok(f"{tier}_adapter_inference_under", adapter=True)
        inject_usage(
            "adapter_inference_tokens",
            allotment(verify(), "adapter_inference_tokens"),
        )
        expect_message_402(f"{tier}_adapter_inference_over_ppu_off", adapter=True)
        r = set_ppu(True)
        if r.status_code == 200:
            inject_usage(
                "adapter_inference_tokens",
                allotment(verify(), "adapter_inference_tokens"),
            )
            expect_message_ok(f"{tier}_adapter_inference_over_ppu_on", adapter=True)
        set_ppu(False)

    # ---------- Tier switch while trialing: premium -> pro (downgrade) ----------
    clear_injected()
    before = verify()
    r = subscribe("pro")
    after = verify()
    record(
        "tier_switch_during_trial_premium_to_pro",
        r.status_code == 200,
        {
            "http": r.status_code,
            "body": r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text[:800],
            "before": {"tier": before.get("tier"), "status": before.get("status"), "period_start": before.get("usage_period_start")},
            "after": {"tier": after.get("tier"), "status": after.get("status"), "period_start": after.get("usage_period_start")},
            "anchor_retained": before.get("usage_period_start") == after.get("usage_period_start"),
        },
    )

    # ---------- Switch to free (cancel at period end while trial) ----------
    r = subscribe("free")
    record(
        "tier_switch_to_free_during_trial",
        r.status_code == 200,
        {"http": r.status_code, "body": r.text[:800], "status": verify()},
    )

    # ---------- Switch back toward pro (reactivate / change) ----------
    r = subscribe("pro")
    record(
        "subscribe_pro_after_free_request",
        r.status_code == 200,
        {"http": r.status_code, "body": r.text[:800], "status": verify()},
    )

    # ---------- Upgrade to premium during trial (anchor must not reset) ----------
    before = verify()
    r = subscribe("premium")
    after = verify()
    record(
        "tier_switch_during_trial_pro_to_premium",
        r.status_code == 200,
        {
            "http": r.status_code,
            "body": r.text[:800],
            "before_period": before.get("usage_period_start"),
            "after_period": after.get("usage_period_start"),
            "anchor_retained": before.get("usage_period_start") == after.get("usage_period_start"),
            "tier": after.get("tier"),
            "status": after.get("status"),
        },
    )

    clear_injected()
    out = Path("/tmp/metering_scenario_results.json")
    out.write_text(json.dumps(RESULTS, indent=2, default=str))
    passed = sum(1 for x in RESULTS if x["ok"])
    failed = sum(1 for x in RESULTS if not x["ok"])
    print(f"\n\nSUMMARY: {passed} passed, {failed} failed -> {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
