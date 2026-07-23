# Metering Manual-Validation — Execution Plan

Goal: manually re-run **every** metering situation against the live dev stack on
this `test` branch, using a fresh disposable account, and confirm true behavior by
HTTP status + `usage` payloads. Harness lives in `scripts/metering_manual/`.

## Deliverables (already built)
- `env.sh` — sourceable curl + SQL helpers (inspect / act / billing toggles / meter injection)
- `provision_account.py` — signup → Auth0 email-verify → provision → create avatar → prints exports
- `stripe_setup.py` — Stripe mutations: show / attach-card / detach-cards / create / cancel / end-trial
- `WALKTHROUGH.md` — per-situation mutation + test + expected result
- `PLAN.md` — this file

## Pre-flight
- [ ] **Rotate + scrub the leaked `sk-…` key** on line 33 of `_METERING_MANUAL_VALIDATION.md`.
- [ ] Set `MESSAGE_RATE_LIMIT_TOKENS_PER_WINDOW=0` and `MEDIA_UPLOAD_RATE_LIMIT_TOKENS_PER_WINDOW=0` in `.env.dev`.
- [ ] `docker compose up -d --force-recreate langgraph-api-dev` (a plain restart will NOT re-read env_file).
- [ ] Confirm dev API is up: `curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8123/ok` → `200`.
- [ ] Confirm the account you create is NOT `ADMIN_USER_ID` (`auth0|69e5e49…`) — it bypasses all enforcement.

## Setup
- [ ] `python scripts/metering_manual/provision_account.py --email "metering+$(date +%s)@example.com"`
- [ ] Paste the 5 exports → `source scripts/metering_manual/env.sh`
- [ ] `nn_tier` → expect `tier=pro, status=trialing`

## Invariants to hold every case (from code)
1. Status is the signal: `200/202` ok · `402` allotment · `403` capability · `429` rate limit.
2. `nn_inject_full <meter>` sets usage = allotment → forces the "over" path.
3. Over-allotment tests REQUIRE rate limits at 0 (the 429 window sums the injected rows).
4. `nn_ppu false` before every "over, PPU off" case (active paid tier infers PPU=on).
5. `adapter=true` needs premium or it falls back to `messaging_tokens`.
6. Reset between cases: `nn_ppu false; nn_clear`.

## Execution matrix (tick as verified)

### Free (`stripe_setup.py cancel` → tier=free)
- [ ] F1 messaging under → 200, allotment 200,000, remaining drops ~19,315
- [ ] F2 messaging over, PPU off → 402 free-tier messaging exhausted
- [ ] F3 messaging over, PPU on (card + `nn_ppu true`) → 200, remaining 0
- [ ] F4 upload → 403 "'free' tier does not permit this action. Upgrade to the pro tier."

### Pro active (`attach-card` → `create --tier pro`)
- [ ] P1 messaging under → 200, allotment 5,000,000
- [ ] P2 messaging over, PPU off → 402
- [ ] P3 messaging over, PPU on → 200, remaining 0
- [ ] P4 upload under → 202 queued, estimated_tokens ≈ 6
- [ ] P5 upload over, PPU off → 402
- [ ] P6 upload over, PPU on → 202

### Premium active (`create --tier premium`)
- [ ] Prem1 messaging under → 200, allotment 20,000,000
- [ ] Prem2 messaging over, PPU off → 402
- [ ] Prem3 messaging over, PPU on → 200, remaining 0
- [ ] Prem4 upload under → 202
- [ ] Prem5 upload over, PPU off → 402
- [ ] Prem6 upload over, PPU on → 202
- [ ] **Prem7 adapter under** (`nn_msg true`) → 200, meter `adapter_inference_tokens`, allotment 10,000,000  ← open in ledger
- [ ] **Prem8 adapter over, PPU off** → 402  ← open in ledger
- [ ] **Prem9 adapter over, PPU on** → 200, remaining 0  ← open in ledger
- [ ] Prem10 adapter training → N/A (no HTTP route; blocked by design)

### Free-trial / lifecycle
- [ ] T1 trial messaging under → 200, premium allotment 20M
- [ ] T2 trial over, PPU off → 402
- [ ] T3 trial over, PPU on (card) → 200; without card `nn_ppu true` → 402
- [ ] T4 trial ends no payment (`detach-cards; cancel`) → free, canceled; 200k enforced
- [ ] T5 trial ends with payment (`attach-card; end-trial`) → trialing→active, stays paid

### Tier switch
- [ ] S1 downgrade premium→pro → 200 change_tier, still premium, usage retained
- [ ] S2 downgrade to free → 200, still premium until boundary
- [ ] S3 upgrade pro→premium → 200, anchor reset, usage → 0
- [ ] S4 same tier → 200 no_change_required (or reactivate if cancel pending)

### Delete & re-signup
- [ ] delete keeps+tags customer, subs cancel-at-period-end
- [ ] re-signup same period → same sub adopted, no new invoice, original trial_end
- [ ] re-signup after lapse → free, no second trial

### Closed gaps
- [ ] GAP2 upgrade then immediate `nn_tier` → new tier, no restart
- [ ] GAP3 `create --tier pro` outside Checkout → tier syncs from `created` event
- [ ] GAP4 mid-trial premium→pro downgrade → messaging allotment stays 20M until trial_end

## Teardown
- [ ] `nn_ppu false; nn_clear`
- [ ] `stripe_setup.py cancel` (or `DELETE /delete_user`) on the disposable account
- [ ] Restore rate limits to `30000` / `90000` and `--force-recreate langgraph-api-dev`

## Ledger to update on completion
- `TEST_SITUATIONS.md` — flip Prem7/8/9 (adapter inference) to `[x]` once verified.
