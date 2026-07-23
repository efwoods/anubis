# Metering — Manual Validation Walkthrough (current `test` branch)

Runnable, situation-by-situation. This corrects `_METERING_MANUAL_VALIDATION.md`
for the environment as it actually is on this branch and gives you the exact
mutation + test + expected result for every case, using the helpers in this
directory.

## What changed vs. the old playbook

| Old playbook says | Reality on this branch |
|---|---|
| Base URL `http://localhost:8900` | dev API is **`http://localhost:8123`** (`:8900` is dead) |
| Avatar `c206e1a4-…` | does not exist here — you **create your own** avatar |
| Fixture `data/shivon_zilis/test_tokens_1_tokens.md` | missing — `nn_fixture` writes a ~6-token file |
| A specific test user | you create a **fresh** account (`provision_account.py`) |

⚠️ **Rotate the leaked key.** `_METERING_MANUAL_VALIDATION.md` line 33 has a live
`sk-…` credential embedded as a code-fence label. Rotate it and scrub the file.

## Ground rules baked into these scripts (verified in code)

1. **Status is the signal:** `200/202` success, `402` allotment exhausted,
   `403` capability not in tier, `429` token rate limit.
2. **Injecting the allotment forces "over".** Enforcement blocks when
   `month_to_date_usage + this_request_estimate >= allotment`
   (`gating.exhausted_allotment_block_reason`). Injecting exactly the allotment
   → already `>=` → blocked (unless PPU). `nn_inject_full <meter>` does this.
3. **The rate-limit window sums the SAME injected rows.** `enforce_token_rate_limit`
   sums `api_metrics` over a rolling window for the messaging+adapter meters, so a
   multi-million injected allotment trips **429 before** the intended **402**.
   → **Set both rate limits to 0 before every over-allotment test** (below).
4. **Active paid tier infers PPU = on.** `resolve_pay_per_use_enabled` returns
   true when `status=="active"` unless `app_metadata.pay_per_use_enabled` is
   explicitly set. → Always `nn_ppu false` before an "over, PPU off" case.
   `trialing` does **not** infer PPU.
5. **`adapter=true` silently falls back** to `messaging_tokens` unless the user is
   **premium** (`resolve_use_adapter_inference`). Adapter meter only bites on premium.
6. **Don't use the admin account.** `ADMIN_USER_ID` (`auth0|69e5e49…`) bypasses
   ALL enforcement and metering. `provision_account.py` warns if you hit it.
7. **Period = calendar month** (`USAGE_PERIOD_DAYS=0`); injected rows use `now()`.

---

## 0. Prerequisites

### 0a. Disable rate limits (required for all over-allotment tests)
Edit `.env.dev`:
```
MESSAGE_RATE_LIMIT_TOKENS_PER_WINDOW=0
MEDIA_UPLOAD_RATE_LIMIT_TOKENS_PER_WINDOW=0
```
Recreate the dev container so it re-reads the env file (a plain `restart` will
**not** pick up env_file changes):
```bash
docker compose up -d --force-recreate langgraph-api-dev
```
Restore to `30000` / `90000` when finished (same recreate).

### 0b. Create the fresh, verified, provisioned account
```bash
python scripts/metering_manual/provision_account.py \
  --email "metering+$(date +%s)@example.com" --name "Metering Test"
```
This signs up, marks the email verified via the Auth0 Management API (the dev
tenant sends no email), fires provisioning on the first authenticated call, and
creates an avatar. It prints five `export` lines. Paste them, then:
```bash
source scripts/metering_manual/env.sh
nn_tier      # expect: tier=pro, status=trialing (the signup free trial)
```

> A brand-new account lands on **pro / trialing** (the 30-day pro trial). That
> covers every pro-tier and trial situation with **no Stripe surgery**. Only the
> paid-**active** and **premium** situations need a card / a created subscription.

### 0c. The webhook forwarder must be running

```bash
docker compose up -d stripe-cli
docker compose logs -f stripe-cli    # each forwarded event prints here
```
Without this container no subscription change reaches the API, so
`app_metadata.subscription_status` keeps pointing at the PREVIOUS subscription
and `check_subscription_status` re-reads that stale (usually canceled) one on
every call. The forwarded event list lives in
`scripts/stripe_listen_entrypoint.sh` and **must** include
`customer.subscription.created`: the API creates subscriptions server-side (the
pro signup trial, the free-tier billing vehicle, every `stripe_setup.py create`),
and those emit only `.created`.

### 0d. Billing state is TWO things, not one

| What it is | Where it lives | What it decides |
|---|---|---|
| Stripe subscription | Stripe | `status` and `tier` |
| `app_metadata.trial_context` | Auth0 | which **allotments** apply |

While `now < trial_context.trial_end`, `resolve_effective_monthly_allotment`
grants the trial tier's allotment as a floor over the current tier's, and
`/verify_subscription_status` lists every meter the trial tier grants — including
meters the current tier does not have. **No product code ever clears
`trial_context`**, so canceling in Stripe leaves the trial grant behind: the
account reports `tier: free` while advertising the pro trial's 5M messaging /
10M document-upload allotments. That hybrid is a real state, not a bug in the
test, and it is the usual reason "the free tier still shows document uploads".

Drive both halves at once with named scenarios (each polls
`/verify_subscription_status` until the API agrees and prints the result):

```bash
nn_scenario free                 # active free tier, no trial → messaging 200k ONLY
nn_scenario free-expired-trial   # free tier, trial_context back-dated 1 day
nn_scenario canceled             # no live subscription, no trial → canceled/free
nn_scenario canceled-in-trial    # canceled DURING the trial → free tier at pro allotments
nn_scenario premium-trial        # premium, status=trialing (all four meters)
nn_scenario premium-active       # premium, status=active, card attached, no trial
nn_state                         # Stripe subs + trial_context + whether its window is open
```

Finer-grained control over just the Auth0 half:
```bash
python scripts/metering_manual/stripe_setup.py trial show
python scripts/metering_manual/stripe_setup.py trial clear
python scripts/metering_manual/stripe_setup.py trial expire
python scripts/metering_manual/stripe_setup.py trial grant --tier premium --days 14
```

> Anything mutating Auth0 from outside the API must also evict the API's
> five-minute `_api_key_cache`, or the next call serves the previous billing
> metadata. Every command above does that automatically by calling
> `POST /set_pay_per_use?enabled=false` (an API-side metadata write, which
> evicts) — so pay-per-use is left OFF after a scenario; turn it back on with
> `nn_ppu true` when a test needs it.

---

## 1. Free tier

Move the account to the **plain** free tier — canceling alone is not enough,
because the trial grant outlives the subscription (§0d):
```bash
nn_scenario free     # cancels live subs, creates the $0 free subscription,
                     # clears trial_context, and verifies
nn_tier              # expect: tier=free, status=active, meters=[messaging_tokens]
```
For the canceled variant use `nn_scenario canceled` (tier=free, status=canceled).

| # | Situation | Commands | Expected |
|---|---|---|---|
| F1 | messaging under | `nn_ppu false; nn_clear; nn_msg` | `HTTP 200`; usage frame `meter=messaging_tokens`, `monthly_allotment=200000`, `remaining` drops ~19,315 |
| F2 | messaging over, PPU off | `nn_ppu false; nn_inject_full messaging_tokens; nn_msg` | `HTTP 402` … *"free-tier monthly allotment of 200,000 messaging tokens is exhausted"* |
| F3 | messaging over, PPU on | attach card (below); `nn_ppu true; nn_inject_full messaging_tokens; nn_msg` | `HTTP 200`; `pay_per_use_enabled=true`; `remaining=0` |
| F4 | upload blocked on free | `nn_upload` | `HTTP 403` … *"'free' tier does not permit this action. Upgrade to the pro tier."* |

F3 needs a payment method even on free (the free tier's $0 subscription qualifies,
or just attach a card):
```bash
python scripts/metering_manual/stripe_setup.py attach-card
nn_ppu true                       # 200 (with card) — 402 if no card on file
nn_inject_full messaging_tokens
nn_msg                            # 200, remaining 0
nn_ppu false; nn_clear            # reset
```

---

## 2. Pro tier (active)

Get to pro **active** (paid, so PPU-on works). A fresh account is pro *trialing*;
for the active-tier matrix create a paid subscription with a card:
```bash
python scripts/metering_manual/stripe_setup.py attach-card
python scripts/metering_manual/stripe_setup.py create --tier pro
nn_tier      # poll until tier=pro, status=active
```

| # | Situation | Commands | Expected |
|---|---|---|---|
| P1 | messaging under | `nn_ppu false; nn_clear; nn_msg` | `200`; `monthly_allotment=5000000`; `remaining ≈ 4,980,685` |
| P2 | messaging over, PPU off | `nn_ppu false; nn_inject_full messaging_tokens; nn_msg` | `402` pro-tier messaging exhausted |
| P3 | messaging over, PPU on | `nn_ppu true; nn_inject_full messaging_tokens; nn_msg` | `200`; `remaining=0` |
| P4 | upload under | `nn_ppu false; nn_clear; nn_upload` | `202` `status=queued`, `estimated_tokens ≈ 6` |
| P5 | upload over, PPU off | `nn_ppu false; nn_inject_full document_upload_tokens; nn_upload` | `402` document-upload exhausted |
| P6 | upload over, PPU on | `nn_ppu true; nn_inject_full document_upload_tokens; nn_upload` | `202` queued |

Reset after: `nn_ppu false; nn_clear`.

---

## 3. Premium tier (active)

```bash
python scripts/metering_manual/stripe_setup.py create --tier premium   # card already attached
nn_tier      # poll until tier=premium, status=active
nn_verify | jq '.meters | keys'   # expect messaging_/document_upload_/adapter_inference_ (+ adapter_training in trial view)
```

| # | Situation | Commands | Expected |
|---|---|---|---|
| Prem1 | messaging under | `nn_ppu false; nn_clear; nn_msg` | `200`; `monthly_allotment=20000000`; `remaining ≈ 19,980,685` |
| Prem2 | messaging over, PPU off | `nn_ppu false; nn_inject_full messaging_tokens; nn_msg` | `402` premium messaging exhausted |
| Prem3 | messaging over, PPU on | `nn_ppu true; nn_inject_full messaging_tokens; nn_msg` | `200`; `remaining=0` |
| Prem4 | upload under | `nn_ppu false; nn_clear; nn_upload` | `202` |
| Prem5 | upload over, PPU off | `nn_ppu false; nn_inject_full document_upload_tokens; nn_upload` | `402` document-upload exhausted |
| Prem6 | upload over, PPU on | `nn_ppu true; nn_inject_full document_upload_tokens; nn_upload` | `202` |
| **Prem7** | **adapter under** | `nn_ppu false; nn_clear; nn_msg true` | `200`; usage frame `meter=adapter_inference_tokens`, `monthly_allotment=10000000` |
| **Prem8** | **adapter over, PPU off** | `nn_ppu false; nn_inject_full adapter_inference_tokens; nn_msg true` | `402` adapter-inference exhausted |
| **Prem9** | **adapter over, PPU on** | `nn_ppu true; nn_inject_full adapter_inference_tokens; nn_msg true` | `200`; `remaining=0` |
| Prem10 | adapter training | — | **Blocked by design** — no HTTP training route exists |

Prem7–9 are the three still-open `[ ]` in `TEST_SITUATIONS.md`. Confirm the usage
frame's `meter` is `adapter_inference_tokens` (proves adapter routing, not the
messaging fallback). Reset after: `nn_ppu false; nn_clear`.

---

## 4. Free-trial / lifecycle

A fresh account (§0b) is pro *trialing*; premium-trial can be made with
`create --tier premium --trial-days 14`.

| # | Situation | Commands | Expected |
|---|---|---|---|
| T1 | trial messaging under | on a trialing premium sub: `nn_clear; nn_msg` | `200`; premium allotment `20000000` |
| T2 | trial over, PPU off | `nn_ppu false; nn_inject_full messaging_tokens; nn_msg` | `402` premium messaging exhausted |
| T3 | trial over, PPU on | `attach-card; nn_ppu true; nn_inject_full messaging_tokens; nn_msg` | `200`. **Without** a card, `nn_ppu true` → `402` from `/set_pay_per_use` |
| T4 | trial ends, no payment | `detach-cards; stripe_setup.py cancel` | `nn_tier`: `tier=free`, `status=canceled`; free 200k enforced |
| T5 | trial ends, with payment | `attach-card; stripe_setup.py end-trial` | `trialing → active`; stays on the paid tier |

---

## 5. Tier switch / usage retain-clear

| # | Situation | Call | Expected |
|---|---|---|---|
| S1 | downgrade premium→pro | `nn_subscribe pro` | `200` `change_tier`, *"switch at the period boundary"*; still premium, usage retained |
| S2 | downgrade to free | `nn_subscribe free` | `200` *"end at the period boundary; drop to free"*; still premium until boundary |
| S3 | upgrade pro→premium | `nn_subscribe premium` | `200` *"changed to the premium tier"*; anchor reset, usage cleared → 0 |
| S4 | same tier again | `nn_subscribe <current>` | `200` `no_change_required` (or `reactivate` if a cancel is pending) |

For S3 verify the reset explicitly:
```bash
nn_verify | jq '{start:.usage_period_start, used:.meters.messaging_tokens.used_to_date}'
nn_subscribe premium
nn_verify | jq '{start:.usage_period_start, used:.meters.messaging_tokens.used_to_date}'  # start changed, used=0
```

---

## 6. Account deletion & re-signup

`DELETE /delete_user` keeps + tags the Stripe customer and sets live subs to
cancel-at-period-end; re-signup within the pay period re-adopts the same
subscription (no new charge, original trial_end kept); a lapsed re-signup enrolls
free with no second trial. Live-drive it, or lean on the automated coverage:
```bash
# delete
curl -s -X DELETE -H "API-KEY: $NN_API_KEY" "$NN_API/delete_user" | jq .
# re-signup with the SAME email within the period → same sub adopted, no new invoice
python scripts/metering_manual/provision_account.py --email "<same-email>" --name "Metering Test"
python scripts/metering_manual/stripe_setup.py show   # one sub, cancel_at_period_end cleared, no amount_paid>0 invoice
```
Automated: `scripts/e2e_billing/scenario_delete_and_resignup.py` (Stripe test
clocks, 8 checks) + `tests/unit_tests/test_subscription_lifecycle.py` and
`test_initial_subscription_provisioning.py` (18 tests).

---

## 7. Verifying the closed gaps (GAP 2 / 3 / 4)

- **GAP 2 — no stale tier after a webhook.** `nn_subscribe premium` (upgrade),
  then **immediately** `nn_tier` on the very next request — the new tier/meters
  must appear with **no container restart** (the API-key cache is evicted on every
  subscription-status write).
- **GAP 3 — `customer.subscription.created` synced.** `stripe_setup.py create
  --tier pro` (a sub made outside Checkout), then poll `nn_tier` — the tier must
  sync from the `created` event without waiting for a later `updated`.
- **GAP 4 — trial allotment floor.** While premium *trialing*
  (`create --tier premium --trial-days 14`), `nn_subscribe pro` (mid-trial
  downgrade), then `nn_verify | jq '.meters.messaging_tokens.monthly_allotment'`
  → must stay **20000000** (premium floor) until `trial_end`, not drop to pro's
  `5000000`.

---

## Reference: raw curl / SQL (what the helpers run)

```bash
# verify
curl -s -H "API-KEY: $NN_API_KEY" "$NN_API/verify_subscription_status" | jq .

# message (adapter=true for adapter inference)
curl -s -N -H "API-KEY: $NN_API_KEY" \
  --data-urlencode "message=Reply with exactly: ok" \
  --data stream=true --data include_usage_metrics=true --data adapter=false \
  "$NN_API/message/$NN_AVATAR_ID"

# upload
curl -s -H "API-KEY: $NN_API_KEY" -F "assistant_id=$NN_AVATAR_ID" \
  -F "files=@/tmp/nn_upload_fixture.md;type=text/markdown" \
  "$NN_API/update_avatar_identity_with_media"

# pay-per-use / subscribe
curl -s -X POST -H "API-KEY: $NN_API_KEY" "$NN_API/set_pay_per_use?enabled=true"
curl -s -X POST -H "API-KEY: $NN_API_KEY" "$NN_API/subscribe?tier=premium"

# inject usage so a meter is at its allotment (forces the over-allotment path)
docker exec -i postgres16 psql -U postgres -d postgres -c "
  INSERT INTO api_metrics (id, created_at, user_id, stripe_customer_id, inference_type,
    prompt_tokens, completion_tokens, total_tokens, cost_usd, latency_ms, meter_event_name)
  VALUES (gen_random_uuid(), now(), '$NN_USER_ID', '$NN_CUSTOMER', 'test_inject',
    0, 0, 20000000, 0, 0, 'messaging_tokens');"

# clear injected rows only (nn_clear) — reports DELETE 0 when the usage came from
# real /message and /update_avatar_identity_with_media calls, which is why
# used_to_date can look impossible to reset
docker exec -i postgres16 psql -U postgres -d postgres -c "
  DELETE FROM api_metrics
  WHERE inference_type='test_inject'
    AND COALESCE(stripe_customer_id, user_id) IN ('$NN_CUSTOMER', '$NN_USER_ID');"

# clear ALL usage for the account, genuine rows included (nn_clear_all) — the only
# in-period way to put used_to_date back to zero, since used_to_date is
# SUM(total_tokens) over the period window, not a stored counter
docker exec -i postgres16 psql -U postgres -d postgres -c "
  DELETE FROM api_metrics
  WHERE COALESCE(stripe_customer_id, user_id) IN ('$NN_CUSTOMER', '$NN_USER_ID');"
```

> A new subscription also zeroes `used_to_date` without deleting anything: the
> usage window follows the Stripe billing period, so every `nn_scenario` that
> creates a subscription starts a fresh period and leaves earlier rows behind it.

Meter names: `messaging_tokens`, `document_upload_tokens`,
`adapter_inference_tokens`, `adapter_training_units`.
