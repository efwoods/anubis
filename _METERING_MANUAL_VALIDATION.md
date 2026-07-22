# Metering — Manual Validation Playbook

A self-contained, runnable checklist to recreate every metering scenario against
the local dev stack and confirm the results by HTTP status and `usage` payloads.
The authoritative checkmark ledger is `TEST_SITUATIONS.md`; this file is the
how-to. Automated equivalents live in `tests/unit_tests/test_billing_*.py`,
`scripts/e2e_billing/` (Stripe test clocks), and `scripts/manual_metering_*.py`.

## Setup

| Item | Value |
|---|---|
| Base URL | `http://localhost:8900` |
| Header | `API-KEY: <your key>` |
| Avatar | `c206e1a4-5cdf-4f3b-8ceb-9f23b605ddb1` |
| Auth0 user | `auth0|6a5fb6cd601f497f56f5aa37` |
| Stripe customer | `cus_UvHevUUQ1nJnE1` |
| Upload fixture | `data/shivon_zilis/test_tokens_1_tokens.md` (~6 estimated tokens) |
| Typical message cost | ~19,315 total tokens (prompt ≈19,311 / completion ≈4) |

Allotments (catalog, `src/anubis/utils/billing/tiers.py`):

| Tier | messaging | document upload | adapter inference | adapter training |
|---|---|---|---|---|
| free | 200,000 | — | — | — |
| pro | 5,000,000 | 10,000,000 | — | — |
| premium | 20,000,000 | 40,000,000 | 10,000,000 | 5 |

### Before over-allotment tests
Set both caps to 0 and recreate the API container, otherwise a second message
within 60s trips the 429 rate limit (~19k prior + ~22k estimate vs the 30k cap):

```sk-rhZGqpwMTIARI_Lzm0GsPYNhdkL0GEiAj8FbvCMlP18
MESSAGE_RATE_LIMIT_TOKENS_PER_WINDOW=0
MEDIA_UPLOAD_RATE_LIMIT_TOKENS_PER_WINDOW=0
```

Restore to 30k / 90k afterward.

### Force over-allotment (replace injected meter + tokens)


Meter names: `messaging_tokens`, `document_upload_tokens`, `adapter_inference_tokens`.

### Useful curls

Stale tier after a webhook should no longer happen (the API-key cache is now
evicted on every subscription-status write). If a tier ever still looks stale,
restart `langgraph-api-dev` (cache TTL is 300s).

## Expected results — HTTP status is the pass/fail signal

### Free tier
| # | Situation | How | Expected |
|---|---|---|---|
| F1 | Messaging under | clear inject, PPU off, message | 200 SSE; allotment 200,000; `used_to_date` rises ~19,315 |
| F2 | Messaging over, PPU off | inject `messaging_tokens`=200000; message | 402 "free-tier monthly allotment of 200,000 messaging tokens is exhausted" |
| F3 | Messaging over, PPU on | card on file; `set_pay_per_use?enabled=true`; inject 200000; message | 200; `pay_per_use_enabled=true`; remaining 0 |
| F4 | Upload on free | upload fixture | 403 "'free' tier does not permit this action. Upgrade to the pro tier." |

### Pro tier (active)
| # | Situation | How | Expected |
|---|---|---|---|
| P1 | Messaging under | PPU off; message | 200; allotment 5,000,000; remaining ≈4,980,685 |
| P2 | Messaging over, PPU off | inject `messaging_tokens`=5000000 | 402 pro-tier messaging exhausted |
| P3 | Messaging over, PPU on | PPU on + inject 5M | 200; remaining 0 |
| P4 | Upload under | upload fixture | 202 `status=queued`, `estimated_tokens`≈6 |
| P5 | Upload over, PPU off | inject `document_upload_tokens`=10000000 | 402 document-upload exhausted |
| P6 | Upload over, PPU on | PPU on + inject 10M | 202 queued |

### Premium tier (active)
| # | Situation | How | Expected |
|---|---|---|---|
| Prem1 | Messaging under | message | 200; allotment 20,000,000; remaining ≈19,980,685 |
| Prem2 | Messaging over, PPU off | inject 20M messaging | 402 messaging exhausted |
| Prem3 | Messaging over, PPU on | PPU on + inject 20M | 200; remaining 0 |
| Prem4 | Upload under | upload | 202 |
| Prem5 | Upload over, PPU off | inject 40M document | 402 document-upload exhausted |
| Prem6 | Upload over, PPU on | PPU on + inject 40M | 202 |
| Prem7 | Adapter under | `-F adapter=true` | 200; meter `adapter_inference_tokens`, allotment 10,000,000 |
| Prem8 | Adapter over, PPU off | inject `adapter_inference_tokens`=10M | 402 adapter-inference exhausted |
| Prem9 | Adapter over, PPU on | PPU on + inject 10M | 200; remaining 0 |
| Prem10 | Adapter training | — | Blocked — no HTTP training route (known limitation) |

### Free-trial / lifecycle
| # | Situation | How | Expected |
|---|---|---|---|
| T1 | Trial messaging under | while `status=trialing`, tier=premium | 200; allotment 20M |
| T2 | Trial over, PPU off | inject 20M | 402 premium messaging exhausted |
| T3 | Trial over, PPU on | card + `set_pay_per_use=true` | 200 (as Prem3). Without card: 402 from `/set_pay_per_use` |
| T4 | Trial ends without payment | cancel sub in Stripe (or cancel at end, no PM) | `/verify`: tier=free, status=canceled; free 200k enforced |
| T5 | Trial ends with payment | attach test card while trialing | trialing → active; stays on paid tier |

### Tier switch / usage retain-clear
| # | Situation | Call | Expected |
|---|---|---|---|
| S1 | Downgrade premium→pro | `POST /subscribe?tier=pro` | 200 `change_tier` "switch at the period boundary"; still premium, usage retained |
| S2 | Downgrade to free | `POST /subscribe?tier=free` | 200 "end at the period boundary; drop to free"; still premium until boundary |
| S3 | Upgrade pro→premium | `POST /subscribe?tier=premium` | 200 "changed to the premium tier"; anchor reset, usage cleared → 0 |
| S4 | Same tier again | `POST /subscribe?tier=<current>` | 200 `no_change_required` (or `reactivate` if a cancel is pending) |

## Account deletion and re-signup

The Auth0 dev tenant has **no email provider configured**, so verification
emails never deliver (they only reach tenant admins). After each `/signup`,
mark the account verified directly through the Management API before
authenticating — provisioning (the pro free trial) fires on the first
authenticated request after verification:

```python
# patch email_verified:true for the just-created user via
# GET  /api/v2/users-by-email?email=<email>   then
# PATCH /api/v2/users/<user_id>  {"email_verified": true}
```

Steps and expectations:
1. `/signup` → mark verified → `GET /verify_subscription_status` shows
   `tier=pro`, `status=trialing`; the Stripe customer gains
   `metadata.neural_nexus_trial_used=true`.
2. `DELETE /delete_user` → the Stripe customer is KEPT and tagged
   (`deleted_auth0_user_id`, `account_deleted_at`); every live subscription is
   set to `cancel_at_period_end=true` (never billed for another period).
3. **Re-signup within the same pay period** → adopts the SAME subscription id,
   clears `cancel_at_period_end` and any pending schedule, retains the ORIGINAL
   `trial_end` (never restarted/extended), rebuilds `usage_period_anchor` from
   the subscription's period start, clears the deleted-account markers, and
   produces NO new charge.
4. **Re-signup after the trial or paid period lapsed** → enrolls the FREE tier
   (active, non-trialing new subscription); no second free trial ever. A paid
   tier is regained only by re-selecting one through `POST /subscribe` Checkout,
   like any new user.

**Policy (confirmed):** a lapsed re-signup with a card on file still enrolls
free + Checkout re-selection — the previous paid tier is NOT auto-restarted.

Automated coverage: `scripts/e2e_billing/scenario_delete_and_resignup.py`
(Stripe test clocks, 8 checks) and the 18 unit tests in
`test_subscription_lifecycle.py` + `test_initial_subscription_provisioning.py`.

## Verifying the closed gaps (2026-07-20)

Three gaps that were open in `TEST_SITUATIONS.md` are now fixed; verify each:

- **GAP 2 — no stale tier after a webhook.** Change tier via `POST /subscribe`
  (which emits `customer.subscription.updated`), then immediately
  `GET /verify_subscription_status` on the very next request. The new tier must
  appear with NO container restart (the API-key cache is evicted on every
  subscription-status write). Meter names / enforcement should match the new
  tier at once.
- **GAP 3 — `customer.subscription.created` synced.** Create a subscription
  outside Checkout (server-side or Dashboard) and confirm the tier/status
  reaches Auth0 without waiting for a later `updated` event.
- **GAP 4 — trial allotment floor.** While `status=trialing` on premium,
  `POST /subscribe?tier=pro` (mid-trial downgrade), then
  `GET /verify_subscription_status`. Messaging `monthly_allotment` must remain
  20,000,000 (the premium trial floor) until `trial_end`, not drop to the pro
  5,000,000. After `trial_end` it falls to the pro allotment. Covered by unit
  tests `test_trialing_premium_then_downgrade_to_pro_keeps_the_premium_floor`
  and siblings in `test_billing_enforcement_and_periods.py`.

## Validation checklist (what to look at)
- HTTP status: 200/202 success, 402 allotment, 403 capability, 429 rate limit.
- SSE `done.usage`: `meter`, `tier`, `monthly_allotment`, `used_to_date`,
  `remaining`, `pay_per_use_enabled`.
- `GET /verify_subscription_status`: per-meter allotment/usage + period bounds
  before and after tier changes.
- Stripe Dashboard (test): meter events after a successful overage with PPU on.
