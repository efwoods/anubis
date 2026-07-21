
# Notes: FREE TIER USAGE IS FREE FOR THE MONTH AS WELL AS THE SUBSCRIPTION FOR PRO TIER (PRO TIER FREE USAGE IS ALLOTTED FOR PREMIUM AND FREE AND TRANSFERS IF THE USER CHANGES THE SUBSCRIPTION TO PREMIUM FROM PRO OR TO FREE FROM PRO)

# Test account (2026-07-16)
# - email: epitome_75_springs@icloud.com
# - avatar_id: c206e1a4-5cdf-4f3b-8ceb-9f23b605ddb1
# - API: http://localhost:8900 (f-metering docker-compose, PORT=8900)
# - Over-allotment forced via temporary `api_metrics` rows (`inference_type=test_inject`)
# - Rate limits temporarily set to 0 during allotment tests (system prompt ~19k tokens trips 30k/60s)
# - Stripe test card `pm_card_visa` attached for pay-per-use scenarios

# Test Situations:
Free tier:
messaging: 
 - [x] under usage allotment messaging succeeds, metering increments
 - [x] over usage allotment, pay-per-usage disabled, messaging does not succeed (HTTP 402)
 - [x] over usage allotment, pay-per-usage enabled, metering increments, pay-per-usage is charged per rate for free-tier
 - [x] document upload capability blocked on free (HTTP 403: upgrade to pro)

Pro tier:
messaging: 
 - [x] under usage allotment messaging succeeds, metering increments
 - [x] over usage allotment, pay-per-usage disabled, messaging does not succeed (HTTP 402)
 - [x] over usage allotment, pay-per-usage enabled, metering increments, pay-per-usage is charged per rate for pro-tier
Document token uploads:
 - [x] under usage allotment upload succeeds, metering increments (HTTP 202)
 - [x] over usage allotment, pay-per-usage disabled, document upload does not succeed (HTTP 402)
 - [x] over usage allotment, pay-per-usage enabled, metering increments, pay-per-usage is charged per rate for pro-tier

Pro tier (Free-Trial subscription is free; free usage to point of pay-per-usage for tokens for the period of the free-trial):
messaging: 
 - [x] under usage allotment messaging succeeds, metering increments (tested while originally `trialing` on premium-after-upgrade from pro trial)
 - [x] over usage allotment, pay-per-usage disabled, messaging does not succeed (HTTP 402)
 - [x] over usage allotment, pay-per-usage enabled, metering increments (requires card + POST /set_pay_per_use)
Document token uploads:
 - [x] under usage allotment upload succeeds, metering increments
 - [x] over usage allotment, pay-per-usage disabled, document upload does not succeed (HTTP 402)
 - [x] over usage allotment, pay-per-usage enabled, metering increments


Premium Tier:
messaging: 
 - [x] under usage allotment messaging succeeds, metering increments
 - [x] over usage allotment, pay-per-usage disabled, messaging does not succeed (HTTP 402)
 - [x] over usage allotment, pay-per-usage enabled, metering increments, pay-per-usage is charged per rate for premium-tier

Document token uploads:
 - [x] under usage allotment upload succeeds, metering increments
 - [x] over usage allotment, pay-per-usage disabled, document upload does not succeed (HTTP 402)
 - [x] over usage allotment, pay-per-usage enabled, metering increments, pay-per-usage is charged per rate for premium-tier

Adapter Training token allotment: 
 - [ ] under usage allotment adapter training succeeds, metering increments — **BLOCKED: no HTTP training endpoint in API**
 - [ ] over usage allotment, pay-per-usage disabled, adapter training does not succeed — **BLOCKED: no endpoint**
 - [ ] over usage allotment, pay-per-usage enabled, metering increments — **BLOCKED: no endpoint**

Adapter Inference:
 - [ ] under usage allotment adapter inference messaging succeeds, metering increments (`adapter=true`)
 - [ ] over usage allotment, pay-per-usage disabled, adapter inference messaging does not succeed (HTTP 402)
 - [ ] over usage allotment, pay-per-usage enabled, metering increments, pay-per-usage is charged per rate for premium-tier


# Additional situations to test:

## Usage
 - [x] Make certain usage resets at the end of the period / on upgrade — **PASS**: `POST /subscribe?tier=premium` from pro reset `usage_period_anchor` and cleared `used_to_date` (38630 → 0)
 - [x] Make certain usage … retained when downgrading — **PASS**: scheduled premium→pro kept tier=premium + same used_to_date until period boundary
 - [x] Make certain usage … cleared when upgrading — **PASS** (see above)

## Free Trial
 - [x] pro/premium with free trial moves to free tier if the trial ends without payment — **PASS**: canceled subscription → `tier=free`, `status=canceled`; messaging still gated on free allotment
 - [x] free trial continues to paid tier if payment information present — **PASS**: attaching test card converted `trialing` → `active` premium (Stripe billed / trial ended with PM)

## Tier switching during trial / paid
 - [x] `POST /subscribe?tier=pro` while on premium schedules downgrade (usage retained, still premium until boundary)
 - [x] `POST /subscribe?tier=premium` from pro upgrades immediately and clears usage window
 - [x] Downgrade schedule Stripe API: fixed `phases[iterations]` → `duration` (was HTTP 502 `Received unknown parameter: phases[iterations]`)

## Account deletion and re-signup (added 2026-07-18; verified 2026-07-18)
Unit coverage: `tests/unit_tests/test_subscription_lifecycle.py`, `tests/unit_tests/test_initial_subscription_provisioning.py` (18 passed). Stripe-side coverage: `scripts/e2e_billing/scenario_delete_and_resignup.py` (test clocks; all 8 checks pass). Live coverage: full signup→verify→provision→delete→re-signup lifecycle run against the dev server on :8900 with a disposable account (20/20 checks) — email verification applied via the Auth0 Management API because the tenant cannot send verification emails.
 - [x] `DELETE /delete_user` sets every live subscription to cancel at period end and tags the kept Stripe customer (deleted_auth0_user_id, account_deleted_at); the user is never billed for another period — **PASS** (Stripe clocks + live)
 - [x] re-signup with the same email within the same pay period adopts the same subscription, clears cancel_at_period_end (and any pending downgrade schedule), produces NO new invoice (no double charge), and rebuilds `usage_period_anchor` from the subscription's period start — **PASS** (Stripe clocks + live; anchor matched the subscription period start)
 - [x] re-signup during a still-running free trial retains the trial to the ORIGINAL trial_end (never restarted, never extended) — **PASS** (live: trial_end unchanged across delete/re-signup; trial_context carries the original trial_end)
 - [x] re-signup after the trial window lapsed enrolls the free tier; no second free trial ever (`neural_nexus_trial_used` on the customer survives deletion; Checkout also refuses a trial via `resolve_checkout_trial_period_days`) — **PASS** (Stripe clocks for the real time lapse; live with a simulated lapse → free tier, active, not trialing)
 - [x] re-signup after the paid period lapsed enrolls the free tier; regaining a paid tier requires the user to choose one through `POST /subscribe` Checkout like any new user — no paid subscription is auto-created — **PASS** (Stripe clocks: canceled at the boundary with exactly one charged invoice; live free re-enrollment is a NEW free-tier subscription)
 - [x] reattach on re-signup clears the deleted_auth0_user_id / account_deleted_at customer metadata markers — **PASS** (live)

Notes from the 2026-07-18 verification run:
 - A subscription canceling at the period boundary emits a closing `subscription_cycle` invoice with a **zero** total (settles metered usage); the double-charge assertion in `scenario_delete_and_resignup.py` now counts invoices with `amount_paid > 0` instead of raw invoice count.
 - **Signup bug fixed**: `POST /signup` without a `name` sent `"name": null` to Auth0 (SignupRequest.name defaults to None but signup_user only skipped the field when `name != ""`), producing a 400 mislabeled as "Invalid Password". `src/security/auth.py` now sends the field only when a name is present.
 - The manual live account `epitome_75_springs@icloud.com` is currently UNVERIFIED with no subscription: the Auth0 tenant has no email provider configured, so verification emails never deliver (configure one under Auth0 Dashboard → Branding → Email Provider, or patch email_verified via the Management API). Provisioning (pro free trial) fires automatically on the first authenticated request after verification.

---

# Bugs / gaps found during testing (2026-07-16; status updated 2026-07-20)

1. **SubscriptionSchedule downgrade 502** — ✅ FIXED. Stripe flexible billing rejects `phases[].iterations`; `src/api/webapp.py` uses `"duration": {"interval": "month", "interval_count": 1}`.
2. **API-key cache stale after webhook** — ✅ FIXED (2026-07-20). `update_user_subscription_status` now evicts `_api_key_cache` via the shared `_evict_api_key_cache_for_user` helper, so `customer.subscription.updated` / `invoice.payment_failed` tier changes take effect on the next request instead of after the 300s TTL.
3. **`customer.subscription.created` not handled** — ✅ FIXED (2026-07-20). `_handle_stripe_event` now dispatches `customer.subscription.created` through the same non-deleted sync path as `updated`.
4. **Rate limit vs allotment testing** — expected behavior, not a bug: set `MESSAGE_RATE_LIMIT_TOKENS_PER_WINDOW=0` / `MEDIA_UPLOAD_RATE_LIMIT_TOKENS_PER_WINDOW=0` before over-allotment tests (see `_METERING_MANUAL_VALIDATION.md`).
5. **Adapter training** — still OPEN by design: no public API route to exercise `adapter_training_units` allotment (Prem10).
6. **Trial allotment floor** — ✅ FIXED (2026-07-20). `resolve_effective_monthly_allotment` is now wired into `enforce_remaining_allotment`, the SSE usage snapshot, and `/verify_subscription_status`, so a mid-trial downgrade keeps the higher trial allotment (and trial-only meters) until `trial_end`. Covered by new unit tests in `test_billing_enforcement_and_periods.py`.

# Harness scripts (local only)
- `scripts/manual_metering_allotment_focus.py`
- `scripts/manual_metering_premium_pro.py`
- `scripts/manual_metering_scenario_test.py` / `manual_metering_retest.py`
