# POST /subscribe as single subscription entry point + anonymous Stripe metering fix

## Context

Two bugs from `_BUG_07_15_2026.md` (2026-07-15 testing):

1. **`/subscribe` rejects existing subscribers** with 409 "You already have a subscription. Use POST /change_subscription_tier to switch tiers." — but that route no longer exists. The pure planner `plan_subscribe_action` (`src/anubis/utils/billing/gating.py:337-383`, fully unit-tested) was built for exactly this dispatch but was **never wired into the endpoint**: `GET /subscribe` (webapp.py:1241) still has the old guard at 1282-1296, `change_subscription_tier` sits at webapp.py:1490 as un-routed dead code, and `plan_tier_change` is called without `currently_trialing` so trial free-usage would be reset on upgrade. Owner requirements: selecting the current tier = no-op (subscription and trial untouched); selecting a different tier = change tier retaining trial free-usage; pending cancellation auto-reactivates; `/cancel_subscription` and `/reactivate_subscription` standalone routes are removed (the billing portal via `GET /manage_subscription` handles those for end users).
2. **Anonymous usage does not aggregate in Stripe.** Two verified root causes: `ANONYMOUS_BILLING_ENABLED=FALSE` in `.env:153` and `.env.dev:156` (gate at `src/security/anonymous_billing.py:100-101` returns None silently), and `ensure_anonymous_billing_customers_table` (metering.py:273-287) is never called in the webapp lifespan (only `ensure_api_metrics_table` at webapp.py:1113), so the hashed-ip→customer mapping table never exists. The rest of the chain is verified correct: auth.py:886-892 stores the customer id on the canonical `app_metadata.stripe_customer_id` key for both anonymous branches; meter payload keys match the provisioned meters; `create_free_tier_subscription` exists (auth.py:296-334).

## Issue A — rewire /subscribe (all in `src/api/webapp.py` unless noted)

### A1. Extract internal helpers from existing bodies
- **`_reactivate_subscription_for_user(stripe_client, subscription: dict) -> None`** — body extracted from the `/reactivate_subscription` route (webapp.py:1760-1780): `_release_pending_subscription_schedule(...)` then `Subscription.modify(id, cancel_at_period_end=False)`. Takes the already-retrieved subscription dict; drops the status-lookup preamble (the planner guarantees a live subscription). Stripe error → log + HTTPException 502.
- **`_change_subscription_tier_for_user(request, current_user, requested_tier: SubscriptionTier, currently_trialing: bool, pay_per_use: bool | None = None) -> dict`** — rename of dead `change_subscription_tier` (webapp.py:1490): strip `Body(...)`/`Depends(...)` machinery, take a validated `SubscriptionTier`, delete the same-tier 400 guard (planner routes same-tier away first). **Behavioral fix:** webapp.py:1535 becomes `plan_tier_change(current_tier, requested_tier, currently_trialing=currently_trialing)` — with trialing True the usage-period anchor is NOT rewritten, so trial free-usage is retained (matches `resolve_effective_monthly_allotment` trial-floor semantics, gating.py:424-457). Everything else in the body (downgrade schedule, atomic item swap, pay_per_use passthrough) unchanged.

### A2. Replace `GET /subscribe` with `POST /subscribe`
Owner preference (earlier bug note): **query parameters as Swagger dropdowns, not JSON body**:

```python
@app.post("/subscribe")
async def subscribe(
    request: Request,
    tier: SubscriptionTier = Query(default=SubscriptionTier.PRO),
    pay_per_use: Optional[bool] = Query(default=None),
    current_user: dict = Depends(get_current_user),
):
```

Flow (replaces the 409 guard; the checkout construction at 1298-1340 stays as the START_CHECKOUT branch):
1. Existing email-verified guard + legacy payment-link fallback unchanged.
2. `status = await check_subscription_status(...)` (returns `{status, subscription_id, customer_id, email, tier}` — no cancel/schedule info).
3. When `subscription_id` exists and status ∈ (active, trialing, past_due): `subscription = stripe_client.Subscription.retrieve(subscription_id).to_dict()`; `cancel_at_period_end = bool(subscription.get("cancel_at_period_end"))`; `has_pending_downgrade_schedule = subscription_has_pending_downgrade_schedule(subscription)` (new pure helper, A4).
4. `action = plan_subscribe_action(current_status, current_tier, requested_tier, cancel_at_period_end, has_pending_downgrade_schedule)` — add `plan_subscribe_action`, `SubscribeAction` to the billing import block (~webapp.py:81).
5. Dispatch:
   - **START_CHECKOUT** → existing checkout code: `{"action": "start_checkout", "url": ..., "message": ...}`.
   - **NO_CHANGE_REQUIRED** → no Stripe mutation, no anchor write, trial untouched: `{"action": "no_change_required", "message": ..., "subscription_status": status}` (apply `pay_per_use` only when explicitly provided).
   - **REACTIVATE** → `_reactivate_subscription_for_user(...)`: `{"action": "reactivate", "cancel_at_period_end": false, "subscription_status": ...}`.
   - **CHANGE_TIER** → `_change_subscription_tier_for_user(..., currently_trialing=(current_status == "trialing"), pay_per_use=pay_per_use)`: `{"action": "change_tier", **result}`.
   - **REACTIVATE_AND_CHANGE_TIER** → reactivate helper, then change-tier helper: `{"action": "reactivate_and_change_tier", **result}`.

POST-only, no GET alias: with the planner, GET would mutate state (unsafe for idempotent method; prefetchers could change subscriptions). Stale GET clients get a clear 405.

### A3. Delete routes + fix stale strings
- Delete `@app.post("/cancel_subscription")` + body (webapp.py:1704-1743) and `@app.post("/reactivate_subscription")` + body (1746-1781) after extracting A1's helper — the billing portal (`GET /manage_subscription`) handles both for end users; reactivation also happens automatically via POST /subscribe.
- Stale string fixes: webapp.py:1429 (`GET /subscribe?tier=free` → POST wording), 1526 ("Use POST /subscribe"), 1661 (manage_subscription docstring), 1681; new /subscribe docstring describes the five-action dispatch.

### A4. New pure helper + tests (`src/anubis/utils/billing/gating.py`)
- `subscription_has_pending_downgrade_schedule(subscription: Mapping[str, Any] | None) -> bool` — truthy `subscription.get("schedule")`, handling None / string id / expanded dict. Export from billing `__init__`.
- Tests in `tests/unit_tests/test_billing_tiers_and_gating.py`: None subscription, string schedule id, expanded dict, absent key. The planner itself is already covered by `TestPlanSubscribeAction` (:451-525).

## Issue B — anonymous Stripe aggregation

### B1. Lifespan table creation
In webapp lifespan, directly after `await ensure_api_metrics_table(app.state.pool)` (webapp.py:1113): `await ensure_anonymous_billing_customers_table(app.state.pool)`. Add to the metering import block (~webapp.py:67).

### B2. Log the silent fail-open paths (`src/security/anonymous_billing.py`)
Once-per-process warning flags for the two silent drops: gate off (:100-101) → "Anonymous billing is disabled (ANONYMOUS_BILLING_ENABLED != TRUE); anonymous usage will not be metered to Stripe."; missing `app.state.stripe_billing_config` (:102-105) → equivalent warning. First occurrence warns, later occurrences stay silent.

### B3. Flip the gate (last step, after code lands)
- `.env:153` and `.env.dev:156` → `ANONYMOUS_BILLING_ENABLED=TRUE` (`.env.example` already has the empty key; `GlobalContext.anonymous_billing_enabled` already exists).
- **Ops prerequisite, not code:** `.env:139` `STRIPE_BILLING_CONFIG_JSON=` is empty — the live environment still fails open until the live Stripe catalog is provisioned (`scripts/provision_stripe_billing.py --allow-live`) and its JSON pasted into `.env`. Dev (`.env.dev:142`) is already configured and will work immediately.

Nothing else is needed for correctness: the per-process TTL cache fronts the durable Postgres row; races are absorbed by the `ON CONFLICT` upsert (metering.py:266-270) and the `Customer.search` dedup fallback (anonymous_billing.py:66-84).

## Sequencing
A1 helpers → A2 POST /subscribe → A3 route deletions + strings → A4 gating helper + tests → B1 lifespan + B2 logging → B3 env flips.

## Verification
- Unit: `make test TEST_FILE="tests/unit_tests/test_billing_tiers_and_gating.py tests/unit_tests/test_token_estimation.py tests/unit_tests/test_usage_period_math.py"` (plus test_billing_enforcement_and_periods.py).
- Manual (dev stack, Stripe test mode; restart dev container first):
  1. No sub → `POST /subscribe?tier=pro` → `{"action":"start_checkout","url":...}`; complete checkout → `trialing`.
  2. Same tier again → `{"action":"no_change_required",...}`; Stripe dashboard shows subscription + trial end untouched.
  3. While trialing, `POST /subscribe?tier=premium` → `{"action":"change_tier",...}`; Auth0 app_metadata `usage_period_anchor` NOT rewritten, `trial_context` intact; message allotment still honors the trial floor.
  4. Cancel via billing portal (`GET /manage_subscription`) → `POST /subscribe?tier=<current>` → `{"action":"reactivate","cancel_at_period_end":false}`.
  5. `POST /cancel_subscription` → 404; `GET /subscribe` → 405.
  6. Anonymous: restart webapp (check `anonymous_billing_customers` table exists), send anonymous message with spoofed `X-Forwarded-For` → row in table, Stripe customer with `metadata.anonymous_hashed_ip`, $0 free-tier subscription, meter event on that customer in Billing → Meters. Flip gate FALSE + restart → one-time warning appears in logs.
