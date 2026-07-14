# Metering & Subscriptions — Close the Remaining Gaps (f-metering)

## Context

The Stripe metering system on `f-metering` is largely built and TEST-mode verified: 4 Billing Meters, tier catalog (`src/anubis/utils/billing/tiers.py`), provisioning script, meter-event reporting + `api_metrics` table (`metering.py`), gating helpers (`gating.py`), Checkout `/subscribe`, `/change_subscription_tier`, webhook sync, and metering on all `/message` variants (resume included) and `/update_avatar_identity_with_media`.

The IMMEDIATE NEED in `_METERING_FEATURE_PLAN.md` — a functional, metered, subscribable API with a working customer portal in test AND live — is unmet in seven specific places (verified against code):

1. **Paid tiers are never blocked**: `enforce_remaining_allotment` (`webapp.py:144`) early-returns for non-free tiers; the correct all-tier logic `exhausted_allotment_block_reason` (`gating.py:155`) exists but is unwired. Uploads are never allotment-checked at all.
2. **No pay-per-use toggle**: `POST /set_pay_per_use` is referenced in `gating.py` docstrings but doesn't exist; `/change_subscription_tier` has no pay-per-use parameter.
3. **`/verify_subscription_status` shows no allotment/usage** (and has a `== None` set-literal bug at `webapp.py:1255`). This is the endpoint the future Vercel portal (`anubis-customer-portal`, currently docs-only) will poll.
4. **Rate limiting is unenforced**: `fetch_rolling_window_usage` is never called; the two context vars aren't in `.env.example` and aren't per-endpoint as required.
5. **Usage period is hardcoded** to `date_trunc('month')`; requirement says env-configurable.
6. **Upgrade/downgrade semantics wrong**: `clear_usage` fires on every classic-mode tier change; requirement = usage/allotment RETAINED on downgrade until period end ("unused allotment continues"), CLEARED on upgrade.
7. **Cancel + portal are static URLs**: `/cancel_subscription` and `/manage_subscription` return a production-only link — no real API cancel, no test-mode portal.

Plus: no free-tier billing vehicle for overage (**user decision: build the $0 free subscription with card collection**), no adapter-training report path, and the live-mode ops steps (provision `--allow-live`, live webhook, rotate committed live keys) remain.

Out of scope: the Vercel customer-portal app itself (user builds later; this plan only guarantees the API + Stripe-hosted portal the app will consume), and any adapter-training endpoint (job doesn't exist yet — only the metering helper ships).

---

## Phase 1 — Billing package core (pure logic)

### 1.1 Configurable usage period — `src/anubis/utils/billing/metering.py`
- New pure helper `resolve_usage_period_start(now, usage_period_days, period_anchor=None) -> datetime`:
  - `usage_period_days == 0` (default): calendar-month semantics. No anchor → first of current UTC month (status quo). With per-user anchor → most recent monthly boundary on the anchor's day-of-month (clamp day 29–31, matching Stripe `billing_cycle_anchor`), never earlier than the anchor.
  - `usage_period_days > 0`: fixed windows `anchor + floor((now − anchor)/days)·days`; global module-constant anchor when no per-user anchor (deterministic across restarts).
- Replace `fetch_month_to_date_usage` with `fetch_usage_since(pool, user_id, meter_event_name, period_start)` and add `fetch_usage_by_meter_since(pool, user_id, period_start) -> dict[str, int]` (GROUP BY `meter_event_name`, for the verify endpoint). Keep fail-open on DB errors. Update exports in `billing/__init__.py` and the call site `webapp.py:165`.
- For paid tiers, enforcement prefers the Stripe billing period cached by the webhook (Phase 3.6) over the env-derived period.

### 1.2 Rate-limit primitives — `metering.py`
- Extend the never-called `fetch_rolling_window_usage` to accept `meter_event_names: Sequence[str] | None` and also return `MIN(created_at)` in the window: `-> tuple[int, datetime | None]`.
- Pure decision helper `token_rate_limit_retry_after_seconds(window_usage, tokens_per_window, window_seconds, oldest_usage_at, now) -> int | None` (None = allowed; else seconds until the oldest row exits the window, clamped 1..window).

### 1.3 Tier-change planner + adapter-training helper — `gating.py`, `metering.py`
- `plan_tier_change(current_tier, target_tier) -> TierChangePlan` frozen dataclass: `direction` ("upgrade"/"downgrade"), `swap_items_immediately` (upgrades), `schedule_change_at_period_end` (downgrades, incl. →free), `reset_usage_period_anchor` (True only on upgrade).
- `resolve_usage_period_anchor(user) -> datetime | None` — defensively parse `app_metadata.usage_period_anchor` ISO string.
- `report_adapter_training_usage(...)` in `metering.py`: wraps `report_meter_event(UsageMeter.ADAPTER_TRAINING_UNITS, ...)` + `persist_api_metrics_row(inference_type="adapter_training", ...)`. Ready for the future training job; **no endpoint**. Export from `__init__.py`.

### 1.4 Env vars — `src/anubis/utils/context.py`, `.env.example`, `.env`, `.env.dev`
Replace `rate_limit_window_seconds`/`rate_limit_tokens_per_window` with:

| Env var | GlobalContext field | Default |
|---|---|---|
| `MESSAGE_RATE_LIMIT_WINDOW_SECONDS` | `message_rate_limit_window_seconds` | 60 |
| `MESSAGE_RATE_LIMIT_TOKENS_PER_WINDOW` | `message_rate_limit_tokens_per_window` | 0 = disabled |
| `MEDIA_UPLOAD_RATE_LIMIT_WINDOW_SECONDS` | `media_upload_rate_limit_window_seconds` | 60 |
| `MEDIA_UPLOAD_RATE_LIMIT_TOKENS_PER_WINDOW` | `media_upload_rate_limit_tokens_per_window` | 0 = disabled |
| `USAGE_PERIOD_DAYS` | `usage_period_days` | 0 = calendar month |

Add all five to `.env.example` (no values, per repo convention) next to the STRIPE block; set OpenAI-TPM-aligned working values in `.env.dev`/`.env` (e.g. messaging 30,000 tokens/60 s; media upload 90,000/60 s). Note in `.env.example` that `STRIPE_MANAGE_SUBSCRIPTION_URL` becomes fallback-only.

## Phase 2 — Enforcement wiring — `src/api/webapp.py`

### 2.1 Rewrite `enforce_remaining_allotment` (webapp.py:144)
Drop the `if tier != FREE: return` early-out; resolve period start (cached Stripe period → `usage_period_anchor` → env period), `fetch_usage_since`, then delegate to `exhausted_allotment_block_reason(tier, meter, usage, resolve_pay_per_use_enabled(current_user))` → 402 with its reason. Docstring updated (all tiers, pay-per-use aware).

### 2.2 `enforce_token_rate_limit(app_state, current_user, meter_event_names, window_seconds, tokens_per_window)`
Helper called at endpoint start (not middleware — needs resolved `current_user`, applies to 4 routes only). Raises 429 with `Retry-After` header via the Phase 1.2 helpers. `tokens_per_window <= 0` → no-op.

### 2.3 Wire into four endpoints
- `/message` (webapp.py:1776) and `/message/{assistant_id}` (webapp.py:1946): pick meter first — `ADAPTER_INFERENCE_TOKENS` when `resolve_use_adapter_inference(current_user, adapter)` else `MESSAGING_TOKENS` (current checks always use MESSAGING even for adapter turns — fix); then allotment + rate-limit checks (rate limit sums both messaging + adapter-inference meter names, message env vars).
- `/message/{assistant_id}/resume` (webapp.py:2144): already metered via `message_graph_sse`; add the same two enforcement calls (meter = MESSAGING_TOKENS).
- `/update_avatar_identity_with_media` (webapp.py:3769): after the existing `enforce_tier_capability(UPLOAD)`, add `enforce_remaining_allotment(..., DOCUMENT_UPLOAD_TOKENS)` + rate limit with the media-upload env vars.

## Phase 3 — Subscription-management endpoints

### 3.1 `update_user_app_metadata_fields(request, auth0_user_id, fields)` — `src/security/auth.py`
Generic Auth0 app_metadata PATCH mirroring `update_user_subscription_status` (auth.py:1018). **Must refresh/invalidate the `_api_key_cache` TTL entry** (pattern at auth.py:85) so a toggled pay-per-use flag can't be stale for 5 minutes.

### 3.2 `POST /set_pay_per_use` — `webapp.py`
Body `{"enabled": bool}`. `false` → always allowed. `true` → require `resolve_stripe_customer_id` + a payment method on file (`Customer.retrieve(expand=["invoice_settings.default_payment_method"])`, fallback `PaymentMethod.list`); missing → 402 pointing at `/subscribe` / `/manage_subscription`. Extract pure `customer_has_payment_method(customer_document, payment_methods) -> bool` for unit tests. Persist via 3.1; return `{"pay_per_use_enabled": enabled}`. Trialing users with a card may enable (matches `resolve_pay_per_use_enabled` resolution order).

### 3.3 `pay_per_use` parameter on `/change_subscription_tier` (webapp.py:982)
Optional body field; after a successful tier change apply the shared `_apply_pay_per_use_setting(...)` (same validation as 3.2).

### 3.4 Upgrade/downgrade semantics (`change_subscription_tier`, using `plan_tier_change`)
- **Downgrade (usage/allotment retained)**: do NOT swap items now. `SubscriptionSchedule.create(from_subscription=...)` then `modify` with two phases — current items until period end, target tier's `all_price_ids()` after — `end_behavior="release"`. Stripe keeps billing the already-paid tier until period end; the webhook flips the cached tier at the phase change, so local gating keeps the higher allotment automatically. Release any pre-existing schedule first. `→free` keeps the existing `cancel_at_period_end=True` path.
- **Upgrade (usage cleared)**: keep the existing immediate item swap + classic-mode `clear_usage` (webapp.py:1041-1057); additionally write `app_metadata.usage_period_anchor = now-UTC-ISO` via 3.1 so local usage queries start fresh.
- Also set `usage_period_anchor` on `checkout.session.completed` (free→paid) in the webhook.

### 3.5 Real cancel/reactivate + test-and-live portal
- `POST /cancel_subscription` (replaces GET static-URL version, webapp.py:1081): `Subscription.modify(cancel_at_period_end=True)`; return message + `current_period_end` (read from `items.data[0]` first, top-level fallback — flexible-mode subscriptions moved period bounds onto items). 404 without a subscription.
- `POST /reactivate_subscription`: `cancel_at_period_end=False`; release pending downgrade schedule; 409 if fully canceled (must `/subscribe` again).
- `GET /manage_subscription` (webapp.py:1071): `stripe_client.billing_portal.Session.create(customer=..., return_url=..., configuration=<from billing config>)` — works in both test and live. Fallback to `stripe_manage_subscription_url` only when billing config is absent.
- `scripts/provision_stripe_billing.py`: add idempotent `find_or_create_billing_portal_configuration()` (invoice history, payment-method update, customer_update, `subscription_cancel` at period end; leave `subscription_update` OFF — the hosted portal can't switch metered-price plans; `/change_subscription_tier` owns that). Emit `"portal_configuration"` in the printed JSON; add tolerant optional `portal_configuration_id` to `StripeBillingConfig` (`billing/config.py`).

### 3.6 Usage summary on `/verify_subscription_status` (webapp.py:1250)
Fix the `== None` set-literal bug. Return status/tier/subscription_id/customer_id plus `pay_per_use_enabled`, `cancel_at_period_end`, `usage_period_start/end`, and per-meter `{monthly_allotment, used_to_date, remaining, overage_rate}` — built from `check_subscription_status`, `TIER_DEFINITIONS[tier].meter_allotments`, one `fetch_usage_by_meter_since` call, same period resolution as enforcement. Only meters the tier grants appear. This is the portal-facing usage endpoint.

**Webhook additions** (`_handle_stripe_event`, webapp.py:1103): cache `current_period_start/end` (items-first, defensive) + `cancel_at_period_end` into the stored `subscription_status`; on `customer.subscription.deleted` also write `pay_per_use_enabled: false` so a stale flag can't grant overage after cancellation.

### 3.7 Free-tier pay-per-use vehicle (user-approved)
- Allow `tier=free` through `/subscribe` (currently rejected at webapp.py:905): Checkout session with the free tier's $0 licensed base price + its metered overage prices, `payment_method_collection="always"`. Provisioning script already creates the free-tier prices.
- `/set_pay_per_use enabled=true` for a subscription-less free user → 402 directing to `GET /subscribe?tier=free`.
- Free users without a card remain hard-capped at the allotment (unchanged).

## Phase 4 — Tests, docs, verification

### 4.1 Unit tests — new `tests/unit_tests/test_billing_enforcement_and_periods.py`
Follow the fixture style of `test_billing_tiers_and_gating.py` (dict users, `_FakeStripeClient`, parametrize):
- `exhausted_allotment_block_reason`: 3 tiers × pay-per-use on/off × under/at/over allotment.
- `resolve_usage_period_start`: calendar month, `usage_period_days=7` determinism, per-user anchor, day-31 clamping.
- `token_rate_limit_retry_after_seconds`: disabled/under/over, Retry-After bounds.
- `plan_tier_change`: upgrade vs downgrade matrix incl. anchor-reset flags.
- `resolve_pay_per_use_enabled` explicit-flag precedence + `customer_has_payment_method`.
- `report_adapter_training_usage` payload + api_metrics row.
Endpoint layer stays thin and is covered by the manual checklist (existing suite deliberately doesn't import `webapp.py`).

### 4.2 Ops doc — new `_METERING_OPERATIONS.md`
1. **Rotate the committed live Stripe keys** (revoke old, new restricted key, stop committing `.env`).
2. Live provisioning: `STRIPE_SECRET_KEY=sk_live_... python scripts/provision_stripe_billing.py --allow-live` → paste JSON (now incl. portal configuration) into prod `STRIPE_BILLING_CONFIG_JSON`.
3. Register live webhook (`checkout.session.completed`, `customer.subscription.updated/deleted`, `invoice.payment_failed`) → set `STRIPE_WEBHOOK_SECRET`.
4. Portal branding/ToS URLs; set the five new env vars; live smoke test.

### 4.3 Verification
- `make test` + `make test TEST_FILE=tests/unit_tests/test_billing_enforcement_and_periods.py`; `make lint_diff`.
- **Stripe MCP server** (authenticate first): use to inspect meters/prices/subscriptions/schedules created during manual checks instead of the dashboard where convenient.
- Manual test-mode (docker compose stack with stripe-cli listener; 72-token `/message` scenarios and the `data/` test files per `_METERING_FEATURE_TESTING.md`):
  1. `/subscribe?tier=pro` → checkout 4242 → webhook flips status; `/verify_subscription_status` shows pro allotments, zero usage, period bounds.
  2. `/message` turns → usage increments locally + Stripe meter graph.
  3. Tiny `MESSAGE_RATE_LIMIT_TOKENS_PER_WINDOW` → second message 429 + Retry-After.
  4. Force over-allotment (insert big `api_metrics` row) → 402 on `/message` and `/update_avatar_identity_with_media`; `POST /set_pay_per_use true` → succeeds; `false` → 402 again. Repeat per tier.
  5. Upgrade pro→premium → fresh usage (anchor) + invoice; downgrade premium→pro → SubscriptionSchedule exists, allotment unchanged until period end (use a Stripe **test clock** to fast-forward).
  6. `POST /cancel_subscription` → `cancel_at_period_end` true in verify; `POST /reactivate_subscription` → cleared.
  7. `GET /manage_subscription` → live `billing.stripe.com/session/...` portal URL (test mode).
  8. `/subscribe?tier=free` with card → enable pay-per-use → over-allotment message succeeds and bills free-tier overage rate.
  9. Resume flow: interrupt + `POST /message/{id}/resume` → new `api_metrics` row + meter event.
  10. Trial: pro trial without card stays capped, deletes→free at trial end; with card converts to active pro.

## Design constraints to respect
- stripe-python 15: `StripeObject` → `.to_dict()` everywhere.
- Meters can't be decremented; "clear on upgrade" is local (`usage_period_anchor`); downgrades avoid Stripe-side mis-billing via period-end schedules.
- `clear_usage` is classic-billing-mode-only (existing guard, keep).
- Period bounds live on subscription **items** in flexible mode — read items-first.
- Fail-open gating on DB errors is deliberate; enforcement is pre-request, metering post-response (one-request overshoot accepted, same as OpenAI TPM).
- Naming: fully spelled-out identifiers, no acronyms, no bare "it" in prompts/docs.
