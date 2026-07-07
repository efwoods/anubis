# Metering & 3-Tier Subscriptions (Stripe) — Plan

## Context

`_METERING_FEATURE.md` requires three subscription tiers — **Free**, **Pro** (with a
free trial), **Premium** — that users can switch between, where each tier grants a
**monthly allotment** plus **pay-per-use overage**, and the allotment is split into
**four separately-budgeted, per-user, monthly-resetting usage dimensions** so one
can't cannibalize another:

1. **messaging tokens** — all tiers
2. **document-upload tokens** — uploads distilled to tokens via diarization + structured-output cost estimate (Pro, Premium)
3. **adapter-training units** — Premium
4. **adapter-inference tokens** — billed at a *different* rate than standard inference, with fallback to the non-adapter model (Premium)

The live Stripe account is **"Afterlife Systems Inc"** (`acct_1RyHMQLimk9GVblr`). The
user asked us to diagnose the live config against these standards and plan both the
corrected Stripe object model **and** the full application-side metering integration.

**Decisions taken with the user:** (1) plan covers Stripe objects **+ full code
integration**; (2) build & validate everything in **Stripe TEST mode first**, then
replicate to live; (3) **four separate meters**, per-user, monthly reset, with the
allotment varying per tier (option 1).

Metering is greenfield: the current integration is subscription-only (create / manage
/ cancel / verify). There is **no** usage metering, **no** tier gating enforcement,
**no** webhooks, and **no** per-user token recording (the Prometheus token/cost
counters exist but are never incremented).

---

## Part A — Current live Stripe state (diagnosed via Stripe MCP): what's wrong

Three tier **products** exist, but their **prices are misconfigured** and cannot
express "flat fee + 4-dimension allotment + overage":

| Tier | Product | Default price | Actual config | Problem |
|---|---|---|---|---|
| Free | `prod_Uoxh1F1O4jtoPc` | `price_1TpJmRLimk9GVblrZ90XGYj7` | 1 metered price, `per_unit` $0, meter `mtr_…41Q` | Single meter; `per_unit` (no allotment tier); $0/unit → overage never bills |
| Pro | `prod_UJ3tEj2bomrBZd` | `price_1TKRlhLimk9GVblrqdhhyiEi` | licensed flat **$20/mo**, tax-inclusive | No metered items → no usage/overage at all |
| Premium | `prod_UoxkiVucGqKBAF` | `price_1TpJpMLimk9GVblr6l1p4May` | 1 metered price, `per_unit` **$50/unit**, meter `mtr_…IMS` | Charges $50 **per metered event**; no flat base; single meter; no allotment tier |

Also:
- Only **one payment link is active** (`plink_1TKRnBLimk9GVblrGtWquA6q` = the
  `STRIPE_PAYMENT_URL`) and it sells **only Pro** (30-day trial, pause-if-no-payment).
  No way to choose Free/Pro/Premium, no switch/upgrade/downgrade path.
- Prices are `billing_scheme=per_unit`, not **tiered/graduated** — "N included then
  pay-per-use" is impossible as configured.
- Each price references **≤1 meter**; spec needs **four** dimensions per paid tier.
- **Payment Links do not support usage-based billing** (Stripe docs) — the link
  approach fundamentally can't deliver metered tiers.
- `STRIPE_PRODUCT_ID` in `.env` (`prod_UFBn6TLLfo9fgI`) is a **stale legacy product**,
  not any of the three current tier products.
- **Live `sk_live_`/`pk_live_` keys are committed in `.env` and `.env.dev`** — flag
  for rotation; do dev in test mode.

## Part B — Corrected Stripe object model (build in TEST mode)

**Four account-level Billing Meters** (shared across tiers; the *price* sets the
allotment+rate, the *meter* aggregates events per customer, per billing period which
gives the monthly reset). `default_aggregation=sum`,
`customer_mapping.event_payload_key → stripe_customer_id`:
`messaging_tokens`, `document_upload_tokens`, `adapter_training_units`,
`adapter_inference_tokens`.

**Per tier = one product with multiple `interval=month` prices:**
- **1 licensed flat base price** (monthly fee): Free `$0`, Pro `$20`, Premium `$TBD`.
- **Metered prices**, each `billing_scheme=tiered`, `tiers_mode=graduated`, attached to
  one meter, two tiers each:
  - Tier 1 `up_to=<monthly allotment for this dimension & tier>`, `unit_amount=0` (included)
  - Tier 2 `up_to=inf`, `unit_amount=<overage rate>` (pay-per-use)
  - Free → `messaging_tokens` only. Pro → adds `document_upload_tokens`. Premium →
    adds `adapter_training_units` + `adapter_inference_tokens`, with larger allotments
    and document-upload limit than Pro. The **allotment differs per tier** by using a
    different Tier-1 `up_to` on each tier's price against the **same** four meters.

**A customer's subscription** = flat base item + that tier's metered items (Stripe
allows mixing licensed + metered items in one subscription).

**Free trial** = `trial_period_days` on Pro (and optionally Premium),
`trial_settings.end_behavior.missing_payment_method=pause`. Trial expiry with no
payment method → user falls back to Free-tier messaging.

**Tier switching:** the Stripe **Customer Portal "Switch plan" does NOT support
subscriptions containing metered prices** (confirmed in Stripe docs). Therefore:
- Initial subscribe / upgrade → **Stripe Checkout (`mode=subscription`)** per tier
  (replaces the payment-link `/subscribe`).
- Plan change → our own endpoint using the **Subscription API**: replace each
  subscription **item**'s price to the target tier (pass item IDs so prices are
  *replaced*, not added); pick `proration_behavior` deliberately; use subscription
  schedules for end-of-period downgrades.
- Customer Portal retained for payment-method, invoices, cancellation only.

## Part C — Application code integration

### C1. Config & env (`src/anubis/utils/context.py`, `.env`, `.env.dev`, `.env.example`)
- Add `GlobalContext` fields (lowercase; env uppercase; per repo convention): 
  `stripe_publishable_key`, `stripe_manage_subscription_url` (both already in `.env`
  but **not** wired), `stripe_webhook_secret`, and the corrected tier price/meter
  identifiers. Replace the single stale `stripe_product_id`.
- Add a **tier→pricing config** (a typed mapping, e.g. a new module
  `src/anubis/utils/billing/tiers.py`): for each tier, the flat price ID, the metered
  price IDs, and the four monthly allotments + overage rates. Numbers seed from
  `research/04_token_workload_cost_model.md` and the existing cost fields in
  `context.py` L324–332 (`audio_transcription_price_per_minute`,
  diarization per-million-token in/out prices).
- Update `.env.example` with all new `STRIPE_*` keys (empty values, per convention).

### C2. Stripe customer provisioning (`src/security/auth.py`)
- `signup_user` (L199) currently writes only `app_metadata.api_key` and **never
  creates a Stripe customer**, yet other code reads `app_metadata["customer_dict"]["id"]`
  (webapp `/subscribe` L669) and `app_metadata["customer"]["id"]` (auth `/delete_user`).
  Create the Stripe customer at signup, store a **single canonical key** (pick one,
  e.g. `app_metadata.stripe_customer_id`) and default tier `free`; migrate the two
  inconsistent readers. Provision customers for **anonymous users** too (needed so
  meter events key on a real `stripe_customer_id`).
- **Anonymous users are ALWAYS Free tier.** They get a Stripe customer + `stripe_customer_id`
  (so their usage still meters against the four meters at Free-tier allotments), but
  they can **never** hold a Pro/Premium subscription — no Checkout/upgrade/tier-change
  path is offered to them, and `tier` is hard-pinned to `free` regardless of any
  subscription lookup. Upgrading requires converting to a verified account first.

### C3. Subscription status & tier model (`src/security/auth.py`)
- Extend `SubscriptionStatus` (L872) with a `tier` field; store tier + status in
  `app_metadata`. **Resolve tier to `free` unconditionally for anonymous users**
  (short-circuit before any Stripe subscription lookup).
- Fix `check_subscription_status` (L909): the **cache-write key typo**
  `"subscription_stat s"` (L949) never matches the `"subscription_status"` reader
  (L911); and the **full `Customer.list()`/`Subscription.list()` table scan** filtered
  in pandas won't scale — filter by `email`/`customer_id` server-side, or rely on the
  webhook-synced `app_metadata` (C6) as the primary source and treat Stripe as fallback.

### C4. Subscription endpoints (`src/api/webapp.py` L657–698)
- `/subscribe` → create a **Checkout Session** for the chosen tier (accept a `tier`
  param) instead of returning the single payment-link URL.
- Add `POST /change_subscription_tier` → Subscription-API item-price replacement (Part B).
- `/manage_subscription` & `/cancel_subscription` (L675–688) currently return a
  **hardcoded** portal URL — switch to `context.stripe_manage_subscription_url`, or
  create a real Customer Portal session.

### C5. Usage metering + gating (the core of the feature)
- **Meter-report helper** (new, e.g. `src/anubis/utils/billing/metering.py`): send
  Stripe **meter events** (`stripe.billing.MeterEvent.create`) with
  `payload.stripe_customer_id` + value, one function per dimension. Called after each
  billed operation.
- **Token capture is already present** — reuse it rather than re-deriving:
  `model.py` `TokenUsage`/`ResponseMetadata` (L32–410), `schema.py` L29–33 token
  accumulation, and the media-graph/image token extraction. Wire these into the
  meter-report helper and finally `.inc()` the **defined-but-unused**
  `MODEL_TOKENS_TOTAL`/`MODEL_COST_TOTAL` Prometheus counters (`webapp.py` L354–366)
  so the ready Grafana panels populate.
- **Per-message metering:** report `messaging_tokens` after graph runs on
  `POST /message` (L1217) and `POST /message/{assistant_id}` (L1366) — split standard
  vs `adapter_inference_tokens` when an adapter is attached.
- **Upload metering:** estimate upload cost (diarization + structured-output pricing
  from `context.py`) → report `document_upload_tokens` in the media-upload path.
- **Adapter-training metering:** report `adapter_training_units` (endpoint is future;
  add the report hook now).
- **Gating:** add a FastAPI dependency/middleware that, before graph execution on
  `/message*` and uploads, checks the user's tier capability (free=message,
  pro=+upload, premium=+train) and remaining allotment; allow overage per tier or
  block per the spec. This is the enforcement layer that today **does not exist**
  (message endpoints are completely ungated).
- **`api_metrics` PG table** (described in CLAUDE.md L151 but not built): persist
  per-call latency/tokens/cost/model/inference-type for Grafana + reconciliation.

### C6. Webhooks (new — none exist today)
- Add `POST /stripe/webhook` with `stripe.Webhook.construct_event` (verify against
  `stripe_webhook_secret`). Handle `checkout.session.completed` (activate tier),
  `customer.subscription.updated`/`deleted` (sync tier/status → `app_metadata`),
  `invoice.payment_failed` (downgrade/flag). This replaces the pull-only
  `check_subscription_status` as the real-time source of truth.

### Files at a glance
- `src/anubis/utils/context.py` L324–332 (cost prices), L537–549 (Stripe fields → expand)
- new `src/anubis/utils/billing/{tiers.py,metering.py}`
- `src/security/auth.py` L199 (signup+customer), L872 (`SubscriptionStatus`+tier), L909 (status fixes)
- `src/api/webapp.py` L354–366 (counters), L541 (stripe init), L657–698 (sub endpoints), L1217/1366/1543 (gate message endpoints), + new webhook route
- `src/anubis/utils/model.py` L32–410, `src/anubis/utils/schema.py` L29–33 (reuse token capture)
- `.env`, `.env.dev`, `.env.example`; `grafana/provisioning/dashboards/anubis-api-metrics.json` (panels already present)

## Open numeric decisions (finalize during implementation, seed from `research/04`)
Premium monthly price; per-tier monthly allotment for each of the 4 meters; overage
$/unit for each; Pro/Premium document count and Premium adapter count (expressed as
token/unit allotments); adapter-inference rate multiplier vs standard inference.

## Verification (end-to-end, TEST mode)
1. Via Stripe MCP in **test mode**: create the 4 meters, 3 products, and all prices;
   confirm each metered price is graduated with a $0 included tier + overage tier.
2. Create a test subscription per tier (flat + metered items); preview the upcoming
   invoice and confirm flat base + $0 within allotment.
3. Send synthetic **meter events** for each dimension past the allotment; confirm the
   upcoming invoice shows correct graduated overage and that dimensions bill
   independently (no cross-cannibalization).
4. Run the app locally (`docker compose up` / `langgraph dev`): exercise `/subscribe`
   (Checkout), a `/message` call → assert `messaging_tokens` meter event + Prometheus
   counter increment + `api_metrics` row; an upload → `document_upload_tokens`.
5. Fire test webhooks (`stripe trigger` / MCP) for
   `checkout.session.completed`, `customer.subscription.updated`,
   `invoice.payment_failed`; assert `app_metadata` tier/status sync.
6. Exercise `POST /change_subscription_tier` and confirm subscription **item prices
   are replaced** (not duplicated) with correct proration.
7. Confirm gating: a Free user is blocked from uploads; a Pro user from adapter training.
8. Once green in test mode, replicate the object model to live and swap keys/IDs
   (and rotate the committed live keys).
