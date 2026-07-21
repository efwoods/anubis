# Metering & Subscriptions — Production Operations Checklist

Step-by-step runbook for taking the TEST-mode-verified metering system live, and
for rotating credentials. Every step is manual and deliberate; nothing here runs
automatically.

## 1. Rotate the committed live Stripe keys (do this FIRST)

The live secret key and publishable key were committed to `.env` in git history
and must be treated as burned.

1. In the Stripe Dashboard (live mode) → Developers → API keys, create a new
   **restricted** key with only the permissions the API uses: Customers (write),
   Checkout Sessions (write), Subscriptions (write), Subscription Schedules
   (write), Billing Meters / Meter Events (write), Payment Methods (read),
   Billing Portal sessions (write), Products & Prices (read), Webhook
   verification happens locally and needs no key permission.
2. Update the deployment's `STRIPE_SECRET_KEY` (and `STRIPE_PUBLISHABLE_KEY`)
   with the new values.
3. **Roll (revoke) the old live secret key** in the Dashboard.
4. Stop committing `.env`: keep real values only in deployment secrets, and keep
   the repository's `.env.example` as the documented shape. (History scrubbing
   is optional; revocation is mandatory.)

## 2. Provision the live Stripe billing objects

```bash
STRIPE_SECRET_KEY=sk_live_... python scripts/provision_stripe_billing.py --allow-live
```

The script is idempotent (meters matched by event name, products by the
`neural_nexus_tier` metadata key, prices by lookup key, portal configuration by
metadata tag). It prints a single-line JSON document that now also contains the
billing-portal configuration id under `"portal_configuration"`.

Paste that JSON into the production environment as `STRIPE_BILLING_CONFIG_JSON`.

## 3. Register the live webhook endpoint

Dashboard (live mode) → Developers → Webhooks → Add endpoint:

- URL: `https://<production-host>/stripe/webhook`
- Events:
  - `checkout.session.completed`
  - `customer.subscription.updated`
  - `customer.subscription.deleted`
  - `invoice.payment_failed`

Copy the endpoint's signing secret (`whsec_...`) into the production
`STRIPE_WEBHOOK_SECRET`. (Local docker-compose continues to use the stripe-cli
service and `STRIPE_WEBHOOK_SECRET_FILE` instead.)

## 4. Brand the billing portal

Dashboard (live mode) → Settings → Billing → Customer portal: set the business
name, icon, and the privacy-policy / terms-of-service URLs. The functional
feature set (invoices, payment-method update, billing-information update,
at-period-end cancellation, no plan switching) is already provisioned by the
script; branding is Dashboard-only.

## 5. Set the new environment variables in production

| Variable | Purpose | Suggested production value |
|---|---|---|
| `USAGE_PERIOD_DAYS` | Local usage-allotment period; 0 = calendar month | `0` |
| `MESSAGE_RATE_LIMIT_WINDOW_SECONDS` | Message rate-limit window | `60` |
| `MESSAGE_RATE_LIMIT_TOKENS_PER_WINDOW` | Message tokens per window (0 disables) | `30000` |
| `MEDIA_UPLOAD_RATE_LIMIT_WINDOW_SECONDS` | Upload rate-limit window | `60` |
| `MEDIA_UPLOAD_RATE_LIMIT_TOKENS_PER_WINDOW` | Upload token-equivalents per window (0 disables) | `90000` |

`STRIPE_MANAGE_SUBSCRIPTION_URL` is now only a degraded-mode fallback used when
`STRIPE_BILLING_CONFIG_JSON` is absent; `GET /manage_subscription` creates a
billing-portal session per request in both test and live modes.

## 6. Live smoke test

1. `GET /subscribe?tier=pro` with a real account → complete checkout with a real
   card → confirm `GET /verify_subscription_status` shows `tier: pro`, the four
   `meters` entries for pro, and the period bounds.
2. Send one message → confirm the meter increments in the Dashboard (Billing →
   Meters) and `used_to_date` moves in `/verify_subscription_status`.
3. `GET /manage_subscription` → confirm the portal session opens.
4. `POST /cancel_subscription` → confirm `cancel_at_period_end: true`, then
   `POST /reactivate_subscription` to keep the account, or let the cancellation
   stand and refund the invoice from the Dashboard.

## 7. Prometheus note

The production `/metrics` route is shadowed by the langgraph platform's own
metrics server; if the Grafana token/cost panels stay empty, point the
Prometheus scrape target at the FastAPI app's metrics port directly.
