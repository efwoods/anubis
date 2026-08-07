#!/bin/sh
# scripts/stripe_listen_entrypoint.sh
#
# Dev-only: print the Stripe CLI webhook signing secret to a shared file, then
# forward account events to the API. The API reads that file when
# STRIPE_WEBHOOK_SECRET is unset (see GlobalContext.stripe_webhook_secret_file),
# so .env.dev does not need a fresh whsec_ on every restart.
#
# Persistent / prod webhooks use a Dashboard "Your account" endpoint secret in
# STRIPE_WEBHOOK_SECRET instead — not this script.

set -eu

API_KEY="${STRIPE_API_KEY:-${STRIPE_SECRET_KEY:-}}"
if [ -z "$API_KEY" ]; then
  echo "stripe-cli: STRIPE_SECRET_KEY (or STRIPE_API_KEY) is not set." >&2
  exit 1
fi

SECRET_FILE="${STRIPE_WEBHOOK_SECRET_FILE:-/run/stripe/webhook_secret}"
FORWARD_TO="${STRIPE_WEBHOOK_FORWARD_TO:-http://langgraph-api-dev:8123/stripe/webhook}"
# This list must stay equal to the branches in webapp.py::_handle_stripe_event,
# which is the source of truth. It is duplicated in two places that cannot import
# it — here, and in scripts/provision_stripe_webhook.py (which registers the
# production endpoint). Change one, change all three.
#
# customer.subscription.created must be forwarded: the API creates subscriptions
# server-side without Checkout (the pro signup trial, the free-tier billing
# vehicle, tier changes made with the Stripe SDK), and those emit ONLY .created.
# Omitting it leaves app_metadata.subscription_status pointing at whatever
# subscription came before, so the API keeps re-reading a stale (often canceled)
# subscription until some later .updated happens to fire.
#
# customer.updated must be forwarded too: webapp.py handles it to keep the
# cached billing email in step when a customer's details change in Stripe.
EVENTS="${STRIPE_WEBHOOK_EVENTS:-checkout.session.completed,customer.updated,customer.subscription.created,customer.subscription.updated,customer.subscription.deleted,invoice.payment_failed}"

mkdir -p "$(dirname "$SECRET_FILE")"

echo "stripe-cli: resolving webhook signing secret..."
SECRET="$(stripe listen --api-key "$API_KEY" --print-secret)"
if [ -z "$SECRET" ]; then
  echo "stripe-cli: --print-secret returned empty." >&2
  exit 1
fi
# Atomic-ish replace so the API never reads a half-written secret.
printf '%s' "$SECRET" > "${SECRET_FILE}.tmp"
mv "${SECRET_FILE}.tmp" "$SECRET_FILE"
echo "stripe-cli: wrote signing secret to ${SECRET_FILE}"
echo "stripe-cli: forwarding to ${FORWARD_TO}"

exec stripe listen \
  --api-key "$API_KEY" \
  --events "$EVENTS" \
  --forward-to "$FORWARD_TO"
