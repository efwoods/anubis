#!/usr/bin/env python
# scripts/provision_stripe_webhook.py

"""Idempotently register the production Stripe webhook endpoint and print its secret.

Companion to ``scripts/provision_stripe_billing.py``: that script builds the
billing objects, this one builds the single webhook endpoint the API needs in
order to keep Auth0 ``app_metadata.subscription_status`` in step with Stripe.

WHY THIS IS A SCRIPT AND NOT A COMPOSE SIDECAR
----------------------------------------------
Stripe returns a webhook endpoint's signing secret **only in the create
response** — never on retrieve, never on update. So a secret can be captured
once and then only stored, never re-read. That makes it a genuine environment
variable (``STRIPE_WEBHOOK_SECRET``) rather than something a container can
regenerate on every start, which is why the compose stack has a
``stripe-provision`` service but no webhook equivalent.

The development stack solves the same problem differently: ``stripe listen``
(the ``stripe-cli`` service) mints its own secret and relays events over an
outbound connection to a container-network address. That is a development
relay, not an ingress — it never uses the public URL, and events that occur
while the relay is down are lost, with no retry schedule and no delivery log.
Production therefore uses a registered endpoint, which Stripe retries for
roughly three days with exponential backoff and records in the Dashboard.

WHAT THIS SCRIPT DOES
---------------------
* Finds an enabled endpoint at ``--url``. If one exists, it does **not**
  recreate it (that would discard a secret you have already stored). It instead
  reconciles the endpoint's subscribed events if they have drifted, and exits.
* If no such endpoint exists, creates one and prints the signing secret to
  store in ``STRIPE_WEBHOOK_SECRET``.
* With ``--rotate``, deletes the existing endpoint and creates a replacement so
  a fresh secret becomes readable — use only when the stored secret is lost.
  In-flight deliveries signed with the old secret fail verification, and the
  endpoint's delivery history restarts.
* With ``--disable-other-endpoints``, disables any other enabled endpoint whose
  URL ends in ``/stripe/webhook``. That is how the misspelled
  ``api.neuralneuxus.site`` endpoint was retired; it is opt-in because
  disabling a live endpoint stops real deliveries.

Usage:

    # test mode (default guard: refuses a live key unless --allow-live is passed)
    STRIPE_SECRET_KEY=sk_test_... python scripts/provision_stripe_webhook.py \\
        --url https://api.neuralnexus.site/stripe/webhook

    # live mode (explicit opt-in)
    STRIPE_SECRET_KEY=sk_live_... python scripts/provision_stripe_webhook.py \\
        --url https://api.neuralnexus.site/stripe/webhook --allow-live

Then put the printed ``whsec_...`` into ``STRIPE_WEBHOOK_SECRET`` in the env
file that stack loads (``.env`` for production) and restart the API. Verify with
an unsigned POST: a ``503`` means the secret is still unset, a ``400``
("invalid signature") means it is configured and verification is running.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

import stripe

DEFAULT_WEBHOOK_URL = "https://api.neuralnexus.site/stripe/webhook"
WEBHOOK_PATH_SUFFIX = "/stripe/webhook"

# Must stay equal to the branches in webapp.py::_handle_stripe_event, which is
# the source of truth. Duplicated in two places that cannot import it — here and
# in scripts/stripe_listen_entrypoint.sh (the development relay's event list).
# Change one, change all three. Subscribing to events the API does not handle
# only costs needless deliveries; MISSING one silently drops a state change.
HANDLED_EVENT_NAMES: List[str] = [
    # Checkout completion is how a subscription created through Stripe Checkout
    # first becomes known to the API.
    "checkout.session.completed",
    # Keeps the cached billing email in step when customer details change.
    "customer.updated",
    # The API creates subscriptions server-side without Checkout (the pro signup
    # trial, the free-tier billing vehicle, tier changes made with the Stripe
    # SDK), and those emit ONLY .created.
    "customer.subscription.created",
    "customer.subscription.updated",
    "customer.subscription.deleted",
    # Drives the past-due / payment-failure path.
    "invoice.payment_failed",
]

ENDPOINT_DESCRIPTION = (
    "Neural Nexus API subscription-lifecycle sync: keeps Auth0 "
    "app_metadata.subscription_status in step with every Stripe subscription "
    "change, including those made by the customer portal."
)


def subscribed_event_names(endpoint: Dict[str, Any]) -> List[str]:
    """Read an endpoint's enabled events by subscript, not ``.get``.

    ``stripe.WebhookEndpoint.list`` yields ``StripeObject`` instances, which are
    NOT a dict subclass in stripe-python 15 — ``.get`` raises ``AttributeError``.
    Subscript access works, so every field read in this script uses it. The same
    trap is called out in webapp.py where ``construct_event`` returns one.
    """
    try:
        return list(endpoint["enabled_events"] or [])
    except (KeyError, AttributeError):
        return []


def find_enabled_endpoint_at_url(
    endpoints: List[Dict[str, Any]], webhook_url: str
) -> Optional[Dict[str, Any]]:
    for endpoint in endpoints:
        if endpoint["url"] == webhook_url and endpoint["status"] == "enabled":
            return endpoint
    return None


def reconcile_subscribed_events(endpoint: Dict[str, Any]) -> bool:
    """Bring an existing endpoint's event list back in line. Returns True if changed.

    Updating an endpoint never exposes its secret, so this repairs a drifted
    subscription list without disturbing the stored ``STRIPE_WEBHOOK_SECRET``.
    """
    existing_events = sorted(subscribed_event_names(endpoint))
    if existing_events == sorted(HANDLED_EVENT_NAMES):
        return False
    missing_events = sorted(set(HANDLED_EVENT_NAMES) - set(existing_events))
    surplus_events = sorted(set(existing_events) - set(HANDLED_EVENT_NAMES))
    if missing_events:
        print(f"  events missing from the endpoint: {', '.join(missing_events)}")
    if surplus_events:
        print(f"  events the API does not handle:   {', '.join(surplus_events)}")
    stripe.WebhookEndpoint.modify(endpoint["id"], enabled_events=HANDLED_EVENT_NAMES)
    print(f"  reconciled endpoint {endpoint['id']} to the {len(HANDLED_EVENT_NAMES)} handled events")
    return True


def create_endpoint(webhook_url: str) -> str:
    created = stripe.WebhookEndpoint.create(
        url=webhook_url,
        enabled_events=HANDLED_EVENT_NAMES,
        description=ENDPOINT_DESCRIPTION,
    )
    print(
        f"  CREATED endpoint {created['id']} -> {webhook_url} "
        f"({len(HANDLED_EVENT_NAMES)} events)"
    )
    return created["secret"]


def disable_other_webhook_endpoints(
    endpoints: List[Dict[str, Any]], webhook_url: str
) -> None:
    for endpoint in endpoints:
        if endpoint["url"] == webhook_url or endpoint["status"] != "enabled":
            continue
        if not endpoint["url"].endswith(WEBHOOK_PATH_SUFFIX):
            continue
        stripe.WebhookEndpoint.modify(endpoint["id"], disabled=True)
        print(f"  disabled {endpoint['id']} -> {endpoint['url']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.environ.get("STRIPE_WEBHOOK_URL", DEFAULT_WEBHOOK_URL),
        help=f"Public HTTPS endpoint to register (default: {DEFAULT_WEBHOOK_URL}).",
    )
    parser.add_argument(
        "--allow-live",
        action="store_true",
        help="Permit running against a live (sk_live_) key. Off by default.",
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        help=(
            "Delete the existing endpoint and create a replacement so a fresh "
            "secret is readable. Only when the stored secret is lost — in-flight "
            "deliveries fail verification and delivery history restarts."
        ),
    )
    parser.add_argument(
        "--disable-other-endpoints",
        action="store_true",
        help=(
            "Disable every OTHER enabled endpoint whose URL ends in "
            "/stripe/webhook (retires a stale or misspelled endpoint)."
        ),
    )
    arguments = parser.parse_args()

    secret_key = os.environ.get("STRIPE_SECRET_KEY")
    if not secret_key:
        raise SystemExit("STRIPE_SECRET_KEY is not set.")

    is_live_key = secret_key.startswith("sk_live_")
    if is_live_key and not arguments.allow_live:
        raise SystemExit(
            "Refusing to register a webhook against a LIVE key without "
            "--allow-live. Register and verify in test mode first (sk_test_ key)."
        )
    if not arguments.url.startswith("https://"):
        raise SystemExit(
            f"Refusing to register a non-HTTPS endpoint: {arguments.url}. Stripe "
            "signs deliveries but does not encrypt them for you."
        )

    stripe.api_key = secret_key
    mode = "LIVE" if is_live_key else "TEST"
    print(f"=== Registering Stripe webhook endpoint in {mode} mode ===")
    print(f"    {arguments.url}")

    endpoints = stripe.WebhookEndpoint.list(limit=100)["data"]
    existing_endpoint = find_enabled_endpoint_at_url(endpoints, arguments.url)

    signing_secret: Optional[str] = None
    if existing_endpoint is not None and not arguments.rotate:
        print(f"  endpoint exists -> {existing_endpoint['id']} (secret not re-readable)")
        if not reconcile_subscribed_events(existing_endpoint):
            print("  subscribed events already match the API's handled events")
    else:
        if existing_endpoint is not None:
            stripe.WebhookEndpoint.delete(existing_endpoint["id"])
            print(f"  deleted {existing_endpoint['id']} to rotate its secret")
        signing_secret = create_endpoint(arguments.url)

    if arguments.disable_other_endpoints:
        disable_other_webhook_endpoints(endpoints, arguments.url)

    print("\n=== Endpoints now ===")
    for endpoint in stripe.WebhookEndpoint.list(limit=100)["data"]:
        marker = " <-- this one" if endpoint["url"] == arguments.url else ""
        print(
            f"  {endpoint['status']:9} {endpoint['url']} "
            f"({len(subscribed_event_names(endpoint))} events){marker}"
        )

    if signing_secret is None:
        print(
            "\nNo new secret to report — the endpoint already existed, and Stripe "
            "returns a signing secret only when an endpoint is created. Keep the "
            "STRIPE_WEBHOOK_SECRET you already have, or re-run with --rotate if "
            "it is lost."
        )
        return

    # File handoff, mirroring provision_stripe_billing.py: when
    # STRIPE_WEBHOOK_SECRET_FILE is set, also write the secret there so an
    # operator can wire it up without copying it through a terminal.
    secret_file = os.environ.get("STRIPE_WEBHOOK_SECRET_FILE")
    if secret_file:
        try:
            os.makedirs(os.path.dirname(secret_file) or ".", exist_ok=True)
            temporary_path = f"{secret_file}.tmp"
            with open(temporary_path, "w", encoding="utf-8") as handle:
                handle.write(signing_secret)
            os.replace(temporary_path, secret_file)
            print(f"\n=== Wrote signing secret to {secret_file} ===")
        except OSError as write_error:
            print(f"\n(could not write {secret_file}: {write_error})", file=sys.stderr)

    print("\n=== DONE. Set STRIPE_WEBHOOK_SECRET to the following value: ===\n")
    print(signing_secret)
    print(
        "\nStore it in the env file that stack loads (.env for production) and "
        "restart the API. Verify with an unsigned POST to the endpoint: 503 means "
        "still unset, 400 (invalid signature) means configured and verifying."
    )


if __name__ == "__main__":
    main()
