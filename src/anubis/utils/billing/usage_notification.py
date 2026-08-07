# src/anubis/utils/billing/usage_notification.py

"""Push a customer's usage to the customer portal the moment it changes.

Stripe's Billing Meter aggregation is the source of truth for what a customer is
billed, but it does not reflect an event immediately, so a portal that reads only
Stripe always shows a number lagging behind the message that just went through.
This module closes that gap: after a turn is metered, the API posts the caller's
new cumulative usage straight to the portal, which serves ``max(pushed, stripe)``
until Stripe's own aggregate catches up and overtakes it. Stripe still governs
what is billed; this only governs what is *displayed*, and the two converge.

Design points that matter:

* **The value is cumulative, never a delta.** A delta would need exactly-once
  delivery to stay correct, which a fire-and-forget POST deliberately is not.
  Re-delivering a cumulative figure is harmless, so duplicates and retries cannot
  double-count.
* **The figure is the reconciled one** already computed by
  ``_build_meter_usage_snapshot`` (``resolve_period_usage_to_date``), so the
  portal, the ``/verify_subscription_status`` response, and the 402 gate are all
  quoting one number.
* **Fail-open, always.** Delivery runs as a detached task with a short timeout and
  swallows every error. The portal being slow, down, or misconfigured must never
  delay or break a customer's message. A missed push is self-correcting: the
  portal keeps reading Stripe on its own schedule.
* **Signed.** The portal endpoint mutates what customers are shown about their
  own spending, so the body is authenticated with an HMAC over a timestamp and
  the exact bytes sent, the same construction Stripe uses for its webhooks and
  that ``webapp.py`` already verifies on the way in.

Disabled by omission: with no ``PORTAL_USAGE_EVENT_URL`` or
``PORTAL_USAGE_EVENT_SECRET`` configured, every call here is a no-op.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from typing import Any

from src.anubis.utils.context import GlobalContext

logger = logging.getLogger(__name__)

_DELIVERY_TIMEOUT_SECONDS = 3.0

TIMESTAMP_HEADER_NAME = "X-Neural-Nexus-Usage-Timestamp"
SIGNATURE_HEADER_NAME = "X-Neural-Nexus-Usage-Signature"


def build_usage_event_signature(
    shared_secret: str, timestamp: str, body: bytes
) -> str:
    """Return the hex HMAC-SHA256 over ``timestamp.body``.

    Binding the timestamp into the signed material is what makes the timestamp
    itself unforgeable, so the receiver can reject a replayed body by age. The
    portal reproduces this exact construction; changing it on one side alone
    silently rejects every event.
    """
    signed_payload = timestamp.encode("utf-8") + b"." + body
    return hmac.new(
        shared_secret.encode("utf-8"), signed_payload, hashlib.sha256
    ).hexdigest()


async def _deliver_usage_event(
    portal_usage_event_url: str, shared_secret: str, event_document: dict[str, Any]
) -> None:
    # Imported here rather than at module scope to keep httpx off the cold-start
    # import path, matching the lazy-import convention used across this package.
    import httpx

    body = json.dumps(event_document, separators=(",", ":")).encode("utf-8")
    timestamp = str(int(time.time()))
    headers = {
        "Content-Type": "application/json",
        TIMESTAMP_HEADER_NAME: timestamp,
        SIGNATURE_HEADER_NAME: build_usage_event_signature(
            shared_secret, timestamp, body
        ),
    }
    try:
        async with httpx.AsyncClient(timeout=_DELIVERY_TIMEOUT_SECONDS) as http_client:
            response = await http_client.post(
                portal_usage_event_url, content=body, headers=headers
            )
        if response.status_code >= 400:
            logger.warning(
                "Customer portal rejected a usage event (HTTP %s); the portal will "
                "fall back to its own Stripe read: %s",
                response.status_code,
                response.text[:200],
            )
    except Exception as delivery_error:  # noqa: BLE001 - display-only, never fatal
        logger.warning(
            "Could not deliver a usage event to the customer portal (%s); the "
            "portal will fall back to its own Stripe read.",
            delivery_error,
        )


def schedule_usage_notification(
    *,
    stripe_customer_id: str | None,
    meter_event_name: str | None,
    cumulative_period_usage: int | None,
    usage_period_start: str | None,
    usage_period_end: str | None,
) -> None:
    """Fire off one usage event without waiting for it.

    Returns immediately in every case. Called from the metering path, which runs
    after the model has already replied, so nothing here is allowed to add
    latency or raise.
    """
    if not stripe_customer_id or not meter_event_name:
        return
    if cumulative_period_usage is None:
        return

    context = GlobalContext()
    portal_usage_event_url = (context.portal_usage_event_url or "").strip()
    shared_secret = (context.portal_usage_event_secret or "").strip()
    if not portal_usage_event_url or not shared_secret:
        return

    event_document: dict[str, Any] = {
        "stripe_customer_id": stripe_customer_id,
        "meter_event_name": meter_event_name,
        "cumulative_period_usage": int(cumulative_period_usage),
        "usage_period_start": usage_period_start,
        "usage_period_end": usage_period_end,
    }

    try:
        task = asyncio.create_task(
            _deliver_usage_event(
                portal_usage_event_url, shared_secret, event_document
            )
        )
        # Hold a reference until completion so the task is not garbage collected
        # mid-flight, and drop it afterwards so the set cannot grow unbounded.
        _in_flight_deliveries.add(task)
        task.add_done_callback(_in_flight_deliveries.discard)
    except RuntimeError:
        # No running event loop (a synchronous caller or interpreter shutdown).
        # Usage display is not worth spinning up a loop for.
        return


_in_flight_deliveries: set[asyncio.Task] = set()
