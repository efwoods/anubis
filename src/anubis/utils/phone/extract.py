"""Pull cost, ready time, and location out of a restaurant-call transcript."""

from __future__ import annotations

import re
from typing import Any

_MONEY = re.compile(
    r"\$\s*(\d+(?:\.\d{1,2})?)|(\d+(?:\.\d{1,2})?)\s*(?:dollars?)",
    re.IGNORECASE,
)
_READY = re.compile(
    r"(?:ready|pickup|pick up|come by|come get).{0,40}?"
    r"(\d{1,2}\s*(?:min(?:ute)?s?|hours?|hr))",
    re.IGNORECASE,
)
_READY_CLOCK = re.compile(
    r"(?:ready|pickup|pick up).{0,24}?"
    r"(\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?)?|\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?))",
    re.IGNORECASE,
)
_ADDRESS = re.compile(
    r"(\d{1,5}\s+[A-Za-z0-9 .'-]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct)\b"
    r"(?:,?\s*[A-Za-z .'-]+)?)",
    re.IGNORECASE,
)


def extract_order_result(transcript: str, *, destination_name: str = "") -> dict[str, Any]:
    """Return structured fields from a hangup transcript.

    Used by the outbound worker and by tests with Kanji / Mellow Mushroom
    fixtures. Missing fields stay ``unknown`` rather than being invented.
    """
    text = str(transcript or "")
    money = _MONEY.search(text)
    cost = None
    if money:
        raw = money.group(1) or money.group(2)
        try:
            cost = float(raw)
        except (TypeError, ValueError):
            cost = None
    ready_at = "unknown"
    ready_match = _READY.search(text) or _READY_CLOCK.search(text)
    if ready_match:
        ready_at = ready_match.group(1).strip()
    location = "unknown"
    address_match = _ADDRESS.search(text)
    if address_match:
        location = address_match.group(1).strip()
    lowered = text.casefold()
    if "card" in lowered and (
        "credit" in lowered or "debit" in lowered or "card number" in lowered
    ):
        outcome = "ended_payment_required"
    elif "sorry" in lowered and ("can't" in lowered or "cannot" in lowered or "closed" in lowered):
        outcome = "refused_by_restaurant"
    elif cost is not None or ready_at != "unknown":
        outcome = "placed"
    else:
        outcome = "unknown"
    return {
        "cost": cost,
        "ready_at": ready_at,
        "location": location,
        "travel_minutes": None,
        "outcome": outcome,
        "destination_name": destination_name,
    }
