"""LiveKit SIP: one shared inbound trunk, owner first, then the restaurant.

Never buy a number per user. Never create a per-user dispatch rule. Inbound
uses ``PLATFORM_PHONE_NUMBER``. Outbound rings ``owner_mobile_e164`` first,
then the destination from the same trunk.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.anubis.utils.phone.numbers import PhoneNumberError, normalize_e164

logger = logging.getLogger(__name__)


class LiveKitNotConfigured(RuntimeError):
    """LiveKit URL, API key, or SIP trunk is missing."""


def livekit_is_configured(context: Any) -> bool:
    """Whether outbound SIP can be placed from this process."""
    return bool(
        str(getattr(context, "livekit_url", None) or "").strip()
        and str(getattr(context, "livekit_api_key", None) or "").strip()
        and str(getattr(context, "livekit_api_secret", None) or "").strip()
        and str(getattr(context, "livekit_sip_outbound_trunk_id", None) or "").strip()
    )


def platform_phone_number(context: Any) -> str | None:
    """The one shared inbound DID, or None when unset."""
    raw = str(getattr(context, "platform_phone_number", None) or "").strip()
    if not raw:
        return None
    try:
        return normalize_e164(raw)
    except PhoneNumberError:
        return raw


def listen_token(
    context: Any,
    *,
    room_name: str,
    identity: str,
    ttl_seconds: int = 3600,
) -> str:
    """Return a subscribe-only LiveKit access token for web listen-in."""
    api_key = str(getattr(context, "livekit_api_key", None) or "").strip()
    api_secret = str(getattr(context, "livekit_api_secret", None) or "").strip()
    if not api_key or not api_secret:
        raise LiveKitNotConfigured("LIVEKIT_API_KEY and LIVEKIT_API_SECRET are required.")
    now = int(time.time())
    claims = {
        "iss": api_key,
        "sub": identity,
        "nbf": now,
        "exp": now + max(60, int(ttl_seconds)),
        "video": {
            "roomJoin": True,
            "room": room_name,
            "canPublish": False,
            "canSubscribe": True,
            "canPublishData": False,
        },
    }
    return _hs256_jwt(claims, api_secret)


def _hs256_jwt(claims: dict[str, Any], secret: str) -> str:
    """Mint a compact HS256 JWT without requiring PyJWT at import time."""
    import base64
    import hashlib
    import hmac
    import json

    def _b64url(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")

    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _b64url(json.dumps(claims, separators=(",", ":")).encode())
    signing_input = f"{header}.{payload}".encode("ascii")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64url(signature)}"


async def create_sip_participant(
    context: Any,
    *,
    room_name: str,
    phone_e164: str,
    participant_identity: str,
    participant_name: str,
) -> dict[str, Any]:
    """Dial ``phone_e164`` into ``room_name`` on the shared outbound trunk."""
    if not livekit_is_configured(context):
        raise LiveKitNotConfigured(
            "LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, and "
            "LIVEKIT_SIP_OUTBOUND_TRUNK_ID must be set to place a call."
        )
    number = normalize_e164(phone_e164)
    trunk_id = str(getattr(context, "livekit_sip_outbound_trunk_id", None) or "").strip()
    livekit_url = str(getattr(context, "livekit_url", None) or "").rstrip("/")
    api_key = str(getattr(context, "livekit_api_key", None) or "").strip()
    api_secret = str(getattr(context, "livekit_api_secret", None) or "").strip()
    timeout = float(getattr(context, "phone_lookup_http_timeout_seconds", None) or 20.0)
    import httpx

    url = f"{livekit_url}/twirp/livekit.SIP/CreateSIPParticipant"
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            url,
            headers={"Content-Type": "application/json"},
            auth=(api_key, api_secret),
            json={
                "sip_trunk_id": trunk_id,
                "sip_call_to": number,
                "room_name": room_name,
                "participant_identity": participant_identity,
                "participant_name": participant_name,
                "wait_until_answered": False,
            },
        )
    if response.status_code >= 400:
        raise RuntimeError(
            f"LiveKit CreateSIPParticipant refused {number}: {response.text}"
        )
    try:
        return response.json()
    except Exception:
        return {"status": "accepted", "sip_call_to": number}


async def ring_owner_then_destination(
    context: Any,
    *,
    room_name: str,
    owner_mobile_e164: str,
    destination_e164: str,
) -> dict[str, Any]:
    """Ring the owner's existing mobile first, then the restaurant.

    The destination leg is still placed when the owner does not answer: web
    listen-in is the fallback. This function never fans out to an iOS device.
    """
    owner = await create_sip_participant(
        context,
        room_name=room_name,
        phone_e164=owner_mobile_e164,
        participant_identity="owner-listen",
        participant_name="Owner",
    )
    destination = await create_sip_participant(
        context,
        room_name=room_name,
        phone_e164=destination_e164,
        participant_identity="destination",
        participant_name="Destination",
    )
    return {
        "owner": owner,
        "destination": destination,
        "order": ["owner_mobile_e164", "destination_e164"],
    }
