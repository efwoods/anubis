"""SMS one-time codes that prove the owner still has the mobile they typed.

Codes are stored as SHA-256 digests with a short lifetime. Twilio (or any
later sender) is optional: when no sender is configured the code is still
issued so tests and local development can complete the form. Production
refuses to send a code it cannot deliver unless ``phone_verify_allow_undelivered``
is true.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Any

from src.anubis.utils.phone.numbers import PhoneNumberError, normalize_e164

logger = logging.getLogger(__name__)

CODE_LIFETIME_SECONDS = 600
CODE_DIGITS = 6

_pending: dict[str, "_PendingCode"] = {}


@dataclass
class _PendingCode:
    digest: str
    expires_at: float


def _digest_of(code: str) -> str:
    return hashlib.sha256(str(code).strip().encode("utf-8")).hexdigest()


def issue_verification_code(phone_e164: str) -> str:
    """Store a fresh code for ``phone_e164`` and return the plaintext once."""
    normalized = normalize_e164(phone_e164)
    code = f"{secrets.randbelow(10**CODE_DIGITS):0{CODE_DIGITS}d}"
    _pending[normalized] = _PendingCode(
        digest=_digest_of(code),
        expires_at=time.time() + CODE_LIFETIME_SECONDS,
    )
    return code


def store_verification_code(phone_e164: str, code: str) -> None:
    """Put a known code in the pending map (tests)."""
    normalized = normalize_e164(phone_e164)
    _pending[normalized] = _PendingCode(
        digest=_digest_of(code),
        expires_at=time.time() + CODE_LIFETIME_SECONDS,
    )


def check_verification_code(phone_e164: str, code: str) -> bool:
    """Return True when ``code`` matches the pending digest and has not expired."""
    normalized = normalize_e164(phone_e164)
    pending = _pending.get(normalized)
    if pending is None:
        return False
    if time.time() > pending.expires_at:
        _pending.pop(normalized, None)
        return False
    if pending.digest != _digest_of(code):
        return False
    _pending.pop(normalized, None)
    return True


def clear_verification_codes() -> None:
    """Drop every pending code (tests)."""
    _pending.clear()


async def send_verification_sms(
    phone_e164: str, code: str, context: Any
) -> str:
    """Deliver ``code`` to ``phone_e164``.

    Returns ``"twilio"`` when Twilio accepted the message, ``"logged"`` when
    no sender is configured and undelivered codes are allowed. Raises
    ``PhoneNumberError`` when delivery is required and no sender is set.
    """
    normalized = normalize_e164(phone_e164)
    account_sid = str(getattr(context, "twilio_account_sid", None) or "").strip()
    auth_token = str(getattr(context, "twilio_auth_token", None) or "").strip()
    from_number = str(getattr(context, "twilio_from_number", None) or "").strip()
    if account_sid and auth_token and from_number:
        try:
            import httpx

            url = (
                f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}"
                "/Messages.json"
            )
            timeout = float(
                getattr(context, "phone_verify_http_timeout_seconds", None) or 15.0
            )
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    url,
                    auth=(account_sid, auth_token),
                    data={
                        "To": normalized,
                        "From": from_number,
                        "Body": (
                            f"Neural Nexus confirmation code: {code}. "
                            "It expires in ten minutes."
                        ),
                    },
                )
            if response.status_code >= 400:
                raise PhoneNumberError(
                    "The confirmation text could not be sent. Try again in a moment."
                )
            return "twilio"
        except PhoneNumberError:
            raise
        except Exception as send_error:
            logger.warning("Twilio SMS send failed: %s", send_error)
            raise PhoneNumberError(
                "The confirmation text could not be sent. Try again in a moment."
            ) from send_error

    allow_undelivered = str(
        getattr(context, "phone_verify_allow_undelivered", None) or ""
    ).strip().upper() in {"TRUE", "1", "YES"}
    if allow_undelivered:
        logger.info("Phone verification code for %s issued without SMS delivery.", normalized)
        return "logged"
    raise PhoneNumberError(
        "Phone verification is not configured. Set TWILIO_ACCOUNT_SID, "
        "TWILIO_AUTH_TOKEN, and TWILIO_FROM_NUMBER."
    )
