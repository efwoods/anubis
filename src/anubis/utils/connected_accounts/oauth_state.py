"""Signed, expiring state for logins that leave the API and come back in a popup.

An OAuth authorization redirect, a Plaid Link page, and a live browser sign-in
all share one problem: the browser window that returns to this API carries no
Authorization header, so the API cannot tell from the request alone which
owner, which provider, and which avatar the returning login belongs to. The
answer is a ``state`` token this module signs when the login starts and
verifies when the popup comes back. The token is opaque to the browser, cannot
be forged without the signing secret, and expires, so a stale link in a browser
history cannot complete a login days later.

Standard library only (``hmac``, ``hashlib``, ``json``, ``base64``): the token
format is small enough that a dependency would cost more than it saves, and the
prod image is Python 3.11 with no room for surprises.

The PKCE helpers live here too because every popup login that uses PKCE needs
them and nothing else does.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Any

# Separates the signed body from its signature. Both halves are base64url with
# no padding, so a "." can never appear inside either.
_SEPARATOR = "."


class OAuthStateError(ValueError):
    """The state token is missing, malformed, tampered with, or expired."""


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode((text + padding).encode("ascii"))


def state_secret(context: Any) -> bytes:
    """Return the HMAC key that signs state tokens.

    ``CONNECT_OAUTH_STATE_SECRET`` when set; otherwise the SHA-256 digest of the
    credential encryption key, so a deployment that configured encryption gets
    signed state without a second secret. An unconfigured deployment cannot
    sign anything, which surfaces as a clear error rather than a token anyone
    could forge with an empty key.
    """
    explicit = str(getattr(context, "connect_oauth_state_secret", "") or "").strip()
    if explicit:
        return hashlib.sha256(explicit.encode("utf-8")).digest()
    encryption_key = str(
        getattr(context, "connected_account_encryption_key", "") or ""
    ).strip()
    if encryption_key:
        return hashlib.sha256(
            b"connect-account-state:" + encryption_key.encode("utf-8")
        ).digest()
    raise OAuthStateError(
        "Popup logins need CONNECT_OAUTH_STATE_SECRET or "
        "CONNECTED_ACCOUNT_ENCRYPTION_KEY to be configured."
    )


def sign_state(payload: dict[str, Any], secret: bytes, max_age_seconds: int) -> str:
    """Sign ``payload`` into an opaque token that expires after ``max_age_seconds``."""
    body = dict(payload)
    body["exp"] = int(time.time()) + int(max_age_seconds)
    encoded_body = _b64encode(
        json.dumps(body, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    signature = hmac.new(secret, encoded_body.encode("ascii"), hashlib.sha256).digest()
    return encoded_body + _SEPARATOR + _b64encode(signature)


def verify_state(token: str, secret: bytes) -> dict[str, Any]:
    """Return the payload of a token signed by :func:`sign_state`, or raise.

    Signature comparison is constant-time; expiry is checked after the
    signature so an attacker learns nothing from timing about which half failed.
    """
    text = str(token or "").strip()
    if _SEPARATOR not in text:
        raise OAuthStateError("The login state is malformed.")
    encoded_body, encoded_signature = text.rsplit(_SEPARATOR, 1)
    expected = hmac.new(secret, encoded_body.encode("ascii"), hashlib.sha256).digest()
    try:
        provided = _b64decode(encoded_signature)
    except Exception as decode_error:
        raise OAuthStateError("The login state is malformed.") from decode_error
    if not hmac.compare_digest(expected, provided):
        raise OAuthStateError("The login state was not issued by this server.")
    try:
        payload = json.loads(_b64decode(encoded_body).decode("utf-8"))
    except Exception as decode_error:
        raise OAuthStateError("The login state is malformed.") from decode_error
    if not isinstance(payload, dict):
        raise OAuthStateError("The login state is malformed.")
    if int(payload.get("exp") or 0) < int(time.time()):
        raise OAuthStateError("The login took too long; start the sign-in again.")
    return payload


def random_nonce() -> str:
    """Return a 128-bit URL-safe identifier for one login attempt."""
    return secrets.token_urlsafe(16)


def make_pkce() -> tuple[str, str]:
    """Return ``(code_verifier, code_challenge)`` per RFC 7636 (S256)."""
    verifier = _b64encode(secrets.token_bytes(48))
    challenge = _b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def pkce_challenge_for(verifier: str) -> str:
    """Return the S256 challenge of a known verifier (used by tests)."""
    return _b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
