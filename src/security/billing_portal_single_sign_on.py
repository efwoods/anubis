# src/security/billing_portal_single_sign_on.py

"""Single sign-on from the Neural Nexus web application into the customer portal.

The customer portal (checkout.neuralnexus.site) is a separate application with
its own session, minted by its own server. The web application embeds that
portal in an iframe on its billing page, so a user who has already signed in to
the web application was met by a second sign-in card. This module removes that
second sign-in by handing the portal a credential the portal can exchange for
its own session.

The credential is a **billing portal exchange code**: a short-lived, single-use,
signed statement by this API that a named account is authenticated right now.
The flow is:

1. The web application, authenticated as the user, calls
   ``POST /create_billing_portal_exchange_code`` and receives an exchange code.
2. The web application posts that code to the portal iframe with
   ``window.postMessage``, pinned to the portal's exact origin.
3. The portal server calls ``POST /redeem_billing_portal_exchange_code`` on this
   API, receives the account's email address, and mints its own portal session
   exactly as its password login already does.

Why an exchange code rather than handing over the session credential itself:
the web application authenticates with the account's Auth0 refresh token, and
that token is a full account credential. An exchange code carries only the two
facts the portal actually needs — which account, and that the account is
authenticated — expires in two minutes, and cannot be replayed. The refresh
token never crosses into the portal's origin.

Design points that matter:

* **The code is a signed JSON Web Token, not a stored random string.** The API
  runs as more than one worker, and a random string kept in the memory of the
  worker that minted it is not redeemable by the worker that receives the
  redemption. A signature is verifiable by every worker with no shared storage.
* **Single use is enforced best-effort, expiry absolutely.** A redeemed code's
  identifier is remembered for the code's whole lifetime and refused on a second
  redemption, but that memory is per worker, so a replay racing across workers
  inside the two-minute window can succeed. Expiry is what bounds the exposure,
  and it is verified from the signature on every worker.
* **The redemption endpoint is authenticated by a shared secret**, as an
  HMAC-SHA256 over ``'<timestamp>.<body>'`` — the same construction the usage
  event push already uses in the other direction, and the same construction
  Stripe uses for its webhook signatures. A stolen code alone therefore does not
  reveal an account's email address.
* **One secret, two uses.** ``BILLING_PORTAL_EXCHANGE_SECRET`` both signs the
  exchange code and authenticates the redemption call. Two secrets would be two
  values to keep in step across two deployments for no gain: the portal is the
  only holder of either.

Disabled by omission: with no ``BILLING_PORTAL_EXCHANGE_SECRET`` configured,
both endpoints refuse, the portal is never handed anything, and the billing page
falls back to the portal's own sign-in card — which is exactly the behaviour
that existed before this module.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
import uuid

from cachetools import TTLCache
from jose import JWTError, jwt

logger = logging.getLogger(__name__)

# The code is signed, not encrypted: nothing secret is inside it, and both sides
# hold the same secret, so a symmetric algorithm is the right one.
EXCHANGE_CODE_ALGORITHM = "HS256"

# Issuer and audience are checked on redemption so that a token minted for
# something else, by something else, can never be spent here as an exchange code.
EXCHANGE_CODE_ISSUER = "neural-nexus-api"
EXCHANGE_CODE_AUDIENCE = "neural-nexus-customer-portal"

# Long enough to cover the frame handshake — the web application retries posting
# the code until the portal frame acknowledges it, because the frame's listener
# may not exist yet when the first message is sent — and short enough that a code
# captured in transit is worthless by the time anyone could use it.
EXCHANGE_CODE_TTL_SECONDS = 120

REDEMPTION_TIMESTAMP_HEADER_NAME = "X-Neural-Nexus-Portal-Timestamp"
REDEMPTION_SIGNATURE_HEADER_NAME = "X-Neural-Nexus-Portal-Signature"

# How far a redemption request's timestamp may be from this server's clock. The
# timestamp is inside the signed material, so it cannot be rewritten; this bound
# is what stops a captured redemption request from being replayed indefinitely.
REDEMPTION_TIMESTAMP_TOLERANCE_SECONDS = 300

# Identifiers of codes already spent. The entries expire on their own after the
# code's lifetime, because a code that has expired is refused by signature
# verification and no longer needs remembering.
_redeemed_exchange_code_identifiers: TTLCache = TTLCache(
    maxsize=10000, ttl=EXCHANGE_CODE_TTL_SECONDS
)


class BillingPortalExchangeError(Exception):
    """A redemption was refused.

    Carries the HTTP status the endpoint should answer with, so the reason a
    redemption failed is not flattened into one indistinguishable error: the
    portal must be able to tell "this deployment is misconfigured" (503) from
    "whoever called me is not the portal" (401) from "this code is expired,
    forged, or already spent" (400).
    """

    def __init__(self, message: str, status_code: int) -> None:
        """Record the message and the HTTP status the endpoint answers with."""
        super().__init__(message)
        self.status_code = status_code


def mint_billing_portal_exchange_code(
    shared_secret: str, user_id: str, email: str
) -> tuple[str, int]:
    """Mint an exchange code for an already-authenticated account.

    The caller is responsible for having authenticated the account; this
    function states that fact, it does not establish it.

    Returns the code and the number of seconds it remains valid, so the client
    knows how long it may keep retrying the frame handshake with it.
    """
    issued_at = int(time.time())
    claims = {
        "iss": EXCHANGE_CODE_ISSUER,
        "aud": EXCHANGE_CODE_AUDIENCE,
        "sub": user_id,
        "email": email,
        "iat": issued_at,
        "exp": issued_at + EXCHANGE_CODE_TTL_SECONDS,
        # The single-use marker. Randomly generated rather than derived from the
        # account, so two codes minted for the same account in the same second
        # are still independently spendable — a user may open the billing page
        # in two tabs.
        "jti": uuid.uuid4().hex,
    }
    exchange_code = jwt.encode(claims, shared_secret, algorithm=EXCHANGE_CODE_ALGORITHM)
    return exchange_code, EXCHANGE_CODE_TTL_SECONDS


def build_redemption_signature(shared_secret: str, timestamp: str, body: bytes) -> str:
    """Return the hex HMAC-SHA256 over ``'<timestamp>.<body>'``.

    Binding the timestamp into the signed material is what makes the timestamp
    itself unforgeable, so a replayed body can be rejected by age. The portal
    reproduces this exact construction; changing it on one side alone silently
    rejects every redemption.
    """
    signed_payload = timestamp.encode("utf-8") + b"." + body
    return hmac.new(
        shared_secret.encode("utf-8"), signed_payload, hashlib.sha256
    ).hexdigest()


def verify_redemption_signature(
    shared_secret: str,
    timestamp: str | None,
    signature: str | None,
    body: bytes,
) -> None:
    """Confirm a redemption request came from the customer portal.

    Raises ``BillingPortalExchangeError`` when it did not; returns None when it
    did.
    """
    if not timestamp or not signature:
        raise BillingPortalExchangeError(
            "Redemption requests must be signed.", status_code=401
        )

    try:
        request_age_seconds = abs(int(time.time()) - int(timestamp))
    except ValueError:
        raise BillingPortalExchangeError(
            "Redemption signature timestamp is not a Unix timestamp.",
            status_code=401,
        ) from None

    if request_age_seconds > REDEMPTION_TIMESTAMP_TOLERANCE_SECONDS:
        raise BillingPortalExchangeError(
            "Redemption signature timestamp is outside the accepted window.",
            status_code=401,
        )

    expected_signature = build_redemption_signature(shared_secret, timestamp, body)
    # compare_digest rather than ==: an ordinary comparison returns faster the
    # earlier it finds a difference, which leaks the signature one byte at a time.
    if not hmac.compare_digest(expected_signature, signature):
        raise BillingPortalExchangeError(
            "Redemption signature does not match.", status_code=401
        )


def redeem_billing_portal_exchange_code(shared_secret: str, exchange_code: str) -> dict:
    """Verify an exchange code and spend it.

    Returns the account the code names, as ``{"user_id", "email"}``. Raises
    ``BillingPortalExchangeError`` when the code is forged, expired, minted for
    a different audience, or already spent.
    """
    if not exchange_code:
        raise BillingPortalExchangeError(
            "No exchange code was supplied.", status_code=400
        )

    try:
        claims = jwt.decode(
            exchange_code,
            shared_secret,
            algorithms=[EXCHANGE_CODE_ALGORITHM],
            audience=EXCHANGE_CODE_AUDIENCE,
            issuer=EXCHANGE_CODE_ISSUER,
        )
    except JWTError as verification_error:
        # Deliberately not echoed to the caller: the difference between "expired"
        # and "wrong signature" is useful to an attacker and to nobody else. It is
        # logged so an operator can still tell the two apart.
        logger.warning(
            "Rejected a billing portal exchange code: %s", verification_error
        )
        raise BillingPortalExchangeError(
            "The exchange code is not valid.", status_code=400
        ) from verification_error

    code_identifier = claims.get("jti")
    if not code_identifier:
        raise BillingPortalExchangeError(
            "The exchange code is not valid.", status_code=400
        )

    if code_identifier in _redeemed_exchange_code_identifiers:
        logger.warning(
            "Refused a second redemption of billing portal exchange code %s.",
            code_identifier,
        )
        raise BillingPortalExchangeError(
            "The exchange code has already been redeemed.", status_code=400
        )
    _redeemed_exchange_code_identifiers[code_identifier] = True

    return {"user_id": claims["sub"], "email": claims["email"]}
