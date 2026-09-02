"""Unit tests for customer-portal single sign-on.

The billing page embeds the customer portal, and a user already signed in to the
web application should not sign in again inside that frame. The handoff is an
exchange code: this API mints one for an authenticated account, the web
application posts it to the portal frame, and the portal server spends it here
to learn which account it is serving.

What is asserted, in the order the properties matter:

* A minted code names the account, and spending it returns that account. This is
  the whole feature.
* A code is spendable exactly once, and only before it expires. Both are what
  make the code safe to put through a browser and across an origin boundary.
* A code minted under a different secret, for a different audience, or by a
  different issuer is refused — a signature check that accepts a token meant for
  something else is not a signature check.
* The redemption endpoint's signature covers a timestamp, and an old or
  mismatched signature is refused. Without this, anyone who observed a code
  could turn it into a customer's email address.
"""

import time

import pytest
from jose import jwt

from src.security import billing_portal_single_sign_on as single_sign_on

SHARED_SECRET = "portal-and-api-share-this-exact-value"
OTHER_SECRET = "a-different-deployment-secret"
AUTH0_SUBJECT = "auth0|6a5e59310832afadd626e583"
EMAIL = "person@example.com"


@pytest.fixture(autouse=True)
def clear_redeemed_codes():
    """Single-use state is module-level, so one test must not spend another's
    codes for it."""
    single_sign_on._redeemed_exchange_code_identifiers.clear()
    yield
    single_sign_on._redeemed_exchange_code_identifiers.clear()


def test_a_minted_code_redeems_to_the_account_it_was_minted_for():
    exchange_code, expires_in_seconds = (
        single_sign_on.mint_billing_portal_exchange_code(
            SHARED_SECRET, AUTH0_SUBJECT, EMAIL
        )
    )

    assert expires_in_seconds == single_sign_on.EXCHANGE_CODE_TTL_SECONDS

    account = single_sign_on.redeem_billing_portal_exchange_code(
        SHARED_SECRET, exchange_code
    )

    assert account == {"user_id": AUTH0_SUBJECT, "email": EMAIL}


def test_the_code_carries_no_neural_nexus_credential():
    """The reason this feature mints a code instead of forwarding the session
    credential: nothing that authenticates against this API may cross into the
    portal's origin."""
    exchange_code, _ = single_sign_on.mint_billing_portal_exchange_code(
        SHARED_SECRET, AUTH0_SUBJECT, EMAIL
    )

    claims = jwt.get_unverified_claims(exchange_code)

    assert set(claims) == {"iss", "aud", "sub", "email", "iat", "exp", "jti"}


def test_a_code_can_be_redeemed_only_once():
    exchange_code, _ = single_sign_on.mint_billing_portal_exchange_code(
        SHARED_SECRET, AUTH0_SUBJECT, EMAIL
    )
    single_sign_on.redeem_billing_portal_exchange_code(SHARED_SECRET, exchange_code)

    with pytest.raises(single_sign_on.BillingPortalExchangeError) as refusal:
        single_sign_on.redeem_billing_portal_exchange_code(SHARED_SECRET, exchange_code)

    assert refusal.value.status_code == 400


def test_two_codes_for_one_account_are_independently_spendable():
    """A user may have the billing page open in two tabs; the second tab's code
    must not be invalidated by the first tab spending its own."""
    first_code, _ = single_sign_on.mint_billing_portal_exchange_code(
        SHARED_SECRET, AUTH0_SUBJECT, EMAIL
    )
    second_code, _ = single_sign_on.mint_billing_portal_exchange_code(
        SHARED_SECRET, AUTH0_SUBJECT, EMAIL
    )

    single_sign_on.redeem_billing_portal_exchange_code(SHARED_SECRET, first_code)
    account = single_sign_on.redeem_billing_portal_exchange_code(
        SHARED_SECRET, second_code
    )

    assert account["email"] == EMAIL


def test_an_expired_code_is_refused():
    expired_claims = {
        "iss": single_sign_on.EXCHANGE_CODE_ISSUER,
        "aud": single_sign_on.EXCHANGE_CODE_AUDIENCE,
        "sub": AUTH0_SUBJECT,
        "email": EMAIL,
        "iat": int(time.time()) - 600,
        "exp": int(time.time()) - 60,
        "jti": "an-expired-code",
    }
    expired_code = jwt.encode(
        expired_claims,
        SHARED_SECRET,
        algorithm=single_sign_on.EXCHANGE_CODE_ALGORITHM,
    )

    with pytest.raises(single_sign_on.BillingPortalExchangeError) as refusal:
        single_sign_on.redeem_billing_portal_exchange_code(SHARED_SECRET, expired_code)

    assert refusal.value.status_code == 400


def test_a_code_signed_with_another_secret_is_refused():
    forged_code, _ = single_sign_on.mint_billing_portal_exchange_code(
        OTHER_SECRET, AUTH0_SUBJECT, EMAIL
    )

    with pytest.raises(single_sign_on.BillingPortalExchangeError):
        single_sign_on.redeem_billing_portal_exchange_code(SHARED_SECRET, forged_code)


@pytest.mark.parametrize(
    "audience, issuer",
    [
        ("some-other-application", single_sign_on.EXCHANGE_CODE_ISSUER),
        (single_sign_on.EXCHANGE_CODE_AUDIENCE, "some-other-issuer"),
    ],
)
def test_a_token_minted_for_something_else_is_refused(audience, issuer):
    """Holding the shared secret is not enough: a token has to have been minted
    as an exchange code, for this portal, by this API."""
    foreign_token = jwt.encode(
        {
            "iss": issuer,
            "aud": audience,
            "sub": AUTH0_SUBJECT,
            "email": EMAIL,
            "iat": int(time.time()),
            "exp": int(time.time()) + 120,
            "jti": "a-token-for-something-else",
        },
        SHARED_SECRET,
        algorithm=single_sign_on.EXCHANGE_CODE_ALGORITHM,
    )

    with pytest.raises(single_sign_on.BillingPortalExchangeError):
        single_sign_on.redeem_billing_portal_exchange_code(SHARED_SECRET, foreign_token)


def test_a_correctly_signed_redemption_request_is_accepted():
    body = b'{"exchange_code":"whatever"}'
    timestamp = str(int(time.time()))
    signature = single_sign_on.build_redemption_signature(
        SHARED_SECRET, timestamp, body
    )

    single_sign_on.verify_redemption_signature(
        SHARED_SECRET, timestamp, signature, body
    )


@pytest.mark.parametrize(
    "description, timestamp_offset_seconds, corrupt_body",
    [
        ("a replayed request", -(60 * 60), False),
        ("a body altered after signing", 0, True),
    ],
)
def test_redemption_signature_refuses(
    description, timestamp_offset_seconds, corrupt_body
):
    body = b'{"exchange_code":"whatever"}'
    timestamp = str(int(time.time()) + timestamp_offset_seconds)
    signature = single_sign_on.build_redemption_signature(
        SHARED_SECRET, timestamp, body
    )
    delivered_body = b'{"exchange_code":"a-different-code"}' if corrupt_body else body

    with pytest.raises(single_sign_on.BillingPortalExchangeError) as refusal:
        single_sign_on.verify_redemption_signature(
            SHARED_SECRET, timestamp, signature, delivered_body
        )

    assert refusal.value.status_code == 401, description


def test_an_unsigned_redemption_request_is_refused():
    with pytest.raises(single_sign_on.BillingPortalExchangeError) as refusal:
        single_sign_on.verify_redemption_signature(
            SHARED_SECRET, None, None, b'{"exchange_code":"whatever"}'
        )

    assert refusal.value.status_code == 401
