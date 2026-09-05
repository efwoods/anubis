"""Unit tests for authenticating with the refresh token POST /login returns.

Every authenticated endpoint accepts either credential: the long-lived ``API-KEY``
header that integrations hold, or the refresh token that a browser session holds
and sends as ``Authorization: Bearer``. The API key path is unchanged and is
asserted here only as a regression guard; everything else covers the refresh-token
path.

Three properties get the most attention, because breaking any of them is silent:

* An Auth0 refresh token carries no claims, so resolving one costs a token
  exchange plus a Management API read. The session is cached, and a cache hit must
  perform NEITHER upstream call.
* A refresh-token session has no API key, but the LangGraph call sites in
  ``src/api/webapp.py`` require one. The resolver mints an ephemeral key and seeds
  the API-key cache with the already-resolved user, so ``get_user_with_api_key``
  answers from cache rather than querying an account that does not carry that
  key's hash. It must also be seeded BEFORE personal-avatar provisioning runs, for
  the same re-entrancy reason the API-key path documents.
* Ending a session must drop the cached copy. Otherwise a token revoked at Auth0
  would keep authenticating until the cache entry expired on its own.
"""

from types import SimpleNamespace

import pytest

from src.anubis.utils import personal_avatar as personal_avatar_module
from src.security import auth as auth_module

REFRESH_TOKEN = "v1.MdRefreshTokenFromLogin"
REAL_API_KEY = "sk-the-key-shown-once-at-signup"
AUTH0_SUBJECT = "auth0|6a5e59310832afadd626e583"
BARE_USER_ID = "6a5e59310832afadd626e583"


def _account(email_verified=True):
    return {
        "user_id": AUTH0_SUBJECT,
        "email": "person@example.com",
        "email_verified": email_verified,
        "identities": [{"user_id": BARE_USER_ID}],
        # Auth0 holds the HASH of the key, never the key, which is exactly why a
        # refresh-token session cannot recover one and needs an ephemeral stand-in.
        "app_metadata": {
            "api_key": auth_module._hash_key(REAL_API_KEY),
            "logged_in": True,
        },
    }


class _Auth0Double:
    """Stands in for the Auth0 endpoints the resolver touches, counting calls."""

    def __init__(self, account, token_status_code=200, id_token="id-token"):
        self.account = account
        self.token_status_code = token_status_code
        self.id_token = id_token
        self.token_exchanges = []
        self.account_reads = []

    async def post(self, url, json=None, headers=None):
        self.token_exchanges.append(json)
        body = {"id_token": self.id_token} if self.id_token else {}
        return SimpleNamespace(
            status_code=self.token_status_code, json=lambda: body
        )

    async def get(self, url, params=None, headers=None):
        self.account_reads.append(url)
        # Two different Auth0 reads land here and their shapes differ: the
        # Management API user SEARCH (used by the API-key path, carries params)
        # answers with a list, while reading one user by id answers with the
        # object. The API-key search only matches when the queried hash is one
        # this account actually carries.
        if params is not None:
            queried_hash = params.get("q", "")
            matches = self.account["app_metadata"]["api_key"] in queried_hash
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: [self.account] if matches else [],
            )
        return SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: self.account
        )


def _request_for(auth0_double):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(httpx_client=auth0_double))
    )


@pytest.fixture(autouse=True)
def isolated_caches():
    auth_module._api_key_cache.clear()
    auth_module._refresh_token_cache.clear()
    yield
    auth_module._api_key_cache.clear()
    auth_module._refresh_token_cache.clear()


@pytest.fixture(autouse=True)
def stubbed_side_effects(monkeypatch):
    """Silence the two non-fatal side effects so tests assert authentication only."""

    async def _no_enrollment(request, user):
        return None

    async def _no_provisioning(request, user, api_key):
        return None

    monkeypatch.setattr(
        auth_module, "ensure_initial_subscription_after_verification", _no_enrollment
    )
    monkeypatch.setattr(
        personal_avatar_module, "ensure_personal_avatar_for_user", _no_provisioning
    )
    monkeypatch.setattr(
        auth_module, "jwt", SimpleNamespace(get_unverified_claims=lambda token: {"sub": AUTH0_SUBJECT})
    )


@pytest.mark.asyncio
async def test_a_refresh_token_resolves_to_its_account():
    auth0 = _Auth0Double(_account())

    user = await auth_module.get_user_with_refresh_token(
        REFRESH_TOKEN, _request_for(auth0)
    )

    assert user["user_id"] == AUTH0_SUBJECT
    assert auth0.token_exchanges[0]["grant_type"] == "refresh_token"
    assert auth0.token_exchanges[0]["refresh_token"] == REFRESH_TOKEN


@pytest.mark.asyncio
async def test_the_session_carries_an_api_key_that_resolves_to_the_same_user():
    """The ephemeral key must satisfy the LangGraph call sites in webapp.py."""
    auth0 = _Auth0Double(_account())
    request = _request_for(auth0)

    user = await auth_module.get_user_with_refresh_token(REFRESH_TOKEN, request)
    ephemeral_api_key = user["API_KEY"]

    assert ephemeral_api_key
    # Resolving that key must not query Auth0 for an account carrying its hash —
    # no account does. It comes back from the seeded cache as the same user.
    account_reads_before = len(auth0.account_reads)
    resolved_again = await auth_module.get_user_with_api_key(
        ephemeral_api_key, request
    )
    assert resolved_again["user_id"] == AUTH0_SUBJECT
    assert len(auth0.account_reads) == account_reads_before


@pytest.mark.asyncio
async def test_a_cached_session_costs_no_upstream_calls():
    auth0 = _Auth0Double(_account())
    request = _request_for(auth0)

    await auth_module.get_user_with_refresh_token(REFRESH_TOKEN, request)
    await auth_module.get_user_with_refresh_token(REFRESH_TOKEN, request)

    assert len(auth0.token_exchanges) == 1
    assert len(auth0.account_reads) == 1


@pytest.mark.asyncio
async def test_a_cached_session_reseeds_an_expired_api_key_entry():
    """The two caches expire independently; the session must survive that."""
    auth0 = _Auth0Double(_account())
    request = _request_for(auth0)

    user = await auth_module.get_user_with_refresh_token(REFRESH_TOKEN, request)
    auth_module._api_key_cache.clear()

    user_again = await auth_module.get_user_with_refresh_token(REFRESH_TOKEN, request)

    assert (
        auth_module._hash_key(user_again["API_KEY"]) in auth_module._api_key_cache
    )
    assert user_again["API_KEY"] == user["API_KEY"]


@pytest.mark.asyncio
async def test_an_unverified_account_is_rejected_and_never_cached():
    auth0 = _Auth0Double(_account(email_verified=False))
    request = _request_for(auth0)

    with pytest.raises(auth_module.HTTPException) as rejection:
        await auth_module.get_user_with_refresh_token(REFRESH_TOKEN, request)

    assert rejection.value.status_code == 401
    assert "not yet verified" in rejection.value.detail
    # Nothing cached, so the next poll re-reads Auth0 and sees the flag flip the
    # moment the user follows the link in the verification email.
    assert not auth_module._refresh_token_cache


@pytest.mark.asyncio
async def test_an_unverified_account_may_still_authenticate_where_allowed():
    """/resend_verification_email, /verify_login_status and /logout need this."""
    auth0 = _Auth0Double(_account(email_verified=False))

    user = await auth_module.get_user_with_refresh_token(
        REFRESH_TOKEN, _request_for(auth0), require_verified_email=False
    )

    assert user["user_id"] == AUTH0_SUBJECT
    assert user["API_KEY"]
    assert not auth_module._refresh_token_cache


@pytest.mark.asyncio
async def test_an_unverified_session_key_is_not_a_way_around_the_verified_gate():
    """The cache is consulted BEFORE the verified-email check, so it must stay clean.

    An unverified session still carries an API_KEY value because /delete_user reads
    one, but seeding it would let that key sail past a gate the account's real key
    fails. It must fail closed instead.
    """
    auth0 = _Auth0Double(_account(email_verified=False))
    request = _request_for(auth0)

    user = await auth_module.get_user_with_refresh_token(
        REFRESH_TOKEN, request, require_verified_email=False
    )

    assert auth_module._hash_key(user["API_KEY"]) not in auth_module._api_key_cache
    # Resolving it goes upstream like any unknown key. No account carries its hash,
    # so it resolves to nothing, which every dependency reports as a 401.
    assert await auth_module.get_user_with_api_key(user["API_KEY"], request) is None


@pytest.mark.asyncio
async def test_a_revoked_or_unknown_token_is_not_authenticated():
    auth0 = _Auth0Double(_account(), token_status_code=403)

    user = await auth_module.get_user_with_refresh_token(
        REFRESH_TOKEN, _request_for(auth0)
    )

    assert user is None
    assert not auth0.account_reads


@pytest.mark.asyncio
async def test_the_api_key_cache_is_warm_before_provisioning_runs(monkeypatch):
    """Same re-entrancy ordering the API-key path guarantees.

    Provisioning calls back into this API and that nested call authenticates with
    the ephemeral key, which therefore has to resolve before provisioning starts.
    """
    observed_cache_state = []

    async def _observe_cache(request, user, api_key):
        observed_cache_state.append(
            auth_module._hash_key(api_key) in auth_module._api_key_cache
        )
        return None

    monkeypatch.setattr(
        personal_avatar_module, "ensure_personal_avatar_for_user", _observe_cache
    )
    auth0 = _Auth0Double(_account())

    await auth_module.get_user_with_refresh_token(REFRESH_TOKEN, _request_for(auth0))

    assert observed_cache_state == [True]


@pytest.mark.asyncio
async def test_the_dependency_accepts_either_credential():
    """What the browser and the integrations respectively send."""
    auth0 = _Auth0Double(_account())
    request = _request_for(auth0)
    bearer = SimpleNamespace(scheme="Bearer", credentials=REFRESH_TOKEN)

    from_refresh_token = await auth_module.get_current_user(
        request, api_key=None, bearer_credentials=bearer
    )
    from_api_key = await auth_module.get_current_user(
        request, api_key=REAL_API_KEY, bearer_credentials=None
    )

    assert from_refresh_token["user_id"] == AUTH0_SUBJECT
    assert from_api_key["user_id"] == AUTH0_SUBJECT


@pytest.mark.asyncio
async def test_the_dependency_rejects_a_request_with_no_credential():
    auth0 = _Auth0Double(_account())

    with pytest.raises(auth_module.HTTPException) as rejection:
        await auth_module.get_current_user(
            _request_for(auth0), api_key=None, bearer_credentials=None
        )

    assert rejection.value.status_code == 401
    assert not auth0.token_exchanges


@pytest.mark.asyncio
async def test_a_dead_session_is_reported_as_a_session_problem():
    """The message has to name the credential the caller actually sent."""
    auth0 = _Auth0Double(_account(), token_status_code=403)
    bearer = SimpleNamespace(scheme="Bearer", credentials=REFRESH_TOKEN)

    with pytest.raises(auth_module.HTTPException) as rejection:
        await auth_module.get_current_user(
            _request_for(auth0), api_key=None, bearer_credentials=bearer
        )

    assert rejection.value.status_code == 401
    assert "log in again" in rejection.value.detail


@pytest.mark.asyncio
async def test_ending_a_session_drops_its_cached_copy(monkeypatch):
    """A token revoked at Auth0 must stop authenticating immediately."""
    auth0 = _Auth0Double(_account())
    request = _request_for(auth0)
    await auth_module.get_user_with_refresh_token(REFRESH_TOKEN, request)
    assert auth_module._refresh_token_cache

    async def _fake_mgmt_headers(request):
        return {}

    async def _fake_patch(url, json=None, headers=None):
        return SimpleNamespace(raise_for_status=lambda: None)

    monkeypatch.setattr(auth_module, "_mgmt_headers", _fake_mgmt_headers)
    auth0.patch = _fake_patch

    await auth_module.set_login_status(AUTH0_SUBJECT, logged_in=False, request=request)

    assert not auth_module._refresh_token_cache
    assert not auth_module._api_key_cache


@pytest.mark.asyncio
async def test_verify_login_status_names_the_account_email():
    status = await auth_module.verify_login_status(
        request=SimpleNamespace(),
        current_user=_account(),
    )
    assert status["email"] == "person@example.com"
    assert status["logged_in"] is True
    assert status["email_verified"] is True
