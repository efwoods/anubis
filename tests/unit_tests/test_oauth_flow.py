"""The popup sign-in: signed state, PKCE, exchange, proof, refresh, result page.

Every vendor call goes through an injected ``httpx.AsyncClient`` built on a
``MockTransport``, so nothing here reaches the network.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import httpx
import pytest

from src.anubis.utils import secret_store
from src.anubis.utils.connected_accounts import oauth_flow, oauth_state
from src.anubis.utils.connected_accounts.pending_logins import (
    InMemoryPendingLoginRepository,
)
from src.anubis.utils.connected_accounts.providers import get_provider
from src.anubis.utils.connected_accounts.store import public_account_view

USER_ID = "auth0|owner"
ASSISTANT_ID = "assistant-1"


def _context(**overrides):
    values = dict(
        connected_account_encryption_key=secret_store.generate_encryption_key(),
        connect_oauth_state_secret="",
        connect_oauth_state_max_age_seconds=600,
        connect_oauth_http_timeout_seconds=5.0,
        connect_oauth_redirect_base_url="http://localhost:9600",
        connect_oauth_popup_target_origins="http://localhost:5173,https://neuralnexus.site",
        google_oauth_client_id="google-client",
        google_oauth_client_secret="google-secret",
        github_oauth_client_id="github-client",
        github_oauth_client_secret="github-secret",
        x_oauth_client_id="",
        x_oauth_client_secret="",
        vercel_oauth_client_id="",
        vercel_oauth_client_secret="",
        mailbox_request_timeout_seconds=5.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


# --------------------------------------------------------------------------
# Signed state and PKCE
# --------------------------------------------------------------------------


def test_state_round_trips_and_rejects_tampering_and_expiry():
    secret = oauth_state.state_secret(_context())
    token = oauth_state.sign_state({"nonce": "n1", "user_id": USER_ID}, secret, 60)
    assert oauth_state.verify_state(token, secret)["nonce"] == "n1"

    body, signature = token.rsplit(".", 1)
    with pytest.raises(oauth_state.OAuthStateError):
        oauth_state.verify_state(body + ".AAAA", secret)
    with pytest.raises(oauth_state.OAuthStateError):
        oauth_state.verify_state(token, b"another-secret" * 2)
    with pytest.raises(oauth_state.OAuthStateError):
        oauth_state.verify_state("garbage", secret)

    expired = oauth_state.sign_state({"nonce": "n2"}, secret, -1)
    with pytest.raises(oauth_state.OAuthStateError):
        oauth_state.verify_state(expired, secret)


def test_state_secret_needs_some_configured_key():
    with pytest.raises(oauth_state.OAuthStateError):
        oauth_state.state_secret(
            SimpleNamespace(connect_oauth_state_secret="", connected_account_encryption_key="")
        )
    explicit = oauth_state.state_secret(
        SimpleNamespace(connect_oauth_state_secret="abc", connected_account_encryption_key="")
    )
    derived = oauth_state.state_secret(
        SimpleNamespace(connect_oauth_state_secret="", connected_account_encryption_key="k")
    )
    assert explicit != derived


def test_pkce_matches_rfc_7636_example():
    # RFC 7636 appendix B verifier and challenge.
    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    assert (
        oauth_state.pkce_challenge_for(verifier)
        == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    )
    generated_verifier, generated_challenge = oauth_state.make_pkce()
    assert oauth_state.pkce_challenge_for(generated_verifier) == generated_challenge


# --------------------------------------------------------------------------
# Start
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_builds_a_google_consent_url_and_a_pending_row():
    repository = InMemoryPendingLoginRepository()
    started = await oauth_flow.start_oauth(
        _context(),
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        provider=get_provider("gmail"),
        repository=repository,
    )
    url = started["authorization_url"]
    assert url.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
    assert "access_type=offline" in url
    assert "prompt=consent" in url
    assert "code_challenge_method=S256" in url
    assert "mail.google.com" in url
    assert "redirect_uri=http%3A%2F%2Flocalhost%3A9600%2Fconnect_account%2Foauth%2Fcallback" in url
    pending = await repository.peek(started["nonce"])
    assert pending["provider"] == "gmail"
    assert pending["payload"]["code_verifier"]


@pytest.mark.asyncio
async def test_start_refuses_when_the_vendor_client_is_unconfigured():
    with pytest.raises(oauth_flow.OAuthFlowError) as raised:
        await oauth_flow.start_oauth(
            _context(google_oauth_client_id=""),
            user_id=USER_ID,
            assistant_id=ASSISTANT_ID,
            provider=get_provider("gmail"),
            repository=InMemoryPendingLoginRepository(),
        )
    assert raised.value.status_code == 503


# --------------------------------------------------------------------------
# Complete
# --------------------------------------------------------------------------


def _google_transport(calls, *, token_status=200, refresh_token="refresh-1"):
    def _handle(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url), dict(request.headers)))
        if request.url.host == "oauth2.googleapis.com":
            form = dict(httpx.QueryParams(request.content.decode()))
            calls.append(("form", form))
            if token_status != 200:
                return httpx.Response(token_status, json={"error": "invalid_grant"})
            return httpx.Response(
                200,
                json={
                    "access_token": "access-1",
                    "refresh_token": refresh_token,
                    "expires_in": 3600,
                    "scope": "openid email https://mail.google.com/",
                    "token_type": "Bearer",
                },
            )
        if request.url.host == "openidconnect.googleapis.com":
            return httpx.Response(200, json={"email": "Evan@Example.com", "sub": "1"})
        return httpx.Response(404)

    return httpx.MockTransport(_handle)


@pytest.mark.asyncio
async def test_complete_exchanges_proves_and_returns_an_encrypted_record(monkeypatch):
    from src.anubis.utils.tools.email import imap_client

    proved = {}

    def _verify(credentials):
        proved["mechanism"] = credentials.auth_mechanism
        proved["token"] = credentials.access_token

    monkeypatch.setattr(imap_client, "verify_credentials", _verify)

    context = _context()
    repository = InMemoryPendingLoginRepository()
    started = await oauth_flow.start_oauth(
        context,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        provider=get_provider("gmail"),
        repository=repository,
    )
    state = dict(httpx.QueryParams(started["authorization_url"].split("?", 1)[1]))["state"]
    calls = []
    async with httpx.AsyncClient(transport=_google_transport(calls)) as client:
        record = await oauth_flow.complete_oauth(
            context, code="code-1", state=state, repository=repository, http_client=client
        )
    form = next(entry[1] for entry in calls if entry[0] == "form")
    assert form["grant_type"] == "authorization_code"
    assert form["code_verifier"]
    assert form["client_secret"] == "google-secret"
    assert proved == {"mechanism": "xoauth2", "token": "access-1"}

    assert record["account_key"] == "gmail:evan@example.com"
    assert record["credential_mechanism"] == "oauth"
    assert record["user_id"] == USER_ID
    assert record["assistant_id"] == ASSISTANT_ID
    bundle = oauth_flow.decrypt_token_bundle(record, context)
    assert bundle["refresh_token"] == "refresh-1"
    assert "access-1" not in json.dumps(public_account_view(record))
    assert "refresh-1" not in json.dumps(public_account_view(record))

    # Single use: the same code and state cannot complete twice.
    async with httpx.AsyncClient(transport=_google_transport([])) as client:
        with pytest.raises(oauth_flow.OAuthFlowError):
            await oauth_flow.complete_oauth(
                context, code="code-1", state=state, repository=repository, http_client=client
            )


@pytest.mark.asyncio
async def test_a_failed_mailbox_proof_stores_nothing(monkeypatch):
    from src.anubis.utils.tools.email import imap_client

    def _refuse(credentials):
        raise imap_client.MailboxAuthenticationError("no")

    monkeypatch.setattr(imap_client, "verify_credentials", _refuse)
    context = _context()
    repository = InMemoryPendingLoginRepository()
    started = await oauth_flow.start_oauth(
        context, user_id=USER_ID, assistant_id=ASSISTANT_ID,
        provider=get_provider("gmail"), repository=repository,
    )
    state = dict(httpx.QueryParams(started["authorization_url"].split("?", 1)[1]))["state"]
    async with httpx.AsyncClient(transport=_google_transport([])) as client:
        with pytest.raises(oauth_flow.OAuthFlowError) as raised:
            await oauth_flow.complete_oauth(
                context, code="c", state=state, repository=repository, http_client=client
            )
    assert "mail permission" in raised.value.detail


@pytest.mark.asyncio
async def test_a_tampered_state_completes_nothing():
    context = _context()
    with pytest.raises(oauth_flow.OAuthFlowError) as raised:
        await oauth_flow.complete_oauth(
            context, code="c", state="not.signed", repository=InMemoryPendingLoginRepository()
        )
    assert raised.value.status_code == 400


# --------------------------------------------------------------------------
# Refresh
# --------------------------------------------------------------------------


class _Store:
    def __init__(self):
        self.saved = []

    async def aput(self, namespace, key, value):
        self.saved.append((key, value))

    async def aget(self, namespace, key):
        return None


def _oauth_record(context, *, expires_at, refresh_token="refresh-1"):
    from src.anubis.utils.connected_accounts.store import build_account_record

    bundle = {
        "access_token": "old-access",
        "refresh_token": refresh_token,
        "expires_at": expires_at,
        "token_type": "Bearer",
        "scopes": ["https://mail.google.com/"],
    }
    record = build_account_record(
        provider=get_provider("gmail"),
        account_address="evan@example.com",
        display_label="evan",
        encrypted_secret=secret_store.encrypt_secret(json.dumps(bundle), context),
        assistant_id=ASSISTANT_ID,
        transport={"oauth_vendor": "google", "scopes": bundle["scopes"]},
    )
    record["user_id"] = USER_ID
    return record


@pytest.mark.asyncio
async def test_a_valid_token_is_returned_without_a_refresh():
    context = _context()
    record = _oauth_record(context, expires_at=time.time() + 3600)
    oauth_flow.forget_cached_access_token(record["account_key"])
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
        token = await oauth_flow.get_fresh_access_token(
            context, _Store(), USER_ID, record, http_client=client
        )
    assert token == "old-access"


@pytest.mark.asyncio
async def test_an_expired_token_is_refreshed_and_re_encrypted():
    context = _context()
    record = _oauth_record(context, expires_at=time.time() - 10)
    oauth_flow.forget_cached_access_token(record["account_key"])
    calls = []
    store = _Store()
    async with httpx.AsyncClient(transport=_google_transport(calls, refresh_token="refresh-2")) as client:
        token = await oauth_flow.get_fresh_access_token(
            context, store, USER_ID, record, http_client=client
        )
    assert token == "access-1"
    form = next(entry[1] for entry in calls if entry[0] == "form")
    assert form["grant_type"] == "refresh_token"
    assert oauth_flow.decrypt_token_bundle(record, context)["refresh_token"] == "refresh-2"


@pytest.mark.asyncio
async def test_a_revoked_token_flags_the_account_for_reconnect(monkeypatch):
    from src.anubis.utils.connected_accounts import oauth_flow as flow_module

    flagged = []

    async def _mark(store, user_id, key):
        flagged.append(key)

    monkeypatch.setattr(
        "src.anubis.utils.connected_accounts.store.mark_account_needs_reconnect", _mark
    )
    context = _context()
    record = _oauth_record(context, expires_at=time.time() - 10)
    flow_module.forget_cached_access_token(record["account_key"])
    async with httpx.AsyncClient(transport=_google_transport([], token_status=400)) as client:
        with pytest.raises(flow_module.OAuthReconnectRequired):
            await flow_module.get_fresh_access_token(
                context, _Store(), USER_ID, record, http_client=client
            )
    assert flagged == [record["account_key"]]


# --------------------------------------------------------------------------
# Result page
# --------------------------------------------------------------------------


def test_the_result_page_posts_only_to_configured_origins_and_holds_no_token():
    html = oauth_flow.render_popup_result_html(
        {"ok": True, "nonce": "n", "provider": "gmail", "display_label": "evan", "access_token": "SECRET"},
        oauth_flow.allowed_popup_origins(_context()),
    )
    assert "http://localhost:5173" in html
    assert "https://neuralnexus.site" in html
    assert '"*"' not in html
    assert "SECRET" not in html
    assert "neural-nexus:login-result" in html
    assert oauth_flow.allowed_popup_origins(SimpleNamespace(connect_oauth_popup_target_origins="*")) == []
