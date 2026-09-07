"""The OAuth 2.0 authorization-code flow behind the connect card's popup.

Two requests and one long-lived helper:

``start_oauth``
    Called by ``POST /connect_account/oauth/start`` for the signed-in owner.
    Mints a nonce and PKCE pair, stores the verifier server-side
    (``pending_logins``), signs the state, and returns the vendor's
    authorization URL for the popup to open.

``complete_oauth``
    Called by ``GET /connect_account/oauth/callback`` when the vendor redirects
    back. Verifies the state, consumes the pending row (single use), exchanges
    the code, reads who signed in, PROVES the token where the vendor's
    userinfo is not enough (Gmail: a real XOAUTH2 login), and returns the
    record to store. The refresh token, access token, and expiry are
    encrypted together into ``encrypted_secret`` so a record still has one
    secret field and the public projection still hides it.

``get_fresh_access_token``
    Called by every tool that dials the vendor. Returns a cached access token
    while it is valid, refreshes it through the token endpoint when it is not,
    re-encrypts a rotated refresh token, and — when the vendor answers
    ``invalid_grant`` (revoked, or Google's seven-day Testing expiry) — marks
    the record ``needs_reconnect`` and raises ``OAuthReconnectRequired`` so the
    avatar raises the card again instead of failing every call.

The result page the callback renders posts a NON-secret result to the window
that opened the popup, only to the configured UI origins, then closes itself.
Tokens never appear in that page.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import time
from typing import Any
from urllib.parse import urlencode

from src.anubis.utils.connected_accounts.oauth_providers import (
    OAuthProviderConfig,
    client_credentials,
    get_oauth_provider,
)
from src.anubis.utils.connected_accounts.oauth_state import (
    OAuthStateError,
    make_pkce,
    random_nonce,
    sign_state,
    state_secret,
    verify_state,
)
from src.anubis.utils.connected_accounts.pending_logins import (
    MODE_MCP_OAUTH,
    MODE_OAUTH,
    build_pending_row,
    get_pending_login_repository,
)

logger = logging.getLogger(__name__)

OAUTH_CALLBACK_PATH = "/connect_account/oauth/callback"
LOGIN_RESULT_MESSAGE_TYPE = "neural-nexus:login-result"

# Access tokens are refreshed this many seconds before they expire so a call
# started just before expiry does not fail mid-flight.
_EXPIRY_SKEW_SECONDS = 60

# Per-process cache of decrypted access tokens: ``account_key`` → (token, expiry).
_access_token_cache: dict[str, tuple[str, float]] = {}


class OAuthFlowError(Exception):
    """A popup login could not be started or completed; ``detail`` is owner-safe."""

    def __init__(self, status_code: int, detail: str) -> None:
        """Carry the HTTP status the route should answer with, and why."""
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class OAuthReconnectRequired(Exception):
    """The vendor no longer honours the stored refresh token."""


def redirect_uri(context: Any) -> str:
    """Return the callback URL registered with every vendor."""
    base = str(getattr(context, "connect_oauth_redirect_base_url", "") or "").strip()
    if not base:
        raise OAuthFlowError(
            503,
            "CONNECT_OAUTH_REDIRECT_BASE_URL is not configured, so a sign-in "
            "window has nowhere to return to.",
        )
    return base.rstrip("/") + OAUTH_CALLBACK_PATH


def allowed_popup_origins(context: Any) -> list[str]:
    """Return the UI origins a result page may post to (never ``*``)."""
    raw = str(getattr(context, "connect_oauth_popup_target_origins", "") or "")
    origins = [entry.strip().rstrip("/") for entry in raw.split(",") if entry.strip()]
    return [origin for origin in origins if origin != "*"]


def _state_max_age(context: Any) -> int:
    return int(getattr(context, "connect_oauth_state_max_age_seconds", None) or 600)


def _http_timeout(context: Any) -> float:
    return float(getattr(context, "connect_oauth_http_timeout_seconds", None) or 15.0)


def _client(context: Any, http_client: Any | None):
    if http_client is not None:
        return http_client, False
    import httpx

    return httpx.AsyncClient(timeout=_http_timeout(context)), True


def scopes_for(provider: Any, config: OAuthProviderConfig) -> tuple[str, ...]:
    """Return the scopes one provider row requests from its vendor."""
    return tuple(provider.oauth_scopes) or tuple(config.scopes)


async def start_oauth(
    context: Any,
    *,
    user_id: str,
    assistant_id: str,
    provider: Any,
    server_url: str | None = None,
    name: str | None = None,
    repository: Any | None = None,
) -> dict[str, Any]:
    """Begin a popup sign-in; return ``{"authorization_url", "nonce", "expires_in"}``."""
    repository = repository or get_pending_login_repository()
    try:
        await repository.purge_expired()
    except Exception:
        logger.debug("Could not purge expired pending logins", exc_info=True)

    if provider.name == "custom_mcp":
        from src.anubis.utils.connected_accounts.mcp_oauth import begin_mcp_login

        return await begin_mcp_login(
            context,
            user_id=user_id,
            assistant_id=assistant_id,
            provider=provider,
            server_url=str(server_url or ""),
            name=str(name or ""),
            repository=repository,
        )

    config = get_oauth_provider(provider.oauth_config_key or "")
    if config is None:
        raise OAuthFlowError(
            400, f"{provider.display_name} does not sign in through OAuth."
        )
    client_id, _client_secret = client_credentials(config, context)
    if not client_id:
        raise OAuthFlowError(
            503,
            f"{provider.display_name} sign-in is not configured on this server "
            f"({config.client_id_field.upper()} is empty).",
        )

    nonce = random_nonce()
    verifier, challenge = make_pkce()
    max_age = _state_max_age(context)
    state = sign_state(
        {
            "nonce": nonce,
            "user_id": user_id,
            "assistant_id": assistant_id,
            "provider": provider.name,
            "mode": MODE_OAUTH,
        },
        state_secret(context),
        max_age,
    )
    await repository.create(
        build_pending_row(
            nonce=nonce,
            user_id=user_id,
            assistant_id=assistant_id,
            provider=provider.name,
            mode=MODE_OAUTH,
            payload={"code_verifier": verifier if config.pkce else None},
            max_age_seconds=max_age,
        )
    )
    parameters: dict[str, str] = {
        "client_id": client_id,
        "redirect_uri": redirect_uri(context),
        "response_type": "code",
        "state": state,
    }
    scopes = scopes_for(provider, config)
    if scopes:
        parameters["scope"] = config.scope_separator.join(scopes)
    if config.pkce:
        parameters["code_challenge"] = challenge
        parameters["code_challenge_method"] = "S256"
    parameters.update(config.extra_authorize_params)
    return {
        "authorization_url": f"{config.authorization_url}?{urlencode(parameters)}",
        "nonce": nonce,
        "expires_in": max_age,
        "provider": provider.name,
    }


async def _exchange_code(
    context: Any,
    config: OAuthProviderConfig,
    *,
    code: str,
    code_verifier: str | None,
    http_client: Any | None,
) -> dict[str, Any]:
    client_id, client_secret = client_credentials(config, context)
    form: dict[str, str] = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri(context),
        "client_id": client_id,
    }
    if code_verifier:
        form["code_verifier"] = code_verifier
    auth = None
    if config.token_auth == "basic":
        auth = (client_id, client_secret)
    else:
        form["client_secret"] = client_secret
    client, owned = _client(context, http_client)
    try:
        response = await client.post(
            config.token_url,
            data=form,
            headers={"Accept": "application/json"},
            auth=auth,
        )
    finally:
        if owned:
            await client.aclose()
    if response.status_code >= 400:
        raise OAuthFlowError(
            400,
            f"The sign-in could not be completed ({response.status_code} from the "
            "token endpoint). Start the sign-in again.",
        )
    return _token_document(response)


def _token_document(response: Any) -> dict[str, Any]:
    try:
        document = response.json()
    except Exception:
        # GitHub answers form-encoded unless asked for JSON; parse that too.
        from urllib.parse import parse_qs

        document = {
            key: values[0] for key, values in parse_qs(response.text or "").items()
        }
    if not isinstance(document, dict) or not document.get("access_token"):
        raise OAuthFlowError(
            400, "The token endpoint answered without an access token."
        )
    return document


def _bundle_from_token_document(
    document: dict[str, Any], previous: dict[str, Any] | None = None
) -> dict[str, Any]:
    expires_in = document.get("expires_in")
    expires_at = (
        time.time() + float(expires_in)
        if isinstance(expires_in, (int, float, str)) and str(expires_in).strip()
        else None
    )
    scope = document.get("scope")
    scopes = (
        [entry for entry in str(scope).replace(",", " ").split() if entry]
        if scope
        else list((previous or {}).get("scopes") or [])
    )
    return {
        "access_token": str(document["access_token"]),
        "refresh_token": str(
            document.get("refresh_token") or (previous or {}).get("refresh_token") or ""
        )
        or None,
        "expires_at": expires_at,
        "token_type": str(document.get("token_type") or "Bearer"),
        "scopes": scopes,
    }


async def _fetch_userinfo(
    context: Any,
    config: OAuthProviderConfig,
    access_token: str,
    http_client: Any | None,
) -> dict[str, Any]:
    client, owned = _client(context, http_client)
    try:
        response = await client.get(
            config.userinfo_url,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
                **config.userinfo_headers,
            },
        )
    finally:
        if owned:
            await client.aclose()
    if response.status_code >= 400:
        raise OAuthFlowError(
            400,
            "The account that signed in could not be identified "
            f"({response.status_code} from the identity endpoint).",
        )
    document = response.json()
    return document if isinstance(document, dict) else {}


async def _prove_mailbox(provider: Any, account_address: str, access_token: str, context: Any) -> None:
    """Prove a Gmail token by a real XOAUTH2 login before storing the token."""
    from src.anubis.utils.tools.email.imap_client import (
        AUTH_MECHANISM_XOAUTH2,
        MailboxAuthenticationError,
        MailboxCredentials,
        MailboxUnreachableError,
        verify_credentials,
    )

    credentials = MailboxCredentials(
        account_address=account_address,
        password="",
        imap_host=provider.imap_host,
        imap_port=provider.imap_port,
        smtp_host=provider.smtp_host,
        smtp_port=provider.smtp_port,
        drafts_mailbox=provider.drafts_mailbox,
        timeout_seconds=float(
            getattr(context, "mailbox_request_timeout_seconds", None) or 30.0
        ),
        auth_mechanism=AUTH_MECHANISM_XOAUTH2,
        access_token=access_token,
    )
    try:
        await asyncio.to_thread(verify_credentials, credentials)
    except MailboxAuthenticationError as auth_error:
        raise OAuthFlowError(
            400,
            f"Google signed in, but {provider.display_name} refused the mail "
            "permission. Make sure the mail scope was granted on the consent "
            "screen and sign in again.",
        ) from auth_error
    except MailboxUnreachableError as unreachable_error:
        raise OAuthFlowError(
            503, f"Could not reach {provider.display_name} to check the sign-in: "
            f"{unreachable_error}",
        ) from unreachable_error


async def complete_oauth(
    context: Any,
    *,
    code: str,
    state: str,
    repository: Any | None = None,
    existing_records: list[dict[str, Any]] | None = None,
    http_client: Any | None = None,
) -> dict[str, Any]:
    """Finish a popup sign-in; return the account record to store.

    The returned record carries ``user_id`` and ``assistant_id`` from the
    signed state so the route knows whose account to store without a session.
    """
    from src.anubis.utils.connected_accounts.providers import get_provider
    from src.anubis.utils.connected_accounts.store import (
        account_key,
        build_account_record,
        deduplicate_label,
    )
    from src.anubis.utils.secret_store import encrypt_secret

    repository = repository or get_pending_login_repository()
    try:
        payload = verify_state(state, state_secret(context))
    except OAuthStateError as state_error:
        raise OAuthFlowError(400, str(state_error)) from state_error
    nonce = str(payload.get("nonce") or "")
    pending = await repository.consume(nonce)
    if pending is None:
        raise OAuthFlowError(
            400, "This sign-in was already completed or has expired. Start it again."
        )
    if pending.get("user_id") != payload.get("user_id"):
        raise OAuthFlowError(400, "The login state does not match the started sign-in.")

    if pending.get("mode") == MODE_MCP_OAUTH:
        from src.anubis.utils.connected_accounts.mcp_oauth import complete_mcp_login

        return await complete_mcp_login(
            context,
            code=code,
            pending=pending,
            existing_records=list(existing_records or []),
            http_client=http_client,
        )

    provider = get_provider(str(pending.get("provider") or payload.get("provider") or ""))
    if provider is None:
        raise OAuthFlowError(400, "The provider of this sign-in is unknown.")
    config = get_oauth_provider(provider.oauth_config_key or "")
    if config is None:
        raise OAuthFlowError(400, f"{provider.display_name} does not sign in through OAuth.")

    verifier = (pending.get("payload") or {}).get("code_verifier")
    token_document = await _exchange_code(
        context, config, code=code, code_verifier=verifier, http_client=http_client
    )
    bundle = _bundle_from_token_document(token_document)
    userinfo = await _fetch_userinfo(context, config, bundle["access_token"], http_client)
    account_address, display_label = config.identity_from_userinfo(userinfo)
    if not account_address:
        raise OAuthFlowError(400, "The vendor did not say which account signed in.")

    if provider.is_mailbox:
        await _prove_mailbox(provider, account_address, bundle["access_token"], context)

    user_id = str(pending["user_id"])
    assistant_id = str(pending["assistant_id"])
    key = account_key(provider.name, account_address)
    label = deduplicate_label(display_label, list(existing_records or []), key)
    record = build_account_record(
        provider=provider,
        account_address=account_address,
        display_label=label,
        encrypted_secret=encrypt_secret(json.dumps(bundle), context),
        assistant_id=assistant_id,
        transport={
            "oauth_vendor": config.key,
            "scopes": bundle["scopes"],
            "access_token_expires_at": bundle["expires_at"],
        },
    )
    record["user_id"] = user_id
    _access_token_cache[key] = (bundle["access_token"], bundle["expires_at"] or 0.0)
    return record


def decrypt_token_bundle(record: dict[str, Any], context: Any) -> dict[str, Any]:
    """Return the stored token bundle of an OAuth record."""
    from src.anubis.utils.secret_store import decrypt_secret

    raw = decrypt_secret(str(record.get("encrypted_secret") or ""), context)
    try:
        bundle = json.loads(raw)
    except Exception:
        bundle = {}
    return bundle if isinstance(bundle, dict) else {}


async def get_fresh_access_token(
    context: Any,
    store: Any,
    user_id: str,
    record: dict[str, Any],
    *,
    http_client: Any | None = None,
) -> str:
    """Return a valid access token for an OAuth record, refreshing if needed."""
    from src.anubis.utils.connected_accounts.store import (
        mark_account_needs_reconnect,
        save_connected_account,
    )
    from src.anubis.utils.secret_store import encrypt_secret

    key = str(record.get("account_key") or "")
    cached = _access_token_cache.get(key)
    now = time.time()
    if cached and (cached[1] == 0.0 or cached[1] - _EXPIRY_SKEW_SECONDS > now):
        return cached[0]

    bundle = decrypt_token_bundle(record, context)
    expires_at = bundle.get("expires_at")
    access_token = bundle.get("access_token")
    if access_token and (
        expires_at is None or float(expires_at) - _EXPIRY_SKEW_SECONDS > now
    ):
        _access_token_cache[key] = (str(access_token), float(expires_at or 0.0))
        return str(access_token)

    vendor_key = str((record.get("transport") or {}).get("oauth_vendor") or "")
    config = get_oauth_provider(vendor_key)
    refresh_token = bundle.get("refresh_token")
    if config is None or not refresh_token:
        await mark_account_needs_reconnect(store, user_id, key)
        raise OAuthReconnectRequired(
            f"{record.get('display_label')} has no refresh token; sign in again."
        )

    client_id, client_secret = client_credentials(config, context)
    form: dict[str, str] = {
        "grant_type": "refresh_token",
        "refresh_token": str(refresh_token),
        "client_id": client_id,
    }
    auth = None
    if config.token_auth == "basic":
        auth = (client_id, client_secret)
    else:
        form["client_secret"] = client_secret
    client, owned = _client(context, http_client)
    try:
        response = await client.post(
            config.token_url,
            data=form,
            headers={"Accept": "application/json"},
            auth=auth,
        )
    finally:
        if owned:
            await client.aclose()
    if response.status_code >= 400:
        error_code = ""
        try:
            error_code = str((response.json() or {}).get("error") or "")
        except Exception:
            error_code = ""
        if response.status_code in (400, 401) and error_code in (
            "invalid_grant",
            "invalid_token",
            "",
        ):
            await mark_account_needs_reconnect(store, user_id, key)
            _access_token_cache.pop(key, None)
            raise OAuthReconnectRequired(
                f"{record.get('display_label')} needs to be signed in again."
            )
        raise OAuthFlowError(
            503, f"The token could not be refreshed ({response.status_code})."
        )
    refreshed = _bundle_from_token_document(_token_document(response), bundle)
    record["encrypted_secret"] = encrypt_secret(json.dumps(refreshed), context)
    transport = dict(record.get("transport") or {})
    transport["access_token_expires_at"] = refreshed["expires_at"]
    record["transport"] = transport
    try:
        await save_connected_account(store, user_id, record)
    except Exception:
        logger.debug("Could not persist the refreshed token for %s", key, exc_info=True)
    _access_token_cache[key] = (refreshed["access_token"], refreshed["expires_at"] or 0.0)
    return refreshed["access_token"]


def forget_cached_access_token(key: str) -> None:
    """Drop a cached token (after a disconnect)."""
    _access_token_cache.pop(key, None)


def render_popup_result_html(result: dict[str, Any], allowed_origins: list[str]) -> str:
    """Render the page a popup ends on: posts a non-secret result to the opener and closes."""
    safe_result = {
        "type": LOGIN_RESULT_MESSAGE_TYPE,
        "ok": bool(result.get("ok")),
        "nonce": result.get("nonce"),
        "provider": result.get("provider"),
        "account_key": result.get("account_key"),
        "display_label": result.get("display_label"),
        "account_address": result.get("account_address"),
        "tool_count": result.get("tool_count"),
        "error": result.get("error"),
    }
    heading = "Connected" if safe_result["ok"] else "Sign-in not completed"
    detail = (
        f"{safe_result.get('display_label') or safe_result.get('provider') or 'The account'} "
        "is connected. You can close this window."
        if safe_result["ok"]
        else str(safe_result.get("error") or "Close this window and try again.")
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(heading)}</title>
<style>body{{font-family:system-ui,sans-serif;background:#0b0b0d;color:#e8e8ea;display:flex;
align-items:center;justify-content:center;height:100vh;margin:0}}main{{text-align:center;max-width:28rem;padding:2rem}}
h1{{font-size:1.25rem;margin:0 0 .5rem}}p{{color:#a3a3a8;margin:0}}</style></head>
<body><main><h1>{html.escape(heading)}</h1><p>{html.escape(detail)}</p></main>
<script>
(function () {{
  var result = {json.dumps(safe_result)};
  var origins = {json.dumps(list(allowed_origins))};
  try {{
    if (window.opener && !window.opener.closed) {{
      for (var i = 0; i < origins.length; i += 1) {{
        try {{ window.opener.postMessage(result, origins[i]); }} catch (e) {{}}
      }}
    }}
  }} catch (e) {{}}
  setTimeout(function () {{ try {{ window.close(); }} catch (e) {{}} }}, 400);
}})();
</script></body></html>"""
