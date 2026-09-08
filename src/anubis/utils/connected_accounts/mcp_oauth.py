"""OAuth for custom Model Context Protocol servers that demand a login.

A custom connector is collected as a name and a server address in conversation
(neither is a secret) and connected DIRECTLY when the server answers a plain
``initialize`` request. Only when the server answers ``401`` does a card appear:

- ``401`` with protected-resource metadata (RFC 9728) → the authorization
  server is discovered (RFC 8414), this API registers itself as a client
  (RFC 7591 dynamic registration) when the server allows, and the popup runs
  the PKCE authorization-code flow. The freshly registered client id and
  secret travel through ``pending_logins``, never through the browser.
- ``401`` without metadata → a static bearer token form (the existing path).

The helpers of the installed ``mcp`` package do the parsing so the discovery
order matches the reference client exactly.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any
from urllib.parse import urlencode, urljoin

from src.anubis.utils.connected_accounts.oauth_state import (
    random_nonce,
    sign_state,
    state_secret,
)
from src.anubis.utils.connected_accounts.pending_logins import (
    MODE_MCP_OAUTH,
    build_pending_row,
)

logger = logging.getLogger(__name__)

AUTHORIZATION_OPEN = "open"
AUTHORIZATION_NEEDS_OAUTH = "needs_oauth"
AUTHORIZATION_NEEDS_TOKEN = "needs_token"
AUTHORIZATION_UNREACHABLE = "unreachable"

_MCP_PROTOCOL_VERSION_HEADER = "MCP-Protocol-Version"
_MCP_PROTOCOL_VERSION = "2025-06-18"

_INITIALIZE_REQUEST = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": _MCP_PROTOCOL_VERSION,
        "capabilities": {},
        "clientInfo": {"name": "neural-nexus-probe", "version": "1.0"},
    },
}


class McpOAuthError(Exception):
    """Discovery, registration, or the exchange failed; ``detail`` is owner-safe."""

    def __init__(self, status_code: int, detail: str) -> None:
        """Carry the HTTP status the route should answer with, and why."""
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _timeout(context: Any) -> float:
    return float(getattr(context, "connect_oauth_http_timeout_seconds", None) or 15.0)


async def probe_authorization(
    server_url: str, context: Any, *, http_client: Any | None = None
) -> dict[str, Any]:
    """Ask the server how a client is expected to authenticate.

    Returns ``{"status": open|needs_oauth|needs_token|unreachable,
    "www_authenticate": str | None, "resource_metadata_url": str | None}``.
    A streamable-HTTP server is probed with an ``initialize`` POST; an SSE
    endpoint with a streamed GET that reads only the status line and headers.
    """
    import httpx

    owned = http_client is None
    client = http_client or httpx.AsyncClient(timeout=_timeout(context))
    headers = {
        "Accept": "application/json, text/event-stream",
        _MCP_PROTOCOL_VERSION_HEADER: _MCP_PROTOCOL_VERSION,
    }
    try:
        if str(server_url).rstrip("/").lower().endswith("/sse"):
            async with client.stream("GET", server_url, headers=headers) as response:
                status_code = response.status_code
                www_authenticate = response.headers.get("WWW-Authenticate")
        else:
            response = await client.post(
                server_url,
                json=_INITIALIZE_REQUEST,
                headers={**headers, "Content-Type": "application/json"},
            )
            status_code = response.status_code
            www_authenticate = response.headers.get("WWW-Authenticate")
    except Exception as probe_error:
        logger.info("Custom connector probe failed for %s: %s", server_url, probe_error)
        return {
            "status": AUTHORIZATION_UNREACHABLE,
            "www_authenticate": None,
            "resource_metadata_url": None,
            "error": str(probe_error),
        }
    finally:
        if owned:
            await client.aclose()

    if status_code == 401:
        resource_metadata_url = _resource_metadata_from_header(www_authenticate)
        if resource_metadata_url is None:
            # Servers that follow the specification without the header still
            # publish the well-known document; try the first discovery URL.
            resource_metadata_url = await _first_discovery_hit(
                server_url, context, http_client=http_client
            )
        return {
            "status": AUTHORIZATION_NEEDS_OAUTH
            if resource_metadata_url
            else AUTHORIZATION_NEEDS_TOKEN,
            "www_authenticate": www_authenticate,
            "resource_metadata_url": resource_metadata_url,
        }
    if status_code >= 500:
        return {
            "status": AUTHORIZATION_UNREACHABLE,
            "www_authenticate": None,
            "resource_metadata_url": None,
            "error": f"The server answered {status_code}.",
        }
    return {
        "status": AUTHORIZATION_OPEN,
        "www_authenticate": None,
        "resource_metadata_url": None,
    }


def _resource_metadata_from_header(www_authenticate: str | None) -> str | None:
    if not www_authenticate:
        return None
    for part in str(www_authenticate).split(","):
        part = part.strip()
        if part.lower().startswith("bearer "):
            part = part[7:].strip()
        if "=" not in part:
            continue
        key, _, value = part.partition("=")
        if key.strip().lower() == "resource_metadata":
            return value.strip().strip('"')
    return None


async def _first_discovery_hit(
    server_url: str, context: Any, *, http_client: Any | None
) -> str | None:
    import httpx
    from mcp.client.auth.utils import (
        build_protected_resource_metadata_discovery_urls,
    )

    owned = http_client is None
    client = http_client or httpx.AsyncClient(timeout=_timeout(context))
    try:
        for url in build_protected_resource_metadata_discovery_urls(None, server_url):
            try:
                response = await client.get(url)
            except Exception:
                continue
            if response.status_code == 200:
                return url
    finally:
        if owned:
            await client.aclose()
    return None


async def discover(
    server_url: str,
    resource_metadata_url: str | None,
    context: Any,
    *,
    http_client: Any | None = None,
) -> dict[str, Any]:
    """Resolve the authorization server's endpoints for one MCP server.

    Returns ``{"authorization_endpoint", "token_endpoint", "registration_endpoint",
    "resource", "scopes"}``; the fallback endpoints are ``/authorize``,
    ``/token``, and ``/register`` on the server's origin, per the MCP
    authorization specification.
    """
    import httpx
    from mcp.client.auth.utils import (
        build_oauth_authorization_server_metadata_discovery_urls,
        build_protected_resource_metadata_discovery_urls,
        handle_auth_metadata_response,
        handle_protected_resource_response,
    )

    owned = http_client is None
    client = http_client or httpx.AsyncClient(timeout=_timeout(context))
    resource_metadata = None
    authorization_server_url: str | None = None
    scopes: list[str] = []
    try:
        for url in build_protected_resource_metadata_discovery_urls(
            resource_metadata_url, server_url
        ):
            try:
                response = await client.get(url)
            except Exception:
                continue
            resource_metadata = await handle_protected_resource_response(response)
            if resource_metadata is not None:
                break
        if resource_metadata is not None:
            authorization_server_url = str(resource_metadata.authorization_servers[0])
            scopes = list(resource_metadata.scopes_supported or [])

        auth_metadata = None
        for url in build_oauth_authorization_server_metadata_discovery_urls(
            authorization_server_url, server_url
        ):
            try:
                response = await client.get(url)
            except Exception:
                continue
            keep_trying, auth_metadata = await handle_auth_metadata_response(response)
            if auth_metadata is not None or not keep_trying:
                break
    finally:
        if owned:
            await client.aclose()

    base = authorization_server_url or server_url
    if auth_metadata is not None:
        return {
            "authorization_endpoint": str(auth_metadata.authorization_endpoint),
            "token_endpoint": str(auth_metadata.token_endpoint),
            "registration_endpoint": str(auth_metadata.registration_endpoint)
            if auth_metadata.registration_endpoint
            else None,
            "resource": str(resource_metadata.resource) if resource_metadata else server_url,
            "scopes": scopes or list(auth_metadata.scopes_supported or []),
        }
    return {
        "authorization_endpoint": urljoin(base, "/authorize"),
        "token_endpoint": urljoin(base, "/token"),
        "registration_endpoint": urljoin(base, "/register"),
        "resource": str(resource_metadata.resource) if resource_metadata else server_url,
        "scopes": scopes,
    }


async def register_client(
    discovery: dict[str, Any],
    context: Any,
    *,
    http_client: Any | None = None,
) -> dict[str, Any]:
    """Register this API as an OAuth client; return ``{client_id, client_secret}``."""
    import httpx
    from mcp.shared.auth import OAuthClientMetadata

    from src.anubis.utils.connected_accounts.oauth_flow import redirect_uri

    registration_endpoint = discovery.get("registration_endpoint")
    if not registration_endpoint:
        raise McpOAuthError(
            400,
            "The server's authorization server does not offer client registration. "
            "Connect with an access token instead.",
        )
    metadata = OAuthClientMetadata(
        redirect_uris=[redirect_uri(context)],
        client_name=str(getattr(context, "mcp_oauth_client_name", None) or "Neural Nexus"),
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="client_secret_post",
    )
    owned = http_client is None
    client = http_client or httpx.AsyncClient(timeout=_timeout(context))
    try:
        response = await client.post(
            registration_endpoint,
            json=metadata.model_dump(by_alias=True, mode="json", exclude_none=True),
            headers={"Content-Type": "application/json"},
        )
    except Exception as registration_error:
        raise McpOAuthError(
            503, f"Client registration failed: {registration_error}"
        ) from registration_error
    finally:
        if owned:
            await client.aclose()
    if response.status_code not in (200, 201):
        raise McpOAuthError(
            400,
            "The server's authorization server refused client registration "
            f"({response.status_code}). Connect with an access token instead.",
        )
    document = response.json()
    if not isinstance(document, dict) or not document.get("client_id"):
        raise McpOAuthError(400, "Client registration answered without a client id.")
    return {
        "client_id": str(document["client_id"]),
        "client_secret": str(document.get("client_secret") or "") or None,
    }


async def begin_mcp_login(
    context: Any,
    *,
    user_id: str,
    assistant_id: str,
    provider: Any,
    server_url: str,
    name: str,
    repository: Any,
    http_client: Any | None = None,
) -> dict[str, Any]:
    """Discover, register, and return the authorization URL for the popup."""
    from src.anubis.utils.connected_accounts.oauth_flow import redirect_uri
    from src.anubis.utils.connected_accounts.oauth_state import make_pkce
    from src.anubis.utils.secret_store import encrypt_secret

    if not server_url:
        raise McpOAuthError(400, "A server_url is required to sign in to a connector.")
    probe = await probe_authorization(server_url, context, http_client=http_client)
    if probe["status"] == AUTHORIZATION_UNREACHABLE:
        raise McpOAuthError(400, f"The server at {server_url} could not be reached.")
    discovery = await discover(
        server_url, probe.get("resource_metadata_url"), context, http_client=http_client
    )
    client_info = await register_client(discovery, context, http_client=http_client)

    nonce = random_nonce()
    verifier, challenge = make_pkce()
    max_age = int(getattr(context, "connect_oauth_state_max_age_seconds", None) or 600)
    state = sign_state(
        {
            "nonce": nonce,
            "user_id": user_id,
            "assistant_id": assistant_id,
            "provider": provider.name,
            "mode": MODE_MCP_OAUTH,
        },
        state_secret(context),
        max_age,
    )
    payload = {
        "code_verifier": verifier,
        "server_url": server_url,
        "name": name,
        "token_endpoint": discovery["token_endpoint"],
        "client_id": client_info["client_id"],
        "client_secret_encrypted": encrypt_secret(client_info["client_secret"], context)
        if client_info.get("client_secret")
        else None,
        "resource": discovery.get("resource"),
        "scopes": discovery.get("scopes") or [],
    }
    await repository.create(
        build_pending_row(
            nonce=nonce,
            user_id=user_id,
            assistant_id=assistant_id,
            provider=provider.name,
            mode=MODE_MCP_OAUTH,
            payload=payload,
            max_age_seconds=max_age,
        )
    )
    parameters = {
        "response_type": "code",
        "client_id": client_info["client_id"],
        "redirect_uri": redirect_uri(context),
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    if discovery.get("resource"):
        parameters["resource"] = discovery["resource"]
    if discovery.get("scopes"):
        parameters["scope"] = " ".join(discovery["scopes"])
    return {
        "authorization_url": f"{discovery['authorization_endpoint']}?{urlencode(parameters)}",
        "nonce": nonce,
        "expires_in": max_age,
        "provider": provider.name,
    }


async def _token_request(
    context: Any,
    payload: dict[str, Any],
    form: dict[str, str],
    *,
    http_client: Any | None,
) -> dict[str, Any]:
    import httpx

    from src.anubis.utils.secret_store import decrypt_secret

    form = dict(form)
    form["client_id"] = str(payload.get("client_id") or "")
    if payload.get("client_secret_encrypted"):
        form["client_secret"] = decrypt_secret(payload["client_secret_encrypted"], context)
    if payload.get("resource"):
        form["resource"] = str(payload["resource"])
    owned = http_client is None
    client = http_client or httpx.AsyncClient(timeout=_timeout(context))
    try:
        response = await client.post(
            str(payload["token_endpoint"]),
            data=form,
            headers={"Accept": "application/json"},
        )
    finally:
        if owned:
            await client.aclose()
    if response.status_code >= 400:
        raise McpOAuthError(
            400, f"The connector's token endpoint answered {response.status_code}."
        )
    document = response.json()
    if not isinstance(document, dict) or not document.get("access_token"):
        raise McpOAuthError(400, "The connector's token endpoint returned no token.")
    return document


async def complete_mcp_login(
    context: Any,
    *,
    code: str,
    pending: dict[str, Any],
    existing_records: list[dict[str, Any]],
    http_client: Any | None = None,
) -> dict[str, Any]:
    """Exchange the code, prove the token by listing tools, and build the record."""
    from src.anubis.utils.connected_accounts.connect_handlers import server_address_for
    from src.anubis.utils.connected_accounts.mcp_server_tools import (
        McpServerUnreachableError,
        infer_transport,
        probe_server_tools,
    )
    from src.anubis.utils.connected_accounts.oauth_flow import redirect_uri
    from src.anubis.utils.connected_accounts.providers import get_provider
    from src.anubis.utils.connected_accounts.store import (
        account_key,
        build_account_record,
        deduplicate_label,
    )
    from src.anubis.utils.secret_store import encrypt_secret

    payload = dict(pending.get("payload") or {})
    provider = get_provider("custom_mcp")
    server_url = str(payload.get("server_url") or "")
    document = await _token_request(
        context,
        payload,
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri(context),
            "code_verifier": str(payload.get("code_verifier") or ""),
        },
        http_client=http_client,
    )
    access_token = str(document["access_token"])
    expires_in = document.get("expires_in")
    expires_at = time.time() + float(expires_in) if expires_in else None

    timeout_seconds = float(
        getattr(context, "mcp_connector_probe_timeout_seconds", None) or 20.0
    )
    try:
        tools = await probe_server_tools(server_url, access_token, timeout_seconds)
    except McpServerUnreachableError as unreachable_error:
        raise McpOAuthError(
            400, f"Signed in, but the server did not list tools: {unreachable_error}"
        ) from unreachable_error
    tool_names = sorted(
        {str(getattr(tool, "name", "") or "") for tool in tools if getattr(tool, "name", "")}
    )

    name = str(payload.get("name") or "") or server_url.split("//", 1)[-1].split("/", 1)[0]
    account_address = server_address_for(server_url)
    key = account_key(provider.name, account_address)
    label = deduplicate_label(name, existing_records, key)
    bundle = {
        "access_token": access_token,
        "refresh_token": document.get("refresh_token"),
        "expires_at": expires_at,
        "token_type": str(document.get("token_type") or "Bearer"),
        "scopes": str(document.get("scope") or "").split(),
    }
    record = build_account_record(
        provider=provider,
        account_address=account_address,
        display_label=label,
        encrypted_secret=encrypt_secret(json.dumps(bundle), context),
        assistant_id=str(pending["assistant_id"]),
        transport={
            "server_url": server_url,
            "transport": infer_transport(server_url),
            "tool_names": tool_names,
            "auth_type": "oauth",
            "oauth": {
                "token_endpoint": payload.get("token_endpoint"),
                "client_id": payload.get("client_id"),
                "client_secret_encrypted": payload.get("client_secret_encrypted"),
                "resource": payload.get("resource"),
                "access_token_expires_at": expires_at,
            },
        },
    )
    record["user_id"] = str(pending["user_id"])
    return record


async def bearer_for_record(
    record: dict[str, Any],
    context: Any,
    *,
    store: Any = None,
    user_id: str | None = None,
    http_client: Any | None = None,
) -> str | None:
    """Return the bearer a custom connector should present now.

    Static-bearer records decrypt their token; OAuth records refresh through the
    stored token endpoint when the access token is within a minute of expiry.
    """
    from src.anubis.utils.connected_accounts.store import (
        mark_account_needs_reconnect,
        save_connected_account,
    )
    from src.anubis.utils.secret_store import decrypt_secret, encrypt_secret

    if not record.get("encrypted_secret"):
        return None
    transport = record.get("transport") or {}
    if transport.get("auth_type") != "oauth":
        return decrypt_secret(record["encrypted_secret"], context)

    raw = decrypt_secret(record["encrypted_secret"], context)
    try:
        bundle = json.loads(raw)
    except Exception:
        bundle = {}
    expires_at = bundle.get("expires_at")
    if bundle.get("access_token") and (
        expires_at is None or float(expires_at) - 60 > time.time()
    ):
        return str(bundle["access_token"])
    refresh_token = bundle.get("refresh_token")
    oauth_details = dict(transport.get("oauth") or {})
    if not refresh_token or not oauth_details.get("token_endpoint"):
        await mark_account_needs_reconnect(store, user_id or record.get("user_id") or "", record.get("account_key") or "")
        return None
    try:
        document = await _token_request(
            context,
            oauth_details,
            {"grant_type": "refresh_token", "refresh_token": str(refresh_token)},
            http_client=http_client,
        )
    except McpOAuthError:
        await mark_account_needs_reconnect(store, user_id or record.get("user_id") or "", record.get("account_key") or "")
        return None
    expires_in = document.get("expires_in")
    bundle = {
        **bundle,
        "access_token": str(document["access_token"]),
        "refresh_token": document.get("refresh_token") or refresh_token,
        "expires_at": time.time() + float(expires_in) if expires_in else None,
    }
    record["encrypted_secret"] = encrypt_secret(json.dumps(bundle), context)
    try:
        await save_connected_account(store, user_id or record.get("user_id") or "", record)
    except Exception:
        logger.debug("Could not persist a refreshed connector token", exc_info=True)
    return bundle["access_token"]
