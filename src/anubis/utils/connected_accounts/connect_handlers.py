"""How each credential mechanism turns a filled-in connect card into a record.

``POST /connect_account`` looks up the provider, then dispatches on the
provider's ``credential_mechanism`` through :data:`CONNECT_HANDLERS`. Each
handler has the same contract — take the submitted fields, PROVE the connection
works, and return the record to store — so a new mechanism is one function
here and a new provider of an existing mechanism is a registry row alone.

The proving step is the point. A mailbox password is proved by a real IMAP
login; a Model Context Protocol server address is proved by listing the
server's tools. Either failure is reported while the owner still has the card
in front of them, with a message that says what to fix, instead of being stored
and failing on the first use days later.

Nothing here persists anything. The route owns the cap check and the write so
the handlers stay pure functions over their inputs and are trivially testable.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from src.anubis.utils.connected_accounts.providers import (
    LOGIN_MODE_FORM,
    MECHANISM_APP_PASSWORD,
    MECHANISM_AUTH0_IDENTITY,
    MECHANISM_BROWSER_SESSION,
    MECHANISM_DEVICE_PAIRING,
    MECHANISM_MCP_URL,
    MECHANISM_OAUTH,
    MECHANISM_PLAID_LINK,
    MECHANISM_URL_ONLY,
    ConnectedAccountProvider,
)
from src.anubis.utils.connected_accounts.store import (
    account_key,
    build_account_record,
    deduplicate_label,
    derive_display_label,
)


def server_address_for(server_url: str) -> str:
    """Return the display-safe address a custom server is recorded under.

    The shape is ``host#digest``: the host is safe to show, and the digest of the
    full URL keeps one record per server without ever storing the URL in a field
    that is read back to the owner.
    """
    import hashlib
    from urllib.parse import urlparse

    parsed = urlparse(str(server_url).strip())
    host = parsed.hostname or parsed.netloc or "server"
    digest = hashlib.sha256(str(server_url).strip().lower().encode("utf-8")).hexdigest()
    return f"{host}#{digest[:10]}"


class ConnectRefused(Exception):
    """The connection cannot be made; carries the HTTP status and the reason."""

    def __init__(self, status_code: int, detail: str) -> None:
        """Carry the HTTP status the route should answer with, and why."""
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class ConnectNeedsLogin(Exception):
    """The connection needs the owner to sign in through a popup first.

    Raised instead of a record when a provider's flow is not a form: the route
    answers ``200 {"connected": false, "action": "open_login_popup", ...}`` and
    the card opens the login endpoint. ``login_request`` is the body the card
    posts to start the popup (provider, and for a custom connector the server
    URL and name the owner already typed).
    """

    def __init__(
        self,
        provider: ConnectedAccountProvider,
        login_request: dict[str, Any],
        *,
        login_mode: str | None = None,
        message: str = "",
    ) -> None:
        """Carry the provider, the popup mode, and the body the card posts."""
        super().__init__(message or f"{provider.display_name} needs a sign-in.")
        self.provider = provider
        self.login_mode = login_mode or provider.login_mode
        self.login_endpoint = provider.login_endpoint
        self.login_request = dict(login_request)
        self.message = message or (
            f"Sign in to {provider.display_name} in the window that opens."
        )

    def as_response(self) -> dict[str, Any]:
        """Return the JSON body the route answers with."""
        from src.anubis.utils.connected_accounts.connection_tools import (
            build_connect_card,
        )

        card = build_connect_card(self.provider)
        card["login_request"] = self.login_request
        return {
            "connected": False,
            "action": "open_login_popup",
            "login_mode": self.login_mode,
            "login_endpoint": self.login_endpoint,
            "login_request": self.login_request,
            "message": self.message,
            "card": card,
        }


@dataclass
class ConnectRequest:
    """Everything a handler needs to prove and describe one connection."""

    provider: ConnectedAccountProvider
    fields: dict[str, Any]
    assistant_id: str
    context: Any
    existing_records: list[dict[str, Any]] = field(default_factory=list)

    def text(self, name: str) -> str:
        """Return one submitted field as stripped text (empty when absent)."""
        return str(self.fields.get(name) or "").strip()


ConnectHandler = Callable[[ConnectRequest], Awaitable[dict[str, Any]]]


def _encrypt(secret: str, context: Any) -> str:
    from src.anubis.utils.secret_store import (
        SecretEncryptionNotConfiguredError,
        encrypt_secret,
    )

    try:
        return encrypt_secret(secret, context)
    except SecretEncryptionNotConfiguredError as configuration_error:
        raise ConnectRefused(503, str(configuration_error))


async def connect_app_password_account(request: ConnectRequest) -> dict[str, Any]:
    """Prove a mailbox address + app password by logging in, then describe it.

    On Gmail the password must be a 16-character app password, not the account
    password: Google stopped accepting account passwords over IMAP on
    2025-03-14, and creating an app password requires 2-Step Verification. A
    rejected credential says exactly that and links to the page that issues one,
    because "authentication failed" alone sends people to re-type the same wrong
    secret.
    """
    from src.anubis.utils.tools.email.imap_client import (
        MailboxAuthenticationError,
        MailboxCredentials,
        MailboxUnreachableError,
        verify_credentials,
    )

    provider = request.provider
    email_address = request.text("email_address")
    app_password = str(request.fields.get("app_password") or "")
    if not provider.is_mailbox:
        raise ConnectRefused(
            400,
            f"{provider.display_name} is not a mailbox and cannot be connected "
            "with an email address and password.",
        )
    if not email_address or not app_password:
        raise ConnectRefused(400, "Both email_address and app_password are required.")

    credentials = MailboxCredentials(
        account_address=email_address,
        password=app_password,
        imap_host=provider.imap_host,
        imap_port=provider.imap_port,
        smtp_host=provider.smtp_host,
        smtp_port=provider.smtp_port,
        drafts_mailbox=provider.drafts_mailbox,
        timeout_seconds=float(
            getattr(request.context, "mailbox_request_timeout_seconds", None) or 30.0
        ),
    )
    try:
        await asyncio.to_thread(verify_credentials, credentials)
    except MailboxAuthenticationError:
        raise ConnectRefused(
            400,
            f"{provider.display_name} rejected that address and password. "
            "Use a 16-character app password, not your account password — "
            "Google stopped accepting account passwords for mail access on "
            "14 March 2025. Creating one requires 2-Step Verification: "
            f"{provider.credential_help_url}",
        )
    except MailboxUnreachableError as unreachable_error:
        raise ConnectRefused(
            503,
            f"Could not reach {provider.display_name} to check the credential: "
            f"{unreachable_error}",
        )

    encrypted_secret = _encrypt(app_password, request.context)
    key = account_key(provider.name, email_address)
    label = deduplicate_label(
        derive_display_label(email_address), request.existing_records, key
    )
    return build_account_record(
        provider=provider,
        account_address=email_address,
        display_label=label,
        encrypted_secret=encrypted_secret,
        assistant_id=request.assistant_id,
    )


async def connect_mcp_server_account(request: ConnectRequest) -> dict[str, Any]:
    """Prove a Model Context Protocol server by listing its tools, then describe it.

    The record's ``account_address`` is the server URL, so reconnecting the same
    server refreshes one record; the transport details carry the URL, the
    inferred transport, and the tool names the probe returned so the catalog
    can say how many tools the connector adds without dialing the server.
    """
    from src.anubis.utils.connected_accounts.mcp_server_tools import (
        McpServerUnreachableError,
        infer_transport,
        probe_server_tools,
    )

    provider = request.provider
    server_url = request.text("server_url")
    name = request.text("name")
    bearer_token = str(request.fields.get("bearer_token") or "").strip() or None

    if not server_url:
        raise ConnectRefused(400, "A server_url is required.")
    lowered = server_url.lower()
    if not (lowered.startswith("https://") or lowered.startswith("http://")):
        raise ConnectRefused(
            400,
            "The server URL must start with https:// (or http:// for a local server).",
        )
    if not name:
        name = server_url.split("//", 1)[-1].split("/", 1)[0] or "Custom connector"

    timeout_seconds = float(
        getattr(request.context, "mcp_connector_probe_timeout_seconds", None) or 20.0
    )
    if bearer_token is None:
        # No token typed: ask the server how a client is expected to sign in.
        # A 401 with authorization metadata means an OAuth popup; a bare 401
        # means a static token, which the card's form collects.
        from src.anubis.utils.connected_accounts.mcp_oauth import (
            AUTHORIZATION_NEEDS_OAUTH,
            AUTHORIZATION_NEEDS_TOKEN,
            AUTHORIZATION_UNREACHABLE,
            probe_authorization,
        )

        probe = await probe_authorization(server_url, request.context)
        if probe["status"] == AUTHORIZATION_UNREACHABLE:
            raise ConnectRefused(
                400,
                f"The server at {server_url} could not be reached: "
                f"{probe.get('error') or 'no answer'}. Check the URL.",
            )
        if probe["status"] == AUTHORIZATION_NEEDS_OAUTH:
            raise ConnectNeedsLogin(
                provider,
                {"provider": provider.name, "server_url": server_url, "name": name},
                login_mode="oauth_popup",
                message=(
                    f"{name} asks you to sign in. Sign in in the window that opens."
                ),
            )
        if probe["status"] == AUTHORIZATION_NEEDS_TOKEN:
            raise ConnectNeedsLogin(
                provider,
                {"provider": provider.name, "server_url": server_url, "name": name},
                login_mode=LOGIN_MODE_FORM,
                message=(
                    f"{name} requires an access token. Paste the token the "
                    "server gave you."
                ),
            )
    try:
        tools = await probe_server_tools(server_url, bearer_token, timeout_seconds)
    except McpServerUnreachableError as unreachable_error:
        raise ConnectRefused(
            400,
            f"{unreachable_error} Check the URL, and the access token if the "
            "server requires one.",
        )

    tool_names = sorted(
        {
            str(getattr(tool, "name", "") or "")
            for tool in tools
            if getattr(tool, "name", "")
        }
    )
    encrypted_secret = _encrypt(bearer_token, request.context) if bearer_token else None
    # The record is addressed by the server's host plus a digest of the full URL
    # rather than by the URL itself: a URL can embed a credential in its path or
    # query, and the account address is shown back to the owner in every
    # listing. The digest keeps "reconnect the same server" refreshing one
    # record while the full URL lives only in the transport details, which the
    # public projection never returns.
    account_address = server_address_for(server_url)
    key = account_key(provider.name, account_address)
    label = deduplicate_label(name, request.existing_records, key)
    return build_account_record(
        provider=provider,
        account_address=account_address,
        display_label=label,
        encrypted_secret=encrypted_secret,
        assistant_id=request.assistant_id,
        transport={
            "server_url": server_url,
            "transport": infer_transport(server_url),
            "tool_names": tool_names,
        },
    )


async def _needs_popup_login(request: ConnectRequest) -> dict[str, Any]:
    """OAuth, Plaid Link, and browser sign-ins happen in a popup, not a form.

    The card posts to this route only when a client predates popup logins or
    when the owner typed a site address for a custom site; either way the
    answer is "open the login popup", with whatever the owner typed carried
    along so the popup starts on the right page.
    """
    provider = request.provider
    login_request: dict[str, Any] = {"provider": provider.name}
    for name in ("site_url", "name", "server_url"):
        value = request.text(name)
        if value:
            login_request[name] = value
    if provider.credential_mechanism == MECHANISM_BROWSER_SESSION and not (
        provider.login_url or login_request.get("site_url")
    ):
        raise ConnectRefused(
            400, f"A site_url is required to sign in to {provider.display_name}."
        )
    raise ConnectNeedsLogin(provider, login_request)


async def _refuse_redirect_mechanism(request: ConnectRequest) -> dict[str, Any]:
    """Identity linking through the login provider is not offered."""
    raise ConnectRefused(501, request.provider.coming_soon_message())


def normalize_site_url(site_url: str) -> str:
    """Return a website address with a scheme, or raise ``ConnectRefused``."""
    from urllib.parse import urlparse

    candidate = str(site_url or "").strip()
    if not candidate:
        raise ConnectRefused(400, "A site_url is required.")
    if "://" not in candidate:
        candidate = "https://" + candidate
    parsed = urlparse(candidate)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ConnectRefused(400, f"{site_url!r} is not a website address.")
    return candidate


async def connect_website(request: ConnectRequest) -> dict[str, Any]:
    """Prove a website address by fetching the home page, then describe it.

    No credential is involved: the record carries the address, the page title
    the fetch found, and the hostname the crawl is confined to.
    """
    provider = request.provider
    site_url = normalize_site_url(request.text("site_url"))
    name = request.text("name")
    from urllib.parse import urlparse

    hostname = urlparse(site_url).hostname or site_url
    title = ""
    status_code = None
    try:
        import httpx

        timeout_seconds = float(
            getattr(request.context, "connect_oauth_http_timeout_seconds", None)
            or 15.0
        )
        async with httpx.AsyncClient(
            timeout=timeout_seconds, follow_redirects=True
        ) as client:
            response = await client.get(
                site_url, headers={"User-Agent": "NeuralNexus/1.0 (+website connector)"}
            )
        status_code = response.status_code
        if status_code >= 400:
            raise ConnectRefused(
                400, f"{site_url} answered {status_code}; check the address."
            )
        text = response.text or ""
        lowered = text.lower()
        start = lowered.find("<title")
        if start != -1:
            start = lowered.find(">", start)
            end = lowered.find("</title>", start)
            if start != -1 and end != -1:
                title = " ".join(text[start + 1 : end].split())[:200]
    except ConnectRefused:
        raise
    except Exception as fetch_error:
        raise ConnectRefused(
            400, f"{site_url} could not be reached: {fetch_error}"
        ) from fetch_error

    account_address = hostname.lower()
    key = account_key(provider.name, account_address)
    label = deduplicate_label(name or title or hostname, request.existing_records, key)
    return build_account_record(
        provider=provider,
        account_address=account_address,
        display_label=label,
        encrypted_secret=None,
        assistant_id=request.assistant_id,
        transport={
            "site_url": site_url,
            "hostname": hostname,
            "title": title,
            "status_code": status_code,
        },
    )


async def _refuse_device_pairing(request: ConnectRequest) -> dict[str, Any]:
    """Devices connect themselves; the card carries the instructions."""
    raise ConnectRefused(
        400,
        f"{request.provider.display_name} are connected by running the Neural "
        f"Nexus daemon, not from this form. {request.provider.pairing_instructions}",
    )


CONNECT_HANDLERS: dict[str, ConnectHandler] = {
    MECHANISM_APP_PASSWORD: connect_app_password_account,
    MECHANISM_MCP_URL: connect_mcp_server_account,
    MECHANISM_URL_ONLY: connect_website,
    MECHANISM_OAUTH: _needs_popup_login,
    MECHANISM_PLAID_LINK: _needs_popup_login,
    MECHANISM_BROWSER_SESSION: _needs_popup_login,
    MECHANISM_AUTH0_IDENTITY: _refuse_redirect_mechanism,
    MECHANISM_DEVICE_PAIRING: _refuse_device_pairing,
}


async def connect_account(request: ConnectRequest) -> dict[str, Any]:
    """Dispatch one connect request to its mechanism's handler.

    Availability is checked first, so a coming-soon provider is refused with its
    plain message regardless of mechanism, and an unknown mechanism — impossible
    once ``validate_registry`` has run, but cheap to guard — is refused rather
    than raising a ``KeyError`` into the route.
    """
    provider = request.provider
    if not provider.is_available:
        raise ConnectRefused(501, provider.coming_soon_message())
    handler = CONNECT_HANDLERS.get(provider.credential_mechanism)
    if handler is None:
        raise ConnectRefused(
            400,
            f"{provider.display_name} declares a connection mechanism this server "
            "does not implement.",
        )
    return await handler(request)
