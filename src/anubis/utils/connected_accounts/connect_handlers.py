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
    KIND_CALENDAR,
    LOGIN_MODE_FORM,
    MECHANISM_AUTH0_IDENTITY,
    MECHANISM_BROWSER_SESSION,
    MECHANISM_DEVICE_PAIRING,
    MECHANISM_MCP_URL,
    MECHANISM_OAUTH,
    MECHANISM_PASSWORD,
    MECHANISM_PLAID_LINK,
    MECHANISM_SITE_DISCOVERY,
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


async def connect_password_account(request: ConnectRequest) -> dict[str, Any]:
    """Prove a mailbox address and account password by logging in, then describe it.

    This is the path a desktop mail client takes, and the reason the owner is
    asked for two things and no more. When the provider row names no servers —
    the generic email row — they are discovered from the address
    (``mail_autoconfig``) and written onto the record, so every later turn
    reaches the mailbox without rediscovering anything.

    Three failures are three different answers, because they need three
    different actions from the owner:

    * the provider has withdrawn password access — say which company did that
      and which sign-in it wants instead, since no password will ever work;
    * nothing answered at the address's domain — ask for the server names;
    * the server answered and rejected the credential — say the password was
      refused, and nothing else.

    A single "authentication failed" for all three is what sends a person to
    type the same rejected password a second time.
    """
    from src.anubis.utils.connected_accounts.mail_autoconfig import (
        discover_mail_settings,
        domain_of,
        withdrawn_password_provider,
    )
    from src.anubis.utils.tools.email.imap_client import (
        MailboxAuthenticationError,
        MailboxCredentials,
        MailboxUnreachableError,
        verify_credentials,
    )

    provider = request.provider
    email_address = request.text("email_address")
    # ``app_password`` is the field name older clients posted. Accepted so a
    # stale browser tab still connects; never offered, never labelled.
    password = str(
        request.fields.get("password") or request.fields.get("app_password") or ""
    )
    if provider.kind == KIND_CALENDAR:
        return await _connect_calendar_account(request, email_address, password)
    if not provider.is_mailbox:
        raise ConnectRefused(
            400,
            f"{provider.display_name} is not a mailbox and cannot be connected "
            "with an email address and password.",
        )
    if not email_address or not password:
        raise ConnectRefused(400, "Both email_address and password are required.")

    overrides: dict[str, Any] = {}
    imap_host = provider.imap_host
    imap_port = provider.imap_port
    smtp_host = provider.smtp_host
    smtp_port = provider.smtp_port
    drafts_mailbox = provider.drafts_mailbox
    username = email_address

    if not imap_host:
        # The owner may have typed the servers themselves for a domain that
        # publishes nothing; that always wins over discovery.
        typed_imap_host = request.text("imap_host")
        typed_smtp_host = request.text("smtp_host")
        if typed_imap_host:
            imap_host = typed_imap_host
            smtp_host = typed_smtp_host or typed_imap_host
        else:
            settings = await discover_mail_settings(email_address)
            if settings is None:
                raise ConnectRefused(
                    400,
                    f"No mail settings could be found for {domain_of(email_address)}. "
                    "Enter the incoming and outgoing server names for this "
                    "account and connect again.",
                )
            if not settings.password_authentication:
                withdrawn_by = settings.password_withdrawn_by or "This provider"
                raise ConnectRefused(
                    400,
                    f"{withdrawn_by} no longer accepts an account password for "
                    "mail access, so this address cannot be connected with a "
                    "password. Connect it with the sign-in button for "
                    f"{withdrawn_by} instead — the same address and password, "
                    "typed on their own page.",
                )
            imap_host = settings.imap_host
            imap_port = settings.imap_port
            smtp_host = settings.smtp_host or settings.imap_host
            smtp_port = settings.smtp_port
            username = settings.username_for(email_address)
        overrides = {
            "imap_host": imap_host,
            "imap_port": imap_port,
            "smtp_host": smtp_host,
            "smtp_port": smtp_port,
        }
    elif withdrawn_password_provider(imap_host):
        withdrawn_by = withdrawn_password_provider(imap_host)
        raise ConnectRefused(
            400,
            f"{withdrawn_by} no longer accepts an account password for mail "
            f"access. Connect {provider.display_name} with its sign-in button "
            "instead — the same address and password, typed on their own page.",
        )

    credentials = MailboxCredentials(
        account_address=username,
        password=password,
        imap_host=imap_host,
        imap_port=imap_port,
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        drafts_mailbox=drafts_mailbox,
        timeout_seconds=float(
            getattr(request.context, "mailbox_request_timeout_seconds", None) or 30.0
        ),
    )
    try:
        await asyncio.to_thread(verify_credentials, credentials)
    except MailboxAuthenticationError:
        raise ConnectRefused(
            400,
            f"{imap_host} rejected that password for {email_address}. Check the "
            "password you use to sign in to this email account and try again.",
        )
    except MailboxUnreachableError as unreachable_error:
        raise ConnectRefused(
            503,
            f"Could not reach {imap_host} to check the credential: "
            f"{unreachable_error}",
        )

    encrypted_secret = _encrypt(password, request.context)
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
        connection_overrides=overrides or None,
    )


# The name this handler carried when the mechanism was misnamed. Kept so an
# existing import resolves; new code calls ``connect_password_account``.
connect_app_password_account = connect_password_account


async def _connect_calendar_account(
    request: ConnectRequest, email_address: str, password: str
) -> dict[str, Any]:
    """Prove a calendar account over CalDAV, then describe it.

    The same two things the owner typed for mail, against the calendar's own
    protocol. The proven principal and calendar-home addresses are kept on the
    record so no later turn repeats discovery.
    """
    from src.anubis.utils.connected_accounts.caldav_client import (
        CalDavAuthenticationError,
        CalDavUnreachableError,
        connect_caldav_account,
        list_calendars,
    )

    provider = request.provider
    if not email_address or not password:
        raise ConnectRefused(400, "Both email_address and password are required.")

    try:
        account = await connect_caldav_account(
            email_address=email_address,
            password=password,
            server_url=request.text("server_url"),
        )
        calendars = await list_calendars(account)
    except CalDavAuthenticationError:
        raise ConnectRefused(
            400,
            f"The calendar server rejected that password for {email_address}. "
            "Check the password you use to sign in to this account and try again.",
        )
    except CalDavUnreachableError as unreachable_error:
        raise ConnectRefused(
            400,
            f"No calendar server could be found for {email_address}: "
            f"{unreachable_error} Enter the calendar server address and try again.",
        )

    encrypted_secret = _encrypt(password, request.context)
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
        transport={
            "caldav": {
                "base_url": account.base_url,
                "principal_url": account.principal_url,
                "calendar_home_url": account.calendar_home_url,
                "username": account.username,
                "calendars": [
                    {"name": calendar.display_name, "read_only": calendar.read_only}
                    for calendar in calendars
                ],
            }
        },
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


async def connect_site_by_discovery(request: ConnectRequest) -> dict[str, Any]:
    """Connect a site the way the site itself offers, or say that it offers none.

    The owner names a site; this asks the site how it wants to be reached. A
    Model Context Protocol server is the answer whenever there is one, because
    it needs no application registered anywhere and no credential typed into
    Neural Nexus — the server states how to sign in and registers this client
    itself.

    Once found, the address is handed to the Model Context Protocol handler,
    which already knows how to prove a server, ask for a sign-in, and describe
    the tools. Discovery adds no second copy of any of that.
    """
    from src.anubis.utils.connected_accounts.mcp_discovery import (
        discover_mcp_server,
        normalize_site,
    )
    from src.anubis.utils.connected_accounts.providers import get_provider

    site = request.text("site_url")
    if not site:
        raise ConnectRefused(400, "A site address is required.")
    origin, host = normalize_site(site)
    if not origin:
        raise ConnectRefused(400, f"{site!r} is not a web address.")

    found = await discover_mcp_server(origin, request.context)
    if found is None:
        raise ConnectRefused(
            400,
            f"{host} does not offer a Model Context Protocol server, so there "
            "is no supported way for the avatar to use an account there. If "
            f"{host} publishes an API key or a connector address, connect it "
            "as a custom connector instead.",
        )

    mcp_provider = get_provider("custom_mcp") or request.provider
    return await connect_mcp_server_account(
        ConnectRequest(
            provider=mcp_provider,
            fields={
                "server_url": found.server_url,
                "name": request.text("name") or found.name or host,
            },
            assistant_id=request.assistant_id,
            context=request.context,
            existing_records=request.existing_records,
        )
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
    MECHANISM_PASSWORD: connect_password_account,
    MECHANISM_MCP_URL: connect_mcp_server_account,
    MECHANISM_URL_ONLY: connect_website,
    MECHANISM_SITE_DISCOVERY: connect_site_by_discovery,
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
