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
    MECHANISM_APP_PASSWORD,
    MECHANISM_AUTH0_IDENTITY,
    MECHANISM_DEVICE_PAIRING,
    MECHANISM_MCP_URL,
    MECHANISM_OAUTH,
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


async def _refuse_redirect_mechanism(request: ConnectRequest) -> dict[str, Any]:
    """OAuth and identity linking are declared in the registry but not yet built."""
    raise ConnectRefused(501, request.provider.coming_soon_message())


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
    MECHANISM_OAUTH: _refuse_redirect_mechanism,
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
