"""Find a mail provider's servers from an email address alone.

A person who adds an account to a desktop mail client types an address and a
password and nothing else. The client finds the servers. This module is that
step, so Neural Nexus can ask for the same two things and no more: no host
names, no port numbers, and above all no generated secret.

The ladder below is the one Thunderbird uses, in the same order, because that
order is what the world's mail administrators have actually published against:

1. ``https://autoconfig.<domain>/mail/config-v1.1.xml`` — the provider's own
   answer, served from a host the provider controls.
2. ``https://<domain>/.well-known/autoconfig/mail/config-v1.1.xml`` — the same
   answer for an administrator who cannot add a subdomain.
3. The Mozilla ISPDB at ``https://autoconfig.thunderbird.net/v1.1/<domain>`` —
   a community database covering most consumer providers.
4. The domain's MX record, resolved to the provider actually hosting the mail,
   looked up in the ISPDB again. This is what finds a custom domain hosted by
   somebody else, and it is the rung that makes a company address work.
5. Probing the conventional names — ``imap.<domain>``, ``mail.<domain>``,
   ``smtp.<domain>`` — for IMAP over TLS and submission with STARTTLS.

Every rung is skipped silently on failure and the next is tried; the whole
walk is bounded by one short timeout each, so a domain that answers nothing
costs a couple of seconds and returns ``None`` rather than hanging.

The MX rung resolves DNS over HTTPS rather than through a resolver library.
That is deliberate: it needs no new dependency, no native resolver in the
container image, and it works identically in the API process and in a test.

One more thing this module reports, and the reason it returns a settings
object rather than a bare tuple: **whether the provider still accepts a
password at all**. Google, Microsoft, and Yahoo have each withdrawn password
authentication for mail access, and an autoconfiguration record says so in
its ``authentication`` elements. Knowing that here means the owner is told
which sign-in their provider actually wants, instead of being handed an
authentication failure for a password that was never going to work.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

# Socket types as the autoconfiguration format spells them.
SOCKET_SSL = "SSL"
SOCKET_STARTTLS = "STARTTLS"
SOCKET_PLAIN = "plain"

# Username templates the autoconfiguration format defines.
USERNAME_FULL_ADDRESS = "%EMAILADDRESS%"
USERNAME_LOCAL_PART = "%EMAILLOCALPART%"

# Where each rung looks.
ISPDB_URL_TEMPLATE = "https://autoconfig.thunderbird.net/v1.1/{domain}"
PROVIDER_AUTOCONFIG_URL_TEMPLATE = (
    "https://autoconfig.{domain}/mail/config-v1.1.xml?emailaddress={email_address}"
)
WELL_KNOWN_AUTOCONFIG_URL_TEMPLATE = (
    "https://{domain}/.well-known/autoconfig/mail/config-v1.1.xml"
    "?emailaddress={email_address}"
)
DNS_OVER_HTTPS_URL = "https://dns.google/resolve"

# Authentication values that mean "a password will work here".
PASSWORD_AUTHENTICATION_VALUES = frozenset(
    {"password-cleartext", "plain", "password-encrypted", "cram-md5", "ntlm"}
)

# Providers that have withdrawn password authentication for mail access, keyed
# by the mail host their records resolve to.
#
# The published record cannot be trusted for this. The Mozilla database still
# advertises ``password-cleartext`` for gmail.com years after Google stopped
# honouring it, so a purely record-driven answer sends the owner to type a
# password that is guaranteed to be rejected. Matching on the resolved host
# instead of the address domain is what catches a company's own domain hosted
# by one of these providers, which is the common case and the one a domain
# allow-list would miss.
PASSWORD_WITHDRAWN_HOST_SUFFIXES: dict[str, str] = {
    "gmail.com": "Google",
    "googlemail.com": "Google",
    "google.com": "Google",
    "outlook.com": "Microsoft",
    "outlook.office365.com": "Microsoft",
    "office365.com": "Microsoft",
    "hotmail.com": "Microsoft",
    "live.com": "Microsoft",
    "yahoo.com": "Yahoo",
    "yahoodns.net": "Yahoo",
    "aol.com": "Yahoo",
}


def withdrawn_password_provider(host: str) -> str:
    """Return the provider that no longer accepts a password on ``host``, or ""."""
    hostname = str(host or "").strip().lower().rstrip(".")
    for suffix, provider_name in PASSWORD_WITHDRAWN_HOST_SUFFIXES.items():
        if hostname == suffix or hostname.endswith(f".{suffix}"):
            return provider_name
    return ""

# How long a discovered answer is reused. Mail settings change on the order of
# years; a day keeps a busy signup from re-walking the ladder per attempt.
CACHE_SECONDS = 24 * 60 * 60

# One short budget per rung. The walk as a whole is bounded by the sum.
HTTP_TIMEOUT_SECONDS = 5.0
PROBE_TIMEOUT_SECONDS = 4.0

# Conventional host names, in the order a provider is most likely to use them.
IMAP_HOST_PREFIXES = ("imap", "mail", "")
SMTP_HOST_PREFIXES = ("smtp", "mail", "")

_cache: dict[str, tuple[float, MailServerSettings | None]] = {}


@dataclass(frozen=True)
class MailServerSettings:
    """Where one domain's mail lives, and how a client is meant to sign in.

    Attributes:
        imap_host: IMAP server host name.
        imap_port: IMAP port.
        imap_socket_type: One of :data:`SOCKET_SSL`, :data:`SOCKET_STARTTLS`,
            :data:`SOCKET_PLAIN`.
        smtp_host: Submission server host name.
        smtp_port: Submission port.
        smtp_socket_type: As ``imap_socket_type``, for submission.
        username_template: ``%EMAILADDRESS%`` or ``%EMAILLOCALPART%`` — some
            providers want only the part before the "@" as the user name.
        display_name: The provider's own name for itself, when it gave one.
        password_authentication: Whether the provider still accepts a password
            for mail access. ``False`` means the account has to be connected a
            different way, and ``authentication_methods`` says which.
        authentication_methods: Every authentication value the record listed,
            lower-cased, so a caller can explain the alternative by name.
        password_withdrawn_by: The provider that no longer accepts a password
            on this host ("Google", "Microsoft", "Yahoo"), or an empty string.
            Named so the owner is told which company made the change rather
            than being shown a bare refusal.
        source: Which rung answered. Carried for the connection record and for
            support questions, never shown as jargon to the owner.
    """

    imap_host: str
    imap_port: int
    smtp_host: str
    smtp_port: int
    imap_socket_type: str = SOCKET_SSL
    smtp_socket_type: str = SOCKET_STARTTLS
    username_template: str = USERNAME_FULL_ADDRESS
    display_name: str = ""
    password_authentication: bool = True
    authentication_methods: tuple[str, ...] = ()
    password_withdrawn_by: str = ""
    source: str = ""

    def username_for(self, email_address: str) -> str:
        """Return the user name this provider expects for ``email_address``."""
        if self.username_template == USERNAME_LOCAL_PART:
            return email_address.split("@", 1)[0]
        return email_address


@dataclass
class _ServerRecord:
    """One ``incomingServer`` or ``outgoingServer`` element, parsed."""

    host: str = ""
    port: int = 0
    socket_type: str = SOCKET_SSL
    username_template: str = USERNAME_FULL_ADDRESS
    authentication_methods: tuple[str, ...] = field(default_factory=tuple)


def domain_of(email_address: str) -> str:
    """Return the domain part of an address, lower-cased, or an empty string."""
    _, _, domain = str(email_address or "").strip().partition("@")
    return domain.strip().lower()


def _text(element: Any, tag: str) -> str:
    child = element.find(tag)
    return (child.text or "").strip() if child is not None and child.text else ""


def _socket_type(raw_value: str) -> str:
    value = raw_value.strip().upper()
    if value == "SSL":
        return SOCKET_SSL
    if value == "STARTTLS":
        return SOCKET_STARTTLS
    return SOCKET_PLAIN


def _parse_server(element: Any) -> _ServerRecord | None:
    host = _text(element, "hostname")
    port_text = _text(element, "port")
    if not host or not port_text.isdigit():
        return None
    username_template = _text(element, "username") or USERNAME_FULL_ADDRESS
    methods = tuple(
        (child.text or "").strip().lower()
        for child in element.findall("authentication")
        if (child.text or "").strip()
    )
    return _ServerRecord(
        host=host,
        port=int(port_text),
        socket_type=_socket_type(_text(element, "socketType")),
        username_template=username_template,
        authentication_methods=methods,
    )


def parse_autoconfig_document(document_text: str, *, source: str) -> MailServerSettings | None:
    """Turn one autoconfiguration XML document into settings, or ``None``.

    Accepts any of the ``clientConfig`` documents the four HTTP rungs return —
    the provider's own, the well-known path, and the Mozilla database all use
    the same schema. A document that names no IMAP server is not an answer.
    """
    try:
        root = ElementTree.fromstring(document_text)
    except ElementTree.ParseError:
        return None

    provider_element = root.find("emailProvider")
    if provider_element is None:
        return None

    incoming: _ServerRecord | None = None
    for element in provider_element.findall("incomingServer"):
        if (element.get("type") or "").strip().lower() != "imap":
            continue
        parsed = _parse_server(element)
        if parsed is None:
            continue
        # Prefer an encrypted transport when the provider offers several.
        if incoming is None or (
            incoming.socket_type == SOCKET_PLAIN and parsed.socket_type != SOCKET_PLAIN
        ):
            incoming = parsed
    if incoming is None:
        return None

    outgoing: _ServerRecord | None = None
    for element in provider_element.findall("outgoingServer"):
        if (element.get("type") or "").strip().lower() != "smtp":
            continue
        parsed = _parse_server(element)
        if parsed is None:
            continue
        if outgoing is None or (
            outgoing.socket_type == SOCKET_PLAIN and parsed.socket_type != SOCKET_PLAIN
        ):
            outgoing = parsed

    methods = tuple(dict.fromkeys(incoming.authentication_methods))
    withdrawn_by = withdrawn_password_provider(incoming.host)
    return MailServerSettings(
        imap_host=incoming.host,
        imap_port=incoming.port,
        imap_socket_type=incoming.socket_type,
        smtp_host=outgoing.host if outgoing else "",
        smtp_port=outgoing.port if outgoing else 587,
        smtp_socket_type=outgoing.socket_type if outgoing else SOCKET_STARTTLS,
        username_template=incoming.username_template,
        display_name=_text(provider_element, "displayName"),
        # An absent authentication element is the older, permissive form and
        # means a password; an element list that names only OAuth means the
        # provider has withdrawn password access; and a host belonging to a
        # provider known to have withdrawn it overrides whatever the record
        # claims, because several of those records are years out of date.
        password_authentication=(
            not withdrawn_by
            and (
                True
                if not methods
                else any(method in PASSWORD_AUTHENTICATION_VALUES for method in methods)
            )
        ),
        authentication_methods=methods,
        password_withdrawn_by=withdrawn_by,
        source=source,
    )


async def _fetch_document(http_client: Any, url: str) -> str | None:
    try:
        response = await http_client.get(url, timeout=HTTP_TIMEOUT_SECONDS)
    except Exception:  # noqa: BLE001 - any transport failure is just "no answer"
        return None
    if response.status_code != 200:
        return None
    body = response.text or ""
    return body if "<clientConfig" in body else None


async def _mail_exchange_domains(http_client: Any, domain: str) -> list[str]:
    """Return the provider domains behind ``domain``'s MX records, best effort.

    Resolved over HTTPS so no resolver library is required. ``mx1.mail.host.com``
    contributes ``host.com``, which is what the database is keyed by.
    """
    try:
        response = await http_client.get(
            DNS_OVER_HTTPS_URL,
            params={"name": domain, "type": "MX"},
            headers={"accept": "application/dns-json"},
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            return []
        payload = response.json()
    except Exception:  # noqa: BLE001 - no DNS answer is not an error here
        return []

    ranked: list[tuple[int, str]] = []
    for answer in payload.get("Answer") or []:
        if int(answer.get("type") or 0) != 15:
            continue
        parts = str(answer.get("data") or "").split()
        if len(parts) != 2:
            continue
        preference_text, exchange = parts
        exchange = exchange.strip().rstrip(".").lower()
        if not exchange:
            continue
        try:
            preference = int(preference_text)
        except ValueError:
            preference = 0
        ranked.append((preference, exchange))

    candidates: list[str] = []
    for _, exchange in sorted(ranked):
        labels = exchange.split(".")
        # Walk up from the fully qualified exchange to its registrable domain,
        # so both "google.com" and "aspmx.l.google.com" get a chance.
        for index in range(len(labels) - 1):
            candidate = ".".join(labels[index:])
            if candidate.count(".") >= 1 and candidate not in candidates:
                candidates.append(candidate)
    return candidates[:6]


def _probe_imap(host: str, port: int, socket_type: str) -> bool:
    import imaplib

    try:
        if socket_type == SOCKET_SSL:
            connection = imaplib.IMAP4_SSL(host, port, timeout=PROBE_TIMEOUT_SECONDS)
        else:
            connection = imaplib.IMAP4(host, port, timeout=PROBE_TIMEOUT_SECONDS)
            connection.starttls()
    except Exception:  # noqa: BLE001 - a host that does not answer is not a match
        return False
    try:
        connection.logout()
    except Exception:  # noqa: BLE001 - the probe already succeeded
        pass
    return True


def _probe_smtp(host: str, port: int, socket_type: str) -> bool:
    import smtplib

    try:
        if socket_type == SOCKET_SSL:
            connection = smtplib.SMTP_SSL(host, port, timeout=PROBE_TIMEOUT_SECONDS)
        else:
            connection = smtplib.SMTP(host, port, timeout=PROBE_TIMEOUT_SECONDS)
            connection.ehlo()
            connection.starttls()
    except Exception:  # noqa: BLE001 - a host that does not answer is not a match
        return False
    try:
        connection.quit()
    except Exception:  # noqa: BLE001 - the probe already succeeded
        pass
    return True


async def _probe_conventional_hosts(domain: str) -> MailServerSettings | None:
    """Try the conventional host names — the last rung of the ladder."""
    imap_match: tuple[str, int, str] | None = None
    for prefix in IMAP_HOST_PREFIXES:
        host = f"{prefix}.{domain}" if prefix else domain
        for port, socket_type in ((993, SOCKET_SSL), (143, SOCKET_STARTTLS)):
            if await asyncio.to_thread(_probe_imap, host, port, socket_type):
                imap_match = (host, port, socket_type)
                break
        if imap_match:
            break
    if imap_match is None:
        return None

    smtp_match: tuple[str, int, str] | None = None
    for prefix in SMTP_HOST_PREFIXES:
        host = f"{prefix}.{domain}" if prefix else domain
        for port, socket_type in ((587, SOCKET_STARTTLS), (465, SOCKET_SSL)):
            if await asyncio.to_thread(_probe_smtp, host, port, socket_type):
                smtp_match = (host, port, socket_type)
                break
        if smtp_match:
            break

    imap_host, imap_port, imap_socket_type = imap_match
    smtp_host, smtp_port, smtp_socket_type = smtp_match or (
        f"smtp.{domain}",
        587,
        SOCKET_STARTTLS,
    )
    return MailServerSettings(
        imap_host=imap_host,
        imap_port=imap_port,
        imap_socket_type=imap_socket_type,
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_socket_type=smtp_socket_type,
        password_withdrawn_by=withdrawn_password_provider(imap_host),
        password_authentication=not withdrawn_password_provider(imap_host),
        source="probe",
    )


async def discover_mail_settings(
    email_address: str,
    *,
    http_client: Any = None,
    use_cache: bool = True,
) -> MailServerSettings | None:
    """Find the servers for ``email_address``, or ``None`` if nothing answers.

    Walks the five rungs in order and returns the first answer. Never raises
    for a domain that simply does not publish anything: the caller's next move
    is to ask the owner for the server names, not to show a stack trace.
    """
    domain = domain_of(email_address)
    if not domain or not re.match(r"^[a-z0-9.-]+\.[a-z]{2,}$", domain):
        return None

    now = time.monotonic()
    if use_cache:
        cached = _cache.get(domain)
        if cached is not None and now - cached[0] < CACHE_SECONDS:
            return cached[1]

    settings = await _walk_rungs(domain, email_address, http_client)
    if use_cache:
        _cache[domain] = (now, settings)
    return settings


async def _walk_rungs(
    domain: str, email_address: str, http_client: Any
) -> MailServerSettings | None:
    import httpx

    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(follow_redirects=True)
    try:
        provider_url = PROVIDER_AUTOCONFIG_URL_TEMPLATE.format(
            domain=domain, email_address=email_address
        )
        document = await _fetch_document(client, provider_url)
        if document:
            settings = parse_autoconfig_document(document, source="provider_autoconfig")
            if settings:
                return settings

        well_known_url = WELL_KNOWN_AUTOCONFIG_URL_TEMPLATE.format(
            domain=domain, email_address=email_address
        )
        document = await _fetch_document(client, well_known_url)
        if document:
            settings = parse_autoconfig_document(document, source="well_known_autoconfig")
            if settings:
                return settings

        document = await _fetch_document(client, ISPDB_URL_TEMPLATE.format(domain=domain))
        if document:
            settings = parse_autoconfig_document(document, source="ispdb")
            if settings:
                return settings

        for exchange_domain in await _mail_exchange_domains(client, domain):
            document = await _fetch_document(
                client, ISPDB_URL_TEMPLATE.format(domain=exchange_domain)
            )
            if not document:
                continue
            settings = parse_autoconfig_document(document, source="ispdb_mail_exchange")
            if settings:
                return settings
    finally:
        if owns_client:
            await client.aclose()

    return await _probe_conventional_hosts(domain)


def clear_cache() -> None:
    """Forget every discovered answer. Used by tests."""
    _cache.clear()
