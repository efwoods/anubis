"""Refuse URLs that resolve to an address inside this deployment's network.

Every remote byte the API downloads used to come from a URL a signed-in person
typed, so a URL pointing at ``127.0.0.1`` or the cloud metadata endpoint was a
person attacking their own deployment. Deep research changed that: the portrait
the research acquires comes from a search engine and a language model, and a
page under someone else's control can hand the pipeline any URL it likes. That
is a server-side request forgery, and the guard in this module is what stops it.

Two rules do the work:

* Resolve the host and refuse when ANY address it resolves to is private,
  loopback, link-local, reserved, multicast or unspecified. A hostname is not
  safe because it looks public — ``metadata.example.com`` may resolve to
  ``169.254.169.254`` — so the check is on the resolved addresses, never on the
  name.
* Follow redirects manually and re-validate every hop. ``follow_redirects=True``
  is precisely what lets a public URL answer ``302 Location: http://169.254.169.254/``
  and have httpx fetch it without the caller ever seeing the address.

``allow_private`` exists for development, where a fixture may legitimately be
served from ``localhost``; it is read from ``URL_FETCH_ALLOW_PRIVATE_HOSTS`` at
the call site rather than defaulting to permissive here.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# Schemes that may reach the network at all. ``file://``, ``gopher://`` and
# friends are not downloads, they are ways to read the container's filesystem.
ALLOWED_URL_SCHEMES = ("http", "https")

DEFAULT_MAX_REDIRECTS = 5


class UnsafeUrlError(ValueError):
    """The URL is malformed, uses a forbidden scheme, or resolves internally."""


def parse_public_http_url(url: str) -> tuple[str, str]:
    """Return ``(scheme, host)`` for a URL that is allowed to be fetched at all.

    Raises ``UnsafeUrlError`` for a missing or forbidden scheme, a missing host,
    or embedded credentials. Credentials are refused because a URL of the form
    ``https://user:password@host/`` leaks whatever the caller was tricked into
    embedding, and nothing this pipeline fetches ever needs them.
    """
    cleaned = (url or "").strip()
    if not cleaned:
        raise UnsafeUrlError("The URL is empty.")
    try:
        parsed = urlparse(cleaned)
    except Exception as parse_error:  # noqa: BLE001 - any parse failure is a refusal
        raise UnsafeUrlError(
            f"The URL could not be parsed: {parse_error}"
        ) from parse_error
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_URL_SCHEMES:
        raise UnsafeUrlError(
            f"Only {' and '.join(ALLOWED_URL_SCHEMES)} URLs may be fetched (got {scheme or 'no scheme'!r})."
        )
    if parsed.username or parsed.password:
        raise UnsafeUrlError("A URL carrying credentials will not be fetched.")
    host = (parsed.hostname or "").strip()
    if not host:
        raise UnsafeUrlError("The URL has no host.")
    return scheme, host


def address_is_internal(address: str) -> bool:
    """Whether one resolved IP address belongs to this deployment's network.

    ``is_global`` is deliberately not used on its own: it answers False for a
    handful of public-but-special ranges, and the categories named here are the
    ones that reach something inside the perimeter.
    """
    try:
        parsed_address = ipaddress.ip_address(address)
    except ValueError:
        # A name that does not parse as an address cannot be proven safe.
        return True
    return bool(
        parsed_address.is_private
        or parsed_address.is_loopback
        or parsed_address.is_link_local
        or parsed_address.is_reserved
        or parsed_address.is_multicast
        or parsed_address.is_unspecified
    )


def _resolve_host_addresses(host: str, port: int) -> list[str]:
    """Every address ``host`` resolves to, as strings. Blocking; call in a thread."""
    try:
        address_info = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as resolution_error:
        raise UnsafeUrlError(
            f"The host {host!r} could not be resolved: {resolution_error}"
        ) from resolution_error
    return [entry[4][0] for entry in address_info if entry[4]]


async def assert_public_host(url: str, *, allow_private: bool = False) -> None:
    """Raise ``UnsafeUrlError`` unless every address ``url``'s host resolves to is public.

    A literal IP address in the URL is checked directly; a name is resolved with
    ``socket.getaddrinfo`` on a worker thread so the event loop is never blocked.
    """
    scheme, host = parse_public_http_url(url)
    if allow_private:
        return
    port = 443 if scheme == "https" else 80
    # A bracketed IPv6 literal arrives from urlparse already unbracketed.
    try:
        ipaddress.ip_address(host)
    except ValueError:
        addresses = await asyncio.to_thread(_resolve_host_addresses, host, port)
    else:
        addresses = [host]
    if not addresses:
        raise UnsafeUrlError(f"The host {host!r} resolved to no addresses.")
    internal = [address for address in addresses if address_is_internal(address)]
    if internal:
        raise UnsafeUrlError(
            f"The host {host!r} resolves to an address inside this network "
            f"({internal[0]}); it will not be fetched."
        )


async def get_with_public_host_guard(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict | None = None,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    allow_private: bool = False,
) -> httpx.Response:
    """GET ``url`` with every redirect hop re-validated against the host guard.

    ``client`` must be built with ``follow_redirects=False``; this function does
    the following itself so each ``Location`` is checked before it is fetched.
    Returns the first non-redirect response, exactly as ``client.get`` would.
    """
    current_url = url
    for _ in range(max(1, max_redirects) + 1):
        await assert_public_host(current_url, allow_private=allow_private)
        response = await client.get(current_url, headers=headers)
        if not response.is_redirect:
            return response
        location = response.headers.get("location") or ""
        if not location:
            return response
        current_url = (
            str(response.next_request.url) if response.next_request else location
        )
    raise UnsafeUrlError(
        f"The URL {url!r} redirected more than {max_redirects} times; it will not be fetched."
    )


__all__ = [
    "ALLOWED_URL_SCHEMES",
    "UnsafeUrlError",
    "address_is_internal",
    "assert_public_host",
    "get_with_public_host_guard",
    "parse_public_http_url",
]
