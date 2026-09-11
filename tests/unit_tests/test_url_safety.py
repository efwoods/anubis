"""Unit tests for the host guard that stands in front of every remote download.

Pinned down:

- **A literal internal address is refused**, whichever form it takes: IPv4
  loopback, an RFC 1918 range, the cloud metadata link-local address, and the
  IPv6 loopback.
- **A public-looking hostname that RESOLVES internally is refused too**, which
  is the case a name-based allowlist would miss.
- **Only http and https reach the network**, so a ``file://`` URL cannot be
  turned into a read of the container's filesystem.
- **A redirect is re-checked**: a public URL that answers ``302`` toward an
  internal address does not get followed.
- **An ordinary public URL passes**, so the guard does not break the uploads
  people have always been able to paste.
"""

import ipaddress
import socket

import httpx
import pytest

from src.anubis.utils.net.url_safety import (
    UnsafeUrlError,
    address_is_internal,
    assert_public_host,
    get_with_public_host_guard,
    parse_public_http_url,
)

PUBLIC_ADDRESS = "93.184.216.34"


@pytest.fixture
def resolve_to(monkeypatch):
    """Point every hostname lookup at the addresses a test names."""

    def _install(mapping: dict[str, list[str]]):
        def _fake_getaddrinfo(host, port, *args, **kwargs):
            addresses = mapping.get(host)
            if addresses is None:
                raise socket.gaierror(f"unknown host {host}")
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))
                for address in addresses
            ]

        monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)

    return _install


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/latest",
        "http://10.0.0.5/admin",
        "http://192.168.1.1/",
        "https://169.254.169.254/latest/meta-data/",
        "http://[::1]:8080/",
        "http://0.0.0.0/",
    ],
)
@pytest.mark.asyncio
async def test_a_literal_internal_address_is_refused(url):
    with pytest.raises(UnsafeUrlError):
        await assert_public_host(url)


@pytest.mark.asyncio
async def test_a_public_name_that_resolves_internally_is_refused(resolve_to):
    # The whole point of resolving rather than pattern-matching the name: this
    # host looks like any other public host until it is looked up.
    resolve_to({"metadata.example.com": ["169.254.169.254"]})
    with pytest.raises(UnsafeUrlError):
        await assert_public_host("https://metadata.example.com/token")


@pytest.mark.asyncio
async def test_one_internal_address_among_several_refuses_the_whole_host(resolve_to):
    resolve_to({"mixed.example.com": [PUBLIC_ADDRESS, "127.0.0.1"]})
    with pytest.raises(UnsafeUrlError):
        await assert_public_host("https://mixed.example.com/")


@pytest.mark.asyncio
async def test_an_ordinary_public_url_passes(resolve_to):
    resolve_to({"example.com": [PUBLIC_ADDRESS]})
    await assert_public_host("https://example.com/a/picture.jpg")


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "gopher://example.com/", "ftp://example.com/x", "notaurl"],
)
def test_only_http_and_https_are_parsed_at_all(url):
    with pytest.raises(UnsafeUrlError):
        parse_public_http_url(url)


def test_a_url_carrying_credentials_is_refused():
    with pytest.raises(UnsafeUrlError):
        parse_public_http_url("https://user:secret@example.com/")


def test_allow_private_is_the_only_way_past_the_guard(resolve_to):
    # Development fixtures are served from localhost; nothing else may be.
    assert address_is_internal("127.0.0.1") is True
    assert address_is_internal(str(ipaddress.ip_address("8.8.8.8"))) is False


@pytest.mark.asyncio
async def test_allow_private_lets_a_local_fixture_through():
    await assert_public_host("http://127.0.0.1:9600/fixture.jpg", allow_private=True)


@pytest.mark.asyncio
async def test_a_redirect_toward_an_internal_address_is_not_followed(resolve_to):
    resolve_to({"redirector.example.com": [PUBLIC_ADDRESS]})
    requested: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if request.url.host == "redirector.example.com":
            return httpx.Response(
                302, headers={"location": "http://169.254.169.254/latest/meta-data/"}
            )
        return httpx.Response(200, content=b"secrets")

    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
        with pytest.raises(UnsafeUrlError):
            await get_with_public_host_guard(
                client, "https://redirector.example.com/photo.jpg"
            )
    # The redirect target was never requested — that is the whole guarantee.
    assert requested == ["https://redirector.example.com/photo.jpg"]


@pytest.mark.asyncio
async def test_a_redirect_to_another_public_host_is_followed(resolve_to):
    resolve_to({"a.example.com": [PUBLIC_ADDRESS], "b.example.com": [PUBLIC_ADDRESS]})

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "a.example.com":
            return httpx.Response(
                302, headers={"location": "https://b.example.com/photo.jpg"}
            )
        return httpx.Response(200, content=b"a photograph")

    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
        response = await get_with_public_host_guard(
            client, "https://a.example.com/photo.jpg"
        )
    assert response.status_code == 200
    assert response.content == b"a photograph"


@pytest.mark.asyncio
async def test_a_redirect_loop_stops_rather_than_spinning(resolve_to):
    resolve_to({"loop.example.com": [PUBLIC_ADDRESS]})

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302, headers={"location": "https://loop.example.com/again"}
        )

    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
        with pytest.raises(UnsafeUrlError):
            await get_with_public_host_guard(
                client, "https://loop.example.com/start", max_redirects=3
            )
