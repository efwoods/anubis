"""Finding a mail provider's servers from an address, the way a mail client does.

Every test here is offline. The discovery ladder is a sequence of HTTP calls, so
a mock transport can answer each rung exactly as the real world would and the
order can be asserted directly — which is the part that matters, since a rung
answering out of turn means a provider's own settings would be overruled by a
community database entry.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from src.anubis.utils.connected_accounts import mail_autoconfig
from src.anubis.utils.connected_accounts.mail_autoconfig import (
    USERNAME_LOCAL_PART,
    discover_mail_settings,
    domain_of,
    parse_autoconfig_document,
    withdrawn_password_provider,
)


def _document(
    *,
    imap_host: str = "imap.example.com",
    smtp_host: str = "smtp.example.com",
    authentication: str = "password-cleartext",
    username: str = "%EMAILADDRESS%",
) -> str:
    return f"""<clientConfig version="1.1">
      <emailProvider id="example.com">
        <domain>example.com</domain>
        <displayName>Example Mail</displayName>
        <incomingServer type="imap">
          <hostname>{imap_host}</hostname>
          <port>993</port>
          <socketType>SSL</socketType>
          <username>{username}</username>
          <authentication>{authentication}</authentication>
        </incomingServer>
        <outgoingServer type="smtp">
          <hostname>{smtp_host}</hostname>
          <port>587</port>
          <socketType>STARTTLS</socketType>
          <username>{username}</username>
          <authentication>{authentication}</authentication>
        </outgoingServer>
      </emailProvider>
    </clientConfig>"""


def _client(routes: dict[str, httpx.Response], seen: list[str] | None = None):
    """An httpx client answering only the URLs named in ``routes``."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url).split("?", 1)[0]
        if seen is not None:
            seen.append(url)
        response = routes.get(url)
        if response is None:
            return httpx.Response(404)
        return httpx.Response(
            response.status_code, content=response.content, headers=response.headers
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _discover(address: str, routes, seen=None, monkeypatch=None):
    mail_autoconfig.clear_cache()
    if monkeypatch is not None:
        # The probe rung opens real sockets. A test of the HTTP rungs must not.
        async def _no_probe(domain):
            return None

        monkeypatch.setattr(mail_autoconfig, "_probe_conventional_hosts", _no_probe)

    async def run():
        async with _client(routes, seen) as client:
            return await discover_mail_settings(
                address, http_client=client, use_cache=False
            )

    return asyncio.run(run())


def test_the_providers_own_settings_win(monkeypatch):
    """A provider that publishes settings must not be overruled by a database."""
    seen: list[str] = []
    routes = {
        "https://autoconfig.example.com/mail/config-v1.1.xml": httpx.Response(
            200, content=_document(imap_host="imap.theirs.example.com")
        ),
        "https://autoconfig.thunderbird.net/v1.1/example.com": httpx.Response(
            200, content=_document(imap_host="imap.database.example.com")
        ),
    }

    settings = _discover("evan@example.com", routes, seen, monkeypatch)

    assert settings is not None
    assert settings.imap_host == "imap.theirs.example.com"
    assert settings.source == "provider_autoconfig"
    # The later rungs must not even be asked once an answer is in hand.
    assert "https://autoconfig.thunderbird.net/v1.1/example.com" not in seen


def test_the_well_known_path_answers_when_the_subdomain_does_not(monkeypatch):
    routes = {
        "https://example.com/.well-known/autoconfig/mail/config-v1.1.xml": httpx.Response(
            200, content=_document()
        )
    }

    settings = _discover("evan@example.com", routes, None, monkeypatch)

    assert settings is not None
    assert settings.imap_host == "imap.example.com"
    assert settings.source == "well_known_autoconfig"


def test_the_community_database_answers_when_the_domain_publishes_nothing(monkeypatch):
    routes = {
        "https://autoconfig.thunderbird.net/v1.1/example.com": httpx.Response(
            200, content=_document()
        )
    }

    settings = _discover("evan@example.com", routes, None, monkeypatch)

    assert settings is not None
    assert settings.smtp_host == "smtp.example.com"
    assert settings.source == "ispdb"


def test_a_custom_domain_is_found_through_its_mail_exchange(monkeypatch):
    """The rung that makes a company address work.

    Nothing is published for the company's own domain; the mail is hosted
    elsewhere, and the host behind the MX record is what the database knows.
    """
    routes = {
        "https://dns.google/resolve": httpx.Response(
            200,
            json={"Answer": [{"type": 15, "data": "10 mx1.mail.hoster.example."}]},
        ),
        "https://autoconfig.thunderbird.net/v1.1/hoster.example": httpx.Response(
            200, content=_document(imap_host="imap.hoster.example")
        ),
    }

    settings = _discover("evan@company.example", routes, None, monkeypatch)

    assert settings is not None
    assert settings.imap_host == "imap.hoster.example"
    assert settings.source == "ispdb_mail_exchange"


def test_a_domain_that_answers_nothing_returns_nothing(monkeypatch):
    """Not an error: the owner is asked for the server names instead."""
    assert _discover("evan@example.com", {}, None, monkeypatch) is None


def test_a_local_part_username_is_honoured(monkeypatch):
    routes = {
        "https://autoconfig.thunderbird.net/v1.1/example.com": httpx.Response(
            200, content=_document(username=USERNAME_LOCAL_PART)
        )
    }

    settings = _discover("evan@example.com", routes, None, monkeypatch)

    assert settings is not None
    assert settings.username_for("evan@example.com") == "evan"


def test_a_published_password_method_does_not_override_a_withdrawn_provider():
    """The database still advertises passwords for Gmail. Google does not accept them.

    Believing the record here would send every Gmail owner to type a password
    that is guaranteed to be rejected, so the resolved host decides.
    """
    settings = parse_autoconfig_document(
        _document(imap_host="imap.gmail.com", smtp_host="smtp.gmail.com"),
        source="ispdb",
    )

    assert settings is not None
    assert settings.password_authentication is False
    assert settings.password_withdrawn_by == "Google"


def test_an_oauth_only_record_reports_that_a_password_will_not_work():
    settings = parse_autoconfig_document(
        _document(imap_host="imap.elsewhere.example", authentication="OAuth2"),
        source="ispdb",
    )

    assert settings is not None
    assert settings.password_authentication is False
    assert settings.authentication_methods == ("oauth2",)


def test_an_ordinary_provider_still_accepts_a_password():
    settings = parse_autoconfig_document(_document(), source="ispdb")

    assert settings is not None
    assert settings.password_authentication is True
    assert settings.password_withdrawn_by == ""


@pytest.mark.parametrize(
    "host,expected",
    [
        ("imap.gmail.com", "Google"),
        ("aspmx.l.google.com", "Google"),
        ("outlook.office365.com", "Microsoft"),
        ("imap.mail.yahoo.com", "Yahoo"),
        ("imap.fastmail.com", ""),
        ("", ""),
    ],
)
def test_withdrawn_providers_are_recognised_by_host(host, expected):
    assert withdrawn_password_provider(host) == expected


@pytest.mark.parametrize(
    "address,expected",
    [
        ("Evan@Example.COM ", "example.com"),
        ("no-at-sign", ""),
        ("", ""),
    ],
)
def test_the_domain_is_taken_from_the_address(address, expected):
    assert domain_of(address) == expected


def test_a_malformed_document_is_not_an_answer():
    assert parse_autoconfig_document("<clientConfig", source="ispdb") is None
    assert parse_autoconfig_document("<clientConfig/>", source="ispdb") is None
