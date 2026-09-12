"""Naming any website connects it, by whichever route that site actually offers.

The owner should be able to type an address and get a working connection without
knowing whether the site publishes a connector, an OAuth application, an API
key, or nothing at all. That is a ladder, and what is pinned down here is its
ORDER — because each rung is better than the one below it:

1. **A Model Context Protocol server**, when the site publishes one: nothing to
   register, nothing to paste, no credential typed into Neural Nexus.
2. **The owner's own login**, for everything else — they sign in once on the
   site's own page and the avatar uses that session afterwards. This comes
   before OAuth and before keys: it is the thing an owner can always do, and it
   gives the avatar exactly the access the owner has.
3. **An API key last**, and only where the owner's login is not an option —
   at Anthropic, OpenAI and LangSmith, whose terms forbid keeping a signed-in
   session and reading their pages (``terms_require_api_key``).

Two rungs matter most for correctness: claiming a site for the wrong provider
would hand the owner a connector for a service they did not name, and losing
the terms exception would have the avatar quietly do the forbidden thing on the
owner's own account.
"""

from types import SimpleNamespace

import pytest

from src.anubis.utils.connected_accounts import connect_handlers
from src.anubis.utils.connected_accounts.connect_handlers import (
    ConnectNeedsLogin,
    ConnectRequest,
    connect_site_by_discovery,
)
from src.anubis.utils.connected_accounts.providers import (
    get_provider,
    provider_for_host,
)

ASSISTANT_ID = "assistant-1"


def _request(site_url: str, **fields):
    return ConnectRequest(
        provider=get_provider("custom_site"),
        fields={"site_url": site_url, **fields},
        assistant_id=ASSISTANT_ID,
        context=SimpleNamespace(),
    )


def _no_mcp_server(monkeypatch):
    async def _discover(origin, context):
        return None

    monkeypatch.setattr(
        "src.anubis.utils.connected_accounts.mcp_discovery.discover_mcp_server",
        _discover,
    )


""" Rung 2: a site a registered provider already speaks for """


@pytest.mark.parametrize(
    "hostname, expected",
    [
        ("github.com", "github"),
        ("gist.github.com", "github"),
        ("platform.openai.com", "openai"),
        ("console.anthropic.com", "anthropic"),
        ("smith.langchain.com", "langsmith"),
        ("vercel.com", "vercel"),
    ],
)
def test_a_site_a_provider_speaks_for_is_matched(hostname, expected):
    matched = provider_for_host(hostname)
    assert matched is not None and matched.name == expected


def test_a_provider_named_by_a_subdomain_does_not_claim_its_neighbours():
    """LangSmith is smith.langchain.com; the Academy is not LangSmith.

    Sharing a domain is not sharing a service. Claiming the whole domain would
    offer the LangSmith connector to someone who named a course site.
    """
    assert provider_for_host("smith.langchain.com").name == "langsmith"
    assert provider_for_host("academy.langchain.com") is None


def test_google_hosts_are_never_matched_by_address():
    """Several Google providers share one login address, so none may claim it."""
    for hostname in ("mail.google.com", "calendar.google.com", "accounts.google.com"):
        assert provider_for_host(hostname) is None


def test_an_unknown_site_is_claimed_by_nobody():
    assert provider_for_host("console.x.ai") is None
    assert provider_for_host("academy.example") is None


@pytest.mark.asyncio
async def test_a_vendor_whose_terms_forbid_sessions_still_asks_for_a_key(monkeypatch):
    """Naming platform.openai.com asks for the API key, not a browser session.

    The owner's login comes first everywhere else. This is the narrow exception:
    OpenAI's terms forbid keeping a session and reading their pages, so asking
    for a key protects the owner's account instead of quietly breaking them.
    """
    _no_mcp_server(monkeypatch)
    handed_to = {}

    async def _connect_account(request):
        handed_to["provider"] = request.provider.name
        handed_to["site_url"] = request.text("site_url")
        return {"connected": True}

    monkeypatch.setattr(connect_handlers, "connect_account", _connect_account)

    result = await connect_site_by_discovery(_request("platform.openai.com"))

    assert result == {"connected": True}
    assert handed_to["provider"] == "openai"
    # What the owner typed travels with them; they do not retype it.
    assert handed_to["site_url"] == "platform.openai.com"


""" Rung 3: everything else signs in """


@pytest.mark.asyncio
async def test_an_ordinary_site_falls_through_to_signing_in(monkeypatch):
    """A site with no connector and no provider is reached by signing in.

    This is the case the whole feature exists for, and it used to be a refusal.
    """
    _no_mcp_server(monkeypatch)

    with pytest.raises(ConnectNeedsLogin) as raised:
        await connect_site_by_discovery(
            _request("academy.langchain.com", name="Course site")
        )

    assert raised.value.provider.name == "signed_in_site"
    login_request = raised.value.login_request
    assert login_request["provider"] == "signed_in_site"
    assert "academy.langchain.com" in login_request["site_url"]
    assert login_request["name"] == "Course site"


@pytest.mark.asyncio
async def test_a_site_with_no_name_is_named_after_itself(monkeypatch):
    _no_mcp_server(monkeypatch)

    with pytest.raises(ConnectNeedsLogin) as raised:
        await connect_site_by_discovery(_request("console.x.ai"))

    assert raised.value.login_request["name"] == "console.x.ai"


@pytest.mark.asyncio
async def test_a_site_that_offers_a_connector_never_asks_for_a_sign_in(monkeypatch):
    """Rung 1 still wins: a published server needs no credential at all."""
    found = SimpleNamespace(server_url="https://example.com/mcp", name="Example")

    async def _discover(origin, context):
        return found

    monkeypatch.setattr(
        "src.anubis.utils.connected_accounts.mcp_discovery.discover_mcp_server",
        _discover,
    )
    handed_to = {}

    async def _connect_mcp(request):
        handed_to["provider"] = request.provider.name
        handed_to["server_url"] = request.text("server_url")
        return {"connected": True}

    monkeypatch.setattr(connect_handlers, "connect_mcp_server_account", _connect_mcp)

    result = await connect_site_by_discovery(_request("example.com"))

    assert result == {"connected": True}
    assert handed_to["server_url"] == "https://example.com/mcp"


""" The provider row the sign-in rung needs """


def test_the_signed_in_site_provider_is_shaped_for_an_arbitrary_site():
    """It must carry a site_url and no login_url — the address is the owner's.

    ``validate_registry`` enforces that a browser-session provider declares one
    or the other; a fixed login_url would mean a fixed site, which is the
    opposite of what this row is for.
    """
    provider = get_provider("signed_in_site")
    assert provider is not None
    assert provider.credential_mechanism == "browser_session"
    assert provider.login_url is None
    assert [field.name for field in provider.connect_fields] == ["site_url", "name"]


""" The owner's own login comes before OAuth and before keys """


@pytest.mark.parametrize("provider_name", ["openai", "anthropic", "langsmith"])
def test_only_the_three_named_vendors_carry_the_terms_exception(provider_name):
    """The exception is per-vendor and stated, not buried in the ordering."""
    assert get_provider(provider_name).terms_require_api_key is True


def test_no_other_provider_forces_a_key():
    from src.anubis.utils.connected_accounts.providers import PROVIDER_REGISTRY

    forced = sorted(
        provider.name
        for provider in PROVIDER_REGISTRY.values()
        if provider.terms_require_api_key
    )
    assert forced == ["anthropic", "langsmith", "openai"]


@pytest.mark.asyncio
async def test_a_key_vendor_without_the_exception_falls_through_to_signing_in(
    monkeypatch,
):
    """An API key is the last resort, not the first offer.

    A vendor that issues keys but does not forbid sessions should still let the
    owner sign in as themselves — that is the whole preference. Simulated by
    lifting the exception from OpenAI for one call.
    """
    _no_mcp_server(monkeypatch)
    import dataclasses

    permissive = dataclasses.replace(
        get_provider("openai"), terms_require_api_key=False
    )
    monkeypatch.setattr(
        "src.anubis.utils.connected_accounts.providers.provider_for_host",
        lambda hostname: permissive,
    )

    with pytest.raises(ConnectNeedsLogin) as raised:
        await connect_site_by_discovery(_request("platform.openai.com"))

    assert raised.value.provider.name == "signed_in_site"


@pytest.mark.asyncio
async def test_an_oauth_vendor_still_goes_to_its_own_route(monkeypatch):
    """OAuth is not a key: GitHub's own sign-in IS the owner logging in."""
    _no_mcp_server(monkeypatch)
    handed_to = {}

    async def _connect_account(request):
        handed_to["provider"] = request.provider.name
        return {"connected": True}

    monkeypatch.setattr(connect_handlers, "connect_account", _connect_account)

    await connect_site_by_discovery(_request("github.com"))

    assert handed_to["provider"] == "github"
