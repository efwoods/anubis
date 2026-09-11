"""Working out how to reach a site from its address alone.

A site that runs a Model Context Protocol server is reachable with nothing
registered anywhere and no credential typed into Neural Nexus: the server says
how to sign in and registers this client itself. So discovery is the first
thing tried for an arbitrary site, and these assert the parts that decide
whether it is trustworthy — that a published document is never believed
without opening a session, that the order of the conventions is respected, and
that a site with nothing is an ordinary answer rather than a failure.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from src.anubis.utils.connected_accounts import mcp_discovery
from src.anubis.utils.connected_accounts.mcp_discovery import (
    DiscoveredMcpServer,
    discover_mcp_server,
    normalize_site,
)

CONTEXT = SimpleNamespace(mcp_connector_probe_timeout_seconds=1.0)


def _client(routes: dict[str, object], seen: list[str] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url).split("?", 1)[0]
        if seen is not None:
            seen.append(url)
        body = routes.get(url)
        if body is None:
            return httpx.Response(404)
        return httpx.Response(200, json=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _stub_probes(monkeypatch, *, open_at=(), auth_at=(), tools=("search", "create")):
    """Make every candidate answer: open, needing sign-in, or nothing."""
    asked: list[str] = []

    async def _probe_authorization(server_url, context, **kwargs):
        asked.append(server_url)
        if server_url in open_at:
            return {"status": "open"}
        if server_url in auth_at:
            return {"status": "needs_oauth"}
        return {"status": "unreachable"}

    async def _probe_server_tools(server_url, token, timeout):
        return [SimpleNamespace(name=name) for name in tools]

    import src.anubis.utils.connected_accounts.mcp_oauth as mcp_oauth
    import src.anubis.utils.connected_accounts.mcp_server_tools as mcp_server_tools

    monkeypatch.setattr(mcp_oauth, "probe_authorization", _probe_authorization)
    monkeypatch.setattr(mcp_server_tools, "probe_server_tools", _probe_server_tools)
    return asked


def _discover(site, routes, monkeypatch, seen=None, **stub):
    asked = _stub_probes(monkeypatch, **stub)

    async def run():
        async with _client(routes, seen) as client:
            return await discover_mcp_server(site, CONTEXT, http_client=client)

    return asyncio.run(run()), asked


# -- reading what a site publishes ----------------------------------------


@pytest.mark.parametrize(
    "typed,origin",
    [
        ("linear.app", "https://linear.app"),
        ("https://sentry.io/", "https://sentry.io"),
        ("https://example.com/team/x", "https://example.com"),
        (" HTTPS://Foo.COM ", "https://foo.com"),
        ("not a url at all", ""),
        ("", ""),
    ],
)
def test_however_the_owner_typed_the_site_it_means_the_same_place(typed, origin):
    assert normalize_site(typed)[0] == origin


def test_the_published_document_is_used_when_the_server_answers(monkeypatch):
    routes = {
        "https://acme.test/.well-known/mcp.json": {
            "servers": [
                {"url": "/mcp", "name": "Acme", "description": "Acme's tools"}
            ]
        }
    }

    found, _ = _discover(
        "acme.test", routes, monkeypatch, open_at=("https://acme.test/mcp",)
    )

    assert found is not None
    assert found.server_url == "https://acme.test/mcp"
    assert found.name == "Acme"
    assert found.description == "Acme's tools"
    assert found.source == "well_known:/.well-known/mcp.json"
    assert found.tool_names == ["search", "create"]


def test_a_document_pointing_at_a_dead_host_is_not_a_discovery(monkeypatch):
    """A stale file is worse than none: it would make a connector that answers nothing."""
    routes = {
        "https://acme.test/.well-known/mcp.json": {
            "servers": [{"url": "https://gone.test/mcp"}]
        }
    }

    # Nothing answers anywhere, including the address the document names.
    found, asked = _discover("acme.test", routes, monkeypatch)

    assert found is None
    assert "https://gone.test/mcp" in asked


def test_the_second_draft_path_is_tried_when_the_first_is_absent(monkeypatch):
    routes = {
        "https://acme.test/.well-known/mcp-server": {"url": "https://acme.test/mcp"}
    }

    found, _ = _discover(
        "acme.test", routes, monkeypatch, open_at=("https://acme.test/mcp",)
    )

    assert found is not None
    assert found.source == "well_known:/.well-known/mcp-server"


def test_a_document_in_any_of_the_shapes_seen_in_the_wild_is_read(monkeypatch):
    """Neither draft is final and they disagree, so all spellings are accepted."""
    routes = {
        "https://acme.test/.well-known/mcp.json": {
            "mcpServers": {"main": {"endpoint": "https://acme.test/mcp"}}
        }
    }

    found, _ = _discover(
        "acme.test", routes, monkeypatch, open_at=("https://acme.test/mcp",)
    )

    assert found is not None
    assert found.server_url == "https://acme.test/mcp"


# -- when a site publishes nothing ----------------------------------------


def test_the_conventional_addresses_are_tried_when_nothing_is_published(monkeypatch):
    """Most vendors running a server today publish no document at all."""
    found, asked = _discover(
        "acme.test", {}, monkeypatch, open_at=("https://acme.test/mcp",)
    )

    assert found is not None
    assert found.server_url == "https://acme.test/mcp"
    assert found.source == "conventional_address"
    # The subdomain is the likeliest home and is asked about first.
    assert asked[0] == "https://mcp.acme.test/mcp"


def test_a_site_with_no_server_is_an_ordinary_answer(monkeypatch):
    found, _ = _discover("acme.test", {}, monkeypatch)

    assert found is None


def test_a_site_that_is_not_an_address_is_not_probed(monkeypatch):
    found, asked = _discover("not a url at all", {}, monkeypatch)

    assert found is None
    assert asked == []


# -- servers that demand a sign-in ----------------------------------------


def test_a_server_demanding_authorization_still_counts_as_found(monkeypatch):
    """A challenge proves the server is real; listing tools is what it refuses."""
    found, _ = _discover(
        "acme.test", {}, monkeypatch, auth_at=("https://mcp.acme.test/mcp",)
    )

    assert found is not None
    assert found.needs_authorization is True
    assert found.tool_names == []
    assert found.server_url == "https://mcp.acme.test/mcp"


def test_a_known_server_is_preferred_over_guessing(monkeypatch):
    """Some addresses cannot be derived from the domain a person knows."""
    monkeypatch.setitem(
        mcp_discovery.KNOWN_SERVERS, "acme.test", "https://elsewhere.test/mcp/"
    )

    found, asked = _discover(
        "acme.test", {}, monkeypatch, open_at=("https://elsewhere.test/mcp/",)
    )

    assert found is not None
    assert found.source == "known_server"
    assert asked[0] == "https://elsewhere.test/mcp/"


def test_the_public_view_carries_what_a_card_needs():
    view = DiscoveredMcpServer(
        server_url="https://acme.test/mcp",
        site_url="https://acme.test",
        name="Acme",
        source="conventional_address",
        tool_names=["search"],
    ).as_public_dict()

    assert view["server_url"] == "https://acme.test/mcp"
    assert view["tool_names"] == ["search"]
    assert view["found_by"] == "conventional_address"


# -- connecting a site through what was discovered ------------------------


def test_connecting_a_site_hands_the_address_to_the_connector_path(monkeypatch):
    """Discovery adds no second copy of proving, signing in, or describing."""
    from src.anubis.utils.connected_accounts import connect_handlers, get_provider

    async def _discovered(site, context, **kwargs):
        return DiscoveredMcpServer(
            server_url="https://acme.test/mcp", site_url=site, name="Acme"
        )

    delegated: dict[str, object] = {}

    async def _connect_mcp(request):
        delegated["provider"] = request.provider.name
        delegated["fields"] = dict(request.fields)
        return {"account_key": "custom_mcp:acme.test", "provider": "custom_mcp"}

    monkeypatch.setattr(mcp_discovery, "discover_mcp_server", _discovered)
    monkeypatch.setattr(connect_handlers, "connect_mcp_server_account", _connect_mcp)

    record = asyncio.run(
        connect_handlers.connect_account(
            connect_handlers.ConnectRequest(
                provider=get_provider("custom_site"),
                fields={"site_url": "acme.test"},
                assistant_id="assistant-1",
                context=CONTEXT,
            )
        )
    )

    assert record["provider"] == "custom_mcp"
    # Stored as a connector, so its tools come from the existing factory.
    assert delegated["provider"] == "custom_mcp"
    assert delegated["fields"]["server_url"] == "https://acme.test/mcp"
    assert delegated["fields"]["name"] == "Acme"


def test_a_site_offering_nothing_falls_through_to_signing_in(monkeypatch):
    """No connector is not the end of the road — it is the next rung.

    This used to refuse and tell the owner to go find a connector address
    themselves. Naming a site should be enough: when the site publishes nothing,
    the owner signs in to it once and the avatar keeps that session.
    """
    from src.anubis.utils.connected_accounts import connect_handlers, get_provider

    async def _nothing(site, context, **kwargs):
        return None

    monkeypatch.setattr(mcp_discovery, "discover_mcp_server", _nothing)

    with pytest.raises(connect_handlers.ConnectNeedsLogin) as raised:
        asyncio.run(
            connect_handlers.connect_account(
                connect_handlers.ConnectRequest(
                    provider=get_provider("custom_site"),
                    fields={"site_url": "acme.test"},
                    assistant_id="assistant-1",
                    context=CONTEXT,
                )
            )
        )

    assert raised.value.provider.name == "signed_in_site"
    assert "acme.test" in raised.value.login_request["site_url"]
