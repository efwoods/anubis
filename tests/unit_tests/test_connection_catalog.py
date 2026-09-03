"""Unit tests for the connection catalog, the generic connect dispatcher, and the
unified connections listing.

What these pin down:

- **One row per provider is the whole extension cost.** Every catalog row that
  is available and form-based can be connected through the one generic route by
  its declared mechanism; a coming-soon row is refused with its plain message and
  nothing is stored; a pairing row returns instructions rather than a form.
- **A custom server is proved before it is stored.** The connect handler lists
  the server's tools first; an unreachable server stores nothing, and a reachable
  one records the tool names so the catalog never has to dial the server again to
  say how many tools the connector adds.
- **Secrets stay out of every response.** The unified listing carries neither
  ciphertext nor the server URL the owner typed.
- **The toggle is connect/disconnect.** Off deletes the record and its
  credential; on, for a deleted account, hands back the connect card.
- **The table and the legacy store agree.** The facade reads the repository when
  one is published and the store otherwise, through one set of call sites.
"""

import asyncio
from types import SimpleNamespace

import pytest

from src.anubis.utils import secret_store
from src.anubis.utils.connected_accounts import (
    build_account_record,
    catalog_providers,
    get_provider,
)
from src.anubis.utils.connected_accounts import repository as repository_module
from src.anubis.utils.connected_accounts.connect_handlers import (
    ConnectRefused,
    ConnectRequest,
    connect_account,
)
from src.anubis.utils.connected_accounts.listing import (
    account_connection_view,
    device_connection_view,
    split_connection_key,
)
from src.anubis.utils.connected_accounts.providers import (
    AVAILABILITY_AVAILABLE,
    AVAILABILITY_COMING_SOON,
    CATEGORY_ORDER,
    MECHANISM_DEVICE_PAIRING,
)
from src.api import webapp as webapp_module

USER_ID = "auth0-user-abc"
ASSISTANT_ID = "assistant-personal"
SERVER_URL = "https://mcp.example.com/mcp"
TOKEN = "super-secret-bearer"


def _context(**overrides):
    values = dict(
        connected_account_encryption_key=secret_store.generate_encryption_key(),
        max_connected_accounts_per_user=10,
        max_custom_mcp_connectors_per_user=10,
        mailbox_request_timeout_seconds=5.0,
        mailbox_fetch_max_messages=25,
        mcp_connector_probe_timeout_seconds=2.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _current_user(user_id=USER_ID):
    return {"API_KEY": "sk-test-key", "identities": [{"user_id": user_id}]}


def _json_request(payload):
    async def _json():
        return payload

    return SimpleNamespace(json=_json)


class _StoreAPI:
    def __init__(self, items=None):
        self.items = dict(items or {})
        self.deleted = []
        self.put = []

    async def search_items(self, namespace, limit=100):
        return {
            "items": [{"key": key, "value": value} for key, value in self.items.items()]
        }

    async def put_item(self, namespace, key, value):
        self.put.append((tuple(namespace), key))
        self.items[key] = value

    async def get_item(self, namespace, key):
        value = self.items.get(key)
        return {"key": key, "value": value} if value is not None else None

    async def delete_item(self, namespace, key):
        self.deleted.append((tuple(namespace), key))
        self.items.pop(key, None)


@pytest.fixture(autouse=True)
def _no_published_repository():
    """Each test starts with no repository published and leaves none behind."""
    repository_module.set_repository(None)
    yield
    repository_module.set_repository(None)


def _install(monkeypatch, store_api, context=None):
    monkeypatch.setattr(
        webapp_module, "get_client", lambda **kwargs: SimpleNamespace(store=store_api)
    )
    monkeypatch.setattr(
        webapp_module.app, "state", SimpleNamespace(context=context or _context())
    )

    async def _resolve(client, request, user, api_key):
        return {"assistant_id": ASSISTANT_ID}

    monkeypatch.setattr(
        webapp_module, "_resolve_personal_avatar_for_connection", _resolve
    )


def _fake_probe(monkeypatch, tool_names=("search", "fetch"), fail=False):
    from src.anubis.utils.connected_accounts import mcp_server_tools

    async def _probe(server_url, bearer_token, timeout_seconds):
        if fail:
            raise mcp_server_tools.McpServerUnreachableError("no answer")
        _probe.calls.append((server_url, bearer_token))
        return [SimpleNamespace(name=name) for name in tool_names]

    _probe.calls = []
    monkeypatch.setattr(mcp_server_tools, "probe_server_tools", _probe)
    return _probe


# --------------------------------------------------------------------------
# The catalog
# --------------------------------------------------------------------------


def test_every_catalog_row_is_either_connectable_or_plainly_coming_soon():
    for provider in catalog_providers():
        assert provider.availability in (
            AVAILABILITY_AVAILABLE,
            AVAILABILITY_COMING_SOON,
        )
        if provider.is_available:
            assert provider.uses_form or provider.credential_mechanism == (
                MECHANISM_DEVICE_PAIRING
            ), f"{provider.name} is available but has no way to connect"
        assert provider.category in CATEGORY_ORDER
        assert provider.icon_key, f"{provider.name} needs an icon key"


def test_featured_rows_come_first_then_categories_in_order():
    providers = catalog_providers()
    featured_flags = [provider.featured for provider in providers]
    assert featured_flags == sorted(featured_flags, reverse=True)
    featured = [provider for provider in providers if provider.featured]
    category_indexes = [
        CATEGORY_ORDER.index(provider.category) for provider in featured
    ]
    assert category_indexes == sorted(category_indexes)


def test_the_welcome_page_sources_are_all_in_the_catalog():
    names = {provider.name for provider in catalog_providers()}
    for expected in (
        "gmail",
        "google_calendar",
        "twitter",
        "youtube",
        "twitch",
        "instagram",
        "facebook",
        "linkedin",
        "discord",
        "slack",
        "desktop_mcp",
        "custom_mcp",
    ):
        assert expected in names


def test_the_connect_card_carries_the_catalog_fields():
    from src.anubis.utils.connected_accounts.connection_tools import build_connect_card

    card = build_connect_card(get_provider("custom_mcp"))
    assert card["connect_endpoint"] == "/connect_account"
    assert card["uses_form"] is True
    assert card["availability"] == "available"
    assert [field["name"] for field in card["fields"]] == [
        "name",
        "server_url",
        "bearer_token",
    ]
    assert card["fields"][2]["required"] is False

    device_card = build_connect_card(get_provider("desktop_mcp"))
    assert device_card["uses_form"] is False
    assert device_card["pairing_instructions"]
    assert device_card["install_url"]


# --------------------------------------------------------------------------
# The dispatcher
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_coming_soon_provider_is_refused_with_its_message():
    with pytest.raises(ConnectRefused) as raised:
        await connect_account(
            ConnectRequest(
                provider=get_provider("google_calendar"),
                fields={},
                assistant_id=ASSISTANT_ID,
                context=_context(),
            )
        )
    assert raised.value.status_code == 501
    assert "coming soon" in raised.value.detail.lower()


@pytest.mark.asyncio
async def test_a_device_provider_returns_pairing_instructions_not_a_form():
    with pytest.raises(ConnectRefused) as raised:
        await connect_account(
            ConnectRequest(
                provider=get_provider("desktop_mcp"),
                fields={},
                assistant_id=ASSISTANT_ID,
                context=_context(),
            )
        )
    assert raised.value.status_code == 400
    assert "daemon" in raised.value.detail.lower()


@pytest.mark.asyncio
async def test_a_custom_server_is_proved_by_listing_its_tools(monkeypatch):
    probe = _fake_probe(monkeypatch, tool_names=("search", "fetch"))
    context = _context()

    record = await connect_account(
        ConnectRequest(
            provider=get_provider("custom_mcp"),
            fields={
                "name": "My Server",
                "server_url": SERVER_URL,
                "bearer_token": TOKEN,
            },
            assistant_id=ASSISTANT_ID,
            context=context,
        )
    )

    assert probe.calls == [(SERVER_URL, TOKEN)], (
        "the server must be dialed before storing"
    )
    assert record["kind"] == "mcp_server"
    assert record["transport"]["tool_names"] == ["fetch", "search"]
    assert record["transport"]["transport"] == "streamable_http"
    assert record["encrypted_secret"] != TOKEN
    assert secret_store.decrypt_secret(record["encrypted_secret"], context) == TOKEN
    assert record["display_label"] == "My Server"


@pytest.mark.asyncio
async def test_an_unreachable_custom_server_stores_nothing(monkeypatch):
    _fake_probe(monkeypatch, fail=True)
    with pytest.raises(ConnectRefused) as raised:
        await connect_account(
            ConnectRequest(
                provider=get_provider("custom_mcp"),
                fields={"name": "Dead", "server_url": SERVER_URL},
                assistant_id=ASSISTANT_ID,
                context=_context(),
            )
        )
    assert raised.value.status_code == 400


@pytest.mark.asyncio
async def test_a_server_url_without_a_scheme_is_refused_before_dialing(monkeypatch):
    probe = _fake_probe(monkeypatch)
    with pytest.raises(ConnectRefused):
        await connect_account(
            ConnectRequest(
                provider=get_provider("custom_mcp"),
                fields={"name": "x", "server_url": "mcp.example.com"},
                assistant_id=ASSISTANT_ID,
                context=_context(),
            )
        )
    assert probe.calls == []


def test_sse_urls_select_the_sse_transport():
    from src.anubis.utils.connected_accounts.mcp_server_tools import infer_transport

    assert infer_transport("https://mcp.example.com/sse") == "sse"
    assert infer_transport("https://mcp.example.com/mcp") == "streamable_http"


# --------------------------------------------------------------------------
# The generic route
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_account_route_connects_a_custom_server(monkeypatch):
    store_api = _StoreAPI()
    context = _context()
    _install(monkeypatch, store_api, context)
    _fake_probe(monkeypatch)

    response = await webapp_module.connect_account_route(
        request=_json_request(
            {
                "provider": "custom_mcp",
                "fields": {
                    "name": "Notes",
                    "server_url": SERVER_URL,
                    "bearer_token": TOKEN,
                },
            }
        ),
        current_user=_current_user(),
    )

    assert response.status_code == 200
    body = response.body.decode("utf-8")
    assert TOKEN not in body
    assert SERVER_URL not in body, "the URL may embed a credential; never echo it"
    stored = list(store_api.items.values())[0]
    assert stored["account_key"].startswith("custom_mcp:mcp.example.com#")
    assert stored["transport"]["server_url"] == SERVER_URL


@pytest.mark.asyncio
async def test_connect_account_route_accepts_flat_fields_for_gmail(monkeypatch):
    from src.anubis.utils.tools.email import imap_client

    store_api = _StoreAPI()
    _install(monkeypatch, store_api)
    monkeypatch.setattr(imap_client, "verify_credentials", lambda credentials: None)

    response = await webapp_module.connect_account_route(
        request=_json_request(
            {
                "provider": "gmail",
                "email_address": "evan@example.com",
                "app_password": "pw",
            }
        ),
        current_user=_current_user(),
    )
    assert response.status_code == 200
    assert "gmail:evan@example.com" in store_api.items


@pytest.mark.asyncio
async def test_connect_account_route_refuses_a_coming_soon_provider(monkeypatch):
    store_api = _StoreAPI()
    _install(monkeypatch, store_api)
    with pytest.raises(webapp_module.HTTPException) as raised:
        await webapp_module.connect_account_route(
            request=_json_request({"provider": "slack"}), current_user=_current_user()
        )
    assert raised.value.status_code == 501
    assert store_api.items == {}


@pytest.mark.asyncio
async def test_the_custom_connector_cap_is_separate_from_the_account_cap(monkeypatch):
    store_api = _StoreAPI()
    context = _context(max_custom_mcp_connectors_per_user=1)
    _install(monkeypatch, store_api, context)
    _fake_probe(monkeypatch)

    await webapp_module.connect_account_route(
        request=_json_request(
            {
                "provider": "custom_mcp",
                "fields": {"name": "a", "server_url": SERVER_URL},
            }
        ),
        current_user=_current_user(),
    )
    with pytest.raises(webapp_module.HTTPException) as raised:
        await webapp_module.connect_account_route(
            request=_json_request(
                {
                    "provider": "custom_mcp",
                    "fields": {
                        "name": "b",
                        "server_url": "https://other.example.com/mcp",
                    },
                }
            ),
            current_user=_current_user(),
        )
    assert raised.value.status_code == 409
    assert "custom connectors" in raised.value.detail


# --------------------------------------------------------------------------
# The unified listing and the toggle
# --------------------------------------------------------------------------


def _mailbox_record(context):
    return build_account_record(
        provider=get_provider("gmail"),
        account_address="evan@example.com",
        display_label="evan",
        encrypted_secret=secret_store.encrypt_secret("pw", context),
        assistant_id=ASSISTANT_ID,
    )


def _server_record(context):
    from src.anubis.utils.connected_accounts.connect_handlers import server_address_for

    return build_account_record(
        provider=get_provider("custom_mcp"),
        account_address=server_address_for(SERVER_URL),
        display_label="Notes",
        encrypted_secret=secret_store.encrypt_secret(TOKEN, context),
        assistant_id=ASSISTANT_ID,
        transport={
            "server_url": SERVER_URL,
            "transport": "streamable_http",
            "tool_names": ["fetch", "search"],
        },
    )


def test_account_views_never_carry_the_ciphertext_or_the_server_url():
    context = _context()
    view = account_connection_view(_server_record(context))
    flattened = repr(view)
    assert TOKEN not in flattened
    assert SERVER_URL not in flattened
    assert view["connection_key"].startswith("account:custom_mcp:")
    assert view["tool_count"] == 2
    assert view["icon_key"] == "custom"
    assert view["disconnect_endpoint"] == "/disconnect_account"

    mailbox_view = account_connection_view(_mailbox_record(context))
    assert mailbox_view["tool_count"] > 0
    assert mailbox_view["category"] == "mail"


def test_device_views_share_the_row_shape():
    view = device_connection_view(
        {
            "device_id": "dev-1",
            "device_label": "ubuntu-desktop",
            "platform": "ubuntu",
            "online": True,
            "connected": True,
            "bound_assistant_id": ASSISTANT_ID,
        }
    )
    assert view["connection_key"] == "device:dev-1"
    assert view["icon_key"] == "ubuntu"
    assert view["connected"] is True
    assert view["disconnect_endpoint"] == "/disconnect_mcp"
    assert set(view) >= set(account_connection_view(_mailbox_record(_context())))


def test_connection_keys_must_be_prefixed():
    assert split_connection_key("account:gmail:a@b.c") == ("account", "gmail:a@b.c")
    assert split_connection_key("device:abc") == ("device", "abc")
    with pytest.raises(ValueError):
        split_connection_key("gmail:a@b.c")


@pytest.mark.asyncio
async def test_list_connections_merges_accounts_and_devices(monkeypatch):
    context = _context()
    record = _mailbox_record(context)
    store_api = _StoreAPI({record["account_key"]: record})
    _install(monkeypatch, store_api, context)

    async def _devices(client, user_id):
        return [
            {
                "device_id": "dev-1",
                "device_label": "mac",
                "platform": "macos",
                "online": False,
                "connected": True,
                "bound_assistant_id": ASSISTANT_ID,
            }
        ]

    monkeypatch.setattr(webapp_module, "_device_rows_for_user", _devices)

    response = await webapp_module.list_connections(
        request=SimpleNamespace(), current_user=_current_user()
    )
    body = response.body.decode("utf-8")
    assert "account:gmail:evan@example.com" in body
    assert "device:dev-1" in body
    assert record["encrypted_secret"] not in body


@pytest.mark.asyncio
async def test_toggling_an_account_off_deletes_it_and_on_returns_the_card(monkeypatch):
    context = _context()
    record = _mailbox_record(context)
    store_api = _StoreAPI({record["account_key"]: record})
    _install(monkeypatch, store_api, context)

    off = await webapp_module.set_connection_state(
        request=_json_request(
            {"connection_key": f"account:{record['account_key']}", "connected": False}
        ),
        current_user=_current_user(),
    )
    assert off.status_code == 200
    assert store_api.items == {}, "off means the credential is gone"

    on = await webapp_module.set_connection_state(
        request=_json_request(
            {"connection_key": f"account:{record['account_key']}", "connected": True}
        ),
        current_user=_current_user(),
    )
    body = on.body.decode("utf-8")
    assert '"action":"open_connect_card"' in body.replace(" ", "")
    assert '"provider":"gmail"' in body.replace(" ", "")


@pytest.mark.asyncio
async def test_toggling_a_device_off_writes_the_suppression_marker(monkeypatch):
    store_api = _StoreAPI()
    _install(monkeypatch, store_api)

    response = await webapp_module.set_connection_state(
        request=_json_request({"connection_key": "device:dev-9", "connected": False}),
        current_user=_current_user(),
    )
    assert response.status_code == 200
    namespaces = {namespace for namespace, _key in store_api.put}
    assert any("mcp_connection_declined" in namespace for namespace in namespaces)


@pytest.mark.asyncio
async def test_a_bare_connection_key_is_rejected(monkeypatch):
    store_api = _StoreAPI()
    _install(monkeypatch, store_api)
    with pytest.raises(webapp_module.HTTPException) as raised:
        await webapp_module.set_connection_state(
            request=_json_request(
                {"connection_key": "gmail:a@b.c", "connected": False}
            ),
            current_user=_current_user(),
        )
    assert raised.value.status_code == 400


# --------------------------------------------------------------------------
# The repository behind the facade
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_facade_prefers_a_published_repository_over_the_store():
    from src.anubis.utils.connected_accounts import (
        bound_accounts_for,
        clear_connected_account,
        read_connected_accounts,
        save_connected_account,
    )

    class _ExplodingStore:
        async def asearch(self, *args, **kwargs):
            raise AssertionError("the store must not be consulted")

        async def aput(self, *args, **kwargs):
            raise AssertionError("the store must not be consulted")

    repository = repository_module.InMemoryConnectedAccountRepository()
    repository_module.set_repository(repository)
    context = _context()
    record = _mailbox_record(context)

    await save_connected_account(_ExplodingStore(), USER_ID, record)
    assert await read_connected_accounts(_ExplodingStore(), USER_ID) == [record]
    assert await bound_accounts_for(_ExplodingStore(), USER_ID, ASSISTANT_ID) == [
        record
    ]
    assert await bound_accounts_for(_ExplodingStore(), USER_ID, "other") == []
    assert await clear_connected_account(
        _ExplodingStore(), USER_ID, record["account_key"]
    )
    assert await read_connected_accounts(_ExplodingStore(), USER_ID) == []


@pytest.mark.asyncio
async def test_the_routes_use_the_published_repository(monkeypatch):
    repository = repository_module.InMemoryConnectedAccountRepository()
    repository_module.set_repository(repository)
    store_api = _StoreAPI()
    context = _context()
    _install(monkeypatch, store_api, context)
    _fake_probe(monkeypatch)

    await webapp_module.connect_account_route(
        request=_json_request(
            {
                "provider": "custom_mcp",
                "fields": {"name": "Notes", "server_url": SERVER_URL},
            }
        ),
        current_user=_current_user(),
    )
    assert store_api.items == {}, "the store is bypassed once a repository exists"
    assert len(repository.records[USER_ID]) == 1

    listing = await webapp_module.list_connected_accounts(
        request=SimpleNamespace(), current_user=_current_user()
    )
    assert "custom_mcp" in listing.body.decode("utf-8")


# --------------------------------------------------------------------------
# Tools per account kind
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tools_are_built_per_kind_and_prefixed_for_custom_servers(monkeypatch):
    from src.anubis.utils.connected_accounts import mcp_server_tools
    from src.anubis.utils.connected_accounts.tool_factories import (
        build_tools_for_accounts,
    )

    context = _context()
    _fake_probe(monkeypatch, tool_names=("search",))
    mcp_server_tools._tools_cache.clear()
    mcp_server_tools._last_failure_monotonic.clear()

    tools = await build_tools_for_accounts(
        context, [_mailbox_record(context), _server_record(context)]
    )
    names = {tool.name for tool in tools}
    assert "search_mailbox_messages" in names
    assert "notes__search" in names, "custom server tools carry the connector prefix"


@pytest.mark.asyncio
async def test_an_unreachable_custom_server_contributes_no_tools(monkeypatch):
    from src.anubis.utils.connected_accounts import mcp_server_tools
    from src.anubis.utils.connected_accounts.tool_factories import (
        build_tools_for_accounts,
    )

    context = _context()
    _fake_probe(monkeypatch, fail=True)
    mcp_server_tools._tools_cache.clear()
    mcp_server_tools._last_failure_monotonic.clear()

    tools = await build_tools_for_accounts(context, [_server_record(context)])
    assert tools == []


# --------------------------------------------------------------------------
# Sent mail as identity: the import route reuses the upload pipeline
# --------------------------------------------------------------------------


def test_quoted_replies_and_signatures_are_stripped_from_writing_samples():
    text = webapp_module._writing_sample_text_from_sent_messages(
        [
            {
                "subject": "Re: lunch",
                "recipients": ["a@b.c"],
                "sent_at": "2026-09-01",
                "body_text": "Sounds good, see you at noon.\n\nOn Mon, Alice wrote:\n> where?\n> when?",
            },
            {"subject": "empty", "recipients": [], "body_text": "> only a quote"},
        ]
    )
    assert "Sounds good, see you at noon." in text
    assert "where?" not in text and "Alice wrote" not in text
    assert "only a quote" not in text
    assert text.count("Subject:") == 1


@pytest.mark.asyncio
async def test_importing_sent_mail_schedules_one_text_media_job(monkeypatch):
    from src.anubis.utils.tools.email import imap_client

    context = _context(media_processing_concurrency=1)
    record = _mailbox_record(context)
    store_api = _StoreAPI({record["account_key"]: record})
    _install(monkeypatch, store_api, context)
    webapp_module.app.state.media_jobs = {}
    webapp_module.app.state.store = object()
    monkeypatch.setattr(webapp_module, "enforce_tier_capability", lambda *a, **k: None)

    searched = {}

    def _search(credentials, query, limit, mailbox):
        searched["mailbox"] = mailbox
        searched["password"] = credentials.password
        return [
            {
                "subject": "hi",
                "recipients": ["x@y.z"],
                "sent_at": "d",
                "body_text": "Hello there.",
            }
        ]

    monkeypatch.setattr(imap_client, "search_messages", _search)

    scheduled = {}

    async def _run_batch(master, items, config, store, ctx, **kwargs):
        scheduled["items"] = items
        scheduled["config"] = config

    monkeypatch.setattr(webapp_module, "run_batch_media_job", _run_batch)

    response = await webapp_module.import_mailbox_writing_samples(
        request=_json_request({}), current_user=_current_user()
    )
    await asyncio.sleep(0)

    assert response.status_code == 202
    assert searched["mailbox"] == "[Gmail]/Sent Mail"
    assert searched["password"] == "pw", (
        "the stored credential is decrypted for the fetch"
    )
    assert len(scheduled["items"]) == 1
    entry = scheduled["items"][0]["media_file"]
    assert entry["content_type"] == "text/plain"
    assert b"Hello there." in entry["content"]
    assert entry["filename"].startswith("sent_mail_evan_")
    assert scheduled["config"]["configurable"]["assistant_id"] == ASSISTANT_ID
