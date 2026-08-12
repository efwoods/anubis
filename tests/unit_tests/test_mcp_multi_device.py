"""Unit tests for multi-device Model Context Protocol identity and endpoints.

A user runs the Neural Nexus daemon on several machines at once (Ubuntu desktop,
macOS, mobile, Windows). Two things make that workable and are covered here:

- **Device identity.** Every machine needs a human-readable name, because an
  avatar cannot report results from an opaque device token. Explicit daemon
  fields win, and derivation from the announced ``server_name`` is the
  compatibility path for daemons that predate those fields — derivation must
  keep working, because an already-installed daemon keeps sending the old
  payload until the user updates the daemon.
- **Device-scoped endpoints.** Registration, heartbeat, and unregistration act
  on one machine's record. Unregistration in particular deletes ONLY the calling
  device's record: while all machines shared one record under a constant key,
  stopping a development daemon deleted production's registration.
"""

import asyncio
from types import SimpleNamespace

import pytest

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.tools.data_analysis.devices import (
    UNKNOWN_DEVICE_LABEL,
    UNKNOWN_PLATFORM,
    connection_label_map,
    deduplicate_label,
    derive_device_identity,
)

# --------------------------------------------------------------------------
# Device identity derivation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("server_name", "expected_label", "expected_platform"),
    [
        ("Ubuntu-OS-Filesystem", "Ubuntu", "ubuntu"),
        ("macOS-Filesystem", "macOS", "macos"),
        # The Windows daemon is templated from the macOS daemon and has not
        # fixed its exact server name yet, so matching is by KEYWORD: any of
        # these must resolve rather than falling back to "Unknown device".
        ("Windows-OS-Filesystem", "Windows", "windows"),
        ("Windows-Filesystem", "Windows", "windows"),
        ("win32-filesystem", "Windows", "windows"),
        ("iPhone-Filesystem", "iPhone", "ios"),
        ("Android-Filesystem", "Android", "android"),
        ("Debian-OS-Filesystem", "Debian", "linux"),
        ("some-linux-box", "Linux", "linux"),
    ],
)
def test_labels_are_derived_from_the_announced_server_name(
    server_name, expected_label, expected_platform
):
    label, platform = derive_device_identity({"server_name": server_name})
    assert label == expected_label
    assert platform == expected_platform


def test_unrecognized_server_name_still_yields_a_usable_identity():
    label, platform = derive_device_identity({"server_name": "something-else"})
    assert label == UNKNOWN_DEVICE_LABEL
    assert platform == UNKNOWN_PLATFORM


def test_explicit_daemon_fields_win_over_derivation():
    label, platform = derive_device_identity(
        {
            "server_name": "Ubuntu-OS-Filesystem",
            "device_label": "Work laptop",
            "platform": "ubuntu",
        }
    )
    assert label == "Work laptop"
    assert platform == "ubuntu"


def test_platform_is_resolved_independently_of_the_label():
    """The mobile daemon sends ``platform`` but no label; both must survive.

    Taking both fields from one source would discard the field the daemon does
    supply, so the label is derived while the explicit platform is kept.
    """
    label, platform = derive_device_identity(
        {"server_name": "AnubisMCP-Filesystem", "platform": "ios"}
    )
    assert platform == "ios"
    assert label == UNKNOWN_DEVICE_LABEL


def test_duplicate_labels_are_counted_so_two_ubuntu_boxes_stay_distinct():
    existing = [{"device_id": "d1", "device_label": "Ubuntu"}]
    assert deduplicate_label("Ubuntu", existing, "d2") == "Ubuntu 2"

    existing.append({"device_id": "d2", "device_label": "Ubuntu 2"})
    assert deduplicate_label("Ubuntu", existing, "d3") == "Ubuntu 3"


def test_a_device_keeps_its_own_label_when_re_registering():
    """Every daemon restart re-registers; the label must not climb each time."""
    existing = [{"device_id": "d1", "device_label": "Ubuntu"}]
    assert deduplicate_label("Ubuntu", existing, "d1") == "Ubuntu"


def test_label_lookup_tolerates_the_casing_a_human_typed():
    connections = [
        SimpleNamespace(device_label="My MacBook"),
        SimpleNamespace(device_label="Ubuntu"),
    ]
    label_map = connection_label_map(connections)
    assert label_map["my macbook"].device_label == "My MacBook"
    assert label_map["ubuntu"].device_label == "Ubuntu"


# --------------------------------------------------------------------------
# Device-scoped endpoints
# --------------------------------------------------------------------------


class _FakeStoreClient:
    """In-memory stand-in for the SDK ``StoreClient`` used by the endpoints."""

    def __init__(self):
        # {namespace tuple: {key: value}}
        self.items: dict[tuple, dict] = {}

    async def put_item(self, namespace, key, value):
        self.items.setdefault(tuple(namespace), {})[key] = value

    async def get_item(self, namespace, key):
        namespace_items = self.items.get(tuple(namespace), {})
        if key not in namespace_items:
            raise RuntimeError("item not found")
        return {"value": namespace_items[key]}

    async def delete_item(self, namespace, key):
        self.items.get(tuple(namespace), {}).pop(key, None)

    async def search_items(self, namespace, limit=10, **kwargs):
        namespace_items = self.items.get(tuple(namespace), {})
        return {
            "items": [
                {"key": key, "value": value}
                for key, value in list(namespace_items.items())[:limit]
            ]
        }


class _FakeRequest:
    """Only ``.json()`` and ``.base_url`` are read by these endpoints."""

    def __init__(self, body):
        self._body = body
        self.base_url = "http://testserver/"

    async def json(self):
        return self._body


def _user():
    return {
        "identities": [{"user_id": "auth0|u1"}],
        "API_KEY": "sk-test-key",
    }


@pytest.fixture
def webapp_with_fake_store(monkeypatch):
    from src.api import webapp as webapp_module

    store_client = _FakeStoreClient()
    client = SimpleNamespace(store=store_client)
    monkeypatch.setattr(webapp_module, "get_client", lambda **kwargs: client)
    # ``app.state.context`` is populated by the lifespan handler, which does not
    # run in a unit test.
    webapp_module.app.state.context = GlobalContext()
    return webapp_module, store_client


def _registration_namespace():
    from src.anubis.utils.tools.data_analysis.backend import (
        mcp_registration_namespace,
    )

    return mcp_registration_namespace("auth0|u1")


def test_register_keys_the_record_by_device_and_derives_a_label(
    webapp_with_fake_store,
):
    webapp_module, store_client = webapp_with_fake_store

    async def run():
        response = await webapp_module.mcp_register(
            request=_FakeRequest(
                {
                    "connection_mode": "relay",
                    "device_id": "d-ubuntu",
                    "device_secret": "mcp_dev_secret",
                    "server_name": "Ubuntu-OS-Filesystem",
                    "allowed_roots": ["/data"],
                }
            ),
            current_user=_user(),
        )
        assert response.status_code == 200

        records = store_client.items[tuple(_registration_namespace())]
        assert set(records) == {"d-ubuntu"}
        assert records["d-ubuntu"]["device_label"] == "Ubuntu"
        assert records["d-ubuntu"]["platform"] == "ubuntu"
        # Relay URLs are rewritten to the API instance that accepted the call.
        assert records["d-ubuntu"]["mcp_url"] == (
            "http://testserver/mcp/relay/d-ubuntu"
        )

    asyncio.run(run())


def test_registering_a_second_machine_keeps_the_first(webapp_with_fake_store):
    webapp_module, store_client = webapp_with_fake_store

    async def run():
        for device_id, server_name in (
            ("d-ubuntu", "Ubuntu-OS-Filesystem"),
            ("d-macos", "macOS-Filesystem"),
        ):
            await webapp_module.mcp_register(
                request=_FakeRequest(
                    {
                        "connection_mode": "relay",
                        "device_id": device_id,
                        "device_secret": "mcp_dev_secret",
                        "server_name": server_name,
                    }
                ),
                current_user=_user(),
            )

        records = store_client.items[tuple(_registration_namespace())]
        assert set(records) == {"d-ubuntu", "d-macos"}
        assert records["d-macos"]["device_label"] == "macOS"

    asyncio.run(run())


def test_two_machines_of_the_same_platform_get_distinct_labels(
    webapp_with_fake_store,
):
    webapp_module, store_client = webapp_with_fake_store

    async def run():
        for device_id in ("d-ubuntu-1", "d-ubuntu-2"):
            await webapp_module.mcp_register(
                request=_FakeRequest(
                    {
                        "connection_mode": "relay",
                        "device_id": device_id,
                        "device_secret": "mcp_dev_secret",
                        "server_name": "Ubuntu-OS-Filesystem",
                    }
                ),
                current_user=_user(),
            )

        records = store_client.items[tuple(_registration_namespace())]
        assert {record["device_label"] for record in records.values()} == {
            "Ubuntu",
            "Ubuntu 2",
        }

    asyncio.run(run())


def test_register_without_a_device_identifier_is_rejected(webapp_with_fake_store):
    webapp_module, _store_client = webapp_with_fake_store
    from fastapi import HTTPException

    async def run():
        with pytest.raises(HTTPException) as raised:
            await webapp_module.mcp_register(
                request=_FakeRequest({"connection_mode": "relay"}),
                current_user=_user(),
            )
        assert raised.value.status_code == 400

    asyncio.run(run())


def test_register_enforces_the_device_cap_but_allows_re_registration(
    webapp_with_fake_store,
):
    webapp_module, store_client = webapp_with_fake_store
    from fastapi import HTTPException

    async def run():
        webapp_module.app.state.context.data_analysis_max_devices_per_user = 2
        for device_id in ("d1", "d2"):
            await webapp_module.mcp_register(
                request=_FakeRequest(
                    {"device_id": device_id, "server_name": "Ubuntu-OS-Filesystem"}
                ),
                current_user=_user(),
            )

        with pytest.raises(HTTPException) as raised:
            await webapp_module.mcp_register(
                request=_FakeRequest(
                    {"device_id": "d3", "server_name": "Ubuntu-OS-Filesystem"}
                ),
                current_user=_user(),
            )
        assert raised.value.status_code == 409

        # An already-registered machine may always re-register — every daemon
        # restart does, and the cap counts only OTHER devices.
        response = await webapp_module.mcp_register(
            request=_FakeRequest(
                {"device_id": "d2", "server_name": "Ubuntu-OS-Filesystem"}
            ),
            current_user=_user(),
        )
        assert response.status_code == 200
        assert set(store_client.items[tuple(_registration_namespace())]) == {
            "d1",
            "d2",
        }

    asyncio.run(run())


def test_unregister_removes_only_the_calling_device(webapp_with_fake_store):
    """The production incident: a dev daemon's shutdown deleted prod's record."""
    webapp_module, store_client = webapp_with_fake_store

    async def run():
        for device_id, server_name in (
            ("d-dev", "Ubuntu-OS-Filesystem"),
            ("d-prod", "Ubuntu-OS-Filesystem"),
        ):
            await webapp_module.mcp_register(
                request=_FakeRequest(
                    {"device_id": device_id, "server_name": server_name}
                ),
                current_user=_user(),
            )

        response = await webapp_module.mcp_unregister(
            request=_FakeRequest({"device_id": "d-dev"}), current_user=_user()
        )
        assert response.status_code == 200

        records = store_client.items[tuple(_registration_namespace())]
        assert set(records) == {"d-prod"}

    asyncio.run(run())


def test_unregister_without_a_device_identifier_is_rejected(webapp_with_fake_store):
    """Treating a missing device id as "all" would reproduce the incident."""
    webapp_module, store_client = webapp_with_fake_store
    from fastapi import HTTPException

    async def run():
        await webapp_module.mcp_register(
            request=_FakeRequest(
                {"device_id": "d-prod", "server_name": "Ubuntu-OS-Filesystem"}
            ),
            current_user=_user(),
        )

        with pytest.raises(HTTPException) as raised:
            await webapp_module.mcp_unregister(
                request=_FakeRequest({}), current_user=_user()
            )
        assert raised.value.status_code == 400
        # Nothing was deleted.
        assert set(store_client.items[tuple(_registration_namespace())]) == {"d-prod"}

    asyncio.run(run())


def test_heartbeat_refreshes_only_its_own_device(webapp_with_fake_store):
    """One machine's heartbeat must not keep another machine looking online."""
    webapp_module, store_client = webapp_with_fake_store

    async def run():
        for device_id in ("d-ubuntu", "d-macos"):
            await webapp_module.mcp_register(
                request=_FakeRequest(
                    {"device_id": device_id, "server_name": "Ubuntu-OS-Filesystem"}
                ),
                current_user=_user(),
            )
        records = store_client.items[tuple(_registration_namespace())]
        stale_last_seen = "2020-01-01T00:00:00+00:00"
        for record in records.values():
            record["last_seen_at"] = stale_last_seen

        response = await webapp_module.mcp_heartbeat(
            request=_FakeRequest({"device_id": "d-ubuntu"}), current_user=_user()
        )
        assert response.status_code == 200

        assert records["d-ubuntu"]["last_seen_at"] != stale_last_seen
        assert records["d-macos"]["last_seen_at"] == stale_last_seen

    asyncio.run(run())


def test_heartbeat_without_a_device_identifier_is_a_no_op(webapp_with_fake_store):
    webapp_module, _store_client = webapp_with_fake_store

    async def run():
        response = await webapp_module.mcp_heartbeat(
            request=_FakeRequest({}), current_user=_user()
        )
        assert response.status_code == 200

    asyncio.run(run())


def test_list_mcp_connections_merges_registration_and_connection_state(
    webapp_with_fake_store, monkeypatch
):
    webapp_module, store_client = webapp_with_fake_store
    import json

    from src.anubis.utils.tools.data_analysis import relay
    from src.anubis.utils.tools.data_analysis.backend import mcp_connection_namespace

    async def run():
        for device_id, server_name in (
            ("d-ubuntu", "Ubuntu-OS-Filesystem"),
            ("d-macos", "macOS-Filesystem"),
        ):
            await webapp_module.mcp_register(
                request=_FakeRequest(
                    {"device_id": device_id, "server_name": server_name}
                ),
                current_user=_user(),
            )

        # Only the Ubuntu machine has been adopted by an avatar.
        await store_client.put_item(
            list(mcp_connection_namespace("auth0|u1")),
            key="d-ubuntu",
            value={
                "status": "connected",
                "device_id": "d-ubuntu",
                "device_label": "Ubuntu",
                "assistant_id": "a1",
                "connected_at": "2026-08-11T00:00:00+00:00",
            },
        )
        # …and only the Ubuntu machine holds a live relay socket.
        monkeypatch.setattr(
            relay, "is_online", lambda device_id: device_id == "d-ubuntu"
        )

        response = await webapp_module.list_mcp_connections(current_user=_user())
        payload = json.loads(bytes(response.body))
        devices = {device["device_id"]: device for device in payload["devices"]}

        assert payload["device_count"] == 2
        assert devices["d-ubuntu"]["online"] is True
        assert devices["d-ubuntu"]["connected"] is True
        assert devices["d-ubuntu"]["bound_assistant_id"] == "a1"
        assert devices["d-macos"]["online"] is False
        assert devices["d-macos"]["connected"] is False
        # Host directory paths never appear in the listing.
        assert all("allowed_roots" not in device for device in payload["devices"])

    asyncio.run(run())
