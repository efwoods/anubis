"""Unit tests for auto-provisioning the one personal avatar per verified account.

Every signed-up user always has exactly one personal avatar — the single avatar
carrying ``is_personal_avatar_of_creator`` — so no feature ever has to answer
"create a personal avatar first". These tests cover the provisioning that makes
that invariant true: creation on the first verified request, idempotence through
the Auth0 ``app_metadata`` marker, adoption of an avatar the owner flagged by
hand, demotion of any second flagged avatar, and the retry behavior that follows
a failure.

The last two groups guard the re-entrancy hazard that is specific to this
feature. Creating an avatar is an HTTP call back into this same API, and the
LangGraph server authenticates that call through ``get_user_with_api_key`` — the
very function that triggers provisioning. Two defenses keep that from recursing
without bound, and both are asserted here: the API-key cache is populated BEFORE
provisioning starts, and the provisioning function holds a guard against a second
concurrent pass for the same user.
"""

import asyncio
from types import SimpleNamespace

import langgraph_sdk
import pytest

from src.anubis.utils import avatar_deletion as avatar_deletion_module
from src.anubis.utils import personal_avatar as personal_avatar_module
from src.anubis.utils.personal_avatar import (
    PERSONAL_AVATAR_IDENTIFIER_FIELD,
    PERSONAL_AVATAR_METADATA_FLAG,
    PERSONAL_AVATAR_PROVISIONED_MARKER,
    ensure_personal_avatar_for_user,
    resolve_personal_avatar,
)
from src.security import auth as auth_module


class _RecordingAssistantsAPI:
    """Fake LangGraph assistants API recording every create/search/update."""

    def __init__(self, existing_avatars=None, create_hook=None):
        self.existing_avatars = list(existing_avatars or [])
        self.create_calls = []
        self.update_calls = []
        self._create_hook = create_hook

    async def search(self, metadata=None, limit=None, offset=0, headers=None):
        owned = [
            avatar
            for avatar in self.existing_avatars
            if (avatar.get("metadata") or {}).get("user_id")
            == (metadata or {}).get("user_id")
        ]
        return owned[offset : offset + (limit or len(owned))]

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        if self._create_hook is not None:
            await self._create_hook()
        created = {
            "assistant_id": kwargs["assistant_id"],
            "name": kwargs.get("name"),
            "description": kwargs.get("description"),
            "metadata": kwargs.get("metadata"),
        }
        self.existing_avatars.append(created)
        return created

    async def update(self, assistant_id=None, metadata=None, **kwargs):
        self.update_calls.append((assistant_id, metadata))
        for avatar in self.existing_avatars:
            if avatar.get("assistant_id") == assistant_id:
                (avatar.setdefault("metadata", {})).update(metadata or {})
        return {"assistant_id": assistant_id}


class _RecordingStoreAPI:
    def __init__(self):
        self.put_item_calls = []

    async def put_item(self, namespace, key=None, value=None):
        self.put_item_calls.append((tuple(namespace), key, value))


class _FakeLangGraphClient:
    def __init__(self, existing_avatars=None, create_hook=None):
        self.assistants = _RecordingAssistantsAPI(existing_avatars, create_hook)
        self.store = _RecordingStoreAPI()


def _verified_user(app_metadata=None, name="Evan Woods"):
    return {
        "user_id": "auth0|new-account",
        "email": "owner@example.com",
        "name": name,
        "email_verified": True,
        "identities": [{"user_id": "new-account"}],
        "app_metadata": dict(app_metadata or {}),
    }


def _owned_avatar(assistant_id, *, personal=False, user_id="new-account"):
    return {
        "assistant_id": assistant_id,
        "name": assistant_id,
        "metadata": {
            "user_id": user_id,
            "is_public": False,
            PERSONAL_AVATAR_METADATA_FLAG: personal,
        },
    }


@pytest.fixture
def recorded_app_metadata_writes(monkeypatch):
    writes = []

    async def _record_write(request, auth0_user_id, fields):
        writes.append((auth0_user_id, fields))
        return True

    monkeypatch.setattr(auth_module, "update_user_app_metadata_fields", _record_write)
    return writes


@pytest.fixture
def fake_langgraph_client(monkeypatch):
    """Install one fake LangGraph client and hand it back for assertions."""

    holder = SimpleNamespace(client=None)

    def _install(existing_avatars=None, create_hook=None):
        holder.client = _FakeLangGraphClient(existing_avatars, create_hook)
        monkeypatch.setattr(
            langgraph_sdk, "get_client", lambda **kwargs: holder.client
        )
        # search_all_avatars_for_user is imported lazily from this module by the
        # provisioning helpers, so patching the attribute here is what the
        # helpers resolve at call time.
        monkeypatch.setattr(
            avatar_deletion_module,
            "search_all_avatars_for_user",
            lambda client, user_id, headers=None: client.assistants.search(
                metadata={"user_id": user_id}, limit=1000
            ),
        )
        return holder.client

    return _install


@pytest.mark.asyncio
async def test_first_verified_request_creates_the_flagged_personal_avatar(
    fake_langgraph_client, recorded_app_metadata_writes
):
    client = fake_langgraph_client()
    user = _verified_user()

    created = await ensure_personal_avatar_for_user(object(), user, "sk-test-key")

    (create_kwargs,) = client.assistants.create_calls
    assert create_kwargs["graph_id"] == "Anubis"
    assert create_kwargs["name"] == "Evan Woods"
    assert create_kwargs["metadata"] == {
        "user_id": "new-account",
        "is_public": False,
        PERSONAL_AVATAR_METADATA_FLAG: True,
    }
    # The creator_id store item is written exactly as /create_avatar writes it,
    # so every downstream reader treats an auto-provisioned avatar identically.
    assert client.store.put_item_calls == [
        (
            (created["assistant_id"], "creator_id"),
            "creator_id",
            {"value": "new-account"},
        )
    ]
    ((auth0_user_id, fields),) = recorded_app_metadata_writes
    assert auth0_user_id == "auth0|new-account"
    assert fields[PERSONAL_AVATAR_PROVISIONED_MARKER] is True
    assert fields[PERSONAL_AVATAR_IDENTIFIER_FIELD] == created["assistant_id"]


@pytest.mark.asyncio
async def test_the_marker_keeps_provisioning_off_the_langgraph_server(
    fake_langgraph_client, recorded_app_metadata_writes
):
    client = fake_langgraph_client()
    user = _verified_user(
        app_metadata={PERSONAL_AVATAR_PROVISIONED_MARKER: True}
    )

    assert await ensure_personal_avatar_for_user(object(), user, "sk-test-key") is None
    assert client.assistants.create_calls == []
    assert recorded_app_metadata_writes == []


@pytest.mark.asyncio
async def test_an_avatar_the_owner_flagged_by_hand_is_adopted_not_duplicated(
    fake_langgraph_client, recorded_app_metadata_writes
):
    client = fake_langgraph_client(
        existing_avatars=[
            _owned_avatar("avatar-plain"),
            _owned_avatar("avatar-personal", personal=True),
        ]
    )

    adopted = await ensure_personal_avatar_for_user(
        object(), _verified_user(), "sk-test-key"
    )

    assert adopted["assistant_id"] == "avatar-personal"
    assert client.assistants.create_calls == []
    ((_, fields),) = recorded_app_metadata_writes
    assert fields[PERSONAL_AVATAR_IDENTIFIER_FIELD] == "avatar-personal"


@pytest.mark.asyncio
async def test_adoption_demotes_any_other_flagged_avatar(
    fake_langgraph_client, recorded_app_metadata_writes
):
    """Adoption must restore exactly-one, not merely pick one.

    An account reaches this branch holding two flagged avatars only when an
    earlier demotion failed. Adopting the first without clearing the rest would
    leave the invariant broken permanently, because the marker written here stops
    provisioning from ever running again.
    """
    client = fake_langgraph_client(
        existing_avatars=[
            _owned_avatar("avatar-personal", personal=True),
            _owned_avatar("avatar-stale-personal", personal=True),
        ]
    )

    adopted = await ensure_personal_avatar_for_user(
        object(), _verified_user(), "sk-test-key"
    )

    assert adopted["assistant_id"] == "avatar-personal"
    assert client.assistants.create_calls == []
    assert client.assistants.update_calls == [
        ("avatar-stale-personal", {PERSONAL_AVATAR_METADATA_FLAG: False})
    ]
    remaining_personal = [
        avatar["assistant_id"]
        for avatar in client.assistants.existing_avatars
        if (avatar.get("metadata") or {}).get(PERSONAL_AVATAR_METADATA_FLAG) is True
    ]
    assert remaining_personal == ["avatar-personal"]


@pytest.mark.asyncio
async def test_creation_demotes_a_previously_flagged_avatar(
    fake_langgraph_client, recorded_app_metadata_writes, monkeypatch
):
    """Creation demotes too, so a stale flag cannot survive provisioning.

    Reaching creation while a flagged avatar exists requires the lookup to have
    missed it, which is what a transient enumeration failure looks like.
    """
    client = fake_langgraph_client(
        existing_avatars=[_owned_avatar("avatar-stale-personal", personal=True)]
    )

    async def _lookup_finds_nothing(client, user_id):
        return None

    monkeypatch.setattr(
        personal_avatar_module, "find_personal_avatar", _lookup_finds_nothing
    )

    created = await ensure_personal_avatar_for_user(
        object(), _verified_user(), "sk-test-key"
    )

    assert client.assistants.update_calls == [
        ("avatar-stale-personal", {PERSONAL_AVATAR_METADATA_FLAG: False})
    ]
    remaining_personal = [
        avatar["assistant_id"]
        for avatar in client.assistants.existing_avatars
        if (avatar.get("metadata") or {}).get(PERSONAL_AVATAR_METADATA_FLAG) is True
    ]
    assert remaining_personal == [created["assistant_id"]]


@pytest.mark.asyncio
async def test_a_failure_leaves_the_marker_unwritten_so_the_next_request_retries(
    fake_langgraph_client, recorded_app_metadata_writes
):
    async def _fail_on_create():
        raise RuntimeError("langgraph unavailable")

    client = fake_langgraph_client(create_hook=_fail_on_create)
    user = _verified_user()

    with pytest.raises(RuntimeError):
        await ensure_personal_avatar_for_user(object(), user, "sk-test-key")

    assert recorded_app_metadata_writes == []
    assert PERSONAL_AVATAR_PROVISIONED_MARKER not in user["app_metadata"]
    # The guard is released on the failure path, so the retry is not blocked.
    assert personal_avatar_module._user_identifiers_being_provisioned == set()

    # The retry succeeds and creates exactly one avatar.
    client.assistants._create_hook = None
    created = await ensure_personal_avatar_for_user(object(), user, "sk-test-key")
    assert created is not None
    assert len(client.assistants.create_calls) == 2


@pytest.mark.asyncio
async def test_a_reentrant_call_during_creation_does_not_create_a_second_avatar(
    fake_langgraph_client, recorded_app_metadata_writes
):
    """The guard covers the nested authenticate that avatar creation triggers.

    Creating an avatar calls the LangGraph server, which authenticates the call
    through ``get_user_with_api_key`` — the function that triggers provisioning.
    Here the nested pass is fired from inside ``assistants.create`` itself, the
    worst case, and must be a no-op.
    """
    reentrant_results = []
    user = _verified_user()

    async def _provision_again_from_inside_create():
        reentrant_results.append(
            await ensure_personal_avatar_for_user(object(), user, "sk-test-key")
        )

    client = fake_langgraph_client(create_hook=_provision_again_from_inside_create)

    await ensure_personal_avatar_for_user(object(), user, "sk-test-key")

    assert reentrant_results == [None]
    assert len(client.assistants.create_calls) == 1


@pytest.mark.asyncio
async def test_concurrent_provisioning_for_one_user_creates_exactly_one_avatar(
    fake_langgraph_client, recorded_app_metadata_writes
):
    started = asyncio.Event()
    release = asyncio.Event()

    async def _hold_open_during_create():
        started.set()
        await release.wait()

    client = fake_langgraph_client(create_hook=_hold_open_during_create)
    user = _verified_user()

    first = asyncio.create_task(
        ensure_personal_avatar_for_user(object(), user, "sk-test-key")
    )
    await started.wait()
    second = await ensure_personal_avatar_for_user(object(), user, "sk-test-key")
    release.set()
    await first

    assert second is None
    assert len(client.assistants.create_calls) == 1


@pytest.mark.asyncio
async def test_resolution_self_heals_instead_of_reporting_a_missing_avatar(
    fake_langgraph_client, recorded_app_metadata_writes
):
    """A stale marker with no avatar must still resolve, never raise.

    The personal avatar is guaranteed to exist, so resolution provisions one
    rather than telling the owner to create one.
    """
    client = fake_langgraph_client(existing_avatars=[_owned_avatar("avatar-plain")])
    user = _verified_user(app_metadata={PERSONAL_AVATAR_PROVISIONED_MARKER: True})

    resolved = await resolve_personal_avatar(client, object(), user, "sk-test-key")

    assert resolved is not None
    assert (resolved.get("metadata") or {})[PERSONAL_AVATAR_METADATA_FLAG] is True
    assert len(client.assistants.create_calls) == 1


@pytest.mark.asyncio
async def test_the_endpoint_provisions_rather_than_reporting_a_missing_avatar(
    fake_langgraph_client, recorded_app_metadata_writes, monkeypatch
):
    """``GET /personal_avatar`` never tells the owner to create one first."""
    import json

    from src.api import webapp as webapp_module
    from src.api.webapp import get_personal_avatar

    client = fake_langgraph_client()
    # The web application binds ``get_client`` at import time, so the fake has to
    # replace that binding rather than the software development kit's attribute.
    monkeypatch.setattr(webapp_module, "get_client", lambda **kwargs: client)
    current_user = _verified_user()
    current_user["API_KEY"] = "sk-test-key"

    response = await get_personal_avatar(
        request=object(), current_user=current_user
    )

    payload = json.loads(bytes(response.body).decode())
    assert payload["personal_avatar"]["metadata"][PERSONAL_AVATAR_METADATA_FLAG] is True
    capability_names = [entry["name"] for entry in payload["capabilities"]]
    assert "desktop_data_servers" in capability_names
    assert "mailbox" in capability_names
    # Nothing is connected yet, so every connection-backed capability reports
    # not_configured while adapter training — which no connection step gates —
    # is already active.
    statuses = {entry["name"]: entry["status"] for entry in payload["capabilities"]}
    assert statuses["desktop_data_servers"] == "not_configured"
    assert statuses["adapter_training_from_conversations"] == "active"


@pytest.mark.asyncio
async def test_the_api_key_cache_is_warm_before_provisioning_runs(monkeypatch):
    """The ordering that prevents unbounded recursion.

    Provisioning creates an avatar over HTTP, and that call is authenticated by
    re-entering ``get_user_with_api_key``. The nested call must find the cache
    already populated; were provisioning invoked before the cache write, it would
    re-enter itself without bound.
    """
    observed_cache_state = []

    async def _observe_cache(request, user, api_key):
        cache_key = auth_module._hash_key(api_key)
        observed_cache_state.append(cache_key in auth_module._api_key_cache)
        return None

    monkeypatch.setattr(
        personal_avatar_module, "ensure_personal_avatar_for_user", _observe_cache
    )

    async def _no_enrollment(request, user):
        return None

    monkeypatch.setattr(
        auth_module,
        "ensure_initial_subscription_after_verification",
        _no_enrollment,
    )

    async def _fake_mgmt_headers(request):
        return {}

    monkeypatch.setattr(auth_module, "_mgmt_headers", _fake_mgmt_headers)

    async def _fake_get(url, params=None, headers=None):
        return SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: [_verified_user()]
        )

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(httpx_client=SimpleNamespace(get=_fake_get))
        )
    )

    auth_module._api_key_cache.clear()
    user = await auth_module.get_user_with_api_key("sk-ordering-key", request)

    assert user is not None
    assert observed_cache_state == [True]
