"""Renaming the personal avatar renames the account holder on Auth0.

The personal avatar is the account holder's own portrait. A person who renamed
the personal avatar "Marshall" still appeared as "Marshal" in live-voice
speaker labels, because those labels read the Auth0 profile and nothing ever
wrote the new name there.
"""

from types import SimpleNamespace

import pytest

import src.api.webapp as webapp_module
from src.security.auth import auth0_profile_name_fields

CALLER_ID = "6ab6daff0f5dfbfd6c8f65e6"
PROVIDER_CALLER_ID = f"auth0|{CALLER_ID}"


class _AssistantsAPI:
    def __init__(self, avatar_metadata):
        self.avatar_metadata = avatar_metadata
        self.updates = []

    async def update(self, **kwargs):
        self.updates.append(kwargs)
        return {
            "assistant_id": kwargs.get("assistant_id"),
            "name": kwargs.get("name") or "Old Name",
            "metadata": self.avatar_metadata,
        }


def _install(monkeypatch, avatar_metadata):
    assistants_api = _AssistantsAPI(avatar_metadata)
    monkeypatch.setattr(
        webapp_module,
        "get_client",
        lambda **kwargs: SimpleNamespace(assistants=assistants_api),
    )
    monkeypatch.setattr(
        webapp_module.app.state,
        "context",
        SimpleNamespace(
            admin_user_id="someone-else",
            personal_avatar_research_on_naming_enabled="false",
        ),
        raising=False,
    )
    profile_writes = []

    async def _record_profile_write(request, auth0_user_id, full_name):
        profile_writes.append((auth0_user_id, full_name))
        return True

    monkeypatch.setattr(webapp_module, "update_user_profile_name", _record_profile_write)
    monkeypatch.setattr(
        webapp_module, "_demote_other_personal_avatars", _no_op, raising=False
    )
    return assistants_api, profile_writes


async def _no_op(*args, **kwargs):
    return None


def _caller():
    return {
        "API_KEY": "sk-test-key",
        "user_id": PROVIDER_CALLER_ID,
        "identities": [{"user_id": CALLER_ID}],
    }


def test_the_profile_fields_split_a_full_name():
    assert auth0_profile_name_fields("  Marshall   Woods ") == {
        "name": "Marshall Woods",
        "given_name": "Marshall",
        "family_name": "Woods",
    }
    assert auth0_profile_name_fields("Marshall") == {
        "name": "Marshall",
        "given_name": "Marshall",
    }
    assert auth0_profile_name_fields("   ") == {}


@pytest.mark.asyncio
async def test_renaming_the_personal_avatar_renames_the_auth0_profile(monkeypatch):
    _assistants_api, profile_writes = _install(
        monkeypatch, {"is_personal_avatar_of_creator": True, "user_id": CALLER_ID}
    )
    await webapp_module.modify_avatar(
        request=SimpleNamespace(query_params={}),
        assistant_id="marshall-avatar",
        current_user=_caller(),
        new_avatar_name="Marshall",
    )
    assert profile_writes == [(PROVIDER_CALLER_ID, "Marshall")]


@pytest.mark.asyncio
async def test_renaming_any_other_avatar_leaves_the_profile_alone(monkeypatch):
    _assistants_api, profile_writes = _install(
        monkeypatch, {"is_personal_avatar_of_creator": False, "user_id": CALLER_ID}
    )
    await webapp_module.modify_avatar(
        request=SimpleNamespace(query_params={}),
        assistant_id="character-avatar",
        current_user=_caller(),
        new_avatar_name="Grant Imahara",
    )
    assert profile_writes == []


@pytest.mark.asyncio
async def test_flagging_an_avatar_as_personal_writes_that_avatar_name(monkeypatch):
    _assistants_api, profile_writes = _install(
        monkeypatch, {"is_personal_avatar_of_creator": True, "user_id": CALLER_ID}
    )
    monkeypatch.setattr(
        "src.anubis.utils.personal_avatar.record_personal_avatar_pointer", _no_op
    )
    await webapp_module.modify_avatar(
        request=SimpleNamespace(
            query_params={"is_personal_avatar_of_creator": "true"}
        ),
        assistant_id="marshall-avatar",
        current_user=_caller(),
        is_personal_avatar_of_creator=True,
    )
    assert profile_writes == [(PROVIDER_CALLER_ID, "Old Name")]
