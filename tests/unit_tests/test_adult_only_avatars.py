"""Adult-only is an administrator setting; search hides those avatars until age is verified, except from the administrator."""

from datetime import date
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.anubis.utils.age_verification.repository import (
    InMemoryAgeVerificationRepository,
    set_age_verification_repository,
)
from src.api import webapp as webapp_module

ASSISTANT_ID = "assistant-adult"
CREATOR_ID = "creator-1"
STRANGER_ID = "stranger-1"
ADMIN_ID = "the-admin"
TODAY = date(2026, 9, 14)


def _current_user(user_id, *, anonymous=False):
    return {
        "API_KEY": "sk-test-key",
        "identities": [{"user_id": user_id}],
        "is_anonymous": anonymous,
        "email": None if anonymous else f"{user_id}@example.com",
    }


def _request_with_query(**parameters):
    return SimpleNamespace(query_params=parameters)


class _AssistantsAPI:
    def __init__(self, metadata):
        self._metadata = dict(metadata)
        self.updates = []

    async def get(self, assistant_id):
        return {"assistant_id": assistant_id, "metadata": dict(self._metadata)}

    async def update(self, **kwargs):
        self.updates.append(kwargs)
        metadata = kwargs.get("metadata") or {}
        self._metadata.update(metadata)
        return {"assistant_id": kwargs.get("assistant_id"), "metadata": dict(self._metadata)}


def _install(monkeypatch, metadata, *, repository=None):
    assistants_api = _AssistantsAPI(metadata)
    monkeypatch.setattr(
        webapp_module,
        "get_client",
        lambda **kwargs: SimpleNamespace(assistants=assistants_api),
    )
    monkeypatch.setattr(
        webapp_module.app.state,
        "context",
        SimpleNamespace(admin_user_id=ADMIN_ID, age_verification_minimum_years=18),
        raising=False,
    )
    set_age_verification_repository(repository)
    return assistants_api


@pytest.mark.asyncio
async def test_only_the_administrator_may_mark_an_avatar_adult_only(monkeypatch):
    assistants_api = _install(monkeypatch, {"user_id": CREATOR_ID})

    with pytest.raises(HTTPException) as refused:
        await webapp_module.modify_avatar(
            request=_request_with_query(adult_only="true"),
            assistant_id=ASSISTANT_ID,
            current_user=_current_user(CREATOR_ID),
            adult_only=True,
        )
    assert refused.value.status_code == 403
    assert assistants_api.updates == []

    await webapp_module.modify_avatar(
        request=_request_with_query(adult_only="true"),
        assistant_id=ASSISTANT_ID,
        current_user=_current_user(ADMIN_ID),
        adult_only=True,
    )
    assert assistants_api.updates[-1]["metadata"]["adult_only"] is True


@pytest.mark.asyncio
async def test_the_administrator_may_clear_the_adult_only_flag(monkeypatch):
    assistants_api = _install(
        monkeypatch, {"user_id": CREATOR_ID, "adult_only": True}
    )

    await webapp_module.modify_avatar(
        request=_request_with_query(adult_only="false"),
        assistant_id=ASSISTANT_ID,
        current_user=_current_user(ADMIN_ID),
        adult_only=False,
    )
    assert assistants_api.updates[-1]["metadata"]["adult_only"] is False


def test_public_listing_lifts_adult_only_without_the_creator():
    stripped = webapp_module._assistant_without_metadata(
        {
            "assistant_id": ASSISTANT_ID,
            "name": "Adult Avatar",
            "metadata": {
                "user_id": CREATOR_ID,
                "adult_only": True,
                "is_public": True,
            },
        }
    )
    assert stripped["adult_only"] is True
    assert "metadata" not in stripped
    assert CREATOR_ID not in str(stripped)


@pytest.mark.asyncio
async def test_search_hides_adult_only_avatars_until_the_viewer_verifies_age(
    monkeypatch,
):
    repository = InMemoryAgeVerificationRepository()
    _install(monkeypatch, {}, repository=repository)
    adult = {
        "assistant_id": "adult-1",
        "name": "Adult Avatar",
        "metadata": {"user_id": CREATOR_ID, "adult_only": True, "is_public": True},
    }
    ordinary = {
        "assistant_id": "guide-1",
        "name": "Guide",
        "metadata": {"user_id": CREATOR_ID, "is_public": True},
    }

    hidden = await webapp_module._avatars_visible_to_viewer(
        [adult, ordinary],
        _current_user(STRANGER_ID),
    )
    assert [avatar["assistant_id"] for avatar in hidden] == ["guide-1"]

    owner_without_verification = await webapp_module._avatars_visible_to_viewer(
        [adult, ordinary],
        _current_user(CREATOR_ID),
    )
    assert [avatar["assistant_id"] for avatar in owner_without_verification] == [
        "guide-1"
    ]
    admin_without_verification = await webapp_module._avatars_visible_to_viewer(
        [adult, ordinary],
        _current_user(ADMIN_ID),
    )
    assert [avatar["assistant_id"] for avatar in admin_without_verification] == [
        "adult-1",
        "guide-1",
    ]
    admin_by_email = await webapp_module._avatars_visible_to_viewer(
        [adult, ordinary],
        {
            "API_KEY": "sk-test-key",
            "identities": [{"user_id": "not-the-stored-admin-id"}],
            "is_anonymous": False,
            "email": "e.woods.business@icloud.com",
        },
    )
    assert [avatar["assistant_id"] for avatar in admin_by_email] == [
        "adult-1",
        "guide-1",
    ]

    await repository.set_verification(STRANGER_ID, date(1990, 5, 1))
    shown = await webapp_module._avatars_visible_to_viewer(
        [adult, ordinary],
        _current_user(STRANGER_ID),
    )
    assert [avatar["assistant_id"] for avatar in shown] == ["adult-1", "guide-1"]


@pytest.mark.asyncio
async def test_a_share_link_lookup_still_returns_an_adult_only_avatar(monkeypatch):
    _install(monkeypatch, {})
    adult = {
        "assistant_id": ASSISTANT_ID,
        "metadata": {"user_id": CREATOR_ID, "adult_only": True},
    }
    visible = await webapp_module._avatars_visible_to_viewer(
        [adult],
        None,
        allow_direct_lookup=True,
    )
    assert visible == [adult]


@pytest.mark.asyncio
async def test_age_verification_refuses_a_minor_and_accepts_an_adult(monkeypatch):
    repository = InMemoryAgeVerificationRepository()
    _install(monkeypatch, {}, repository=repository)

    with pytest.raises(HTTPException) as refused:
        await webapp_module.set_age_verification_route(
            body=webapp_module.AgeVerificationRequest(date_of_birth="2015-01-01"),
            current_user=_current_user(STRANGER_ID),
        )
    assert refused.value.status_code == 403
    assert await repository.get_verification(STRANGER_ID) is None

    response = await webapp_module.set_age_verification_route(
        body=webapp_module.AgeVerificationRequest(date_of_birth="1990-01-01"),
        current_user=_current_user(STRANGER_ID),
    )
    assert response.status_code == 200
    body = response.body
    # JSONResponse stores bytes; the date of birth must not leave the server.
    assert b"1990" not in body
    assert await repository.is_verified(STRANGER_ID, minimum_years=18, on_date=TODAY)


@pytest.mark.asyncio
async def test_age_verification_status_omits_the_date_of_birth(monkeypatch):
    repository = InMemoryAgeVerificationRepository()
    await repository.set_verification(STRANGER_ID, date(1990, 1, 1))
    _install(monkeypatch, {}, repository=repository)

    response = await webapp_module.get_age_verification_route(
        current_user=_current_user(STRANGER_ID)
    )
    assert response.status_code == 200
    assert b"1990" not in response.body
