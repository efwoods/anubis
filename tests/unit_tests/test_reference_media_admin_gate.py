"""reference_media is reserved for admin-created avatars.

Ordinary identity uploads stay available to every creator. The flag that
stores a consultable item (a menu) in the document namespace, without
identity analysis, is refused unless the avatar's ``metadata.user_id`` is
``ADMIN_USER_ID``.
"""

import io
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from src.api.webapp import avatar_was_created_by_administrator


def test_avatar_was_created_by_administrator_matches_admin_user_id():
    context = SimpleNamespace(admin_user_id="admin-1")
    assert avatar_was_created_by_administrator("admin-1", context) is True
    assert avatar_was_created_by_administrator("  admin-1  ", context) is True
    assert avatar_was_created_by_administrator("someone-else", context) is False
    assert avatar_was_created_by_administrator("admin-1", None) is False
    assert (
        avatar_was_created_by_administrator(
            "admin-1", SimpleNamespace(admin_user_id=None)
        )
        is False
    )
    assert avatar_was_created_by_administrator(None, context) is False
    assert avatar_was_created_by_administrator("", context) is False


def _upload_file(filename: str, data: bytes, content_type: str) -> UploadFile:
    return UploadFile(
        filename=filename,
        file=io.BytesIO(data),
        headers=Headers({"content-type": content_type}),
    )


@pytest.fixture
def upload_endpoint_environment(monkeypatch):
    """Stub the upload endpoint the same way ``test_media_type_validation`` does."""
    import src.api.webapp as webapp

    monkeypatch.setattr(webapp, "enforce_tier_capability", lambda *a, **k: None)

    class _Assistants:
        async def get(self, assistant_id):
            return {
                "metadata": {"user_id": "u1"},
                "name": "Avatar",
                "description": "d",
            }

    monkeypatch.setattr(
        webapp, "get_client", lambda **k: SimpleNamespace(assistants=_Assistants())
    )

    async def _estimate(entries):
        for entry in entries:
            entry["estimated_tokens"] = 1
        return len(entries)

    monkeypatch.setattr(webapp, "_estimate_media_entries_tokens", _estimate)

    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(webapp, "enforce_remaining_allotment", _noop)
    monkeypatch.setattr(webapp, "enforce_token_rate_limit", _noop)
    monkeypatch.setattr(
        webapp,
        "resolve_metering_bypass",
        lambda user: SimpleNamespace(
            skips_metering_writes=True, usage_response_fields=lambda: {}
        ),
    )

    async def _usage_snapshot(*a, **k):
        return {}

    monkeypatch.setattr(webapp, "_build_meter_usage_snapshot", _usage_snapshot)
    monkeypatch.setattr(webapp, "run_batch_media_job", _noop)

    class _Store:
        async def asearch(self, namespace, limit=None):
            return []

    webapp.app.state.store = _Store()
    webapp.app.state.media_jobs = {}
    webapp.app.state.context = SimpleNamespace(media_processing_concurrency=1)
    webapp.app.state.stripe = None
    webapp.app.state.pool = None
    return webapp


def _markdown_upload() -> UploadFile:
    return _upload_file("menu.md", b"Cortado $5.72\nAmericano $5.17\n", "text/markdown")


@pytest.mark.asyncio
async def test_reference_media_flag_is_refused_when_admin_user_id_is_unset(
    upload_endpoint_environment,
):
    """No configured administrator means the flag is refused for every avatar."""
    webapp = upload_endpoint_environment
    webapp.app.state.context = SimpleNamespace(
        media_processing_concurrency=1, admin_user_id=None
    )

    with pytest.raises(HTTPException) as excinfo:
        await webapp.update_avatar_identity_with_media(
            files=[_markdown_upload()],
            assistant_id="a1",
            reference_media=True,
            current_user={"identities": [{"user_id": "u1"}], "API_KEY": "k"},
        )

    assert excinfo.value.status_code == 403


@pytest.mark.asyncio
async def test_reference_media_flag_is_refused_when_avatar_is_not_admin_created(
    upload_endpoint_environment,
):
    """A non-admin avatar must not accept reference_media."""
    webapp = upload_endpoint_environment
    webapp.app.state.context = SimpleNamespace(
        media_processing_concurrency=1, admin_user_id="admin-1"
    )

    with pytest.raises(HTTPException) as excinfo:
        await webapp.update_avatar_identity_with_media(
            files=[_markdown_upload()],
            assistant_id="a1",
            reference_media=True,
            current_user={"identities": [{"user_id": "u1"}], "API_KEY": "k"},
        )

    assert excinfo.value.status_code == 403
    assert "administrator" in str(excinfo.value.detail)


@pytest.mark.asyncio
async def test_ordinary_upload_still_works_when_avatar_is_not_admin_created(
    upload_endpoint_environment,
):
    """Leaving the flag off must not change an ordinary creator's upload."""
    webapp = upload_endpoint_environment
    webapp.app.state.context = SimpleNamespace(
        media_processing_concurrency=1, admin_user_id="admin-1"
    )

    response = await webapp.update_avatar_identity_with_media(
        files=[_markdown_upload()],
        assistant_id="a1",
        reference_media=False,
        current_user={"identities": [{"user_id": "u1"}], "API_KEY": "k"},
    )

    assert response.status_code == 202
    payload = json.loads(response.body)
    assert payload["items_accepted"] == 1


@pytest.mark.asyncio
async def test_reference_media_flag_is_accepted_for_an_admin_created_avatar(
    upload_endpoint_environment, monkeypatch
):
    """The administrator's own avatar may upload consultable reference media."""
    webapp = upload_endpoint_environment

    class _Assistants:
        async def get(self, assistant_id):
            return {
                "metadata": {"user_id": "admin-1"},
                "name": "Avatar",
                "description": "d",
            }

    monkeypatch.setattr(
        webapp, "get_client", lambda **k: SimpleNamespace(assistants=_Assistants())
    )
    webapp.app.state.context = SimpleNamespace(
        media_processing_concurrency=1, admin_user_id="admin-1"
    )

    captured = {}

    async def _capture_batch(**kwargs):
        captured["media_files"] = kwargs.get("media_files")
        return {
            "job_id": "job-1",
            "items_accepted": len(kwargs.get("media_files") or []),
            "filenames": [
                entry.get("filename") for entry in (kwargs.get("media_files") or [])
            ],
        }

    monkeypatch.setattr(webapp, "_start_media_batch", _capture_batch)

    response = await webapp.update_avatar_identity_with_media(
        files=[_markdown_upload()],
        assistant_id="a1",
        reference_media=True,
        current_user={"identities": [{"user_id": "admin-1"}], "API_KEY": "k"},
    )

    assert response.status_code == 202
    payload = json.loads(response.body)
    assert payload["items_accepted"] == 1
    assert captured["media_files"][0]["reference_media"] is True


@pytest.mark.asyncio
async def test_reference_media_cannot_combine_with_portrait_or_voice(
    upload_endpoint_environment, monkeypatch
):
    webapp = upload_endpoint_environment

    class _Assistants:
        async def get(self, assistant_id):
            return {
                "metadata": {"user_id": "admin-1"},
                "name": "Avatar",
                "description": "d",
            }

    monkeypatch.setattr(
        webapp, "get_client", lambda **k: SimpleNamespace(assistants=_Assistants())
    )
    webapp.app.state.context = SimpleNamespace(
        media_processing_concurrency=1, admin_user_id="admin-1"
    )

    with pytest.raises(HTTPException) as excinfo:
        await webapp.update_avatar_identity_with_media(
            files=[_markdown_upload()],
            assistant_id="a1",
            reference_media=True,
            reference_image=True,
            current_user={"identities": [{"user_id": "admin-1"}], "API_KEY": "k"},
        )

    assert excinfo.value.status_code == 400
