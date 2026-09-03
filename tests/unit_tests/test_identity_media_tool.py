"""Learning from media in conversation.

- The message endpoint records the turn's raw files; the graph offers
  ``update_avatar_identity_with_media`` to the avatar's creator, and the starter
  re-checks ownership and the upload tier — no per-request flag anywhere.
- The tool selects attachments by filename, validates the reference flags, and
  hands the batch to the published starter — never touching the graph state.
- The starter builds the same media entries the upload endpoint builds and
  turns allotment refusals into a status the avatar can explain.
"""

from __future__ import annotations

import time

import pytest
from fastapi import HTTPException

from src.anubis.utils import runtime_handles
from src.anubis.utils.tools.identity.identity_media_tools import (
    IDENTITY_MEDIA_TOOL_NAME,
    build_identity_media_tools,
)
from src.api import chat_attachments
from src.api.chat_attachments import (
    TurnAttachment,
    describe_turn_attachments,
    forget_turn_attachments,
    get_turn_attachments,
    remember_turn_attachments,
)

USER_ID = "auth0|creator"
ASSISTANT_ID = "assistant-1"
THREAD_ID = "thread-1"
CURRENT_USER = {
    "identities": [{"user_id": USER_ID}],
    "email": "creator@example.com",
    "API_KEY": "k",
}
PORTRAIT = TurnAttachment("me.png", "image/png", b"\x89PNG")
RECORDING = TurnAttachment("me.mp3", "audio/mpeg", b"ID3")


@pytest.fixture(autouse=True)
def _clean_registry():
    forget_turn_attachments(THREAD_ID)
    runtime_handles.set_identity_media_job_starter(None)
    yield
    forget_turn_attachments(THREAD_ID)
    runtime_handles.set_identity_media_job_starter(None)


def _tool():
    tools = build_identity_media_tools(
        None,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        assistant_ctx={
            "name": "Evan",
            "description": "",
            "metadata": {"user_id": USER_ID},
        },
        thread_id=THREAD_ID,
    )
    assert [tool.name for tool in tools] == [IDENTITY_MEDIA_TOOL_NAME]
    return tools[0]


class _Starter:
    def __init__(self, result=None, error=None):
        self.calls = []
        self.result = result or {
            "status": "started",
            "job_id": "job-1",
            "items_accepted": 1,
        }
        self.error = error

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.result


# --------------------------------------------------------------------------- registry


def test_the_registry_keeps_one_record_per_thread_and_expires_it():
    remember_turn_attachments(THREAD_ID, [PORTRAIT], CURRENT_USER)
    record = get_turn_attachments(THREAD_ID)
    assert [a.filename for a in record.attachments] == ["me.png"]
    assert record.current_user is CURRENT_USER
    assert describe_turn_attachments(THREAD_ID) == [
        {"filename": "me.png", "mime_type": "image/png", "size_bytes": 4}
    ]
    # A new turn replaces the previous record.
    remember_turn_attachments(THREAD_ID, [RECORDING], CURRENT_USER)
    assert [a.filename for a in get_turn_attachments(THREAD_ID).attachments] == [
        "me.mp3"
    ]
    # Expiry.
    get_turn_attachments(THREAD_ID).remembered_at = (
        time.monotonic() - chat_attachments.TURN_ATTACHMENT_TTL_SECONDS - 1
    )
    assert get_turn_attachments(THREAD_ID) is None
    assert describe_turn_attachments(THREAD_ID) == []
    assert get_turn_attachments(None) is None


# --------------------------------------------------------------------------- tool


@pytest.mark.asyncio
async def test_the_tool_reports_unavailable_without_a_published_starter():
    remember_turn_attachments(THREAD_ID, [PORTRAIT], CURRENT_USER)
    result = await _tool().ainvoke({})
    assert result["status"] == "unavailable"


@pytest.mark.asyncio
async def test_the_tool_needs_something_to_learn_from():
    runtime_handles.set_identity_media_job_starter(_Starter())
    remember_turn_attachments(THREAD_ID, [], CURRENT_USER)
    result = await _tool().ainvoke({})
    assert result["status"] == "nothing_to_learn"


@pytest.mark.asyncio
async def test_the_tool_selects_attachments_by_filename_and_names_missing_ones():
    starter = _Starter()
    runtime_handles.set_identity_media_job_starter(starter)
    remember_turn_attachments(THREAD_ID, [PORTRAIT, RECORDING], CURRENT_USER)
    result = await _tool().ainvoke({"filenames": ["nope.txt"]})
    assert result["status"] == "not_found"
    assert "nope.txt" in result["detail"]
    assert starter.calls == []

    result = await _tool().ainvoke({"filenames": ["me.mp3"], "reference_audio": True})
    assert result["status"] == "started"
    call = starter.calls[-1]
    assert [a.filename for a in call["attachments"]] == ["me.mp3"]
    assert call["reference_audio"] is True and call["reference_image"] is False
    assert call["current_user"] is CURRENT_USER
    assert call["assistant_id"] == ASSISTANT_ID
    assert call["assistant_ctx"]["metadata"] == {"user_id": USER_ID}


@pytest.mark.asyncio
async def test_the_tool_learns_from_every_attachment_and_link_by_default():
    starter = _Starter()
    runtime_handles.set_identity_media_job_starter(starter)
    remember_turn_attachments(THREAD_ID, [PORTRAIT, RECORDING], CURRENT_USER)
    result = await _tool().ainvoke({"urls": [" https://youtu.be/abc "]})
    assert result["job_id"] == "job-1"
    call = starter.calls[-1]
    assert [a.filename for a in call["attachments"]] == ["me.png", "me.mp3"]
    assert call["urls"] == ["https://youtu.be/abc"]


@pytest.mark.asyncio
async def test_reference_flags_are_exclusive_and_single_item():
    starter = _Starter()
    runtime_handles.set_identity_media_job_starter(starter)
    remember_turn_attachments(THREAD_ID, [PORTRAIT, RECORDING], CURRENT_USER)
    both = await _tool().ainvoke({"reference_image": True, "reference_audio": True})
    assert both["status"] == "invalid"
    two_items = await _tool().ainvoke({"reference_image": True})
    assert two_items["status"] == "invalid"
    assert starter.calls == []


@pytest.mark.asyncio
async def test_a_starter_failure_becomes_a_status_not_an_exception():
    runtime_handles.set_identity_media_job_starter(_Starter(error=RuntimeError("boom")))
    remember_turn_attachments(THREAD_ID, [PORTRAIT], CURRENT_USER)
    result = await _tool().ainvoke({})
    assert result == {"status": "error", "detail": "boom"}


# --------------------------------------------------------------------------- starter


@pytest.mark.asyncio
async def test_the_chat_starter_builds_entries_and_starts_the_shared_batch(monkeypatch):
    from src.api import webapp as webapp_module

    built = []

    async def _file_entries(filename, content, mime_type, **kwargs):
        built.append(("file", filename, kwargs["reference_image"]))
        return [
            {
                "filename": filename,
                "namespace_filename": filename,
                "estimated_tokens": 3,
            }
        ]

    async def _url_entries(url, **kwargs):
        built.append(("url", url, kwargs["rich"]))
        return [{"filename": url, "namespace_filename": url, "estimated_tokens": 5}]

    batches = []

    async def _batch(**kwargs):
        batches.append(kwargs)
        return {
            "job_id": "master-1",
            "items_accepted": len(kwargs["media_files"]),
            "filenames": [m["filename"] for m in kwargs["media_files"]],
            "items_rejected": len(kwargs["rejected_items"]),
            "rejected": kwargs["rejected_items"],
            "playlists_expanding": len(kwargs["playlist_urls"]),
            "message": "Media processing started",
        }

    monkeypatch.setattr(webapp_module, "enforce_tier_capability", lambda *a, **k: None)
    monkeypatch.setattr(webapp_module, "_build_media_entries_for_file", _file_entries)
    monkeypatch.setattr(webapp_module, "_build_media_entries_for_url", _url_entries)
    monkeypatch.setattr(webapp_module, "_start_media_batch", _batch)

    result = await webapp_module.start_identity_media_job_from_chat(
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        assistant_ctx={
            "name": "Evan",
            "description": None,
            "metadata": {"user_id": USER_ID},
        },
        current_user=CURRENT_USER,
        attachments=[PORTRAIT],
        urls=[
            "https://example.com/post",
            "https://www.youtube.com/playlist?list=PL123",
        ],
    )
    assert result["status"] == "started"
    assert result["job_id"] == "master-1"
    assert result["items_accepted"] == 2
    assert result["playlists_expanding"] == 1
    assert ("file", "me.png", False) in built
    assert ("url", "https://example.com/post", False) in built
    batch = batches[0]
    assert batch["user_id"] == USER_ID
    assert batch["config"]["configurable"]["assistant_ctx"]["metadata"] == {
        "user_id": USER_ID
    }
    assert batch["current_user"] is CURRENT_USER


@pytest.mark.asyncio
async def test_the_chat_starter_turns_refusals_and_rejections_into_statuses(
    monkeypatch,
):
    from src.api import webapp as webapp_module

    async def _broken(filename, content, mime_type, **kwargs):
        raise RuntimeError("unsupported media type")

    monkeypatch.setattr(webapp_module, "enforce_tier_capability", lambda *a, **k: None)
    monkeypatch.setattr(webapp_module, "_build_media_entries_for_file", _broken)
    rejected = await webapp_module.start_identity_media_job_from_chat(
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        assistant_ctx={"metadata": {"user_id": USER_ID}},
        current_user=CURRENT_USER,
        attachments=[PORTRAIT],
        urls=[],
    )
    assert rejected["status"] == "rejected"
    assert rejected["rejected"][0]["filename"] == "me.png"

    async def _ok(filename, content, mime_type, **kwargs):
        return [{"filename": filename, "namespace_filename": filename}]

    async def _over_budget(**kwargs):
        raise HTTPException(status_code=402, detail="allotment spent")

    monkeypatch.setattr(webapp_module, "_build_media_entries_for_file", _ok)
    monkeypatch.setattr(webapp_module, "_start_media_batch", _over_budget)
    refused = await webapp_module.start_identity_media_job_from_chat(
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        assistant_ctx={"metadata": {"user_id": USER_ID}},
        current_user=CURRENT_USER,
        attachments=[PORTRAIT],
        urls=[],
    )
    assert refused == {
        "status": "refused",
        "status_code": 402,
        "detail": "allotment spent",
    }


@pytest.mark.asyncio
async def test_the_chat_starter_enforces_ownership_and_the_upload_tier(monkeypatch):
    from src.api import webapp as webapp_module

    async def _never(*args, **kwargs):
        raise AssertionError("nothing should be built for a refused caller")

    monkeypatch.setattr(webapp_module, "_build_media_entries_for_file", _never)
    monkeypatch.setattr(webapp_module, "_start_media_batch", _never)

    monkeypatch.setattr(webapp_module, "enforce_tier_capability", lambda *a, **k: None)
    visitor = await webapp_module.start_identity_media_job_from_chat(
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        assistant_ctx={"metadata": {"user_id": "someone-else"}},
        current_user=CURRENT_USER,
        attachments=[PORTRAIT],
        urls=[],
    )
    assert visitor["status"] == "refused" and visitor["status_code"] == 403

    def _free_tier(*args, **kwargs):
        raise HTTPException(status_code=403, detail="Upgrade to upload media.")

    monkeypatch.setattr(webapp_module, "enforce_tier_capability", _free_tier)
    free = await webapp_module.start_identity_media_job_from_chat(
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        assistant_ctx={"metadata": {"user_id": USER_ID}},
        current_user=CURRENT_USER,
        attachments=[PORTRAIT],
        urls=[],
    )
    assert free == {
        "status": "refused",
        "status_code": 403,
        "detail": "Upgrade to upload media.",
    }
