"""Processing-time estimate for media jobs (about one second per media second).

Covers the offline pieces:

* ``estimate_processing_seconds`` — probed media length x the GlobalContext
  factor, ``None`` for anything that is not timed media.
* ``job_estimated_processing_seconds`` / ``job_estimated_media_seconds`` — a
  child reports its own estimate, a master the sum of its children.
* ``_stamp_media_time_estimate`` — the upload path records both numbers on the
  media entry that becomes the child job.
* The upload endpoint, the job snapshot, and the SSE progress stream all carry
  the estimate (and ``estimated_remaining_seconds`` on every frame).
"""

import io
import json
from types import SimpleNamespace

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from src.api.media_jobs import (
    add_event,
    create_child_job,
    create_master_job,
    estimate_processing_seconds,
    finish_job,
    job_estimated_media_seconds,
    job_estimated_processing_seconds,
)


def test_estimate_processing_seconds_scales_media_length_by_factor():
    assert estimate_processing_seconds(2342.0, 1.0) == 2342.0
    assert estimate_processing_seconds(600, 1.5) == 900.0
    assert estimate_processing_seconds("120", "0.5") == 60.0


def test_estimate_processing_seconds_is_none_for_untimed_media():
    assert estimate_processing_seconds(None, 1.0) is None
    assert estimate_processing_seconds(0, 1.0) is None
    assert estimate_processing_seconds(120.0, None) is None
    assert estimate_processing_seconds(120.0, 0) is None
    assert estimate_processing_seconds("not-a-number", 1.0) is None


def test_master_reports_sum_of_children_and_child_reports_its_own():
    registry = {}
    master = create_master_job(registry, "u1", "a1")
    video = create_child_job(
        registry,
        user_id="u1",
        assistant_id="a1",
        parent_id=master.job_id,
        filename="talk.mp4",
        namespace_filename="talk",
        estimated_media_seconds=2342.0,
        estimated_processing_seconds=2342.0,
    )
    document = create_child_job(
        registry,
        user_id="u1",
        assistant_id="a1",
        parent_id=master.job_id,
        filename="notes.md",
        namespace_filename="notes",
    )
    clip = create_child_job(
        registry,
        user_id="u1",
        assistant_id="a1",
        parent_id=master.job_id,
        filename="clip.mp3",
        namespace_filename="clip",
        estimated_media_seconds=60.0,
        estimated_processing_seconds=90.0,
    )
    master.child_ids.extend([video.job_id, document.job_id, clip.job_id])

    assert job_estimated_processing_seconds(registry, video) == 2342.0
    assert job_estimated_processing_seconds(registry, document) is None
    assert job_estimated_processing_seconds(registry, master) == 2432.0
    assert job_estimated_media_seconds(registry, master) == 2402.0


def test_master_without_timed_children_has_no_estimate():
    registry = {}
    master = create_master_job(registry, "u1", "a1")
    document = create_child_job(
        registry,
        user_id="u1",
        assistant_id="a1",
        parent_id=master.job_id,
        filename="notes.md",
        namespace_filename="notes",
    )
    master.child_ids.append(document.job_id)
    assert job_estimated_processing_seconds(registry, master) is None
    assert job_estimated_media_seconds(registry, master) is None


def test_stamp_media_time_estimate_records_length_and_processing_time():
    import src.api.webapp as webapp

    entry = {"filename": "talk.mp4"}
    context = SimpleNamespace(media_preprocessing_seconds_per_media_second=1.0)
    webapp._stamp_media_time_estimate(entry, 2342.04, context)
    assert entry["estimated_media_seconds"] == 2342.0
    assert entry["estimated_processing_seconds"] == 2342.0

    untimed = {"filename": "notes.md"}
    webapp._stamp_media_time_estimate(untimed, None, context)
    assert untimed["estimated_media_seconds"] is None
    assert untimed["estimated_processing_seconds"] is None


# --------------------------------------------------------------------------- #
# Endpoint, snapshot, and progress stream carry the estimate
# --------------------------------------------------------------------------- #


def _upload_file(name: str, data: bytes, content_type: str) -> UploadFile:
    return UploadFile(
        file=io.BytesIO(data),
        filename=name,
        headers=Headers({"content-type": content_type}),
    )


@pytest.fixture
def timed_upload_environment(monkeypatch):
    """Stub the upload endpoint's collaborators; the estimator stamps a
    39-minute video length on every entry so the job carries a time estimate."""
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
            entry["estimated_media_seconds"] = 2342.0
            entry["estimated_processing_seconds"] = 2342.0
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


CURRENT_USER = {"identities": [{"user_id": "u1"}], "API_KEY": "k"}


async def _collect_sse_frames(response) -> list[dict]:
    frames: list[dict] = []
    async for chunk in response.body_iterator:
        text = chunk.decode() if isinstance(chunk, bytes) else chunk
        for line in text.split("\n"):
            if line.startswith("data:"):
                frames.append(json.loads(line[len("data:") :]))
    return frames


@pytest.mark.asyncio
async def test_upload_snapshot_and_progress_frames_carry_the_estimate(
    timed_upload_environment,
):
    webapp = timed_upload_environment

    response = await webapp.update_avatar_identity_with_media(
        files=[_upload_file("talk.md", b"Transcript-like prose.", "text/markdown")],
        assistant_id="a1",
        current_user=CURRENT_USER,
    )
    assert response.status_code == 202
    payload = json.loads(response.body)
    assert payload["estimated_media_seconds_total"] == 2342.0
    assert payload["estimated_processing_seconds_total"] == 2342.0
    assert payload["items"][0]["estimated_processing_seconds"] == 2342.0

    master_id = payload["job_id"]
    registry = webapp.app.state.media_jobs
    master = registry[master_id]
    child = registry[master.child_ids[0]]

    # The list view a Settings screen restores from.
    listing = await webapp.list_media_jobs(
        include_finished=True, assistant_id="a1", current_user=CURRENT_USER
    )
    assert listing["jobs"][0]["estimated_processing_seconds"] == 2342.0
    assert listing["jobs"][0]["estimated_media_seconds"] == 2342.0

    # The point-in-time snapshot, master and child.
    snapshot = await webapp.media_job_status(master_id, current_user=CURRENT_USER)
    assert snapshot["estimated_processing_seconds"] == 2342.0
    assert snapshot["children"][0]["estimated_processing_seconds"] == 2342.0

    # Every SSE frame carries the estimate and the seconds still expected.
    add_event(child, {"type": "media_progress", "stage": "converting"})
    finish_job(child, result={"items_processed": 1})
    stream = await webapp.media_job_progress(child.job_id, current_user=CURRENT_USER)
    frames = await _collect_sse_frames(stream)
    assert [frame["type"] for frame in frames] == ["status", "media_progress", "done"]
    for frame in frames:
        assert frame["estimated_media_seconds"] == 2342.0
        assert frame["estimated_processing_seconds"] == 2342.0
        assert 0.0 <= frame["estimated_remaining_seconds"] <= 2342.0
        assert frame["elapsed_seconds"] >= 0.0


@pytest.mark.asyncio
async def test_untimed_upload_has_no_estimate_on_frames(
    timed_upload_environment, monkeypatch
):
    webapp = timed_upload_environment

    async def _estimate_without_duration(entries):
        for entry in entries:
            entry["estimated_tokens"] = 1
        return len(entries)

    monkeypatch.setattr(
        webapp, "_estimate_media_entries_tokens", _estimate_without_duration
    )
    response = await webapp.update_avatar_identity_with_media(
        files=[_upload_file("notes.md", b"Plain notes.", "text/markdown")],
        assistant_id="a1",
        current_user=CURRENT_USER,
    )
    payload = json.loads(response.body)
    assert payload["estimated_processing_seconds_total"] is None
    master = webapp.app.state.media_jobs[payload["job_id"]]
    child = webapp.app.state.media_jobs[master.child_ids[0]]
    finish_job(child, result={"items_processed": 1})
    stream = await webapp.media_job_progress(child.job_id, current_user=CURRENT_USER)
    frames = await _collect_sse_frames(stream)
    assert all(frame["estimated_processing_seconds"] is None for frame in frames)
    assert all(frame["estimated_remaining_seconds"] is None for frame in frames)
