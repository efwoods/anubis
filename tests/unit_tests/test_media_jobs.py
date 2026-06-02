"""Unit tests for the background media-job registry and YouTube playlist routing.

Covers the deterministic, offline pieces:

* ``run_media_job`` — forwards ``media_progress`` custom events into the job
  buffer and records the final result; failures land on the job, not raised.
* ``MediaJob`` registry helpers — create/get and finish semantics.
* ``_classify_url`` — playlist vs single-video YouTube routing.

The ``process_media`` graph's ``astream`` is replaced with a fake async stream
so these tests need no model, store, or network.
"""

import pytest

import src.subgraphs.process_media_graph.process_media_graph_api_endpoint as pme
from src.anubis.utils.classes.URLDocumentLoaderClass import _classify_url
from src.api.media_jobs import (
    MediaJob,
    create_job,
    finish_job,
    get_job,
    run_media_job,
)


class _FakeCompiled:
    """Stand-in for the compiled graph; emits two progress events and one update.

    ``failed_files`` lets a test inject an ``index_docs``-style
    ``failed_to_index_files`` update so the failed-file capture path is covered.
    """

    def __init__(
        self,
        *,
        raise_exc: Exception | None = None,
        failed_files: list[dict] | None = None,
    ):
        self._raise_exc = raise_exc
        self._failed_files = failed_files

    async def astream(self, _input, *, config, context, stream_mode, subgraphs):
        yield (("ns",), "custom", {"type": "media_progress", "stage": "labeling", "total": 1})
        yield (("ns",), "updates", {"some_node": {}})
        if self._failed_files is not None:
            yield (
                ("ns",),
                "updates",
                {"index_docs": {"failed_to_index_files": self._failed_files}},
            )
        if self._raise_exc is not None:
            raise self._raise_exc
        yield (
            ("ns",),
            "custom",
            {"type": "media_progress", "stage": "indexing", "current": 1, "total": 1},
        )


class _FakeWorkflow:
    def __init__(self, compiled: _FakeCompiled):
        self._compiled = compiled

    def compile(self, store=None):
        return self._compiled


@pytest.mark.asyncio
async def test_run_media_job_records_progress_and_result(monkeypatch):
    monkeypatch.setattr(pme, "workflow", _FakeWorkflow(_FakeCompiled()))

    registry: dict[str, MediaJob] = {}
    job = create_job(registry, user_id="u1", assistant_id="a1")
    media_files = [{"filename": "clip.mp3"}]

    await run_media_job(job, media_files, config={}, store=None, context=None)

    assert job.status == "completed"
    assert job.done.is_set()
    assert job.result == {
        "items_processed": 1,
        "filenames": ["clip.mp3"],
        "indexed_filenames": ["clip.mp3"],
        "failed_files": [],
        "message": "Media processed and indexed successfully",
    }
    stages = [e["stage"] for e in job.events]
    assert stages == ["labeling", "indexing"]  # only media_progress events buffered


@pytest.mark.asyncio
async def test_run_media_job_surfaces_failed_files(monkeypatch):
    """Failed-file logic must survive the move to a background job: an
    ``index_docs`` ``failed_to_index_files`` update is captured onto the job
    result so the SSE ``done`` event reports the failure instead of a silent
    success (the bug the failed-file logic originally fixed)."""
    failed = [{"filename": "bad.pdf", "error": "indexing error", "document_ids": ["d1"]}]
    monkeypatch.setattr(
        pme, "workflow", _FakeWorkflow(_FakeCompiled(failed_files=failed))
    )

    registry: dict[str, MediaJob] = {}
    job = create_job(registry, user_id="u1", assistant_id="a1")
    media_files = [{"filename": "good.mp3"}, {"filename": "bad.pdf"}]

    await run_media_job(job, media_files, config={}, store=None, context=None)

    assert job.status == "completed"  # the job itself succeeded; some files did not
    assert job.result["failed_files"] == failed
    assert job.result["indexed_filenames"] == ["good.mp3"]  # bad.pdf excluded
    assert job.result["filenames"] == ["good.mp3", "bad.pdf"]
    assert "bad.pdf" in job.result["message"]


@pytest.mark.asyncio
async def test_run_media_job_captures_failure(monkeypatch):
    boom = RuntimeError("conversion blew up")
    monkeypatch.setattr(pme, "workflow", _FakeWorkflow(_FakeCompiled(raise_exc=boom)))

    registry: dict[str, MediaJob] = {}
    job = create_job(registry, user_id="u1", assistant_id="a1")

    # Must not raise — failures surface via the job, not the caller.
    await run_media_job(job, [{"filename": "x"}], config={}, store=None, context=None)

    assert job.status == "error"
    assert job.done.is_set()
    assert "conversion blew up" in (job.error or "")


def test_registry_create_get_and_finish():
    registry: dict[str, MediaJob] = {}
    job = create_job(registry, user_id="u1", assistant_id="a1")

    assert get_job(registry, job.job_id) is job
    assert get_job(registry, "missing") is None

    finish_job(job, result={"ok": True})
    assert job.status == "completed"
    assert job.done.is_set()


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.youtube.com/playlist?list=PL123", "youtube_playlist"),
        ("https://youtube.com/playlist?list=PLabc", "youtube_playlist"),
        ("https://www.youtube.com/watch?v=abc123", "youtube"),
        # watch?v=..&list=.. is a single video, not a playlist expansion.
        ("https://www.youtube.com/watch?v=abc123&list=PL123", "youtube"),
        ("https://youtu.be/abc123", "youtube"),
    ],
)
def test_classify_url_youtube_playlist(url, expected):
    assert _classify_url(url) == expected
