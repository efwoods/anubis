"""Unit tests for the background media-job registry and YouTube playlist routing.

Covers the deterministic, offline pieces:

* ``run_single_item_job`` — forwards ``media_progress`` custom events into both the
  child's buffer and the master's aggregate buffer, and records the final result;
  failures land on the job, not raised.
* ``run_batch_media_job`` — runs one child per item under a shared limiter and
  aggregates per-item status into the master.
* ``MediaJob`` registry helpers — create master/child, get, finish, and cancel.
* ``_classify_url`` — playlist vs single-video YouTube routing.

The ``process_media`` graph's ``astream`` is replaced with a fake async stream
so these tests need no model, store, or network.
"""

import pytest

import src.subgraphs.process_media_graph.process_media_graph_api_endpoint as pme
from src.anubis.utils.classes.URLDocumentLoaderClass import _classify_url
from src.api.media_jobs import (
    MediaJob,
    create_child_job,
    create_master_job,
    finish_job,
    get_job,
    request_cancel,
    run_batch_media_job,
    run_single_item_job,
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


class _FakeErrorCompiled:
    """Emits an item_error + converting_complete(indexed=0): a total item failure
    the graph swallows (partial success) rather than raising."""

    async def astream(self, _input, *, config, context, stream_mode, subgraphs):
        yield (("ns",), "custom", {"type": "media_progress", "stage": "converting", "current": 1, "total": 1})
        yield (
            ("ns",),
            "custom",
            {"type": "media_progress", "stage": "item_error", "filename": "v.mp4", "error": "missing_audio_reference_audio"},
        )
        yield (
            ("ns",),
            "custom",
            {"type": "media_progress", "stage": "converting_complete", "total": 1, "skipped": 0, "errors": 1, "indexed": 0},
        )


class _FakeWorkflow:
    def __init__(self, compiled):
        self._compiled = compiled

    def compile(self, store=None):
        return self._compiled


def _master_and_child(registry, filename="clip.mp3"):
    master = create_master_job(registry, user_id="u1", assistant_id="a1")
    child = create_child_job(
        registry,
        user_id="u1",
        assistant_id="a1",
        parent_id=master.job_id,
        filename=filename,
        namespace_filename=f"ns::{filename}",
    )
    master.child_ids.append(child.job_id)
    return master, child


@pytest.mark.asyncio
async def test_run_single_item_job_records_progress_and_result(monkeypatch):
    monkeypatch.setattr(pme, "workflow", _FakeWorkflow(_FakeCompiled()))

    registry: dict[str, MediaJob] = {}
    master, child = _master_and_child(registry)

    await run_single_item_job(
        child, master, {"filename": "clip.mp3"}, config={"configurable": {}},
        store=None, context=None,
    )

    assert child.status == "completed"
    assert child.done.is_set()
    assert child.result["filename"] == "clip.mp3"
    # Only media_progress events are buffered, in order, on the child...
    assert [e["stage"] for e in child.events] == ["labeling", "indexing"]
    # ...and mirrored onto the master, attributed to the item.
    assert [e["stage"] for e in master.events] == ["labeling", "indexing"]
    assert all(e["item_job_id"] == child.job_id for e in master.events)


@pytest.mark.asyncio
async def test_run_single_item_job_captures_failure(monkeypatch):
    boom = RuntimeError("conversion blew up")
    monkeypatch.setattr(pme, "workflow", _FakeWorkflow(_FakeCompiled(raise_exc=boom)))

    registry: dict[str, MediaJob] = {}
    master, child = _master_and_child(registry, filename="x")

    # Must not raise — failures surface via the job, not the caller.
    await run_single_item_job(
        child, master, {"filename": "x"}, config={"configurable": {}},
        store=None, context=None,
    )

    assert child.status == "error"
    assert child.done.is_set()
    assert "conversion blew up" in (child.error or "")


@pytest.mark.asyncio
async def test_run_single_item_job_marks_swallowed_error(monkeypatch):
    # The graph reports a total item failure via item_error / converting_complete
    # (indexed=0) without raising; the child must end as "error", not "completed".
    monkeypatch.setattr(pme, "workflow", _FakeWorkflow(_FakeErrorCompiled()))

    registry: dict[str, MediaJob] = {}
    master, child = _master_and_child(registry, filename="v.mp4")

    await run_single_item_job(
        child, master, {"filename": "v.mp4"}, config={"configurable": {}},
        store=None, context=None,
    )

    assert child.status == "error"
    assert "missing_audio_reference_audio" in (child.error or "")


@pytest.mark.asyncio
async def test_run_batch_media_job_aggregates_children(monkeypatch):
    monkeypatch.setattr(pme, "workflow", _FakeWorkflow(_FakeCompiled()))

    registry: dict[str, MediaJob] = {}
    master = create_master_job(registry, user_id="u1", assistant_id="a1")
    items = []
    for name in ("a.mp3", "b.mp3"):
        child = create_child_job(
            registry,
            user_id="u1",
            assistant_id="a1",
            parent_id=master.job_id,
            filename=name,
            namespace_filename=f"ns::{name}",
        )
        master.child_ids.append(child.job_id)
        items.append({"child": child, "media_file": {"filename": name}})

    await run_batch_media_job(
        master, items, config={"configurable": {}}, store=None, context=None,
        concurrency=5,
    )

    assert master.status == "completed"
    assert master.done.is_set()
    assert master.result["items_total"] == 2
    assert master.result["items_completed"] == 2
    assert all(spec["child"].status == "completed" for spec in items)


def test_registry_create_get_finish_and_cancel():
    registry: dict[str, MediaJob] = {}
    master, child = _master_and_child(registry)

    assert get_job(registry, master.job_id) is master
    assert get_job(registry, child.job_id) is child
    assert get_job(registry, "missing") is None

    # Cancelling the master flags the master and returns its child(ren).
    targets = request_cancel(registry, master)
    assert master.cancelled is True
    assert [t.job_id for t in targets] == [child.job_id]
    assert child.cancelled is True

    finish_job(child, cancelled=True)
    assert child.status == "cancelled"
    assert child.done.is_set()
    # finish_job is idempotent once a job is done.
    finish_job(child, result={"ok": True})
    assert child.status == "cancelled"


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
