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

import asyncio

import pytest

import src.subgraphs.process_media_graph.process_media_graph_api_endpoint as pme
from src.anubis.utils.classes.URLDocumentLoaderClass import _classify_url
from src.api.media_jobs import (
    MediaJob,
    _calibrate_ground_truth_after_batch,
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
        yield (
            ("ns",),
            "custom",
            {"type": "media_progress", "stage": "labeling", "total": 1},
        )
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


class _FakeQuoteCompiled(_FakeCompiled):
    """A graph run that also reports how many Documents it indexed per namespace.

    This is what tells the batch orchestrator whether an upload actually added
    direct quotes, and therefore whether the avatar's direct-quote cloud is worth
    refitting.
    """

    def __init__(self, namespace_counts: dict, **kwargs):
        super().__init__(**kwargs)
        self._namespace_counts = namespace_counts

    async def astream(self, _input, *, config, context, stream_mode, subgraphs):
        async for item in super().astream(
            _input,
            config=config,
            context=context,
            stream_mode=stream_mode,
            subgraphs=subgraphs,
        ):
            yield item
        yield (
            ("ns",),
            "custom",
            {
                "type": "media_progress",
                "stage": "indexed_namespace_counts",
                "namespace_counts": self._namespace_counts,
            },
        )


class _RecordingCalibration:
    """Records every direct-quote recalibration the batch orchestrator requests."""

    def __init__(self, raise_exc: Exception | None = None, delay_seconds: float = 0.0):
        self.calls: list[dict] = []
        self._raise_exc = raise_exc
        self._delay_seconds = delay_seconds

    async def __call__(self, *, store, assistant_id, user_id):
        self.calls.append(
            {"store": store, "assistant_id": assistant_id, "user_id": user_id}
        )
        if self._delay_seconds:
            await asyncio.sleep(self._delay_seconds)
        if self._raise_exc is not None:
            raise self._raise_exc


def _batch_of(registry, master, filenames):
    """Build one child job + item spec per filename under an existing master."""
    items = []
    for name in filenames:
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
    return items


def _patch_calibration(monkeypatch, recorder):
    """Point the batch hook's lazily-imported calibration at ``recorder``."""
    import src.subgraphs.process_media_graph.utils.calibrate_ground_truth as cgt

    monkeypatch.setattr(
        cgt, "calibrate_ground_truth_from_stored_corpus", recorder, raising=True
    )


@pytest.mark.asyncio
async def test_batch_calibrates_direct_quote_cloud_exactly_once(monkeypatch):
    """Eighteen videos must produce ONE refit, not eighteen.

    ``calibrate_ground_truth`` rereads and refits the avatar's entire quote corpus
    on every call, at a cost quadratic in corpus size, so a per-item hook would
    redo the whole fit once per file and discard all but the last result.
    """
    monkeypatch.setattr(
        pme, "workflow", _FakeWorkflow(_FakeQuoteCompiled({"quote": 40}))
    )
    recorder = _RecordingCalibration()
    _patch_calibration(monkeypatch, recorder)

    registry: dict[str, MediaJob] = {}
    master = create_master_job(registry, user_id="u1", assistant_id="a1")
    items = _batch_of(registry, master, ("a.mp4", "b.mp4", "c.mp4"))
    sentinel_store = object()

    await run_batch_media_job(
        master,
        items,
        config={"configurable": {}},
        store=sentinel_store,
        context=None,
        concurrency=5,
    )

    assert len(recorder.calls) == 1
    assert recorder.calls[0]["assistant_id"] == "a1"
    assert recorder.calls[0]["user_id"] == "u1"
    assert recorder.calls[0]["store"] is sentinel_store
    # Every child's contribution is accumulated onto the master.
    assert master.indexed_quote_document_count == 120
    assert master.status == "completed"
    stages = [event.get("stage") for event in master.events]
    assert "calibrating" in stages and "calibrating_complete" in stages


@pytest.mark.asyncio
async def test_batch_without_quotes_skips_calibration(monkeypatch):
    """An upload that added no direct quotes must not pay for a refit.

    A PDF or a biography produces identity Documents only; refitting the
    direct-quote cloud would reproduce exactly the previous fit.
    """
    monkeypatch.setattr(
        pme,
        "workflow",
        _FakeWorkflow(_FakeQuoteCompiled({"identity": 12, "document": 3})),
    )
    recorder = _RecordingCalibration()
    _patch_calibration(monkeypatch, recorder)

    registry: dict[str, MediaJob] = {}
    master = create_master_job(registry, user_id="u1", assistant_id="a1")
    items = _batch_of(registry, master, ("bio.pdf",))

    await run_batch_media_job(
        master,
        items,
        config={"configurable": {}},
        store=object(),
        context=None,
        concurrency=5,
    )

    assert recorder.calls == []
    assert master.indexed_quote_document_count == 0
    assert master.status == "completed"


@pytest.mark.asyncio
async def test_calibration_failure_never_fails_the_upload(monkeypatch):
    """The documents are already indexed; a failed refit must not undo that."""
    monkeypatch.setattr(
        pme, "workflow", _FakeWorkflow(_FakeQuoteCompiled({"quote": 5}))
    )
    recorder = _RecordingCalibration(raise_exc=RuntimeError("singular covariance"))
    _patch_calibration(monkeypatch, recorder)

    registry: dict[str, MediaJob] = {}
    master = create_master_job(registry, user_id="u1", assistant_id="a1")
    items = _batch_of(registry, master, ("clip.mp3",))

    await run_batch_media_job(
        master,
        items,
        config={"configurable": {}},
        store=object(),
        context=None,
        concurrency=5,
    )

    assert len(recorder.calls) == 1
    assert master.status == "completed"
    assert master.result["items_completed"] == 1
    stages = [event.get("stage") for event in master.events]
    assert "calibrating_skipped" in stages
    assert "calibrating_complete" not in stages


@pytest.mark.asyncio
async def test_calibration_timeout_still_completes_the_upload(monkeypatch):
    """A pathological corpus must not wedge the batch's terminal event."""

    class _ImpatientContext:
        ground_truth_calibration_timeout_seconds = 0.01

    monkeypatch.setattr(
        pme, "workflow", _FakeWorkflow(_FakeQuoteCompiled({"quote": 5}))
    )
    recorder = _RecordingCalibration(delay_seconds=5.0)
    _patch_calibration(monkeypatch, recorder)

    registry: dict[str, MediaJob] = {}
    master = create_master_job(registry, user_id="u1", assistant_id="a1")
    items = _batch_of(registry, master, ("clip.mp3",))

    await run_batch_media_job(
        master,
        items,
        config={"configurable": {}},
        store=object(),
        context=_ImpatientContext(),
        concurrency=5,
    )

    assert master.status == "completed"
    stages = [event.get("stage") for event in master.events]
    assert "calibrating_skipped" in stages


@pytest.mark.asyncio
async def test_cancelled_batch_never_refits_the_quote_cloud(monkeypatch):
    """A cancelled batch must not fit a cloud over rows that are being rolled back.

    Cancelling a batch rolls back everything its children indexed. Fitting after
    that would bake documents into the direct-quote cloud that are about to
    disappear from the store, leaving a cloud describing a corpus the avatar no
    longer has — and unlike a skipped fit, nothing later recomputes it.

    The hook is called directly rather than through ``run_batch_media_job``
    because this guard is only distinguishable when quotes DID index before the
    cancel: a cancelled batch that reached no children would also be skipped by
    the ``indexed_quote_document_count <= 0`` gate, so routing through the
    orchestrator could not tell the two reasons apart.
    """
    recorder = _RecordingCalibration()
    _patch_calibration(monkeypatch, recorder)

    registry: dict[str, MediaJob] = {}
    master = create_master_job(registry, user_id="u1", assistant_id="a1")
    # Quotes indexed before the cancel landed, so the "this batch added no
    # quotes" gate cannot be what refuses the refit here.
    master.indexed_quote_document_count = 40
    master.cancelled = True

    await _calibrate_ground_truth_after_batch(master, object(), None)

    assert recorder.calls == []
    # Skipped before any stage is emitted: a cancelled batch reports the cancel,
    # not a calibration that never started.
    stages = [event.get("stage") for event in master.events]
    assert "calibrating" not in stages
    assert "calibrating_skipped" not in stages

    # ``force`` is what the explicit recalibration endpoint passes; the cancel
    # guard is checked ahead of it, so an in-flight cancel still wins.
    await _calibrate_ground_truth_after_batch(master, object(), None, force=True)
    assert recorder.calls == []


class _FakeErrorCompiled:
    """Emits an item_error + converting_complete(indexed=0): a total item failure
    the graph swallows (partial success) rather than raising."""

    async def astream(self, _input, *, config, context, stream_mode, subgraphs):
        yield (
            ("ns",),
            "custom",
            {"type": "media_progress", "stage": "converting", "current": 1, "total": 1},
        )
        yield (
            ("ns",),
            "custom",
            {
                "type": "media_progress",
                "stage": "item_error",
                "filename": "v.mp4",
                "error": "missing_audio_reference_audio",
            },
        )
        yield (
            ("ns",),
            "custom",
            {
                "type": "media_progress",
                "stage": "converting_complete",
                "total": 1,
                "skipped": 0,
                "errors": 1,
                "indexed": 0,
            },
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
        child,
        master,
        {"filename": "clip.mp3"},
        config={"configurable": {}},
        store=None,
        context=None,
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
        child,
        master,
        {"filename": "x"},
        config={"configurable": {}},
        store=None,
        context=None,
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
        child,
        master,
        {"filename": "v.mp4"},
        config={"configurable": {}},
        store=None,
        context=None,
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
        master,
        items,
        config={"configurable": {}},
        store=None,
        context=None,
        concurrency=5,
    )

    assert master.status == "completed"
    assert master.done.is_set()
    assert master.result["items_total"] == 2
    assert master.result["items_completed"] == 2
    assert all(spec["child"].status == "completed" for spec in items)


@pytest.mark.asyncio
async def test_run_batch_media_job_expands_deferred_playlist(monkeypatch):
    """A deferred expander (playlist enumeration) mints one child job per video
    under the master, in the background, and runs them alongside ready items."""
    monkeypatch.setattr(pme, "workflow", _FakeWorkflow(_FakeCompiled()))

    registry: dict[str, MediaJob] = {}
    master = create_master_job(registry, user_id="u1", assistant_id="a1")

    # One ready (non-playlist) item plus a deferred expander yielding two videos.
    ready_child = create_child_job(
        registry,
        user_id="u1",
        assistant_id="a1",
        parent_id=master.job_id,
        filename="ready.mp3",
        namespace_filename="ns::ready",
    )
    master.child_ids.append(ready_child.job_id)
    items = [{"child": ready_child, "media_file": {"filename": "ready.mp3"}}]

    async def _expander():
        return [
            {
                "filename": "PL::Episode A",
                "namespace_filename": "PLNS::AAA",
                "page_url": "https://www.youtube.com/watch?v=aaa",
            },
            {
                "filename": "PL::Episode B",
                "namespace_filename": "PLNS::BBB",
                "page_url": "https://www.youtube.com/watch?v=bbb",
            },
        ]

    await run_batch_media_job(
        master,
        items,
        config={"configurable": {}},
        store=None,
        context=None,
        concurrency=5,
        registry=registry,
        deferred_expanders=[_expander],
    )

    # Master saw the ready item + two enumerated videos = three children.
    assert master.status == "completed"
    assert master.result["items_total"] == 3
    assert master.result["items_completed"] == 3
    # Two fresh child jobs were registered with the composite playlist keys.
    composite_children = [
        j
        for j in registry.values()
        if j.namespace_filename in ("PLNS::AAA", "PLNS::BBB")
    ]
    assert len(composite_children) == 2
    assert all(c.parent_id == master.job_id for c in composite_children)
    assert all(c.job_id in master.child_ids for c in composite_children)
    # The master stream announced each enumerated video.
    added = [e for e in master.events if e.get("stage") == "playlist_child_added"]
    assert sorted(e["item_filename"] for e in added) == [
        "PL::Episode A",
        "PL::Episode B",
    ]


@pytest.mark.asyncio
async def test_deferred_expander_failure_is_isolated(monkeypatch):
    """A failing expander surfaces an item_error on the master but doesn't abort
    the batch — ready items still complete."""
    monkeypatch.setattr(pme, "workflow", _FakeWorkflow(_FakeCompiled()))

    registry: dict[str, MediaJob] = {}
    master = create_master_job(registry, user_id="u1", assistant_id="a1")
    ready_child = create_child_job(
        registry,
        user_id="u1",
        assistant_id="a1",
        parent_id=master.job_id,
        filename="ready.mp3",
        namespace_filename="ns::ready",
    )
    master.child_ids.append(ready_child.job_id)
    items = [{"child": ready_child, "media_file": {"filename": "ready.mp3"}}]

    async def _boom():
        raise RuntimeError("yt_dlp exploded")

    await run_batch_media_job(
        master,
        items,
        config={"configurable": {}},
        store=None,
        context=None,
        concurrency=5,
        registry=registry,
        deferred_expanders=[_boom],
    )

    assert master.status == "completed"
    assert master.result["items_total"] == 1
    assert ready_child.status == "completed"
    errors = [e for e in master.events if e.get("stage") == "item_error"]
    assert any("yt_dlp exploded" in (e.get("error") or "") for e in errors)


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


# --------------------------------------------------------------------------- #
# Playlist namespace: each video is keyed by a composite {playlist_ns}::{video_ns}
# so the namespace carries both the playlist and the individual video, and the
# content (subs/audio) inherits that composite key instead of collapsing to a
# video-only key.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_playlist_entries_get_composite_namespace(monkeypatch):
    """_load_youtube_playlist keys each video by {playlist_ns}::{video_ns} and
    stamps playlist context (url / ns / title) onto every entry."""
    import src.anubis.utils.classes.URLDocumentLoaderClass as loader_mod
    from src.anubis.utils.classes.URLDocumentLoaderClass import (
        URLDocumentLoaderClass,
        _namespace_for,
    )

    playlist_url = "https://www.youtube.com/playlist?list=PLxyz"
    watch_a = "https://www.youtube.com/watch?v=aaa"
    watch_b = "https://www.youtube.com/watch?v=bbb"

    async def _fake_entries(url):
        return (
            [
                {"id": "aaa", "url": watch_a, "title": "Episode A"},
                {"id": "bbb", "url": watch_b, "title": "Episode B"},
            ],
            "My Playlist",
        )

    monkeypatch.setattr(loader_mod, "_extract_playlist_entries", _fake_entries)

    items = await URLDocumentLoaderClass()._load_youtube_playlist(
        playlist_url, user_id="u", assistant_id="a"
    )

    playlist_ns = _namespace_for(playlist_url)
    assert [i["metadata"]["namespace_filename"] for i in items] == [
        f"{playlist_ns}::{_namespace_for(watch_a)}",
        f"{playlist_ns}::{_namespace_for(watch_b)}",
    ]
    for item, title in zip(items, ("Episode A", "Episode B")):
        meta = item["metadata"]
        assert meta["playlist_url"] == playlist_url
        assert meta["playlist_namespace_filename"] == playlist_ns
        assert meta["playlist_title"] == "My Playlist"
        assert meta["video_title"] == title


@pytest.mark.asyncio
async def test_expand_keyless_child_inherits_parent_namespace(monkeypatch):
    """A playlist entry's keyless content (subs/audio) inherits the composite
    namespace and gets playlist context stamped onto the produced Documents."""
    from langchain_core.documents import Document

    import src.subgraphs.process_media_graph.utils.nodes as nodes_mod

    playlist_ns = "PLNS"
    composite = f"{playlist_ns}::VIDNS"
    parent_item = {
        "type": "url",
        "url": "https://www.youtube.com/watch?v=aaa",
        "metadata": {
            "filename": "https://www.youtube.com/watch?v=aaa",
            "namespace_filename": composite,
            "playlist_url": "https://www.youtube.com/playlist?list=PLxyz",
            "playlist_namespace_filename": playlist_ns,
            "playlist_title": "My Playlist",
            "video_title": "Episode A",
        },
    }

    # Loader returns a single keyless child (the video's transcript text).
    class _FakeLoader:
        async def load(
            self, url, user_id=None, assistant_id=None, expect_multispeaker=False
        ):
            return [{"type": "text", "content": "hello", "metadata": {}}]

    monkeypatch.setattr(nodes_mod, "URLDocumentLoaderClass", _FakeLoader)

    captured = {}

    async def _fake_process(item, runtime, config, store, **kwargs):
        # The child must have inherited the parent's composite namespace_filename.
        captured["child_ns"] = item["metadata"].get("namespace_filename")
        return [
            Document(
                page_content="hello",
                metadata={
                    "namespace_filename": item["metadata"]["namespace_filename"],
                    "namespace": "quote",
                },
            )
        ]

    monkeypatch.setattr(nodes_mod, "process_media_item_task", _fake_process)

    docs = await nodes_mod._expand_url_media_item(
        parent_item,
        None,
        None,
        None,
        url=parent_item["url"],
        filename=parent_item["metadata"]["filename"],
        namespace_filename=composite,
        user_id="u",
        assistant_id="a",
    )

    assert captured["child_ns"] == composite
    assert len(docs) == 1
    meta = docs[0].metadata
    assert meta["namespace_filename"] == composite
    assert meta["playlist_url"] == "https://www.youtube.com/playlist?list=PLxyz"
    assert meta["playlist_namespace_filename"] == playlist_ns
    assert meta["playlist_title"] == "My Playlist"
    assert meta["video_title"] == "Episode A"


@pytest.mark.asyncio
async def test_create_reference_media_from_playlist_forces_multispeaker_and_inherits(
    monkeypatch,
):
    """A URL item flagged ``create_reference_media_from_playlist`` must call the loader with
    ``expect_multispeaker=True`` (forcing the audio/diarize path over subtitles)
    and stamp the flag onto every expanded child so playlist videos inherit it."""
    from langchain_core.documents import Document

    import src.subgraphs.process_media_graph.utils.nodes as nodes_mod

    parent_item = {
        "type": "url",
        "url": "https://www.youtube.com/watch?v=aaa",
        "metadata": {
            "filename": "https://www.youtube.com/watch?v=aaa",
            "namespace_filename": "NS",
            "create_reference_media_from_playlist": True,
        },
    }

    captured = {}

    class _FakeLoader:
        async def load(
            self, url, user_id=None, assistant_id=None, expect_multispeaker=False
        ):
            captured["expect_multispeaker"] = expect_multispeaker
            return [{"type": "audio", "content": "hi", "metadata": {}}]

    monkeypatch.setattr(nodes_mod, "URLDocumentLoaderClass", _FakeLoader)

    async def _fake_process(item, runtime, config, store, **kwargs):
        captured["child_create_reference_media_from_playlist"] = item["metadata"].get(
            "create_reference_media_from_playlist"
        )
        return [Document(page_content="hi", metadata={"namespace": "quote"})]

    monkeypatch.setattr(nodes_mod, "process_media_item_task", _fake_process)

    await nodes_mod._expand_url_media_item(
        parent_item,
        None,
        None,
        None,
        url=parent_item["url"],
        filename=parent_item["metadata"]["filename"],
        namespace_filename="NS",
        user_id="u",
        assistant_id="a",
    )

    assert captured["expect_multispeaker"] is True
    assert captured["child_create_reference_media_from_playlist"] is True


def _make_runtime():
    from types import SimpleNamespace

    return SimpleNamespace(
        context=SimpleNamespace(audio_diarization_known_speaker_name="avatar")
    )


def _audio_item():
    return {
        "type": "audio",
        "base64_encoded_str": "data:audio/mp3;base64,QUJD",
        "metadata": {
            "filename": "talk.mp3",
            "content_type": "audio/mp3",
            "user_id": "u",
            "assistant_id": "a",
            "namespace_filename": "NS",
            "reference_audio": False,
            "create_reference_media_from_playlist": True,
        },
    }


@pytest.mark.asyncio
async def test_create_reference_media_from_playlist_dialogue_reuses_preceding_statement(
    monkeypatch,
):
    """Multiple speakers (all target): one ``quote`` Document per statement, each
    later turn reusing the PRECEDING statement as ``adapter_prompt`` (the genuine
    question); the first statement has no predecessor (synthesized downstream).
    No monologue/adapter-conversation doc is produced for multi-speaker content,
    and the standard dialogue document builder is never run."""
    import src.subgraphs.process_media_graph.utils.nodes as nodes_mod

    captured = {}

    async def _fake_diarize(
        *, media_base64, context, encoded_reference_audio, filename, content_type
    ):
        captured["encoded_reference_audio"] = encoded_reference_audio
        return {
            "text": "Q1 A1 Q2",
            "segments": [
                {"speaker": "A", "text": "How are you?", "start": 0, "end": 1},
                {"speaker": "B", "text": "I am well.", "start": 1, "end": 2},
                {"speaker": "A", "text": "Glad to hear.", "start": 2, "end": 3},
            ],
        }

    async def _fail_dialogue(*args, **kwargs):  # must NOT be called
        raise AssertionError(
            "dialogue path must not run in create_reference_media_from_playlist mode"
        )

    monkeypatch.setattr(nodes_mod, "transcribe_audio_diarize", _fake_diarize)
    monkeypatch.setattr(nodes_mod, "process_dialogue_json_to_documents", _fail_dialogue)

    docs = await nodes_mod.process_media_item_task(
        _audio_item(), _make_runtime(), {}, store=None
    )

    # Diarized without a known-speaker reference clip.
    assert captured["encoded_reference_audio"] is None

    # Only per-statement quote docs (no adapter-conversation/monologue doc).
    assert all(d.metadata.get("namespace") == "quote" for d in docs)
    assert [d.page_content for d in docs] == [
        "How are you?",
        "I am well.",
        "Glad to hear.",
    ]
    assert all(d.metadata.get("adapter_acceptable") is True for d in docs)
    # First statement has no predecessor (-> synthesized); later statements
    # reuse the immediately preceding turn as the question.
    assert docs[0].metadata.get("adapter_prompt") is None
    assert docs[1].metadata.get("adapter_prompt") == "How are you?"
    assert docs[2].metadata.get("adapter_prompt") == "I am well."


@pytest.mark.asyncio
async def test_create_reference_media_from_playlist_single_speaker_classified_normally(
    monkeypatch,
):
    """A single speaker is classified normally (monologue / tweets_or_quotes):
    the full transcript is routed through ``process_text_to_document`` rather than
    the per-statement / dialogue paths."""
    from langchain_core.documents import Document

    import src.subgraphs.process_media_graph.utils.nodes as nodes_mod

    captured = {}

    async def _fake_diarize(
        *, media_base64, context, encoded_reference_audio, filename, content_type
    ):
        return {
            "text": "Statement one. Statement two.",
            "segments": [
                {"speaker": "S", "text": "Statement one.", "start": 0, "end": 1},
                {"speaker": "S", "text": "Statement two.", "start": 1, "end": 2},
            ],
        }

    async def _fail_dialogue(*args, **kwargs):
        raise AssertionError(
            "dialogue path must not run in create_reference_media_from_playlist mode"
        )

    async def _fake_classify(*, metadata, user_id, assistant_id, media_item):
        captured["classify_text"] = media_item.get("content")
        return [Document(page_content="chunk", metadata={"namespace": "quote"})]

    monkeypatch.setattr(nodes_mod, "transcribe_audio_diarize", _fake_diarize)
    monkeypatch.setattr(nodes_mod, "process_dialogue_json_to_documents", _fail_dialogue)
    monkeypatch.setattr(nodes_mod, "process_text_to_document", _fake_classify)

    docs = await nodes_mod.process_media_item_task(
        _audio_item(), _make_runtime(), {}, store=None
    )

    # The whole single-speaker transcript was sent to the normal classifier.
    assert captured["classify_text"] == "Statement one.\nStatement two."
    assert len(docs) == 1
    assert docs[0].metadata.get("namespace") == "quote"


# --------------------------------------------------------------------------- #
# Endpoint-level playlist expansion: a playlist URL is exploded into one
# upload entry per video BEFORE child jobs are created, so each video gets its
# own processing id and lists as ``{playlist}::{video}``.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_expand_youtube_playlist_to_media_entries(monkeypatch):
    """A playlist URL becomes one page_url entry per video, each keyed by the
    composite {playlist_ns}::{video_ns}, named {playlist}::{video}, and carrying
    playlist context so downstream Documents group under the playlist."""
    import src.anubis.utils.classes.URLDocumentLoaderClass as loader_mod
    from src.api.webapp import (
        _expand_youtube_playlist_to_media_entries,
        _namespace_safe_formatted_filename,
    )

    playlist_url = "https://www.youtube.com/playlist?list=PLxyz"
    watch_a = "https://www.youtube.com/watch?v=aaa"
    watch_b = "https://www.youtube.com/watch?v=bbb"

    async def _fake_entries(url):
        return (
            [
                {"id": "aaa", "url": watch_a, "title": "Episode A"},
                {"id": "bbb", "url": watch_b, "title": "Episode B"},
            ],
            "My Playlist",
        )

    monkeypatch.setattr(loader_mod, "_extract_playlist_entries", _fake_entries)

    entries = await _expand_youtube_playlist_to_media_entries(
        playlist_url, user_id="u", assistant_id="a"
    )

    playlist_ns = _namespace_safe_formatted_filename(playlist_url)
    assert [e["namespace_filename"] for e in entries] == [
        f"{playlist_ns}::{_namespace_safe_formatted_filename(watch_a)}",
        f"{playlist_ns}::{_namespace_safe_formatted_filename(watch_b)}",
    ]
    assert [e["filename"] for e in entries] == [
        "My Playlist::Episode A",
        "My Playlist::Episode B",
    ]
    for entry, watch, title in (
        (entries[0], watch_a, "Episode A"),
        (entries[1], watch_b, "Episode B"),
    ):
        assert entry["page_url"] == watch
        assert entry["playlist_url"] == playlist_url
        assert entry["playlist_namespace_filename"] == playlist_ns
        assert entry["playlist_title"] == "My Playlist"
        assert entry["video_title"] == title
        assert entry["url_kind"] == "youtube_playlist_entry"


@pytest.mark.asyncio
async def test_expand_non_playlist_url_returns_none(monkeypatch):
    """A non-playlist URL yields ``None`` so the caller takes the normal path."""
    from src.api.webapp import _expand_youtube_playlist_to_media_entries

    result = await _expand_youtube_playlist_to_media_entries(
        "https://www.youtube.com/watch?v=abc123", user_id="u", assistant_id="a"
    )
    assert result is None
