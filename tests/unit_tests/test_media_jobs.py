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
            {"filename": "PL::Episode A", "namespace_filename": "PLNS::AAA",
             "page_url": "https://www.youtube.com/watch?v=aaa"},
            {"filename": "PL::Episode B", "namespace_filename": "PLNS::BBB",
             "page_url": "https://www.youtube.com/watch?v=bbb"},
        ]

    await run_batch_media_job(
        master, items, config={"configurable": {}}, store=None, context=None,
        concurrency=5, registry=registry, deferred_expanders=[_expander],
    )

    # Master saw the ready item + two enumerated videos = three children.
    assert master.status == "completed"
    assert master.result["items_total"] == 3
    assert master.result["items_completed"] == 3
    # Two fresh child jobs were registered with the composite playlist keys.
    composite_children = [
        j for j in registry.values()
        if j.namespace_filename in ("PLNS::AAA", "PLNS::BBB")
    ]
    assert len(composite_children) == 2
    assert all(c.parent_id == master.job_id for c in composite_children)
    assert all(c.job_id in master.child_ids for c in composite_children)
    # The master stream announced each enumerated video.
    added = [e for e in master.events if e.get("stage") == "playlist_child_added"]
    assert sorted(e["item_filename"] for e in added) == [
        "PL::Episode A", "PL::Episode B",
    ]


@pytest.mark.asyncio
async def test_deferred_expander_failure_is_isolated(monkeypatch):
    """A failing expander surfaces an item_error on the master but doesn't abort
    the batch — ready items still complete."""
    monkeypatch.setattr(pme, "workflow", _FakeWorkflow(_FakeCompiled()))

    registry: dict[str, MediaJob] = {}
    master = create_master_job(registry, user_id="u1", assistant_id="a1")
    ready_child = create_child_job(
        registry, user_id="u1", assistant_id="a1", parent_id=master.job_id,
        filename="ready.mp3", namespace_filename="ns::ready",
    )
    master.child_ids.append(ready_child.job_id)
    items = [{"child": ready_child, "media_file": {"filename": "ready.mp3"}}]

    async def _boom():
        raise RuntimeError("yt_dlp exploded")

    await run_batch_media_job(
        master, items, config={"configurable": {}}, store=None, context=None,
        concurrency=5, registry=registry, deferred_expanders=[_boom],
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
        return [Document(page_content="hello", metadata={"namespace_filename": item["metadata"]["namespace_filename"], "namespace": "quote"})]

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
async def test_all_speakers_target_forces_multispeaker_and_inherits(monkeypatch):
    """A URL item flagged ``all_speakers_target`` must call the loader with
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
            "all_speakers_target": True,
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
        captured["child_all_speakers_target"] = item["metadata"].get(
            "all_speakers_target"
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
    assert captured["child_all_speakers_target"] is True


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
            "all_speakers_target": True,
        },
    }


@pytest.mark.asyncio
async def test_all_speakers_target_dialogue_reuses_preceding_statement(monkeypatch):
    """Multiple speakers (all target): one ``quote`` Document per statement, each
    later turn reusing the PRECEDING statement as ``adapter_prompt`` (the genuine
    question); the first statement has no predecessor (synthesized downstream).
    No monologue/adapter-conversation doc is produced for multi-speaker content,
    and the standard dialogue document builder is never run."""
    import src.subgraphs.process_media_graph.utils.nodes as nodes_mod

    captured = {}

    async def _fake_diarize(*, media_base64, context, encoded_reference_audio,
                            filename, content_type):
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
        raise AssertionError("dialogue path must not run in all_speakers_target mode")

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
async def test_all_speakers_target_single_speaker_classified_normally(monkeypatch):
    """A single speaker is classified normally (monologue / tweets_or_quotes):
    the full transcript is routed through ``process_text_to_document`` rather than
    the per-statement / dialogue paths."""
    from langchain_core.documents import Document

    import src.subgraphs.process_media_graph.utils.nodes as nodes_mod

    captured = {}

    async def _fake_diarize(*, media_base64, context, encoded_reference_audio,
                            filename, content_type):
        return {
            "text": "Statement one. Statement two.",
            "segments": [
                {"speaker": "S", "text": "Statement one.", "start": 0, "end": 1},
                {"speaker": "S", "text": "Statement two.", "start": 1, "end": 2},
            ],
        }

    async def _fail_dialogue(*args, **kwargs):
        raise AssertionError("dialogue path must not run in all_speakers_target mode")

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
