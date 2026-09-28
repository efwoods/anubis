"""Unit tests for media added in the Voice section feeding the audible voice.

Pinned down:

- **An already-indexed item reuses the stored diarization.** The avatar's
  windows come from the ``quote`` / ``identity`` documents written when the item
  was first processed; the diarizer is never called and nothing is written to
  the store.
- **An item with no stored speech of the avatar fails with a reason** instead
  of finishing as a silent success.
- **A YouTube link has only the audio downloaded again.**
- **An existing instant clone is rebuilt once per batch, newest clips first**, so
  the speech the owner just added is what the avatar sounds like; an avatar
  with no clone before the batch is left to ``add_voice_clip``.
- **The batch runner routes an already-indexed Voice-section item** to the
  voice-only job rather than the media graph, and reports the seconds added.
"""

import asyncio
import base64
from types import SimpleNamespace

import pytest

from src.anubis.utils.media_assets.repository import InMemoryMediaAssetRepository
from src.anubis.utils.voice import clips, corpus, elevenlabs_client, voice_upload
from src.api import media_jobs

USER_ID = "auth0-user"
ASSISTANT_ID = "assistant-1"
NAMESPACE_FILENAME = "fee42951-8312-5ed0-b532-f043a32f7635"
AUDIO_DATA_URI = "data:audio/mpeg;base64," + base64.b64encode(b"recording").decode()
CUT_CLIP_DATA_URI = "data:audio/mpeg;base64," + base64.b64encode(b"claire").decode()


def _context(**overrides):
    values = dict(
        elevenlabs_api_key="sk-test",
        elevenlabs_instant_voice_clone_minimum_seconds=60,
        elevenlabs_instant_voice_clone_target_seconds=120,
        elevenlabs_professional_voice_clone_minimum_seconds=1800,
        elevenlabs_professional_voice_clone_maximum_seconds=10800,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _stored_document(namespace, **metadata):
    return SimpleNamespace(
        namespace=(USER_ID, ASSISTANT_ID, namespace, NAMESPACE_FILENAME),
        value={
            "document": {
                "kwargs": {
                    "page_content": "words",
                    "metadata": {"namespace_filename": NAMESPACE_FILENAME, **metadata},
                }
            }
        },
    )


class _FakeStore:
    """A store holding documents by namespace, recording every write attempt."""

    def __init__(self, documents):
        self.documents = documents
        self.writes = []

    async def asearch(self, namespace_prefix, *, limit=10, **_ignored):
        return [
            document
            for document in self.documents
            if tuple(document.namespace[: len(namespace_prefix)])
            == tuple(namespace_prefix)
        ][:limit]

    async def aput(self, *arguments, **keyword_arguments):
        self.writes.append((arguments, keyword_arguments))


def _claire_store():
    return _FakeStore(
        [
            _stored_document("quote", start=43.0, end=50.7, is_target=True),
            _stored_document("quote", start=185.6, end=245.6, is_target=True),
            _stored_document("quote", start=326.4, end=334.7, is_target=True),
            # The interviewer's turn is stored too and must not be cut.
            _stored_document("identity", start=57.9, end=183.6, speaker="A"),
        ]
    )


class _FakeVendor:
    def __init__(self):
        self.instant = []
        self.deleted = []

    def install(self, monkeypatch):
        async def create_instant_voice(context, *, name, clips, description=""):
            self.instant.append([clip_bytes for _name, clip_bytes, _mime in clips])
            return f"ivc-{len(self.instant)}"

        async def delete_voice(context, voice_id):
            self.deleted.append(voice_id)

        async def voice_is_blocked(context, *, voice_id):
            return False

        monkeypatch.setattr(
            elevenlabs_client, "create_instant_voice", create_instant_voice
        )
        monkeypatch.setattr(elevenlabs_client, "delete_voice", delete_voice)
        monkeypatch.setattr(elevenlabs_client, "voice_is_blocked", voice_is_blocked)
        return self


@pytest.fixture
def cut_turns(monkeypatch):
    """Replace the ffmpeg cut with a recorder that returns a clip of the windows' length."""
    calls = []

    async def fake_cut(audio_data_uri, turns):
        calls.append((audio_data_uri, turns))
        seconds = sum(end - start for start, end in clips.target_windows(turns))
        return CUT_CLIP_DATA_URI, seconds

    monkeypatch.setattr(clips, "cut_target_turns_to_mp3_data_uri", fake_cut)
    return calls


@pytest.fixture
def diarizer_forbidden(monkeypatch):
    from src.anubis.utils import utility

    async def refuse(*_arguments, **_keyword_arguments):
        raise AssertionError("the diarizer must not run for an already-diarized item")

    monkeypatch.setattr(utility, "transcribe_audio_diarize", refuse)
    monkeypatch.setattr(utility, "isolate_dominant_speaker_audio_b64", refuse)


@pytest.mark.asyncio
async def test_an_indexed_item_feeds_the_voice_from_the_stored_diarization(
    monkeypatch, cut_turns, diarizer_forbidden
):
    _FakeVendor().install(monkeypatch)
    repository = InMemoryMediaAssetRepository()
    store = _claire_store()
    progress_events = []

    added_seconds = await voice_upload.collect_voice_from_indexed_item(
        store,
        _context(),
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        media_file={
            "filename": "interview.mp3",
            "content_type": "audio/mpeg",
            "base64_encoded_str": AUDIO_DATA_URI,
            "namespace_filename": NAMESPACE_FILENAME,
        },
        avatar_name="Claire Wineland",
        emit=progress_events.append,
        repository=repository,
    )

    assert added_seconds == pytest.approx(7.7 + 60.0 + 8.3)
    uploaded_audio, cut_turn_list = cut_turns[0]
    assert uploaded_audio == AUDIO_DATA_URI
    # Only the avatar's own turns are cut; the interviewer's stays behind.
    assert {(turn["start"], turn["end"]) for turn in cut_turn_list} == {
        (43.0, 50.7),
        (185.6, 245.6),
        (326.4, 334.7),
    }
    stored_clips = await repository.list_voice_clips(ASSISTANT_ID)
    assert [clip["source"] for clip in stored_clips] == ["voice_upload"]
    assert stored_clips[0]["source_document_name"] == "interview.mp3"
    # Past the sixty-second minimum, the clip builds the instant clone at once.
    voice_record = await repository.get_voice(ASSISTANT_ID)
    assert voice_record["instant_voice_id"] == "ivc-1"
    assert store.writes == []
    assert [event["stage"] for event in progress_events] == [
        "voice_reusing_diarization",
        "voice_clip_collected",
        "instant_clone_created",
    ]


@pytest.mark.asyncio
async def test_an_item_with_no_stored_speech_of_the_avatar_fails_with_a_reason(
    monkeypatch, cut_turns, diarizer_forbidden
):
    _FakeVendor().install(monkeypatch)
    store = _FakeStore([_stored_document("identity", start=0.0, end=90.0, speaker="A")])

    with pytest.raises(
        voice_upload.VoiceUploadError, match="No speech of Claire Wineland"
    ):
        await voice_upload.collect_voice_from_indexed_item(
            store,
            _context(),
            user_id=USER_ID,
            assistant_id=ASSISTANT_ID,
            media_file={
                "filename": "panel.mp3",
                "content_type": "audio/mpeg",
                "base64_encoded_str": AUDIO_DATA_URI,
                "namespace_filename": NAMESPACE_FILENAME,
            },
            avatar_name="Claire Wineland",
            repository=InMemoryMediaAssetRepository(),
        )
    assert cut_turns == []


@pytest.mark.asyncio
async def test_a_youtube_link_downloads_only_the_audio(monkeypatch):
    from src.anubis.utils.classes import URLDocumentLoaderClass

    downloaded_links = []

    async def fake_download(link):
        downloaded_links.append(link)
        return AUDIO_DATA_URI, ".mp3"

    monkeypatch.setattr(
        URLDocumentLoaderClass, "_download_youtube_audio_b64", fake_download
    )
    audio_data_uri = await voice_upload.audio_for_voice_upload(
        {
            "filename": "https://www.youtube.com/watch?v=W0RcaTSXWZ8",
            "content_type": "text/html",
            "page_url": "https://www.youtube.com/watch?v=W0RcaTSXWZ8",
        }
    )
    assert audio_data_uri == AUDIO_DATA_URI
    assert downloaded_links == ["https://www.youtube.com/watch?v=W0RcaTSXWZ8"]


@pytest.mark.asyncio
async def test_a_document_carries_no_voice():
    with pytest.raises(voice_upload.VoiceUploadError, match="not audio or video"):
        await voice_upload.audio_for_voice_upload(
            {
                "filename": "notes.pdf",
                "content_type": "application/pdf",
                "base64_encoded_str": "data:application/pdf;base64,AAAA",
            }
        )


@pytest.mark.asyncio
async def test_an_existing_clone_is_rebuilt_from_the_newest_speech(monkeypatch):
    vendor = _FakeVendor().install(monkeypatch)
    repository = InMemoryMediaAssetRepository()
    context = _context()
    old_clip = "data:audio/mpeg;base64," + base64.b64encode(b"old").decode()
    new_clip = "data:audio/mpeg;base64," + base64.b64encode(b"new").decode()
    await corpus.add_voice_clip(
        repository,
        context,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        audio_data_uri=old_clip,
        duration_seconds=120,
        source="recorder",
    )
    instant_voice_id_before = await voice_upload.instant_voice_before_batch(
        ASSISTANT_ID, repository=repository
    )
    assert instant_voice_id_before == "ivc-1"
    await corpus.add_voice_clip(
        repository,
        context,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        audio_data_uri=new_clip,
        duration_seconds=70,
        source="voice_upload",
    )

    rebuilt_record = await voice_upload.finish_voice_upload_batch(
        context,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        avatar_name="Claire Wineland",
        seconds_added=70,
        instant_voice_id_before=instant_voice_id_before,
        repository=repository,
    )

    assert vendor.deleted == ["ivc-1"]
    assert rebuilt_record["instant_voice_id"] == "ivc-2"
    # The rebuilt clone hears the new speech first; before the change the
    # oldest 120 seconds filled the whole target and the upload was ignored.
    assert vendor.instant[-1][0] == b"new"


@pytest.mark.asyncio
async def test_no_rebuild_when_the_batch_built_the_first_clone(monkeypatch):
    vendor = _FakeVendor().install(monkeypatch)
    rebuilt_record = await voice_upload.finish_voice_upload_batch(
        _context(),
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        avatar_name="Claire Wineland",
        seconds_added=76,
        instant_voice_id_before=None,
        repository=InMemoryMediaAssetRepository(),
    )
    assert rebuilt_record is None
    assert vendor.deleted == []


@pytest.mark.asyncio
async def test_the_batch_routes_an_indexed_voice_upload_past_the_graph(monkeypatch):
    registry = {}
    master = media_jobs.create_master_job(registry, USER_ID, ASSISTANT_ID)
    child = media_jobs.create_child_job(
        registry,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        parent_id=master.job_id,
        filename="https://www.youtube.com/watch?v=W0RcaTSXWZ8",
        namespace_filename=NAMESPACE_FILENAME,
    )
    master.child_ids.append(child.job_id)

    async def graph_must_not_run(*_arguments, **_keyword_arguments):
        raise AssertionError("an already-indexed voice upload must not enter the graph")

    async def fake_collect(store, context, *, emit, **_keyword_arguments):
        emit(
            {
                "stage": "voice_clip_collected",
                "seconds": 76.0,
                "collected_seconds": 76.0,
            }
        )
        return 76.0

    monkeypatch.setattr(media_jobs, "run_single_item_job", graph_must_not_run)
    monkeypatch.setattr(voice_upload, "collect_voice_from_indexed_item", fake_collect)

    async def fake_calibrate(*_arguments, **_keyword_arguments):
        return None

    monkeypatch.setattr(
        media_jobs, "_calibrate_ground_truth_after_batch", fake_calibrate
    )
    monkeypatch.setattr(media_jobs, "_verify_facts_after_batch", fake_calibrate)

    settled_masters = []

    async def on_voice_upload_settled(settled_master):
        settled_masters.append(settled_master.voice_seconds_collected)
        return {
            "seconds_added": settled_master.voice_seconds_collected,
            "rebuilt": False,
        }

    await media_jobs.run_batch_media_job(
        master,
        [
            {
                "child": child,
                "media_file": {
                    "filename": child.filename,
                    "namespace_filename": NAMESPACE_FILENAME,
                    "page_url": child.filename,
                    "voice_upload": True,
                },
            }
        ],
        {
            "configurable": {
                "assistant_ctx": {"name": "Claire Wineland", "metadata": {}}
            }
        },
        store=None,
        context=_context(),
        concurrency=2,
        existing_namespaces=[NAMESPACE_FILENAME],
        on_voice_upload_settled=on_voice_upload_settled,
    )

    assert child.status == "completed"
    assert child.result["voice_seconds_collected"] == 76.0
    assert settled_masters == [76.0]
    assert master.result["voice_seconds_collected"] == 76.0
    assert master.result["voice"] == {"seconds_added": 76.0, "rebuilt": False}


@pytest.mark.asyncio
async def test_a_failed_voice_upload_item_is_an_error_on_the_card(monkeypatch):
    registry = {}
    master = media_jobs.create_master_job(registry, USER_ID, ASSISTANT_ID)
    child = media_jobs.create_child_job(
        registry,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        parent_id=master.job_id,
        filename="panel.mp3",
        namespace_filename=NAMESPACE_FILENAME,
    )

    async def fake_collect(*_arguments, **_keyword_arguments):
        raise voice_upload.VoiceUploadError("No speech of Claire Wineland was found")

    monkeypatch.setattr(voice_upload, "collect_voice_from_indexed_item", fake_collect)
    await media_jobs.run_voice_upload_item_job(
        child, master, {"filename": "panel.mp3"}, {"configurable": {}}, None, _context()
    )
    assert child.status == "error"
    assert "No speech of Claire Wineland" in child.error
    await asyncio.sleep(0)
