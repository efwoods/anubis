"""How the media pipeline picks the reference clip and feeds the voice corpus.

Pinned down against ``process_media_item_task`` with the heavy audio work
stubbed out:

- The first audio upload for an avatar becomes the reference clip without any
  request flag, joins the voice corpus once, and still yields identity
  documents.
- A later upload never replaces the stored reference; the avatar's speech in
  that upload still joins the corpus and identity — for a non-personal avatar
  too.
- Two uploads processed concurrently store exactly one reference.
- A first upload that reads the calibration sentence stores the reference
  only.
"""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import src.subgraphs.process_media_graph.utils.nodes as nodes_mod
from src.anubis.utils.media_assets import repository as media_repository
from src.anubis.utils.media_assets.repository import InMemoryMediaAssetRepository
from src.anubis.utils.voice import elevenlabs_client, reference_audio

USER_ID = "u"
ASSISTANT_ID = "a"
CLIP = "data:audio/mp3;base64,QUJD"


class _Store:
    def __init__(self):
        self.rows = {}

    async def aget(self, namespace, key):
        value = self.rows.get((tuple(namespace), key))
        return None if value is None else SimpleNamespace(value=value)

    async def aput(self, namespace, key, value):
        await asyncio.sleep(0)
        self.rows[(tuple(namespace), key)] = value


def _audio_item(filename):
    return {
        "type": "audio",
        "base64_encoded_str": CLIP,
        "metadata": {
            "filename": filename,
            "content_type": "audio/mp3",
            "user_id": USER_ID,
            "assistant_id": ASSISTANT_ID,
            "namespace_filename": f"key-{filename}",
            "reference_audio": False,
        },
    }


def _runtime():
    return SimpleNamespace(
        context=SimpleNamespace(
            audio_diarization_known_speaker_name="avatar",
            elevenlabs_api_key="sk-test",
            elevenlabs_instant_voice_clone_minimum_seconds=60,
            elevenlabs_instant_voice_clone_target_seconds=120,
            elevenlabs_professional_voice_clone_minimum_seconds=1800,
            elevenlabs_professional_voice_clone_maximum_seconds=10800,
            embedding_model="stub",
        )
    )


@pytest.fixture
def pipeline(monkeypatch):
    """Stub the audio work; record what the pipeline did."""
    repository = InMemoryMediaAssetRepository()
    media_repository.set_media_asset_repository(repository)
    recorded = {"isolations": [], "similarity": 0.1, "classified": []}

    async def fake_isolate(audio_uri, *, context, filename, content_type, reference_audio=False, allow_single_speaker=None):
        recorded["isolations"].append((filename, bool(reference_audio)))
        seconds = 3.0 if reference_audio else 30.0
        return {
            "audio_base64_preprocessed": CLIP,
            "duration": seconds,
            "text": f"speech from {filename}",
        }

    async def fake_diarize(*, media_base64, context, encoded_reference_audio, filename, content_type, **_ignored):
        return {
            "text": "One. Two.",
            "segments": [
                {"speaker": "S", "text": "One.", "start": 0.0, "end": 1.0},
                {"speaker": "S", "text": "Two.", "start": 1.0, "end": 2.0},
            ],
        }

    async def fake_cut(audio_uri, turns):
        return CLIP, 20.0

    async def fake_classify(**kwargs):
        recorded["classified"].append(kwargs["media_item"].get("content"))
        return [Document(page_content="chunk", metadata={"namespace": "quote"})]

    original_to_thread = asyncio.to_thread

    async def fake_to_thread(function, *args, **kwargs):
        if getattr(function, "__name__", "") == "_compute_reference_similarity":
            return recorded["similarity"]
        return await original_to_thread(function, *args, **kwargs)

    async def create_instant_voice(context, *, name, clips, description=""):
        return "ivc-1"

    monkeypatch.setattr(nodes_mod, "isolate_dominant_speaker_audio_b64", fake_isolate)
    monkeypatch.setattr(nodes_mod, "transcribe_audio_diarize", fake_diarize)
    monkeypatch.setattr(nodes_mod, "process_text_to_document", fake_classify)
    monkeypatch.setattr(nodes_mod.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(
        "src.anubis.utils.voice.clips.cut_target_turns_to_mp3_data_uri", fake_cut
    )
    monkeypatch.setattr(elevenlabs_client, "create_instant_voice", create_instant_voice)
    reference_audio._reference_audio_locks.clear()
    yield SimpleNamespace(repository=repository, recorded=recorded, store=_Store())
    media_repository.set_media_asset_repository(None)


async def _process(pipeline, filename, config=None):
    return await nodes_mod.process_media_item_task(
        _audio_item(filename), _runtime(), config or {}, store=pipeline.store
    )


def _clips(pipeline):
    return [
        (clip["source_document_name"], clip["source"], clip["duration_seconds"])
        for clip in pipeline.repository.clips.values()
    ]


@pytest.mark.asyncio
async def test_the_first_upload_becomes_the_reference_and_joins_the_corpus_once(pipeline):
    documents = await _process(pipeline, "Mom.m4a")

    stored = await reference_audio.read_reference_audio(pipeline.store, USER_ID, ASSISTANT_ID)
    assert stored["filename"] == "Mom.m4a"
    assert stored["transcript_text"] == "speech from Mom.m4a"
    # One reference cut, one corpus cut; the target-turn pass did not run again.
    assert pipeline.recorded["isolations"] == [("Mom.m4a", True), ("Mom.m4a", False)]
    assert _clips(pipeline) == [("Mom.m4a", "reference_upload", 30.0)]
    # The reference document is listable, and the transcript still reached identity.
    assert documents[0].metadata["namespace"] == "reference_audio"
    assert documents[0].metadata["filename"] == "Mom.m4a"
    assert pipeline.recorded["classified"] == ["One. Two."]


@pytest.mark.asyncio
async def test_a_later_upload_keeps_the_reference_and_still_feeds_voice_and_identity(pipeline):
    await _process(pipeline, "Mom.m4a")
    documents = await _process(pipeline, "talk.mp3")

    stored = await reference_audio.read_reference_audio(pipeline.store, USER_ID, ASSISTANT_ID)
    assert stored["filename"] == "Mom.m4a"
    # No second reference cut; the later upload's turns were cut into the corpus.
    assert ("talk.mp3", True) not in pipeline.recorded["isolations"]
    assert _clips(pipeline) == [
        ("Mom.m4a", "reference_upload", 30.0),
        ("talk.mp3", "media_upload", 20.0),
    ]
    assert all(document.metadata.get("namespace") != "reference_audio" for document in documents)
    assert pipeline.recorded["classified"] == ["One. Two.", "One. Two."]


@pytest.mark.asyncio
async def test_a_non_personal_avatar_collects_from_ordinary_uploads(pipeline):
    await _process(pipeline, "Mom.m4a")
    await _process(pipeline, "talk.mp3", config={"configurable": {"assistant_ctx": {"metadata": {}}}})
    assert ("talk.mp3", "media_upload", 20.0) in _clips(pipeline)


@pytest.mark.asyncio
async def test_two_uploads_in_one_batch_store_exactly_one_reference(pipeline):
    await asyncio.gather(_process(pipeline, "first.m4a"), _process(pipeline, "second.m4a"))
    reference_cuts = [name for name, is_reference in pipeline.recorded["isolations"] if is_reference]
    assert len(reference_cuts) == 1
    stored = await reference_audio.read_reference_audio(pipeline.store, USER_ID, ASSISTANT_ID)
    assert stored["filename"] == reference_cuts[0]
    assert sorted(name for name, _source, _seconds in _clips(pipeline)) == ["first.m4a", "second.m4a"]


@pytest.mark.asyncio
async def test_a_calibration_sentence_upload_stores_the_reference_only(pipeline):
    pipeline.recorded["similarity"] = 0.95
    documents = await _process(pipeline, "script.webm")
    assert [document.metadata["namespace"] for document in documents] == ["reference_audio"]
    assert pipeline.recorded["classified"] == []
    assert _clips(pipeline) == [("script.webm", "reference_upload", 30.0)]
