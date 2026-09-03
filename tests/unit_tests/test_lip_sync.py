"""Unit tests for lip-synced video replies.

Pinned down:

- **A repeated phrase is served from the table**, never rendered twice: the
  clip is keyed by emotion plus a digest of the text.
- **The still is uploaded to the vendor once** and its asset id cached on the row.
- **Starting a clip records a durable job**; polling stores the finished bytes as
  a ``lip_sync`` asset and reports completion exactly once as new (so spend is
  metered once).
- **The route gates on the video capability** and on the process-wide switch.
"""

from types import SimpleNamespace

import pytest

from src.anubis.utils.media_assets import repository as media_repository
from src.anubis.utils.media_assets.repository import (
    ASSET_KIND_LIP_SYNC,
    ASSET_KIND_STILL,
    InMemoryMediaAssetRepository,
)
from src.anubis.utils.media_generation import lip_sync
from src.anubis.utils.voice import elevenlabs_client

USER_ID = "auth0-user"
ASSISTANT_ID = "assistant-1"


def _context(**overrides):
    values = dict(
        elevenlabs_api_key="sk-test",
        lip_sync_enabled="true",
        elevenlabs_lip_sync_model="creatify-aurora",
        elevenlabs_lip_sync_resolution="720p",
        elevenlabs_lip_sync_cost_per_second_usd=0.14,
        elevenlabs_text_to_speech_model="eleven_flash_v2_5",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeVendor:
    def __init__(self):
        self.uploads = []
        self.generations = []
        self.status = "generating"

    def install(self, monkeypatch):
        async def synthesize_speech(context, *, voice_id, text, model_id=None, output_format="mp3_44100_128"):
            return b"speech-bytes"

        async def upload_asset(context, *, payload, name, mime_type):
            self.uploads.append((name, mime_type))
            return f"asset-{len(self.uploads)}"

        async def create_lip_sync_video(context, *, model_id, image_asset_id, audio_asset_id, resolution="720p"):
            self.generations.append((model_id, image_asset_id, audio_asset_id))
            return f"gen-{len(self.generations)}"

        async def get_lip_sync_video(context, *, generation_id):
            return {
                "status": self.status,
                "content_url": "https://vendor/clip.mp4" if self.status == "completed" else None,
                "content_mime_type": "video/mp4",
            }

        async def download(url):
            return b"video-bytes", "video/mp4"

        for name, function in locals().items():
            if name not in ("self", "monkeypatch"):
                monkeypatch.setattr(elevenlabs_client, name, function)
        return self


async def _repository_with_still():
    repository = InMemoryMediaAssetRepository()
    await repository.upsert_emotion_asset(
        {"user_id": USER_ID, "assistant_id": ASSISTANT_ID, "emotion": "joy",
         "asset_kind": ASSET_KIND_STILL, "mime_type": "image/jpeg", "bytes": b"joy-still"}
    )
    return repository


@pytest.mark.asyncio
async def test_a_clip_is_started_then_stored_on_completion_and_metered_once(monkeypatch):
    vendor = _FakeVendor().install(monkeypatch)
    repository = await _repository_with_still()
    context = _context()

    started = await lip_sync.start_lip_sync(
        context, repository, user_id=USER_ID, assistant_id=ASSISTANT_ID,
        text="Hello there, friend.", emotion="joy", voice_id="ivc-1",
    )
    assert started["status"] == "pending"
    assert vendor.uploads[0][1] == "image/jpeg", "the still is uploaded first"
    assert vendor.uploads[1][1] == "audio/mpeg"
    assert vendor.generations == [("creatify-aurora", "asset-1", "asset-2")]
    still = next(a for a in await repository.list_emotion_assets(ASSISTANT_ID) if a["asset_kind"] == ASSET_KIND_STILL)
    assert still["elevenlabs_asset_id"] == "asset-1", "the still's vendor id is cached"

    job = await repository.get_job(started["job_id"])
    assert (await lip_sync.poll_lip_sync(context, repository, job=job))["status"] == "pending"

    vendor.status = "completed"
    job = await repository.get_job(started["job_id"])
    result = await lip_sync.poll_lip_sync(context, repository, job=job)
    assert result["status"] == "completed"
    assert result["newly_completed"] is True
    clip = await repository.get_emotion_asset(result["asset_id"])
    assert clip["asset_kind"] == ASSET_KIND_LIP_SYNC
    assert clip["bytes"] == b"video-bytes"
    assert clip["variant_key"] == lip_sync.text_digest("Hello there, friend.")

    # A second poll of the finished job is not "new" — spend is metered once.
    job = await repository.get_job(started["job_id"])
    again = await lip_sync.poll_lip_sync(context, repository, job=job)
    assert again["status"] == "completed"
    assert not again.get("newly_completed")


@pytest.mark.asyncio
async def test_a_repeated_phrase_is_served_from_the_table(monkeypatch):
    vendor = _FakeVendor().install(monkeypatch)
    repository = await _repository_with_still()
    context = _context()
    started = await lip_sync.start_lip_sync(
        context, repository, user_id=USER_ID, assistant_id=ASSISTANT_ID,
        text="Good morning!", emotion="joy", voice_id="ivc-1",
    )
    vendor.status = "completed"
    await lip_sync.poll_lip_sync(context, repository, job=await repository.get_job(started["job_id"]))

    again = await lip_sync.start_lip_sync(
        context, repository, user_id=USER_ID, assistant_id=ASSISTANT_ID,
        text="good   morning!", emotion="joy", voice_id="ivc-1",
    )
    assert again["status"] == "completed"
    assert again["cached"] is True
    assert len(vendor.generations) == 1, "no second render for the same words"
    assert len(vendor.uploads) == 2, "the still was not uploaded again either"


@pytest.mark.asyncio
async def test_the_still_upload_falls_back_to_neutral(monkeypatch):
    _FakeVendor().install(monkeypatch)
    repository = InMemoryMediaAssetRepository()
    await repository.upsert_emotion_asset(
        {"user_id": USER_ID, "assistant_id": ASSISTANT_ID, "emotion": "neutral",
         "asset_kind": ASSET_KIND_STILL, "mime_type": "image/jpeg", "bytes": b"neutral"}
    )
    asset_id = await lip_sync.ensure_still_uploaded(
        _context(), repository, assistant_id=ASSISTANT_ID, emotion="fear"
    )
    assert asset_id == "asset-1"


@pytest.mark.asyncio
async def test_a_failed_generation_marks_the_job_failed(monkeypatch):
    vendor = _FakeVendor().install(monkeypatch)
    repository = await _repository_with_still()
    context = _context()
    started = await lip_sync.start_lip_sync(
        context, repository, user_id=USER_ID, assistant_id=ASSISTANT_ID,
        text="x", emotion="joy", voice_id="ivc-1",
    )
    vendor.status = "failed"
    result = await lip_sync.poll_lip_sync(context, repository, job=await repository.get_job(started["job_id"]))
    assert result["status"] == "failed"
    assert (await repository.get_job(started["job_id"]))["state"] == "failed"


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_repository():
    media_repository.set_media_asset_repository(None)
    yield
    media_repository.set_media_asset_repository(None)


def _json_request(payload):
    async def _json():
        return payload

    return SimpleNamespace(json=_json)


@pytest.mark.asyncio
async def test_the_route_refuses_without_the_video_capability(monkeypatch):
    from src.api import webapp as webapp_module

    _FakeVendor().install(monkeypatch)
    repository = await _repository_with_still()
    media_repository.set_media_asset_repository(repository)
    monkeypatch.setattr(webapp_module.app, "state", SimpleNamespace(context=_context(), pool=None, stripe=None))

    def _refuse(current_user, capability):
        raise webapp_module.HTTPException(status_code=403, detail=f"needs {capability.value}")

    monkeypatch.setattr(webapp_module, "enforce_tier_capability", _refuse)
    with pytest.raises(webapp_module.HTTPException) as raised:
        await webapp_module.start_lip_sync_clip(
            request=_json_request({"assistant_id": ASSISTANT_ID, "text": "hi", "emotion": "joy"}),
            current_user={"API_KEY": "k", "identities": [{"user_id": USER_ID}]},
        )
    assert raised.value.status_code == 403
    assert "video_responses" in raised.value.detail


@pytest.mark.asyncio
async def test_the_route_starts_a_generation_and_reports_completion(monkeypatch):
    from src.api import webapp as webapp_module

    vendor = _FakeVendor().install(monkeypatch)
    repository = await _repository_with_still()
    await repository.upsert_voice({"assistant_id": ASSISTANT_ID, "user_id": USER_ID, "instant_voice_id": "ivc-1"})
    media_repository.set_media_asset_repository(repository)
    monkeypatch.setattr(webapp_module.app, "state", SimpleNamespace(context=_context(), pool=None, stripe=None))
    monkeypatch.setattr(webapp_module, "enforce_tier_capability", lambda *a, **k: None)
    metered = {}

    async def _meter(current_user, **kwargs):
        metered.update(kwargs)

    monkeypatch.setattr(webapp_module, "_meter_video_seconds", _meter)

    response = await webapp_module.start_lip_sync_clip(
        request=_json_request({"assistant_id": ASSISTANT_ID, "text": "Hello there my friend", "emotion": "joy"}),
        current_user={"API_KEY": "k", "identities": [{"user_id": USER_ID}]},
    )
    assert response.status_code == 202
    import json

    generation_id = json.loads(response.body)["generation_id"]

    vendor.status = "completed"
    status = await webapp_module.get_lip_sync_clip(
        generation_id=generation_id,
        current_user={"API_KEY": "k", "identities": [{"user_id": USER_ID}]},
    )
    body = json.loads(status.body)
    assert body["status"] == "completed"
    assert body["video_url"].startswith("/avatar_emotion_media/")
    assert metered["seconds"] == 2.0
    assert metered["cost_usd"] == pytest.approx(0.28)
