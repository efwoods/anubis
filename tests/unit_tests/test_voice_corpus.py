"""Unit tests for the voice-clone corpus and the clone state machine.

Pinned down:

- **Only the target's seconds count.** A clip is stored with the duration the
  isolator reports for the target speaker, never the length of the upload.
- **Thresholds drive the clones.** Below 60 s nothing is cloned; at 60 s an
  instant clone is created once; reaching 120 s rebuilds it from the fuller
  corpus and retires the first; a non-personal avatar stops collecting at the
  target while the personal avatar keeps going.
- **The professional clone is prepared only for the personal avatar** once 30
  minutes is reached, and only moves to training after the spoken CAPTCHA.
- **The active voice prefers the professional clone once fine-tuned.**
- **The speak route** refuses without a clone (409, with the collected seconds)
  and serves audio with one.
"""

from types import SimpleNamespace

import pytest

from src.anubis.utils.media_assets import repository as media_repository
from src.anubis.utils.media_assets.repository import InMemoryMediaAssetRepository
from src.anubis.utils.voice import corpus, elevenlabs_client

USER_ID = "auth0-user"
ASSISTANT_ID = "assistant-1"
CLIP = "data:audio/mpeg;base64," + __import__("base64").b64encode(b"speech").decode()


def _context(**overrides):
    values = dict(
        elevenlabs_api_key="sk-test",
        elevenlabs_instant_voice_clone_minimum_seconds=60,
        elevenlabs_instant_voice_clone_target_seconds=120,
        elevenlabs_professional_voice_clone_minimum_seconds=1800,
        elevenlabs_professional_voice_clone_maximum_seconds=10800,
        elevenlabs_professional_voice_clone_training_model="eleven_multilingual_v2",
        elevenlabs_text_to_speech_model="eleven_flash_v2_5",
        elevenlabs_text_to_speech_cost_per_1000_characters_usd=0.05,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeVendor:
    def __init__(self):
        self.instant = []
        self.deleted = []
        self.professional = []
        self.samples = []
        self.trained = []
        self.verified = []
        self.state = "fine_tuning"

    def install(self, monkeypatch):
        async def create_instant_voice(context, *, name, clips, description=""):
            self.instant.append((name, sum(1 for _ in clips)))
            return f"ivc-{len(self.instant)}"

        async def delete_voice(context, voice_id):
            self.deleted.append(voice_id)

        async def create_professional_voice(
            context, *, name, language="en", description=""
        ):
            self.professional.append(name)
            return "pvc-1"

        async def add_professional_samples(context, *, voice_id, clips):
            self.samples.append((voice_id, len(clips)))
            return ["s1"]

        async def submit_verification_recording(context, *, voice_id, recording):
            self.verified.append(voice_id)
            return {"status": "ok"}

        async def train_professional_voice(context, *, voice_id, model_id=None):
            self.trained.append((voice_id, model_id))
            return {"status": "started"}

        async def get_voice_fine_tuning_state(context, *, voice_id, model_id=None):
            return self.state

        async def synthesize_speech(
            context, *, voice_id, text, model_id=None, output_format="mp3_44100_128"
        ):
            return f"audio:{voice_id}:{text}".encode()

        for name, function in locals().items():
            if name not in ("self", "monkeypatch"):
                monkeypatch.setattr(elevenlabs_client, name, function)
        return self


async def _add(repository, context, seconds, *, personal=False):
    return await corpus.add_voice_clip(
        repository,
        context,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        audio_data_uri=CLIP,
        duration_seconds=seconds,
        source="recorder",
        is_personal_avatar=personal,
        avatar_name="Evan",
    )


@pytest.mark.asyncio
async def test_nothing_is_cloned_below_the_minimum(monkeypatch):
    vendor = _FakeVendor().install(monkeypatch)
    repository = InMemoryMediaAssetRepository()
    record = await _add(repository, _context(), 45)
    assert record["instant_voice_id"] is None
    assert record["collected_seconds"] == 45
    assert vendor.instant == []


@pytest.mark.asyncio
async def test_the_instant_clone_is_created_once_at_the_minimum_and_rebuilt_at_the_target(
    monkeypatch,
):
    vendor = _FakeVendor().install(monkeypatch)
    repository = InMemoryMediaAssetRepository()
    await _add(repository, _context(), 45)
    record = await _add(repository, _context(), 30)
    assert record["instant_voice_id"] == "ivc-1"
    assert record["instant_voice_seconds"] == 75
    # More audio below the target changes nothing.
    record = await _add(repository, _context(), 20)
    assert record["instant_voice_id"] == "ivc-1"
    # Reaching the target rebuilds from the fuller corpus and retires the first.
    record = await _add(repository, _context(), 40)
    assert record["instant_voice_id"] == "ivc-2"
    assert vendor.deleted == ["ivc-1"]
    assert record["instant_voice_seconds"] >= 120


@pytest.mark.asyncio
async def test_a_non_personal_avatar_stops_collecting_at_the_target(monkeypatch):
    _FakeVendor().install(monkeypatch)
    repository = InMemoryMediaAssetRepository()
    await _add(repository, _context(), 130)
    await _add(repository, _context(), 60)
    assert await repository.total_voice_seconds(ASSISTANT_ID) == 130
    assert len(repository.clips) == 1


@pytest.mark.asyncio
async def test_the_personal_avatar_prepares_a_professional_clone_at_thirty_minutes(
    monkeypatch,
):
    vendor = _FakeVendor().install(monkeypatch)
    repository = InMemoryMediaAssetRepository()
    context = _context()
    record = await _add(repository, context, 900, personal=True)
    assert record["professional_state"] == "collecting"
    assert vendor.professional == []
    record = await _add(repository, context, 900, personal=True)
    assert record["professional_state"] == "awaiting_verification"
    assert record["professional_voice_id"] == "pvc-1"
    assert vendor.samples == [("pvc-1", 2)]
    # The personal avatar's instant clone exists too, from the same corpus.
    assert record["instant_voice_id"] == "ivc-1"


@pytest.mark.asyncio
async def test_verification_starts_training_and_polling_marks_it_fine_tuned(
    monkeypatch,
):
    vendor = _FakeVendor().install(monkeypatch)
    repository = InMemoryMediaAssetRepository()
    context = _context()
    await _add(repository, context, 1800, personal=True)

    record = await corpus.submit_verification_and_train(
        repository,
        context,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        recording=("captcha.webm", b"audio", "audio/webm"),
    )
    assert record["professional_state"] == "training"
    assert vendor.verified == ["pvc-1"]
    assert vendor.trained == [("pvc-1", "eleven_multilingual_v2")]

    assert (await corpus.resolve_active_voice_id(repository, ASSISTANT_ID)) == (
        "instant",
        "ivc-1",
    )
    vendor.state = "fine_tuned"
    record = await corpus.refresh_training_state(
        repository, context, user_id=USER_ID, assistant_id=ASSISTANT_ID
    )
    assert record["professional_state"] == "fine_tuned"
    assert (await corpus.resolve_active_voice_id(repository, ASSISTANT_ID)) == (
        "professional",
        "pvc-1",
    )


@pytest.mark.asyncio
async def test_verification_is_refused_before_the_corpus_is_ready(monkeypatch):
    _FakeVendor().install(monkeypatch)
    repository = InMemoryMediaAssetRepository()
    with pytest.raises(ValueError):
        await corpus.submit_verification_and_train(
            repository,
            _context(),
            user_id=USER_ID,
            assistant_id=ASSISTANT_ID,
            recording=("captcha.webm", b"audio", "audio/webm"),
        )


@pytest.mark.asyncio
async def test_a_vendor_failure_is_recorded_not_raised(monkeypatch):
    _FakeVendor().install(monkeypatch)

    async def _fail(context, *, name, clips, description=""):
        raise elevenlabs_client.ElevenLabsError("plan does not allow cloning")

    monkeypatch.setattr(elevenlabs_client, "create_instant_voice", _fail)
    repository = InMemoryMediaAssetRepository()
    record = await _add(repository, _context(), 90)
    assert record["instant_voice_id"] is None
    assert "plan does not allow" in record["detail"]["instant_error"]


def test_target_windows_merge_and_skip_non_target_turns():
    from src.anubis.utils.voice.clips import target_windows

    windows = target_windows(
        [
            {"is_target": True, "start": 0.0, "end": 2.0},
            {"is_target": False, "start": 2.0, "end": 5.0},
            {"is_target": True, "start": 4.5, "end": 6.0},
            {"is_target": True, "start": 6.0, "end": 6.1},
            {"is_target": True, "start": 9.0, "end": 12.0},
        ]
    )
    assert windows == [(0.0, 2.0), (4.5, 6.0), (9.0, 12.0)]


# --------------------------------------------------------------------------
# The speak route
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
async def test_speak_refuses_without_a_clone_and_reports_progress(monkeypatch):
    from src.api import webapp as webapp_module

    _FakeVendor().install(monkeypatch)
    repository = InMemoryMediaAssetRepository()
    media_repository.set_media_asset_repository(repository)
    monkeypatch.setattr(
        webapp_module.app,
        "state",
        SimpleNamespace(context=_context(), pool=None, stripe=None),
    )
    monkeypatch.setattr(webapp_module, "enforce_tier_capability", lambda *a, **k: None)
    await repository.add_voice_clip(
        {
            "user_id": USER_ID,
            "assistant_id": ASSISTANT_ID,
            "source": "recorder",
            "mime_type": "audio/mpeg",
            "bytes": b"x",
            "duration_seconds": 30,
        }
    )

    response = await webapp_module.speak_text(
        request=_json_request({"assistant_id": ASSISTANT_ID, "text": "hello"}),
        current_user={"API_KEY": "k", "identities": [{"user_id": USER_ID}]},
    )
    assert response.status_code == 409
    body = response.body.decode("utf-8")
    assert "voice_not_ready" in body
    assert '"collected_seconds":30' in body.replace(" ", "")


@pytest.mark.asyncio
async def test_speak_returns_audio_in_the_active_voice(monkeypatch):
    from src.api import webapp as webapp_module

    _FakeVendor().install(monkeypatch)
    repository = InMemoryMediaAssetRepository()
    media_repository.set_media_asset_repository(repository)
    await repository.upsert_voice(
        {"assistant_id": ASSISTANT_ID, "user_id": USER_ID, "instant_voice_id": "ivc-9"}
    )
    monkeypatch.setattr(
        webapp_module.app,
        "state",
        SimpleNamespace(context=_context(), pool=None, stripe=None),
    )
    monkeypatch.setattr(webapp_module, "enforce_tier_capability", lambda *a, **k: None)
    recorded = {}

    async def _meter(current_user, **kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(webapp_module, "_meter_speech_characters", _meter)

    response = await webapp_module.speak_text(
        request=_json_request({"assistant_id": ASSISTANT_ID, "text": "hello there"}),
        current_user={"API_KEY": "k", "identities": [{"user_id": USER_ID}]},
    )
    assert response.status_code == 200
    assert response.body == b"audio:ivc-9:hello there"
    assert response.media_type == "audio/mpeg"
    assert response.headers["x-voice-kind"] == "instant"
    assert recorded["characters"] == len("hello there")
    assert recorded["cost_usd"] == pytest.approx(0.05 * 11 / 1000)
