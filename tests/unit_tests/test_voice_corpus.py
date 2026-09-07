"""Unit tests for the voice-clone corpus and the clone state machine.

Pinned down:

- **Only the target's seconds count.** A clip is stored with the duration the
  isolator reports for the target speaker, never the length of the upload.
- **Thresholds drive the clones.** Below 60 s nothing is cloned; at 60 s an
  instant clone is created once and never rebuilt — the first voice is final.
  Every avatar keeps storing clips afterwards (the pool the owner picks a
  reference from, and the seconds the panel reports).
- **The professional clone is prepared only for the personal avatar** once 30
  minutes is reached, and only moves to training after the spoken CAPTCHA.
- **The active voice prefers the professional clone once fine-tuned.**
- **The speak route** refuses without a clone (409, with the collected seconds)
  and serves audio with one.
- **A voice ElevenLabs has banned is reported as banned**, not as a voice model
  that is trained and available. The ban is permanent and belongs to the
  recording, so it is found by asking the vendor before anyone presses speak,
  it is recorded when a speak attempt meets it, and it is answered with 409
  ``voice_blocked`` rather than the ``voice_not_ready`` that asks for more
  speech.
- **The ban is noted on the avatar itself**, so every screen holding the avatar
  treats it as one with no voice audio model — text replies in live voice mode,
  no speak button in chat — without asking the voice route again.
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
        # Voice ids ElevenLabs has banned. Empty by default: a clone is in good
        # standing unless a test says otherwise.
        self.blocked_voice_ids = set()
        self.safety_lookups = []

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
            if voice_id in self.blocked_voice_ids:
                raise elevenlabs_client.ElevenLabsVoiceBlockedError(
                    f"ElevenLabs rejected the request (403 detected_blocked_voice: "
                    f"Voice '{voice_id}' may violate our Terms of Service and has "
                    f"been blocked.)"
                )
            return f"audio:{voice_id}:{text}".encode()

        async def voice_is_blocked(context, *, voice_id):
            self.safety_lookups.append(voice_id)
            return voice_id in self.blocked_voice_ids

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
async def test_the_instant_clone_is_created_once_at_the_minimum_and_never_rebuilt(
    monkeypatch,
):
    vendor = _FakeVendor().install(monkeypatch)
    repository = InMemoryMediaAssetRepository()
    await _add(repository, _context(), 45)
    record = await _add(repository, _context(), 30)
    assert record["instant_voice_id"] == "ivc-1"
    assert record["instant_voice_seconds"] == 75
    # More audio, even past the old 120 s target, never trains another voice.
    record = await _add(repository, _context(), 20)
    record = await _add(repository, _context(), 40)
    assert record["instant_voice_id"] == "ivc-1"
    assert vendor.instant == [("Evan (instant)", 2)]
    assert vendor.deleted == []
    assert record["collected_seconds"] == 135


@pytest.mark.asyncio
async def test_a_non_personal_avatar_keeps_its_clips_after_the_voice_exists(monkeypatch):
    _FakeVendor().install(monkeypatch)
    repository = InMemoryMediaAssetRepository()
    await _add(repository, _context(), 130)
    record = await _add(repository, _context(), 60)
    assert await repository.total_voice_seconds(ASSISTANT_ID) == 190
    assert len(repository.clips) == 2
    assert record["instant_voice_id"] == "ivc-1"


class _ReferenceStore:
    """A dictionary-backed stand-in for the LangGraph store."""

    def __init__(self):
        self.rows = {}

    async def aget(self, namespace, key):
        value = self.rows.get((namespace, key))
        return None if value is None else SimpleNamespace(value=value)

    async def aput(self, namespace, key, value):
        self.rows[(namespace, key)] = value


@pytest.mark.asyncio
async def test_voice_status_lists_clips_and_the_reference_document(monkeypatch):
    from src.anubis.utils.voice.reference_audio import store_reference_audio

    _FakeVendor().install(monkeypatch)
    repository = InMemoryMediaAssetRepository()
    store = _ReferenceStore()
    await corpus.add_voice_clip(
        repository,
        _context(),
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        audio_data_uri=CLIP,
        duration_seconds=12,
        source="reference_upload",
        source_document_name="Mom.m4a",
    )
    await store_reference_audio(
        store,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        audio_data_uri=CLIP,
        transcript_text="hello",
        filename="Mom.m4a",
        namespace_filename="mom-key",
        duration_seconds=3.0,
        source="upload",
    )
    status = await corpus.voice_status_for(
        repository,
        _context(),
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        is_personal_avatar=False,
        store=store,
    )
    assert status.reference_audio_document == "Mom.m4a"
    assert [clip["source_document_name"] for clip in status.clips] == ["Mom.m4a"]
    assert status.clips[0]["duration_seconds"] == 12
    assert corpus.voice_seconds_by_document(status.clips) == {"Mom.m4a": 12}


@pytest.mark.asyncio
async def test_forgetting_a_document_removes_its_clips_and_recomputes_seconds(
    monkeypatch,
):
    _FakeVendor().install(monkeypatch)
    repository = InMemoryMediaAssetRepository()
    for name, seconds in (("Mom.m4a", 40), ("talk.mp4", 50)):
        await corpus.add_voice_clip(
            repository,
            _context(),
            user_id=USER_ID,
            assistant_id=ASSISTANT_ID,
            audio_data_uri=CLIP,
            duration_seconds=seconds,
            source="media_upload",
            source_document_name=name,
        )
    record = await corpus.forget_document_clips(
        repository,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        source_document_name="Mom.m4a",
    )
    assert record["collected_seconds"] == 50
    assert record["instant_voice_id"] == "ivc-1"  # the trained voice stays
    assert len(repository.clips) == 1
    assert await corpus.longest_clip_for_document(
        repository, ASSISTANT_ID, "Mom.m4a"
    ) is None
    longest = await corpus.longest_clip_for_document(repository, ASSISTANT_ID, "talk.mp4")
    assert corpus.clip_data_uri(longest).startswith("data:audio/mpeg;base64,")


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


@pytest.mark.asyncio
async def test_a_plan_refusal_parks_the_professional_voice_until_retried(monkeypatch):
    """ElevenLabs offers professional cloning to the Creator plan and above.

    The refusal must not be retried on every later clip (each attempt is a
    vendor call that fails the same way), the instant voice must keep working,
    and an explicit retry after the upgrade must pick the flow back up.
    """
    vendor = _FakeVendor().install(monkeypatch)
    attempts = []

    async def _refuse(context, *, name, language="en", description=""):
        attempts.append(name)
        raise elevenlabs_client.ElevenLabsError(
            "Creating a PVC requires you to be on the Creator plan or above."
        )

    monkeypatch.setattr(elevenlabs_client, "create_professional_voice", _refuse)
    repository = InMemoryMediaAssetRepository()
    context = _context()
    record = await _add(repository, context, 1800, personal=True)
    assert record["professional_state"] == "plan_required"
    assert record["detail"]["professional_error_kind"] == "plan_required"
    assert "Creator plan" in record["detail"]["professional_error"]
    assert record["detail"]["professional_help_url"].startswith("https://elevenlabs.io/")
    assert record["instant_voice_id"] == "ivc-1"
    assert len(attempts) == 1

    # Later clips keep accumulating but do not knock on the vendor again.
    record = await _add(repository, context, 300, personal=True)
    assert record["professional_state"] == "plan_required"
    assert len(attempts) == 1

    status = await corpus.voice_status_for(
        repository,
        context,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        is_personal_avatar=True,
    )
    assert status.professional_state == "plan_required"
    assert status.detail["professional_error_kind"] == "plan_required"
    assert status.active_voice == "instant"

    # The owner upgrades the ElevenLabs account and retries.
    async def _accept(context, *, name, language="en", description=""):
        vendor.professional.append(name)
        return "pvc-1"

    monkeypatch.setattr(elevenlabs_client, "create_professional_voice", _accept)
    record = await corpus.retry_professional_voice(
        repository, context, user_id=USER_ID, assistant_id=ASSISTANT_ID, avatar_name="Evan"
    )
    assert record["professional_state"] == "awaiting_verification"
    assert record["professional_voice_id"] == "pvc-1"
    assert "professional_error" not in record["detail"]


def test_plan_refusal_detection_reads_the_vendor_wording():
    assert corpus.is_plan_refusal(
        "Creating a PVC requires you to be on the Creator plan or above."
    )
    assert not corpus.is_plan_refusal("Connection reset by peer")


@pytest.mark.asyncio
async def test_a_status_read_retries_a_failed_instant_clone(monkeypatch):
    """The clone is built when the corpus crosses the minimum; when the vendor
    refused that attempt, the next Voice panel read retries instead of waiting
    for another upload, and a persistent refusal is throttled."""
    vendor = _FakeVendor().install(monkeypatch)
    attempts = {"count": 0}

    async def _fail_once(context, *, name, clips, description=""):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise elevenlabs_client.ElevenLabsError(
                "ElevenLabs rejected the request (400 invalid_labels: Labels must be serialized dictionary object.)"
            )
        vendor.instant.append((name, sum(1 for _ in clips)))
        return "ivc-retry"

    monkeypatch.setattr(elevenlabs_client, "create_instant_voice", _fail_once)
    repository = InMemoryMediaAssetRepository()
    record = await _add(repository, _context(), 310)
    assert record["instant_voice_id"] is None
    assert "invalid_labels" in record["detail"]["instant_error"]
    assert record["detail"]["instant_error_at"] > 0

    # Too soon: the refusal is not retried on every refresh.
    status = await corpus.voice_status_for(
        repository,
        _context(),
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        is_personal_avatar=False,
    )
    assert status.instant_voice_id is None
    assert attempts["count"] == 1

    monkeypatch.setattr(corpus, "INSTANT_CLONE_RETRY_SECONDS", 0.0)
    status = await corpus.voice_status_for(
        repository,
        _context(),
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        is_personal_avatar=False,
    )
    assert attempts["count"] == 2
    assert status.instant_voice_id == "ivc-retry"
    assert "instant_error" not in status.detail

    # Built once: another read does not touch the vendor again.
    await corpus.voice_status_for(
        repository,
        _context(),
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        is_personal_avatar=False,
    )
    assert attempts["count"] == 2


@pytest.mark.asyncio
async def test_status_reports_a_clone_the_vendor_has_banned(monkeypatch):
    """The settings panel must not call a banned voice trained and available."""
    vendor = _FakeVendor().install(monkeypatch)
    repository = InMemoryMediaAssetRepository()
    await repository.upsert_voice(
        {
            "assistant_id": ASSISTANT_ID,
            "user_id": USER_ID,
            "instant_voice_id": "ivc-banned",
            "collected_seconds": 299.1,
        }
    )
    vendor.blocked_voice_ids.add("ivc-banned")

    status = await corpus.voice_status_for(
        repository,
        _context(),
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        is_personal_avatar=False,
    )

    assert vendor.safety_lookups == ["ivc-banned"]
    assert status.instant_voice_blocked is True
    assert status.instant_voice_blocked_reason
    # The id and the seconds survive: the corpus is still there, and the owner
    # is told which voice was refused.
    assert status.instant_voice_id == "ivc-banned"


@pytest.mark.asyncio
async def test_status_does_not_ask_the_vendor_again_straight_away(monkeypatch):
    """A healthy verdict is cached, so the panel is not a vendor call per load."""
    vendor = _FakeVendor().install(monkeypatch)
    repository = InMemoryMediaAssetRepository()
    await repository.upsert_voice(
        {
            "assistant_id": ASSISTANT_ID,
            "user_id": USER_ID,
            "instant_voice_id": "ivc-fine",
            "collected_seconds": 299.1,
        }
    )

    for _ in range(3):
        status = await corpus.voice_status_for(
            repository,
            _context(),
            user_id=USER_ID,
            assistant_id=ASSISTANT_ID,
            is_personal_avatar=False,
        )

    assert vendor.safety_lookups == ["ivc-fine"]
    assert status.instant_voice_blocked is False


@pytest.mark.asyncio
async def test_speak_refuses_a_banned_voice_without_calling_the_vendor(monkeypatch):
    """A ban already on the record is answered without paying for a 403."""
    from src.api import webapp as webapp_module

    vendor = _FakeVendor().install(monkeypatch)
    repository = InMemoryMediaAssetRepository()
    media_repository.set_media_asset_repository(repository)
    await repository.upsert_voice(
        {
            "assistant_id": ASSISTANT_ID,
            "user_id": USER_ID,
            "instant_voice_id": "ivc-banned",
            "detail": {"instant_blocked": True, "instant_blocked_reason": "banned"},
        }
    )
    vendor.blocked_voice_ids.add("ivc-banned")
    monkeypatch.setattr(
        webapp_module.app,
        "state",
        SimpleNamespace(context=_context(), pool=None, stripe=None),
    )
    monkeypatch.setattr(webapp_module, "enforce_tier_capability", lambda *a, **k: None)

    spoken = []

    async def _never_synthesize(*args, **kwargs):
        spoken.append(kwargs)
        raise AssertionError("a banned voice must not reach the vendor")

    monkeypatch.setattr(elevenlabs_client, "synthesize_speech", _never_synthesize)

    response = await webapp_module.speak_text(
        request=_json_request({"assistant_id": ASSISTANT_ID, "text": "hello"}),
        current_user={"API_KEY": "k", "identities": [{"user_id": USER_ID}]},
    )

    assert response.status_code == 409
    body = response.body.decode("utf-8")
    assert "voice_blocked" in body
    # Not the "record more speech" refusal: no amount of recording lifts a ban.
    assert "voice_not_ready" not in body
    assert spoken == []


@pytest.mark.asyncio
async def test_speak_records_a_ban_it_meets_at_synthesis(monkeypatch):
    """A ban applied after the last check is remembered, not re-paid for."""
    from src.api import webapp as webapp_module

    vendor = _FakeVendor().install(monkeypatch)
    repository = InMemoryMediaAssetRepository()
    media_repository.set_media_asset_repository(repository)
    await repository.upsert_voice(
        {
            "assistant_id": ASSISTANT_ID,
            "user_id": USER_ID,
            "instant_voice_id": "ivc-banned",
        }
    )
    vendor.blocked_voice_ids.add("ivc-banned")
    monkeypatch.setattr(
        webapp_module.app,
        "state",
        SimpleNamespace(context=_context(), pool=None, stripe=None),
    )
    monkeypatch.setattr(webapp_module, "enforce_tier_capability", lambda *a, **k: None)

    response = await webapp_module.speak_text(
        request=_json_request({"assistant_id": ASSISTANT_ID, "text": "hello"}),
        current_user={"API_KEY": "k", "identities": [{"user_id": USER_ID}]},
    )

    assert response.status_code == 409
    assert "voice_blocked" in response.body.decode("utf-8")
    stored = await repository.get_voice(ASSISTANT_ID)
    assert stored["detail"]["instant_blocked"] is True

    status = await corpus.voice_status_for(
        repository,
        _context(),
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        is_personal_avatar=False,
    )
    assert status.instant_voice_blocked is True


@pytest.mark.asyncio
async def test_a_ban_is_not_retried_like_a_failed_clone(monkeypatch):
    """A banned clone must not sit in the retry loop meant for vendor errors."""
    vendor = _FakeVendor().install(monkeypatch)
    repository = InMemoryMediaAssetRepository()

    async def _blocked_clone(context, *, name, clips, description=""):
        raise elevenlabs_client.ElevenLabsVoiceBlockedError("blocked at creation")

    monkeypatch.setattr(elevenlabs_client, "create_instant_voice", _blocked_clone)
    context = _context()
    await _add(repository, context, 90)

    record = await repository.get_voice(ASSISTANT_ID)
    assert record["detail"]["instant_blocked"] is True
    # ``instant_error`` is the retryable failure; a ban is not one of those.
    assert "instant_error" not in record["detail"]
    assert vendor.instant == []


class _FakeAssistants:
    """The subset of the LangGraph assistants client the note helper uses."""

    def __init__(self, metadata):
        self.metadata = dict(metadata)
        self.updates = []

    async def get(self, assistant_id):
        return {"assistant_id": assistant_id, "metadata": dict(self.metadata)}

    async def update(self, *, assistant_id, metadata):
        self.updates.append(metadata)
        # LangGraph merges metadata by key.
        self.metadata.update(metadata)
        return {"assistant_id": assistant_id, "metadata": dict(self.metadata)}


def _install_fake_client(monkeypatch, webapp_module, assistants):
    monkeypatch.setattr(
        webapp_module,
        "get_client",
        lambda *args, **kwargs: SimpleNamespace(assistants=assistants),
    )


@pytest.mark.asyncio
async def test_a_confirmed_ban_is_noted_on_the_avatar(monkeypatch):
    """The note is what makes every other screen fall back to text."""
    from src.api import webapp as webapp_module

    assistants = _FakeAssistants(
        {"user_id": USER_ID, "is_personal_avatar_of_creator": False}
    )
    _install_fake_client(monkeypatch, webapp_module, assistants)

    await webapp_module.note_blocked_voice_on_avatar(
        ASSISTANT_ID,
        {"API_KEY": "k", "identities": [{"user_id": USER_ID}]},
        "blocked by the vendor",
    )

    assert len(assistants.updates) == 1
    assert assistants.updates[0]["voice_model_blocked"] is True
    assert assistants.updates[0]["voice_model_blocked_reason"] == "blocked by the vendor"
    # The note is merged in; nothing else about the avatar is disturbed.
    assert assistants.metadata["user_id"] == USER_ID
    assert assistants.metadata["is_personal_avatar_of_creator"] is False


@pytest.mark.asyncio
async def test_the_note_is_written_once(monkeypatch):
    """An avatar already carrying the note costs a read and no write."""
    from src.api import webapp as webapp_module

    assistants = _FakeAssistants({"user_id": USER_ID, "voice_model_blocked": True})
    _install_fake_client(monkeypatch, webapp_module, assistants)

    await webapp_module.note_blocked_voice_on_avatar(
        ASSISTANT_ID, {"API_KEY": "k", "identities": [{"user_id": USER_ID}]}
    )

    assert assistants.updates == []


@pytest.mark.asyncio
async def test_a_failed_note_does_not_break_the_refusal(monkeypatch):
    """The note is advisory: speak must still answer 409 when it cannot be written."""
    from src.api import webapp as webapp_module

    vendor = _FakeVendor().install(monkeypatch)
    repository = InMemoryMediaAssetRepository()
    media_repository.set_media_asset_repository(repository)
    await repository.upsert_voice(
        {
            "assistant_id": ASSISTANT_ID,
            "user_id": USER_ID,
            "instant_voice_id": "ivc-banned",
        }
    )
    vendor.blocked_voice_ids.add("ivc-banned")
    monkeypatch.setattr(
        webapp_module.app,
        "state",
        SimpleNamespace(context=_context(), pool=None, stripe=None),
    )
    monkeypatch.setattr(webapp_module, "enforce_tier_capability", lambda *a, **k: None)

    def _unreachable_client(*args, **kwargs):
        raise RuntimeError("LangGraph is unreachable")

    monkeypatch.setattr(webapp_module, "get_client", _unreachable_client)

    response = await webapp_module.speak_text(
        request=_json_request({"assistant_id": ASSISTANT_ID, "text": "hello"}),
        current_user={"API_KEY": "k", "identities": [{"user_id": USER_ID}]},
    )

    assert response.status_code == 409
    assert "voice_blocked" in response.body.decode("utf-8")
    stored = await repository.get_voice(ASSISTANT_ID)
    assert stored["detail"]["instant_blocked"] is True
