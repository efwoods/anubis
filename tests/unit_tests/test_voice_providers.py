"""Unit tests for the switchable voice provider (ElevenLabs or Cartesia).

Pinned down:

- **One switch.** ``VOICE_PROVIDER`` picks the provider for stock voices,
  instant clones and speech; unset or unknown means ElevenLabs.
- **Cartesia requests have the documented shape** (``/tts/bytes``,
  ``/voices/clone``, ``/voices`` paging) and vendor failures map onto the shared
  voice errors the routes already answer.
- **Each provider keeps its own slot.** An ElevenLabs row reads exactly as
  before; a Cartesia clone and a Cartesia stock pick live under
  ``detail.provider_voices.cartesia``.
- **Switching is revertible.** A clone minted by the other provider keeps
  speaking until the active provider has one; a rebuild deletes only the active
  provider's copy; switching back restores the earlier voice untouched.
"""

import json
from types import SimpleNamespace

import httpx
import pytest

from src.anubis.utils.media_assets import repository as media_repository
from src.anubis.utils.media_assets.repository import InMemoryMediaAssetRepository
from src.anubis.utils.voice import corpus, elevenlabs_client, standard_voices
from src.anubis.utils.voice import providers as voice_providers
from src.anubis.utils.voice.provider_errors import (
    VoiceProviderCreditsExhaustedError,
    VoiceProviderError,
    VoiceProviderKeyRefusedError,
    VoiceProviderNotConfiguredError,
)
from src.anubis.utils.voice.providers import cartesia as cartesia_module
from src.anubis.utils.voice.voice_slots import (
    PROVIDER_VOICES_DETAIL_KEY,
    store_voice_slot,
    voice_slot,
)

USER_ID = "auth0-user"
ASSISTANT_ID = "assistant-1"


def _context(**overrides):
    values = dict(
        voice_provider="ELEVENLABS",
        elevenlabs_api_key="sk-eleven",
        cartesia_api_key="sk_car_test",
        cartesia_api_version="2026-08-14",
        cartesia_text_to_speech_model="sonic-3.6",
        cartesia_text_to_speech_cost_per_1000_characters_usd=0.04,
        cartesia_voice_language="en",
        cartesia_instant_voice_clone_maximum_seconds=60,
        elevenlabs_instant_voice_clone_minimum_seconds=60,
        elevenlabs_instant_voice_clone_target_seconds=120,
        elevenlabs_professional_voice_clone_minimum_seconds=1800,
        elevenlabs_professional_voice_clone_maximum_seconds=10800,
        elevenlabs_text_to_speech_model="eleven_flash_v2_5",
        elevenlabs_text_to_speech_cost_per_1000_characters_usd=0.05,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _cartesia_context(**overrides):
    return _context(voice_provider="CARTESIA", **overrides)


@pytest.fixture(autouse=True)
def _clear_state():
    media_repository.set_media_asset_repository(None)
    standard_voices.clear_catalogue_cache()
    yield
    media_repository.set_media_asset_repository(None)
    standard_voices.clear_catalogue_cache()


class _RecordingProvider:
    """A stand-in provider that records every call, for either vendor name."""

    def __init__(self, name, display_name, *, clone_cap_seconds=None):
        self.name = name
        self.display_name = display_name
        self.supports_professional_clone = name == "elevenlabs"
        self.clone_cap_seconds = clone_cap_seconds
        self.created = []
        self.deleted = []
        self.spoken = []
        self.stock = {
            "female": [
                {
                    "voice_id": f"{name}-stock-f",
                    "name": "Stock F",
                    "gender": "female",
                    "preview_requires_auth": name == "cartesia",
                }
            ],
            "male": [],
        }

    def is_configured(self, context):
        return bool(getattr(context, f"{self.name}_api_key", None))

    async def create_instant_voice(self, context, *, name, clips, description=""):
        self.created.append((name, len(clips)))
        return f"{self.name}-clone-{len(self.created)}"

    async def delete_voice(self, context, voice_id):
        self.deleted.append(voice_id)

    async def voice_is_blocked(self, context, *, voice_id):
        return False

    async def synthesize_speech(self, context, *, voice_id, text):
        self.spoken.append((voice_id, text))
        return f"{self.name}:{voice_id}:{text}".encode()

    async def list_stock_voices(self, context, *, gender):
        return list(self.stock.get(gender, []))

    async def stock_voice_preview(self, context, *, voice_id):
        return (b"sample", "audio/mpeg")

    def speech_model_name(self, context):
        return f"{self.name}-model"

    def speech_cost_per_1000_characters_usd(self, context):
        return 0.04 if self.name == "cartesia" else 0.05

    def instant_clone_max_seconds(self, context):
        return self.clone_cap_seconds


@pytest.fixture
def fake_providers(monkeypatch):
    from src.anubis.utils.voice.providers import elevenlabs as elevenlabs_module

    eleven = _RecordingProvider("elevenlabs", "ElevenLabs")
    cartesia = _RecordingProvider("cartesia", "Cartesia", clone_cap_seconds=60.0)
    monkeypatch.setattr(elevenlabs_module, "PROVIDER", eleven)
    monkeypatch.setattr(cartesia_module, "PROVIDER", cartesia)
    return SimpleNamespace(elevenlabs=eleven, cartesia=cartesia)


async def _add_clip(repository, seconds, *, created_at=None):
    await repository.add_voice_clip(
        {
            "user_id": USER_ID,
            "assistant_id": ASSISTANT_ID,
            "source": "recorder",
            "mime_type": "audio/mpeg",
            "bytes": b"speech",
            "duration_seconds": seconds,
        }
    )


# --------------------------------------------------------------------------
# The switch
# --------------------------------------------------------------------------


def test_the_active_provider_defaults_to_elevenlabs():
    assert voice_providers.active_voice_provider_name(None) == "elevenlabs"
    assert voice_providers.active_voice_provider_name(_context(voice_provider=None)) == (
        "elevenlabs"
    )
    assert voice_providers.active_voice_provider_name(_context(voice_provider="nope")) == (
        "elevenlabs"
    )


def test_the_switch_names_cartesia_in_any_case():
    assert voice_providers.active_voice_provider_name(_cartesia_context()) == "cartesia"
    assert (
        voice_providers.active_voice_provider_name(_context(voice_provider="cartesia"))
        == "cartesia"
    )
    assert voice_providers.get_voice_provider(_cartesia_context()).name == "cartesia"
    assert (
        voice_providers.get_voice_provider(_cartesia_context(), "elevenlabs").name
        == "elevenlabs"
    )


def test_voice_configured_follows_the_active_provider():
    assert corpus.voice_configured(_cartesia_context(elevenlabs_api_key=None))
    assert not corpus.voice_configured(_cartesia_context(cartesia_api_key=None))
    assert corpus.voice_configured(_context(cartesia_api_key=None))
    # Professional clones stay on ElevenLabs whatever the switch says.
    assert not corpus.professional_voice_configured(
        _cartesia_context(elevenlabs_api_key=None)
    )


def test_speaking_order_is_only_the_active_provider():
    # A switched-off provider never speaks, even while that provider's key is
    # still set: the key may be refused.
    assert voice_providers.speaking_provider_order(_cartesia_context()) == [
        "cartesia"
    ]
    assert voice_providers.speaking_provider_order(
        _cartesia_context(elevenlabs_api_key=None)
    ) == ["cartesia"]
    assert voice_providers.speaking_provider_order(_context()) == ["elevenlabs"]
    assert voice_providers.speaking_provider_order(None) == ["elevenlabs"]


def test_elevenlabs_errors_are_the_shared_voice_errors():
    assert issubclass(elevenlabs_client.ElevenLabsError, VoiceProviderError)
    assert issubclass(
        elevenlabs_client.ElevenLabsKeyRefusedError, VoiceProviderKeyRefusedError
    )
    assert issubclass(
        elevenlabs_client.ElevenLabsNotConfiguredError, VoiceProviderNotConfiguredError
    )


# --------------------------------------------------------------------------
# Cartesia request shapes
# --------------------------------------------------------------------------


def _install_transport(monkeypatch, handler):
    requests = []

    def recording_handler(request):
        requests.append(request)
        return handler(request)

    def _client(timeout_seconds):
        return httpx.AsyncClient(
            base_url=cartesia_module.CARTESIA_BASE_URL,
            transport=httpx.MockTransport(recording_handler),
        )

    monkeypatch.setattr(cartesia_module, "_http_client", _client)
    return requests


@pytest.mark.asyncio
async def test_cartesia_speech_request_shape(monkeypatch):
    requests = _install_transport(
        monkeypatch, lambda request: httpx.Response(200, content=b"mp3-bytes")
    )
    audio = await cartesia_module.PROVIDER.synthesize_speech(
        _cartesia_context(), voice_id="voice-1", text="hello"
    )
    assert audio == b"mp3-bytes"
    request = requests[0]
    assert request.method == "POST"
    assert request.url.path == "/tts/bytes"
    assert request.headers["authorization"] == "Bearer sk_car_test"
    assert request.headers["cartesia-version"] == "2026-08-14"
    body = json.loads(request.content)
    assert body["model_id"] == "sonic-3.6"
    assert body["transcript"] == "hello"
    assert body["voice"] == {"mode": "id", "id": "voice-1"}
    assert body["output_format"] == {
        "container": "mp3",
        "sample_rate": 44100,
        "bit_rate": 128000,
    }
    assert body["language"] == "en"


@pytest.mark.asyncio
async def test_cartesia_clone_sends_one_joined_clip(monkeypatch):
    joined = []

    def _join(clips, maximum_seconds):
        joined.append((len(clips), maximum_seconds))
        return b"joined-mp3", 58.0

    monkeypatch.setattr(cartesia_module, "join_clips_to_mp3", _join)
    requests = _install_transport(
        monkeypatch, lambda request: httpx.Response(200, json={"id": "car-voice-7"})
    )
    voice_id = await cartesia_module.PROVIDER.create_instant_voice(
        _cartesia_context(),
        name="Claire (instant)",
        clips=[("a.mp3", b"a", "audio/mpeg"), ("b.mp3", b"b", "audio/mpeg")],
        description="Neural Nexus instant voice clone",
    )
    assert voice_id == "car-voice-7"
    assert joined == [(2, 60.0)]
    request = requests[0]
    assert request.url.path == "/voices/clone"
    multipart_body = request.content.decode("latin-1")
    assert 'name="clip"; filename="voice.mp3"' in multipart_body
    assert "joined-mp3" in multipart_body
    assert 'name="name"' in multipart_body and "Claire (instant)" in multipart_body
    assert 'name="language"' in multipart_body


@pytest.mark.asyncio
async def test_cartesia_stock_voices_page_and_keep_only_public_voices(monkeypatch):
    pages = {
        None: {
            "data": [
                {"id": "v1", "name": "Nova", "gender": "feminine"},
                {"id": "v2", "name": "Mine", "gender": "feminine", "is_owner": True},
            ],
            "has_more": True,
        },
        "v2": {
            "data": [
                {"id": "v3", "name": "Atlas", "gender": "masculine"},
                {
                    "id": "v4",
                    "name": "Private",
                    "gender": "feminine",
                    "access": "private",
                },
                {
                    "id": "v5",
                    "name": "Iris",
                    "gender": "feminine",
                    "preview_file_url": "https://api.cartesia.ai/f",
                },
            ],
            "has_more": False,
        },
    }
    requests = _install_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200, json=pages[request.url.params.get("starting_after")]
        ),
    )
    female = await cartesia_module.PROVIDER.list_stock_voices(
        _cartesia_context(), gender="female"
    )
    assert [voice["voice_id"] for voice in female] == ["v1", "v5"]
    # Every Cartesia sample is served by the API, with or without a preview file.
    assert female[1]["preview_requires_auth"] is True
    assert female[0]["preview_requires_auth"] is True
    assert len(requests) == 2
    assert requests[0].url.params["language"] == "en"


@pytest.mark.asyncio
async def test_a_cartesia_voice_without_a_preview_file_gets_a_synthesized_sample(
    monkeypatch,
):
    monkeypatch.setattr(cartesia_module, "_synthesized_stock_voice_samples", {})

    def handler(request):
        if request.url.path == "/tts/bytes":
            return httpx.Response(200, content=b"synthesized-sample")
        return httpx.Response(200, json={"id": "v1", "preview_file_url": None})

    requests = _install_transport(monkeypatch, handler)
    for _ in range(2):
        preview = await cartesia_module.PROVIDER.stock_voice_preview(
            _cartesia_context(), voice_id="v1"
        )
        assert preview == (b"synthesized-sample", "audio/mpeg")
    speech_requests = [
        request for request in requests if request.url.path == "/tts/bytes"
    ]
    assert len(speech_requests) == 1
    body = json.loads(speech_requests[0].content)
    assert body["transcript"] == cartesia_module.STOCK_VOICE_SAMPLE_TEXT
    assert body["voice"] == {"mode": "id", "id": "v1"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "body", "error_class"),
    [
        (401, "invalid api key", VoiceProviderKeyRefusedError),
        (402, "payment required", VoiceProviderCreditsExhaustedError),
        (429, "credit limit reached", VoiceProviderCreditsExhaustedError),
        (500, "internal error", VoiceProviderError),
    ],
)
async def test_cartesia_failures_map_to_shared_errors(
    monkeypatch, status_code, body, error_class
):
    _install_transport(
        monkeypatch, lambda request: httpx.Response(status_code, text=body)
    )
    with pytest.raises(error_class):
        await cartesia_module.PROVIDER.synthesize_speech(
            _cartesia_context(), voice_id="voice-1", text="hello"
        )


@pytest.mark.asyncio
async def test_a_cartesia_timeout_is_a_vendor_error(monkeypatch):
    real_async_client = httpx.AsyncClient

    def _timing_out_handler(request):
        raise httpx.ReadTimeout("no response", request=request)

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **client_arguments: real_async_client(
            transport=httpx.MockTransport(_timing_out_handler), **client_arguments
        ),
    )
    with pytest.raises(VoiceProviderError, match="could not be reached"):
        await cartesia_module.PROVIDER.synthesize_speech(
            _cartesia_context(), voice_id="voice-1", text="hello"
        )


@pytest.mark.asyncio
async def test_cartesia_without_a_key_is_not_configured():
    with pytest.raises(VoiceProviderNotConfiguredError):
        await cartesia_module.PROVIDER.synthesize_speech(
            _cartesia_context(cartesia_api_key=None), voice_id="voice-1", text="hi"
        )


@pytest.mark.asyncio
async def test_cartesia_delete_never_raises(monkeypatch):
    _install_transport(monkeypatch, lambda request: httpx.Response(404, text="gone"))
    await cartesia_module.PROVIDER.delete_voice(_cartesia_context(), "voice-1")


# --------------------------------------------------------------------------
# Slots
# --------------------------------------------------------------------------


def test_the_elevenlabs_slot_is_the_row_itself():
    record = {"instant_voice_id": "ivc-1", "detail": {"instant_error": "x"}}
    assert voice_slot(record, "elevenlabs") is record
    assert store_voice_slot(record, "elevenlabs", record) is record


def test_a_cartesia_slot_round_trips_through_the_detail():
    record = {"instant_voice_id": "ivc-1", "detail": {"voice_choice": "custom"}}
    slot = voice_slot(record, "cartesia")
    assert slot["instant_voice_id"] is None
    slot["instant_voice_id"] = "car-1"
    slot["instant_voice_seconds"] = 60.0
    slot["detail"]["standard_voice"] = {"voice_id": "car-std"}
    store_voice_slot(record, "cartesia", slot)
    assert record["instant_voice_id"] == "ivc-1"
    assert record["detail"]["voice_choice"] == "custom"
    stored = record["detail"][PROVIDER_VOICES_DETAIL_KEY]["cartesia"]
    assert stored["instant_voice_id"] == "car-1"
    assert voice_slot(record, "cartesia")["detail"]["standard_voice"] == {
        "voice_id": "car-std"
    }


# --------------------------------------------------------------------------
# Clones and the revert round trip
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_cartesia_clone_lands_in_the_cartesia_slot(fake_providers):
    repository = InMemoryMediaAssetRepository()
    await repository.upsert_voice(
        {"assistant_id": ASSISTANT_ID, "user_id": USER_ID, "instant_voice_id": "ivc-el"}
    )
    for _ in range(3):
        await _add_clip(repository, 40)

    record = await corpus.ensure_instant_voice(
        repository, _cartesia_context(), user_id=USER_ID, assistant_id=ASSISTANT_ID
    )

    assert record["instant_voice_id"] == "ivc-el"
    cartesia_slot = voice_slot(record, "cartesia")
    assert cartesia_slot["instant_voice_id"] == "cartesia-clone-1"
    # Cartesia uses about a minute: two 40-second clips cover the cap.
    assert fake_providers.cartesia.created == [("assistant-1 (instant)", 2)]
    assert cartesia_slot["instant_voice_seconds"] == 60.0
    assert fake_providers.elevenlabs.created == []


@pytest.mark.asyncio
async def test_switching_providers_is_revertible(fake_providers):
    repository = InMemoryMediaAssetRepository()
    await repository.upsert_voice(
        {"assistant_id": ASSISTANT_ID, "user_id": USER_ID, "instant_voice_id": "ivc-el"}
    )
    for _ in range(3):
        await _add_clip(repository, 40)
    eleven_context = _context()
    cartesia_context = _cartesia_context()

    # Switched to Cartesia before a Cartesia clone exists: the first speak
    # builds the Cartesia clone from the stored speech and speaks through
    # Cartesia, never through ElevenLabs.
    speaking = await corpus.resolve_speaking_voice_and_provider(
        repository, ASSISTANT_ID, cartesia_context
    )
    assert (speaking.kind, speaking.voice_id, speaking.provider_name) == (
        "instant",
        "cartesia-clone-1",
        "cartesia",
    )

    # A status read reports the Cartesia clone without building another.
    status = await corpus.voice_status_for(
        repository,
        cartesia_context,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        is_personal_avatar=False,
    )
    assert status.voice_provider == "cartesia"
    assert status.voice_provider_display_name == "Cartesia"
    assert status.instant_voice_id == "cartesia-clone-1"
    assert status.speaking_voice_provider == "cartesia"

    # Rebuilding on Cartesia replaces only the Cartesia clone.
    await corpus.rebuild_instant_voice(
        repository, cartesia_context, user_id=USER_ID, assistant_id=ASSISTANT_ID
    )
    assert fake_providers.cartesia.deleted == ["cartesia-clone-1"]
    assert fake_providers.elevenlabs.deleted == []

    # Switching back restores the ElevenLabs clone untouched.
    speaking = await corpus.resolve_speaking_voice_and_provider(
        repository, ASSISTANT_ID, eleven_context
    )
    assert (speaking.voice_id, speaking.provider_name) == ("ivc-el", "elevenlabs")
    stored = await repository.get_voice(ASSISTANT_ID)
    assert voice_slot(stored, "cartesia")["instant_voice_id"] == "cartesia-clone-2"
    assert fake_providers.elevenlabs.created == []


@pytest.mark.asyncio
async def test_another_providers_voice_is_skipped_once_its_key_is_gone(fake_providers):
    repository = InMemoryMediaAssetRepository()
    await repository.upsert_voice(
        {"assistant_id": ASSISTANT_ID, "user_id": USER_ID, "instant_voice_id": "ivc-el"}
    )
    speaking = await corpus.resolve_speaking_voice_and_provider(
        repository, ASSISTANT_ID, _cartesia_context(elevenlabs_api_key=None)
    )
    assert speaking.kind == "none"


# --------------------------------------------------------------------------
# Standard voices
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_cartesia_stock_pick_leaves_the_elevenlabs_pick_in_place(
    fake_providers,
):
    repository = InMemoryMediaAssetRepository()
    await repository.upsert_voice(
        {
            "assistant_id": ASSISTANT_ID,
            "user_id": USER_ID,
            "detail": {
                "standard_voice": {"voice_id": "el-std", "name": "Rachel"},
                "voice_choice": "standard",
            },
        }
    )
    cartesia_context = _cartesia_context()

    catalogue = await standard_voices.list_standard_voices(
        cartesia_context, gender="female"
    )
    assert [voice["voice_id"] for voice in catalogue] == ["cartesia-stock-f"]
    assert catalogue[0]["preview_requires_auth"] is True

    await standard_voices.set_standard_voice(
        repository,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        voice=catalogue[0],
        context=cartesia_context,
    )
    record = await repository.get_voice(ASSISTANT_ID)
    assert record["detail"]["standard_voice"]["voice_id"] == "el-std"
    assert standard_voices.standard_voice_of(record, "cartesia")["voice_id"] == (
        "cartesia-stock-f"
    )
    speaking = corpus.speaking_voice_of(record, cartesia_context)
    assert (speaking.kind, speaking.voice_id, speaking.provider_name) == (
        "standard",
        "cartesia-stock-f",
        "cartesia",
    )
    speaking = corpus.speaking_voice_of(record, _context())
    assert (speaking.voice_id, speaking.provider_name) == ("el-std", "elevenlabs")


@pytest.mark.asyncio
async def test_a_clone_less_avatar_speaks_a_cartesia_stock_voice_after_the_switch(
    fake_providers,
):
    repository = InMemoryMediaAssetRepository()
    await repository.upsert_voice(
        {
            "assistant_id": ASSISTANT_ID,
            "user_id": USER_ID,
            "detail": {
                "standard_voice": {
                    "voice_id": "el-std",
                    "name": "Alice",
                    "gender": "female",
                }
            },
        }
    )

    speaking = await corpus.resolve_speaking_voice_and_provider(
        repository, ASSISTANT_ID, _cartesia_context()
    )
    assert (speaking.kind, speaking.voice_id, speaking.provider_name) == (
        "standard",
        "cartesia-stock-f",
        "cartesia",
    )
    stored = await repository.get_voice(ASSISTANT_ID)
    assert stored["detail"]["standard_voice"]["voice_id"] == "el-std"
    assert "voice_choice" not in stored["detail"]
    assert standard_voices.standard_voice_of(stored, "cartesia") == {
        "voice_id": "cartesia-stock-f",
        "name": "Stock F",
        "gender": "female",
    }

    # Switching back speaks the ElevenLabs pick again.
    speaking = await corpus.resolve_speaking_voice_and_provider(
        repository, ASSISTANT_ID, _context()
    )
    assert (speaking.voice_id, speaking.provider_name) == ("el-std", "elevenlabs")


@pytest.mark.asyncio
async def test_a_failed_catalogue_read_never_speaks_through_the_earlier_provider(
    fake_providers, monkeypatch
):
    async def _refuse(context, *, gender):
        raise RuntimeError("catalogue unavailable")

    monkeypatch.setattr(fake_providers.cartesia, "list_stock_voices", _refuse)
    repository = InMemoryMediaAssetRepository()
    await repository.upsert_voice(
        {
            "assistant_id": ASSISTANT_ID,
            "user_id": USER_ID,
            "detail": {"standard_voice": {"voice_id": "el-std", "gender": "female"}},
        }
    )
    speaking = await corpus.resolve_speaking_voice_and_provider(
        repository, ASSISTANT_ID, _cartesia_context()
    )
    assert speaking.kind == "none"
    stored = await repository.get_voice(ASSISTANT_ID)
    assert standard_voices.standard_voice_of(stored, "cartesia") is None
    # The ElevenLabs pick stays stored for a switch back.
    assert standard_voices.standard_voice_of(stored, "elevenlabs")["voice_id"] == (
        "el-std"
    )


@pytest.mark.asyncio
async def test_a_professional_clone_does_not_speak_after_switching_to_cartesia(
    fake_providers,
):
    repository = InMemoryMediaAssetRepository()
    await repository.upsert_voice(
        {
            "assistant_id": ASSISTANT_ID,
            "user_id": USER_ID,
            "professional_state": corpus.VOICE_STATE_FINE_TUNED,
            "professional_voice_id": "pvc-el",
        }
    )
    for _ in range(3):
        await _add_clip(repository, 40)

    speaking = await corpus.resolve_speaking_voice_and_provider(
        repository, ASSISTANT_ID, _cartesia_context()
    )
    assert (speaking.kind, speaking.voice_id, speaking.provider_name) == (
        "instant",
        "cartesia-clone-1",
        "cartesia",
    )

    speaking = await corpus.resolve_speaking_voice_and_provider(
        repository, ASSISTANT_ID, _context()
    )
    assert (speaking.kind, speaking.voice_id, speaking.provider_name) == (
        "professional",
        "pvc-el",
        "elevenlabs",
    )


@pytest.mark.asyncio
async def test_the_catalogue_cache_is_kept_per_provider(fake_providers):
    eleven_catalogue = await standard_voices.list_standard_voices(
        _context(), gender="female"
    )
    cartesia_catalogue = await standard_voices.list_standard_voices(
        _cartesia_context(), gender="female"
    )
    assert eleven_catalogue[0]["voice_id"] == "elevenlabs-stock-f"
    assert cartesia_catalogue[0]["voice_id"] == "cartesia-stock-f"


# --------------------------------------------------------------------------
# The speak route
# --------------------------------------------------------------------------


def _json_request(payload):
    async def _json():
        return payload

    return SimpleNamespace(json=_json)


@pytest.mark.asyncio
async def test_speak_synthesizes_with_the_provider_that_minted_the_voice(
    monkeypatch, fake_providers
):
    from src.api import webapp as webapp_module

    repository = InMemoryMediaAssetRepository()
    media_repository.set_media_asset_repository(repository)
    record = {"assistant_id": ASSISTANT_ID, "user_id": USER_ID, "detail": {}}
    slot = voice_slot(record, "cartesia")
    slot["instant_voice_id"] = "car-9"
    store_voice_slot(record, "cartesia", slot)
    await repository.upsert_voice(record)
    monkeypatch.setattr(
        webapp_module.app,
        "state",
        SimpleNamespace(context=_cartesia_context(), pool=None, stripe=None),
    )
    monkeypatch.setattr(webapp_module, "enforce_tier_capability", lambda *a, **k: None)
    recorded = {}

    async def _meter(current_user, **kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(webapp_module, "_meter_speech_characters", _meter)

    response = await webapp_module.speak_text(
        request=_json_request({"assistant_id": ASSISTANT_ID, "text": "hello"}),
        current_user={"API_KEY": "k", "identities": [{"user_id": USER_ID}]},
    )
    assert response.status_code == 200
    assert response.body == b"cartesia:car-9:hello"
    assert response.headers["x-voice-kind"] == "instant"
    assert response.headers["x-voice-provider"] == "cartesia"
    assert recorded["model_name"] == "cartesia-model"
    assert recorded["cost_usd"] == pytest.approx(0.04 * 5 / 1000)


@pytest.mark.asyncio
async def test_speak_builds_and_speaks_the_cartesia_clone_of_an_elevenlabs_only_avatar(
    monkeypatch, fake_providers
):
    from src.api import webapp as webapp_module

    repository = InMemoryMediaAssetRepository()
    media_repository.set_media_asset_repository(repository)
    await repository.upsert_voice(
        {"assistant_id": ASSISTANT_ID, "user_id": USER_ID, "instant_voice_id": "ivc-el"}
    )
    for _ in range(3):
        await _add_clip(repository, 40)
    monkeypatch.setattr(
        webapp_module.app,
        "state",
        SimpleNamespace(context=_cartesia_context(), pool=None, stripe=None),
    )
    monkeypatch.setattr(webapp_module, "enforce_tier_capability", lambda *a, **k: None)

    async def _meter(current_user, **kwargs):
        return None

    monkeypatch.setattr(webapp_module, "_meter_speech_characters", _meter)

    response = await webapp_module.speak_text(
        request=_json_request({"assistant_id": ASSISTANT_ID, "text": "hello"}),
        current_user={"API_KEY": "k", "identities": [{"user_id": USER_ID}]},
    )
    assert response.status_code == 200
    assert response.body == b"cartesia:cartesia-clone-1:hello"
    assert response.headers["x-voice-provider"] == "cartesia"
    stored = await repository.get_voice(ASSISTANT_ID)
    assert stored["instant_voice_id"] == "ivc-el"


@pytest.mark.asyncio
async def test_the_preview_route_serves_a_sample_that_needs_the_key(
    monkeypatch, fake_providers
):
    from src.api import webapp as webapp_module

    monkeypatch.setattr(
        webapp_module.app,
        "state",
        SimpleNamespace(context=_cartesia_context(), pool=None, stripe=None),
    )
    response = await webapp_module.get_standard_voice_preview(
        voice_id="cartesia-stock-f",
        current_user={"identities": [{"user_id": USER_ID}]},
    )
    assert response.status_code == 200
    assert response.body == b"sample"
    assert response.media_type == "audio/mpeg"


# --------------------------------------------------------------------------
# Cartesia catalogue duplicates and the configured standard voices
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cartesia_emotion_variants_collapse_to_the_plain_voice(monkeypatch):
    luke_description = "Seasoned male voice for casual, authentic conversations"
    carson_description = "Friendly young adult male for customer support conversations"
    page = {
        "data": [
            {
                "id": "luke-angry",
                "name": "Luke",
                "gender": "masculine",
                "tagline": "Angry Broadway Voice",
                "description": luke_description,
                "created_at": "2026-01-09T00:00:00Z",
                "accents": [{"locale": "en-US"}],
            },
            {
                "id": "luke-plain-newer",
                "name": "Luke",
                "gender": "masculine",
                "tagline": "Broadway Voice",
                "description": luke_description,
                "created_at": "2026-01-09T00:00:00Z",
                "accents": [{"locale": "en-US"}],
            },
            {
                "id": "luke-plain-oldest",
                "name": "Luke",
                "gender": "masculine",
                "tagline": "Broadway Voice",
                "description": luke_description,
                "created_at": "2026-01-05T00:00:00Z",
                "accents": [{"locale": "en-US"}],
            },
            {
                "id": "carson-multilingual",
                "name": "Carson",
                "gender": "masculine",
                "tagline": "Friendly Support",
                "description": "Friendly, young adult male for customer support conversations",
                "created_at": "2025-05-05T00:00:00Z",
                "accents": [{"locale": "en-US"}, {"locale": "en-GB"}],
            },
            {
                "id": "carson-american",
                "name": "Carson",
                "gender": "masculine",
                "tagline": "Friendly Support",
                "description": carson_description,
                "created_at": "2025-05-05T00:00:00Z",
                "accents": [{"locale": "en-US"}],
            },
            {
                "id": "benedict-royal",
                "name": "Benedict",
                "gender": "masculine",
                "tagline": "Royal Narrator",
                "description": "Confident, firm male for narrations",
            },
            {
                "id": "benedict-mediator",
                "name": "Benedict",
                "gender": "masculine",
                "tagline": "Measured Mediator",
                "description": "Polished, and formal British male.",
            },
        ],
        "has_more": False,
    }
    _install_transport(monkeypatch, lambda request: httpx.Response(200, json=page))
    male = await cartesia_module.PROVIDER.list_stock_voices(
        _cartesia_context(), gender="male"
    )
    assert [voice["voice_id"] for voice in male] == [
        "luke-plain-oldest",
        "carson-american",
        "benedict-royal",
        "benedict-mediator",
    ]


@pytest.mark.asyncio
async def test_the_configured_standard_voice_leads_the_catalogue(
    fake_providers, monkeypatch
):
    fake_providers.cartesia.stock["female"] = [
        {"voice_id": "aila", "name": "Aila", "gender": "female"},
        {"voice_id": "ailsa", "name": "Ailsa", "gender": "female"},
    ]
    monkeypatch.setattr(
        fake_providers.cartesia,
        "default_stock_voice_id",
        cartesia_module.CartesiaVoiceProvider().default_stock_voice_id,
        raising=False,
    )
    cartesia_context = _cartesia_context(cartesia_standard_female_voice_id="ailsa")
    catalogue = await standard_voices.list_standard_voices(
        cartesia_context, gender="female"
    )
    assert [(voice["voice_id"], voice["is_default"]) for voice in catalogue] == [
        ("ailsa", True),
        ("aila", False),
    ]

    repository = InMemoryMediaAssetRepository()
    await repository.upsert_voice(
        {
            "assistant_id": ASSISTANT_ID,
            "user_id": USER_ID,
            "detail": {"standard_voice": {"voice_id": "el-std", "gender": "female"}},
        }
    )
    speaking = await corpus.resolve_speaking_voice_and_provider(
        repository, ASSISTANT_ID, cartesia_context
    )
    assert (speaking.voice_id, speaking.provider_name) == ("ailsa", "cartesia")
