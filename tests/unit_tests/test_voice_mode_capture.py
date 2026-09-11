"""Unit tests for growing a personal avatar's voice from voice mode.

Pinned down:

- **The reference audio clip is the gate.** With no stored reference nothing
  accrues on either path, because the reference is the only thing that can tell
  the owner apart from a third party or from background chatter in the room.
- **No utterance ever creates the reference.** A first utterance must not seed
  it: a quiet owner in a room with a television would otherwise make a
  stranger's voice the avatar's voice, and the instant clone is built once and
  never rebuilt on its own.
- **Only a real match accrues.** ``claim_lone_speaker_as_owner`` relabels a lone
  voice as the owner so a monologue is answered rather than triaged; accrual
  reads ``owner_matched_reference`` instead, which records whether the diarizer
  matched the reference.
- **Only the owner's windows are cut.** Avatar echo and every ``Speaker N``
  segment are left behind, so a conversation held in a room with other people
  contributes the owner's half and nothing else.
- **Dictation is diarized against the reference too.** Whisper reports words
  and not who said them, so being the signed-in owner is not evidence that the
  voice on the microphone is the owner's; that path diarizes before keeping
  anything, and keeps only what matched the reference.
- **Consent is asked once and honoured**, and a non-personal avatar never
  accrues the owner's voice at all.
- **The instant clone is refreshed once** when the corpus first reaches the
  target, which is what that setting's description has always claimed.
"""

import base64
from types import SimpleNamespace

import pytest

from src.anubis.utils.media_assets.repository import InMemoryMediaAssetRepository
from src.anubis.utils.voice import capture, corpus
from src.anubis.utils.voice.speakers import LabelledSegment, SpokenTurn

USER_ID = "auth0-user"
ASSISTANT_ID = "assistant-1"
AUDIO = "data:audio/mpeg;base64," + base64.b64encode(b"speech").decode()


def _context(**overrides):
    values = dict(
        elevenlabs_api_key="sk-test",
        elevenlabs_instant_voice_clone_minimum_seconds=60,
        elevenlabs_instant_voice_clone_target_seconds=120,
        elevenlabs_professional_voice_clone_minimum_seconds=1800,
        elevenlabs_professional_voice_clone_maximum_seconds=10800,
        voice_mode_capture_enabled="TRUE",
        voice_mode_capture_min_segment_seconds=1.0,
        voice_mode_capture_max_seconds_per_turn=30.0,
        instant_voice_refresh_at_target_enabled="TRUE",
        reference_audio_clip_max_seconds=10.0,
        reference_audio_minimum_seconds=1.3,
        audio_diarization_known_speaker_name="avatar",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class _Store:
    """The smallest store that answers a reference-audio read."""

    def __init__(self, reference_seconds: float | None = None):
        self.value = None
        self.writes = []
        if reference_seconds is not None:
            self.value = {
                "reference_audio_data": AUDIO,
                "document": {
                    "kwargs": {
                        "page_content": "a sentence the owner read",
                        "metadata": {
                            "filename": "reference.mp3",
                            "duration": reference_seconds,
                        },
                    }
                },
            }

    async def aget(self, namespace, key):
        if self.value is None:
            return None
        return SimpleNamespace(value=self.value)

    async def aput(self, namespace, key, value):
        self.writes.append((namespace, key, value))


def _turn(segments, *, matched=True, audio=AUDIO):
    return SpokenTurn(
        script="",
        segments=segments,
        owner_label="Evan",
        owner_identified=True,
        other_speakers=[],
        duration_seconds=sum(s.end - s.start for s in segments),
        audio_data_uri=audio,
        owner_matched_reference=matched,
    )


def _owner(start, end):
    return LabelledSegment("Evan", "words", start, end, is_owner=True)


def _other(start, end, label="Speaker 2"):
    return LabelledSegment(label, "words", start, end)


def _echo(start, end):
    return LabelledSegment("Evan (avatar)", "words", start, end, is_avatar=True)


@pytest.fixture
def cut_calls(monkeypatch):
    """Capture the windows handed to the cutter instead of running ffmpeg."""
    calls = []

    async def fake_cut(audio_data_uri, turns):
        calls.append(turns)
        seconds = sum(float(t["end"]) - float(t["start"]) for t in turns)
        return (AUDIO, seconds) if seconds > 0 else (None, 0.0)

    monkeypatch.setattr(
        "src.anubis.utils.voice.clips.cut_target_turns_to_mp3_data_uri", fake_cut
    )
    return calls


async def _grant(repository):
    await capture.set_consent(
        repository, user_id=USER_ID, assistant_id=ASSISTANT_ID, granted=True
    )


async def _accrue(repository, context, store, turn, *, personal=True):
    return await capture.accrue_voice_from_spoken_turn(
        turn,
        repository,
        context,
        store,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        is_personal_avatar=personal,
        avatar_name="Evan",
        thread_id="thread-1",
    )


@pytest.mark.asyncio
async def test_no_reference_audio_accrues_nothing(cut_calls):
    """The gate: without a reference clip the avatar cannot attribute a voice."""
    repository = InMemoryMediaAssetRepository()
    await _grant(repository)
    store = _Store(reference_seconds=None)

    result = await _accrue(repository, _context(), store, _turn([_owner(0.0, 5.0)]))

    assert result is None
    assert cut_calls == []
    assert await repository.total_voice_seconds(ASSISTANT_ID) == 0


@pytest.mark.asyncio
async def test_reference_in_place_accrues_the_owner(cut_calls):
    repository = InMemoryMediaAssetRepository()
    await _grant(repository)
    store = _Store(reference_seconds=4.0)

    await _accrue(repository, _context(), store, _turn([_owner(0.0, 5.0)]))

    assert await repository.total_voice_seconds(ASSISTANT_ID) == pytest.approx(5.0)
    clips = await repository.list_voice_clips(ASSISTANT_ID)
    assert [clip["source"] for clip in clips] == [capture.CAPTURE_SOURCE_SPOKEN_TURN]
    # The document name carries the thread, so one conversation's contribution
    # can be dropped again and the clone rebuilt from what is left.
    assert "thread-1" in clips[0]["source_document_name"]


@pytest.mark.asyncio
async def test_accrual_never_writes_a_reference(cut_calls):
    """No utterance seeds the anchor — it is supplied deliberately or not at all."""
    repository = InMemoryMediaAssetRepository()
    await _grant(repository)
    store = _Store(reference_seconds=4.0)

    await _accrue(repository, _context(), store, _turn([_owner(0.0, 5.0)]))

    assert store.writes == []


@pytest.mark.asyncio
async def test_claimed_lone_voice_does_not_accrue(cut_calls):
    """A lone voice claimed as the owner may be a television, so it is not kept."""
    repository = InMemoryMediaAssetRepository()
    await _grant(repository)
    store = _Store(reference_seconds=4.0)

    result = await _accrue(
        repository, _context(), store, _turn([_owner(0.0, 5.0)], matched=False)
    )

    assert result is None
    assert await repository.total_voice_seconds(ASSISTANT_ID) == 0


@pytest.mark.asyncio
async def test_only_owner_windows_are_cut(cut_calls):
    repository = InMemoryMediaAssetRepository()
    await _grant(repository)
    store = _Store(reference_seconds=4.0)
    turn = _turn([_owner(0.0, 3.0), _other(3.0, 9.0), _echo(9.0, 11.0), _owner(11.0, 14.0)])

    await _accrue(repository, _context(), store, turn)

    assert cut_calls == [[
        {"is_target": True, "start": 0.0, "end": 3.0},
        {"is_target": True, "start": 11.0, "end": 14.0},
    ]]


@pytest.mark.asyncio
async def test_a_turn_of_only_other_voices_accrues_nothing(cut_calls):
    repository = InMemoryMediaAssetRepository()
    await _grant(repository)
    store = _Store(reference_seconds=4.0)

    result = await _accrue(
        repository, _context(), store, _turn([_other(0.0, 8.0), _other(8.0, 12.0, "Speaker 3")])
    )

    assert result is None
    assert cut_calls == []


@pytest.mark.asyncio
async def test_short_windows_are_dropped_and_long_turns_capped(cut_calls):
    repository = InMemoryMediaAssetRepository()
    await _grant(repository)
    store = _Store(reference_seconds=4.0)
    context = _context(voice_mode_capture_max_seconds_per_turn=10.0)
    turn = _turn([_owner(0.0, 0.4), _owner(1.0, 9.0), _owner(9.0, 20.0)])

    await _accrue(repository, context, store, turn)

    windows = cut_calls[0]
    assert {"is_target": True, "start": 0.0, "end": 0.4} not in windows
    assert sum(w["end"] - w["start"] for w in windows) == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_consent_is_required_and_can_be_declined(cut_calls):
    repository = InMemoryMediaAssetRepository()
    store = _Store(reference_seconds=4.0)
    context = _context()

    # Never asked.
    assert await _accrue(repository, context, store, _turn([_owner(0.0, 5.0)])) is None

    await capture.set_consent(
        repository, user_id=USER_ID, assistant_id=ASSISTANT_ID, granted=False
    )
    assert await _accrue(repository, context, store, _turn([_owner(0.0, 5.0)])) is None
    assert await repository.total_voice_seconds(ASSISTANT_ID) == 0

    await _grant(repository)
    await _accrue(repository, context, store, _turn([_owner(0.0, 5.0)]))
    assert await repository.total_voice_seconds(ASSISTANT_ID) == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_a_non_personal_avatar_never_accrues(cut_calls):
    """The owner's voice is the avatar's voice for the personal avatar alone."""
    repository = InMemoryMediaAssetRepository()
    await _grant(repository)
    store = _Store(reference_seconds=4.0)

    result = await _accrue(
        repository, _context(), store, _turn([_owner(0.0, 5.0)]), personal=False
    )

    assert result is None
    assert await repository.total_voice_seconds(ASSISTANT_ID) == 0


@pytest.mark.asyncio
async def test_capture_can_be_switched_off_for_the_deployment(cut_calls):
    repository = InMemoryMediaAssetRepository()
    await _grant(repository)
    store = _Store(reference_seconds=4.0)

    result = await _accrue(
        repository,
        _context(voice_mode_capture_enabled="FALSE"),
        store,
        _turn([_owner(0.0, 5.0)]),
    )

    assert result is None


@pytest.mark.asyncio
async def test_ceiling_stops_accrual():
    repository = InMemoryMediaAssetRepository()
    await _grant(repository)
    store = _Store(reference_seconds=4.0)
    context = _context(elevenlabs_professional_voice_clone_maximum_seconds=10.0)
    await repository.add_voice_clip(
        {
            "user_id": USER_ID,
            "assistant_id": ASSISTANT_ID,
            "source": "recorder",
            "source_document_name": "seed",
            "mime_type": "audio/mpeg",
            "bytes": b"speech",
            "duration_seconds": 12.0,
        }
    )

    reason = await capture.accrual_blocked_reason(
        repository,
        context,
        store,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        is_personal_avatar=True,
    )

    assert reason == capture.BLOCKED_CEILING_REACHED


@pytest.mark.asyncio
async def test_blocked_reason_names_the_missing_reference():
    repository = InMemoryMediaAssetRepository()
    await _grant(repository)

    reason = await capture.accrual_blocked_reason(
        repository,
        _context(),
        _Store(reference_seconds=None),
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        is_personal_avatar=True,
    )

    assert reason == capture.BLOCKED_REFERENCE_MISSING


class _FakeDiarizer:
    """Stands in for the diarizer, recording what it was asked to recognise."""

    def __init__(self):
        self.references = []
        self.segments = []

    async def __call__(
        self,
        media_base64,
        context,
        encoded_reference_audio=None,
        filename=None,
        content_type=None,
        reference_audio=False,
    ):
        self.references.append(encoded_reference_audio)
        return {
            "segments": list(self.segments),
            "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            "total_cost": 0.004,
            "model": "gpt-4o-transcribe-diarize",
        }


@pytest.fixture
def diarizer(monkeypatch):
    fake = _FakeDiarizer()
    monkeypatch.setattr("src.anubis.utils.utility.transcribe_audio_diarize", fake)
    return fake


def _diarized(speaker, start, end, text="words"):
    return {"speaker": speaker, "text": text, "start": start, "end": end}


async def _dictate(repository, context, store, **kwargs):
    return await capture.accrue_voice_from_utterance(
        AUDIO,
        kwargs.pop("seconds", 8.0),
        repository,
        context,
        store,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        is_personal_avatar=True,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_dictation_needs_the_reference_too(diarizer, cut_calls):
    repository = InMemoryMediaAssetRepository()
    await _grant(repository)

    result = await _dictate(repository, _context(), _Store(reference_seconds=None))

    assert result is None
    # Not even worth asking the diarizer: there is nothing to recognise against.
    assert diarizer.references == []
    assert await repository.total_voice_seconds(ASSISTANT_ID) == 0


@pytest.mark.asyncio
async def test_dictation_is_diarized_against_the_reference(diarizer, cut_calls):
    repository = InMemoryMediaAssetRepository()
    await _grant(repository)
    store = _Store(reference_seconds=4.0)
    diarizer.segments = [_diarized("avatar", 0.0, 6.0)]

    await _dictate(repository, _context(), store)

    # The stored reference clip is what the diarizer was given to match against.
    assert diarizer.references == [AUDIO]
    clips = await repository.list_voice_clips(ASSISTANT_ID)
    assert [clip["source"] for clip in clips] == [capture.CAPTURE_SOURCE_DICTATION]
    assert await repository.total_voice_seconds(ASSISTANT_ID) == pytest.approx(6.0)
    assert store.writes == []


@pytest.mark.asyncio
async def test_dictation_keeps_only_the_owners_speech(diarizer, cut_calls):
    """Someone else at the owner's microphone does not become the avatar's voice."""
    repository = InMemoryMediaAssetRepository()
    await _grant(repository)
    diarizer.segments = [
        _diarized("avatar", 0.0, 4.0),
        _diarized("B", 4.0, 12.0),
        _diarized("avatar", 12.0, 15.0),
    ]

    await _dictate(repository, _context(), _Store(reference_seconds=4.0))

    assert cut_calls == [[
        {"is_target": True, "start": 0.0, "end": 4.0},
        {"is_target": True, "start": 12.0, "end": 15.0},
    ]]


@pytest.mark.asyncio
async def test_dictation_by_someone_else_accrues_nothing(diarizer, cut_calls):
    repository = InMemoryMediaAssetRepository()
    await _grant(repository)
    diarizer.segments = [_diarized("B", 0.0, 9.0), _diarized("C", 9.0, 14.0)]

    result = await _dictate(repository, _context(), _Store(reference_seconds=4.0))

    assert result is None
    assert cut_calls == []
    assert await repository.total_voice_seconds(ASSISTANT_ID) == 0


@pytest.mark.asyncio
async def test_dictation_diarization_is_offered_for_metering(diarizer, cut_calls):
    repository = InMemoryMediaAssetRepository()
    await _grant(repository)
    diarizer.segments = [_diarized("avatar", 0.0, 6.0)]
    metered = []

    await _dictate(
        repository,
        _context(),
        _Store(reference_seconds=4.0),
        on_diarized=metered.append,
    )

    assert metered and metered[0]["total_cost"] == pytest.approx(0.004)


@pytest.mark.asyncio
async def test_the_instant_clone_is_refreshed_once_at_the_target(monkeypatch):
    """A clone built from sixty seconds is replaced when the corpus reaches the target."""
    from src.anubis.utils.voice import elevenlabs_client

    created: list[float] = []
    deleted: list[str] = []

    async def create_instant_voice(context, *, name, clips, description=""):
        created.append(sum(1 for _ in clips))
        return f"voice-{len(created)}"

    async def delete_voice(context, voice_id):
        deleted.append(voice_id)

    async def voice_safety_control(context, voice_id):
        return "ALLOW"

    monkeypatch.setattr(elevenlabs_client, "create_instant_voice", create_instant_voice)
    monkeypatch.setattr(elevenlabs_client, "delete_voice", delete_voice)
    monkeypatch.setattr(elevenlabs_client, "voice_safety_control", voice_safety_control)

    repository = InMemoryMediaAssetRepository()
    context = _context()
    await repository.upsert_voice(
        {
            "assistant_id": ASSISTANT_ID,
            "user_id": USER_ID,
            "instant_voice_id": "voice-original",
            "instant_voice_seconds": 60.0,
            "collected_seconds": 0.0,
            "detail": {},
        }
    )
    await repository.add_voice_clip(
        {
            "user_id": USER_ID,
            "assistant_id": ASSISTANT_ID,
            "source": capture.CAPTURE_SOURCE_SPOKEN_TURN,
            "source_document_name": "voice mode thread-1",
            "mime_type": "audio/mpeg",
            "bytes": b"speech",
            "duration_seconds": 130.0,
        }
    )

    first = await corpus.refresh_instant_voice_at_target(
        repository, context, user_id=USER_ID, assistant_id=ASSISTANT_ID
    )
    assert deleted == ["voice-original"]
    assert first["instant_voice_id"] == "voice-1"

    # Once, and only once: a second crossing must not churn the vendor.
    await corpus.refresh_instant_voice_at_target(
        repository, context, user_id=USER_ID, assistant_id=ASSISTANT_ID
    )
    assert deleted == ["voice-original"]


@pytest.mark.asyncio
async def test_refresh_leaves_a_clone_already_built_from_the_target(monkeypatch):
    repository = InMemoryMediaAssetRepository()
    await repository.upsert_voice(
        {
            "assistant_id": ASSISTANT_ID,
            "user_id": USER_ID,
            "instant_voice_id": "voice-good",
            "instant_voice_seconds": 120.0,
            "collected_seconds": 0.0,
            "detail": {},
        }
    )

    record = await corpus.refresh_instant_voice_at_target(
        repository, _context(), user_id=USER_ID, assistant_id=ASSISTANT_ID
    )

    assert record["instant_voice_id"] == "voice-good"


@pytest.mark.asyncio
async def test_readiness_tells_everyone_whether_it_can_speak():
    """Whether an avatar can speak is plain from pressing speak, so it is told."""
    repository = InMemoryMediaAssetRepository()
    await repository.upsert_voice(
        {
            "assistant_id": ASSISTANT_ID,
            "user_id": USER_ID,
            "instant_voice_id": "voice-1",
            "instant_voice_seconds": 90.0,
            "collected_seconds": 90.0,
            "detail": {},
        }
    )

    owner_view = await corpus.voice_readiness(
        repository, _context(), user_id=USER_ID, assistant_id=ASSISTANT_ID
    )
    visitor_view = await corpus.voice_readiness(
        repository, _context(), user_id="someone-else", assistant_id=ASSISTANT_ID
    )

    assert owner_view["has_voice"] is True
    assert visitor_view["has_voice"] is True
    # How much speech the avatar holds is the owner's business alone.
    assert "collected_seconds" in owner_view
    assert "collected_seconds" not in visitor_view
    assert "professional_state" not in visitor_view
