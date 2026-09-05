"""``isolate_dominant_speaker_audio_b64`` and single-speaker recordings.

A voice sample is usually one person talking. The corpus callers pass
``allow_single_speaker=True`` so that recording still yields a clip with a
duration; without the keyword the historical rule applies (reference clips
keep the speaker, dominant-speaker isolation passes the audio through).
"""

import pytest

import src.anubis.utils.utility as utility


@pytest.fixture
def recorded_selection_kwargs(monkeypatch):
    """Stub the pipeline around the selection step and capture its kwargs."""
    captured: dict = {}

    async def fake_diarize(**_kwargs):
        return {"segments": []}

    def fake_select(_segments, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(utility, "_decode_base64_media_payload", lambda _b64: b"x")
    monkeypatch.setattr(utility, "transcribe_audio_diarize", fake_diarize)
    monkeypatch.setattr(utility, "_select_dominant_speaker_segments", fake_select)
    return captured


@pytest.mark.asyncio
async def test_the_corpus_keyword_keeps_a_single_speaker(recorded_selection_kwargs):
    result = await utility.isolate_dominant_speaker_audio_b64(
        "data:audio/mp3;base64,QUJD",
        context=object(),
        reference_audio=False,
        allow_single_speaker=True,
    )
    assert recorded_selection_kwargs["allow_single_speaker"] is True
    assert result["duration"] is None  # the stub selected nothing; passthrough


@pytest.mark.asyncio
async def test_without_the_keyword_only_a_reference_keeps_a_single_speaker(
    recorded_selection_kwargs,
):
    await utility.isolate_dominant_speaker_audio_b64(
        "data:audio/mp3;base64,QUJD", context=object(), reference_audio=False
    )
    assert recorded_selection_kwargs["allow_single_speaker"] is False

    await utility.isolate_dominant_speaker_audio_b64(
        "data:audio/mp3;base64,QUJD", context=object(), reference_audio=True
    )
    assert recorded_selection_kwargs["allow_single_speaker"] is True


def test_selection_returns_the_lone_speaker_only_when_allowed():
    segments = [
        {"speaker": "A", "start": 0.0, "end": 2.0, "text": "hello there"},
        {"speaker": "A", "start": 2.5, "end": 4.0, "text": "it is me"},
    ]
    assert utility._select_dominant_speaker_segments(segments) is None
    selected = utility._select_dominant_speaker_segments(
        segments, allow_single_speaker=True
    )
    assert selected is not None
    speaker, target_segments, _totals, total_seconds = selected
    assert speaker == "A"
    assert len(target_segments) == 2
    assert total_seconds == pytest.approx(3.5)
