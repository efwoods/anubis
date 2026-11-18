"""Cost-efficient diarization: fragment merging, published transcripts, providers.

- Overlapping speech that a diarizer cut into alternating sub-second pieces is
  regrouped per speaker, so one sentence is one turn and never several one-word
  quote documents, and no word is lost.
- A transcript published by the source of a video is read only when the
  transcript's own timestamp links point at the same video id.
- Each hosted diarizer's response is normalized to seconds and one segment
  shape, and the recorded cost is the audio duration times the hourly price.
- A chunked recording diarized without a reference clip never lets two chunks
  share one speaker label.
"""

from types import SimpleNamespace

import pytest

from src.anubis.utils.speech import diarization_providers as providers
from src.anubis.utils.speech import published_transcripts
from src.anubis.utils.utility import (
    _merge_diarized_segments_from_chunks,
    _select_dominant_speaker_segments,
)
from src.subgraphs.process_media_graph.utils.helper_functions import (
    coalesce_segments_by_speaker,
    merge_interleaved_fragments,
)


def _segment(speaker, start, end, text):
    return {"speaker": speaker, "start": start, "end": end, "text": text}


# The opening of the 2026-09-17 diarization of youtube.com/watch?v=mQ7ECcjXazw.
OVERLAPPING_SPEECH = [
    _segment("A", 0.00, 5.00, "take this that's yours now"),
    _segment("B", 5.00, 5.15, "Really?"),
    _segment("A", 5.15, 5.80, "see that moon in"),
    _segment("B", 5.80, 5.90, "I"),
    _segment("A", 5.90, 6.15, "the bottom of"),
    _segment("B", 6.15, 6.25, "can keep"),
    _segment("A", 6.25, 6.45, "it that's"),
    _segment("B", 6.45, 6.65, "this?"),
    _segment("A", 6.65, 9.75, "yours that's yours"),
    _segment("B", 9.75, 11.05, "This is a happy man right now."),
]


def test_overlapping_fragments_are_regrouped_into_whole_sentences():
    turns = coalesce_segments_by_speaker(merge_interleaved_fragments(OVERLAPPING_SPEECH))

    assert [(turn["speaker"], turn["text"]) for turn in turns] == [
        ("A", "take this that's yours now see that moon in the bottom of it that's"),
        ("B", "Really? I can keep this?"),
        ("A", "yours that's yours"),
        ("B", "This is a happy man right now."),
    ]


def test_fragment_merging_never_drops_or_adds_a_word():
    def words(segments):
        return sorted(word for segment in segments for word in segment["text"].split())

    assert words(merge_interleaved_fragments(OVERLAPPING_SPEECH)) == words(OVERLAPPING_SPEECH)


def test_a_short_reply_between_two_long_turns_stays_a_turn():
    segments = [
        _segment("A", 0.0, 6.0, "that is the actual moon phase"),
        _segment("B", 6.0, 6.9, "Oh, that's awesome."),
        _segment("A", 6.9, 12.0, "yeah it's awesome"),
    ]
    assert merge_interleaved_fragments(segments) == segments


def test_fragments_separated_by_a_long_silence_are_not_one_run():
    segments = [
        _segment("A", 0.0, 0.5, "Yes."),
        _segment("B", 10.0, 10.5, "No."),
        _segment("A", 20.0, 20.5, "Maybe."),
    ]
    assert merge_interleaved_fragments(segments) == segments


def test_segments_without_timestamps_pass_through_unchanged():
    segments = [
        {"speaker": "A", "text": "first"},
        {"speaker": "B", "text": "second"},
        {"speaker": "A", "text": "third"},
    ]
    assert merge_interleaved_fragments(segments) == segments


def test_the_target_flag_survives_regrouping():
    segments = [
        {**_segment("avatar", 0.0, 0.2, "I"), "is_target": True},
        {**_segment("B", 0.2, 0.4, "you"), "is_target": False},
        {**_segment("avatar", 0.4, 0.6, "agree"), "is_target": True},
    ]
    merged = merge_interleaved_fragments(segments)
    assert [(s["speaker"], s["text"], s["is_target"]) for s in merged] == [
        ("avatar", "I agree", True),
        ("B", "you", False),
    ]


# ---------------------------------------------------------------------------
# Chunk label namespacing and the chosen-label selection
# ---------------------------------------------------------------------------


def _chunk_response(*segments):
    return SimpleNamespace(
        segments=[SimpleNamespace(model_dump=lambda segment=segment: dict(segment)) for segment in segments]
    )


def test_two_chunks_never_share_a_label_when_no_reference_exists():
    merged = _merge_diarized_segments_from_chunks(
        [
            _chunk_response(_segment("A", 0.0, 5.0, "host speaking")),
            _chunk_response(_segment("A", 0.0, 5.0, "guest speaking")),
        ],
        [0.0, 100.0],
        namespace_every_label=True,
    )
    assert [segment["speaker"] for segment in merged] == ["chunk_0.A", "chunk_1.A"]
    assert merged[1]["start"] == 100.0


def test_labels_stay_verbatim_by_default_without_a_reference():
    merged = _merge_diarized_segments_from_chunks(
        [_chunk_response(_segment("A", 0.0, 5.0, "x")), _chunk_response(_segment("A", 0.0, 5.0, "y"))],
        [0.0, 100.0],
    )
    assert [segment["speaker"] for segment in merged] == ["A", "A"]


def test_the_identified_speaker_is_selected_over_the_one_who_talks_most():
    segments = [
        _segment("host", 0.0, 60.0, "a long introduction"),
        _segment("guest", 60.0, 70.0, "a short answer"),
    ]
    dominant = _select_dominant_speaker_segments(segments)
    identified = _select_dominant_speaker_segments(segments, target_speaker_labels=["guest"])

    assert dominant[0] == "host"
    assert identified[0] == "guest"
    assert [segment["text"] for segment in identified[1]] == ["a short answer"]


# ---------------------------------------------------------------------------
# Published transcripts
# ---------------------------------------------------------------------------

TRANSCRIPT_PAGE = """
<div class="ts-segment"><span class="ts-name">Lex Fridman</span>
 <span class="ts-timestamp"><a href="https://youtube.com/watch?v=VIDEO123456&#038;t=0">(00:00:00)</a> </span>
 <span class="ts-text">The following is a conversation.</span></div>
<div class="ts-segment"><span class="ts-name">Guest Person</span>
 <span class="ts-timestamp"><a href="https://youtube.com/watch?v=VIDEO123456&#038;t=49">(00:00:49)</a> </span>
 <span class="ts-text">Drinking coffee or water?</span></div>
<div class="ts-segment"><span class="ts-name"></span>
 <span class="ts-timestamp"><a href="https://youtube.com/watch?v=VIDEO123456&#038;t=58">(00:00:58)</a> </span>
 <span class="ts-text">I prefer water.</span></div>
<div class="ts-segment"><span class="ts-name">Lex Fridman</span>
 <span class="ts-timestamp"><a href="https://youtube.com/watch?v=VIDEO123456&#038;t=62">(00:01:02)</a> </span>
 <span class="ts-text">Water.</span></div>
"""

INDEX_PAGE = """
<div class="guest">
  <a href="https://www.youtube.com/watch?v=OTHERVIDEO1">Video</a> -
  <a href="https://lexfridman.com/other-transcript">Transcript</a>
</div>
<div class="guest">
  <a href="https://www.youtube.com/watch?v=VIDEO123456">Video</a> -
  <a href="https://lexfridman.com/guest-person">Episode</a> -
  <a href="https://lexfridman.com/guest-person-transcript">Transcript</a>
</div>
<div class="guest">
  <a href="https://www.youtube.com/watch?v=NOTRANSCRIP">Video</a>
</div>
"""


def test_a_transcript_page_becomes_named_timed_segments():
    segments = published_transcripts.parse_lex_fridman_transcript_page(
        TRANSCRIPT_PAGE, video_id="VIDEO123456"
    )
    assert [(s["speaker"], s["start"], s["end"], s["text"]) for s in segments[:3]] == [
        ("Lex Fridman", 0.0, 49.0, "The following is a conversation."),
        ("Guest Person", 49.0, 58.0, "Drinking coffee or water?"),
        # An empty name continues the preceding speaker's turn.
        ("Guest Person", 58.0, 62.0, "I prefer water."),
    ]
    assert segments[-1]["end"] > segments[-1]["start"]


def test_a_transcript_of_a_different_video_is_never_applied():
    """A clip has a different video id and timeline than the full episode."""
    assert (
        published_transcripts.parse_lex_fridman_transcript_page(
            TRANSCRIPT_PAGE, video_id="CLIPOFVIDEO"
        )
        == []
    )


def test_the_index_pairs_a_video_with_the_transcript_page():
    find = published_transcripts.lex_fridman_transcript_url_for_video
    assert find(INDEX_PAGE, video_id="VIDEO123456") == "https://lexfridman.com/guest-person-transcript"
    assert find(INDEX_PAGE, video_id="NOTRANSCRIP") is None
    assert find(INDEX_PAGE, video_id="UNKNOWNVID1") is None


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.youtube.com/watch?v=mQ7ECcjXazw", "mQ7ECcjXazw"),
        ("https://youtu.be/mQ7ECcjXazw?t=3", "mQ7ECcjXazw"),
        ("https://youtube.com/watch?v=Kbk9BiPhm7o&#038;t=49", "Kbk9BiPhm7o"),
        ("https://x.com/lexfridman", None),
        ("dialogue_long_clip.mp3", None),
        (None, None),
    ],
)
def test_youtube_video_id(url, expected):
    assert published_transcripts.youtube_video_id(url) == expected


def test_only_the_exact_name_is_the_avatar():
    matches = published_transcripts.speaker_name_matches_avatar
    assert matches("Lex Fridman", "lex  fridman")
    assert not matches("Alexander Fridman", "Lex Fridman")
    assert not matches("Lex", "Lex Fridman")
    assert not matches("", "Lex Fridman")


@pytest.mark.asyncio
async def test_a_published_transcript_is_returned_as_a_free_diarization(monkeypatch):
    pages = {
        "https://lexfridman.com/podcast": INDEX_PAGE,
        "https://lexfridman.com/guest-person-transcript": TRANSCRIPT_PAGE,
    }

    async def fake_fetch(url):
        return pages[url]

    monkeypatch.setattr(published_transcripts, "_fetch_text", fake_fetch)
    published_transcripts._index_page_cache.clear()

    diarization = await published_transcripts.load_published_transcript_as_diarization(
        "https://www.youtube.com/watch?v=VIDEO123456"
    )
    assert diarization["total_cost"] == 0.0
    assert diarization["model"] == "published_transcript"
    assert len(diarization["segments"]) == 4
    assert diarization["published_transcript_url"] == "https://lexfridman.com/guest-person-transcript"

    assert (
        await published_transcripts.load_published_transcript_as_diarization(
            "https://www.youtube.com/watch?v=NOTRANSCRIP"
        )
        is None
    )
    published_transcripts._index_page_cache.clear()


@pytest.mark.asyncio
async def test_a_failed_lookup_falls_back_to_the_diarizer(monkeypatch):
    async def failing_fetch(url):
        raise RuntimeError("site unreachable")

    monkeypatch.setattr(published_transcripts, "_fetch_text", failing_fetch)
    published_transcripts._index_page_cache.clear()
    assert (
        await published_transcripts.load_published_transcript_as_diarization(
            "https://www.youtube.com/watch?v=VIDEO123456"
        )
        is None
    )


# ---------------------------------------------------------------------------
# Hosted providers
# ---------------------------------------------------------------------------


def test_assemblyai_milliseconds_become_seconds():
    segments = providers.normalize_assemblyai_transcript(
        {
            "utterances": [
                {"speaker": "A", "text": "Hello there.", "start": 1500, "end": 4250},
                {"speaker": "B", "text": "  ", "start": 4250, "end": 4300},
            ]
        }
    )
    assert segments == [{"speaker": "A", "text": "Hello there.", "start": 1.5, "end": 4.25}]


def test_deepgram_speaker_numbers_become_labels():
    segments = providers.normalize_deepgram_response(
        {
            "results": {
                "utterances": [
                    {"speaker": 0, "transcript": "Hello there.", "start": 1.5, "end": 4.25},
                    {"speaker": 1, "transcript": "Hi.", "start": 4.5, "end": 5.0},
                ]
            }
        }
    )
    assert [segment["speaker"] for segment in segments] == ["speaker_0", "speaker_1"]
    assert segments[0]["text"] == "Hello there."


def test_pyannoteai_turns_become_segments():
    segments = providers.normalize_pyannoteai_output(
        {
            "turnLevelTranscription": [
                {"speaker": "SPEAKER_00", "text": "Hello there.", "start": 1.5, "end": 4.25}
            ]
        }
    )
    assert segments == [
        {"speaker": "SPEAKER_00", "text": "Hello there.", "start": 1.5, "end": 4.25}
    ]


def test_the_recorded_cost_is_duration_times_the_hourly_price():
    response = providers._response_dictionary(
        provider=providers.PROVIDER_DEEPGRAM,
        model="deepgram:nova-3",
        segments=[{"speaker": "speaker_0", "text": "Hello.", "start": 0.0, "end": 1.0}],
        duration_seconds=1800.0,
        price_per_hour=0.258,
        started_at_nanoseconds=0,
    )
    # 1,800 s / 3,600 s per hour x $0.258 per hour = $0.129
    assert response["total_cost"] == pytest.approx(0.129)
    assert response["inference_type"] == "diarization"
    assert response["text"] == "Hello."


def test_an_empty_price_records_zero_rather_than_a_guess():
    response = providers._response_dictionary(
        provider=providers.PROVIDER_ASSEMBLYAI,
        model="assemblyai:default",
        segments=[],
        duration_seconds=1800.0,
        price_per_hour=0.0,
        started_at_nanoseconds=0,
    )
    assert response["total_cost"] == 0.0


def test_the_provider_defaults_to_openai_and_rejects_unknown_names():
    assert providers.selected_diarization_provider(SimpleNamespace()) == "OPENAI"
    assert (
        providers.selected_diarization_provider(
            SimpleNamespace(audio_diarization_provider=" deepgram ")
        )
        == "DEEPGRAM"
    )
    with pytest.raises(providers.DiarizationProviderError):
        providers.selected_diarization_provider(
            SimpleNamespace(audio_diarization_provider="WHISPERX")
        )


@pytest.mark.asyncio
async def test_a_hosted_provider_without_a_key_fails_with_the_setting_name(tmp_path):
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"ID3")
    with pytest.raises(providers.DiarizationProviderError, match="ASSEMBLYAI_API_KEY"):
        await providers.diarize_mp3_with_hosted_provider(
            "ASSEMBLYAI", str(audio_path), SimpleNamespace()
        )


# ---------------------------------------------------------------------------
# Scoring a provider against a transcript whose speakers are known
# ---------------------------------------------------------------------------

from src.anubis.utils.speech import diarization_scoring as scoring  # noqa: E402

REFERENCE_SEGMENTS = [
    {"speaker": "Lex Fridman", "text": "What is the meaning of life?", "start": 0.0, "end": 4.0},
    {"speaker": "Guest Person", "text": "I think love is the answer.", "start": 4.0, "end": 9.0},
    {"speaker": "Lex Fridman", "text": "Beautifully put.", "start": 9.0, "end": 11.0},
]


def test_word_error_rate_counts_substitutions_deletions_and_insertions():
    reference = scoring.normalized_words("What is the meaning of life?")
    assert scoring.word_error_rate(reference, reference) == 0.0
    # One substitution in six reference words.
    assert scoring.word_error_rate(
        reference, scoring.normalized_words("what is the meaning of live")
    ) == pytest.approx(1 / 6)
    # Two deletions in six reference words.
    assert scoring.word_error_rate(
        reference, scoring.normalized_words("what is the meaning")
    ) == pytest.approx(2 / 6)
    assert scoring.word_error_rate([], ["anything"]) is None


def test_perfect_diarization_scores_one_under_any_label_names():
    hypothesis = [
        {"speaker": "A", "text": "What is the meaning of life?"},
        {"speaker": "B", "text": "I think love is the answer."},
        {"speaker": "A", "text": "Beautifully put."},
    ]
    scores = scoring.speaker_attribution_scores(
        REFERENCE_SEGMENTS, hypothesis, avatar_speaker_name="Lex Fridman"
    )
    assert scores["speaker_attribution_accuracy"] == 1.0
    assert scores["label_to_speaker"] == {"A": "Lex Fridman", "B": "Guest Person"}
    assert (scores["avatar_precision"], scores["avatar_recall"]) == (1.0, 1.0)


def test_a_merged_speaker_costs_the_other_speakers_words():
    """A diarizer that hears one speaker credits every word to that speaker."""
    hypothesis = [
        {
            "speaker": "A",
            "text": "What is the meaning of life? I think love is the answer. Beautifully put.",
        }
    ]
    scores = scoring.speaker_attribution_scores(
        REFERENCE_SEGMENTS, hypothesis, avatar_speaker_name="Lex Fridman"
    )
    # 8 of the 14 words are Lex Fridman's, so label A maps to Lex Fridman and
    # the guest's 6 words are wrong: 8 / 14.
    assert scores["speaker_attribution_accuracy"] == pytest.approx(8 / 14)
    assert scores["avatar_precision"] == pytest.approx(8 / 14)
    assert scores["avatar_recall"] == 1.0
    assert (scores["reference_speaker_count"], scores["hypothesis_label_count"]) == (2, 1)


def test_the_reference_is_trimmed_to_the_audio_that_was_sent():
    assert [s["start"] for s in scoring.segments_within(REFERENCE_SEGMENTS, 9.0)] == [0.0, 4.0]
    assert len(scoring.segments_within(REFERENCE_SEGMENTS, None)) == 3


def test_turn_structure_counts_turns_too_short_to_be_a_quote():
    structure = scoring.turn_structure(
        [{"text": "man."}, {"text": "This is a sentence that is long enough to be a quote."}]
    )
    assert (structure["turns"], structure["turns_under_short_limit"]) == (2, 1)
    assert structure["share_of_turns_under_short_limit"] == 0.5
