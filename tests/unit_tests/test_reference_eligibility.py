"""What may become the avatar's reference-audio clip.

The reference clip is the anchor the diarizer receives on every later upload, so
two things have to hold: the source is one recording, and the isolation actually
cut a clip the diarizer accepts. Both rules live in
``src/anubis/utils/voice/reference_eligibility.py`` so the media pipeline, the
voice recorder endpoint, and the owner's explicit selection all agree.
"""

from src.anubis.utils.classes.URLDocumentLoaderClass import (
    _classify_url,
    channel_videos_url,
)
from src.anubis.utils.voice.reference_eligibility import (
    OPENAI_REFERENCE_MAXIMUM_SECONDS,
    reference_clip_rejection,
    reference_source_rejection,
    stored_reference_rejection,
)


def test_a_channel_or_playlist_link_is_not_a_recording():
    for link in (
        "https://www.youtube.com/@imahara",
        "https://www.youtube.com/channel/UCabc",
        "https://www.youtube.com/c/SomeChannel",
        "https://www.youtube.com/user/SomeUser",
        "https://www.youtube.com/playlist?list=PL123",
    ):
        assert reference_source_rejection(filename=link) is not None, link


def test_a_single_video_and_an_ordinary_upload_may_anchor():
    assert (
        reference_source_rejection(
            filename="https://www.youtube.com/watch?v=Px_5Z0pPlPc&t=10s",
            media_type="audio",
        )
        is None
    )
    assert reference_source_rejection(filename="Mom.m4a", media_type="audio") is None
    assert reference_source_rejection(filename="Talk.mp4", media_type="video") is None


def test_the_expanded_kind_is_rejected_however_the_link_reads():
    # An item expanded out of a channel carries the enumerated kind even when
    # the filename by then names one video.
    assert (
        reference_source_rejection(
            filename="Mom.m4a", url_kind="youtube_playlist", media_type="audio"
        )
        is not None
    )


def test_only_a_recording_can_anchor_the_diarizer():
    assert reference_source_rejection(filename="notes.pdf", media_type="pdf") is not None
    assert reference_source_rejection(filename="face.png", media_type="image") is not None


def test_the_passthrough_fallback_is_never_a_clip():
    # Every fallback path of ``isolate_dominant_speaker_audio_b64`` reports the
    # untouched input with no duration and no transcript.
    assert (
        reference_clip_rejection(
            audio_data_uri="data:audio/mp3;base64,QUJD",
            transcript_text="",
            duration_seconds=None,
        )
        is not None
    )


def test_a_clip_outside_the_diarizer_bounds_is_rejected():
    too_short = reference_clip_rejection(
        audio_data_uri="data:audio/mp3;base64,QUJD",
        transcript_text="hello",
        duration_seconds=0.4,
    )
    assert too_short is not None
    too_long = reference_clip_rejection(
        audio_data_uri="data:audio/mp3;base64,QUJD",
        transcript_text="hello",
        duration_seconds=OPENAI_REFERENCE_MAXIMUM_SECONDS + 1.0,
    )
    assert too_long is not None
    # A configured maximum never widens past what the diarizer accepts.
    assert (
        reference_clip_rejection(
            audio_data_uri="data:audio/mp3;base64,QUJD",
            transcript_text="hello",
            duration_seconds=20.0,
            maximum_seconds=60.0,
        )
        is not None
    )


def test_a_silent_clip_cannot_identify_a_voice():
    assert (
        reference_clip_rejection(
            audio_data_uri="data:audio/mp3;base64,QUJD",
            transcript_text="   ",
            duration_seconds=4.0,
        )
        is not None
    )


def test_a_real_clip_passes():
    assert (
        reference_clip_rejection(
            audio_data_uri="data:audio/mp3;base64,QUJD",
            transcript_text="I was born in London.",
            duration_seconds=7.65,
        )
        is None
    )


def test_a_stored_row_is_judged_the_same_way():
    assert stored_reference_rejection(None) is not None
    assert (
        stored_reference_rejection(
            {
                "audio_data_uri": "data:audio/mp3;base64,QUJD",
                "transcript_text": "",
                "duration_seconds": None,
            }
        )
        is not None
    )
    assert (
        stored_reference_rejection(
            {
                "audio_data_uri": "data:audio/mp3;base64,QUJD",
                "transcript_text": "I was born in London.",
                "duration_seconds": 7.65,
            }
        )
        is None
    )


def test_a_channel_link_enumerates_instead_of_downloading_one_arbitrary_video():
    assert _classify_url("https://www.youtube.com/@imahara") == "youtube_playlist"
    assert _classify_url("https://www.youtube.com/channel/UCabc") == "youtube_playlist"
    assert _classify_url("https://www.youtube.com/c/Name") == "youtube_playlist"
    assert _classify_url("https://www.youtube.com/user/Name") == "youtube_playlist"
    # A single video, with or without a playlist alongside, stays a single video.
    assert _classify_url("https://www.youtube.com/watch?v=abc") == "youtube"
    assert _classify_url("https://www.youtube.com/watch?v=abc&list=PL1") == "youtube"
    assert _classify_url("https://youtu.be/abc") == "youtube"


def test_a_channel_is_enumerated_through_its_videos_tab():
    # yt_dlp flat-extracts a bare channel URL into the channel's tabs, so the
    # Videos tab is asked for by name.
    assert (
        channel_videos_url("https://www.youtube.com/@imahara")
        == "https://www.youtube.com/@imahara/videos"
    )
    # A link that already names a tab, and a link that is not a channel, are
    # left exactly as they are.
    assert (
        channel_videos_url("https://www.youtube.com/@imahara/videos")
        == "https://www.youtube.com/@imahara/videos"
    )
    assert (
        channel_videos_url("https://www.youtube.com/watch?v=abc")
        == "https://www.youtube.com/watch?v=abc"
    )
