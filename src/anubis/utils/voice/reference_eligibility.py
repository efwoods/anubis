"""What may become an avatar's reference-audio clip, and what may not.

The reference clip is the single anchor OpenAI's diarizer receives as
``known_speaker_references``. Every later upload leans on that anchor to decide
which turns belong to the avatar and which belong to everyone else, so an anchor
cut from the wrong recording — or from no speech at all — mislabels every upload
that follows, and the avatar ends up learning other people's words as the
avatar's own.

Two separate questions decide whether a candidate may become the anchor, and
both are answered here so the media pipeline, the voice recorder endpoint, and
the owner's explicit "make this the reference" endpoint all apply one rule and
speak with one wording:

- ``reference_source_rejection`` — is the SOURCE a single recording? A YouTube
  channel or playlist link names a body of work, not one person speaking, so
  whichever video such a link happened to yield is an arbitrary choice.
- ``reference_clip_rejection`` — did the isolation actually produce a usable
  clip? ``isolate_dominant_speaker_audio_b64`` returns a passthrough fallback
  whenever diarization could not find a dominant speaker — the whole original
  audio, carrying no duration and no transcript. That fallback must never be
  stored: the whole recording is longer than the diarizer accepts, so storing
  the fallback makes every later diarization call fail.

Every rejection reason is written for the avatar's creator to read, because the
reason is surfaced in the Voice panel and in the media-job progress events.
"""

from __future__ import annotations

from typing import Any

# OpenAI's diarizer rejects a ``known_speaker_references`` clip whose duration is
# not strictly between 1.2 s and 10.0 s. Both bounds keep a margin inside the
# real limits: mp3 frame padding can nudge a clip a few milliseconds past the
# requested length, so capping at exactly 10.0 s can still produce a file over
# 10.0 s and a 400. ``src/anubis/utils/utility.py`` imports these two names so
# the bounds are stated exactly once.
OPENAI_REFERENCE_MINIMUM_SECONDS = 1.3
OPENAI_REFERENCE_MAXIMUM_SECONDS = 9.5

# ``url_kind`` values that name a body of work rather than one recording. The
# media pipeline stamps ``url_kind`` on every item expanded out of a URL.
ENUMERATED_URL_KINDS = frozenset(
    {"youtube_playlist", "youtube_channel", "playlist", "channel"}
)


def _looks_like_enumerated_youtube_link(filename: str) -> bool:
    """Report whether a filename is a YouTube link naming a channel or a playlist.

    Uploads keep the original URL as the filename, so the URL is what a later
    stage has to judge. The check mirrors ``_classify_url`` in
    ``src/anubis/utils/classes/URLDocumentLoaderClass.py`` without importing that
    module, because the media pipeline may hand over a filename that never went
    through the URL loader.
    """
    from urllib.parse import parse_qs, urlparse

    try:
        parsed = urlparse(filename)
    except Exception:  # noqa: BLE001 - an unparseable filename is not a link
        return False
    host = (parsed.hostname or "").lower()
    if "youtube" not in host and "youtu.be" not in host:
        return False
    path = (parsed.path or "").lower()
    query = parse_qs(parsed.query or "")
    if "playlist" in path:
        return True
    if "list" in query and "v" not in query:
        return True
    return (
        path.startswith("/@")
        or path.startswith("/channel/")
        or path.startswith("/c/")
        or path.startswith("/user/")
        or path.strip("/") in ("", "feed", "feed/subscriptions")
    )


def reference_source_rejection(
    *,
    filename: str | None,
    url_kind: str | None = None,
    media_type: str | None = None,
) -> str | None:
    """Why this source may not become the reference clip, or ``None`` when the source may.

    Args:
        filename: What the upload is called. A URL upload keeps the original
            URL as the filename.
        url_kind: The label the URL loader stamped on an expanded item.
        media_type: The pipeline's media label; only audio and video can anchor
            a diarizer.

    Returns:
        A sentence naming what to upload instead, or ``None`` when the source is
        a single recording and may become the reference clip.
    """
    if media_type is not None and media_type not in ("audio", "video"):
        return (
            "Only an audio or video recording can be the reference clip that "
            "identifies this avatar's voice."
        )
    if (url_kind or "").strip().lower() in ENUMERATED_URL_KINDS:
        return (
            "A channel or playlist link names many videos, so whichever video "
            "the link yields is an arbitrary choice. Upload a single video in "
            "which this avatar speaks more than anyone else."
        )
    if _looks_like_enumerated_youtube_link(str(filename or "")):
        return (
            "A YouTube channel or playlist link is not a single recording. "
            "Paste a single video in which this avatar speaks more than anyone "
            "else, and that video becomes the reference clip."
        )
    return None


def reference_clip_rejection(
    *,
    audio_data_uri: str | None,
    transcript_text: str | None,
    duration_seconds: float | None,
    maximum_seconds: float | None = None,
    minimum_seconds: float | None = None,
) -> str | None:
    """Why this isolated clip may not be stored as the reference, or ``None`` when the clip may.

    Args:
        audio_data_uri: The clip ``isolate_dominant_speaker_audio_b64`` produced.
        transcript_text: The transcript of exactly that clip.
        duration_seconds: The clip's length. ``None`` is exactly how the
            isolation reports the passthrough fallback — every fallback path
            returns the untouched input with no duration — so a missing
            duration means no clip was cut at all.
        maximum_seconds: Upper bound, defaulting to what the diarizer accepts.
        minimum_seconds: Lower bound, defaulting to what the diarizer accepts.

    Returns:
        A sentence naming what went wrong, or ``None`` when the clip is usable.
    """
    upper_bound = min(
        float(maximum_seconds or OPENAI_REFERENCE_MAXIMUM_SECONDS),
        OPENAI_REFERENCE_MAXIMUM_SECONDS,
    )
    lower_bound = max(
        float(minimum_seconds or OPENAI_REFERENCE_MINIMUM_SECONDS),
        OPENAI_REFERENCE_MINIMUM_SECONDS,
    )
    clip = str(audio_data_uri or "").strip()
    if not clip:
        return "No audio was produced for the reference clip."
    if duration_seconds is None:
        return (
            "No single speaker stood out in that recording, so no reference "
            "clip could be cut from the recording. Upload a recording in which "
            "this avatar speaks more than anyone else."
        )
    length = float(duration_seconds)
    if length < lower_bound:
        return (
            f"The clip holds only {length:.1f}s of speech; the diarizer needs at "
            f"least {lower_bound:.1f}s of this avatar speaking without "
            "interruption."
        )
    if length > upper_bound:
        return (
            f"The clip runs {length:.1f}s; the diarizer accepts at most "
            f"{upper_bound:.1f}s."
        )
    if not str(transcript_text or "").strip():
        return (
            "No speech was transcribed from that clip, so the clip cannot "
            "identify this avatar's voice."
        )
    return None


def stored_reference_rejection(
    stored_reference: dict[str, Any] | None,
    *,
    maximum_seconds: float | None = None,
    minimum_seconds: float | None = None,
) -> str | None:
    """Why a reference row already in the store is unusable, or ``None`` when the row is fine.

    Rows written before these checks existed can hold the passthrough fallback,
    so every reader that depends on the anchor asks this question rather than
    trusting the row's presence.
    """
    if not stored_reference:
        return "This avatar has no reference audio yet."
    return reference_clip_rejection(
        audio_data_uri=stored_reference.get("audio_data_uri"),
        transcript_text=stored_reference.get("transcript_text"),
        duration_seconds=stored_reference.get("duration_seconds"),
        maximum_seconds=maximum_seconds,
        minimum_seconds=minimum_seconds,
    )
