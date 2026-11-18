"""Speaker-labelled transcripts that the source of a recording publishes.

A podcast that publishes a transcript with the speaker's name and a timestamp
on every turn has already done the work a diarizer is paid for, and has done
that work with the speakers' real names. When such a transcript exists for an
uploaded video, the upload pipeline reads the published transcript and sends
no audio to a diarizer.

A published transcript is trusted only when the transcript's own timestamp
links point at the SAME YouTube video id as the upload. A clip cut from an
episode has a different video id and a different timeline, so the episode's
transcript is never applied to the clip.

Adding a source is one ``PublishedTranscriptSource`` entry: the address of an
index page that pairs video ids with transcript pages, and the parser for the
transcript page.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

PUBLISHED_TRANSCRIPT_MODEL_NAME = "published_transcript"
_REQUEST_TIMEOUT_SECONDS = 30.0
_REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; transcript-reader)"}


def youtube_video_id(url: str | None) -> str | None:
    """Return the video id of a YouTube watch or short link, else ``None``."""
    if not url:
        return None
    parsed = urlparse(html.unescape(str(url).strip()))
    host = (parsed.netloc or "").lower()
    if host.endswith("youtu.be"):
        candidate = parsed.path.strip("/").split("/")[0]
        return candidate or None
    if "youtube.com" not in host:
        return None
    query_video_ids = parse_qs(parsed.query).get("v")
    if query_video_ids:
        return query_video_ids[0]
    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) >= 2 and path_parts[0] in ("shorts", "embed", "live"):
        return path_parts[1]
    return None


def _timestamp_seconds(url: str) -> float | None:
    """Return the ``t=`` offset of a YouTube timestamp link in seconds."""
    parsed = urlparse(html.unescape(url))
    offsets = parse_qs(parsed.query).get("t")
    if not offsets:
        return None
    digits = re.match(r"\d+", offsets[0])
    return float(digits.group(0)) if digits else None


def parse_lex_fridman_transcript_page(
    page_html: str, *, video_id: str
) -> List[Dict[str, Any]]:
    """Parse one lexfridman.com transcript page into speaker-labelled segments.

    Every turn on the page is a ``div.ts-segment`` holding the speaker's name
    (``span.ts-name``), a link to the moment in the YouTube video
    (``span.ts-timestamp``), and the words (``span.ts-text``). A segment with
    an empty ``ts-name`` continues the turn of the speaker named on the
    preceding segment.

    Only segments whose timestamp link points at ``video_id`` are returned, so
    a transcript page of a different video yields an empty list. Each returned
    segment carries ``speaker``, ``text``, ``start`` and ``end`` in seconds;
    ``end`` is the start of the next segment, because the page publishes start
    times only. The last segment's ``end`` equals the last segment's ``start``
    plus an estimate from the word count at 2.5 words per second.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(page_html, "html.parser")
    segments: List[Dict[str, Any]] = []
    previous_speaker: str | None = None
    for segment_element in soup.select("div.ts-segment"):
        name_element = segment_element.select_one(".ts-name")
        text_element = segment_element.select_one(".ts-text")
        link_element = segment_element.select_one(".ts-timestamp a")
        text = (text_element.get_text(" ", strip=True) if text_element else "").strip()
        if not text or link_element is None:
            continue
        link = link_element.get("href") or ""
        if youtube_video_id(link) != video_id:
            continue
        start = _timestamp_seconds(link)
        if start is None:
            continue
        speaker = (name_element.get_text(" ", strip=True) if name_element else "").strip()
        if not speaker:
            speaker = previous_speaker or ""
        if not speaker:
            continue
        previous_speaker = speaker
        segments.append({"speaker": speaker, "text": text, "start": start, "end": None})

    for index, segment in enumerate(segments):
        if index + 1 < len(segments):
            segment["end"] = max(float(segments[index + 1]["start"]), segment["start"])
        else:
            segment["end"] = segment["start"] + len(segment["text"].split()) / 2.5
    return segments


def lex_fridman_transcript_url_for_video(
    index_html: str, *, video_id: str
) -> str | None:
    """Find the transcript page for ``video_id`` on the lexfridman.com index.

    The index lists every episode as a ``div.guest`` holding a link to the
    YouTube video and, when a transcript is published, a link whose text is
    "Transcript".
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(index_html, "html.parser")
    for episode_element in soup.select("div.guest"):
        links = episode_element.find_all("a", href=True)
        if not any(youtube_video_id(link["href"]) == video_id for link in links):
            continue
        for link in links:
            if link.get_text(strip=True).lower() == "transcript":
                return str(link["href"])
        return None
    return None


@dataclass(frozen=True)
class PublishedTranscriptSource:
    """One publisher of speaker-labelled transcripts.

    ``index_url`` is the page that pairs YouTube video ids with transcript
    pages. ``find_transcript_url`` reads that index page. ``parse_transcript``
    reads one transcript page into segments.
    """

    name: str
    index_url: str
    find_transcript_url: Callable[..., str | None]
    parse_transcript: Callable[..., List[Dict[str, Any]]]


PUBLISHED_TRANSCRIPT_SOURCES: List[PublishedTranscriptSource] = [
    PublishedTranscriptSource(
        name="lexfridman.com",
        index_url="https://lexfridman.com/podcast",
        find_transcript_url=lex_fridman_transcript_url_for_video,
        parse_transcript=parse_lex_fridman_transcript_page,
    ),
]

# Index pages are large (the lexfridman.com index is 772,415 bytes) and list
# every episode, so one fetch per process serves every upload of a batch.
_index_page_cache: Dict[str, str] = {}


async def _fetch_text(url: str) -> str:
    import httpx

    async with httpx.AsyncClient(
        timeout=_REQUEST_TIMEOUT_SECONDS, follow_redirects=True, headers=_REQUEST_HEADERS
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text


def speaker_name_matches_avatar(speaker_name: str, avatar_name: str) -> bool:
    """Whether a published speaker name is the avatar's name.

    Compared case-insensitively on letters and digits only, so "Lex Fridman"
    matches "lex  fridman" and does not match "Lex" or "Alexander Fridman".
    """

    def _normalized(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()

    return bool(_normalized(speaker_name)) and _normalized(speaker_name) == _normalized(
        avatar_name
    )


async def load_published_transcript_as_diarization(
    source_url: str | None,
) -> Dict[str, Any] | None:
    """Return the published transcript of a YouTube video as a diarizer response.

    The returned dictionary has the shape ``transcribe_audio_diarize`` returns
    (``segments`` with ``speaker``, ``text``, ``start``, ``end``; ``text``;
    ``duration``; ``model``; ``inference_type``; ``total_cost``), with
    ``total_cost`` 0.0 because no vendor is paid, and with each segment's
    ``speaker`` holding the speaker's published name. ``published_transcript_url``
    names the page the segments were read from.

    Returns ``None`` when ``source_url`` is not a YouTube video, when no
    registered source publishes a transcript of that exact video id, or when a
    page cannot be fetched. A ``None`` return sends the upload to the diarizer
    as before, so a failure here never fails an upload.
    """
    video_id = youtube_video_id(source_url)
    if not video_id:
        return None
    for source in PUBLISHED_TRANSCRIPT_SOURCES:
        try:
            index_html = _index_page_cache.get(source.index_url)
            if index_html is None:
                index_html = await _fetch_text(source.index_url)
                _index_page_cache[source.index_url] = index_html
            transcript_url = source.find_transcript_url(index_html, video_id=video_id)
            if not transcript_url:
                continue
            transcript_html = await _fetch_text(transcript_url)
            segments = source.parse_transcript(transcript_html, video_id=video_id)
        except Exception as fetch_error:  # noqa: BLE001 - fall back to the diarizer
            logger.warning(
                "Published transcript lookup on %s failed for video %s: %s",
                source.name,
                video_id,
                fetch_error,
            )
            continue
        if not segments:
            continue
        logger.info(
            "Using the transcript published at %s for video %s (%d segments); "
            "no diarizer call is made",
            transcript_url,
            video_id,
            len(segments),
        )
        return {
            "segments": segments,
            "text": " ".join(segment["text"] for segment in segments),
            "duration": float(segments[-1]["end"]),
            "task": "transcribe",
            "model": PUBLISHED_TRANSCRIPT_MODEL_NAME,
            "inference_type": "diarization",
            "total_cost": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_ms": 0.0,
            "published_transcript_url": transcript_url,
            "published_transcript_source": source.name,
        }
    return None
