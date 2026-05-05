"""Routing layer for URL ingestion.

Detects the type of URL the user provided and turns it into a list of media
items the existing ``process_media_item_task`` can handle:

* YouTube URL  → try subtitles via ``get_transcript``; on failure or when the
  caller suspects multi-speaker content, download audio with ``yt_dlp`` and
  return one ``type="audio"`` media item so the audio branch (now using
  ``transcribe_audio_diarize``) takes over.
* Linktree URL → fetch root, parse outbound links, return one ``type="url"``
  media item per link (those re-enter this class on the next pass).
* Tweet URL    → fetch with ``WebBaseLoader`` and tag with
  ``quotes_per_line=True`` so each line becomes one ``quote`` Document.
* Generic article URL → fetch with ``WebBaseLoader`` and return as plain text.

Per workspace ``.cursorrules`` we surface env-var validation by instantiating
``GlobalContext()`` at the top of ``load`` and passing the context through to
nested helpers that need it.
"""

import asyncio
import base64
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.utility import get_transcript

logger = logging.getLogger(__name__)


_YOUTUBE_HOSTS = frozenset({"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "music.youtube.com"})
_TWITTER_HOSTS = frozenset({"twitter.com", "www.twitter.com", "x.com", "www.x.com", "mobile.twitter.com"})
_LINKTREE_HOSTS = frozenset({"linktr.ee", "www.linktr.ee"})


def _classify_url(url: str) -> str:
    """Return a routing label: youtube / twitter / linktree / article."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return "article"
    if host in _YOUTUBE_HOSTS:
        return "youtube"
    if host in _TWITTER_HOSTS:
        return "twitter"
    if host in _LINKTREE_HOSTS:
        return "linktree"
    return "article"


class URLDocumentLoaderClass:
    """Expand a URL into one or more media-item dicts."""

    def __init__(self):
        # Surfaces env-var validation per workspace .cursorrules.
        self.context = GlobalContext()

    async def load(
        self,
        url: str,
        *,
        user_id: Optional[str] = None,
        assistant_id: Optional[str] = None,
        creator_id: Optional[str] = None,
        expect_multispeaker: bool = False,
    ) -> List[Dict[str, Any]]:
        """Expand ``url`` into media items consumable by ``process_media_item_task``."""
        url = (url or "").strip()
        if not url:
            return []

        kind = _classify_url(url)
        try:
            if kind == "youtube":
                return await self._load_youtube(
                    url, user_id, assistant_id, creator_id, expect_multispeaker
                )
            if kind == "linktree":
                return await self._load_linktree(
                    url, user_id, assistant_id, creator_id
                )
            if kind == "twitter":
                return await self._load_article(
                    url,
                    user_id,
                    assistant_id,
                    creator_id,
                    quotes_per_line=True,
                )
            return await self._load_article(
                url,
                user_id,
                assistant_id,
                creator_id,
                quotes_per_line=False,
            )
        except Exception as exc:  # pragma: no cover - logged for the operator
            logger.exception("URLDocumentLoaderClass.load failed for %s: %s", url, exc)
            return [
                {
                    "type": "text",
                    "content": f"[URL fetch failed: {url}]",
                    "metadata": {
                        "filename": url,
                        "source": url,
                        "user_id": user_id,
                        "assistant_id": assistant_id,
                        "creator_id": creator_id,
                        "status": "error",
                        "error": str(exc),
                    },
                }
            ]

    async def _load_article(
        self,
        url: str,
        user_id: Optional[str],
        assistant_id: Optional[str],
        creator_id: Optional[str],
        *,
        quotes_per_line: bool,
    ) -> List[Dict[str, Any]]:
        """Fetch a generic article URL via ``WebBaseLoader``."""
        try:
            from langchain_community.document_loaders import WebBaseLoader

            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, _load_webdocs_sync, url)
            content = "\n\n".join((d.page_content or "").strip() for d in docs).strip()
        except Exception as exc:
            logger.warning(
                "WebBaseLoader failed for %s, falling back to httpx: %s", url, exc
            )
            content = await _httpx_fallback_text(url)

        if not content:
            return []

        return [
            {
                "type": "text",
                "content": content,
                "metadata": {
                    "filename": url,
                    "source": url,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "creator_id": creator_id,
                    "quotes_per_line": quotes_per_line,
                    "url_kind": "twitter" if quotes_per_line else "article",
                },
            }
        ]

    async def _load_linktree(
        self,
        url: str,
        user_id: Optional[str],
        assistant_id: Optional[str],
        creator_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Fetch a Linktree page and return one ``type="url"`` item per outbound link."""
        try:
            text = await _httpx_fallback_text(url, return_html=True)
        except Exception as exc:
            logger.warning("Linktree fetch failed for %s: %s", url, exc)
            return []

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(text, "html.parser")
        candidate_links: List[str] = []
        seen = set()
        for a in soup.find_all("a"):
            href = (a.get("href") or "").strip()
            if not href or href.startswith("#"):
                continue
            try:
                host = (urlparse(href).hostname or "").lower()
            except Exception:
                continue
            if host in _LINKTREE_HOSTS:
                continue
            if host == "":
                continue
            if href in seen:
                continue
            seen.add(href)
            candidate_links.append(href)

        media_items: List[Dict[str, Any]] = []
        for link in candidate_links:
            media_items.append(
                {
                    "type": "url",
                    "url": link,
                    "metadata": {
                        "filename": link,
                        "source": link,
                        "user_id": user_id,
                        "assistant_id": assistant_id,
                        "creator_id": creator_id,
                        "url_kind": "linktree_child",
                        "linktree_root": url,
                    },
                }
            )
        return media_items

    async def _load_youtube(
        self,
        url: str,
        user_id: Optional[str],
        assistant_id: Optional[str],
        creator_id: Optional[str],
        expect_multispeaker: bool,
    ) -> List[Dict[str, Any]]:
        """Subtitles fast-path; otherwise download audio for diarization."""
        if not expect_multispeaker:
            try:
                loop = asyncio.get_event_loop()
                transcript = await loop.run_in_executor(
                    None, _get_transcript_sync, url
                )
                if (transcript or "").strip():
                    return [
                        {
                            "type": "text",
                            "content": transcript.strip(),
                            "metadata": {
                                "filename": url,
                                "source": url,
                                "user_id": user_id,
                                "assistant_id": assistant_id,
                                "creator_id": creator_id,
                                "url_kind": "youtube_subs",
                            },
                        }
                    ]
            except Exception as exc:
                logger.info(
                    "YouTube subs unavailable for %s, falling back to audio: %s",
                    url,
                    exc,
                )

        # Multi-speaker path or subs unavailable: download audio and let the
        # audio branch use OpenAI's hosted gpt-4o-transcribe-diarize.
        audio_b64, suffix = await _download_youtube_audio_b64(url)
        if not audio_b64:
            logger.warning("YouTube audio download failed for %s", url)
            return []

        return [
            {
                "type": "audio",
                "base64_encoded_str": audio_b64,
                "metadata": {
                    "filename": f"youtube{suffix}",
                    "content_type": f"audio/{suffix.lstrip('.') or 'mp3'}",
                    "source": url,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "creator_id": creator_id,
                    "url_kind": "youtube_audio",
                },
            }
        ]


def _load_webdocs_sync(url: str):
    """Run ``WebBaseLoader`` in a worker thread (sync API)."""
    from langchain_community.document_loaders import WebBaseLoader

    loader = WebBaseLoader(url)
    return loader.load()


def _get_transcript_sync(url: str) -> str:
    """Run the existing ``get_transcript`` helper synchronously in a worker."""
    return get_transcript(url=url, lang="en", save_txt=False)


async def _httpx_fallback_text(url: str, *, return_html: bool = False) -> str:
    """Last-resort plain-text or HTML fetch via httpx."""
    async with httpx.AsyncClient(
        follow_redirects=True, timeout=30.0
    ) as client:
        response = await client.get(url, headers={"User-Agent": "anubis/0.0.1"})
        response.raise_for_status()
        text = response.text
    if return_html:
        return text
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return re.sub(r"\n{3,}", "\n\n", soup.get_text("\n", strip=True))
    except Exception:
        return text


async def _download_youtube_audio_b64(url: str) -> tuple[str, str]:
    """Download a YouTube video's audio and return ``(base64_data_uri, suffix)``."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _download_youtube_audio_b64_sync, url)


def _download_youtube_audio_b64_sync(url: str) -> tuple[str, str]:
    """Sync helper for ``yt_dlp`` audio download + base64 encode."""
    import yt_dlp  # local import: heavy module

    with tempfile.TemporaryDirectory() as tmpdir:
        outtmpl = os.path.join(tmpdir, "%(id)s.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        # Locate the produced .mp3 (post-processor renames the file).
        mp3_path: Optional[str] = None
        for fname in os.listdir(tmpdir):
            if fname.lower().endswith(".mp3"):
                mp3_path = os.path.join(tmpdir, fname)
                break
        if not mp3_path:
            return "", ""

        with open(mp3_path, "rb") as fh:
            audio_b64 = base64.b64encode(fh.read()).decode("ascii")

        return f"data:audio/mp3;base64,{audio_b64}", Path(mp3_path).suffix
