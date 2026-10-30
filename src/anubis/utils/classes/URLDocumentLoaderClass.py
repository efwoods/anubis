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
* Instagram URL → fetch with ``WebBaseLoader`` and return as a single
  complete-document text item (``quotes_per_line=False``).
* Twitch URL   → fetch with ``WebBaseLoader`` and return as a single
  complete-document text item (``quotes_per_line=False``).
* Generic article URL → fetch with ``WebBaseLoader`` and return as plain text.
* MediaWiki article URL (Fandom, Wikipedia, wiki.gg, …) → fetch rendered HTML
  through ``api.php?action=parse``. Fandom's ``/wiki/…`` skin is behind a
  Cloudflare bot challenge (HTTP 403); the parse API on the same host is not.

Per workspace ``.cursorrules`` we surface env-var validation by instantiating
``GlobalContext()`` at the top of ``load`` and passing the context through to
nested helpers that need it.
"""

import asyncio
import base64
import html as html_lib
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse
from uuid import NAMESPACE_URL, uuid5

import httpx

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.utility import get_transcript

logger = logging.getLogger(__name__)


def _namespace_for(source: str) -> str:
    """Deterministic store key for a media source.

    Mirrors ``webapp._namespace_safe_formatted_filename`` so the dedup/skip keys
    produced here for expanded children line up exactly with the ones the upload
    endpoint computes for top-level items.
    """
    return str(uuid5(NAMESPACE_URL, source))


_YOUTUBE_HOSTS = frozenset({"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "music.youtube.com"})
_TWITTER_HOSTS = frozenset({"twitter.com", "www.twitter.com", "x.com", "www.x.com", "mobile.twitter.com"})
_LINKTREE_HOSTS = frozenset({"linktr.ee", "www.linktr.ee"})
_INSTAGRAM_HOSTS = frozenset(
    {"instagram.com", "www.instagram.com", "m.instagram.com"}
)
_TWITCH_HOSTS = frozenset(
    {"twitch.tv", "www.twitch.tv", "m.twitch.tv", "clips.twitch.tv"}
)

# Descriptive UA: Wikimedia rejects generic library defaults, and Fandom's
# parse API accepts this while still blocking the HTML skin.
ARTICLE_FETCH_USER_AGENT = (
    "AnubisAvatarBot/1.0 (https://neuralnexus.site; contact@neuralnexus.site)"
)

_FANDOM_HOST_SUFFIXES = (".fandom.com", ".wikia.com", ".wikia.org")
_WIKI_GG_HOST_SUFFIXES = (".wiki.gg",)
_WIKIMEDIA_HOST_SUFFIXES = (
    ".wikipedia.org",
    ".wikimedia.org",
    ".mediawiki.org",
    ".wiktionary.org",
    ".wikibooks.org",
    ".wikiquote.org",
    ".wikisource.org",
    ".wikinews.org",
    ".wikiversity.org",
    ".wikivoyage.org",
    ".wikidata.org",
)
_SKIP_MEDIAWIKI_TITLE_PREFIXES = ("special:", "media:")


def _host_has_suffix(host: str, suffixes: tuple[str, ...]) -> bool:
    """Return whether ``host`` is one of the suffix roots or a subdomain of them."""
    normalized = (host or "").lower().rstrip(".")
    for suffix in suffixes:
        root = suffix.lstrip(".")
        if normalized == root or normalized.endswith(suffix):
            return True
    return False


def _mediawiki_api_path_for_host(host: str) -> str | None:
    """Return the MediaWiki API path for a known wiki host, else ``None``.

    Fandom / wiki.gg expose ``/api.php`` at the wiki origin. Wikimedia family
    wikis expose ``/w/api.php``. Unknown hosts are left alone: GitHub wikis
    also use a ``/wiki/…`` path and must not be rewritten to a missing API.
    """
    if _host_has_suffix(host, _FANDOM_HOST_SUFFIXES) or _host_has_suffix(
        host, _WIKI_GG_HOST_SUFFIXES
    ):
        return "/api.php"
    if _host_has_suffix(host, _WIKIMEDIA_HOST_SUFFIXES):
        return "/w/api.php"
    return None


def _mediawiki_article_title(parsed) -> str | None:
    """Extract a MediaWiki article title from ``/wiki/Title`` or ``title=``."""
    path = unquote(parsed.path or "")
    title = ""
    path_lower = path.lower()
    if path_lower.startswith("/wiki/"):
        title = path[len("/wiki/") :]
    elif path_lower.endswith("/index.php") or path_lower.endswith("/w/index.php"):
        title = (parse_qs(parsed.query or "").get("title") or [""])[0]
        title = unquote(title)
    title = (title or "").strip().strip("/")
    if not title:
        return None
    if title.lower().startswith(_SKIP_MEDIAWIKI_TITLE_PREFIXES):
        return None
    return title


def mediawiki_article_api_target(url: str) -> tuple[str, str] | None:
    """Return ``(api.php URL, page title)`` for a known MediaWiki article URL.

    ``None`` when the URL is not a Fandom / Wikimedia / wiki.gg article page
    (so callers fall through to a normal HTTP fetch).
    """
    try:
        parsed = urlparse((url or "").strip())
    except Exception:
        return None
    host = (parsed.hostname or "").lower()
    api_path = _mediawiki_api_path_for_host(host)
    if not api_path:
        return None
    title = _mediawiki_article_title(parsed)
    if not title:
        return None
    api_url = f"{parsed.scheme or 'https'}://{parsed.netloc}{api_path}"
    return api_url, title


def html_document_from_mediawiki_parse(
    payload: Dict[str, Any], *, fallback_title: str = ""
) -> str:
    """Turn a MediaWiki ``action=parse`` JSON body into a full HTML document.

    The parse API returns the article fragment (``.mw-parser-output``, portable
    infobox, quote tables) without ``<title>``. Structured extraction reads
    ``<title>`` and the infobox, so wrap the fragment in a tiny document.
    """
    parse = payload.get("parse") if isinstance(payload, dict) else None
    if not isinstance(parse, dict):
        return ""
    title = (parse.get("title") or fallback_title or "").strip()
    text = parse.get("text") or ""
    if isinstance(text, dict):
        text = text.get("*") or ""
    if not isinstance(text, str) or not text.strip():
        return ""
    escaped_title = html_lib.escape(title, quote=True)
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        f"<title>{escaped_title}</title></head><body>{text}</body></html>"
    )


def _visible_text_from_html(html_text: str) -> str:
    """Strip scripts/styles from HTML and return visible text."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_text or "", "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return re.sub(r"\n{3,}", "\n\n", soup.get_text("\n", strip=True))
    except Exception:
        return html_text or ""


async def fetch_mediawiki_article_html(url: str) -> tuple[bytes, str] | None:
    """Fetch a MediaWiki article via ``action=parse`` when the URL is a wiki page.

    Returns ``(html_bytes, 'text/html')`` on success, or ``None`` when the URL
    is not a known wiki article or the parse API does not yield article HTML.
    Callers then fall back to fetching the URL as a normal page.
    """
    target = mediawiki_article_api_target(url)
    if target is None:
        return None
    api_url, page_title = target
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            response = await client.get(
                api_url,
                params={
                    "action": "parse",
                    "page": page_title,
                    "prop": "text",
                    "format": "json",
                    "redirects": "1",
                    "formatversion": "2",
                    "disablelimitreport": "1",
                    "disableeditsection": "1",
                },
                headers={"User-Agent": ARTICLE_FETCH_USER_AGENT},
            )
            response.raise_for_status()
            payload = response.json()
    except Exception as fetch_error:
        logger.warning(
            "MediaWiki parse API failed for %s via %s: %s",
            url,
            api_url,
            fetch_error,
        )
        return None
    if isinstance(payload, dict) and payload.get("error"):
        logger.warning(
            "MediaWiki parse API refused %s: %s",
            url,
            payload.get("error"),
        )
        return None
    html_text = html_document_from_mediawiki_parse(
        payload, fallback_title=page_title
    )
    if not html_text:
        return None
    return html_text.encode("utf-8"), "text/html"


# Medium serves every article page behind a Cloudflare challenge (HTTP 403 for
# every User-Agent, including browser and search-engine agents), but the
# author's or publication's RSS feed is served openly and carries the full
# article body in ``content:encoded``. The feed lists only the most recent
# posts, so an older article is not reachable through the feed.
_MEDIUM_HOST = "medium.com"
_MEDIUM_POST_ID_PATTERN = re.compile(r"(?:^|-)([0-9a-f]{8,16})$")
_RSS_CONTENT_NAMESPACE = "{http://purl.org/rss/1.0/modules/content/}"
MEDIUM_ARTICLE_NOT_IN_FEED_REASON = (
    "Medium blocks automated reading of this article, and the article is not "
    "among the author's recent posts in the author's Medium feed. Save the "
    "article as a PDF or text file and upload that file instead."
)


def is_medium_url(url: str) -> bool:
    """Report whether a URL is hosted on ``medium.com`` or a ``*.medium.com`` subdomain."""
    try:
        host = (urlparse((url or "").strip()).hostname or "").lower()
    except Exception:
        return False
    return host == _MEDIUM_HOST or host.endswith("." + _MEDIUM_HOST)


def medium_article_feed_target(url: str) -> tuple[str, str] | None:
    """Return ``(feed URL, post id)`` for a Medium article URL.

    Handled article URL shapes:

    * ``medium.com/@author/slug-<post id>`` → ``medium.com/feed/@author``
    * ``medium.com/<publication>/slug-<post id>`` → ``medium.com/feed/<publication>``
    * ``<author>.medium.com/slug-<post id>`` → ``<author>.medium.com/feed``

    ``None`` when the URL is not a Medium article (so callers fall through to a
    normal HTTP fetch). ``medium.com/p/<post id>`` names no author or
    publication, so no feed can be chosen and the URL returns ``None``.
    """
    if not is_medium_url(url):
        return None
    parsed = urlparse((url or "").strip())
    host = (parsed.hostname or "").lower()
    path_segments = [segment for segment in (parsed.path or "").split("/") if segment]
    if not path_segments:
        return None
    post_id_match = _MEDIUM_POST_ID_PATTERN.search(path_segments[-1].lower())
    if post_id_match is None:
        return None
    post_id = post_id_match.group(1)
    if host == _MEDIUM_HOST:
        if len(path_segments) < 2 or path_segments[0].lower() in {"p", "feed"}:
            return None
        return f"https://{_MEDIUM_HOST}/feed/{path_segments[0]}", post_id
    return f"https://{host}/feed", post_id


def html_document_from_medium_feed(feed_xml: bytes, post_id: str) -> str:
    """Return the article HTML for ``post_id`` from a Medium RSS feed body.

    The matching ``<item>`` is the one whose ``guid`` or ``link`` contains the
    post id. The ``content:encoded`` body is wrapped in a small document with
    a ``<title>`` because structured extraction reads ``<title>``. An empty
    string means the feed does not hold the post.
    """
    import xml.etree.ElementTree as element_tree

    try:
        feed_root = element_tree.fromstring(feed_xml)
    except element_tree.ParseError:
        return ""
    for feed_item in feed_root.iter("item"):
        item_identifiers = " ".join(
            (feed_item.findtext("guid") or "", feed_item.findtext("link") or "")
        ).lower()
        if post_id not in item_identifiers:
            continue
        article_body = feed_item.findtext(f"{_RSS_CONTENT_NAMESPACE}encoded") or ""
        if not article_body.strip():
            return ""
        escaped_title = html_lib.escape(
            (feed_item.findtext("title") or "").strip(), quote=True
        )
        return (
            '<!DOCTYPE html><html><head><meta charset="utf-8">'
            f"<title>{escaped_title}</title></head><body>{article_body}</body></html>"
        )
    return ""


async def fetch_medium_article_html(url: str) -> tuple[bytes, str] | None:
    """Fetch a Medium article body from the author's or publication's RSS feed.

    Returns ``(html_bytes, 'text/html')`` on success, or ``None`` when the URL
    is not a Medium article, the feed cannot be fetched, or the feed does not
    hold the article (the feed lists only the most recent posts).
    """
    target = medium_article_feed_target(url)
    if target is None:
        return None
    feed_url, post_id = target
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            response = await client.get(
                feed_url, headers={"User-Agent": ARTICLE_FETCH_USER_AGENT}
            )
            response.raise_for_status()
            feed_xml = response.content
    except Exception as fetch_error:
        logger.warning(
            "Medium feed fetch failed for %s via %s: %s", url, feed_url, fetch_error
        )
        return None
    html_text = html_document_from_medium_feed(feed_xml, post_id)
    if not html_text:
        logger.warning(
            "Medium feed %s does not hold post %s (requested %s)",
            feed_url,
            post_id,
            url,
        )
        return None
    return html_text.encode("utf-8"), "text/html"


def bot_challenged_article_target(url: str) -> bool:
    """Report whether a URL is an article whose HTML page sits behind a bot challenge.

    Those article pages are read through a side channel (the MediaWiki parse
    API, or the Medium RSS feed) instead of the challenged page, so content-type
    probes and media probes must skip the challenged page.
    """
    return (
        mediawiki_article_api_target(url) is not None
        or medium_article_feed_target(url) is not None
    )


async def fetch_bot_challenged_article_html(url: str) -> tuple[bytes, str] | None:
    """Fetch article HTML through the MediaWiki parse API, else the Medium feed.

    ``None`` when neither side channel applies to the URL or yields the article.
    """
    mediawiki_article = await fetch_mediawiki_article_html(url)
    if mediawiki_article is not None:
        return mediawiki_article
    return await fetch_medium_article_html(url)


def _is_youtube_playlist_url(parsed) -> bool:
    """A pure playlist (``/playlist?list=...``) or any YouTube URL carrying a
    ``list=`` query param but no single ``v=`` video — those enumerate. A
    ``watch?v=X&list=Y`` URL stays a single video."""
    path = (parsed.path or "").lower()
    query = parse_qs(parsed.query or "")
    if "playlist" in path:
        return True
    return "list" in query and "v" not in query


def _is_youtube_channel_url(parsed) -> bool:
    """Report whether a link names a channel, handle, or user page.

    Such a link is a body of work rather than one video.

    A channel link names everything a person published, so downloading "the"
    audio behind such a link picks an arbitrary video. Channel links enumerate
    into their videos instead, the way a playlist does, and every video is then
    ingested on its own terms.
    """
    path = (parsed.path or "").lower().rstrip("/")
    query = parse_qs(parsed.query or "")
    if "v" in query:
        return False
    return (
        path.startswith("/@")
        or path.startswith("/channel/")
        or path.startswith("/c/")
        or path.startswith("/user/")
        or path in ("", "/feed", "/feed/subscriptions")
    )


def channel_videos_url(url: str) -> str:
    """Point a channel link at the channel's Videos tab.

    ``yt_dlp`` flat-extracts a bare channel URL into the channel's tabs
    (Videos, Shorts, Live) rather than into videos, so the Videos tab is asked
    for by name. A link that already names a tab, and a link that is not a
    channel, are returned unchanged.
    """
    try:
        parsed = urlparse(url)
    except Exception:  # noqa: BLE001 - an unparseable link is left alone
        return url
    if not _is_youtube_channel_url(parsed):
        return url
    path = (parsed.path or "").rstrip("/")
    if path.lower().endswith(("/videos", "/shorts", "/streams", "/live", "/featured")):
        return url
    if not path or path in ("/feed", "/feed/subscriptions"):
        return url
    return f"{parsed.scheme or 'https'}://{parsed.netloc}{path}/videos"


def _classify_url(url: str) -> str:
    """Return a routing label: youtube_playlist / youtube / twitter / linktree / instagram / twitch / article."""
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
    except Exception:
        return "article"
    if host in _YOUTUBE_HOSTS:
        # A playlist and a channel are both bodies of work: each enumerates
        # into single videos rather than being downloaded as one recording.
        if _is_youtube_playlist_url(parsed) or _is_youtube_channel_url(parsed):
            return "youtube_playlist"
        return "youtube"
    if host in _TWITTER_HOSTS:
        return "twitter"
    if host in _LINKTREE_HOSTS:
        return "linktree"
    if host in _INSTAGRAM_HOSTS:
        return "instagram"
    if host in _TWITCH_HOSTS:
        return "twitch"
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
    ) -> List[Dict[str, Any]]:
        """Expand ``url`` into media items consumable by ``process_media_item_task``."""
        url = (url or "").strip()
        if not url:
            return []

        kind = _classify_url(url)
        try:
            if kind == "youtube_playlist":
                return await self._load_youtube_playlist(
                    url, user_id, assistant_id
                )
            if kind == "youtube":
                return await self._load_youtube(
                    url, user_id, assistant_id,
                )
            if kind == "linktree":
                return await self._load_linktree(
                    url, user_id, assistant_id
                )
            # Every other URL is routed by what the URL serves, not by the host
            # name: an x.com post that carries a video is processed as a video,
            # and only a URL serving no media is read as a web page.
            return await self._load_content_aware_url(
                url,
                user_id,
                assistant_id,
                url_kind=kind,
                quotes_per_line=kind == "twitter",
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
                        "status": "error",
                        "error": str(exc),
                    },
                }
            ]

    async def _load_content_aware_url(
        self,
        url: str,
        user_id: str | None,
        assistant_id: str | None,
        *,
        url_kind: str,
        quotes_per_line: bool,
    ) -> List[Dict[str, Any]]:
        """Process a URL as the media type the URL serves, else read the URL as a page.

        ``_load_remote_media`` returns a video, audio or image media item when the
        URL serves one; an empty list means the URL serves no media and the
        existing ``_load_article`` reader handles the page exactly as before.
        """
        remote_media_items = await self._load_remote_media(
            url, user_id, assistant_id, url_kind=url_kind
        )
        if remote_media_items:
            return remote_media_items
        return await self._load_article(
            url,
            user_id,
            assistant_id,
            quotes_per_line=quotes_per_line,
            url_kind=url_kind,
        )

    async def _load_remote_media(
        self,
        url: str,
        user_id: str | None,
        assistant_id: str | None,
        *,
        url_kind: str,
    ) -> List[Dict[str, Any]]:
        """Return one media item when ``url`` serves a video, audio or image, else ``[]``.

        Two checks, in order of cost:

        1. The Content-Type of the URL itself. A direct ``.mp4`` / ``.mp3`` /
           ``.jpg`` link names the media type in the response header.
        2. For a web page, the yt_dlp metadata probe shared with the upload
           estimate (``resolve_remote_recorded_video``). An x.com post, an
           Instagram reel or a Twitch clip reports ``text/html`` while the page
           hosts a recorded video; the probe finds the video and yt_dlp
           downloads the video.

        Every failure returns ``[]`` so the URL falls back to the page reader:
        a URL that cannot be read as media is never lost.
        """
        from src.anubis.utils.net.url_safety import UnsafeUrlError, assert_public_host
        from src.anubis.utils.utility import resolve_remote_recorded_video

        # A MediaWiki or Medium article never hosts a recording, and the
        # article HTML page sits behind a bot challenge that fails every probe.
        if bot_challenged_article_target(url):
            return []
        allow_private_hosts = _url_fetch_allows_private_hosts(self.context)
        try:
            await assert_public_host(url, allow_private=allow_private_hosts)
        except UnsafeUrlError as unsafe_url_error:
            logger.warning(
                "Media probe skipped for %s: %s", url, unsafe_url_error
            )
            return []
        media_max_bytes = int(self.context.url_media_max_bytes or 0)

        try:
            content_type, content_length = await _probe_remote_content_type(
                url, allow_private_hosts=allow_private_hosts
            )
        except Exception as probe_error:  # noqa: BLE001 - fall through to the page probe
            logger.info(
                "Content-Type probe failed for %s (%s: %s)",
                url,
                type(probe_error).__name__,
                probe_error,
            )
            content_type, content_length = "", None

        media_metadata: Dict[str, Any] = {
            "filename": url,
            "source": url,
            "user_id": user_id,
            "assistant_id": assistant_id,
        }

        if content_type in _URL_STILL_IMAGE_MIMES:
            return [
                {
                    "type": "image_url",
                    "image_url": {"url": url},
                    "metadata": {
                        **media_metadata,
                        "content_type": content_type,
                        "url_kind": f"{url_kind}_image",
                    },
                }
            ]

        if content_type.startswith("audio/") or content_type.startswith("video/"):
            media_type = "video" if content_type.startswith("video/") else "audio"
            if content_length is not None and content_length > media_max_bytes:
                logger.warning(
                    "Direct %s at %s is %d bytes, above URL_MEDIA_MAX_BYTES %d; reading the URL as a page",
                    media_type,
                    url,
                    content_length,
                    media_max_bytes,
                )
                return []
            try:
                media_data_uri = await _fetch_direct_media_data_uri(
                    url,
                    content_type,
                    media_max_bytes=media_max_bytes,
                    allow_private_hosts=allow_private_hosts,
                )
            except Exception as download_error:  # noqa: BLE001 - fall back to the page reader
                logger.warning(
                    "Direct %s download failed for %s (%s: %s)",
                    media_type,
                    url,
                    type(download_error).__name__,
                    download_error,
                )
                return []
            if not media_data_uri:
                return []
            return [
                {
                    "type": media_type,
                    "base64_encoded_str": media_data_uri,
                    "metadata": {
                        **media_metadata,
                        "content_type": content_type,
                        "url_kind": f"{url_kind}_{media_type}",
                    },
                }
            ]

        # A header that names any other concrete type (PDF, plain text, JSON)
        # is not a page that could host a video; the page reader handles the URL.
        if content_type and content_type not in _URL_PAGE_CONTENT_TYPES:
            return []

        recorded_video_metadata = await resolve_remote_recorded_video(
            url, self.context
        )
        if recorded_video_metadata is None:
            return []
        try:
            video_data_uri, video_content_type = await _download_recorded_video_b64(
                url, media_max_bytes=media_max_bytes
            )
        except Exception as download_error:  # noqa: BLE001 - fall back to the page reader
            logger.warning(
                "Recorded video download failed for %s (%s: %s); reading the URL as a page",
                url,
                type(download_error).__name__,
                download_error,
            )
            return []
        if not video_data_uri:
            logger.warning(
                "No video under URL_MEDIA_MAX_BYTES %d was downloaded for %s; reading the URL as a page",
                media_max_bytes,
                url,
            )
            return []
        logger.info(
            "URL %s hosts a recorded video (%.3f s); processing the URL as video",
            url,
            float(recorded_video_metadata.get("duration") or 0.0),
        )
        return [
            {
                "type": "video",
                "base64_encoded_str": video_data_uri,
                "metadata": {
                    **media_metadata,
                    "content_type": video_content_type,
                    "url_kind": f"{url_kind}_video",
                },
            }
        ]

    async def _load_article(
        self,
        url: str,
        user_id: Optional[str],
        assistant_id: Optional[str],
        *,
        quotes_per_line: bool,
        url_kind: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch a generic article URL via MediaWiki parse or the Medium feed, else ``WebBaseLoader``.

        ``url_kind`` lets the caller tag the produced media item with the
        original routing label (``twitter``, ``instagram``, ``twitch``,
        ``article``). When omitted, falls back to the legacy behavior of
        ``twitter`` for ``quotes_per_line=True`` and ``article`` otherwise.

        Fandom ``/wiki/…`` skins and Medium article pages are
        Cloudflare-challenged; the MediaWiki parse API and the Medium RSS feed
        are not, so those articles skip ``WebBaseLoader``.
        """
        raw_html = ""
        content = ""
        challenged_article = await fetch_bot_challenged_article_html(url)
        if challenged_article is not None:
            raw_html = challenged_article[0].decode("utf-8", errors="replace")
            content = _visible_text_from_html(raw_html).strip()
        else:
            try:
                loop = asyncio.get_event_loop()
                docs = await loop.run_in_executor(None, _load_webdocs_sync, url)
                content = "\n\n".join(
                    (d.page_content or "").strip() for d in docs
                ).strip()
            except Exception as exc:
                logger.warning(
                    "WebBaseLoader failed for %s, falling back to httpx: %s",
                    url,
                    exc,
                )
                content = await _httpx_fallback_text(url)

            # Retain the raw HTML so downstream structured-web extraction can
            # parse biography and quote structures. The plain-text ``content``
            # is the fallback when the page is not a subject page.
            try:
                raw_html = await _httpx_fallback_text(url, return_html=True)
            except Exception as html_exc:
                logger.debug(
                    "raw HTML fetch failed for %s (continuing): %s", url, html_exc
                )

        if not content:
            return []

        resolved_url_kind = url_kind or (
            "twitter" if quotes_per_line else "article"
        )
        return [
            {
                "type": "text",
                "content": content,
                "metadata": {
                    "filename": url,
                    "source": url,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "quotes_per_line": quotes_per_line,
                    "url_kind": resolved_url_kind,
                    "raw_html": raw_html,
                },
            }
        ]

    async def _load_linktree(
        self,
        url: str,
        user_id: Optional[str],
        assistant_id: Optional[str],
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
                        "url_kind": "linktree_child",
                        "linktree_root": url,
                        "namespace_filename": _namespace_for(link),
                    },
                }
            )
        return media_items

    async def _load_youtube_playlist(
        self,
        url: str,
        user_id: Optional[str],
        assistant_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Enumerate every video in a YouTube playlist (no cap).

        Returns one ``type="url"`` watch item per entry; each re-enters this
        class on the next pass and flows through the single-video YouTube path
        (subtitles fast-path, else audio + diarization)."""
        entries, playlist_title = await _extract_playlist_entries(url)
        if not entries:
            logger.warning("YouTube playlist produced no entries: %s", url)
            return []

        # The playlist's own deterministic key. Every video is keyed by a single
        # uuid5 over ``{playlist_ns}::{video_ns}`` — derived from BOTH identities so
        # it stays disambiguated from other videos and from the same video in a
        # different playlist, but emitted as one opaque hash so the store key holds
        # no ``::`` separator. The playlist a video belongs to is recovered from the
        # playlist_namespace_filename / playlist_url / title metadata below, not by
        # parsing the key: /list_avatar_documents groups on playlist_url, the
        # skip-set matches the opaque key, and /delete_avatar_document drops a whole
        # playlist via playlist_namespace_filename or a single video via this key.
        playlist_ns = _namespace_for(url)

        media_items: List[Dict[str, Any]] = []
        for entry in entries:
            video_id = entry.get("id")
            watch_url = entry.get("url") or (
                f"https://www.youtube.com/watch?v={video_id}" if video_id else None
            )
            if not watch_url:
                continue
            video_ns = _namespace_for(watch_url)
            media_items.append(
                {
                    "type": "url",
                    "url": watch_url,
                    "metadata": {
                        "filename": watch_url,
                        "source": watch_url,
                        "user_id": user_id,
                        "assistant_id": assistant_id,
                        "url_kind": "youtube_playlist_entry",
                        "playlist_url": url,
                        "playlist_namespace_filename": playlist_ns,
                        "playlist_title": playlist_title,
                        "video_title": entry.get("title") or "",
                        "namespace_filename": _namespace_for(
                            f"{playlist_ns}::{video_ns}"
                        ),
                    },
                }
            )
        logger.info(
            "Expanded YouTube playlist %s (%s) into %d videos",
            url,
            playlist_title or "untitled",
            len(media_items),
        )
        return media_items

    async def _load_youtube(
        self,
        url: str,
        user_id: Optional[str],
        assistant_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Subtitles fast-path; otherwise download audio for diarization."""

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
                    # Keep the original URL as the filename — never rename to a
                    # generic "youtube.mp3". ``content_type`` still carries the real
                    # audio container so the audio branch decodes correctly.
                    "filename": url,
                    "content_type": f"audio/{suffix.lstrip('.') or 'mp3'}",
                    "source": url,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "url_kind": "youtube_audio",
                },
            }
        ]


def _load_webdocs_sync(url: str):
    """Run ``WebBaseLoader`` in a worker thread (sync API)."""
    from langchain_community.document_loaders import WebBaseLoader

    loader = WebBaseLoader(url)
    return loader.load()


async def _httpx_fallback_text(url: str, *, return_html: bool = False) -> str:
    """Last-resort plain-text or HTML fetch via httpx.

    MediaWiki article URLs go through the parse API and Medium article URLs
    through the RSS feed first, so Cloudflare-challenged Fandom and Medium
    pages still yield the article body.
    """
    challenged_article = await fetch_bot_challenged_article_html(url)
    if challenged_article is not None:
        html_text = challenged_article[0].decode("utf-8", errors="replace")
        return html_text if return_html else _visible_text_from_html(html_text)
    async with httpx.AsyncClient(
        follow_redirects=True, timeout=30.0
    ) as client:
        response = await client.get(
            url, headers={"User-Agent": ARTICLE_FETCH_USER_AGENT}
        )
        response.raise_for_status()
        text = response.text
    if return_html:
        return text
    return _visible_text_from_html(text)


async def _extract_playlist_entries(url: str) -> tuple[List[Dict[str, Any]], str]:
    """Return ``(entries, playlist_title)`` for a YouTube playlist.

    ``entries`` is the flat list (``{id, url, ...}``); ``playlist_title`` is the
    human-readable playlist name (empty string when yt_dlp can't resolve it),
    surfaced so individual videos can be listed under their playlist.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _extract_playlist_entries_sync, url)


def _extract_playlist_entries_sync(url: str) -> tuple[List[Dict[str, Any]], str]:
    """Sync helper: flat-extract playlist entries + title via ``yt_dlp`` (no download)."""
    import yt_dlp  # local import: heavy module

    ydl_opts = {
        "extract_flat": "in_playlist",
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    # A channel link is asked for by its Videos tab, because a bare channel URL
    # flat-extracts into the channel's tabs rather than into videos.
    extraction_url = channel_videos_url(url)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(extraction_url, download=False)
    except Exception as exc:  # pragma: no cover - logged for the operator
        logger.exception(
            "yt_dlp playlist extraction failed for %s: %s", extraction_url, exc
        )
        return [], ""

    entries = (info or {}).get("entries") or []
    playlist_title = (info or {}).get("title") or ""
    # Drop unavailable/private entries that yt_dlp returns as ``None``, and
    # flatten the one level of nesting a channel tab can still produce.
    videos: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("_type") == "playlist":
            videos.extend(
                nested
                for nested in (entry.get("entries") or [])
                if isinstance(nested, dict)
            )
            continue
        videos.append(entry)
    return videos, playlist_title


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
            # One recording, never a whole playlist or channel. Without this a
            # link that carries a list downloads every video, and the caller
            # then keeps an arbitrary one of them.
            "noplaylist": True,
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

        # Locate the produced .mp3 (the post-processor renames the file). The
        # file is chosen by the id of the video that was asked for, so a
        # directory holding more than one download can never yield audio of a
        # different recording; directory order decides nothing.
        produced_mp3_names = sorted(
            name for name in os.listdir(tmpdir) if name.lower().endswith(".mp3")
        )
        if not produced_mp3_names:
            return "", ""
        requested_video_id = str((info or {}).get("id") or "")
        preferred_name = f"{requested_video_id}.mp3"
        chosen_name = (
            preferred_name
            if requested_video_id and preferred_name in produced_mp3_names
            else produced_mp3_names[0]
        )
        if requested_video_id and chosen_name != preferred_name:
            logger.warning(
                "yt_dlp produced no mp3 named for %s; falling back to %s",
                requested_video_id,
                chosen_name,
            )
        mp3_path = os.path.join(tmpdir, chosen_name)

        with open(mp3_path, "rb") as fh:
            audio_b64 = base64.b64encode(fh.read()).decode("ascii")

        return f"data:audio/mp3;base64,{audio_b64}", Path(mp3_path).suffix


# The still-image types the media graph describes (mirrors
# ``_ALLOWED_STILL_IMAGE_MIMES`` in ``process_media_graph/utils/nodes.py``,
# which imports this module and so cannot be imported from here).
_URL_STILL_IMAGE_MIMES = frozenset(
    {"image/jpeg", "image/png", "image/gif", "image/webp"}
)

# Content-Types under which a URL may be a web page hosting a recorded video.
# ``application/octet-stream`` is included because some hosts label every
# response that way; yt_dlp then decides from the page itself.
_URL_PAGE_CONTENT_TYPES = frozenset(
    {"text/html", "application/xhtml+xml", "application/octet-stream"}
)

# Container extension produced by yt_dlp → MIME type of the media item.
_RECORDED_VIDEO_EXTENSION_MIMES = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
}


def _url_fetch_allows_private_hosts(context: GlobalContext) -> bool:
    """Whether ``URL_FETCH_ALLOW_PRIVATE_HOSTS`` lets a URL resolve inside this network."""
    flag = str(getattr(context, "url_fetch_allow_private_hosts", None) or "false")
    return flag.strip().lower() in ("1", "true", "yes", "on")


def _content_type_without_parameters(content_type_header: str | None) -> str:
    return (content_type_header or "").split(";")[0].strip().lower()


async def _probe_remote_content_type(
    url: str, *, allow_private_hosts: bool
) -> tuple[str, int | None]:
    """Return ``(content_type, content_length)`` for ``url`` without downloading the body.

    HEAD first, then a ranged GET of the first 512 bytes when HEAD fails or
    names no useful type (the same order as ``probe_remote_url_content_type``
    in ``webapp.py``). Redirects are followed only through
    ``get_with_public_host_guard``, which re-checks every hop.
    ``content_length`` is ``None`` when the server does not report a size.
    """
    from src.anubis.utils.net.url_safety import get_with_public_host_guard

    request_headers = {"User-Agent": ARTICLE_FETCH_USER_AGENT}
    async with httpx.AsyncClient(follow_redirects=False, timeout=30.0) as client:
        try:
            head_response = await client.head(url, headers=request_headers)
        except httpx.HTTPError:
            head_response = None
        if head_response is not None and head_response.status_code < 300:
            head_content_type = _content_type_without_parameters(
                head_response.headers.get("content-type")
            )
            if head_content_type and head_content_type != "application/octet-stream":
                head_content_length = head_response.headers.get("content-length")
                return head_content_type, (
                    int(head_content_length)
                    if head_content_length and head_content_length.isdigit()
                    else None
                )
        ranged_response = await get_with_public_host_guard(
            client,
            url,
            headers={**request_headers, "Range": "bytes=0-511"},
            allow_private=allow_private_hosts,
        )
        ranged_response.raise_for_status()
        ranged_content_type = _content_type_without_parameters(
            ranged_response.headers.get("content-type")
        )
        # ``Content-Range: bytes 0-511/1048576`` carries the full size after the slash.
        content_range = ranged_response.headers.get("content-range") or ""
        total_size_text = content_range.rpartition("/")[2]
        return ranged_content_type, (
            int(total_size_text) if total_size_text.isdigit() else None
        )


async def _fetch_direct_media_data_uri(
    url: str,
    content_type: str,
    *,
    media_max_bytes: int,
    allow_private_hosts: bool,
) -> str:
    """Download a direct audio / video URL into a data URI, or ``""`` above the byte cap."""
    from src.anubis.utils.net.url_safety import get_with_public_host_guard

    async with httpx.AsyncClient(follow_redirects=False, timeout=120.0) as client:
        media_response = await get_with_public_host_guard(
            client,
            url,
            headers={"User-Agent": ARTICLE_FETCH_USER_AGENT},
            allow_private=allow_private_hosts,
        )
        media_response.raise_for_status()
        media_bytes = media_response.content
    if len(media_bytes) > media_max_bytes:
        logger.warning(
            "Direct media at %s is %d bytes, above URL_MEDIA_MAX_BYTES %d",
            url,
            len(media_bytes),
            media_max_bytes,
        )
        return ""
    media_base64 = base64.b64encode(media_bytes).decode("ascii")
    return f"data:{content_type};base64,{media_base64}"


async def _download_recorded_video_b64(
    url: str, *, media_max_bytes: int
) -> tuple[str, str]:
    """Download the recorded video a web page hosts; return ``(data_uri, content_type)``."""
    return await asyncio.to_thread(
        _download_recorded_video_b64_sync, url, media_max_bytes
    )


def _download_recorded_video_b64_sync(url: str, media_max_bytes: int) -> tuple[str, str]:
    """Sync helper for the yt_dlp video download + base64 encode.

    Prefers a single-file mp4 under ``media_max_bytes`` (``<?`` also accepts a
    format whose size the host does not report), and ``max_filesize`` stops any
    download that turns out larger. Returns ``("", "")`` when no file under the
    cap was produced.
    """
    import yt_dlp  # local import: heavy module

    with tempfile.TemporaryDirectory() as download_directory:
        download_options = {
            "format": (
                f"best[ext=mp4][filesize<?{media_max_bytes}]"
                f"/best[filesize<?{media_max_bytes}]"
            ),
            "outtmpl": os.path.join(download_directory, "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            # One recording, never a whole playlist or channel.
            "noplaylist": True,
            "max_filesize": media_max_bytes,
        }
        with yt_dlp.YoutubeDL(download_options) as downloader:
            video_information = downloader.extract_info(url, download=True)

        # A post holding several videos comes back as a playlist; the first
        # entry is the recording that the post leads with.
        if (video_information or {}).get("entries"):
            first_entries = [
                entry for entry in video_information["entries"] if isinstance(entry, dict)
            ]
            video_information = first_entries[0] if first_entries else {}
        requested_video_id = str((video_information or {}).get("id") or "")

        produced_video_names = sorted(
            name
            for name in os.listdir(download_directory)
            if Path(name).suffix.lower() in _RECORDED_VIDEO_EXTENSION_MIMES
        )
        if not produced_video_names:
            return "", ""
        # The file is chosen by the id of the recording that was asked for, so
        # directory order decides nothing.
        chosen_video_name = next(
            (
                name
                for name in produced_video_names
                if requested_video_id and Path(name).stem == requested_video_id
            ),
            produced_video_names[0],
        )
        chosen_video_path = os.path.join(download_directory, chosen_video_name)
        if os.path.getsize(chosen_video_path) > media_max_bytes:
            return "", ""
        video_content_type = _RECORDED_VIDEO_EXTENSION_MIMES[
            Path(chosen_video_name).suffix.lower()
        ]
        with open(chosen_video_path, "rb") as video_file:
            video_base64 = base64.b64encode(video_file.read()).decode("ascii")
        return f"data:{video_content_type};base64,{video_base64}", video_content_type
