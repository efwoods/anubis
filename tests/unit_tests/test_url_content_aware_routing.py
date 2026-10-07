"""Content-aware URL routing: a URL is processed as the media type the URL serves.

An x.com post carrying a video reports ``Content-Type: text/html``. Before
this routing, ``URLDocumentLoaderClass`` read every such post as a web page, so
nothing said in the video reached the avatar. These tests pin the routing
without touching the network: the Content-Type probe, the yt_dlp metadata
probe and the downloads are all replaced.
"""

import pytest

from src.anubis.utils.classes import URLDocumentLoaderClass as loader_module
from src.anubis.utils.classes.URLDocumentLoaderClass import URLDocumentLoaderClass
from src.anubis.utils.context import GlobalContext

TWEET_URL = "https://x.com/i/status/975952542063448064"
ARTICLE_URL = "https://example.com/blog/post"
VIDEO_DATA_URI = "data:video/mp4;base64,AAAA"


@pytest.fixture
def routing_calls(monkeypatch):
    """Replace every network-facing helper; record which helpers were called."""
    routing_calls = {
        "content_type": ("text/html", None),
        "recorded_video_metadata": None,
        "video_probe_urls": [],
        "video_download_urls": [],
        "direct_download_urls": [],
        "article_calls": [],
    }

    async def _assert_public_host(url, *, allow_private=False):
        return None

    async def _probe_remote_content_type(url, *, allow_private_hosts):
        return routing_calls["content_type"]

    async def _resolve_remote_recorded_video(url, context):
        routing_calls["video_probe_urls"].append(url)
        return routing_calls["recorded_video_metadata"]

    async def _download_recorded_video_b64(url, *, media_max_bytes):
        routing_calls["video_download_urls"].append(url)
        return VIDEO_DATA_URI, "video/mp4"

    async def _fetch_direct_media_data_uri(
        url, content_type, *, media_max_bytes, allow_private_hosts
    ):
        routing_calls["direct_download_urls"].append(url)
        return f"data:{content_type};base64,AAAA"

    async def _load_article(
        self, url, user_id, assistant_id, *, quotes_per_line, url_kind=None
    ):
        routing_calls["article_calls"].append(
            {"url": url, "quotes_per_line": quotes_per_line, "url_kind": url_kind}
        )
        return [
            {
                "type": "text",
                "content": "post text",
                "metadata": {"url_kind": url_kind, "quotes_per_line": quotes_per_line},
            }
        ]

    monkeypatch.setattr(
        "src.anubis.utils.net.url_safety.assert_public_host", _assert_public_host
    )
    monkeypatch.setattr(
        loader_module, "_probe_remote_content_type", _probe_remote_content_type
    )
    monkeypatch.setattr(
        "src.anubis.utils.utility.resolve_remote_recorded_video",
        _resolve_remote_recorded_video,
    )
    monkeypatch.setattr(
        loader_module, "_download_recorded_video_b64", _download_recorded_video_b64
    )
    monkeypatch.setattr(
        loader_module, "_fetch_direct_media_data_uri", _fetch_direct_media_data_uri
    )
    monkeypatch.setattr(URLDocumentLoaderClass, "_load_article", _load_article)
    return routing_calls


@pytest.mark.asyncio
async def test_tweet_with_recorded_video_is_processed_as_video(routing_calls):
    routing_calls["recorded_video_metadata"] = {"duration": 61.998, "title": "post"}

    media_items = await URLDocumentLoaderClass().load(
        TWEET_URL, user_id="user", assistant_id="avatar"
    )

    assert len(media_items) == 1
    video_item = media_items[0]
    assert video_item["type"] == "video"
    assert video_item["base64_encoded_str"] == VIDEO_DATA_URI
    assert video_item["metadata"]["content_type"] == "video/mp4"
    assert video_item["metadata"]["url_kind"] == "twitter_video"
    assert video_item["metadata"]["filename"] == TWEET_URL
    assert routing_calls["article_calls"] == []


@pytest.mark.asyncio
async def test_tweet_without_video_keeps_the_quote_reader(routing_calls):
    routing_calls["recorded_video_metadata"] = None

    media_items = await URLDocumentLoaderClass().load(TWEET_URL)

    assert media_items[0]["type"] == "text"
    assert routing_calls["article_calls"] == [
        {"url": TWEET_URL, "quotes_per_line": True, "url_kind": "twitter"}
    ]
    assert routing_calls["video_download_urls"] == []


@pytest.mark.asyncio
async def test_failed_video_download_falls_back_to_the_page_reader(
    routing_calls, monkeypatch
):
    routing_calls["recorded_video_metadata"] = {"duration": 9.857}

    async def _download_produces_nothing(url, *, media_max_bytes):
        return "", ""

    monkeypatch.setattr(
        loader_module, "_download_recorded_video_b64", _download_produces_nothing
    )

    media_items = await URLDocumentLoaderClass().load(TWEET_URL)

    assert media_items[0]["type"] == "text"
    assert routing_calls["article_calls"][0]["url_kind"] == "twitter"


@pytest.mark.asyncio
async def test_direct_video_link_is_downloaded_without_the_page_probe(routing_calls):
    routing_calls["content_type"] = ("video/mp4", 1024)
    direct_video_url = "https://example.com/clip.mp4"

    media_items = await URLDocumentLoaderClass().load(direct_video_url)

    assert media_items[0]["type"] == "video"
    assert media_items[0]["metadata"]["url_kind"] == "article_video"
    assert routing_calls["direct_download_urls"] == [direct_video_url]
    assert routing_calls["video_probe_urls"] == []


@pytest.mark.asyncio
async def test_direct_audio_link_is_processed_as_audio(routing_calls):
    routing_calls["content_type"] = ("audio/mpeg", 2048)

    media_items = await URLDocumentLoaderClass().load("https://example.com/talk.mp3")

    assert media_items[0]["type"] == "audio"
    assert media_items[0]["base64_encoded_str"].startswith("data:audio/mpeg;base64,")
    assert media_items[0]["metadata"]["url_kind"] == "article_audio"


@pytest.mark.asyncio
async def test_direct_media_above_the_byte_cap_is_read_as_a_page(routing_calls):
    media_max_bytes = int(GlobalContext().url_media_max_bytes)
    routing_calls["content_type"] = ("video/mp4", media_max_bytes + 1)

    media_items = await URLDocumentLoaderClass().load("https://example.com/long.mp4")

    assert media_items[0]["type"] == "text"
    assert routing_calls["direct_download_urls"] == []


@pytest.mark.asyncio
async def test_direct_image_link_is_processed_as_an_image(routing_calls):
    routing_calls["content_type"] = ("image/jpeg", 4096)
    image_url = "https://pbs.twimg.com/media/photo.jpg"

    media_items = await URLDocumentLoaderClass().load(image_url)

    assert media_items[0]["type"] == "image_url"
    assert media_items[0]["image_url"] == {"url": image_url}
    assert media_items[0]["metadata"]["content_type"] == "image/jpeg"


@pytest.mark.asyncio
async def test_pdf_link_skips_the_video_probe(routing_calls):
    routing_calls["content_type"] = ("application/pdf", 4096)

    media_items = await URLDocumentLoaderClass().load("https://example.com/paper.pdf")

    assert media_items[0]["type"] == "text"
    assert routing_calls["video_probe_urls"] == []


@pytest.mark.asyncio
async def test_mediawiki_article_never_probes_for_media(routing_calls):
    wiki_url = "https://en.wikipedia.org/wiki/Shivon_Zilis"

    media_items = await URLDocumentLoaderClass().load(wiki_url)

    assert media_items[0]["type"] == "text"
    assert routing_calls["video_probe_urls"] == []
    assert routing_calls["direct_download_urls"] == []


@pytest.mark.asyncio
async def test_youtube_keeps_the_audio_only_path(routing_calls, monkeypatch):
    async def _download_youtube_audio(url):
        return "data:audio/mp3;base64,AAAA", ".mp3"

    monkeypatch.setattr(
        loader_module, "_download_youtube_audio_b64", _download_youtube_audio
    )

    media_items = await URLDocumentLoaderClass().load(
        "https://www.youtube.com/watch?v=abcdefghijk"
    )

    assert media_items[0]["type"] == "audio"
    assert media_items[0]["metadata"]["url_kind"] == "youtube_audio"
    assert routing_calls["video_probe_urls"] == []


@pytest.mark.asyncio
async def test_resolve_remote_recorded_video_rejects_live_and_empty(monkeypatch):
    from src.anubis.utils import utility

    probe_answers = iter(
        [
            {"duration": 6.77, "live_status": "not_live"},
            {"duration": 0.0, "live_status": None},
            {"duration": 120.0, "live_status": "is_live"},
        ]
    )

    async def _get_remote_video_metadata(url):
        return next(probe_answers)

    monkeypatch.setattr(
        utility, "get_remote_video_metadata", _get_remote_video_metadata
    )
    context = GlobalContext()

    assert (await utility.resolve_remote_recorded_video(TWEET_URL, context))[
        "duration"
    ] == 6.77
    assert await utility.resolve_remote_recorded_video(TWEET_URL, context) is None
    assert await utility.resolve_remote_recorded_video(TWEET_URL, context) is None


@pytest.mark.asyncio
async def test_upload_estimate_bills_a_video_post_as_video(monkeypatch):
    from src.api import webapp

    async def _probe_content_type(url):
        return "text/html"

    async def _resolve_remote_recorded_video(url, context):
        return {"duration": 61.998}

    async def _fetch_must_not_run(*args, **kwargs):
        raise AssertionError("a video post must not be downloaded as a page")

    monkeypatch.setattr(webapp, "probe_remote_url_content_type", _probe_content_type)
    monkeypatch.setattr(webapp, "fetch_remote_url_bytes", _fetch_must_not_run)
    monkeypatch.setattr(
        "src.anubis.utils.utility.resolve_remote_recorded_video",
        _resolve_remote_recorded_video,
    )
    entry = webapp._lightweight_url_media_entry(
        TWEET_URL, user_id="user", assistant_id="avatar"
    )

    estimated_tokens = await webapp._estimate_tokens_for_media_entry(entry)

    assert estimated_tokens > 0
    assert entry["estimated_media_seconds"] == 62.0
