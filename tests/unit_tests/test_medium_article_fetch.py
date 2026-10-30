"""Medium article fetch for identity-media URL uploads.

Every Medium article page answers an automated fetch with a Cloudflare 403
challenge (measured 2026-10-11 against
``medium.com/@shivon/the-current-state-of-machine-intelligence-f76c20db2fe1``
with four different User-Agents). The author's RSS feed at
``medium.com/feed/@shivon`` answers 200 and carries the full article body in
``content:encoded``. These tests cover URL → feed mapping, feed parsing, and
the upload and loader fetch paths without hitting the network.
"""

import httpx
import pytest
from fastapi import HTTPException

from src.anubis.utils.classes.URLDocumentLoaderClass import (
    MEDIUM_ARTICLE_NOT_IN_FEED_REASON,
    _httpx_fallback_text,
    bot_challenged_article_target,
    fetch_medium_article_html,
    html_document_from_medium_feed,
    medium_article_feed_target,
)

SHIVON_ARTICLE_URL = (
    "https://medium.com/@shivon/the-current-state-of-machine-intelligence-f76c20db2fe1"
)
SHIVON_FEED_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss xmlns:dc="http://purl.org/dc/elements/1.1/"
     xmlns:content="http://purl.org/rss/1.0/modules/content/" version="2.0">
<channel>
  <title><![CDATA[Stories by Shivon Zilis on Medium]]></title>
  <item>
    <title><![CDATA[The Current State of Machine Intelligence 3.0]]></title>
    <link>https://medium.com/@shivon/the-current-state-of-machine-intelligence-3-0-e4d305da032e?source=rss</link>
    <guid isPermaLink="false">https://medium.com/p/e4d305da032e</guid>
    <content:encoded><![CDATA[<p>Third edition body.</p>]]></content:encoded>
  </item>
  <item>
    <title><![CDATA[The Current State of Machine Intelligence]]></title>
    <link>https://medium.com/@shivon/the-current-state-of-machine-intelligence-f76c20db2fe1?source=rss</link>
    <guid isPermaLink="false">https://medium.com/p/f76c20db2fe1</guid>
    <content:encoded><![CDATA[<p>I spent the last three months learning about every artificial intelligence startup I could find.</p>]]></content:encoded>
  </item>
</channel>
</rss>
"""


class _FeedResponse:
    def __init__(self, *, status_code=200, content=SHIVON_FEED_XML):
        self.status_code = status_code
        self.content = content
        self.request = httpx.Request("GET", "https://medium.com/feed/@shivon")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=self.request,
                response=httpx.Response(self.status_code, request=self.request),
            )


class _RecordingFeedClient:
    def __init__(self, feed_response):
        self.feed_response = feed_response
        self.requested_urls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exception_details):
        return False

    async def get(self, url, params=None, headers=None):
        self.requested_urls.append(str(url))
        return self.feed_response


def _install_feed_client(monkeypatch, feed_response):
    feed_client = _RecordingFeedClient(feed_response)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: feed_client)
    return feed_client


# --------------------------------------------------------------------------- #
# URL → feed target
# --------------------------------------------------------------------------- #


def test_author_article_maps_to_author_feed():
    assert medium_article_feed_target(SHIVON_ARTICLE_URL) == (
        "https://medium.com/feed/@shivon",
        "f76c20db2fe1",
    )


def test_publication_article_maps_to_publication_feed():
    assert medium_article_feed_target(
        "https://medium.com/machine-intelligence-report/the-current-state-of-machine-intelligence-2-0-a1b2c3d4e5f6?source=rss"
    ) == ("https://medium.com/feed/machine-intelligence-report", "a1b2c3d4e5f6")


def test_subdomain_article_maps_to_subdomain_feed():
    assert medium_article_feed_target(
        "https://shivon.medium.com/the-current-state-of-machine-intelligence-f76c20db2fe1"
    ) == ("https://shivon.medium.com/feed", "f76c20db2fe1")


@pytest.mark.parametrize(
    "url",
    [
        "https://medium.com/p/f76c20db2fe1",
        "https://medium.com/@shivon",
        "https://medium.com/feed/@shivon",
        "https://example.com/@shivon/an-article-f76c20db2fe1",
        "https://notmedium.com/@shivon/an-article-f76c20db2fe1",
        "",
    ],
)
def test_urls_without_a_feed_target_are_not_rewritten(url):
    assert medium_article_feed_target(url) is None


def test_medium_article_counts_as_bot_challenged():
    assert bot_challenged_article_target(SHIVON_ARTICLE_URL)
    assert not bot_challenged_article_target("https://example.com/article")


# --------------------------------------------------------------------------- #
# Feed → article HTML
# --------------------------------------------------------------------------- #


def test_feed_item_matching_the_post_id_is_wrapped_with_title():
    article_html = html_document_from_medium_feed(SHIVON_FEED_XML, "f76c20db2fe1")
    assert "<title>The Current State of Machine Intelligence</title>" in article_html
    assert "every artificial intelligence startup" in article_html
    assert "Third edition body" not in article_html


def test_feed_without_the_post_yields_no_document():
    assert html_document_from_medium_feed(SHIVON_FEED_XML, "0123456789ab") == ""


def test_unparseable_feed_yields_no_document():
    assert html_document_from_medium_feed(b"<html>challenge</html", "f76c20db2fe1") == ""


@pytest.mark.asyncio
async def test_fetch_medium_reads_the_feed_not_the_article_page(monkeypatch):
    feed_client = _install_feed_client(monkeypatch, _FeedResponse())
    article = await fetch_medium_article_html(SHIVON_ARTICLE_URL)
    assert article is not None
    article_bytes, content_type = article
    assert content_type == "text/html"
    assert b"every artificial intelligence startup" in article_bytes
    assert feed_client.requested_urls == ["https://medium.com/feed/@shivon"]


@pytest.mark.asyncio
async def test_fetch_medium_returns_none_when_the_feed_fails(monkeypatch):
    _install_feed_client(monkeypatch, _FeedResponse(status_code=403))
    assert await fetch_medium_article_html(SHIVON_ARTICLE_URL) is None


@pytest.mark.asyncio
async def test_httpx_fallback_uses_the_medium_feed(monkeypatch):
    _install_feed_client(monkeypatch, _FeedResponse())
    article_text = await _httpx_fallback_text(SHIVON_ARTICLE_URL)
    assert "every artificial intelligence startup" in article_text


# --------------------------------------------------------------------------- #
# Upload path (src/api/webapp.py)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_probe_medium_article_is_html_without_network(monkeypatch):
    def refuse_network(*a, **k):
        raise AssertionError("probe must not fetch a Medium article page")

    monkeypatch.setattr(httpx, "AsyncClient", refuse_network)
    from src.api.webapp import probe_remote_url_content_type

    assert await probe_remote_url_content_type(SHIVON_ARTICLE_URL) == "text/html"


@pytest.mark.asyncio
async def test_fetch_remote_url_bytes_uses_the_medium_feed(monkeypatch):
    from src.api import webapp

    async def public_host(url, allow_private=False):
        return None

    monkeypatch.setattr(webapp, "assert_public_host", public_host)
    monkeypatch.setattr(webapp, "url_fetch_allows_private_hosts", lambda: False)
    _install_feed_client(monkeypatch, _FeedResponse())

    article_bytes, content_type = await webapp.fetch_remote_url_bytes(
        SHIVON_ARTICLE_URL
    )
    assert content_type == "text/html"
    assert b"every artificial intelligence startup" in article_bytes


@pytest.mark.asyncio
async def test_medium_article_missing_from_feed_is_refused_with_a_readable_reason(
    monkeypatch,
):
    from src.api import webapp

    async def public_host(url, allow_private=False):
        return None

    monkeypatch.setattr(webapp, "assert_public_host", public_host)
    monkeypatch.setattr(webapp, "url_fetch_allows_private_hosts", lambda: False)
    feed_client = _install_feed_client(monkeypatch, _FeedResponse())

    with pytest.raises(HTTPException) as refusal:
        await webapp.fetch_remote_url_bytes(
            "https://medium.com/@shivon/an-older-article-0123456789ab"
        )
    assert refusal.value.status_code == 400
    assert refusal.value.detail == MEDIUM_ARTICLE_NOT_IN_FEED_REASON
    assert feed_client.requested_urls == ["https://medium.com/feed/@shivon"]
