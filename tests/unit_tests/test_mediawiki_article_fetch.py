"""MediaWiki article fetch for identity-media URL uploads.

Fandom ``/wiki/…`` skins are Cloudflare-challenged (HTTP 403,
``cf-mitigated: challenge``). The parse API on the same host is not, and
returns the article HTML structured extraction already parses. These tests
cover URL rewriting, parse-JSON wrapping, and the upload/loader fetch path
without hitting the network.
"""

import httpx
import pytest
from fastapi import HTTPException

from src.anubis.utils.classes.URLDocumentLoaderClass import (
    URLDocumentLoaderClass,
    _httpx_fallback_text,
    fetch_mediawiki_article_html,
    html_document_from_mediawiki_parse,
    mediawiki_article_api_target,
)
from src.subgraphs.process_media_graph.utils.structured_web_extraction import (
    page_looks_like_subject_page,
    parse_html_into_structured_blocks,
)

JESTER_URL = "https://criticalrole.fandom.com/wiki/Jester_Lavorre"
PARSE_FRAGMENT = """
<div class="mw-parser-output">
  <aside class="portable-infobox"><h2 class="pi-title">Jester Lavorre</h2></aside>
  <h2><span class="mw-headline">Personality</span></h2>
  <p>Jester is excitable and eager.</p>
  <h2><span class="mw-headline">Notable quotes</span></h2>
  <table class="wikitable">
    <tr><th>Context</th><th>Comment(s)</th></tr>
    <tr><td>When asked</td><td>"It's okay. I'm a trickster."</td></tr>
  </table>
</div>
"""


class _FakeResponse:
    def __init__(self, *, status=200, json_body=None, content=b"", headers=None):
        self.status_code = status
        self._json_body = json_body
        self.content = content
        self.headers = headers or {"content-type": "application/json"}
        self.text = (
            content.decode("utf-8") if isinstance(content, bytes) else str(content)
        )
        self.request = httpx.Request("GET", "https://example.invalid")

    def json(self):
        return self._json_body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=self.request,
                response=httpx.Response(self.status_code, request=self.request),
            )


class _RecordingClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None, headers=None):
        self.calls.append(
            {"url": str(url), "params": dict(params or {}), "headers": headers}
        )
        return self.response


def _install_client(monkeypatch, response):
    client = _RecordingClient(response)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: client)
    return client


# --------------------------------------------------------------------------- #
# URL → parse API target
# --------------------------------------------------------------------------- #


def test_fandom_wiki_url_maps_to_api_php():
    api_url, title = mediawiki_article_api_target(JESTER_URL)
    assert api_url == "https://criticalrole.fandom.com/api.php"
    assert title == "Jester_Lavorre"


def test_wikipedia_wiki_url_maps_to_w_api_php():
    api_url, title = mediawiki_article_api_target(
        "https://en.wikipedia.org/wiki/Lex_Fridman"
    )
    assert api_url == "https://en.wikipedia.org/w/api.php"
    assert title == "Lex_Fridman"


def test_wiki_gg_url_maps_to_api_php():
    api_url, title = mediawiki_article_api_target(
        "https://calamity.wiki.gg/wiki/Some_Page"
    )
    assert api_url == "https://calamity.wiki.gg/api.php"
    assert title == "Some_Page"


def test_subpage_and_index_php_title_are_kept():
    api_url, title = mediawiki_article_api_target(
        "https://fallout.fandom.com/wiki/CompanionCurie.txt/COM"
    )
    assert title == "CompanionCurie.txt/COM"
    api_url, title = mediawiki_article_api_target(
        "https://en.wikipedia.org/w/index.php?title=Lex_Fridman"
    )
    assert api_url == "https://en.wikipedia.org/w/api.php"
    assert title == "Lex_Fridman"


def test_non_mediawiki_urls_are_not_rewritten():
    assert mediawiki_article_api_target("https://lexfridman.com/") is None
    assert (
        mediawiki_article_api_target("https://github.com/owner/repo/wiki/Home")
        is None
    )
    assert mediawiki_article_api_target("https://criticalrole.fandom.com/") is None
    assert (
        mediawiki_article_api_target(
            "https://criticalrole.fandom.com/wiki/Special:Search"
        )
        is None
    )


# --------------------------------------------------------------------------- #
# Parse JSON → HTML document
# --------------------------------------------------------------------------- #


def test_parse_payload_is_wrapped_with_title():
    html = html_document_from_mediawiki_parse(
        {"parse": {"title": "Jester Lavorre", "text": PARSE_FRAGMENT}},
        fallback_title="Jester_Lavorre",
    )
    assert "<title>Jester Lavorre</title>" in html
    assert "portable-infobox" in html
    parsed = parse_html_into_structured_blocks(html, url=JESTER_URL)
    assert parsed["page_title"] == "Jester Lavorre"
    assert parsed["infobox_subject_name"] == "Jester Lavorre"
    assert page_looks_like_subject_page(parsed) is True


def test_legacy_parse_text_star_key_is_unwrapped():
    html = html_document_from_mediawiki_parse(
        {"parse": {"title": "Curie", "text": {"*": "<p>scientist</p>"}}}
    )
    assert "scientist" in html
    assert "<title>Curie</title>" in html


def test_parse_error_or_empty_text_yields_no_document():
    assert html_document_from_mediawiki_parse({"error": {"code": "missingtitle"}}) == ""
    assert (
        html_document_from_mediawiki_parse({"parse": {"title": "X", "text": ""}}) == ""
    )
    assert html_document_from_mediawiki_parse({}) == ""


# --------------------------------------------------------------------------- #
# Live fetch helpers (httpx mocked)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_fetch_mediawiki_hits_parse_api_not_the_wiki_skin(monkeypatch):
    payload = {
        "parse": {"title": "Jester Lavorre", "text": PARSE_FRAGMENT},
    }
    client = _install_client(monkeypatch, _FakeResponse(json_body=payload))
    body, content_type = await fetch_mediawiki_article_html(JESTER_URL)
    assert content_type == "text/html"
    assert b"Jester Lavorre" in body
    assert b"excitable" in body
    assert len(client.calls) == 1
    assert client.calls[0]["url"] == "https://criticalrole.fandom.com/api.php"
    assert client.calls[0]["params"]["action"] == "parse"
    assert client.calls[0]["params"]["page"] == "Jester_Lavorre"
    assert "wiki/Jester" not in client.calls[0]["url"]


@pytest.mark.asyncio
async def test_fetch_mediawiki_returns_none_for_non_wiki_urls(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("non-wiki URLs must not call the parse API")

    monkeypatch.setattr(httpx, "AsyncClient", boom)
    assert await fetch_mediawiki_article_html("https://lexfridman.com/") is None


@pytest.mark.asyncio
async def test_fetch_mediawiki_returns_none_when_api_errors(monkeypatch):
    _install_client(
        monkeypatch,
        _FakeResponse(json_body={"error": {"code": "missingtitle"}}),
    )
    assert await fetch_mediawiki_article_html(JESTER_URL) is None


@pytest.mark.asyncio
async def test_httpx_fallback_uses_mediawiki_html(monkeypatch):
    html = (
        b"<!DOCTYPE html><html><head><title>Jester Lavorre</title></head>"
        b"<body><p>excitable</p></body></html>"
    )

    async def fake_mw(url):
        assert url == JESTER_URL
        return html, "text/html"

    monkeypatch.setattr(
        "src.anubis.utils.classes.URLDocumentLoaderClass.fetch_mediawiki_article_html",
        fake_mw,
    )
    text = await _httpx_fallback_text(JESTER_URL)
    raw = await _httpx_fallback_text(JESTER_URL, return_html=True)
    assert "excitable" in text
    assert "<title>Jester Lavorre</title>" in raw


@pytest.mark.asyncio
async def test_load_article_skips_webloader_for_mediawiki(monkeypatch):
    html = (
        b"<!DOCTYPE html><html><head><title>Jester Lavorre</title></head>"
        b"<body><p>excitable</p></body></html>"
    )

    async def fake_mw(url):
        return html, "text/html"

    monkeypatch.setattr(
        "src.anubis.utils.classes.URLDocumentLoaderClass.fetch_mediawiki_article_html",
        fake_mw,
    )

    def boom(*a, **k):
        raise AssertionError("WebBaseLoader must not run for MediaWiki articles")

    monkeypatch.setattr(
        "src.anubis.utils.classes.URLDocumentLoaderClass._load_webdocs_sync",
        boom,
    )
    items = await URLDocumentLoaderClass()._load_article(
        JESTER_URL, "u1", "a1", quotes_per_line=False
    )
    assert len(items) == 1
    assert "excitable" in items[0]["content"]
    assert "Jester Lavorre" in items[0]["metadata"]["raw_html"]
    assert items[0]["metadata"]["url_kind"] == "article"


@pytest.mark.asyncio
async def test_fetch_remote_url_bytes_uses_mediawiki_for_fandom(monkeypatch):
    html = b"<html><head><title>Jester Lavorre</title></head><body>bio</body></html>"

    async def fake_mw(url):
        assert "fandom.com" in url
        return html, "text/html"

    monkeypatch.setattr(
        "src.anubis.utils.classes.URLDocumentLoaderClass.fetch_mediawiki_article_html",
        fake_mw,
    )
    from src.api.webapp import fetch_remote_url_bytes

    body, content_type = await fetch_remote_url_bytes(JESTER_URL)
    assert body == html
    assert content_type == "text/html"


@pytest.mark.asyncio
async def test_probe_fandom_wiki_is_html_without_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("probe must not fetch a known wiki article")

    monkeypatch.setattr(httpx, "AsyncClient", boom)
    from src.api.webapp import probe_remote_url_content_type

    assert await probe_remote_url_content_type(JESTER_URL) == "text/html"


@pytest.mark.asyncio
async def test_fetch_remote_url_bytes_maps_http_error_to_400(monkeypatch):
    async def no_mw(url):
        return None

    monkeypatch.setattr(
        "src.anubis.utils.classes.URLDocumentLoaderClass.fetch_mediawiki_article_html",
        no_mw,
    )
    request = httpx.Request("GET", "https://example.com/blocked")
    error_response = httpx.Response(403, request=request)

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers=None):
            return error_response

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _Client())
    from src.api.webapp import fetch_remote_url_bytes

    with pytest.raises(HTTPException) as excinfo:
        await fetch_remote_url_bytes("https://example.com/blocked")
    assert excinfo.value.status_code == 400
    assert "403" in str(excinfo.value.detail)


@pytest.mark.asyncio
async def test_build_media_entries_accepts_fandom_wiki_url(monkeypatch):
    html = (
        b"<html><head><title>Jester Lavorre</title></head>"
        b"<body><p>bio</p></body></html>"
    )

    async def fake_fetch(url, max_bytes=25 * 1024 * 1024):
        return html, "text/html"

    async def fake_probe(url):
        return "text/html"

    import src.api.webapp as webapp

    monkeypatch.setattr(webapp, "fetch_remote_url_bytes", fake_fetch)
    monkeypatch.setattr(webapp, "probe_remote_url_content_type", fake_probe)
    entries = await webapp._build_media_entries_for_url(
        JESTER_URL,
        reference_image=False,
        reference_audio=False,
        user_id="u1",
        assistant_id="a1",
        rich=True,
    )
    assert len(entries) == 1
    assert entries[0]["page_url"] == JESTER_URL
    assert entries[0]["content_type"] == "text/html"
    assert entries[0]["base64_encoded_str"].startswith("data:text/html;base64,")
