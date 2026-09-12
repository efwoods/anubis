"""Web search for deep research: Tavily and the existing browser tools, merged.

Two providers, both optional, both used when available:

* **Tavily** (``TAVILY_API_KEY``) — a search API that returns page content
  with each result, so most sources need no second fetch.
* **The headless Chromium browser tools** (``BROWSER_TOOLS_ENABLED``) — the
  same per-process Chromium ``src/anubis/utils/tools/browser`` drives for the
  avatar's browsing, here pointed at DuckDuckGo's HTML results page. When the
  browser gate is off, the same results page is fetched over plain HTTP.

Results from every provider are merged and de-duplicated by URL; page text
for results that arrived without content is read with the plain-text fetch
``URLDocumentLoaderClass`` already uses for articles.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from src.anubis.utils.context import GlobalContext

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"
_PAGE_TEXT_CHARACTER_LIMIT = 20_000
_USER_AGENT = "Mozilla/5.0 (compatible; NeuralNexusResearch/1.0)"


@dataclass
class ImageCandidate:
    """One picture the search or a read page offered, and where it came from.

    ``origin`` names how the picture was found — a Tavily image result, or the
    tag on a page that declared it — because that is what ranks candidates:
    a page's ``og:image`` is the picture that page chose to represent itself,
    which is a far better portrait bet than an arbitrary inline image.
    """

    url: str
    description: str = ""
    origin: str = ""
    source_url: str = ""


@dataclass
class SearchResult:
    """One web result: where the result came from and what the page says."""

    url: str
    title: str = ""
    snippet: str = ""
    content: str = ""
    provider: str = ""
    queries: list[str] = field(default_factory=list)
    images: list[ImageCandidate] = field(default_factory=list)


def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    parsed = urlparse(url)
    if not parsed.scheme:
        return ""
    cleaned = parsed._replace(fragment="")
    return cleaned.geturl().rstrip("/")


def _unwrap_duckduckgo_redirect(href: str) -> str:
    """DuckDuckGo wraps result links as ``/l/?uddg=<encoded url>``."""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target)
    return href


def parse_duckduckgo_html(html: str, limit: int) -> list[SearchResult]:
    """Parse result links, titles, and snippets from DuckDuckGo's HTML results page."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "html.parser")
    results: list[SearchResult] = []
    for result_node in soup.select("div.result"):
        anchor = result_node.select_one("a.result__a")
        if anchor is None:
            continue
        url = _normalize_url(_unwrap_duckduckgo_redirect(anchor.get("href", "")))
        if not url:
            continue
        snippet_node = result_node.select_one(".result__snippet")
        results.append(
            SearchResult(
                url=url,
                title=anchor.get_text(" ", strip=True),
                snippet=snippet_node.get_text(" ", strip=True) if snippet_node else "",
                provider="duckduckgo",
            )
        )
        if len(results) >= limit:
            break
    return results


async def tavily_search(query: str, *, limit: int, api_key: str) -> list[SearchResult]:
    """Search Tavily, which returns the page content along with each result."""
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max(1, min(limit, 20)),
        "search_depth": "advanced",
        "include_raw_content": True,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=10.0)) as client:
        response = await client.post(TAVILY_SEARCH_URL, json=payload)
        response.raise_for_status()
        body = response.json()
    results: list[SearchResult] = []
    for entry in body.get("results") or []:
        url = _normalize_url(entry.get("url") or "")
        if not url:
            continue
        content = entry.get("raw_content") or entry.get("content") or ""
        results.append(
            SearchResult(
                url=url,
                title=entry.get("title") or "",
                snippet=(entry.get("content") or "")[:500],
                content=str(content)[:_PAGE_TEXT_CHARACTER_LIMIT],
                provider="tavily",
            )
        )
    return results


async def _duckduckgo_html_via_http(query: str) -> str:
    async with httpx.AsyncClient(
        follow_redirects=True, timeout=httpx.Timeout(30.0, connect=10.0)
    ) as client:
        response = await client.get(
            DUCKDUCKGO_HTML_URL,
            params={"q": query},
            headers={"User-Agent": _USER_AGENT},
        )
        response.raise_for_status()
        return response.text


async def _duckduckgo_html_via_browser(query: str, context: GlobalContext) -> str:
    """Render the results page in the conversation-browser Chromium (browser tools gate)."""
    from src.anubis.utils.tools.browser.browser_tools import (
        get_browser_toolkit_tools,
        release_conversation_browser,
    )

    conversation_key = "deep-research"
    tools = await get_browser_toolkit_tools(context, conversation_key=conversation_key)
    if not tools:
        raise RuntimeError("browser tools unavailable")
    try:
        navigate_tool = next(tool for tool in tools if tool.name == "navigate_browser")
        encoded_query = httpx.QueryParams({"q": query})["q"]
        await navigate_tool.arun({"url": f"{DUCKDUCKGO_HTML_URL}?q={encoded_query}"})
        # The toolkit shares one browser; read the page the tool just navigated.
        page = navigate_tool.async_browser.contexts[0].pages[-1]
        return await page.content()
    finally:
        await release_conversation_browser(conversation_key)


async def browser_search(
    query: str, *, limit: int, context: GlobalContext
) -> list[SearchResult]:
    """DuckDuckGo results through the browser tools when enabled, else plain HTTP."""
    from src.anubis.utils.tools.browser.browser_tools import browser_tools_enabled

    html = ""
    if browser_tools_enabled(context):
        try:
            html = await _duckduckgo_html_via_browser(query, context)
        except Exception as browser_error:  # noqa: BLE001 - fall back to HTTP
            logger.info(
                "Browser search failed (%s); using the HTTP results page", browser_error
            )
    if not html:
        html = await _duckduckgo_html_via_http(query)
    return parse_duckduckgo_html(html, limit)


def html_to_markdown(html: str) -> str:
    """HTML to Markdown, the way the LangGraph deep-research guide feeds pages to the model.

    Markdown keeps the headings, lists, and links that plain-text extraction
    flattens, which is what lets the fact extractor tell a biography from a
    navigation menu. ``markdownify`` is optional: without the package the
    article loader's BeautifulSoup text extraction is used instead.
    """
    try:
        from markdownify import markdownify
    except ImportError:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html or "", "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer"]):
            tag.decompose()
        return re.sub(r"\n{3,}", "\n\n", soup.get_text("\n", strip=True))
    return markdownify(html or "", strip=["script", "style"])


# The tags a page uses to declare the picture that represents it, best first.
# og:image is what a page hands a social card, so it is the page's own answer to
# "which picture is this page about" — for a biography page, that is a portrait.
_PAGE_IMAGE_META_PROPERTIES = (
    ("og:image", "og:image"),
    ("og:image:secure_url", "og:image"),
    ("twitter:image", "twitter:image"),
    ("twitter:image:src", "twitter:image"),
)


def extract_page_images(
    html: str, base_url: str, *, limit: int = 4
) -> list[ImageCandidate]:
    """Return the pictures a page declares as representing itself.

    Reads the social-card tags, then ``link[rel=image_src]``, then JSON-LD
    ``image``, then — on a MediaWiki page — the first infobox image, which is
    the lead portrait on a biography article. Relative sources are resolved
    against ``base_url``. Inline body images are deliberately not collected:
    they are mostly logos, icons and unrelated illustrations, and every extra
    candidate costs a download and a vision call downstream.
    """
    from urllib.parse import urljoin

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "html.parser")
    candidates: list[ImageCandidate] = []
    seen: set[str] = set()

    def remember(raw_url: str, origin: str, description: str = "") -> None:
        if len(candidates) >= limit:
            return
        resolved = _normalize_url(urljoin(base_url, (raw_url or "").strip()))
        if not resolved or resolved in seen:
            return
        seen.add(resolved)
        candidates.append(
            ImageCandidate(
                url=resolved,
                description=description,
                origin=origin,
                source_url=base_url,
            )
        )

    page_title = (soup.title.get_text(strip=True) if soup.title else "") or ""
    for property_name, origin in _PAGE_IMAGE_META_PROPERTIES:
        for tag in soup.find_all("meta", attrs={"property": property_name}):
            remember(tag.get("content", ""), origin, page_title)
        for tag in soup.find_all("meta", attrs={"name": property_name}):
            remember(tag.get("content", ""), origin, page_title)

    for tag in soup.find_all("link", attrs={"rel": "image_src"}):
        remember(tag.get("href", ""), "image_src", page_title)

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            payload = json.loads(script.string or "")
        except Exception:  # noqa: BLE001 - malformed JSON-LD is common and not an error
            continue
        for entry in payload if isinstance(payload, list) else [payload]:
            if not isinstance(entry, dict):
                continue
            image = entry.get("image")
            if isinstance(image, str):
                remember(image, "json-ld", str(entry.get("name") or page_title))
            elif isinstance(image, dict):
                remember(str(image.get("url") or ""), "json-ld", str(entry.get("name") or page_title))
            elif isinstance(image, list):
                for item in image:
                    if isinstance(item, str):
                        remember(item, "json-ld", page_title)
                    elif isinstance(item, dict):
                        remember(str(item.get("url") or ""), "json-ld", page_title)

    infobox_image = soup.select_one(".infobox img, #mw-content-text figure img")
    if infobox_image is not None:
        remember(infobox_image.get("src", ""), "wikipedia", page_title)

    return candidates


@dataclass
class PageRead:
    """One page read once: its Markdown text and the pictures it declares."""

    text: str
    images: list[ImageCandidate] = field(default_factory=list)


async def read_page(url: str, *, collect_images: bool = True) -> PageRead:
    """Fetch a page once and return both its Markdown and its declared pictures.

    The HTML is fetched a single time and used twice. Reading the page for facts
    and harvesting a portrait candidate from it are the same download, so
    acquisition costs the research pipeline nothing extra for the pages it was
    already going to read.
    """
    from src.anubis.utils.classes.URLDocumentLoaderClass import _httpx_fallback_text

    html = await _httpx_fallback_text(url, return_html=True)
    text = html_to_markdown(html)
    bounded_text = re.sub(r"[ \t]+", " ", text or "")[:_PAGE_TEXT_CHARACTER_LIMIT]
    images: list[ImageCandidate] = []
    if collect_images:
        try:
            images = extract_page_images(html or "", url)
        except Exception as image_error:  # noqa: BLE001 - facts matter more than pictures
            logger.debug("Could not read images from %s: %s", url, image_error)
    return PageRead(text=bounded_text, images=images)


async def read_page_text(url: str) -> str:
    """Read the page as bounded Markdown, fetched through the article loader's HTTP path."""
    page = await read_page(url, collect_images=False)
    return page.text


async def search_images(
    query: str, *, limit: int, context: GlobalContext | None = None
) -> list[ImageCandidate]:
    """Search for pictures matching ``query``.

    Tavily is the only provider that answers with images; the DuckDuckGo HTML
    endpoint the browser path uses has no image results at all. A deployment
    with no Tavily key therefore gets an empty list here and falls back to the
    pictures the read pages declared, which is thinner but still works.
    """
    context = context or GlobalContext()
    api_key = getattr(context, "tavily_api_key", None)
    if not api_key:
        return []
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max(1, min(limit, 20)),
        "search_depth": "advanced",
        "include_images": True,
        "include_image_descriptions": True,
        "include_raw_content": False,
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=10.0)) as client:
            response = await client.post(TAVILY_SEARCH_URL, json=payload)
            response.raise_for_status()
            body = response.json()
    except Exception as search_error:  # noqa: BLE001 - a failed image search is not fatal
        logger.warning("Tavily image search failed for %r: %s", query, search_error)
        return []
    candidates: list[ImageCandidate] = []
    seen: set[str] = set()
    # Tavily answers with bare URL strings when descriptions are off and with
    # {"url", "description"} objects when they are on; older responses mix both.
    for entry in body.get("images") or []:
        if isinstance(entry, str):
            raw_url, description = entry, ""
        elif isinstance(entry, dict):
            raw_url, description = entry.get("url") or "", entry.get("description") or ""
        else:
            continue
        url = _normalize_url(raw_url)
        if not url or url in seen:
            continue
        seen.add(url)
        candidates.append(
            ImageCandidate(
                url=url, description=description, origin="tavily", source_url=""
            )
        )
        if len(candidates) >= limit:
            break
    return candidates


def merge_results(
    result_lists: list[list[SearchResult]], query: str
) -> list[SearchResult]:
    """One result per URL, remembering which query found the result."""
    merged: dict[str, SearchResult] = {}
    for results in result_lists:
        for result in results:
            existing = merged.get(result.url)
            if existing is None:
                result.queries = [query]
                merged[result.url] = result
                continue
            if not existing.content and result.content:
                existing.content = result.content
            if not existing.snippet and result.snippet:
                existing.snippet = result.snippet
            if not existing.images and result.images:
                existing.images = result.images
            existing.provider = f"{existing.provider}+{result.provider}"
    return list(merged.values())


async def search_web(
    query: str, *, limit: int, context: GlobalContext | None = None
) -> list[SearchResult]:
    """Search every configured provider for ``query`` and merge the results."""
    context = context or GlobalContext()
    searches: list[Any] = []
    tavily_key = getattr(context, "tavily_api_key", None)
    if tavily_key:
        searches.append(tavily_search(query, limit=limit, api_key=tavily_key))
    searches.append(browser_search(query, limit=limit, context=context))
    outcomes = await asyncio.gather(*searches, return_exceptions=True)
    result_lists: list[list[SearchResult]] = []
    for outcome in outcomes:
        if isinstance(outcome, Exception):
            logger.warning("Web search provider failed for %r: %s", query, outcome)
            continue
        result_lists.append(outcome)
    return merge_results(result_lists, query)


__all__ = [
    "ImageCandidate",
    "PageRead",
    "SearchResult",
    "browser_search",
    "extract_page_images",
    "html_to_markdown",
    "merge_results",
    "parse_duckduckgo_html",
    "read_page",
    "read_page_text",
    "search_images",
    "search_web",
    "tavily_search",
]
