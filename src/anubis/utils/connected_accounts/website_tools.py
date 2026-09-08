"""Tools for websites the owner connected by address: crawl, audit, traffic.

A website connection is an address, not a credential. ``crawl_website`` reads
the site's pages (same host only, bounded by ``WEBSITE_CRAWL_MAX_PAGES``) into
the deep agent's working files so the avatar can answer questions about the
content; ``website_audit`` reports on titles and descriptions, headings,
canonical and social tags, broken internal links, accessibility basics, and
what changed since the last crawl (content hashes kept in ``website_crawls``);
``website_traffic`` reads visitors from a Google Analytics or Vercel connection
that covers the same host.

Pages are fetched with a signed-in browser session when the website record
holds one (gated sites) and with plain HTTP otherwise.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections import deque
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urljoin, urlparse

from langchain.tools import tool

logger = logging.getLogger(__name__)

WEBSITE_TOOL_NAMES: tuple[str, ...] = ("crawl_website", "website_audit", "website_traffic")

WEBSITE_CRAWLS_DDL = """
CREATE TABLE IF NOT EXISTS website_crawls (
    user_id TEXT NOT NULL,
    connection_key TEXT NOT NULL,
    url TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    title TEXT,
    status_code INTEGER,
    crawled_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, connection_key, url)
);
"""

_SKIP_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".pdf", ".zip", ".mp4",
    ".mp3", ".css", ".js", ".woff", ".woff2", ".ttf", ".xml", ".json",
)


async def ensure_website_crawls_table(pool: Any) -> None:
    """Create the crawl-hash table on boot."""
    from src.anubis.utils.postgres_ddl import execute_ddl_script

    await execute_ddl_script(pool, WEBSITE_CRAWLS_DDL)


# ---------------------------------------------------------------------------
# Pure page analysis
# ---------------------------------------------------------------------------


def _first(pattern: str, html: str) -> str:
    match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
    return " ".join(match.group(1).split())[:300] if match else ""


def extract_links(html: str, base_url: str) -> list[str]:
    """Return absolute ``href`` targets found in a page."""
    links: list[str] = []
    for match in re.finditer(r"<a[^>]+href=[\"']([^\"']+)[\"']", html or "", re.IGNORECASE):
        href = match.group(1).strip().split("#", 1)[0]
        if not href or href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        links.append(urljoin(base_url, href))
    return links


def analyze_page(url: str, html: str) -> dict[str, Any]:
    """Describe one page: metadata, headings, and accessibility basics."""
    text = html or ""
    title = _first(r"<title[^>]*>(.*?)</title>", text)
    description = _first(r"<meta[^>]+name=[\"']description[\"'][^>]+content=[\"']([^\"']*)[\"']", text) or _first(
        r"<meta[^>]+content=[\"']([^\"']*)[\"'][^>]+name=[\"']description[\"']", text
    )
    canonical = _first(r"<link[^>]+rel=[\"']canonical[\"'][^>]+href=[\"']([^\"']*)[\"']", text)
    open_graph_title = _first(r"<meta[^>]+property=[\"']og:title[\"'][^>]+content=[\"']([^\"']*)[\"']", text)
    open_graph_image = _first(r"<meta[^>]+property=[\"']og:image[\"'][^>]+content=[\"']([^\"']*)[\"']", text)
    language = _first(r"<html[^>]+lang=[\"']([^\"']*)[\"']", text)
    h1_count = len(re.findall(r"<h1[\s>]", text, re.IGNORECASE))
    images = re.findall(r"<img[^>]*>", text, re.IGNORECASE)
    images_without_alt = [
        image for image in images if not re.search(r"\salt=[\"'][^\"']*[\"']", image, re.IGNORECASE)
    ]
    inputs = re.findall(r"<input[^>]*>", text, re.IGNORECASE)
    labels = len(re.findall(r"<label[\s>]", text, re.IGNORECASE))
    unlabeled_inputs = [
        entry
        for entry in inputs
        if not re.search(r"type=[\"'](hidden|submit|button)[\"']", entry, re.IGNORECASE)
        and not re.search(r"aria-label", entry, re.IGNORECASE)
    ]
    body_text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
    body_text = re.sub(r"<[^>]+>", " ", body_text)
    word_count = len(body_text.split())
    issues: list[str] = []
    if not title:
        issues.append("missing <title>")
    elif len(title) > 70:
        issues.append("title longer than 70 characters")
    if not description:
        issues.append("missing meta description")
    elif len(description) > 160:
        issues.append("meta description longer than 160 characters")
    if h1_count == 0:
        issues.append("no <h1>")
    elif h1_count > 1:
        issues.append(f"{h1_count} <h1> headings")
    if not canonical:
        issues.append("no canonical link")
    if not open_graph_title:
        issues.append("no Open Graph title")
    if not language:
        issues.append("no lang attribute on <html>")
    if images_without_alt:
        issues.append(f"{len(images_without_alt)} images without alt text")
    if unlabeled_inputs and labels < len(unlabeled_inputs):
        issues.append(f"{len(unlabeled_inputs) - labels} form inputs without a label")
    if word_count < 80:
        issues.append(f"thin content ({word_count} words)")
    return {
        "url": url,
        "title": title,
        "description": description,
        "canonical": canonical,
        "open_graph_title": open_graph_title,
        "open_graph_image": open_graph_image,
        "language": language,
        "h1_count": h1_count,
        "image_count": len(images),
        "images_without_alt": len(images_without_alt),
        "word_count": word_count,
        "issues": issues,
    }


def content_hash(html: str) -> str:
    """Hash the visible text of a page so trivial markup changes do not count."""
    body = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html or "")
    body = re.sub(r"<[^>]+>", " ", body)
    body = " ".join(body.split()).lower()
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _crawlable(url: str, hostname: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    if (parsed.hostname or "").lower() != hostname:
        return False
    return not parsed.path.lower().endswith(_SKIP_EXTENSIONS)


# ---------------------------------------------------------------------------
# Crawl history
# ---------------------------------------------------------------------------


async def _previous_hashes(pool: Any, user_id: str, connection_key: str) -> dict[str, str]:
    if pool is None:
        return {}
    try:
        async with pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT url, content_hash FROM website_crawls WHERE user_id = %s AND connection_key = %s",
                (user_id, connection_key),
            )
            rows = await cursor.fetchall()
        return {str(row[0]): str(row[1]) for row in rows}
    except Exception:
        logger.debug("Could not read previous crawl hashes", exc_info=True)
        return {}


async def _store_hashes(pool: Any, user_id: str, connection_key: str, pages: list[dict[str, Any]]) -> None:
    if pool is None or not pages:
        return
    try:
        async with pool.connection() as connection:
            for page in pages:
                await connection.execute(
                    "INSERT INTO website_crawls (user_id, connection_key, url, content_hash, title, status_code, crawled_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (user_id, connection_key, url) DO UPDATE SET content_hash = EXCLUDED.content_hash, "
                    "title = EXCLUDED.title, status_code = EXCLUDED.status_code, crawled_at = EXCLUDED.crawled_at",
                    (
                        user_id,
                        connection_key,
                        page["url"],
                        page["content_hash"],
                        page.get("title"),
                        page.get("status_code"),
                        datetime.now(UTC),
                    ),
                )
    except Exception:
        logger.debug("Could not store crawl hashes", exc_info=True)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


async def fetch_pages(
    context: Any,
    start_url: str,
    *,
    max_pages: int,
    record: dict[str, Any] | None = None,
    store: Any = None,
    http_client: Any | None = None,
) -> list[dict[str, Any]]:
    """Breadth-first fetch of a site's pages (same host), returning html per page."""
    hostname = (urlparse(start_url).hostname or "").lower()
    queue: deque[str] = deque([start_url])
    seen: set[str] = set()
    pages: list[dict[str, Any]] = []
    session_handle = None
    if record and (record.get("transport") or {}).get("browser_session"):
        try:
            from src.anubis.utils.connected_accounts.browser_sessions import (
                open_session,
            )

            session_handle = await open_session(context, store, str(record.get("user_id") or ""), record, lease=True)
        except Exception:
            session_handle = None
    owned = http_client is None and session_handle is None
    client = http_client
    if owned:
        import httpx

        client = httpx.AsyncClient(
            timeout=20.0, follow_redirects=True, headers={"User-Agent": "NeuralNexus/1.0 (+website audit)"}
        )
    try:
        while queue and len(pages) < max_pages:
            url = queue.popleft()
            normalized = url.split("#", 1)[0].rstrip("/") or url
            if normalized in seen:
                continue
            seen.add(normalized)
            try:
                if session_handle is not None:
                    async with session_handle.lock:
                        response = await session_handle.page.goto(url, wait_until="domcontentloaded", timeout=30000)
                        await asyncio.sleep(0.4)
                        html = await session_handle.page.content()
                        status_code = response.status if response else 200
                        final_url = session_handle.page.url
                else:
                    response = await client.get(url)
                    html = response.text if "text/html" in str(response.headers.get("content-type", "")) else ""
                    status_code = response.status_code
                    final_url = str(response.url)
            except Exception as fetch_error:
                pages.append({"url": url, "status_code": None, "html": "", "error": str(fetch_error)})
                continue
            pages.append({"url": final_url or url, "status_code": status_code, "html": html or ""})
            for link in extract_links(html or "", final_url or url):
                candidate = link.split("#", 1)[0]
                if _crawlable(candidate, hostname) and candidate.rstrip("/") not in seen:
                    queue.append(candidate)
    finally:
        if owned and client is not None:
            await client.aclose()
        if session_handle is not None:
            from src.anubis.utils.connected_accounts.browser_sessions import (
                release_session,
            )

            release_session(session_handle)
    return pages


async def check_links(
    links: list[str], *, http_client: Any | None = None, limit: int = 60
) -> list[dict[str, Any]]:
    """HEAD (then GET on 405) internal links; return the broken ones."""
    import httpx

    broken: list[dict[str, Any]] = []
    owned = http_client is None
    client = http_client or httpx.AsyncClient(timeout=15.0, follow_redirects=True)
    semaphore = asyncio.Semaphore(6)

    async def _check(link: str) -> None:
        async with semaphore:
            try:
                response = await client.head(link)
                if response.status_code == 405:
                    response = await client.get(link)
                if response.status_code >= 400:
                    broken.append({"url": link, "status_code": response.status_code})
            except Exception as link_error:
                broken.append({"url": link, "error": str(link_error)})

    try:
        await asyncio.gather(*(_check(link) for link in links[:limit]))
    finally:
        if owned:
            await client.aclose()
    return broken


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def build_website_tools(
    context: Any,
    accounts: list[dict[str, Any]],
    *,
    store: Any = None,
    bundle: Any = None,
    all_accounts: list[dict[str, Any]] | None = None,
) -> list[Any]:
    """Build the website tools for every connected website."""
    websites = [record for record in accounts if record.get("kind") == "website"]
    if not websites:
        return []
    max_pages_cap = int(getattr(context, "website_crawl_max_pages", None) or 50)

    def _labels() -> list[str]:
        return [str(record.get("display_label") or "") for record in websites]

    def _select(connection: str | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if connection is None or not str(connection).strip():
            if len(websites) == 1:
                return websites[0], None
            return None, {"status": "ambiguous", "error": f"Several websites are connected; name one: {_labels()}."}
        wanted = str(connection).strip().lower()
        for record in websites:
            transport = record.get("transport") or {}
            if wanted in (
                str(record.get("display_label") or "").lower(),
                str(record.get("account_address") or "").lower(),
                str(transport.get("hostname") or "").lower(),
                str(transport.get("site_url") or "").lower(),
            ):
                return record, None
        return None, {"status": "unknown_connection", "error": f"No connected website named {connection!r}. Connected: {_labels()}."}

    def _site_url(record: dict[str, Any]) -> str:
        transport = record.get("transport") or {}
        return str(transport.get("site_url") or f"https://{record.get('account_address')}/")

    async def _write_workspace(record: dict[str, Any], pages: list[dict[str, Any]]) -> list[str]:
        if bundle is None:
            return []
        from src.anubis.utils.connected_accounts.browser_session_tools import (
            html_to_text,
        )

        written: list[str] = []
        root = getattr(bundle, "workspace_root", None) or getattr(bundle, "workspace_directory", None)
        if root is None:
            return []
        from pathlib import Path

        slug = re.sub(r"[^a-z0-9]+", "-", str(record.get("account_address") or "site").lower()).strip("-")
        base = Path(str(root)) / "site" / slug
        base.mkdir(parents=True, exist_ok=True)
        for page in pages:
            if not page.get("html"):
                continue
            path = urlparse(page["url"]).path.strip("/") or "index"
            name = re.sub(r"[^a-z0-9/_-]+", "-", path.lower()).replace("/", "__")[:120] + ".md"
            target = base / name
            target.write_text(f"# {page['url']}\n\n" + html_to_text(page["html"], limit=60000), encoding="utf-8")
            written.append(str(target.relative_to(Path(str(root)))))
        return written

    @tool
    async def crawl_website(connection: str | None = None, max_pages: int = 25) -> dict[str, Any]:
        """Read a connected website's pages (same host) into the working files.

        Use before answering questions about what a site says, or before an
        audit. ``connection`` is the website's label (optional when one site is
        connected). Returns the pages read (address, title, status) and the
        working-file paths written.
        """
        record, error = _select(connection)
        if error:
            return error
        limit = max(1, min(int(max_pages or 25), max_pages_cap))
        pages = await fetch_pages(context, _site_url(record), max_pages=limit, record=record, store=store)
        written = await _write_workspace(record, pages)
        summaries = [
            {"url": page["url"], "status_code": page.get("status_code"), "title": analyze_page(page["url"], page.get("html", "")).get("title"), "error": page.get("error")}
            for page in pages
        ]
        return {"status": "ok", "connection": record.get("display_label"), "page_count": len(pages), "pages": summaries, "files": written}

    @tool
    async def website_audit(connection: str | None = None, max_pages: int = 25) -> dict[str, Any]:
        """Audit a connected website: content, search tags, links, accessibility, changes.

        Use for "how is my site doing", "audit my site", "what is wrong with my
        site", or "what changed on my site". Reports per-page issues ordered
        by impact, broken internal links, and the pages whose content changed
        since the last audit. Chart counts with make_chart and save the result
        with save_report (kind "website").
        """
        record, error = _select(connection)
        if error:
            return error
        limit = max(1, min(int(max_pages or 25), max_pages_cap))
        site_url = _site_url(record)
        pages = await fetch_pages(context, site_url, max_pages=limit, record=record, store=store)
        from src.anubis.utils.runtime_handles import get_postgres_pool

        pool = get_postgres_pool()
        user_id = str(record.get("user_id") or "")
        connection_key = str(record.get("account_key") or "")
        previous = await _previous_hashes(pool, user_id, connection_key)
        analyses: list[dict[str, Any]] = []
        hashed: list[dict[str, Any]] = []
        changed: list[str] = []
        new_pages: list[str] = []
        internal_links: set[str] = set()
        hostname = (urlparse(site_url).hostname or "").lower()
        for page in pages:
            html = page.get("html") or ""
            if not html:
                analyses.append({"url": page["url"], "status_code": page.get("status_code"), "issues": [page.get("error") or f"answered {page.get('status_code')}"]})
                continue
            analysis = analyze_page(page["url"], html)
            analysis["status_code"] = page.get("status_code")
            analyses.append(analysis)
            digest = content_hash(html)
            hashed.append({"url": page["url"], "content_hash": digest, "title": analysis.get("title"), "status_code": page.get("status_code")})
            if page["url"] not in previous:
                new_pages.append(page["url"])
            elif previous[page["url"]] != digest:
                changed.append(page["url"])
            for link in extract_links(html, page["url"]):
                if (urlparse(link).hostname or "").lower() == hostname:
                    internal_links.add(link.split("#", 1)[0])
        crawled_urls = {page["url"] for page in pages}
        broken = await check_links([link for link in internal_links if link not in crawled_urls])
        await _store_hashes(pool, user_id, connection_key, hashed)
        issue_counts: dict[str, int] = {}
        for analysis in analyses:
            for issue in analysis.get("issues") or []:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        ranked = sorted(issue_counts.items(), key=lambda entry: entry[1], reverse=True)
        return {
            "status": "ok",
            "connection": record.get("display_label"),
            "site_url": site_url,
            "audited_at": datetime.now(UTC).isoformat(),
            "page_count": len(pages),
            "issue_summary": [{"issue": issue, "pages": count} for issue, count in ranked],
            "pages": analyses,
            "broken_links": broken,
            "changed_since_last_audit": changed,
            "new_since_last_audit": new_pages,
            "first_audit": not previous,
        }

    @tool
    async def website_traffic(connection: str | None = None, since: str | None = None, until: str | None = None) -> dict[str, Any]:
        """Report visitors, sessions, and top pages of a connected website.

        Reads a Google Analytics or Vercel connection bound to the same owner;
        say plainly (and offer connect_account with provider google_analytics)
        when none is connected. Dates are ISO strings; default the last 30 days.
        """
        record, error = _select(connection)
        if error:
            return error
        others = [entry for entry in (all_accounts or []) if entry.get("provider") in ("google_analytics", "vercel")]
        if not others:
            return {
                "status": "no_traffic_source",
                "connection": record.get("display_label"),
                "message": "No Google Analytics or Vercel account is connected. Offer to connect one (connect_account with provider google_analytics or vercel).",
            }
        try:
            from src.anubis.utils.connected_accounts.vendor_api_tools import (
                traffic_report_for_site,
            )
        except ImportError:
            return {"status": "unavailable", "message": "The traffic client is not installed."}
        return await traffic_report_for_site(context, store, others, _site_url(record), since=since, until=until)

    return [crawl_website, website_audit, website_traffic]
