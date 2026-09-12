"""Public organization websites the avatar should share in conversation.

Identity facts often carry a foundation, restaurant, or recruiting-office URL
in the description or in document metadata (the page that was uploaded). The
identity prompt tells the avatar not to mention the *medium* a fact came from,
so those URLs never made it into a reply. This module lifts them into a
dedicated ROLE section framed as the avatar's own organizations to share.
"""

from __future__ import annotations

import re
from typing import Any, Iterable
from urllib.parse import urlparse

from langchain_core.documents import Document

ORGANIZATION_LINK_PATTERN = re.compile(r"https?://[^\s<>\"'`]+", re.IGNORECASE)

MEDIA_SOURCE_HOST_SUFFIXES = (
    "youtube.com",
    "youtu.be",
    "twitter.com",
    "x.com",
    "instagram.com",
    "tiktok.com",
    "facebook.com",
    "fb.com",
)

DOCUMENT_URL_METADATA_KEYS = (
    "source_url",
    "url",
    "source",
    "source_urls",
    "supporting_source_urls",
)

ORGANIZATION_LINKS_SECTION_EMPTY = ""

ORGANIZATION_LINKS_SECTION_HEADER = (
    "These are public websites of organizations you represent. When the "
    "conversation is about that organization, share the matching URL as a "
    "full https link in the reply. These URLs are yours to give; they are "
    "not a source to cite. Never invent a URL that is not listed here."
)


def _strip_trailing_punctuation(href: str) -> str:
    """Drop wrapping punctuation a sentence or markdown span often leaves on."""
    return href.rstrip(".,;:!?)]}'\"")


def _host_of(href: str) -> str:
    try:
        return (urlparse(href).hostname or "").lower()
    except ValueError:
        return ""


def is_media_source_host(host: str) -> bool:
    """Whether ``host`` is a social or video site, not an organization page."""
    hostname = (host or "").lower().removeprefix("www.")
    return any(
        hostname == suffix or hostname.endswith(f".{suffix}")
        for suffix in MEDIA_SOURCE_HOST_SUFFIXES
    )


def normalize_organization_link(url: str | None) -> str | None:
    """Return a shareable http(s) organization URL, or ``None``."""
    raw = _strip_trailing_punctuation(str(url or "").strip())
    if not raw:
        return None
    try:
        parsed = urlparse(raw)
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https"):
        return None
    if not parsed.netloc:
        return None
    if is_media_source_host(parsed.hostname or ""):
        return None
    return parsed.geturl()


def organization_links_from_text(text: str | None) -> list[str]:
    """Unique organization URLs in appearance order from ``text``."""
    found: list[str] = []
    seen: set[str] = set()
    for match in ORGANIZATION_LINK_PATTERN.finditer(str(text or "")):
        href = normalize_organization_link(match.group(0))
        if href is None or href in seen:
            continue
        seen.add(href)
        found.append(href)
    return found


def _metadata_url_values(metadata: dict[str, Any] | None) -> Iterable[str]:
    """Yield URL-shaped metadata values, including list fields."""
    if not metadata:
        return
    for key in DOCUMENT_URL_METADATA_KEYS:
        value = metadata.get(key)
        if isinstance(value, str):
            yield value
        elif isinstance(value, list):
            for entry in value:
                if isinstance(entry, str):
                    yield entry


def organization_links_from_documents(
    documents: Iterable[Document] | None,
) -> list[str]:
    """Unique organization URLs from document text and source metadata."""
    found: list[str] = []
    seen: set[str] = set()
    for document in documents or []:
        page_content = getattr(document, "page_content", "") or ""
        metadata = getattr(document, "metadata", None) or {}
        candidates = list(organization_links_from_text(page_content))
        for raw in _metadata_url_values(metadata):
            href = normalize_organization_link(raw)
            if href is not None:
                candidates.append(href)
        for href in candidates:
            if href in seen:
                continue
            seen.add(href)
            found.append(href)
    return found


def organization_links_from_identity(
    *,
    assistant_description: str | None,
    identity_documents: Iterable[Document] | None = None,
) -> list[str]:
    """Organization URLs from the avatar description and identity documents."""
    found: list[str] = []
    seen: set[str] = set()
    for href in organization_links_from_text(assistant_description):
        if href in seen:
            continue
        seen.add(href)
        found.append(href)
    for href in organization_links_from_documents(identity_documents):
        if href in seen:
            continue
        seen.add(href)
        found.append(href)
    return found


def render_organization_links_section(links: list[str]) -> str:
    """ROLE text for ``=== YOUR ORGANIZATION LINKS ===``.

    Empty when no organization website is known, so the section stays blank
    the same way YOUR PLACE does when nobody pinned the avatar.
    """
    if not links:
        return ORGANIZATION_LINKS_SECTION_EMPTY
    listed = "\n".join(links)
    return f"{ORGANIZATION_LINKS_SECTION_HEADER}\n{listed}"
