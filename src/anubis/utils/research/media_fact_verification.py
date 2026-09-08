"""Verify the facts in media the avatar was taught from, the way research is verified.

Uploading media used to write whatever was extracted straight into the avatar's
identity. Nothing compared a new claim against what the avatar already believed,
so a transcript saying one thing simply sat beside a stored fact saying another
and the avatar held both.

Deep research already answers exactly this question for the web — extract the
claims, cluster the ones making the same claim (including the facts the avatar
already holds), judge each cluster across its sources, apply what agrees and
hold back what contradicts for a person to settle. This module runs those same
stages over the documents a media batch just indexed, so a fact learned from an
upload is verified on the same terms as a fact found on the web, and a
contradiction reaches the owner through the same review — in the conversation
and in avatar settings — rather than being written silently.

Extraction and judging both cost model calls, so a batch verifies at most
``MEDIA_FACT_VERIFICATION_MAX_DOCUMENTS`` documents; the rest are indexed as
before and simply not fact-checked.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from src.anubis.utils.research.deep_research import (
    MEDIA_FACT_SOURCE,
    build_identity_document,
    build_proposal_document,
    cluster_facts,
    extract_facts,
    identity_namespace,
    load_existing_identity_facts,
    partition_verified_facts,
    research_proposal_namespace,
    verify_cluster,
)

logger = logging.getLogger(__name__)

EventSink = Callable[[dict[str, Any]], None]

# A media document is one "source" to the extractor; this is the pseudo scheme
# its URL carries so a verified fact can name the upload it came from, the way a
# researched fact names a web page.
MEDIA_SOURCE_URL_SCHEME = "media"


def _is_enabled(context: Any) -> bool:
    value = str(getattr(context, "media_fact_verification_enabled", "true") or "")
    return value.strip().lower() in ("1", "true", "yes", "on")


def media_documents_as_sources(documents: list[dict[str, Any]], *, limit: int) -> list:
    """Shape the indexed documents as search results the fact extractor can read.

    Each document becomes one source whose URL identifies the upload, so a fact
    verified out of a transcript can point back at the file it came from.
    """
    from src.anubis.utils.research.web_search import SearchResult

    sources: list[SearchResult] = []
    for document in documents:
        if len(sources) >= max(0, limit):
            break
        text = (document.get("page_content") or "").strip()
        if not text:
            continue
        metadata = document.get("metadata") or {}
        identifier = (
            metadata.get("document_id")
            or metadata.get("namespace_filename")
            or metadata.get("filename")
            or "unknown"
        )
        sources.append(
            SearchResult(
                url=f"{MEDIA_SOURCE_URL_SCHEME}://{identifier}",
                title=str(metadata.get("filename") or metadata.get("namespace_filename") or ""),
                content=text,
                provider="media",
            )
        )
    return sources


async def verify_facts_from_media(
    store: Any,
    context: Any,
    *,
    creator_id: str,
    assistant_id: str,
    subject_name: str,
    documents: list[dict[str, Any]],
    emit: EventSink = lambda payload: None,
    is_cancelled: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    """Extract, cross-check, and verify the facts in freshly indexed media.

    Returns a summary of what was applied and what is waiting for the owner.
    Never raises: the documents are already indexed, and an avatar that could
    not fact-check an upload is in the same state it was in before this ran.
    """
    if not _is_enabled(context):
        return {"skipped": "disabled"}

    limit = int(getattr(context, "media_fact_verification_max_documents", 0) or 0)
    sources = media_documents_as_sources(documents, limit=limit)
    if not sources:
        return {"applied": 0, "proposals": 0, "documents_read": 0}

    concurrency = int(getattr(context, "deep_research_concurrency", 4) or 4)
    emit({"type": "media_fact_verification", "stage": "extracting", "documents": len(sources)})
    media_facts = await extract_facts(
        sources, subject=subject_name, concurrency=concurrency
    )
    if is_cancelled() or not media_facts:
        return {"applied": 0, "proposals": 0, "documents_read": len(sources)}

    # The avatar's own beliefs join the clustering, which is what turns "a new
    # claim" into "a claim that disputes one you already hold".
    existing_facts = await load_existing_identity_facts(
        store, creator_id, assistant_id
    )
    emit(
        {
            "type": "media_fact_verification",
            "stage": "verifying",
            "facts": len(media_facts),
            "known_facts": len(existing_facts),
        }
    )
    clusters = await cluster_facts(media_facts + existing_facts)
    verified = list(
        await asyncio.gather(*(verify_cluster(cluster) for cluster in clusters))
    )
    if is_cancelled():
        return {"applied": 0, "proposals": 0, "documents_read": len(sources)}

    to_apply, to_review = partition_verified_facts(verified)

    applied = 0
    for entry in to_apply:
        document = build_identity_document(
            entry,
            creator_id=creator_id,
            assistant_id=assistant_id,
            subject_name=subject_name,
            source=MEDIA_FACT_SOURCE,
        )
        await store.aput(
            identity_namespace(creator_id, assistant_id),
            key=document.metadata["document_id"],
            value={"document": document.to_json()},
        )
        applied += 1

    # Contradictions land in the same place a researched contradiction lands, so
    # the owner settles both through the one review — in the conversation and in
    # avatar settings.
    proposals = 0
    for entry in to_review:
        document = build_proposal_document(
            entry,
            creator_id=creator_id,
            assistant_id=assistant_id,
            subject_name=subject_name,
        )
        await store.aput(
            research_proposal_namespace(creator_id, assistant_id),
            key=document.metadata["fact_id"],
            value={"document": document.to_json()},
        )
        proposals += 1

    emit(
        {
            "type": "media_fact_verification",
            "stage": "verified",
            "applied": applied,
            "proposals": proposals,
        }
    )
    return {
        "applied": applied,
        "proposals": proposals,
        "documents_read": len(sources),
        "facts_extracted": len(media_facts),
    }
