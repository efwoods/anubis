"""Build the per-avatar knowledge profile (atomic-fact index) once, store it, never recompute.

The knowledge profile is a list of atomic facts the avatar is allowed to
draw on. We populate it from two sources:

* ``identity`` namespace Documents — first-person statements written by
  :class:`FirstPersonRewriterClass`. Each Document already corresponds to
  one atomic fact (one fact per identity Document is the design intent), so
  we copy them in directly.

* ``quote`` namespace Documents — verbatim direct quotes / monologue
  chunks. To keep this profile atomic, we also store the raw text under a
  ``quote`` source label so the knowledge evaluator can decide a candidate
  claim is supported either by a stated identity fact or by something the
  avatar literally said.

The profile is stored at namespace ``(creator_id, assistant_id,
"knowledge_profile")`` and individual fact rows are also written under
``(creator_id, assistant_id, "knowledge_profile_index")`` so the LangGraph
store's vector backend can serve bounded ``asearch`` queries during
evaluation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langgraph.store.base import BaseStore

from src.anubis.utils.context import GlobalContext

logger = logging.getLogger(__name__)


KNOWLEDGE_PROFILE_NAMESPACE_TAG = "knowledge_profile"
KNOWLEDGE_PROFILE_INDEX_TAG = "knowledge_profile_index"
KNOWLEDGE_PROFILE_KEY = "profile"


async def _enumerate_namespace(
    creator_id: str, assistant_id: str, namespace_tag: str, store: BaseStore
) -> List[Dict[str, Any]]:
    namespace = (creator_id, assistant_id, namespace_tag)
    try:
        items = await store.asearch(namespace, query="*", limit=10000)
    except Exception as exc:
        logger.warning(
            "asearch over %s failed (%s); skipping", namespace_tag, exc
        )
        return []
    out: List[Dict[str, Any]] = []
    for item in items or []:
        value = getattr(item, "value", {}) or {}
        page_content = value.get("page_content")
        metadata = value.get("metadata") or {}
        if not page_content and isinstance(value.get("document"), dict):
            doc = value["document"]
            page_content = doc.get("page_content")
            metadata = doc.get("metadata") or metadata
        if isinstance(page_content, str) and page_content.strip():
            out.append(
                {
                    "page_content": page_content.strip(),
                    "metadata": metadata,
                    "store_key": getattr(item, "key", None),
                }
            )
    return out


async def maybe_build_knowledge_profile(
    *,
    creator_id: str,
    assistant_id: str,
    store: BaseStore,
    context: GlobalContext,
) -> Optional[Dict[str, Any]]:
    """Build / refresh the atomic-fact index if thresholds are met."""
    if not creator_id or not assistant_id:
        return None

    identity_items = await _enumerate_namespace(
        creator_id, assistant_id, "identity", store
    )
    quote_items = await _enumerate_namespace(
        creator_id, assistant_id, "quote", store
    )

    if len(identity_items) < int(context.min_identity_docs_for_knowledge_profile or 0):
        logger.info(
            "Knowledge profile build skipped: %d identity docs < threshold %s",
            len(identity_items),
            context.min_identity_docs_for_knowledge_profile,
        )
        return None

    profile_namespace = (
        creator_id,
        assistant_id,
        KNOWLEDGE_PROFILE_NAMESPACE_TAG,
    )
    index_namespace = (
        creator_id,
        assistant_id,
        KNOWLEDGE_PROFILE_INDEX_TAG,
    )

    existing_item = None
    try:
        existing_item = await store.aget(profile_namespace, KNOWLEDGE_PROFILE_KEY)
    except Exception:
        existing_item = None
    existing = (
        getattr(existing_item, "value", None) if existing_item is not None else None
    )

    if existing is not None:
        last_id = int(existing.get("identity_document_count", 0) or 0)
        last_q = int(existing.get("quote_document_count", 0) or 0)
        if (
            len(identity_items) - last_id
            < int(context.profile_refresh_threshold or 0)
            and len(quote_items) - last_q
            < int(context.profile_refresh_threshold or 0)
        ):
            logger.info(
                "Knowledge profile refresh skipped: insufficient new docs since last build."
            )
            return existing

    facts: List[Dict[str, Any]] = []

    for item in identity_items:
        meta = item["metadata"] or {}
        facts.append(
            {
                "fact": item["page_content"],
                "kind": "identity",
                "synthetic": bool(meta.get("synthetic", True)),
                "original_statement": meta.get("original_statement"),
                "extracted_fact": meta.get("extracted_fact"),
                "rewritten_statement": meta.get("rewritten_statement"),
                "first_person_statement": meta.get("first_person_statement")
                or item["page_content"],
                "source": meta.get("original_source") or meta.get("source"),
                "filename": meta.get("filename"),
                "store_key": item.get("store_key"),
            }
        )

    for item in quote_items:
        meta = item["metadata"] or {}
        facts.append(
            {
                "fact": item["page_content"],
                "kind": "quote",
                "synthetic": False,
                "speaker": meta.get("speaker"),
                "source": meta.get("original_source") or meta.get("source"),
                "filename": meta.get("filename"),
                "store_key": item.get("store_key"),
            }
        )

    profile = {
        "version": 1,
        "built_at": datetime.now(tz=timezone.utc).isoformat(),
        "identity_document_count": len(identity_items),
        "quote_document_count": len(quote_items),
        "facts": facts,
    }
    await store.aput(
        profile_namespace, key=KNOWLEDGE_PROFILE_KEY, value=profile
    )

    # Mirror each fact into a vector-searchable index so the knowledge
    # evaluator can do bounded retrieval without scanning the whole list.
    for idx, fact in enumerate(facts):
        try:
            await store.aput(
                index_namespace,
                key=f"{fact['kind']}:{idx}",
                value={
                    "page_content": fact["fact"],
                    "metadata": {k: v for k, v in fact.items() if k != "fact"},
                },
            )
        except Exception as exc:  # pragma: no cover - operator log only
            logger.debug(
                "Failed to mirror fact into knowledge index (continuing): %s",
                exc,
            )

    logger.info(
        "Knowledge profile written for %s/%s (n_identity=%d n_quote=%d)",
        creator_id,
        assistant_id,
        len(identity_items),
        len(quote_items),
    )
    return profile


async def load_knowledge_profile(
    *, creator_id: str, assistant_id: str, store: BaseStore
) -> Optional[Dict[str, Any]]:
    """Read-only fetch used by the knowledge evaluator."""
    profile_namespace = (
        creator_id,
        assistant_id,
        KNOWLEDGE_PROFILE_NAMESPACE_TAG,
    )
    try:
        item = await store.aget(profile_namespace, KNOWLEDGE_PROFILE_KEY)
    except Exception:
        return None
    if item is None:
        return None
    value = getattr(item, "value", None)
    return value if isinstance(value, dict) else None
