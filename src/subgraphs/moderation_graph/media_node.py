"""The process-media node that moderates an upload before anything is persisted.

An upload is already a background job that nobody is waiting on, so here — unlike
the chat path — moderation is a GATE rather than a branch that runs alongside the
work. That is a correctness requirement, not a performance choice: content that
violates the terms of service must never be indexed into the avatar's store,
analyzed into its traits, or written into adapter training data, and every one of
those consumers reads the documents this node judges.

The gate is still cheap. The moderation graph screens with the free OpenAI
moderation endpoint first and only reaches the structured-output judge when that
screen is not clean; the judge itself fans out over the documents concurrently;
and separate uploads in one batch remain separate concurrent child jobs.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from src.anubis.utils.moderation.content_moderation import (
    MEDIA_MODERATION_STATE_KEY,
    moderation_flag_enabled,
)

logger = logging.getLogger(__name__)

# The consumers the gate protects. Named here so the routing function and the
# graph builder cannot drift apart.
MEDIA_MODERATION_CONSUMERS = (
    "index_docs",
    "analyze_documents",
    "process_adapter_documents",
    "psycho_analysis",
)


def _documents_under_review(state: dict) -> list[Document]:
    """Every converted document this upload would persist, de-duplicated by identity."""
    documents: list[Document] = []
    documents += list(state.get("vectorstore_documents_to_be_indexed") or [])
    documents += list(
        state.get(
            "documents_to_be_analyzed_for_context_storage_and_prompt_injection_of_assistant"
        )
        or []
    )
    documents += list(state.get("documents_to_be_processed_for_adapter_training") or [])

    seen_identities: set[int] = set()
    unique_documents: list[Document] = []
    for document in documents:
        if id(document) in seen_identities:
            continue
        seen_identities.add(id(document))
        unique_documents.append(document)
    return unique_documents


async def moderate_documents(
    state: dict, config: RunnableConfig = None, runtime: Any = None
) -> dict:
    """Judge every converted document before anything downstream reads it.

    On a violation the verdict is recorded in state and a ``media_progress`` event
    with stage ``moderation_violation`` is emitted for the job runner, which bans
    the uploader; ``route_media_moderation`` then sends the graph straight to the
    end so nothing is persisted.
    """
    context = getattr(runtime, "context", None)
    if context is not None and not moderation_flag_enabled(
        getattr(context, "content_moderation_enabled", "TRUE")
    ):
        return {}
    if (config or {}).get("configurable", {}).get("skip_content_moderation"):
        return {}

    documents = _documents_under_review(state)
    if not documents:
        return {}

    from src.subgraphs.moderation_graph.graph import (
        MODERATION_MODE_UPLOAD,
        moderate_documents_with_graph,
    )

    try:
        result = await moderate_documents_with_graph(
            documents, mode=MODERATION_MODE_UPLOAD, context=context
        )
    except Exception as moderation_error:  # noqa: BLE001 - fail open; see the graph docstring
        logger.error(
            "Upload content moderation failed (treating as clean): %s", moderation_error
        )
        return {}

    verdict = result.get("verdict") or {}
    if not verdict.get("violation"):
        return {}

    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
        writer(
            {
                "type": "media_progress",
                "stage": "moderation_violation",
                "reasoning": verdict.get("reasoning"),
                "violated_clauses": verdict.get("violated_clauses"),
                "source": verdict.get("source"),
            }
        )
    except Exception:  # noqa: BLE001 - outside a graph run there is no stream writer
        pass
    return {MEDIA_MODERATION_STATE_KEY: verdict}


def route_media_moderation(state: dict) -> list[str] | str:
    """After ``moderate_documents``: end on a violation, else the normal fan-out."""
    verdict = state.get(MEDIA_MODERATION_STATE_KEY) or {}
    if verdict.get("violation"):
        return "__end__"
    return list(MEDIA_MODERATION_CONSUMERS)


__all__ = [
    "MEDIA_MODERATION_CONSUMERS",
    "moderate_documents",
    "route_media_moderation",
]
