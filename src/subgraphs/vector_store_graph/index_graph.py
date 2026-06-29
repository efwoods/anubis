"""This "graph" simply exposes an endpoint for a user to upload docs to be indexed."""

from typing import Any, Sequence, cast

from langchain_core.documents import Document
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.state import GlobalState

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

across_thread_memory = InMemoryStore()


import logging
logger = logging.getLogger(__name__)

from src.subgraphs.vector_store_graph.utils.helper_functions import batch_index_documents_vectorstore
from langchain_core.runnables import RunnableConfig
from src.anubis.utils.utility import extract_user_id_assistant_id, remove_docs_update


def ensure_docs_have_user_id(
    vectorstore_documents_to_be_indexed: Sequence[Document], runtime: GlobalContext
) -> list[Document]:
    """Ensure that all documents have a user_id in their metadata.

        vectorstore_documents_to_be_indexed (Sequence[Document]): A sequence of Document objects to process.
        runtime (GlobalContext): A context object containing the user_id.

    Returns:
        list[Document]: A new list of Document objects with updated metadata.
    """
    
    if isinstance(runtime.context.user_ctx, dict):
        user_id = runtime.context.user_ctx.get("user_id", "")
    else:
        user_id = getattr(runtime.context.user_ctx, "user_id", "")

    return [
        Document(
            page_content=doc.page_content, metadata={**doc.metadata, "user_id": user_id}
        )
        for doc in vectorstore_documents_to_be_indexed
    ]


def _file_identity(doc: Document) -> str:
    """Original-file key for a Document (its uploaded filename)."""
    meta = doc.metadata or {}
    return meta.get("filename") or meta.get("namespace_filename") or "unknown"


async def index_docs(
    state: GlobalState, runtime: Runtime[GlobalContext], config: RunnableConfig,
    store: BaseStore
) -> dict[str, Any]:
    """Index the queued documents, tolerating per-file failures.

    Indexes everything it can. A document/batch that fails does NOT stop the
    pipeline: the original files (by filename) whose documents failed are
    collected into ``failed_to_index_files`` so the upload layer can surface and
    reprocess them, while every other file is indexed normally.

    The processed snapshot is removed from the buffer via a targeted removal
    (not a blind ``"delete"``), so documents appended by a concurrent node in
    the same superstep are preserved for the next pass.
    """
    logger.info(f"INDEXING DOCUMENTS")

    updated_user_id, updated_assistant_id = await extract_user_id_assistant_id(config)

    user_id = updated_user_id['user_id']
    assistant_id = updated_assistant_id['assistant_id']

    # region agent log
    try:
        from src.anubis.utils.utility import _agent_debug_log as _adl
        _raw = state.get('vectorstore_documents_to_be_indexed')
        _kind = type(_raw).__name__
        _len = len(_raw) if hasattr(_raw, "__len__") else None
        _adl(
            "index_docs:entry:vectorstore_documents_to_be_indexed",
            {"kind": _kind, "len": _len},
            hypothesis_id="H1",
        )
    except Exception:
        pass
    # endregion

    # The full snapshot we pulled from the buffer. We must remove THIS exact set
    # from the buffer at the end (even fragments we choose not to index), or
    # dropped junk would linger and be reprocessed every superstep.
    attempted_docs: list[Document] = list(
        cast(Sequence[Document], state.get('vectorstore_documents_to_be_indexed') or [])
    )

    # Final fragment safety net: drop any document whose page_content is clear
    # boilerplate (page numbers, headers/footers, nav) that slipped past the
    # per-loader filters. Heuristic-only (no LLM) so the index path stays cheap;
    # indeterminant short lines (genuine one-line quotes) are kept.
    docs: list[Document] = attempted_docs
    if attempted_docs:
        try:
            from src.subgraphs.process_media_graph.utils.fragment_filter import (
                is_useful_content,
            )

            kept_docs = [
                d for d in attempted_docs if is_useful_content(d.page_content or "")
            ]
            if len(kept_docs) != len(attempted_docs):
                logger.info(
                    "index_docs fragment net dropped %d/%d documents",
                    len(attempted_docs) - len(kept_docs),
                    len(attempted_docs),
                )
            docs = kept_docs
        except Exception as exc:  # pragma: no cover - never block indexing
            logger.warning("index_docs fragment net failed (%s); indexing all", exc)

    if not attempted_docs:
        # Nothing in our snapshot to index. Do NOT clear the buffer: another
        # node (e.g. analyze_documents) may have appended docs in this same
        # superstep, and a blind clear would discard them un-indexed.
        logger.info("No documents to index; skipping batch indexing")
        return {}

    if not docs:
        # Every attempted document was a fragment. Nothing to index, but still
        # remove the attempted snapshot so the fragments don't linger and get
        # reprocessed every superstep.
        logger.info("All attempted documents were fragments; removing snapshot")
        return {"vectorstore_documents_to_be_indexed": remove_docs_update(attempted_docs)}

    # Files whose documents failed to index, keyed by original filename.
    failed_file_keys: set[str] = set()
    error_detail: str | None = None

    try:
        result = await batch_index_documents_vectorstore(
            store, user_id, assistant_id, docs, BATCH_SIZE=1000
        )
        if not result.get("success", False):
            # batch_index_documents_vectorstore already kept going past failed
            # batches; it returns the failed PutOps (keyed by document_id).
            error_ops = result.get("error_batch_documents", []) or []
            failed_keys = {getattr(op, "key", None) for op in error_ops}
            for doc in docs:
                if doc.metadata.get("document_id") in failed_keys:
                    failed_file_keys.add(_file_identity(doc))
            error_detail = "batch indexing error"
            logger.error(
                "Indexing failed for %d file(s): %s",
                len(failed_file_keys),
                sorted(failed_file_keys),
            )
    except Exception as exc:
        # Hard failure indexing this snapshot: do not stop the pipeline. Mark
        # every attempted file as failed so the uploader can reprocess them.
        error_detail = str(exc)
        failed_file_keys = {_file_identity(doc) for doc in docs}
        logger.error("Indexing raised; marking all attempted files failed: %s", exc)

    # Build a per-file failure report (one entry per original filename) so the
    # original upload can be made aware and reprocess only the failed files.
    file_failures: dict[str, dict[str, Any]] = {}
    if failed_file_keys:
        for doc in docs:
            fkey = _file_identity(doc)
            if fkey not in failed_file_keys:
                continue
            entry = file_failures.setdefault(
                fkey,
                {
                    "filename": (doc.metadata or {}).get("filename"),
                    "namespace_filename": (doc.metadata or {}).get("namespace_filename"),
                    "document_ids": [],
                    "error": error_detail or "indexing error",
                },
            )
            entry["document_ids"].append((doc.metadata or {}).get("document_id"))

    logger.info(f"breakpoint after batch_index_documents_vectorstore")

    # Remove the entire attempted snapshot from the buffer (targeted, so a
    # concurrent append in the same superstep survives). Successful files are
    # persisted; failed files are surfaced via failed_to_index_files for
    # reprocessing rather than being silently retried here.
    return {
        "vectorstore_documents_to_be_indexed": remove_docs_update(attempted_docs),
        "failed_to_index_files": list(file_failures.values()),
    }

# Define a new graph
builder = StateGraph(GlobalState, context_schema=GlobalContext)

builder.add_node(index_docs)
builder.add_edge("__start__", "index_docs")

index_graph = builder.compile()

index_graph.name = "IndexGraph"

__all__ = ["index_graph"]
