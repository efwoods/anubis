"""Process-wide handles to shared LangGraph runtime resources.

The avatar deep agent is rebuilt inside the ``think`` node on every turn, but a
human-in-the-loop ``interrupt`` raised by a deep-agent tool (e.g.
``edit_identity_fact`` / ``delete_identity_fact``) must survive across the outer
graph's interrupt → resume
passes. That requires the deep agent to be compiled with ONE persistent
checkpointer shared across ``think`` invocations — not a fresh in-memory one each
call. The FastAPI lifespan owns the ``AsyncPostgresSaver`` (built on the shared
connection pool) and publishes it here; ``think`` reads it back.

When no checkpointer has been published (e.g. ``langgraph dev`` or unit tests that
don't run the FastAPI lifespan), ``get_deep_agent_checkpointer`` returns ``None``
and ``think`` runs the deep agent without durable interrupts.
"""

import asyncio
import threading
from dataclasses import dataclass
from typing import Any, Optional

_deep_agent_checkpointer: Optional[object] = None


def set_deep_agent_checkpointer(checkpointer: object) -> None:
    """Publish the shared checkpointer the deep agent should reuse each turn."""
    global _deep_agent_checkpointer
    _deep_agent_checkpointer = checkpointer


def get_deep_agent_checkpointer() -> Optional[object]:
    """Return the shared deep-agent checkpointer, or ``None`` if unset."""
    return _deep_agent_checkpointer


# Process-wide ``SentenceTransformer`` reused across fact-correction calls. The
# fact-correction tool does sentence-level semantic matching to locate a claim buried in
# a long verbatim document, which needs to embed arbitrary sentences at runtime (the
# LangGraph store only auto-embeds whole documents at write time). Loading the model is
# expensive, so it is constructed once on first use and cached here rather than per call.
# Parallel deep-agent tool calls (e.g. several ``edit_identity_fact`` invocations in one
# turn) fan out ``asyncio.to_thread`` workers that must not share an encode forward pass.
_sentence_embedder: Optional[object] = None
_embedder_lock = threading.Lock()


def run_with_sentence_embedder(callback):
    """Run ``callback(model)`` with the shared embedder, serialized for thread safety."""
    with _embedder_lock:
        return callback(get_sentence_embedder())


@dataclass
class _PendingEmbedRequest:
    texts: list[str]
    future: asyncio.Future[Any]


_embed_batch_lock: asyncio.Lock | None = None
_embed_pending: list[_PendingEmbedRequest] = []
_embed_flush_task: asyncio.Task[None] | None = None


def _embed_batch_lock_for_loop() -> asyncio.Lock:
    global _embed_batch_lock
    if _embed_batch_lock is None:
        _embed_batch_lock = asyncio.Lock()
    return _embed_batch_lock


async def async_encode_texts(texts: list[str]) -> Any:
    """Embed ``texts`` asynchronously, coalescing concurrent callers into one batch.

    Parallel fact-tool invocations enqueue their texts on the same event-loop tick;
    a single ``encode`` forward pass embeds the deduplicated union, then each caller
    receives only its requested rows. The shared ``SentenceTransformer`` is still
    accessed under ``_embedder_lock`` — we batch to reduce forward-pass count, not
    to run the model concurrently.
    """
    if not texts:
        return []

    loop = asyncio.get_running_loop()
    future: asyncio.Future[Any] = loop.create_future()
    batch_lock = _embed_batch_lock_for_loop()

    async with batch_lock:
        _embed_pending.append(_PendingEmbedRequest(texts=texts, future=future))
        global _embed_flush_task
        if _embed_flush_task is None:
            _embed_flush_task = loop.create_task(_flush_embed_batch())

    return await future


async def _flush_embed_batch() -> None:
    """Drain the pending queue after yielding so concurrent submitters can enqueue."""
    await asyncio.sleep(0)

    batch_lock = _embed_batch_lock_for_loop()
    async with batch_lock:
        global _embed_flush_task
        batch = list(_embed_pending)
        _embed_pending.clear()
        _embed_flush_task = None

    if not batch:
        return

    text_to_unique_index: dict[str, int] = {}
    unique_texts: list[str] = []
    request_index_maps: list[tuple[list[int], asyncio.Future[Any]]] = []

    for pending in batch:
        indices_for_request: list[int] = []
        for text in pending.texts:
            if text not in text_to_unique_index:
                text_to_unique_index[text] = len(unique_texts)
                unique_texts.append(text)
            indices_for_request.append(text_to_unique_index[text])
        request_index_maps.append((indices_for_request, pending.future))

    try:

        def _encode() -> Any:
            def _run(model: Any) -> Any:
                return model.encode(unique_texts, convert_to_numpy=True)

            return run_with_sentence_embedder(_run)

        unique_embeddings = await asyncio.to_thread(_encode)

        for indices, future in request_index_maps:
            if future.done():
                continue
            future.set_result(unique_embeddings[indices])
    except Exception as exc:
        for _, future in request_index_maps:
            if not future.done():
                future.set_exception(exc)


async def async_score_query_against_texts(query: str, texts: list[str]) -> list[float]:
    """Cosine similarity of ``query`` against each entry in ``texts`` (same order)."""
    if not texts:
        return []

    embeddings = await async_encode_texts([query, *texts])

    def _similarity() -> list[float]:
        def _run(model: Any) -> list[float]:
            query_embedding = embeddings[0:1]
            text_embeddings = embeddings[1:]
            similarities = model.similarity(query_embedding, text_embeddings)[0]
            return [float(score) for score in similarities]

        return run_with_sentence_embedder(_run)

    return await asyncio.to_thread(_similarity)


def get_sentence_embedder() -> object:
    """Return the shared sentence-embedding model, loading it once on first use.

    Uses the same model as the store's vector index (``GlobalContext().embedding_model``,
    default ``microsoft/harrier-oss-v1-270m``) so sentence-level similarity is on the
    same scale as document retrieval. The ``sentence_transformers`` import is local to
    keep it off the module-import cold-start path (see CLAUDE.md import conventions).
    """
    global _sentence_embedder
    if _sentence_embedder is None:
        import torch
        from sentence_transformers import SentenceTransformer

        from src.anubis.utils.context import GlobalContext

        model_name = GlobalContext().embedding_model
        # Harrier (Gemma3) defaults to bfloat16 weights; on CPU, activations can stay
        # float32 and Gemma3 MLP layers raise "BFloat16 != float". Force float32 load.
        _sentence_embedder = SentenceTransformer(
            model_name,
            model_kwargs={"torch_dtype": torch.float32},
        )
    return _sentence_embedder
