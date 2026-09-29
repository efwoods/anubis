"""Embed store search queries BEFORE a store batch opens a Postgres pipeline.

``langgraph.store.postgres.AsyncPostgresStore._execute_batch`` opens one
pipelined cursor for a whole batch, sends the batch's get queries (the
``UPDATE store … SET expires_at`` TTL refresh), and only then awaits
``self.embeddings.aembed_documents`` for the batch's search queries — with the
pipeline still open and no ``Sync`` sent. For the whole embedding wait the
pooled connection stays checked out and the Postgres backend sits ``active`` /
``ClientRead``, holding row locks on the refreshed rows.

On 2026-09-28 prod held one such connection from 14:12:38.286 to 14:22:28 UTC:
every avatar turn stalled after the ``chat`` node, 27 ``GET
/avatar_reference_image`` requests waited 177,906–178,182 ms, and all released
in the same second. ``statement_timeout`` cannot end that state: a statement
waiting for ``Sync`` is waiting on the client, not running (verified against
Postgres on the same server).

``install_store_embedding_before_pipeline`` patches the store class so the
search embeddings are computed first, with a time limit, and the pipeline opens
only once every vector is ready. The embedding wait then holds no connection.
Both stores in the ``langgraph-api`` process — the platform store the graph
receives and ``app.state.store`` in ``src/api/webapp.py`` — are instances of the
same class, so one patch covers both.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from typing import Any, Sequence, cast

logger = logging.getLogger(__name__)

# Default time limit, in seconds, for embedding the search queries of one store
# batch. Env ``STORE_SEARCH_EMBEDDING_TIMEOUT_SECONDS``.
DEFAULT_STORE_SEARCH_EMBEDDING_TIMEOUT_SECONDS = 30.0

# (id of the store instance, prepared search queries with vectors filled in),
# set by the patched ``_execute_batch`` for the patched ``_batch_search_ops`` of
# the same batch in the same task.
_prepared_search_queries: contextvars.ContextVar[tuple[int, list] | None] = (
    contextvars.ContextVar("prepared_store_search_queries", default=None)
)

_installed_embedding_timeout_seconds: float | None = None


class StoreSearchEmbeddingTimeout(TimeoutError):
    """Embedding the search queries of one store batch exceeded the time limit."""


def store_search_embedding_timeout_seconds(configured_seconds: Any) -> float:
    """Read the configured time limit; a missing or non-positive value uses the default."""
    try:
        parsed_seconds = float(configured_seconds)
    except (TypeError, ValueError):
        return DEFAULT_STORE_SEARCH_EMBEDDING_TIMEOUT_SECONDS
    if parsed_seconds <= 0:
        return DEFAULT_STORE_SEARCH_EMBEDDING_TIMEOUT_SECONDS
    return parsed_seconds


def install_store_embedding_before_pipeline(
    embedding_timeout_seconds: float | None = None,
) -> bool:
    """Patch ``AsyncPostgresStore`` once per process. Returns True when patched now.

    A second call only updates the time limit.
    """
    global _installed_embedding_timeout_seconds
    from langgraph.store.base import SearchOp
    from langgraph.store.postgres import aio as postgres_store_module

    store_class = postgres_store_module.AsyncPostgresStore
    timeout_seconds = store_search_embedding_timeout_seconds(embedding_timeout_seconds)
    already_installed = getattr(store_class, "_neural_nexus_embedding_guard", False)
    _installed_embedding_timeout_seconds = timeout_seconds
    if already_installed:
        return False

    original_execute_batch = store_class._execute_batch
    original_batch_search_ops = store_class._batch_search_ops
    row_to_search_item = postgres_store_module._row_to_search_item
    decode_namespace_bytes = postgres_store_module._decode_ns_bytes

    async def _execute_batch_with_embeddings_first(
        self: Any, grouped_ops: dict, results: list, conn: Any = None
    ) -> None:
        search_operations = grouped_ops.get(SearchOp)
        if not search_operations or not getattr(self, "embeddings", None):
            await original_execute_batch(self, grouped_ops, results, conn)
            return
        prepared_queries, embedding_requests = self._prepare_batch_search_queries(
            search_operations
        )
        if embedding_requests:
            limit_seconds = (
                _installed_embedding_timeout_seconds
                or DEFAULT_STORE_SEARCH_EMBEDDING_TIMEOUT_SECONDS
            )
            try:
                vectors = await asyncio.wait_for(
                    self.embeddings.aembed_documents(
                        [search_query for _, search_query in embedding_requests]
                    ),
                    timeout=limit_seconds,
                )
            except TimeoutError as embedding_timeout:
                logger.error(
                    "Store search embedding exceeded %s s for %s queries; "
                    "the batch was refused before any connection was taken",
                    limit_seconds,
                    len(embedding_requests),
                )
                raise StoreSearchEmbeddingTimeout(
                    f"store search embedding exceeded {limit_seconds} s"
                ) from embedding_timeout
            from langgraph.store.postgres.base import PLACEHOLDER

            for (query_index, _), vector in zip(
                embedding_requests, vectors, strict=False
            ):
                query_parameters = prepared_queries[query_index][1]
                for parameter_index in range(len(query_parameters)):
                    if query_parameters[parameter_index] is PLACEHOLDER:
                        query_parameters[parameter_index] = vector
        context_token = _prepared_search_queries.set((id(self), prepared_queries))
        try:
            await original_execute_batch(self, grouped_ops, results, conn)
        finally:
            _prepared_search_queries.reset(context_token)

    async def _batch_search_ops_with_prepared_queries(
        self: Any,
        search_operations: Sequence[tuple[int, Any]],
        results: list,
        cursor: Any,
    ) -> None:
        prepared_entry = _prepared_search_queries.get()
        if prepared_entry is None or prepared_entry[0] != id(self):
            await original_batch_search_ops(self, search_operations, results, cursor)
            return
        prepared_queries = prepared_entry[1]
        for (result_index, _), (query, query_parameters) in zip(
            search_operations, prepared_queries, strict=False
        ):
            await cursor.execute(query, query_parameters)
            rows = cast(list, await cursor.fetchall())
            results[result_index] = [
                row_to_search_item(
                    decode_namespace_bytes(row["prefix"]),
                    row,
                    loader=self._deserializer,
                )
                for row in rows
            ]

    store_class._execute_batch = _execute_batch_with_embeddings_first
    store_class._batch_search_ops = _batch_search_ops_with_prepared_queries
    store_class._neural_nexus_embedding_guard = True
    logger.info(
        "Store search embeddings now run before the pipeline opens (limit %s s)",
        timeout_seconds,
    )
    return True
