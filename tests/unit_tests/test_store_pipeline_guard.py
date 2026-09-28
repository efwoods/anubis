"""Store search embeddings finish before a store batch takes a connection.

Regression for the 2026-09-28 prod outage: ``AsyncPostgresStore`` awaited the
search embedding inside an open pipeline, so the pooled connection sat
``active`` / ``ClientRead`` for the whole embedding wait (9 minutes 50 seconds
on prod) and every avatar turn stalled behind the held connection.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest
from langchain_core.embeddings import Embeddings
from langgraph.store.base import IndexConfig, SearchOp
from langgraph.store.postgres import AsyncPostgresStore

from src.anubis.utils import store_pipeline_guard
from src.anubis.utils.store_pipeline_guard import (
    DEFAULT_STORE_SEARCH_EMBEDDING_TIMEOUT_SECONDS,
    StoreSearchEmbeddingTimeout,
    install_store_embedding_before_pipeline,
    store_search_embedding_timeout_seconds,
)


class _RecordingEmbeddings(Embeddings):
    def __init__(self, event_log: list[str], delay_seconds: float = 0.0):
        self.event_log = event_log
        self.delay_seconds = delay_seconds

    def embed_documents(self, texts):
        return [[0.5, 0.5] for _ in texts]

    def embed_query(self, text):
        return [0.5, 0.5]

    async def aembed_documents(self, texts):
        self.event_log.append("embedding_started")
        await asyncio.sleep(self.delay_seconds)
        self.event_log.append("embedding_finished")
        return [[0.5, 0.5] for _ in texts]


class _RecordingCursor:
    def __init__(self, event_log: list[str]):
        self.event_log = event_log
        self.executed_parameters: list = []

    async def execute(self, query, parameters):
        self.event_log.append("query_sent")
        self.executed_parameters.append(parameters)

    async def fetchall(self):
        return []


class _RecordingStore(AsyncPostgresStore):
    """The real store class with the connection replaced by an event log."""

    def __init__(self, event_log: list[str], embeddings: Embeddings):
        self.event_log = event_log
        self.recording_cursor = _RecordingCursor(event_log)
        super().__init__(
            object(),  # never used: ``_cursor`` below replaces every connection use
            index=IndexConfig(dims=2, embed=embeddings, fields=["text"]),
        )

    @asynccontextmanager
    async def _cursor(self, *, pipeline: bool = False):
        self.event_log.append("connection_taken")
        yield self.recording_cursor
        self.event_log.append("connection_returned")


@pytest.fixture(autouse=True)
def _guard_installed():
    install_store_embedding_before_pipeline(DEFAULT_STORE_SEARCH_EMBEDDING_TIMEOUT_SECONDS)


@pytest.mark.asyncio
async def test_embedding_finishes_before_the_connection_is_taken():
    event_log: list[str] = []
    store = _RecordingStore(event_log, _RecordingEmbeddings(event_log))
    results = await store.abatch([SearchOp(("user", "avatar"), query="launch", limit=2)])
    assert event_log == [
        "embedding_started",
        "embedding_finished",
        "connection_taken",
        "query_sent",
        "connection_returned",
    ]
    assert results == [[]]
    # The vector reached the query instead of the placeholder.
    sent_parameters = store.recording_cursor.executed_parameters[0]
    assert [0.5, 0.5] in list(sent_parameters)


@pytest.mark.asyncio
async def test_slow_embedding_fails_without_taking_a_connection():
    event_log: list[str] = []
    install_store_embedding_before_pipeline(0.05)
    try:
        store = _RecordingStore(event_log, _RecordingEmbeddings(event_log, 1.0))
        with pytest.raises(StoreSearchEmbeddingTimeout):
            await store.abatch([SearchOp(("user", "avatar"), query="launch", limit=2)])
    finally:
        install_store_embedding_before_pipeline(
            DEFAULT_STORE_SEARCH_EMBEDDING_TIMEOUT_SECONDS
        )
    assert "connection_taken" not in event_log


def test_install_patches_the_class_once():
    assert AsyncPostgresStore._neural_nexus_embedding_guard is True
    assert install_store_embedding_before_pipeline(12) is False
    assert store_pipeline_guard._installed_embedding_timeout_seconds == 12.0
    install_store_embedding_before_pipeline(DEFAULT_STORE_SEARCH_EMBEDDING_TIMEOUT_SECONDS)


@pytest.mark.parametrize(
    ("configured_seconds", "expected_seconds"),
    [(None, 30.0), ("", 30.0), ("abc", 30.0), (0, 30.0), (-4, 30.0), ("45", 45.0), (7.5, 7.5)],
)
def test_timeout_setting_is_parsed_strictly(configured_seconds, expected_seconds):
    assert store_search_embedding_timeout_seconds(configured_seconds) == expected_seconds
