"""Document statefulness: persisted state docs merged with fresh retrievals.

``merge_dedup_threshold_documents`` is the mechanism behind the stateful
``load_consciousness`` rework: documents persist in graph state across turns and each
turn are unioned with the fresh store retrieval, de-duplicated by stable id (the fresh
copy wins so in-place edits are reflected), and — for episodic memory only —
salience-thresholded against the current query so stale low-salience memories fall out
of state. Identity channels are merged/deduped but never pruned.

The store-index embedder is faked by seeding ``runtime_handles._sentence_embedder`` with
a deterministic bag-of-words model, so no model download is needed.
"""

import math
import re

import pytest
from langchain_core.documents import Document
from langgraph.store.base import SearchItem

import src.anubis.utils.runtime_handles as runtime_handles
from src.anubis.utils.utility import merge_dedup_threshold_documents


class _FakeEmbedder:
    """Deterministic set-cosine embedder exposing the SentenceTransformer surface the
    merge helper uses (``encode`` + ``similarity``)."""

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z']+", (text or "").lower()))

    def encode(self, texts, convert_to_numpy=True):
        return [self._tokens(t) for t in texts]

    def similarity(self, query_embeddings, doc_embeddings):
        query_tokens = query_embeddings[0]
        row = []
        for doc_tokens in doc_embeddings:
            if not query_tokens or not doc_tokens:
                row.append(0.0)
                continue
            shared = len(query_tokens & doc_tokens)
            row.append(
                shared / (math.sqrt(len(query_tokens)) * math.sqrt(len(doc_tokens)))
            )
        return [row]


@pytest.fixture(autouse=True)
def _fake_sentence_embedder(monkeypatch):
    monkeypatch.setattr(runtime_handles, "_sentence_embedder", _FakeEmbedder())


def _doc(doc_id: str, content: str) -> Document:
    return Document(page_content=content, metadata={"document_id": doc_id})


def _search_item(doc: Document, score: float | None) -> SearchItem:
    """A real store SearchItem (``_coerce_to_documents`` dispatches on isinstance)."""
    return SearchItem(
        namespace=("ns",),
        key=doc.metadata["document_id"],
        value={"document": doc.to_json()},
        created_at=None,
        updated_at=None,
        score=score,
    )


@pytest.mark.asyncio
async def test_identity_merge_persists_prior_docs_unfiltered():
    """Identity channels: prior state docs survive even when totally unrelated to the
    current query (the avatar never forgets its identity), and fresh docs are unioned."""
    prior = [_doc("1", "I am a carpenter."), _doc("2", "I grew up in Ottawa.")]
    fresh = [_search_item(_doc("3", "I love sailing boats."), 0.9)]

    merged = await merge_dedup_threshold_documents(
        prior, fresh, "tell me about sailing", apply_threshold=False
    )
    assert [d.metadata["document_id"] for d in merged] == ["1", "2", "3"]


@pytest.mark.asyncio
async def test_fresh_retrieval_wins_on_id_collision():
    """An in-place store edit is reflected: the freshly retrieved copy replaces the stale
    state copy that shares its document id."""
    prior = [_doc("1", "I am 5' 6\".")]
    fresh = [_search_item(_doc("1", "I am 6' 1\"."), 0.95)]

    merged = await merge_dedup_threshold_documents(
        prior, fresh, "how tall are you", apply_threshold=False
    )
    assert len(merged) == 1
    assert merged[0].page_content == "I am 6' 1\"."


@pytest.mark.asyncio
async def test_memory_threshold_prunes_stale_low_salience_docs():
    """Episodic memory: a prior-state memory unrelated to the current query is re-scored
    and pruned; a prior memory restating the query survives; fresh items keep their store
    score."""
    salient_prior = _doc("1", "We talked about the hockey game last night.")
    stale_prior = _doc("2", "The invoice for the plumbing was paid in April.")
    fresh = [_search_item(_doc("3", "You said the hockey game went to overtime."), 0.8)]

    merged = await merge_dedup_threshold_documents(
        [salient_prior, stale_prior],
        fresh,
        "what did we say about the hockey game last night",
        apply_threshold=True,
        threshold=0.5,
    )
    ids = [d.metadata["document_id"] for d in merged]
    assert "1" in ids  # salient prior memory persists
    assert "2" not in ids  # stale low-salience memory is pruned
    assert "3" in ids  # fresh retrieval kept via its store score


@pytest.mark.asyncio
async def test_deleted_doc_stays_gone_when_absent_from_state_and_retrieval():
    """After the edit/delete tools prune a document from state, the merge cannot
    resurrect it — it is in neither the prior docs nor the fresh retrieval."""
    prior = [_doc("1", "I love hockey.")]  # doc "2" was deleted and pruned from state
    fresh = [_search_item(_doc("1", "I love hockey."), 0.9)]

    merged = await merge_dedup_threshold_documents(
        prior, fresh, "hockey", apply_threshold=False
    )
    assert [d.metadata["document_id"] for d in merged] == ["1"]
