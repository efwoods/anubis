"""Unit tests for the modular psycho-analysis pipeline.

Covers the deterministic, offline pieces:

* ``LatentFeatureAnalysisClass.analyze`` — FACT_CONTEXT/FACT formatting,
  analysis-namespace metadata, and acceptability flags.
* ``analyze_documents`` — parallel fan-out over a narrowed analyzer set, with
  outputs merged into the vector-store index batch and the analysis queue
  cleared.

All LLM calls (``init_model``) are stubbed so the tests stay offline.
"""

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import src.anubis.utils.analysis.analysis_methods as analysis_methods
import src.anubis.utils.classes.LatentFeatureAnalysisClass as lfa_module
import src.subgraphs.vector_store_graph.index_graph as index_graph
from src.anubis.utils.classes.FactRewriterClass import (
    ConciseContextOfTheSourceOfFacts,
)
from src.anubis.utils.classes.LatentFeatureAnalysisClass import (
    ExtractedLatentFeature,
    ExtractedLatentFeatureList,
    LatentFeatureAnalysisClass,
)


class _FakeModel:
    """Stand-in for an init_model result; returns a fixed structured output."""

    def __init__(self, response_format):
        self._response_format = response_format

    async def ainvoke(self, messages):
        if self._response_format is ExtractedLatentFeatureList:
            return ExtractedLatentFeatureList(
                features=[
                    ExtractedLatentFeature(
                        feature_statement="I believe hard work pays off.",
                        supporting_reason="The speaker repeatedly credits effort.",
                    )
                ]
            )
        if self._response_format is ConciseContextOfTheSourceOfFacts:
            return ConciseContextOfTheSourceOfFacts(
                concise_context_summary="A career retrospective by the speaker."
            )
        raise AssertionError(f"unexpected response_format {self._response_format}")


@pytest.fixture
def stub_init_model(monkeypatch):
    """Replace init_model in the analyzer module with an offline fake."""

    def _fake_init_model(*args, **kwargs):
        return _FakeModel(kwargs.get("response_format"))

    monkeypatch.setattr(lfa_module, "init_model", _fake_init_model)
    # Drop any cached analyzers so the fake init_model is used when (re)built.
    analysis_methods._NARRATIVE_ANALYZER_CACHE.clear()
    yield
    analysis_methods._NARRATIVE_ANALYZER_CACHE.clear()


@pytest.mark.asyncio
async def test_latent_feature_analysis_formats_and_tags(stub_init_model):
    analyzer = LatentFeatureAnalysisClass("belief", "PROMPT {target_name}")
    docs = await analyzer.analyze(
        "I worked hard and it paid off.",
        target_name="Jane",
        source_metadata={"filename": "bio.txt", "user_id": "u1", "assistant_id": "a1"},
    )

    assert len(docs) == 1
    doc = docs[0]
    # FACT_CONTEXT/FACT structure wraps context + first-person finding.
    assert doc.page_content.startswith("<FACT_CONTEXT_AND_FACT> <FACT_CONTEXT>")
    assert "A career retrospective by the speaker." in doc.page_content
    assert "<FACT>I believe hard work pays off.</FACT>" in doc.page_content
    # Analysis-namespace metadata + acceptability flags.
    assert doc.metadata["namespace"] == "analysis"
    assert doc.metadata["vectorstore_acceptable"] is True
    assert doc.metadata["adapter_acceptable"] is False
    assert doc.metadata["analysis_acceptable"] is False
    assert doc.metadata["feature"] == "belief"
    assert doc.metadata["belief"] == "I believe hard work pays off."
    assert doc.metadata["target_name"] == "Jane"
    # Source metadata carried forward.
    assert doc.metadata["filename"] == "bio.txt"


@pytest.mark.asyncio
async def test_latent_feature_analysis_empty_text_returns_empty(stub_init_model):
    analyzer = LatentFeatureAnalysisClass("belief", "PROMPT")
    assert await analyzer.analyze("   ") == []


@pytest.mark.asyncio
async def test_analyze_documents_returns_only_new_docs(stub_init_model):
    """analyze_documents contributes only its own new analysis docs.

    It must NOT re-read/return the existing buffer — the append reducer merges
    them, and re-reading would risk re-queuing already-processed docs.
    """
    source = Document(
        page_content="I worked hard and it paid off.",
        metadata={
            "analysis_acceptable": True,
            "vectorstore_acceptable": False,
            "analysis_scaffolds": ["beliefs"],  # narrow to one analyzer
            "target_name": "Jane",
            "filename": "bio.txt",
        },
    )
    existing = Document(page_content="source doc", metadata={"namespace": "quote"})
    state = {
        "documents_to_be_analyzed_for_context_storage_and_prompt_injection_of_assistant": [
            source
        ],
        "vectorstore_documents_to_be_indexed": [existing],
    }

    from src.subgraphs.process_media_graph.utils.nodes import analyze_documents

    result = await analyze_documents(state, None, None, None)

    # Analysis queue cleared.
    assert (
        result[
            "documents_to_be_analyzed_for_context_storage_and_prompt_injection_of_assistant"
        ]
        == "delete"
    )
    returned = result["vectorstore_documents_to_be_indexed"]
    # Only the freshly produced analysis docs are returned; the existing buffer
    # doc is left for the reducer to append, not re-emitted here.
    assert existing not in returned
    assert len(returned) == 1
    assert returned[0].metadata["namespace"] == "analysis"
    assert returned[0].metadata["feature"] == "belief"
    assert returned[0].metadata["analysis_acceptable"] is False


def test_reduce_docs_appends_dedupes_and_deletes():
    """The buffer reducer appends new docs, de-dupes by id, and clears on delete."""
    from src.anubis.utils.utility import reduce_docs

    a = Document(page_content="A", metadata={"document_id": "1"})
    b = Document(page_content="B", metadata={"document_id": "2"})
    c = Document(page_content="C", metadata={"document_id": "3"})

    # convert puts [a, b]; analyze later appends [c] -> all three present.
    buf = reduce_docs([], [a, b])
    buf = reduce_docs(buf, [c])
    assert [d.metadata["document_id"] for d in buf] == ["1", "2", "3"]

    # Re-emitting already-present docs (e.g. load_consciousness re-returning its
    # cached identity list) is idempotent — no growth.
    buf2 = reduce_docs(buf, [a, b, c])
    assert [d.metadata["document_id"] for d in buf2] == ["1", "2", "3"]

    # "delete" still clears the entire buffer.
    assert reduce_docs(buf2, "delete") == []


def test_reduce_docs_targeted_removal_keeps_unprocessed():
    """remove_docs_update drops only processed docs; the rest survive."""
    from src.anubis.utils.utility import reduce_docs, remove_docs_update

    a = Document(page_content="A", metadata={"document_id": "1"})
    b = Document(page_content="B", metadata={"document_id": "2"})
    c = Document(page_content="C", metadata={"document_id": "3"})

    buf = reduce_docs([], [a, b, c])
    # index_docs processed only [a, b]; c was appended after its snapshot.
    buf = reduce_docs(buf, remove_docs_update([a, b]))
    assert [d.metadata["document_id"] for d in buf] == ["3"]


def test_reduce_docs_remove_and_append_are_order_independent():
    """The SS3 race: a concurrent index removal + analyze append commute."""
    from src.anubis.utils.utility import reduce_docs, remove_docs_update

    a = Document(page_content="A", metadata={"document_id": "1"})
    b = Document(page_content="B", metadata={"document_id": "2"})
    c = Document(page_content="C", metadata={"document_id": "3"})
    d = Document(page_content="D", metadata={"document_id": "4"})

    start = reduce_docs([], [a, b])

    # Order 1: append [c, d] then remove [a, b].
    o1 = reduce_docs(reduce_docs(start, [c, d]), remove_docs_update([a, b]))
    # Order 2: remove [a, b] then append [c, d].
    o2 = reduce_docs(reduce_docs(start, remove_docs_update([a, b])), [c, d])

    assert sorted(x.metadata["document_id"] for x in o1) == ["3", "4"]
    assert sorted(x.metadata["document_id"] for x in o2) == ["3", "4"]


# --------------------------------------------------------------------------- #
# index_docs — tolerate per-file failures without stopping the pipeline
# --------------------------------------------------------------------------- #


@pytest.fixture
def stub_index_ids(monkeypatch):
    async def _fake_ids(config):
        return {"user_id": "u"}, {"assistant_id": "a"}

    monkeypatch.setattr(index_graph, "extract_user_id_assistant_id", _fake_ids)


def _doc(doc_id, filename):
    return Document(
        page_content=doc_id,
        metadata={"document_id": doc_id, "filename": filename, "namespace": "identity"},
    )


@pytest.mark.asyncio
async def test_index_docs_partial_failure_reports_failed_files(monkeypatch, stub_index_ids):
    a1, a2, b1 = _doc("a1", "fileA.txt"), _doc("a2", "fileA.txt"), _doc("b1", "fileB.txt")

    async def _fake_batch(store, user_id, assistant_id, docs, BATCH_SIZE=1000):
        # fileA's a2 failed; everything else succeeded.
        return {"success": False, "error_batch_documents": [SimpleNamespace(key="a2")]}

    monkeypatch.setattr(index_graph, "batch_index_documents_vectorstore", _fake_batch)

    state = {"vectorstore_documents_to_be_indexed": [a1, a2, b1]}
    result = await index_graph.index_docs(state, None, {}, None)

    # Whole attempted snapshot removed from the buffer (targeted removal).
    removal = result["vectorstore_documents_to_be_indexed"]
    assert removal["op"] == "remove"
    assert set(removal["keys"]) == {"a1", "a2", "b1"}

    # Only fileA reported failed; fileB is not present.
    failed = result["failed_to_index_files"]
    assert len(failed) == 1
    assert failed[0]["filename"] == "fileA.txt"
    # Whole file marked failed: both of fileA's docs listed for reprocessing.
    assert sorted(failed[0]["document_ids"]) == ["a1", "a2"]


@pytest.mark.asyncio
async def test_index_docs_hard_exception_marks_all_failed(monkeypatch, stub_index_ids):
    a1, b1 = _doc("a1", "fileA.txt"), _doc("b1", "fileB.txt")

    async def _boom(*args, **kwargs):
        raise RuntimeError("store down")

    monkeypatch.setattr(index_graph, "batch_index_documents_vectorstore", _boom)

    state = {"vectorstore_documents_to_be_indexed": [a1, b1]}
    result = await index_graph.index_docs(state, None, {}, None)  # must not raise

    failed_names = sorted(f["filename"] for f in result["failed_to_index_files"])
    assert failed_names == ["fileA.txt", "fileB.txt"]
    assert result["vectorstore_documents_to_be_indexed"]["op"] == "remove"


@pytest.mark.asyncio
async def test_index_docs_success_reports_no_failures(monkeypatch, stub_index_ids):
    a1 = _doc("a1", "fileA.txt")

    async def _ok(*args, **kwargs):
        return {"success": True, "documents_uploaded": 1}

    monkeypatch.setattr(index_graph, "batch_index_documents_vectorstore", _ok)

    state = {"vectorstore_documents_to_be_indexed": [a1]}
    result = await index_graph.index_docs(state, None, {}, None)

    assert result["failed_to_index_files"] == []
    assert result["vectorstore_documents_to_be_indexed"]["keys"] == ["a1"]
