"""Unit tests for the modular psycho-analysis pipeline.

Covers the deterministic, offline pieces:

* ``LatentFeatureAnalysisClass.analyze`` — FACT_CONTEXT/FACT formatting,
  analysis-namespace metadata, and acceptability flags.
* ``analyze_documents`` — parallel fan-out over a narrowed analyzer set, with
  outputs merged into the vector-store index batch and the analysis queue
  cleared.

All LLM calls (``init_model``) are stubbed so the tests stay offline.
"""

import pytest
from langchain_core.documents import Document

import src.anubis.utils.analysis.analysis_methods as analysis_methods
import src.anubis.utils.classes.LatentFeatureAnalysisClass as lfa_module
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
async def test_analyze_documents_fans_out_and_merges(stub_init_model):
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
    indexed = result["vectorstore_documents_to_be_indexed"]
    # Pre-existing index doc preserved; one analysis doc appended.
    assert existing in indexed
    analysis_docs = [d for d in indexed if d.metadata.get("namespace") == "analysis"]
    assert len(analysis_docs) == 1
    assert analysis_docs[0].metadata["feature"] == "belief"
    assert analysis_docs[0].metadata["analysis_acceptable"] is False
