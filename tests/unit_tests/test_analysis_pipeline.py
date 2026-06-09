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


# ---------------------------------------------------------------------------
# Standardized-question analyzer: one model call per question; a question/answer
# document is created only when an answer is found/inferred in the content.
# ---------------------------------------------------------------------------

_TEST_QUESTIONS = [
    "What is your greatest fear?",
    "What is your favorite food?",
    "How do you handle stress?",
]


class _FakeStandardizedQuestionModel:
    """Returns an answer only for questions whose text is in ``answered``."""

    def __init__(self, response_format, answered):
        from src.anubis.utils.prompts.psycho_analysis.standardized_question_analysis_prompt import (
            StandardizedQuestionAnswer,
        )

        self._schema = StandardizedQuestionAnswer
        self._answered = answered

    async def ainvoke(self, messages):
        system = messages[0].content
        for question, payload in self._answered.items():
            if question in system:
                return self._schema(answer_found=True, **payload)
        return self._schema(answer_found=False)


def _patch_standardized(monkeypatch, answered):
    """Stub init_model + shrink the question bank for offline determinism."""
    import src.anubis.utils.analysis.standardized_questions as sq

    monkeypatch.setattr(
        analysis_methods,
        "init_model",
        lambda *a, **k: _FakeStandardizedQuestionModel(
            k.get("response_format"), answered
        ),
    )
    monkeypatch.setattr(sq, "ALL_STANDARDIZED_QUESTIONS", list(_TEST_QUESTIONS))


@pytest.mark.asyncio
async def test_standardized_question_found_answer_creates_pair(monkeypatch):
    """A question whose answer is in the content yields one Q&A analysis doc."""
    from langchain_core.messages import HumanMessage

    _patch_standardized(
        monkeypatch,
        {
            "What is your greatest fear?": {
                "answer": "I fear being forgotten.",
                "supporting_reason": "The content says he dreads being forgotten.",
            }
        },
    )

    docs = await analysis_methods.perform_standardized_question_analysis(
        HumanMessage(content="He often says he dreads being forgotten."),
        target_name="Bob",
        source_metadata={"filename": "bio.txt", "user_id": "u1", "assistant_id": "a1"},
    )

    # Exactly one of the three questions had an answer in the content.
    assert len(docs) == 1
    doc = docs[0]
    # FACT_CONTEXT/FACT structure pairs the question (context) with the answer.
    assert doc.page_content.startswith("<FACT_CONTEXT_AND_FACT> <FACT_CONTEXT>")
    assert 'In response to the question "What is your greatest fear?"' in doc.page_content
    assert "<FACT>I fear being forgotten.</FACT>" in doc.page_content
    # Question/answer pair + analysis-namespace metadata and flags.
    assert doc.metadata["feature"] == "standardized_question"
    assert doc.metadata["question"] == "What is your greatest fear?"
    assert doc.metadata["answer"] == "I fear being forgotten."
    assert doc.metadata["namespace"] == "analysis"
    assert doc.metadata["vectorstore_acceptable"] is True
    assert doc.metadata["adapter_acceptable"] is False
    assert doc.metadata["analysis_acceptable"] is False
    assert doc.metadata["target_name"] == "Bob"
    assert doc.metadata["filename"] == "bio.txt"


@pytest.mark.asyncio
async def test_standardized_question_no_answer_creates_no_doc(monkeypatch):
    """When no question is answered by the content, no documents are created."""
    from langchain_core.messages import HumanMessage

    _patch_standardized(monkeypatch, {})  # nothing found for any question

    docs = await analysis_methods.perform_standardized_question_analysis(
        HumanMessage(content="Unrelated content with no identity signal."),
        target_name="Bob",
        source_metadata={"filename": "bio.txt"},
    )
    assert docs == []


@pytest.mark.asyncio
async def test_standardized_question_blank_answer_dropped(monkeypatch):
    """answer_found=True but an empty answer string still creates no document."""
    from langchain_core.messages import HumanMessage

    _patch_standardized(
        monkeypatch,
        {"What is your favorite food?": {"answer": "   ", "supporting_reason": "x"}},
    )

    docs = await analysis_methods.perform_standardized_question_analysis(
        HumanMessage(content="Some content."),
        target_name="Bob",
        source_metadata={"filename": "bio.txt"},
    )
    assert docs == []


@pytest.mark.asyncio
async def test_standardized_question_empty_source_returns_empty(monkeypatch):
    """Empty source text short-circuits before any model call."""
    from langchain_core.messages import HumanMessage

    _patch_standardized(monkeypatch, {})
    docs = await analysis_methods.perform_standardized_question_analysis(
        HumanMessage(content="   "), target_name="Bob"
    )
    assert docs == []


def test_standardized_question_registered_in_scaffold_runners():
    """The analyzer is wired into the modular registry under a stable key."""
    assert "standardized_questions" in analysis_methods.ANALYSIS_SCAFFOLD_RUNNERS


# ---------------------------------------------------------------------------
# Situational context (spec Step 2): per-target analyzers receive the scene
# summary + the preceding "user" turn instead of analyzing a quote in isolation.
# ---------------------------------------------------------------------------


def test_situational_context_from_doc_combines_scene_and_user():
    """A target quote doc yields a context block with both scene + user turn."""
    doc = Document(
        page_content="I'm supposed to be in Singapore.",
        metadata={
            "scene_summary": "Miranda is redirected to Berlin.",
            "user_context": "Get on it. You're meeting me in Berlin.",
        },
    )
    ctx = analysis_methods._situational_context_from_doc(doc)
    assert "Miranda is redirected to Berlin." in ctx
    assert "Get on it. You're meeting me in Berlin." in ctx


def test_situational_context_falls_back_to_adapter_prompt():
    """When user_context is absent, the adapter_prompt (preceding turn) is used."""
    doc = Document(
        page_content="Yeah.",
        metadata={"adapter_prompt": "See that plane?"},
    )
    ctx = analysis_methods._situational_context_from_doc(doc)
    assert "See that plane?" in ctx


def test_situational_context_none_for_isolated_doc():
    """Docs with no scene/user context (e.g. biographical) analyze in isolation."""
    doc = Document(page_content="A fact.", metadata={"filename": "bio.txt"})
    assert analysis_methods._situational_context_from_doc(doc) is None


def test_format_analysis_input_wraps_and_passthrough():
    """The wrapper labels context + statement; empty context is passthrough."""
    from src.anubis.utils.classes.LatentFeatureAnalysisClass import (
        format_analysis_input_with_context,
    )

    wrapped = format_analysis_input_with_context("Speaking.", "Scene: a call.")
    assert "<SITUATIONAL_CONTEXT>" in wrapped
    assert "Scene: a call." in wrapped
    assert "<TARGET_STATEMENT>" in wrapped
    assert "Speaking." in wrapped
    # No context -> the raw statement is returned unchanged.
    assert format_analysis_input_with_context("Speaking.", None) == "Speaking."
    assert format_analysis_input_with_context("Speaking.", "  ") == "Speaking."


@pytest.mark.asyncio
async def test_standardized_question_sees_situational_context(monkeypatch):
    """The model input is contextualized, but provenance stays the raw statement."""
    from langchain_core.messages import HumanMessage

    seen = {}

    class _CapturingModel:
        def __init__(self, response_format):
            from src.anubis.utils.prompts.psycho_analysis.standardized_question_analysis_prompt import (
                StandardizedQuestionAnswer,
            )

            self._schema = StandardizedQuestionAnswer

        async def ainvoke(self, messages):
            seen["human"] = messages[1].content
            return self._schema(answer_found=False)

    import src.anubis.utils.analysis.standardized_questions as sq

    monkeypatch.setattr(
        analysis_methods,
        "init_model",
        lambda *a, **k: _CapturingModel(k.get("response_format")),
    )
    monkeypatch.setattr(sq, "ALL_STANDARDIZED_QUESTIONS", ["What is your name?"])

    docs = await analysis_methods.perform_standardized_question_analysis(
        HumanMessage(content="I'm supposed to be in Singapore."),
        target_name="Miranda",
        source_metadata={"filename": "scene.wav"},
        situational_context="Scene: Miranda is redirected to Berlin.",
    )
    assert docs == []  # no answer found in this stub
    # The model saw the wrapped input carrying the scene context...
    assert "Scene: Miranda is redirected to Berlin." in seen["human"]
    assert "I'm supposed to be in Singapore." in seen["human"]


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


# --------------------------------------------------------------------------- #
# batch_index_documents_vectorstore — replace-by-filename must be scoped to the
# (namespace_filename, namespace) pair, so the analysis pass does NOT delete the
# source quote/identity docs it was derived from (they share namespace_filename).
# --------------------------------------------------------------------------- #


class _FakeStoreItem(SimpleNamespace):
    pass


class _FakeStore:
    """Minimal store: records puts, supports prefix search and delete."""

    def __init__(self):
        # keyed by (namespace_tuple, key) -> value dict
        self._items: dict[tuple, dict] = {}

    def seed(self, namespace, key, value):
        self._items[(namespace, key)] = value

    async def asearch(self, prefix, limit=1000):
        return [
            _FakeStoreItem(namespace=ns, key=key, value=value)
            for (ns, key), value in self._items.items()
            if ns[: len(prefix)] == tuple(prefix)
        ]

    async def adelete(self, namespace, key):
        self._items.pop((namespace, key), None)

    async def abatch(self, ops):
        for op in ops:
            self._items[(op.namespace, op.key)] = op.value

    def keys(self):
        return set(self._items.keys())


def _store_doc(doc_id, namespace_filename, namespace):
    return Document(
        page_content=doc_id,
        metadata={
            "document_id": doc_id,
            "namespace_filename": namespace_filename,
            "namespace": namespace,
        },
    )


@pytest.mark.asyncio
async def test_analysis_pass_does_not_delete_source_docs_same_filename():
    """The bug: indexing analysis docs wiped the source docs sharing a filename.

    Source (quote) docs are indexed first; the derived analysis docs carry the
    SAME namespace_filename. The replace-by-filename sweep must only replace
    prior rows of the SAME namespace, so the quote source survives.
    """
    from src.subgraphs.vector_store_graph.utils.helper_functions import (
        batch_index_documents_vectorstore,
    )

    store = _FakeStore()
    # First pass already persisted a quote source doc for nsf1.
    src = _store_doc("src1", "nsf1", "quote")
    src_ns = ("u", "a", "quote", "nsf1")
    store.seed(src_ns, "src1", {"document": src.to_json()})

    # Second pass: index the analysis doc derived from it (same namespace_filename).
    analysis = _store_doc("an1", "nsf1", "analysis")
    result = await batch_index_documents_vectorstore(store, "u", "a", [analysis])

    assert result["success"] is True
    keys = store.keys()
    # The quote source doc must NOT have been swept away...
    assert (src_ns, "src1") in keys
    # ...and the analysis doc is now indexed under the analysis namespace.
    assert (("u", "a", "analysis", "nsf1"), "an1") in keys


@pytest.mark.asyncio
async def test_reindex_same_namespace_replaces_prior_version():
    """Same-namespace re-index of a file still replaces its prior rows."""
    from src.subgraphs.vector_store_graph.utils.helper_functions import (
        batch_index_documents_vectorstore,
    )

    store = _FakeStore()
    old = _store_doc("old1", "nsf1", "quote")
    old_ns = ("u", "a", "quote", "nsf1")
    store.seed(old_ns, "old1", {"document": old.to_json()})

    new = _store_doc("new1", "nsf1", "quote")
    await batch_index_documents_vectorstore(store, "u", "a", [new])

    keys = store.keys()
    assert (old_ns, "old1") not in keys  # prior version replaced
    assert (old_ns, "new1") in keys
