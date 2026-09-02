"""Unit tests for the ground-truth calibration store read-back helpers.

Covers `_quote_text_from_store_value`, which must understand the
LangChain-serialized Document envelope the indexer actually persists
(``value.document.kwargs.page_content`` — the same path ``langgraph.json``
points the store's vector index at). Missing that envelope previously made
every stored quote extract as ``None``: the prior corpus read back EMPTY, and
the slow-path rebuild then replaced a ~6k-row feature dict with a single
upload's rows.
"""

import json

import pytest

from src.subgraphs.process_media_graph.utils.calibrate_ground_truth import (
    _quote_text_from_store_value,
)


def test_extracts_langchain_serialized_document_envelope():
    # The shape the indexer persists (matches langgraph.json's
    # document.kwargs.page_content index path).
    value = {
        "document": {
            "id": ["langchain", "schema", "document", "Document"],
            "lc": 1,
            "type": "constructor",
            "kwargs": {
                "type": "Document",
                "page_content": "Sock Con, the conference for socks",
                "metadata": {"document_id": "abc"},
            },
        }
    }
    assert (
        _quote_text_from_store_value(value)
        == "Sock Con, the conference for socks"
    )


def test_extracts_flat_fallback_shapes():
    assert _quote_text_from_store_value({"page_content": " hello "}) == "hello"
    assert (
        _quote_text_from_store_value({"document": {"page_content": "hi"}}) == "hi"
    )


def test_missing_or_blank_content_returns_none():
    assert _quote_text_from_store_value(None) is None
    assert _quote_text_from_store_value({}) is None
    assert _quote_text_from_store_value({"document": {"kwargs": {}}}) is None
    assert (
        _quote_text_from_store_value(
            {"document": {"kwargs": {"page_content": "   "}}}
        )
        is None
    )


# ---------------------------------------------------------------------------
# Driving calibration from the stored corpus
# ---------------------------------------------------------------------------
# The post-upload hook and the backfill script both call calibration with NO
# documents, because by then everything is already indexed and the corpus is read
# out of the store. These cover that form.


class _FakeQuoteStore:
    """Dict-backed store exposing just the surface calibration touches.

    ``asearch`` returns the seeded quote items (prefix-matched the way the real
    store matches a namespace prefix against deeper namespaces); ``aget``/``aput``
    operate on a plain dict keyed by ``(namespace, key)`` so a test can assert
    exactly which artifacts were written and under which namespace.
    """

    def __init__(self, quote_items=None, asearch_failure: Exception | None = None):
        self.quote_items = list(quote_items or [])
        self.asearch_failure = asearch_failure
        self.contents: dict = {}
        self.put_calls: list = []

    async def asearch(self, namespace, query=None, limit=None):
        if self.asearch_failure is not None:
            raise self.asearch_failure
        return list(self.quote_items)

    async def aget(self, namespace, key):
        return self.contents.get((tuple(namespace), key))

    async def aput(self, namespace, key, value):
        self.put_calls.append((tuple(namespace), key))
        self.contents[(tuple(namespace), key)] = _StoreItem(value)


class _StoreItem:
    def __init__(self, value):
        self.value = value


class _QuoteItem:
    """One stored quote, in the LangChain-serialized envelope the indexer writes."""

    def __init__(self, key: str, text: str):
        self.key = key
        self.value = {"document": {"kwargs": {"page_content": text}}}


# Structurally VARIED sample texts. The calibration fits a covariance matrix over
# the feature columns and inverts it, so a corpus built from one repeated template
# makes several columns constant, the covariance singular, and the fit raise —
# an artifact of the fixture, not of the code. These vary sentence length and
# count, question and exclamation use, paragraphing, and vocabulary, the way a
# real quote corpus does.
_QUOTE_SENTENCE_POOL = (
    "The hospital is not where life stops. It is another room, and I decided a "
    "long time ago that I would keep living in it.",
    "Do you know what nobody tells you about being sick? Everyone wants to make "
    "you into a lesson! I refuse.",
    "I want to change the hospital for people who are sick, and I want to change "
    "pity into something that actually resembles respect.\n\nThat is the whole "
    "project, really.",
    "My mother gave me room to be exactly who I was, including the parts that "
    "were stupid and ridiculous, and that is why I am the way I am.",
    "Pride matters. Not the pretty kind — the kind where you look at the mess of "
    "your own life and say yes, that was mine, I did that.",
    "When you have something bigger you are living for, taking care of yourself "
    "stops being a chore; it becomes the thing that buys you more time to do the "
    "work.",
    "People ask me if I am scared! Of course I am scared. Being scared and being "
    "functional turn out to be completely compatible.",
    "I have a few years and a very long list. I would rather spend them building "
    "something than waiting for somebody to arrive and fix me first.",
    "Bucket lists are an empty way to live. Checking experiences off a page never "
    "once made my life feel worth living; the work did, and the giving did.",
    "There is a version of this where I am a tragedy. I am not interested in "
    "that version, and I have never been willing to perform it for anybody.",
    "Families living with this need support that treats them as capable, not as "
    "victims who happen to be standing near a diagnosis.",
    "I talk about dying because it is true, and because pretending otherwise "
    "makes everybody lonelier — me most of all.",
    "What does dignity actually look like in a hospital room? Smaller things "
    "than you would think. Being asked. Being told. Being believed.",
    "I am eighteen and I have opinions about mortality, which people find "
    "unsettling, and honestly that reaction tells me more about them than me.",
    "The messiness counts too! You do not get to keep only the beautiful parts "
    "and call that a life.",
    "I built something out of the years I had, and whatever happens next, that "
    "part is finished and it is mine.",
    "Empowerment is not a poster. It is the difference between somebody deciding "
    "for you and somebody asking you what you want.",
)


def _distinct_quote(index: int) -> str:
    """Return a sample quote long and varied enough to yield a usable feature row.

    The extractor returns an all-NaN row for text it cannot measure, and such rows
    are deliberately dropped before the corpus row count is checked, so short
    filler would make these tests assert against an empty corpus.
    """
    return _QUOTE_SENTENCE_POOL[index % len(_QUOTE_SENTENCE_POOL)]


@pytest.fixture
def deterministic_key_phrases(monkeypatch):
    """Pin phrase discovery so the fast/slow path choice is what a test intends.

    Patched on the DEFINING module, not on the calibration module: calibration
    imports these lazily inside the function body (the module pulls in numpy and
    scikit-learn, so the imports are deferred), which means the names are resolved
    from the source module at call time and never exist as calibration-module
    attributes.
    """
    import src.anubis.utils.dataset.key_phrases as key_phrases_module

    monkeypatch.setattr(
        key_phrases_module,
        "discover_key_phrases",
        lambda documents: [{"phrase": "the hospital"}],
    )
    monkeypatch.setattr(
        key_phrases_module,
        "build_corpus_phrase_attestation_set",
        lambda documents: {"the hospital"},
    )
    return ["the hospital"]


@pytest.mark.asyncio
async def test_stored_corpus_alone_calibrates_every_artifact(
    deterministic_key_phrases,
):
    """With enough quotes and no documents passed, all five artifacts are written."""
    from src.subgraphs.process_media_graph.utils.calibrate_ground_truth import (
        MIN_ROWS_FOR_CALIBRATION,
        calibrate_ground_truth_from_stored_corpus,
    )
    from src.anubis.utils.dataset.style_features import GROUND_TRUTH_FEATURES_DICT_KEY

    quote_count = MIN_ROWS_FOR_CALIBRATION + 2
    store = _FakeQuoteStore(
        [_QuoteItem(f"doc-{i}", _distinct_quote(i)) for i in range(quote_count)]
    )

    await calibrate_ground_truth_from_stored_corpus(
        store=store, assistant_id="a1", user_id="owner1"
    )

    written = {(namespace, key) for namespace, key in store.put_calls}
    # Owner-scoped namespaces, artifact name as both the third element and the
    # key — the exact shape the message path reads back.
    assert ("owner1", "a1", "key_phrase_profile") in {ns for ns, _ in written}
    for artifact_name in (
        GROUND_TRUTH_FEATURES_DICT_KEY,
        "ground_truth_text_empirical_threshold_list_str",
        "ground_truth_text_features_model_b64_pkl",
        "style_profile",
        "key_phrase_profile",
    ):
        assert (("owner1", "a1", artifact_name), artifact_name) in written


@pytest.mark.asyncio
async def test_corpus_below_floor_defers_threshold_and_model(
    deterministic_key_phrases,
):
    """Too few quotes: keep accumulating, but do not fit a degenerate cloud."""
    from src.subgraphs.process_media_graph.utils.calibrate_ground_truth import (
        calibrate_ground_truth_from_stored_corpus,
    )
    from src.anubis.utils.dataset.style_features import GROUND_TRUTH_FEATURES_DICT_KEY

    store = _FakeQuoteStore(
        [_QuoteItem(f"doc-{i}", _distinct_quote(i)) for i in range(3)]
    )

    await calibrate_ground_truth_from_stored_corpus(
        store=store, assistant_id="a1", user_id="owner1"
    )

    written_keys = {key for _, key in store.put_calls}
    assert GROUND_TRUTH_FEATURES_DICT_KEY in written_keys
    assert "key_phrase_profile" in written_keys
    assert "ground_truth_text_empirical_threshold_list_str" not in written_keys
    assert "ground_truth_text_features_model_b64_pkl" not in written_keys


@pytest.mark.asyncio
async def test_fast_path_extracts_quotes_missing_from_the_persisted_corpus(
    deterministic_key_phrases,
):
    """The incremental path must follow the corpus, not the passed documents.

    When the phrase set is unchanged the existing rows stay valid, so only the
    unmeasured quotes are extracted. Deriving that set from the passed documents
    instead means a store-driven call (which passes none) measures nothing — the
    post-upload hook and the backfill would both become silent no-ops from the
    second run onward.
    """
    from src.subgraphs.process_media_graph.utils.calibrate_ground_truth import (
        calibrate_ground_truth_from_stored_corpus,
    )
    from src.anubis.utils.dataset.style_features import (
        FEATURE_NAMES,
        GROUND_TRUTH_FEATURES_DICT_KEY,
        deserialize_features_by_doc_id,
        extract_style_features,
        serialize_features_by_doc_id,
    )
    import numpy as np

    total_quotes = 15
    already_measured = 10
    store = _FakeQuoteStore(
        [_QuoteItem(f"doc-{i}", _distinct_quote(i)) for i in range(total_quotes)]
    )

    # Seed a persisted corpus covering only the first ten quotes, measured against
    # the same phrase set discovery will return — so the run takes the fast path.
    seeded = {}
    for i in range(already_measured):
        features = extract_style_features(
            _distinct_quote(i), key_phrases=deterministic_key_phrases
        )
        seeded[f"doc-{i}"] = np.array(
            [features[name] for name in FEATURE_NAMES], dtype=np.float64
        )
    namespace = ("owner1", "a1", GROUND_TRUTH_FEATURES_DICT_KEY)
    store.contents[(namespace, GROUND_TRUTH_FEATURES_DICT_KEY)] = _StoreItem(
        {"value": serialize_features_by_doc_id(seeded)}
    )
    store.contents[
        (("owner1", "a1", "key_phrase_profile"), "key_phrase_profile")
    ] = _StoreItem({"value": json.dumps(deterministic_key_phrases)})

    await calibrate_ground_truth_from_stored_corpus(
        store=store, assistant_id="a1", user_id="owner1"
    )

    persisted = deserialize_features_by_doc_id(
        store.contents[(namespace, GROUND_TRUTH_FEATURES_DICT_KEY)].value["value"]
    )
    assert len(persisted) == total_quotes


@pytest.mark.asyncio
async def test_unreadable_corpus_with_no_documents_writes_nothing(
    deterministic_key_phrases,
):
    """A failed read-back must not be mistaken for an avatar with no quotes.

    With no documents passed the store is the only source of the corpus, so
    treating a read failure as "empty" would persist a degenerate corpus over one
    that may hold thousands of rows.
    """
    from src.subgraphs.process_media_graph.utils.calibrate_ground_truth import (
        calibrate_ground_truth_from_stored_corpus,
    )

    store = _FakeQuoteStore(asearch_failure=RuntimeError("connection reset"))

    with pytest.raises(RuntimeError, match="connection reset"):
        await calibrate_ground_truth_from_stored_corpus(
            store=store, assistant_id="a1", user_id="owner1"
        )

    assert store.put_calls == []


@pytest.mark.asyncio
async def test_unreadable_corpus_still_proceeds_when_documents_were_passed(
    deterministic_key_phrases,
):
    """An upload carrying its own documents has something real to calibrate over."""
    from langchain_core.documents import Document

    from src.subgraphs.process_media_graph.utils.calibrate_ground_truth import (
        calibrate_ground_truth,
    )

    store = _FakeQuoteStore(asearch_failure=RuntimeError("connection reset"))
    documents = [
        Document(
            page_content=_distinct_quote(i), metadata={"document_id": f"doc-{i}"}
        )
        for i in range(3)
    ]

    await calibrate_ground_truth(
        store=store, assistant_id="a1", documents=documents, user_id="owner1"
    )

    assert store.put_calls  # degraded to "new documents only", but still wrote


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
