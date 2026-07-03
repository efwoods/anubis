"""Unit tests for the ground-truth calibration store read-back helpers.

Covers `_quote_text_from_store_value`, which must understand the
LangChain-serialized Document envelope the indexer actually persists
(``value.document.kwargs.page_content`` — the same path ``langgraph.json``
points the store's vector index at). Missing that envelope previously made
every stored quote extract as ``None``: the prior corpus read back EMPTY, and
the slow-path rebuild then replaced a ~6k-row feature dict with a single
upload's rows.
"""

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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
