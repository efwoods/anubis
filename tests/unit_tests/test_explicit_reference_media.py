"""Explicit reference_media uploads skip identity classification."""

import pytest

from src.subgraphs.process_media_graph.utils.helper_functions import (
    process_text_to_document,
)


@pytest.mark.asyncio
async def test_explicit_reference_media_skips_classifiers_and_identity(monkeypatch):
    """A menu flagged reference_media must land in the document namespace
    without running the reference or situation classifiers."""

    def _fail_if_constructed(*_args, **_kwargs):
        raise AssertionError("identity classifiers must not run for reference media")

    monkeypatch.setattr(
        "src.subgraphs.process_media_graph.utils.helper_functions.ReferenceDocumentClassificationClass",
        _fail_if_constructed,
    )
    monkeypatch.setattr(
        "src.subgraphs.process_media_graph.utils.helper_functions.ContentSituationClassificationClass",
        _fail_if_constructed,
    )

    documents = await process_text_to_document(
        metadata={},
        user_id="u1",
        assistant_id="a1",
        media_item={
            "content": "Cortado: $5.72, 90 cal\nCaffe Americano: $5.17, 15 cal\n",
            "metadata": {
                "filename": "menu.txt",
                "namespace_filename": "menu.txt",
                "reference_media": True,
            },
        },
        store=None,
    )

    assert documents
    assert all(document.metadata.get("namespace") == "document" for document in documents)
    assert all(document.metadata.get("analysis_acceptable") is False for document in documents)
    assert all(document.metadata.get("adapter_acceptable") is False for document in documents)
    assert all(document.metadata.get("reference_media") is True for document in documents)
    assert all(
        document.metadata.get("classified_situation") == "proprietary_content"
        for document in documents
    )
    assert any("Cortado" in (document.page_content or "") for document in documents)
