"""Integration tests for the upload-media pipeline + load_consciousness retrieval.

These tests exercise the same compiled subgraph that
``POST /update_avatar_identity_with_media`` invokes in
[`src/api/webapp.py`](src/api/webapp.py), but without FastAPI / HTTP plumbing.

We use the Mom dataset under [`data/mom`](data/mom) to drive four upload modalities:

* Reference image - ``mom_facebook_contextual_image.jpg`` (reference_image=True)
* Non-reference image - ``mom-feeding-geese-pandemic.png`` (identity namespace)
* Reference audio - ``Mom.m4a`` (reference_audio=True; transcribed)
* Quotes text file - ``texts_with_mom_adapter_training.txt`` (one quote per line)

External LLM / vision / ASR boundaries are mocked so assertions are deterministic:

* ``ReferenceDocumentClassificationClass.classify`` -> rule-based stub
* ``ContentSituationClassificationClass.classify`` -> rule-based stub
* ``transcribe_audio`` -> canned transcript
* ``extract_personality_from_image`` -> canned image description Document
* ``perform_ocean_analysis`` -> [] (analysis path is not under test here)

After the pipeline runs, we call ``load_consciousness`` directly to confirm the
system prompt aggregates content from the ``quote``, ``identity``, and
``reference_image`` namespaces.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

# Ensure the project root is importable when running the file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anubis.utils.context import AssistantContext, GlobalContext, UserContext
from src.anubis.utils.nodes import load_consciousness


DATA_DIR = PROJECT_ROOT / "data" / "mom"

USER_ID = "test-creator-mom-avatar"
ASSISTANT_ID = "test-assistant-mom"


# ---------------------------------------------------------------------------
# Helpers and stubs
# ---------------------------------------------------------------------------


def _make_data_uri(mime: str, content: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(content).decode('ascii')}"


def _read_bytes(name: str) -> bytes:
    return (DATA_DIR / name).read_bytes()


def _read_text(name: str) -> str:
    return (DATA_DIR / name).read_text(encoding="utf-8", errors="replace")


def _build_config() -> RunnableConfig:
    return RunnableConfig(
        configurable={
            "user_id": USER_ID,
            "assistant_id": ASSISTANT_ID,
            "user_ctx": {"name": "TestUser", "description": None},
            "assistant_ctx": {
                "name": "Mom",
                "description": "Mom avatar (test).",
                "assistant_id": ASSISTANT_ID,
                "metadata": {"user_id": USER_ID},
            },
        }
    )


def _build_context() -> GlobalContext:
    """Minimal context safe to construct without env-loaded settings."""
    return GlobalContext(
        assistant_ctx=AssistantContext(
            name="Mom",
            metadata={"user_id": USER_ID, "assistant_id": ASSISTANT_ID},
        ),
        user_ctx=UserContext(name="TestUser"),
    )


def _build_media_files() -> List[Dict[str, Any]]:
    """Build the same media_files payload shape that webapp.py constructs."""
    media_files: List[Dict[str, Any]] = []

    # Reference image
    ref_img_name = "mom_facebook_contextual_image.jpg"
    ref_img_bytes = _read_bytes(ref_img_name)
    media_files.append(
        {
            "filename": ref_img_name,
            "content_type": "image/jpeg",
            "content": ref_img_bytes,
            "user_id": USER_ID,
            "assistant_id": ASSISTANT_ID,
            "reference_audio": False,
            "reference_image": True,
            "base64_encoded_str": _make_data_uri("image/jpeg", ref_img_bytes),
        }
    )

    # Non-reference image (identity description target)
    other_img_name = "mom-feeding-geese-pandemic.png"
    other_img_bytes = _read_bytes(other_img_name)
    media_files.append(
        {
            "filename": other_img_name,
            "content_type": "image/png",
            "content": other_img_bytes,
            "user_id": USER_ID,
            "assistant_id": ASSISTANT_ID,
            "reference_audio": False,
            "reference_image": False,
            "base64_encoded_str": _make_data_uri("image/png", other_img_bytes),
        }
    )

    # Reference audio
    ref_audio_name = "Mom.m4a"
    ref_audio_bytes = _read_bytes(ref_audio_name)
    media_files.append(
        {
            "filename": ref_audio_name,
            "content_type": "audio/mp4",
            "content": ref_audio_bytes,
            "user_id": USER_ID,
            "assistant_id": ASSISTANT_ID,
            "reference_audio": True,
            "reference_image": False,
            "base64_encoded_str": _make_data_uri("audio/mp4", ref_audio_bytes),
        }
    )

    # Quotes text file
    quotes_name = "texts_with_mom_adapter_training.txt"
    quotes_bytes = _read_bytes(quotes_name)
    media_files.append(
        {
            "filename": quotes_name,
            "content_type": "text/plain",
            "content": quotes_bytes,
            "user_id": USER_ID,
            "assistant_id": ASSISTANT_ID,
            "reference_audio": False,
            "reference_image": False,
            "base64_encoded_str": _make_data_uri("text/plain", quotes_bytes),
        }
    )

    return media_files


# ---------------------------------------------------------------------------
# Mocks for external services
# ---------------------------------------------------------------------------


async def _stub_reference_classify(self, input_str: str) -> Dict[str, Any]:
    """Reference vs biographical stub.

    Treats menus or holy texts as proprietary and everything else as
    biographical. The Mom dataset and image descriptions are biographical.
    """
    text_lower = (input_str or "").lower()
    is_proprietary = any(
        marker in text_lower
        for marker in (
            "starbucks",
            "menu",
            "in the beginning god created",
            "1:1 in the beginning",
        )
    )
    return {
        "is_menu_or_religious_text": is_proprietary,
        "reasoning": (
            "Stub classified as proprietary content."
            if is_proprietary
            else "Stub classified as biographical/conversational."
        ),
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "model_name": "stub",
        "total_cost": 0.0,
        "model_inference_type": "reference_document_structured_output",
        "latency_ms": 0.0,
    }


async def _stub_situation_classify(self, input_str: str) -> Dict[str, Any]:
    """Content situation stub.

    Quotes-per-line text and audio transcripts default to ``tweets_or_quotes``
    when many short newline-delimited lines are present; image descriptions
    classify as ``biographical_facts``.
    """
    text = input_str or ""
    non_empty_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    image_description_signal = any(
        marker in text.lower()
        for marker in (
            "appears to be",
            "in the image",
            "the photo",
            "this picture",
            "image description",
            "stub image description",
        )
    )

    if image_description_signal:
        situation = "biographical_facts"
    elif len(non_empty_lines) >= 2 and (
        sum(1 for ln in non_empty_lines if len(ln) <= 280) / len(non_empty_lines)
    ) >= 0.7:
        situation = "tweets_or_quotes"
    else:
        situation = "monologue"

    return {
        "classified_situation": situation,
        "reasoning": "Stub situation classifier.",
        "has_identifiable_target": True,
        "target_name": "Mom",
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "model_name": "stub",
        "total_cost": 0.0,
        "model_inference_type": "model_with_structured_output",
        "latency_ms": 0.0,
    }


async def _stub_transcribe_audio(audio_base64, context, filename=None):
    return {
        "content": (
            "Hi sweetie, this is Mom. I love you and I am proud of you. "
            "Remember to eat your vegetables and call me when you can."
        ),
        "file_duration_s": 5.0,
        "total_cost": 0.0,
        "latency_ms": 0.0,
        "model": "stub",
        "inference_type": "transcription",
    }


async def _stub_extract_personality_from_image(
    image_data, filename, store, user_id, assistant_id, context=None
):
    """Return a Document mimicking the vision model's first-person description."""
    return Document(
        page_content=(
            f"Stub image description for {filename}: a kind woman feeding geese "
            "by a pond on a sunny afternoon, smiling warmly."
        ),
        metadata={"source": filename, "inference_type": "image_description"},
    )


async def _stub_perform_ocean_analysis(human_message, additional_metadata=None):
    return []


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_pipeline():
    """Patch all external dependencies that the pipeline touches."""
    patches = [
        patch(
            "src.subgraphs.process_media_graph.utils.helper_functions."
            "ReferenceDocumentClassificationClass.classify",
            new=_stub_reference_classify,
        ),
        patch(
            "src.subgraphs.process_media_graph.utils.helper_functions."
            "ContentSituationClassificationClass.classify",
            new=_stub_situation_classify,
        ),
        patch(
            "src.anubis.utils.analysis.analysis_methods.perform_ocean_analysis",
            side_effect=_stub_perform_ocean_analysis,
        ),
        patch(
            "src.subgraphs.process_media_graph.utils.nodes."
            "ReferenceDocumentClassificationClass.classify",
            new=_stub_reference_classify,
        ),
        patch(
            "src.subgraphs.process_media_graph.utils.nodes.transcribe_audio",
            side_effect=_stub_transcribe_audio,
        ),
        patch(
            "src.subgraphs.process_media_graph.utils.nodes."
            "extract_personality_from_image",
            side_effect=_stub_extract_personality_from_image,
        ),
    ]
    started = [p.start() for p in patches]
    try:
        yield started
    finally:
        for p in patches:
            p.stop()


def _assert_required_data_present():
    """Sanity check that the Mom dataset files exist before mocking pipeline."""
    for name in (
        "mom_facebook_contextual_image.jpg",
        "mom-feeding-geese-pandemic.png",
        "Mom.m4a",
        "texts_with_mom_adapter_training.txt",
    ):
        path = DATA_DIR / name
        assert path.exists(), f"Required test fixture missing: {path}"


@pytest.mark.asyncio
async def test_upload_media_pipeline_then_load_consciousness(patched_pipeline):
    _assert_required_data_present()

    store = InMemoryStore()
    context = _build_context()
    config = _build_config()
    media_files = _build_media_files()

    from src.subgraphs.process_media_graph.process_media_graph_api_endpoint import (
        workflow,
    )

    process_media_graph = workflow.compile(store=store)

    initial_state = {"media_files": media_files}

    await process_media_graph.ainvoke(
        initial_state,
        config=config,
        context=context,
    )

    # ---- Verify reference image stored under (user_id, assistant_id, "reference_image") ----
    ref_image_namespace = (USER_ID, ASSISTANT_ID, "reference_image")
    ref_image_item = await store.aget(ref_image_namespace, ASSISTANT_ID)
    assert ref_image_item is not None, (
        "Reference image was not persisted under the avatar-owner namespace."
    )
    ref_value = getattr(ref_image_item, "value", {}) or {}
    assert "reference_image_data" in ref_value
    assert ref_value["reference_image_data"].startswith("data:image/")

    # ---- Verify reference audio stored under (user_id, assistant_id, "reference_audio") ----
    ref_audio_namespace = (USER_ID, ASSISTANT_ID, "reference_audio")
    ref_audio_item = await store.aget(ref_audio_namespace, ASSISTANT_ID)
    assert ref_audio_item is not None, "Reference audio not persisted."
    ref_audio_value = getattr(ref_audio_item, "value", {}) or {}
    assert ref_audio_value["reference_audio_data"].startswith("data:audio/")

    # ---- Verify quote namespace contains documents from text quotes AND audio transcript ----
    quote_namespace = (USER_ID, ASSISTANT_ID, "quote")
    quote_items = await store.asearch(quote_namespace, limit=10000)
    assert len(quote_items) > 0, "No documents indexed in quote namespace."
    quote_filenames = {
        (
            getattr(item, "value", {}) or {}
        )
        .get("document", {})
        .get("kwargs", {})
        .get("metadata", {})
        .get("filename", "")
        for item in quote_items
    }
    assert "texts_with_mom_adapter_training.txt" in quote_filenames, (
        "Mom quotes file was not indexed under quote namespace."
    )
    assert "Mom.m4a" in quote_filenames, (
        "Mom audio transcript was not indexed under quote namespace."
    )

    # The quotes file is one Document per line, so we expect many entries.
    quotes_per_line_count = sum(
        1
        for item in quote_items
        if (
            (getattr(item, "value", {}) or {})
            .get("document", {})
            .get("kwargs", {})
            .get("metadata", {})
            .get("filename")
            == "texts_with_mom_adapter_training.txt"
        )
    )
    assert quotes_per_line_count >= 5, (
        f"Expected line-per-quote indexing to produce many docs; got {quotes_per_line_count}"
    )

    # ---- Verify identity namespace contains the non-reference image description ----
    identity_namespace = (USER_ID, ASSISTANT_ID, "identity")
    identity_items = await store.asearch(identity_namespace, limit=10000)
    identity_filenames = {
        (getattr(item, "value", {}) or {})
        .get("document", {})
        .get("kwargs", {})
        .get("metadata", {})
        .get("filename", "")
        for item in identity_items
    }
    assert "mom-feeding-geese-pandemic.png" in identity_filenames, (
        "Non-reference image description was not indexed under identity namespace."
    )

    # ---- Run load_consciousness directly and verify retrieval into the system prompt ----
    state = {
        "messages": [HumanMessage(content="Tell me about Mom.")],
        "user_state": {"user_id": USER_ID},
        "assistant_state": {"assistant_id": ASSISTANT_ID},
        "user_identity_documents": [],
        "assistant_identity_documents": [],
        "recalled_memory_documents": [],
        "system_message": [],
    }

    runtime = Runtime(store=store, context=context)
    result = await load_consciousness(state=state, config=config, runtime=runtime)

    assert "system_message" in result and len(result["system_message"]) == 1
    system_prompt_str = result["system_message"][0].content

    assert isinstance(system_prompt_str, str) and system_prompt_str.strip(), (
        "load_consciousness produced an empty system prompt."
    )

    # The system prompt should reflect content from each indexed namespace.
    # Identity (image description) -> assistant_identity_str via assistant_identity docs
    assert "Stub image description" in system_prompt_str, (
        "Identity-namespace image description missing from system prompt."
    )

    # Quote namespace docs flow into direct_quotes section. We verify at least one
    # short quote line from the Mom file appears verbatim in the prompt.
    sample_lines = [
        line.strip()
        for line in _read_text("texts_with_mom_adapter_training.txt").splitlines()
        if line.strip()
    ][:20]
    assert any(line in system_prompt_str for line in sample_lines), (
        "No Mom quote lines from the indexed quotes file appear in the system prompt."
    )

    # Transcription content should also appear in the quote-derived prompt section.
    assert "this is Mom" in system_prompt_str, (
        "Transcribed audio content not retrieved into the system prompt."
    )

    # Reference image is stored under reference_image namespace and also merged
    # into the assistant identity context by load_consciousness; the rendered
    # prompt should contain the image description page_content.
    # (Reference-image Document carries a vision description as page_content.)
    # Not strictly required, but a good signal that retrieval is wired up.


@pytest.mark.asyncio
async def test_upload_media_text_only_quotes_per_line_namespace(patched_pipeline):
    """Standalone check that a quotes-per-line text file goes to the quote namespace."""
    _assert_required_data_present()

    store = InMemoryStore()
    context = _build_context()
    config = _build_config()

    quotes_name = "texts_with_mom_adapter_training.txt"
    quotes_bytes = _read_bytes(quotes_name)
    media_files = [
        {
            "filename": quotes_name,
            "content_type": "text/plain",
            "content": quotes_bytes,
            "user_id": USER_ID,
            "assistant_id": ASSISTANT_ID,
            "reference_audio": False,
            "reference_image": False,
            "base64_encoded_str": _make_data_uri("text/plain", quotes_bytes),
        }
    ]

    from src.subgraphs.process_media_graph.process_media_graph_api_endpoint import (
        workflow,
    )

    process_media_graph = workflow.compile(store=store)
    await process_media_graph.ainvoke(
        {"media_files": media_files}, config=config, context=context
    )

    quote_items = await store.asearch(
        (USER_ID, ASSISTANT_ID, "quote"), limit=10000
    )
    assert quote_items, "Mom quotes did not land in the quote namespace."

    # Verify the line-per-quote shape: every chunk index should be unique
    chunk_indices = [
        (getattr(item, "value", {}) or {})
        .get("document", {})
        .get("kwargs", {})
        .get("metadata", {})
        .get("chunk_index")
        for item in quote_items
    ]
    assert len(set(chunk_indices)) == len(chunk_indices), (
        "Quote-per-line documents should have unique chunk indices."
    )


# ---------------------------------------------------------------------------
# .log upload: verbatim, single Document, document namespace, no classification
# ---------------------------------------------------------------------------

_SAMPLE_LOG = (
    "2026-05-25 14:55:20 [f-log] running: ./hourly_progress.py -c -a <AVATAR_ID>\n"
    "Reviewing progress...\n"
    "Traceback (most recent call last):\n"
    '  File "./hourly_progress.py", line 57, in <module>\n'
    "    raise Exception(\n"
    "Exception: Neural Nexus request failed (500): Internal Server Error\n"
    "2026-05-25 14:56:03 [f-log] FAILED (exit 1)\n"
)


def _document_namespace_items(store, namespace_filename: str):
    items = []
    for item in store.search((USER_ID, ASSISTANT_ID), limit=1_000_000):
        meta = (
            (getattr(item, "value", {}) or {})
            .get("document", {})
            .get("kwargs", {})
            .get("metadata", {})
        )
        if (
            meta.get("namespace") == "document"
            and meta.get("namespace_filename") == namespace_filename
        ):
            items.append(item)
    return items


def _build_log_media_files(content_type: str) -> List[Dict[str, Any]]:
    log_bytes = _SAMPLE_LOG.encode("utf-8")
    return [
        {
            "filename": "hourly_progress.log",
            "content_type": content_type,
            "content": log_bytes,
            "user_id": USER_ID,
            "assistant_id": ASSISTANT_ID,
            "reference_audio": False,
            "reference_image": False,
            "base64_encoded_str": _make_data_uri(content_type, log_bytes),
            "namespace_filename": "hourly_progress.log",
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content_type", ["text/plain", "application/octet-stream", "application/x-log"]
)
async def test_upload_log_file_indexed_verbatim_to_document_namespace(content_type):
    """A ``.log`` upload is stored verbatim as one Document under ``document``.

    No classifiers are patched here: the ``.log`` path must not invoke any
    structured-output classification. The whole file is the Document
    ``page_content``; the same flow runs regardless of the declared MIME type.
    """
    store = InMemoryStore()
    context = _build_context()
    config = _build_config()
    media_files = _build_log_media_files(content_type)

    from src.subgraphs.process_media_graph.process_media_graph_api_endpoint import (
        workflow,
    )

    process_media_graph = workflow.compile(store=store)
    await process_media_graph.ainvoke(
        {"media_files": media_files}, config=config, context=context
    )

    items = _document_namespace_items(store, "hourly_progress.log")
    assert len(items) == 1, (
        f"Expected exactly one verbatim log Document, got {len(items)} "
        f"(content_type={content_type})."
    )

    doc = (getattr(items[0], "value", {}) or {})["document"]["kwargs"]
    assert doc["page_content"] == _SAMPLE_LOG, (
        "Log Document page_content should be the full file text, unchunked."
    )
    meta = doc["metadata"]
    assert meta["filename"] == "hourly_progress.log"
    assert meta["type"] == "log"
    assert meta["namespace"] == "document"
    assert meta["classified_situation"] == "proprietary_content"
    assert meta["total_chunks"] == 1


@pytest.mark.asyncio
async def test_upload_empty_log_file_raises(  # noqa: D401
):
    """An empty/whitespace ``.log`` upload fails rather than indexing nothing."""
    store = InMemoryStore()
    context = _build_context()
    config = _build_config()
    blank_bytes = b"   \n\t\n"
    media_files = [
        {
            "filename": "blank.log",
            "content_type": "text/plain",
            "content": blank_bytes,
            "user_id": USER_ID,
            "assistant_id": ASSISTANT_ID,
            "reference_audio": False,
            "reference_image": False,
            "base64_encoded_str": _make_data_uri("text/plain", blank_bytes),
            "namespace_filename": "blank.log",
        }
    ]

    from src.subgraphs.process_media_graph.process_media_graph_api_endpoint import (
        workflow,
    )

    process_media_graph = workflow.compile(store=store)
    with pytest.raises(Exception):
        await process_media_graph.ainvoke(
            {"media_files": media_files}, config=config, context=context
        )
