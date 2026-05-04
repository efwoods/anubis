import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import NAMESPACE_URL, uuid4, uuid5

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.anubis.utils.analysis.analysis_methods import perform_ocean_analysis
from src.anubis.utils.classes.ContentSituationClassificationClass import (
    ContentSituationClassificationClass,
)
from src.anubis.utils.classes.ReferenceDocumentClassificationClass import (
    ReferenceDocumentClassificationClass,
)

logger = logging.getLogger(__name__)


CLASSIFICATION_INPUT_CHAR_LIMIT = 5000


def _coerce_classification_input_to_string(content: Any) -> str:
    """Normalize any text/json content to a string for classifier input."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, bytes):
        try:
            return content.decode("utf-8", errors="replace")
        except Exception:
            return ""
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except Exception:
        return str(content)


def _classification_slice(text: str, limit: int = CLASSIFICATION_INPUT_CHAR_LIMIT) -> str:
    return text[:limit] if len(text) > limit else text


def normal_chunking(
    text_content: str, metadata: dict, chunk_size: int = 1024, chunk_overlap: int = 150
):
    """Token-aware recursive chunking for downstream Document creation."""
    separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=count_tokens_approximately,
        is_separator_regex=False,
    )

    if isinstance(text_content, list):
        texts = text_content
    elif isinstance(text_content, str):
        texts = [text_content]
    else:
        texts = [str(text_content)]

    docs = text_splitter.create_documents(texts=texts, metadatas=[metadata])
    return docs


async def split_text_into_chunks(
    text_splitter: RecursiveCharacterTextSplitter,
    text_content: str,
    source_metadata: dict,
    source: str,
    user_id: str,
    assistant_id: str,
    classification_metadata: dict,
    idx: int,
    document_id: str,
    filename: str,
    filename_uuid5: str,
):
    """Helper for chunking when an external splitter is supplied."""

    text_chunks = text_splitter.split_text(text_content)
    documents: List[Document] = []
    current_timestamp = datetime.now(tz=timezone.utc).isoformat()

    for chunk in text_chunks:
        doc = Document(
            page_content=chunk,
            metadata={
                "user_id": user_id,
                "assistant_id": assistant_id,
                "created_at": current_timestamp,
                "processing_task_id": str(uuid4()),
                "source": source,
                "type": "text",
                "chunk_index": idx,
                "total_chunks": len(text_chunks),
                "filename": filename,
                "filename_uuid5": str(uuid5(NAMESPACE_URL, filename)),
                "document_id": str(uuid4()),
            },
        )
        idx += 1
        doc.metadata.update(source_metadata)
        if classification_metadata is not None:
            doc.metadata.update(classification_metadata)
        documents.append(doc)

    return idx, documents


async def process_text_media_item_target_for_vectorstore(
    media_item: Dict[str, Any],
    user_id: str,
    assistant_id: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 150,
    separators: Optional[List[str]] = None,
    classification_metadata: Optional[dict] = None,
    use_semantic_chunks: bool = False,
    namespace: str = "document",
) -> List[Document]:
    """Chunk text/string content into Documents tagged with the desired namespace.

    The ``namespace`` arg sets ``metadata['namespace']`` on every produced chunk so the
    indexer can place it under ``(creator_id, assistant_id, namespace, filename)``.
    """

    logger.info("process_text_media_item_target_for_vectorstore entrypoint")
    try:
        text_content = _coerce_classification_input_to_string(
            media_item.get("content", "")
        )
        filename = media_item.get("metadata", {}).get("filename", "")
        filename_uuid5 = str(uuid5(NAMESPACE_URL, filename or "unknown"))

        source_metadata = media_item.get("metadata", {}) or {}
        source = source_metadata.get("source", "user_upload")

        if not text_content.strip():
            logger.warning("Empty text content in media_item")
            return []

        _ = use_semantic_chunks
        all_documents = normal_chunking(
            text_content=text_content,
            metadata={},
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        current_timestamp = datetime.now(tz=timezone.utc).isoformat()
        total_chunks = len(all_documents)
        idx = 0

        for document in all_documents:
            document.metadata.update(
                {
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "created_at": current_timestamp,
                    "processing_task_id": str(uuid4()),
                    "source": source,
                    "type": "text",
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                    "filename": filename,
                    "filename_uuid5": filename_uuid5,
                    "document_id": str(uuid4()),
                    "namespace": namespace,
                }
            )
            document.metadata.update(source_metadata)
            if classification_metadata is not None:
                document.metadata.update(classification_metadata)
            idx += 1

        for document in all_documents:
            document.metadata.update({"total_chunks": idx})

        return all_documents

    except Exception as e:
        logger.exception(
            "Error in text chunking during process media item target for vector store: %s",
            e,
        )
        raise


def _is_quotes_per_line_text(text: str) -> bool:
    """Detect a file whose non-empty lines are each a complete standalone quote.

    Heuristic (any one fails -> not quotes-per-line):
        * >= 2 non-empty lines.
        * No line is a continuation: lines do not start with whitespace.
        * At least 70% of non-empty lines either fit a tweet (<= 280 chars stripped)
          OR end with sentence-terminating punctuation (. ? ! " ' ) ]).
    """
    raw_lines = text.splitlines()
    non_empty = [l for l in raw_lines if l.strip()]
    if len(non_empty) < 2:
        return False
    if any(l.startswith((" ", "\t")) for l in non_empty):
        return False
    terminators = (".", "?", "!", '"', "\u201d", "'", "\u2019", ")", "]")
    discrete = 0
    for line in non_empty:
        s = line.strip()
        if len(s) <= 280 or s.endswith(terminators):
            discrete += 1
    return (discrete / len(non_empty)) >= 0.7


def _build_quote_documents_per_line(
    text_content: str,
    user_id: str,
    assistant_id: str,
    media_item: Dict[str, Any],
    classification_metadata: Dict[str, Any],
) -> List[Document]:
    """One Document per non-empty line under the ``quote`` namespace."""
    lines = text_content.splitlines()
    filename = media_item.get("metadata", {}).get("filename", "")
    filename_uuid5 = str(uuid5(NAMESPACE_URL, filename or "unknown"))
    source_metadata = media_item.get("metadata", {}) or {}
    source = source_metadata.get("source", "user_upload")

    documents: List[Document] = []
    idx = 0
    current_timestamp = datetime.now(tz=timezone.utc).isoformat()

    for line in lines:
        text = line.strip()
        if not text:
            continue

        doc = Document(
            page_content=text,
            metadata={
                "user_id": user_id,
                "assistant_id": assistant_id,
                "created_at": current_timestamp,
                "processing_task_id": str(uuid4()),
                "source": source,
                "type": "text",
                "chunk_index": idx,
                "filename": filename,
                "filename_uuid5": filename_uuid5,
                "document_id": str(uuid4()),
                "namespace": "quote",
                "vectorstore_acceptable": True,
                "adapter_acceptable": True,
            },
        )
        doc.metadata.update(source_metadata)
        if classification_metadata:
            doc.metadata.update(classification_metadata)
        documents.append(doc)
        idx += 1

    for d in documents:
        d.metadata.update({"total_chunks": idx})

    return documents


async def process_text_to_document(
    metadata,
    user_id,
    assistant_id,
    media_item,
    namespace_hint: Optional[str] = None,
) -> List[Document]:
    """Reference-document gate, then content-situation classifier, then route to chunkers.

    Routing rules:
        * Reference (menu / holy text) -> ``document`` namespace, full chunked text.
        * Biographical/conversational -> ``ContentSituationClassification``:
            - ``tweets_or_quotes`` -> one Document per line under ``quote``.
            - ``monologue`` / ``presentation`` / ``dialogue`` -> chunk under ``quote``.
            - ``biographical_facts`` -> chunk under ``identity``.
        * ``namespace_hint`` overrides reference/biographical when supplied
          (e.g. predetermined identity content from images).
    """
    logger.info("process_text_to_document entrypoint")

    text_content = _coerce_classification_input_to_string(
        media_item.get("content", "")
    )
    classification_input = _classification_slice(text_content)

    if not text_content.strip():
        logger.warning("Empty text content in media_item; returning no documents")
        return []

    reference_classifier = ReferenceDocumentClassificationClass()
    reference_response = await reference_classifier.classify(classification_input)
    is_menu_or_religious_text = bool(
        reference_response.get("is_menu_or_religious_text", False)
    )
    reference_reasoning = reference_response.get("reasoning", "")

    if is_menu_or_religious_text and namespace_hint is None:
        logger.info("Reference document detected (menu/holy text) -> document namespace")
        classification_metadata = {
            "classified_situation": "proprietary_content",
            "reference_classification_reasoning": reference_reasoning,
            "is_menu_or_religious_text": True,
        }
        documents = await process_text_media_item_target_for_vectorstore(
            media_item=media_item,
            user_id=user_id,
            assistant_id=assistant_id,
            classification_metadata=classification_metadata,
            use_semantic_chunks=False,
            namespace=namespace_hint or "document",
        )
        for document in documents:
            document.metadata.update({"vectorstore_acceptable": True})
        return documents

    situation_classifier = ContentSituationClassificationClass()
    situation_response = await situation_classifier.classify(classification_input)
    classified_situation = situation_response.get("classified_situation", "")
    situation_reasoning = situation_response.get("reasoning", "")
    target_name = situation_response.get("target_name")

    classification_metadata = {
        "classified_situation": classified_situation,
        "classification_reasoning": situation_reasoning,
        "reference_classification_reasoning": reference_reasoning,
        "is_menu_or_religious_text": False,
        "target_name": target_name,
    }

    if namespace_hint is not None:
        target_namespace = namespace_hint
    elif classified_situation == "biographical_facts":
        target_namespace = "identity"
    elif classified_situation in ("tweets_or_quotes", "monologue", "presentation", "dialogue"):
        target_namespace = "quote"
    else:
        target_namespace = "quote"

    logger.info(
        "ContentSituationClassification: %s -> namespace=%s",
        classified_situation,
        target_namespace,
    )

    item_metadata = media_item.get("metadata", {}) or {}
    explicit_quotes_per_line = bool(item_metadata.get("quotes_per_line"))
    quotes_per_line_detected = (
        classified_situation == "tweets_or_quotes"
        and _is_quotes_per_line_text(text_content)
    )

    if (explicit_quotes_per_line or quotes_per_line_detected) and target_namespace == "quote":
        logger.info(
            "Detected quotes-per-line text (explicit=%s, heuristic=%s) -> 1 Document/line",
            explicit_quotes_per_line,
            quotes_per_line_detected,
        )
        classification_metadata["quotes_per_line"] = True
        documents = _build_quote_documents_per_line(
            text_content=text_content,
            user_id=user_id,
            assistant_id=assistant_id,
            media_item=media_item,
            classification_metadata=classification_metadata,
        )
    else:
        documents = await process_text_media_item_target_for_vectorstore(
            media_item=media_item,
            user_id=user_id,
            assistant_id=assistant_id,
            classification_metadata=classification_metadata,
            use_semantic_chunks=False,
            namespace=target_namespace,
        )
        for document in documents:
            document.metadata.update(
                {
                    "vectorstore_acceptable": True,
                    "adapter_acceptable": classified_situation
                    in ("monologue", "presentation", "tweets_or_quotes", "dialogue"),
                }
            )

    if classified_situation in ("monologue", "presentation", "tweets_or_quotes"):
        try:
            analysis_metadata = {
                "user_id": user_id,
                "assistant_id": assistant_id,
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
                "source": (media_item.get("metadata", {}) or {}).get(
                    "source", "user_upload"
                ),
                "type": "text",
                "filename": (media_item.get("metadata", {}) or {}).get("filename", ""),
                "namespace": "identity",
                "analysis_acceptable": True,
                "vectorstore_acceptable": True,
            }
            analysis_documents = await perform_ocean_analysis(
                human_message=HumanMessage(content=text_content),
                additional_metadata=analysis_metadata,
            )
            documents.extend(analysis_documents or [])
        except Exception as e:
            logger.warning("OCEAN analysis failed; continuing without it: %s", e)

    return documents
