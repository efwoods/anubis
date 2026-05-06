import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import NAMESPACE_URL, uuid4, uuid5

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from src.anubis.utils.tokenizer import count_tokens
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.anubis.utils.analysis.analysis_methods import perform_ocean_analysis
from src.anubis.utils.classes.ContentSituationClassificationClass import (
    ContentSituationClassificationClass,
)
from src.anubis.utils.classes.FactRewriterClass import FactRewriterClass
from src.anubis.utils.classes.FirstPersonRewriterClass import (
    FirstPersonRewriterClass,
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
        length_function=count_tokens,
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
                "synthetic": False,
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


async def _build_identity_documents_from_facts(
    facts: List[dict],
    *,
    user_id: str,
    assistant_id: str,
    media_item: Dict[str, Any],
    original_statement: str = "",
    target_name: Optional[str] = None,
) -> List[Document]:
    """Run FirstPersonRewriter over rewritten facts and emit identity Documents.

    Each output Document is one first-person identity statement with full
    provenance preserved in metadata: original_statement (the verbatim
    source text the fact rewriter saw — appended in code by
    :class:`FactRewriterClass`), rewritten_statement (lawsuit-safer
    third-person phrasing produced by the model), first_person_statement
    (final identity content produced by :class:`FirstPersonRewriterClass`),
    source (filename or URL the fact came from), target_name (caller hint
    appended in code), and ``synthetic=True`` because an LLM produced the
    wording.

    The identity-namespace embed field (``document.kwargs.page_content``) is
    the first-person statement, so retrieval at chat time finds these as
    primary self-knowledge of the avatar.
    """
    if not facts:
        return []

    rewritten_inputs: List[str] = [
        (f.get("rewritten_statement") or "").strip() for f in facts
    ]
    rewriter = FirstPersonRewriterClass()
    rewriter_response = await rewriter.rewrite(rewritten_inputs)
    statements = rewriter_response.get("statements") or []

    item_metadata = media_item.get("metadata", {}) or {}
    filename = item_metadata.get("filename", "")
    filename_uuid5 = str(uuid5(NAMESPACE_URL, filename or "unknown"))
    source = item_metadata.get("source") or filename or "user_upload"
    current_timestamp = datetime.now(tz=timezone.utc).isoformat()

    documents: List[Document] = []
    for idx, fact in enumerate(facts):
        if idx >= len(statements):
            break
        stmt = statements[idx] or {}
        first_person = (stmt.get("first_person_statement") or "").strip()
        if not first_person:
            continue
        doc = Document(
            page_content=first_person,
            metadata={
                "user_id": user_id,
                "assistant_id": assistant_id,
                "created_at": current_timestamp,
                "processing_task_id": str(uuid4()),
                "source": source,
                "type": "text",
                "filename": filename,
                "filename_uuid5": filename_uuid5,
                "document_id": str(uuid4()),
                "namespace": "identity",
                "vectorstore_acceptable": True,
                "adapter_acceptable": False,
                "analysis_acceptable": False,
                "classified_situation": "biographical_facts",
                "synthetic": True,
                "original_statement": original_statement,
                "rewritten_statement": fact.get("rewritten_statement", ""),
                "target_name": target_name,
            },
        )
        documents.append(doc)
    return documents


async def _build_biographical_identity_documents(
    text_content: str,
    *,
    user_id: str,
    assistant_id: str,
    media_item: Dict[str, Any],
    target_name: Optional[str] = None,
) -> List[Document]:
    """Run FactRewriter then FirstPersonRewriter and emit identity Documents."""
    extractor = FactRewriterClass()
    fact_response = await extractor.extract(
        input_str=_classification_slice(text_content),
        target_name=target_name,
    )
    facts = fact_response.get("facts") or []
    return await _build_identity_documents_from_facts(
        facts,
        user_id=user_id,
        assistant_id=assistant_id,
        media_item=media_item,
        original_statement=fact_response.get("original_statement", ""),
        target_name=fact_response.get("target_name") or target_name,
    )


def _build_target_quote_documents_from_dialogue(
    dialogue_segments: List[Dict[str, Any]],
    *,
    user_id: str,
    assistant_id: str,
    media_item: Dict[str, Any],
    target_name: Optional[str],
) -> List[Document]:
    """One Document per target turn in a diarized dialogue (verbatim)."""
    item_metadata = media_item.get("metadata", {}) or {}
    filename = item_metadata.get("filename", "")
    filename_uuid5 = str(uuid5(NAMESPACE_URL, filename or "unknown"))
    source = item_metadata.get("source") or filename or "user_upload"
    creator_id = item_metadata.get("creator_id") or user_id
    current_timestamp = datetime.now(tz=timezone.utc).isoformat()

    documents: List[Document] = []
    target_segments = [
        seg for seg in dialogue_segments if seg.get("is_target") and (seg.get("text") or "").strip()
    ]
    total = len(target_segments)
    for idx, seg in enumerate(target_segments):
        text = (seg.get("text") or "").strip()
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "creator_id": creator_id,
                    "created_at": current_timestamp,
                    "processing_task_id": str(uuid4()),
                    "source": source,
                    "type": "text",
                    "chunk_index": idx,
                    "total_chunks": total,
                    "filename": filename,
                    "filename_uuid5": filename_uuid5,
                    "document_id": str(uuid4()),
                    "namespace": "quote",
                    "vectorstore_acceptable": True,
                    "adapter_acceptable": True,
                    "analysis_acceptable": True,
                    "classified_situation": "dialogue",
                    "synthetic": False,
                    "speaker": seg.get("speaker"),
                    "is_target": True,
                    "target_name": target_name,
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                },
            )
        )
    return documents


def _build_adapter_dialogue_document(
    dialogue_segments: List[Dict[str, Any]],
    *,
    user_id: str,
    assistant_id: str,
    media_item: Dict[str, Any],
    target_name: Optional[str],
) -> Optional[Document]:
    """One adapter-namespace Document holding the role-converted conversation."""
    if not dialogue_segments:
        return None

    role_messages: List[Dict[str, str]] = []
    for seg in dialogue_segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        role = "assistant" if seg.get("is_target") else "user"
        if role_messages and role_messages[-1]["role"] == role:
            role_messages[-1]["content"] = role_messages[-1]["content"] + "\n" + text
        else:
            role_messages.append({"role": role, "content": text})

    if not any(m["role"] == "assistant" for m in role_messages):
        return None

    item_metadata = media_item.get("metadata", {}) or {}
    filename = item_metadata.get("filename", "")
    filename_uuid5 = str(uuid5(NAMESPACE_URL, filename or "unknown"))
    source = item_metadata.get("source") or filename or "user_upload"
    creator_id = item_metadata.get("creator_id") or user_id
    current_timestamp = datetime.now(tz=timezone.utc).isoformat()

    speakers_seen = sorted({seg.get("speaker") for seg in dialogue_segments if seg.get("speaker")})

    return Document(
        page_content=json.dumps({"messages": role_messages}, ensure_ascii=False),
        metadata={
            "user_id": user_id,
            "assistant_id": assistant_id,
            "creator_id": creator_id,
            "created_at": current_timestamp,
            "processing_task_id": str(uuid4()),
            "source": source,
            "type": "adapter_conversation",
            "filename": filename,
            "filename_uuid5": filename_uuid5,
            "document_id": str(uuid4()),
            "namespace": "adapter",
            "vectorstore_acceptable": False,
            "adapter_acceptable": True,
            "analysis_acceptable": False,
            "classified_situation": "dialogue",
            "synthetic": False,
            "target_name": target_name,
            "speakers": speakers_seen,
            "messages": role_messages,
        },
    )


async def process_dialogue_json_to_documents(
    dialogue_payload: Dict[str, Any],
    *,
    user_id: str,
    assistant_id: str,
    media_item: Dict[str, Any],
) -> List[Document]:
    """Convert a diarized dialogue JSON into Documents across three namespaces.

    Outputs:
      * Target quote Documents (one per target turn, verbatim) under
        ``quote`` namespace with timestamps.
      * One adapter conversation Document under ``adapter`` namespace
        containing the full role-converted exchange (target → assistant,
        others → user, consecutive same-role turns concatenated).
      * Identity Documents under ``identity`` namespace, produced by scanning
        the entire transcript for biographical facts about the target via
        :class:`FactRewriterClass` + :class:`FirstPersonRewriterClass`. If
        nothing is said about the target, no identity Documents are emitted.

    ``dialogue_payload`` shape:
        {
            "segments": [
                {"speaker": "<label>", "is_target": bool, "text": str,
                 "start": float, "end": float},
                ...
            ],
            "target_name": Optional[str],
            "speakers": Optional[List[dict]],
        }
    """
    segments: List[Dict[str, Any]] = list(
        dialogue_payload.get("segments") or []
    )
    target_name = dialogue_payload.get("target_name")

    documents: List[Document] = []

    # Target turns under ``quote`` (verbatim).
    documents.extend(
        _build_target_quote_documents_from_dialogue(
            segments,
            user_id=user_id,
            assistant_id=assistant_id,
            media_item=media_item,
            target_name=target_name,
        )
    )

    # Full role-converted dialogue under ``adapter``.
    adapter_doc = _build_adapter_dialogue_document(
        segments,
        user_id=user_id,
        assistant_id=assistant_id,
        media_item=media_item,
        target_name=target_name,
    )
    if adapter_doc is not None:
        documents.append(adapter_doc)

    # Scan whole transcript for biographical facts about the target.
    transcript_text = "\n".join(
        f"{seg.get('speaker') or 'unknown'}: {(seg.get('text') or '').strip()}"
        for seg in segments
        if (seg.get("text") or "").strip()
    )
    if transcript_text.strip():
        try:
            identity_docs = await _build_biographical_identity_documents(
                text_content=transcript_text,
                user_id=user_id,
                assistant_id=assistant_id,
                media_item=media_item,
                target_name=target_name,
            )
            documents.extend(identity_docs)
        except Exception as exc:
            logger.warning(
                "Biographical fact extraction over dialogue transcript failed: %s",
                exc,
            )

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
            - ``biographical_facts`` -> FactRewriter then FirstPersonRewriter,
              one identity Document per first-person statement with full
              provenance metadata (original_statement, extracted_fact,
              rewritten_statement, first_person_statement, synthetic=True).
            - ``tweets_or_quotes`` -> one verbatim Document per line under
              ``quote``.
            - ``monologue`` / ``presentation`` -> verbatim chunked Document(s)
              under ``quote``.
            - ``dialogue`` -> caller is expected to supply diarized JSON via
              :func:`process_dialogue_json_to_documents`. The text branch here
              falls back to chunking under ``quote`` (best-effort).
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

    # Biographical facts: extract atomic facts, lawsuit-safer rewrite, then
    # convert to first person. Each first-person statement becomes one
    # identity Document with full provenance preserved in metadata.
    if classified_situation == "biographical_facts" and namespace_hint is None:
        try:
            documents = await _build_biographical_identity_documents(
                text_content=text_content,
                user_id=user_id,
                assistant_id=assistant_id,
                media_item=media_item,
                target_name=target_name,
            )
            for document in documents:
                document.metadata.update(
                    {
                        "classification_reasoning": situation_reasoning,
                        "reference_classification_reasoning": reference_reasoning,
                        "is_menu_or_religious_text": False,
                    }
                )
            return documents
        except Exception as exc:
            logger.warning(
                "biographical_facts rewriter pipeline failed (%s); "
                "falling back to chunked identity Documents",
                exc,
            )
            # Fall through to chunked path so we still capture content.

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
                    "synthetic": False
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
