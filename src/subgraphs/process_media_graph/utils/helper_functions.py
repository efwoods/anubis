import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import NAMESPACE_URL, uuid4, uuid5

from langchain_core.documents import Document
from src.anubis.utils.tokenizer import count_tokens
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.anubis.utils.classes.ContentSituationClassificationClass import (
    ContentSituationClassificationClass,
)
from src.anubis.utils.classes.FactRewriterClass import FactRewriterClass
from src.anubis.utils.classes.FirstPersonRewriterClass import (
    FirstPersonRewriterClass,
)
from src.anubis.utils.context import GlobalContext
from src.subgraphs.process_media_graph.utils.fragment_filter import (
    filter_fragment_documents,
)
from src.anubis.utils.classes.ReferenceDocumentClassificationClass import (
    ReferenceDocumentClassificationClass,
)
from langgraph.store.base import BaseStore

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


async def drop_fragment_documents(documents: List[Document]) -> List[Document]:
    """Filter out fragment/boilerplate Documents using context-configured thresholds.

    Reads ``min_useful_tokens`` and ``fragment_llm_fallback`` from ``GlobalContext``
    so the floor and LLM-judge toggle are env-tunable. Never raises — on any error
    the input list is returned unfiltered so ingestion is not blocked.
    """
    if not documents:
        return documents
    try:
        context = GlobalContext()
        min_tokens = int(getattr(context, "min_useful_tokens", 4) or 4)
        use_llm_fallback = str(
            getattr(context, "fragment_llm_fallback", "TRUE") or "TRUE"
        ).upper() == "TRUE"
        return await filter_fragment_documents(
            documents, min_tokens=min_tokens, use_llm_fallback=use_llm_fallback
        )
    except Exception as exc:  # pragma: no cover - filtering must not break ingestion
        logger.warning("drop_fragment_documents failed (%s); keeping all docs", exc)
        return documents


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
    else:
        texts = [text_content]
    
    texts = list(set([text.strip() for text in texts if text.strip()]))

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
    classification_metadata: Optional[dict] = None,
    use_semantic_chunks: bool = False,
    namespace: str = "document",
) -> List[Document]:
    """Chunk text/string content into Documents tagged with the desired namespace.

    The ``namespace`` arg sets ``metadata['namespace']`` on every produced chunk so the
    indexer can place it under ``(user_id, assistant_id, namespace, filename)``.
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

        # Drop fragment chunks (page numbers, headers/footers, nav text) before
        # they are stamped and indexed.
        all_documents = await drop_fragment_documents(all_documents)
        if not all_documents:
            logger.warning(
                "All chunks filtered as fragments for %s; no documents emitted", filename
            )
            return []

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

    # Unique non-empty lines
    lines = list(set([line for line in lines if line.strip()]))
    filename = media_item.get("metadata", {}).get("filename", "")
    filename_uuid5 = str(uuid5(NAMESPACE_URL, filename or "unknown"))
    source_metadata = media_item.get("metadata", {}) or {}
    source = source_metadata.get("source", "user_upload")

    documents: List[Document] = []
    idx = 0
    current_timestamp = datetime.now(tz=timezone.utc).isoformat()

    for line in lines:
        doc = Document(
            page_content=line,
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
    concise_context_summary: str = "",
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

    Each Document is both ``vectorstore_acceptable`` (stored under ``identity``)
    and ``analysis_acceptable`` (queued for the analyzer fan-out), so a
    biographical fact is preserved verbatim as self-knowledge while also driving
    the latent-feature/OCEAN/emotional-trigger analysis — storage and analysis
    are maintained side by side rather than being mutually exclusive.
    """
    if not facts:
        return []

    rewritten_inputs: List[str] = [
        (f.get("rewritten_statement") or "").strip() for f in facts
    ]
    rewriter = FirstPersonRewriterClass()
    rewriter_response = await rewriter.rewrite(rewritten_inputs, concise_context_summary=concise_context_summary)
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
        if first_person != "":
            first_person = "<FACT_CONTEXT_AND_FACT>" + " <FACT_CONTEXT>" + (concise_context_summary.strip()) + "</FACT_CONTEXT>" + "<FACT>" + first_person + "</FACT>" + "</FACT_CONTEXT_AND_FACT>"

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
                # Biographical facts are stored under ``identity`` AND fed to the
                # analysis pipeline: the first-person fact is exactly the input
                # the latent-feature/OCEAN/emotional-trigger analyzers consume to
                # derive beliefs, values, history, relationships, etc. Storage and
                # analysis are maintained side by side (the doc lands on both the
                # vectorstore-index buffer and the analysis queue; analyzer
                # outputs are separate ``analysis``-namespace Documents).
                "analysis_acceptable": True,
                "classified_situation": "biographical_facts",
                "synthetic": True,
                "original_statement": original_statement,
                "rewritten_statement": fact.get("rewritten_statement", ""),
                "concise_context_summary": concise_context_summary,
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
        concise_context_summary = fact_response.get("concise_context_summary", "")
    )


def coalesce_segments_by_speaker(
    segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge consecutive same-speaker segments into single alternating turns.

    Walks ``segments`` in order; while the next segment shares the current
    turn's ``speaker`` label its ``text`` is concatenated into the current
    turn, ``end`` is extended to the later segment's end, and ``is_target`` is
    OR-ed. A change of speaker label closes the current turn and opens a new
    one. The result is an alternating-speaker list of turns, each carrying
    ``speaker``, ``text``, ``start``, ``end`` and ``is_target``.

    Idempotent: re-running over an already-coalesced list yields an equivalent
    list, so callers may safely coalesce defensively.
    """
    turns: List[Dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker = str(seg.get("speaker") or "unknown")
        if turns and turns[-1]["speaker"] == speaker:
            prev = turns[-1]
            prev["text"] = f"{prev['text']} {text}".strip()
            prev["end"] = seg.get("end", prev.get("end"))
            prev["is_target"] = bool(prev.get("is_target")) or bool(
                seg.get("is_target")
            )
        else:
            turns.append(
                {
                    "speaker": speaker,
                    "text": text,
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "is_target": bool(seg.get("is_target")),
                }
            )
    return turns


async def _reconcile_speakers(
    turns: List[Dict[str, Any]], target_label: str
) -> List[Dict[str, Any]]:
    """Correct diarizer over/under-splitting via the SpeakerReconciliation judge.

    ``gpt-4o-transcribe-diarize`` has no parameter to set the number of speakers
    and commonly splits one person across several labels. This maps each raw label
    to a canonical speaker (target relabeled to ``target_label``) and re-coalesces.
    Never raises — on failure the original diarizer turns are returned unchanged.
    """
    try:
        from src.anubis.utils.classes.SpeakerReconciliationClass import (
            SpeakerReconciliationClass,
        )

        result = await SpeakerReconciliationClass().reconcile(
            turns, target_speaker_label=target_label
        )
        label_map = {
            entry.get("raw_speaker"): entry
            for entry in (result.get("label_map") or [])
            if isinstance(entry, dict) and entry.get("raw_speaker") is not None
        }
        if not label_map:
            return turns
        for turn in turns:
            entry = label_map.get(turn.get("speaker"))
            if not entry:
                continue
            turn["speaker"] = entry.get("canonical_speaker") or turn["speaker"]
            turn["is_target"] = bool(entry.get("is_target")) or bool(
                turn.get("is_target")
            )
        return coalesce_segments_by_speaker(turns)
    except Exception as exc:  # pragma: no cover - reconciliation must not break ingestion
        logger.warning(
            "speaker reconciliation failed (%s); using raw diarizer labels", exc
        )
        return turns


async def build_golden_transcript(
    diar_response: Dict[str, Any],
    *,
    target_speaker_label: Optional[str],
    reference_audio: bool = False,
    reconcile: bool = True,
) -> Dict[str, Any]:
    """Normalize a raw diarizer response into the canonical golden-transcript format.

    Produces the same shape as ``*_golden_dataset_hand_annotated.json``:
    ``{source, filename, model, duration, text, segments:[{speaker, start, end,
    text, is_target}]}``. Consecutive same-speaker segments are coalesced, the
    target's turns are relabeled to ``target_speaker_label`` (``avatar`` by
    default), other speakers keep their (reconciled) labels, and the top-level
    ``text`` is recomputed from the coalesced turns. This is the single canonical
    intermediate that both the on-disk artifact and downstream document creation
    consume.
    """
    target_label = target_speaker_label or "avatar"
    segments_raw = (diar_response or {}).get("segments") or []

    normalized: List[Dict[str, Any]] = []
    for seg in segments_raw:
        if not isinstance(seg, dict):
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker_label = seg.get("speaker") or "unknown"
        is_target = (
            target_speaker_label is not None
            and target_label.lower() in str(speaker_label).lower()
        ) or bool(reference_audio)
        normalized.append(
            {
                "speaker": str(speaker_label),
                "text": text,
                "start": seg.get("start"),
                "end": seg.get("end"),
                "is_target": bool(is_target),
            }
        )

    turns = coalesce_segments_by_speaker(normalized)

    # Speaker reconciliation only helps when the diarizer reported >1 label.
    if reconcile and len({t["speaker"] for t in turns}) > 1:
        turns = await _reconcile_speakers(turns, target_label)

    # Canonical relabel: every target turn becomes ``target_label`` (``avatar``),
    # then re-coalesce so adjacent target turns merge under the unified label.
    for turn in turns:
        if turn.get("is_target"):
            turn["speaker"] = target_label
    turns = coalesce_segments_by_speaker(turns)

    golden = {
        "source": (diar_response or {}).get("source")
        or (diar_response or {}).get("filename"),
        "filename": (diar_response or {}).get("filename"),
        "model": (diar_response or {}).get("model"),
        "duration": (diar_response or {}).get("duration"),
        "text": "\n".join(t["text"] for t in turns),
        "segments": [
            {
                "speaker": t["speaker"],
                "start": t.get("start"),
                "end": t.get("end"),
                "text": t["text"],
                "is_target": bool(t.get("is_target")),
            }
            for t in turns
        ],
    }
    return golden


def _build_target_quote_documents_from_dialogue(
    dialogue_segments: List[Dict[str, Any]],
    *,
    user_id: str,
    assistant_id: str,
    media_item: Dict[str, Any],
    target_name: Optional[str],
) -> List[Document]:
    """One Document per target turn in a diarized dialogue (verbatim).

    Target turns are pre-tagged ``classified_situation="tweets_or_quotes"`` and
    are NOT re-run through the content classifier. Each carries an
    ``adapter_prompt`` metadata field holding the text of the immediately
    preceding non-target turn (the "prerequisite/prompting" statement) when one
    exists, else ``None`` so the adapter builder falls back to a synthetic
    prompt. ``dialogue_segments`` must be the FULL ordered turn list (not just
    the target turns) so the predecessor can be resolved.
    """
    item_metadata = media_item.get("metadata", {}) or {}
    filename = item_metadata.get("filename", "")
    filename_uuid5 = str(uuid5(NAMESPACE_URL, filename or "unknown"))
    source = item_metadata.get("source") or filename or "user_upload"
    current_timestamp = datetime.now(tz=timezone.utc).isoformat()

    target_indices = [
        i
        for i, seg in enumerate(dialogue_segments)
        if seg.get("is_target") and (seg.get("text") or "").strip()
    ]
    total = len(target_indices)

    documents: List[Document] = []
    for chunk_idx, seg_idx in enumerate(target_indices):
        seg = dialogue_segments[seg_idx]
        text = (seg.get("text") or "").strip()
        adapter_prompt: Optional[str] = None
        if seg_idx > 0:
            prev = dialogue_segments[seg_idx - 1]
            if not prev.get("is_target"):
                adapter_prompt = (prev.get("text") or "").strip() or None
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "created_at": current_timestamp,
                    "processing_task_id": str(uuid4()),
                    "source": source,
                    "type": "text",
                    "chunk_index": chunk_idx,
                    "total_chunks": total,
                    "filename": filename,
                    "filename_uuid5": filename_uuid5,
                    "document_id": str(uuid4()),
                    "namespace": "quote",
                    "vectorstore_acceptable": True,
                    "adapter_acceptable": True,
                    "adapter_formatted": False,
                    "analysis_acceptable": True,
                    "classified_situation": "tweets_or_quotes",
                    "synthetic": False,
                    "speaker": seg.get("speaker"),
                    "is_target": True,
                    "target_name": target_name,
                    "adapter_prompt": adapter_prompt,
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                },
            )
        )
    return documents


def build_all_speakers_quote_documents(
    statements: List[Dict[str, Any]],
    *,
    user_id: str,
    assistant_id: str,
    media_item: Dict[str, Any],
    target_name: Optional[str],
    multi_speaker: bool,
) -> List[Document]:
    """One verbatim ``quote`` Document per statement when EVERY speaker is the
    avatar (the ``create_reference_media_from_playlist`` path only).

    This is a standalone builder: it never calls — and is never called by — the
    standard dialogue/quote/monologue helpers, so it cannot change any behaviour
    of the normal media-processing pipeline.

    Each statement (a diarized turn) becomes one adapter-eligible ``quote``
    Document. The synthesized question is decided per statement by
    :func:`process_adapter_documents` from the ``adapter_prompt`` metadata:

    * ``multi_speaker=True`` (a dialogue): the immediately preceding statement's
      text is attached as ``adapter_prompt`` so the real prior turn is **reused**
      as the question (the first statement has no predecessor → ``None`` →
      synthesized).
    * ``multi_speaker=False`` (one speaker): no ``adapter_prompt`` is attached,
      so a question is **generated** for every statement.
    """
    item_metadata = media_item.get("metadata", {}) or {}
    filename = item_metadata.get("filename", "")
    filename_uuid5 = str(uuid5(NAMESPACE_URL, filename or "unknown"))
    source = item_metadata.get("source") or filename or "user_upload"
    current_timestamp = datetime.now(tz=timezone.utc).isoformat()

    valid = [s for s in statements if (s.get("text") or "").strip()]
    total = len(valid)

    documents: List[Document] = []
    for idx, seg in enumerate(valid):
        text = (seg.get("text") or "").strip()
        adapter_prompt: Optional[str] = None
        if multi_speaker and idx > 0:
            adapter_prompt = (valid[idx - 1].get("text") or "").strip() or None
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "user_id": user_id,
                    "assistant_id": assistant_id,
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
                    "adapter_formatted": False,
                    "analysis_acceptable": True,
                    "classified_situation": (
                        "dialogue" if multi_speaker else "monologue"
                    ),
                    "synthetic": False,
                    "speaker": seg.get("speaker"),
                    "is_target": True,
                    "target_name": target_name,
                    "adapter_prompt": adapter_prompt,
                    "create_reference_media_from_playlist": True,
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
    current_timestamp = datetime.now(tz=timezone.utc).isoformat()

    speakers_seen = sorted({seg.get("speaker") for seg in dialogue_segments if seg.get("speaker")})

    return Document(
        page_content=json.dumps({"messages": role_messages}, ensure_ascii=False),
        metadata={
            "user_id": user_id,
            "assistant_id": assistant_id,
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


def _format_dialogue_transcript(segments: List[Dict[str, Any]]) -> str:
    """Render coalesced turns as ``speaker: text`` lines for whole-scene summary."""
    lines: List[str] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker = str(seg.get("speaker") or "unknown")
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


async def generate_scene_summary(
    transcript_text: str, *, target_name: Optional[str] = None
) -> str:
    """Spec Step 1: ONE structured-output summary of the entire scene.

    Reuses the concise-context-summary model (the same model the fact rewriter
    uses) over the FULL transcript so each target statement can later be
    analyzed with the overall situation as situational context. Returns ``""``
    on empty input or any failure, in which case analysis simply proceeds
    without the scene context.
    """
    text = (transcript_text or "").strip()
    if not text:
        return ""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from src.anubis.utils.classes.FactRewriterClass import (
            ConciseContextOfTheSourceOfFacts,
        )
        from src.anubis.utils.model import init_model
        from src.anubis.utils.prompts.concise_context_summary_prompt import (
            CONCISE_CONTEXT_SUMMARY_SYSTEM_PROMPT,
        )

        model = init_model(response_format=ConciseContextOfTheSourceOfFacts)
        response = await model.ainvoke(
            [
                SystemMessage(
                    content=CONCISE_CONTEXT_SUMMARY_SYSTEM_PROMPT.format(
                        target_name=target_name
                    )
                ),
                HumanMessage(content=text),
            ]
        )
        return (getattr(response, "concise_context_summary", "") or "").strip()
    except Exception as exc:
        logger.warning("Scene summary generation failed: %s", exc)
        return ""


async def _attach_target_analysis_context(
    target_quote_docs: List[Document], *, scene_summary: str
) -> None:
    """Spec Step 2 prep: stamp whole-scene summary + guaranteed user-context.

    Each target quote already carries ``adapter_prompt`` (the preceding
    non-target turn) when one exists. This adds, in place:

      * ``scene_summary`` — the one whole-scene summary, identical across every
        target statement in the dialogue.
      * ``user_context`` — the "user" turn the target statement responds to,
        used by the per-target analyzers. When the target led the conversation
        (no preceding non-target turn), a synthetic user statement is generated
        from the target statement so there is ALWAYS a target statement with a
        previous user input (per the spec).
    """
    if not target_quote_docs:
        return

    missing = [
        d
        for d in target_quote_docs
        if not (d.metadata.get("adapter_prompt") or "").strip()
    ]
    synthetic_iter = iter(())
    if missing:
        from src.anubis.utils.dataset.formatting import create_question_list

        synth = await create_question_list(
            [d.page_content or "" for d in missing]
        )
        synthetic_iter = iter(synth)

    for doc in target_quote_docs:
        if scene_summary:
            doc.metadata["scene_summary"] = scene_summary
        real = (doc.metadata.get("adapter_prompt") or "").strip()
        if real:
            doc.metadata["user_context"] = real
        else:
            doc.metadata["user_context"] = next(synthetic_iter, "")
            doc.metadata["user_context_synthetic"] = True


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
    # Coalesce defensively so the helper is correct whether callers pass raw
    # diarizer segments or already-coalesced turns (idempotent).
    segments: List[Dict[str, Any]] = coalesce_segments_by_speaker(
        list(dialogue_payload.get("segments") or [])
    )
    target_name = dialogue_payload.get("target_name")

    documents: List[Document] = []

    # Spec Step 1: summarize the entire scene ONCE so each target statement can
    # be analyzed with the overall situation as situational context.
    scene_summary = await generate_scene_summary(
        _format_dialogue_transcript(segments), target_name=target_name
    )

    # Target turns under ``quote`` (verbatim), each carrying its preceding
    # non-target turn as ``adapter_prompt``. Spec Step 2 prep: stamp the scene
    # summary and a guaranteed user-context (synthetic when the target led) onto
    # each so the per-target analyzers run with that context.

    """ Direct Quotes and Question and Answer Dataset """
    target_quote_docs = _build_target_quote_documents_from_dialogue(
        segments,
        user_id=user_id,
        assistant_id=assistant_id,
        media_item=media_item,
        target_name=target_name,
    )
    await _attach_target_analysis_context(
        target_quote_docs, scene_summary=scene_summary
    )
    documents.extend(target_quote_docs)

    # Full role-converted dialogue under ``adapter``.

    """ Create Multi-Turn Dialogue for adapter training """
    adapter_doc = _build_adapter_dialogue_document(
        segments,
        user_id=user_id,
        assistant_id=assistant_id,
        media_item=media_item,
        target_name=target_name,
    )
    if adapter_doc is not None:
        documents.append(adapter_doc)

    # Biographical facts about the target are extracted from each non-target
    # statement individually so per-speaker attribution is preserved on the
    # resulting identity Documents. FactRewriter returns nothing when a
    # non-target statement says nothing about the target, so empty statements
    # produce no Documents and non-target speech never lands in the quote
    # namespace.

    """ Biographical Documents from non-target """
    for seg in segments:
        if seg.get("is_target"):
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker_label = seg.get("speaker") or "unknown"
        try:
            statement_docs = await _build_biographical_identity_documents(
                text_content=text,
                user_id=user_id,
                assistant_id=assistant_id,
                media_item=media_item,
                target_name=target_name,
            )
        except Exception as exc:
            logger.warning(
                "Biographical fact extraction over non-target statement (speaker=%s) failed: %s",
                speaker_label,
                exc,
            )
            continue
        for doc in statement_docs:
            doc.metadata["speaker"] = speaker_label
            if seg.get("start") is not None:
                doc.metadata["start"] = seg.get("start")
            if seg.get("end") is not None:
                doc.metadata["end"] = seg.get("end")
        documents.extend(statement_docs)

    return documents


async def process_nontarget_text_to_identity_documents(
    text_content: str,
    *,
    user_id: str,
    assistant_id: str,
    media_item: Dict[str, Any],
    target_name: Optional[str] = None,
) -> List[Document]:
    """Treat non-target speech as biographical source about the target.

    Used for the single-speaker path when the lone speaker is NOT the target
    (and as the diarization-failure fallback for unidentified audio). Runs the
    FactRewriter -> FirstPersonRewriter pipeline and emits one ``identity``
    Document per first-person fact about the target. Returns ``[]`` when the
    speaker says nothing about the target, so non-target content never becomes
    a quote and produces no document when it carries no facts about the target.

    This deliberately bypasses :func:`process_text_to_document` (whose
    classifier keys on grammatical form, not speaker identity, and would route
    first-person non-target speech into the ``quote`` namespace).
    """
    if not (text_content or "").strip():
        return []

    namespace_filename = media_item.get("metadata", {}).get(
        "namespace_filename", ""
    )
    try:
        documents = await _build_biographical_identity_documents(
            text_content=text_content,
            user_id=user_id,
            assistant_id=assistant_id,
            media_item=media_item,
            target_name=target_name,
        )
    except Exception as exc:
        logger.warning(
            "Non-target biographical extraction failed (%s); no documents emitted",
            exc,
        )
        return []

    for document in documents:
        document.metadata.update({"namespace_filename": namespace_filename})
    return documents

# Long-form prose is attributed to speakers in windows so dialogue context is
# preserved per call. The window count is bounded so a very large corpus (e.g. a
# whole Bible) can't fan out into thousands of LLM calls in one request; raise
# the cap or split the upload for fuller coverage.
_TARGET_ATTRIBUTION_WINDOW_CHARS = 6000
_MAX_TARGET_ATTRIBUTION_WINDOWS = 25


def _window_text(text: str, window_chars: int, max_windows: int) -> List[str]:
    """Split ``text`` into ``<= max_windows`` character windows on paragraph/line
    boundaries where possible (keeps a speaker turn intact across the split)."""
    windows: List[str] = []
    remaining = text
    while remaining and len(windows) < max_windows:
        if len(remaining) <= window_chars:
            windows.append(remaining)
            break
        # Prefer to cut on a paragraph/line/sentence boundary before the hard cap.
        cut = window_chars
        for sep in ("\n\n", "\n", ". "):
            idx = remaining.rfind(sep, int(window_chars * 0.5), window_chars)
            if idx != -1:
                cut = idx + len(sep)
                break
        windows.append(remaining[:cut])
        remaining = remaining[cut:]
    return windows


async def _process_target_aware_text(
    *,
    text_content: str,
    user_id: str,
    assistant_id: str,
    media_item: Dict[str, Any],
    target_name: Optional[str],
) -> List[Document]:
    """Attribute long-form prose to speakers, then run the dialogue pipeline.

    Uses :class:`TargetIdentificationInTextClass` to turn unstructured prose
    (script, wiki, Bible passage, article) into speaker-attributed statements
    with ``is_target`` flags, then feeds them through
    :func:`process_dialogue_json_to_documents` so the SAME outputs as diarized
    audio are produced: target direct-quote Documents, biographical-fact
    Documents from non-target turns, and the role-converted adapter conversation.
    This is what makes prompt-completion + multi-turn adapter datasets available
    from text/URL/PDF per ``_OVERALL_PREPROCESSING_PROCESS.md``.
    """
    import asyncio

    from src.anubis.utils.classes.TargetIdentificationInTextClass import (
        TargetIdentificationInTextClass,
    )

    windows = _window_text(
        text_content,
        _TARGET_ATTRIBUTION_WINDOW_CHARS,
        _MAX_TARGET_ATTRIBUTION_WINDOWS,
    )
    if not windows:
        return []

    identifier = TargetIdentificationInTextClass()
    responses = await asyncio.gather(
        *[identifier.identify(window, target_name) for window in windows],
        return_exceptions=True,
    )

    segments: List[Dict[str, Any]] = []
    resolved_target = target_name
    for response in responses:
        if isinstance(response, Exception):
            logger.warning("target attribution window failed: %s", response)
            continue
        resolved_target = (
            response.get("target_name")
            if response.get("target_name") not in (None, "", "UNKNOWN")
            else resolved_target
        )
        for statement in response.get("statements") or []:
            content = (statement.get("content") or "").strip()
            if not content:
                continue
            segments.append(
                {
                    "speaker": statement.get("speaker") or "unknown",
                    "text": content,
                    "is_target": bool(statement.get("is_target")),
                    "start": None,
                    "end": None,
                }
            )

    if not segments:
        return []

    dialogue_payload = {
        "segments": segments,
        "target_name": resolved_target or target_name,
    }
    return await process_dialogue_json_to_documents(
        dialogue_payload=dialogue_payload,
        user_id=user_id,
        assistant_id=assistant_id,
        media_item=media_item,
    )


async def process_text_to_document(
    metadata,
    user_id,
    assistant_id,
    media_item,
    store: BaseStore,
    namespace_hint: Optional[str] = None,
    target_name: Optional[str] = None,
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
            - ``monologue`` -> verbatim chunked Document(s)
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
    namespace_filename = media_item.get("metadata", {}).get("namespace_filename", "")

    if not text_content.strip():
        logger.warning("Empty text content in media_item; returning no documents")
        return []

    # An explicit target supplied at upload time (param or media_item metadata)
    # signals intent to reconstruct THAT person from this text. It forces the
    # target-aware attribution path (direct quotes + biographical facts + adapter
    # datasets) and overrides the menu/holy-text short-circuit.
    explicit_target_name = target_name or (
        media_item.get("metadata", {}) or {}
    ).get("target_name")

    reference_classifier = ReferenceDocumentClassificationClass()
    reference_response = await reference_classifier.classify(classification_input)
    is_menu_or_religious_text = bool(
        reference_response.get("is_menu_or_religious_text", False)
    )
    reference_reasoning = reference_response.get("reasoning", "")

    # Bible-with-target-"Jesus", Curie-wiki-with-target-"Curie", and similar: when
    # a target is explicitly supplied at upload time, run target-aware extraction
    # for ANY source (not just menu/holy text) instead of deferring to the content-
    # situation classifier's single-bucket guess. This guarantees BOTH the target's
    # direct quotes AND biographical facts about the target are produced, per
    # _OVERALL_PREPROCESSING_PROCESS.md. The menu/holy-text branch below still
    # handles reference docs when no explicit target is supplied.
    if explicit_target_name and namespace_hint is None:
        logger.info(
            "Explicit target %r supplied -> target-aware extraction",
            explicit_target_name,
        )
        documents = await _process_target_aware_text(
            text_content=text_content,
            user_id=user_id,
            assistant_id=assistant_id,
            media_item=media_item,
            target_name=explicit_target_name,
        )
        for document in documents:
            document.metadata.setdefault("namespace_filename", namespace_filename)
            document.metadata.setdefault("target_name", explicit_target_name)
        return documents

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
            document.metadata.update({"vectorstore_acceptable": True, "namespace_filename": namespace_filename})
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
    elif classified_situation in ("tweets_or_quotes", "monologue"):
        target_namespace = "quote"
    elif classified_situation == "dialogue":
        target_namespace = "dialogue"
    else:
        target_namespace = "document"

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
            # Expected metadata: 
            # analysis_acceptable: True
            # namespace: identity (statements in first person about the target)
            # Analysis acceptable: True
            # adapter_acceptable: False (not creating synthetic prompt questions)
            # synthetic: True (identified facts with structured llm and rewrote the facts to first person)
            # classified_situation: biographical_facts
            for document in documents:
                document.metadata.update(
                    {
                        "classification_reasoning": situation_reasoning,
                        "reference_classification_reasoning": reference_reasoning,
                        "is_menu_or_religious_text": False,
                        "namespace_filename": namespace_filename,
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

    # Handle quotes-per-line text document 
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
        # presume the target is the only person for each line in the text content if the text is not labeled and only a single line of text per line (one statement per line)
        documents = _build_quote_documents_per_line(
            text_content=text_content,
            user_id=user_id,
            assistant_id=assistant_id,
            media_item=media_item,
            classification_metadata=classification_metadata,
        )
        """ CALIBRATE GROUND TRUTH """
        from src.subgraphs.process_media_graph.utils.calibrate_ground_truth import calibrate_ground_truth
        await calibrate_ground_truth(store=store, assistant_id=assistant_id, documents=documents)
        
        # Expected metadata (treated same as quotes below in next classified situation; only target information): 
        # vectorstore_acceptable: True
        # adapter_acceptable: True
        # adapter_formatted: False
        # analysis_acceptable: True
        # synthetic: False
        # namespace_filename: namespace_filename
        # namespace: quote

    elif classified_situation in ("monologue", "tweets_or_quotes"):
        # Monologue or tweets or quotes -> quote namespace
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
                    in ("monologue","tweets_or_quotes"),
                    "adapter_formatted": False, # will be processed to create prompt synthetic questions for the adapter namespace
                    "analysis_acceptable": True,
                    "synthetic": False,
                    "namespace_filename": namespace_filename, # identity namespace
                }
            )

    elif classified_situation == "dialogue":
        # Dialogue in raw prose form (script, interview, chat export). Attribute
        # the text to speakers (flagging the target), then run the shared dialogue
        # pipeline to emit: target direct-quote Documents, biographical-fact
        # Documents from non-target turns, and the role-converted adapter
        # conversation — the prompt-completion + multi-turn datasets are then built
        # downstream by process_adapter_documents.
        documents = await _process_target_aware_text(
            text_content=text_content,
            user_id=user_id,
            assistant_id=assistant_id,
            media_item=media_item,
            target_name=explicit_target_name or target_name,
        )
        for document in documents:
            document.metadata.update(
                {
                    "vectorstore_acceptable": True,
                    "adapter_acceptable": True,
                    "adapter_formatted": False,
                    "synthetic": False,
                    "namespace_filename": namespace_filename,
                }
            )

    for document in documents:
        document.metadata.update({"namespace_filename": namespace_filename})
    return documents
