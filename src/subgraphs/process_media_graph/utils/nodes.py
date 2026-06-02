# Nodes for Identifying and Handling each media type 

import asyncio
from curses import napms
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from uuid import NAMESPACE_URL, uuid4, uuid5
import logging
logger = logging.getLogger(__name__)
import base64
from pathlib import Path
import json

from langchain_community.document_loaders import PyPDFLoader
import tempfile, os


# At top of file
import tempfile

import base64
from src.anubis.utils.state import GlobalState
from src.anubis.utils.context import GlobalContext

# from langgraph.config import get_store

from langgraph.store.base import BaseStore
from langgraph.config import get_stream_writer


def _emit_media_progress(stage: str, **fields: Any) -> None:
    """Emit a ``media_progress`` custom event for the background-job SSE stream.

    Safe no-op when not running under a streaming context (e.g. plain ``ainvoke``).
    """
    try:
        writer = get_stream_writer()
        writer({"type": "media_progress", "stage": stage, **fields})
    except Exception:  # pragma: no cover - progress must never break processing
        pass


def _namespace_for(source: str) -> str:
    """Deterministic store key for a media source.

    Mirrors ``webapp._namespace_safe_formatted_filename`` (uuid5 over NAMESPACE_URL)
    so child keys computed here line up with the top-level keys the upload endpoint
    records — required for the skip/dedup set to match.
    """
    return str(uuid5(NAMESPACE_URL, source))

from src.subgraphs.process_media_graph.utils.helper_functions import (
    CLASSIFICATION_INPUT_CHAR_LIMIT,
    coalesce_segments_by_speaker,
    process_dialogue_json_to_documents,
    process_nontarget_text_to_identity_documents,
    process_text_media_item_target_for_vectorstore,
)
from src.anubis.utils.classes.ReferenceDocumentClassificationClass import (
    ReferenceDocumentClassificationClass,
)
from src.anubis.utils.classes.URLDocumentLoaderClass import URLDocumentLoaderClass
from src.anubis.utils.utility import (
    transcribe_audio,
    transcribe_audio_diarize,
    isolate_dominant_speaker_audio_b64,
    extract_video_audio_b64,
)

from src.anubis.utils.state import GlobalState
from src.anubis.utils.context import GlobalContext
from langgraph.runtime import Runtime

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.anubis.utils.model import init_model, init_image_description_model

from langchain.tools import tool


from langchain_core.runnables import RunnableConfig
from src.anubis.utils.utility import extract_user_id_assistant_id, transcribe_video
from src.subgraphs.process_media_graph.utils.utility import extract_personality_from_image
from src.subgraphs.process_media_graph.utils.helper_functions import process_text_to_document

_ALLOWED_STILL_IMAGE_MIMES = frozenset(
    {"image/jpeg", "image/png", "image/gif", "image/webp"}
)


def _normalize_declared_image_mime(ct: str) -> str:
    m = (ct or "").split(";")[0].strip().lower()
    return "image/jpeg" if m == "image/jpg" else m


def _full_data_uri_from_media_dict(media: Dict[str, Any]) -> str:
    """Full ``data:<mime>;base64,...`` string from webapp ``base64_encoded_str``."""
    return (media.get("base64_encoded_str") or "").strip()


def _decode_data_uri_base64_payload(data_uri: str) -> bytes:
    """Decode the base64 segment of an RFC 2397 data URI."""
    if not data_uri.startswith("data:") or "," not in data_uri:
        raise ValueError("Expected a data URI with a base64 payload")
    return base64.b64decode(data_uri.split(",", 1)[1])


def _is_full_image_data_uri(value: str) -> bool:
    """True if ``value`` is a complete ``data:image/...;base64,...`` string."""
    normalized = (value or "").strip()
    return normalized.startswith("data:image/") and ";base64," in normalized


def _is_full_audio_data_uri(value: str) -> bool:
    """True if ``value`` is a complete ``data:audio/...;base64,...`` string."""
    normalized = (value or "").strip()
    return normalized.startswith("data:audio/") and ";base64," in normalized


async def process_uploaded_files_and_label_media_type(
    state: GlobalState, 
    runtime: Runtime[GlobalContext], 
    config: RunnableConfig,
    store: BaseStore
) -> Dict[str, Any]:
    """
    Convert FastAPI UploadFile objects into standardized media format.
    This is the entry point for direct file uploads (not from messages).
    """
    
    logger.info(f"Process uploaded files NODE")
    user_id, assistant_id = await extract_user_id_assistant_id(config)

    media_files = state.get('media_files', [])
    
    if not media_files:
        logger.info("No media files to process")
        return {"media_list": []}
    
    logger.info(f"Processing {len(media_files)} uploaded files")
    
    media_list = []
    
    for file_data in media_files:
        try:
            file_start_idx = len(media_list)
            filename = file_data.get('filename', 'unknown')
            suffix = Path(filename).suffix
            content_type = file_data.get('content_type', '')
            file_bytes = file_data.get('content')
            user_id = file_data.get("user_id")
            assistant_id = file_data.get("assistant_id")
            reference_image = file_data.get("reference_image")
            reference_audio = file_data.get("reference_audio")
            namespace_filename = file_data.get("namespace_filename")
        
            logger.info(f"Processing file: {filename} ({content_type})")

            full_payload_uri = _full_data_uri_from_media_dict(file_data)

            image_url_remote = (file_data.get("image_url") or "").strip()
            if image_url_remote:
                mime = _normalize_declared_image_mime(content_type)
                if mime not in _ALLOWED_STILL_IMAGE_MIMES:
                    logger.warning(
                        "Skipping remote image with disallowed MIME %s", mime
                    )
                    continue
                entry: Dict[str, Any] = {
                    "type": "image_url",
                    "image_url": {"url": image_url_remote},
                    "metadata": {
                        "filename": filename,
                        "content_type": mime,
                        "size": 0,
                        "user_id": user_id,
                        "assistant_id": assistant_id,
                        "reference_image": reference_image,
                        "namespace_filename": namespace_filename,
                    },
                }
                if full_payload_uri:
                    entry["base64_encoded_str"] = full_payload_uri
                media_list.append(entry)
                continue

            audio_url_remote = (file_data.get("audio_url") or "").strip()
            if audio_url_remote:
                mime = content_type.split(";")[0].strip().lower()
                if not mime.startswith("audio/"):
                    logger.warning(
                        "Skipping remote audio with non-audio MIME %s", mime
                    )
                    continue
                entry = {
                    "type": "audio",
                    "audio_url": {"url": audio_url_remote},
                    "metadata": {
                        "filename": filename,
                        "content_type": mime,
                        "size": 0,
                        "user_id": user_id,
                        "assistant_id": assistant_id,
                        "reference_audio": reference_audio,
                        "namespace_filename": namespace_filename,
                    },
                }
                if full_payload_uri:
                    entry["base64_encoded_str"] = full_payload_uri
                media_list.append(entry)
                continue

            video_url_remote = (file_data.get("video_url") or "").strip()
            if video_url_remote:
                mime = content_type.split(";")[0].strip().lower()
                if not mime.startswith("video/"):
                    logger.warning(
                        "Skipping remote video with non-video MIME %s", mime
                    )
                    continue
                entry = {
                    "type": "video",
                    "video_url": {"url": video_url_remote},
                    "metadata": {
                        "filename": filename,
                        "content_type": mime,
                        "size": 0,
                        "user_id": user_id,
                        "assistant_id": assistant_id,
                        "namespace_filename": namespace_filename,
                    },
                }
                if full_payload_uri:
                    entry["base64_encoded_str"] = full_payload_uri
                media_list.append(entry)
                continue

            page_url_remote = (file_data.get("page_url") or "").strip()
            if page_url_remote:
                mime = content_type.split(";")[0].strip().lower()
                entry = {
                    "type": "url",
                    "url": page_url_remote,
                    "metadata": {
                        "filename": filename,
                        "content_type": mime,
                        "size": 0,
                        "user_id": user_id,
                        "assistant_id": assistant_id,
                        "namespace_filename": namespace_filename,
                    },
                }
                if full_payload_uri:
                    entry["base64_encoded_str"] = full_payload_uri
                media_list.append(entry)
                continue
            
            # Log files are captured verbatim as a single reference document
            # (no classification, no chunking), regardless of the declared
            # content type the client sent for the ``.log`` upload.
            if suffix.lower() == '.log':
                if full_payload_uri:
                    log_text = _decode_data_uri_base64_payload(
                        full_payload_uri
                    ).decode("utf-8", errors="replace")
                else:
                    log_text = (file_bytes or b"").decode("utf-8", errors="replace")
                media_list.append({
                    "type": "log",
                    "content": log_text,
                    "metadata": {
                        "filename": filename,
                        "content_type": content_type,
                        "size": len(file_bytes or b""),
                        "user_id": user_id,
                        "assistant_id": assistant_id,
                        "namespace_filename": namespace_filename
                    }
                })
                continue

            # Determine media type and convert to standardized format
            if content_type.startswith('image/'):
                mime = _normalize_declared_image_mime(content_type)
                if mime not in _ALLOWED_STILL_IMAGE_MIMES:
                    logger.warning("Skipping image with disallowed MIME %s", mime)
                    continue
                payload_uri = full_payload_uri
                if not payload_uri:
                    if not file_bytes:
                        logger.warning(
                            "Skipping image %s: no base64_encoded_str or bytes",
                            filename,
                        )
                        continue
                    payload_uri = (
                        f"data:{mime};base64,"
                        f"{base64.b64encode(file_bytes).decode('ascii')}"
                    )
                media_list.append({
                    "type": "image",
                    "base64_encoded_str": payload_uri,
                    "metadata": {
                        "filename": filename,
                        "content_type": mime,
                        "size": len(file_bytes or b""),
                        "user_id": user_id,
                        "assistant_id": assistant_id,
                        "reference_image": reference_image,
                        "namespace_filename": namespace_filename,
                    }
                })
            
            elif content_type.startswith('audio/'):
                mime = content_type.split(";")[0].strip().lower()
                payload_uri = full_payload_uri
                if not payload_uri:
                    if not file_bytes:
                        logger.warning(
                            "Skipping audio %s: no base64_encoded_str or bytes",
                            filename,
                        )
                        continue
                    payload_uri = (
                        f"data:{mime};base64,"
                        f"{base64.b64encode(file_bytes).decode('ascii')}"
                    )
                media_list.append({
                    "type": "audio",
                    "base64_encoded_str": payload_uri,
                    "metadata": {
                        "filename": filename,
                        "content_type": content_type,
                        "size": len(file_bytes or b""),
                        "user_id": user_id,
                        "assistant_id": assistant_id, 
                        "reference_audio": reference_audio,
                        "namespace_filename": namespace_filename
                    }
                })
            
            elif content_type.startswith('video/'):
                mime = content_type.split(";")[0].strip().lower()
                payload_uri = full_payload_uri
                if not payload_uri:
                    if not file_bytes:
                        logger.warning(
                            "Skipping video %s: no base64_encoded_str or bytes",
                            filename,
                        )
                        continue
                    payload_uri = (
                        f"data:{mime};base64,"
                        f"{base64.b64encode(file_bytes).decode('ascii')}"
                    )
                media_list.append({
                    "type": "video",
                    "base64_encoded_str": payload_uri,
                    "metadata": {
                        "filename": filename,
                        "content_type": content_type,
                        "size": len(file_bytes or b""),
                        "user_id": user_id,
                        "reference_audio": reference_audio,
                        "assistant_id": assistant_id,
                        "namespace_filename": namespace_filename
                    }
                })
            
            elif content_type in [
                'text/plain', 
            'application/json', 
            'text/markdown', 
            'application/octet-stream', 
            'text/csv'
            ]:
                # Handle text files — prefer webapp base64_encoded_str (full data URI)
                if suffix == '.txt':
                    if full_payload_uri:
                        text_content = _decode_data_uri_base64_payload(
                            full_payload_uri
                        ).decode("utf-8", errors="replace")
                    else:
                        text_content = file_bytes.decode('utf-8')
                    media_list.append({
                        "type": "text",
                        "content": text_content,
                        "metadata": {
                            "filename": filename,
                            "content_type": content_type,
                            "size": len(file_bytes or b""),
                            "user_id": user_id,
                            "assistant_id": assistant_id,
                            "namespace_filename": namespace_filename
                        }
                    })
                elif suffix == '.log':
                    if full_payload_uri:
                        text_content = _decode_data_uri_base64_payload(
                            full_payload_uri
                        ).decode("utf-8", errors="replace")
                    else:
                        text_content = file_bytes.decode('utf-8', errors="replace")
                    media_list.append({
                        "type": "log",
                        "content": text_content,
                        "metadata": {
                            "filename": filename,
                            "content_type": content_type,
                            "size": len(file_bytes or b""),
                            "user_id": user_id,
                            "assistant_id": assistant_id,
                            "namespace_filename": namespace_filename
                        }
                    })
                elif suffix == '.json' or suffix == '.jsonl':
                    if full_payload_uri:
                        raw = _decode_data_uri_base64_payload(
                            full_payload_uri
                        ).decode("utf-8")
                        text_content = json.loads(raw)
                    else:
                        text_content = json.loads(file_bytes.decode('utf-8'))
                    media_list.append({
                        "type": "json",
                        "content": text_content,
                        "metadata": {
                            "filename": filename,
                            "content_type": content_type,
                            "size": len(file_bytes or b""),
                            "user_id": user_id,
                            "assistant_id": assistant_id,
                            "namespace_filename": namespace_filename
                        }
                    })
                else: # handle markdown
                    if full_payload_uri:
                        text_content = _decode_data_uri_base64_payload(
                            full_payload_uri
                        )
                    else:
                        text_content = file_bytes
                    media_list.append({ # if content_type is application_json, then the type needs to be json
                        "type": content_type.split("/")[-1] if content_type.split("/")[-1] != "" else "text",
                        "content": text_content,
                        "metadata": {
                            "filename": filename,
                            "content_type": content_type,
                            "size": len(file_bytes or b""),
                            "user_id": user_id,
                            "assistant_id": assistant_id,
                            "namespace_filename": namespace_filename
                        }
                    })
            
            elif content_type == 'application/pdf':
                if full_payload_uri:
                    pdf_bytes = _decode_data_uri_base64_payload(full_payload_uri)
                elif file_bytes:
                    pdf_bytes = file_bytes
                else:
                    logger.warning(
                        "Skipping PDF %s: no base64_encoded_str or bytes", filename
                    )
                    continue
                media_list.append({
                    "type": "pdf",
                    "base64_encoded_str": full_payload_uri or "",
                    "bytes": pdf_bytes,
                    "metadata": {
                        "filename": filename,
                        "content_type": content_type,
                        "size": len(pdf_bytes),
                        "user_id": user_id,
                        "assistant_id": assistant_id,
                        "namespace_filename": namespace_filename
                    }
                })
            
            else:
                logger.warning(f"Unsupported content type: {content_type}")
                continue

        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            continue
    
    logger.info(f"Converted {len(media_list)} files to media format")
    _emit_media_progress("labeling", total=len(media_list))

    return {
        "media_list": media_list,
        "media_files": []
    }


# Metadata keys only used for queue routing; omitted from ``additional_metadata`` for scaffolds.
_ANALYSIS_QUEUE_METADATA_KEYS = frozenset(
    {
        "analysis_scaffolds",
        "analysis_job_kind",
        "analysis_acceptable",
        "vectorstore_acceptable",
        "adapter_acceptable",
    }
)


def _metadata_for_analysis_outputs(doc: Document) -> Dict[str, Any]:
    return {
        k: v
        for k, v in (doc.metadata or {}).items()
        if k not in _ANALYSIS_QUEUE_METADATA_KEYS
    }


async def _analysis_scaffold_ocean(doc: Document) -> List[Document]:
    from src.anubis.utils.analysis.analysis_methods import perform_ocean_analysis

    text = (doc.page_content or "").strip()
    if not text:
        return []
    meta = _metadata_for_analysis_outputs(doc)
    meta.setdefault("vectorstore_acceptable", True)
    return await perform_ocean_analysis(
        HumanMessage(content=text),
        additional_metadata=meta,
    )


# Extend with additional keyed scaffolds (emotion, relational graphs, etc.).
ANALYSIS_SCAFFOLD_RUNNERS: Dict[str, Any] = {
    "ocean": _analysis_scaffold_ocean,
}


async def analyze_documents(
    state: GlobalState,
    runtime: Runtime[GlobalContext],
    config: RunnableConfig,
    store: BaseStore,
) -> Dict[str, Any]:
    """Run registered analysis scaffolds on queued documents; merge vector-bound results."""
    queue: List[Document] = list(
        state.get(
            "documents_to_be_analyzed_for_context_storage_and_prompt_injection_of_assistant",
        )
        or []
    )
    if not queue:
        logger.info("analyze_documents: empty queue; skipping")
        return {}

    out: List[Document] = []
    for doc in queue:
        scaffolds = doc.metadata.get("analysis_scaffolds")
        if not scaffolds:
            scaffolds = ["ocean"]
        for name in scaffolds:
            runner = ANALYSIS_SCAFFOLD_RUNNERS.get(name)
            if runner is None:
                logger.warning(
                    "analyze_documents: unknown scaffold %r (filename=%s); skipping",
                    name,
                    doc.metadata.get("filename", ""),
                )
                continue
            try:
                produced = await runner(doc)
                if produced:
                    out.extend(produced)
            except Exception as exc:
                logger.warning(
                    "analyze_documents: scaffold %r failed: %s; continuing",
                    name,
                    exc,
                )

    existing_vs: List[Document] = list(
        state.get("vectorstore_documents_to_be_indexed") or []
    )
    return {
        "documents_to_be_analyzed_for_context_storage_and_prompt_injection_of_assistant": "delete",
        "vectorstore_documents_to_be_indexed": existing_vs + out,
    }


async def convert_media_list_to_text_document(state: GlobalState, runtime: Runtime[GlobalContext], store: BaseStore, config: RunnableConfig) -> Dict[str, Any]:
    """ 
    Media type in media list is determined at this point: 
    Convert the media in a list of one or more media to text in parallel.
    media items must have user_id and assistant_id as metadata.
    Exptected format:
    [
        {
            "type": "MEDIA_TYPE",
            "base64_encoded_str": "full data URI for binary media (when applicable)",
            "content": "text / structured content for text and json types",
            "metadata":{
                fields may include mime-type or the metadata may not exists at all
                }
        }, 
        ...
    ]
    I want to keep the media in a list and queue tasks for each item in the list 
    then I want to execute those tasks in parallel and update the final state 
    with the list of text Documents from the media:
    async def determine_media_type(state: GlobalState, context: GlobalContext, media_list: List[Dict]):
    """
    
    logging.info(f"DETERMINE_MEDIA_TYPE NODE")

    media_list = state.get('media_list', [])

    if not media_list:
        logger.info(f"No Meida to process")
        return {
            "media_list": []
        }

    # namespace_filename values already indexed for this avatar. Items whose key
    # is present are skipped (top-level here; expanded playlist/linktree children
    # inside _expand_url_media_item) so a re-upload doesn't reprocess hundreds of
    # already-stored files. Populated by the upload endpoint.
    existing_namespaces: set = set(state.get("existing_namespaces") or [])

    # Drop top-level items already indexed before doing any work.
    to_process: list = []
    for media_item in media_list:
        ns = (media_item.get("metadata", {}) or {}).get("namespace_filename")
        if ns and ns in existing_namespaces:
            _emit_media_progress(
                "skipped_existing",
                filename=(media_item.get("metadata", {}) or {}).get("filename"),
                namespace_filename=ns,
            )
            continue
        to_process.append(media_item)

    skipped_count = len(media_list) - len(to_process)
    total_items = len(to_process)
    logger.info(
        "Processing %d media items (%d skipped as already indexed)",
        total_items,
        skipped_count,
    )
    _emit_media_progress("converting_started", total=total_items, skipped=skipped_count)

    # Per-item / batch identity passed by the upload orchestrator. Stamped onto
    # every produced Document below so a per-item or batch cancel can delete
    # exactly the rows this run wrote (including expanded playlist children).
    try:
        configurable = (config or {}).get("configurable", {}) or {}
    except Exception:
        configurable = {}
    item_job_id = configurable.get("item_job_id")
    master_job_id = configurable.get("master_job_id")

    # Bound how many items convert in parallel (OpenAI diarization / yt_dlp /
    # web fetches) so a big playlist or batch upload can't exhaust rate limits or
    # memory. The same semaphore is threaded through URL expansion so the cap is
    # global across the whole recursion. When part of a batch, reuse the batch's
    # shared limiter (registered by run_batch_media_job, keyed by master_job_id) so
    # every item — and every playlist child — draws from one global pool of slots
    # instead of N independent per-item pools. See media_processing_concurrency.
    concurrency = max(1, GlobalContext().media_processing_concurrency)
    semaphore = None
    if master_job_id:
        try:
            from src.api.media_jobs import get_batch_semaphore

            semaphore = get_batch_semaphore(master_job_id)
        except Exception:  # pragma: no cover - registry import must never break work
            semaphore = None
    if semaphore is None:
        semaphore = asyncio.Semaphore(concurrency)

    async def _convert_one(index: int, media_item: Dict[str, Any]) -> List[Document]:
        _emit_media_progress(
            "converting",
            current=index + 1,
            total=total_items,
            filename=media_item.get("metadata", {}).get("filename")
            or media_item.get("filename"),
        )
        try:
            return await process_media_item_task(
                media_item,
                runtime,
                config,
                store,
                existing_namespaces=existing_namespaces,
                semaphore=semaphore,
            )
        except Exception as exc:  # noqa: BLE001 - one bad item must not abort the batch
            logger.exception("Media item processing crashed: %s", exc)
            return [
                Document(
                    page_content=f"[Error processing media: {exc}]",
                    metadata={
                        "status": "error",
                        "error": str(exc),
                        "filename": media_item.get("metadata", {}).get("filename"),
                    },
                )
            ]

    # Process every item in parallel; collect results and per-item errors without
    # aborting the batch (partial success — one failed video shouldn't sink a
    # 300-item playlist). Errors are surfaced as media_progress events + summary.
    all_documents: list = []
    processing_errors: list[str] = []
    results = await asyncio.gather(
        *(_convert_one(i, item) for i, item in enumerate(to_process))
    )
    for docs in results:
        for doc in docs:
            status = doc.metadata.get("status", "")
            if status == "error":
                error = doc.metadata.get("error", "")
                filename = doc.metadata.get("filename", "")
                processing_errors.append(f"{filename}: {error or 'unknown'}")
                logger.warning("Error processing media: %s %s", filename, error)
                _emit_media_progress(
                    "item_error", filename=filename, error=error or "unknown"
                )
            else:
                all_documents.append(doc)

    _emit_media_progress(
        "converting_complete",
        total=total_items,
        skipped=skipped_count,
        errors=len(processing_errors),
        indexed=len(all_documents),
    )

    # Stamp batch/item identity onto every produced Document so it is persisted
    # under value.document.kwargs.metadata at index time — this is the key a
    # per-item or batch cancel deletes by (covers playlist children, whose
    # namespace_filename differs from the top-level item's).
    if item_job_id or master_job_id:
        for doc in all_documents:
            try:
                if item_job_id:
                    doc.metadata["item_job_id"] = item_job_id
                if master_job_id:
                    doc.metadata["master_job_id"] = master_job_id
            except Exception:  # pragma: no cover - metadata must never break work
                pass

    # Identify Vector Store formatted documents
    # NOTE This contains only information from the target speaker or about the target speaker
    vector_store_document_list_formatted = [
        doc
        for doc in all_documents
        if doc.metadata.get("vectorstore_acceptable", False) is True
    ]

    # # Analysis list (needs a node)
    # These documents have been formatted for analysis but have not yet been analyzed.
    # NOTE: Using non-target information will indicate triggers or responses. This information must not be lost. For analysis, keep both the User and other speakers but focus on the target.

    analysis_document_list_formatted = [doc for doc in all_documents if doc.metadata.get("analysis_acceptable", False) == True]

    # # Adapter list (needs a node)
    # documents_to_be_processed_for_adapter_training: List[Sequence[Document]] UPDATED RETURN VALUES IN RETURN processed into adapter training format and uploaded to storage

    adapter_document_list_formatted = [doc for doc in all_documents if doc.metadata.get("adapter_acceptable", False) == True]

    return {
        "vectorstore_documents_to_be_indexed": vector_store_document_list_formatted,
        "documents_to_be_analyzed_for_context_storage_and_prompt_injection_of_assistant": analysis_document_list_formatted,
        "documents_to_be_processed_for_adapter_training": adapter_document_list_formatted,
        "media_list": [] # Clear processed media list in the state
    }

async def process_media_item_task(
    media_item: Dict[str, Any],
    runtime: Runtime[GlobalContext],
    config: RunnableConfig,
    store: BaseStore,
    *,
    existing_namespaces: Optional[set] = None,
    semaphore: Optional["asyncio.Semaphore"] = None,
) -> List[Document]:
    """Task: Convert a single media item to a list of Documents.

    ``existing_namespaces`` / ``semaphore`` thread the skip-set and the global
    concurrency limit through the recursion (URL items expand into children that
    re-enter this function). URL items are expansion *coordinators*: they must not
    hold a semaphore slot while awaiting their children, or the bounded pool would
    deadlock — so they are delegated to ``_expand_url_media_item`` which takes a
    slot only around the heavy ``loader.load()`` call. Every other (leaf) media
    type holds a slot for the duration of its conversion work below.
    """

    logger.info(f"process_media_item_task entry")

    media_type = media_item.get("type", "")

    metadata = media_item.get("metadata", {})

    user_id = metadata.get("user_id", "")
    assistant_id = metadata.get("assistant_id", "")

    logger.info(f"extracted user_id: {user_id}")
    logger.info(f"extracted assistant_id: {assistant_id}")

    filename = media_item['metadata']['filename']
    logger.info(f"Processing file: {filename}")
    namespace_filename = metadata.get("namespace_filename")
    if not namespace_filename:
        raise Exception(f"namespace_filename is required for media item: {media_item}")

    # URL items expand into children and must not hold a concurrency slot while
    # awaiting them (see docstring). Delegate before taking the leaf semaphore.
    if media_type == "url":
        return await _expand_url_media_item(
            media_item,
            runtime,
            config,
            store,
            url=(media_item.get("url") or "").strip(),
            filename=filename,
            namespace_filename=namespace_filename,
            user_id=user_id,
            assistant_id=assistant_id,
            existing_namespaces=existing_namespaces,
            semaphore=semaphore,
        )

    # Leaf media types: hold a concurrency slot for the duration of the (heavy)
    # conversion so a large batch/playlist can't fan out unboundedly.
    leaf_acquired = False
    if semaphore is not None:
        await semaphore.acquire()
        leaf_acquired = True
    try:
        if media_type in ("image", "image_url"):
            reference_image = metadata.get("reference_image", False)

            full_uri = _full_data_uri_from_media_dict(media_item)

            if reference_image:
                if not _is_full_image_data_uri(full_uri):
                    return [
                        Document(
                            page_content=(
                                "[Reference image requires media_item.base64_encoded_str "
                                "as the full data:image/...;base64,... string from upstream]"
                            ),
                            metadata={
                                "status": "error",
                                "error": "missing_reference_image_data_uri",
                                "filename": filename,
                            },
                        )
                    ]
                image_source = full_uri
            else:
                image_source = ""
                if full_uri:
                    image_source = full_uri
                elif media_type == "image_url":
                    url = (media_item.get("image_url") or {}).get("url", "").strip()
                    if not url:
                        return [
                            Document(
                                page_content="[Empty image URL]",
                                metadata={
                                    "status": "error",
                                    "error": "empty_url",
                                    "filename": filename,
                                },
                            )
                        ]
                    image_source = url
                elif "image_url" in media_item:
                    url = (media_item.get("image_url") or {}).get("url", "").strip()
                    image_source = url
                else:
                    return [
                        Document(
                            page_content="[Unsupported image payload]",
                            metadata={
                                "status": "error",
                                "error": "missing_image_data",
                                "filename": filename,
                            },
                        )
                    ]

            doc = await extract_personality_from_image(
                image_data=image_source,
                filename=filename,
                store=store,
                user_id=user_id,
                assistant_id=assistant_id,
                context=runtime.context,
                reference_image=reference_image
            )

            doc.metadata.update(
                {
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "created_at": datetime.now(tz=timezone.utc).isoformat(),
                    "processing_task_id": str(uuid4()),
                    "reference_image": reference_image,
                    "filename": filename,
                    "analysis_acceptable": True,
                    "namespace_filename": namespace_filename, 
                }
            )

            if reference_image:
                
                # I need a function to create generative images of different emotions: happiness, sadness, anger, surprise, fear, disgust, pondering
                # the function should take in the reference image and the emotion and return a base64_encoded_str
                # the function should be called for each emotion
                # the base64_encoded_str is passed into the extract_personality_from_image function


                # Use the reference image to create generative images of different emotions: happiness, sadness, anger, surprise, fear, disgust, pondering
                # namespace is (user_id, assistant_id, "identity")
                # key is the assistant_id
                # metadata contains "emotion" and "base64_encoded_str"
                # page_content is the description of the image
                # filename is the filename of the reference image with the emotion appended to the end with an underscore
                # the description is created with extract_personality_from_image
                # add "synthetic" True to the metadata
                # add "content_type": "image/jpeg" to the metadata
                # set as vectorstore_acceptable True, adapter_acceptable False, analysis_acceptable False
                # append to the list of documents
                # an api endpoint provides an endpoint to allow for the search of the store for metadata for "emotion", "content_type", and "synthetic" to display the images on load of the avatar once and caches all results then uses the results on emotion trigger.
                # The frontend searches the metadata for "emotion", "content_type", and "synthetic" to display the images
                
                
                namespace = (user_id, assistant_id, "reference_image")
                doc_json = doc.to_json()
                
                await store.aput(
                    namespace,
                    key=assistant_id,
                    value={
                        "reference_image_data": full_uri,
                        "document": doc_json,
                    },
                )
                doc.metadata.update(
                    {
                        "namespace": "reference_image",
                        "vectorstore_acceptable": False,
                        "adapter_acceptable": False,
                        "namespace_filename": namespace_filename,
                    }
                )
                return [doc]

            description_text = (doc.page_content or "").strip()
            if not description_text:
                doc.metadata.update(
                    {"namespace": "identity", "vectorstore_acceptable": False}
                )
                return [doc]

            try:
                reference_classifier = ReferenceDocumentClassificationClass()
                reference_response = await reference_classifier.classify(
                    description_text[:CLASSIFICATION_INPUT_CHAR_LIMIT]
                )
                is_menu_or_religious_text = bool(
                    reference_response.get("is_menu_or_religious_text", False)
                )
                reference_reasoning = reference_response.get("reasoning", "")
            except Exception as e:
                logger.warning(
                    "Image description reference classification failed: %s", e
                )
                is_menu_or_religious_text = False
                reference_reasoning = ""

            target_namespace = "document" if is_menu_or_religious_text else "identity"
            classified_situation = (
                "proprietary_content" if is_menu_or_religious_text else "image_identity"
            )

            description_media_item = {
                "type": "text",
                "content": description_text,
                "metadata": {
                    "filename": filename,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "source": "image_description",
                    "image_filename": filename,
                },
            }
            classification_metadata = {
                "classified_situation": classified_situation,
                "reference_classification_reasoning": reference_reasoning,
                "is_menu_or_religious_text": is_menu_or_religious_text,
                "image_filename": filename,
            }

            documents = await process_text_media_item_target_for_vectorstore(
                media_item=description_media_item,
                user_id=user_id,
                assistant_id=assistant_id,
                classification_metadata=classification_metadata,
                use_semantic_chunks=False,
                namespace=target_namespace,
            )
            for d in documents:
                d.metadata.update(
                    {
                        "vectorstore_acceptable": True,
                        "adapter_acceptable": False,
                        "analysis_acceptable": True,
                        "namespace_filename": namespace_filename,
                    }
                )
            return documents

        elif media_type == "text":
            documents = await process_text_to_document(
                metadata=metadata,
                user_id=user_id,
                assistant_id=assistant_id,
                media_item=media_item,
            )
            return documents
        elif media_type == "log":
            # .log files bypass the content-situation classifier and are
            # token-aware chunked straight into the ``document`` namespace.
            classification_metadata = {
                "classified_situation": "log_file",
                "classification_reasoning": (
                    "log files bypass classification and are chunked directly"
                ),
                "is_menu_or_religious_text": False,
            }
            documents = await process_text_media_item_target_for_vectorstore(
                media_item=media_item,
                user_id=user_id,
                assistant_id=assistant_id,
                classification_metadata=classification_metadata,
                use_semantic_chunks=False,
                namespace="document",
            )
            for document in documents:
                document.metadata.update(
                    {
                        "vectorstore_acceptable": True,
                        "adapter_acceptable": False,
                        "analysis_acceptable": False,
                        "synthetic": False,
                        "namespace_filename": namespace_filename,
                    }
                )
            return documents
        elif media_type == "json": # formatted proprietary llm content (chatgpt, claude, grok, etc.)
            content = media_item.get('content')

            # CSV preprocessing in webapp.py emits ``{"statements": [...]}``
            # where each statement is the avatar-identity contract shape:
            # ``{"messages": [{"role": "assistant", "content": "..."}],
            #    "metadata": {"target": "...", "source": "..."}}``.
            # Treat these as quote-namespace Documents so they feed both
            # retrieval and adapter training, with per-statement target /
            # source metadata flowing to each Document.
            if isinstance(content, bytes): 
                content = json.loads(content)

            if (
                isinstance(content, dict)
                and isinstance(content.get("statements"), list)
            ):
                final_documents: List[Document] = []
                base_metadata = dict(media_item.get('metadata') or {})
                for statement in content["statements"]:
                    if not isinstance(statement, dict):
                        continue
                    stmt_messages = statement.get("messages") or []
                    stmt_meta_raw = statement.get("metadata") or {}
                    statement_metadata = dict(base_metadata)
                    statement_metadata.update(
                        {k: v for k, v in stmt_meta_raw.items() if v is not None}
                    )
                    target_name = (
                        stmt_meta_raw.get("target")
                        or statement_metadata.get("target")
                        or statement_metadata.get("target_name")
                    )
                    if target_name:
                        statement_metadata["target_name"] = target_name
                    source_label = (
                        stmt_meta_raw.get("source")
                        or statement_metadata.get("source")
                        or statement_metadata.get("filename")
                    )
                    if source_label:
                        statement_metadata["source"] = source_label

                    classification_metadata = {
                        "classified_situation": "tweets_or_quotes",
                        "classification_reasoning": (
                            "csv_preprocessed_avatar_identity_statements"
                        ),
                        "quotes_per_line": True,
                        "target_name": target_name,
                    }

                    for message in stmt_messages:
                        if not isinstance(message, dict):
                            continue
                        text = (message.get("content") or "").strip()
                        if not text:
                            continue
                        statement_media_item = {
                            "type": "json",
                            "content": text,
                            "metadata": statement_metadata,
                        }
                        documents = await process_text_media_item_target_for_vectorstore(
                            media_item=statement_media_item,
                            user_id=user_id,
                            assistant_id=assistant_id,
                            classification_metadata=classification_metadata,
                            use_semantic_chunks=False,
                            namespace="quote",
                        )
                        for document in documents:
                            document.metadata.update(
                                {
                                    "vectorstore_acceptable": True,
                                    "adapter_acceptable": True,
                                    "analysis_acceptable": True,
                                    "adapter_formatted": False,
                                    "synthetic": False,
                                    "namespace_filename": namespace_filename,
                                }
                            )
                            final_documents.append(document)
                return final_documents

            # Existing single-conversation shape: ``{"messages": [...]}``
            classification_metadata = {
                "classified_situation": "conversation_facts",
                "classification_reasoning": "user_selected_classification_of_ai_human_conversation"
            }
            messages = media_item['content']['messages']
            final_documents = []
            for message in messages:
                media_item['content'] = message['content']
                documents = await process_text_media_item_target_for_vectorstore(
                    media_item=media_item,
                    user_id=user_id,
                    assistant_id=assistant_id,
                    classification_metadata=classification_metadata,
                    use_semantic_chunks=False
                )

                for document in documents:
                            document.metadata.update({"vectorstore_acceptable": True, "namespace_filename": namespace_filename})
                            final_documents.append(document)

            return final_documents
        # NOTE: ``type == "url"`` is handled before the leaf semaphore guard by
        # _expand_url_media_item (see the early return at the top of this
        # function) so it never reaches this branch chain.

        elif media_type == "pdf":
            """Extract text from each PDF page and route it through process_text_to_document
            so reference + content situation classifiers decide the namespace per page."""
            pdf_bytes = media_item.get("bytes")
            if not pdf_bytes:
                pdf_uri = _full_data_uri_from_media_dict(media_item)
                if pdf_uri:
                    pdf_bytes = _decode_data_uri_base64_payload(pdf_uri)
            if not pdf_bytes:
                return [
                    Document(
                        page_content="[PDF bytes missing]",
                        metadata={
                            "status": "error",
                            "error": "missing_pdf_bytes",
                            "filename": filename,
                        },
                    )
                ]
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                tmp_file.write(pdf_bytes)
                tmp_path = tmp_file.name
            try:
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            final_documents: List[Document] = []
            for page_idx, page_doc in enumerate(pages):
                page_text = (page_doc.page_content or "").strip()
                if not page_text:
                    continue
                page_media_item = {
                    "type": "text",
                    "content": page_text,
                    "metadata": {
                        "filename": filename,
                        "user_id": user_id,
                        "assistant_id": assistant_id,
                        "source": "pdf_page",
                        "pdf_page_index": page_idx,
                        "namespace_filename": namespace_filename,
                    },
                }
                documents = await process_text_to_document(
                    metadata=page_media_item["metadata"],
                    user_id=user_id,
                    assistant_id=assistant_id,
                    media_item=page_media_item,
                )
                for d in documents:
                    d.metadata.setdefault("pdf_page_index", page_idx)
                    d.metadata["namespace_filename"]=namespace_filename
                final_documents.extend(documents)

            return final_documents
           
        elif media_type in ("audio", "video"):
            """Diarize audio (or audio extracted from video) using the hosted
            ``transcribe_audio_diarize`` helper and route the structured
            transcript through ``process_dialogue_json_to_documents``.

            Reference audio is a special case: we never run the diarizer on
            it (the goal is to anchor a known speaker for *future* uploads).
            We persist the reference audio URI under ``(user_id, assistant_id,
            "reference_audio")`` so the next non-reference upload can use it.
            """
            reference_audio = metadata.get("reference_audio", False)
            if not reference_audio:
                reference_namespace = (user_id, assistant_id, "reference_audio")
                ref_item = await store.aget(reference_namespace, key=assistant_id)
                if not ref_item and not reference_audio:
                    return [
                        Document(
                            page_content=f"{media_type.capitalize()} missing reference audio reference audio is required for audio and video distillation to text.",
                            metadata={
                                "status": "error",
                                "error": f"missing_{media_type}_reference_audio",
                                "filename": filename,
                            },
                        )
                    ]

            payload_uri = _full_data_uri_from_media_dict(media_item)
            audio_url = ""
            if isinstance(media_item.get("audio_url"), dict):
                audio_url = (media_item.get("audio_url") or {}).get("url", "").strip()
            video_url = ""
            if isinstance(media_item.get("video_url"), dict):
                video_url = (media_item.get("video_url") or {}).get("url", "").strip()
            content_type = metadata.get("content_type")

            if not payload_uri and not audio_url and not video_url:
                return [
                    Document(
                        page_content=f"[{media_type.capitalize()} missing payload and URL]",
                        metadata={
                            "status": "error",
                            "error": f"missing_{media_type}_payload",
                            "filename": filename,
                        },
                    )
                ]

            if media_type == "audio" and reference_audio and payload_uri and not _is_full_audio_data_uri(payload_uri):
                return [
                    Document(
                        page_content=(
                            "[Reference audio requires media_item.base64_encoded_str "
                            "as the full data:audio/...;base64,... string from upstream]"
                        ),
                        metadata={
                            "status": "error",
                            "error": "invalid_reference_audio_data_uri",
                            "filename": filename,
                            "namespace_filename": namespace_filename,
                        },
                    )
                ]

            # Accumulates every document produced for this upload. For a
            # reference upload it first holds the stored reference document,
            # then the body below extends it with the documents produced by
            # diarizing the original payload_uri. For a normal upload it starts
            # empty and the body extends it before the final return.
            all_documents: List[Document] = []

            # ---------------------------------------------------------------
            # Reference-audio: persist for known-speaker labelling on later
            # uploads. Do not diarize the reference itself. Will be the same text spoken for most people.
            # ---------------------------------------------------------------
            if (media_type == "audio" or media_type == "video") and reference_audio:
                # Reference clip pipeline: extract a single-speaker, speech-bearing
                # mp3 (the dominant speaker by total speech time across diarized,
                # text-bearing segments), then run the existing ``transcribe_audio``
                # heuristics (denoise + highest-RMS N-second window) over that
                # target-only track. Video first extracts the audio track so both
                # branches share the same code path.
                if media_type == "video":
                    audio_uri, audio_name = extract_video_audio_b64(
                        payload_uri, filename
                    )
                else:
                    audio_uri = payload_uri
                    audio_name = filename

                transcription_dict = await isolate_dominant_speaker_audio_b64(
                    audio_uri,
                    context=runtime.context,
                    filename=audio_name,
                    content_type="audio/mp3",
                    reference_audio=reference_audio
                )

                # The helper returns a coherent triple: the encoded mp3 of the
                # dominant speaker's clip, its duration, and the transcript
                # ``text`` that matches that clip (same key as the OpenAI
                # transcription API and ``transcribe_audio_diarize``).
                ref_payload_uri = transcription_dict.get("audio_base64_preprocessed", "")
                transcription_text = transcription_dict.get("text") or ""
                ref_duration = transcription_dict.get("duration")

                ref_namespace = (user_id, assistant_id, "reference_audio")
                doc = Document(
                    page_content=transcription_text,
                    metadata={
                        "user_id": user_id,
                        "assistant_id": assistant_id,
                        "created_at": datetime.now(tz=timezone.utc).isoformat(),
                        "processing_task_id": str(uuid4()),
                        "type": "audio",
                        "reference_audio": True,
                        "duration": ref_duration,
                        "filename": filename,
                        "namespace": "reference_audio",
                        "vectorstore_acceptable": False,
                        "adapter_acceptable": False,
                        "analysis_acceptable": False,
                        "namespace_filename": namespace_filename,
                    },
                )
                await store.aput(
                    ref_namespace,
                    key=assistant_id,
                    value={
                        "reference_audio_data": ref_payload_uri,
                        "document": doc.to_json(),
                    },
                )
                all_documents.append(doc)

                """ Compare the approximate embedding of the transcription to the reference audio embedding """
                # Run the synchronous SentenceTransformer load + encode + similarity
                # off the event loop: it is CPU/GPU-bound and would otherwise freeze
                # the single asyncio loop, starving the media-job SSE stream and any
                # concurrent request (e.g. the progress endpoint's auth call).
                def _compute_reference_similarity() -> Any:
                    from sentence_transformers import SentenceTransformer

                    model = SentenceTransformer(runtime.context.embedding_model)
                    embedding = model.encode(transcription_text)
                    reference_audio_sentence = "The quick fox jumped over the brown lazy dog."
                    reference_audio_embedding = model.encode(reference_audio_sentence)
                    return model.similarity(embedding, reference_audio_embedding)

                similarity = await asyncio.to_thread(_compute_reference_similarity)
                if similarity >= 0.80:
                    # The reference upload IS the calibration sentence, so it
                    # carries no real content beyond the voice sample. We've
                    # already stored the reference clip for known-speaker
                    # labelling, so we're done: return just the reference doc.
                    logger.info(
                        "Reference audio matches calibration sentence (similarity=%s); skipping content processing",
                        similarity,
                    )
                    return all_documents

                # Dissimilar -> the reference upload carries real content. Treat
                # the original audio as a normal (non-reference) upload from
                # here on: flip the flag so the diarization body below pulls the
                # reference clip we just stored (for known-speaker labelling),
                # diarizes the full payload_uri, identifies speakers, and routes
                # multi-speaker dialogue / target monologue / non-target
                # biographical facts exactly like any other audio document
                # rather than blanket-labelling every segment as the target.
                # The body extends ``all_documents`` (already holding the
                # reference doc) and returns it.
                logger.info(
                    "Reference audio dissimilar to calibration sentence (similarity=%s); diarizing original payload as a non-reference upload",
                    similarity,
                )
                reference_audio = False

            # ---------------------------------------------------------------
            # Pull the stored reference audio (if any) so the diarizer can
            # label the target speaker via known_speaker_references.
            # ---------------------------------------------------------------
            encoded_reference_audio = None
            try:
                ref_item = await store.aget(
                    (user_id, assistant_id, "reference_audio"), assistant_id
                )
                if ref_item is not None:
                    encoded_reference_audio = (
                        getattr(ref_item, "value", {}) or {}
                    ).get("reference_audio_data") or None
            except Exception as exc:
                logger.debug("Reference audio lookup failed (continuing): %s", exc)

            if not payload_uri:
                # URL-only path - the upload pipeline currently base64-encodes
                # bytes for the file path before reaching here, so this is
                # only hit when a URL was passed without bytes. The
                # URLDocumentLoaderClass.load() YouTube path already returns a
                # base64 payload in payload_uri.
                logger.info(
                    "%s %s has only a URL and no inline bytes; diarization skipped",
                    media_type,
                    filename,
                )
                all_documents.append(
                    Document(
                        page_content=f"[{media_type.capitalize()} URL pointer: {audio_url or video_url}]",
                        metadata={
                            "user_id": user_id,
                            "assistant_id": assistant_id,
                            "created_at": datetime.now(tz=timezone.utc).isoformat(),
                            "type": media_type,
                            "filename": filename,
                            "vectorstore_acceptable": False,
                            "status": "url_only_no_bytes",
                        },
                    )
                )
                return all_documents
            try:
                diar_response = await transcribe_audio_diarize(
                    media_base64=payload_uri,
                    context=runtime.context,
                    encoded_reference_audio=encoded_reference_audio,
                    filename=filename,
                    content_type=content_type,
                )
            except Exception as e:
                logger.exception(
                    "transcribe_audio_diarize failed for %s: %s; falling back to plain transcribe",
                    filename,
                    e,
                )
                diar_response = None

            target_speaker_label = (
                runtime.context.audio_diarization_known_speaker_name or "avatar"
            )

            # ---------------------------------------------------------------
            # Normalize diarizer segments, then coalesce consecutive
            # same-speaker segments into alternating turns. The number of
            # DISTINCT speaker labels decides the routing:
            #   * >1 speaker  -> full dialogue processing (quote turns +
            #     role-converted adapter conversation + biographical facts
            #     about the target from non-target turns).
            #   * 1 speaker   -> gate on is_target. The target's own speech is
            #     classified (monologue/tweets) into the quote namespace via
            #     process_text_to_document; a non-target lone speaker only ever
            #     yields biographical facts about the target (and nothing when
            #     they say nothing about the target).
            #   * 0 usable    -> transcribe once and apply the same is_target
            #     (reference-upload) gate.
            # ---------------------------------------------------------------
            segments_raw = (diar_response or {}).get("segments") or []
            normalized_segments: List[Dict[str, Any]] = []
            for seg in segments_raw:
                if not isinstance(seg, dict):
                    continue
                seg_text = (seg.get("text") or "").strip()
                if not seg_text:
                    continue
                speaker_label = seg.get("speaker") or "unknown"
                is_target = (
                    encoded_reference_audio is not None
                    and target_speaker_label is not None
                    and target_speaker_label.lower() in str(speaker_label).lower()
                )
                normalized_segments.append(
                    {
                        "speaker": str(speaker_label),
                        "text": seg_text,
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "is_target": bool(is_target) or bool(reference_audio),
                    }
                )

            turns = coalesce_segments_by_speaker(normalized_segments)
            distinct_speakers = {t["speaker"] for t in turns}

            # Lone-speaker promotion. If the diarizer returned exactly one
            # speaker and we supplied a stored reference audio (i.e. the user
            # previously registered their voice), the lone speaker is the
            # target — even if the diarizer's label didn't literally contain
            # ``target_speaker_label``. The reference audio is the only voice
            # we have a sample of; if there's one speaker on this upload and
            # we anchored them via known_speaker_references, the upload IS the
            # target. This rescues the case where the diarizer ignores or
            # mangles the provided ``known_speaker_names`` label.
            if (
                len(distinct_speakers) == 1
                and encoded_reference_audio is not None
                and turns
                and not turns[0].get("is_target")
            ):
                logger.info(
                    "Lone speaker %r promoted to target via stored reference audio",
                    turns[0].get("speaker"),
                )
                for t in turns:
                    t["is_target"] = True

            if len(distinct_speakers) > 1:
                # Multiple speakers -> full dialogue processing. Outputs:
                # multi-turn adapter conversation, per-target quote Documents
                # (each carrying its preceding non-target turn as the prompt),
                # and biographical facts about the target from non-target turns.
                dialogue_payload = {
                    "segments": turns,
                    "target_name": target_speaker_label,
                }
                dialogue_media_item = {
                    "type": "dialogue",
                    "content": (diar_response or {}).get("text", ""),
                    "metadata": {
                        "filename": filename,
                        "user_id": user_id,
                        "assistant_id": assistant_id,
                        "source": (
                            metadata.get("source") or filename or f"{media_type}_upload"
                        ),
                        "diarization_model": (diar_response or {}).get("model"),
                        "namespace_filename": namespace_filename,
                    },
                }
                documents = await process_dialogue_json_to_documents(
                    dialogue_payload=dialogue_payload,
                    user_id=user_id,
                    assistant_id=assistant_id,
                    media_item=dialogue_media_item,
                )
                for d in documents:
                    d.metadata.setdefault("audio_filename", filename)
                    d.metadata["namespace_filename"] = namespace_filename
                all_documents.extend(documents)
                return all_documents

            if len(distinct_speakers) == 1:
                # Single speaker -> gate on whether it is the target.
                turn = turns[0]
                single_media_item = {
                    "type": "text",
                    "content": turn["text"],
                    "metadata": {
                        "filename": filename,
                        "user_id": user_id,
                        "assistant_id": assistant_id,
                        "source": (
                            metadata.get("source") or filename or f"{media_type}_transcription"
                        ),
                        "namespace_filename": namespace_filename,
                    },
                }
                if turn["is_target"]:
                    # The avatar's own speech: classify (monologue/tweets) into
                    # the quote namespace via the shared text classifier.
                    documents = await process_text_to_document(
                        metadata=single_media_item["metadata"],
                        user_id=user_id,
                        assistant_id=assistant_id,
                        media_item=single_media_item,
                    )
                else:
                    # A non-target lone speaker: only biographical facts about
                    # the target, never quotes; no document if nothing is said
                    # about the target.
                    documents = await process_nontarget_text_to_identity_documents(
                        text_content=turn["text"],
                        user_id=user_id,
                        assistant_id=assistant_id,
                        media_item=single_media_item,
                        target_name=target_speaker_label,
                    )
                for d in documents:
                    d.metadata.setdefault("audio_filename", filename)
                    d.metadata["namespace_filename"] = namespace_filename
                all_documents.extend(documents)
                return all_documents

            # ---------------------------------------------------------------
            # No usable diarized segments: transcribe once and apply the same
            # is_target gate. Only a reference-audio upload (this upload IS the
            # target sample) is treated as the target; otherwise the speaker is
            # unidentified and yields biographical facts only.
            # ---------------------------------------------------------------
            try:
                fallback = await transcribe_audio(
                    audio_base64=payload_uri,
                    context=runtime.context,
                    filename=filename,
                )
                fallback_text = (fallback.get("text") or "").strip()
            except Exception as e:
                logger.exception(
                    "Fallback transcription failed for %s: %s", filename, e
                )
                fallback_text = ""

            if not fallback_text:
                all_documents.append(
                    Document(
                        page_content=f"[{media_type.capitalize()} transcription unavailable]",
                        metadata={
                            "user_id": user_id,
                            "assistant_id": assistant_id,
                            "created_at": datetime.now(tz=timezone.utc).isoformat(),
                            "type": media_type,
                            "filename": filename,
                            "vectorstore_acceptable": False,
                            "status": "transcription_unavailable",
                            "namespace_filename": namespace_filename,
                        },
                    )
                )
                return all_documents

            transcript_media_item = {
                "type": "text",
                "content": fallback_text,
                "metadata": {
                    "filename": filename,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "source": (
                        metadata.get("source") or filename or f"{media_type}_transcription"
                    ),
                    "namespace_filename": namespace_filename,
                },
            }
            # Treat as target speech when this IS the reference upload OR
            # when the user has previously registered a reference audio
            # (single-speaker uploads after registration are presumed to be
            # the target — same rationale as the lone-speaker promotion in
            # the diarized-segments branch above). The non-target identity-
            # facts path only kicks in for unidentified-speaker uploads with
            # no stored reference audio.
            if reference_audio or encoded_reference_audio is not None:
                documents = await process_text_to_document(
                    metadata=transcript_media_item["metadata"],
                    user_id=user_id,
                    assistant_id=assistant_id,
                    media_item=transcript_media_item,
                )
            else:
                documents = await process_nontarget_text_to_identity_documents(
                    text_content=fallback_text,
                    user_id=user_id,
                    assistant_id=assistant_id,
                    media_item=transcript_media_item,
                    target_name=target_speaker_label,
                )
            for d in documents:
                d.metadata.setdefault("audio_filename", filename)
                d.metadata["namespace_filename"] = namespace_filename
            all_documents.extend(documents)
            return all_documents

        else:
            logger.warning(f"Unsupported media type: {media_type}")
            docs = [Document(
                page_content=f"[Unsupported media type: {media_type}]",
                metadata={"type": media_type, "status": "unsupported", "namespace_filename": namespace_filename}
            )]
            return docs
    
    except Exception as e:
        # ERROR DOCUMENT
        logger.error(f"Error processing media item: {e}")
        documents =  [Document(
            page_content=f"[Error processing media: {str(e)}]",
            metadata={"type": media_type, "status": "error", "error": str(e)}
        )]
        return documents
    finally:
        if leaf_acquired:
            semaphore.release()


async def _expand_url_media_item(
    media_item: Dict[str, Any],
    runtime: Runtime[GlobalContext],
    config: RunnableConfig,
    store: BaseStore,
    *,
    url: str,
    filename: str,
    namespace_filename: str,
    user_id: str,
    assistant_id: str,
    existing_namespaces: Optional[set] = None,
    semaphore: Optional["asyncio.Semaphore"] = None,
) -> List[Document]:
    """Expand a ``type="url"`` item (YouTube subs/audio, playlist, article, tweet,
    linktree) into Documents.

    Concurrency contract: a semaphore slot is held only around the heavy
    ``loader.load()`` call, never while awaiting the recursive children — children
    take their own slots, so the bounded pool can't deadlock. Each child carries
    its own ``namespace_filename`` (playlist entries: ``{playlist}::{video}``,
    linktree: the child href); ones already present in ``existing_namespaces`` are
    skipped so a re-upload doesn't reprocess hundreds of indexed items.
    """
    if not url:
        return [
            Document(
                page_content="[Empty URL]",
                metadata={
                    "type": "url",
                    "status": "error",
                    "error": "empty_url",
                    "filename": filename,
                    "namespace_filename": namespace_filename,
                },
            )
        ]

    loader = URLDocumentLoaderClass()
    if semaphore is not None:
        async with semaphore:
            expanded_items = await loader.load(
                url, user_id=user_id, assistant_id=assistant_id
            )
    else:
        expanded_items = await loader.load(
            url, user_id=user_id, assistant_id=assistant_id
        )

    if not expanded_items:
        return [
            Document(
                page_content=f"[URL produced no content: {url}]",
                metadata={
                    "type": "url",
                    "status": "error",
                    "error": "url_no_content",
                    "filename": filename,
                },
            )
        ]

    existing = existing_namespaces or set()
    total_children = len(expanded_items)
    _emit_media_progress("expanding", url=url, total=total_children)

    # Resolve each child's namespace_filename, then drop ones already indexed.
    pending: List[Dict[str, Any]] = []
    for item in expanded_items:
        child_meta = item.setdefault("metadata", {})
        child_ns = child_meta.get("namespace_filename")
        if not child_ns:
            # Loader didn't pre-key this child (e.g. single-video subs/audio):
            # derive from its own source so the key matches the top-level item.
            child_source = (
                item.get("url")
                or child_meta.get("source")
                or child_meta.get("filename")
                or url
            )
            child_ns = _namespace_for(child_source)
            child_meta["namespace_filename"] = child_ns
        if child_ns in existing:
            _emit_media_progress(
                "skipped_existing",
                filename=child_meta.get("filename") or url,
                namespace_filename=child_ns,
            )
            continue
        pending.append(item)

    if not pending:
        return []

    async def _run_child(child_index: int, item: Dict[str, Any]) -> List[Document]:
        _emit_media_progress(
            "converting_child",
            current=child_index + 1,
            total=len(pending),
            filename=item.get("metadata", {}).get("filename") or url,
        )
        try:
            return await process_media_item_task(
                item,
                runtime,
                config,
                store,
                existing_namespaces=existing_namespaces,
                semaphore=semaphore,
            )
        except Exception as exc:
            logger.exception(
                "URL child item processing failed for %s: %s",
                item.get("metadata", {}).get("filename") or url,
                exc,
            )
            return []

    results = await asyncio.gather(
        *(_run_child(i, item) for i, item in enumerate(pending))
    )
    collected: List[Document] = []
    for child_docs in results:
        if isinstance(child_docs, list):
            collected.extend(child_docs)
        elif child_docs is not None:
            collected.append(child_docs)
    return collected


async def extract_media_from_message(state: GlobalState, runtime: Runtime[GlobalContext]):
    
    logger.info(f"Extract_media_from_message NODE")
    if isinstance(runtime.context.user_ctx, dict):
        user_id = runtime.context.user_ctx.get("user_id", "")
    else:
        user_id = getattr(runtime.context.user_ctx, "user_id", "")

    if isinstance(runtime.context.assistant_ctx, dict):
        assistant_id = runtime.context.assistant_ctx.get("assistant_id", "")
    else:
        assistant_id = getattr(runtime.context.assistant_ctx, "assistant_id", "")

    messages = state.get('messages', [])

    if not messages:
        logger.warning("No messages found in state")
        return {"media_list": []}
    
    logger.info(f"Processing {len(messages)} messages")

    # Get the most recent HumanMessage
    recent_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            recent_message = msg
            break
    
    if not recent_message:
        logger.info("No HumanMessage found")
        return {"media_list": []}

    content = recent_message.content

    # Handle string content (no media)
    if isinstance(content, str):
        logger.info("Message contains only text, no media")
        return {"media_list": []}
    
    # Handle list content (may contain media)
    if isinstance(content, list):
        logger.info(f"Message content has {len(content)} items")

        # Extract media (skip first item if it's text)
        media_list = []
        for item in content: 
            if isinstance(item, dict):
                item_type = item.get("type", "")

                # Skip pure text items
                if item_type == "text":
                    continue

                # Add media items
                if item_type in ["image", "image_url", "audio", "video", "url"]:
                    normalized = dict(item)
                    legacy_payload = normalized.pop("data", None)
                    if legacy_payload is not None and not (
                        normalized.get("base64_encoded_str") or ""
                    ).strip():
                        normalized["base64_encoded_str"] = legacy_payload
                    if "metadata" not in normalized or not isinstance(
                        normalized.get("metadata"), dict
                    ):
                        normalized["metadata"] = {}
                    normalized["metadata"]["user_id"] = user_id
                    normalized["metadata"]["assistant_id"] = assistant_id
                    media_list.append(normalized)
                    # EACH ITEM NEEDS USER_ID AND ASSISTANT_ID FROM CONTEXT
                    # user_id
                    # assistant_id
            
        logger.info(f"Extracted {len(media_list)} media items")
        return {"media_list": media_list}

    logger.warning(f"Unexpected content type: {type(content)}")
    return {"media_list": []}


# ---------------------------------------------------------------------------
# Adapter dataset writer
# ---------------------------------------------------------------------------

async def process_adapter_documents(
    state: GlobalState,
    runtime: Runtime[GlobalContext],
    config: RunnableConfig,
    store: BaseStore,
) -> Dict[str, Any]:
    """Persist adapter-training rows for any newly-classified Documents.

    Two source kinds:

    * Dialogue Documents (``namespace == "adapter"``): the page_content is
      already a ``{"messages": [...]}`` JSON blob produced by
      :func:`process_dialogue_json_to_documents`. We append it to
      ``data/<assistant_id>/adapter_training.jsonl`` as-is and write the
      same row to the LangGraph store under ``(user_id, assistant_id,
      "adapter")`` so retrieval can later use it.

    * Quote Documents (``namespace == "quote"``, ``adapter_acceptable=True``):
      grouped by source filename, each group produces single-turn
      ``(synthetic_question, verbatim_quote)`` adapter rows plus matching
      LangSmith dataset rows via :func:`build_adapter_and_langsmith_for_quotes`.

    The node is idempotent because each adapter Document carries its own
    ``document_id``; rerunning over the same input simply rewrites the JSONL
    line for that id.
    """

    logger.info("process_adapter_documents NODE")

    adapter_docs: List[Document] = list(
        state.get("documents_to_be_processed_for_adapter_training") or []
    )
    if not adapter_docs:
        logger.info("No adapter Documents queued; skipping")
        return {}

    user_id, assistant_id = await extract_user_id_assistant_id(config)

    # Lazy import so optional helpers don't slow the graph cold start.
    from src.anubis.utils.dataset.formatting import (
        build_adapter_and_langsmith_for_quotes,
        build_langsmith_for_conversation,
        llm_single_turn_dataset,
    )

    out_dir = os.path.join("data", assistant_id)
    os.makedirs(out_dir, exist_ok=True)
    adapter_jsonl = os.path.join(out_dir, "adapter_training.jsonl")
    langsmith_jsonl = os.path.join(out_dir, "langsmith_eval_dataset.jsonl")

    adapter_rows_total: List[Dict[str, Any]] = []
    langsmith_rows_total: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 1) Dialogue conversations: pass-through (already role-converted).
    # ------------------------------------------------------------------
    dialogue_docs = [
        d for d in adapter_docs if d.metadata.get("namespace") == "adapter"
    ]
    for d in dialogue_docs:
        try:
            payload = json.loads(d.page_content)
        except Exception:
            payload = {"messages": d.metadata.get("messages") or []}
        messages = payload.get("messages") or []
        if not messages:
            continue
        # (1) Multi-turn conversation row for adapter training (pass-through).
        adapter_rows_total.append(payload)
        # (3) Conversation-level LangSmith Q&A pairs derived from the same
        #     role-converted messages.
        try:
            conv_source = (
                d.metadata.get("filename")
                or d.metadata.get("source")
                or "unknown"
            )
            conversation_langsmith_rows = await build_langsmith_for_conversation(
                messages=messages,
                dataset_source_filename=conv_source,
            )
            langsmith_rows_total.extend(conversation_langsmith_rows)
        except Exception as exc:
            logger.warning(
                "Conversation LangSmith build failed for %s: %s",
                d.metadata.get("filename", ""),
                exc,
            )
        try:
            ns = (user_id, assistant_id, "adapter")
            await store.aput(
                ns,
                key=str(d.metadata.get("document_id") or uuid4()),
                value={"document": d.to_json()},
            )
        except Exception as exc:  # pragma: no cover - operator log only
            logger.warning("Failed to put adapter dialogue Document: %s", exc)

    # ------------------------------------------------------------------
    # 2) Quote-shaped Documents: synthesize a question per quote and emit
    #    single-turn adapter + langsmith rows. Group by filename so each
    #    source file becomes one ``langsmith_eval_dataset`` source label.
    # ------------------------------------------------------------------
    quote_docs = [
        d
        for d in adapter_docs
        if d.metadata.get("namespace") == "quote"
        and d.metadata.get("adapter_acceptable") is True
        and (d.page_content or "").strip()
    ]
    grouped: Dict[str, List[Document]] = {}
    for d in quote_docs:
        key = (
            d.metadata.get("filename")
            or d.metadata.get("original_source")
            or d.metadata.get("source")
            or "unknown"
        )
        grouped.setdefault(key, []).append(d)

    for source_key, docs_for_source in grouped.items():
        quotes = [d.page_content.strip() for d in docs_for_source]
        # Use the real preceding non-target turn as the prompt when present
        # (carried as ``adapter_prompt`` by diarized target quotes); otherwise
        # the builder synthesizes a question per quote.
        prompts = [d.metadata.get("adapter_prompt") for d in docs_for_source]
        try:
            adapter_rows, langsmith_rows = await build_adapter_and_langsmith_for_quotes(
                quotes=quotes,
                dataset_source_filename=source_key,
                prompts=prompts,
            )
        except Exception as exc:
            logger.exception(
                "Adapter+LangSmith builder failed for %s: %s; falling back to no-question pairs",
                source_key,
                exc,
            )
            adapter_rows = await llm_single_turn_dataset(
                question_list=[""] * len(quotes), answer_list=quotes
            )
            langsmith_rows = []
        adapter_rows_total.extend(adapter_rows)
        langsmith_rows_total.extend(langsmith_rows)

    # ------------------------------------------------------------------
    # 3) Persist to disk (append-only JSONL).
    # ------------------------------------------------------------------
    if adapter_rows_total:
        with open(adapter_jsonl, "a", encoding="utf-8") as fh:
            for row in adapter_rows_total:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(
            "Wrote %d adapter rows -> %s",
            len(adapter_rows_total),
            adapter_jsonl,
        )

    if langsmith_rows_total:
        with open(langsmith_jsonl, "a", encoding="utf-8") as fh:
            for row in langsmith_rows_total:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(
            "Wrote %d langsmith eval rows -> %s",
            len(langsmith_rows_total),
            langsmith_jsonl,
        )

    return {
        "documents_to_be_processed_for_adapter_training": [],
    }


# ---------------------------------------------------------------------------
# Profile build trigger node
# ---------------------------------------------------------------------------

async def build_stylistic_fingerprint(
    state: GlobalState,
    runtime: Runtime[GlobalContext],
    config: RunnableConfig,
    store: BaseStore,
) -> Dict[str, Any]:
    """Build / refresh the per-avatar stylistic + knowledge profiles.

    Threshold logic lives in
    :mod:`src.anubis.utils.dataset.build_profile` and
    :mod:`src.anubis.utils.dataset.build_knowledge_profile`. This node
    delegates to them; the corpus is touched only here, never at evaluation
    time.
    """
    logger.info("build_stylistic_fingerprint NODE")
    try:
        user_id, assistant_id = await extract_user_id_assistant_id(config)
    except Exception as exc:
        logger.warning("Could not resolve user/assistant id: %s", exc)
        return {}

    from src.anubis.utils.dataset.build_profile import maybe_build_stylistic_profile
    from src.anubis.utils.dataset.build_knowledge_profile import (
        maybe_build_knowledge_profile,
    )

    try:
        await maybe_build_stylistic_profile(
            user_id=user_id,
            assistant_id=assistant_id,
            store=store,
            context=runtime.context,
        )
    except Exception as exc:
        logger.exception("Stylistic profile build failed: %s", exc)

    try:
        await maybe_build_knowledge_profile(
            user_id=user_id,
            assistant_id=assistant_id,
            store=store,
            context=runtime.context,
        )
    except Exception as exc:
        logger.exception("Knowledge profile build failed: %s", exc)

    return {}

