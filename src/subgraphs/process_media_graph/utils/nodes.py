# Nodes for Identifying and Handling each media type 

import asyncio
from curses import napms
import asyncio
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


# ``nodes.py`` → ``utils`` → ``process_media_graph`` → ``subgraphs`` → ``src`` → repo root
_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _sanitize_for_filename(source: str, *, max_len: int = 120) -> str:
    """Make ``source`` (a filename or URL) safe to use as a single path component.

    Keep ``[A-Za-z0-9._-]``; replace everything else (path separators, URL
    punctuation, whitespace) with ``_`` and truncate to ``max_len``.
    """
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in (source or ""))
    safe = safe.strip("._-") or "source"
    return safe[:max_len]


def _write_dev_diarization_transcript(
    diar_response: dict,
    *,
    source: str,
    filename: Optional[str],
    assistant_id: Optional[str],
    runtime,
) -> None:
    """Dump the diarized transcript (labeled speakers) when ``DEV=TRUE`` (dev-only aid).

    Mirrors the ``_write_dev_system_prompt`` convention: gated on
    ``context.dev.upper() == "TRUE"``. Writes a clean, human-readable JSON view
    (speaker-labeled segments + full text) to
    ``<repo_root>/data/<assistant_id>/transcriptions/<source>_<timestamp>.json``.
    The large ``encoded_audio_base64`` blob is intentionally dropped. Never raises —
    a dev-artifact write must not break the upload pipeline.
    """
    try:
        context = getattr(runtime, "context", None) or GlobalContext()
        if str(getattr(context, "dev", "") or "").upper() != "TRUE":
            return

        out_dir = os.path.join(
            _PROJECT_ROOT, "data", str(assistant_id or "anonymous"), "transcriptions"
        )
        os.makedirs(out_dir, exist_ok=True)

        from time import time_ns

        out_path = os.path.join(
            out_dir, f"{_sanitize_for_filename(source)}_{time_ns()}.json"
        )

        payload = {
            "source": source,
            "filename": filename,
            "model": diar_response.get("model"),
            "duration": diar_response.get("duration"),
            "text": diar_response.get("text"),
            "segments": [
                {
                    "speaker": str(seg.get("speaker") or "unknown"),
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": seg.get("text"),
                }
                for seg in (diar_response.get("segments") or [])
                if isinstance(seg, dict)
            ],
        }

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
        logger.info("dev diarization transcript written to: %s", out_path)
    except Exception:  # pragma: no cover - dev aid must never break processing
        logger.exception("failed to write dev diarization transcript")

from src.subgraphs.process_media_graph.utils.helper_functions import (
    CLASSIFICATION_INPUT_CHAR_LIMIT,
    build_all_speakers_quote_documents,
    build_golden_transcript,
    coalesce_segments_by_speaker,
    process_dialogue_json_to_documents,
    process_nontarget_text_to_identity_documents,
    process_text_media_item_target_for_vectorstore,
)
from src.subgraphs.process_media_graph.utils.fragment_filter import (
    classify_fragment_heuristic,
    strip_repeated_lines,
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
            create_reference_media_from_playlist = file_data.get("create_reference_media_from_playlist", False)
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
                        "create_reference_media_from_playlist": create_reference_media_from_playlist,
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
                        "reference_audio": reference_audio,
                        "create_reference_media_from_playlist": create_reference_media_from_playlist,
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
                url_metadata = {
                    "filename": filename,
                    "content_type": mime,
                    "size": 0,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "create_reference_media_from_playlist": create_reference_media_from_playlist,
                    "namespace_filename": namespace_filename,
                }
                # Carry playlist context when the upload endpoint expanded a
                # playlist into one page_url entry per video (each its own child
                # job). _expand_url_media_item stamps these onto the produced
                # Documents so /list_avatar_documents groups each video under its
                # playlist and a whole-playlist delete can target them.
                for playlist_key in (
                    "playlist_url",
                    "playlist_namespace_filename",
                    "playlist_title",
                    "video_title",
                    "url_kind",
                ):
                    playlist_value = file_data.get(playlist_key)
                    if playlist_value is not None:
                        url_metadata[playlist_key] = playlist_value
                entry = {
                    "type": "url",
                    "url": page_url_remote,
                    "metadata": url_metadata,
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
                        "create_reference_media_from_playlist": create_reference_media_from_playlist,
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
                        "create_reference_media_from_playlist": create_reference_media_from_playlist,
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

async def analyze_documents(
    state: GlobalState,
    runtime: Runtime[GlobalContext],
    config: RunnableConfig,
    store: BaseStore,
) -> Dict[str, Any]:
    """Fan registered analyzers out over queued docs in parallel; merge results.

    For every ``analysis_acceptable`` Document, run each applicable analyzer
    from :data:`ANALYSIS_SCAFFOLD_RUNNERS`. The default analyzer set is the
    full registry; a Document narrows it by listing analyzer keys in
    ``metadata["analysis_scaffolds"]``. Every analyzer produces
    ``analysis``-namespace Documents which are merged into the vector-store
    index batch so ``index_docs`` persists them alongside the source docs.

    To add a feature, register one analyzer in
    ``src/anubis/utils/analysis/analysis_methods.py`` — no change is needed
    here.
    """
    # Kill switch: skip the whole analysis fan-out (OCEAN, emotional triggers,
    # standardized questions, narrative analyzers) when disabled. Documents are
    # still indexed via the direct convert->index_docs edge; only the analysis
    # branch is short-circuited. Clearing the queue keeps state consistent.
    if (GlobalContext().enable_document_analysis or "TRUE").upper() != "TRUE":
        logger.info(
            "analyze_documents: disabled via ENABLE_DOCUMENT_ANALYSIS; skipping"
        )
        return {
            "documents_to_be_analyzed_for_context_storage_and_prompt_injection_of_assistant": "delete"
        }

    from src.anubis.utils.analysis.analysis_methods import ANALYSIS_SCAFFOLD_RUNNERS

    queue: List[Document] = list(
        state.get(
            "documents_to_be_analyzed_for_context_storage_and_prompt_injection_of_assistant",
        )
        or []
    )
    if not queue:
        logger.info("analyze_documents: empty queue; skipping")
        return {}

    # Build (label, coroutine) pairs across docs × analyzers, then run them all
    # concurrently in a single gather.
    labels: List[str] = []
    coros = []
    for doc in queue:
        scaffolds = doc.metadata.get("analysis_scaffolds") or list(
            ANALYSIS_SCAFFOLD_RUNNERS.keys()
        )
        for name in scaffolds:
            runner = ANALYSIS_SCAFFOLD_RUNNERS.get(name)
            if runner is None:
                logger.warning(
                    "analyze_documents: unknown analyzer %r (filename=%s); skipping",
                    name,
                    doc.metadata.get("filename", ""),
                )
                continue
            labels.append(name)
            coros.append(runner(doc))

    analyzed_documents: List[Document] = []
    if coros:
        results = await asyncio.gather(*coros, return_exceptions=True)
        for name, result in zip(labels, results):
            if isinstance(result, Exception):
                logger.warning(
                    "analyze_documents: analyzer %r failed: %s; continuing",
                    name,
                    result,
                )
                continue
            if result:
                analyzed_documents.extend(result)

    # region agent log
    try:
        from src.anubis.utils.utility import _agent_debug_log as _adl
        _adl(
            "analyze_documents:return",
            {
                "queue_len": len(queue),
                "analyzed_documents_len": len(analyzed_documents),
                "sample_doc_id": (
                    (analyzed_documents[0].metadata or {}).get("document_id")
                    if analyzed_documents else None
                ),
            },
            hypothesis_id="H1",
        )
    except Exception:
        pass
    # endregion

    return {
        "documents_to_be_analyzed_for_context_storage_and_prompt_injection_of_assistant": "delete",
        "vectorstore_documents_to_be_indexed": analyzed_documents,
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
        logger.info(f"No Media to process")
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

    # region agent log
    try:
        from src.anubis.utils.utility import _agent_debug_log as _adl
        _adl(
            "convert_media_list_to_text_document:return",
            {
                "vectorstore_len": len(vector_store_document_list_formatted),
                "analysis_len": len(analysis_document_list_formatted),
                "adapter_len": len(adapter_document_list_formatted),
                "sample_vs_doc_id": (
                    (vector_store_document_list_formatted[0].metadata or {}).get("document_id")
                    if vector_store_document_list_formatted else None
                ),
            },
            hypothesis_id="H1",
        )
    except Exception:
        pass
    # endregion

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
                store = store, 
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

                        # FINAL DOCUMENTS ARE USED TO CALIBRATE THE GROUND_TRUTH_FEATURES
                """ CALIBRATE GROUND TRUTH """
                from src.subgraphs.process_media_graph.utils.calibrate_ground_truth import calibrate_ground_truth
                await calibrate_ground_truth(store=store, assistant_id=assistant_id, documents=final_documents)

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
                # ``layout`` mode preserves column/line structure better than the
                # default; fall back to the plain loader if the installed pypdf
                # signature doesn't accept it.
                try:
                    loader = PyPDFLoader(tmp_path, extraction_mode="layout")
                except TypeError:
                    loader = PyPDFLoader(tmp_path)
                # PyPDFLoader.load() is synchronous and pypdf parsing is
                # CPU-bound; offload so the event loop isn't blocked.
                pages = await asyncio.to_thread(loader.load)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            # Optimize extraction quality before classification/chunking:
            #   1. Strip lines that recur across many pages (running headers /
            #      footers / page numbers) — the root cause of fragments like
            #      "Page 13 of 10".
            #   2. Drop pages whose remaining text is pure page furniture.
            #   3. Concatenate the surviving pages into one continuous body so a
            #      sentence spanning a page break survives and header-only pages
            #      can't become standalone Documents. The whole document is then
            #      classified + chunked once (cross-page coherent).
            raw_page_texts = [(p.page_content or "") for p in pages]
            cleaned_page_texts = strip_repeated_lines(raw_page_texts)

            kept_page_texts: List[str] = []
            first_kept_page_index: Optional[int] = None
            for page_idx, page_text in enumerate(cleaned_page_texts):
                stripped = (page_text or "").strip()
                if not stripped:
                    continue
                if classify_fragment_heuristic(stripped) == "junk":
                    continue
                if first_kept_page_index is None:
                    first_kept_page_index = page_idx
                kept_page_texts.append(stripped)

            if not kept_page_texts:
                logger.warning(
                    "PDF %s yielded no useful page text after cleaning", filename
                )
                return []

            document_text = "\n\n".join(kept_page_texts)
            pdf_media_item = {
                "type": "text",
                "content": document_text,
                "metadata": {
                    "filename": filename,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "source": "pdf_document",
                    "pdf_page_count": len(pages),
                    "pdf_first_content_page_index": first_kept_page_index,
                    "namespace_filename": namespace_filename,
                    # Carry any user-supplied target through so target-aware
                    # extraction runs for PDFs (e.g. a memoir with a named subject).
                    "target_name": metadata.get("target_name"),
                },
            }
            final_documents = await process_text_to_document(
                metadata=pdf_media_item["metadata"],
                user_id=user_id,
                assistant_id=assistant_id,
                media_item=pdf_media_item,
                store=store,
            )
            for d in final_documents:
                d.metadata["namespace_filename"] = namespace_filename

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
            # Batch-wide "no single target": every detected speaker is the avatar.
            # Diarization still runs, but no stored reference clip is required and
            # known-speaker labelling is skipped (every turn is forced is_target).
            create_reference_media_from_playlist = bool(metadata.get("create_reference_media_from_playlist", False))
            if not reference_audio and not create_reference_media_from_playlist:
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
                    audio_uri, audio_name = await asyncio.to_thread(
                        extract_video_audio_b64, payload_uri, filename
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
            if not create_reference_media_from_playlist: # Every entity is the target during create_reference_media_from_playlist
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

            # Reformat the raw diarizer response into the canonical golden
            # transcript: coalesce consecutive same-speaker segments, run the LLM
            # speaker-reconciliation pass (gpt-4o-transcribe-diarize has no
            # speaker-count parameter and over-splits), relabel the target's turns
            # to ``avatar``, and recompute the top-level text. Everything below
            # (and the dev artifact) consumes THIS golden form.
            golden_transcript: Optional[Dict[str, Any]] = None
            if diar_response:
                reconcile_enabled = str(
                    getattr(runtime.context, "enable_speaker_reconciliation", "TRUE")
                    or "TRUE"
                ).upper() == "TRUE"
                golden_transcript = await build_golden_transcript(
                    diar_response,
                    target_speaker_label=target_speaker_label,
                    reference_audio=bool(reference_audio),
                    reconcile=reconcile_enabled,
                )
                # DEV-only: persist the golden transcript for inspection. Placed
                # here so it captures every routing branch below.
                _write_dev_diarization_transcript(
                    golden_transcript,
                    source=(
                        metadata.get("source")
                        or filename
                        or f"{media_type}_transcription"
                    ),
                    filename=filename,
                    assistant_id=assistant_id,
                    runtime=runtime,
                )

            # ---------------------------------------------------------------
            # create_reference_media_from_playlist: no single target speaker, so EVERY detected
            # speaker is the avatar. Routed only through standalone helpers / the
            # normal text classifier so the established dialogue path is never
            # touched. This branch returns early.
            #
            #   * MULTIPLE speakers -> one adapter-eligible ``quote`` Document per
            #     statement, each reusing the PRECEDING statement as its genuine
            #     question (``adapter_prompt``); the first statement has no
            #     predecessor so a question is synthesized for it downstream.
            #   * a SINGLE speaker -> a monologue: classified normally (monologue
            #     / tweets_or_quotes) via process_text_to_document, which already
            #     stores it in the vectorstore, marks it analysis-acceptable, and
            #     makes it adapter-acceptable with a synthesized prompt.
            # ---------------------------------------------------------------
            if create_reference_media_from_playlist:
                statements: List[Dict[str, Any]] = []
                for seg in (diar_response or {}).get("segments") or []:
                    if not isinstance(seg, dict):
                        continue
                    seg_text = (seg.get("text") or "").strip()
                    if not seg_text:
                        continue
                    statements.append(
                        {
                            "speaker": str(seg.get("speaker") or "unknown"),
                            "text": seg_text,
                            "start": seg.get("start"),
                            "end": seg.get("end"),
                            "is_target": True,
                        }
                    )

                if not statements:
                    # Diarization yielded no usable segments — fall back to a
                    # plain transcription pass and treat it as one statement.
                    try:
                        plain = await transcribe_audio(
                            audio_base64=payload_uri,
                            context=runtime.context,
                            filename=filename,
                        )
                        plain_text = (plain.get("text") or "").strip()
                    except Exception as e:
                        logger.exception(
                            "create_reference_media_from_playlist transcription failed for %s: %s",
                            filename,
                            e,
                        )
                        all_documents.append(
                            Document(
                                page_content=f"[{media_type.capitalize()} transcription failed: {e}]",
                                metadata={
                                    "user_id": user_id,
                                    "assistant_id": assistant_id,
                                    "created_at": datetime.now(tz=timezone.utc).isoformat(),
                                    "type": media_type,
                                    "filename": filename,
                                    "vectorstore_acceptable": False,
                                    "status": "error",
                                    "error": f"transcription_failed: {e}",
                                    "namespace_filename": namespace_filename,
                                },
                            )
                        )
                        return all_documents
                    if plain_text:
                        statements = [
                            {
                                "speaker": "unknown",
                                "text": plain_text,
                                "start": None,
                                "end": None,
                                "is_target": True,
                            }
                        ]

                if not statements:
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

                source_label = (
                    metadata.get("source")
                    or filename
                    or f"{media_type}_transcription"
                )
                multi_speaker = len({s["speaker"] for s in statements}) > 1

                if multi_speaker:
                    # Speaker-coalesce into alternating turns so each turn's
                    # predecessor is a clean, genuine question, then build one
                    # quote Q&A Document per statement (each reusing the preceding
                    # statement as its question; the first is synthesized
                    # downstream). No monologue is created for multi-speaker
                    # content — a monologue is the single-speaker case below.
                    turns = coalesce_segments_by_speaker(statements)
                    quote_media_item = {
                        "metadata": {"filename": filename, "source": source_label}
                    }
                    quote_documents = build_all_speakers_quote_documents(
                        turns,
                        user_id=user_id,
                        assistant_id=assistant_id,
                        media_item=quote_media_item,
                        target_name=target_speaker_label,
                        multi_speaker=True,
                    )
                    for d in quote_documents:
                        d.metadata.setdefault("audio_filename", filename)
                        d.metadata["namespace_filename"] = namespace_filename
                    all_documents.extend(quote_documents)
                    return all_documents

                # Single speaker -> a monologue: classify normally (monologue /
                # tweets_or_quotes) over the full transcript, exactly like any
                # other single-speaker target text. That path already stores the
                # content in the vectorstore, marks it analysis-acceptable, and
                # makes it adapter-acceptable (a synthetic prompt is generated
                # downstream) — no custom handling needed here.
                full_text = "\n".join(
                    (s.get("text") or "").strip()
                    for s in statements
                    if (s.get("text") or "").strip()
                ).strip()
                single_media_item = {
                    "type": "text",
                    "content": full_text,
                    "metadata": {
                        "filename": filename,
                        "user_id": user_id,
                        "assistant_id": assistant_id,
                        "source": source_label,
                        "namespace_filename": namespace_filename,
                    },
                }
                documents = await process_text_to_document(
                    metadata=single_media_item["metadata"],
                    user_id=user_id,
                    assistant_id=assistant_id,
                    media_item=single_media_item,
                    store=store,
                )
                for d in documents:
                    d.metadata.setdefault("audio_filename", filename)
                    d.metadata["namespace_filename"] = namespace_filename
                all_documents.extend(documents)
                return all_documents

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
            # Turns come from the already-coalesced, speaker-reconciled golden
            # transcript built above. The target's turns are labeled ``avatar``;
            # ``is_target`` is carried through for the routing gate below.
            turns: List[Dict[str, Any]] = [
                {
                    "speaker": seg["speaker"],
                    "text": seg["text"],
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "is_target": bool(seg.get("is_target")),
                }
                for seg in (golden_transcript or {}).get("segments", [])
                if (seg.get("text") or "").strip()
            ]
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
                        store=store,
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
                # A transcription API failure is a REAL error, not "no speech".
                # Surfacing it as status="error" makes convert_media_list_to_text_document
                # count it in items_error, emit an item_error progress event, and
                # report the file in failed_to_index_files for reprocessing —
                # instead of silently reporting the upload as completed. Transient
                # failures were already retried with backoff in _speech_call_with_retry;
                # reaching here means retries were exhausted or the error is permanent
                # (e.g. insufficient_quota), so the user must see it.
                logger.exception(
                    "Fallback transcription failed for %s: %s", filename, e
                )
                all_documents.append(
                    Document(
                        page_content=f"[{media_type.capitalize()} transcription failed: {e}]",
                        metadata={
                            "user_id": user_id,
                            "assistant_id": assistant_id,
                            "created_at": datetime.now(tz=timezone.utc).isoformat(),
                            "type": media_type,
                            "filename": filename,
                            "vectorstore_acceptable": False,
                            "status": "error",
                            "error": f"transcription_failed: {e}",
                            "namespace_filename": namespace_filename,
                        },
                    )
                )
                return all_documents

            if not fallback_text:
                # Transcription SUCCEEDED but produced no text (genuinely silent /
                # speechless audio). This is not an error — record it as such.
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
                    store=store,
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
    its own ``namespace_filename`` (playlist entries: a uuid5 over
    ``{playlist_ns}::{video_ns}``, linktree: the child href); ones already present
    in ``existing_namespaces`` are
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

    # Batch-wide "no single target": every detected speaker is the avatar. This
    # forces the YouTube loader past its subtitles fast-path onto the audio +
    # diarization path so the avatar's voice is actually transcribed (subtitles
    # carry no speaker turns), and is inherited by every expanded child below.
    parent_create_reference_media_from_playlist = bool(
        (media_item.get("metadata") or {}).get("create_reference_media_from_playlist")
    )

    loader = URLDocumentLoaderClass()
    if semaphore is not None:
        async with semaphore:
            expanded_items = await loader.load(
                url,
                user_id=user_id,
                assistant_id=assistant_id,
                expect_multispeaker=parent_create_reference_media_from_playlist,
            )
    else:
        expanded_items = await loader.load(
            url,
            user_id=user_id,
            assistant_id=assistant_id,
            expect_multispeaker=parent_create_reference_media_from_playlist,
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
    # Children (e.g. playlist videos) inherit the parent URL item's batch-wide
    # "no single target" flag so every expanded video is diarized with every
    # speaker treated as the target.
    pending: List[Dict[str, Any]] = []
    for item in expanded_items:
        child_meta = item.setdefault("metadata", {})
        if parent_create_reference_media_from_playlist:
            child_meta.setdefault("create_reference_media_from_playlist", True)
        child_ns = child_meta.get("namespace_filename")
        if not child_ns:
            # A keyless child is the single logical content of THIS url item
            # (e.g. a video's subtitles or audio). Inherit the parent item's
            # namespace_filename so the content lands under the SAME key as the
            # item. This is what preserves a playlist entry's key (a uuid5 over
            # ``{playlist_ns}::{video_ns}``): deriving a fresh key from the
            # child's own URL here would collapse it to a video-only key and
            # lose the playlist grouping. For a standalone video/article the
            # parent key already equals the per-source key, so this is a no-op.
            child_ns = namespace_filename
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

    # Propagate playlist context onto the produced Documents so the listing can
    # group videos under their playlist and the delete endpoint can target a
    # whole playlist. Only stamp when THIS item carries playlist context (i.e.
    # it is a playlist entry being expanded into its subs/audio content) and
    # never overwrite a child's own namespace_filename (already the uuid5 over
    # ``{playlist_ns}::{video_ns}`` inherited above).
    item_meta = media_item.get("metadata", {}) or {}
    if item_meta.get("playlist_url"):
        playlist_fields = {
            key: item_meta[key]
            for key in (
                "playlist_url",
                "playlist_namespace_filename",
                "playlist_title",
                "video_title",
            )
            if item_meta.get(key) is not None
        }
        for doc in collected:
            if isinstance(getattr(doc, "metadata", None), dict):
                for key, value in playlist_fields.items():
                    doc.metadata.setdefault(key, value)
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
    """Build and persist the three adapter datasets into the runtime store.

    Datasets are written to the LangGraph cross-thread store (NOT disk), one
    entry per source file, keyed by ``source_uuid5 = uuid5(NAMESPACE_URL,
    source_filename)``. Each value is a wrapped dict holding the dataset as a
    JSONL string: ``{"jsonl", "source_filename", "row_count", "created_at"}``.

    Source kinds and outputs:

    * Quote Documents (``namespace == "quote"``, ``adapter_acceptable=True``):
      grouped by source filename. Each verbatim target quote is paired with its
      genuine preceding non-target turn (carried as ``adapter_prompt``); a
      synthetic prompt is generated ONLY where no genuine prompt exists. Via
      :func:`build_adapter_and_langsmith_for_quotes` this yields:
        - single-turn Q&A adapter rows -> ``(user_id, assistant_id,
          "q_and_a_adapter", source_uuid5)`` (for training adapters), and
        - LangSmith example rows -> ``(user_id, assistant_id,
          "langsmith_factual_q_and_a", source_uuid5)`` (for factual testing).

    * Dialogue Documents (``namespace == "adapter"``): the page_content is the
      role-converted ``{"messages": [...]}`` conversation. Genuine preceding
      user turns are reused as prompts (synthetic only where the target led) via
      :func:`pairs_from_conversation`, then formatted with
      :func:`llm_multiturn_dataset_one_conversation` into one conversation row ->
      ``(user_id, assistant_id, "multi_turn_dataset_adapter", source_uuid5)``
      (to attune adapter behaviour).

    Idempotent per source: the store key is the source uuid5, so rerunning a
    source overwrites its dataset entry rather than duplicating it.
    """

    logger.info("process_adapter_documents NODE")

    adapter_docs: List[Document] = list(
        state.get("documents_to_be_processed_for_adapter_training") or []
    )
    if not adapter_docs:
        logger.info("No adapter Documents queued; skipping")
        return {}

    # extract_user_id_assistant_id returns ({"user_id": ...}, {"assistant_id": ...}).
    user_state, assistant_state = await extract_user_id_assistant_id(config)
    user_id = user_state.get("user_id")
    assistant_id = assistant_state.get("assistant_id")

    # Lazy import so optional helpers don't slow the graph cold start.
    from src.anubis.utils.dataset.formatting import (
        build_adapter_and_langsmith_for_quotes,
        llm_multiturn_dataset_one_conversation,
        pairs_from_conversation,
    )

    def _source_of(d: Document) -> str:
        """Source-file label that ties a doc's datasets to one store key."""
        return (
            d.metadata.get("filename")
            or d.metadata.get("original_source")
            or d.metadata.get("source")
            or "unknown"
        )

    async def _store_dataset(
        dataset_type: str,
        source_filename: str,
        source_uuid5: str,
        rows: List[Dict[str, Any]],
    ) -> None:
        """Persist one dataset as a JSONL-in-dict value under its namespace."""
        if not rows:
            return
        jsonl = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        try:
            await store.aput(
                (user_id, assistant_id, dataset_type, source_uuid5),
                key=source_uuid5,
                value={
                    "jsonl": jsonl,
                    "source_filename": source_filename,
                    "row_count": len(rows),
                    "created_at": datetime.now(tz=timezone.utc).isoformat(),
                },
            )
            logger.info(
                "Stored %d %s rows -> (%s, %s, %s, %s)",
                len(rows),
                dataset_type,
                user_id,
                assistant_id,
                dataset_type,
                source_uuid5,
            )
        except Exception as exc:  # pragma: no cover - operator log only
            logger.warning(
                "Failed to store %s dataset for %s: %s",
                dataset_type,
                source_filename,
                exc,
            )

    # ------------------------------------------------------------------
    # 1) Quote Documents -> Q&A adapter dataset + LangSmith factual dataset.
    #    Grouped by source so each file becomes one store entry per dataset.
    #    Genuine ``adapter_prompt`` is reused; synthetic prompts fill only gaps.
    # ------------------------------------------------------------------
    quote_docs = [
        d
        for d in adapter_docs
        if d.metadata.get("namespace") == "quote"
        and d.metadata.get("adapter_acceptable") is True
        and (d.page_content or "").strip()
    ]
    grouped_quotes: Dict[str, List[Document]] = {}
    for d in quote_docs:
        grouped_quotes.setdefault(_source_of(d), []).append(d)

    for source_filename, docs_for_source in grouped_quotes.items():
        source_uuid5 = str(uuid5(NAMESPACE_URL, source_filename))
        quotes = [d.page_content.strip() for d in docs_for_source]
        prompts = [d.metadata.get("adapter_prompt") for d in docs_for_source]
        try:
            adapter_rows, langsmith_rows = await build_adapter_and_langsmith_for_quotes(
                quotes=quotes,
                dataset_source_filename=source_filename,
                prompts=prompts,
            )
        except Exception as exc:
            logger.exception(
                "Adapter+LangSmith builder failed for %s: %s; skipping source",
                source_filename,
                exc,
            )
            continue
        await _store_dataset(
            "q_and_a_adapter", source_filename, source_uuid5, adapter_rows
        )
        await _store_dataset(
            "langsmith_factual_q_and_a",
            source_filename,
            source_uuid5,
            langsmith_rows,
        )

    # ------------------------------------------------------------------
    # 2) Dialogue Documents -> multi-turn adapter dataset (one conversation).
    #    Each role-converted conversation reuses genuine user turns as prompts;
    #    a synthetic prompt is generated only where the target led.
    # ------------------------------------------------------------------
    dialogue_docs = [
        d for d in adapter_docs if d.metadata.get("namespace") == "adapter"
    ]
    grouped_dialogue: Dict[str, List[Document]] = {}
    for d in dialogue_docs:
        grouped_dialogue.setdefault(_source_of(d), []).append(d)

    for source_filename, docs_for_source in grouped_dialogue.items():
        source_uuid5 = str(uuid5(NAMESPACE_URL, source_filename))
        conversation_rows: List[Dict[str, Any]] = []
        for d in docs_for_source:
            try:
                payload = json.loads(d.page_content)
            except Exception:
                payload = {"messages": d.metadata.get("messages") or []}
            messages = payload.get("messages") or []
            if not messages:
                continue
            try:
                question_list, answer_list = await pairs_from_conversation(messages)
                if not answer_list:
                    continue
                conversation_rows.append(
                    llm_multiturn_dataset_one_conversation(
                        question_list=question_list, answer_list=answer_list
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Multi-turn build failed for %s: %s",
                    source_filename,
                    exc,
                )
        await _store_dataset(
            "multi_turn_dataset_adapter",
            source_filename,
            source_uuid5,
            conversation_rows,
        )

    # Clear the adapter buffer of what we just processed (append-reducer
    # semantics: an empty list would leave processed docs queued).
    return {
        "documents_to_be_processed_for_adapter_training": "delete",
    }
