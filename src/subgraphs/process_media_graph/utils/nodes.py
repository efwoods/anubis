# Nodes for Identifying and Handling each media type 

from curses import napms
from datetime import datetime, timezone
from typing import Any, Dict, List

from langchain_core.documents import Document
from uuid import uuid4
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

from src.subgraphs.process_media_graph.utils.helper_functions import (
    CLASSIFICATION_INPUT_CHAR_LIMIT,
    process_dialogue_json_to_documents,
    process_text_media_item_target_for_vectorstore,
)
from src.anubis.utils.classes.ReferenceDocumentClassificationClass import (
    ReferenceDocumentClassificationClass,
)
from src.anubis.utils.classes.URLDocumentLoaderClass import URLDocumentLoaderClass
from src.anubis.utils.utility import transcribe_audio, transcribe_audio_diarize

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

    logger.info(f"Processing {len(media_list)} media items")

    # Create tasks for parallel processing
    all_documents: list = []
    processing_errors: list[str] = []
    for media_item in media_list:
        docs = await process_media_item_task(media_item, runtime, config, store)

        for doc in docs:
            status = doc.metadata.get("status", "")
            if status == "error":
                error = doc.metadata.get("error", "")
                filename = doc.metadata.get("filename", "")
                processing_errors.append(f"{filename}: {error or 'unknown'}")
                logger.warning(
                    "Error processing media: %s %s", filename, error
                )
            else:
                all_documents.append(doc)

    if processing_errors:
        raise RuntimeError(
            "Media processing failed: " + "; ".join(processing_errors)
        )

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
    store: BaseStore
) -> Document:
    """Task: Convert a single media item to a Document"""
    
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
        # Handle URLs - YouTube subs/audio, articles, tweets, linktrees.
        elif media_type == "url":
            url = (media_item.get("url") or "").strip()
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
            expanded_items = await loader.load(
                url,
                user_id=user_id,
                assistant_id=assistant_id,
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
            collected: List[Document] = []
            for item in expanded_items:
                # Each expanded item already has user_id / assistant_id baked
                # into its metadata; recurse into the same task. Linktree
                # children come back as ``type="url"`` so they pass back
                # through this branch.
                try:
                    item["metadata"]["namespace_filename"] = namespace_filename
                    child_docs = await process_media_item_task(
                        item, runtime, config, store
                    )
                except Exception as exc:
                    logger.exception(
                        "URL child item processing failed for %s: %s",
                        item.get("metadata", {}).get("filename") or url,
                        exc,
                    )
                    continue
                if isinstance(child_docs, list):
                    collected.extend(child_docs)
                elif child_docs is not None:
                    collected.append(child_docs)
            return collected
        
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
                    return Document(
                        page_content=f"{media_type.capitalize()} missing reference audio reference audio is required for audio and video distillation to text.",
                        metadata={
                            "status": "error",
                            "error": f"missing_{media_type}_reference_audio",
                            "filename": filename,
                        },
                    )

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

            # ---------------------------------------------------------------
            # Reference-audio: persist for known-speaker labelling on later
            # uploads. Do not diarize the reference itself. Will be the same text spoken for most people.
            # ---------------------------------------------------------------
            if (media_type == "audio" or media_type == "video") and reference_audio:
                # Create 9 second reference clip:
                all_documents: List[Document] = []
                if media_type == "audio":
                    """ transcribe reference audio """
                    transcription_dict = await transcribe_audio(
                        audio_base64=payload_uri,
                        context=runtime.context,
                        filename=filename,
                    )
                else:
                    """ transcribe reference video """
                    transcription_dict = await transcribe_video(
                        video_base64=payload_uri,
                        context=runtime.context,
                        filename=filename,
                    )
                
                # Update the payload_uri to the preprocessed audio base64 (now mp3 codec)
                payload_uri = transcription_dict.get("audio_base64_preprocessed", "")
                transcription_text = transcription_dict.get("content", "")
                
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
                        "filename": filename,
                        "namespace": "reference_audio",
                        "vectorstore_acceptable": False,
                        "adapter_acceptable": False,
                        "analysis_acceptable": False,
                        "namespace_filename": namespace_filename,
                    },
                )
                if payload_uri:
                    await store.aput(
                        ref_namespace,
                        key=assistant_id,
                        value={
                            "reference_audio_data": payload_uri,
                            "document": doc.to_json(),
                        },
                    )
                all_documents.append(doc)

                """ Compare the approximate embedding of the transcription to the reference audio embedding """     
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(runtime.context.embedding_model)
                embedding = model.encode(transcription_text)
                REFERENCE_AUDIO_SENTENCE = "The quick fox jumped over the brown lazy dog."
                REFERENCE_AUDIO_EMBEDDING = model.encode(REFERENCE_AUDIO_SENTENCE)

                similarity = model.similarity(embedding, REFERENCE_AUDIO_EMBEDDING)
                if similarity < 0.80:
                    logger.info(f"Transcription text is dissimilar to the reference audio text. Similarity: {similarity}")
                    media_item_transcription = media_item.copy()
                    media_item_transcription["content"] = transcription_text

                    documents = await process_text_to_document(
                        metadata=metadata,
                        user_id=user_id,
                        assistant_id=assistant_id,
                        media_item=media_item_transcription,
                    )
                    all_documents.extend(documents)

                # return the document
                return all_documents

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
                return [
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
                ]

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
            # If diarization succeeded with usable segments, route as dialogue
            # so target turns become quote Documents, the role-converted
            # conversation lands in the adapter namespace, and biographical
            # facts about the target are extracted from the whole transcript.
            # ---------------------------------------------------------------
            segments_raw = (diar_response or {}).get("segments") or []
            if segments_raw:
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

                if normalized_segments and any(
                    seg["is_target"] for seg in normalized_segments
                ):
                    dialogue_payload = {
                        "segments": normalized_segments,
                        "target_name": target_speaker_label,
                    }
                    dialogue_media_item = {
                        "type": "dialogue",
                        "content": diar_response.get("text", ""),
                        "metadata": {
                            "filename": filename,
                            "user_id": user_id,
                            "assistant_id": assistant_id,
                            "source": (
                                metadata.get("source") or filename or f"{media_type}_upload"
                            ),
                            "diarization_model": diar_response.get("model"),
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
                        d.metadata["namespace_filename"]=namespace_filename
                    if documents:
                        return documents
                # Diarized but no target turn detected - treat the whole
                # transcript as a monologue under quote.
                full_text = (diar_response or {}).get("text", "")
                if full_text.strip():
                    transcript_media_item = {
                        "type": "text",
                        "content": full_text,
                        "metadata": {
                            "filename": filename,
                            "user_id": user_id,
                            "assistant_id": assistant_id,
                            "source": (
                                metadata.get("source") or filename or f"{media_type}_transcription"
                            ),
                        },
                    }
                    documents = await process_text_to_document(
                        metadata=transcript_media_item["metadata"],
                        user_id=user_id,
                        assistant_id=assistant_id,
                        media_item=transcript_media_item,
                    )
                    for d in documents:
                        d.metadata.setdefault("audio_filename", filename)
                        d.metadata["namespace_filename"]=namespace_filename
                    return documents

            # ---------------------------------------------------------------
            # Fallback: no diarized output - try plain transcription so
            # something still lands in the namespaces. Treat as monologue.
            # ---------------------------------------------------------------
            try:
                fallback = await transcribe_audio(
                    audio_base64=payload_uri,
                    context=runtime.context,
                    filename=filename,
                )
                fallback_text = (fallback.get("content") or "").strip()
            except Exception as e:
                logger.exception(
                    "Fallback transcription failed for %s: %s", filename, e
                )
                fallback_text = ""

            if not fallback_text:
                return [
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
                ]

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
            documents = await process_text_to_document(
                metadata=transcript_media_item["metadata"],
                user_id=user_id,
                assistant_id=assistant_id,
                media_item=transcript_media_item,
            )
            for d in documents:
                d.metadata.setdefault("audio_filename", filename)
                d.metadata["namespace_filename"]=namespace_filename
            return documents
        
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
        if not payload.get("messages"):
            continue
        adapter_rows_total.append(payload)
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
        try:
            adapter_rows, langsmith_rows = await build_adapter_and_langsmith_for_quotes(
                quotes=quotes,
                dataset_source_filename=source_key,
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

