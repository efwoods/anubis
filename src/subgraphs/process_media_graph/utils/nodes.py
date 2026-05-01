# Nodes for Identifying and Handling each media type 

from typing import Dict
from datetime import datetime, timezone
from langchain_core.documents import Document
from typing import Dict, Any
from uuid import uuid4
import logging
logger = logging.getLogger(__name__)
import base64
from pathlib import Path
import json
import time

from langchain_community.document_loaders import PyPDFLoader
import tempfile, os


# At top of file
import tempfile

import base64
from src.anubis.utils.state import GlobalState
from src.anubis.utils.context import GlobalContext

# from langgraph.config import get_store

from langgraph.store.base import BaseStore

from src.subgraphs.process_media_graph.utils.helper_functions import process_text_media_item_target_for_vectorstore

from src.anubis.utils.state import GlobalState
from src.anubis.utils.context import GlobalContext
from langgraph.runtime import Runtime

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.anubis.utils.model import init_model, init_image_description_model

from langchain.tools import tool


from langchain_core.runnables import RunnableConfig
from src.anubis.utils.utility import extract_user_id_assistant_id
from src.subgraphs.process_media_graph.utils.utility import extract_personality_from_image
from src.subgraphs.process_media_graph.utils.helper_functions import process_text_to_document

DEBUG_LOG_PATH = "/home/user/gh/anubis/.cursor/debug-eeabc1.log"
DEBUG_SESSION_ID = "eeabc1"

_ALLOWED_STILL_IMAGE_MIMES = frozenset(
    {"image/jpeg", "image/png", "image/gif", "image/webp"}
)


def _normalize_declared_image_mime(ct: str) -> str:
    m = (ct or "").split(";")[0].strip().lower()
    return "image/jpeg" if m == "image/jpg" else m


def _write_debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass



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
            # Extract file info
            filename = file_data.get('filename', 'unknown')
            suffix = Path(filename).suffix
            content_type = file_data.get('content_type', '')
            file_bytes = file_data.get('content')  # Raw bytes
            user_id = file_data.get("user_id")
            assistant_id = file_data.get("assistant_id")
            reference_image = file_data.get("reference_image")
            reference_audio = file_data.get("reference_audio")
            
            logger.info(f"Processing file: {filename} ({content_type})")

            image_url_remote = (file_data.get("image_url") or "").strip()
            if image_url_remote:
                mime = _normalize_declared_image_mime(content_type)
                if mime not in _ALLOWED_STILL_IMAGE_MIMES:
                    logger.warning(
                        "Skipping remote image with disallowed MIME %s", mime
                    )
                    continue
                media_list.append({
                    "type": "image_url",
                    "image_url": {"url": image_url_remote},
                    "metadata": {
                        "filename": filename,
                        "content_type": mime,
                        "size": 0,
                        "user_id": user_id,
                        "assistant_id": assistant_id,
                        "reference_image": reference_image,
                    },
                })
                continue

            audio_url_remote = (file_data.get("audio_url") or "").strip()
            if audio_url_remote:
                mime = content_type.split(";")[0].strip().lower()
                if not mime.startswith("audio/"):
                    logger.warning(
                        "Skipping remote audio with non-audio MIME %s", mime
                    )
                    continue
                media_list.append({
                    "type": "audio",
                    "audio_url": {"url": audio_url_remote},
                    "metadata": {
                        "filename": filename,
                        "content_type": mime,
                        "size": 0,
                        "user_id": user_id,
                        "assistant_id": assistant_id,
                        "reference_audio": reference_audio,
                    },
                })
                continue

            video_url_remote = (file_data.get("video_url") or "").strip()
            if video_url_remote:
                mime = content_type.split(";")[0].strip().lower()
                if not mime.startswith("video/"):
                    logger.warning(
                        "Skipping remote video with non-video MIME %s", mime
                    )
                    continue
                media_list.append({
                    "type": "video",
                    "video_url": {"url": video_url_remote},
                    "metadata": {
                        "filename": filename,
                        "content_type": mime,
                        "size": 0,
                        "user_id": user_id,
                        "assistant_id": assistant_id,
                    },
                })
                continue

            page_url_remote = (file_data.get("page_url") or "").strip()
            if page_url_remote:
                mime = content_type.split(";")[0].strip().lower()
                media_list.append({
                    "type": "url",
                    "url": page_url_remote,
                    "metadata": {
                        "filename": filename,
                        "content_type": mime,
                        "size": 0,
                        "user_id": user_id,
                        "assistant_id": assistant_id,
                    },
                })
                continue
            
            # Determine media type and convert to standardized format
            if content_type.startswith('image/'):
                mime = _normalize_declared_image_mime(content_type)
                if mime not in _ALLOWED_STILL_IMAGE_MIMES:
                    logger.warning("Skipping image with disallowed MIME %s", mime)
                    continue
                # Full data URI string for storage and vision APIs (langgraph index uses document.kwargs.page_content)
                base64_data = base64.b64encode(file_bytes).decode('utf-8')
                image_data_uri = f"data:{mime};base64,{base64_data}"
                media_list.append({
                    "type": "image",
                    "data": image_data_uri,
                    "metadata": {
                        "filename": filename,
                        "content_type": mime,
                        "size": len(file_bytes),
                        "user_id": user_id,
                        "assistant_id": assistant_id,
                        "reference_image": reference_image,
                    }
                })
            
            elif content_type.startswith('audio/'):
                # Handle audio files
                base64_data = base64.b64encode(file_bytes).decode('utf-8')
                media_list.append({
                    "type": "audio",
                    "data": base64_data,
                    "metadata": {
                        "filename": filename,
                        "content_type": content_type,
                        "size": len(file_bytes),
                        "user_id": user_id,
                        "assistant_id": assistant_id, 
                        "reference_audio": reference_audio                    
                    }
                })
            
            elif content_type.startswith('video/'):
                # Handle video files
                base64_data = base64.b64encode(file_bytes).decode('utf-8')
                media_list.append({
                    "type": "video",
                    "data": base64_data,
                    "metadata": {
                        "filename": filename,
                        "content_type": content_type,
                        "size": len(file_bytes),
                        "user_id": user_id,
                        "assistant_id": assistant_id,
                    }
                })
            
            elif content_type in [
                'text/plain', 
            'application/json', 
            'text/markdown', 
            'application/octet-stream'
            ]:
                # Handle text files
                if suffix == '.txt':
                    text_content = file_bytes.decode('utf-8')
                    media_list.append({
                        "type": "text",
                        "content": text_content,
                        "metadata": {
                            "filename": filename,
                            "content_type": content_type,
                            "size": len(file_bytes),
                            "user_id": user_id,
                            "assistant_id": assistant_id,
                        }
                    })
                elif suffix == '.json' or suffix == '.jsonl':
                    import json
                    text_content = json.loads(file_bytes.decode('utf-8'))
                    media_list.append({
                        "type": "json",
                        "content": text_content,
                        "metadata": {
                            "filename": filename,
                            "content_type": content_type,
                            "size": len(file_bytes),
                            "user_id": user_id,
                            "assistant_id": assistant_id,
                        }
                    })
                else: # handle markdown
                    text_content = file_bytes
                    media_list.append({
                        "type": "text",
                        "content": text_content,
                        "metadata": {
                            "filename": filename,
                            "content_type": content_type,
                            "size": len(file_bytes),
                            "user_id": user_id,
                            "assistant_id": assistant_id,
                        }
                    })
            
            elif content_type == 'application/pdf':
                # Handle PDFs
                base64_data = base64.b64encode(file_bytes).decode('utf-8')
                media_list.append({
                    "type": "pdf",
                    "data": base64_data,
                    "bytes":file_bytes,
                    "metadata": {
                        "filename": filename,
                        "content_type": content_type,
                        "size": len(file_bytes),
                        "user_id": user_id,
                        "assistant_id": assistant_id,
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
        "media_files": []  # Clear after processing
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
            "data|text|indicator": "CONTENT OF MEDIA", 
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
    all_documents = []
    for media_item in media_list:
        docs = await process_media_item_task(media_item, runtime, config, store)
        
        for doc in docs:
            status = doc.metadata.get("status", "")
            if status == "error":
                error = doc.metadata.get("error", "")
                filename = doc.metadata.get("filename", "")
                logger.warning(f"Error processing media: {filename} {error}")
            else:
                all_documents.append(doc)

    # Identify Vector Store formatted documents
    # NOTE This contains only information from the target speaker or about the target speaker
    vector_store_document_list_formatted = [doc for doc in all_documents if doc.metadata.get("vectorstore_acceptable", False) == True]

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

    try:
        if media_type in ("image", "image_url"):
            reference_image = metadata.get("reference_image", False)

            vision_source: str = ""
            stored_reference_string: str = ""

            if media_type == "image_url":
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
                vision_source = url
                stored_reference_string = url
            elif "data" in media_item:
                vision_source = media_item["data"]
                stored_reference_string = vision_source
            elif "image_url" in media_item:
                url = (media_item.get("image_url") or {}).get("url", "").strip()
                vision_source = url
                stored_reference_string = url
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
                image_data=vision_source,
                filename=filename,
                store=store,
                user_id=user_id,
                assistant_id=assistant_id,
                context=runtime.context,
            )
            doc.metadata.update(
                {
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "created_at": datetime.now(tz=timezone.utc).isoformat(),
                    "processing_task_id": str(uuid4()),
                    "reference_image": reference_image,
                    "filename": filename,
                    "vectorstore_acceptable": True,
                    "adapter_acceptable": False,
                    "analysis_acceptable": True,
                }
            )

            if reference_image:
                namespace = (user_id, assistant_id, "reference_image")
                doc_json = doc.to_json()
                await store.aput(
                    namespace,
                    key=assistant_id,
                    value={
                        "reference_image_data": stored_reference_string,
                        "document": doc_json,
                    },
                )
            return [doc]

        # Handle text (Project Gutenberg; text files; list of media urls): https://claude.ai/chat/30c554c8-1386-4af2-9f19-f63b51942fc5 
        # Handle large continuous text string if proprietary content; classify non-proprietary text content
        elif media_type == "text":
            documents = await process_text_to_document(
                metadata=metadata,
                user_id=user_id,
                assistant_id=assistant_id,
                media_item=media_item,
            )
            return documents
        elif media_type == "json": # formatted proprietary llm content (chatgpt, claude, grok, etc.)
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
                            document.metadata.update({"vectorstore_acceptable": True})
                            final_documents.append(document)
                
            return final_documents
        # Handle URLs
        elif media_type == "url":
            # TODO: Implement URL content fetching
            url = media_item.get("url", "")
            docs = [Document(
                page_content=f"Content from URL: {url}",
                metadata={"source": url, "type": "url", "status": "not_implemented"}
            )]
            return docs
        
        elif media_type == "pdf": # Presumes written in first person from the target source
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                tmp_file.write(media_item["bytes"])
                tmp_path = tmp_file.name
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            os.unlink(tmp_path)
            document = docs[0]
            # Process data for vectorstore; Identify content
            classification_metadata = {
                "classified_situation": "target pdf source document",
                "classification_reasoning": "predefined classification"
            }
            final_documents = []
            for temp_document in docs:
                media_item['content'] = temp_document.page_content
                documents = await process_text_media_item_target_for_vectorstore(
                    media_item = media_item, 
                    user_id = user_id, 
                    assistant_id= assistant_id, 
                    classification_metadata=classification_metadata,
                    use_semantic_chunks=False,
                )

                for document in documents:
                    document.metadata.update({"vectorstore_acceptable":True})
                    document.metadata.update({"namespace":"identity"})
                    final_documents.append(document)
            # Analysis Will be handled here with the appropriate model

            return final_documents
           
        # Handle audio: https://claude.ai/chat/df5f518f-f846-4015-bb05-7adc6de96678
        elif media_type == "audio":
            """
            Detect the number of speakers
            Audio needs to be diarized if reference audio is available;
            otherwise convert to text
            """
            reference_audio = metadata.get("reference_audio", False)
            if "data" in media_item:
                audio_data = media_item["data"]

                if reference_audio:
                    namespace = (user_id, assistant_id, "reference_audio")
                    await store.aput(
                        namespace,
                        key=assistant_id,
                        value={"reference_audio_data": audio_data},
                    )

                media_item_fwd = {
                    **media_item,
                    "content": media_item.get("content")
                    or "[Audio upload; transcription pending]",
                }
                return await process_text_to_document(
                    media_item=media_item_fwd,
                    user_id=user_id,
                    assistant_id=assistant_id,
                    metadata=metadata,
                )

            if "audio_url" in media_item:
                url = (media_item["audio_url"] or {}).get("url", "").strip()
                if not url:
                    return [
                        Document(
                            page_content="[Empty audio URL]",
                            metadata={
                                "status": "error",
                                "filename": filename,
                            },
                        )
                    ]
                stored_ref = url
                if url.startswith("data:audio"):
                    stored_ref = url

                if reference_audio:
                    namespace = (user_id, assistant_id, "reference_audio")
                    await store.aput(
                        namespace,
                        key=assistant_id,
                        value={"reference_audio_data": stored_ref},
                    )

                doc = Document(
                    page_content=(
                        "Reference audio from URL."
                        if reference_audio
                        else f"[Audio URL: {url}]"
                    ),
                    metadata={
                        "user_id": user_id,
                        "assistant_id": assistant_id,
                        "created_at": datetime.now(tz=timezone.utc).isoformat(),
                        "processing_task_id": str(uuid4()),
                        "type": "audio",
                        "reference_audio": reference_audio,
                        "filename": filename,
                        "vectorstore_acceptable": True,
                        "adapter_acceptable": False,
                        "analysis_acceptable": True,
                    },
                )
                return [doc]

        # Handle video
        elif media_type == "video":
            if "video_url" in media_item:
                vurl = (media_item["video_url"] or {}).get("url", "").strip()
                return [
                    Document(
                        page_content=f"[Video URL: {vurl}]",
                        metadata={
                            "type": "video",
                            "status": "not_implemented",
                            "filename": filename,
                            "user_id": user_id,
                            "assistant_id": assistant_id,
                            "vectorstore_acceptable": False,
                        },
                    )
                ]
            docs = [
                Document(
                    page_content="[Video processing not yet implemented]",
                    metadata={"type": "video", "status": "not_implemented"},
                )
            ]
            return docs
        
        else:
            logger.warning(f"Unsupported media type: {media_type}")
            docs = [Document(
                page_content=f"[Unsupported media type: {media_type}]",
                metadata={"type": media_type, "status": "unsupported"}
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
    return await tool.ainvoke(media_item["content"])

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
                    if "metadata" not in item or not isinstance(
                        item.get("metadata"), dict
                    ):
                        item["metadata"] = {}
                    item["metadata"]["user_id"] = user_id
                    item["metadata"]["assistant_id"] = assistant_id
                    media_list.append(item)
                    # EACH ITEM NEEDS USER_ID AND ASSISTANT_ID FROM CONTEXT
                    # user_id
                    # assistant_id
            
        logger.info(f"Extracted {len(media_list)} media items")
        return {"media_list": media_list}

    logger.warning(f"Unexpected content type: {type(content)}")
    return {"media_list": []}
