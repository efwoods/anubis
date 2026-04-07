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
from src.subgraphs.process_media_graph.utils.helper_functions import process_text_to_document



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

    # logger.info("STORE ACCESS TESTING")
    # namespace = ("evan")
    # await store.aput(namespace=namespace, key="evan", value={"name": "evan"})
    # get_value = await store.aget("evan", key="name")
    # logger.info("get_value: {get_value}")

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
            proprietary_content = file_data.get("proprietary_content") # This is a single body of text (scripture/menu/etc ... there is no single target, text only)
            
            logger.info(f"Processing file: {filename} ({content_type})")
            
            # Determine media type and convert to standardized format
            if content_type.startswith('image/'):
                # Convert image to base64
                # proprietary content creates reference documents from the images otherwise assumes image of source target individual
                base64_data = base64.b64encode(file_bytes).decode('utf-8')
                media_list.append({
                    "type": "image",
                    "data": base64_data,
                    "metadata": {
                        "filename": filename,
                        "content_type": content_type,
                        "size": len(file_bytes),
                        "user_id": user_id,
                        "assistant_id": assistant_id, 
                        "reference_image": reference_image,
                        "proprietary_content": proprietary_content
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
            
            elif content_type in ['text/plain', 'application/json', 'text/markdown', 'application/octet-stream']:
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
                            "proprietary_content": proprietary_content
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
                            "proprietary_content": proprietary_content
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
                            "proprietary_content": proprietary_content
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
                        "proprietary_content": proprietary_content
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

    logger.info(f"Testing store access")

    # namespace = ("testing","document")
    # await store.aput(namespace=namespace, key="media", value={"media":media_item, "document":media_item['content']})
    # testing_get = await store.aget(namespace=namespace, key="media")
    # testing_search = await store.asearch(("testing", "document"), query="Shivon Zilis")
    # logger.info(f"testing_get: {testing_get}")
    # logger.info(f"get_value: {testing_search}")

    try:
        # Handle base64 images
        if media_type == "image":
            reference_image = media_item['metadata']["reference_image"]
            if "data" in media_item:
                
                # Base64 image
                image_data = media_item["data"]                
                
                    
                doc =  await extract_personality_from_image(image_data=image_data, store=store, user_id=user_id, assistant_id=assistant_id)
                    # Filter valid Documents and add metadata
                doc.metadata.update({
                    "user_id": user_id,
                    "assistant_id": assistant_id, 
                    "created_at": datetime.now(tz=timezone.utc).isoformat(),
                    "processing_task_id": str(uuid4()),
                    "reference_image": reference_image,
                    "filename": filename,
                    "vectorstore_acceptable": True,
                    "adapter_acceptable": False,
                    "analysis_acceptable": True
                }) 

                if reference_image:
                    logger.warning(f"STORE REFERENCE IMAGE HERE: presuming upsert")
                    namespace=(user_id, assistant_id, "reference_image")
                    doc_json = doc.to_json()
                    await store.aput(namespace, key=assistant_id, value={"reference_image_data": image_data, "document": doc_json})                    
                docs = [doc]
                return docs
            
            elif "image_url" in media_item:
                # URL-based image
                url = media_item["image_url"].get("url", "")
                if url.startswith("data:image"):
                    # Extract base64 data
                    image_data = url.split(",", 1)[1]
                    
                    doc =  await extract_personality_from_image(image_data)
                    # Filter valid Documents and add metadata
                doc.metadata.update({
                    "user_id": user_id,
                    "assistant_id": assistant_id, 
                    "created_at": datetime.now(tz=timezone.utc).isoformat(),
                    "processing_task_id": str(uuid4()),
                    "reference_image": reference_image,
                    "filename": filename
                })  

                if reference_image:
                    logger.warning(f"STORE REFERENCE IMAGE HERE: presuming upsert")
                    namespace=(user_id, assistant_id, "reference_image")
                    metadata = doc.metadata
                    page_content = doc.metadata.get("page_content", "")
                    metadata.update({"page_content":page_content})
                    await store.aput(namespace, key=assistant_id, value={"reference_image_data": image_data, "metadata": metadata})
                    
                docs = [doc]
                return docs
        
        # Handle text (Project Gutenberg; text files; list of media urls): https://claude.ai/chat/30c554c8-1386-4af2-9f19-f63b51942fc5 
        # Handle large continuous text string if proprietary content; classify non-proprietary text content
        elif media_type == "text":
            documents = await process_text_to_document(
                metadata=metadata, 
                user_id=user_id, 
                assistant_id=assistant_id, 
                media_item=media_item)
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
            logging.info("breakpoint")
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
            reference_audio = media_item['metadata']['reference_audio']
            if "data" in media_item:
                # Base64 audio
                audio_data = media_item["data"]

                if reference_audio:
                    logger.warning(f"STORE REFERENCE AUDIO HERE: presuming upsert")
                    namespace=(user_id, assistant_id, "reference_audio")

                    doc_json = doc.to_json()

                    await store.aput(namespace, key=assistant_id,value={"reference_audio_data": audio_data})                   

                text_content = await extract_text_from_audio(audio_data, user_id, assistant_id)

                media_item['content'] = text_content

                documents = await process_text_to_document(media_item=media_item, user_id=user_id, assistant_id=assistant_id, metadata=metadata)

                return documents
            
            elif "audio_url" in media_item:
                # URL-based audio
                url = media_item["audio_url"].get("url", "")
                if url.startswith("data:audio"):
                    # Extract base64 data
                    audio_data = url.split(",", 1)[1]

                    doc = await extract_text_from_audio(audio_data)

                    doc.metadata.update({
                        "user_id": user_id,
                        "assistant_id": assistant_id,
                        "created_at": datetime.now(tz=timezone.utc).isoformat(),
                        "processing_task_id": str(uuid4()),
                        "type": "audio",
                        "reference_audio": reference_audio,
                        "filename": filename
                    })
                    if reference_audio:
                        logger.warning(f"STORE REFERENCE AUDIO HERE: presuming upsert")
                        namespace=(user_id, assistant_id, "reference_audio")
                        metadata = doc.metadata
                        page_content = doc.metadata.get("page_content", "")
                        metadata.update({"page_content":page_content})

                        await store.aput(namespace, key=assistant_id,value={"reference_audio_data": audio_data, "metadata": metadata})                   

                docs = [doc]
                return docs
        
        # Handle video
        elif media_type == "video":
            # TODO: Implement video processing
            docs = [Document(
                page_content="[Video processing not yet implemented]",
                metadata={"type": "video", "status": "not_implemented"}
            )]
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

async def extract_text_from_audio(audio_data: str, user_id: str, assistant_id: str) -> Document:#
    """Extract text from audio using Hugging Face Whisper Large v3
    reference audio is used if available.
    text is diarized and the target is identified.
    """
    logger.info(f"needs reference audio from storage for speaker diarization (timestamps and who is speaking)")
    import base64
    import tempfile
    import os
    import asyncio
    import aiofiles

    logger.info(f"extract text from audio ENTRYPOINT")

    # store

    # Identify reference audio existence and extract reference audio 
    """
    {"reference_audio": {"reference_audio_data": data, "metadata": metadata}}
    """

    """ ANALYZE AUDIO """

    # if reference_audio is not None:
    logger.info(f"Identify the number of speakers in the audio")
    logger.info(f"Identify if the target speaker is in the audio")
    logger.info(f"Diarize the audio (timestamps of who is speaking when)")
    # return text_content

    pass

async def extract_personality_from_image(
    image_data: str, store: str, user_id: str, assistant_id: str) -> Document:
    """Extract personality description from image using vision LLM."""
    logger.info(f"needs reference image from storage for target identification (possibly object bounding box of the target)")
    # base64_image = self._image_to_base64(image_source)
    # base64_image = self.image_to_base64(image_path)

    logger.info(f"extract_personality_from_image entrypoint")    
    # Use reference image to help identify the person in the image.

    from src.anubis.utils.prompts.system_prompts import TEXT_PROMPT_FOR_IMAGE_TO_TEXT_CONTEXT_FOR_FIRST_PERSON_PERSPECTIVE_DESCRIPTION
    
    model = init_image_description_model()

    # use reference image if available to target individual
    assistant_reference_image_identity_namespace = (user_id, assistant_id, "reference_image")
    key=assistant_id
    reference_image_item = await store.aget(assistant_reference_image_identity_namespace, key)
    if reference_image_item and len(reference_image_item) != 0:
        reference_image_data = getattr(reference_image_item,'value', {}).get("reference_image_data", None)

    if reference_image_item and reference_image_data:
        image_to_target_textual_description_payload = [
                            {
                                "type": "text",
                                "text": (TEXT_PROMPT_FOR_IMAGE_TO_TEXT_CONTEXT_FOR_FIRST_PERSON_PERSPECTIVE_DESCRIPTION),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{reference_image_data}"
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                },
                            },
                        ]
    else:

        image_to_target_textual_description_payload = [
                            {
                                "type": "text",
                                "text": (TEXT_PROMPT_FOR_IMAGE_TO_TEXT_CONTEXT_FOR_FIRST_PERSON_PERSPECTIVE_DESCRIPTION),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                },
                            },
                        ]

    input = {"messages": [{"role": "user", "content": image_to_target_textual_description_payload}]}
    

    response = await model.ainvoke(input=input['messages'])

    logger.info(f"response: {response}")

    if hasattr(response, 'content'):
        contextual_description = response.content
    else:
        contextual_description = str(response)

    logger.info(f"Extracted personality from image: {contextual_description[:100]}")

    return Document(
        page_content=contextual_description,
        metadata={
            "source": "vision_model", 
            "type": "personality_extraction"
        }
    )

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
                    item["metadata"]['user_id'] = user_id
                    item["metadata"]['assistant_id'] = assistant_id
                    media_list.append(item)
                    # EACH ITEM NEEDS USER_ID AND ASSISTANT_ID FROM CONTEXT
                    # user_id
                    # assistant_id
            
        logger.info(f"Extracted {len(media_list)} media items")
        return {"media_list": media_list}

    logger.warning(f"Unexpected content type: {type(content)}")
    return {"media_list": []}
