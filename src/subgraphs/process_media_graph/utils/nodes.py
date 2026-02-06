# Nodes for Identifying and Handling each media type 


import json
import tempfile
import zipfile
import tarfile
import gzip
import bz2
import lzma
from pathlib import Path
from typing import Optional, Union, Dict, List
from enum import Enum
from io import BytesIO

from datetime import datetime, timezone

from fastapi import UploadFile
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
)

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from typing import Optional
from langchain_core.documents import Document

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader

from uuid import uuid4

import logging
logger = logging.getLogger(__name__)


import requests

# Audio
# Transcribe to text then use a text loader
# from langchain_google_community import SpeechToTextLoader

# loader = SpeechToTextLoader(
#     file_path="path/to/audio.wav"
# )
# docs = loader.load()

# Images
# Convert to text from image then use a Text Loader...
# Video

# Unknown
# from langchain_community.document_loaders import UnstructuredLoader
from langchain_unstructured import UnstructuredLoader

# Supported by Unstructured include: DOCX, PPTX, HTML, XML, XLSX, CSV, JPG, PNG, BMP, GIF, EMAIL, RTF, EPUB, and more.

# Web

# Youtube video loader
from langchain_community.document_loaders import YoutubeLoader

# # Loads YouTube video transcripts
# loader = YoutubeLoader.from_youtube_url(
#     "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
# )
# docs = loader.load()

# Youtube Audio Loader
from langchain_community.document_loaders import YoutubeAudioLoader

# loader = YoutubeAudioLoader(
#     youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
#     save_dir="./youtube_audio"
# )
# docs = loader.load()

from PIL import Image
import requests
import base64
import io
import os

async def _image_to_base64(
    self, image_source: Union[str, Path, UploadFile, bytes]
) -> str:
    """Convert image from path, UploadFile, or bytes to base64-encoded JPEG."""
    if isinstance(image_source, (str, Path)):
        img = Image.open(image_source)
    elif isinstance(image_source, UploadFile):
        img = Image.open(io.BytesIO(image_source.file.read()))
    elif isinstance(image_source, bytes):
        img = Image.open(io.BytesIO(image_source))
    else:
        raise TypeError("image_source must be str, Path, UploadFile, or bytes")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95, optimize=True)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

# metadata: Optional[Dict]
async def _load_image(
        file: Dict
    ) -> Document:
        """
        Load image file - requires vision model analysis.
        Use: GPT-4V, Claude Vision, or LLaMA-4-vision
        """
        try:
            # if metadata.get("source_type") == "base64":
                # base64 string
            # expects a base64 string
            logger.info(f"_LOAD_IMAGE CALLED")
            image_data = file.get('data')
            # logger.info(f"_LOAD_IMAGE CALLED")
            doc = await extract_personality_from_image(image_source=image_data)
            return doc
        except Exception as e:
            logger.info(f"Error in _load_image: {e}")
            raise e
        
# At top of file
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
import tempfile
import os

async def _load_text(
    file: Union[UploadFile, str, Path], metadata: Dict
) -> list[Document]:  # Changed return type
    """Load and split plain text file into vectorstore chunks"""
    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, encoding="utf-8"
    ) as temp_file:
        content = (await file.read()).decode("utf-8")
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        loader = TextLoader(temp_path, encoding="utf-8")
        full_docs = loader.load()  # Returns list[Document], usually [1] for text
        
        # Split into chunks - good for Gutenberg books
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,      # Chars per chunk
            chunk_overlap=20,    # Overlap for context
            length_function=len,  # Simple char length
        )
        split_docs = text_splitter.split_documents(full_docs)
        
        # Add your metadata to each chunk
        for doc in split_docs:
            doc.metadata.update(metadata)
            doc.metadata["source"] = temp_path  # Or filename
        
        return split_docs  # List ready for vectorstore.add_documents()
    
    finally:
        os.unlink(temp_path)  # Clean up temp file

async def _load_pdf(self, filename: str, raw_bytes: bytes) -> Optional[Document]:
    """Load PDF file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name
    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        if docs:
            combined_text = "\n\n".join([d.page_content for d in docs])
            return Document(
                page_content=combined_text,
                metadata={
                    "source": "pdf_file",
                    "filename": filename,
                    "type": "text",
                    "pages": len(docs),
                    "requires_conversion": False,
                    "ready_for_vectorstore": True,
                    "ready_for_adapter": True,
                },
            )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return None

def _load_audio(
    self, filename: str, content_type: str, raw_bytes: bytes
) -> Document:
    """
    Load audio file - requires transcription to text.
    Use: OpenAI Whisper API or similar
    """
    # return Document(
    #     page_content=f"[AUDIO FILE: {filename} - Transcription required]",
    #     metadata={
    #         "source": "audio_file",
    #         "filename": filename,
    #         "type": "audio",
    #         "mime_type": content_type,
    #         "size_bytes": len(raw_bytes),
    #         "requires_conversion": True,
    #         "conversion_method": "whisper_transcription",
    #         "transcription_endpoint": "/audio-transcription",
    #         "ready_for_vectorstore": False,
    #         "ready_for_adapter": False
    #     }
    # )
    pass

def _load_video(
    self, filename: str, content_type: str, raw_bytes: bytes
) -> Document:
    """
    Load video file - requires audio transcription + frame analysis.
    Use: Whisper for audio + vision model for key frames
    """
    # return Document(
    #     page_content=f"[VIDEO FILE: {filename} - Audio transcription and frame analysis required]",
    #     metadata={
    #         "source": "video_file",
    #         "filename": filename,
    #         "type": "video",
    #         "mime_type": content_type,
    #         "size_bytes": len(raw_bytes),
    #         "requires_conversion": True,
    #         "conversion_methods": ["whisper_transcription", "vision_frame_analysis"],
    #         "transcription_endpoint": "/audio-transcription",
    #         "ready_for_vectorstore": False,
    #         "ready_for_adapter": False
    #     }
    # )
    pass


import base64
from fastapi import UploadFile
from typing import List, Dict, Any

from src.anubis.utils.state import GlobalState
from src.anubis.utils.context import GlobalContext
from langgraph.runtime import Runtime

# from langgraph.config import get_store

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

async def process_uploaded_files(
    state: GlobalState, 
    runtime: Runtime[GlobalContext], 
    store: BaseStore
) -> Dict[str, Any]:
    """
    Convert FastAPI UploadFile objects into standardized media format.
    This is the entry point for direct file uploads (not from messages).
    """
    
    logger.info(f"Process uploaded files NODE")

    user_id = runtime.context.assistant_ctx.user_id
    assistant_id = runtime.context.assistant_ctx.assistant_id

    namespace = (user_id, assistant_id)
    
    # store = runtime.store

    logger.info(f"breakpoint")

    # result_put = await store.aput(namespace=namespace, key="test_key_process_uploaded_files", value="test_values process uploaded files")
    # result_get = await store.asearch(namespace,)

    # result_get = await runtime.store.asearch(namespace,)
    # result_put = await runtime.store.aput(namespace=namespace, key="test_key", value="test_values process uploaded files")

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
            content_type = file_data.get('content_type', '')
            file_bytes = file_data.get('content')  # Raw bytes
            user_id = file_data.get("user_id")
            assistant_id = file_data.get("assistant_id")
            reference_image = file_data.get("reference_image")
            reference_audio = file_data.get("reference_audio")
            
            logger.info(f"Processing file: {filename} ({content_type})")
            
            # Determine media type and convert to standardized format
            if content_type.startswith('image/'):
                # Convert image to base64
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
                        "reference_image": reference_image
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
                        "assistant_id": assistant_id
                    }
                })
            
            elif content_type in ['text/plain', 'application/json', 'text/markdown']:
                # Handle text files
                text_content = file_bytes.decode('utf-8')
                media_list.append({
                    "type": "text",
                    "content": text_content,
                    "metadata": {
                        "filename": filename,
                        "content_type": content_type,
                        "size": len(file_bytes),
                        "user_id": user_id,
                        "assistant_id": assistant_id
                    }
                })
            
            elif content_type == 'application/pdf':
                # Handle PDFs
                base64_data = base64.b64encode(file_bytes).decode('utf-8')
                media_list.append({
                    "type": "pdf",
                    "data": base64_data,
                    "metadata": {
                        "filename": filename,
                        "content_type": content_type,
                        "size": len(file_bytes),
                        "user_id": user_id,
                        "assistant_id": assistant_id
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




from src.subgraphs.process_media_graph.utils.helper_functions import get_whisper_pipeline


async def extract_text_from_audio(audio_data: str) -> Document:
    """Extract text from audio using Hugging Face Whisper Large v3"""
    logger.info(f"needs reference audio from storage for speaker diarization (timestamps and who is speaking)")
    import base64
    import tempfile
    import os
    import asyncio
    import aiofiles

    logger.info(f"extract text from audio ENTRYPOINT")
    
    try:
        # Decode base64 audio data
        audio_bytes = base64.b64decode(audio_data)
        
        # Create temporary file in thread
        temp_file_directory, temp_audio_path = await asyncio.to_thread(
            tempfile.mkstemp,
            ".mp3"
        )

        # Write audio bytes asynchronously
        async with aiofiles.open(temp_audio_path, 'wb') as f:
            await f.write(audio_bytes)

            logger.info(f"Audio file written to {temp_audio_path}")

        
        try:
            # Get cached pipeline
            pipe = get_whisper_pipeline()
            
            # Run transcription in thread pool (it's CPU/GPU intensive)
            logger.info("Starting audio transcription...")
            result = await asyncio.to_thread(pipe, temp_audio_path)

            transcript = result["text"]
            
            # Create Document with transcription
            doc = Document(
                page_content=transcript,
                metadata={
                    "source": "audio_transcription",
                    "model": "whisper-large-v3",
                    "transcript_length": len(transcript)
                }
            )
            return doc
            
        finally:
            # Clean up temporary file
            if temp_file_directory is not None:
                await asyncio.to_thread(os.close, temp_file_directory)
                
    except Exception as e:
        logger.error(f"Audio transcription failed: {e}")
        raise

    finally:
        # Clean up temporary file in thread
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                await asyncio.to_thread(os.unlink, temp_audio_path)
                logger.info(f"Cleaned up temporary file: {temp_audio_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {temp_audio_path}: {cleanup_error}")

# metadata: Optional[Dict]
from src.anubis.utils.model import init_model
async def extract_personality_from_image(
    image_data: str) -> Document:
    from src.anubis.utils.configuration import GlobalConfiguration
    """Extract personality description from image using vision LLM."""
    logger.info(f"needs reference image from storage for target identification (possibly object bounding box of the target)")
    # base64_image = self._image_to_base64(image_source)
    # base64_image = self.image_to_base64(image_path)

    logger.info(f"extract_personality_from_image entrypoint")    
    # Use reference image to help identify the person in the image.
    text_prompt_for_image_to_text_context = (
        "Describe the individual in the image in vivid detail. "
        "Return only the description of the person. "
        "Do not mention that this is an image. "
        "Describe the qualities of the character of the person in full detail and"
        "Describe the personality of this person so as to clearly visualize the person."
        "Do not describe the physical appearance"
    )

    # these requests need to use the model in the graph rather than the requests because of 400 errors

    configuration = GlobalConfiguration()

    tools = []
    
    model = init_model(
        configuration.provider_model,
        configuration.llama_api_base_url,
        configuration.llama_api_key,
        tools, 
        configuration.dev
    )

    image_to_target_textual_description_payload = [
                        {
                            "type": "text",
                            "text": (text_prompt_for_image_to_text_context),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        },
                    ]

    message = HumanMessage(
        content=image_to_target_textual_description_payload
    )

    response = await model.ainvoke([message])

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
            "type": "personality_extraction", 
            "model": configuration.provider_model
        }
    )


from src.anubis.utils.state import GlobalState
from src.anubis.utils.context import GlobalContext
from langgraph.runtime import Runtime

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.anubis.utils.model import init_model

from src.subgraphs.process_media_graph.utils.helper_functions import identify_file_type_and_convert_to_base64

async def extract_media_from_message(state: GlobalState, runtime: Runtime[GlobalContext]):
    
    logger.info(f"Extract_media_from_message NODE")

    user_id = runtime.context.user_ctx.user_id
    assistant_id = runtime.context.assistant_ctx.assistant_id

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

from langgraph.prebuilt import ToolRuntime
from langchain.tools import tool

@tool
async def base64_image(media_data: Dict[str, Any], runtime: ToolRuntime[GlobalContext]) -> Document:
    """ Used to convert a base64 image string to a Document with text. """
    doc = extract_personality_from_image(media_data["data"], runtime.state)
    return doc

@tool
async def non_base64_image(media_data: Dict[str, Any], runtime: ToolRuntime[GlobalContext]) -> Document:
    """ Used to convert a NON base64 image object to a Document with text. """
    pass

@tool
async def text_only_input(media_data: Dict[str, Any], runtime: ToolRuntime[GlobalContext]) -> Document:
    """ Used to convert a text string to a Document with text. """
    pass

@tool
async def audio(media_data: Dict[str, Any], runtime: ToolRuntime[GlobalContext]) -> Document:
    """ Used to convert audio to a Document with text. """
    pass

@tool
async def video(media_data: Dict[str, Any], runtime: ToolRuntime[GlobalContext]) -> Document:
    """ Used to convert a video to a Document with text. """
    pass

@tool
async def handle_url(media_data: Dict[str, Any], runtime: ToolRuntime[GlobalContext]) -> Document:
    """ 
    Used to convert a url to a Document with text. May require other tool use. 
    Download the content from the URL.
    Queue the content as media to be processed. 
    """
    pass

MEDIA_CONVERSION_TOOLS = { # identified type to tool function call
    "base64_image" : base64_image, 
    "non_base64_image": non_base64_image, 
    "text_only_input": text_only_input, 
    "audio": audio,
    "video": video,
    "handle_url": handle_url
}


# from langgraph.func import task

async def process_media_item_task(
    media_item: Dict[str, Any], 
    runtime: Runtime[GlobalContext]
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
        # Handle base64 images
        if media_type == "image":
            reference_image = media_item['metadata']["reference_image"]
            if "data" in media_item:
                # Base64 image
                image_data = media_item["data"]
                logger.warning(f"STORE REFERENCE IMAGE HERE")
                # UPDATE TO RETRIEVE AND PASS REFERENCE IMAGE DATA
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


                return doc
            elif "image_url" in media_item:
                # URL-based image
                url = media_item["image_url"].get("url", "")
                if url.startswith("data:image"):
                    # Extract base64 data
                    image_data = url.split(",", 1)[1]
                    logger.warning(f"STORE REFERENCE IMAGE HERE")
                    # UPDATE TO RETRIEVE AND PASS REFERENCE IMAGE DATA
                    doc =  await extract_personality_from_image(image_data)
                    # Filter valid Documents and add metadata
                doc.metadata.update({
                    "user_id": user_id,
                    "assistant_id": assistant_id, 
                    "created_at": datetime.now(tz=timezone.utc).isoformat(),
                    "processing_task_id": str(uuid4()),
                    "reference_image": reference_image
                })                
                return doc
        
        # Handle text (Project Gutenberg; text files; list of media urls): https://claude.ai/chat/30c554c8-1386-4af2-9f19-f63b51942fc5
        elif media_type == "text":
            return Document(
                page_content=media_item.get("text", ""),
                metadata={"source": "text_input", "type": "text"}
            )
        
        # Handle URLs
        elif media_type == "url":
            # TODO: Implement URL content fetching
            url = media_item.get("url", "")
            return Document(
                page_content=f"Content from URL: {url}",
                metadata={"source": url, "type": "url", "status": "not_implemented"}
            )      
           
        # Handle audio: https://claude.ai/chat/df5f518f-f846-4015-bb05-7adc6de96678
        elif media_type == "audio":
            reference_audio = media_item['metadata']['reference_audio']
            if "data" in media_item:
                # Base64 audio
                audio_data = media_item["data"]
                logger.warning(f"STORE REFERENCE AUDIO HERE")
                # UPDATE TO RETRIEVE AND PASS REFERENCE IMAGE DATA

                doc = await extract_text_from_audio(audio_data)

                # Add metadata
                doc.metadata.update({
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "created_at": datetime.now(tz=timezone.utc).isoformat(),
                    "processing_task_id": str(uuid4()),
                    "type": "audio", 
                    "reference_audio": reference_audio
                })
                return doc
            elif "audio_url" in media_item:
                # URL-based audio
                url = media_item["audio_url"].get("url", "")
                if url.startswith("data:audio"):
                    # Extract base64 data
                    audio_data = url.split(",", 1)[1]
                    logger.warning(f"STORE REFERENCE AUDIO HERE")
                    # UPDATE TO RETRIEVE AND PASS REFERENCE IMAGE DATA

                    doc = await extract_text_from_audio(audio_data)
                    doc.metadata.update({
                        "user_id": user_id,
                        "assistant_id": assistant_id,
                        "created_at": datetime.now(tz=timezone.utc).isoformat(),
                        "processing_task_id": str(uuid4()),
                        "type": "audio",
                        "reference_audio": reference_audio
                    })
                    return doc
        
        # Handle video
        elif media_type == "video":
            # TODO: Implement video processing
            return Document(
                page_content="[Video processing not yet implemented]",
                metadata={"type": "video", "status": "not_implemented"}
            )
        
        else:
            logger.warning(f"Unsupported media type: {media_type}")
            return Document(
                page_content=f"[Unsupported media type: {media_type}]",
                metadata={"type": media_type, "status": "unsupported"}
            )
    
    except Exception as e:
        logger.error(f"Error processing media item: {e}")
        return Document(
            page_content=f"[Error processing media: {str(e)}]",
            metadata={"type": media_type, "status": "error", "error": str(e)}
        )
    return await tool.ainvoke(media_item["content"])

async def convert_media_list_to_text_document(state: GlobalState, runtime: GlobalContext) -> Dict[str, Any]:
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
    docs = []
    for media_item in media_list:
        doc = await process_media_item_task(media_item, runtime)

        if hasattr(doc.metadata, "error"):
            error = getattr(doc.metadata, "error")
            logger.warning(f"Error processing media {error}")

        else:
            docs.append(doc)

    # # Analysis list (needs a node)
    # documents_to_be_analyzed_for_context_storage_and_prompt_injection_of_assistant: List[Sequence[Document]] UPDATED RETURN VALUE LIST IN RETURN analyzed and stored as facts

    # # Adapter list (needs a node)
    # documents_to_be_processed_for_adapter_training: List[Sequence[Document]] UPDATED RETURN VALUES IN RETURN processed into adapter training format and uploaded to storage

    return {
        "vectorstore_documents_to_be_indexed": docs,
        "media_list": [] # Clear processed media list in the state
    }
