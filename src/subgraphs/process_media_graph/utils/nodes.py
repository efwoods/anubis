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

# metadata: Optional[Dict]
from src.anubis.utils.model import init_model
async def extract_personality_from_image(
    image_source: Union[str, Path, UploadFile, bytes], runtime: Runtime[GlobalContext]
) -> Document:
    """Extract personality description from image using vision LLM."""
    # base64_image = self._image_to_base64(image_source)
    # base64_image = self.image_to_base64(image_path)
    
    text_prompt_for_image_to_text_context = (
        "Describe the individual in the image in vivid detail. "
        "Return only the description of the person. "
        "Do not mention that this is an image. "
        "Describe the qualities of the character of the person in full detail and their personality so as to clearly visualize them. "
        "Do not describe the physical appearance"
    )
    # these requests need to use the model in the graph rather than the requests because of 400 errors

    ctx = runtime.context
    configuration = runtime.context.configuration

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
                                "url": f"data:image/jpeg;base64,{image_source}"
                            },
                        },
                    ]

    message = HumanMessage(
        content=image_to_target_textual_description_payload
    )

    response = model.ainvoke([message])

    logger.info(f"response: {response}")


    avatar_image_description = response.json()
    logger.info(f"XXXXXXXXXXXXXX EXTRACT XXXXXXXXXXXXXXXXXXX extract_personality_from_image CALLED")

    logger.info(f"avatar_image_description: {avatar_image_description}")
    try:
        # logger.warning(f"avatar_image_description['choices'][0]: {avatar_image_description['choices'][0]}")

        # logger.warning(f"avatar_image_description['choices']: {avatar_image_description['choices']}")

        contextual_description = avatar_image_description['choices'][0]['message']['content']
        # contextual_description = avatar_image_description["completion_message"][
            # "content"
        # ]["text"]
        logger.warning("HUMAN VALIDATION / INTERRUPTION IS REQUIRED HERE TO CONTINUE")
        logger.warning("HANDLE RESPONSES SUCH AS: I'm sorry, but I can't provide a description of the person's character or personality based on the information given.")
    except:
        print(avatar_image_description)
    # if metadata.get("is_reference_image"):
    #     try:
    #         avatar_id = str(metadata["avatar_id"])
    #         self.firestore.update_avatar_fields(
    #             avatar_id,
    #             {"system_prompt_reference_image_description": contextual_description},
    #         )
    #     except Exception as e:
    #         logger.error(f"Failed to update avatar {avatar_id}: {e}")
    doc = Document(page_content=contextual_description) # , metadata=metadata
    # all_splits = self.text_splitter.split_documents([doc])
    # document_ids = self.chroma_DB.vector_store.add_documents(documents=all_splits)
    return doc


from src.anubis.utils.state import GlobalState
from src.anubis.utils.context import GlobalContext
from langgraph.runtime import Runtime

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.anubis.utils.model import init_model

async def extract_media_from_message(state: GlobalState, runtime: Runtime[GlobalContext]):
    messages = runtime.state['messages']
    logger.info(f"MESSAGES XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX {messages}")
    
    logger.info(f" LEN MESSAGE XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX {len(messages)}")

    recent_msg = messages[-1]
    # logger.info(f" RECENT MESSAGE messages[-1] XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX {messages[-1]}")
    # logger.info(f" RECENT MESSAGE messages[0] XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX {messages[0]}")
    # logger.info(f" RECENT MESSAGE messages[1] XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX {messages[1]}")
    # [logger.info(f"{message}") for message in messages['content']['text']]
    # count = 0
    # for message in messages:
    #     logger.info(f"{count}: {message}")
    #     count+=1
    # logger.info(f"penultimate has data: messages[-2]: {messages[-2]}")

    # logger.info(f" XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX {isinstance(recent_msg, HumanMessage)}")
    recent_message = messages[-2] # HumanMessage with Content 
    # Expects format:
    """ [
        {"type":"text", "text":"text from human message"}, 
        {"type":"image", "data":"BASE64_IMAGE_DATA"}, ZERO OR MORE IMAGE DICTIONARIES
        {"type":"url", "url":"hypterlink.com"}, etc..
        ]
    """

    if isinstance(recent_message, HumanMessage):
        # logger.info(f"CONTENT XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX {recent_msg.content}")
        content = recent_message.content
        logger.info(f"len content: {len(content)}")
        # logger.info(f"content: {content}")

        media_list = []
        try:
            if len(content > 1):
                media_list = content[1:]
            human_text = content[0] # This could contain text
            return {"media_list": media_list} # replace media list     
                       
        except Exception as e:
            logger.info(f"Error during EXTRACT MEDIA FROM MESSAGE: {e}")
            raise e

from langgraph.prebuilt import ToolRuntime
from langchain.tools import tool

@tool
async def base64_image(media_data: Dict[str: Any], runtime: ToolRuntime[GlobalContext]) -> Document:
    """ Used to convert a base64 image string to a Document with text. """
    doc = extract_personality_from_image(media_data["data"], runtime.state)
    return doc

@tool
async def non_base64_image(media_data: Dict[str: Any], runtime: ToolRuntime[GlobalContext]) -> Document:
    """ Used to convert a NON base64 image object to a Document with text. """
    pass

@tool
async def text_only_input(media_data: Dict[str: Any], runtime: ToolRuntime[GlobalContext]) -> Document:
    """ Used to convert a text string to a Document with text. """
    pass

@tool
async def audio(media_data: Dict[str: Any], runtime: ToolRuntime[GlobalContext]) -> Document:
    """ Used to convert audio to a Document with text. """
    pass

@tool
async def video(media_data: Dict[str: Any], runtime: ToolRuntime[GlobalContext]) -> Document:
    """ Used to convert a video to a Document with text. """
    pass

@tool
async def handle_url(media_data: Dict[str: Any], runtime: ToolRuntime[GlobalContext]) -> Document:
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


from langgraph.func import task

@task
async def process_media_task(process_media_item: Dict) -> Document:
    """ Task: Convert the media item to a Document of text """
    
    logging.info(f"PROCESS_MEDIA_TASK NODE")

    process_type = process_media_item["process_type"] # The process_type is the name of the tool to be called

    tool = MEDIA_CONVERSION_TOOLS.get(process_type)

    return await tool.ainvoke(process_media_item["content"])

async def determine_media_type(state: GlobalState, context: GlobalContext):
    """ Determines the media type in a given list of media.
     I should be able to indicate the media type for each type of media
    then convert the media in a list of one or more media to text in parallel.
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
    I want to keep the media in a list and queue tasks for each item in the list then I want to execute those tasks in parallel and update the final state with the list of text Documents from the media:
async def determine_media_type(state: GlobalState, context: GlobalContext, media_list: List[Dict]):

    """
    
    logging.info(f"DETERMINE_MEDIA_TYPE NODE")
    
    media_list = state.get('media_list', [])
    process_type_list = []
    for media in media_list:
        if hasattr(media, "type"):
            if media['type'] == "image": # deterimine if base64
                if hasattr(media, "data"):
                    # base64 encoded data
                    process_type = "base64_image"
                else:
                    process_type = "non_base64_image"
            elif media['type'] == "text":
                process_type = "text_only_input"
            elif media['type'] == "audio":
                process_type == "audio" 
            elif media['type'] == "video":
                process_type == "video" 
            elif media['type'] == "url":
                process_type = "handle_url"
            else:
                logger.warning(f"unhandled media type in determine media type: {media['type']}")
                continue
        process_type_list.append({
            "process_type": process_type, 
            "content":{**media}, 
            "task_id": uuid4()
        }) 

        tasks = [process_media_task(process_media_item) for process_media_item in process_type_list] # queue tool call execution

        # Convert Media to text
        document_futures = [task.result() for task in tasks]
        docs = await asyncio.gather(*document_futures, return_exceptions=True)

        # Filter valid Documents (successful processes)
        documents = [doc for doc in docs if isinstance(doc, Document)]

        # Add process_task_ids and user/assistant metadata for each document
        documents = [doc.metadata.update({
            "process_task_id": uuid4(), 
            "user_id": context.user_ctx.user_id,
            "assistant_id": context.assistant_ctx.assistant_id,
            "created_at": datetime.now(tz=timezone.utc).isoformat(), 
            "updated_at": None, 
        }) for doc in documents]


        return {
            "processed_media_to_be_formatted": documents, # Documents are appended to the list in the state
            "media_list": [] # clear the list of media
        }

import asyncio
from asyncio import Task
from typing import List
from datetime import datetime, timezone

async def identify_media(
    file: Any, 
) -> Union[Document]:
    """Convert media (image/audio) to searchable text docs. DEPRECATED"""
    # Example for image: use vision model or OCR
    # text = vision_llm.invoke([media])['text']
    # return [Document(page_content=text, metadata={"source": "media"})]
    supported_images = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    supported_audio = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
    supported_video = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    try:
        logger.info(f"XXXXXXXXXXXXXXXXXX file: {file}")
        logger.info(f"XXXXXXXXXXXXXXXXXXXXXXXXXXX GOT HERE XXXXXXXXXXXXXXXXXXXXX")
        file_metadata = file.get('metadata')
        source_type = file.get("source_type")
        # logger.info(f"file_metadata: {file_metadata}")
        # [logger.info(f"Key {key}; value {value}") for key, value in file_metadata.items()]
        # Handle FastAPI UploadFile
        # metadata = {
            # "user_id": self.user_id,
            # "avatar_id": self.avatar_id,
            # "source_type": file.get("source_type"),
            # "filename": file_metadata.filename,
            # "mime_type": file_metadata.mime_type,
            # "content_type": file.get("content_type") or None, 
            # "is_reference_image": is_reference_image,
            # "is_reference_audio": is_reference_audio,
        # }
        # logger.info(f"metadata: {metadata}")
        # # Compressed files (must check before text files)
        # if (content_type in self.COMPRESSED_MIME_TYPES or
        #     any(filename.endswith(ext) for ext in self.supported_compressed)):
        #     return await self._load_compressed(filename, content_type)
        content_type = file.get('type')
        filename = file.get('metadata').get('filename')

        logger.info(f"content_type: {content_type}")
        logger.info(f"filename: {filename}")

        # Text files
        if content_type == "text" or filename.endswith(".txt"):
            logger.info(f"working on loading text documents")
            doc = await _load_text(file)
            return doc
        # Image files
        elif content_type == "image" or any(
            filename.endswith(ext) for ext in supported_images
        ):
            logger.info(f"LOAD IMAGE CALL")
            doc = await _load_image(file)   # List ready for vectorstore.add_documents()  # Returns list[Document], usually [1] for text
            return doc
        # Markdown files
        # elif (
        #     filename.endswith(".md")
        #     or filename.endswith(".markdown")
        #     or content_type == "text/markdown"
        # ):
        #     return self._load_markdown(filename)
        # JSON files
        # elif content_type == "application/json" or filename.endswith(".json"):
        #     return self._load_json(filename)
        # PDF files
        # elif content_type == "application/pdf" or filename.endswith(".pdf"):
        #     return await self._load_pdf(filename)
        # Audio files
        elif content_type.startswith("audio/") or any(
            filename.endswith(ext) for ext in supported_audio
        ):
            return _load_audio(filename, content_type)
        # Video files
        elif content_type.startswith("video/") or any(
            filename.endswith(ext) for ext in supported_video
        ):
            return _load_video(filename, content_type)
        # Unknown type
        else:
            logger.warning(f"Unhandled content_type")
            # return self._load_unknown(filename, content_type)
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        raise e
