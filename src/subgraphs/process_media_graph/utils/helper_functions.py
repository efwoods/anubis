from typing import Any

import base64

import logging
logger = logging.getLogger(__name__)


def identify_file_type_and_convert_to_base64(media_files: list[Any]): 
    media_list = []
    for file_data in media_files:
        try:
           # Extract file info
           filename = file_data.get('filename', 'unknown')
           content_type = file_data.get('content_type', '')
           file_bytes = file_data.get('content')  # Raw bytes
           
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
                       "size": len(file_bytes)
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
                       "size": len(file_bytes)
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
                       "size": len(file_bytes)
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
                       "size": len(file_bytes)
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
                       "size": len(file_bytes)
                   }
               })
           
           else:
               logger.warning(f"Unsupported content type: {content_type}")
               continue
        
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            continue
    
    logger.info(f"Converted {len(media_list)} files to media format")


# AUDIO PROCESSING

from functools import lru_cache
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

@lru_cache(maxsize=1)
def get_whisper_pipeline():
    """Load and cache the Whisper model pipeline"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model_id = "openai/whisper-large-v3"
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=False,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    return pipe


# TEXT TO VECTORSTORE

"""
Text chunking module for processing large text files into Documents
for LangGraph-MongoDB vectorstore with all-MiniLM-L6-v2 embeddings (384 dimensions)
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from uuid import uuid4
import logging
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


async def process_text_media_item(
    media_item: Dict[str, Any],
    user_id: str,
    assistant_id: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: Optional[List[str]] = None
) -> List[Document]:
    """
    Process text content using recursive text splitter and convert to Documents.
    
    This function handles large text files by:
    1. Extracting text from media_item
    2. Splitting text into chunks using RecursiveCharacterTextSplitter
    3. Creating Document objects with proper metadata for each chunk
    
    Args:
        media_item: Dictionary containing text data with 'text' key
        user_id: User identifier for metadata
        assistant_id: Assistant identifier for metadata
        chunk_size: Maximum size of each text chunk (default: 500)
                   Optimized for all-MiniLM-L6-v2 (384 dimensions)
        chunk_overlap: Number of overlapping characters between chunks (default: 50)
        separators: Custom separators for text splitting (optional)
    
    Returns:
        List of Document objects ready for vectorstore upload
    """
    
    logger.warning(f"untested process text media item")

    # Extract text content
    text_content = media_item.get("text", "")
    
    if not text_content:
        logger.warning("Empty text content in media_item")
        return []
    
    # Default separators optimized for semantic coherence
    if separators is None:
        separators = [
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ". ",    # Sentence endings
            "? ",    # Question endings
            "! ",    # Exclamation endings
            "; ",    # Semicolons
            ", ",    # Commas
            " ",     # Spaces
            ""       # Characters
        ]
    
    # Initialize recursive text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )
    
    # Split text into chunks
    text_chunks = text_splitter.split_text(text_content)
    
    logger.info(f"Split text into {len(text_chunks)} chunks")
    
    # Create Document objects for each chunk
    documents = []
    current_timestamp = datetime.now(tz=timezone.utc).isoformat()
    
    # Extract source metadata if available
    source_metadata = media_item.get("metadata", {})
    source = source_metadata.get("source", "text_input")
    
    for idx, chunk in enumerate(text_chunks):
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
                # Include any additional metadata from original media_item
                **{k: v for k, v in source_metadata.items() if k not in ["source"]}
            }
        )
        documents.append(doc)
    
    return documents


# async def process_media_item_task_with_text_chunking(media_item: Dict[str, Any]) -> List[Document]:
#     """
#     Updated version of process_media_item_task that handles text chunking.
#     This should replace or extend your existing process_media_item_task function.
    
#     Args:
#         media_item: Media item dictionary with type and content
    
#     Returns:
#         List of Document objects (single doc for images, multiple for chunked text)
#     """
#     media_type = media_item.get("type", "").lower()
    
#     # Extract user_id and assistant_id from metadata
#     metadata = media_item.get("metadata", {})
#     user_id = metadata.get("user_id", "")
#     assistant_id = metadata.get("assistant_id", "")
    
#     # Handle base64 images
#     if media_type == "image":
#         if "data" in media_item:
#             # Base64 image
#             image_data = media_item["data"]
#             doc = await extract_personality_from_image(image_data)
#             # Filter valid Documents and add metadata
#             doc.metadata.update({
#                 "user_id": user_id,
#                 "assistant_id": assistant_id, 
#                 "created_at": datetime.now(tz=timezone.utc).isoformat(),
#                 "processing_task_id": str(uuid4()),
#             })
#             return [doc]  # Return as list for consistency
            
#         elif "image_url" in media_item:
#             # URL-based image
#             url = media_item["image_url"].get("url", "")
#             if url.startswith("data:image"):
#                 # Extract base64 data
#                 image_data = url.split(",", 1)[1]
#                 doc = await extract_personality_from_image(image_data)
#                 doc.metadata.update({
#                     "user_id": user_id,
#                     "assistant_id": assistant_id,
#                     "created_at": datetime.now(tz=timezone.utc).isoformat(),
#                     "processing_task_id": str(uuid4()),
#                 })
#                 return [doc]
    
#     # Handle text with chunking
#     elif media_type == "text":
#         logger.info(f"Processing text media item with chunking")
#         docs = await process_text_media_item(
#             media_item=media_item,
#             user_id=user_id,
#             assistant_id=assistant_id,
#             chunk_size=500,  # Adjust based on your needs
#             chunk_overlap=50
#         )
#         return docs
    
#     # Fallback for unknown types
#     logger.warning(f"Unknown media type: {media_type}")
#     return []


# async def convert_media_list_to_text_document_multiple_documents_per_media_item(state, context=None) -> Dict[str, Any]:
#     """
#     Updated version that properly handles multiple documents per media item.
    
#     Media type in media list is determined at this point: 
#     Convert the media in a list of one or more media to text in parallel.
#     Media items must have user_id and assistant_id as metadata.
    
#     Expected format:
#     [
#         {
#             "type": "MEDIA_TYPE", 
#             "data|text|indicator": "CONTENT OF MEDIA", 
#             "metadata": {
#                 "user_id": "...",
#                 "assistant_id": "...",
#                 # fields may include mime-type or the metadata may not exist at all
#             }
#         }, 
#         ...
#     ]
#     """
#     logging.info(f"CONVERT_MEDIA_LIST_TO_TEXT_DOCUMENT NODE")
#     media_list = state.get('media_list', [])
    
#     if not media_list:
#         logger.info(f"No media to process")
#         return {
#             "vectorstore_documents_to_be_indexed": [],
#             "media_list": []
#         }
    
#     logger.info(f"Processing {len(media_list)} media items")
    
#     # Process each media item (each can return multiple documents)
#     all_docs = []
#     for media_item in media_list:
#         try:
#             docs = await process_media_item_task(media_item)
#             all_docs.extend(docs)  # Use extend instead of append since docs is a list
#             logger.info(f"Processed media item, generated {len(docs)} documents")
#         except Exception as e:
#             logger.error(f"Error processing media item: {e}", exc_info=True)
#             continue
    
#     logger.info(f"Total documents generated: {len(all_docs)}")
    
#     return {
#         "vectorstore_documents_to_be_indexed": all_docs,
#         "media_list": []  # Clear processed media list in the state
#     }


# Helper function for batch processing large files
async def process_large_text_file(
    file_path: str,
    user_id: str,
    assistant_id: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Document]:
    """
    Convenience function to process a large text file directly.
    
    Args:
        file_path: Path to the text file
        user_id: User identifier
        assistant_id: Assistant identifier
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of Document objects
    """
    logger.warning(f"untested process LARGE text file item")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        
        media_item = {
            "type": "text",
            "text": text_content,
            "metadata": {
                "user_id": user_id,
                "assistant_id": assistant_id,
                "source": file_path
            }
        }
        
        return await process_text_media_item(
            media_item=media_item,
            user_id=user_id,
            assistant_id=assistant_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
        return []
