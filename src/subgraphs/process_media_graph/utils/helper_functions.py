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
    separators: Optional[List[str]] = None,
    classification_metadata: Optional[dict] = None
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
        if classification_metadata is not None:
            doc.metadata.update(classification_metadata)
        documents.append(doc)
    
    return documents