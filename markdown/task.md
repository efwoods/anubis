This needs to be an asynchronous function. 

I need audio transcription that will be hosted as a server endpoint that will scale to multiple requests for production. 

I need to establish the LRU cache of the model on startup of the webapp





async def extract_text_from_audio(audio_data: str) -> Document:

    """Extract text from audio using Hugging Face Whisper Large v3"""

logger.warning(f"THIS IS UNTESTED")

import base64

import tempfile

import os

import asyncio

logger.info(f"extract text from audio ENTRYPOINT")

try:

# Decode base64 audio data

audio_bytes = base64.b64decode(audio_data)

# Create temporary file [SYNCHRONOUS WRITE]

with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:

temp_audio.write(audio_bytes)

temp_audio_path = temp_audio.name

try:

# Get cached pipeline

pipe = get_whisper_pipeline()

# Run transcription in thread pool (it's CPU/GPU intensive)

loop = asyncio.get_event_loop()

result = await loop.run_in_executor(None, pipe, temp_audio_path)

transcript = result["text"]

# Create Document with transcription

doc = Document(

page_content=transcript,

metadata={

"source": "audio_transcription",

"model": "whisper-large-v3"

}

)

return doc

finally:

# Clean up temporary file

if os.path.exists(temp_audio_path):

os.unlink(temp_audio_path)

except Exception as e:

logger.error(f"Audio transcription failed: {e}")

raise



# src/anubis/webapp.py
import os
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import asyncio

from src.anubis.utils.context import GlobalContext, UserContext, AssistantContext

import logging
logger = logging.getLogger(__name__)

# Preload audio to text processor [this needs a startup in a lifecycle call]
from src.subgraphs.process_media_graph.utils.helper_functions import get_whisper_pipeline
pipe=get_whisper_pipeline()

app = FastAPI(title="Media Processing API")



@app.get("/hello")
def test_hello_world():
    logger.info(f"HELLO WORLD ENTRY")
    return {"Hello": "World"}

@app.post("/upload-media")
async def upload_media(
    files: List[UploadFile] = File(...),
    user_id: str = Form(default="test_user_1234"),
    assistant_id: str = Form(default="default_assistant"),
):
    # Context user_id, assistant_id
    logger.info(f"UPLOAD MEDIA ENDPOINT ENTRY")
    """
    Upload one or more media files for processing and indexing.
    
    - **files**: One or more files to process
    - **user_id**: User identifier
    - **assistant_id**: Assistant identifier
    """
    try:
        # Read all uploaded files


        media_files = []
        for file in files:
            content = await file.read()
            media_files.append({
                "filename": file.filename,
                "content_type": file.content_type,
                "content": content,
                "user_id": user_id,
                "assistant_id": assistant_id
            })
        
        # Import graph here to avoid circular imports
        from src.subgraphs.process_media_graph.process_media_graph_api_endpoint import process_media_graph_api_endpoint
        
        # Prepare input state
        initial_state = {
            "media_files": media_files,
        }
        
        # Prepare context/config
        # config = {
        #     "configurable": {
        #         "user_id": user_id,
        #         "assistant_id": assistant_id,
        #         # Add other configuration as needed
        #     }
        # }
        
        # Invoke the graph
        result = await process_media_graph_api_endpoint.ainvoke(initial_state)
        
        # Extract indexed documents info
        indexed_docs = result.get("vectorstore_documents_to_be_indexed", [])
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "files_processed": len(files),
                "documents_indexed": len(indexed_docs),
                "filenames": [f.filename for f in files],
                "message": "Media processed and indexed successfully"
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing media: {str(e)}"
        )


@app.post("/process-media-json")
async def process_media_json(
    media_list: List[dict],
    user_id: str = "test_user_1234",
    assistant_id: str = "default_assistant"
):
    """
    Process media from JSON payload (for pre-encoded base64 data).
    
    Expected format:
    {
        "media_list": [
            {
                "type": "image",
                "data": "base64_encoded_data",
                "metadata": {...}
            }
        ],
        "user_id": "user123",
        "assistant_id": "assistant456"
    }
    """
    try:
        from src.subgraphs.process_media_graph.process_media_graph_api_endpoint import process_media_graph_api_endpoint
        
        initial_state = {
            "media_list": media_list,
            
        }
        
        # config = {
        #     "configurable": {
        #         "user_id": user_id,
        #         "assistant_id": assistant_id,
        #     }
        # }
        
        result = await process_media_graph_api_endpoint.ainvoke(initial_state)
        
        indexed_docs = result.get("vectorstore_documents_to_be_indexed", [])
        
        return {
            "status": "success",
            "media_items_processed": len(media_list),
            "documents_indexed": len(indexed_docs)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing media: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)