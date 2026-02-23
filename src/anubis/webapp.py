# src/anubis/webapp.py
import os
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.configuration import GlobalConfiguration

import logging
logger = logging.getLogger(__name__)

# Preload audio to text processor [this needs a startup in a lifecycle call]
 
from contextlib import asynccontextmanager

from src.subgraphs.vector_store_graph.utils.retrieval import make_pg_store, make_text_encoder

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup: Preload the Whisper model pipeline
    logger.info("Application startup: Preloading Whisper model...")
    global configuration
    global store 

    try:
        # Initialize context / configuration
        configuration = GlobalConfiguration()

        # Create pipeline for audio transcription
        if configuration.dev == "TRUE":
            pass
            # from src.subgraphs.process_media_graph.utils.audio_transcription_local import get_whisper_pipeline
            # # Call the function to trigger @lru_cache and load model into memory
            # pipe = get_whisper_pipeline()
        
            # logger.info("✓ Whisper model preloaded and cached successfully")
            # logger.info(f"  - Model: openai/whisper-large-v3")
            # logger.info(f"  - Device: {pipe.device}")
            # logger.info(f"  - Ready to process audio requests")

        
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"✗ CRITICAL: Failed to preload Whisper model: {e}", exc_info=True)
        logger.error("=" * 60)
        # Decide if you want to fail fast or continue
        raise  # Uncomment to prevent startup if model loading fails
    
    yield  # Application runs here
    
    # Shutdown: Cleanup if needed
    logger.info("Shutting down application...")

app = FastAPI(
    title="Media Processing API",
    description="LangGraph-based media processing with Whisper audio transcription",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/hello")
def test_hello_world():
    logger.info(f"HELLO WORLD ENTRY")
    return {"Hello": "World"}

@app.post("/upload-media")
async def upload_media(
    files: List[UploadFile] = File(...),
    user_id: str = Form(default="test_user_1234"),
    assistant_id: str = Form(default="project_gutenberg_assistant_uuid_1234"),
    reference_audio: bool = False,
    reference_image: bool = False, 
    proprietary_content: bool = False, 
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
                "assistant_id": assistant_id,
                "reference_audio": reference_audio,
                "reference_image": reference_image, 
                "proprietary_content": proprietary_content
            })
        
        # Import graph here to avoid circular imports
        from src.subgraphs.process_media_graph.process_media_graph_api_endpoint import process_media_graph_api_endpoint
        
        # Prepare input state
        initial_state = {
            "media_files": media_files,
        }

        config = {
            "configurable": {
                "user_ctx": {"user_id":user_id},
                "assistant_ctx": {"user_id":user_id, "assistant_id":assistant_id}
            }
        }
           
        # Invoke the graph
        if configuration.dev == "TRUE":

            async with make_pg_store() as store:
                await store.setup()
                result = await process_media_graph_api_endpoint.ainvoke(
                    initial_state, 
                    config=config,
                    store=store
                    )
        else:
            result = await process_media_graph_api_endpoint.ainvoke(
                initial_state, 
                config=config
                )
    
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
    assistant_id: str = "project_gutenberg_assistant_uuid_1234", 
    reference_audo: bool = False, 
    reference_image: bool = False
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

        
        config = {
            "configurable": {
                "user_ctx": {"user_id":user_id},
                "assistant_ctx": {"user_id":user_id, "assistant_id":assistant_id}
            }
        }
        result = await process_media_graph_api_endpoint.ainvoke(initial_state, config)
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
from fastapi import Request
import httpx

@app.get("/example-api")
async def example_call_to_extend_api_for_avatars(request: Request):
    context = GlobalContext()
    root_url = str(request.base_url) 
    logger.warning(f"root_url: {root_url}")

    async with httpx.AsyncClient() as client:
        namespaces = await client.post(f"{root_url}store/namespaces",
            headers={
              "Content-Type": "application/json",
              'x-api-key': f"LANGGRAPH_API_SERVER_KEY",
            },
            json={
              "max_depth": 1,
              "limit": 100,
              "offset": 0
            }
        )
        logger.info(f"breakpoing namespaces: {namespaces}")
    return ({"namespaces": namespaces.text})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
