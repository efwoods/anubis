# src/anubis/webapp.py
import os
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import asyncio

app = FastAPI(title="Media Processing API")

@app.get("/hello")
def test_hello_world():
    return {"Hello": "World"}

@app.post("/upload-media")
async def upload_media(
    files: List[UploadFile] = File(...),
    user_id: str = Form(default="test_user_1234"),
    assistant_id: str = Form(default="default_assistant"),
):
    # Context user_id, assistant_id

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
                "content": content
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