from typing import Optional

from fastapi import FastAPI

from langgraph_sdk import get_client

from supabase import Client

from contextlib import asynccontextmanager

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Form, status
from typing import Annotated
from fastapi.responses import Response, JSONResponse
from typing import List


import logging
logger = logging.getLogger(__name__)

from scalar_fastapi import get_scalar_api_reference

import os
import debugpy
load_dotenv()

if os.getenv("DEBUG", "false").lower() == "true":
    debugpy.listen(("0.0.0.0", 5687))
    logger.info("waiting for debugger to attach on port 5687")
    debugpy.wait_for_client()
    logger.info("Debugger attached")

@asynccontextmanager
async def lifespan(app: FastAPI):

    try:
        logger.info(f"lifespan initializing...")
        

        url = os.getenv("LANGSMITH_API_URL")    
        api_key = os.getenv("LANGSMITH_API_KEY")

        logger.info(f"url: {url}")
        logger.info(f"api_key: {api_key}")


        if os.getenv("DEV") == "TRUE":
            app.state.langgraph_sdk_client = get_client(url=url, api_key=api_key)
        else: 
            url = os.getenv("LANGSMITH_API_URL")    
            api_key = os.getenv("LANGSMITH_API_KEY")
            app.state.langgraph_sdk_client = get_client(url=url, api_key=api_key)

        logger.info("breakpoints")
        assistants = await app.state.langgraph_sdk_client.assistants.search()
        logger.info(f"assistants: {assistants}")

        yield

    except Exception as e:
        logger.error(f"Error: {e}")    
        raise e

app = FastAPI(
    title="Neural Nexus API", 
    description="API to interface with the Anubis graph",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/status")
async def status():
    return {"status": "okay"}

from src.classes.ProcessMediaApiEndpoint import process_uploaded_files_and_label_media_type, convert_media_list_to_text_document, index_docs


# TODO: YIELDING SUCCESS BEFORE COMPLETION
# TODO: MULTIPLE API FAILURES WHEN SENDING DATA
# TODO: DATA IS NOT SEMANTICALLY MEANINGFUL AND THE DATA NEEDS TO BE CLEANED

# I would also optionally like to see progress of the document processing as a streamable response

@app.post("/upload-media")
async def upload_media(
    files: Annotated[List[UploadFile], File(...)],
    user_id: str = Form(default="61f439e3-8557-4710-9d81-13124b35ceca"),
    assistant_id: str = Form(default="83945802-958b-49a4-8b12-ecaa1bd2e44e"),
    reference_audio: bool = False,
    reference_image: bool = False,
    proprietary_content: bool = False
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


        langgraph_sdk_client = app.state.langgraph_sdk_client    


        config = {
            "configurable":{
                "user_ctx": {"user_id": user_id},
                "assistant_ctx":{"assistant_id":assistant_id}
            }
        }

        response = await process_uploaded_files_and_label_media_type(media_files, config)
        media_list = response['media_list']
        if len(media_list) == 0:
            raise HTTPException(status_code = 500, detail="No media to process. Media List is empty.")

        result = await convert_media_list_to_text_document(media_list, config, client = langgraph_sdk_client)
        # handle vectorstore documents to be indexed
        vectorstore_documents_to_be_indexed = result.get("vectorstore_documents_to_be_indexed", [])

        # Extract indexed documents info for response
        index_result = await index_docs(vectorstore_documents_to_be_indexed, config, client=langgraph_sdk_client)

        # handle documents to be analyzed for context storage and prompt injection of assistantimpor
        # handle documents to be processed for adapter training

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "files_processed": len(files),
                "documents_indexed": len(index_result),
                "filenames": [f.filename for f in files],
                "message": "Media processed and indexed successfully"
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing media: {str(e)}"
        )

@app.post("/test_client")
async def test_client():
    logger.info("breakpoints")
    assistants = await app.state.langgraph_sdk_client.assistants.search()
    namespaces = await app.state.langgraph_sdk_client.store.list_namespaces()
    logger.info(f"assistants: {assistants}")
    logger.info(f"namespaces: {namespaces}")
    return {"namespaces": namespaces}
    

# @app.post("/process-media-json")
# async def process_media_json(
#     media_list: List[dict],
#     user_id: str = "test_user_1234",
#     assistant_id: str = "project_gutenberg_assistant_uuid_1234", 
#     reference_audo: bool = False, 
#     reference_image: bool = False
# ):
#     """
#     Process media from JSON payload (for pre-encoded base64 data).
    
#     Expected format:
#     {
#         "media_list": [
#             {
#                 "type": "image",
#                 "data": "base64_encoded_data",
#                 "metadata": {...}
#             }
#         ],
#         "user_id": "user123",
#         "assistant_id": "assistant456"
#     }
#     """
#     try:
#         from src.subgraphs.process_media_graph.process_media_graph_api_endpoint import process_media_graph_api_endpoint
        
#         initial_state = {
#             "media_list": media_list,   
#         }

        
#         config = {
#             "configurable": {
#                 "user_ctx": {"user_id":user_id},
#                 "assistant_ctx": {"user_id":user_id, "assistant_id":assistant_id}
#             }
#         }
#         result = await process_media_graph_api_endpoint.ainvoke(initial_state, config)
#         indexed_docs = result.get("vectorstore_documents_to_be_indexed", [])
#         return {
#             "status": "success",
#             "media_items_processed": len(media_list),
#             "documents_indexed": len(indexed_docs)
#         }
    
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error processing media: {str(e)}"
#         )
    


@app.get("/", include_in_schema=False)
async def scalar_docs():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
    )