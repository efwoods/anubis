# src/anubis/webapp.py
import os
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from fastapi import Request
import httpx
# from src.url_loading_graph.graph import url_loading_graph
from datetime import datetime, timezone
from langchain_core.messages import HumanMessage
from src.anubis.utils.context import GlobalContext
from psycopg_pool import AsyncConnectionPool

import logging
logger = logging.getLogger(__name__)

# Preload audio to text processor [this needs a startup in a lifecycle call]
 
from contextlib import asynccontextmanager
from langgraph.store.postgres import AsyncPostgresStore

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup: Preload the Whisper model pipeline
    global context
    global store_context_manager 

    # Initialize context / context
    app.state.context = GlobalContext()
    async_postgres_store_uri = app.state.context.async_postgres_store_uri
    pool = AsyncConnectionPool(
        conninfo=async_postgres_store_uri,
        max_size=20,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        open=False,  # don't open on construction
    )

    await pool.open()
    try:
        store = AsyncPostgresStore(pool)
        await store.setup()
        logger.info("Store setup complete")

        app.state.store = store
        app.state.graph = message_workflow.compile(store=store)
        logger.info("Application startup: lifecycle complete")

        yield
    finally:
        await pool.close()
    logger.info("Application startup: lifecycle...")
    # for direct db connections for efficient processing
    logger.info("Application startup: pre-create engine...")
    # engine = create_async_engine(context.vectorstore_postgres_uri)
    logger.info("Application startup: post-create engine...")
    logger.info("Application startup: pre-create async session...")
    # async_session = sessionmaker(engine, class_=AsyncSession)
    logger.info("Application startup: post-create async session...")
    # app.state.db_session = async_session
    # logger.info(app.state.db_session)
    # await engine.dispose()
    # Langgraph SDK extension 
    # app.state.db_session = async_session
    # app.state.langgraph_client = get_client()
    
    
    # async with make_pg_store() as store:
    #     await store.setup()
    #     app.state.store = store
    #     from src.subgraphs.process_media_graph.process_media_graph_api_endpoint import create_process_media_graph
    #     logger.info("Application startup: pre-create create_process_media_graph during async make_pg_store...")
    #     app.state.process_media_graph_api_endpoint = create_process_media_graph(store=store)
    #     logger.info("Application startup: post-create create_process_media_graph during async make_pg_store...")
    #     yield  # Application runs here
            
    #     # Shutdown: Cleanup if needed
    #     logger.info("Shutting down application...")
    # await engine.dispose()
    # Create pipeline for audio transcription
    # if context.dev == "TRUE":
        # pass
        # logger.info("Application startup: Preloading Whisper model...")
        # from src.subgraphs.process_media_graph.utils.audio_transcription_local import get_whisper_pipeline
        # # Call the function to trigger @lru_cache and load model into memory
        # pipe = get_whisper_pipeline()
    
        # logger.info("✓ Whisper model preloaded and cached successfully")
        # logger.info(f"  - Model: openai/whisper-large-v3")
        # logger.info(f"  - Device: {pipe.device}")
        # logger.info(f"  - Ready to process audio requests")
    # await engine.dispose()
    # Decide if you want to fail fast or continue

app = FastAPI(
    title="Neural Nexus API",
    description="LangGraph-based API",
    version="1.0.0",
    lifespan=lifespan
)


from src.anubis.graph import graph, message_workflow

# shivon zilis assistant_id: 59b682f8-9a9c-4f01-bc86-29d487131e5e
# test user_id: 61f439e3-8557-4710-9d81-13124b35ceca
@app.get("/hello")
async def test_hello_world():
    
    config = {
            "configurable": {
                "user_ctx": {"name":"Evan Woods", "description": "Software developer"},
                "assistant_ctx": {
                    "name":"Shivon Zilis",
                    "description":"Director of Operations at Neuralink"},
                "assistant_id":"59b682f8-9a9c-4f01-bc86-29d487131e5e", 
                "user_id":"61f439e3-8557-4710-9d81-13124b35ceca"
            }
        }
    
    logger.info("breakpoint")
    store = app.state.store
    graph = app.state.graph
    
    # system_time = datetime.now(tz=timezone.utc).isoformat
    # content = [{"type":"text", "text": system_time}]
    # input = {"messages": HumanMessage(content=content)}
    # # store = make_pg_store()

    # result = await url_loading_graph.ainvoke(input, config=config)

    
    response = await graph.ainvoke(input={"messages":[HumanMessage(content="test message")]}, config = config )
    logger.info(f"{response}")

    logger.info(f"HELLO WORLD ENTRY")
    return {"Hello": "okay"}

# @app.post("/upload-media")
# async def upload_media(
#     files: List[UploadFile] = File(...),
#     user_id: str = Form(default="test_user_1234"),
#     assistant_id: str = Form(default="project_gutenberg_assistant_uuid_1234"),
#     reference_audio: bool = False,
#     reference_image: bool = False, 
#     proprietary_content: bool = False, 
# ):
#     # Context user_id, assistant_id
#     logger.info(f"UPLOAD MEDIA ENDPOINT ENTRY")
#     """
#     Upload one or more media files for processing and indexing.
    
#     - **files**: One or more files to process
#     - **user_id**: User identifier
#     - **assistant_id**: Assistant identifier
#     """
#     try:

#         # Read all uploaded files
#         media_files = []
#         for file in files:
#             content = await file.read()
#             media_files.append({
#                 "filename": file.filename,
#                 "content_type": file.content_type,
#                 "content": content,
#                 "user_id": user_id,
#                 "assistant_id": assistant_id,
#                 "reference_audio": reference_audio,
#                 "reference_image": reference_image, 
#                 "proprietary_content": proprietary_content
#             })
        
#         # Import graph here to avoid circular imports
#         from src.subgraphs.process_media_graph.process_media_graph_api_endpoint import process_media_graph_api_endpoint
        
#         # Prepare input state
#         initial_state = {
#             "media_files": media_files,
#         }

#         config = {
#             "configurable": {
#                 "user_ctx": {"user_id":user_id},
#                 "assistant_ctx": {"user_id":user_id, "assistant_id":assistant_id}
#             }
#         }

#         # process_media_graph_api_endpoint = app.state.process_media_graph_api_endpoint
           
#         # Invoke the graph
#         # if context.dev == "TRUE":

#         #     async with store_context_manager as store:
#         #         await store.setup()
#         #         logger.info(f"breakpoint")
#         #         result = await process_media_graph_api_endpoint.ainvoke(
#         #             initial_state, 
#         #                 config=config,
#         #                 store=store
#         #             )
#         # else:
#         logger.info(f"breakpoint before process_media_graph")
#         result = await process_media_graph_api_endpoint.ainvoke(
#             initial_state, 
#             config=config,
#             )
#             # store = app.state.store
    
#         # Extract indexed documents info
#         indexed_docs = result.get("vectorstore_documents_to_be_indexed", [])
        
#         return JSONResponse(
#             status_code=200,
#             content={
#                 "status": "success",
#                 "files_processed": len(files),
#                 "documents_indexed": len(indexed_docs),
#                 "filenames": [f.filename for f in files],
#                 "message": "Media processed and indexed successfully"
#             }
#         )
    
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error processing media: {str(e)}"
#         )

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


# @app.get("/example-api")
# async def example_call_to_extend_api_for_avatars(request: Request):
#     context = GlobalContext()
#     root_url = str(request.base_url) 
#     logger.warning(f"root_url: {root_url}")

#     async with httpx.AsyncClient() as client:
#         namespaces = await client.post(f"{root_url}store/namespaces",
#           headers={
#               "Content-Type": "application/json",
#               'x-api-key': f"LANGGRAPH_API_SERVER_KEY",
#             },
#             json={
#               "max_depth": 1,
#               "limit": 100,
#               "offset": 0
#             }
#         )
#         logger.info(f"breakpoing namespaces: {namespaces}")
#     return ({"namespaces": namespaces.text})

# from sqlalchemy import text

# @app.get("/test_store_endpoint")
# async def test_store_access_production(request: Request):
    
    # langgraph_client = app.state.langgraph_client
    # test_search_results = await langgraph_client.assistants.search()

    # logger.info(f"test_search_results: {test_search_results}")

    # agent = test_search_results[0]

    # thread = await langgraph_client.threads.create()

    # logger.info(f"thread: {thread}")

    # test_input = {"messages": [{"role": "human", "content": "what's the weather in la"}]}

    # async for chunk in langgraph_client.runs.stream(thread['thread_id'], test_search_results["assistant_id"], input=input):
    #     logger.info(f"chunk: {chunk}")

    # db_session = app.state.db_session
    # identify = "2feaa9d8-50c0-4550-81fa-9fb79bfe23f0.Anubis"
    # logger.info(db_session)

    
    # with db_session() as session:
    #     result = await session.execute(
    #         text("SELECT * FROM store WHERE prefix LIKE :prefix"),
    #         {"prefix": f"{identify}%"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
