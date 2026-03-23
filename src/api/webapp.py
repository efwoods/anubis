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

from langgraph.store.memory import InMemoryStore
from langgraph.store.base import IndexConfig

from typing import Optional
from langgraph_sdk import get_client

from src.anubis.graph import graph, message_workflow
from src.security.auth import security_route, auth

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup: Preload the Whisper model pipeline
    global context
    global store_context_manager 
        
    # Initialize context / context
    app.state.context = GlobalContext()
    if app.state.context.deployment == 'FALSE':
        async_postgres_store_uri = app.state.context.async_postgres_store_uri
        logger.warning(f"app.state.context.dev: {app.state.context.dev}")
        if app.state.context.dev == "FALSE":
            pool = AsyncConnectionPool(
                conninfo=async_postgres_store_uri,
                max_size=20,
                kwargs={"autocommit": True, "prepare_threshold": 0},
                open=False,  # don't open on construction
            )

            await pool.open()
            try:
                embed = "huggingface:" + app.state.context.embedding_model
                field = ["document.kwargs.page_content"]
                store = AsyncPostgresStore(pool, index = IndexConfig(dims=384, embed=embed, field=field))
                await store.setup()
                logger.info("Store setup complete")

                app.state.store = store
                app.state.graph = message_workflow.compile(store=store)
                logger.info("Application startup: lifecycle complete")

                yield
            finally:
                await pool.close()
        else:
            store = InMemoryStore()
            app.state.store = store
            app.state.graph = message_workflow.compile(store=store)
            yield        
    else:
        yield
        

app = FastAPI(
    title="Neural Nexus API",
    description="LangGraph-based API",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router=security_route)

# shivon zilis assistant_id: 59b682f8-9a9c-4f01-bc86-29d487131e5e
# test user_id: 61f439e3-8557-4710-9d81-13124b35ceca
@app.get("/message")
async def message(message: Optional[str] = None):


    if getattr(app.state, "config", None) is None:
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
    else:
        config = app.state.config
    
    logger.warning("breakpoint")
    if app.state.context.deployment == 'FALSE':
        store = app.state.store
        graph = app.state.graph

        # system_time = datetime.now(tz=timezone.utc).isoformat
        # content = [{"type":"text", "text": system_time}]
        # input = {"messages": HumanMessage(content=content)}
        # # store = make_pg_store()

        # result = await url_loading_graph.ainvoke(input, config=config)

        if message is None:
            message = "Hello"

        logger.info(f"config: {config}")
        
        response = await graph.ainvoke(input={"messages":[HumanMessage(content=message)]}, config = config )

        logger.info(f"{response}")

        return JSONResponse(response['messages'][-1].content, status_code=200)

from langgraph_sdk.auth import Auth

from src.security.auth import get_current_user
from fastapi import Response, Depends

async def get_public_avatars(assistant_id: Optional[str] = None):
    pool = app.state.pool

    if assistant_id:
        search_query = """
        SELECT * FROM assistant 
        WHERE metadata @> '{"is_public": true}'"
        AND assistant_id = %s
        """
    else:
        search_query = """
        SELECT * FROM assistant 
        WHERE metadata @> '{"is_public": true}'"
        """

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            if assistant_id:
                await cur.execute(search_query, (assistant_id, ))
            else:
                await cur.execute(search_query)
            return await cur.fetchall()

from src.security.auth import security
from fastapi.security import HTTPAuthorizationCredentials

@app.post("/select_avatar")
async def select_avatar(
    response: Response,
    current_user: dict = Depends(get_current_user),
    assistant_id: Optional[str] = None, 
    assistant_name: Optional[str] = None,
    auth_cred: Optional[HTTPAuthorizationCredentials] = Depends(security)
    ):

    if not current_user and not assistant_id:
        return HTTPException(status_code=400, detail="Unauthenticated users must log in to use the select avatars via name feature. Please log in or use an assistant_id for selection.")
    
    if not current_user:
        result = get_public_avatars(assistant_id=assistant_id)

        assistant_config = {"configurable": {
            "assistant_id": assistant_id
        }}

        if len(result) > 0:
            assistant_config['configurable'].update({
                "assistant_ctx": {
                    "name": result[0].get("name", None),
                    "description": result[0].get("description", None)
                }
            })
        response.set_cookie(
            key="assistant_config",
            value = assistant_config,
            httponly=True,
            samesite="lax"
        )

    else:
        #  get the current authorization token; needs to auto configure to the running instance
        #  client = get_client(headers={"Authorization": "Bearer
        if assistant_id:
            assistant_config = {
                "configurable": {
                    "assistant_id": assistant_id,
                }
            }
        elif assistant_name:
            client = get_client()
            try:
                client = get_client()
                result = await client.assistants.search(name=assistant_name)
                if type(response) is list:
                    assistant_id = response[0]["assistant_id"]
                    config = {
                        "configurable": {
                            "assistant_ctx": {
                                "name": response[0].get('name', ''),
                                'description': response[0].get('description', '')
                            },
                            "assistant_id": assistant_id
                        }
                    }
                else:
                    raise AssertionError("response is not a list during search for assistant via name.")
            except Exception as e:
                error_str = "{error}".format(error = e)
                return JSONResponse(content = error_str, status_code=500)
        else: 
            return JSONResponse(content = "Error: either assistant_id or assistant_name is required.", status_code=500)

    configurable = getattr(app.state, "config", {}).get("configurable", {}) 

    if configurable:
        if getattr(app.state, "config", None) is not None:
            app.state.config['configurable'].update(config['configurable'])
        else:
            app.state.config = config
    else:
        if getattr(app.state, "config", None) is not None:
            app.state.config.update(config) 
        else:
            app.state.config = config

    return JSONResponse(app.state.config, status_code=200)
    
    
    # config = {
    #         "configurable": {
    #             "user_ctx": {"name":"Evan Woods", "description": "Software developer"},
    #             "assistant_ctx": {
    #                 "name":"Shivon Zilis",
    #                 "description":"Director of Operations at Neuralink"},
    #             "assistant_id":"59b682f8-9a9c-4f01-bc86-29d487131e5e", 
    #             "user_id":"61f439e3-8557-4710-9d81-13124b35ceca"
    #         }
    #     }

from fastapi.responses import RedirectResponse

@app.get("/")
async def docs_redirect():
    return RedirectResponse(url="/docs")

@app.get("/list_avatars")
async def list_avatars(
    current_user: dict = Depends(get_current_user),
    auth_cred: Optional[HTTPAuthorizationCredentials] = Depends(security)
    ):
    if not current_user:
        result = await get_public_avatars()
        return result


    try: 
        client = get_client()
        response = await client.assistants.search()
        if type(response) is list:
            avatar_list = response[0]
        else:
            raise AssertionError("response is not a list")
        return JSONResponse(avatar_list, status_code=200)
    except Exception as e:
        error = f"Error in listing avatars: {e}"
        return JSONResponse(error, status_code=500)


@app.post("create_avatar")
async def create_avatar(
    name: str, 
    description: Optional[str] = None):

    # config = getattr(app.state, "config")
    # if config.get('configurable', {}).get("user_id", None)

    try:
        client = get_client()
        response = client.assistants.create(graph_id = "Anubis", description=description, name=name)
        return JSONResponse(response, status_code=200)
    except Exception as e:
        error = "Error: {e}".format(error = e)
        return JSONResponse(error, status_code=500)
    
     

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
