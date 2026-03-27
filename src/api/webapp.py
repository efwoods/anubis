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

from langgraph_sdk.auth import Auth

from src.security.auth import get_current_user
from fastapi import Response, Depends

from src.security.auth import security
from fastapi.security import HTTPAuthorizationCredentials
import json

from fastapi.responses import RedirectResponse

from uuid import uuid4, uuid5, NAMESPACE_URL

import httpx

from pydantic import BaseModel
from typing import Any
from uuid import UUID

from langgraph_sdk.schema import Assistant
from psycopg.rows import class_row


class ASSISTANT_QUERY(BaseModel):
    assistant_id: UUID
    graph_id: str
    created_at: datetime
    updated_at: datetime
    config: dict[str, Any]
    metadata: dict[str, Any]
    version: int
    name: str
    description: str | None
    context: dict[str, Any]

    def to_assistant(self) -> Assistant:
        return self.model_dump(mode='json')

import debugpy
if os.getenv("DEBUG", 'false').lower() == 'true':
        debugpy.listen(('0.0.0.0', 5678))

async def get_public_avatars(assistant_id: Optional[str] = None, 
                             user_id: Optional[str] = None):
    pool = app.state.pool

    if assistant_id:
        # Retrieve the public avatar matching the assistant_id
        search_query = """
        SELECT * FROM assistant 
        WHERE metadata @> '{"is_public": true}'
        AND assistant_id = %s
        """
    elif user_id:
        # Retrieve all public avatars not owned by the current user.
        search_query = """
        SELECT * FROM assistant
        WHERE metadata @> '{"is_public": true}'
        AND metadata->user_id = %s
        """
    else:
        # Retrieve all public avatars
        search_query = """
        SELECT * FROM assistant
        WHERE (metadata->'is_public')::boolean = TRUE
        """
# WHERE metadata->>'is_public'::boolean IS TRUE
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=class_row(ASSISTANT_QUERY)) as cur:
            if assistant_id:
                await cur.execute(search_query, (assistant_id, ))
            elif user_id:
                await cur.execute(search_query, (user_id, ))
            else:
                await cur.execute(search_query)
            data = await cur.fetchall()

            return [assistant_query.to_assistant() for assistant_query in data]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup: Preload the Whisper model pipeline
    global context
    global store_context_manager 
        
    # Initialize context / context
    app.state.context = GlobalContext()
    app.state.httpx_client = httpx.AsyncClient()
    
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

            app.state.pool = pool

            await app.state.pool.open()
            try:
                embed = "huggingface:" + app.state.context.embedding_model
                field = ["document.kwargs.page_content"]
                store = AsyncPostgresStore(app.state.pool, index = IndexConfig(dims=384, embed=embed, field=field))
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

@app.get("/")
async def documentation():
    return RedirectResponse(url="/docs")

app.include_router(router=security_route)

# shivon zilis assistant_id: 59b682f8-9a9c-4f01-bc86-29d487131e5e
# test user_id: 61f439e3-8557-4710-9d81-13124b35ceca

# @app.post("/create_avatar")
# async def create_avatar(
#     name: str, 
#     description: Optional[str] = None,
#     is_public: Optional[bool] = False,
#     # is_self_avatar: Optional[bool] = False,
#     current_user: dict = Depends(get_current_user),
#     ):

#     # If the avatar is of the individual, then the avatar is allowed to be made public. 
#     # Reference image, audio, and third-party authenticated account is required to create a shareable avatar. Limited to one shareable avatar of themselves.
#     # Include reference image, reference audio

#     logger.info(f"breakpoint")

#     if not current_user:
#         return JSONResponse(
#             content="User must be logged in to create avatars.", 
#             status_code=400
#         )
    
#     try:
#         context = app.state.context
#         # token = auth_cred.credentials
#         # client = get_client(headers={"Authorization": f"Bearer {token}"})
#         assistant_id = str(uuid4())
#         metadata = {
#                 "user_id": current_user['sub'], 
#                 "assistant_id":assistant_id,
#                 "is_public": False
#             }
        
#         if current_user.get('sub', None) is context.admin_user_id:
#             metadata['is_public'] = is_public
        
#         response = await client.assistants.create(
#             graph_id = "Anubis", 
#             description=description, 
#             name=name, 
#             assistant_id=assistant_id, 
#             metadata=metadata)
        
#         return JSONResponse(response, status_code=200)
#     except Exception as e:
#         return HTTPException(detail = e, status_code=500)


@app.get("/test")
async def test(current_user: dict = Depends(get_current_user)):
    return {"current_user": current_user}

@app.get("/list_public_avatars")
async def list_public_avatars(assistant_id: Optional[str] = None):
    logger.info("breakpoint")
    public_avatars_result = await get_public_avatars(assistant_id=assistant_id)
    return public_avatars_result
    # if not current_user:
        # return public_avatars_result
    # try: 
    #     token = current_user['app_metadata']['api_key']
    #     client = get_client(headers = {"Authentication": f"{token}"})
    #     response = await client.assistants.search(metadata={"user_id": current_user['identities'][0]['user_id']})
    #     if type(response) is list:
    #         avatar_list = response[0]
    #     else:
    #         raise AssertionError("response is not a list")
    #     logger.info(f"breakpoint")
    #     return_list = [avatar_list, public_avatars_result] # public and private avatars
    #     return JSONResponse(return_list, status_code=200)
    # except Exception as e:
    #     error = f"Error in listing avatars: {e}"
    #     return JSONResponse(error, status_code=500)


@app.get("/list_user_avatars")
async def list_user_avatars(
    current_user: dict = Depends(get_current_user),
    ):
    logger.info("breakpoint")
    public_avatars_result = await get_public_avatars()
    if not current_user:
        return public_avatars_result
    try: 
        token = current_user['API_KEY']
        client = get_client(headers = {"Authentication": f"{token}"})
        response = await client.assistants.search(metadata={"user_id": current_user['identities'][0]['user_id']})
        if len(response) > 0:
            avatar_list = response[0]
            public_avatars_result.append(avatar_list) # public and private avatars
        return JSONResponse(public_avatars_result, status_code=200)
    except Exception as e:
        error = f"Error in listing avatars: {e}"
        return HTTPException(detail=error, status_code=500)


@app.delete("/delete_avatar")
async def delete_avatar(
    assistant_id: Optional[str] = None,
    assistant_name: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    auth_cred: Optional[HTTPAuthorizationCredentials] = Depends(security)
):

    logger.info("breakpoint")

    if not assistant_id and not assistant_name:
        raise HTTPException(
            detail = "Either supply assistant_id or assistant name.",
            status_code=400
        )

    if not current_user:
        raise HTTPException(
            content="User must be logged in to delete avatars.", 
            status_code=400
        )
    
    try:
        context = app.state.context
        token = auth_cred.credentials
        client = get_client(headers={"Authorization": f"Bearer {token}"})
        
        metadata = {'user_id':current_user.get('sub', "")}
        if assistant_id:
            metadata.update({"assistant_id": assistant_id})
            await client.assistants.delete(
                graph_id = "Anubis", 
                assistant_id=assistant_id, 
                metadata=metadata)
        elif assistant_name:
            result = await client.assistants.search(graph_id="Anubis", name=assistant_name, metadata=metadata)
            assert type(result) is list
            assert len(result) > 0
            await client.assistants.delete(
                graph_id = "Anubis", 
                assistant_id=result[0].get("assistant_id"))
        else:
            raise HTTPException(
            detail = "Either supply assistant_id or assistant name.",
            status_code=400
        )
        return JSONResponse("Deleted Avatar", status_code=200)
    except Exception as e:
        return HTTPException(detail = e, status_code=500)

@app.post("/share_avatar")
async def share_avatar(
    assistant_id: str,
    is_public: bool = True,
    current_user: dict = Depends(get_current_user),
    auth_cred: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    user_id = current_user.get('sub', None)
    if ((user_id is context.admin_user_id)
        or (assistant_id is user_id)):
            metadata = {"is_public": True}

    # Only admins may share avatars; 
    # Users will authenticate and share avatars in the near future.
    if user_id is context.admin_user_id:
        try:
            token = auth_cred.credentials
            client = get_client(headers={"Authorization": f"Bearer {token}"})
            result = await client.assistants.update(
                assistant_id=assistant_id, 
                metadata=metadata)
        except Exception as e:
            raise HTTPException(status_code=500, detail = f"Error during update of sharing avatar: {e}")

    
from src.security.auth import update_assistant_config
@app.post("/select_avatar")
async def select_avatar(
    request: Request,
    current_user: dict = Depends(get_current_user),
    assistant_id: Optional[str] = None, 
    assistant_name: Optional[str] = None,
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
        result = await update_assistant_config(assistant_config = assistant_config, request=request)
        return assistant_config

    else:
        
        token = current_user['API_KEY']
        client = get_client(headers={"Authorization": f"{token}"})
        user_id = current_user['identities'][0]['user_id']
        #         api_key_hash = _hash_key(api_key)
        # payload = {
        #     "email": email, 
        #     "password": password, 
        #     "connection": CONNECTION,
        #     "name": name,
        #     "app_metadata":{
        #         "api_key": api_key_hash
        #     }
        # }

        # headers = await _mgmt_headers(request)

        # response = await request.app.state.httpx_client.post(
        #     f"{BASE_AUTH_URL}/api/v2/users",
        #     json=payload,            
        #     headers=headers,
        # ) 

        # response.raise_for_status()
                        # "user_id": user_id, 

        if assistant_id:
            try:
                result =  await client.assistants.search(
                    metadata={
                        "assistant_id":assistant_id
                        })
                if len(result)==0:
                    assistant = {"name": None, "description": None}
                else:
                    assistant = result[0]
                logger.info(f"result:{result}")
                assistant_config = {
                    "configurable": {
                        "assistant_id": assistant_id,
                        "assistant_ctx": {
                            "name":assistant.get("name", ""),
                            "description":assistant.get("description", ""),
                        }
                    }
                }
                result = await update_assistant_config(assistant_config = assistant_config, request=request)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error using assistant_id for logged in user {e}")
            
        elif assistant_name:
            try:
                result = await client.assistants.search(name=assistant_name)
                try:
                    assert type(result) is list
                    if len(result) != 0:
                        assistant = result[0]
                    else:
                        assistant = {"name": None, "description": None}
                    assistant_id = result[0].get("assistant_id", None)
                    assert assistant_id is not None
                    assistant_config = {
                        "configurable": {
                            "assistant_ctx": {
                                "name": assistant.get('name', ''),
                                'description': assistant.get('description', '')
                            },
                            "assistant_id": assistant_id
                        }
                    }
                    result = await update_assistant_config(assistant_config = assistant_config, request=request)
                except Exception as e:
                    raise HTTPException(status_code=500, detail = f"Error during avatar selection via assistant_name: {e}")
            except Exception as e:
                error_str = "{error}".format(error = e)
                return HTTPException(detail = error_str, status_code=500)
        else: 
            return HTTPException(detail = "Error: either assistant_id or assistant_name is required.", status_code=400)

    return JSONResponse(assistant_config, status_code=200)

@app.get("/message")
async def message(
    response: Response,
    message: Optional[str] = "Hello!",
    name: Optional[str] = None,
    description: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    ):

    logger.info("breakpoint")
    assistant_config = current_user.get("app_metadata", {}).get("assitant_config", {})
    if not assistant_config:
        raise HTTPException(detail="Please select assistant before messaging.", status_code=400)

    if not current_user:
        user_id = str(uuid5(NAMESPACE_URL, 'ANONYMOUS_USER'))
        user_name = None,
        user_description = None
    else:
        user_id = current_user['identities']['user_id']
        user_name = name
        user_description = description

    config = {
            "configurable": {
                "user_ctx": {"name":user_name, "description": user_description},
                "user_id":user_id
            }
        }
    
    # update with assistant information
    config['configurable'].update(assistant_config['configurable'])

    if app.state.context.deployment == 'FALSE':
        # store = app.state.store
        graph = app.state.graph

        # system_time = datetime.now(tz=timezone.utc).isoformat
        # content = [{"type":"text", "text": system_time}]
        # input = {"messages": HumanMessage(content=content)}
        # # store = make_pg_store()

        # result = await url_loading_graph.ainvoke(input, config=config)

        # logger.info(f"config: {config}")
        
        result = await graph.ainvoke(input={"messages":[HumanMessage(content=message)]}, config = config )

        logger.info(f"{result}")

        return JSONResponse(result['messages'][-1].content, status_code=200)



@app.post("/upload-media")
async def upload_media(
    files: List[UploadFile] = File(...),
    user_id: str = Form(default="test_user_1234"),
    assistant_id: str = Form(default="project_gutenberg_assistant_uuid_1234"),
    reference_audio: bool = False,
    reference_image: bool = False, 
    proprietary_content: bool = False, 
    current_user: dict = Depends(get_current_user)
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
        # from src.subgraphs.process_media_graph.process_media_graph_api_endpoint import process_media_graph_api_endpoint
        
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

    
        logger.info(f"breakpoint before process_media_graph")
        # result = await process_media_graph_api_endpoint.ainvoke(
        #     initial_state, 
        #     config=config,
        #     )
    
        # Extract indexed documents info
        # indexed_docs = result.get("vectorstore_documents_to_be_indexed", [])
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "files_processed": len(files),
                # "documents_indexed": len(indexed_docs),
                "filenames": [f.filename for f in files],
                "message": "Media processed and indexed successfully"
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing media: {str(e)}"
        )

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



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
