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

from urllib.parse import quote
    
from src.security.auth import update_assistant_config

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
        WHERE (metadata->>'is_public')::boolean = TRUE
        AND assistant_id = %s
        """
    elif user_id:
        # Retrieve all public avatars not owned by the current user.
        search_query = """
        SELECT * FROM assistant
        WHERE (metadata->>'is_public')::boolean = TRUE
        AND (metadata->>'user_id') != %s
        """
    else:
        # Retrieve all public avatars
        search_query = """
        SELECT * FROM assistant
        WHERE (metadata->>'is_public')::boolean = TRUE
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
    app.state.stripe = stripe
    app.state.stripe.api_key = app.state.context.stripe_secret_key
    
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
    lifespan=lifespan,
)

@app.get("/*", include_in_schema=False)
async def documentation():
    return RedirectResponse(url="/docs")

@app.get("/", include_in_schema=False)
async def documentation():
    return RedirectResponse(url="/docs")

app.include_router(router=security_route)
from fastapi.responses import HTMLResponse, RedirectResponse

import stripe
@app.get("/subscribe")
async def subscribe(current_user: dict = Depends(get_current_user)):
    """
    Create a monthly subscription.
    """

    verified_email = current_user.get("email_verified", None)
    if not verified_email:
        raise HTTPException(detail="Please verify your email before subscribing.", status_code=401)
    
    return RedirectResponse(url=app.state.context.stripe_payment_url)

    # try: 
    #     checkout_session = stripe.checkout.Session.create(
    #         line_items=[
    #             {
    #                 # Provide the exact Price ID 
    #                 'price': app.state.context.stripe_product_id
    #             }
    #         ], 
    #         mode='subscription',
    #         success_url="api.neuralnexus.site" + '?success=true',
    #         automatic_tax={'enabled': True},
    #     )
    # except Exception as e:
    #         return str(e)
    # return RedirectResponse(url=checkout_session.url, code=303)



@app.post("/unsubscribe")
async def unsubscribe():
    """
    Cancel a monthly subscription.
    """



# shivon zilis assistant_id: 59b682f8-9a9c-4f01-bc86-29d487131e5e
# test user_id: 61f439e3-8557-4710-9d81-13124b35ceca

@app.post("/create_avatar")
async def create_avatar(
    name: str, 
    description: Optional[str] = None,
    is_public: Optional[bool] = False,
    # is_self_avatar: Optional[bool] = False,
    current_user: dict = Depends(get_current_user),
    ):

    # If the avatar is of the individual, then the avatar is allowed to be made public. 
    # Reference image, audio, and third-party authenticated account is required to create a shareable avatar. Limited to one shareable avatar of themselves.
    # Include reference image, reference audio

    logger.info(f"breakpoint")

    context = app.state.context

    if current_user['identities'][0]['user_id'] == context.anonymous_user_id:
        return JSONResponse(
            content="User must be logged in to create avatars.", 
            status_code=400
        )
    
    try:
        # token = auth_cred.credentials
        # client = get_client(headers={"Authorization": f"Bearer {token}"})
        assistant_id = str(uuid4())
        user_id = current_user['identities'][0]['user_id']
        metadata = {
                "user_id": user_id,
                "is_public": False
            }
        
        if user_id == context.admin_user_id:
            metadata['is_public'] = is_public

        token = current_user['API_KEY']
        headers = {"Authorization": f"Bearer {token}"}
        client = get_client(headers=headers)
        
        create_avatar_response = await client.assistants.create(
            graph_id = "Anubis", 
            description=description, 
            name=name, 
            assistant_id=assistant_id, 
            metadata=metadata)
        
        return JSONResponse(content=create_avatar_response, status_code=200)
    except Exception as e:
        return HTTPException(detail = "Error creating avatar {name}: {e}", status_code=500)


@app.post("/share_avatar")
async def share_avatar(
    assistant_id: str,
    is_public: bool = True,
    current_user: dict = Depends(get_current_user)
):
    context = app.state.context
    user_id = current_user['identities'][0]['user_id']
    if ((user_id is context.admin_user_id)):
            """ verify users are creating avatars of their own likeness in the future"""
            metadata = {"is_public": is_public}

    # Only admins may share avatars; 
    # Users will authenticate and share avatars in the near future.
    if user_id is context.admin_user_id:
        try:
            token = current_user['API_KEY']
            client = get_client(headers={"Authorization": f"Bearer {token}"})
            result = await client.assistants.update(
                assistant_id=assistant_id, 
                metadata=metadata)
            return JSONResponse(result, status_code=200)
        except Exception as e:
            raise HTTPException(status_code=500, detail = f"Error during update of sharing avatar: {e}")
    raise HTTPException(status_code=401, detail="Users may only share avatars of themselves.")


@app.patch("/modify_avatar")
async def modify_avatar(
    assistant_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    new_avatar_name: Optional[str] = None,
    new_avatar_description: Optional[str] = None,
):
    # Avatar name changes also need to be applied to the db for consistent identities
    logger.info("breakpoint")

    if not assistant_id:
        raise HTTPException(
            detail = "Supply assistant_id for the assistant to modify.",
            status_code=400
        )
    if not new_avatar_name and not new_avatar_description:
        raise HTTPException(
            detail = "Either supply the new avatar name or the new avatar description.",
            status_code=400
        )

    if not current_user:
        raise HTTPException(
            content="User must be logged in to modify avatar avatars.", 
            status_code=401
        )
    
    token = current_user["API_KEY"]
    client = get_client(headers={"Authorization": f"Bearer {token}"})
    if assistant_id:
        if new_avatar_name and new_avatar_description:
            result = await client.assistants.update(
                graph_id="Anubis", 
                assistant_id=assistant_id, 
                name=new_avatar_name, 
                description=new_avatar_description)
        elif new_avatar_description:
            result = await client.assistants.update(
                graph_id="Anubis", 
                assistant_id=assistant_id, 
                description=new_avatar_description)
        else:
            result = await client.assistants.update(
                graph_id="Anubis", 
                assistant_id=assistant_id, 
                name=new_avatar_name)
        try:
            assert(type(result) == dict)

            # Update the selected avatar if the avatar was selected
            # selected_assistant_id = current_user.get('app_metadata', {}).get("assistant_config", {}).get("configurable", {}).get("assistant_id", None)

            # selected_assistant_config = current_user.get('app_metadata', {}).get("assistant_config", {})

            # if selected_assistant_id:
            #     if selected_assistant_id == assistant_id:
            #         if selected_assistant_config.get("configurable", {}).get("assistant_ctx", None):
            #             if new_avatar_description:
            #                 selected_assistant_config['configurable']['assistant_ctx']['description'] = new_avatar_description

            #             if new_avatar_name: 
            #                 selected_assistant_config['configurable']['assistant_ctx']['name'] = new_avatar_description

            #             provider_encoded_user_id = quote(current_user['user_id'], safe="")

            #             hashed_api_key = current_user['app_metadata']['api_key']
            #             update_assistant_config(hashed_api_key, provider_encoded_user_id, assistant_config=selected_assistant_config)

            return JSONResponse(content=result, status_code=200)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error updating assistant.")

@app.delete("/delete_avatar")
async def delete_avatar(
    assistant_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    # TODO: Delete avatar in database
    logger.info("breakpoint")

    token = current_user["API_KEY"]
    user_id = current_user['identities'][0]['user_id']
    client = get_client(headers={"Authorization": f"Bearer {token}"})
    
    metadata = {'user_id':user_id}
    metadata.update({"assistant_id": assistant_id})
        # Delete all entries in the store and store vectors for the created avatars
    pool = request.app.state.pool
    SQL_STORE_DELETE_QUERY="""DELETE FROM store WHERE prefix = %s OR prefix LIKE %s or prefix LIKE %s or prefix LIKE %s;"""
    SQL_STORE_VECTOR_DELETE_QUERY="""DELETE FROM store WHERE prefix = %s OR prefix LIKE %s or prefix LIKE %s or prefix LIKE %s;""" 
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                params = (
                    assistant_id, 
                    f"{assistant_id}.%",
                    f"%.{assistant_id}.%",
                    f"%.{assistant_id}",
                )
                await cur.execute(SQL_STORE_DELETE_QUERY, params)
                await cur.execute(SQL_STORE_VECTOR_DELETE_QUERY, params)
    except Exception as e:
        raise HTTPException(detail="Error deleting items from store and store vectors during delete avatar.", status_code=500)

    try:
        await client.assistants.delete(assistant_id=assistant_id, delete_threads=True)
    except Exception as e:
        raise HTTPException(detail = "Error Deleting Assistant", status_code=500)
    
    return JSONResponse("Deleted Avatar Successfully", status_code=200)


# @app.get("/test")
# async def test(current_user: dict = Depends(get_current_user)):
#     return {"current_user": current_user}

@app.get("/list_public_avatars")
async def list_public_avatars(assistant_id: Optional[str] = None):
    logger.info("breakpoint")
    public_avatars_result = await get_public_avatars(assistant_id=assistant_id)
    return public_avatars_result

@app.get("/list_user_avatars")
async def list_user_avatars(
    current_user: dict = Depends(get_current_user),
    ):
    logger.info("breakpoint")
    if not current_user:
        public_avatars_result = await get_public_avatars()
        return public_avatars_result
    try:
        public_avatars_result = await get_public_avatars(user_id=current_user['identities'][0]['user_id']) 
        token = current_user['API_KEY']
        client = get_client(headers = {"Authentication": f"{token}"})
        response = await client.assistants.search(metadata={"user_id": current_user['identities'][0]['user_id']})
        if len(response) > 0:
            avatar_list = response
            public_avatars_result.extend(avatar_list) # public and private avatars
        return JSONResponse(public_avatars_result, status_code=200)
    except Exception as e:
        error = f"Error in listing avatars: {e}"
        return HTTPException(detail=error, status_code=500)



@app.post("/select_avatar")
async def select_avatar(
    request: Request,
    response: Response, 
    current_user: dict = Depends(get_current_user),
    assistant_id: Optional[str] = None, 
    assistant_name: Optional[str] = None,
    ):
    logger.info("breakpoint")
    if not current_user and not assistant_id:
        return HTTPException(status_code=400, detail="Unauthenticated users must log in to use the select avatars via name feature. Please log in or use an assistant_id for selection.")
    
    assistant_config = {"configurable": {
        "assistant_id": assistant_id
    }}

    public_avatar_result = await get_public_avatars(assistant_id=assistant_id)
    
    # if not current_user['identities'][0]['user_id'] is request.app.state.context['anonymous_user_id']: # anonymous user case
    if not current_user:
        if len(public_avatar_result) > 0:
            assistant_config['configurable'].update({
                "assistant_ctx": {
                    "name": public_avatar_result[0].get("name", None),
                    "description": public_avatar_result[0].get("description", None)
                }
            })
        
        public_avatar_result = await update_assistant_config(assistant_config = assistant_config, request=request)
        return assistant_config
    else:
        token = current_user['API_KEY']
        client = get_client(headers={"Authorization": f"{token}"})
        user_id = current_user['identities'][0]['user_id']
        if assistant_id:
            try:
                if len(public_avatar_result) == 0: # the avatar was not public
                    result = await client.assistants.get(assistant_id=assistant_id) # attempt to get user-specific avatar with api key
                    if not result:
                        raise HTTPException(detail="Assistant not found: {assistant_id}", status_code=500)
                        # assistant = {"name": None, "description": None}
                    else:
                        assistant = result
                    logger.info(f"result:{result}")
                    assistant_config = {
                        "configurable": {
                            "assistant_id": assistant_id,
                            "assistant_ctx": {
                                "name":assistant.get("name", ""),
                                "description":assistant.get("description", ""),
                                "metadata": assistant.get("metadata", {})
                            }
                        }
                    }
                else:
                    assistant_config['configurable'].update({
                        "assistant_ctx": {
                            "name": public_avatar_result[0].get("name", None),
                            "description": public_avatar_result[0].get("description", None),
                            "metadata": public_avatar_result[0].get("metadata", {})
                        }
                    })
                provider_encoded_user_id = quote(current_user['user_id'], safe="")

                hashed_api_key = current_user['app_metadata']['api_key']
                update_assistant_result = await update_assistant_config(
                    hashed_api_key = hashed_api_key,
                    provider_encoded_user_id=provider_encoded_user_id, 
                    assistant_config = assistant_config, 
                    request=request)
                return assistant_config
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error using assistant_id for logged in user {e}")
        elif assistant_name:
            try:
                result = await client.assistants.search(name=assistant_name)
                try:
                    if (len(result) == 0):
                        raise HTTPException(detail="Assistant not found.", status_code=400)
                    assistant = result[0]
                    is_public = assistant.get("metadata", {}).get("is_public", False)
                    if not is_public and (current_user['identities'][0]['user_id'] != assistant.get('metadata', {}).get('user_id', None)):
                        raise HTTPException(detail="Non-public avatar id.", status_code=401)
                    else:
                        assistant_config = {
                            "configurable": {
                                "assistant_ctx": {
                                    "name": assistant.get('name', None),
                                    'description': assistant.get('description', None)
                                },
                                "assistant_id": assistant.get("assistant_id", None)
                            }
                        }
                    hashed_api_key = current_user['app_metadata']['api_key']
                    provider_encoded_user_id = quote(current_user['user_id'], safe="")
                    result = await update_assistant_config(
                        hashed_api_key=hashed_api_key,
                        provider_encoded_user_id=provider_encoded_user_id, 
                        assistant_config = assistant_config, 
                        request=request)
                    
                    return JSONResponse(content=assistant_config, status_code=200)
                except Exception as e:
                    raise HTTPException(status_code=500, detail = f"Error during avatar selection via assistant_name: {e}")
            except Exception as e:
                error_str = "{error}".format(error = e)
                return HTTPException(detail = error_str, status_code=500)
        else: 
            return HTTPException(detail = "Error: either assistant_id or assistant_name is required.", status_code=400)

from src.anubis.utils.tools.identity.identity_tools import learn_information_about_yourself_through_text_from_the_user_as_a_memory

from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig

# from langgraph.types import StreamWriter

# from langgraph.prebuilt import ToolRuntime
# ToolRuntime(state=[], store=app.state.store, context = app.state.context, config=config, streamWriter = StreamWriter())


# @app.post("/update_avatar_identity")
# async def update_avatar_identity(

#     assistant_fact: str,
#     current_user: dict = Depends(get_current_user), 
#     ):

#     assistant_config = current_user.get('app_metadata', {}).get('assistant_config', {})
#     user_id = current_user.get("identities", {}).get("user_id", None)




        # context=app.state.context
        # store = app.state.store
        # runtime = Runtime(context=context, store=store)
        # config = {
        #     "configurable": {
        #         "user_id": None,
        #         "assistant_id": None,
        #         "user_ctx": {
        #             "name": None, "description": None
        #         },
        #         "assistant_ctx": {
        #             "name": None, 
        #             "description": None
        #         }
        #     }
        # }

#     learn_information_about_yourself_through_text_from_the_user_as_a_memory(
#         assistant_fact: str, fact_context: str, )


class MessagePayload(BaseModel):
    message: Optional[str] = "Hello!"
    name: Optional[str] = None
    description: Optional[str] = None

from time import time_ns

@app.post("/message")
async def message(
    response: Response,    
    body: MessagePayload,
    current_user: dict = Depends(get_current_user),
    ):

    logger.info("breakpoint")
    # allow for select avatar in query and anonymous user for a dedicated endpoint
    start_time = time_ns()
    assistant_config = current_user.get("app_metadata", {}).get("assistant_config", {})
    if not assistant_config:
        raise HTTPException(detail="Please select assistant before messaging.", status_code=400)

    if not current_user:
        user_id = str(uuid5(NAMESPACE_URL, 'ANONYMOUS_USER'))
        user_name = None,
        user_description = None
    else:
        user_id = current_user['identities'][0]['user_id']
        user_name = body.name
        user_description = body.description

    config = {
            "configurable": {
                "user_ctx": {"name":user_name, "description": user_description},
                "user_id":user_id
            }
        }
    
    # update with assistant information
    config['configurable'].update(assistant_config['configurable'])

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

    response = {}
    response["content"] = result['messages'][-1].content
    response["response_metadata"] = result['messages'][-1].response_metadata
    response['total_response_time_ms'] = ((time_ns() - start_time) // 1000000)
    return JSONResponse(response, status_code=200)

from typing import Annotated

@app.post("/chat")
async def chat(
    response: Response,    
    message: Annotated[str, Form(...)],
    file: Annotated[UploadFile, File(...)],
    current_user: dict = Depends(get_current_user)
    ):

    logger.info("breakpoint")
    # allow for select avatar in query and anonymous user for a dedicated endpoint
    start_time = time_ns()
    assistant_config = current_user.get("app_metadata", {}).get("assistant_config", {})
    if not assistant_config:
        raise HTTPException(detail="Please select assistant before messaging.", status_code=400)

    if not current_user:
        user_id = str(uuid5(NAMESPACE_URL, 'ANONYMOUS_USER'))
        user_name = None,
        user_description = None
    else:
        user_id = current_user['identities'][0]['user_id']
        user_name = None
        user_description = None 

    config = {
            "configurable": {
                "user_ctx": {"name":user_name, "description": user_description},
                "user_id":user_id
            }
        }
    
    # update with assistant information
    config['configurable'].update(assistant_config['configurable'])

        # store = app.state.store
    graph = app.state.graph

        # system_time = datetime.now(tz=timezone.utc).isoformat
        # content = [{"type":"text", "text": system_time}]
        # input = {"messages": HumanMessage(content=content)}
        # # store = make_pg_store()

        # result = await url_loading_graph.ainvoke(input, config=config)

        # logger.info(f"config: {config}")

    if file.content_type == "text/plain":
        contents = await file.read()
        content = contents.decode('utf-8')
    else:
        content = ""
    
    human_message_content = message + "\n\n" + content
    result = await graph.ainvoke(input={"messages":[HumanMessage(content=human_message_content)]}, config = config )

    logger.info(f"{result}")

    response = {}
    response["content"] = result['messages'][-1].content
    response["response_metadata"] = result['messages'][-1].response_metadata
    response['total_response_time_ms'] = ((time_ns() - start_time) // 1000000)
    return JSONResponse(response, status_code=200)


@app.post("/upload_media_file")
async def upload_media_file(
    files: List[UploadFile] = File(...),
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
    user_id = current_user['identities'][0]['user_id']
    assitant_config = current_user['app_metadata']['assistant_config']
    assistant_id = assitant_config['configurable']['assistant_id']
    config = {
        "configurable": {
            "user_id": user_id,
            "user_ctx": {"name":None, "description": None},
        }
    }
    config['configurable'].update(assitant_config['configurable'])
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
    logger.info("breakpoint")


@app.post("/update_avatar_identity_with_media")
async def update_avatar_identity_with_media(
    files: List[UploadFile] = File(...),
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

        user_id = current_user['identities'][0]['user_id']
        assitant_config = current_user['app_metadata']['assistant_config']
        assistant_id = assitant_config['configurable']['assistant_id']
        config = {
            "configurable": {
                "user_id": user_id,
                "user_ctx": {"name":None, "description": None},
            }
        }

        config['configurable'].update(assitant_config['configurable'])

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

        logger.info("breakpoint")

        context=app.state.context
        store = app.state.store
        
        # Import graph here to avoid circular imports

        from src.subgraphs.process_media_graph.process_media_graph_api_endpoint import workflow
        # process_media_graph_api_endpoint

        process_media_graph_api_endpoint = workflow.compile(store=store)
        
        # Prepare input state
        initial_state = {
            "media_files": media_files,
        }
    
        logger.info(f"breakpoint before process_media_graph")
        result = await process_media_graph_api_endpoint.ainvoke(
            initial_state, 
            config=config,   
            )
    
        # Extract indexed documents info
        indexed_docs = result.get("vectorstore_documents_to_be_indexed", [])
        if len(indexed_docs) == 0:
            return HTTPException(status_code=500, detail={
                    "files_processed": len(files),
                    "documents_indexed": len(indexed_docs),
                    "filenames": [f.filename for f in files],
                    "message": "Error processing and indexing media"
                })
        else:
            return JSONResponse(
                status_code=200,
                content={
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

@app.get("/list_avatar_documents")
async def list_avatar_documents(current_user: dict = Depends(get_current_user)):
    token = current_user['API_KEY']
    langgraph_sdk_client = get_client(headers={"Authorization": f"{token}"})
    user_id = current_user['identities'][0]['user_id']
    assistant_id = current_user['app_metadata'].get("assistant_config", {}).get("configurable", {}).get("assistant_id", None)
    if assistant_id is None:
        raise HTTPException(detail="Please select an avatar before continuing.", status_code=400)
    results = {}
    namespace = (user_id, assistant_id)
    all_namespaces = await langgraph_sdk_client.store.list_namespaces(namespace, limit=1000000)
    all_document_items = await langgraph_sdk_client.store.search_items(namespace, limit=1000000)
    # results['namespaces'] = all_namespaces
    # results['documents'] = all_document_items
    uploaded_documents = [item['value']['document']['kwargs']['metadata'].get("filename", None) for item in all_document_items['items']]
    uploaded_documents = [item for item in set(uploaded_documents) if item is not None]
    results['uploaded_documents'] = uploaded_documents
    return results

@app.delete("/delete_avatar_document")
async def delete_avatar_documents(source_document_name: str, current_user: dict = Depends(get_current_user)):
    token = current_user['API_KEY']
    langgraph_sdk_client = get_client(headers={"Authorization": f"{token}"})
    user_id = current_user['identities'][0]['user_id']
    assistant_id = current_user['app_metadata'].get("assistant_config", {}).get("configurable", {}).get("assistant_id", None)
    if assistant_id is None:
        raise HTTPException(detail="Please select an avatar before continuing.", status_code=400)
    
    pool = app.state.pool

    SQL_STORE_DELETE_QUERY="""DELETE FROM store WHERE prefix = %s OR prefix LIKE %s or prefix LIKE %s or prefix LIKE %s"""
    SQL_STORE_VECTOR_DELETE_QUERY="""DELETE FROM store WHERE prefix = %s OR prefix LIKE %s or prefix LIKE %s or prefix LIKE %s;""" 
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                params = (
                    f"{user_id}.{assistant_id}.{source_document_name}",
                    f"{user_id}.{assistant_id}.{source_document_name}.%",
                    f"{user_id}.{assistant_id}.%.{source_document_name}",
                    f"{user_id}.{assistant_id}.%.{source_document_name}.%",
                )
                await cur.execute(SQL_STORE_DELETE_QUERY, params)
                await cur.execute(SQL_STORE_VECTOR_DELETE_QUERY, params)

        return JSONResponse(content=f"Successfully deleted: {source_document_name}", status_code=200)
    except Exception as e:
        raise HTTPException(detail="Error deleting documents.", status_code=500)

# # from src.anbis.subgraphs.email import email_graph
# class EmailBody(BaseModel):
#     """ Standard Email Body Format """
#     assistant_id: str
#     email_to: str
#     email_from: str 
#     email_message: str

# @app.post("/handle_email")
# async def handle_email(body: EmailBody, current_user: dict = Depends(get_current_user)):
#     user_id = current_user['identities'][0]['user_id']
#     config = {
#             "configurable": {
#                 "user_id": user_id,
#                 "assistant_id": body.assistant_id,
#                 "user_ctx": {"name":None, "description": None},
#                 "assistant_ctx": {"name":None, "description": None}
#             }
#         }
    
     


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
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
