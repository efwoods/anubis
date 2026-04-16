# src/anubis/webapp.py
import os
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response, Depends, Request

import httpx
# from src.url_loading_graph.graph import url_loading_graph
from datetime import datetime, timezone
from langchain_core.messages import HumanMessage
from src.anubis.utils.context import GlobalContext
from psycopg_pool import AsyncConnectionPool

import logging
logger = logging.getLogger(__name__)

# Add metrics imports
import time
from time import time_ns
import uuid
from decimal import Decimal

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import CollectorRegistry

# Create a custom registry for metrics
registry = CollectorRegistry()

# Define metrics
REQUEST_COUNT = Counter(
    'anubis_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

REQUEST_LATENCY = Histogram(
    'anubis_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint'],
    registry=registry
)

ACTIVE_REQUESTS = Gauge(
    'anubis_active_requests',
    'Number of active requests',
    registry=registry
)

MODEL_TOKENS_TOTAL = Counter(
    'anubis_model_tokens_total',
    'Total number of tokens used by model',
    ['model', 'type'],  # type: prompt or completion
    registry=registry
)

MODEL_COST_TOTAL = Counter(
    'anubis_model_cost_total_usd',
    'Total cost in USD for model usage',
    ['model'],
    registry=registry
)

API_RESPONSE_STATUS = Counter(
    'anubis_api_response_status_total',
    'Response status codes',
    ['status'],
    registry=registry
)

# Preload audio to text processor [this needs a startup in a lifecycle call]
 
from contextlib import asynccontextmanager
from langgraph.store.postgres import AsyncPostgresStore

from langgraph.store.memory import InMemoryStore
from langgraph.store.base import IndexConfig

from typing import Optional
from langgraph_sdk import get_client

from src.anubis.graph import graph, message_workflow, response_only_workflow
from src.security.auth import security_route, auth

from langgraph_sdk.auth import Auth

from src.security.auth import get_current_user

from src.security.auth import security
from fastapi.security import HTTPAuthorizationCredentials
import json

from fastapi.responses import RedirectResponse, JSONResponse

from uuid import uuid4, uuid5, NAMESPACE_URL

import httpx

from pydantic import BaseModel
from typing import Any
from uuid import UUID

from langgraph_sdk.schema import Assistant
from typing import Annotated

from psycopg.rows import class_row

from urllib.parse import quote
    
from src.security.auth import update_assistant_config
from src.security.auth import check_subscription_status

from src.security.auth import get_current_user_or_anonymous_user, get_current_user_or_anonymous_user_id

from uuid import uuid4

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

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

# Metrics helper functions
async def store_api_metrics(
    request_id: str,
    user_id: str = None,
    endpoint: str = None,
    method: str = None,
    model: str = None,
    prompt_tokens: int = None,
    completion_tokens: int = None,
    total_tokens: int = None,
    request_latency_ms: int = None,
    response_status: int = None,
    cost_usd: float = None,
    thread_id: str = None,
    langsmith_trace_id: str = None,
    error_message: str = None,
    pool=None
):
    """Store API metrics in the database"""
    if not pool:
        return

    query = """
        INSERT INTO api_metrics (
            request_id, user_id, endpoint, method, model, prompt_tokens, completion_tokens,
            total_tokens, request_latency_ms, response_status, cost_usd, thread_id,
            langsmith_trace_id, error_message
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, (
                    request_id, user_id, endpoint, method, model, prompt_tokens,
                    completion_tokens, total_tokens, request_latency_ms, response_status,
                    cost_usd, thread_id, langsmith_trace_id, error_message
                ))
    except Exception as e:
        logger.error(f"Failed to store metrics: {e}")

def calculate_inference_cost(model: str, prompt_tokens: int, completion_tokens: int, context: GlobalContext) -> float:
    """Calculate OpenAI API cost based on model and token usage"""
    # Pricing as of 2026
    pricing = {
        'gpt-5.4-nano': {'prompt': context.model_prompt_cost, 'completion': context.model_completion_cost},
        'google/gemma-3-27b-it': {'prompt': context.image_model_prompt_cost, 'completion':context.image_model_completion_cost},
        'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8': {'prompt':context.llama_model_prompt_cost, "completion": context.llama_model_completion_cost}
    }

    if model not in pricing:
        return 0.0

    cost = (prompt_tokens * pricing[model]['prompt']) + (completion_tokens * pricing[model]['completion'])
    return round(cost, 9)


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

import debugpy
if os.getenv("DEBUG", 'false').lower() == 'true':
        debugpy.listen(('0.0.0.0', 5678))



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
    
    async_postgres_store_uri = app.state.context.async_postgres_store_uri
    logger.warning(f"app.state.context.dev: {app.state.context.dev}")
    pool = AsyncConnectionPool(
        conninfo=async_postgres_store_uri,
        min_size=1,
        max_size=5,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        open=False, # do not open on create
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
        checkpointer = AsyncPostgresSaver(app.state.pool)
        await checkpointer.setup()
        app.state.checkpointer = checkpointer
        app.state.graph = message_workflow.compile(store=store, checkpointer=checkpointer)
        app.state.response_only_graph = response_only_workflow.compile(store=store, checkpointer=checkpointer)
        app.state.response_only_graph.name = "Anubis"
        app.state.graph.name = "Anubis"
        logger.info("Application startup: lifecycle complete")
        yield
    finally:
        await pool.close()     
        

app = FastAPI(
    title="Neural Nexus API",
    description="LangGraph-based API",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware for request metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time_ns()
    request_id = str(uuid.uuid4())

    request.state.request_id = request_id
    ACTIVE_REQUESTS.inc()

    try:
        response = await call_next(request)
        latency_ms = (time_ns() - start_time) // 1_000_000

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=str(request.url.path),
            status=response.status_code
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=str(request.url.path)
        ).observe(latency_ms / 1000)
        API_RESPONSE_STATUS.labels(status=response.status_code).inc()

        await store_api_metrics(
            request_id=request_id,
            endpoint=str(request.url.path),
            method=request.method,
            request_latency_ms=latency_ms,
            response_status=response.status_code,
            pool=getattr(request.app.state, 'pool', None)
        )

        return response
    except Exception as e:
        latency_ms = (time_ns() - start_time) // 1_000_000
        API_RESPONSE_STATUS.labels(status=500).inc()
        await store_api_metrics(
            request_id=request_id,
            endpoint=str(request.url.path),
            method=request.method,
            request_latency_ms=latency_ms,
            response_status=500,
            error_message=str(e),
            pool=getattr(request.app.state, 'pool', None)
        )
        raise
    finally:
        ACTIVE_REQUESTS.dec()

@app.get("/metrics")
async def prometheus_metrics():
    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

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
    email = current_user.get("email")
    user_id = current_user['app_metadata']['customer_dict']['id']
    redirect_url = f"{app.state.context.stripe_payment_url}?client_reference_id={user_id}&locked_prefilled_email={email}"

    return {
        "url" : redirect_url, 
        "message":"Follow this link to subscribe."
    }

@app.get("/manage_subscription")
async def manage_subscription(current_user: dict = Depends(get_current_user)):
    return {
        "url" : "https://billing.stripe.com/p/login/eVq28s6XA53C5XpdqH1oI00", 
        "message":"Follow this link to manage your subscription."
    }

@app.get("/cancel_subscription")
async def cancel_subscription(current_user: dict = Depends(get_current_user)):
    return {
        "url" : "https://billing.stripe.com/p/login/eVq28s6XA53C5XpdqH1oI00", 
        "message":"Follow this link to manage and cancel your subscription."
    }

@app.get("/verify_subscription_status")
async def verify_subscription_status(request: Request, current_user: dict = Depends(get_current_user)):
    status = await check_subscription_status(request=request, current_user=current_user)
    if status['status'] == None:
        return {"subscription_status:" "Not Subscribed"}
    return status


@app.post("/create_avatar")
async def create_avatar(
    name: str, 
    description: Optional[str] = None,
    is_public: bool = False,
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

    if ((user_id == context.admin_user_id)):
            """ verify users are creating avatars of their own likeness in the future"""
            metadata = {"is_public": is_public}

    # Only admins may share avatars; 
    # Users will authenticate and share avatars in the near future.
    if user_id == context.admin_user_id:
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

from src.anubis.utils.tools.identity.identity_tools import update_self_identity_mem_from_user_txt

from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig

class MessagePayload(BaseModel):
    message: str = "Hey! Please tell me about yourself and what you can do for me."
    your_name: Optional[str] = None
    your_description: Optional[str] = None
    conversation_title: Optional[str] = None

import base64
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage

async def process_files_for_message(files: Optional[List[UploadFile]] = None) -> tuple:
    """Process uploaded files and return content for inclusion in messages.
    
    Returns:
        tuple: (text_content, multimodal_content)
        - text_content: str - concatenated text from text files
        - multimodal_content: list or None - multimodal content for vision models
    """
    if not files:
        return "", None

    text_contents = []
    multimodal_parts = []
    has_images = False

    for file in files:
        try:
            content = await file.read()
            filename = file.filename or "unknown_file"
            content_type = file.content_type or ""

            if content_type.startswith('image/'):
                # Encode image as base64 for vision models
                base64_image = base64.b64encode(content).decode("utf-8")
                image_url = f"data:{content_type};base64,{base64_image}"
                
                multimodal_parts.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
                has_images = True
                text_contents.append(f"[Image: {filename}]")

            elif content_type.startswith('text/') or content_type == 'application/pdf':
                # Handle text files and PDFs
                if content_type == 'application/pdf':
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                            temp_pdf.write(content)
                            temp_pdf.flush()
                            pdf_loader = PyPDFLoader(temp_pdf.name)
                            pdf_docs = pdf_loader.load()

                        pdf_text = "\n\n".join([doc.page_content for doc in pdf_docs if hasattr(doc, 'page_content')])
                        if pdf_text:
                            text_contents.append(f"[PDF File: {filename}]\n{pdf_text}")
                        else:
                            text_contents.append(f"[PDF File: {filename} - no extractable text]")
                    except Exception as pdf_error:
                        logger.error(f"Failed to extract PDF text from {filename}: {pdf_error}")
                        text_contents.append(f"[PDF File: {filename}]")
                    finally:
                        try:
                            os.unlink(temp_pdf.name)
                        except Exception:
                            pass
                else:
                    # Text files
                    try:
                        text_content = content.decode('utf-8')
                        text_contents.append(f"[File: {filename}]\n{text_content}")
                    except UnicodeDecodeError:
                        text_contents.append(f"[Binary Text File: {filename}]")

            elif content_type.startswith('audio/'):
                # Audio files - describe that audio was uploaded
                text_contents.append(f"[Audio File: {filename} - {content_type}]")

            else:
                # Other file types
                text_contents.append(f"[File: {filename} - {content_type}]")

        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            text_contents.append(f"[Error processing file: {file.filename}]")

    # Combine text content
    combined_text = "\n\n".join(text_contents) if text_contents else ""
    
    # Return multimodal content if images are present
    if has_images:
        # Create multimodal content with text and images
        multimodal_content = [{"type": "text", "text": combined_text}] + multimodal_parts
        return combined_text, multimodal_content
    
    return combined_text, None

class FeedbackData(BaseModel):
    """Feedback data for human-in-the-loop responses"""
    feedback_type: str  # 'like', 'dislike', 'rating', 'edit'
    rating: Optional[float] = None  # 1-5 scale for 'rating' type
    comment: Optional[str] = None
    edited_response: Optional[str] = None  # User edited the response
    
class MessageResponse(BaseModel):
    """Response model for message endpoints with feedback support"""
    content: str
    response_metadata: Optional[dict] = None
    total_response_time_ms: int
    thread_id: str
    request_id: str  # For feedback submission
    feedback: Optional[FeedbackData] = None

async def store_user_feedback(
    pool,
    user_id: str,
    thread_id: str,
    feedback_type: str,  # 'like', 'dislike', 'rating', 'edit'
    request_id: str = None,
    rating: Optional[float] = None,  # 1-5 scale for 'rating' type
    comment: Optional[str] = None,
    edited_response: Optional[str] = None,  # For 'edit' feedback type
) -> dict:
    """Helper function to store user feedback in the database
    
    Args:
        pool: Database connection pool
        user_id: ID of the user providing feedback
        thread_id: ID of the conversation thread being feedback on
        feedback_type: Type of feedback ('like', 'dislike', 'rating', 'edit')
        request_id: Optional request ID for tracking
        rating: Rating value (1-5) if feedback_type is 'rating'
        comment: Optional comment text
        edited_response: Edited response content if feedback_type is 'edit'
    
    Returns:
        dict: Status response with feedback_type included
        
    Raises:
        HTTPException: If validation fails or database error occurs
    """
    if not pool:
        raise HTTPException(status_code=500, detail="Database pool not available")
    
    # Validate feedback_type
    valid_feedback_types = {'like', 'dislike', 'rating', 'edit'}
    if feedback_type not in valid_feedback_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid feedback_type. Must be one of {valid_feedback_types}"
        )
    
    # Validate rating if feedback_type is 'rating'
    if feedback_type == 'rating':
        if rating is None or not (1 <= rating <= 5):
            raise HTTPException(
                status_code=400,
                detail="Rating feedback requires a rating value between 1 and 5"
            )
    
    # Validate edited_response if feedback_type is 'edit'
    if feedback_type == 'edit':
        if not edited_response:
            raise HTTPException(
                status_code=400,
                detail="Edit feedback requires the edited_response field"
            )
        # Store edited response in comment field
        comment = edited_response

    # Store feedback in database
    query = """
    INSERT INTO user_feedback (
        request_id, user_id, thread_id, feedback_type, rating, comment
    ) VALUES (%s, %s, %s, %s, %s, %s)
    """

    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, (
                    request_id,
                    user_id,
                    thread_id,
                    feedback_type,
                    rating if feedback_type == 'rating' else None,
                    comment
                ))
        logger.info(f"Stored {feedback_type} feedback for user {user_id} on thread {thread_id}")
        return {"status": "feedback recorded", "feedback_type": feedback_type}
    except Exception as e:
        logger.error(f"Failed to store feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to record feedback")

async def update_user_preferences_from_feedback(
    store,
    user_id: str,
    assistant_id: str,
    feedback_type: str,
    edited_response: Optional[str] = None,
    comment: Optional[str] = None
):
    """Update user preferences in store based on feedback"""
    if not store:
        return
    
    try:
        namespace = (user_id, assistant_id, "preferences")
        
        # Retrieve existing preferences
        try:
            existing = await store.get_items(namespace, limit=1)
            preferences = existing[0]['value'] if existing else {}
        except:
            preferences = {}
        
        # Update feedback history
        if 'feedback_history' not in preferences:
            preferences['feedback_history'] = []
        
        feedback_entry = {
            'type': feedback_type,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'comment': comment
        }
        
        # If user edited the response, capture it
        if edited_response:
            feedback_entry['edited_response'] = edited_response
            feedback_entry['edit_type'] = 'response_correction'
        
        preferences['feedback_history'].append(feedback_entry)
        
        # Update store
        await store.put_items(
            namespace,
            [{
                'key': 'user_preferences',
                'value': preferences
            }]
        )
        logger.info(f"Updated preferences for user {user_id}, assistant {assistant_id}")
    except Exception as e:
        logger.error(f"Failed to update user preferences: {e}")

from time import time_ns

@app.post("/message")
async def message_selected_avatar(
    request: Request,
    message: str = "Hey! Please tell me about yourself and what you can do for me.",
    your_name: Optional[str] = None,
    your_description: Optional[str] = None,
    conversation_title: Optional[str] = None,
    files: Optional[List[UploadFile]] = File(None),
    thread_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    ):

    logger.info("breakpoint update")
    # allow for select avatar in query and anonymous user for a dedicated endpoint
    start_time = time_ns()
    config = current_user.get("app_metadata", {}).get("assistant_config", {})
    if not config:
        raise HTTPException(detail="Error retrieving assistant information.", status_code=400)

    user_name = your_name
    user_description = your_description
    user_id = current_user['identities'][0]['user_id']
    config_update = {
            "configurable": {
                "user_ctx": {"name":user_name, "description": user_description},
                "user_id":user_id,
            }
        }
    assistant_id = config['configurable'].get("assistant_id")


    # Handle thread_id
    if not thread_id:
        thread_id = str(uuid4())
        thread_metadata = {"thread_metadata": {"user_id":user_id, "assistant_id":assistant_id}, "graph_id": "Anubis"}
        # create thread_id
        try:
            langgraph_client = get_client()
            thread_create_response = await langgraph_client.threads.create(thread_id=thread_id, 
            metadata=thread_metadata)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error creating new conversation thread.")

    # update with user information
    config_update['configurable']['thread_id'] = thread_id
    config['configurable'].update(config_update['configurable'])

    # store = app.state.store
    graph = app.state.graph

    # Process any uploaded files
    file_text_content, multimodal_content = await process_files_for_message(files)

    # Create the human message content
    if multimodal_content:
        # Use multimodal content for vision models
        human_message = HumanMessage(content=multimodal_content)
    else:
        # Use text-only content
        if file_text_content:
            human_message_content = message + "\n\n" + file_text_content
        else:
            human_message_content = message
        human_message = HumanMessage(content=human_message_content)
        
    result = await graph.ainvoke(input={"messages":[human_message]}, config = config )

    logger.info(f"{result}")

    # Update most_recent_message
    langgraph_client = get_client()
    thread_metadata = {"thread_metadata": {"user_id":user_id, "assistant_id":assistant_id, "most_recent_message": datetime.now(timezone.utc).isoformat(), "conversation_title": conversation_title}, "graph_id": "Anubis"}
    await langgraph_client.threads.update(thread_id=thread_id, metadata = thread_metadata)

    response_data = {}
    response_data["content"] = result['messages'][-1].content
    response_metadata = result['messages'][-1].response_metadata
    if response_metadata:
        response_data["response_metadata"] = response_metadata

        # Extract OpenAI usage for metrics
        usage = response_metadata.get('usage', {})
        model = response_metadata.get('model', '')
        prompt_tokens = usage.get('prompt_tokens')
        completion_tokens = usage.get('completion_tokens')
        total_tokens = usage.get('total_tokens')

        # Calculate cost
        cost_usd = calculate_inference_cost(model, prompt_tokens or 0, completion_tokens or 0) if model else 0

        # Store detailed metrics
        await store_api_metrics(
            request_id=request.state.request_id,
            user_id=user_id,
            endpoint='/message',
            method='POST',
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            request_latency_ms=((time_ns() - start_time) // 1000000),
            response_status=200,
            cost_usd=cost_usd,
            thread_id=thread_id,
            pool=request.app.state.pool
        )

    response_data['total_response_time_ms'] = ((time_ns() - start_time) // 1000000)
    response_data['thread_id'] = thread_id
    response_data['request_id'] = request.state.request_id
    return JSONResponse(response_data, status_code=200)

@app.post("/message/{assistant_id}")
async def message_avatar(
    request: Request,
    assistant_id: str,
    message: str = Form("Hey! Please tell me about yourself and what you can do for me."),
    your_name: Optional[str] = Form(None),
    your_description: Optional[str] = Form(None),
    conversation_title: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
    thread_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user_or_anonymous_user),
    ):

    logger.info("breakpoint")
    # allow for select avatar in query and anonymous user for a dedicated endpoint
    start_time = time_ns()
    config = current_user.get("app_metadata", {}).get("assistant_config", {})
    if not config:
        raise HTTPException(detail="Error retrieving assistant information.", status_code=400)

    user_name = your_name
    user_description = your_description
    user_id = current_user['identities'][0]['user_id']
    if request.headers.get("api-key") != '':
        try:
            langgraph_client = get_client()
            assistant = await langgraph_client.assistants.get(assistant_id = assistant_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error selecting avatar.")
        
        config_update = {
                "configurable": {
                    "user_ctx": {"name":user_name, "description": user_description},
                    "user_id":user_id,
                    "assistant_id":assistant_id,
                    "assistant_ctx": {
                      "name": assistant.get("name", None),
                      "description": assistant.get("description", None),
                      "metadata": assistant.get("metadata", {})
                    }
                }
            }
    
    else:
        # anonymous user_id and assistant_id is handled in the current_user dependency function
        config_update = {
            "configurable": {
                "user_ctx": {"name":user_name, "description": user_description},
            }
        }

    # Handle thread_id
    if not thread_id:
        thread_id = str(uuid4())
        thread_metadata = {"thread_metadata": {"user_id":user_id, "assistant_id":assistant_id}, "graph_id": "Anubis"}
        # create thread_id
        try:
            langgraph_client = get_client()
            thread_create_response = await langgraph_client.threads.create(thread_id=thread_id, 
            metadata=thread_metadata)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error creating new conversation thread.")

    # update with user information
    config_update['configurable']['thread_id'] = thread_id
    config['configurable'].update(config_update['configurable'])

    # store = app.state.store
    graph = app.state.graph

    # Process any uploaded files
    file_text_content, multimodal_content = await process_files_for_message(files)

    # Create the human message content
    if multimodal_content:
        # Use multimodal content for vision models
        human_message = HumanMessage(content=multimodal_content)
    else:
        # Use text-only content
        if file_text_content:
            human_message_content = message + "\n\n" + file_text_content
        else:
            human_message_content = message
        human_message = HumanMessage(content=human_message_content)
        
    result = await graph.ainvoke(input={"messages":[human_message]}, config = config )

    logger.info(f"{result}")

    # Update most_recent_message
    if conversation_title is not "":
        conversation_title_data = conversation_title    
    else:
        conversation_title_data = thread_id
    
    langgraph_client = get_client()
    thread_metadata = {"thread_metadata": {"user_id":user_id, "assistant_id":assistant_id, "most_recent_message": datetime.now(timezone.utc).isoformat(), "conversation_title": conversation_title_data}, "graph_id": "Anubis"}
    await langgraph_client.threads.update(thread_id=thread_id, metadata = thread_metadata)

    response_data = {}
    response_data["content"] = result['messages'][-1].content
    response_metadata = result['messages'][-1].response_metadata
    if response_metadata:
        response_data["response_metadata"] = response_metadata

        # Extract OpenAI usage for metrics
        usage = response_metadata.get('usage', {})
        model = response_metadata.get('model', '')
        prompt_tokens = usage.get('prompt_tokens')
        completion_tokens = usage.get('completion_tokens')
        total_tokens = usage.get('total_tokens')

        # Calculate cost
        cost_usd = calculate_inference_cost(model, prompt_tokens or 0, completion_tokens or 0) if model else 0

        # Store detailed metrics
        await store_api_metrics(
            request_id=request.state.request_id,
            user_id=user_id,
            endpoint=f'/message/{assistant_id}',
            method='POST',
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            request_latency_ms=((time_ns() - start_time) // 1000000),
            response_status=200,
            cost_usd=cost_usd,
            thread_id=thread_id,
            pool=request.app.state.pool
        )

    response_data['total_response_time_ms'] = ((time_ns() - start_time) // 1000000)
    response_data['thread_id'] = thread_id
    response_data['request_id'] = request.state.request_id
    return JSONResponse(response_data, status_code=200)

@app.get("/conversations")
async def get_all_conversations(
    assistant_id: str,
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    """Return all threads for this user + assistant, newest-first."""
    user_id = current_user["identities"][0]["user_id"]
    try:
        langgraph_client = get_client()
        threads = await langgraph_client.threads.search(
            metadata={"thread_metadata": {"user_id": user_id, "assistant_id": assistant_id}},
            sort_by="updated_at",
            sort_order="desc",
        )
        return JSONResponse(threads)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error loading threads: {exc}")
 
 
@app.get("/conversations/{thread_id}/messages")
async def get_thread_messages(
    thread_id: str,
    assistant_id: str, 
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    """Return the message history for a single thread."""
    try:
        langgraph_client = get_client()
        state = await langgraph_client.threads.get_state(thread_id=thread_id)
        messages = state.get("values", {}).get("messages", []) if state else []
        return JSONResponse({"messages": messages})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error loading messages: {exc}")
 
# @app.post("/test_upload_media_file")
# async def test_upload_media_file(
#     files: List[UploadFile] = File(...),
#     reference_audio: bool = False,
#     reference_image: bool = False, 
#     proprietary_content: bool = False, 
#     current_user: dict = Depends(get_current_user)
# ):
#     # Context user_id, assistant_id
#     logger.info(f"UPLOAD MEDIA ENDPOINT ENTRY")
#     """
#     Upload one or more media files for processing and indexing.
    
#     - **files**: One or more files to process
#     - **user_id**: User identifier
#     - **assistant_id**: Assistant identifier
#     """
#     user_id = current_user['identities'][0]['user_id']
#     assitant_config = current_user['app_metadata']['assistant_config']
#     assistant_id = assitant_config['configurable']['assistant_id']
#     config = {
#         "configurable": {
#             "user_id": user_id,
#             "user_ctx": {"name":None, "description": None},
#         }
#     }
#     config['configurable'].update(assitant_config['configurable'])
#     # Read all uploaded files
#     media_files = []
#     for file in files:
#         content = await file.read()
#         media_files.append({
#             "filename": file.filename,
#             "content_type": file.content_type,
#             "content": content,
#             "user_id": user_id,
#             "assistant_id": assistant_id,
#             "reference_audio": reference_audio,
#             "reference_image": reference_image,               
#             "proprietary_content": proprietary_content
#         })
#     logger.info("breakpoint")


@app.post("/update_avatar_identity_with_media")
async def update_avatar_identity_with_media(
    files: Optional[List[UploadFile]] = File(...),
    url: Optional[str] = None,
    assistant_id: str = None,
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

        # assitant_config = current_user['app_metadata']['assistant_config']
        # assistant_id = assitant_config['configurable']['assistant_id']  
        # config['configurable'].update(assitant_config['configurable'])
        
        config = {
            "configurable": {
                "user_id": user_id,
                "user_ctx": {"name":None, "description": None},
            }
        }

        config['configurable']['assistant_id'] = assistant_id
        config['configurable']['assistant_ctx'] = {"name":None, "description": None},
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
        # if type(e) is KeyError and e.args[0] == 'assistant_config':
            # raise HTTPException(status_code=400, detail=f"Please select an avatar to upload media to before uploading media.")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
