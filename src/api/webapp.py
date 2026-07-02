# src/anubis/webapp.py

import asyncio
import base64
import functools
import json
import os
import re
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager

# from src.url_loading_graph.graph import url_loading_graph
from datetime import datetime, timezone

# Add metrics imports
from time import time_ns
from typing import Annotated, Any, List, Optional
from uuid import UUID, uuid4

import httpx
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.base import IndexConfig
from langgraph.store.postgres import AsyncPostgresStore
from langgraph_sdk import get_client

# NOTE: ``PyPDFLoader`` is imported lazily inside ``process_files_for_message``
# (the only call site) — eager import of ``langchain_community`` adds ~7.3 s to
# every cold start because the umbrella package eagerly registers many
# integrations. The first PDF upload pays the import cost once.
# Prometheus metrics
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, BeforeValidator

from src.anubis.graph import message_workflow
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.huggingface_prefetch import ensure_huggingface_models_cached
from src.anubis.utils.store_cache import (
    invalidate_store_cache_entry,
    invalidate_store_cache_for_assistant,
)
from src.api.media_jobs import (
    MediaJob,
    create_child_job,
    create_master_job,
    get_job,
    request_cancel,
    run_batch_media_job,
)
from src.security.auth import (
    check_subscription_status,
    get_current_user,
    get_current_user_or_anonymous_user,
    security_route,
    update_assistant_config,
)


def _drop_empty_file_fields(value: Any) -> Any:
    """Normalize the multipart ``files`` field so an absent upload is treated as
    "no files" instead of raising a 422.

    Swagger UI (and some HTTP clients) submit an *empty* file field as a form
    value of ``""`` rather than omitting it. FastAPI then receives ``[""]`` and,
    while validating each element against ``UploadFile``, fails with
    ``Expected UploadFile, received: <class 'str'>`` before the endpoint body
    ever runs. Stripping the stray string(s) here turns that into an empty list.
    """
    if isinstance(value, list):
        return [item for item in value if not isinstance(item, str)]
    if isinstance(value, str):
        return []
    return value


# Reusable annotation for optional multipart file uploads. Use this instead of
# ``Optional[List[UploadFile]] = File(None)`` so empty file fields don't 422.
OptionalUploadFiles = Annotated[
    Optional[List[UploadFile]],
    BeforeValidator(_drop_empty_file_fields),
    File(),
]

import logging
from urllib.parse import quote

import stripe
from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from langgraph_sdk.schema import Assistant
from psycopg.rows import class_row

from src.anubis.utils import runtime_handles

load_dotenv()


logger = logging.getLogger(__name__)

from uuid import NAMESPACE_URL, uuid5


def _namespace_safe_formatted_filename(u: str) -> str:
    formatted_name = str(uuid5(NAMESPACE_URL, u))
    return formatted_name


def _document_label_and_key(metadata: dict) -> tuple[str | None, str | None]:
    """Map a stored Document's metadata to the pair (human label, storage key).

    The *label* is what ``/list_avatar_documents`` displays; the *key* is the
    ``namespace_filename`` that ``/delete_avatar_document`` matches store rows on.
    Keeping both in one place is what lets the two endpoints round-trip: a string
    a user copies out of the list resolves back to the exact key delete needs.

    Playlist videos carry playlist_url / playlist_title / video_title (see
    ``URLDocumentLoaderClass._load_youtube_playlist``); they are labeled
    ``{playlist_title} :: {video_title}`` but keyed by an opaque uuid5
    namespace_filename (hashed over ``{playlist_ns}::{video_ns}``) — the label is
    human-readable titles, the key is a hash, so the two never coincide. Everything
    else is both labeled and keyed by its plain filename. Titles fall back to
    URLs/filenames when yt_dlp couldn't resolve them.
    """
    filename = metadata.get("filename")
    namespace_filename = metadata.get("namespace_filename")
    key = (
        namespace_filename
        if isinstance(namespace_filename, str) and namespace_filename
        else None
    )
    playlist_url = metadata.get("playlist_url")
    if isinstance(playlist_url, str) and playlist_url:
        playlist_label = (metadata.get("playlist_title") or playlist_url).strip()
        video_label = (
            metadata.get("video_title")
            or (filename if isinstance(filename, str) else "")
            or "untitled"
        ).strip()
        return f"{playlist_label} :: {video_label}", key
    if isinstance(filename, str) and filename:
        return filename, key
    return None, key


def _iter_document_labels(store_items) -> Iterator[tuple[str, str | None]]:
    """Yield (label, key) for each stored Document, de-structuring the same
    value.document.kwargs.metadata path /list and /delete read. Multiple
    Documents per source (quote / identity / analysis) yield the same pair; the
    caller de-dupes."""
    for item in store_items or []:
        value = getattr(item, "value", None)
        if value is None and isinstance(item, dict):
            value = item.get("value")
        if not isinstance(value, dict):
            continue
        document = value.get("document")
        kwargs_blob = document.get("kwargs") if isinstance(document, dict) else None
        metadata = (
            kwargs_blob.get("metadata") if isinstance(kwargs_blob, dict) else None
        )
        if not isinstance(metadata, dict):
            continue
        label, key = _document_label_and_key(metadata)
        if label:
            yield label, key


def _latest_ai_from_stream_update(payload: dict) -> AIMessage | None:
    """Pick the last AIMessage from a LangGraph ``updates`` chunk (any node)."""
    last_ai: AIMessage | None = None
    for _node, v in payload.items():
        if not isinstance(v, dict):
            continue
        msgs = v.get("messages")
        if not msgs:
            continue
        tail = msgs[-1]
        if isinstance(tail, AIMessage):
            last_ai = tail
    return last_ai


def _collect_pending_interrupts(snapshot) -> list:
    """Return any Interrupt objects pending on a graph StateSnapshot.

    Newer LangGraph exposes ``snapshot.interrupts``; older surfaces them per task.
    """
    interrupts = list(getattr(snapshot, "interrupts", None) or [])
    if interrupts:
        return interrupts
    for task in getattr(snapshot, "tasks", None) or []:
        interrupts.extend(getattr(task, "interrupts", None) or [])
    return interrupts


async def message_graph_sse(
    graph,
    human_message: HumanMessage,
    config: dict,
    context: GlobalContext,
    *,
    thread_id: str,
    user_id: str,
    assistant_id: str,
    conversation_title_value: str | None,
    start_time_ns: int,
    request_id: str,
    langgraph_client_headers: dict,
    resume_command: Optional[Command] = None,
):
    """Stream assistant tokens (SSE) then a terminal event with full metadata.

    Terminal event is ``done`` on completion, or ``interrupt`` when the graph pauses
    for human approval (carrying the approve/edit/reject preview payload). Pass
    ``resume_command`` (``Command(resume=...)``) to continue a paused run instead of
    sending a fresh ``human_message``.
    """
    accumulated_chunks: list[str] = []
    last_ai: AIMessage | None = None

    graph_input = (
        resume_command if resume_command is not None else {"messages": [human_message]}
    )

    async for item in graph.astream(
        input=graph_input,
        config=config,
        context=context,
        stream_mode=["custom", "updates"],
        subgraphs=True,
    ):
        if not isinstance(item, tuple) or len(item) != 3:
            continue
        _ns, mode, payload = item
        if mode == "custom" and isinstance(payload, dict):
            if payload.get("type") == "assistant_token":
                accumulated_chunks.append(payload.get("text") or "")
                yield f"data: {json.dumps(payload)}\n\n"
        elif mode == "updates" and isinstance(payload, dict):
            ai = _latest_ai_from_stream_update(payload)
            if ai is not None:
                last_ai = ai

    thread_metadata = {
        "thread_metadata": {
            "user_id": user_id,
            "assistant_id": assistant_id,
            "most_recent_message": datetime.now(timezone.utc).isoformat(),
            "conversation_title": conversation_title_value,
        },
        "graph_id": "Anubis",
    }
    langgraph_client = get_client(headers=langgraph_client_headers)
    await langgraph_client.threads.update(thread_id=thread_id, metadata=thread_metadata)

    # If the graph paused for human approval, surface the preview instead of ``done``.
    # The client resumes via ``POST /message/{assistant_id}/resume`` on this thread_id.
    snapshot = await graph.aget_state(config)
    pending_interrupts = _collect_pending_interrupts(snapshot)
    if pending_interrupts:
        interrupt_event: dict = {
            "type": "interrupt",
            "thread_id": thread_id,
            "request_id": request_id,
            "interrupt": getattr(pending_interrupts[0], "value", None),
            "total_response_time_ms": (time_ns() - start_time_ns) // 1_000_000,
        }
        yield f"data: {json.dumps(interrupt_event, default=str)}\n\n"
        return

    content = last_ai.content if last_ai is not None else "".join(accumulated_chunks)
    done: dict = {
        "type": "done",
        "content": content,
        "thread_id": thread_id,
        "request_id": request_id,
        "total_response_time_ms": (time_ns() - start_time_ns) // 1_000_000,
    }
    if last_ai is not None and getattr(last_ai, "response_metadata", None):
        done["response_metadata"] = last_ai.response_metadata
    yield f"data: {json.dumps(done, default=str)}\n\n"


class MessagePayload(BaseModel):
    message: str = "Hey! Please tell me about yourself and what you can do for me."
    your_name: Optional[str] = None
    your_description: Optional[str] = None
    conversation_title: Optional[str] = None


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


# Create a custom registry for metrics
registry = CollectorRegistry()

# Define metrics
REQUEST_COUNT = Counter(
    "anubis_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"],
    registry=registry,
)

REQUEST_LATENCY = Histogram(
    "anubis_request_duration_seconds",
    "Request duration in seconds",
    ["method", "endpoint"],
    registry=registry,
)

ACTIVE_REQUESTS = Gauge(
    "anubis_active_requests", "Number of active requests", registry=registry
)

MODEL_TOKENS_TOTAL = Counter(
    "anubis_model_tokens_total",
    "Total number of tokens used by model",
    ["model", "type"],  # type: prompt or completion
    registry=registry,
)

MODEL_COST_TOTAL = Counter(
    "anubis_model_cost_total_usd",
    "Total cost in USD for model usage",
    ["model"],
    registry=registry,
)

API_RESPONSE_STATUS = Counter(
    "anubis_api_response_status_total",
    "Response status codes",
    ["status"],
    registry=registry,
)


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
        return self.model_dump(mode="json")


AUTH_CATCH_ALL_PATTERNS = (
    # assistants
    ("POST", re.compile(r"^/assistants$")),
    ("POST", re.compile(r"^/assistants/search$")),
    ("POST", re.compile(r"^/assistants/count$")),
    ("GET", re.compile(r"^/assistants/[^/]+$")),
    ("DELETE", re.compile(r"^/assistants/[^/]+$")),
    ("PATCH", re.compile(r"^/assistants/[^/]+$")),
    ("GET", re.compile(r"^/assistants/[^/]+/graph$")),
    ("GET", re.compile(r"^/assistants/[^/]+/subgraphs$")),
    ("GET", re.compile(r"^/assistants/[^/]+/subgraphs/[^/]+$")),
    ("GET", re.compile(r"^/assistants/[^/]+/schemas$")),
    ("POST", re.compile(r"^/assistants/[^/]+/versions$")),
    ("POST", re.compile(r"^/assistants/[^/]+/latest$")),
    # threads
    ("POST", re.compile(r"^/threads$")),
    ("POST", re.compile(r"^/threads/search$")),
    ("POST", re.compile(r"^/threads/count$")),
    ("POST", re.compile(r"^/threads/prune$")),
    ("GET", re.compile(r"^/threads/[^/]+/state$")),
    ("POST", re.compile(r"^/threads/[^/]+/state$")),
    ("GET", re.compile(r"^/threads/[^/]+/state/[^/]+$")),
    ("POST", re.compile(r"^/threads/[^/]+/state/checkpoint$")),
    ("GET", re.compile(r"^/threads/[^/]+/history$")),
    ("POST", re.compile(r"^/threads/[^/]+/history$")),
    ("POST", re.compile(r"^/threads/[^/]+/copy$")),
    ("GET", re.compile(r"^/threads/[^/]+$")),
    ("DELETE", re.compile(r"^/threads/[^/]+$")),
    ("PATCH", re.compile(r"^/threads/[^/]+$")),
    ("GET", re.compile(r"^/threads/[^/]+/stream$")),
    # thread runs
    ("GET", re.compile(r"^/threads/[^/]+/runs$")),
    ("POST", re.compile(r"^/threads/[^/]+/runs$")),
    ("POST", re.compile(r"^/threads/[^/]+/runs/stream$")),
    ("POST", re.compile(r"^/threads/[^/]+/runs/wait$")),
    ("GET", re.compile(r"^/threads/[^/]+/runs/[^/]+$")),
    ("DELETE", re.compile(r"^/threads/[^/]+/runs/[^/]+$")),
    ("GET", re.compile(r"^/threads/[^/]+/runs/[^/]+/join$")),
    ("GET", re.compile(r"^/threads/[^/]+/runs/[^/]+/stream$")),
    ("POST", re.compile(r"^/threads/[^/]+/runs/[^/]+/cancel$")),
    # runs
    ("POST", re.compile(r"^/runs/cancel$")),
    ("POST", re.compile(r"^/runs/stream$")),
    ("POST", re.compile(r"^/runs/wait$")),
    ("POST", re.compile(r"^/runs$")),
    ("POST", re.compile(r"^/runs/batch$")),
    # crons
    ("POST", re.compile(r"^/threads/[^/]+/runs/crons$")),
    ("POST", re.compile(r"^/runs/crons$")),
    ("POST", re.compile(r"^/runs/crons/search$")),
    ("POST", re.compile(r"^/runs/crons/count$")),
    ("PATCH", re.compile(r"^/runs/crons/[^/]+$")),
    ("DELETE", re.compile(r"^/runs/crons/[^/]+$")),
    # store
    ("PUT", re.compile(r"^/store/items$")),
    ("DELETE", re.compile(r"^/store/items$")),
    ("GET", re.compile(r"^/store/items$")),
    ("POST", re.compile(r"^/store/items/search$")),
    ("POST", re.compile(r"^/store/namespaces$")),
    # a2a
    ("POST", re.compile(r"^/a2a/[^/]+$")),
    # mcp
    ("POST", re.compile(r"^/mcp$")),
    ("GET", re.compile(r"^/mcp$")),
    ("DELETE", re.compile(r"^/mcp$")),
)


def _is_auth_catch_all_target(method: str, path: str) -> bool:
    normalized_path = path.rstrip("/") or "/"
    for expected_method, pattern in AUTH_CATCH_ALL_PATTERNS:
        if method == expected_method and pattern.match(normalized_path):
            return True
    return False


async def get_public_avatars(
    assistant_id: Optional[str] = None, user_id: Optional[str] = None
):
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
                await cur.execute(search_query, (assistant_id,))
            elif user_id:
                await cur.execute(search_query, (user_id,))
            else:
                await cur.execute(search_query)
            data = await cur.fetchall()

            return [assistant_query.to_assistant() for assistant_query in data]


def _assistant_without_metadata_if_public(assistant: dict[str, Any]) -> dict[str, Any]:
    meta = assistant.get("metadata")
    if isinstance(meta, dict):
        pub = meta.get("is_public")
        if pub is True or (isinstance(pub, str) and pub.lower() == "true"):
            return {k: v for k, v in assistant.items() if k != "metadata"}
    return assistant


import debugpy

logger.info(f"DEBUG_PORT: {os.getenv('DEBUG_PORT', 5678)}")
logger.info(f"DEV: {os.getenv('DEV', 'false')}")

if os.getenv("DEV", "false").lower() == "true":
    debugpy.listen(("0.0.0.0", int(os.getenv("DEBUG_PORT", 5678))))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup: Preload the Whisper model pipeline
    global context
    global store_context_manager

    # Initialize context / context
    app.state.context = GlobalContext()
    ensure_huggingface_models_cached(app.state.context)
    # Explicit timeouts instead of httpx's silent 5 s default: a short connect
    # timeout fails fast on an unreachable host, while a generous read timeout
    # tolerates a slow-but-alive upstream.
    app.state.httpx_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0)
    )
    app.state.stripe = stripe
    app.state.stripe.api_key = app.state.context.stripe_secret_key

    async_postgres_store_uri = app.state.context.async_postgres_store_uri
    logger.warning(f"app.state.context.dev: {app.state.context.dev}")
    pool = AsyncConnectionPool(
        conninfo=async_postgres_store_uri,
        min_size=1,
        max_size=5,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        open=False,  # do not open on create
    )
    app.state.pool = pool
    await app.state.pool.open()
    try:
        embed = "huggingface:" + app.state.context.embedding_model
        # IndexConfig key must be ``fields`` (plural). Using ``field`` is ignored and
        # LangGraph falls back to embedding the entire JSON value ("$") — catastrophic
        # when values include multi‑MB reference_image_data URIs on store.aput.
        store = AsyncPostgresStore(
            app.state.pool,
            index=IndexConfig(
                dims=640,
                embed=embed,
                fields=["document.kwargs.page_content"],
            ),
        )

        await store.setup()
        logger.info("Store setup complete")
        app.state.store = store
        # Registry for background media-processing jobs (see src/api/media_jobs.py).
        app.state.media_jobs = {}
        checkpointer = AsyncPostgresSaver(app.state.pool)
        await checkpointer.setup()
        app.state.checkpointer = checkpointer
        # Publish the shared checkpointer so the deep agent (rebuilt each turn inside
        # the ``think`` node) can reuse it and make HITL ``interrupt``s durable.
        runtime_handles.set_deep_agent_checkpointer(checkpointer)
        app.state.graph = message_workflow.compile(
            store=store, checkpointer=checkpointer
        )
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
        if _is_auth_catch_all_target(method=request.method, path=request.url.path):
            try:
                await get_current_user(
                    request=request, api_key=request.headers.get("API-KEY")
                )
            except HTTPException as exc:
                return JSONResponse(
                    status_code=exc.status_code, content={"detail": exc.detail}
                )

        response = await call_next(request)
        latency_ms = (time_ns() - start_time) // 1_000_000

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=str(request.url.path),
            status=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method, endpoint=str(request.url.path)
        ).observe(latency_ms / 1000)
        API_RESPONSE_STATUS.labels(status=response.status_code).inc()

        return response
    except Exception as e:
        latency_ms = (time_ns() - start_time) // 1_000_000
        API_RESPONSE_STATUS.labels(status=500).inc()
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


@app.get("/subscribe")
async def subscribe(current_user: dict = Depends(get_current_user)):
    """
    Create a monthly subscription.
    """

    verified_email = current_user.get("email_verified", None)
    if not verified_email:
        raise HTTPException(
            detail="Please verify your email before subscribing.", status_code=401
        )
    email = current_user.get("email")
    user_id = current_user["app_metadata"]["customer_dict"]["id"]
    redirect_url = f"{app.state.context.stripe_payment_url}?client_reference_id={user_id}&locked_prefilled_email={email}"

    return {"url": redirect_url, "message": "Follow this link to subscribe."}


@app.get("/manage_subscription")
async def manage_subscription(current_user: dict = Depends(get_current_user)):
    return {
        "url": "https://billing.stripe.com/p/login/eVq28s6XA53C5XpdqH1oI00",
        "message": "Follow this link to manage your subscription.",
    }


@app.get("/cancel_subscription")
async def cancel_subscription(current_user: dict = Depends(get_current_user)):
    return {
        "url": "https://billing.stripe.com/p/login/eVq28s6XA53C5XpdqH1oI00",
        "message": "Follow this link to manage and cancel your subscription.",
    }


@app.get("/verify_subscription_status")
async def verify_subscription_status(
    request: Request, current_user: dict = Depends(get_current_user)
):
    status = await check_subscription_status(request=request, current_user=current_user)
    if status["status"] == None:
        return {"subscription_status:Not Subscribed"}
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

    if current_user["identities"][0]["user_id"] == context.anonymous_user_id:
        return JSONResponse(
            content="User must be logged in to create avatars.", status_code=400
        )

    try:
        assistant_id = str(uuid4())
        user_id = current_user["identities"][0]["user_id"]
        metadata = {"user_id": user_id, "is_public": False}

        if user_id == context.admin_user_id:
            metadata["is_public"] = is_public

        token = current_user["API_KEY"]
        headers = {"API-KEY": f"{token}"}
        client = get_client(headers=headers)

        create_avatar_response = await client.assistants.create(
            graph_id="Anubis",
            description=description,
            name=name,
            assistant_id=assistant_id,
            metadata=metadata,
        )

        # store the creator of the assistant
        await client.store.aput((assistant_id,'creator_id'), key="creator_id", value={"value": user_id})

        return JSONResponse(content=create_avatar_response, status_code=200)
    except Exception as e:
        return HTTPException(
            detail="Error creating avatar {name}: {e}", status_code=500
        )


@app.post("/share_avatar")
async def share_avatar(
    assistant_id: str,
    is_public: bool = True,
    current_user: dict = Depends(get_current_user),
):
    context = app.state.context
    user_id = current_user["identities"][0]["user_id"]

    if user_id == context.admin_user_id:
        """verify users are creating avatars of their own likeness in the future"""
        metadata = {"is_public": is_public}

    # Only admins may share avatars;
    # Users will authenticate and share avatars in the near future.
    if user_id == context.admin_user_id:
        try:
            token = current_user["API_KEY"]
            client = get_client(headers={"API-KEY": f"{token}"})
            result = await client.assistants.update(
                assistant_id=assistant_id, metadata=metadata
            )
            return JSONResponse(result, status_code=200)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error during update of sharing avatar: {e}"
            )
    raise HTTPException(
        status_code=401, detail="Users may only share avatars of themselves."
    )


# @app.post("/user_is_creator")
# async def user_is_creator(
#     assistant_id: str,
#     current_user: dict = Depends(get_current_user),
# ):
#     """Used to establish creator in vectorstore due to update in code. Unnecessary, already implemented.

#     Args:
#         assistant_id (str): _description_
#         current_user (dict, optional): _description_. Defaults to Depends(get_current_user).

#     Raises:
#         HTTPException: _description_
#         HTTPException: _description_

#     Returns:
#         _type_: _description_
#     """
#     context = app.state.context
#     user_id = current_user["identities"][0]["user_id"]
#     if user_id == context.admin_user_id:
#         try:
#             token = current_user["API_KEY"]
#             client = get_client(headers={"API-KEY": f"{token}"})
#             namespace = (assistant_id, 'creator_id')
#             await client.store.put_item(namespace, key='creator_id', value={"value": user_id}) 
#             return JSONResponse(content="stored creator_id", status_code=200)
#         except Exception as e:
#             raise HTTPException(
#                 status_code=500, detail=f"Error during update of sharing avatar: {e}"
#             )
        
#     raise HTTPException(
#         status_code=401, detail="Users may only share avatars of themselves."
#     )

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
            detail="Supply assistant_id for the assistant to modify.", status_code=400
        )
    if not new_avatar_name and not new_avatar_description:
        raise HTTPException(
            detail="Either supply the new avatar name or the new avatar description.",
            status_code=400,
        )

    if not current_user:
        raise HTTPException(
            content="User must be logged in to modify avatar avatars.", status_code=401
        )

    token = current_user["API_KEY"]
    client = get_client(headers={"API-KEY": f"{token}"})
    if assistant_id:
        if new_avatar_name and new_avatar_description:
            result = await client.assistants.update(
                graph_id="Anubis",
                assistant_id=assistant_id,
                name=new_avatar_name,
                description=new_avatar_description,
            )
        elif new_avatar_description:
            result = await client.assistants.update(
                graph_id="Anubis",
                assistant_id=assistant_id,
                description=new_avatar_description,
            )
        else:
            result = await client.assistants.update(
                graph_id="Anubis", assistant_id=assistant_id, name=new_avatar_name
            )
        try:
            assert type(result) == dict

            return JSONResponse(content=result, status_code=200)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error updating assistant.")


@app.delete("/delete_avatar")
async def delete_avatar(
    assistant_id: str, request: Request, current_user: dict = Depends(get_current_user)
):
    # TODO: Delete avatar in database
    logger.info("breakpoint")
    token = current_user["API_KEY"]
    user_id = current_user["identities"][0]["user_id"]
    client = get_client(headers={"API-KEY": f"{token}"})

    metadata = {"user_id": user_id}
    metadata.update({"assistant_id": assistant_id})
    # Delete all entries in the store and store vectors for the created avatars
    pool = request.app.state.pool
    SQL_STORE_DELETE_QUERY = """DELETE FROM store WHERE prefix = %s OR prefix LIKE %s or prefix LIKE %s or prefix LIKE %s;"""
    SQL_STORE_VECTOR_DELETE_QUERY = """DELETE FROM store WHERE prefix = %s OR prefix LIKE %s or prefix LIKE %s or prefix LIKE %s;"""
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
        raise HTTPException(
            detail="Error deleting items from store and store vectors during delete avatar.",
            status_code=500,
        )

    # Every store row mentioning the assistant was just removed by raw SQL,
    # bypassing the store client — drop every cached entry for the assistant
    # from the load_consciousness read-through cache.
    invalidate_store_cache_for_assistant(assistant_id)

    try:
        await client.assistants.delete(assistant_id=assistant_id, delete_threads=True)
    except Exception as e:
        raise HTTPException(detail="Error Deleting Assistant", status_code=500)
    return JSONResponse("Deleted Avatar Successfully", status_code=200)


@app.get("/list_public_avatars")
async def list_public_avatars(assistant_id: Optional[str] = None):
    public_avatars_result = await get_public_avatars(assistant_id=assistant_id)
    return [
        {k: v for k, v in assistant.items() if k != "metadata"}
        for assistant in public_avatars_result
    ]


@app.get("/list_user_avatars")
async def list_user_avatars(
    current_user: dict = Depends(get_current_user),
):
    logger.info("breakpoint")
    if not current_user:
        public_avatars_result = await get_public_avatars()
        return [_assistant_without_metadata_if_public(a) for a in public_avatars_result]
    try:
        public_avatars_result = await get_public_avatars(
            user_id=current_user["identities"][0]["user_id"]
        )
        token = current_user["API_KEY"]
        client = get_client(headers={"API-KEY": f"{token}"})
        response = await client.assistants.search(
            metadata={"user_id": current_user["identities"][0]["user_id"]}
        )
        if len(response) > 0:
            avatar_list = response
            public_avatars_result.extend(avatar_list)  # public and private avatars
        sanitized = [
            _assistant_without_metadata_if_public(a) for a in public_avatars_result
        ]
        return JSONResponse(sanitized, status_code=200)
    except Exception as e:
        error = f"Error in listing avatars: {e}"
        raise HTTPException(detail=error, status_code=500)


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
        return HTTPException(
            status_code=400,
            detail="Unauthenticated users must log in to use the select avatars via name feature. Please log in or use an assistant_id for selection.",
        )

    assistant_config = {"configurable": {"assistant_id": assistant_id}}

    public_avatar_result = await get_public_avatars(assistant_id=assistant_id)

    # if not current_user['identities'][0]['user_id'] is request.app.state.context['anonymous_user_id']: # anonymous user case
    if not current_user:
        if len(public_avatar_result) > 0:
            assistant_config["configurable"].update(
                {
                    "assistant_ctx": {
                        "name": public_avatar_result[0].get("name", None),
                        "description": public_avatar_result[0].get("description", None),
                    }
                }
            )

        public_avatar_result = await update_assistant_config(
            assistant_config=assistant_config, request=request
        )
        return assistant_config
    else:
        token = current_user["API_KEY"]
        client = get_client(headers={"API-KEY": token})
        user_id = current_user["identities"][0]["user_id"]
        if assistant_id:
            try:
                if len(public_avatar_result) == 0:  # the avatar was not public
                    result = await client.assistants.get(
                        assistant_id=assistant_id
                    )  # attempt to get user-specific avatar with api key
                    if not result:
                        raise HTTPException(
                            detail="Assistant not found: {assistant_id}",
                            status_code=500,
                        )
                        # assistant = {"name": None, "description": None}
                    else:
                        assistant = result
                    logger.info(f"result:{result}")
                    assistant_config = {
                        "configurable": {
                            "assistant_id": assistant_id,
                            "assistant_ctx": {
                                "name": assistant.get("name", ""),
                                "description": assistant.get("description", ""),
                                "metadata": assistant.get("metadata", {}),
                            },
                        }
                    }
                else:
                    assistant_config["configurable"].update(
                        {
                            "assistant_ctx": {
                                "name": public_avatar_result[0].get("name", None),
                                "description": public_avatar_result[0].get(
                                    "description", None
                                ),
                                "metadata": public_avatar_result[0].get("metadata", {}),
                            }
                        }
                    )
                provider_encoded_user_id = quote(current_user["user_id"], safe="")

                hashed_api_key = current_user["app_metadata"]["api_key"]
                _ = await update_assistant_config(
                    hashed_api_key=hashed_api_key,
                    provider_encoded_user_id=provider_encoded_user_id,
                    assistant_config=assistant_config,
                    request=request,
                )
                return assistant_config
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error using assistant_id for logged in user {e}",
                )
        elif assistant_name:
            try:
                result = await client.assistants.search(name=assistant_name)
                try:
                    if len(result) == 0:
                        raise HTTPException(
                            detail="Assistant not found.", status_code=400
                        )
                    assistant = result[0]
                    is_public = assistant.get("metadata", {}).get("is_public", False)
                    if not is_public and (
                        current_user["identities"][0]["user_id"]
                        != assistant.get("metadata", {}).get("user_id", None)
                    ):
                        raise HTTPException(
                            detail="Non-public avatar id.", status_code=401
                        )
                    else:
                        assistant_config = {
                            "configurable": {
                                "assistant_ctx": {
                                    "name": assistant.get("name", None),
                                    "description": assistant.get("description", None),
                                    "metadata": assistant.get("metadata", {}),
                                },
                                "assistant_id": assistant.get("assistant_id", None),
                            }
                        }
                    hashed_api_key = current_user["app_metadata"]["api_key"]
                    provider_encoded_user_id = quote(current_user["user_id"], safe="")
                    result = await update_assistant_config(
                        hashed_api_key=hashed_api_key,
                        provider_encoded_user_id=provider_encoded_user_id,
                        assistant_config=assistant_config,
                        request=request,
                    )

                    return JSONResponse(content=assistant_config, status_code=200)
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error during avatar selection via assistant_name: {e}",
                    )
            except Exception as e:
                error_str = "{error}".format(error=e)
                return HTTPException(detail=error_str, status_code=500)
        else:
            return HTTPException(
                detail="Error: either assistant_id or assistant_name is required.",
                status_code=400,
            )


async def process_files_for_message(
    files: OptionalUploadFiles = None,
    message: str = "",
) -> tuple:
    """Process uploaded files and return content for inclusion in messages.

    Returns:
        tuple: (text_content, multimodal_content, image_filenames)
        - text_content: str - concatenated text from text files (non-image)
        - multimodal_content: list or None - multimodal content (text + image blocks)
        - image_filenames: filenames for each image block, in order
    """
    if not files:
        return "", None, []

    text_contents = []
    multimodal_parts = []
    image_filenames: List[str] = []
    has_images = False

    for file in files:
        try:
            content = await file.read()
            filename = file.filename or "unknown_file"
            content_type = file.content_type or ""

            if content_type.startswith("image/"):
                base64_image = base64.b64encode(content).decode("utf-8")
                image_url = f"data:{content_type};base64,{base64_image}"

                multimodal_parts.append(
                    {"type": "image_url", "image_url": {"url": image_url}}
                )
                image_filenames.append(filename)
                has_images = True
                text_contents.append(f"[Image: {filename}]")

            elif content_type.startswith("text/") or content_type == "application/pdf":
                # Handle text files and PDFs
                if content_type == "application/pdf":
                    try:
                        from langchain_community.document_loaders import (
                            PyPDFLoader,
                        )

                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".pdf"
                        ) as temp_pdf:
                            temp_pdf.write(content)
                            temp_pdf.flush()
                            pdf_loader = PyPDFLoader(temp_pdf.name)
                            pdf_docs = pdf_loader.load()

                        pdf_text = "\n\n".join(
                            [
                                doc.page_content
                                for doc in pdf_docs
                                if hasattr(doc, "page_content")
                            ]
                        )
                        if pdf_text:
                            text_contents.append(f"[PDF File: {filename}]\n{pdf_text}")
                        else:
                            text_contents.append(
                                f"[PDF File: {filename} - no extractable text]"
                            )
                    except Exception as pdf_error:
                        logger.error(
                            f"Failed to extract PDF text from {filename}: {pdf_error}"
                        )
                        text_contents.append(f"[PDF File: {filename}]")
                    finally:
                        try:
                            os.unlink(temp_pdf.name)
                        except Exception:
                            pass
                else:
                    # Text files
                    try:
                        text_content = content.decode("utf-8")
                        text_contents.append(f"[File: {filename}]\n{text_content}")
                    except UnicodeDecodeError:
                        text_contents.append(f"[Binary Text File: {filename}]")

            elif content_type.startswith("audio/"):
                # Audio files - describe that audio was uploaded
                text_contents.append(f"[Audio File: {filename} - {content_type}]")

            else:
                # Other file types
                text_contents.append(f"[File: {filename} - {content_type}]")

        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            text_contents.append(f"[Error processing file: {file.filename}]")

    # Combine text content (file-derived only; caller message is merged below for images)
    combined_text = "\n\n".join(text_contents) if text_contents else ""

    # Return multimodal content if images are present
    if has_images:
        text_segments = []
        if (message or "").strip():
            text_segments.append(message.strip())
        if combined_text:
            text_segments.append(combined_text)
        full_text = "\n\n".join(text_segments)
        multimodal_content = [{"type": "text", "text": full_text}] + multimodal_parts
        return combined_text, multimodal_content, image_filenames

    return combined_text, None, []


@app.post("/message")
async def message_selected_avatar(
    request: Request,
    message: str = Form(""),
    your_name: Optional[str] = Form(None),
    your_description: Optional[str] = Form(None),
    conversation_title: Optional[str] = Form(None),
    files: OptionalUploadFiles = None,
    thread_id: Optional[str] = Form(None),
    stream: bool = Form(True),
    feedback: bool = Form(False),
    like: bool = Form(False),
    dislike: bool = Form(False),
    user_timezone: Optional[str] = Form(None),
    include_metrics: bool = Form(True),
    current_user: dict = Depends(get_current_user),
):
    # NOTE: ``feedback`` / ``like`` / ``dislike`` are inert placeholders. The
    # data-collection / preference-learning pipeline is intentionally deferred
    # while the upload + evaluation pipeline ships first; the parameters exist
    # now so the frontend can wire its UI without a breaking API change later.
    langgraph_client_headers = {"API-KEY": request.headers.get("api-key")}
    # allow for select avatar in query and anonymous user for a dedicated endpoint
    start_time = time_ns()
    config = current_user.get("app_metadata", {}).get("assistant_config", {})
    if not config:
        raise HTTPException(
            detail="Error retrieving assistant information.", status_code=400
        )

    user_name = your_name
    user_description = your_description
    user_id = current_user["identities"][0]["user_id"]
    config_update = {
        "configurable": {
            "user_ctx": {"name": user_name, "description": user_description},
            "user_id": user_id,
        }
    }
    assistant_id = config["configurable"].get("assistant_id")

    # Handle thread_id
    if not thread_id:
        thread_id = str(uuid4())
        thread_metadata = {
            "thread_metadata": {"user_id": user_id, "assistant_id": assistant_id},
            "graph_id": "Anubis",
        }
        # create thread_id
        try:
            langgraph_client = get_client(headers=langgraph_client_headers)
            thread_create_response = await langgraph_client.threads.create(
                thread_id=thread_id, metadata=thread_metadata
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail="Error creating new conversation thread."
            )

    # update with user information
    config_update["configurable"]["thread_id"] = thread_id
    config["configurable"].update(config_update["configurable"])
    # client-supplied IANA timezone (e.g. "America/New_York") used to localize system_time
    config["configurable"]["user_timezone"] = user_timezone
    config["configurable"]["include_metrics"] = include_metrics

    # store = app.state.store
    graph = app.state.graph

    # Process any uploaded files
    (
        file_text_content,
        multimodal_content,
        image_filenames,
    ) = await process_files_for_message(files, message=message)

    # Create the human message content
    if multimodal_content:
        human_message = HumanMessage(
            id=str(uuid4()),
            content=multimodal_content,
            additional_kwargs={"image_filenames": image_filenames},
        )
    else:
        # Use text-only content
        if file_text_content:
            if (message or "").strip():
                human_message_content = message.strip() + "\n\n" + file_text_content
            else:
                human_message_content = file_text_content
        else:
            human_message_content = message
        human_message = HumanMessage(id=str(uuid4()), content=human_message_content)

    if stream:
        return StreamingResponse(
            message_graph_sse(
                graph,
                human_message,
                config,
                app.state.context,
                thread_id=thread_id,
                user_id=user_id,
                assistant_id=assistant_id,
                conversation_title_value=conversation_title,
                start_time_ns=start_time,
                request_id=request.state.request_id,
                langgraph_client_headers=langgraph_client_headers,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    result = await graph.ainvoke(
        input={"messages": [human_message]},
        config=config,
        context=app.state.context,
    )

    # Update most_recent_message
    langgraph_client = get_client(headers=langgraph_client_headers)
    thread_metadata = {
        "thread_metadata": {
            "user_id": user_id,
            "assistant_id": assistant_id,
            "most_recent_message": datetime.now(timezone.utc).isoformat(),
            "conversation_title": conversation_title,
        },
        "graph_id": "Anubis",
    }
    await langgraph_client.threads.update(thread_id=thread_id, metadata=thread_metadata)

    response_data = {}
    response_data["content"] = result["messages"][-1].content
    response_metadata = result["messages"][-1].response_metadata
    if response_metadata:
        response_data["response_metadata"] = response_metadata

    response_data["total_response_time_ms"] = (time_ns() - start_time) // 1000000
    logger.warning(f"RESPONSE_DATA: {response_data}")
    response_data["thread_id"] = thread_id
    response_data["request_id"] = request.state.request_id
    return JSONResponse(response_data, status_code=200)


@app.post("/message/{assistant_id}")
async def message_avatar(
    request: Request,
    assistant_id: str,
    message: str = Form(""),
    your_name: Optional[str] = Form(None),
    your_description: Optional[str] = Form(None),
    conversation_title: Optional[str] = Form(None),
    files: OptionalUploadFiles = None,
    thread_id: Optional[str] = Form(None),
    stream: bool = Form(True),
    feedback: bool = Form(False),
    like: bool = Form(False),
    dislike: bool = Form(False),
    user_timezone: Optional[str] = Form(None),
    include_metrics: bool = Form(True),
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    # NOTE: ``feedback`` / ``like`` / ``dislike`` are inert placeholders. The
    # data-collection / preference-learning pipeline is intentionally deferred
    # while the upload + evaluation pipeline ships first; the parameters exist
    # now so the frontend can wire its UI without a breaking API change later.

    # allow for select avatar in query and anonymous user for a dedicated endpoint

    logger.warning(f"stream:{stream}")
    start_time = time_ns()
    config = current_user.get("app_metadata", {}).get("assistant_config", {})
    if not config:
        raise HTTPException(
            detail="Error retrieving assistant information.", status_code=400
        )

    user_name = your_name
    user_description = your_description
    user_id = current_user["identities"][0]["user_id"]
    if request.headers.get("api-key", "") != "":
        langgraph_client_headers = {"API-KEY": request.headers.get("api-key")}
        try:
            langgraph_client = get_client(headers=langgraph_client_headers)
            assistant = await langgraph_client.assistants.get(assistant_id=assistant_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error selecting avatar.")

        config_update = {
            "configurable": {
                "user_ctx": {"name": user_name, "description": user_description},
                "user_id": user_id,
                "assistant_id": assistant_id,
                "assistant_ctx": {
                    "name": assistant.get("name", None),
                    "description": assistant.get("description", None),
                    "metadata": assistant.get("metadata", {}),
                },
            }
        }

    else:
        # anonymous user_id and assistant_id is handled in the current_user dependency function
        langgraph_client_headers = {"API-KEY": app.state.context.anonymous_api_key}
        config_update = {
            "configurable": {
                "user_ctx": {"name": user_name, "description": user_description},
            }
        }

    # Handle thread_id
    if not thread_id:
        thread_id = str(uuid4())
        thread_metadata = {
            "thread_metadata": {"user_id": user_id, "assistant_id": assistant_id},
            "graph_id": "Anubis",
        }
        # create thread_id
        try:
            langgraph_client = get_client(headers=langgraph_client_headers)
            thread_create_response = await langgraph_client.threads.create(
                thread_id=thread_id, metadata=thread_metadata
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail="Error creating new conversation thread."
            )

    # update with user information
    config_update["configurable"]["thread_id"] = thread_id
    config["configurable"].update(config_update["configurable"])
    # client-supplied IANA timezone (e.g. "America/New_York") used to localize system_time
    config["configurable"]["user_timezone"] = user_timezone
    config["configurable"]["include_metrics"] = include_metrics

    # store = app.state.store
    graph = app.state.graph

    # Process any uploaded files
    (
        file_text_content,
        multimodal_content,
        image_filenames,
    ) = await process_files_for_message(files, message=message)

    # Create the human message content
    if multimodal_content:
        human_message = HumanMessage(
            id=str(uuid4()),
            content=multimodal_content,
            additional_kwargs={"image_filenames": image_filenames},
        )
    else:
        # Use text-only content
        if file_text_content:
            if (message or "").strip():
                human_message_content = message.strip() + "\n\n" + file_text_content
            else:
                human_message_content = file_text_content
        else:
            human_message_content = message
        human_message = HumanMessage(id=str(uuid4()), content=human_message_content)

    conversation_title_data = (
        conversation_title if conversation_title != "" else thread_id
    )

    if stream:
        return StreamingResponse(
            message_graph_sse(
                graph,
                human_message,
                config,
                app.state.context,
                thread_id=thread_id,
                user_id=user_id,
                assistant_id=assistant_id,
                conversation_title_value=conversation_title_data,
                start_time_ns=start_time,
                request_id=request.state.request_id,
                langgraph_client_headers=langgraph_client_headers,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    result = await graph.ainvoke(
        input={"messages": [human_message]},
        config=config,
        context=app.state.context,
    )

    # Update most_recent_message
    langgraph_client = get_client(headers=langgraph_client_headers)
    thread_metadata = {
        "thread_metadata": {
            "user_id": user_id,
            "assistant_id": assistant_id,
            "most_recent_message": datetime.now(timezone.utc).isoformat(),
            "conversation_title": conversation_title_data,
        },
        "graph_id": "Anubis",
    }
    await langgraph_client.threads.update(thread_id=thread_id, metadata=thread_metadata)

    response_data = {}
    response_data["content"] = result["messages"][-1].content
    response_metadata = result["messages"][-1].response_metadata
    if response_metadata:
        response_data["response_metadata"] = response_metadata

    response_data["total_response_time_ms"] = (time_ns() - start_time) // 1000000
    response_data["thread_id"] = thread_id
    response_data["request_id"] = request.state.request_id
    return JSONResponse(response_data, status_code=200)


@app.post("/message/{assistant_id}/resume")
async def resume_avatar_message(
    request: Request,
    assistant_id: str,
    thread_id: str = Form(...),
    decision: str = Form("apply"),
    items: Optional[str] = Form(None),
    your_name: Optional[str] = Form(None),
    your_description: Optional[str] = Form(None),
    user_timezone: Optional[str] = Form(None),
    include_metrics: bool = Form(True),
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    """Resume a run paused for human approval (edit/delete identity fact).

    ``decision`` is ``apply`` | ``cancel``. ``items`` (JSON list) carries the owner's
    per-document decisions — one entry per matched document with ``index`` and an ``action``
    ∈ ``skip`` | ``accept`` | ``edit`` | ``remove`` (plus ``corrected_text`` /
    ``correction_context`` when the action is ``edit``). Any matched document the owner did
    not act on defaults to ``skip`` in the tool, so a missing/empty list changes nothing.
    Older clients' ``approve`` / ``reject`` are accepted as aliases for ``apply`` / ``cancel``.
    Streams the continuation as SSE (same ``assistant_token`` → ``done``/``interrupt`` shape as
    ``/message/{assistant_id}``).
    """
    start_time = time_ns()

    # Map legacy spellings so an older panel still resolves to the current vocabulary.
    decision_aliases = {"approve": "apply", "reject": "cancel"}
    raw_decision = (decision or "apply").strip().lower()
    decision_value = decision_aliases.get(raw_decision, raw_decision)
    if decision_value not in ("apply", "cancel"):
        raise HTTPException(status_code=400, detail="decision must be apply or cancel.")

    config = current_user.get("app_metadata", {}).get("assistant_config", {})
    if not config:
        raise HTTPException(
            detail="Error retrieving assistant information.", status_code=400
        )

    user_id = current_user["identities"][0]["user_id"]
    if request.headers.get("api-key", "") != "":
        langgraph_client_headers = {"API-KEY": request.headers.get("api-key")}
        try:
            langgraph_client = get_client(headers=langgraph_client_headers)
            assistant = await langgraph_client.assistants.get(assistant_id=assistant_id)
        except Exception:
            raise HTTPException(status_code=500, detail="Error selecting avatar.")
        config_update = {
            "configurable": {
                "user_ctx": {"name": your_name, "description": your_description},
                "user_id": user_id,
                "assistant_id": assistant_id,
                "assistant_ctx": {
                    "name": assistant.get("name", None),
                    "description": assistant.get("description", None),
                    "metadata": assistant.get("metadata", {}),
                },
            }
        }
    else:
        langgraph_client_headers = {"API-KEY": app.state.context.anonymous_api_key}
        config_update = {
            "configurable": {
                "user_ctx": {"name": your_name, "description": your_description},
            }
        }

    config_update["configurable"]["thread_id"] = thread_id
    config["configurable"].update(config_update["configurable"])
    config["configurable"]["user_timezone"] = user_timezone
    config["configurable"]["include_metrics"] = include_metrics

    graph = app.state.graph

    # The resume value is the decision dict the paused tool's ``interrupt`` expects; it flows
    # outer-``interrupt`` → ``think`` → the deep-agent tool unchanged. ``cancel`` abandons the
    # whole correction; ``apply`` carries the owner's per-item decisions. The tool defaults any
    # un-acted item to ``skip``, so an empty/missing list is a safe no-op.
    resume_payload: dict = {"type": decision_value}
    if items:
        try:
            parsed_items = json.loads(items)
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="items must be a JSON list.")
        if not isinstance(parsed_items, list):
            raise HTTPException(status_code=400, detail="items must be a JSON list.")
        resume_payload["items"] = parsed_items

    return StreamingResponse(
        message_graph_sse(
            graph,
            None,
            config,
            app.state.context,
            thread_id=thread_id,
            user_id=user_id,
            assistant_id=assistant_id,
            conversation_title_value=None,
            start_time_ns=start_time,
            request_id=request.state.request_id,
            langgraph_client_headers=langgraph_client_headers,
            resume_command=Command(resume=resume_payload),
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/conversations")
async def get_all_conversations(
    request: Request,
    assistant_id: str,
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    """Return all threads for this user + assistant, newest-first."""
    user_id = current_user["identities"][0]["user_id"]
    if request.headers.get("api-key", "") != "":
        langgraph_client_headers = {"API-KEY": request.headers.get("api-key")}
    else:
        langgraph_client_headers = {
            "API-KEY": request.app.state.context.anonymous_api_key
        }
    try:
        langgraph_client = get_client(headers=langgraph_client_headers)
        threads = await langgraph_client.threads.search(
            metadata={
                "thread_metadata": {"user_id": user_id, "assistant_id": assistant_id}
            },
            sort_by="updated_at",
            sort_order="desc",
        )
        return JSONResponse(threads)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error loading threads: {exc}")


@app.get("/conversations/{thread_id}/messages")
async def get_thread_messages(
    request: Request,
    thread_id: str,
    assistant_id: str,
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    """Return the message history for a single thread."""
    if request.headers.get("api-key", "") != "":
        langgraph_client_headers = {"API-KEY": request.headers.get("api-key")}
    else:
        langgraph_client_headers = {
            "API-KEY": request.app.state.context.anonymous_api_key
        }
    try:
        langgraph_client = get_client(headers=langgraph_client_headers)
        state = await langgraph_client.threads.get_state(thread_id=thread_id)
        messages = state.get("values", {}).get("messages", []) if state else []
        return JSONResponse({"messages": messages})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error loading messages: {exc}")


ALLOWED_IMAGE_MIMES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})


def normalize_declared_image_mime(ct: str) -> str:
    ct = (ct or "").split(";")[0].strip().lower()
    if ct == "image/jpg":
        return "image/jpeg"
    return ct


def _sniff_media_category_from_bytes(chunk: bytes) -> Optional[str]:
    """Infer image/audio/video/pdf from magic bytes when Content-Type is unhelpful."""
    if not chunk:
        return None
    if chunk[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if chunk[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if chunk[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if chunk[:4] == b"RIFF" and len(chunk) >= 12 and chunk[8:12] == b"WEBP":
        return "image/webp"
    if chunk[:4] == b"%PDF":
        return "application/pdf"
    if chunk[:3] == b"ID3" or chunk[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio/mpeg"
    if chunk[:4] == b"OggS":
        return "audio/ogg"
    if chunk[:4] == b"RIFF" and len(chunk) >= 12 and chunk[8:12] == b"WAVE":
        return "audio/wav"
    if chunk[:4] == b"\x1a\x45\xdf\xa3":
        return "video/webm"
    if len(chunk) >= 12 and chunk[4:8] == b"ftyp":
        return "video/mp4"
    return None


def _gif_image_descriptor_count(data: bytes) -> int:
    if len(data) < 13:
        return 0
    if data[:6] not in (b"GIF87a", b"GIF89a"):
        return 0
    packed = data[10]
    i = 13
    if packed & 0x80:
        i += 3 * (1 << ((packed & 0x07) + 1))
    count = 0
    n = len(data)
    while i < n:
        tag = data[i]
        if tag == 0x3B:
            break
        if tag == 0x21:
            i += 1
            if i >= n:
                break
            i += 1
            while i < n:
                bsize = data[i]
                i += 1
                if bsize == 0:
                    break
                i += bsize
        elif tag == 0x2C:
            count += 1
            i += 1
            if i + 8 > n:
                break
            i += 8
            local = data[i - 1]
            if local & 0x80:
                i += 3 * (1 << ((local & 0x07) + 1))
            if i >= n:
                break
            i += 1
            while i < n:
                bsize = data[i]
                i += 1
                if bsize == 0:
                    break
                i += bsize
        else:
            i += 1
    return count


def _gif_is_animated(data: bytes) -> bool:
    return _gif_image_descriptor_count(data) > 1


def _webp_is_animated(data: bytes) -> bool:
    cap = min(len(data), 65536)
    return b"ANMF" in data[:cap]


def validate_upload_image_bytes(declared_mime: str, body: bytes) -> str:
    """Return normalized image MIME; raises HTTPException if not an allowed still image."""
    mime = normalize_declared_image_mime(declared_mime)
    sniff = _sniff_media_category_from_bytes(body[:512])
    if mime in ("", "application/octet-stream"):
        if sniff not in ALLOWED_IMAGE_MIMES:
            raise HTTPException(
                status_code=400,
                detail="Could not determine an allowed image type from the file or URL.",
            )
        mime = sniff
    if mime not in ALLOWED_IMAGE_MIMES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image type not allowed (got {mime!r}); "
                "allowed: image/jpeg, image/png, image/gif (non-animated), image/webp."
            ),
        )
    if sniff and normalize_declared_image_mime(sniff) != mime:
        raise HTTPException(
            status_code=400,
            detail="Declared Content-Type does not match image file contents.",
        )
    if mime == "image/gif" and _gif_is_animated(body):
        raise HTTPException(
            status_code=400, detail="Animated GIF is not allowed; use a still frame."
        )
    if mime == "image/webp" and _webp_is_animated(body):
        raise HTTPException(
            status_code=400, detail="Animated WebP is not allowed; use a still image."
        )
    return mime


async def probe_remote_url_content_type(url: str) -> str:
    """Best-effort Content-Type for a remote URL (HEAD, then ranged GET + sniff)."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
        head_ct = ""
        try:
            head = await client.head(url)
            head_ct = (
                (head.headers.get("content-type") or "").split(";")[0].strip().lower()
            )
        except Exception:
            pass
        if head_ct and head_ct != "application/octet-stream":
            return head_ct
        resp = await client.get(url, headers={"Range": "bytes=0-511"})
        resp.raise_for_status()
        body_ct = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        if body_ct and body_ct != "application/octet-stream":
            return body_ct
        sniffed = _sniff_media_category_from_bytes(resp.content[:512])
        return sniffed or body_ct or "application/octet-stream"


async def require_url_content_type_prefix(url: str, prefix: str, label: str) -> None:
    ct = await probe_remote_url_content_type(url)
    if not ct.startswith(prefix):
        raise HTTPException(
            status_code=400,
            detail=f"{label} URL must resolve to {prefix}* (got {ct!r}).",
        )


def _is_youtube_url(url: str) -> bool:
    """Recognize URLs whose Content-Type is HTML but whose payload is video/audio."""
    from urllib.parse import urlparse

    from src.anubis.utils.classes.URLDocumentLoaderClass import _YOUTUBE_HOSTS

    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    return host in _YOUTUBE_HOSTS


MAX_REMOTE_URL_DOWNLOAD_BYTES = 25 * 1024 * 1024


async def fetch_remote_url_bytes(
    url: str,
    max_bytes: int = MAX_REMOTE_URL_DOWNLOAD_BYTES,
) -> tuple[bytes, str]:
    """Download a URL and return (body, Content-Type without parameters)."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=120.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        body = r.content
        if len(body) > max_bytes:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Remote resource exceeds maximum download size ({max_bytes} bytes)."
                ),
            )
        header_ct = (r.headers.get("content-type") or "").split(";")[0].strip().lower()
        return body, header_ct or "application/octet-stream"


def make_data_uri(mime: str, body: bytes) -> str:
    """RFC 2397 data URI: ``data:<mime>;base64,<payload>``."""
    return f"data:{mime};base64,{base64.b64encode(body).decode('ascii')}"


# ---------------------------------------------------------------------------
# CSV ingest preprocessing
#
# Tabular uploads are converted at the API edge into a JSON ``statements``
# document so the rest of the pipeline only ever has to handle media types it
# already knows about. Each CSV row becomes one statement with the shape
# requested by the avatar-identity ingest contract:
#
#     {
#         "messages": [{"role": "assistant", "content": "<row text>"}],
#         "metadata": {"target": "<name>", "source": "<filename>"}
#     }
#
# The text column and target name are picked once per upload by
# ``CSVUserTextColumnIdentificationClass`` (model-driven, schema-constrained).
# Detection happens HERE so the process_media graph never sees raw CSV bytes.
# ---------------------------------------------------------------------------


_CSV_MIME_HINTS = frozenset(
    {
        "text/csv",
        "application/csv",
        "application/vnd.ms-excel",
        "application/x-csv",
    }
)
_CSV_NAME_HINT_RE = re.compile(
    r"\b(user[_-]?name|user|name|author|screen[_-]?name|"
    r"handle|username|creator|speaker|full[_-]?name)\b",
    re.IGNORECASE,
)
_CSV_BOOLEAN_VALUES = frozenset({"true", "false", "yes", "no", "0", "1"})
_CSV_PREVIEW_ROW_LIMIT = 8
_CSV_STATS_SAMPLE_VALUES = 3


def _is_csv_upload(filename: str, content_type: str) -> bool:
    """True when the upload looks like a CSV by MIME type or filename."""
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct in _CSV_MIME_HINTS:
        return True
    name = (filename or "").strip().lower()
    return name.endswith(".csv") or name.endswith(".tsv")


def _decode_csv_bytes(raw: bytes) -> str:
    """Decode CSV bytes preferring UTF-8, falling back to latin-1 then replace.

    BOM is stripped because ``csv.reader`` treats it as part of the first
    header otherwise.
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _parse_csv_to_rows(raw: bytes, filename: str) -> tuple[list[str], list[dict]]:
    """Return ``(headers, rows)`` from CSV bytes.

    Uses ``csv.Sniffer`` for delimiter detection (handles ``,`` and ``\\t``
    files), falls back to comma when sniffing fails on tiny / malformed
    samples. ``rows`` is a list of OrderedDicts keyed by header.
    """
    import csv as _csv
    from io import StringIO

    text = _decode_csv_bytes(raw)
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail=f"CSV upload {filename!r} is empty.",
        )

    sample = text[:8192]
    dialect: Any
    try:
        dialect = _csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except _csv.Error:
        dialect = _csv.excel
        if filename.lower().endswith(".tsv"):
            dialect = _csv.excel_tab

    reader = _csv.DictReader(StringIO(text), dialect=dialect)
    headers = list(reader.fieldnames or [])
    if not headers:
        raise HTTPException(
            status_code=400,
            detail=f"CSV upload {filename!r} has no header row.",
        )
    rows: list[dict] = []
    for row in reader:
        rows.append(
            {h: (row.get(h) if row.get(h) is not None else "") for h in headers}
        )
    if not rows:
        raise HTTPException(
            status_code=400,
            detail=f"CSV upload {filename!r} has no data rows.",
        )
    return headers, rows


def _looks_numeric(value: str) -> bool:
    v = value.strip()
    if not v:
        return False
    try:
        float(v.replace(",", ""))
        return True
    except ValueError:
        return False


def _build_csv_column_stats(
    headers: list[str], rows: list[dict]
) -> dict[str, dict[str, Any]]:
    """Per-column summary used as model context for column identification."""
    stats: dict[str, dict[str, Any]] = {}
    total = len(rows)
    for header in headers:
        values = [str(r.get(header) or "").strip() for r in rows]
        non_empty = [v for v in values if v]
        non_empty_count = len(non_empty)
        if non_empty_count == 0:
            stats[header] = {
                "non_empty_count": 0,
                "non_empty_ratio": 0.0,
                "avg_len": 0.0,
                "max_len": 0,
                "distinct_count": 0,
                "distinct_ratio": 0.0,
                "looks_numeric": False,
                "looks_boolean": False,
                "name_hint": bool(_CSV_NAME_HINT_RE.search(header or "")),
                "sample_values": [],
            }
            continue
        avg_len = sum(len(v) for v in non_empty) / non_empty_count
        max_len = max(len(v) for v in non_empty)
        distinct = sorted(set(non_empty), key=non_empty.index)
        distinct_count = len(distinct)
        looks_numeric = (
            sum(1 for v in non_empty if _looks_numeric(v)) / non_empty_count
        ) >= 0.9
        looks_boolean = (
            sum(1 for v in non_empty if v.lower() in _CSV_BOOLEAN_VALUES)
            / non_empty_count
        ) >= 0.9
        stats[header] = {
            "non_empty_count": non_empty_count,
            "non_empty_ratio": non_empty_count / total if total else 0.0,
            "avg_len": round(avg_len, 2),
            "max_len": max_len,
            "distinct_count": distinct_count,
            "distinct_ratio": distinct_count / non_empty_count,
            "looks_numeric": looks_numeric,
            "looks_boolean": looks_boolean,
            "name_hint": bool(_CSV_NAME_HINT_RE.search(header or "")),
            "sample_values": distinct[:_CSV_STATS_SAMPLE_VALUES],
        }
    return stats


def _csv_dominant_value(values: list[str]) -> tuple[Optional[str], float]:
    """Return the dominant non-empty value and its share of non-empty rows."""
    cleaned = [v for v in (s.strip() for s in values) if v]
    if not cleaned:
        return None, 0.0
    counts: dict[str, int] = {}
    for v in cleaned:
        counts[v] = counts.get(v, 0) + 1
    top_value, top_count = max(counts.items(), key=lambda kv: kv[1])
    return top_value, top_count / len(cleaned)


def _filename_target_hint(filename: str) -> str:
    """Title-case a filename stem when it looks like a person's name namespace_safe_formatted_filename."""
    stem = (filename or "").rsplit(".", 1)[0]
    stem = re.sub(r"[_\-]+", " ", stem).strip()
    if not stem:
        return ""
    parts = [p for p in stem.split() if p and p.isalpha()]
    if 1 <= len(parts) <= 4:
        return " ".join(p.capitalize() for p in parts)
    return ""


async def _csv_to_statements_payload(
    *, raw: bytes, source_filename: str
) -> dict[str, Any]:
    """Convert CSV bytes into the avatar-identity statements JSON document.

    Output shape passed downstream to the JSON media handler:

        {
            "statements": [
                {
                    "messages": [{"role": "assistant", "content": "<text>"}],
                    "metadata": {"target": "<name>", "source": "<filename>"}
                },
                ...
            ],
            "metadata": {
                "target": "<dominant target name or null>",
                "source": "<source_filename>",
                "csv_text_column": "<column>",
                "csv_target_column": "<column or null>",
                "csv_row_count": <int>,
                "csv_classifier_reasoning": "<llm reasoning>"
            }
        }
    """
    headers, rows = _parse_csv_to_rows(raw, source_filename)
    return await _rows_to_statements_payload(
        headers=headers, rows=rows, source_filename=source_filename
    )


async def _rows_to_statements_payload(
    *, headers: list[str], rows: list[dict], source_filename: str
) -> dict[str, Any]:
    """Convert parsed tabular ``(headers, rows)`` into the statements document.

    The shared core behind every tabular upload format (CSV/TSV bytes via
    ``_parse_csv_to_rows``, tabular JSON via ``_normalize_tabular_json_to_rows``):
    build per-column stats, have ``CSVUserTextColumnIdentificationClass`` pick
    the free-text and target-name columns once per upload, then emit one
    statement per non-empty row. Output shape documented on
    ``_csv_to_statements_payload``.
    """
    from src.anubis.utils.classes.CSVUserTextColumnIdentificationClass import (
        CSVUserTextColumnIdentificationClass,
    )

    column_stats = _build_csv_column_stats(headers, rows)

    sample_rows = rows[:_CSV_PREVIEW_ROW_LIMIT]
    classifier = CSVUserTextColumnIdentificationClass()
    classifier_response = await classifier.classify(
        filename=source_filename,
        headers=headers,
        sample_rows=sample_rows,
        column_stats=column_stats,
    )

    text_column: str = classifier_response.get("text_column") or ""
    if text_column not in headers:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Could not identify a text column in {source_filename!r}; "
                "the tabular upload does not appear to contain a free-text column."
            ),
        )

    target_column: Optional[str] = classifier_response.get("target_name_column")
    target_name_value: Optional[str] = classifier_response.get("target_name_value")

    dominant_target: Optional[str] = None
    if target_column and target_column in headers:
        candidate, share = _csv_dominant_value(
            [str(r.get(target_column) or "") for r in rows]
        )
        if candidate and share >= 0.8:
            dominant_target = candidate

    if not target_name_value:
        target_name_value = dominant_target or _filename_target_hint(source_filename)

    statements: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.get(text_column) or "").strip()
        if not text:
            continue
        if target_column and target_column in headers:
            row_target = str(row.get(target_column) or "").strip() or target_name_value
        else:
            row_target = target_name_value
        statements.append(
            {
                "messages": [{"role": "assistant", "content": text}],
                "metadata": {
                    "target": row_target or None,
                    "source": source_filename,
                },
            }
        )

    if not statements:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Tabular upload {source_filename!r} produced no non-empty rows "
                f"in column {text_column!r}."
            ),
        )

    return {
        "statements": statements,
        "metadata": {
            "target": target_name_value or None,
            "source": source_filename,
            "csv_text_column": text_column,
            "csv_target_column": target_column,
            "csv_row_count": len(statements),
            "csv_classifier_reasoning": classifier_response.get("reasoning", ""),
        },
    }


def _build_csv_statements_media_entry(
    *,
    payload: dict[str, Any],
    source_filename: str,
    user_id: str,
    assistant_id: str,
) -> dict[str, Any]:
    """Render the CSV preprocessing payload as a JSON-typed media_files entry.

    The downstream process_media_graph already routes ``application/json``
    files with a ``.json`` suffix through the JSON handler in
    ``process_media_graph/utils/nodes.py``, which now understands the
    ``{"statements": [...]}`` shape produced by CSV preprocessing.
    """
    statements_blob = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    metadata = payload.get("metadata") or {}
    return {
        "filename": source_filename,
        "content_type": "application/json",
        "content": statements_blob,
        "user_id": user_id,
        "assistant_id": assistant_id,
        "reference_audio": False,
        "reference_image": False,
        "base64_encoded_str": make_data_uri("application/json", statements_blob),
        "csv_target_name": metadata.get("target"),
        "csv_text_column": metadata.get("csv_text_column"),
        "csv_target_column": metadata.get("csv_target_column"),
        "csv_row_count": metadata.get("csv_row_count"),
        "namespace_filename": source_filename
        if not "." in source_filename
        else _namespace_safe_formatted_filename(source_filename),
    }


# ---------------------------------------------------------------------------
# Tabular JSON ingest preprocessing
#
# A JSON upload can be the SAME table a CSV would carry — e.g. a pandas
# ``DataFrame.to_json()`` dump (orient="columns": ``{column: {row_key: value}}``)
# or orient="records" (a list of flat dicts). Those are detected here and pushed
# through the exact CSV pipeline (``_rows_to_statements_payload``) so the
# process_media graph only ever sees the ``{"statements": [...]}`` contract.
# JSON that is already contract-shaped (``{"statements": [...]}`` or
# ``{"messages": [...]}``, including JSON-Lines files of statement objects)
# passes through untouched — the graph's JSON handler owns those shapes.
# ---------------------------------------------------------------------------

_JSON_MIME_HINTS = frozenset(
    {
        "application/json",
        "application/x-ndjson",
        "application/jsonl",
        "application/json-lines",
    }
)


def _is_json_upload(filename: str, content_type: str) -> bool:
    """True when the upload looks like JSON / JSON-Lines by MIME type or filename."""
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct in _JSON_MIME_HINTS:
        return True
    name = (filename or "").strip().lower()
    return name.endswith((".json", ".jsonl", ".ndjson"))


def _is_tabular_scalar(value: Any) -> bool:
    """True for cell values a table can hold (no nested containers)."""
    return value is None or isinstance(value, (str, int, float, bool))


def _tabular_cell_to_str(value: Any) -> str:
    """Render a scalar cell the way ``_parse_csv_to_rows`` renders CSV cells."""
    return "" if value is None else str(value)


def _normalize_tabular_json_to_rows(
    parsed: Any,
) -> Optional[tuple[list[str], list[dict]]]:
    """Return ``(headers, rows)`` when ``parsed`` JSON is a flat table, else None.

    Recognized tabular shapes (both produced by ``pandas.DataFrame.to_json``):

    * orient="columns" — ``{column_name: {row_key: scalar}}``;
    * orient="records" — ``[{column_name: scalar}, ...]``.

    A dict carrying the avatar-identity contract keys (``statements`` /
    ``messages`` lists) is never treated as a table, and any nested container
    cell disqualifies the shape — those payloads pass through to the
    process_media graph unchanged.
    """
    if isinstance(parsed, dict):
        if isinstance(parsed.get("statements"), list) or isinstance(
            parsed.get("messages"), list
        ):
            return None
        if not parsed or not all(
            isinstance(column_values, dict) for column_values in parsed.values()
        ):
            return None
        for column_values in parsed.values():
            if not all(
                _is_tabular_scalar(cell_value)
                for cell_value in column_values.values()
            ):
                return None
        headers = [str(column_name) for column_name in parsed.keys()]
        # Row keys in first-seen order across columns (columns may be sparse).
        row_keys: list[Any] = []
        seen_row_keys: set[Any] = set()
        for column_values in parsed.values():
            for row_key in column_values.keys():
                if row_key not in seen_row_keys:
                    seen_row_keys.add(row_key)
                    row_keys.append(row_key)
        if not row_keys:
            return None
        rows = [
            {
                header: _tabular_cell_to_str(column_values.get(row_key))
                for header, column_values in zip(headers, parsed.values())
            }
            for row_key in row_keys
        ]
        return headers, rows

    if isinstance(parsed, list):
        if not parsed or not all(isinstance(record, dict) for record in parsed):
            return None
        headers = []
        seen_headers: set[str] = set()
        for record in parsed:
            for column_name, cell_value in record.items():
                if not _is_tabular_scalar(cell_value):
                    return None
                if column_name not in seen_headers:
                    seen_headers.add(column_name)
                    headers.append(str(column_name))
        if not headers:
            return None
        rows = [
            {header: _tabular_cell_to_str(record.get(header)) for header in headers}
            for record in parsed
        ]
        return headers, rows

    return None


async def _tabular_json_to_statements_payload(
    *, raw: bytes, source_filename: str
) -> Optional[dict[str, Any]]:
    """Convert a tabular JSON / JSON-Lines upload into the statements document.

    Returns None when the payload is not a flat table (contract-shaped JSON,
    arbitrary JSON, undecodable bytes) so the caller passes the file through to
    the process_media graph unchanged.
    """
    text = _decode_csv_bytes(raw)
    parsed: Any
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Possibly JSON-Lines: one object per line. A records-of-scalars file is
        # a table; statement-shaped lines come back None from the normalizer and
        # the graph's JSON-Lines parser handles them instead.
        line_objects: list[Any] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                line_objects.append(json.loads(line))
            except json.JSONDecodeError:
                return None
        if not line_objects:
            return None
        parsed = line_objects

    normalized = _normalize_tabular_json_to_rows(parsed)
    if normalized is None:
        return None
    headers, rows = normalized
    return await _rows_to_statements_payload(
        headers=headers, rows=rows, source_filename=source_filename
    )


@app.get("/avatar_reference_image")
async def get_avatar_reference_image(
    request: Request,
    assistant_id: str,
    current_user: dict = Depends(get_current_user_or_anonymous_user),
):
    """Return stored reference image data URI or image URL string for UI avatars.

    Lookup uses the assistant owner's store namespace so anonymous chatters see the
    same portrait that the chat-time consciousness loader reads.
    """
    store = app.state.store
    if request.headers.get("api-key", "") != "":
        langgraph_client_headers = {"API-KEY": request.headers.get("api-key")}
    else:
        langgraph_client_headers = {"API-KEY": app.state.context.anonymous_api_key}
    try:
        langgraph_client = get_client(headers=langgraph_client_headers)
        assistant = await langgraph_client.assistants.get(assistant_id=assistant_id)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not load assistant: {exc}"
        ) from exc
    assistant_owner_user_id = (assistant.get("metadata") or {}).get("user_id")
    if not assistant_owner_user_id:
        return JSONResponse({"reference_image_data": None})
    namespace = (assistant_owner_user_id, assistant_id, "reference_image")
    item = await store.aget(namespace, assistant_id)
    if item is None:
        return JSONResponse({"reference_image_data": None})
    if isinstance(item, dict):
        value = item.get("value") or {}
    else:
        value = getattr(item, "value", None) or {}
    return JSONResponse({"reference_image_data": value.get("reference_image_data")})


from typing import Optional

_MANIFEST_TEXT_MIMES = frozenset(
    {"text/plain", "text/markdown", "application/octet-stream"}
)


def _looks_like_manifest_candidate(filename: str, mime_type: str) -> bool:
    """True for uploads that could be a newline-delimited URL list (.txt/.md).

    CSVs are handled separately and excluded by the caller. Octet-stream is
    allowed because browsers often send .txt/.md that way.
    """
    name = (filename or "").lower()
    return (
        mime_type in _MANIFEST_TEXT_MIMES
        or mime_type.startswith("text/")
        or name.endswith(".txt")
        or name.endswith(".md")
    )


def _extract_manifest_urls(raw: bytes) -> List[str]:
    """Pull the http(s) URLs out of a text/markdown manifest.

    A line counts only if, stripped, it is *itself* a single URL — name/header
    lines (e.g. ``Gracie Abrams``) and prose are ignored, so the same parser
    handles a pure list (``confirmed_search_results_list.txt``) and a
    name+URL playlist list. Returns ``[]`` when no bare-URL line is present, in
    which case the caller ingests the file as an ordinary text document.
    Order is preserved and duplicates dropped.
    """
    from urllib.parse import urlparse

    text = raw.decode("utf-8", errors="replace")
    seen: set[str] = set()
    urls: List[str] = []
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate or candidate.startswith("#"):
            continue
        # Markdown bullets / numbering before a bare URL: strip common prefixes.
        candidate = candidate.lstrip("-*•").strip()
        try:
            parsed = urlparse(candidate)
        except Exception:
            continue
        if (
            parsed.scheme in ("http", "https")
            and parsed.netloc
            and " " not in candidate
        ):
            if candidate not in seen:
                seen.add(candidate)
                urls.append(candidate)
    return urls


def _collect_input_urls(raw_url_field: Any) -> List[str]:
    """Flatten the ``url`` form field into an ordered, de-duplicated list of URLs.

    The field may arrive as a single string or as repeated ``url=`` form fields
    (FastAPI gives us ``list[str]``), and each value may itself hold several URLs
    pasted one-per-line (or whitespace-separated). Non-URL tokens — e.g. a name
    line above a playlist link — are ignored, mirroring ``_extract_manifest_urls``
    so the same paste works in the field or in an uploaded manifest. Order is
    preserved and duplicates dropped.

    The ``url`` field is typed as an array (``Optional[List[str]]``), so API
    browsers (Swagger "Try it out", the LangGraph API explorer, etc.) often
    present and submit it as a JSON-array literal — e.g. ``["https://…"]`` or a
    bare ``[https://…]`` — rather than a plain value. We strip the surrounding
    ``[ ]`` brackets, quotes and stray commas per token so a single URL works
    whether or not the user wraps it in quotes/brackets. We deliberately do *not*
    strip ``) }`` (keeps Wikipedia-style ``…_(disambiguation)`` URLs intact) and
    split on whitespace only (keeps comma-bearing query strings intact).
    """
    from urllib.parse import urlparse

    if raw_url_field is None:
        values: List[str] = []
    elif isinstance(raw_url_field, str):
        values = [raw_url_field]
    else:
        values = [v for v in raw_url_field if isinstance(v, str)]

    seen: set[str] = set()
    urls: List[str] = []
    for value in values:
        for token in re.split(r"\s+", value or ""):
            candidate = token.strip().strip("<>\"'[],").lstrip("-*•").strip()
            if not candidate:
                continue
            try:
                parsed = urlparse(candidate)
            except Exception:
                continue
            if parsed.scheme in ("http", "https") and parsed.netloc:
                if candidate not in seen:
                    seen.add(candidate)
                    urls.append(candidate)
    return urls


def _lightweight_url_media_entry(
    url_clean: str, *, user_id: str, assistant_id: str
) -> dict:
    """A generic ``page_url`` entry for bulk URLs — no upfront content-type probe.

    The media graph's ``URLDocumentLoaderClass`` classifies and expands it
    (youtube/playlist/twitter/instagram/twitch/linktree/article). Used for every
    URL in a multi-input or manifest request so the endpoint can return 202 fast
    instead of probing hundreds of URLs serially.
    """
    return {
        "filename": url_clean,
        "content_type": "text/html",
        "content": b"",
        "page_url": url_clean,
        "user_id": user_id,
        "assistant_id": assistant_id,
        "reference_audio": False,
        "reference_image": False,
        "namespace_filename": _namespace_safe_formatted_filename(url_clean),
    }


async def _build_media_entries_for_file(
    raw_name: str,
    content: bytes,
    mime_type: str,
    *,
    reference_image: bool,
    reference_audio: bool,
    user_id: str,
    assistant_id: str,
) -> list:
    """Build the ``media_files`` entries for a single uploaded file.

    Extracted verbatim from the original single-file branch so it can run per
    file in a multi-file request. Raises ``HTTPException`` on unsupported types.
    """
    entries: list = []
    if (
        not reference_image
        and not reference_audio
        and _is_csv_upload(raw_name, mime_type)
    ):
        csv_payload = await _csv_to_statements_payload(
            raw=content, source_filename=raw_name
        )
        entries.append(
            _build_csv_statements_media_entry(
                payload=csv_payload,
                source_filename=raw_name,
                user_id=user_id,
                assistant_id=assistant_id,
            )
        )
    elif not reference_image and not reference_audio and _is_json_upload(
        raw_name, mime_type
    ):
        # A JSON upload may be the same table a CSV would carry (a pandas
        # to_json dump). Convert tabular shapes through the CSV statements
        # pipeline; anything else (contract-shaped statements/messages JSON,
        # JSON-Lines of statement objects, arbitrary JSON) passes through as a
        # plain upload for the process_media graph's JSON handler.
        tabular_statements_payload = await _tabular_json_to_statements_payload(
            raw=content, source_filename=raw_name
        )
        if tabular_statements_payload is not None:
            entries.append(
                _build_csv_statements_media_entry(
                    payload=tabular_statements_payload,
                    source_filename=raw_name,
                    user_id=user_id,
                    assistant_id=assistant_id,
                )
            )
        else:
            entries.append(
                {
                    "filename": raw_name,
                    "content_type": mime_type,
                    "content": content,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "reference_audio": False,
                    "reference_image": False,
                    "base64_encoded_str": make_data_uri(mime_type, content),
                    "namespace_filename": raw_name if not "." in raw_name else _namespace_safe_formatted_filename(raw_name),
                }
            )
    elif reference_image:
        if mime_type.startswith("audio/"):
            raise HTTPException(
                status_code=400,
                detail="reference_image requires an image file, not audio.",
            )
        mime = validate_upload_image_bytes(mime_type, content)
        entries.append(
            {
                "filename": raw_name,
                "content_type": mime,
                "content": content,
                "user_id": user_id,
                "assistant_id": assistant_id,
                "reference_audio": False,
                "reference_image": True,
                "base64_encoded_str": make_data_uri(mime, content),
                "namespace_filename": raw_name
                if not "." in raw_name
                else _namespace_safe_formatted_filename(raw_name),
            }
        )
    elif reference_audio:
        if mime_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="reference_audio requires an audio file, not an image.",
            )
        sniff = _sniff_media_category_from_bytes(content[:512])
        effective = mime_type
        if mime_type == "application/octet-stream":
            if not sniff or not sniff.startswith("audio/"):
                raise HTTPException(
                    status_code=400,
                    detail="Could not determine an audio type from the upload.",
                )
            effective = sniff
        elif not mime_type.startswith("audio/") and not mime_type.startswith("video/"):
            raise HTTPException(
                status_code=400,
                detail="reference_audio requires an audio or video Content-Type.",
            )
        entries.append(
            {
                "filename": raw_name,
                "content_type": effective,
                "content": content,
                "user_id": user_id,
                "assistant_id": assistant_id,
                "reference_audio": True,
                "reference_image": False,
                "base64_encoded_str": make_data_uri(effective, content),
                "namespace_filename": raw_name
                if not "." in raw_name
                else _namespace_safe_formatted_filename(raw_name),
            }
        )
    else:
        sniff = _sniff_media_category_from_bytes(content[:512])
        if mime_type.startswith("image/") or (
            mime_type == "application/octet-stream" and sniff in ALLOWED_IMAGE_MIMES
        ):
            mime = validate_upload_image_bytes(mime_type, content)
            entries.append(
                {
                    "filename": raw_name,
                    "content_type": mime,
                    "content": content,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "reference_audio": False,
                    "reference_image": False,
                    "base64_encoded_str": make_data_uri(mime, content),
                    "namespace_filename": raw_name
                    if not "." in raw_name
                    else _namespace_safe_formatted_filename(raw_name),
                }
            )
        elif mime_type.startswith("audio/") or (
            mime_type == "application/octet-stream"
            and sniff
            and sniff.startswith("audio/")
        ):
            effective = (
                mime_type if mime_type.startswith("audio/") else (sniff or mime_type)
            )
            if not effective.startswith("audio/"):
                raise HTTPException(
                    status_code=400,
                    detail="Expected an audio upload.",
                )
            entries.append(
                {
                    "filename": raw_name,
                    "content_type": effective,
                    "content": content,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "reference_audio": False,
                    "reference_image": False,
                    "base64_encoded_str": make_data_uri(effective, content),
                    "namespace_filename": raw_name
                    if not "." in raw_name
                    else _namespace_safe_formatted_filename(raw_name),
                }
            )
        elif mime_type.startswith("video/"):
            entries.append(
                {
                    "filename": raw_name,
                    "content_type": mime_type,
                    "content": content,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "reference_audio": False,
                    "reference_image": False,
                    "base64_encoded_str": make_data_uri(mime_type, content),
                    "namespace_filename": raw_name
                    if not "." in raw_name
                    else _namespace_safe_formatted_filename(raw_name),
                }
            )
        elif mime_type == "application/pdf":
            entries.append(
                {
                    "filename": raw_name,
                    "content_type": mime_type,
                    "content": content,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "reference_audio": False,
                    "reference_image": False,
                    "base64_encoded_str": make_data_uri(mime_type, content),
                    "namespace_filename": raw_name
                    if not "." in raw_name
                    else _namespace_safe_formatted_filename(raw_name),
                }
            )
        elif (
            mime_type
            in (
                "text/plain",
                "application/json",
                "text/markdown",
                "application/octet-stream",
            )
            or mime_type.startswith("text/")
            or (raw_name or "").lower().endswith(".log")
        ):
            entries.append(
                {
                    "filename": raw_name,
                    "content_type": mime_type,
                    "content": content,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "reference_audio": False,
                    "reference_image": False,
                    "base64_encoded_str": make_data_uri(mime_type, content),
                    "namespace_filename": raw_name
                    if not "." in raw_name
                    else _namespace_safe_formatted_filename(raw_name),
                }
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported upload Content-Type {mime_type!r}.",
            )
    return entries


def _is_youtube_playlist_url_str(url_clean: str) -> bool:
    """Cheap, network-free check for a YouTube **playlist** URL.

    Used on the request path to *detect* playlists (so they can be enumerated
    later, in the background) without paying for ``yt_dlp``. The enumeration
    itself lives in ``_expand_youtube_playlist_to_media_entries``.
    """
    from src.anubis.utils.classes.URLDocumentLoaderClass import _classify_url

    return _classify_url(url_clean) == "youtube_playlist"


async def _expand_youtube_playlist_to_media_entries(
    url_clean: str,
    *,
    user_id: str,
    assistant_id: str,
    create_reference_media_from_playlist: bool = False,
) -> Optional[list]:
    """Expand a YouTube **playlist** URL into one media entry per video.

    Each video becomes its own top-level item — and therefore its own child job
    with its own progress/cancel id — keyed by a single uuid5 over
    ``{playlist_ns}::{video_ns}`` and named ``{playlist}::{video}`` so the videos
    list, dedupe, and cancel individually rather than collapsing into a single
    playlist job. Playlist context (``playlist_url`` / title / ns) rides on each
    entry so the produced Documents get stamped, ``/list_avatar_documents`` groups
    every video under its playlist, and a whole-playlist delete can match them by
    ``playlist_namespace_filename``.

    Returns ``None`` for any non-playlist URL so the caller falls back to the
    normal single-URL path; returns ``[]`` if the playlist resolves to no videos.
    """
    # Lazy import (heavy yt_dlp path + cold-start convention). _classify_url is
    # pure; _extract_playlist_entries does the flat yt_dlp enumeration.
    from src.anubis.utils.classes.URLDocumentLoaderClass import (
        _classify_url,
        _extract_playlist_entries,
    )

    if _classify_url(url_clean) != "youtube_playlist":
        return None

    entries, playlist_title = await _extract_playlist_entries(url_clean)
    if not entries:
        logger.warning("YouTube playlist produced no entries: %s", url_clean)
        return []

    # playlist_ns mirrors URLDocumentLoaderClass._namespace_for so the composite
    # keys built here match what the graph would have produced — dedup stays
    # consistent across upload paths.
    playlist_ns = _namespace_safe_formatted_filename(url_clean)
    playlist_label = (playlist_title or url_clean).strip()
    media_entries: list = []
    for entry in entries:
        video_id = entry.get("id")
        watch_url = entry.get("url") or (
            f"https://www.youtube.com/watch?v={video_id}" if video_id else None
        )
        if not watch_url:
            continue
        video_ns = _namespace_safe_formatted_filename(watch_url)
        video_title = (entry.get("title") or "").strip()
        media_entries.append(
            {
                "filename": f"{playlist_label}::{video_title or watch_url}",
                "content_type": "text/html",
                "content": b"",
                "page_url": watch_url,
                "user_id": user_id,
                "assistant_id": assistant_id,
                "reference_audio": False,
                "reference_image": False,
                "create_reference_media_from_playlist": create_reference_media_from_playlist,
                # Single opaque uuid5 over the composite so the store key carries
                # no ``::`` separator. The playlist a video belongs to is recovered
                # from playlist_namespace_filename below (and from playlist_url /
                # title for the listing), not by parsing this key.
                "namespace_filename": _namespace_safe_formatted_filename(
                    f"{playlist_ns}::{video_ns}"
                ),
                "playlist_url": url_clean,
                "playlist_namespace_filename": playlist_ns,
                "playlist_title": playlist_title,
                "video_title": video_title,
                "url_kind": "youtube_playlist_entry",
            }
        )
    logger.info(
        "Expanded YouTube playlist %s (%s) into %d per-video upload items",
        url_clean,
        playlist_title or "untitled",
        len(media_entries),
    )
    return media_entries


async def _build_media_entries_for_url(
    url_clean: str,
    *,
    reference_image: bool,
    reference_audio: bool,
    user_id: str,
    assistant_id: str,
    rich: bool,
) -> list:
    """Build the ``media_files`` entries for a single URL.

    ``rich=True`` runs the original per-URL content-type probing path (handles
    direct image/audio/video/csv URLs and reference flags) — used for a lone URL
    request. ``rich=False`` returns one lightweight ``page_url`` entry that the
    media graph classifies, avoiding an upfront probe per URL in bulk requests.
    """
    if not rich:
        return [
            _lightweight_url_media_entry(
                url_clean, user_id=user_id, assistant_id=assistant_id
            )
        ]

    entries: list = []
    namespace_safe_formatted_filename = _namespace_safe_formatted_filename(url_clean)

    if reference_image:
        body, header_ct = await fetch_remote_url_bytes(url_clean)
        img_mime = validate_upload_image_bytes(header_ct, body)
        entries.append(
            {
                "filename": url_clean,
                "content_type": img_mime,
                "content": b"",
                "image_url": url_clean,
                "user_id": user_id,
                "assistant_id": assistant_id,
                "reference_audio": False,
                "reference_image": True,
                "base64_encoded_str": make_data_uri(img_mime, body),
                "namespace_filename": namespace_safe_formatted_filename
                if not "." in namespace_safe_formatted_filename
                else _namespace_safe_formatted_filename(
                    namespace_safe_formatted_filename
                ),
            }
        )
    elif reference_audio:
        # YouTube watch pages report Content-Type: text/html. Bypass the
        # audio/* guard for those by pulling the audio track via yt_dlp.
        if _is_youtube_url(url_clean):
            from src.anubis.utils.classes.URLDocumentLoaderClass import (
                _download_youtube_audio_b64,
            )

            audio_data_uri, _suffix = await _download_youtube_audio_b64(url_clean)
            if not audio_data_uri:
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract audio from YouTube URL.",
                )
            entries.append(
                {
                    "filename": url_clean,
                    "content_type": "audio/mp3",
                    "content": b"",
                    "audio_url": url_clean,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "reference_audio": True,
                    "reference_image": False,
                    "base64_encoded_str": audio_data_uri,
                    "namespace_filename": _namespace_safe_formatted_filename(url_clean),
                }
            )
        else:
            await require_url_content_type_prefix(
                url_clean, "audio/", "Reference audio"
            )
            body, header_ct = await fetch_remote_url_bytes(url_clean)
            sniff = _sniff_media_category_from_bytes(body[:512])
            audio_mime = (
                header_ct
                if header_ct.startswith("audio/")
                else (sniff if sniff.startswith("audio/") else header_ct)
            )
            entries.append(
                {
                    "filename": url_clean,
                    "content_type": audio_mime,
                    "content": b"",
                    "audio_url": url_clean,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "reference_audio": True,
                    "reference_image": False,
                    "base64_encoded_str": make_data_uri(audio_mime, body),
                    "namespace_filename": url_clean
                    if not "." in url_clean
                    else _namespace_safe_formatted_filename(url_clean),
                }
            )
    else:
        # YouTube URLs probe as text/html but their payload is video/audio.
        # Route them directly to the URL pipeline so URLDocumentLoaderClass
        # can pull subtitles or audio via yt_dlp without us first
        # downloading the HTML page.
        if _is_youtube_url(url_clean):
            entries.append(
                {
                    "filename": url_clean,
                    "content_type": "text/html",
                    "content": b"",
                    "page_url": url_clean,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "reference_audio": False,
                    "reference_image": False,
                    "namespace_filename": _namespace_safe_formatted_filename(url_clean),
                }
            )
            ct = ""  # skip the per-Content-Type branches below
        else:
            ct = await probe_remote_url_content_type(url_clean)
        if not ct:
            pass
        elif _is_csv_upload(namespace_safe_formatted_filename or url_clean, ct):
            body, _header_ct = await fetch_remote_url_bytes(url_clean)
            csv_filename = (
                namespace_safe_formatted_filename
                if namespace_safe_formatted_filename.endswith((".csv", ".tsv"))
                else (f"{namespace_safe_formatted_filename or 'remote_table'}.csv")
            )
            csv_payload = await _csv_to_statements_payload(
                raw=body, source_filename=csv_filename
            )
            entries.append(
                _build_csv_statements_media_entry(
                    payload=csv_payload,
                    source_filename=csv_filename,
                    user_id=user_id,
                    assistant_id=assistant_id,
                )
            )
        elif ct.startswith("image/"):
            body, header_ct = await fetch_remote_url_bytes(url_clean)
            img_mime = validate_upload_image_bytes(header_ct, body)
            entries.append(
                {
                    "filename": url_clean,
                    "content_type": img_mime,
                    "content": b"",
                    "image_url": url_clean,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "reference_audio": False,
                    "reference_image": False,
                    "base64_encoded_str": make_data_uri(img_mime, body),
                    "namespace_filename": url_clean
                    if not "." in url_clean
                    else _namespace_safe_formatted_filename(url_clean),
                }
            )
        elif ct.startswith("audio/"):
            body, header_ct = await fetch_remote_url_bytes(url_clean)
            sniff = _sniff_media_category_from_bytes(body[:512])
            audio_mime = (
                header_ct
                if header_ct.startswith("audio/")
                else (sniff if sniff.startswith("audio/") else ct)
            )
            entries.append(
                {
                    "filename": url_clean,
                    "content_type": audio_mime,
                    "content": b"",
                    "audio_url": url_clean,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "reference_audio": False,
                    "reference_image": False,
                    "base64_encoded_str": make_data_uri(audio_mime, body),
                    "namespace_filename": url_clean
                    if not "." in url_clean
                    else _namespace_safe_formatted_filename(url_clean),
                }
            )
        elif ct.startswith("video/"):
            body, header_ct = await fetch_remote_url_bytes(url_clean)
            video_mime = header_ct if header_ct.startswith("video/") else ct
            entries.append(
                {
                    "filename": url_clean,
                    "content_type": video_mime,
                    "content": b"",
                    "video_url": url_clean,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "reference_audio": False,
                    "reference_image": False,
                    "base64_encoded_str": make_data_uri(video_mime, body),
                    "namespace_filename": url_clean
                    if not "." in url_clean
                    else _namespace_safe_formatted_filename(url_clean),
                }
            )
        elif ct.startswith("text/") or ct in (
            "application/json",
            "application/xml",
            "application/xhtml+xml",
            "application/javascript",
            "application/ld+json",
        ):
            body, header_ct = await fetch_remote_url_bytes(url_clean)
            doc_mime = (
                header_ct
                if (
                    header_ct.startswith("text/")
                    or header_ct
                    in (
                        "application/json",
                        "application/xml",
                        "application/xhtml+xml",
                        "application/javascript",
                        "application/ld+json",
                    )
                )
                else ct
            )
            entries.append(
                {
                    "filename": url_clean,
                    "content_type": doc_mime,
                    "content": b"",
                    "page_url": url_clean,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "reference_audio": False,
                    "reference_image": False,
                    "base64_encoded_str": make_data_uri(doc_mime, body),
                    "namespace_filename": url_clean
                    if not "." in url_clean
                    else _namespace_safe_formatted_filename(url_clean),
                }
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Could not map URL to a supported media type (Content-Type: {ct!r}).",
            )
    return entries


@app.post("/update_avatar_identity_with_media")
async def update_avatar_identity_with_media(
    files: OptionalUploadFiles = None,
    url: Annotated[Optional[List[str]], Form()] = None,
    assistant_id: Annotated[Optional[str], Form()] = None,
    reference_audio: Annotated[bool, Form()] = False,
    reference_image: Annotated[bool, Form()] = False,
    create_reference_media_from_playlist: Annotated[bool, Form()] = False,
    current_user: dict = Depends(get_current_user),
):
    # Context user_id, assistant_id
    """
    Upload media for processing and indexing.

    Accepts **any mix** of files and URLs in a single request — multiple files in
    ``files`` and/or one or more URLs in ``url`` (repeated ``url=`` fields and/or
    several URLs pasted one-per-line into a single field both work). A ``.txt`` /
    ``.md`` file whose lines are bare URLs is treated as a **manifest**: its URLs
    are expanded and processed individually (name/header lines are ignored), so a
    saved list like ``confirmed_search_results_list.txt`` works the same as pasting
    the URLs. A YouTube **playlist** URL is enumerated in the background (so the
    202 isn't blocked on yt_dlp) into one item per video — each its own upload
    with its own progress/cancel id, listed individually as ``{playlist}::{video}``;
    those child ids appear on the master's progress stream as
    ``playlist_child_added`` events. Every item is processed in parallel (bounded
    by ``media_processing_concurrency``); items whose key already exists for this
    avatar (see ``/list_avatar_documents``) are **skipped**, so re-uploading a
    large playlist only processes new videos. The endpoint returns ``202`` with a
    ``job_id`` immediately; progress streams from
    ``GET /media_job/{job_id}/progress``.

    Images must use real MIME types: ``image/jpeg``, ``image/png``, ``image/gif`` (non-animated),
    or ``image/webp`` (non-animated). Proprietary vs biographical classification is done inside
    the processing pipeline via structured model output (no ``proprietary_content`` flag).

    With **reference_image=true** or **reference_audio=true** the request must carry
    **exactly one** file or URL (a reference clip/image is a single item): the file
    or URL must be an allowed still image, or resolve to ``audio/*``, respectively.

    With **create_reference_media_from_playlist=true** the batch has **no single target speaker**:
    every detected speaker is the avatar. Audio/video items are still diarized (so
    no stored reference-audio clip is required and known-speaker labelling is
    skipped). With **multiple speakers**, each statement becomes one ``quote``
    training example whose question is the **preceding statement** (the first
    statement, having no predecessor, gets a synthesized question). With a
    **single speaker** the transcript is a monologue: it is classified normally
    (monologue / tweets_or_quotes), which stores it in the vectorstore, marks it
    analysis-acceptable, and makes it adapter-acceptable with a synthesized prompt.
    YouTube items are forced onto the audio/diarize path (subtitles, which carry no
    speaker turns, are skipped). Use it for playlists/recordings where all voices
    belong to the avatar. It is mutually exclusive with ``reference_image`` /
    ``reference_audio`` (which designate a single target) and applies to every item
    in the request, including expanded playlist children.
    """
    try:
        user_id = current_user["identities"][0]["user_id"]
        if not assistant_id:
            raise HTTPException(status_code=400, detail="assistant_id is required")

        token = current_user["API_KEY"]
        client = get_client(headers={"API-KEY": f"{token}"})
        try:
            assistant = await client.assistants.get(assistant_id)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Could not load assistant: {exc}"
            ) from exc
        assistant_meta = assistant.get("metadata") or {}
        creator_id = assistant_meta.get("user_id")
        if not creator_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Assistant metadata is missing the creator's user_id; "
                    "cannot verify upload permissions."
                ),
            )
        if user_id != creator_id:
            raise HTTPException(
                status_code=403,
                detail=(
                    "Only the creator of this avatar may upload media for it. "
                    "The signed-in user is not the assistant's creator."
                ),
            )

        config = {
            "configurable": {
                "user_id": user_id,
                "user_ctx": {"name": None, "description": None},
                "assistant_id": assistant_id,
                "assistant_ctx": {
                    "name": assistant.get("name"),
                    "description": assistant.get("description"),
                    "assistant_id": assistant_id,
                    "metadata": assistant_meta,
                },
            }
        }

        upload_list = [f for f in (files or []) if f is not None]
        non_empty_files = [
            f for f in upload_list if (getattr(f, "filename", None) or "").strip()
        ]
        # ``url`` may arrive as one field, repeated ``url=`` fields, or several URLs
        # pasted one-per-line into a single field. Flatten to an ordered,
        # de-duplicated list of bare URLs (name/header lines ignored).
        input_urls = _collect_input_urls(url)

        if not non_empty_files and not input_urls:
            raise HTTPException(
                status_code=400,
                detail="Send at least one file or url.",
            )
        if reference_image and reference_audio:
            raise HTTPException(
                status_code=400,
                detail="Use only one of reference_image or reference_audio.",
            )
        if create_reference_media_from_playlist and (
            reference_image or reference_audio
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "create_reference_media_from_playlist cannot be combined with "
                    "reference_image/reference_audio: a reference clip designates a "
                    "single target, while create_reference_media_from_playlist treats every detected "
                    "speaker as the target."
                ),
            )

        reference_mode = reference_image or reference_audio
        if reference_mode and (len(non_empty_files) + len(input_urls)) != 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    "reference_image/reference_audio requires exactly one file or "
                    "url (a reference clip or image is a single item)."
                ),
            )

        media_files: list = []
        # Playlist URLs detected on the request path but enumerated later, in the
        # background master task, so the 202 isn't blocked on yt_dlp. Each yields
        # one child job per video once expanded.
        playlist_urls: list[str] = []

        if reference_mode:
            # Exactly one input (guarded above). No manifest expansion in reference
            # mode — a reference must be a single image/audio item.
            if non_empty_files:
                uf = non_empty_files[0]
                content = await uf.read()
                mime_type = (
                    (uf.content_type or "application/octet-stream")
                    .split(";")[0]
                    .strip()
                    .lower()
                )
                media_files.extend(
                    await _build_media_entries_for_file(
                        uf.filename,
                        content,
                        mime_type,
                        reference_image=reference_image,
                        reference_audio=reference_audio,
                        user_id=user_id,
                        assistant_id=assistant_id,
                    )
                )
            else:
                media_files.extend(
                    await _build_media_entries_for_url(
                        input_urls[0],
                        reference_image=reference_image,
                        reference_audio=reference_audio,
                        user_id=user_id,
                        assistant_id=assistant_id,
                        rich=True,
                    )
                )
        else:
            # General path: build entries for every file (expanding any URL-manifest
            # .txt/.md into its URLs) and every URL. A lone file/URL keeps the rich
            # single-item path (direct image/audio/video/csv links + content
            # probing); anything larger defers URL classification/expansion to the
            # media graph so the endpoint returns 202 fast instead of probing
            # hundreds of URLs serially. Each item is processed in parallel there.
            file_entries: list = []
            manifest_urls: list[str] = []
            for uf in non_empty_files:
                content = await uf.read()
                raw_name = uf.filename
                mime_type = (
                    (uf.content_type or "application/octet-stream")
                    .split(";")[0]
                    .strip()
                    .lower()
                )
                # A .txt/.md (non-CSV) whose lines are bare URLs is a manifest:
                # expand it into URLs instead of ingesting it as a text document.
                if not _is_csv_upload(raw_name, mime_type) and (
                    _looks_like_manifest_candidate(raw_name, mime_type)
                ):
                    extracted = _extract_manifest_urls(content)
                    if extracted:
                        manifest_urls.extend(extracted)
                        continue
                file_entries.extend(
                    await _build_media_entries_for_file(
                        raw_name,
                        content,
                        mime_type,
                        reference_image=False,
                        reference_audio=False,
                        user_id=user_id,
                        assistant_id=assistant_id,
                    )
                )

            # Merge explicit + manifest URLs, de-duplicated, order preserved.
            all_urls: list[str] = []
            seen_urls: set[str] = set()
            for u in (*input_urls, *manifest_urls):
                if u not in seen_urls:
                    seen_urls.add(u)
                    all_urls.append(u)

            # Probe each URL up front only for a single lone URL; bulk requests use
            # lightweight entries the media graph classifies and expands.
            rich_urls = len(file_entries) == 0 and len(all_urls) == 1
            url_entries: list = []
            for u in all_urls:
                # A playlist is set aside for background enumeration (one child job
                # per video, expanded off the request path); non-playlist URLs take
                # the normal single-URL path here.
                if _is_youtube_playlist_url_str(u):
                    playlist_urls.append(u)
                    continue
                url_entries.extend(
                    await _build_media_entries_for_url(
                        u,
                        reference_image=False,
                        reference_audio=False,
                        user_id=user_id,
                        assistant_id=assistant_id,
                        rich=rich_urls,
                    )
                )

            media_files = [*file_entries, *url_entries]

        # A playlist-only upload has no ready media_files yet (its videos are
        # enumerated in the background), so only reject when nothing at all — no
        # files and no playlists — was found.
        if not media_files and not playlist_urls:
            raise HTTPException(
                status_code=400,
                detail="No processable media found in the request.",
            )

        # Stamp the batch-wide "no single target" flag onto every entry (top
        # level, alongside reference_audio/reference_image). convert_uploaded_
        # files_to_media reads it for audio/video/url items and threads it into
        # their metadata; expanded playlist children inherit it downstream.
        if create_reference_media_from_playlist:
            for entry in media_files:
                entry["create_reference_media_from_playlist"] = True

        store = app.state.store

        # Collect every namespace_filename already indexed for this avatar. The
        # store layout ((user_id, assistant_id, <category>)) mirrors what
        # /list_avatar_documents exposes; keys are read from
        # value.document.kwargs.metadata.namespace_filename. This set is handed to
        # the media graph, which skips any incoming item — or expanded playlist /
        # linktree child — whose key is already present, so re-uploading a large
        # playlist only processes new entries (the user's "skip what's already
        # uploaded" requirement). To refresh an existing item, delete it first via
        # DELETE /delete_avatar_document, then re-upload.
        try:
            existing_items = await store.asearch(
                (user_id, assistant_id), limit=1_000_000
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Could not read this avatar's existing media to skip "
                    f"already-indexed items: {exc}"
                ),
            ) from exc

        existing_namespaces: set[str] = set()
        for item in existing_items or []:
            value = getattr(item, "value", None)
            if value is None and isinstance(item, dict):
                value = item.get("value")
            if not isinstance(value, dict):
                continue
            document = value.get("document")
            if not isinstance(document, dict):
                continue
            kwargs_blob = document.get("kwargs")
            if not isinstance(kwargs_blob, dict):
                continue
            metadata = kwargs_blob.get("metadata")
            if not isinstance(metadata, dict):
                continue
            stored_filename = metadata.get("namespace_filename")
            if isinstance(stored_filename, str) and stored_filename.strip():
                existing_namespaces.add(stored_filename.strip())

        incoming_filenames = [
            name
            for name in (
                (entry.get("namespace_filename") or "").strip() for entry in media_files
            )
            if name
        ]
        already_indexed = sorted(
            {name for name in incoming_filenames if name in existing_namespaces}
        )
        if already_indexed:
            logger.info(
                "Skipping %d top-level item(s) already indexed for this avatar: %s",
                len(already_indexed),
                already_indexed,
            )

        # Media processing (diarization, PDFs, YouTube playlists, indexing) can run
        # well past the request timeout, so start it as a background job and return
        # immediately. Each top-level item gets its own child job (its own progress
        # stream + independently cancellable); a master job aggregates them and is
        # the handle to cancel the whole batch. Progress is streamed via
        # GET /media_job/{job_id}/progress for either id; cancel via
        # POST /media_job/{job_id}/cancel. Bytes are already in ``media_files`` and
        # ``store`` / ``context`` are long-lived app resources, so the task is safe
        # after return. ``existing_namespaces`` lets the graph skip already-indexed
        # items and the children that expand from playlists/linktrees.
        registry = app.state.media_jobs
        master = create_master_job(registry, user_id, assistant_id)

        items: list = []
        item_descriptors: list = []
        for media_file in media_files:
            child = create_child_job(
                registry,
                user_id=user_id,
                assistant_id=assistant_id,
                parent_id=master.job_id,
                filename=media_file.get("filename"),
                namespace_filename=media_file.get("namespace_filename"),
            )
            master.child_ids.append(child.job_id)
            items.append({"child": child, "media_file": media_file})
            item_descriptors.append(
                {
                    "job_id": child.job_id,
                    "filename": child.filename,
                    "status": child.status,
                    "status_url": f"/media_job/{child.job_id}",
                    "progress_url": f"/media_job/{child.job_id}/progress",
                    "cancel_url": f"/media_job/{child.job_id}/cancel",
                }
            )

        # Playlists are enumerated inside the background task (off the request
        # path); each binds its URL + flags into an async expander that mints one
        # child job per video under this master once it resolves.
        deferred_expanders = [
            functools.partial(
                _expand_youtube_playlist_to_media_entries,
                playlist_url,
                user_id=user_id,
                assistant_id=assistant_id,
                create_reference_media_from_playlist=create_reference_media_from_playlist,
            )
            for playlist_url in playlist_urls
        ]

        master.task = asyncio.create_task(
            run_batch_media_job(
                master,
                items,
                config,
                store,
                app.state.context,
                concurrency=max(1, app.state.context.media_processing_concurrency),
                existing_namespaces=sorted(existing_namespaces),
                registry=registry,
                deferred_expanders=deferred_expanders,
            )
        )

        # Media now runs as a background job, so per-file indexing failures can
        # no longer be reported synchronously here. The failed-file logic that
        # fixed the silent-success bug lives in ``run_media_job``: it captures
        # ``failed_to_index_files`` from the graph and surfaces it on the job
        # result, delivered to clients via the SSE ``done`` event on
        # ``/media_job/{job_id}/progress``.
        return JSONResponse(
            status_code=202,
            content={
                "job_id": master.job_id,
                "status": master.status,
                "status_url": f"/media_job/{master.job_id}",
                "progress_url": f"/media_job/{master.job_id}/progress",
                "cancel_url": f"/media_job/{master.job_id}/cancel",
                "items_accepted": len(media_files),
                "filenames": [m.get("filename") for m in media_files],
                "items": item_descriptors,
                # Playlists resolve to their per-video child jobs in the background;
                # those child ids surface on the master's progress stream as
                # ``playlist_child_added`` events rather than in this response.
                "playlists_expanding": len(playlist_urls),
                "message": (
                    "Media processing started; enumerating "
                    f"{len(playlist_urls)} playlist(s) in the background"
                    if playlist_urls
                    else "Media processing started"
                ),
            },
        )

    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing media: {str(e)}")


@app.get("/media_jobs")
async def list_media_jobs(
    include_finished: bool = False,
    assistant_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    """List the current user's media jobs — by default only the **active** ones.

    Returns one entry per top-level batch (the **master** job each upload creates),
    newest first, with rolled-up child status counts so a client can render an
    "uploads in progress" view without polling every ``/media_job/{job_id}``. A job
    is "active" while it is still ``queued`` or ``running`` (``done`` not yet set);
    finished jobs linger in the registry for ``_FINISHED_TTL_SECONDS`` and are only
    included when ``include_finished=true``. Pass ``assistant_id`` to scope the list
    to one avatar. The registry is per-process (see media_jobs.py), so this reflects
    jobs owned by the worker handling the request.
    """
    user_id = current_user["identities"][0]["user_id"]
    registry = app.state.media_jobs

    masters = [
        job
        for job in registry.values()
        if job.is_master
        and job.user_id == user_id
        and (assistant_id is None or job.assistant_id == assistant_id)
        and (include_finished or not job.done.is_set())
    ]
    masters.sort(key=lambda j: j.created_at, reverse=True)

    def _summary(job: MediaJob) -> dict:
        children = [registry[cid] for cid in job.child_ids if cid in registry]
        statuses = [c.status for c in children]
        return {
            "job_id": job.job_id,
            "assistant_id": job.assistant_id,
            "status": job.status,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "duration_seconds": job.duration_seconds,
            "children_total": len(children),
            "children_completed": statuses.count("completed"),
            "children_error": statuses.count("error"),
            "children_cancelled": statuses.count("cancelled"),
            "children_running": statuses.count("running"),
            "children_queued": statuses.count("queued"),
            "status_url": f"/media_job/{job.job_id}",
            "progress_url": f"/media_job/{job.job_id}/progress",
            "cancel_url": f"/media_job/{job.job_id}/cancel",
        }

    jobs = [_summary(job) for job in masters]
    return {"count": len(jobs), "jobs": jobs}


@app.get("/media_job/{job_id}")
async def media_job_status(
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Point-in-time snapshot of a media job — a pollable alternative to the SSE
    ``/progress`` stream.

    For a **master** job this lists its current child jobs, **including videos a
    YouTube playlist enumerated in the background** after the upload returned 202
    (those don't appear in the upload response because they don't exist yet at
    request time). Poll this to watch the queue fill in and drain; for a child job
    it returns that single item's status/result.
    """
    user_id = current_user["identities"][0]["user_id"]
    registry = app.state.media_jobs
    job: Optional[MediaJob] = get_job(registry, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown or expired job_id")
    if job.user_id != user_id:
        raise HTTPException(status_code=403, detail="This job belongs to another user.")

    def _descriptor(j: MediaJob) -> dict:
        return {
            "job_id": j.job_id,
            "filename": j.filename,
            "namespace_filename": j.namespace_filename,
            "status": j.status,
            "error": j.error,
            "created_at": j.created_at,
            "started_at": j.started_at,
            "finished_at": j.finished_at,
            "duration_seconds": j.duration_seconds,
            "progress_url": f"/media_job/{j.job_id}/progress",
            "cancel_url": f"/media_job/{j.job_id}/cancel",
        }

    snapshot: dict = {
        **_descriptor(job),
        "is_master": job.is_master,
        "result": job.result,
    }
    if job.is_master:
        children = [registry[cid] for cid in job.child_ids if cid in registry]
        statuses = [c.status for c in children]
        snapshot["children"] = [_descriptor(c) for c in children]
        snapshot["children_total"] = len(children)
        snapshot["children_completed"] = statuses.count("completed")
        snapshot["children_error"] = statuses.count("error")
        snapshot["children_cancelled"] = statuses.count("cancelled")
        snapshot["children_running"] = statuses.count("running")
        snapshot["children_queued"] = statuses.count("queued")
    return snapshot


@app.get("/media_job/{job_id}/progress")
async def media_job_progress(
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Stream progress (SSE) for a background media job started by
    ``/update_avatar_identity_with_media``.

    Replays any buffered ``media_progress`` events, then streams live ones with
    periodic keep-alive comments, ending with a ``done`` event carrying the final
    status and result (or error).
    """
    user_id = current_user["identities"][0]["user_id"]
    job: Optional[MediaJob] = get_job(app.state.media_jobs, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown or expired job_id")
    if job.user_id != user_id:
        raise HTTPException(status_code=403, detail="This job belongs to another user.")

    def _with_timing(payload: dict) -> dict:
        """Return a copy of ``payload`` stamped with the job's start time and the
        wall-clock seconds elapsed since processing began, so every SSE ``data:``
        frame carries timing. ``started_at`` is epoch seconds (set when the job
        flipped to running); fall back to ``created_at`` if it hasn't yet."""
        started = job.started_at or job.created_at
        return {
            **payload,
            "started_at": job.started_at,
            "elapsed_seconds": round(time_ns() / 1_000_000_000 - started, 3),
        }

    async def event_stream(job: MediaJob):
        yield f"data: {json.dumps(_with_timing({'type': 'status', 'status': job.status}), default=str)}\n\n"
        last_index = 0
        while True:
            # Drain everything appended since we last yielded.
            while last_index < len(job.events):
                yield f"data: {json.dumps(_with_timing(job.events[last_index]), default=str)}\n\n"
                last_index += 1

            if job.done.is_set() and last_index >= len(job.events):
                break

            # Clear-then-recheck guards against a wakeup lost between the length
            # check above and the wait below.
            job._updated.clear()
            if last_index < len(job.events) or job.done.is_set():
                continue
            try:
                await asyncio.wait_for(job._updated.wait(), timeout=15)
            except asyncio.TimeoutError:
                # Keep the connection alive AND report timing so clients can show
                # how long the current stage has been running.
                yield f"data: {json.dumps(_with_timing({'type': 'keep_alive'}), default=str)}\n\n"

        done = _with_timing(
            {
                "type": "done",
                "status": job.status,
                "result": job.result,
                "error": job.error,
                "finished_at": job.finished_at,
                "duration_seconds": job.duration_seconds,
            }
        )
        yield f"data: {json.dumps(done, default=str)}\n\n"

    return StreamingResponse(
        event_stream(job),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _rollback_media_job_rows(
    *,
    user_id: str,
    assistant_id: Optional[str],
    item_job_ids: Optional[List[str]] = None,
    master_job_id: Optional[str] = None,
    namespace_filenames: Optional[List[str]] = None,
) -> int:
    """Best-effort delete of store rows a cancelled job/item already indexed.

    Documents are stamped with ``master_job_id`` / ``item_job_id`` at conversion
    time (see convert_media_list_to_text_document), so a cancel deletes exactly the
    rows that run wrote — including expanded playlist/linktree children whose
    ``namespace_filename`` differs from the top-level item. ``namespace_filenames``
    is an extra fallback for the top-level item key. Store rows removed CASCADE the
    matching store_vectors embeddings. Returns the number of rows deleted; never
    raises — rollback is best-effort ("attempt to delete").
    """
    if not assistant_id:
        return 0
    prefix_like = f"{user_id}.{assistant_id}.%"
    pool = app.state.pool
    total_deleted = 0
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                if master_job_id:
                    await cur.execute(
                        "DELETE FROM store WHERE prefix LIKE %s AND "
                        "value #>> '{document,kwargs,metadata,master_job_id}' = %s",
                        (prefix_like, master_job_id),
                    )
                    total_deleted += cur.rowcount or 0
                for item_job_id in item_job_ids or []:
                    await cur.execute(
                        "DELETE FROM store WHERE prefix LIKE %s AND "
                        "value #>> '{document,kwargs,metadata,item_job_id}' = %s",
                        (prefix_like, item_job_id),
                    )
                    total_deleted += cur.rowcount or 0
                for namespace_filename in namespace_filenames or []:
                    if not namespace_filename:
                        continue
                    await cur.execute(
                        "DELETE FROM store WHERE prefix LIKE %s AND "
                        "value #>> '{document,kwargs,metadata,namespace_filename}' = %s",
                        (prefix_like, namespace_filename),
                    )
                    total_deleted += cur.rowcount or 0
    except Exception as exc:  # noqa: BLE001 - rollback is best-effort
        logger.warning("Rollback for cancelled media job failed: %s", exc)
    return total_deleted


@app.post("/media_job/{job_id}/cancel")
async def cancel_media_job(
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Cancel a background media job and roll back what it already wrote.

    Send a **child** ``job_id`` to cancel one document: its processing stops and any
    store rows it already indexed are deleted (rolled back), leaving the rest of the
    batch running. Send the **master** ``job_id`` to do the same for the whole batch.
    Rollback is best-effort — typically a cancel lands before indexing completes, so
    there may be nothing to delete.
    """
    user_id = current_user["identities"][0]["user_id"]
    registry = app.state.media_jobs
    job: Optional[MediaJob] = get_job(registry, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown or expired job_id")
    if job.user_id != user_id:
        raise HTTPException(status_code=403, detail="This job belongs to another user.")

    # Flag + cancel the running task(s); returns the affected child jobs.
    targets = request_cancel(registry, job)

    deleted = await _rollback_media_job_rows(
        user_id=user_id,
        assistant_id=job.assistant_id,
        master_job_id=job.job_id if job.is_master else None,
        item_job_ids=[c.job_id for c in targets],
        namespace_filenames=[c.namespace_filename for c in targets],
    )

    return JSONResponse(
        status_code=200,
        content={
            "job_id": job.job_id,
            "scope": "batch" if job.is_master else "item",
            "status": "cancelled",
            "cancelled_items": [
                {"job_id": c.job_id, "filename": c.filename} for c in targets
            ],
            "rows_rolled_back": deleted,
            "message": (
                "Batch cancelled; processing stopped and indexed rows rolled back."
                if job.is_master
                else "Item cancelled; processing stopped and indexed rows rolled back."
            ),
        },
    )


@app.get("/list_avatar_documents")
async def list_avatar_documents(current_user: dict = Depends(get_current_user)):
    user_id = current_user["identities"][0]["user_id"]
    assistant_id = (
        current_user["app_metadata"]
        .get("assistant_config", {})
        .get("configurable", {})
        .get("assistant_id", None)
    )
    if assistant_id is None:
        raise HTTPException(
            detail="Please select an avatar before continuing.", status_code=400
        )

    # Read the avatar's store namespace in-process via app.state.store rather than
    # the LangGraph SDK HTTP client. The HTTP round-trip ConnectTimeouts while a
    # long media job is occupying the API process; this same in-process path is
    # what the upload endpoint's dedup uses.
    store = app.state.store
    try:
        all_document_items = await store.asearch(
            (user_id, assistant_id), limit=1_000_000
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read this avatar's documents: {exc}",
        ) from exc

    # Each source produces several Documents (quote / identity / analysis); the
    # set de-dupes them down to one entry per source. Playlist videos are listed
    # as ``{playlist} :: {video}`` and everything else by plain filename — see
    # _document_label_and_key, shared with /delete_avatar_document so a label
    # copied out of this list resolves back to the key delete needs.
    uploaded_documents: set[str] = {
        label for label, _key in _iter_document_labels(all_document_items)
    }

    return {"uploaded_documents": sorted(uploaded_documents)}


@app.delete("/delete_avatar_document")
async def delete_avatar_documents(
    source_document_name: str, current_user: dict = Depends(get_current_user)
):

    # Strip wrappers from copied SQL tuple/list output, e.g. ('Mom.m4a',) or "Mom.m4a",
    # leaving only the filename or already-derived namespace id.
    source_document_name = source_document_name.strip(" \t\n\r\"'`(),[]")
    # Keep the user-facing name for the response; source_document_name itself may
    # be rewritten below into an opaque hashed/composite store key.
    display_name = source_document_name
    user_id = current_user["identities"][0]["user_id"]
    assistant_id = (
        current_user["app_metadata"]
        .get("assistant_config", {})
        .get("configurable", {})
        .get("assistant_id", None)
    )
    if assistant_id is None:
        raise HTTPException(
            detail="Please select an avatar before continuing.", status_code=400
        )

    # Users delete by pasting a string straight out of /list_avatar_documents.
    # For a plain file that string IS the stored key (filename), but a playlist
    # video is listed by the human-readable ``{playlist_title} :: {video_title}``
    # label, whose words never equal the uuid5-hashed
    # ``{playlist_ns}::{video_ns}`` namespace_filename it's stored under — so the
    # raw label matches no row and delete 404s. Resolve the label back to its key
    # via the same helper /list builds labels with, so the two round-trip. Only
    # when nothing matches do we treat the input as a filename/URL and hash it
    # (this also avoids mangling a label that happens to contain a ".").
    try:
        existing_items = await app.state.store.asearch(
            (user_id, assistant_id), limit=1_000_000
        )
        label_to_key = {
            label: key for label, key in _iter_document_labels(existing_items) if key
        }
    except Exception:
        label_to_key = {}
    resolved_key = label_to_key.get(source_document_name)
    if resolved_key:
        source_document_name = resolved_key
    elif "." in source_document_name:
        source_document_name = _namespace_safe_formatted_filename(source_document_name)

    pool = app.state.pool

    # LangGraph store: prefix = namespace tuple dot-joined.
    # Match either chunk keys built from the filename prefix, or reference_* namespaces
    # (reference_image, reference_audio, …) where the serialized LangChain Document holds the
    # basename under value.document.kwargs.metadata.filename (same path as list_documents).
    # Rows removed from store CASCADE-delete matching store_vectors embeddings.
    # Playlist videos are keyed by a single opaque namespace_filename (a uuid5 over
    # ``{playlist_ns}::{video_ns}``) and carry their playlist's id under
    # value.document.kwargs.metadata.playlist_namespace_filename. Passing a bare
    # playlist namespace id (or its URL, hashed above) deletes the WHOLE playlist
    # via the playlist_namespace_filename value-match clause; passing a single
    # video's namespace_filename (resolved from its list label above) deletes that
    # one video via the prefix clauses. A plain (non-playlist) id matches no
    # playlist_namespace_filename, so that clause is inert for it.
    SQL_DELETE_DOCUMENT_QUERY = """
DELETE FROM store
WHERE (
    prefix = %s
    OR prefix LIKE %s
    OR prefix LIKE %s
    OR prefix LIKE %s
)
OR (
    prefix LIKE %s
    AND value #>> '{document,kwargs,metadata,playlist_namespace_filename}' = %s
)
OR (
    prefix LIKE %s
    AND value #>> '{document,kwargs,metadata,namespace_filename}' = %s
)
RETURNING value #>> '{document,kwargs,metadata,document_id}' AS document_id
"""
    total_deleted = 0
    # document_ids of the rows just deleted; used below to prune the stylometric
    # "direct quote" feature corpus so it no longer reflects removed documents.
    deleted_document_ids: set[str] = set()
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                params = (
                    f"{user_id}.{assistant_id}.{source_document_name}",
                    f"{user_id}.{assistant_id}.{source_document_name}.%",
                    f"{user_id}.{assistant_id}.%.{source_document_name}",
                    f"{user_id}.{assistant_id}.%.{source_document_name}.%",
                    # Whole playlist: every video whose playlist_namespace_filename
                    # equals this id. Scoped to this user/assistant via the prefix —
                    # playlist_ns is a deterministic hash of the playlist URL and is
                    # therefore shared across users who uploaded the same playlist,
                    # so an unscoped value match would cross avatars.
                    f"{user_id}.{assistant_id}.%",
                    source_document_name,
                    f"{user_id}.{assistant_id}.reference_%",
                    source_document_name,
                )
                await cur.execute(SQL_DELETE_DOCUMENT_QUERY, params)
                returned_rows = await cur.fetchall()
                total_deleted += len(returned_rows)
                deleted_document_ids = {
                    row[0] for row in returned_rows if row and row[0]
                }
    except Exception:
        raise HTTPException(detail="Error deleting documents.", status_code=500)

    if total_deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No stored rows matched document: {display_name}",
        )

    # The raw SQL above can remove the avatar's reference image (the
    # ``reference_%`` prefix clause) without going through the store client, so
    # drop the reference-image entry from the load_consciousness read-through
    # cache. Unconditional because the deleted rows are not inspected per
    # namespace; the invalidation is a dictionary pop either way.
    invalidate_store_cache_entry(
        (user_id, assistant_id, "reference_image"), assistant_id
    )

    # Prune the deleted documents' rows from the stylometric "direct quote"
    # feature corpus (a {document_id: [len(FEATURE_NAMES) floats]} dict in the store), then
    # recalibrate the empirical threshold + IsolationForest from what remains.
    # Best-effort: a failure here must not fail the (already committed) delete.
    try:
        await _prune_ground_truth_features_for_deleted_docs(
            user_id, assistant_id, deleted_document_ids
        )
    except Exception as exc:  # pragma: no cover - operator log only
        logger.warning(
            "ground-truth feature prune failed for %s: %s", assistant_id, exc
        )

    return JSONResponse(
        content=f"Successfully deleted: {display_name}", status_code=200
    )


async def _prune_ground_truth_features_for_deleted_docs(
    user_id: str | None, assistant_id: str | None, deleted_document_ids: set[str]
) -> None:
    """Remove deleted documents from the per-document stylometric feature corpus.

    Reads the ``{document_id: [len(FEATURE_NAMES) floats]}`` dict the avatar's "direct quote"
    corpus is stored under, drops every ``deleted_document_ids`` entry, then:

    * if rows remain — rebuilds the ``(n_docs, len(FEATURE_NAMES))`` array and recalibrates the
      empirical threshold + IsolationForest from it, persisting all the derived
      artifacts (dict, threshold, model, style profile);
    * if the corpus is now empty — deletes those keys so a later re-upload
      starts clean.

    All artifacts live under the owner-scoped ``(user_id, assistant_id,
    <artifact_name>)`` namespaces that ``calibrate_ground_truth`` writes.
    No-op when there is no user/assistant or nothing was deleted.
    """
    if not user_id or not assistant_id or not deleted_document_ids:
        return

    from src.anubis.utils.dataset.style_features import (
        GROUND_TRUTH_FEATURES_DICT_KEY,
        deserialize_features_by_doc_id,
        features_by_doc_id_to_arr,
        recompute_ground_truth_artifacts,
        serialize_features_by_doc_id,
    )

    store = app.state.store

    dict_namespace = (user_id, assistant_id, GROUND_TRUTH_FEATURES_DICT_KEY)
    threshold_namespace = (user_id, assistant_id, "ground_truth_text_empirical_threshold_list_str")
    model_namespace = (user_id, assistant_id, "ground_truth_text_features_model_b64_pkl")
    style_profile_namespace = (user_id, assistant_id, "style_profile")

    item = await store.aget(dict_namespace, key=GROUND_TRUTH_FEATURES_DICT_KEY)
    features_by_doc_id_str = (getattr(item, "value", None) or {}).get("value", None)
    features_by_doc_id = deserialize_features_by_doc_id(features_by_doc_id_str)
    if not features_by_doc_id:
        return

    # Drop the deleted documents' rows.
    removed = False
    for document_id in deleted_document_ids:
        if features_by_doc_id.pop(document_id, None) is not None:
            removed = True
    if not removed:
        return

    if not features_by_doc_id:
        # Corpus is now empty: clear the derived keys.
        await store.adelete(dict_namespace, key=GROUND_TRUTH_FEATURES_DICT_KEY)
        await store.adelete(threshold_namespace, key="ground_truth_text_empirical_threshold_list_str")
        await store.adelete(model_namespace, key="ground_truth_text_features_model_b64_pkl")
        await store.adelete(style_profile_namespace, key="style_profile")
        return

    # Rebuild the corpus array and recalibrate the derived artifacts.
    ground_truth_text_features_arr = features_by_doc_id_to_arr(features_by_doc_id)
    threshold_list_str, model_b64_pkl = recompute_ground_truth_artifacts(
        ground_truth_text_features_arr
    )

    from src.anubis.utils.dataset.style_features import build_style_profile_str
    style_profile_str = await build_style_profile_str(ground_truth_text_features_arr)

    await store.aput(
        dict_namespace,
        key=GROUND_TRUTH_FEATURES_DICT_KEY,
        value={"value": serialize_features_by_doc_id(features_by_doc_id)},
    )
    await store.aput(
        threshold_namespace,
        key="ground_truth_text_empirical_threshold_list_str",
        value={"value": threshold_list_str},
    )
    await store.aput(
        model_namespace,
        key="ground_truth_text_features_model_b64_pkl",
        value={"value": model_b64_pkl},
    )
    await store.aput(
        style_profile_namespace,
        key="style_profile",
        value={"value": style_profile_str},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
