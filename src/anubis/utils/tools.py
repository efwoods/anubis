""" Agent SubGraph Tools """
import uuid
import logging
from typing import List, Annotated

from langchain.tools import tool, ToolRuntime
from langchain_core.documents import Document
from langchain_core.tools import InjectedToolArg
from langchain.messages import AIMessage, HumanMessage

from langgraph.store.base import BaseStore

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.state import GlobalState

logger = logging.getLogger(__name__)

from datetime import datetime, timezone

@tool
async def health_check(runtime: ToolRuntime[GlobalContext]):
    """Call this health check to determine if tools can be called at all.

    Args:
        runtime (ToolRuntime[GlobalContext]): tool runtime of the current state and context 

    Returns:
        _type_: _description_
    """
    return {"messages": AIMessage(content="success")}


@tool
async def upsert_memory(
    content: str,
    context: str,
    *,
    memory_id: uuid.UUID | None = None,
    # Hide these arguments from the model.
    user_id: Annotated[str, InjectedToolArg],
    store: Annotated[BaseStore, InjectedToolArg],
):
    """Upsert a memory in the database.

    If a memory conflicts with an existing one, then just UPDATE the
    existing one by passing in memory_id - don't create two memories
    that are the same. If the user corrects a memory, UPDATE it.

    Args:
        content: The main content of the memory. For example:
            "User expressed interest in learning about French."
        context: Additional context for the memory. For example:
            "This was mentioned while discussing career options in Europe."
        memory_id: ONLY PROVIDE IF UPDATING AN EXISTING MEMORY.
        The memory to overwrite.
    """
    mem_id = memory_id or uuid.uuid4()
    await store.aput(
        ("memories", user_id),
        key=str(mem_id),
        value={"content": content, "context": context},
    )
    return f"Stored memory {mem_id}"


""" Vector Store SubGraph Tools """
""" Process Media SubGraph Tools """

# src/anubis/utils/store.py
import contextlib
from typing import cast
from langchain.embeddings import init_embeddings, Embeddings
from langgraph.store.base import IndexConfig
from langgraph.store.postgres import AsyncPostgresStore
from src.anubis.utils.configuration import GlobalConfiguration
from src.subgraphs.vector_store_graph.utils.retrieval import make_text_encoder
from langchain_huggingface import HuggingFaceEmbeddings

# embeddings = cast(Embeddings, init_embeddings("openai:text-embedding-3-small"))

@contextlib.asynccontextmanager
async def generate_store():
    """Yield a BaseStore, open for the duration of the server."""
    configuration = GlobalConfiguration()
    embeddings = HuggingFaceEmbeddings(model_name = configuration.embedding_model)

    async with AsyncPostgresStore.from_conn_string(
        conn_string=configuration.async_postgres_store_uri,
        index=IndexConfig(dims=384, embed=embeddings, fields=["$"])
    ) as store:
        await store.setup()
        yield store