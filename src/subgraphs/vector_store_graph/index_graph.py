"""This "graph" simply exposes an endpoint for a user to upload docs to be indexed."""

from typing import Sequence

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from src.subgraphs.vector_store_graph.utils import retrieval

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.state import GlobalState


from langgraph_runtime.store import BaseStore

from typing import cast

from langgraph.store.memory import InMemoryStore
from src.anubis.utils.configuration import GlobalConfiguration

across_thread_memory = InMemoryStore()
configuration = GlobalConfiguration()

def ensure_docs_have_user_id(
    vectorstore_documents_to_be_indexed: Sequence[Document], runtime: GlobalContext
) -> list[Document]:
    """Ensure that all documents have a user_id in their metadata.

        vectorstore_documents_to_be_indexed (Sequence[Document]): A sequence of Document objects to process.
        runtime (GlobalContext): A context object containing the user_id.

    Returns:
        list[Document]: A new list of Document objects with updated metadata.
    """
    user_id = runtime.context.user_ctx.user_id
    return [
        Document(
            page_content=doc.page_content, metadata={**doc.metadata, "user_id": user_id}
        )
        for doc in vectorstore_documents_to_be_indexed
    ]

import logging
logger = logging.getLogger(__name__)

import asyncio

from src.subgraphs.vector_store_graph.utils.retrieval import make_pg_vector

async def index_docs(
    state: GlobalState, runtime: Runtime[GlobalContext], store: BaseStore | None = None
) -> dict[str, str]:
    """Asynchronously index documents in the given state using the configured retriever.

    This function takes the documents from the state, ensures they have a user ID,
    adds them to the retriever's index, and then signals for the documents to be
    deleted from the state.

    Args:
        state (IndexState): The current state containing documents and retriever.
        config (Optional[RunnableConfig]): Configuration for the indexing process.r
    """
    from src.anubis.utils.configuration import GlobalConfiguration
    logger.info(f"index docs entrypoint")
    
    configuration = GlobalConfiguration()

    vector_store = make_pg_vector(configuration)

    with vector_store as v_store:
        logger.info(f"INDEXING DOCUMENTS")

        # Delete documents with the same filename in the metadata
        filenames = {
            doc.metadata.get("filename") 
            for doc in 
            state['vectorstore_documents_to_be_indexed'] 
            if doc.metadata.get("filename") is not None
        }
        if filenames:
            delete_value = await asyncio.to_thread(
                v_store.adelete({"filename": {"$in": list(filenames)}})
            )
             
            logger.info(f"delete_value: {delete_value}")
        
        # Upload the new documents into the vector store
        await v_store.aadd_documents(
            state['vectorstore_documents_to_be_indexed']
        )
        return {"docs": "delete"}



# Define a new graph
builder = StateGraph(GlobalState, context_schema=GlobalContext)

builder.add_node(index_docs)
builder.add_edge("__start__", "index_docs")

# if (configuration.dev == "TRUE"):
    # index_graph = builder.compile(store=across_thread_memory)
# else: 
index_graph = builder.compile()

index_graph.name = "IndexGraph"

__all__ = ["index_graph"]
