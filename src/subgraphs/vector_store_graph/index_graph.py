"""This "graph" simply exposes an endpoint for a user to upload docs to be indexed."""

from typing import Sequence

from langchain_core.documents import Document
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.state import GlobalState

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

across_thread_memory = InMemoryStore()


import logging
logger = logging.getLogger(__name__)

from src.subgraphs.vector_store_graph.utils.helper_functions import batch_index_documents_vectorstore
from langchain_core.runnables import RunnableConfig
from src.anubis.utils.utility import extract_user_id_assistant_id


def ensure_docs_have_user_id(
    vectorstore_documents_to_be_indexed: Sequence[Document], runtime: GlobalContext
) -> list[Document]:
    """Ensure that all documents have a user_id in their metadata.

        vectorstore_documents_to_be_indexed (Sequence[Document]): A sequence of Document objects to process.
        runtime (GlobalContext): A context object containing the user_id.

    Returns:
        list[Document]: A new list of Document objects with updated metadata.
    """
    
    if isinstance(runtime.context.user_ctx, dict):
        user_id = runtime.context.user_ctx.get("user_id", "")
    else:
        user_id = getattr(runtime.context.user_ctx, "user_id", "")

    return [
        Document(
            page_content=doc.page_content, metadata={**doc.metadata, "user_id": user_id}
        )
        for doc in vectorstore_documents_to_be_indexed
    ]


async def index_docs(
    state: GlobalState, runtime: Runtime[GlobalContext], config: RunnableConfig,
    store: BaseStore
) -> dict[str, str]:
    """Asynchronously index documents in the given state using the configured retriever.

    This function takes the documents from the state, ensures they have a user ID,
    adds them to the retriever's index, and then signals for the documents to be
    deleted from the state.

    Args:
        state (IndexState): The current state containing documents and retriever.
        config (Optional[RunnableConfig]): Configuration for the indexing process.r
    """
    logger.info(f"INDEXING DOCUMENTS")

    updated_user_id, updated_assistant_id = await extract_user_id_assistant_id(config)

    user_id = updated_user_id['user_id']
    assistant_id = updated_assistant_id['assistant_id']

    docs = state['vectorstore_documents_to_be_indexed']

    filenames = [doc.metadata.get("filename") for doc in docs]
    try:
        assert(len(filenames) == len(docs))
    except AssertionError as e:
        logger.warning(f"Missing {len(docs) - len(filenames)} filenames on documents")

    if len(filenames) > 0:
        result = await batch_index_documents_vectorstore(store, user_id, assistant_id, docs, BATCH_SIZE=1000)

        try:
            assert(result['success'] == True)
        except AssertionError as e:
            logger.error(f"Error uploading documents: {result['error_batch_documents']}")

            # Clear the documents to be indexed on error
            state['vectorstore_documents_to_be_indexed'] = []            
            raise Exception(f"Error uploading documents: {result['error_batch_documents']}")

        logger.info(f"breakpoint after batch_index_documents_vectorstore")

    return {"vectorstore_documents_to_be_indexed": "delete"}

# Define a new graph
builder = StateGraph(GlobalState, context_schema=GlobalContext)

builder.add_node(index_docs)
builder.add_edge("__start__", "index_docs")

index_graph = builder.compile()

index_graph.name = "IndexGraph"

__all__ = ["index_graph"]
