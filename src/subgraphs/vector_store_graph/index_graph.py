"""This "graph" simply exposes an endpoint for a user to upload docs to be indexed."""

from typing import Sequence

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from src.subgraphs.vector_store_graph.utils import retrieval
from src.anubis.utils.configuration import IndexConfiguration
from src.subgraphs.vector_store_graph.utils.state import IndexState

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.state import GlobalState


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

async def index_docs(
    state: IndexState, runtime: Runtime[GlobalContext] | None = None
) -> dict[str, str]:
    """Asynchronously index documents in the given state using the configured retriever.

    This function takes the documents from the state, ensures they have a user ID,
    adds them to the retriever's index, and then signals for the documents to be
    deleted from the state.

    Args:
        state (IndexState): The current state containing documents and retriever.
        config (Optional[RunnableConfig]): Configuration for the indexing process.r
    """

    configuration = runtime.context.configuration
    with retrieval.make_retriever(configuration) as retriever:
        logger.info(f"INDEXING DOCUMENTS")
        stamped_docs = ensure_docs_have_user_id(state.vectorstore_documents_to_be_indexed, runtime)
        logger.info(f"stamped_docs: {stamped_docs}")
        await retriever.aadd_documents(state.vectorstore_documents_to_be_indexed)
    return {"docs": "delete"}


# Define a new graph
builder = StateGraph(IndexState, context_schema=GlobalContext)

builder.add_node(index_docs)
builder.add_edge("__start__", "index_docs")

index_graph = builder.compile()
index_graph.name = "IndexGraph"

__all__ = ["index_graph"]