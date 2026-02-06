# src/subgraphs/process_media_graph/graph.py

"""
Given a list of messages: 
Extracts zero or more media from the most recent message.
Determine the media type.
Convert the media into text.
Creates a Document object from the text.
Returns a List of Document Objects for further processing in other subgraphs.
"""

from langgraph.graph import StateGraph, START, END
from src.anubis.utils.state import GlobalState
from src.anubis.utils.context import GlobalContext

from src.subgraphs.process_media_graph.utils.nodes import (
    extract_media_from_message, 
    convert_media_list_to_text_document,
)

from src.subgraphs.vector_store_graph.index_graph import index_docs

from src.subgraphs.process_media_graph.utils.nodes import process_uploaded_files

from langgraph.store.base import BaseStore

from langgraph.checkpoint.memory import MemorySaver
from typing import cast

from langgraph.store.memory import InMemoryStore

# Define the Graph & Context
workflow = StateGraph(
    state_schema=GlobalState, 
    context_schema=GlobalContext, 
    store=cast(BaseStore, InMemoryStore())
)

# Add Nodes
workflow.add_node("process_uploaded_files", process_uploaded_files)
workflow.add_node("convert_media_list_to_text_document", convert_media_list_to_text_document)
workflow.add_node("index_docs", index_docs)

# Define Edges
workflow.add_edge(START, "process_uploaded_files")
workflow.add_edge("process_uploaded_files", "convert_media_list_to_text_document")
workflow.add_edge("convert_media_list_to_text_document", "index_docs")


process_media_graph_api_endpoint = workflow.compile()
process_media_graph_api_endpoint.name = "process_media_graph_api_endpoint"

__all__ = ["process_media_graph_api_endpoint"]