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
    convert_media_list_to_text_document,
)

from src.subgraphs.vector_store_graph.index_graph import index_docs
from src.subgraphs.process_media_graph.utils.nodes import process_uploaded_files_and_label_media_type
from src.anubis.utils.configuration import GlobalConfiguration

configuration = GlobalConfiguration()


from src.subgraphs.vector_store_graph.utils.retrieval import make_pg_store

def create_process_media_graph(store=None):
    # Define the Graph & Context
    workflow = StateGraph(
        state_schema=GlobalState, 
        context_schema=GlobalContext
    )

    # Add Nodes
    workflow.add_node("process_uploaded_files", process_uploaded_files_and_label_media_type)
    workflow.add_node("convert_media_list_to_text_document", convert_media_list_to_text_document)
    workflow.add_node("index_docs", index_docs)

    # Define Edges
    workflow.add_edge(START, "process_uploaded_files")
    workflow.add_edge("process_uploaded_files", "convert_media_list_to_text_document")
    workflow.add_edge("convert_media_list_to_text_document", "index_docs")

    process_media_graph_api_endpoint = workflow.compile(store=store)

    process_media_graph_api_endpoint.name = "process_media_graph_api_endpoint"
    return process_media_graph_api_endpoint



# Define the Graph & Context
workflow = StateGraph(
    state_schema=GlobalState, 
    context_schema=GlobalContext
)
# Add Nodes
workflow.add_node("process_uploaded_files", process_uploaded_files_and_label_media_type)
workflow.add_node("convert_media_list_to_text_document", convert_media_list_to_text_document)
workflow.add_node("index_docs", index_docs)
# Define Edges
workflow.add_edge(START, "process_uploaded_files")
workflow.add_edge("process_uploaded_files", "convert_media_list_to_text_document")
workflow.add_edge("convert_media_list_to_text_document", "index_docs")




process_media_graph_api_endpoint = workflow.compile(store=(make_pg_store()))
process_media_graph_api_endpoint.name = "process_media_graph_api_endpoint"

__all__ = ["process_media_graph_api_endpoint"]