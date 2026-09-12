# src/subgraphs/process_media_graph/graph.py

"""
Given a list of messages:
Extracts zero or more media from the most recent message.
Determine the media type.
Convert the media into text.
Creates a Document object from the text.
Returns a List of Document Objects for further processing in other subgraphs.
"""

from langgraph.graph import END, START, StateGraph

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.state import GlobalState
from src.subgraphs.moderation_graph.media_node import (
    moderate_documents,
    route_media_moderation,
)
from src.subgraphs.process_media_graph.utils.nodes import (
    analyze_documents,
    convert_media_list_to_text_document,
    process_adapter_documents,
    process_uploaded_files_and_label_media_type,
)
from src.subgraphs.psycho_analysis_graph.graph import psycho_analysis
from src.subgraphs.vector_store_graph.index_graph import index_docs

# Define the Graph & Context
workflow = StateGraph(state_schema=GlobalState, context_schema=GlobalContext)

# Add Nodes
workflow.add_node("process_uploaded_files", process_uploaded_files_and_label_media_type)
workflow.add_node(
    "convert_media_list_to_text_document", convert_media_list_to_text_document
)
workflow.add_node("moderate_documents", moderate_documents)
workflow.add_node("analyze_documents", analyze_documents)
workflow.add_node("psycho_analysis", psycho_analysis)
workflow.add_node("process_adapter_documents", process_adapter_documents)
workflow.add_node("index_docs", index_docs)

# Define Edges
workflow.add_edge(START, "process_uploaded_files")
workflow.add_edge("process_uploaded_files", "convert_media_list_to_text_document")

# AI monitoring: every converted document is judged BEFORE anything downstream
# reads it. This is the one place moderation gates rather than runs alongside the
# work, and it is a correctness requirement rather than a performance choice:
# content that violates the terms of service must never be indexed into the
# avatar's store, analyzed into its traits, or written into adapter training data,
# and all three of those consumers read these documents. An upload is already a
# background job that nobody is waiting on, so the gate costs no request latency.
workflow.add_edge("convert_media_list_to_text_document", "moderate_documents")
workflow.add_conditional_edges(
    "moderate_documents",
    route_media_moderation,
    [
        "index_docs",
        "analyze_documents",
        "process_adapter_documents",
        "psycho_analysis",
        END,
    ],
)

# Past the gate, four branches run concurrently off one fan-out. Analysis and
# psychological analysis both run before indexing so the trait Documents they
# produce (analysis, emotional_trigger namespaces) merge into the same
# vector-store index batch as the source documents.
workflow.add_edge("analyze_documents", "index_docs")
workflow.add_edge("psycho_analysis", "index_docs")
workflow.add_edge("process_adapter_documents", END)
workflow.add_edge("index_docs", END)

process_media_graph_api_endpoint = workflow.compile()
process_media_graph_api_endpoint.name = "process_media_graph_api_endpoint"

__all__ = ["process_media_graph_api_endpoint"]
