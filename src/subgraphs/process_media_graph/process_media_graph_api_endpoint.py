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
    analyze_documents,
    convert_media_list_to_text_document,
    build_stylistic_fingerprint,
    process_adapter_documents,
    process_uploaded_files_and_label_media_type,
)

from src.subgraphs.vector_store_graph.index_graph import index_docs

# Define the Graph & Context
workflow = StateGraph(
    state_schema=GlobalState, 
    context_schema=GlobalContext
)

# Add Nodes
workflow.add_node("process_uploaded_files", process_uploaded_files_and_label_media_type)
workflow.add_node("convert_media_list_to_text_document", convert_media_list_to_text_document)
workflow.add_node("analyze_documents", analyze_documents)
workflow.add_node("process_adapter_documents", process_adapter_documents)
workflow.add_node("index_docs", index_docs)
workflow.add_node("build_stylistic_fingerprint", build_stylistic_fingerprint)

# Define Edges
workflow.add_edge(START, "process_uploaded_files")
workflow.add_edge("process_uploaded_files", "convert_media_list_to_text_document")

# After classification: adapter rows and full-text analysis feed the vector store;
# analysis runs before indexing so trait Documents merge into the same index batch.
# workflow.add_edge("convert_media_list_to_text_document", "analyze_documents")
workflow.add_edge("convert_media_list_to_text_document", "index_docs")

# workflow.add_edge("convert_media_list_to_text_document", "process_adapter_documents")

# workflow.add_edge("analyze_documents", "index_docs")

# workflow.add_edge("index_docs", "build_stylistic_fingerprint")

# workflow.add_edge("process_adapter_documents", "build_stylistic_fingerprint")
# workflow.add_edge("build_stylistic_fingerprint", "analyze_documents")

# workflow.add_edge("build_stylistic_fingerprint", END)
workflow.add_edge("index_docs", END)

process_media_graph_api_endpoint = workflow.compile()
process_media_graph_api_endpoint.name = "process_media_graph_api_endpoint"

__all__ = ["process_media_graph_api_endpoint"]
