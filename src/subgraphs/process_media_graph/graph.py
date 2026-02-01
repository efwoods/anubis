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
    determine_media_type,
    convert_media_to_text,
)

# Define the Graph & Context
workflow = StateGraph(state_schema=GlobalState, context_schema=GlobalContext)

# Add Nodes
workflow.add_node("extract_media_from_message", extract_media_from_message)
workflow.add_node("determine_media_type", determine_media_type)
workflow.add_node("convert_media_to_text", convert_media_to_text)

# Define Edges
workflow.add_edge(START, "extract_media_from_message")
workflow.add_edge("extract_media_from_message", "determine_media_type")
workflow.add_edge("determine_media_type", "convert_media_to_text")

process_media_graph = workflow.compile()
process_media_graph.name = "process_media_graph"

__all__ = ["process_media_graph"]