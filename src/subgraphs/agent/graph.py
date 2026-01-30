"""
src/anubis/graph.py
Super-Graph with a central Langchain Agent and subgraph tool use.
"""

import logging

logger = logging.getLogger(__name__)

from typing import List, Callable, Any

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from src.anubis.utils.state import AnubisState
from src.subgraphs.agent.utils.nodes import model_node, continue_tool_use_conditional

from src.subgraphs.agent.utils.tools import (
    search, 
    get_chat_metadata, 
    health_check, 
    add_to_vectorstore
)

tools = [
    search, 
    get_chat_metadata, 
    health_check, 
    add_to_vectorstore
]

# Build graph
workflow = StateGraph(state_schema=AnubisState)

# Define Nodes
workflow.add_node("model", model_node)
workflow.add_node("tools", ToolNode(tools)) # tool use is parallel

# Entrypoint of graph
workflow.set_entry_point("model")

# Edge Definitions
workflow.add_conditional_edges("model", continue_tool_use_conditional)
workflow.add_edge("tools", "model")
agent_graph = workflow.compile()
agent_graph.name = "AgentGraph"
