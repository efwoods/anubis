"""
src/anubis/graph.py
Super-Graph with a central Langchain Agent and subgraph tool use.
"""

import logging

logger = logging.getLogger(__name__)

from typing import List, Callable, Any

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from src.subgraphs.agent.utils.nodes import model_node, continue_tool_use_conditional

from src.anubis.utils.state import GlobalMessageState
from src.anubis.utils.context import GlobalContext

from src.anubis.utils.tools import (
    search, 
    health_check, 
    add_to_vectorstore, 
    retrieve_from_vectorstore
)

tools = [
    search, 
    health_check, 
    add_to_vectorstore, 
    retrieve_from_vectorstore
]

# Build graph
# workflow = StateGraph(state_schema=GlobalMessageState)



# # Define Nodes
# workflow.add_node("model", model_node)
# workflow.add_node("tools", ToolNode(tools)) # tool use is parallel

# # Entrypoint of graph
# workflow.set_entry_point("model")

# # Edge Definitions
# workflow.add_conditional_edges("model", continue_tool_use_conditional)
# workflow.add_edge("tools", "model")
# agent_graph = workflow.compile()
# agent_graph.name = "AgentGraph"

# __all__ = ["agent_graph"]


"""Graphs that extract memories on a schedule."""
import logging

logger = logging.getLogger(__name__)

from src.subgraphs.memory_store_graph.utils.nodes import (
    call_model, 
    store_memory, 
    route_message
)

from langgraph.graph import END, StateGraph
from src.subgraphs.memory_store_graph.utils.context import Context
from src.subgraphs.memory_store_graph.utils.state import State

# Create the graph + all nodes
builder = StateGraph(state_schema=GlobalMessageState, context_schema=GlobalContextSchema)

# Define the flow of the memory extraction process
builder.add_node(call_model)
builder.add_node(store_memory)
builder.add_node("tools", ToolNode(tools)) # tool use is parallel

builder.add_edge("__start__", "call_model")
builder.add_edge("store_memory", "call_model")
builder.add_edge("tools", "call_model")

builder.add_conditional_edges("call_model", route_message, ["store_memory", END])
builder.add_conditional_edges("call_model", "tools", route_message, ["tools", END])

# Right now, we're returning control to the user after storing a memory
# Depending on the model, you may want to route back to the model
# to let it first store memories, then generate a response

memory_store_graph = builder.compile()
memory_store_graph.name = "agent_graph"

__all__ = ["agent_store"]