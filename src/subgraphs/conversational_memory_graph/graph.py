# src/subgraphs/agent/graph.py

import logging

logger = logging.getLogger(__name__)

from typing import List, Callable, Any

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from langgraph.graph import END, StateGraph

from src.anubis.utils.state import GlobalMessageState
from src.anubis.utils.context import GlobalContext
 
from src.subgraphs.conversational_memory_graph.utils.nodes import (
    call_model, 
    store_memory, 
    route_message
)

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


# Create the graph + all nodes
builder = StateGraph(state_schema=GlobalMessageState, context_schema=GlobalContext)

# Define the flow of the memory extraction process
builder.add_node(call_model)
builder.add_node(store_memory)
# builder.add_node("tools", ToolNode(tools)) # tool use is parallel

builder.add_edge("__start__", "call_model")
builder.add_edge("store_memory", "call_model")
# builder.add_edge("tools", "call_model")

builder.add_conditional_edges("call_model", route_message, ["store_memory", END])
# builder.add_conditional_edges("call_model", "tools", route_message, ["tools", END])

# Right now, we're returning control to the user after storing a memory
# Depending on the model, you may want to route back to the model
# to let it first store memories, then generate a response

conversational_memory_graph = builder.compile()
conversational_memory_graph.name = "agent_graph"

__all__ = ["agent_store"]