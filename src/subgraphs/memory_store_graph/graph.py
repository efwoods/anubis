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
builder = StateGraph(State, context_schema=Context)

# Define the flow of the memory extraction process
builder.add_node(call_model)
builder.add_edge("__start__", "call_model")
builder.add_node(store_memory)
builder.add_conditional_edges("call_model", route_message, ["store_memory", END])
# Right now, we're returning control to the user after storing a memory
# Depending on the model, you may want to route back to the model
# to let it first store memories, then generate a response
builder.add_edge("store_memory", "call_model")
memory_store_graph = builder.compile()
memory_store_graph.name = "memory_store_graph"

__all__ = ["memory_store"]