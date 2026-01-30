"""
src/anubis/graph.py
Super-Graph with a central Langchain Agent and subgraph tool use.
"""

import logging
logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, START, END

from src.subgraphs.agent import graph as agent_graph

from src.anubis.utils.state import AnubisState


# Build graph
workflow = StateGraph(state_schema=AnubisState)

# Define Nodes
workflow.add_node("agent", agent_graph)


# Entrypoint of graph
# workflow.set_entry_point("agent") # agent starts immediately

# Edge Definitions
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

graph = workflow.compile()

graph.name = "Anubis"