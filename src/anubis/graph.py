"""
src/anubis/graph.py
Super-Graph with a central Langchain Agent and subgraph tool use.
"""

import logging

logger = logging.getLogger(__name__)


from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from src.anubis.utils.state import AgentState
from src.anubis.utils.nodes import agent_node, continue_tool_use_conditional

tools = []

# Build graph
workflow = StateGraph(state_schema=AgentState)

# Define Nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools)) # tool use is parallel

# Entrypoint of graph
workflow.set_entry_point("agent")

# Edge Definitions
workflow.add_conditional_edges("agent", continue_tool_use_conditional)
workflow.add_edge("tools", "agent")
graph = workflow.compile()
graph.name = "Anubis"