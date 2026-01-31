# src/anubis/graph.py

"""
src/anubis/graph.py
Super-Graph with a central Langchain Agent and subgraph tool use.
"""

import logging
logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, START, END

# from src.subgraphs.conversational_memory_graph.graph import agent_graph

from src.anubis.utils.state import GlobalMessageState
from src.anubis.utils.context import GlobalContext, UserContext, AssistantContext
from src.anubis.utils.configuration import GlobalConfiguration

from langchain.messages import SystemMessage, AIMessage, HumanMessage
from langchain.agents import create_agent

from src.anubis.utils.model import init_model

import logging
logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate

from src.anubis.utils.state import GlobalMessageState
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.model import init_model  # Your model init with env vars
# from src.anubis.utils.tools import your_tools  # Add your tools here
tools = []  # Replace with your tools

from src.anubis.utils.nodes import agent_node


# # Create context (configuration loads env vars)
# ctx = GlobalContext(configuration=GlobalConfiguration())

# # Create the agent runtime (single node)
# agent = create_agent(
#     llm=None,  # Not used; custom node handles model init
#     tools=tools,
#     state_schema=GlobalMessageState,  # Your message state
#     context_schema=GlobalContext,     # Your context for runtime deps
#     # system_prompt not needed; use in node/middleware
# )

# Build minimal graph: START -> agent -> END
workflow = StateGraph(state_schema = GlobalMessageState, context_schema = GlobalContext)

# Add single node (your input/output)
workflow.add_node("agent", agent_node)

# Edges
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

graph = workflow.compile()
graph.name = "Anubis"

"""New LangGraph Agent.

This module defines a custom graph.
"""

from src.anubis.graph import graph

__all__ = ["graph"]
