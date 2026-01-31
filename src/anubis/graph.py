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
from src.anubis.utils.nodes import invoke_model


# Build minimal graph: START -> agent -> END
workflow = StateGraph(state_schema = GlobalMessageState, context_schema = GlobalContext)

# Add single node (your input/output)
workflow.add_node("invoke_model", invoke_model)

# Edges
workflow.add_edge(START, "invoke_model")
workflow.add_edge("invoke_model", END)

graph = workflow.compile()
graph.name = "Anubis"

from src.anubis.graph import graph

__all__ = ["graph"]
