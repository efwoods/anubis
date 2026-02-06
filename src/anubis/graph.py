# src/anubis/graph.py

"""
src/anubis/graph.py
Super-Graph with a central Langchain Agent and subgraph tool use.
"""

import logging
logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, START, END

# from src.subgraphs.conversational_memory_graph.graph import agent_graph

from src.anubis.utils.state import GlobalState
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

from src.anubis.utils.state import GlobalState
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.nodes import (
    invoke_agent
)
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver

from dotenv import load_dotenv
load_dotenv()

configuration = GlobalConfiguration()

from typing import cast

from langgraph.store.base import BaseStore

# Build minimal graph: START -> agent -> END
workflow = StateGraph(state_schema = GlobalState, context_schema = GlobalContext)

# Add single node (your input/output)
# workflow.add_node("call_router", call_router)
workflow.add_node("invoke_agent", invoke_agent)

# Edges
# workflow.add_edge(START, "invoke_agent")
workflow.add_edge(START, "invoke_agent")

workflow.add_edge("invoke_agent", END)

graph = workflow.compile()
graph.name = "Anubis"

__all__ = ["graph"]
