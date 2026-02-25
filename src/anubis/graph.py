# src/anubis/graph.py

"""
src/anubis/graph.py
Super-Graph with a central Langchain Agent and subgraph tool use.
"""

import logging
logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, START, END
from src.anubis.utils.state import GlobalState
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.configuration import GlobalConfiguration
from src.anubis.utils.nodes import (
    invoke_agent
)

from src.subgraphs.vector_store_graph.retrieval_graph import retrieval_graph

from src.subgraphs.vector_store_graph.utils.retrieval import make_pg_store
from dotenv import load_dotenv
load_dotenv()

from src.anubis.utils.context import IdentityContext, AssistantContext
from langchain_core.runnables import RunnableConfig

configuration = GlobalConfiguration()

# Build minimal graph: START -> agent -> END
workflow = StateGraph(
    state_schema = GlobalState, 
    context_schema = GlobalContext
)

# Add single node (your input/output)
# workflow.add_node("call_router", call_router)
workflow.add_node("retrieve_documents", retrieval_graph)
# workflow.add_node("summarize_conversation", summarize_conversation)
workflow.add_node("invoke_agent", invoke_agent)

# Edges
workflow.add_edge(START, 'retrieve_documents')
workflow.add_edge('retrieve_documents', "invoke_agent")
workflow.add_edge("invoke_agent", END)
store = make_pg_store
graph = workflow.compile(store=store)

graph.name = "Anubis"

__all__ = ["graph"]
