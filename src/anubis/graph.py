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

from dotenv import load_dotenv
load_dotenv()

from src.anubis.utils.context import IdentityContext, AssistantContext
from langchain_core.runnables import RunnableConfig

configuration = GlobalConfiguration()

# Define the context 
# def make_context(config: RunnableConfig) -> GlobalContext:
#     configurable = config.get("configurable", {})

#     user_id = configurable.get("user_id", "test_user_1234")
#     assistant_id = configurable.get("assistant_id", "Anubis")

#     return GlobalContext(
#         user_ctx=IdentityContext(user_id=user_id),
#         assistant_ctx=AssistantContext(assistant_id=assistant_id),
#         configuration=GlobalConfiguration.from_runnable_config(config)
#     )

# Build minimal graph: START -> agent -> END
workflow = StateGraph(
    state_schema = GlobalState, 
    context_schema = GlobalContext
)

# Add single node (your input/output)
# workflow.add_node("call_router", call_router)
workflow.add_node("retrieve_documents", retrieval_graph)
workflow.add_node("invoke_agent", invoke_agent)

# Edges
workflow.add_edge(START, 'retrieve_documents')
workflow.add_edge('retrieve_documents', "invoke_agent")
workflow.add_edge("invoke_agent", END)

graph = workflow.compile()
# graph = workflow.compile(context=make_context)

graph.name = "Anubis"

__all__ = ["graph"]
