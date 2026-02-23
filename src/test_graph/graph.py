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

from langgraph.store.base import BaseStore
from langgraph.runtime import Runtime
from src.subgraphs.vector_store_graph.utils.retrieval import make_pg_store

import logging
logger = logging.getLogger(__name__)

async def test_node(state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext], store: BaseStore):
    logger.info(f"ENTRYPOINT TEST NODE")

    logger.info(f"config: {config.keys()}")
    logger.info(f"config user_ctx: {config['configurable'].get('user_ctx', None)}")
    logger.info(f"config assistant_ctx: {config['configurable'].get('assistant_ctx', None)}")
    logger.info(f"runtime.store: {runtime.store}")
    logger.info(f"store: {store}")
    logger.info(f"new test")

    test_namespace = ("testing", "documents")
    await store.aput(namespace=test_namespace, key="testing_key", value={"testing_key":"testing_value", "documents":"This is a test field to embed. UNICORN."})
    testing_get = await runtime.store.aget(("testing", "documents"), key="testing_key")
    results = await runtime.store.asearch(("testing", "documents"), query="UNICORN.")
    logger.info(f"testing_get: {testing_get}")
    logger.info(f"results: {results}")


# Build minimal graph: START -> agent -> END
workflow = StateGraph(
    state_schema = GlobalState, 
    context_schema = GlobalContext
)

# Add single node (your input/output)
workflow.add_node("test_node", test_node)

# Edges
workflow.add_edge(START, 'test_node')
workflow.add_edge("test_node", END)

test_graph = workflow.compile()
compile()

__all__ = ["test_graph"]
