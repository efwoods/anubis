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

from dotenv import load_dotenv
load_dotenv()

from langchain_core.runnables import RunnableConfig

configuration = GlobalConfiguration()

from langgraph.store.base import BaseStore
from langgraph.runtime import Runtime
from src.subgraphs.vector_store_graph.utils.retrieval import make_pg_store
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import WebBaseLoader

import asyncio
import nest_asyncio

nest_asyncio.apply()

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

    loader = WebBaseLoader("https://lifeboat.com/ex/bios.shivon.a.zilis")

    pages = []
    async for doc in loader.alazy_load():
        pages.append(doc)
    # logger.info(f"pages[0].page_content[:100]: {pages[0].page_content[:100]:}")
    logger.info(f"pages[0].page_content: {pages[0].page_content}")
    logger.info(f"pages[0].metadata: {pages[0].metadata}")

    logger.info(f"pages[0].metadata: {pages[0].metadata}")
    logger.info(f"pages[0].page_content: {pages[0].page_content}")


    namespace = ("evan_woods", "shivon_zilis", "document", "lifeboat_com_ex_bios_shivon_a_zilis")

    aput_result = await runtime.store.aput(
        namespace=namespace, 
        key="document1", 
        value={"page_content":pages[0].page_content, "metadata": pages[0].metadata}
    )

    aget_result = await runtime.store.aget(namespace, key="document1")
    logger.info(f"aget_result: {aget_result}")

    asearch_result = await runtime.store.asearch(("evan_woods", "shivon_zilis", "document"))
    logger.info("asearch_result: {asearch_result}")

    asearch_result_query = await runtime.store.asearch(("evan_woods", "shivon_zilis", "document"), query="Who is Shivon Zilis")
    logger.info(f"asearch_result with query: {asearch_result_query}")

    # human_message = state['messages'][-1]

    # if isinstance(human_message, HumanMessage):
        


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
test_graph = workflow.compile(store = make_pg_store)

__all__ = ["test_graph"]
