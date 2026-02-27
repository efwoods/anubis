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

import logging
logger = logging.getLogger(__name__)

import uuid
import asyncio
from src.subgraphs.process_media_graph.process_media_graph_api_endpoint import process_media_graph_api_endpoint

async def url_loader(state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext], store: BaseStore):
    logger.info(f"ENTRYPOINT TEST NODE")
    

    # logger.info(f"config: {config.keys()}")
    # logger.info(f"config user_ctx: {config['configurable'].get('user_ctx', None)}")
    # logger.info(f"config assistant_ctx: {config['configurable'].get('assistant_ctx', None)}")
    # logger.info(f"runtime.store: {runtime.store}")
    # logger.info(f"store: {store}")
    # logger.info(f"new test")

    # test_namespace = ("testing", "documents")
    # await store.aput(namespace=test_namespace, key="testing_key", value={"testing_key":"testing_value", "documents":"This is a test field to embed. UNICORN."})
    # testing_get = await runtime.store.aget(("testing", "documents"), key="testing_key")
    # results = await runtime.store.asearch(("testing", "documents"), query="UNICORN.")
    # logger.info(f"testing_get: {testing_get}")
    # logger.info(f"results: {results}")

    # namespaces = await runtime.store.alist_namespaces()

    # all_documents = await runtime.store.asearch(("'Anubis_from_studio_3cf764e9-51c3-5404-9699-e16f5e4034ec'", "'3cf764e9-51c3-5404-9699-e16f5e4034ec'", "'assistant_ctx'"))
    # keys = [document.key for document in all_documents]
    # for key in keys:
    #     delete_result = await runtime.store.adelete(("evan_woods", "shivon_zilis", "test"), key=key)

    # logger.info(f"breakpoint")

    # human_message = state['messages'][-1]
    # if isinstance(human_message, HumanMessage):
    #     content = human_message.content[0]['text']
    #     logger.warning(f"content {content}")
    #     urls = content.split(",")
    #     loader = AsyncHtmlLoader(urls[0])

    # docs = []

    # from langchain_community.document_loaders import WebBaseLoader

    # loader = WebBaseLoader(urls)
    # pages = await asyncio.to_thread(loader.load)

    # docs = await asyncio.to_thread(loader.load)
    # html2text = Html2TextTransformer()
    # pages = await html2text.atransform_documents(docs)


    # logger.info(f"pages[0].page_content: {pages[0].page_content}")
    # logger.info(f"pages[0].metadata: {pages[0].metadata}")

    # logger.info(f"pages[0].metadata: {pages[0].metadata}")
    # logger.info(f"pages[0].page_content: {pages[0].page_content}")

    # filenames = [str(uuid.uuid5(uuid.NAMESPACE_URL, url)) for url in urls]

    # namespaces = [("evan_woods", "shivon_zilis", "document", filename) for filename in filenames]

    # _ = [doc.metadata.update({"document_id":str(uuid.uuid4()), "filename_uuid5":filename, "filename":url}) for doc, filename, url in zip(pages, filenames, urls)]
    # keys = [doc.metadata["document_id"] for doc in pages]

    # for namespace, key, page in zip(namespaces, keys, pages):
    #     aput_result = await runtime.store.aput(
    #     namespace=namespace, 
    #     key=key, 
    #     value={"document":page.to_json()}
    # )   
        
    # aget_result = await runtime.store.aget(namespaces[0], key=keys[0])
    # logger.info(f"aget_result: {aget_result}")

    # asearch_result = await runtime.store.asearch(("evan_woods", "shivon_zilis", "document"))
    # logger.info("asearch_result: {asearch_result}")

    # asearch_result_query = await runtime.store.asearch(("evan_woods", "shivon_zilis", "document"), query="Who is Shivon Zilis")
    # logger.info(f"asearch_result with query: {asearch_result_query}")

        
# Build minimal graph: START -> agent -> END
workflow = StateGraph(
    state_schema = GlobalState, 
    context_schema = GlobalContext
)

# Add single node (your input/output)
workflow.add_node("url_loader", url_loader)
# workflow.add_node("process_media", process_media_graph_api_endpoint)

# Edges
workflow.add_edge(START, 'url_loader')

# workflow.add_edge("url_loader", "process_media")
# workflow.add_edge("process_media", END)

workflow.add_edge("url_loader", END)
url_loader = workflow.compile()

__all__ = ["url_loader"]
