""" Agent SubGraph Tools """

from typing import Any, Callable, List, Optional, cast, Dict
from langchain_tavily import TavilySearch
# from src.anubis.utils.state import AnubisState
from src.subgraphs.agent.utils.context import Context
from langchain.tools import tool, ToolRuntime
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain.messages import AIMessage, SystemMessage, HumanMessage

from langgraph.runtime import get_runtime

import logging
logger = logging.getLogger(__name__)

"""Define the agent's tools."""

import uuid
from typing import Annotated

from langchain_core.tools import InjectedToolArg
from langgraph.store.base import BaseStore


@tool
def health_check(runtime: ToolRuntime[Context]) -> AIMessage:
    """Tool is called when the human requests to test tool use.

    Args:
        runtime (ToolRuntime[Context]): ToolRuntime

    Returns:
        AIMessage: success message
    """
    return AIMessage(content="success")

@tool
async def search(query: str) ->  Optional[dict[str, Any]]:
    """Basic websearch

    Args:
        query (str): input query

    Returns:
        Optional[dict[str, Any]]: web search results
    """
    runtime = get_runtime(Context)
    wrapped = TavilySearch(max_results=runtime.context.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))

@tool
async def add_to_vectorstore(runtime: ToolRuntime[Context])-> AIMessage:
    """ TOOL CALL USE CASE, INPUTS, AND RETURN VALUE"""
    from src.subgraphs.vector_store_graph.index_graph import index_graph
    logger.info('ADD_TO_VECTORESTORE TOOL CALLED XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

    # convert state
    logger.info(f"{runtime.state} XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    
    messages = runtime.state['messages']
    logger.info(f"MESSAGES XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX {messages}")
    
    logger.info(f" LEN` MESSAGE XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX {len(messages)}")

    recent_msg = messages[-1]
    # logger.info(f" RECENT MESSAGE messages[-1] XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX {messages[-1]}")
    # logger.info(f" RECENT MESSAGE messages[0] XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX {messages[0]}")
    # logger.info(f" RECENT MESSAGE messages[1] XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX {messages[1]}")
    # [logger.info(f"{message}") for message in messages['content']['text']]
    # count = 0
    # for message in messages:
    #     logger.info(f"{count}: {message}")
    #     count+=1
    # logger.info(f"penultimate has data: messages[-2]: {messages[-2]}")

    # logger.info(f" XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX {isinstance(recent_msg, HumanMessage)}")
    recent_message = messages[-2]
    recent_docs: List[Document] = []
    if isinstance(recent_message, HumanMessage):
        # logger.info(f"CONTENT XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX {recent_msg.content}")
        content = recent_message.content
        logger.info(f"len content: {len(content)}")
        # logger.info(f"content: {content}")
        if isinstance(content, list):
            if len(content) > 1:
                logger.warning(f"LEN OF content 1 or 0 not handled; uploads to vectorstore successfully but no content")
                text_from_human = content[0]
                image_content_dict = content[-1]
                # logger.warning(f"content[-1]:{content[-1]}")
                description_doc = await process_media(image_content_dict)
                logger.info(f"description_doc: {description_doc}")
                recent_docs.append(description_doc)
                logger.info(f"recent_docs: {recent_docs}")
            else: 
                logger.warning(f"content list length is < 1; only text no media inferred.")
                
                return AIMessage(content="Please attach media") # update with custom no media response from llm return to model node with prompt to generate

        # if hasattr(recent_msg, 'media') and recent_msg.media:
        #     media_docs = process_media(recent_msg.media) # process the media
        #     recent_docs.extend(media_docs)
        # else:
        #     recent_docs.append(Document(page_content=content or ""))


    index_input = {"docs": recent_docs}
    logger.info(f"index_input XXXXXXXXXXXXXXXXXXXXXXXXX {index_input}")
    
    result = await index_graph.ainvoke(index_input, {"configurable": {"user_id": "test_user_1234"}}) # user id is bypassed for testing

    # logger.info(f"RESULT XXXXXXXXXXXXXXXXXXXXXXXXX {result}")
    # # Return results of subgraph
    # ai_content = result.get('messages', [-1]).content if result.get('messages') else "Indexed successfully"
    return AIMessage(content="added to vectorstore successfully")
    
@tool
async def retrieve_from_vectorstore(runtime: ToolRuntime[Context]) -> AIMessage:
    """Generates the correct query to search the vectorstore for relevant documents to the Human query.

    Args:
        runtime (ToolRuntime[Context]): tool runtime context

    Returns:
        AIMessage: _description_
    """
    from src.subgraphs.vector_store_graph.retrieval_graph import retrieval_graph
    # messages = runtime.state['messages']
    response = await retrieval_graph.ainvoke(runtime.state, {"configurable": {"user_id": "test_user_1234"}})
    logger.info(f"RETRIEVAL RESPONSE: {response}")
    ai_response = response['messages'][-1]
    
    return ai_response

@tool
async def upsert_memory(
    content: str,
    context: str,
    *,
    memory_id: uuid.UUID | None = None,
    # Hide these arguments from the model.
    user_id: Annotated[str, InjectedToolArg],
    store: Annotated[BaseStore, InjectedToolArg],
):
    """Upsert a memory in the database.

    If a memory conflicts with an existing one, then just UPDATE the
    existing one by passing in memory_id - don't create two memories
    that are the same. If the user corrects a memory, UPDATE it.

    Args:
        content: The main content of the memory. For example:
            "User expressed interest in learning about French."
        context: Additional context for the memory. For example:
            "This was mentioned while discussing career options in Europe."
        memory_id: ONLY PROVIDE IF UPDATING AN EXISTING MEMORY.
        The memory to overwrite.
    """
    mem_id = memory_id or uuid.uuid4()
    await store.aput(
        ("memories", user_id),
        key=str(mem_id),
        value={"content": content, "context": context},
    )
    return f"Stored memory {mem_id}"


""" Vector Store SubGraph Tools """
""" Process Media SubGraph Tools """

