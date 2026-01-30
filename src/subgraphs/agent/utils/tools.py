from typing import Any, Callable, List, Optional, cast, Dict
from langchain_tavily import TavilySearch
# from src.anubis.utils.state import AnubisState
from src.subgraphs.agent.utils.context import Context
from langchain.tools import tool, ToolRuntime
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain.messages import AIMessage, SystemMessage, HumanMessage

from langgraph.runtime import get_runtime

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

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from src.subgraphs.vector_store_graph.index_graph import index_graph 

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=100, chunk_overlap=50)
embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = InMemoryVectorStore.from_documents(
    documents=[], embedding=HuggingFaceEmbeddings()
)
retriever = vectorstore.as_retriever()

# retriever tool
@tool
def vectorstore_retrieval_tool(query: str) -> str:
    """ Search and return information. """
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

# # Upload to vectorstore tool
# def upload_to_vectorstore(runtime: ToolRuntime) -> str:
#     """Upload media attachments to vectorstore when indicated that the media is about or contains the avatar.
#     """
#     attachments = runtime.state.get("attachments", [])
#     if not attachments:
#         return "No attachmets found in state['attachments]'. Pass file paths or bytes."
    
#     docs: List[Document] = []
#     for attach in attachments:
#         if isinstance(attach, str): 
#             content = attach.read() if hasattr(attach, 'read') else attach.decode() if isinstance(attach, bytes) else attach
@tool
def health_check(runtime: ToolRuntime[Context]) -> AIMessage:
    """Tool is called when the human requests to test tool use.

    Args:
        runtime (ToolRuntime[Context]): ToolRuntime

    Returns:
        AIMessage: success message
    """
    return AIMessage(content="success")


# @tool
# def get_chat_metadata(runtime: ToolRuntime[Context]) -> Dict[str, str]:
#     """Get current chat metadata: IDs, names, media details. Use when needing session/file info."""
    
#     # Thread/session ID from config (Studio uses threads)
#     thread_id = runtime.config["configurable"].get("thread_id", "no-thread")
    
#     # User/Human & Assistant names from recent messages or store
#     messages = runtime.state["messages"][-5:]  # Last 5 for context
#     human_name = "Human"  # Default
#     assistant_name = "Assistant"  # Default
    
#     for msg in reversed(messages):
#         if msg.type == "human" and "name" in getattr(msg, "response_metadata", {}):
#             human_name = msg.response_metadata["name"]
#             break
#         elif msg.type == "ai" and "name" in getattr(msg, "response_metadata", {}):
#             assistant_name = msg.response_metadata["name"]
#             break
    
#     # User/assistant IDs from context/config or store
#     user_id = runtime.context.user_id or "studio_user"
#     assistant_id = runtime.config["configurable"].get("assistant_id", "studio_assistant")
    
#     # Latest media from HumanMessage (adapt to your format)
#     media_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage) and has_media(m)), None)
#     if media_msg:
#         mime_type = media_msg.additional_kwargs.get("mime_type", "unknown")
#         filename = media_msg.additional_kwargs.get("filename", "unnamed")
#     else:
#         mime_type = filename = "no_media"
    
#     # Optional: Names from store (if you save them)
#     if runtime.store:
#         user_info = runtime.store.get(("users",), user_id)
#         if user_info:
#             human_name = user_info.value.get("name", human_name)
    
#     return {
#         "user_id": user_id,
#         "assistant_id": assistant_id,
#         "thread_id": thread_id,
#         "human_name": human_name,
#         "assistant_name": assistant_name,
#         "mime_type": mime_type,
#         "filename": filename
#     }

import logging
logger = logging.getLogger(__name__)


from src.subgraphs.agent.utils.utilities import process_media

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
                
                return AIMessage(content="Please attach media")  # update with custom no media response from llm return to model node with prompt to generate

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
