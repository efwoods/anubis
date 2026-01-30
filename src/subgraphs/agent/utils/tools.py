from typing import Any, Callable, List, Optional, cast, Dict
from langchain_tavily import TavilySearch
from langgraph.runtime import get_runtime
from src.subgraphs.agent.utils.context import Context
from langchain.tools import tool, ToolRuntime
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain.messages import AIMessage, SystemMessage, HumanMessage


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


@tool
def get_chat_metadata(runtime: ToolRuntime[Context]) -> Dict[str, str]:
    """Get current chat metadata: IDs, names, media details. Use when needing session/file info."""
    
    # Thread/session ID from config (Studio uses threads)
    thread_id = runtime.config["configurable"].get("thread_id", "no-thread")
    
    # User/Human & Assistant names from recent messages or store
    messages = runtime.state["messages"][-5:]  # Last 5 for context
    human_name = "Human"  # Default
    assistant_name = "Assistant"  # Default
    
    for msg in reversed(messages):
        if msg.type == "human" and "name" in getattr(msg, "response_metadata", {}):
            human_name = msg.response_metadata["name"]
            break
        elif msg.type == "ai" and "name" in getattr(msg, "response_metadata", {}):
            assistant_name = msg.response_metadata["name"]
            break
    
    # User/assistant IDs from context/config or store
    user_id = runtime.context.user_id or "studio_user"
    assistant_id = runtime.config["configurable"].get("assistant_id", "studio_assistant")
    
    # Latest media from HumanMessage (adapt to your format)
    media_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage) and has_media(m)), None)
    if media_msg:
        mime_type = media_msg.additional_kwargs.get("mime_type", "unknown")
        filename = media_msg.additional_kwargs.get("filename", "unnamed")
    else:
        mime_type = filename = "no_media"
    
    # Optional: Names from store (if you save them)
    if runtime.store:
        user_info = runtime.store.get(("users",), user_id)
        if user_info:
            human_name = user_info.value.get("name", human_name)
    
    return {
        "user_id": user_id,
        "assistant_id": assistant_id,
        "thread_id": thread_id,
        "human_name": human_name,
        "assistant_name": assistant_name,
        "mime_type": mime_type,
        "filename": filename
    }












