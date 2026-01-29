from typing import Any, Callable, List, Optional, cast
from langchain_tavily import TavilySearch
from langgraph.runtime import get_runtime
from src.anubis.utils.context import Context
from langchain.tools import tool, ToolRuntime
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document


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

TOOLS: List[Callable[..., Any]] = [search, vectorstore_retrieval_tool]





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
























