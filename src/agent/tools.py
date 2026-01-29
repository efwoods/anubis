from typing import Any, Callable, List, Optional, cast

from langchain_tavily import TavilySearch
from langgraph.runtime import get_runtime

from src.agent.context import Context

from langchain.tools import tool

from langchain_community.document_loaders import WebBaseLoader

# Testing loading documents for RAG retrieval
urls = [
    "https://grokipedia.com/page/Shivon_Zilis",
]

docs = [WebBaseLoader(url).load() for url in urls]

docs[0][0].page_content.strip()[:1000]

from langchain_text_splitters import RecursiveCharacterTextSplitter

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder( chunk_size=100, chunk_overlap=50)

doc_splits = text_splitter.split_documents(docs_list)




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




# Memory
from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits, embedding=HuggingFaceEmbeddings()
)

retriever = vectorstore.as_retriever()

# retriever tool
@tool
def vectorstore_retrieval_tool(query: str) -> str:
    """ Search and return information. """
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


TOOLS: List[Callable[..., Any]] = [search, vectorstore_retrieval_tool]

