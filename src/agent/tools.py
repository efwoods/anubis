from typing import Any, Callable, List, Optional, cast

from langchain_tavily import TavilySearch
from langgraph.runtime import get_runtime

from src.agent.context import Context

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

TOOLS: List[Callable[..., Any]] = [search]
