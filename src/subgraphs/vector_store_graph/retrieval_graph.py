"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing user inputs, generating queries, retrieving
relevant documents, and formulating responses.
"""

from datetime import datetime, timezone
from typing import cast

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from pydantic import BaseModel

from src.subgraphs.vector_store_graph.utils import retrieval
from src.anubis.utils.configuration import GlobalConfiguration
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.state import GlobalState
from src.subgraphs.vector_store_graph.utils.utilities import format_docs, get_message_text, load_chat_model



import logging
logger = logging.getLogger(__name__)

# Define the function that calls the model
class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""
    query: str

from langgraph.runtime import Runtime

async def generate_query(
    state: GlobalState, runtime: Runtime[GlobalContext]
) -> dict[str, list[str]]:
    """Generate a search query based on the current state and configuration.

    This function analyzes the messages in the state and generates an appropriate
    search query. For the first message, it uses the user's input directly.
    For subsequent messages, it uses a language model to generate a refined query.

    Args:
        state (State): The current state containing messages and other information.
        config (RunnableConfig | None, optional): GlobalConfiguration for the query generation process.

    Returns:
        dict[str, list[str]]: A dictionary with a 'queries' key containing a list of generated queries.

    Behavior:
        - If there's only one message (first user input), it uses that as the query.
        - For subsequent messages, it uses a language model to generate a refined query.
        - The function uses the configuration to set up the prompt and model for query generation.
    """
    logging.info(f"XXXXX GENERATE QUERY NODE XXXX")

    messages = state['messages']
    if len(messages) == 1:
        # It's the first user question. We will use the input directly to search.
        human_input = get_message_text(messages[-1])
        return {"queries": [human_input]}
    else:
        
        configuration = runtime.context.configuration
        # Feel free to customize the prompt, model, and other logic!
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", configuration.query_system_prompt),
                ("placeholder", "{messages}"),
            ]
        )

      # Create a model for invocation
        from src.anubis.utils.model import init_model
        configuration = runtime.context.configuration

        tools = []

        model_structured_output = init_model(
            configuration.provider_model, 
            configuration.llama_api_base_url,
            configuration.llama_api_key,
            tools, 
            configuration.dev, 
            response_format=SearchQuery
        )

        message_value = await prompt.ainvoke(
            {
                "messages": state.messages,
                "queries": "\n- ".join(state.queries),
                "system_time": datetime.now(tz=timezone.utc).isoformat(),
            },
            configuration,
        )

        generated = cast(SearchQuery, await model_structured_output.ainvoke(message_value, configuration))
        return {
            "queries": [generated.query],
        }


async def retrieve(
    state: GlobalState, runtime: Runtime[GlobalContext]
) -> dict[str, list[Document]]:
    """Retrieve documents based on the latest query in the state.

    This function takes the current state and configuration, uses the latest query
    from the state to retrieve relevant documents using the retriever, and returns
    the retrieved documents.

    Args:
        state (State): The current state containing queries and the retriever.
        config (RunnableConfig | None, optional): GlobalConfiguration for the retrieval process.

    Returns:
        dict[str, list[Document]]: A dictionary with a single key "retrieved_docs"
        containing a list of retrieved Document objects.
    """
    logging.info(f"XXXXX RETRIEVE NODE XXXX")

    configuration = runtime.context.configuration

    async with retrieval.make_retriever(configuration) as retriever:
        # logger.info(f"XXXXXXXXXXXXXXXXXX CONFIGURATION: {config}")
        
        logger.info(f"{configuration}")

        logger.info(f"{state['queries'][-1]}")

        # Filter Query to return only user_id assistant_id
        user_id = runtime.context.assistant_ctx.user_id
        assistant_id = runtime.context.assistant_ctx.assistant_id

        memory_search = runtime.context.vector_store_memory_search_only
        if memory_search == "FALSE":
            filter_query = {
                "$and": [
                    {"user_id": {"$eq": user_id}},
                    {"assistant_id": {"$eq": assistant_id}},
                    {"type": {"$neq": "memory"}}
                ]
            }
        else:
            filter_query = {
                "$and": [
                    {"user_id": {"$eq": user_id}},
                    {"assistant_id": {"$eq": assistant_id}},
                    {"type": {"$eq": "memory"}}
                ]
            }
        
        

        response = await retriever.ainvoke(
            state['queries'][-1],
            search_kwargs={
                "k": 100,
                "score_threshold": 0.6, # cosine similarity threshold (greater is higher quality fewer results)
                "pre_filter": filter_query
            })
        
        filtered_response = []

        # for doc in response:
        #     if doc.metadata["user_id"] == user_id and doc.metadata['assistant_id'] == assistant_id:
        #         filtered_response.append(doc)

        filtered_response = [doc for doc in response if doc.metadata['user_id'] == user_id and doc.metadata['assistant_id'] == assistant_id]

        
        # response = await retriever.ainvoke(
        #     state['queries'][-1],
        #     search_kwargs={
        #         "pre_filter": {"assistant_id": 'mom'}
        #     })
        
        logger.info(f"Query: {state['queries'][-1]} | Docs: {len(filtered_response)}")
        logger.info(f"{filtered_response}")
        return {"retrieved_docs": filtered_response}

# Define a new graph (It's just a pipe)
builder = StateGraph(GlobalState, context_schema=GlobalContext)

builder.add_node(generate_query)  # type: ignore[arg-type]
builder.add_node(retrieve)  # type: ignore[arg-type]
builder.add_edge("__start__", "generate_query")
builder.add_edge("generate_query", "retrieve")

# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
retrieval_graph = builder.compile(
    interrupt_before=[],  # if you want to update the state before calling the tools
    interrupt_after=[],
)
retrieval_graph.name = "RetrievalGraph"

__all__ = ["retrieval_graph"]
