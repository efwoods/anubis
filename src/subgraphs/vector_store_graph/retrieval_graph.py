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
        model = load_chat_model(configuration.query_model).with_structured_output(
            SearchQuery
        )

        message_value = await prompt.ainvoke(
            {
                "messages": state.messages,
                "queries": "\n- ".join(state.queries),
                "system_time": datetime.now(tz=timezone.utc).isoformat(),
            },
            configuration,
        )
        generated = cast(SearchQuery, await model.ainvoke(message_value, configuration))
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
        response = await retriever.ainvoke(state['queries'][-1])
        # CRITICAL: filter= here
        # response = await retriever.asimilarity_search(
        #     state.queries[-1], 
        #     # filter={"user_id": {"$eq": }},  # MongoDB $eq filter
        #     fetch_k=100,  # Pre-filter candidates
        #     # search_kwargs={"score_threshold": 0.7}  # Optional scoring
        # )
        logger.info(f"Query: {state['queries'][-1]} | Docs: {len(response)}")
        logger.info(f"{response}")
        return {"retrieved_docs": response}


async def respond(
    state: GlobalState, runtime: Runtime[GlobalContext]
) -> dict[str, list[BaseMessage]]:
    """Call the LLM powering our "agent"."""
    logging.info(f"XXXXX REPONSE NODE XXXX")
    configuration = runtime.context.configuration

    # Feel free to customize the prompt, model, and other logic!
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.response_system_prompt),
            ("placeholder", "{messages}"),
        ]
    )
    model = load_chat_model(configuration.response_model)

    retrieved_docs = format_docs(state.retrieved_docs)
    message_value = await prompt.ainvoke(
        {
            "messages": state.messages,
            "retrieved_docs": retrieved_docs,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        configuration,
    )
    response = await model.ainvoke(message_value, configuration)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define a new graph (It's just a pipe)

builder = StateGraph(GlobalState, context_schema=GlobalContext)

builder.add_node(generate_query)  # type: ignore[arg-type]
builder.add_node(retrieve)  # type: ignore[arg-type]
# builder.add_node(respond)  # type: ignore[arg-type]
builder.add_edge("__start__", "generate_query")
builder.add_edge("generate_query", "retrieve")
# builder.add_edge("retrieve", "respond")

# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
retrieval_graph = builder.compile(
    interrupt_before=[],  # if you want to update the state before calling the tools
    interrupt_after=[],
)
retrieval_graph.name = "RetrievalGraph"

__all__ = ["retrieval_graph"]
