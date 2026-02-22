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
from src.subgraphs.vector_store_graph.utils.utilities import format_docs, get_message_text

import logging
logger = logging.getLogger(__name__)

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage

# Define the function that calls the model
class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""
    query: str

from langgraph.runtime import Runtime

from langchain_core.messages.utils import (trim_messages, count_tokens_approximately)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.anubis.utils.model import init_model

from src.anubis.utils.helper_functions import summarize_messages

async def generate_query(
    state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext]
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

    # update the context from the state

    #  update the configuration if used as an argument

    # if config:
    #     if config.get("metadata", None) != None:
    #         logger.warning(f"config: {config}")    
    #         user_id = config['metadata'].get("user_id", "")
    #         assistant_id = config['metadata'].get("assistant_id", "")

    #         if (user_id):
    #             runtime.context.user_ctx.user_id = user_id
    #             runtime.context.assistant_ctx.user_id = user_id

    #         if (assistant_id):
    #             runtime.context.assistant_ctx.assistant_id = assistant_id

    # configuration = runtime.context.configuration

    # messages = state['messages']
    # query_l = state['queries']
    # system_message_instruction_single_message =  "<Instruction>Please summarize this message into a query:</Instruction>"
    
    # future_updated_system_message = "<Instructions>Using the summary and messages, create a brief single-sentence query that identifies the intent of all the messages and message summary and will retrieve documents that match this intent. Treat the intent of the summary and messages as the question to which the retrieved documents are the answer such that the question will match the content of the retrieved documents.</Instructions>\n<Summary>The following is the summary of the current conversation to date.</Summary>"

    # future_updated_system_message_failsafe = "<Instructions>Using the messages, create a brief single-sentence query that identifies the intent of all the messages and will retrieve documents that match this intent. Treat the intent of the summary and messages as the question to which the retrieved documents are the answer such that the question will match the content of the retrieved documents.</Instructions>"

    # system_message = system_message_instruction_single_message
    # input = [SystemMessage(content=[{'type': 'text', 'text': system_message}]), HumanMessage(content=[{'type': 'text', 'text': human_input}])]
    # model = init_model(configuration)
    # response = await model.ainvoke(input=input)


    # master_message_list = await summarize_messages(
    #     messages, 
    #     configuration, 
    #     future_updated_system_message, 
    #     future_updated_system_message_failsafe, 
    #     system_message_instruction_single_message,
    #     query_l = query_l, 
    #     query_generation_mode=True
    # )
    # # Create a model for invocation
    # model_structured_output = init_model(
    #             configuration = configuration,
    #             response_format=SearchQuery
    #         )
    # generated = cast(SearchQuery, await model_structured_output.ainvoke(master_message_list))

    # return {"queries": [generated.query]}

    return {"queries": ["test query"]}

from src.subgraphs.vector_store_graph.utils.retrieval import (
    make_pg_vector
)

from src.anubis.utils.helper_functions import update_current_user_and_assistant_identity
async def retrieve(
    state: GlobalState, config: RunnableConfig,  runtime: Runtime[GlobalContext]
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
    from langchain_core.messages import HumanMessage
    logging.info(f"XXXXX RETRIEVE NODE XXXX")
    logger.warning(f"runtime.context.user_ctx.user_id: {runtime.context.user_ctx.user_id}")
    logger.warning(f"runtime.context.assistant_ctx.assistant_id: {runtime.context.assistant_ctx.assistant_id}")
   
    if config:
        update_current_user_and_assistant_identity(config, runtime)

    human_message = state['messages'][-1]

    assert(isinstance(human_message, HumanMessage))

    retrieval_message = {"messages" : [human_message]}

    logger.info(f"{retrieval_message}")
    
    configuration = runtime.context.configuration

    if isinstance(runtime.context.user_ctx, dict):
        user_id = runtime.context.user_ctx.get("user_id", "")
    else:
        user_id = getattr(runtime.context.user_ctx, "user_id", "")
        
    if isinstance(runtime.context.assistant_ctx, dict):
        assistant_id = runtime.context.assistant_ctx.get("assistant_id", "")
    else:
        assistant_id = getattr(runtime.context.assistant_ctx, "assistant_id", "")

    logger.info(f"user_id: {user_id}")
    logger.info(f"assistant_id: {assistant_id}")

    memory_search = runtime.context.vector_store_memory_search_only

    if memory_search == "FALSE":
        filter_query = {
                "user_id": {"$eq": user_id},
                "assistant_id": {"$eq": assistant_id}, 
        }
                # "type": {"$ne": "memory"}
    else:
        filter_query = {
                "user_id": {"$eq": user_id},
                "assistant_id": {"$eq": assistant_id}, 
        }
                # "type": {"$eq": "memory"}

    
    vector_store = await make_pg_vector(configuration)

    logger.info(f"breakpoint")
    if len(state['queries']) > 0:
        query = state['queries'][-1]
    else:
        query = getattr(state['messages'][-1], "content", "")

    results = await vector_store.asimilarity_search_with_relevance_scores(
        query = query,
        filter=filter_query,
    )
        # score_threshold=0.6

    retrieved_docs = [doc[0] for doc in results] # extract documents only

    logger.info(f"breakpoint")

    # logger.info(f"Query: {state['queries'][-1]} | Docs: {len(retrieved_docs)}")
    logger.info(f"{retrieved_docs}")
    state['retrieved_docs'] = []
    return {"retrieved_docs": retrieved_docs}

# Define a new graph
builder = StateGraph(GlobalState, context_schema=GlobalContext)

# builder.add_node(generate_query)  # type: ignore[arg-type]
builder.add_node(retrieve)  # type: ignore[arg-type]

# builder.add_edge("__start__", "generate_query")
# builder.add_edge("generate_query", "retrieve")
builder.add_edge("__start__", "retrieve")

# This compiles it into a graph you can invoke and deploy.
retrieval_graph = builder.compile()
retrieval_graph.name = "RetrievalGraph"

__all__ = ["retrieval_graph"]
