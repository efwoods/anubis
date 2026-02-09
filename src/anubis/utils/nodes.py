# src/anubis/utils/nodes.py

# from src.anubis.utils.tools import tools  # Your tools list (empty for now)

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.runtime import Runtime   

from src.anubis.utils.model import init_model
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.configuration import GlobalConfiguration
from src.anubis.utils.state import GlobalState

from src.anubis.utils.helper_functions import format_docs

from src.anubis.utils.classes.DynamicPromptBuilder import DynamicPromptBuilder

from src.anubis.utils.model import invoke_model_core

from src.anubis.utils.tools import (
    health_check, 
    # add_to_vectorstore_subgraph, 
    # retrieve_from_vectorstore_subgraph, 
    # upsert_memory
)

from langgraph.store.base import BaseStore

from src.subgraphs.vector_store_graph.utils.retrieval import make_vectorstore

# Optional: Add tools=[] if you have them
tools = []  # Replace with your tools 

async def invoke_model(state: GlobalState, runtime: Runtime[GlobalContext]):
    """Build a model, agent, and dynamic system prompt to load the identity of the assistant into the assistant's current state of consciousness"""
    logger.info(f"INVOKE MODEL NODE ")
    
    response = await invoke_model_core(
        state=state,
        runtime=runtime,
        tools=tools,
    )

    logger.info(f"INVOKE MODEL NODE {response['messages']}")

    result = {"messages": [response["messages"]]}
    logger.info(f"RESULT OF INVOKE MODEL NODE RETURN: {result}")

    return result

import asyncio
from src.subgraphs.vector_store_graph.utils.retrieval import make_text_encoder
from src.subgraphs.vector_store_graph.utils.retrieval import make_pg_vector

async def invoke_agent(state: GlobalState, runtime: Runtime[GlobalContext], store: BaseStore):
    """Build a model, agent, and dynamic system prompt to load the identity of the assistant into the assistant's current state of consciousness"""
    from langchain_huggingface import HuggingFaceEmbeddings
    from sqlalchemy.ext.asyncio import create_async_engine

    logger.info(f"INVOKE AGENT NODE ")

    configuration =  runtime.context.configuration
    # vectorstore = await make_vectorstore(configuration)

    logger.info(f"breakpoint invoke agent")

    # test_store = await runtime.store.alist_namespaces()
    user_id = runtime.context.assistant_ctx.user_id
    assistant_id = runtime.context.assistant_ctx.assistant_id

    # with runtime.context.postgres_db_store as store:
    #     logger.info(f"breakpoint")

    logger.info(f"breakpoint")
    vector_store = make_pg_vector(configuration)
    async with vector_store as vector_store:
        logger.info(f"breakpoint")
        results = await vector_store.asimilarity_search_with_relevance_scores(
        query="kitty", 
        filter={"id": {"$in": [1,2,5,9]}},
        score_threshold=0.3
        )

    retrieved_docs = [doc[0] for doc in results] # extract documents only

    logger.info(f"breakpoint")

    """ CREATE MODEL """

    config = runtime.context.configuration # Loads env vars automatically

    model = init_model(
        config.provider_model,
        config.llama_api_base_url,
        config.llama_api_key,
        tools,
        config.dev
    )

    """ VECTORSTORE DOCUMENT RETRIEVAL """

    # Retrieve documents for the query
    from src.subgraphs.vector_store_graph.retrieval_graph import retrieval_graph

    human_message = state['messages'][-1]

    assert(isinstance(human_message, HumanMessage))
    
    retrieval_message = {"messages" : [human_message]}

    # Memories are text-only statements with user_id, assistant_id, "type": "memory", "source": "conversation" add to vectorstore and filter results to retrieve only memories through prompt-created generation and retrieval; invoke retrieval and only return documents that have the type "memory" 

    # relevant documents invoke retrieval and only return documents that do not have the type "memory"

    runtime.context.vector_store_memory_search_only = "FALSE"

    # new_state_retrieved_docs = await retrieval_graph.ainvoke(
    #     retrieval_message, 
    #     context=runtime.context
    # )
    
    state['retrieved_docs'] = []

    # populate the relevant documents with a new state
    # state['retrieved_docs'] = new_state_retrieved_docs['retrieved_docs']

    logger.info(f"breakpoint")

    state['retrieved_docs'] = retrieved_docs

    # Vectorstore Retrieved Docments
    retrieved_docs = format_docs(state.get('retrieved_docs', []))

    """ RETRIEVE MEMORIES FROM NATURAL LANGUAGE GENERATED QUERY IN VECTORSTORE """

    # TODO: PROMPT INJECT RETRIEVED MEMORIES 

    prompt_builder = DynamicPromptBuilder()

    # TODO: Update the assistant context from the store: details about the AI 

    """ POSTGRES STORE RETRIEVAL (METADATA AI/USER) """


    logger.info(f"async postgres store connection test breakpoint")

    # ai_context = runtime.context.postgres_db_store.get(namespace=(user_id, assistant_id), key="assistant_identity")

    from src.subgraphs.vector_store_graph.utils.retrieval import make_pg_store
    
    postgres_db_store = await make_pg_store(configuration)

    namespace=("user_1234", "assistant_1234")
    async with postgres_db_store as postgres_db_store:
        namespaces = await postgres_db_store.alist_namespaces()
        logger.info(f"namespaces: {namespaces}")

        aput_result = await postgres_db_store.aput(namespace, key="identity_test", value={"identity_key":"identity_value"})
        logger.info(f"aput_result: {aput_result}")

        aget_result = await postgres_db_store.aget(namespace, key="identity_test")
        logger.info(f"aget_result: {aget_result}")

        adelete_result = await postgres_db_store.adelete(namespace, key="identity_test")
        logger.info(f"adelete_result: {adelete_result}")

        aget_result_after_delete = await postgres_db_store.aget(namespace, key="identity_test")
        logger.info(f"aget_result_after_delete: {aget_result_after_delete}")

        asearch_result = await postgres_db_store.asearch(namespace)
        logger.info(f"asearch_result: {asearch_result}")

        ai_context = await postgres_db_store.aget(namespace, key="identity")
        logger.info(f"ai_context: {ai_context}")

    logger.info(f"async postgres store connection test POST breakpoint")

    ai_context = {}

    # Load the current assistant context for prompt injection
    # ai_context = runtime.context.assistant_ctx.to_dict()

    # TODO: Update the user context from the state: details about the user from the AI's perspective

    user_ctx = ai_context.get("USER", "") # information stored in a nested dictionary about the user

    # Load the current user context from prompt injection
    # user_ctx = runtime.context.user_ctx.to_dict()

    system_time = datetime.now(tz=timezone.utc).isoformat()

    temporary_system_prompt_update = runtime.context.temporary_system_prompt_update

    populated_template = prompt_builder.build_prompt(
        ai_context=ai_context,
        user_context=user_ctx, 
        retrieved_docs=retrieved_docs,
        system_time = system_time,
        temporary_message=temporary_system_prompt_update,
    )

    logger.info(f"populated_template: {populated_template}")

    # clear the temporary injected prompt
    runtime.context.temporary_system_prompt_update = ""

    # prepend system message
    messages_input = populated_template.messages + state["messages"]
    
    logger.info(f"messages_input: {messages_input}")
    
    # Call the model
    response = await model.ainvoke(input=messages_input)

    logger.info(f"AGENT RESPONSE: {response}")
    result = {"messages": [response]}
    return result


from pydantic import BaseModel, Field
from typing import Literal
from langgraph.types import Command

class RouteDecision(BaseModel):
    """"Determine whether to upload media or respond to the conversation. """
    reasoning: str = Field(
        description="Step-by-step reasoning behind the decision for the route."
    )
    route_decision: Literal["chat"] = Field(
        description="Classification of the route. chat if responding to the conversation. upload if the user indicates the attached media needs to be added to the identity or uploaded."
    )


async def call_router(state: GlobalState, runtime: Runtime[GlobalContext]) -> Command[Literal["chat"]]:
    """Decides whether to upload new personal information or to respond based upon the chat

    Args:
        state (GlobalState): message state
        runtime (Runtime[GlobalContext]): context of agent and user and configuration

    Returns:
        command: next node location
    """
    logger.info(f"CALL ROUTER")
    
    # Create a model with structured output
    config = runtime.context.configuration # Loads env vars automatically
    tools = []

    model_router_structured_output = init_model(
        config.provider_model,
        config.llama_api_base_url,
        config.llama_api_key,
        tools,
        config.dev,
        response_format=RouteDecision
    )

    decision_instructions = """Route to chat.
"""
    system_message = SystemMessage(content=decision_instructions)
    
    assert isinstance(state['messages'][-1], HumanMessage)
    assert hasattr(state['messages'][-1], "content")
    assert getattr(state['messages'][-1], "content") != ""

    human_message = state['messages'][-1]

    decided_route = await model_router_structured_output.ainvoke(
        [system_message, human_message]
    )
    logger.info(f"DECIDED ROUTE {decided_route}")
    
    return {"route_decision": decided_route}

async def route_node_from_decision(state: GlobalState, runtime: GlobalContext) -> str:
    logger.info(f"ROUTING NODE FROM DECISION")
    route_decision = state["route_decision"]
    if route_decision == "chat":
        return "invoke_agent"
    else:
        logger.info(f"NON CHAT")
        return "invoke_agent"
