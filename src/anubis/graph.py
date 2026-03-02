# src/anubis/graph.py

"""
src/anubis/graph.py
Super-Graph with a central Langchain Agent and subgraph tool use.
"""

import logging
logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, START, END
from src.anubis.utils.state import GlobalState
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.configuration import GlobalConfiguration

from src.subgraphs.vector_store_graph.retrieval_graph import retrieval_graph

from dotenv import load_dotenv
load_dotenv()

from src.anubis.utils.context import IdentityContext, AssistantContext
from langchain_core.runnables import RunnableConfig

from langchain_core.runnables import RunnableConfig
from src.anubis.utils.helper_functions import extract_user_id_assistant_id, configure_assistant_context

from src.anubis.utils.schema import RouteDecision

from pydantic import BaseModel, Field
from typing import Literal
from langgraph.types import Command


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

from langgraph.store.base import BaseStore


from langchain_core.messages.utils import (trim_messages, count_tokens_approximately)


""" TOOLS """

""" CONFIGURATION """

configuration = GlobalConfiguration()


""" NODES """

async def invoke_agent(state: GlobalState, config: RunnableConfig, runtime: Runtime[GlobalContext], store: BaseStore):
    """Build a model, agent, and dynamic system prompt to load the identity of the assistant into the assistant's current state of consciousness"""
    from langchain_huggingface import HuggingFaceEmbeddings
    from sqlalchemy.ext.asyncio import create_async_engine

    logger.info(f"INVOKE AGENT NODE ")

    configuration =  runtime.context.configuration

    logger.info(f"breakpoint invoke agent")
    
    logger.warning(f"THERE SHOULD BE ENVIRONMENT VARIABLES; configuration: {configuration}")

    logger.info(f"Testing store access")
    logger.info(f"runtime.store: {runtime.store}")
    logger.info(f"store: {store}")

    # Asserting Current Identity:
    user_id, assistant_id = await extract_user_id_assistant_id(config)
    logger.info(f"user_id: {user_id}")
    logger.info(f"assistant_id: {assistant_id}")

    logger.info(f"BREAKPOINT: UPDATE USER AND ASSISTANT CONTEXT")

    # await store.aput("testing", key="testing_key", value={"testing_key":"testing_value"})
    # testing_get = await store.aget("testing", key="testing_key", value={"testing_key":"testing_value"})
    # get_value = await store.aget("evan", key="name")
    # logger.info(f"get_value: {get_value}")
    # logger.info(f"testing_get: {testing_get}")

    """ CREATE MODEL """

    model = init_model(
        configuration = configuration,
    )

    """ VECTORSTORE DOCUMENT RETRIEVAL """

    logger.info(f"breakpoint")
    logger.info(f"state['retrieved_docs']: {state['retrieved_docs']}")

    # Vectorstore Retrieved Docments
    retrieved_docs = format_docs(state.get('retrieved_docs', []))

    logger.info(f"format_docs(state.get('retrieved_docs', [])): {retrieved_docs}")

    """ RETRIEVE MEMORIES FROM NATURAL LANGUAGE GENERATED QUERY IN VECTORSTORE """
    
    # # TODO: PROMPT INJECT RETRIEVED MEMORIES 

    prompt_builder = DynamicPromptBuilder()

    # TODO: Update the assistant context from the store: details about the AI 

    """ POSTGRES STORE RETRIEVAL (METADATA AI/USER) """

    logger.info(f"async postgres store connection test breakpoint")

    ai_context_item = await configure_assistant_context(config, store)

    try:
        assert(ai_context_item is not None)
    except AssertionError as e:
        logger.warning(f"ai_context_item is not None: {e}")
        raise e

    system_time = datetime.now(tz=timezone.utc).isoformat()

    temporary_system_prompt_update = runtime.context.temporary_system_prompt_update

    ai_context = ai_context_item.value['assistant_ctx'].get('metadata', {})
    user_context = ai_context_item.value['assistant_ctx'].get('metadata', {}).get("user", {})

    populated_template = prompt_builder.build_prompt(
        ai_context=ai_context,
        user_context=user_context, 
        retrieved_docs=retrieved_docs,
        retrieved_memories="",
        system_time = system_time,
        temporary_message=temporary_system_prompt_update,
    )

    logger.info(f"populated_template: {populated_template}")

    # clear the temporary injected prompt
    runtime.context.temporary_system_prompt_update = ""

    # prepend system message
    messages_input = populated_template.messages + state["messages"]
    
    logger.info(f"messages_input: {messages_input}")

    # TODO: Summarize messages

    response = await model.ainvoke(input=messages_input)

    logger.info(f"AGENT RESPONSE: {response}")
    result = {"messages": [response]}
    return result


async def summarize_conversation(state: GlobalState, runtime: Runtime[GlobalContext]):
    
    messages = state['messages']
    configuration = runtime.context.configuration


    # Retrieve the current message summary
    conversation_summary = state.get("conversation_summary", "") 
    if conversation_summary:

        # Extend the current conversation summary:
        summary_message = (
            f"This is the current summary of the conversation to the current message {conversation_summary}\n\n"
            +f"Extend this conversation summary with the included messages:"
        )
    else:
        summary_message = "Please create a summarization of the conversation using the included messages. Do not include this message as part of the summary"
    
    model = init_model(configuration=configuration)

    input_messages = [HumanMessage(context=summary_message)] + state['messages']

    # response = await model.ainvoke(input=HumanMessage(content=summary_message)


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

    model_router_structured_output = init_model(
        configuration=config,
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


""" GRAPH """

# Build minimal graph: START -> agent -> END
workflow = StateGraph(
    state_schema = GlobalState, 
    context_schema = GlobalContext
)

# Add single node (your input/output)
# workflow.add_node("summarize_conversation", summarize_conversation)
# workflow.add_node("call_router", call_router)
workflow.add_node("retrieve_documents", retrieval_graph)
workflow.add_node("invoke_agent", invoke_agent)

# Edges
workflow.add_edge(START, 'retrieve_documents')
workflow.add_edge('retrieve_documents', "invoke_agent")
workflow.add_edge("invoke_agent", END)
graph = workflow.compile()

graph.name = "Anubis"

__all__ = ["graph"]
