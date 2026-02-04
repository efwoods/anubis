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

async def invoke_agent(state: GlobalState, runtime: Runtime[GlobalContext]):
    """Build a model, agent, and dynamic system prompt to load the identity of the assistant into the assistant's current state of consciousness"""
    logger.info(f"INVOKE AGENT NODE ")
    
    config = runtime.context.configuration # Loads env vars automatically

    model = init_model(
        config.provider_model,
        config.llama_api_base_url,
        config.llama_api_key,
        tools,
        config.dev
    )

    # Retrieve documents for the query
    from src.subgraphs.vector_store_graph.retrieval_graph import retrieval_graph

    human_message = state['messages'][-1]

    assert(isinstance(human_message, HumanMessage))
    
    retrieval_message = {"messages" : [human_message]}

    new_state_retrieved_docs = await retrieval_graph.ainvoke(retrieval_message, context=runtime.context)
    
    # populate the relevant documents with a new state
    state['retrieved_docs'] = new_state_retrieved_docs['retrieved_docs']

    # Vectorstore Retrieved Docments
    retrieved_docs = format_docs(state.get('retrieved_docs', []))

    prompt_builder = DynamicPromptBuilder()

    # TODO: Update the assistant context from the state

    # Load the current assistant context for prompt injection
    ai_context = runtime.context.assistant_ctx.to_dict()

    # TODO: Update the user context from the state

    # Load the current user context from prompt injection
    user_ctx = runtime.context.user_ctx.to_dict()

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
