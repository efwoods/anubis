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

    # build system prompt with injection
    # search store for current context information
    # update the context
    # inject the system prompt with context from user and assistant

    prompt_builder = DynamicPromptBuilder()

    retrieved_docs = format_docs(state.get('retrieved_docs', []))

    ai_context = runtime.context.assistant_ctx.to_dict()
    user_ctx = runtime.context.user_ctx.to_dict()
    system_time = datetime.now(tz=timezone.utc).isoformat()

    temporary_system_prompt_update = runtime.context.temporary_system_prompt_update

    prompt_template, prompt_variables = prompt_builder.build_prompt(
        ai_context=ai_context,
        user_context=user_ctx, 
        retrieved_docs=retrieved_docs,
        system_time = system_time,
        temporary_message=temporary_system_prompt_update,
    )
    
    runtime.context.temporary_system_prompt_update = ""

    # Inject and create the system prompt and append messages of state
    # injected_prompt = await prompt_template.ainvoke({
    #     **prompt_variables, 
    #     "messages": state['messages']
    # })

    # logger.info(f"INJECTED PROMPT: {injected_prompt}")
    
    # agent = create_agent(
    #     model=model, 
    #     tools = tools, 
    #     context_schema=GlobalContext, 
    #     state_schema=GlobalState,
    # )
    injected_prompt = [SystemMessage(content="This is a system message")] + state['messages']
    

    response = await model.ainvoke(input=injected_prompt)

    logger.info(f"AGENT RESPONSE: {response}")
    result = {"messages": [response]}
    return result

# upload_document: str
    # learn_about_identity_q_and_a: str

class RouteDecision:
    chat: str

async def call_router(state: GlobalState, runtime: Runtime[GlobalContext]):
    """Decides whether to upload new personal information or to respond

    Args:
        state (GlobalState): message state
        runtime (Runtime[GlobalContext]): context of agent and user and configuration

    Returns:
        str: next node location
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

    decision_instructions = """Route the input to upload a document, learn about your identity, or respond to a chat.
"""
    system_message = SystemMessage(content=decision_instructions)
    
    assert isinstance(state['messages'][-1], HumanMessage)
    assert hasattr(state['messages'][-1], "content")
    assert getattr(state['messages'][-1], "content") != ""

    human_message = state['messages'][-1]

    decided_route = model_router_structured_output.ainvoke(
        [system_message, human_message]
    )
    
    return {"route_decision": decided_route}


async def route_node_from_decision(state: GlobalState, runtime: GlobalContext) -> str:
    logger.info(f"ROUTING NODE FROM DECISION")
    route_decision = state["route_decision"]
    if route_decision == "chat":
        return "invoke_agent"
    else:
        return "invoke_agent"


































    logger.info(f"INVOKE MODEL NODE {response['messages']}")

    result = {"messages": [response["messages"]]}
    logger.info(f"RESULT OF INVOKE MODEL NODE RETURN: {result}")

    return result
