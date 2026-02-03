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
tools = [health_check]  # Replace with your tools 

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
    injected_prompt = await prompt_template.ainvoke({
        **prompt_variables, 
        "messages": state['messages']
    })

    logger.info(f"INJECTED PROMPT: {injected_prompt}")
    
    agent = create_agent(
        model=model, 
        tools = tools, 
        context_schema=GlobalContext, 
        state_schema=GlobalState,
    )

    response = await agent.ainvoke(input=injected_prompt)

    logger.info(f"AGENT RESPONSE: {response}")
    result = {"messages": response['messages'][-1]}
    return result


































    logger.info(f"INVOKE MODEL NODE {response['messages']}")

    result = {"messages": [response["messages"]]}
    logger.info(f"RESULT OF INVOKE MODEL NODE RETURN: {result}")

    return result
