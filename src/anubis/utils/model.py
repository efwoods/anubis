# src/anubis/utils/model

# from src.anubis.utils.tools import search, health_check, add_to_vectorstore, retrieve_from_vectorstore

# tools = [search, health_check, add_to_vectorstore, retrieve_from_vectorstore]

from pydantic import BaseModel


class Address(BaseModel):
    street: str
    city: str
    state: str
    zip: str



def init_model(provider_model, base_url, api_key, tools=[], dev="TRUE", response_format = None):
    provider, model_name = provider_model.split("/", maxsplit=1) 
    # if dev == 'TRUE':
    from langchain_openai import ChatOpenAI
    if response_format is None:
        model = ChatOpenAI(
                    model = model_name,
                    base_url = base_url,
                    temperature=0.1,
                    api_key = api_key,
                ).bind_tools(tools=tools)
    else: 
        model = ChatOpenAI(
            model = model_name,
            base_url = base_url,
            temperature=0.1,
            api_key = api_key,
            response_format=response_format
        ).bind_tools(tools=tools)
    # else: 
    #     from langchain_together import ChatTogether
    #     model = ChatTogether(model=model_name, temperature=0.1)
    return model

from src.anubis.utils.classes.DynamicPromptBuilder import DynamicPromptBuilder
from src.anubis.utils.helper_functions import format_docs
from datetime import datetime, timezone
from src.anubis.utils.state import GlobalState
from src.anubis.utils.context import GlobalContext
from langgraph.runtime import Runtime
from langchain.agents import create_agent

import logging

logger = logging.getLogger(__name__)

async def invoke_model_core(
        state: GlobalState, 
        runtime: Runtime[GlobalContext],
        tools: list,
        ):
    """Build a model, agent, and dynamic system prompt to load the identity of the assistant into the assistant's current state of consciousness"""
    
    logger.info(f"invoke model core")

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



