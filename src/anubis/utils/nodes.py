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

from src.anubis.utils.tools import (
    health_check, 
    # add_to_vectorstore_subgraph, 
    # retrieve_from_vectorstore_subgraph, 
    # upsert_memory
)

# Optional: Add tools=[] if you have them
tools = [health_check]  # Replace with your tools

async def invoke_model(state: GlobalState, runtime: Runtime[GlobalContext]):
    """Single agent node: init model from context, bind tools, respond."""
    ctx = runtime.context
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

    prompt = ChatPromptTemplate.from_messages([
        ("system", GlobalConfiguration.response_system_prompt),
        ("placeholder", "{messages}"),
    ])

    retrieved_docs = format_docs(state['retrieved_docs'])

    ctx.assistant_ctx.name = "You are Elon Musk"

    # This is how prompt injection happens
    injected_prompt = await prompt.ainvoke(
        {
            "messages": state['messages'],
            "retrieved_docs": retrieved_docs,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
            "ai_context": ctx.assistant_ctx.name
        }
    )
    
    agent = create_agent(
        model=model, 
        tools = tools, 
        context_schema=GlobalContext, 
        state_schema=GlobalState,
    )

    response = await agent.ainvoke(input=injected_prompt)

    return response
