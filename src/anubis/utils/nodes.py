# src/anubis/utils/nodes.py
from src.anubis.utils.model import init_model
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.configuration import GlobalConfiguration
# from src.anubis.utils.tools import tools  # Your tools list (empty for now)
from src.anubis.utils.state import GlobalMessageState

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.runtime import Runtime   
import logging

logger = logging.getLogger(__name__)

from src.anubis.utils.tools import (
    health_check, 
    # add_to_vectorstore_subgraph, 
    # retrieve_from_vectorstore_subgraph, 
    # upsert_memory
)

from langchain.agents import create_agent

# Optional: Add tools=[] if you have them
tools = [health_check]  # Replace with your tools

async def invoke_model(state: GlobalMessageState, runtime: Runtime[GlobalContext]):
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

    system_prompt = "You are Elon Musk"
    
    agent = create_agent(
        model=model, 
        tools = tools, 
        context_schema=GlobalContext, 
        state_schema=GlobalMessageState,
        system_prompt=system_prompt
    )

    response = await agent.ainvoke(input=state)

    return response