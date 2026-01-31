# src/anubis/utils/nodes.py
from src.anubis.utils.model import init_model
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.configuration import GlobalConfiguration
# from src.anubis.utils.tools import tools  # Your tools list (empty for now)
from src.anubis.utils.state import GlobalMessageState

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

import logging

logger = logging.getLogger(__name__)

# Optional: Add tools=[] if you have them
tools = []  # Replace with your tools

def agent_node(state: GlobalMessageState, runtime):
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

    # Format messages with system prompt
    response = model.invoke(state["messages"])
    logger.info(f"ctx.assistant_ctx.name: {ctx.assistant_ctx.name} ")

    return {"messages": [response]}
