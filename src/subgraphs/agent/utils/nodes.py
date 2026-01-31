import logging
from typing import Literal
logger = logging.getLogger(__name__)

from src.subgraphs.agent.utils.state import AgentState
from src.subgraphs.agent.utils.model import init_model
from langgraph.graph import END
from langchain.agents import create_agent

model = init_model()

# agent = create_agent(model=model, tools=tools)

# def model_node(state: AgentState):
#     """LLM responds or chooses to use tools"""
#     logging.info(f"AGENT MODEL NODE: state['messages'][-1]: {state['messages'][-1]}")
#     return {"messages": [model.invoke(state["messages"])]}

# # Conditional edge tools if tool_calls, else end the loop
# def continue_tool_use_conditional(state: AgentState) -> Literal["tools", "__end__"]:
#     logger.info(f'CONTINUE TOOL USE CONDITIONAL')
#     last_msg = state["messages"][-1]
#     if last_msg.tool_calls:
#         return "tools"
#     return "__end__"


from datetime import datetime
from typing import cast


from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore

from src.subgraphs.memory_store_graph import utils

from src.subgraphs.memory_store_graph.utils import tools

import asyncio

from src.subgraphs.memory_store_graph.utils.context import Context
from src.subgraphs.memory_store_graph.utils.state import State
from langgraph.graph import END

async def call_model(state: State, runtime: Runtime[Context]) -> dict:
    """Extract the user's state from the conversation and update the memory."""
    user_id = runtime.context.user_id
    model = runtime.context.provider_model
    system_prompt = runtime.context.system_prompt

    # Retrieve the most recent memories for context
    memories = await cast(BaseStore, runtime.store).asearch(
        ("memories", user_id),
        query=str([m.content for m in state.messages[-3:]]),
        limit=10,
    )

    # Format memories for inclusion in the prompt
    formatted = "\n".join(
        f"[{mem.key}]: {mem.value} (similarity: {mem.score})" for mem in memories
    )
    if formatted:
        formatted = f"""
<memories>
{formatted}
</memories>"""

    # Prepare the system prompt with user memories and current time
    # This helps the model understand the context and temporal relevance
    sys = system_prompt.format(user_info=formatted, time=datetime.now().isoformat())

    # Load the chat model from the runtime context
    api_key = runtime.context.llama_api_key
    base_url = runtime.context.llama_api_base_url
    llm = utils.init_model(model, api_key, base_url)

    # Invoke the language model with the prepared prompt and tools
    # "bind_tools" gives the LLM the JSON schema for all tools in the list so it knows how
    # to use them.
    msg = await llm.bind_tools([tools.upsert_memory]).ainvoke(
        [{"role": "system", "content": sys}, *state.messages]
    )
    return {"messages": [msg]}


async def store_memory(state: State, runtime: Runtime[Context]):
    # Extract tool calls from the last message
    tool_calls = getattr(state.messages[-1], "tool_calls", [])

    # Concurrently execute all upsert_memory calls
    saved_memories = await asyncio.gather(
        *(
            tools.upsert_memory(
                **tc["args"],
                user_id=runtime.context.user_id,
                store=cast(BaseStore, runtime.store),
            )
            for tc in tool_calls
        )
    )

    # Format the results of memory storage operations
    # This provides confirmation to the model that the actions it took were completed
    results = [
        {
            "role": "tool",
            "content": mem,
            "tool_call_id": tc["id"],
        }
        for tc, mem in zip(tool_calls, saved_memories)
    ]
    return {"messages": results}


def route_message(state: State):
    """Determine the next step based on the presence of tool calls."""
    msg = state.messages[-1]
    if getattr(msg, "tool_calls", None):
        # If there are tool calls, we need to store memories
        return "store_memory"
    # Otherwise, finish; user can send the next message
    return END