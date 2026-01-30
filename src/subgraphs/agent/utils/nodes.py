import logging
from typing import Literal
logger = logging.getLogger(__name__)

from src.subgraphs.agent.utils.state import AgentState
from src.subgraphs.agent.utils.utilities import init_model
from langgraph.graph import END
from langchain.agents import create_agent

model = init_model()

# agent = create_agent(model=model, tools=tools)

def model_node(state: AgentState):
    """LLM responds or chooses to use tools"""
    logging.info(state["messages"][-1])
    return {"messages": [model.invoke(state["messages"])]}

# Conditional edge tools if tool_calls, else end the loop
def continue_tool_use_conditional(state: AgentState) -> Literal["tools", "__end__"]:
    logger.info(f'CONTINUE TOOL USE CONDITIONAL')
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return "__end__"