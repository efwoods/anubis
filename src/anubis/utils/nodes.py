import logging
logger = logging.getLogger(__name__)

from src.anubis.utils.state import AgentState
from langgraph.graph import END
from src.anubis.utils.utilities import init_model
from langchain.agents import create_agent
from src.anubis.utils.tools import search, get_chat_metadata

tools = [search, get_chat_metadata]

model = init_model()
agent = create_agent(model=model, tools=tools)

def agent_node(state: AgentState):
    """LLM responds or chooses to use tools"""
    logging.info(state["messages"][-1])
    return {"messages": [model.invoke(state["messages"])]}

# Conditional edge tools if tool_calls, else end the loop
def continue_tool_use_conditional(state: AgentState):
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return END