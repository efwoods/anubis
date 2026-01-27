# anubis.py
import os
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from core.logging import logger
from langgraph.graph.message import MessagesState
from langchain.agents import create_agent
from langgraph.graph import StateGraph, MessagesState, START, END

from prompts.system_prompt import SYSTEM_PROMPT
from classes.Objects import AvatarContext

if os.getenv("DEV") == 'TRUE':
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model = os.getenv("MODEL"),
        base_url = os.getenv("LLAMA_API_BASE_URL"),
        temperature=0.1,
        api_key = os.environ.get("LLAMA_API_KEY"),
    )
else: 
    from langchain_together import ChatTogether
    llm = ChatTogether(model=os.getenv("MODEL"), temperature=0.1)


agent = create_agent(llm, tools=[])




# def call_agent(state: MessagesState):
#     """Node calls your agent"""
#     result = agent.invoke({"messages": state["messages"]})
#     return {"messages": result["messages"]}



__all__ = ["agent"]














# class State(TypedDict):
#     messages: list[BaseMessage]


# from llama_api_client import LlamaAPIClient
# client = LlamaAPIClient()


# def agent_node(state: State):
#     messages = [{"role": m.role, "content": m.content} for m in state['messages']]
#     response = client.chat.completions.create(
#         model=os.getenv("MODEL"),
#         messages=messages,
#         max_completion_tokens=1024,
#         temperature=0.1,
#     )
#     content = response.completion_message.content.text
#     ai_message = AIMessage(content=content)

#     return {"messages": [ai_message]}

# graph = (StateGraph(MessagesState)
#          .add_node("agent", agent_node)
#          .add_edge(START, "agent")
#          .add_edge("agent", END)
#          .compile())
         

# # Export graph for api
# __all__ = ["graph"] 