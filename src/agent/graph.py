
"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from datetime import UTC, datetime
from typing import Dict, List, Literal, cast

from langgraph.runtime import Runtime
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain.agents import create_agent

from langgraph.types import interrupt
from langchain_core.messages import HumanMessage


from src.agent.context import Context
from src.agent.state import InputState, State
from src.agent.tools import TOOLS
from src.agent.utils import init_model
from dataclasses import dataclass

import logging

logger = logging.getLogger(__name__)

model = init_model()

async def get_system_prompt(context: Context) -> str:
    """Async system prompt formatter."""
    system_time = datetime.now(tz=UTC).isoformat()
    return context.system_prompt.format(
        system_time=system_time,
        # name=context.name or "Assistant",
        # description=context.description or "",
        # facts="\n".join(context.facts) if context.facts else ""
    )

async def human_node(state: State) -> dict:
    prompt = {"pending_messages": state["messages"][-3], "step":state.get("is_last_step")}
    # human_input = interrupt({"Human Message": state["text"]})
    human_input = interrupt(prompt)
    return {"messages": [HumanMessage(content=str(human_input), name="human")]}



async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, List[AIMessage]]:
    """
    Invoke the AI model with the current conversation state and system prompt.
    
    This function formats and sends the system prompt along with conversation history
    to the AI model, retrieves the response, and handles tool call scenarios based on
    whether this is the final step in the execution flow.
    
    Args:
        state: The current state object containing:
            - messages: List of conversation messages to send to the model
            - is_last_step: Boolean indicating if this is the final execution step
        runtime: The runtime environment containing:
            - context: Context object with system_prompt attribute that can be formatted
    
    Returns:
        A dictionary with a single key "messages" containing a list with the AI's response
        as an AIMessage object. The response may include tool calls if the model decides
        to use tools.
    
    Note:
        If this is the last step (state.is_last_step is True) and the model attempts to
        make tool calls, the function returns the response immediately without executing
        the tools, effectively terminating the tool use cycle.
    
    Example:
        >>> result = await call_model(current_state, app_runtime)
        >>> ai_response = result["messages"][0]
    """
        
    # Format the system prompt
    # system_message = runtime.context.system_prompt.format(
    #     system_time=datetime.now(tz=UTC).isoformat(), 
    #     name="Evan", description="Software Engineer", facts="enjoys music"
    # )
    # system_message = await get_system_prompt(runtime.context)

    # Get the model's response
    # response = cast(
    #     AIMessage, 
    #    await model.invoke(
    #        [{"role": "system", "content": system_message}, *state.messages]
    #     ),
    # )

    response = await model.invoke([{"role": "system", "content": system_message}, *state.messages])

    if state.is_last_step and response.tool_calls:
        return {"messages": [response]}

def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """
    Determine the next node in the graph based on the model's output.
    
    This routing function examines the last message in the conversation state to decide
    whether to end the graph execution or proceed to tool execution. It validates that
    the last message is from the AI and routes based on whether tool calls are present.
    
    Args:
        state: The current state object containing:
            - messages: List of conversation messages, where the last message should be
              an AIMessage from the model
    
    Returns:
        A string literal indicating the next node:
        - "__end__": Terminate the graph execution (no tool calls present)
        - "tools": Proceed to tool execution node (tool calls are present)
    
    Raises:
        ValueError: If the last message in state.messages is not an AIMessage instance,
                   indicating an unexpected message type in the routing logic
    
    Example:
        >>> next_node = route_model_output(current_state)
        >>> if next_node == "tools":
        ...     # Execute tools
        >>> else:
        ...     # End conversation
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # if there is no tool call, complete the graph logic
    if not last_message.tool_calls or state.is_last_step:
        logger.info(f"state.is_last_step: {state.is_last_step}")

        return "__end__"
    # Otherwise we execute the requested actions
    return "tools"


# Nodes
# Graph Definition
# builder = StateGraph(State, input_schema=InputState, context_schema=Context)

# # entrypoint

# # Define the two nodes we will cycle between
# builder.add_node(call_model)
# builder.add_node("tools", ToolNode(TOOLS))

# # Entrypoint is human_node
# # builder.add_edge("__start__", "human_node")
# # builder.add_edge("human_node", "call_model")

# builder.add_edge("__start__", call_model)
# # determine tool use based upon the state
# builder.add_conditional_edges("call_model", route_model_output)
# # Normal edge from tools to call model
# builder.add_edge("tools", "call_model")


# builder.add_edge(call_model, "__end__")

# graph = builder.compile(name="Anubis")


# Edges
# Graph Definition
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing import Annotated, List, Literal, TypedDict
import operator
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    nlist: Annotated[List[str], operator.add]

def node_a(state:State) -> Literal["b", "c", END]:
    select = state["nlist"][-1]
    if select == "b":
        next_node = "b"
    elif select == "c":
        next_node = "c"
    else:
        next_node = END
    
    return Command(update = State(nlist = [select], goto = next_node))
    # message = "message in node_a"
    # state['nlist'].append(message)
    # logger.info(f"Adding 'A' to {state['nlist']}")
    # return(State(nlist = ["A"]))

def node_b(state: State) -> State:
    print(f"Adding 'B' to {state['nlist']}")
    return (State(nlist=["B"]))

def node_c(state: State) -> State:
    print(f"Adding 'C' to {state['nlist']}")
    return (State(nlist=["C"]))

def node_bb(state: State) -> State:
    print(f"Adding 'BB' to {state['nlist']}")
    return (State(nlist=["BB"]))

def node_cc(state: State) -> State:
    print(f"Adding 'CC' to {state['nlist']}")
    return (State(nlist=["CC"]))


def node_d(state: State) -> State:
    print(f"Adding 'D' to {state['nlist']}")
    return (State(nlist=["D"]))


# builder.add_node("a", node_a)
# builder.add_node("b", node_b)
# builder.add_node("c", node_c)
# builder.add_node("d", node_d)
# builder.add_node("bb", node_bb)
# builder.add_node("cc", node_cc)

# builder.add_edge(START, "a")
# builder.add_edge("a", "z")
# builder.add_edge("a", "c")
# builder.add_edge("b", "bb")
# builder.add_edge("bb", "d")
# builder.add_edge("c", "cc")
# builder.add_edge("cc", "d")
# builder.add_edge("d", END)

# graph = builder.compile()

# initial_state = State(nlist=["Initial String"])
# why is this never called
# result = graph.invoke(initial_state, )
# logger.info(f"XXXXXXXXXXXXXXXXXXXXXXXX RESULT: {result}")

# result2 = graph.invoke(None, config)
# builder.add_edge("a", END)
# logger.info(f"XXXXXXXXXXXXXXXXXXXXXXXX RESULT: {result2}")


# CONDITIONAL EDGES
from langgraph.graph import END
from typing import Literal

# def conditional_edge(state: State) -> Literal["b", "c", END]:
#     select = state["nlist"][-1]
#     if select == "b":
#         return "b"
#     elif select == "c":
#         return "c"
#     else:
#         return END
    

# builder = StateGraph(State)

# builder.add_node("a", node_a)
# builder.add_node("b", node_b)
# builder.add_node("c", node_c)

# # Add edges
# builder.add_edge(START, "a")
# builder.add_edge("b", END)
# builder.add_edge("c", END)


# # builder.add_conditional_edges("a", conditional_edge)


# graph = builder.compile()

# while True:
#     user = input('test input please')
#     print(user)

#     input_state = State(
#         nlist = [user]
#     )

#     result = graph.invoke(input_state)
#     print(result)
#     if result['nlist'][-1] == 'q':
#         print('quit')
#         break

# agent level use case
# user_id = "test_user"


# @dataclass
# class Context:
#     user_id: str

# graph = create_agent(model = model, tools = [])


from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from langgraph.graph.message import add_messages # Built-in reducer

tools = []
model = init_model()

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages] # enables append/update

def agent_node(state: AgentState):
    """LLM responds or chooses to use tools"""
    return {"messages": [model.invoke(state["messages"])]}

# Conditional edge tools if tool_calls, else end the loop
def continue_tool_use_conditional(state: AgentState):
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return END

# Build graph
workflow = StateGraph(state_schema=AgentState)

# Define Nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools)) # tool use is parallel

# Entrypoint of graph
workflow.set_entry_point("agent")

# Edge Definitions
workflow.add_conditional_edges("agent", continue_tool_use_conditional)
workflow.add_edge("tools", "agent")
graph = workflow.compile()
