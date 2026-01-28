
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

from src.agent.context import Context
from src.agent.state import InputState, State
from src.agent.tools import TOOLS
from src.agent.utils import init_model

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
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    return "tools"



def llm_call(state: State):
    """LLM decides when to use a tool"""

    return  {'messages': [
        model.invoke([SystemMessage(content="You are a helpful assistant")] + state["messages"])

    ], 
    }

# Graph Definition
builder = StateGraph(State, input_schema=InputState, context_schema=Context)

# Define the two nodes we will cycle between
builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))

# Entrypoint is call_model
builder.add_edge("__start__", "call_model")

builder.add_conditional_edges("call_model", route_model_output)

# Normal edge from tools to call model
builder.add_edge("tools", "call_model")

graph = builder.compile(name="Anubis")
