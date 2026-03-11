# src/anubis/integration_tests/test_graph.py

import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langgraph.graph import MessagesState

from src.anubis.utils.state import GlobalState
from src.anubis.utils.context import GlobalContext
from src.anubis.graph import graph, message_workflow

@pytest.mark.asyncio
async def test_anubis_graph_integration():

    store = InMemoryStore()
    
    # Set up real store and context
    context = GlobalContext(assistant_ctx={"name": "TestAssistant"}, user_ctx={"name": "TestUser"})
    runtime = Runtime(store=store, context=context)
    
    # Prepare input state
    input_state = MessagesState(messages=[HumanMessage(content="Hello, how are you?")])
    
    # Prepare config
    config = RunnableConfig(configurable={"assistant_id": "test_assistant", "langgraph_auth_user_id": "test_user"})
    
    graph = message_workflow.compile(store=store)

    # Invoke the graph
    result = await graph.ainvoke(input_state, config=config, runtime=runtime)
    
    # Assert output is MessagesState with a response
    assert isinstance(result, MessagesState)
    assert len(result["messages"]) > 1  # At least the input plus a response
    assert isinstance(result["messages"][-1], (HumanMessage, AIMessage))  # Last message should be a response