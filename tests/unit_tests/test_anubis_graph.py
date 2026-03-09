"""
Unit tests for the Anubis graph workflow.

Tests cover:
- Graph structure and node existence
- Edge connections and routing
- Conditional edge behavior
- Graph compilation
"""

# Add project root to path
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(project_root)

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import START, END

from src.anubis.graph import (
    anubis_workflow,
    anubis,
    load_consciousness,
    invoke_agent,
    avatar_tool_node,
    avatar_tools_condition,
    message_workflow,
    graph,
)
from src.anubis.utils.state import GlobalState
from src.anubis.utils.context import GlobalContext, UserContext, AssistantContext
from langgraph.store.memory  import InMemoryStore
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

class TestAvatarToolsCondition:
    """Test the avatar_tools_condition routing function."""

    @pytest.mark.anyio
    async def test_condition_with_tool_calls(self):
        """Test condition returns avatar_tool_node when tool_calls exist."""
        # Create mock state with tool calls
        mock_message = AIMessage(content="test", tool_calls=[{"id": "1", "name": "test_tool", "args": {}}])


        config = RunnableConfig()
        runtime = Runtime(
            context=GlobalContext(
                user_ctx=UserContext(), 
                assistant_ctx=AssistantContext()
            ),
            store = InMemoryStore(index= {
                        "dims": 384,
                        "embed": "huggingface:sentence-transformers/all-MiniLM-l6-v2",
                        "fields": ["document.kwargs.page_content"]
                    })
        )
        state = GlobalState(messages=[mock_message])
        result = await avatar_tools_condition(state, config, runtime)
        assert result == "avatar_tool_node"

    @pytest.mark.anyio
    async def test_condition_without_tool_calls(self):
        """Test condition returns __end__ when no tool_calls exist."""
        # Create mock state without tool calls
        mock_message = AIMessage(content="test response")
        
        state = {
            "messages": [mock_message],
            "assistant_state": {},
            "user_state": {},
        }
        
        config = RunnableConfig()
        runtime = Runtime(
            context=GlobalContext(
                user_ctx=UserContext(), 
                assistant_ctx=AssistantContext()
            ),
            store = InMemoryStore(index= {
                        "dims": 384,
                        "embed": "huggingface:sentence-transformers/all-MiniLM-l6-v2",
                        "fields": ["document.kwargs.page_content"]
                    })
        )


        state = GlobalState(messages=[mock_message])
        result = await avatar_tools_condition(state, config, runtime)
        assert result == "__end__"
        print(f"RESULT: {result}")

class TestLoadConsciousnessNode:
    """Test the load_consciousness node functionality."""

    @pytest.mark.anyio
    async def test_load_consciousness_basic_execution(self):
        """Test load_consciousness node executes without error."""

        mock_state = {
            "messages": [HumanMessage(content="This is a test input message.")],
            "system_message": "You are a helpful assistant.",
            "user_state": {"user_id": "test_user_123"},
            "assistant_state": {"assistant_id": "test_assistant_123"},
            "user_identity_documents": [],
            "assistant_identity_documents": [],
            "recalled_memory_documents": [],
        }
        
        config = RunnableConfig()
        runtime = Runtime(
            context=GlobalContext(
                user_ctx=UserContext(), 
                assistant_ctx=AssistantContext()
            ),
            store = InMemoryStore(index= {
                        "dims": 384,
                        "embed": "huggingface:sentence-transformers/all-MiniLM-l6-v2",
                        "fields": ["document.kwargs.page_content"]
                    })
        )

        state = GlobalState()
        state['messages'] = [HumanMessage(content="This is a test message")]

        namespaces = await runtime.store.alist_namespaces()        
        assert namespaces is not None

        result = await load_consciousness(mock_state, config=config, runtime=runtime)
                
        assert result is not None
        assert "system_message" in result
        assert "user_identity_documents" in result
        assert "assistant_identity_documents" in result

        print(f"result:{result}")

class TestInvokeAgentNode:
    """Test the invoke_agent node functionality."""

    @pytest.mark.anyio
    async def test_invoke_agent_basic_execution(self):
        """Test invoke_agent node executes and returns response."""
        mock_state = {
            "messages": [HumanMessage(content="Hello")],
            "system_message": "You are a helpful assistant.",
            "user_state": {"user_id": "test_user_123"},
            "assistant_state": {"assistant_id": "test_assistant_123"},
            "user_identity_documents": [],
            "assistant_identity_documents": [],
            "recalled_memory_documents": [],
        }
        
        config = RunnableConfig()
        runtime = Runtime(context=GlobalContext(user_ctx=UserContext(), assistant_ctx=AssistantContext()))

        state = GlobalState(messages = [HumanMessage(content="This is a test message")])
        
        result = await invoke_agent(state=mock_state, config=config, runtime=runtime)

        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) > 0
        print(f'result: {result}')  # Print to stdout so it shows in pytest output

# class TestAvatarToolNodeExecution:
#     """Test the avatar_tool_node execution."""

#     @pytest.mark.anyio
#     async def test_avatar_tool_node_with_valid_tool(self):
#         """Test avatar_tool_node executes valid tool calls."""
#         mock_tool_call = {
#             "id": "call_123",
#             "name": "recall_memories",
#             "args": {"query": "memories"},
#         }
        
#         mock_message = AIMessage(content='', additional_kwargs={}, response_metadata={'finish_reason': 'tool_calls', 'model_name': 'Llama-4-Maverick-17B-128E-Instruct-FP8', 'model_provider': 'openai'}, id='lc_run--019cc603-e0d4-7b00-bee2-a8fa7b98d736', tool_calls=[{'name': 'recall_memories', 'args': {}, 'id': '6bb3bf99-c951-4322-83f9-c8ea4bd8671b', 'type': 'tool_call'}], invalid_tool_calls=[])
             
        
#         mock_state = {
#             "messages": [HumanMessage(content="Hello"), mock_message],
#             "system_message": "You are a helpful assistant.",
#             "user_state": {"user_id": "test_user_123"},
#             "assistant_state": {"assistant_id": "test_assistant_123"},
#             "user_identity_documents": [],
#             "assistant_identity_documents": [],
#             "recalled_memory_documents": [],
#         }
        
#         config = RunnableConfig()
#         runtime = Runtime(context=GlobalContext(user_ctx=UserContext(), assistant_ctx=AssistantContext()))

#         state = GlobalState(messages = [HumanMessage(content="This is a test message")])
                    
#         result = await avatar_tool_node(mock_state, config, runtime)
            
#         # Should return a Command object
#         assert result is not None

class TestWorkflowLoopPrevention:
    """Test that the workflow loop between load_consciousness and avatar_tool_node works correctly."""

    @pytest.mark.anyio
    async def test_loop_terminates_without_tool_calls(self):
        """Test that the loop terminates when no more tool calls are made."""
        # After avatar_tool_node executes and loops back to load_consciousness,
        # the next invoke_agent should return a message without tool_calls,
        # causing the condition to return __end__
        
        mock_message_with_tools = AIMessage(
            content="", tool_calls=[{"id": "1", "name": "test", "args": {}}]
        )
        mock_message_without_tools = AIMessage(content="Final response")
        
        # First condition check (with tools)
        state1 = {"messages": [mock_message_with_tools]}
        result1 = await avatar_tools_condition(state1, MagicMock(), MagicMock())
        assert result1 == "avatar_tool_node"
        
        # Second condition check (without tools)
        state2 = {"messages": [mock_message_without_tools]}
        result2 = await avatar_tools_condition(state2, MagicMock(), MagicMock())
        assert result2 == "__end__"
