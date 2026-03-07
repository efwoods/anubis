from langgraph.pregel import Pregel


from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.append(project_root)

import pytest
from src.anubis.graph import graph

from langchain_core.messages import HumanMessage

from src.anubis.utils.context import GlobalContext, UserContext, AssistantContext
from langgraph.runtime import Runtime

class TestGraphIntegration:
    """Integration tests for the complete graph workflow."""

    def test_graph_compiles_successfully(self):
        """Test that the complete graph compiles without errors."""
        assert graph is not None
        assert graph.name == "Anubis"

    def test_graph_has_expected_structure(self):
        """Test the compiled graph has the expected structure."""
        # The graph should be a runnable
        assert hasattr(graph, 'invoke') or hasattr(graph, 'ainvoke')

    @pytest.mark.anyio
    async def test_graph_runs_with_message(self):
        """Test that the compiled graph runs end-to-end with real services."""
        # Input: List of HumanMessage objects

        input_messages = {"messages": [HumanMessage(content="How are you?")]}
        
        # Config with required configurable keys for the graph runtime
        config = {
            "configurable": {
                "assistant_id": "test_assistant",
                "langgraph_auth_user_id": "test_user",  
            }
        }

        assistant_ctx = AssistantContext()
        user_ctx = UserContext()
        context = Runtime(context = GlobalContext(user_ctx=user_ctx, assistant_ctx=assistant_ctx))
        
        # Invoke the compiled graph - no mocks, runs with real services
        response = await graph.ainvoke(input=input_messages, config=config, runtime=context)
        
        # Assertions: verify the graph ran successfully
        assert response is not None
        assert "messages" in response
        assert len(response["messages"]) > 0
        
        # Verify the response contains messages
        assert any(isinstance(msg, HumanMessage) for msg in response["messages"])

          