
import os
import sys
from pathlib import Path

# Unit tests never send LangSmith traces: every traced test run used to count
# against the LangSmith organization's monthly billable traces. The environment
# variables are set before any langsmith or langchain import reads them.
for langsmith_tracing_variable in (
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_TRACING_V2",
):
    os.environ[langsmith_tracing_variable] = "false"

# Add the project root to sys.path so imports work correctly
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.anubis.graph import anubis
import pytest


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"



@pytest.fixture(autouse=True)
def clean_concurrent_moderation_screen(monkeypatch):
    """Keep the streaming endpoint's concurrent moderation screen off the network.

    ``message_graph_sse`` screens every fresh typed turn beside the reply
    (``screen_message_for_hard_block``), which calls the OpenAI moderation
    endpoint. Unit tests answer clean by default; a test exercising the screen
    replaces the function again with its own verdict.
    """
    import src.anubis.graph as graph_module

    async def clean_screen(message_text, context, setting=None):
        return None

    monkeypatch.setattr(graph_module, "screen_message_for_hard_block", clean_screen)


@pytest.fixture(autouse=True, scope="session")
def langsmith_tracing_off():
    """Turn LangSmith tracing off for the test process even when a .env file turned tracing on."""
    from langsmith.run_trees import configure as configure_langsmith

    configure_langsmith(enabled=False)
    yield
