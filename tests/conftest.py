
import sys
from pathlib import Path

# Add the project root to sys.path so imports work correctly
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.anubis.graph import anubis
import pytest


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"

