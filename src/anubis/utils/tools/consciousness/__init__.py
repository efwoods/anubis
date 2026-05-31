"""In-agent tools that let the deep agent recompose the avatar's system prompt mid-run."""

from src.anubis.utils.tools.consciousness.load_consciousness_tool import (
    LOAD_CONSCIOUSNESS_TOOL_NAME,
    load_consciousness_tool,
)

__all__ = ["LOAD_CONSCIOUSNESS_TOOL_NAME", "load_consciousness_tool"]
