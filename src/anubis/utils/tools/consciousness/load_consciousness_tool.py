"""`load_consciousness` tool — the in-agent twin of the outer-graph node.

The deep agent calls this tool after any batch of identity-mutating tools so
its next LLM turn sees a freshly-rebuilt system prompt that includes whatever
identity facts, memories, and quotes were just persisted. It is the exact
same prompt-builder the outer graph's ``load_consciousness`` node uses; the
shared helper lives in :mod:`src.anubis.utils.nodes` so both call sites
produce byte-identical output.

Wiring summary:

- ``ConsciousnessRefreshGate`` middleware injects a synthetic tool call to
  this tool after any successful identity-tool batch.
- This tool returns ``Command(update=...)`` so the deep agent's state picks
  up the new ``system_message`` (pinned ID lets ``add_messages`` replace,
  not append) and the refreshed identity-doc snapshots.
- ``DynamicConsciousnessPromptMiddleware`` then reads
  ``state['system_message'][-1]`` and overrides ``request.system_message`` on
  the next LLM call.
"""

from __future__ import annotations

import logging
from typing import Annotated

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolArg
from langgraph.types import Command

from src.anubis.utils.nodes import _build_consciousness_system_message_update

logger = logging.getLogger(__name__)


LOAD_CONSCIOUSNESS_TOOL_NAME = "load_consciousness"


@tool(LOAD_CONSCIOUSNESS_TOOL_NAME)
async def load_consciousness_tool(
    runtime: Annotated[ToolRuntime, InjectedToolArg] = None,
):
    """Rebuild the avatar's system prompt from the latest identity store contents.

    Call this tool ONLY after any other identity-mutating tool has just
    finished — never on its own initiative. The system will normally inject
    this call automatically; explicit invocation should be reserved for
    cases where the avatar wants to force a fresh consciousness snapshot
    before its next reasoning step.

    Returns a state update containing:
    - ``system_message``: the regenerated ``SystemMessage`` (pinned id so it
      replaces rather than appends).
    - ``user_identity_documents`` / ``assistant_identity_documents``: the
      identity-doc lists used to build the prompt.
    - ``messages``: a ``ToolMessage`` closing this tool call.
    """
    tool_call_id = runtime.tool_call_id

    update = await _build_consciousness_system_message_update(
        runtime.state, runtime.config, runtime
    )

    update["messages"] = [
        ToolMessage(
            content="Consciousness reloaded: system prompt rebuilt with the latest identity, memory, quote, and knowledge documents.",
            tool_call_id=tool_call_id,
        )
    ]

    logger.info("load_consciousness tool refreshed system prompt for deep agent")
    return Command(update=update)
