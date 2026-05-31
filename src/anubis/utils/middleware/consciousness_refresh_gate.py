"""Middleware that auto-injects a ``load_consciousness`` tool call after identity-tool batches.

The user-facing contract: any time the deep agent uses one or more identity-
mutating tools (those that change what the consciousness loader would
return), the next LLM turn must run against a freshly-rebuilt system prompt.
We enforce that by intercepting ``before_model`` after a tool batch
completes:

1. Look back over the trailing ``ToolMessage`` run.
2. If any of those ToolMessages closed a call to a configured identity tool,
   AND the most recent AIMessage's tool batch is fully resolved (every
   tool_call has a matching ToolMessage),
3. Inject a synthetic ``AIMessage`` with a single tool call to
   ``load_consciousness`` and ``jump_to="tools"`` so LangGraph routes
   straight to the tools node instead of asking the model what to do next.

The tools node then runs ``load_consciousness_tool``, which updates
``state['system_message']``. The next model turn fires with the new prompt
courtesy of ``DynamicConsciousnessPrompt``.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


class ConsciousnessRefreshGate(AgentMiddleware):
    """Force a ``load_consciousness`` call after every identity-tool batch.

    Args:
        identity_tool_names: Tool names whose completion should trigger a
            consciousness refresh.
        load_consciousness_tool_name: Tool name used for the synthetic
            refresh call. Defaults to ``"load_consciousness"``.
    """

    def __init__(
        self,
        identity_tool_names: Iterable[str],
        load_consciousness_tool_name: str = "load_consciousness",
    ) -> None:
        super().__init__()
        self._identity_tool_names = frozenset(identity_tool_names)
        self._load_consciousness_tool_name = load_consciousness_tool_name

    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return "ConsciousnessRefreshGate"

    def _should_refresh(self, state: dict[str, Any]) -> bool:
        messages = state.get("messages") if isinstance(state, dict) else None
        if not messages:
            return False

        # Walk back over consecutive trailing ToolMessages to capture the most
        # recent tool-execution batch the agent just finished.
        trailing_tool_messages: list[ToolMessage] = []
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                trailing_tool_messages.append(msg)
            else:
                break
        if not trailing_tool_messages:
            return False

        # Find the AIMessage that originated the trailing tool batch.
        last_ai: AIMessage | None = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last_ai = msg
                break
        if last_ai is None:
            return False

        ai_tool_calls = list(getattr(last_ai, "tool_calls", None) or [])
        if not ai_tool_calls:
            return False

        # Bail out if this AIMessage already kicked off a consciousness refresh
        # — the trailing ToolMessage(s) will be its response, and we don't want
        # to schedule another one back-to-back.
        if any(
            call.get("name") == self._load_consciousness_tool_name
            for call in ai_tool_calls
        ):
            return False

        # Only refresh if every tool_call from the originating AIMessage has a
        # matching ToolMessage (i.e., the batch is fully resolved). Otherwise
        # we'd inject in the middle of a partially-complete tool batch.
        ai_call_ids = {
            call.get("id") for call in ai_tool_calls if call.get("id") is not None
        }
        responded_ids = {
            getattr(tm, "tool_call_id", None) for tm in trailing_tool_messages
        }
        if not ai_call_ids.issubset(responded_ids):
            return False

        # Only refresh if at least one of those tool_calls targeted an
        # identity-mutating tool.
        return any(
            call.get("name") in self._identity_tool_names for call in ai_tool_calls
        )

    def _build_refresh_command(self) -> dict[str, Any]:
        tool_call_id = f"call_{uuid.uuid4().hex}"
        synthetic_ai = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": self._load_consciousness_tool_name,
                    "args": {},
                    "id": tool_call_id,
                    "type": "tool_call",
                }
            ],
        )
        logger.info(
            "ConsciousnessRefreshGate: scheduling synthetic load_consciousness tool call id=%s",
            tool_call_id,
        )
        return {"messages": [synthetic_ai], "jump_to": "tools"}

    @hook_config(can_jump_to=["tools"])
    def before_model(  # type: ignore[override]
        self, state: dict[str, Any], runtime: Runtime
    ) -> dict[str, Any] | None:
        if not self._should_refresh(state):
            return None
        return self._build_refresh_command()

    @hook_config(can_jump_to=["tools"])
    async def abefore_model(  # type: ignore[override]
        self, state: dict[str, Any], runtime: Runtime
    ) -> dict[str, Any] | None:
        if not self._should_refresh(state):
            return None
        return self._build_refresh_command()
