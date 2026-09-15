"""Middleware that swaps the deep agent's system prompt for the latest consciousness snapshot.

The outer ``load_consciousness`` node, and the in-agent ``load_consciousness``
tool, both write a ``SystemMessage`` pinned to a fixed id into
``state['system_message']`` (so ``add_messages`` replaces rather than
appends). This middleware reads ``state['system_message'][-1]`` and forwards
it as ``request.system_message`` on every LLM call, guaranteeing the model
always sees the most recent identity, memory, quote, and knowledge context
— including any updates a tool just persisted in the same agent turn.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import SystemMessage

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.context_compression import (
    estimate_messages_token_count,
    estimate_tool_schema_tokens,
    truncate_string_to_token_limit,
)
from src.anubis.utils.model import hosted_inference_input_token_limit
from src.anubis.utils.tokenizer import count_tokens


class DynamicConsciousnessPrompt(AgentMiddleware):
    """Override ``request.system_message`` with the freshest entry from ``state['system_message']``."""

    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return "DynamicConsciousnessPrompt"

    @staticmethod
    def _latest_system_message(state: dict[str, Any]) -> SystemMessage | None:
        msgs = state.get("system_message") if isinstance(state, dict) else None
        if not msgs:
            return None
        latest = msgs[-1]
        if isinstance(latest, SystemMessage):
            return latest
        # Defensive fallback if a downstream reducer coerced into a dict.
        content = getattr(latest, "content", None)
        if isinstance(content, str):
            return SystemMessage(content=content)
        return None

    def _apply(self, request: ModelRequest) -> ModelRequest:
        latest = self._latest_system_message(request.state)
        if latest is None:
            return request
        context = GlobalContext()
        window = hosted_inference_input_token_limit(context)
        content = latest.content if isinstance(latest.content, str) else str(latest.content or "")
        prompt_tokens = count_tokens(content)
        tool_tokens = estimate_tool_schema_tokens(getattr(request, "tools", None))
        message_tokens = estimate_messages_token_count(
            list(getattr(request, "messages", None) or [])
        )
        overhead_tokens = 2048
        prompt_budget = max(
            4096, window - tool_tokens - message_tokens - overhead_tokens
        )
        # Truncate only for the small NVIDIA 90B NIM window. A 400k
        # MODEL_TOKEN_LIMIT must leave the identity prompt intact.
        truncated = window <= 32768 and prompt_tokens > prompt_budget
        if truncated:
            content = truncate_string_to_token_limit(content, prompt_budget)
            message_id = getattr(latest, "id", None)
            if message_id:
                latest = SystemMessage(content=content, id=message_id)
            else:
                latest = SystemMessage(content=content)
        return request.override(system_message=latest)

    def wrap_model_call(  # type: ignore[override]
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._apply(request))

    async def awrap_model_call(  # type: ignore[override]
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._apply(request))
