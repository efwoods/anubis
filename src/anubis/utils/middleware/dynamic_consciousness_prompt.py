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
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.context_compression import (
    estimate_messages_token_count,
    estimate_tool_schema_tokens,
    truncate_string_to_token_limit,
)
from src.anubis.utils.model import hosted_inference_input_token_limit
from src.anubis.utils.prompts.system_prompts import (
    PER_TURN_SECTION_START_KEY,
    ROLE_SECTION_OPENING_TAG,
)
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

    @staticmethod
    def _truncate_preserving_role(content: str, prompt_budget: int) -> str:
        """Trim the fixed instructions rather than the avatar's identity facts.

        ``load_consciousness`` assembles the prompt fixed-text-first so the
        fixed stretch forms a cacheable prefix, which puts the ROLE section —
        identity facts, retrieved memories, direct quotes, the system time —
        last. Trimming from the end would therefore cut exactly the material
        the reply is built from and keep the boilerplate, so the trim falls on
        the fixed half and the ROLE section is kept whole whenever the budget
        can hold the ROLE section.
        """
        if ROLE_SECTION_OPENING_TAG not in content:
            return truncate_string_to_token_limit(content, prompt_budget)
        fixed_half, role_after_opening_tag = content.split(
            ROLE_SECTION_OPENING_TAG, 1
        )
        role_section = ROLE_SECTION_OPENING_TAG + role_after_opening_tag
        fixed_half_budget = prompt_budget - count_tokens(role_section)
        if fixed_half_budget <= 0:
            # The ROLE section alone fills the window: keep as much of the
            # ROLE section as the window holds and drop the instructions.
            return truncate_string_to_token_limit(role_section, prompt_budget)
        return (
            truncate_string_to_token_limit(fixed_half, fixed_half_budget)
            + role_section
        )

    @staticmethod
    def _per_turn_section_start(latest: SystemMessage, content: str) -> int | None:
        """Where the per-turn sections begin in ``content``, or None when unknown.

        ``load_consciousness`` records the offset in the message's
        ``additional_kwargs``. An offset that does not fall strictly inside the
        text is treated as absent, so a stale or malformed offset degrades to
        sending one system message rather than cutting the prompt in a wrong
        place.
        """
        offset = (getattr(latest, "additional_kwargs", None) or {}).get(
            PER_TURN_SECTION_START_KEY
        )
        if isinstance(offset, int) and 0 < offset < len(content):
            return offset
        return None

    @staticmethod
    def _messages_with_per_turn_section(
        messages: list[BaseMessage], per_turn_message: SystemMessage
    ) -> list[BaseMessage]:
        """Place the per-turn sections immediately before the newest HumanMessage.

        Everything ahead of that position — the fixed system message and the
        conversation so far — is identical from one model call to the next, so
        the provider can serve the whole stretch from its prompt cache. Placing
        the per-turn sections before the newest human words rather than after
        them also keeps the tool-calling rounds of one turn identical up to
        their own new messages, so every round after the first is a cache hit.
        """
        for index in range(len(messages) - 1, -1, -1):
            if isinstance(messages[index], HumanMessage):
                return [*messages[:index], per_turn_message, *messages[index:]]
        return [per_turn_message, *messages]

    @staticmethod
    def _provider_accepts_a_second_system_message(context: GlobalContext) -> bool:
        """Whether the configured provider takes a system message mid-conversation.

        OpenAI does. The Llama chat templates behind ``META`` and ``TOGETHER``
        expect one leading system message, and a request those templates refuse
        fails the turn outright, so every other provider keeps the single
        system message the avatar always sent.
        """
        return str(getattr(context, "model_provider", "") or "") == "OPEN_AI"

    def _apply(self, request: ModelRequest) -> ModelRequest:
        latest = self._latest_system_message(request.state)
        if latest is None:
            return request
        context = GlobalContext()
        window = hosted_inference_input_token_limit(context)
        content = latest.content if isinstance(latest.content, str) else str(latest.content or "")
        message_id = getattr(latest, "id", None)
        prompt_tokens = count_tokens(content)
        tool_tokens = estimate_tool_schema_tokens(getattr(request, "tools", None))
        message_tokens = estimate_messages_token_count(
            list(getattr(request, "messages", None) or [])
        )
        overhead_tokens = 2048
        prompt_budget = max(
            4096, window - tool_tokens - message_tokens - overhead_tokens
        )
        # Truncate when the assembled identity prompt exceeds the remaining
        # budget for this MODEL_TOKEN_LIMIT. A large ceiling leaves the
        # identity prompt intact. A trimmed prompt no longer matches the
        # recorded per-turn offset, so the trimmed prompt is sent the way the
        # prompt was always sent: as one system message.
        if prompt_tokens > prompt_budget:
            content = self._truncate_preserving_role(content, prompt_budget)
            return request.override(
                system_message=SystemMessage(content=content, id=message_id)
            )

        per_turn_start = self._per_turn_section_start(latest, content)
        if per_turn_start is None or not self._provider_accepts_a_second_system_message(
            context
        ):
            return request.override(
                system_message=SystemMessage(content=content, id=message_id)
            )

        # The fixed text is sent as the system message, byte for byte the same
        # on every turn for one avatar and audience, and this turn's status
        # blocks and ROLE go in a second system message beside the person's
        # newest words. gpt-5.6-luna keeps a prompt cache only at message
        # boundaries: with the per-turn text inside the one system message,
        # every reply wrote the whole prompt into the cache (billed above the
        # plain input rate) and read none of it back.
        fixed_system_message = SystemMessage(
            content=content[:per_turn_start], id=message_id
        )
        per_turn_message = SystemMessage(content=content[per_turn_start:])
        return request.override(
            system_message=fixed_system_message,
            messages=self._messages_with_per_turn_section(
                list(getattr(request, "messages", None) or []), per_turn_message
            ),
        )

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
