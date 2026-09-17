"""Force a text reply after ``update_avatar_identity_with_media`` runs once.

Returning early from the tool still counts as a graph step. The next
model call must have no tools so the same identity-media call cannot loop.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

from src.anubis.utils.tools.identity.identity_media_tools import (
    IDENTITY_MEDIA_TOOL_NAME,
)


def should_answer_in_words_only(
    messages: list[Any] | None,
    *,
    context: Any | None = None,
) -> bool:
    """Whether this turn must reply in words, with no tools.

    Every inference model keeps its tools. Capability questions are
    answered from the tool descriptions rather than by stripping the
    catalog for a smaller hosted model.
    """
    _ = messages, context
    return False


def _messages_of(state: Any) -> list[Any]:
    if isinstance(state, dict):
        return list(state.get("messages") or [])
    getter = getattr(state, "get", None)
    if callable(getter):
        return list(getter("messages") or [])
    return []


def identity_media_tool_already_returned(messages: list[Any] | None) -> bool:
    """Whether this turn already closed one identity-media tool call."""
    for message in messages or []:
        if not isinstance(message, ToolMessage):
            continue
        if getattr(message, "name", None) == IDENTITY_MEDIA_TOOL_NAME:
            return True
    return False


class IdentityMediaOnceMiddleware(AgentMiddleware):
    """Strip tools on the model call after identity media has already run."""

    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return "IdentityMediaOnceMiddleware"

    def _request_after_identity_media(self, request: Any) -> Any:
        state_messages = _messages_of(request.state)
        request_messages = list(getattr(request, "messages", None) or [])
        messages = request_messages or state_messages
        words_only = should_answer_in_words_only(
            state_messages
        ) or should_answer_in_words_only(request_messages)
        if words_only:
            return request.override(tools=[], tool_choice="none")
        if not identity_media_tool_already_returned(messages):
            return request
        return request.override(tools=[], tool_choice="none")

    def wrap_model_call(self, request: Any, handler: Callable[..., Any]) -> Any:
        return handler(self._request_after_identity_media(request))

    async def awrap_model_call(
        self, request: Any, handler: Callable[..., Awaitable[Any]]
    ) -> Any:
        return await handler(self._request_after_identity_media(request))
