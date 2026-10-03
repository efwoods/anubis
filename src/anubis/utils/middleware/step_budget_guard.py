"""Middleware that keeps the deep agent learning every fact while still ending runaway turns.

The deep agent's recursion limit counts LangGraph supersteps, not model calls.
One tool round costs five supersteps when the tool is an identity tool:
``model`` → ``tools`` → ``ConsciousnessRefreshGate`` (synthetic
``load_consciousness`` call) → ``tools`` → the next ``before_model`` hooks →
``model``. A turn therefore costs ``3 + 5 × tool_rounds`` supersteps (measured
with a scripted model against the real middleware stack: 1 round = 8, 9 rounds
= 48, 11 rounds = 58). A message carrying many distinct facts can lead the
model to save the facts one call at a time, and with a flat
``DEEP_AGENT_RECURSION_LIMIT=50`` the tenth save raised ``GraphRecursionError``
with no reply at all (thread 618f0fd0, 2026-09-29 17:40:57 UTC).

The recursion limit exists to end a runaway turn, and saving a new fact is
progress, not a runaway. So ``StepBudgetGuard`` charges two budgets:

1. **The work budget** is ``DEEP_AGENT_RECURSION_LIMIT``. Every superstep
   counts against the work budget EXCEPT the five supersteps of a learning
   round — a tool round in which at least one tool reported a newly learned
   fact (``parse_learned_fact_from_tool_content``). A round that only repeats
   an already-stored fact, fails to save, or does any other work is charged,
   so a model re-saving the same fact forever is still stopped.
2. **The graph limit** is the work budget plus
   ``DEEP_AGENT_MAXIMUM_LEARNING_ROUNDS × 5`` supersteps
   (``deep_agent_graph_recursion_limit``). The graph limit is only the
   backstop that LangGraph enforces; the work budget is what normally ends a
   turn.

When either budget could not hold one more tool round, the guard removes every
tool from the model request, so the model has to write the reply now instead
of the turn failing with no reply.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.config import get_config

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.learning.fact_learned import parse_learned_fact_from_tool_content

logger = logging.getLogger(__name__)

DEFAULT_RESERVED_SUPERSTEPS = 5
"""Supersteps a model call must leave free before tools are removed.

One identity tool round moves the next model call five supersteps later, and
the closing model call must land at or below the recursion limit. A model call
with more than five supersteps left can therefore afford one more tool round;
a model call with five or fewer must write the reply. Measured with the
scripted-model harness at recursion limits 50 through 55 with twenty queued
facts: every run ended in a reply and none raised ``GraphRecursionError``.
"""

LEARNING_ROUND_SUPERSTEPS = 5
"""Supersteps one learning round costs: model, tools, refresh gate, tools, before-model hooks."""


def deep_agent_graph_recursion_limit(context: GlobalContext) -> int:
    """Return the LangGraph recursion limit: the work budget plus room for every learning round."""
    return (
        context.deep_agent_recursion_limit
        + context.deep_agent_maximum_learning_rounds * LEARNING_ROUND_SUPERSTEPS
    )


def count_learning_rounds(messages: Sequence[BaseMessage]) -> int:
    """Count tool rounds in which at least one tool reported a newly learned fact.

    A tool round is one ``AIMessage`` carrying tool calls; the round is a
    learning round when any ``ToolMessage`` answering one of those tool calls
    parses as a newly learned fact. Duplicate and failed saves do not parse,
    so a round of duplicates is not a learning round.
    """
    learned_tool_call_ids = {
        message.tool_call_id
        for message in messages
        if isinstance(message, ToolMessage)
        and parse_learned_fact_from_tool_content(
            message.content, tool_name=message.name
        )
        is not None
    }
    return sum(
        1
        for message in messages
        if isinstance(message, AIMessage)
        and any(
            tool_call.get("id") in learned_tool_call_ids
            for tool_call in message.tool_calls or []
        )
    )


class StepBudgetGuard(AgentMiddleware):
    """Remove tools from the model request once the work budget or the graph limit is spent.

    Args:
        work_superstep_budget: ``DEEP_AGENT_RECURSION_LIMIT``, the supersteps
            available to everything except learning rounds.
        reserved_supersteps: Supersteps that must remain under each budget
            for the model to keep access to tools.
    """

    def __init__(
        self,
        work_superstep_budget: int,
        reserved_supersteps: int = DEFAULT_RESERVED_SUPERSTEPS,
    ) -> None:
        """Store the work budget and the supersteps held back for the closing reply."""
        super().__init__()
        self._work_superstep_budget = work_superstep_budget
        self._reserved_supersteps = reserved_supersteps

    @property
    def name(self) -> str:  # pragma: no cover - trivial
        """Name the middleware in the agent graph."""
        return "StepBudgetGuard"

    @staticmethod
    def _current_step_and_graph_limit() -> tuple[int, int] | None:
        try:
            run_config = get_config()
        except RuntimeError:
            return None
        graph_recursion_limit = run_config.get("recursion_limit")
        current_step = (run_config.get("metadata") or {}).get("langgraph_step")
        if not isinstance(graph_recursion_limit, int) or not isinstance(
            current_step, int
        ):
            return None
        return current_step, graph_recursion_limit

    def _apply(self, request: ModelRequest) -> ModelRequest:
        if not request.tools:
            return request
        step_and_limit = self._current_step_and_graph_limit()
        if step_and_limit is None:
            return request
        current_step, graph_recursion_limit = step_and_limit

        learning_rounds = count_learning_rounds(list(request.messages))
        work_supersteps_spent = (
            current_step - learning_rounds * LEARNING_ROUND_SUPERSTEPS
        )
        work_supersteps_remaining = self._work_superstep_budget - work_supersteps_spent
        graph_supersteps_remaining = graph_recursion_limit - current_step

        if (
            work_supersteps_remaining > self._reserved_supersteps
            and graph_supersteps_remaining > self._reserved_supersteps
        ):
            return request
        logger.warning(
            "StepBudgetGuard: removing %d tools so the model writes the reply now "
            "(superstep %d; learning rounds %d; work supersteps remaining %d of %d; "
            "graph supersteps remaining %d of %d; reserve %d)",
            len(request.tools),
            current_step,
            learning_rounds,
            work_supersteps_remaining,
            self._work_superstep_budget,
            graph_supersteps_remaining,
            graph_recursion_limit,
            self._reserved_supersteps,
        )
        return request.override(tools=[], tool_choice=None)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Call the model, without tools once a budget is spent."""
        return handler(self._apply(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Call the model asynchronously, without tools once a budget is spent."""
        return await handler(self._apply(request))


__all__ = [
    "LEARNING_ROUND_SUPERSTEPS",
    "StepBudgetGuard",
    "count_learning_rounds",
    "deep_agent_graph_recursion_limit",
]
