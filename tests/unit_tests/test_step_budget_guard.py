"""Regression: a message carrying many facts is learned in full and ends in a reply.

A message listing more than nine distinct memories led the model to save the
memories one tool call per round. Each round costs five supersteps
(``ConsciousnessRefreshGate`` adds a ``load_consciousness`` round), so the
tenth save exceeded ``DEEP_AGENT_RECURSION_LIMIT=50`` and the turn failed with
no reply. ``StepBudgetGuard`` does not charge a round that learned a new fact
against ``DEEP_AGENT_RECURSION_LIMIT``, so every fact is saved; a round that
learned nothing is charged, so a model repeating a stored fact is still ended
with a reply instead of ``GraphRecursionError``.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

import src.anubis.utils.deep_agent as deep_agent_module
from src.anubis.utils.context import GlobalContext
from src.anubis.utils.middleware.step_budget_guard import (
    deep_agent_graph_recursion_limit,
)

QUEUED_FACT_COUNT = 20
WORK_SUPERSTEP_BUDGET = 50
MAXIMUM_LEARNING_ROUNDS = 60


class OneFactPerRoundModel(BaseChatModel):
    """Scripted model that saves one fact per round while tools are bound."""

    tools_bound: bool = False

    @property
    def _llm_type(self) -> str:
        return "one-fact-per-round"

    def bind_tools(self, tools: Any, **kwargs: Any) -> OneFactPerRoundModel:  # type: ignore[override]
        return self.model_copy(update={"tools_bound": bool(tools)})

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        saved_fact_count = sum(
            1
            for message in messages
            if isinstance(message, ToolMessage)
            and message.name == "create_episodic_memory"
        )
        if not self.tools_bound or saved_fact_count >= QUEUED_FACT_COUNT:
            reply = AIMessage(content="final reply")
        else:
            reply = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "create_episodic_memory",
                        "args": {
                            "significant_event": f"fact {saved_fact_count}",
                            "significant_event_context": "context",
                        },
                        "id": f"call_{uuid.uuid4().hex}",
                        "type": "tool_call",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=reply)])


@tool("create_episodic_memory")
def stub_create_episodic_memory(
    significant_event: str, significant_event_context: str
) -> str:
    """Stand-in for the memory tool, answering with the real tool's success text."""
    return f"Learned: {significant_event}"


@tool("create_episodic_memory")
def stub_create_episodic_memory_duplicate(
    significant_event: str, significant_event_context: str
) -> str:
    """Stand-in for the memory tool when every fact is already stored."""
    return f"Not learned: {significant_event} was previously learned."


@tool("load_consciousness")
def stub_load_consciousness() -> str:
    """Stand-in for the consciousness loader."""
    return "refreshed"


async def run_deep_agent(
    monkeypatch: pytest.MonkeyPatch, memory_tool: Any
) -> tuple[list[Any], int]:
    """Run the real deep-agent middleware stack with a one-fact-per-round model."""
    scripted_model = OneFactPerRoundModel()
    monkeypatch.setattr(deep_agent_module, "IDENTITY_TOOLS", [memory_tool])
    monkeypatch.setattr(
        deep_agent_module, "load_consciousness_tool", stub_load_consciousness
    )
    monkeypatch.setattr(
        deep_agent_module,
        "init_chat_model_unbound",
        lambda *arguments, **keyword_arguments: scripted_model,
    )
    context = GlobalContext()
    context.deep_agent_recursion_limit = WORK_SUPERSTEP_BUDGET
    context.deep_agent_maximum_learning_rounds = MAXIMUM_LEARNING_ROUNDS
    deep_agent = deep_agent_module.build_avatar_deep_agent(
        context=context,
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
    )
    run_config: dict[str, Any] = {
        "configurable": {"thread_id": "step-budget"},
        "recursion_limit": deep_agent_graph_recursion_limit(context),
    }
    async for _ in deep_agent.astream(
        {"messages": [HumanMessage("many facts")], "system_message": []},
        run_config,
    ):
        pass
    final_state = await deep_agent.aget_state(run_config)
    final_messages = list(final_state.values["messages"])
    memory_round_count = sum(
        1
        for message in final_messages
        if isinstance(message, ToolMessage) and message.name == "create_episodic_memory"
    )
    return final_messages, memory_round_count


@pytest.mark.asyncio
async def test_every_fact_is_learned_when_saved_one_per_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_messages, memory_round_count = await run_deep_agent(
        monkeypatch, stub_create_episodic_memory
    )
    assert final_messages[-1].content == "final reply"
    # 3 + 5 × 20 = 103 supersteps, above the flat limit of 50 that failed the turn.
    assert memory_round_count == QUEUED_FACT_COUNT


@pytest.mark.asyncio
async def test_rounds_that_learn_nothing_still_end_at_the_work_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_messages, memory_round_count = await run_deep_agent(
        monkeypatch, stub_create_episodic_memory_duplicate
    )
    assert final_messages[-1].content == "final reply"
    # Duplicate rounds are charged: 3 + 5 × 9 = 48 fits the work budget of 50.
    assert memory_round_count == 9
