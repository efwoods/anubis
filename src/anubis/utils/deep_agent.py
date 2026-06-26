"""Deep agent state schema + assembly for the Anubis avatar.

This module is the seam between the outer LangGraph workflow (see
``src/anubis/graph.py``) and the ``deepagents.create_deep_agent`` runtime.
It defines:

- ``AvatarDeepAgentState``: a subclass of ``DeepAgentState`` that carries
  every avatar-specific state slot the outer graph reads/writes
  (identity-document snapshots, the pinned consciousness ``SystemMessage``,
  user/assistant identity blobs, internal-thoughts audit channel, etc.) so
  the deep agent's tool node can update them in-place via ``Command``.
- ``build_avatar_deep_agent``: factory that wires identity tools +
  ``load_consciousness_tool`` and stacks our custom middleware
  (``ConsciousnessRefreshGate`` + ``DynamicConsciousnessPrompt``) on top
  of the deep-agent default stack — which already includes
  ``SummarizationMiddleware``, so we don't add another instance.

The exported agent is compiled but unbound to a checkpointer; the outer
graph drives it inside the ``think`` node so persistence stays unified at
the workflow level.
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from typing_extensions import Required

from deepagents import create_deep_agent
from deepagents.graph import DeepAgentState

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.middleware.consciousness_refresh_gate import (
    ConsciousnessRefreshGate,
)
from src.anubis.utils.middleware.dynamic_consciousness_prompt import (
    DynamicConsciousnessPrompt,
)
from src.anubis.utils.model import init_chat_model_unbound
from src.anubis.utils.state import AssistantState, UserState
from src.anubis.utils.tools.consciousness import (
    LOAD_CONSCIOUSNESS_TOOL_NAME,
    load_consciousness_tool,
)
from src.anubis.utils.tools.identity.identity_tools import (
    correct_identity_fact,
    create_episodic_memory,
    learn_information_about_the_user,
    recall_memories,
    update_self_identity_mem_from_user_txt,
)
from src.anubis.utils.utility import reduce_docs

logger = logging.getLogger(__name__)


IDENTITY_TOOLS = [
    create_episodic_memory,
    recall_memories,
    update_self_identity_mem_from_user_txt,
    learn_information_about_the_user,
    correct_identity_fact,
]
"""Tools whose successful execution should trigger a ``load_consciousness`` refresh.

Order doesn't matter — ``ConsciousnessRefreshGate`` only checks set
membership of tool names against the most recent AI tool-call batch.
"""

IDENTITY_TOOL_NAMES: frozenset[str] = frozenset(t.name for t in IDENTITY_TOOLS)


class AvatarDeepAgentState(DeepAgentState):
    """Deep-agent state augmented with avatar consciousness slots.

    ``DeepAgentState.messages`` already uses a ``DeltaChannel`` reducer for
    O(N) checkpoint growth; we inherit that unchanged. Everything else
    mirrors the keys ``load_consciousness`` (the node and the in-agent
    tool) writes, so identity-tool ``Command`` updates apply cleanly
    without needing custom reducers wired in.
    """

    system_message: Required[Annotated[list[SystemMessage], add_messages]]
    """Pinned single-slot system prompt list.

    ``load_consciousness`` writes a ``SystemMessage`` with a fixed UUID so
    ``add_messages`` replaces rather than appends. The
    ``DynamicConsciousnessPrompt`` middleware reads the last entry on
    every model call.
    """

    internal_thoughts: Required[Annotated[list[AIMessage | ToolMessage], add_messages]]
    """Audit channel — only the outer ``think`` node writes to this.

    Carried in state so legacy nodes (e.g., previous ``process_thoughts``
    routes) can still read it without a schema mismatch during the
    migration.
    """

    user_identity_documents: Annotated[Sequence[Document], reduce_docs]
    assistant_identity_documents: Annotated[Sequence[Document], reduce_docs]
    recalled_memory_documents: Annotated[Sequence[Document], reduce_docs]

    user_state: UserState
    assistant_state: AssistantState

    queries: Annotated[list[str], operator.add]
    retrieved_docs: Annotated[list[Document], operator.add]

    current_user_emotions: str
    current_assistant_emotions: str


def build_avatar_deep_agent(
    context: Optional[GlobalContext] = None,
    *,
    extra_tools: Sequence[Any] | None = None,
    checkpointer: Any | None = None,
    store: Any | None = None,
):
    """Construct the avatar's deep agent.

    Args:
        context: Optional pre-instantiated ``GlobalContext``. When
            ``None``, a fresh one is built — same as every other
            avatar-side helper.
        extra_tools: Additional tools to expose to the deep agent on top
            of the identity tool suite + ``load_consciousness_tool``.
            Reserved for future analysis / OS-query / report tools.
        checkpointer: Optional persistent checkpointer. Required for
            human-in-the-loop tools (``correct_identity_fact``) so an
            ``interrupt`` raised mid-tool is durable and resumable. When
            ``None`` the agent runs without its own persistence (the outer
            workflow owns it) — the legacy behavior for non-interrupting turns.
        store: Optional cross-thread store. When ``None`` the store
            propagates from the parent runtime (legacy behavior); ``think``
            passes ``runtime.store`` explicitly so the agent under its own
            checkpointer can still reach identity facts.

    Returns:
        A compiled deep-agent graph.
    """
    context = context or GlobalContext()

    model = init_chat_model_unbound(context)

    # ``create_deep_agent`` always installs its own
    # ``SummarizationMiddleware`` (via
    # ``deepagents.middleware.summarization.create_summarization_middleware``)
    # with model-aware fraction/token defaults. Passing another
    # ``SummarizationMiddleware`` instance here would collide on
    # ``middleware.name == "SummarizationMiddleware"`` and trip the
    # duplicate-middleware assertion in ``create_agent``. We therefore
    # rely on the built-in summarization and only inject avatar-specific
    # middleware (consciousness refresh gate + dynamic prompt). The
    # ``deep_agent_summarization_*`` env vars are kept on
    # ``GlobalContext`` as future hooks for a harness-profile-based
    # override path.

    tools: list[Any] = [
        *IDENTITY_TOOLS,
        load_consciousness_tool,
    ]
    if extra_tools:
        tools.extend(extra_tools)

    refresh_gate = ConsciousnessRefreshGate(
        identity_tool_names=IDENTITY_TOOL_NAMES,
        load_consciousness_tool_name=LOAD_CONSCIOUSNESS_TOOL_NAME,
    )
    dynamic_prompt = DynamicConsciousnessPrompt()

    logger.info(
        "Building avatar deep agent: model=%s identity_tools=%d",
        context.model,
        len(IDENTITY_TOOLS),
    )

    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=None,
        middleware=[refresh_gate, dynamic_prompt],
        state_schema=AvatarDeepAgentState,
        checkpointer=checkpointer,
        store=store,
    ).with_config(
        {
            "recursion_limit": context.deep_agent_recursion_limit,
        }
    )


__all__ = [
    "AvatarDeepAgentState",
    "IDENTITY_TOOLS",
    "IDENTITY_TOOL_NAMES",
    "build_avatar_deep_agent",
]
