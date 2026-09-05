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
from typing import Annotated, Any, Sequence

from deepagents import create_deep_agent
from deepagents.graph import DeepAgentState
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, Required

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.middleware.consciousness_refresh_gate import (
    ConsciousnessRefreshGate,
)
from src.anubis.utils.middleware.avatar_summarization import (
    build_avatar_summarization_middleware,
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
    create_episodic_memory,
    delete_identity_fact,
    edit_identity_fact,
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
    edit_identity_fact,
    delete_identity_fact,
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

    conversation_summary_event: NotRequired[dict[str, Any] | None]
    conversation_summary_session_id: NotRequired[str | None]
    """Public mirror of the summarization event, carried across turns.

    ``AvatarSummarizationMiddleware`` reads this key when the deep agent's own
    private event is absent (every turn runs on a fresh deep-agent thread) and
    writes the newest event back to it; the outer ``think`` node forwards the
    value between the outer conversation state and this input.
    """


def build_avatar_deep_agent(
    context: GlobalContext | None = None,
    *,
    extra_tools: Sequence[Any] | None = None,
    checkpointer: Any | None = None,
    store: Any | None = None,
    backend: Any | None = None,
):
    """Construct the avatar's deep agent.

    Args:
        context: Optional pre-instantiated ``GlobalContext``. When
            ``None``, a fresh one is built — same as every other
            avatar-side helper.
        extra_tools: Additional tools to expose to the deep agent on top
            of the identity tool suite + ``load_consciousness_tool``.
            Used by the data-analysis capability (discover / ingest /
            hydrate / persist / preview tools built per turn in ``think``).
        checkpointer: Optional persistent checkpointer. Required for
            human-in-the-loop tools (``edit_identity_fact`` /
            ``delete_identity_fact``) so an
            ``interrupt`` raised mid-tool is durable and resumable. When
            ``None`` the agent runs without its own persistence (the outer
            workflow owns it) — the legacy behavior for non-interrupting turns.
        store: Optional cross-thread store. When ``None`` the store
            propagates from the parent runtime (legacy behavior); ``think``
            passes ``runtime.store`` explicitly so the agent under its own
            checkpointer can still reach identity facts.
        backend: Optional deep-agent file/execution backend. When ``None``
            the deepagents default (a virtual ``StateBackend``, no shell
            execution) applies — the legacy behavior. The data-analysis
            capability passes a ``CompositeBackend`` (local-shell workspace
            + per-user-per-avatar ``StoreBackend`` routes) built by
            ``src.anubis.utils.tools.data_analysis.backend.build_analysis_backend``.

    Returns:
        A compiled deep-agent graph.
    """
    context = context or GlobalContext()

    model = init_chat_model_unbound(context)

    # ``create_deep_agent`` installs its own ``SummarizationMiddleware`` (via
    # ``deepagents.middleware.summarization.create_summarization_middleware``).
    # deepagents merges caller-supplied middleware by NAME and replaces a
    # same-named default in place, so the avatar's summarizer below — which
    # keeps the name ``"SummarizationMiddleware"`` — takes the built-in's slot
    # rather than running beside the built-in. The avatar's summarizer honours
    # ``DEEP_AGENT_SUMMARIZATION_MAX_TOKENS`` / ``..._KEEP_LAST_N_MESSAGES``,
    # summarizes with a conversation-oriented prompt (relationship facts, tone,
    # ambient observations, open threads), and mirrors the summarization event
    # onto the public ``conversation_summary_event`` key so the outer workflow
    # carries the compaction across turns (the deep agent runs on a fresh
    # thread every turn, which would otherwise discard the event).

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
    summarization = build_avatar_summarization_middleware(context, model, backend)

    analysis_tool_count = len(extra_tools) if extra_tools else 0
    logger.info(
        "Building avatar deep agent: model=%s identity_tools=%d analysis_tools=%d total_tools=%d",
        context.model,
        len(IDENTITY_TOOLS),
        analysis_tool_count,
        len(tools),
    )

    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=None,
        middleware=[refresh_gate, dynamic_prompt, summarization],
        state_schema=AvatarDeepAgentState,
        checkpointer=checkpointer,
        store=store,
        backend=backend,
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
