r"""The psycho-analysis graph: read the target's latent psychology from an upload.

```
START -> select_target_documents
      -> analyze_love_languages            \\
      -> analyze_dialogue_emotional_triggers |
      -> analyze_emotional_baseline          |  every registered dimension,
      -> analyze_attachment_style            |  fanned out concurrently
      -> ... one node per dimension ...     /
      -> consolidate_psychological_profile
      -> seed_current_emotional_state
      -> END
```

Every dimension is a real node rather than a coroutine inside one node, so the
graph itself expresses the concurrency and each dimension is separately visible
while an upload runs. The fan-in waits for all of them before consolidating, and
each node bounds its own per-document fan-out with a semaphore.

The graph carries its own state rather than ``GlobalState``: it is invoked as one
node of the process-media graph through the thin wrapper at the bottom of this
module, which maps the upload's documents in and the produced Documents back out.
That keeps ``GlobalState`` free of a dozen analysis-only channels and lets the
graph be exercised on its own in tests.
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, Sequence

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.anubis.utils.context import GlobalContext
from src.subgraphs.psycho_analysis_graph.utils.dimensions import (
    PSYCHOLOGICAL_DIMENSIONS,
    selected_dimensions,
)
from src.subgraphs.psycho_analysis_graph.utils.nodes import (
    build_dimension_node,
    consolidate_psychological_profile,
    seed_current_emotional_state,
    select_target_documents,
)

logger = logging.getLogger(__name__)

SELECT_NODE = "select_target_documents"
CONSOLIDATE_NODE = "consolidate_psychological_profile"
SEED_EMOTION_NODE = "seed_current_emotional_state"


class PsychoAnalysisState(TypedDict, total=False):
    """State of one psycho-analysis run over one upload.

    ``psychological_documents`` and ``dimension_findings`` are reduced by addition
    because every dimension node writes them in the same superstep; a plain channel
    would raise "can receive only one value per step" the moment two dimensions
    finished together.
    """

    documents: Sequence[Document]
    # The role-converted conversation, both speakers intact, for the dimensions
    # that read an exchange rather than the target's words alone.
    dialogue_documents: Sequence[Document]
    selected_documents: Sequence[Document]
    selected_dialogue_documents: Sequence[Document]
    max_documents: int
    creator_id: str
    assistant_id: str
    # The LangGraph store, handed in by the caller. It travels in state rather
    # than through the runtime because this graph is invoked as a fresh run from
    # inside a process-media node, and a fresh run does not inherit the parent
    # run's store. The graph is compiled without a checkpointer, so nothing here
    # is ever serialized.
    store: Any
    psychological_documents: Annotated[list[Document], operator.add]
    dimension_findings: Annotated[list[dict], operator.add]
    psychological_profile: dict | None
    current_emotion: dict | None


def build_psycho_analysis_graph(dimension_names: Sequence[str] | None = None):
    """Compile the graph with one node per dimension.

    ``dimension_names`` narrows the graph, which tests use to compile a graph with
    a single dimension in it. The default is every registered dimension; whether a
    dimension actually RUNS is decided at run time by ``selected_dimensions`` so a
    deployment flag never requires recompiling the graph.
    """
    names = list(dimension_names or PSYCHOLOGICAL_DIMENSIONS.keys())
    workflow = StateGraph(PsychoAnalysisState, context_schema=GlobalContext)
    workflow.add_node(SELECT_NODE, select_target_documents)
    workflow.add_node(CONSOLIDATE_NODE, consolidate_psychological_profile)
    workflow.add_node(SEED_EMOTION_NODE, seed_current_emotional_state)
    workflow.add_edge(START, SELECT_NODE)

    dimension_node_names: list[str] = []
    for name in names:
        dimension = PSYCHOLOGICAL_DIMENSIONS.get(name)
        if dimension is None:
            logger.warning("psycho analysis: unknown dimension %r; skipping", name)
            continue
        node_name = f"analyze_{dimension.name}"
        workflow.add_node(node_name, build_dimension_node(dimension))
        workflow.add_edge(SELECT_NODE, node_name)
        dimension_node_names.append(node_name)

    if dimension_node_names:
        workflow.add_edge(dimension_node_names, CONSOLIDATE_NODE)
    else:
        workflow.add_edge(SELECT_NODE, CONSOLIDATE_NODE)
    workflow.add_edge(CONSOLIDATE_NODE, SEED_EMOTION_NODE)
    workflow.add_edge(SEED_EMOTION_NODE, END)

    compiled = workflow.compile()
    compiled.name = "psycho_analysis_graph"
    return compiled


psycho_analysis_graph = build_psycho_analysis_graph()


async def psycho_analysis(
    state: dict,
    config: RunnableConfig = None,
    runtime: Any = None,
    store: Any = None,
) -> dict:
    """The process-media node that runs the psycho-analysis graph over an upload.

    Reads the same analysis queue the trait analyzers read, runs the dimensions,
    and merges the produced Documents into the upload's index batch so they are
    persisted alongside the source documents in one pass.
    """
    from src.anubis.utils.moderation.content_moderation import moderation_flag_enabled

    context = getattr(runtime, "context", None) or GlobalContext()
    if not moderation_flag_enabled(
        getattr(context, "enable_psychological_analysis", "TRUE")
    ):
        logger.info(
            "psycho analysis: disabled via ENABLE_PSYCHOLOGICAL_ANALYSIS; skipping"
        )
        return {}

    documents = list(
        state.get(
            "documents_to_be_analyzed_for_context_storage_and_prompt_injection_of_assistant"
        )
        or []
    )
    # The adapter queue holds the conversation as one unchunked document with every
    # speaker in it. Trigger dimensions need that: a trigger is a stimulus and a
    # reaction, and the analysis queue's few-hundred-character target-only chunks
    # rarely contain both.
    dialogue_documents = list(
        state.get("documents_to_be_processed_for_adapter_training") or []
    )
    if not documents:
        return {}

    configurable = (config or {}).get("configurable", {}) or {}
    assistant_id = configurable.get("assistant_id") or (
        (state.get("assistant_state") or {}).get("assistant_id")
    )
    creator_id = ((configurable.get("assistant_ctx") or {}).get("metadata") or {}).get(
        "user_id"
    ) or configurable.get("user_id")
    runtime_store = store or getattr(runtime, "store", None)

    dimension_names = [dimension.name for dimension in selected_dimensions(context)]
    if not dimension_names:
        return {}
    graph = (
        psycho_analysis_graph
        if dimension_names == list(PSYCHOLOGICAL_DIMENSIONS.keys())
        else build_psycho_analysis_graph(dimension_names)
    )

    try:
        result = await graph.ainvoke(
            {
                "documents": documents,
                "dialogue_documents": dialogue_documents,
                "max_documents": int(
                    getattr(context, "psychological_analysis_max_documents", 24) or 24
                ),
                "creator_id": creator_id or "",
                "assistant_id": assistant_id or "",
                "store": runtime_store,
            },
            context=context,
        )
    except Exception as analysis_error:  # noqa: BLE001 - an upload must still finish
        logger.error(
            "psycho analysis failed; the upload continues without it: %s",
            analysis_error,
        )
        return {}

    produced = list(result.get("psychological_documents") or [])
    if not produced:
        return {}
    return {"vectorstore_documents_to_be_indexed": produced}


__all__ = [
    "CONSOLIDATE_NODE",
    "SEED_EMOTION_NODE",
    "SELECT_NODE",
    "PsychoAnalysisState",
    "build_psycho_analysis_graph",
    "psycho_analysis",
    "psycho_analysis_graph",
]
