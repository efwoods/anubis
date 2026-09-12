"""The moderation graph.

```
START -> fast_screen -> route_after_fast_screen
                          |-- "block"      -> refuse (verdict from the screen) -> END
                          |-- "clean"      -> END           (message path)
                          |                -> deep_judge    (upload path)
                          |-- "suspect"    -> deep_judge -> END
```

One compiled graph serves both callers, and the ``mode`` field decides how much of
it runs:

``MODERATION_MODE_MESSAGE`` is the chat path. Only ``fast_screen`` is allowed to
refuse a turn, because only ``fast_screen`` is cheap enough to sit on the critical
path of a reply. A ``suspect`` screen still runs the deep judge, but the caller
invokes that part AFTER the reply has streamed (see ``schedule_background`` in
``src.api.webapp``), so the person waits on nothing.

``MODERATION_MODE_UPLOAD`` is the media path. An upload is already a background
job that nobody is waiting on, and content that violates the terms must never be
indexed, analyzed, or turned into training data — so there the deep judge runs
even on a clean screen, and the process-media graph gates its whole fan-out on the
result.

Both stages are fail-open; see the docstrings in
``src.anubis.utils.moderation.content_moderation`` and ``...fast_screen``.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Sequence

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.moderation.content_moderation import (
    DEFAULT_MAX_CHARACTERS,
    clean_verdict,
    judge_documents,
    judge_text,
    moderation_flag_enabled,
)
from src.anubis.utils.moderation.fast_screen import (
    FAST_SCREEN_BLOCK,
    FAST_SCREEN_CLEAN,
    FAST_SCREEN_SUSPECT,
    clean_screen,
    fast_screen_text,
    screen_to_verdict,
)

logger = logging.getLogger(__name__)

MODERATION_MODE_MESSAGE = "message"
MODERATION_MODE_UPLOAD = "upload"

# How much of a document set the cheap screen reads. The screen is a single call
# over one string, so the documents are joined and truncated; the deep judge is
# the stage that reads every document in full.
_SCREEN_JOIN_CHARACTERS = 4000


class ModerationState(TypedDict, total=False):
    """State of one moderation run.

    ``text`` is set by the chat path, ``documents`` by the upload path; a run
    carries one or the other. ``screen`` and ``verdict`` are the two stages'
    outputs, and ``verdict`` is what every caller reads.
    """

    text: str
    documents: Sequence[Document]
    mode: str
    screen: dict
    verdict: dict


def _context_or_default(context: Any):
    if context is not None:
        return context
    return GlobalContext()


def _screen_input_text(state: ModerationState) -> str:
    text = (state.get("text") or "").strip()
    if text:
        return text
    documents = list(state.get("documents") or [])
    if not documents:
        return ""
    joined: list[str] = []
    remaining = _SCREEN_JOIN_CHARACTERS
    for document in documents:
        content = (document.page_content or "").strip()
        if not content:
            continue
        joined.append(content[:remaining])
        remaining -= len(content[:remaining])
        if remaining <= 0:
            break
    return "\n\n".join(joined)


async def fast_screen(state: ModerationState, runtime: Any = None) -> dict:
    """Stage one: the cheap OpenAI moderation screen."""
    context = _context_or_default(getattr(runtime, "context", None))
    if not moderation_flag_enabled(
        getattr(context, "content_moderation_fast_screen_enabled", "TRUE")
    ):
        return {"screen": clean_screen()}
    text = _screen_input_text(state)
    if not text:
        return {"screen": clean_screen()}
    return {"screen": await fast_screen_text(text, context)}


def route_after_fast_screen(
    state: ModerationState,
) -> Literal["refuse", "deep_judge", "__end__"]:
    """A blocking screen refuses immediately; otherwise decide whether to judge."""
    screen = state.get("screen") or {}
    if screen.get("outcome") == FAST_SCREEN_BLOCK:
        return "refuse"
    if screen.get("outcome") == FAST_SCREEN_SUSPECT:
        return "deep_judge"
    # A clean screen ends the chat path. An upload is judged anyway: nothing that
    # violates the terms may reach the index, and no person is waiting on it.
    if state.get("mode") == MODERATION_MODE_UPLOAD:
        return "deep_judge"
    return "__end__"


async def refuse(state: ModerationState, runtime: Any = None) -> dict:
    """Turn a blocking screen straight into a verdict, with no model call."""
    return {"verdict": screen_to_verdict(state.get("screen") or {})}


async def deep_judge(state: ModerationState, runtime: Any = None) -> dict:
    """Stage two: the structured-output terms-of-service / privacy-policy judge."""
    context = _context_or_default(getattr(runtime, "context", None))
    if not moderation_flag_enabled(
        getattr(context, "content_moderation_deep_judge_enabled", "TRUE")
    ):
        return {"verdict": clean_verdict()}
    max_characters = int(
        getattr(context, "content_moderation_max_characters", DEFAULT_MAX_CHARACTERS)
        or DEFAULT_MAX_CHARACTERS
    )
    documents = list(state.get("documents") or [])
    if documents:
        concurrency = int(getattr(context, "content_moderation_concurrency", 4) or 4)
        verdict = await judge_documents(
            documents, concurrency=concurrency, max_characters=max_characters
        )
    else:
        verdict = await judge_text(
            state.get("text") or "", max_characters=max_characters
        )
    return {"verdict": verdict}


workflow = StateGraph(ModerationState, context_schema=GlobalContext)

workflow.add_node("fast_screen", fast_screen)
workflow.add_node("refuse", refuse)
workflow.add_node("deep_judge", deep_judge)

workflow.add_edge(START, "fast_screen")
workflow.add_conditional_edges(
    "fast_screen",
    route_after_fast_screen,
    {"refuse": "refuse", "deep_judge": "deep_judge", "__end__": END},
)
workflow.add_edge("refuse", END)
workflow.add_edge("deep_judge", END)

moderation_graph = workflow.compile()
moderation_graph.name = "moderation_graph"


async def moderate_text_with_graph(
    text: str,
    *,
    mode: str = MODERATION_MODE_MESSAGE,
    context: Any | None = None,
) -> dict:
    """Run the graph over one text and return ``{screen, verdict}``.

    The verdict is always present, so a caller never has to distinguish "the graph
    ended early because the screen was clean" from "the judge found nothing".
    """
    result = await moderation_graph.ainvoke(
        {"text": text or "", "mode": mode},
        context=_context_or_default(context),
    )
    return {
        "screen": result.get("screen") or clean_screen(),
        "verdict": result.get("verdict") or clean_verdict(),
    }


async def moderate_documents_with_graph(
    documents: Sequence[Document],
    *,
    mode: str = MODERATION_MODE_UPLOAD,
    context: Any | None = None,
) -> dict:
    """Run the graph over converted upload documents and return ``{screen, verdict}``."""
    result = await moderation_graph.ainvoke(
        {"documents": list(documents), "mode": mode},
        context=_context_or_default(context),
    )
    return {
        "screen": result.get("screen") or clean_screen(),
        "verdict": result.get("verdict") or clean_verdict(),
    }


__all__ = [
    "FAST_SCREEN_BLOCK",
    "FAST_SCREEN_CLEAN",
    "FAST_SCREEN_SUSPECT",
    "MODERATION_MODE_MESSAGE",
    "MODERATION_MODE_UPLOAD",
    "ModerationState",
    "deep_judge",
    "fast_screen",
    "moderate_documents_with_graph",
    "moderate_text_with_graph",
    "moderation_graph",
    "route_after_fast_screen",
]
