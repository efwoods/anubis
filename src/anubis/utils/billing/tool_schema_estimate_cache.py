# src/anubis/utils/billing/tool_schema_estimate_cache.py

"""Process-wide cached measurement of the avatar deep agent's tool schemas.

Every model call the deep agent makes carries the FULL serialized JSON schema
of every bound tool as billed input tokens — the provider embeds the tool
definitions in the prompt on every loop iteration, even though the tool
definitions never appear in the visible message list. The pre-request input
estimate must therefore include the tool schemas for the initial model call
(the number of loop iterations before the final reply is unknowable in
advance; recorded actual usage governs accrual for the loop).

The measurement is MODULAR by construction: the tools are enumerated from the
COMPILED deep agent — the same ``build_avatar_deep_agent`` graph the ``think``
node runs — so any tool added later (a new identity tool, a new deepagents
builtin, an environment-gated extra tool) is included automatically without
touching this module. Each bound tool is serialized to the provider
function-schema JSON (``convert_to_openai_tool``) and measured with the manual
four-characters-per-token arithmetic — no tokenizer, no counting endpoint,
per the estimation doctrine in ``estimation.py``.

The bound tool set is fixed for the lifetime of the process (environment-gated
extras are decided at build time from environment variables), so the
measurement is taken once and cached process-wide;
``invalidate_deep_agent_tool_schema_token_estimate`` exists for tests and for
any future hot-reload path.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Sequence

from src.anubis.utils.billing.estimation import (
    TokenEstimationError,
    estimate_text_tokens_from_characters,
)

_cached_tool_schema_token_estimate: int | None = None
_measurement_lock = threading.Lock()


def estimate_tool_schema_tokens_for_tools(tools: Sequence[Any]) -> int:
    """Measure the token estimate of a tool sequence's serialized schemas.

    Serializes every tool to the provider function-schema JSON and applies
    the manual four-characters-per-token ratio. Pure measurement over an
    explicit tool list — the testable core under
    ``measure_deep_agent_tool_schema_token_estimate``.
    """
    from langchain_core.utils.function_calling import convert_to_openai_tool

    total_serialized_characters = 0
    for tool in tools:
        serialized_schema = json.dumps(convert_to_openai_tool(tool))
        total_serialized_characters += len(serialized_schema)
    return estimate_text_tokens_from_characters(total_serialized_characters)


def _enumerate_bound_tools_from_compiled_agent(compiled_deep_agent: Any) -> list[Any]:
    """Return every tool object bound on a compiled deep agent's tool node.

    Scans the compiled graph's nodes for the tool-executing node (the node
    exposing ``tools_by_name``) so the enumeration follows whatever
    ``create_deep_agent`` actually bound — identity tools, deepagents
    builtins (``write_todos``, filesystem tools, ``execute``, ``task``), and
    any environment-gated extras — rather than a hand-maintained list.
    """
    for node in compiled_deep_agent.nodes.values():
        runnable = getattr(node, "bound", node)
        inner_node = (
            getattr(runnable, "_node", None)
            or getattr(runnable, "runnable", None)
            or runnable
        )
        tools_by_name = getattr(inner_node, "tools_by_name", None)
        if tools_by_name:
            return list(tools_by_name.values())
    raise TokenEstimationError(
        "Could not enumerate bound tools: no node on the compiled deep agent "
        "exposes tools_by_name."
    )


def measure_deep_agent_tool_schema_token_estimate() -> int:
    """Build the avatar deep agent and measure the bound tool schemas' tokens.

    Builds the SAME compiled graph the ``think`` node runs (no model call is
    made — building only compiles the graph), enumerates the bound tools from
    the compiled graph, and measures the serialized schemas manually. Raises
    on any failure — the message estimator treats estimation as fail-closed.
    """
    from src.anubis.utils.deep_agent import build_avatar_deep_agent

    compiled_deep_agent = build_avatar_deep_agent()
    bound_tools = _enumerate_bound_tools_from_compiled_agent(compiled_deep_agent)
    return estimate_tool_schema_tokens_for_tools(bound_tools)


def fetch_or_measure_deep_agent_tool_schema_token_estimate() -> int:
    """Return the process-wide tool-schema token estimate, measuring on first use.

    The first call per process pays one graph compilation; every later call
    returns the cached integer. Thread-safe: concurrent first requests measure
    once.
    """
    global _cached_tool_schema_token_estimate
    if _cached_tool_schema_token_estimate is not None:
        return _cached_tool_schema_token_estimate
    with _measurement_lock:
        if _cached_tool_schema_token_estimate is None:
            _cached_tool_schema_token_estimate = (
                measure_deep_agent_tool_schema_token_estimate()
            )
        return _cached_tool_schema_token_estimate


def invalidate_deep_agent_tool_schema_token_estimate() -> None:
    """Drop the cached measurement so the next fetch measures afresh."""
    global _cached_tool_schema_token_estimate
    with _measurement_lock:
        _cached_tool_schema_token_estimate = None
