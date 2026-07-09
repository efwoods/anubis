"""Data-analysis capability for the avatar deep agent.

Connects the avatar to a Model Context Protocol filesystem server so the
avatar can discover host data files, ingest them into a per-user-per-avatar
persistent store buffer, analyze them with real shell execution (pandas /
matplotlib) inside an ephemeral workspace, and persist created artifacts
(reports, plots) separately from ingested data.

Modules:

- ``discovery``: subscribes to a server's Server-Sent-Events announcement,
  and persists the per-user (single-avatar-bound) connection + per-avatar
  decline marker in the cross-thread store.
- ``mcp_client``: cached access to a saved connection's Model Context
  Protocol filesystem tools with graceful degradation when unreachable.
- ``backend``: assembles the ``CompositeBackend`` (local-shell execution
  workspace + two ``StoreBackend`` routes) and owns the workspace-cleanup
  and store-quota policies.
- ``analysis_tools``: the LangChain tools exposed to the deep agent
  (discover / ingest / hydrate / persist / preview).
"""

from src.anubis.utils.tools.data_analysis.analysis_tools import (
    build_connect_tool,
    build_data_analysis_tools,
)
from src.anubis.utils.tools.data_analysis.backend import (
    AnalysisBackendBundle,
    build_analysis_backend,
    cleanup_analysis_workspace,
    enforce_ingested_quota,
)
from src.anubis.utils.tools.data_analysis.discovery import (
    McpConnection,
    bound_connection_for,
    clear_declined,
    clear_user_connection,
    discover_announced_server,
    is_declined,
    mark_declined,
    read_user_connection,
    save_user_connection,
)
from src.anubis.utils.tools.data_analysis.mcp_client import (
    call_mcp_filesystem_tool,
)

__all__ = [
    "AnalysisBackendBundle",
    "McpConnection",
    "bound_connection_for",
    "build_analysis_backend",
    "build_connect_tool",
    "build_data_analysis_tools",
    "call_mcp_filesystem_tool",
    "cleanup_analysis_workspace",
    "clear_declined",
    "clear_user_connection",
    "discover_announced_server",
    "enforce_ingested_quota",
    "is_declined",
    "mark_declined",
    "read_user_connection",
    "save_user_connection",
]
