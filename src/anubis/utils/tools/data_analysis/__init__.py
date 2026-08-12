"""Data-analysis capability for the avatar deep agent.

Connects the avatar to a Model Context Protocol filesystem server so the
avatar can discover host data files, ingest them into a per-user-per-avatar
persistent store buffer, analyze them with real shell execution (pandas /
matplotlib) inside an ephemeral workspace, and persist created artifacts
(reports, plots) separately from ingested data.

Modules:

- ``devices``: resolves each connected machine's human-readable label and
  platform, and works out which machine an absolute host path belongs to.
- ``discovery``: resolves every reachable machine (live relay socket, pushed
  registration, or Server-Sent-Events announcement) and persists one
  avatar-bound connection record per device, plus per-device auto-adopt
  suppression, in the cross-thread store.
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
    collect_turn_artifacts,
    persist_workspace_file,
)
from src.anubis.utils.tools.data_analysis.backend import (
    AnalysisBackendBundle,
    build_analysis_backend,
    cleanup_analysis_workspace,
    enforce_ingested_quota,
)
from src.anubis.utils.tools.data_analysis.devices import (
    deduplicate_label,
    derive_device_identity,
    resolve_device_for_path,
)
from src.anubis.utils.tools.data_analysis.discovery import (
    McpConnection,
    bound_connections_for,
    clear_declined,
    clear_user_connection,
    discover_announced_server,
    is_declined,
    mark_declined,
    read_user_connections,
    read_user_registrations,
    resolve_available_connections,
    save_user_connection,
    suppressed_device_ids,
)
from src.anubis.utils.tools.data_analysis.mcp_client import (
    call_mcp_filesystem_tool,
)

__all__ = [
    "AnalysisBackendBundle",
    "McpConnection",
    "bound_connections_for",
    "build_analysis_backend",
    "build_connect_tool",
    "build_data_analysis_tools",
    "call_mcp_filesystem_tool",
    "cleanup_analysis_workspace",
    "clear_declined",
    "clear_user_connection",
    "collect_turn_artifacts",
    "deduplicate_label",
    "derive_device_identity",
    "discover_announced_server",
    "enforce_ingested_quota",
    "is_declined",
    "mark_declined",
    "persist_workspace_file",
    "read_user_connections",
    "read_user_registrations",
    "resolve_available_connections",
    "resolve_device_for_path",
    "save_user_connection",
    "suppressed_device_ids",
]
