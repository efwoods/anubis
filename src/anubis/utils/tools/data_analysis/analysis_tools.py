"""LangChain tools the deep agent uses for host-data analysis.

These tools bridge three places:

1. the **host filesystem**, reachable only through the Model Context
   Protocol filesystem server (never read in bulk — the avatar discovers and
   ingests specific files);
2. the **persistent per-user-per-avatar store buffer** (LangGraph
   ``BaseStore`` namespaces — see ``backend.py``), which survives across
   threads;
3. the **ephemeral execution workspace**, a real directory where the deep
   agent's built-in ``execute`` tool runs pandas / matplotlib. The workspace
   is wiped when the turn ends.

Raw Model Context Protocol tools are deliberately **not** exposed to the
model: ``read_file_bytes`` returns base64 payloads that would flood the
message context. The wrappers below keep bytes out of the model's context
and return compact summaries instead.

The tools are built per turn by ``build_data_analysis_tools`` because they
close over the turn's ``AnalysisBackendBundle`` (workspace path + ids).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Annotated, Any

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import InjectedToolArg

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.tools.data_analysis.backend import (
    CREATED_ROUTE,
    INGESTED_ROUTE,
    WORKSPACE_DATA_DIRECTORY,
    AnalysisBackendBundle,
    created_namespace,
    enforce_ingested_quota,
    ingested_namespace,
)
from src.anubis.utils.tools.data_analysis.discovery import (
    McpConnection,
    clear_declined,
    clear_user_connection,
    discover_announced_server,
    mark_declined,
    save_user_connection,
)
from src.anubis.utils.tools.data_analysis.mcp_client import call_mcp_filesystem_tool

logger = logging.getLogger(__name__)

# Cap on how many discovered paths are returned to the model in one call.
_DISCOVER_RESULT_LIMIT = 200


def _store_key_for_source(source_path: str, existing_source_path: str | None) -> str:
    """Store key for one ingested source file.

    Keys keep the plain file name (so the deep agent sees
    ``/data_ingested/health1.json``); when two different source paths share
    a basename, the later one gets a deterministic short-hash suffix so
    re-ingesting the same source always maps to the same key.
    """
    name = PurePosixPath(source_path).name
    if existing_source_path is None or existing_source_path == source_path:
        return f"/{name}"
    stem = PurePosixPath(name).stem
    suffix = PurePosixPath(name).suffix
    source_hash = hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:8]
    return f"/{stem}-{source_hash}{suffix}"


def _decode_store_content(value: dict[str, Any]) -> bytes:
    """Bytes of one store item's content (text or base64-encoded binary)."""
    content = value.get("content", "")
    if value.get("encoding") == "base64":
        return base64.standard_b64decode(content)
    return str(content).encode("utf-8")


def _write_workspace_file(bundle: AnalysisBackendBundle, name: str, data: bytes) -> str:
    """Write bytes under the workspace data directory; returns the relative path."""
    relative_path = f"{WORKSPACE_DATA_DIRECTORY}/{name.lstrip('/')}"
    disk_path = bundle.workspace_path / relative_path
    disk_path.parent.mkdir(parents=True, exist_ok=True)
    disk_path.write_bytes(data)
    return relative_path


def build_data_analysis_tools(
    context: GlobalContext,
    bundle: AnalysisBackendBundle,
    connection: McpConnection,
) -> list[Any]:
    """Build the per-turn data-analysis tool set.

    Every tool closes over the turn's ``AnalysisBackendBundle`` — the
    authenticated ``(user_id, assistant_id)`` pair scopes all store access,
    so one user's data is never reachable from another user's session and
    each avatar's buffer is distinct per user. The host filesystem tools are
    reached through ``connection`` — the per-user MCP connection saved for the
    bound avatar — so a turn only ever talks to that avatar's own server.
    """
    user_id = bundle.user_id
    assistant_id = bundle.assistant_id

    @tool
    async def discover_data_files(
        directory: str | None = None,
        recursive: bool = True,
    ) -> dict[str, Any]:
        """List data files available on the connected host filesystem.

        Use this tool first to find which files exist before ingesting.
        Omit the directory argument to list the connected data directory —
        the directory the Neural Nexus MCP data server exposes. The host only
        exposes allow-listed directories; asking for a directory outside the
        allow-list returns an error.

        Args:
            directory: Optional absolute host directory to list; omit to use
                the connected data directory.
            recursive: When true, include files in subdirectories.
        """
        if directory is None:
            if not connection.allowed_roots:
                return {"error": "The connected data server exposes no directories."}
            directory = connection.allowed_roots[0]
        try:
            file_paths = await call_mcp_filesystem_tool(
                connection,
                "list_all_files",
                {"directory": directory, "recursive": recursive},
            )
        except Exception as error:
            # Return the failure to the model rather than raising, so an
            # unreachable Neural Nexus MCP server degrades to a graceful reply
            # ("the data server seems unreachable") instead of crashing the turn.
            return {"error": f"Could not reach the data server: {error}"}
        if not isinstance(file_paths, list):
            return {"error": f"Unexpected listing result: {file_paths!r}"}
        return {
            "total_files": len(file_paths),
            "file_paths": file_paths[:_DISCOVER_RESULT_LIMIT],
            "truncated": len(file_paths) > _DISCOVER_RESULT_LIMIT,
        }

    @tool
    async def preview_data_file(file_path: str, n_rows: int = 5) -> Any:
        """Preview the first rows of one CSV or JSON file on the host.

        Use this tool to inspect a file's structure (columns, nesting)
        before deciding to ingest the file.

        Args:
            file_path: Absolute host path of the file to preview.
            n_rows: Number of rows to preview.
        """
        try:
            return await call_mcp_filesystem_tool(
                connection,
                "preview_data",
                {"file_path": file_path, "n_rows": n_rows},
            )
        except Exception as error:
            return {"error": f"Could not reach the data server: {error}"}

    @tool
    async def ingest_data_files(
        file_paths: list[str],
        runtime: Annotated[ToolRuntime, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Copy host data files into the analysis workspace and the persistent buffer.

        For each host file: the file's bytes are fetched from the host,
        written into the workspace directory "work/" (so shell commands can
        read the file with a relative path like "work/health1.json"), and —
        for text files — saved into the persistent per-avatar buffer under
        "/data_ingested/" so future conversations can reuse the data without
        re-fetching. Files that have not changed on the host since the last
        ingest are served from the buffer instead of being re-fetched;
        files whose host modification time changed are re-fetched and the
        buffered copy is overwritten. Binary files (for example images) are
        placed in the workspace only and are not persisted.

        Args:
            file_paths: Absolute host paths of the files to ingest.
        """
        max_bytes = int(context.data_analysis_store_max_bytes)
        ingested: list[str] = []
        reused_unchanged: list[str] = []
        workspace_paths: list[str] = []
        errors: list[str] = []
        namespace = ingested_namespace(user_id, assistant_id)

        # Basenames that repeat within this batch (distinct directories, same
        # file name) always take the hashed key so concurrent ingests cannot
        # race each other onto one plain-name key.
        basename_counts: dict[str, int] = {}
        for source_path in file_paths:
            name = PurePosixPath(source_path).name
            basename_counts[name] = basename_counts.get(name, 0) + 1

        async def ingest_one(source_path: str) -> None:
            try:
                file_info = await call_mcp_filesystem_tool(
                    connection, "get_file_info", {"file_path": source_path}
                )
                source_modified_at = file_info.get("modified_at")
                size_bytes = int(file_info.get("size_bytes") or 0)
                if size_bytes > max_bytes:
                    errors.append(
                        f"{source_path}: file size {size_bytes} exceeds the "
                        f"buffer quota of {max_bytes} bytes; skipped."
                    )
                    return

                if basename_counts[PurePosixPath(source_path).name] > 1:
                    # In-batch basename collision — force the deterministic
                    # hashed key by pretending a different source holds the
                    # plain name.
                    store_key = _store_key_for_source(source_path, "")
                    existing_item = await runtime.store.aget(namespace, store_key)
                else:
                    provisional_key = _store_key_for_source(source_path, None)
                    existing_item = await runtime.store.aget(
                        namespace, provisional_key
                    )
                    existing_source_path = (
                        (existing_item.value or {}).get("source_path")
                        if existing_item is not None
                        else None
                    )
                    store_key = _store_key_for_source(
                        source_path, existing_source_path
                    )
                    if store_key != provisional_key:
                        existing_item = await runtime.store.aget(
                            namespace, store_key
                        )

                if (
                    existing_item is not None
                    and (existing_item.value or {}).get("source_modified_at")
                    == source_modified_at
                ):
                    # Unchanged on the host — reuse the buffered copy.
                    data = _decode_store_content(existing_item.value)
                    workspace_paths.append(
                        _write_workspace_file(bundle, store_key, data)
                    )
                    reused_unchanged.append(source_path)
                    return

                content_b64 = await call_mcp_filesystem_tool(
                    connection, "read_file_bytes", {"file_path": source_path}
                )
                data = base64.standard_b64decode(content_b64)
                workspace_paths.append(_write_workspace_file(bundle, store_key, data))

                try:
                    text_content = data.decode("utf-8")
                    encoding = "utf-8"
                except UnicodeDecodeError:
                    # Binary content (for example an image) stays
                    # workspace-only: useful for the current conversation,
                    # never persisted.
                    ingested.append(f"{source_path} (workspace only, binary)")
                    return

                now_iso = datetime.now(UTC).isoformat()
                created_at = (
                    (existing_item.value or {}).get("created_at", now_iso)
                    if existing_item is not None
                    else now_iso
                )
                await runtime.store.aput(
                    namespace,
                    key=store_key,
                    value={
                        "content": text_content,
                        "encoding": encoding,
                        "created_at": created_at,
                        "modified_at": now_iso,
                        "source_path": source_path,
                        "source_modified_at": source_modified_at,
                        "size_bytes": size_bytes,
                    },
                )
                ingested.append(source_path)
            except Exception as error:
                # Log with traceback so per-file failures are diagnosable from
                # the server logs, not only from the model's summary of them.
                logger.warning(
                    "ingest_data_files failed for %s", source_path, exc_info=True
                )
                errors.append(f"{source_path}: {error}")

        # Fan out per-file fetches concurrently — a directory of daily exports
        # is dozens of files, and each file costs two Model Context Protocol
        # round trips (info + bytes); sequential ingest dominated turn latency.
        await asyncio.gather(*(ingest_one(path) for path in file_paths))

        evicted_keys = await enforce_ingested_quota(
            runtime.store, user_id, assistant_id, context
        )
        return {
            "workspace_paths": sorted(workspace_paths),
            "ingested": sorted(ingested),
            "reused_unchanged": sorted(reused_unchanged),
            "evicted_from_buffer": evicted_keys,
            "errors": errors,
        }

    @tool
    async def hydrate_ingested_data(
        runtime: Annotated[ToolRuntime, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Copy previously ingested data from the persistent buffer into the workspace.

        Use this tool when the user asks about data that was ingested in an
        earlier conversation: the persistent buffer survives across
        conversations, but the workspace starts empty each turn. After this
        tool runs, the buffered files are readable by shell commands under
        the "work/" directory.
        """
        namespace = ingested_namespace(user_id, assistant_id)
        items = await runtime.store.asearch(namespace, limit=1000)
        workspace_paths = [
            _write_workspace_file(bundle, item.key, _decode_store_content(item.value))
            for item in items
        ]
        return {
            "hydrated_count": len(workspace_paths),
            "workspace_paths": workspace_paths,
        }

    @tool
    async def persist_created_artifact(
        workspace_file_path: str,
        runtime: Annotated[ToolRuntime, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Save a produced report or plot from the workspace into durable storage.

        Use this tool after writing an analysis artifact (for example
        "report.md" or "plot.png") with the execute or write_file tools, so
        the artifact survives after the workspace is cleaned at the end of
        the turn. The artifact becomes readable in future conversations
        under "/data_created/".

        Args:
            workspace_file_path: Path of the produced file relative to the
                workspace root, for example "report.md" or "work/plot.png".
        """
        candidate_path = (
            bundle.workspace_path / workspace_file_path.lstrip("/")
        ).resolve()
        if not candidate_path.is_relative_to(bundle.workspace_path.resolve()):
            return {"error": f"Path escapes the workspace: {workspace_file_path}"}
        if not candidate_path.is_file():
            return {"error": f"No such workspace file: {workspace_file_path}"}

        data = candidate_path.read_bytes()
        try:
            content = data.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            content = base64.standard_b64encode(data).decode("ascii")
            encoding = "base64"

        store_key = f"/{candidate_path.name}"
        now_iso = datetime.now(UTC).isoformat()
        await runtime.store.aput(
            created_namespace(user_id, assistant_id),
            key=store_key,
            value={
                "content": content,
                "encoding": encoding,
                "created_at": now_iso,
                "modified_at": now_iso,
                "size_bytes": len(data),
            },
        )
        return {"persisted_path": f"{CREATED_ROUTE.rstrip('/')}{store_key}"}

    @tool
    async def list_persisted_data(
        runtime: Annotated[ToolRuntime, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """List the persistent buffer contents: ingested data and created artifacts.

        Use this tool to check what data is already available from earlier
        conversations before discovering or ingesting from the host again.
        """
        ingested_items = await runtime.store.asearch(
            ingested_namespace(user_id, assistant_id), limit=1000
        )
        created_items = await runtime.store.asearch(
            created_namespace(user_id, assistant_id), limit=1000
        )
        return {
            "ingested": [
                {
                    "path": f"{INGESTED_ROUTE.rstrip('/')}{item.key}",
                    "source_path": (item.value or {}).get("source_path"),
                    "source_modified_at": (item.value or {}).get("source_modified_at"),
                    "size_bytes": (item.value or {}).get("size_bytes"),
                }
                for item in ingested_items
            ],
            "created": [
                {
                    "path": f"{CREATED_ROUTE.rstrip('/')}{item.key}",
                    "modified_at": (item.value or {}).get("modified_at"),
                    "size_bytes": (item.value or {}).get("size_bytes"),
                }
                for item in created_items
            ],
        }

    @tool
    async def check_data_server_connection(
        runtime: Annotated[ToolRuntime, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Report the current Neural Nexus MCP data-server connection.

        Use this tool to answer natural-language questions like "are you
        connected to the MCP server / Neural Nexus server / my data server?".
        This tool set is present only while a connection is established, so a
        successful call means "connected". The result deliberately carries no
        server address or directory details — never reveal those in the chat.
        """
        return {
            "connected": True,
            "server": "Neural Nexus MCP data server",
        }

    @tool
    async def disconnect_data_server(
        runtime: Annotated[ToolRuntime, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Disconnect this avatar from the Neural Nexus MCP data server.

        Use this tool when the user asks in natural language to disconnect,
        unlink, or forget the data server (for example "disconnect from the
        Neural Nexus server"). It deletes the saved connection so future turns
        no longer have data-analysis tools. Automatic connection offers stay
        suppressed after an explicit disconnect; the user can reconnect at any
        time by asking to connect (the connect_data_server tool).
        """
        cleared = await clear_user_connection(runtime.store, user_id)
        # Suppress the automatic re-offer: the user just said "disconnect" —
        # immediately asking "connect?" again on the next message would nag.
        # An explicit "connect" request clears this marker again.
        await mark_declined(runtime.store, user_id, assistant_id, connection)
        return {
            "disconnected": cleared,
            "message": (
                "Disconnected from the Neural Nexus MCP data server."
                if cleared
                else "No active connection was found to disconnect."
            ),
        }

    return [
        discover_data_files,
        preview_data_file,
        ingest_data_files,
        hydrate_ingested_data,
        persist_created_artifact,
        list_persisted_data,
        check_data_server_connection,
        disconnect_data_server,
    ]


def build_connect_tool(
    context: GlobalContext, user_id: str, assistant_id: str
) -> Any:
    """Build the single tool an UNCONNECTED owned avatar carries: explicit connect.

    The automatic discovery offer (the ``mcp_discovery`` consent interrupt) is
    suppressed after a decline or an explicit disconnect — but a user saying
    "connect to the Neural Nexus MCP server" must always work. This tool
    discovers the announced server (bypassing the discovery failure backoff,
    because the user may have just started the server), saves the single
    per-user connection bound to this avatar, and clears the decline marker so
    automatic offers resume.
    """

    @tool
    async def connect_data_server(
        runtime: Annotated[ToolRuntime, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Connect this avatar to the Neural Nexus MCP data server.

        Use this tool when the user asks in natural language to connect,
        reconnect, or link the Neural Nexus MCP data server (for example
        "please connect to the Neural Nexus MCP Server"). Never reveal server
        addresses or directory paths in the reply — confirm the connection by
        name only.
        """
        connection = await discover_announced_server(
            context.data_analysis_mcp_discovery_url,
            float(context.data_analysis_discovery_timeout_seconds),
            ignore_failure_backoff=True,
        )
        if connection is None:
            return {
                "connected": False,
                "message": (
                    "No Neural Nexus data server is reachable right now. "
                    "Confirm the server is running, then ask to connect again."
                ),
            }
        await save_user_connection(
            runtime.store, user_id, connection=connection, assistant_id=assistant_id
        )
        await clear_declined(runtime.store, user_id, assistant_id)
        return {
            "connected": True,
            "message": (
                "Connected to the Neural Nexus MCP data server. "
                "Data analysis is available from the next message onward."
            ),
        }

    return connect_data_server
