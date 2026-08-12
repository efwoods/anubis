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

**Several machines at once.** A user connects the Neural Nexus daemon on an
Ubuntu desktop, a Mac, a phone, and a Windows machine simultaneously, so the
tools close over a LIST of connections rather than one. Two addressing shapes
are used, chosen per tool by what the model already knows:

- Tools that answer "what do I have?" (``discover_data_files``) **fan out**:
  every connected machine is queried concurrently and results come back grouped
  by device label. A machine that is asleep contributes an ``offline`` entry
  instead of failing the call, so one sleeping laptop never costs the user an
  answer — but the entry is present so the model can say a machine was not
  reached rather than presenting partial data as complete.
- Tools that name one absolute host path (``preview_data_file``,
  ``ingest_data_files``) resolve the owning machine from that path against each
  device's allow-listed roots. Broadcasting a path to every device would cost a
  round trip per machine and produce confusing failures from the machines that
  do not hold the file.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import mimetypes
import re
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
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
from src.anubis.utils.tools.data_analysis.devices import (
    connection_label_map,
    resolve_device_for_path,
)
from src.anubis.utils.tools.data_analysis.discovery import (
    McpConnection,
    clear_declined,
    clear_user_connection,
    mark_declined,
    resolve_available_connections,
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


# Underscored calendar date and clock time appended to every created artifact
# name, e.g. ``report_2026_07_27_14_30_05.md``. Underscores throughout (rather
# than dashes or colons) keep the name safe as a file name on every platform
# and as a store key.
_ARTIFACT_TIMESTAMP_FORMAT = "%Y_%m_%d_%H_%M_%S"
_ARTIFACT_TIMESTAMP_PATTERN = re.compile(r"_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$")


def timestamped_artifact_name(file_name: str, moment: datetime) -> str:
    """Artifact file name carrying the date and time the artifact was saved.

    ``report.md`` becomes ``report_2026_07_27_14_30_05.md`` and ``plot.png``
    becomes ``plot_2026_07_27_14_30_05.png``. Store keys are basenames, so
    without the date and time every analysis turn would overwrite the previous
    turn's report and plot; with the date and time each report and plot is a
    distinct, self-describing record the conversation partner can go back to.

    Applying the date and time is idempotent: a name the model already
    timestamped (the capability prompt asks the model to do exactly that) is
    returned unchanged rather than stamped twice.
    """
    stem = PurePosixPath(file_name).stem
    suffix = PurePosixPath(file_name).suffix
    if _ARTIFACT_TIMESTAMP_PATTERN.search(stem):
        return file_name
    return f"{stem}_{moment.strftime(_ARTIFACT_TIMESTAMP_FORMAT)}{suffix}"


def _artifact_mime_type(file_name: str) -> str:
    """MIME type for one created artifact, derived from the file name.

    The store value carries no MIME field (see ``persist_workspace_file``), so
    the extension is the only signal available to a client deciding whether to
    render an artifact as an image, as markdown, or as a download link.
    ``mimetypes`` does not know ``.md`` on every platform, so markdown is
    mapped explicitly rather than falling back to ``text/plain``.
    """
    suffix = PurePosixPath(file_name).suffix.lower()
    if suffix in (".md", ".markdown"):
        return "text/markdown"
    guessed_mime_type, _encoding = mimetypes.guess_type(file_name)
    return guessed_mime_type or "application/octet-stream"


async def persist_workspace_file(
    bundle: AnalysisBackendBundle,
    store: Any,
    candidate_path: Path,
) -> dict[str, Any]:
    """Save one workspace file into the durable created-artifact namespace.

    Shared by the ``persist_created_artifact`` tool (model-driven) and
    ``collect_turn_artifacts`` (the end-of-turn sweep), so both write exactly
    the same store record and both append to ``bundle.persisted_artifacts``.

    Binary files are base64-encoded because the store holds JSON values; text
    files stay readable so the deep agent can ``read_file`` an earlier
    conversation's report back out of ``/data_created/``.

    The saved name carries the date and time (see
    :func:`timestamped_artifact_name`), so each turn's report and plot is kept
    rather than overwriting the previous turn's.
    """
    data = candidate_path.read_bytes()
    try:
        content = data.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        content = base64.standard_b64encode(data).decode("ascii")
        encoding = "base64"

    saved_at = datetime.now(UTC)
    artifact_name = timestamped_artifact_name(candidate_path.name, saved_at)
    store_key = f"/{artifact_name}"
    now_iso = saved_at.isoformat()
    await store.aput(
        created_namespace(bundle.user_id, bundle.assistant_id),
        key=store_key,
        value={
            "content": content,
            "encoding": encoding,
            "created_at": now_iso,
            "modified_at": now_iso,
            "size_bytes": len(data),
        },
    )

    record = {
        "name": artifact_name,
        "workspace_name": candidate_path.name,
        "mime_type": _artifact_mime_type(artifact_name),
        "size_bytes": len(data),
        "created_at": now_iso,
        "persisted_path": f"{CREATED_ROUTE.rstrip('/')}{store_key}",
        "encoding": encoding,
        "content": content,
    }
    # Deduplicated on the WORKSPACE name, not the saved name: a turn that
    # regenerates plot.png and persists the file twice produces two different
    # timestamped names, and the conversation partner should see the finished
    # plot once rather than two near-identical charts.
    bundle.persisted_artifacts = [
        existing
        for existing in bundle.persisted_artifacts
        if existing["workspace_name"] != record["workspace_name"]
    ]
    bundle.persisted_artifacts.append(record)
    return record


async def collect_turn_artifacts(
    context: GlobalContext, bundle: AnalysisBackendBundle
) -> list[dict[str, Any]]:
    """Persist and return every artifact this turn produced, for display.

    Called once at the end of the turn, before the workspace is wiped. Two
    sources are merged:

    1. artifacts the model explicitly saved with ``persist_created_artifact``;
    2. a sweep of the workspace root for files the model wrote but never
       persisted — the prompt tells the model to write ``report.md`` /
       ``plot.png`` and then persist them, and a skipped persist call would
       otherwise silently discard the very thing the user asked for.

    The sweep is deliberately non-recursive: the ``work/`` subdirectory holds
    ingested *input* data, which is already persisted in its own namespace and
    is not an artifact of this turn.

    Returned records carry inline ``content`` so the client can render the
    report and plot without a second fetch. Content above
    ``data_analysis_inline_artifact_max_bytes`` is dropped from the record
    (the durable store copy is unaffected) so one oversized file cannot bloat
    the checkpointed message it rides on.
    """
    workspace_root = bundle.workspace_path
    if bundle.store is not None and workspace_root.is_dir():
        # Compared on the workspace name because the saved name carries a
        # timestamp the workspace file does not — matching on the saved name
        # would re-persist every file the model already saved.
        already_persisted = {
            record["workspace_name"] for record in bundle.persisted_artifacts
        }
        for candidate_path in sorted(workspace_root.iterdir()):
            if not candidate_path.is_file() or candidate_path.name in already_persisted:
                continue
            try:
                await persist_workspace_file(bundle, bundle.store, candidate_path)
            except Exception:
                logger.exception(
                    "Could not persist swept analysis artifact %s", candidate_path
                )

    inline_byte_cap = int(context.data_analysis_inline_artifact_max_bytes or 0)
    display_records: list[dict[str, Any]] = []
    for record in bundle.persisted_artifacts:
        display_record = dict(record)
        if inline_byte_cap and record["size_bytes"] > inline_byte_cap:
            display_record["content"] = None
            display_record["omitted_reason"] = "too_large"
        display_records.append(display_record)
    return display_records


# Fallback per-device deadline for one fan-out leg, used when the context does
# not carry ``data_analysis_device_fanout_timeout_seconds``. A device that has
# not answered within this window is reported as offline rather than awaited
# further: fan-out legs run concurrently, so this is the ceiling the whole
# fan-out call adds to the turn regardless of how many machines are connected.
_DEFAULT_DEVICE_FANOUT_TIMEOUT_SECONDS = 20.0


def _device_fanout_timeout_seconds(context: GlobalContext) -> float:
    """Per-device deadline for one leg of a fan-out call."""
    configured = getattr(context, "data_analysis_device_fanout_timeout_seconds", None)
    if configured is None:
        return _DEFAULT_DEVICE_FANOUT_TIMEOUT_SECONDS
    try:
        return float(configured)
    except (TypeError, ValueError):
        return _DEFAULT_DEVICE_FANOUT_TIMEOUT_SECONDS


async def _call_one_device(
    connection: McpConnection,
    tool_name: str,
    tool_args: dict[str, Any],
    timeout_seconds: float,
) -> Any:
    """Invoke one Model Context Protocol tool on one device, never raising.

    Returns the tool's normalized result, or a status mapping describing why the
    machine did not answer. Failures are converted rather than raised because a
    fan-out call must survive a machine that is asleep, and because a raised
    exception inside ``asyncio.gather`` would lose which device produced it.
    """
    try:
        return await asyncio.wait_for(
            call_mcp_filesystem_tool(connection, tool_name, tool_args),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        logger.info(
            "Device %r did not answer %s within %.1fs; reported offline.",
            connection.device_label,
            tool_name,
            timeout_seconds,
        )
        return {
            "status": "offline",
            "detail": "The machine did not answer before the deadline.",
        }
    except Exception as error:
        logger.warning(
            "Device %r failed %s", connection.device_label, tool_name, exc_info=True
        )
        return {"status": "offline", "detail": str(error)}


def _select_devices(
    connections: list[McpConnection], device_label: str | None
) -> tuple[list[McpConnection], dict[str, Any] | None]:
    """Resolve a model-supplied device label to the devices to query.

    Returns ``(devices, error)``. A missing label selects every device (the
    fan-out default). An unrecognized label returns an error naming the valid
    labels, so the model can correct itself in one step instead of guessing.
    """
    if device_label is None or not str(device_label).strip():
        return connections, None
    label_map = connection_label_map(connections)
    selected = label_map.get(str(device_label).strip().lower())
    if selected is None:
        return [], {
            "error": (
                f"No connected machine is named {device_label!r}. "
                f"Connected machines: {[c.device_label for c in connections]}."
            )
        }
    return [selected], None


def build_data_analysis_tools(
    context: GlobalContext,
    bundle: AnalysisBackendBundle,
    connections: list[McpConnection],
) -> list[Any]:
    """Build the per-turn data-analysis tool set across every connected machine.

    Every tool closes over the turn's ``AnalysisBackendBundle`` — the
    authenticated ``(user_id, assistant_id)`` pair scopes all store access,
    so one user's data is never reachable from another user's session and
    each avatar's buffer is distinct per user. The host filesystem tools are
    reached through ``connections`` — the per-device MCP connections saved for
    the bound avatar — so a turn only ever talks to that avatar's own machines.
    """
    user_id = bundle.user_id
    assistant_id = bundle.assistant_id
    fanout_timeout_seconds = _device_fanout_timeout_seconds(context)

    def _ambiguous_path_error(file_path: str) -> dict[str, Any]:
        return {
            "error": (
                f"Could not tell which machine holds {file_path!r}. "
                f"Name the machine explicitly with the device_label argument. "
                f"Connected machines: {[c.device_label for c in connections]}."
            )
        }

    def _device_for_path(
        file_path: str, device_label: str | None
    ) -> tuple[McpConnection | None, dict[str, Any] | None]:
        """Pick the one machine a named host path belongs to."""
        selected, error = _select_devices(connections, device_label)
        if error is not None:
            return None, error
        if len(selected) == 1:
            return selected[0], None
        resolved = resolve_device_for_path(file_path, selected)
        if resolved is None:
            return None, _ambiguous_path_error(file_path)
        return resolved, None

    @tool
    async def discover_data_files(
        directory: str | None = None,
        recursive: bool = True,
        device_label: str | None = None,
    ) -> dict[str, Any]:
        """List data files available on the connected machines.

        Use this tool first to find which files exist before ingesting. Results
        come back grouped by machine name, so state which machine a file is on
        when reporting findings. A machine that is asleep or unreachable appears
        with a status of "offline" instead of failing the whole call — say so
        plainly rather than presenting the remaining results as complete.

        Omit the directory argument to list each machine's connected data
        directory. Each machine only exposes allow-listed directories; asking
        for a directory outside the allow-list returns an error for that machine.

        Args:
            directory: Optional absolute host directory to list; omit to use
                each machine's connected data directory.
            recursive: When true, include files in subdirectories.
            device_label: Optional name of a single machine to list, for example
                "Ubuntu" or "My MacBook". Omit to list every connected machine.
        """
        selected, error = _select_devices(connections, device_label)
        if error is not None:
            return error

        async def list_one(connection: McpConnection) -> tuple[str, Any]:
            target_directory = directory
            if target_directory is None:
                if not connection.allowed_roots:
                    return connection.device_label, {
                        "error": "This machine exposes no directories."
                    }
                target_directory = connection.allowed_roots[0]
            result = await _call_one_device(
                connection,
                "list_all_files",
                {"directory": target_directory, "recursive": recursive},
                fanout_timeout_seconds,
            )
            if isinstance(result, dict) and result.get("status") == "offline":
                return connection.device_label, result
            if not isinstance(result, list):
                return connection.device_label, {
                    "error": f"Unexpected listing result: {result!r}"
                }
            return connection.device_label, {
                "total_files": len(result),
                "file_paths": result[:_DISCOVER_RESULT_LIMIT],
                "truncated": len(result) > _DISCOVER_RESULT_LIMIT,
            }

        # Legs run concurrently so the call costs one device's latency rather
        # than the sum across every connected machine.
        outcomes = await asyncio.gather(
            *(list_one(connection) for connection in selected)
        )
        return {"devices": dict(outcomes)}

    @tool
    async def preview_data_file(
        file_path: str, n_rows: int = 5, device_label: str | None = None
    ) -> Any:
        """Preview the first rows of one CSV or JSON file on one machine.

        Use this tool to inspect a file's structure (columns, nesting)
        before deciding to ingest the file. The machine holding the file is
        worked out from the path; name the machine explicitly only when the
        result says the path was ambiguous.

        Args:
            file_path: Absolute host path of the file to preview.
            n_rows: Number of rows to preview.
            device_label: Optional name of the machine holding the file.
        """
        connection, error = _device_for_path(file_path, device_label)
        if error is not None:
            return error
        return await _call_one_device(
            connection,
            "preview_data",
            {"file_path": file_path, "n_rows": n_rows},
            fanout_timeout_seconds,
        )

    @tool
    async def ingest_data_files(
        file_paths: list[str],
        device_label: str | None = None,
        runtime: Annotated[ToolRuntime, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Copy host data files into the analysis workspace and the persistent buffer.

        For each host file: the file's bytes are fetched from the machine that
        holds the file, written into the workspace directory "work/" (so shell
        commands can read the file with a relative path like
        "work/health1.json"), and — for text files — saved into the persistent
        per-avatar buffer under "/data_ingested/" so future conversations can
        reuse the data without re-fetching. Files that have not changed on the
        host since the last ingest are served from the buffer instead of being
        re-fetched; files whose host modification time changed are re-fetched and
        the buffered copy is overwritten. Binary files (for example images) are
        placed in the workspace only and are not persisted.

        Paths from different machines may be mixed in one call: the machine
        holding each file is worked out from that file's path.

        Args:
            file_paths: Absolute host paths of the files to ingest.
            device_label: Optional name of the machine holding every listed
                file. Omit unless a previous result said a path was ambiguous.
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
            connection, resolution_error = _device_for_path(source_path, device_label)
            if resolution_error is not None:
                errors.append(f"{source_path}: {resolution_error['error']}")
                return
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
                    existing_item = await runtime.store.aget(namespace, provisional_key)
                    existing_source_path = (
                        (existing_item.value or {}).get("source_path")
                        if existing_item is not None
                        else None
                    )
                    store_key = _store_key_for_source(source_path, existing_source_path)
                    if store_key != provisional_key:
                        existing_item = await runtime.store.aget(namespace, store_key)

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
                        # Which machine the data came from, so a later
                        # conversation reading the buffer can still attribute
                        # the data to a machine without re-contacting the host.
                        "device_id": connection.device_id,
                        "device_label": connection.device_label,
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

        Use this tool after writing an analysis artifact with the execute or
        write_file tools, so the artifact survives after the workspace is
        cleaned at the end of the turn. The artifact becomes readable in
        future conversations under "/data_created/".

        The saved artifact name always carries the current date and time,
        separated by underscores, so that every report and plot is unique and
        no report or plot ever replaces an earlier one — for example
        "report_2026_07_27_14_30_05.md" and "plot_2026_07_27_14_30_05.png".
        The date and time are appended automatically when the name given does
        not already carry a date and time, so passing plain "report.md" is
        also correct.

        Args:
            workspace_file_path: Path of the produced file relative to the
                workspace root, for example
                "report_2026_07_27_14_30_05.md" or "work/plot.png".
        """
        candidate_path = (
            bundle.workspace_path / workspace_file_path.lstrip("/")
        ).resolve()
        if not candidate_path.is_relative_to(bundle.workspace_path.resolve()):
            return {"error": f"Path escapes the workspace: {workspace_file_path}"}
        if not candidate_path.is_file():
            return {"error": f"No such workspace file: {workspace_file_path}"}

        record = await persist_workspace_file(bundle, runtime.store, candidate_path)
        return {"persisted_path": record["persisted_path"]}

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
                    # Absent for data ingested before machines were tracked
                    # individually; the model reports the machine only when the
                    # machine is known.
                    "device_label": (item.value or {}).get("device_label"),
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
        """Report which machines are connected through Neural Nexus.

        Use this tool to answer natural-language questions like "are you
        connected to the MCP server / Neural Nexus server / my data server?" and
        "which of my machines can you see?". Report the machines by name and say
        which are currently reachable. The result deliberately carries no server
        address, port, or directory details — never reveal those in the chat.
        """
        from src.anubis.utils.tools.data_analysis import relay

        return {
            "connected": True,
            "server": "Neural Nexus MCP data server",
            "device_count": len(connections),
            "devices": [
                {
                    "device_label": connection.device_label,
                    "platform": connection.platform,
                    "online": relay.is_online(connection.device_id),
                }
                for connection in connections
            ],
        }

    @tool
    async def disconnect_data_server(
        device_label: str | None = None,
        runtime: Annotated[ToolRuntime, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Disconnect one machine, or every machine, from this avatar.

        Use this tool when the user asks in natural language to disconnect,
        unlink, or forget a machine or the data server (for example "disconnect
        from the Neural Nexus server" or "disconnect my phone"). Name the machine
        when the user named one; omit the name to disconnect every machine.

        A disconnected machine stays disconnected — it is not picked up again
        automatically — until the user asks to connect it (the connect_data_server
        tool).

        Args:
            device_label: Optional name of a single machine to disconnect. Omit
                to disconnect every connected machine.
        """
        selected, error = _select_devices(connections, device_label)
        if error is not None:
            return error

        removed_labels: list[str] = []
        for connection in selected:
            removed = await clear_user_connection(
                runtime.store, user_id, connection.device_id
            )
            if removed:
                removed_labels.append(connection.device_label)
            # Suppress automatic re-adoption: the user just said "disconnect",
            # and without this marker the next conversation turn would silently
            # bind the machine again. An explicit "connect" clears the marker.
            await mark_declined(runtime.store, user_id, assistant_id, connection)

        return {
            "disconnected": bool(removed_labels),
            "disconnected_devices": removed_labels,
            "message": (
                "Disconnected from the Neural Nexus MCP data server on "
                + ", ".join(removed_labels)
                + "."
                if removed_labels
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


def build_connect_tool(context: GlobalContext, user_id: str, assistant_id: str) -> Any:
    """Build the explicit-connect tool the personal avatar always carries.

    Devices are adopted automatically (``mcp_auto_adopt``), but adoption is
    suppressed for a machine the user explicitly disconnected — so a user saying
    "connect my laptop again" must always work. This tool resolves every
    reachable machine (bypassing the discovery failure backoff, because the user
    may have just started a daemon), binds them to this avatar, and clears the
    suppression markers so automatic adoption resumes.

    The tool is attached even when machines are already connected, because with
    several machines a user can have one connected and another suppressed.
    """

    @tool
    async def connect_data_server(
        device_label: str | None = None,
        runtime: Annotated[ToolRuntime, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Connect this avatar to the user's Neural Nexus machines.

        Use this tool when the user asks in natural language to connect,
        reconnect, or link a machine or the Neural Nexus MCP data server (for
        example "please connect to the Neural Nexus MCP Server" or "reconnect my
        desktop"). Never reveal server addresses, ports, or directory paths in
        the reply — confirm each connection by machine name only.

        Args:
            device_label: Optional name of a single machine to connect. Omit to
                connect every machine that is currently reachable.
        """
        available = await resolve_available_connections(
            runtime.store,
            user_id,
            context,
            ignore_failure_backoff=True,
        )
        selected, error = _select_devices(available, device_label)
        if error is not None:
            return error
        if not selected:
            return {
                "connected": False,
                "message": (
                    "No Neural Nexus data server is reachable right now. "
                    "Confirm the server is running, then ask to connect again."
                ),
            }

        connected_labels: list[str] = []
        for connection in selected:
            await save_user_connection(
                runtime.store,
                user_id,
                connection=connection,
                assistant_id=assistant_id,
            )
            await clear_declined(
                runtime.store, user_id, assistant_id, connection.device_id
            )
            connected_labels.append(connection.device_label)

        return {
            "connected": True,
            "connected_devices": connected_labels,
            "message": (
                "Connected to the Neural Nexus MCP data server on "
                + ", ".join(connected_labels)
                + ". Data analysis is available from the next message onward."
            ),
        }

    return connect_data_server
