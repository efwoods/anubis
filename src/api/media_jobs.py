"""In-process registry for background media-processing jobs.

``/update_avatar_identity_with_media`` used to block on the full ``process_media``
graph and timed out (~2 min) on long jobs — diarized audio, PDFs, and especially
YouTube playlists that expand into many videos. Instead, the endpoint now starts a
job (this module) and returns immediately; a separate SSE endpoint streams progress.

The job runs the already-compiled graph **in-process** (so the file bytes already
read from the upload and the live ``store`` need no JSON serialization). Graph nodes
emit ``{"type": "media_progress", ...}`` custom events via ``get_stream_writer()``
(the same mechanism the chat graph uses for ``assistant_token``); ``run_media_job``
forwards them into the job's event buffer.

NOTE: the registry is per-process. The LangGraph deployment here runs the graph
in-process, so a progress request lands on the same process that owns the job. A
future multi-worker deployment would need a shared store (Redis) or the LangGraph
runs API instead.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# Keep finished jobs around briefly so a late/reconnecting progress client can still
# read the final result, then drop them so the registry doesn't grow unbounded.
_FINISHED_TTL_SECONDS = 30 * 60
_MAX_JOBS = 1000


@dataclass
class MediaJob:
    """A single background media-processing job and its progress buffer."""

    job_id: str
    user_id: str
    assistant_id: Optional[str]
    status: str = "queued"  # queued | running | completed | error
    created_at: float = field(default_factory=time.time)
    # Epoch seconds when file processing actually began (status -> running) and
    # when it finished (completion or error), set by run_media_job / finish_job.
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    # Wall-clock seconds from processing start to completion/error, set by finish_job.
    duration_seconds: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    # Append-only history of progress payloads. Subscribers replay from index 0,
    # then wait on ``_updated`` for new appends — this supports late joiners and
    # multiple concurrent subscribers.
    events: List[Dict[str, Any]] = field(default_factory=list)
    _updated: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    task: Optional[asyncio.Task] = None


def create_job(
    registry: Dict[str, MediaJob], user_id: str, assistant_id: Optional[str]
) -> MediaJob:
    """Register and return a new queued job."""
    _cleanup(registry)
    job = MediaJob(job_id=str(uuid4()), user_id=user_id, assistant_id=assistant_id)
    registry[job.job_id] = job
    return job


def get_job(registry: Dict[str, MediaJob], job_id: str) -> Optional[MediaJob]:
    """Return the job for ``job_id``, or ``None`` if unknown/expired."""
    return registry.get(job_id)


def add_event(job: MediaJob, payload: Dict[str, Any]) -> None:
    """Append a progress payload and wake any waiting subscribers."""
    job.events.append(payload)
    job._updated.set()


def finish_job(
    job: MediaJob,
    *,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """Mark the job completed or errored and wake subscribers."""
    job.finished_at = time.time()
    # Measure from when processing actually started; fall back to creation time
    # if the job errored before it began running.
    job.duration_seconds = round(
        job.finished_at - (job.started_at or job.created_at), 3
    )
    if error is not None:
        job.status = "error"
        job.error = error
    else:
        job.status = "completed"
        job.result = result
    job._updated.set()
    job.done.set()


def _cleanup(registry: Dict[str, MediaJob]) -> None:
    """Drop finished jobs past their TTL; trim oldest if the registry is too large."""
    now = time.time()
    stale = [
        jid
        for jid, job in registry.items()
        if job.finished_at is not None
        and now - job.finished_at > _FINISHED_TTL_SECONDS
    ]
    for jid in stale:
        registry.pop(jid, None)

    if len(registry) > _MAX_JOBS:
        oldest = sorted(registry.values(), key=lambda j: j.created_at)
        for job in oldest[: len(registry) - _MAX_JOBS]:
            registry.pop(job.job_id, None)


async def run_media_job(
    job: MediaJob,
    media_files: List[Dict[str, Any]],
    config: Dict[str, Any],
    store: Any,
    context: Any,
) -> None:
    """Run the ``process_media`` graph in the background, recording progress.

    Compiles the graph the same way the endpoint did, streams custom +
    update events, forwards ``media_progress`` payloads into the job buffer,
    and records the final result/error. Never raises — failures land on the
    job (and surface to clients via the SSE ``done`` event).
    """
    try:
        from src.subgraphs.process_media_graph.process_media_graph_api_endpoint import (
            workflow,
        )

        compiled = workflow.compile(store=store)
        job.started_at = time.time()
        job.status = "running"

        # index_docs does not raise on per-file indexing failures; it reports
        # them on ``failed_to_index_files`` (accumulated via operator.add). We
        # collect those entries from the "updates" stream so the silent-success
        # bug stays fixed — the failed files are surfaced on the job result
        # rather than the upload reporting success unconditionally.
        failed_files: List[Dict[str, Any]] = []
        async for item in compiled.astream(
            {"media_files": media_files},
            config=config,
            context=context,
            stream_mode=["custom", "updates"],
            subgraphs=True,
        ):
            if not isinstance(item, tuple) or len(item) != 3:
                continue
            _ns, mode, payload = item
            if (
                mode == "custom"
                and isinstance(payload, dict)
                and payload.get("type") == "media_progress"
            ):
                add_event(job, payload)
            elif mode == "updates" and isinstance(payload, dict):
                # Each value is a node's state update; index_docs reports
                # ``failed_to_index_files``. Gather them across all nodes/passes.
                for node_update in payload.values():
                    if not isinstance(node_update, dict):
                        continue
                    node_failures = node_update.get("failed_to_index_files")
                    if node_failures:
                        failed_files.extend(node_failures)

        requested_filenames = [m.get("filename") for m in media_files]
        failed_filenames = {
            f.get("filename")
            for f in failed_files
            if isinstance(f, dict) and f.get("filename") is not None
        }
        indexed_filenames = [
            name for name in requested_filenames if name not in failed_filenames
        ]

        if failed_files:
            message = (
                "Some files failed to index and should be reprocessed: "
                + ", ".join(sorted(n for n in failed_filenames if n))
            )
        else:
            message = "Media processed and indexed successfully"

        finish_job(
            job,
            result={
                "items_processed": len(media_files),
                "filenames": requested_filenames,
                "indexed_filenames": indexed_filenames,
                "failed_files": failed_files,
                "message": message,
            },
        )
    except Exception as exc:  # noqa: BLE001 - surface every failure via the job
        logger.exception("Media job %s failed: %s", job.job_id, exc)
        finish_job(job, error=str(exc))
