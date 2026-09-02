"""In-process registry for background media-processing jobs.

``/update_avatar_identity_with_media`` used to block on the full ``process_media``
graph and timed out (~2 min) on long jobs — diarized audio, PDFs, and especially
YouTube playlists that expand into many videos. Instead, the endpoint now starts a
job (this module) and returns immediately; a separate SSE endpoint streams progress.

The job runs the already-compiled graph **in-process** (so the file bytes already
read from the upload and the live ``store`` need no JSON serialization). Graph nodes
emit ``{"type": "media_progress", ...}`` custom events via ``get_stream_writer()``
(the same mechanism the chat graph uses for ``assistant_token``);
``run_single_item_job`` forwards them into the job's event buffer (and the batch
master's aggregate buffer).

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
from typing import Any, Awaitable, Callable, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# Keep finished jobs around briefly so a late/reconnecting progress client can still
# read the final result, then drop them so the registry doesn't grow unbounded.
_FINISHED_TTL_SECONDS = 30 * 60
_MAX_JOBS = 1000


@dataclass
class MediaJob:
    """A single background media-processing job and its progress buffer.

    A batch upload produces one **master** job (``is_master=True``, ``child_ids``
    populated) plus one **child** job per top-level item (``parent_id`` set,
    ``filename`` / ``namespace_filename`` identifying the item). The master
    aggregates every child's progress events; each child also has its own buffer so
    a client can stream a single item. Either can be cancelled via its ``task``.
    """

    job_id: str
    user_id: str
    assistant_id: Optional[str]
    status: str = "queued"  # queued | running | completed | error | cancelled
    created_at: float = field(default_factory=time.time)
    # Epoch seconds when file processing actually began (status -> running) and
    # when it finished (completion or error), set by run_single_item_job /
    # run_batch_media_job / finish_job.
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    # Wall-clock seconds from processing start to completion/error, set by finish_job.
    duration_seconds: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    # Master/child wiring. ``parent_id`` is set on children; ``child_ids`` and
    # ``is_master`` on the batch master. ``filename`` / ``namespace_filename``
    # identify a child's single item (used for rollback + per-item reporting).
    is_master: bool = False
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    filename: Optional[str] = None
    namespace_filename: Optional[str] = None
    # Set when a cancel was requested so progress subscribers and the runner can
    # see the item/batch is being torn down.
    cancelled: bool = False
    # Pre-request token estimate for this child's item (image dimensions,
    # audio/video duration, extracted-text words — see billing/estimation.py),
    # stamped at submit (or at playlist expansion) so job status/progress can
    # show what the item was billed against.
    estimated_tokens: int | None = None
    # How many Documents this job indexed into the ``quote`` namespace — the
    # avatar's direct quotes. ``run_batch_media_job`` recalibrates the avatar's
    # direct-quote cloud after the batch only when this is non-zero, so a
    # PDF-only or identity-only upload does not pay for a leave-one-out fit that
    # would produce exactly the previous result. Counted from the store writes
    # the graph reports (``indexed_namespace_counts``), and accumulated onto both
    # the child and its master.
    #
    # Lives on the job rather than in ``result`` because a child that ends in
    # ``error`` is finished with no result dict at all, which would discard the
    # count for a partially successful item.
    indexed_quote_document_count: int = 0
    # Append-only history of progress payloads. Subscribers replay from index 0,
    # then wait on ``_updated`` for new appends — this supports late joiners and
    # multiple concurrent subscribers.
    events: List[Dict[str, Any]] = field(default_factory=list)
    _updated: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    task: Optional[asyncio.Task] = None


# Shared per-batch concurrency limiters, keyed by master job_id. The graph node
# looks the semaphore up (lazy import) so every item in a batch — and every
# playlist/linktree child that expands inside it — draws from one global pool of
# ``media_processing_concurrency`` slots instead of N independent per-item pools.
_batch_semaphores: Dict[str, asyncio.Semaphore] = {}


def set_batch_semaphore(master_job_id: str, semaphore: asyncio.Semaphore) -> None:
    """Register the shared concurrency limiter for a batch."""
    _batch_semaphores[master_job_id] = semaphore


def get_batch_semaphore(master_job_id: Optional[str]) -> Optional[asyncio.Semaphore]:
    """Return the shared limiter for a batch, or ``None`` if unknown."""
    if not master_job_id:
        return None
    return _batch_semaphores.get(master_job_id)


def clear_batch_semaphore(master_job_id: str) -> None:
    """Drop a batch's limiter once the batch is done."""
    _batch_semaphores.pop(master_job_id, None)


def create_master_job(
    registry: Dict[str, MediaJob], user_id: str, assistant_id: Optional[str]
) -> MediaJob:
    """Register and return the batch master job (no items attached yet)."""
    _cleanup(registry)
    job = MediaJob(
        job_id=str(uuid4()),
        user_id=user_id,
        assistant_id=assistant_id,
        is_master=True,
    )
    registry[job.job_id] = job
    return job


def create_child_job(
    registry: Dict[str, MediaJob],
    *,
    user_id: str,
    assistant_id: Optional[str],
    parent_id: str,
    filename: Optional[str],
    namespace_filename: Optional[str],
    estimated_tokens: int | None = None,
) -> MediaJob:
    """Register and return one per-item child job under ``parent_id``."""
    job = MediaJob(
        job_id=str(uuid4()),
        user_id=user_id,
        assistant_id=assistant_id,
        parent_id=parent_id,
        filename=filename,
        namespace_filename=namespace_filename,
        estimated_tokens=estimated_tokens,
    )
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
    cancelled: bool = False,
) -> None:
    """Mark the job completed, errored, or cancelled and wake subscribers.

    Idempotent: a job already ``done`` (e.g. a child that finished a moment before
    a batch cancel reached it) is left untouched.
    """
    if job.done.is_set():
        return
    job.finished_at = time.time()
    # Measure from when processing actually started; fall back to creation time
    # if the job errored before it began running.
    job.duration_seconds = round(
        job.finished_at - (job.started_at or job.created_at), 3
    )
    if cancelled:
        job.status = "cancelled"
        job.cancelled = True
        if result is not None:
            job.result = result
    elif error is not None:
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


async def run_single_item_job(
    child: MediaJob,
    master: MediaJob,
    media_file: Dict[str, Any],
    config: Dict[str, Any],
    store: Any,
    context: Any,
    existing_namespaces: Optional[List[str]] = None,
) -> None:
    """Run the ``process_media`` graph for a single top-level item.

    Progress events are forwarded to **both** the child's own buffer and the
    master's aggregate buffer (tagged with the child's ``job_id``/``filename`` so
    the master stream can attribute each event). ``item_job_id`` / ``master_job_id``
    are threaded through ``config.configurable`` so the graph stamps them onto every
    indexed Document — that is what lets a per-item or batch cancel delete exactly
    the rows this run wrote (including playlist children). Never raises: a failure
    or cancellation lands on the child job.
    """
    item_config = {
        **config,
        "configurable": {
            **config.get("configurable", {}),
            "item_job_id": child.job_id,
            "master_job_id": master.job_id,
        },
    }
    try:
        from src.subgraphs.process_media_graph.process_media_graph_api_endpoint import (
            workflow,
        )

        compiled = workflow.compile(store=store)
        child.started_at = time.time()
        child.status = "running"

        # The graph swallows per-item failures (partial success) and reports them
        # as ``item_error`` events + an ``errors``/``indexed`` count on
        # ``converting_complete`` rather than raising. Track those so this item's
        # job status reflects whether it actually produced anything — otherwise a
        # video that failed (e.g. missing reference audio) would still report
        # "completed". See convert_media_list_to_text_document.
        item_errors: List[str] = []
        last_complete: Optional[Dict[str, Any]] = None

        # index_docs does not raise on per-file indexing failures; it reports
        # them on ``failed_to_index_files`` (accumulated via operator.add). We
        # collect those entries from the "updates" stream so the silent-success
        # bug stays fixed — the failed files are surfaced on the job result
        # rather than the upload reporting success unconditionally.
        failed_files: List[Dict[str, Any]] = []
        async for item in compiled.astream(
            {
                "media_files": [media_file],
                "existing_namespaces": list(existing_namespaces or []),
            },
            config=item_config,
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
                add_event(child, payload)
                # Mirror into the master stream, attributed to this item.
                add_event(
                    master,
                    {
                        **payload,
                        "item_job_id": child.job_id,
                        "item_filename": child.filename,
                    },
                )
                stage = payload.get("stage")
                if stage == "item_error":
                    item_errors.append(str(payload.get("error") or "unknown"))
                elif stage == "converting_complete":
                    last_complete = payload
                elif stage == "indexed_namespace_counts":
                    # Accumulated onto the master as well as the child: the
                    # once-per-upload recalibration decision is made at the batch
                    # level, and one video out of eighteen contributing quotes is
                    # enough to make the whole batch worth recalibrating. The
                    # payload crosses the graph boundary, so validate its shape
                    # rather than trusting it.
                    namespace_counts = payload.get("namespace_counts")
                    if isinstance(namespace_counts, dict):
                        quote_document_count = namespace_counts.get("quote", 0)
                        if isinstance(quote_document_count, int) and quote_document_count > 0:
                            child.indexed_quote_document_count += quote_document_count
                            master.indexed_quote_document_count += quote_document_count

        # Decide the item's final status from what the graph actually did. A total
        # failure (errors and nothing indexed — e.g. a video with no subtitles and
        # no reference audio) is an error; a partial failure (some children of a
        # playlist failed but others indexed) stays completed with the errors
        # surfaced in the result.
        indexed = (last_complete or {}).get("indexed")
        skipped = (last_complete or {}).get("skipped")
        if item_errors and not indexed:
            finish_job(child, error="; ".join(item_errors[:10]))
        else:
            finish_job(
                child,
                result={
                    "items_processed": 1,
                    "indexed": indexed,
                    "skipped": skipped,
                    "errors": item_errors or None,
                    "filename": child.filename,
                    "namespace_filename": child.namespace_filename,
                    "message": (
                        "Item already indexed; skipped"
                        if (indexed == 0 and skipped)
                        else "Media processed and indexed successfully"
                    ),
                },
            )
    except asyncio.CancelledError:
        # A per-item or batch cancel tore this task down. Record it and re-raise so
        # the orchestrator's gather sees the cancellation.
        finish_job(
            child,
            cancelled=True,
            result={"message": "Item processing cancelled", "filename": child.filename},
        )
        raise
    except Exception as exc:  # noqa: BLE001 - surface every failure via the child job
        logger.exception("Media item job %s failed: %s", child.job_id, exc)
        finish_job(child, error=str(exc))


async def _calibrate_ground_truth_after_batch(
    master: MediaJob, store: Any, context: Any, *, force: bool = False
) -> None:
    """Refit the avatar's direct-quote cloud once, after a whole batch indexes.

    The avatar's replies are scored against a cloud fitted from the target's own
    direct quotes (empirical Mahalanobis threshold + IsolationForest, read back by
    the message path). That fit is derived state: it only exists if something
    recalculates it after new quotes land, and until it does the message path
    silently reports no direct-quote comparison at all.

    Why the fit happens HERE, once per batch, rather than per media item or inside
    the graph: ``calibrate_ground_truth`` reads the avatar's ENTIRE quote corpus
    from the store on every call and refits from scratch, at a cost quadratic in
    corpus size (leave-one-out covariance estimation). Running it per item would
    make an eighteen-video upload do the same whole-corpus fit eighteen times over
    and throw away the first seventeen results. This point is downstream of every
    ingestion path and of all indexing, and is reached exactly once per upload —
    playlist children are expanded into the same batch before the gather, so they
    are included too.

    Never raises. Recalibration failing must not turn a successful upload into a
    failed one: the documents are already indexed, and the degraded state (no
    refreshed threshold or model) is the same state the message path already
    handles by omitting the direct-quote comparison until the next upload or an
    explicit recalibration.
    """
    if master.cancelled:
        # A cancelled batch has its indexed rows rolled back, so fitting over them
        # would bake documents into the cloud that are about to disappear.
        return
    if not master.assistant_id or store is None:
        return
    # ``force`` is set when a caller asked for a recalibration outright rather than
    # as a side effect of an upload — the point of that request is to fit a corpus
    # that is ALREADY in the store, so the "did this batch add quotes" gate would
    # reject exactly the case the caller wants.
    if not force and master.indexed_quote_document_count <= 0:
        logger.info(
            "Batch %s indexed no direct quotes; skipping ground-truth calibration",
            master.job_id,
        )
        return

    # Surfaced on the master's progress stream so a client watching a large upload
    # can tell the batch is still doing real work rather than having hung between
    # the last item finishing and the terminal event.
    add_event(
        master,
        {
            "type": "media_progress",
            "stage": "calibrating",
            "quote_documents_indexed": master.indexed_quote_document_count,
        },
    )

    # Imported lazily, matching every other call site: the module pulls in numpy,
    # scikit-learn and the phrase miner, none of which belong in API import cost.
    from src.subgraphs.process_media_graph.utils.calibrate_ground_truth import (
        calibrate_ground_truth_from_stored_corpus,
    )

    calibration_timeout_seconds = getattr(
        context, "ground_truth_calibration_timeout_seconds", None
    ) or 1800.0

    try:
        await asyncio.wait_for(
            calibrate_ground_truth_from_stored_corpus(
                store=store,
                assistant_id=master.assistant_id,
                user_id=master.user_id,
            ),
            timeout=calibration_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Ground-truth calibration for %s exceeded %.0fs; the upload is complete "
            "and the direct-quote comparison stays on its previous fit",
            master.assistant_id,
            calibration_timeout_seconds,
        )
        add_event(
            master,
            {
                "type": "media_progress",
                "stage": "calibrating_skipped",
                "reason": f"timed out after {calibration_timeout_seconds:.0f}s",
            },
        )
        return
    except asyncio.CancelledError:
        # A batch cancel tore this down; let the orchestrator's handler see it.
        raise
    except Exception as calibration_error:  # noqa: BLE001 - best-effort derived state
        logger.warning(
            "calibrate_ground_truth failed for %s (%s); the upload is complete and "
            "the direct-quote comparison stays on its previous fit",
            master.assistant_id,
            calibration_error,
        )
        add_event(
            master,
            {
                "type": "media_progress",
                "stage": "calibrating_skipped",
                "reason": str(calibration_error),
            },
        )
        return

    add_event(
        master,
        {
            "type": "media_progress",
            "stage": "calibrating_complete",
            "quote_documents_indexed": master.indexed_quote_document_count,
        },
    )


async def run_batch_media_job(
    master: MediaJob,
    items: List[Dict[str, Any]],
    config: Dict[str, Any],
    store: Any,
    context: Any,
    concurrency: int,
    existing_namespaces: Optional[List[str]] = None,
    registry: Optional[Dict[str, MediaJob]] = None,
    deferred_expanders: Optional[
        List[Callable[[], Awaitable[Optional[List[Dict[str, Any]]]]]]
    ] = None,
) -> None:
    """Orchestrate a batch: one child task per item, one shared concurrency pool.

    ``items`` is a list of ``{"child": MediaJob, "media_file": {...}}``. A single
    ``Semaphore(concurrency)`` is registered for this batch so every item — and
    every playlist/linktree child expanded inside the graph — draws from one global
    pool of slots. Child tasks run concurrently; the master finishes once all have
    settled (completed, errored, or cancelled). Never raises.

    ``deferred_expanders`` are async callables (e.g. a YouTube playlist enumeration)
    awaited **here**, off the request path, so the upload endpoint can return ``202``
    without blocking on ``yt_dlp``. Each one yields per-video ``media_file`` dicts
    that are turned into fresh child jobs registered under this master (``registry``
    required) — so every playlist video gets its own progress/cancel id, created
    asynchronously rather than at request time.
    """
    semaphore = asyncio.Semaphore(max(1, concurrency))
    set_batch_semaphore(master.job_id, semaphore)
    master.started_at = time.time()
    master.status = "running"
    try:
        items = list(items)
        # Expand deferred sources (playlists) in the background, minting one child
        # job per discovered video. Guarded by ``master.cancelled`` so a cancel
        # that arrives mid-enumeration stops us from spawning more work.
        if deferred_expanders and registry is not None:
            for expander in deferred_expanders:
                if master.cancelled:
                    break
                try:
                    expanded = await expander()
                except Exception as exc:  # noqa: BLE001 - surface via master stream
                    logger.exception("Deferred media expansion failed: %s", exc)
                    add_event(
                        master,
                        {
                            "type": "media_progress",
                            "stage": "item_error",
                            "error": f"playlist expansion failed: {exc}",
                        },
                    )
                    continue
                for media_file in expanded or []:
                    child = create_child_job(
                        registry,
                        user_id=master.user_id,
                        assistant_id=master.assistant_id,
                        parent_id=master.job_id,
                        filename=media_file.get("filename"),
                        namespace_filename=media_file.get("namespace_filename"),
                        estimated_tokens=media_file.get("estimated_tokens"),
                    )
                    master.child_ids.append(child.job_id)
                    items.append({"child": child, "media_file": media_file})
                    add_event(
                        master,
                        {
                            "type": "media_progress",
                            "stage": "playlist_child_added",
                            "item_job_id": child.job_id,
                            "item_filename": child.filename,
                            "estimated_tokens": child.estimated_tokens,
                        },
                    )

        child_tasks: List[asyncio.Task] = []
        for spec in items:
            child: MediaJob = spec["child"]
            media_file: Dict[str, Any] = spec["media_file"]
            # A cancel may have landed during expansion (children created after the
            # cancel request weren't in its target set); settle them without running.
            if master.cancelled:
                finish_job(
                    child,
                    cancelled=True,
                    result={
                        "message": "Batch cancelled before item start",
                        "filename": child.filename,
                    },
                )
                continue
            task = asyncio.create_task(
                run_single_item_job(
                    child,
                    master,
                    media_file,
                    config,
                    store,
                    context,
                    existing_namespaces=existing_namespaces,
                )
            )
            child.task = task
            child_tasks.append(task)

        # Wait for every child to settle. return_exceptions keeps one cancelled or
        # failed item from aborting the gather (each child already recorded its own
        # outcome via run_single_item_job).
        await asyncio.gather(*child_tasks, return_exceptions=True)

        # Refit the avatar's direct-quote cloud from everything this batch just
        # indexed. Awaited inside the master's own task, before the batch is
        # reported finished: an upload is not really done until the avatar can be
        # scored against its own quotes, and keeping the await here means a batch
        # cancel tears the fit down with the rest of the batch, the time it takes
        # lands in the master's duration, and subscribers see the stage. A
        # detached task would escape cancellation and could outlive the registry.
        await _calibrate_ground_truth_after_batch(master, store, context)

        children = [spec["child"] for spec in items]
        statuses = [c.status for c in children]
        if master.cancelled:
            finish_job(
                master,
                cancelled=True,
                result={
                    "message": "Batch cancelled",
                    "items": _summarize_children(children),
                },
            )
        else:
            finish_job(
                master,
                result={
                    "items_total": len(children),
                    "items_completed": statuses.count("completed"),
                    "items_error": statuses.count("error"),
                    "items_cancelled": statuses.count("cancelled"),
                    "items": _summarize_children(children),
                    "message": "Batch processing finished",
                },
            )
    except Exception as exc:  # noqa: BLE001 - surface every failure via the master job
        logger.exception("Batch media job %s failed: %s", master.job_id, exc)
        finish_job(master, error=str(exc))
    finally:
        clear_batch_semaphore(master.job_id)


def _summarize_children(children: List[MediaJob]) -> List[Dict[str, Any]]:
    """Per-item status summary embedded in the master's final result."""
    return [
        {
            "job_id": c.job_id,
            "filename": c.filename,
            "namespace_filename": c.namespace_filename,
            "status": c.status,
            "estimated_tokens": c.estimated_tokens,
            "error": c.error,
        }
        for c in children
    ]


def cancel_targets(
    registry: Dict[str, MediaJob], job: MediaJob
) -> List[MediaJob]:
    """The child jobs a cancel of ``job`` affects (itself, if it's a child)."""
    if job.is_master:
        return [registry[cid] for cid in job.child_ids if cid in registry]
    return [job]


def request_cancel(
    registry: Dict[str, MediaJob], job: MediaJob
) -> List[MediaJob]:
    """Flag ``job`` cancelled and cancel the running task(s).

    Returns the affected child jobs so the caller can roll back exactly what each
    wrote (by ``job_id`` / ``namespace_filename``). Cancelling a master flags every
    child and the master itself; the orchestrator then settles the master as
    ``cancelled``. Cancelling a child leaves the rest of the batch running.
    """
    targets = cancel_targets(registry, job)
    if job.is_master:
        job.cancelled = True
    for child in targets:
        child.cancelled = True
        task = child.task
        if task is not None and not task.done():
            task.cancel()
    return targets
