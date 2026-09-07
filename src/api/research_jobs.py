"""In-process registry for background deep-research jobs.

The same shape as the media-job registry (``src/api/media_jobs.py``),
flattened: one ``ResearchJob`` per ``/deep_research`` request, an append-only
progress-event buffer with an ``asyncio.Event`` so server-sent-event
subscribers replay from index 0 and then wait for new appends, TTL cleanup of
finished jobs, and a cooperative cancel flag the pipeline checks between
stages.

NOTE: the registry is per-process, exactly like the media jobs, so a restart
of the API loses the progress of a running research job.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List
from uuid import uuid4

_FINISHED_TTL_SECONDS = 60 * 60
_MAX_JOBS = 1000


@dataclass
class ResearchJob:
    """One research run: the status, the progress events, and the result."""

    job_id: str
    user_id: str
    assistant_id: str
    subject_name: str
    status: str = "queued"  # queued | running | completed | error | cancelled
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    duration_seconds: float | None = None
    result: Dict[str, Any] | None = None
    error: str | None = None
    cancelled: bool = False
    events: List[Dict[str, Any]] = field(default_factory=list)
    updated: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None

    def snapshot(self) -> Dict[str, Any]:
        """Return the job as the status and progress endpoints report the job."""
        return {
            "job_id": self.job_id,
            "assistant_id": self.assistant_id,
            "subject_name": self.subject_name,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "result": self.result,
            "error": self.error,
            "cancelled": self.cancelled,
            "latest_stage": (self.events[-1].get("stage") if self.events else None),
        }


def create_research_job(
    registry: Dict[str, ResearchJob],
    *,
    user_id: str,
    assistant_id: str,
    subject_name: str,
) -> ResearchJob:
    """Register one new research job, evicting finished and overflowing jobs first."""
    _cleanup(registry)
    job = ResearchJob(
        job_id=str(uuid4()),
        user_id=user_id,
        assistant_id=assistant_id,
        subject_name=subject_name,
    )
    registry[job.job_id] = job
    return job


def get_research_job(
    registry: Dict[str, ResearchJob], job_id: str
) -> ResearchJob | None:
    """Return the registered job, or ``None`` when the job is unknown or expired."""
    return registry.get(job_id)


def add_event(job: ResearchJob, payload: Dict[str, Any]) -> None:
    """Append one progress event and wake every subscriber."""
    job.events.append(payload)
    job.updated.set()


def finish_job(
    job: ResearchJob,
    *,
    result: Dict[str, Any] | None = None,
    error: str | None = None,
    cancelled: bool = False,
) -> None:
    """Close the job once: completed, errored, or cancelled."""
    if job.done.is_set():
        return
    job.finished_at = time.time()
    job.duration_seconds = round(
        job.finished_at - (job.started_at or job.created_at), 3
    )
    if cancelled:
        job.status = "cancelled"
        job.cancelled = True
        job.result = result
    elif error is not None:
        job.status = "error"
        job.error = error
    else:
        job.status = "completed"
        job.result = result
    job.updated.set()
    job.done.set()


def request_cancel(job: ResearchJob) -> None:
    """Ask the pipeline to stop at the next stage boundary."""
    job.cancelled = True
    if job.status == "queued" and job.task is not None and not job.task.done():
        job.task.cancel()


def _cleanup(registry: Dict[str, ResearchJob]) -> None:
    now = time.time()
    expired = [
        job_id
        for job_id, job in registry.items()
        if job.finished_at is not None and now - job.finished_at > _FINISHED_TTL_SECONDS
    ]
    for job_id in expired:
        registry.pop(job_id, None)
    if len(registry) > _MAX_JOBS:
        oldest = sorted(registry.values(), key=lambda job: job.created_at)
        for job in oldest[: len(registry) - _MAX_JOBS]:
            registry.pop(job.job_id, None)


__all__ = [
    "ResearchJob",
    "add_event",
    "create_research_job",
    "finish_job",
    "get_research_job",
    "request_cancel",
]
