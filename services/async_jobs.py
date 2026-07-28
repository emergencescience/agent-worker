"""
Async job store — in-memory (V1, matches agent-render pattern).

V2 will migrate to PostgreSQL-backed jobs table.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("async_jobs")

# How long to keep completed/failed jobs (7 days)
JOB_TTL_SECONDS = 7 * 24 * 3600


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    DONE = "done"
    ERROR = "error"


@dataclass
class Job:
    id: str
    job_type: str  # "geo_audit", "geo_interview", "geo_publish"
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None

    # Progress tracking (for SSE streaming via orchestrator)
    phase: str = ""
    progress: float = 0.0
    log_entries: list[str] = field(default_factory=list)

    # Result or error
    result: dict[str, Any] | None = None
    error: str | None = None


class JobStore:
    """Thread-safe in-memory job store with TTL cleanup."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def create(self, job_type: str) -> Job:
        job = Job(id=str(uuid.uuid4()), job_type=job_type)
        async with self._lock:
            self._jobs[job.id] = job
        logger.info(f"Job created: {job.id} (type={job_type})")
        return job

    async def get(self, job_id: str) -> Job | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def start(self, job_id: str):
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.RUNNING
                job.started_at = time.time()

    async def log(self, job_id: str, entry: str, phase: str = "", progress: float | None = None):
        """Append a log entry and optionally update phase/progress."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.log_entries.append(entry)
                if phase:
                    job.phase = phase
                if progress is not None:
                    job.progress = progress

    async def complete(self, job_id: str, result: dict[str, Any]):
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.DONE
                job.completed_at = time.time()
                job.progress = 1.0
                job.result = result
                logger.info(f"Job completed: {job_id}")

    async def await_confirmation(self, job_id: str):
        """Set job to AWAITING_CONFIRMATION (e.g., waiting for user to confirm profile)."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.AWAITING_CONFIRMATION

    async def fail(self, job_id: str, error: str):
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.ERROR
                job.completed_at = time.time()
                job.error = error
                logger.warning(f"Job failed: {job_id} — {error[:100]}")

    async def cleanup(self):
        """Remove expired jobs."""
        now = time.time()
        async with self._lock:
            expired = [
                jid
                for jid, job in self._jobs.items()
                if job.completed_at and (now - job.completed_at > JOB_TTL_SECONDS)
            ]
            for jid in expired:
                self._jobs.pop(jid, None)
            if expired:
                logger.info(f"Cleaned up {len(expired)} expired jobs")

    async def cleanup_loop(self, interval: int = 3600):
        """Background cleanup loop."""
        while True:
            await asyncio.sleep(interval)
            try:
                await self.cleanup()
            except Exception as e:
                logger.error(f"Job cleanup error: {e}")


# Global singleton
job_store = JobStore()


def job_to_dict(job: Job) -> dict[str, Any]:
    """Convert Job to API response dict for orchestrator polling."""
    d: dict[str, Any] = {
        "job_id": job.id,
        "job_type": job.job_type,
        "status": job.status.value,
        "phase": job.phase,
        "progress": job.progress,
        "log_entries": job.log_entries,
        "created_at": job.created_at,
    }
    if job.started_at:
        d["started_at"] = job.started_at
    if job.completed_at:
        d["completed_at"] = job.completed_at
    if job.status == JobStatus.DONE and job.result:
        d["partial_result"] = job.result
    if job.status == JobStatus.ERROR:
        d["error"] = job.error
    return d
