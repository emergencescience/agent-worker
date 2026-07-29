"""Job routes — async GEO audit and interview endpoints."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.async_jobs import JobStatus, job_store, job_to_dict
from services.geo_audit import run_geo_audit
from services.geo_interview import run_geo_interview
from services.tavily_monitor import run_tavily_geo_monitor

logger = logging.getLogger("jobs_api")

router = APIRouter(prefix="/jobs", tags=["jobs"])


# ── Request Models ──────────────────────────────────────────────


class GeoAuditRequest(BaseModel):
    url: str
    keywords: list[str] = []
    lang: str = "zh"
    probes: list[str] | None = None  # Optional subset of probe IDs
    user_id: str | None = None
    session_id: str | None = None


class GeoInterviewRequest(BaseModel):
    brand_name: str
    current_question: str  # Which field is being answered
    user_answer: str  # User's answer to that question
    previous_answers: dict | None = None  # Accumulated profile so far
    lang: str = "zh"
    user_id: str | None = None
    session_id: str | None = None


# ── POST /jobs/geo-audit ─────────────────────────────────────────


@router.post("/geo-audit")
async def create_geo_audit(body: GeoAuditRequest):
    """Launch multi-LLM GEO probe → returns job_id for polling."""
    if not body.url:
        raise HTTPException(status_code=400, detail="url is required")

    # Sanitize URL before processing
    from services.sanitizer import sanitize_url
    body.url = sanitize_url(body.url)
    if not body.url:
        raise HTTPException(status_code=400, detail="Invalid URL")

    job = await job_store.create("geo_audit")

    # Fire background task with supervision (won't crash the pod)
    async def _run():
        try:
            await job_store.start(job.id)
            await run_geo_audit(job, body.url, body.keywords, body.probes)
        except asyncio.CancelledError:
            logger.warning(f"Job {job.id} cancelled (pod shutdown?)")
            await job_store.fail(job.id, "Job cancelled — pod may be shutting down")
        except Exception as e:
            logger.exception(f"GEO audit job {job.id} failed: {e}")
            try:
                await job_store.fail(job.id, f"Internal error: {type(e).__name__}")
            except Exception:
                logger.critical(f"Failed to record job failure for {job.id}")

    asyncio.create_task(_run())

    return {"job_id": job.id, "status": JobStatus.QUEUED.value}


# ── POST /jobs/geo-interview ─────────────────────────────────────


@router.post("/geo-interview")
async def create_geo_interview(body: GeoInterviewRequest):
    """Process one interview turn → extract structured data, return next question."""
    if not body.brand_name:
        raise HTTPException(status_code=400, detail="brand_name is required")
    if not body.current_question:
        raise HTTPException(status_code=400, detail="current_question is required")
    if not body.user_answer:
        raise HTTPException(status_code=400, detail="user_answer is required")

    job = await job_store.create("geo_interview")

    # Fire background task with supervision
    async def _run():
        try:
            await job_store.start(job.id)
            await run_geo_interview(
                job=job,
                brand_name=body.brand_name,
                current_question=body.current_question,
                user_answer=body.user_answer,
                previous_answers=body.previous_answers,
                lang=body.lang,
            )
        except asyncio.CancelledError:
            logger.warning(f"Job {job.id} cancelled")
            await job_store.fail(job.id, "Job cancelled")
        except Exception as e:
            logger.exception(f"GEO interview job {job.id} failed: {e}")
            try:
                await job_store.fail(job.id, f"Internal error: {type(e).__name__}")
            except Exception:
                logger.critical(f"Failed to record job failure for {job.id}")

    asyncio.create_task(_run())

    return {"job_id": job.id, "status": JobStatus.QUEUED.value}


# ── POST /jobs/geo-monitor ──────────────────────────────────────


class GeoMonitorRequest(BaseModel):
    brand: str
    url: str | None = None
    keywords: list[str] | None = None
    competitors: list[str] | None = None
    lang: str = "zh"
    user_id: str | None = None
    session_id: str | None = None


@router.post("/geo-monitor")
async def create_geo_monitor(body: GeoMonitorRequest):
    """Launch Tavily GEO monitor → search-based brand visibility tracking."""
    if not body.brand:
        raise HTTPException(status_code=400, detail="brand is required")

    job = await job_store.create("geo_monitor")

    async def _run():
        try:
            await run_tavily_geo_monitor(
                job=job,
                brand=body.brand,
                url=body.url,
                keywords=body.keywords,
                competitors=body.competitors,
                lang=body.lang,
            )
        except Exception as e:
            logger.exception(f"GEO monitor job {job.id} failed: {e}")
            await job_store.fail(job.id, str(e))

    asyncio.create_task(_run())

    return {"job_id": job.id, "status": JobStatus.QUEUED.value}


# ── GET /jobs/{job_id}/status ────────────────────────────────────


@router.get("/{job_id}/status")
async def get_job_status(job_id: str):
    """Poll job status — returns phase, progress, log_entries for SSE streaming."""
    job = await job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    return job_to_dict(job)


# ── GET /jobs/{job_id}/result ────────────────────────────────────


@router.get("/{job_id}/result")
async def get_job_result(job_id: str):
    """Get full job result (only available when status=DONE)."""
    job = await job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    if job.status != JobStatus.DONE:
        raise HTTPException(status_code=409, detail=f"Job status is {job.status.value}, not done")
    return {"job_id": job.id, "result": job.result}
