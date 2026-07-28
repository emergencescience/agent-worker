"""Project routes — GEO audit project lifecycle management."""

from __future__ import annotations

import logging

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.database import async_session as db_session
from services.geo_projects import (
    confirm_prompts,
    create_project,
    generate_report,
    generate_search_prompts,
    get_latest_report,
    get_project,
    get_project_prompts,
    get_project_results,
    list_projects,
    update_prompt_query,
    update_project,
)

logger = logging.getLogger("projects_api")

router = APIRouter(prefix="/projects", tags=["projects"])

# Guard: db_session is None if DATABASE_URL not set
if db_session is None:
    logger.error("DATABASE_URL not configured — project endpoints will fail")


# ── Request Models ──────────────────────────────────────────────────


class CreateProjectRequest(BaseModel):
    brand_name: str
    industry: str | None = None
    value_proposition: str | None = None
    target_audience: str | None = None
    website_url: str | None = None
    competitors: list[str] | None = None
    lang: str = "zh"
    user_id: str | None = None
    session_id: str | None = None


class UpdateProjectRequest(BaseModel):
    industry: str | None = None
    value_proposition: str | None = None
    target_audience: str | None = None
    website_url: str | None = None
    competitors: list[str] | None = None


class UpdatePromptRequest(BaseModel):
    query_text: str


# ── Helpers ─────────────────────────────────────────────────────────


def _get_db():
    if db_session is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    return db_session()


def _project_to_dict(project) -> dict:
    return {
        "id": project.id,
        "brand_name": project.brand_name,
        "industry": project.industry,
        "value_proposition": project.value_proposition,
        "target_audience": project.target_audience,
        "website_url": project.website_url,
        "competitors": project.competitors,
        "lang": project.lang,
        "status": project.status,
        "user_id": project.user_id,
        "session_id": project.session_id,
        "created_at": project.created_at.isoformat() if project.created_at else None,
        "updated_at": project.updated_at.isoformat() if project.updated_at else None,
    }


def _prompt_to_dict(p) -> dict:
    return {
        "id": p.id,
        "template_type": p.template_type,
        "query_text": p.query_text,
        "confirmed": p.confirmed,
        "user_edited": p.user_edited,
    }


def _result_to_dict(r) -> dict:
    return {
        "id": r.id,
        "search_prompt_id": r.search_prompt_id,
        "engine": r.engine,
        "total_results": r.total_results,
        "brand_mentions": r.brand_mentions,
        "visibility_score": r.visibility_score,
        "has_own_domain": r.has_own_domain,
        "top_results": r.top_results,
    }


# ── POST /projects ──────────────────────────────────────────────────


@router.post("")
async def create_project_endpoint(body: CreateProjectRequest):
    if not body.brand_name:
        raise HTTPException(status_code=400, detail="brand_name is required")

    async with _get_db() as db:
        project = await create_project(
            db=db,
            brand_name=body.brand_name,
            user_id=body.user_id,
            session_id=body.session_id,
            industry=body.industry,
            value_proposition=body.value_proposition,
            target_audience=body.target_audience,
            website_url=body.website_url,
            competitors=body.competitors,
            lang=body.lang,
        )
        return _project_to_dict(project)


# ── GET /projects ───────────────────────────────────────────────────


@router.get("")
async def list_projects_endpoint(user_id: str | None = None, limit: int = 20):
    async with _get_db() as db:
        projects = await list_projects(db, user_id=user_id, limit=limit)
        return [_project_to_dict(p) for p in projects]


# ── GET /projects/{id} ──────────────────────────────────────────────


@router.get("/{project_id}")
async def get_project_endpoint(project_id: str):
    async with _get_db() as db:
        project = await get_project(db, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        prompts = await get_project_prompts(db, project_id)
        results = await get_project_results(db, project_id)
        report = await get_latest_report(db, project_id)

        return {
            "project": _project_to_dict(project),
            "prompts": [_prompt_to_dict(p) for p in prompts],
            "results": [_result_to_dict(r) for r in results],
            "report": {
                "overall_visibility": report.overall_visibility,
                "branded_visibility": report.branded_visibility,
                "category_visibility": report.category_visibility,
                "competitor_visibility": report.competitor_visibility,
                "review_visibility": report.review_visibility,
                "recommendations": report.recommendations,
                "competitor_benchmark": report.competitor_benchmark or {},
            } if report else None,
        }


# ── PATCH /projects/{id} ───────────────────────────────────────────


@router.patch("/{project_id}")
async def update_project_endpoint(project_id: str, body: UpdateProjectRequest):
    async with _get_db() as db:
        updates = {k: v for k, v in body.model_dump().items() if v is not None}
        project = await update_project(db, project_id, **updates)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return _project_to_dict(project)


# ── POST /projects/{id}/prompts ─────────────────────────────────────


@router.post("/{project_id}/prompts")
async def generate_prompts_endpoint(project_id: str):
    async with _get_db() as db:
        project = await get_project(db, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        prompts = await generate_search_prompts(db, project_id, lang=project.lang)
        return {
            "project_id": project_id,
            "prompts": [_prompt_to_dict(p) for p in prompts],
            "message": f"Generated {len(prompts)} search prompts. Review and confirm to start probing.",
        }


# ── POST /projects/{id}/confirm ─────────────────────────────────────


@router.post("/{project_id}/confirm")
async def confirm_prompts_endpoint(project_id: str):
    async with _get_db() as db:
        project = await get_project(db, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        await confirm_prompts(db, project_id)

        # Trigger background probing
        from services.probe_orchestrator import run_project_probes
        asyncio.create_task(run_project_probes(project_id))

        return {
            "project_id": project_id,
            "status": "probing",
            "message": "Prompts confirmed. Multi-engine probes are running in background.",
        }


# ── POST /projects/{id}/probe ─────────────────────────────────────


@router.post("/{project_id}/probe")
async def run_probes_endpoint(project_id: str):
    """Manually trigger probing for a project."""
    async with _get_db() as db:
        project = await get_project(db, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

    from services.probe_orchestrator import run_project_probes
    asyncio.create_task(run_project_probes(project_id))

    return {
        "project_id": project_id,
        "status": "probing_started",
        "message": "Probes running in background. Poll /projects/{id} for status.",
    }


# ── PATCH /projects/{id}/prompts/{prompt_id} ─────────────────────────


@router.patch("/{project_id}/prompts/{prompt_id}")
async def edit_prompt_endpoint(project_id: str, prompt_id: str, body: UpdatePromptRequest):
    async with _get_db() as db:
        prompt = await update_prompt_query(db, prompt_id, body.query_text)
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        return _prompt_to_dict(prompt)


# ── GET /projects/{id}/report ───────────────────────────────────────


@router.get("/{project_id}/report")
async def get_report_endpoint(project_id: str):
    async with _get_db() as db:
        report = await get_latest_report(db, project_id)
        if not report:
            raise HTTPException(status_code=404, detail="No report yet — run probing first")

        results = await get_project_results(db, project_id)

        return {
            "project_id": project_id,
            "overall_visibility": report.overall_visibility,
            "branded_visibility": report.branded_visibility,
            "category_visibility": report.category_visibility,
            "competitor_visibility": report.competitor_visibility,
            "review_visibility": report.review_visibility,
            "recommendations": report.recommendations,
            "competitor_benchmark": report.competitor_benchmark,
            "probe_results": [_result_to_dict(r) for r in results],
            "generated_at": report.created_at.isoformat() if report.created_at else None,
        }
