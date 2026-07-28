"""GEO project service — CRUD for audit projects, prompts, and results."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from services.database import (
    GeoProject,
    ProjectReport,
    ProbeResult,
    SearchPrompt,
)

logger = logging.getLogger("geo_projects")


# ── Project CRUD ────────────────────────────────────────────────────


async def create_project(
    db: AsyncSession,
    brand_name: str,
    user_id: str | None = None,
    session_id: str | None = None,
    industry: str | None = None,
    value_proposition: str | None = None,
    target_audience: str | None = None,
    website_url: str | None = None,
    competitors: list[str] | None = None,
    lang: str = "zh",
) -> GeoProject:
    """Create a new GEO audit project."""
    project = GeoProject(
        brand_name=brand_name,
        user_id=user_id,
        session_id=session_id,
        industry=industry,
        value_proposition=value_proposition,
        target_audience=target_audience,
        website_url=website_url,
        competitors=competitors or [],
        lang=lang,
        status="interview",
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)
    logger.info(f"Created GEO project {project.id} for brand '{brand_name}'")
    return project


async def get_project(db: AsyncSession, project_id: str) -> GeoProject | None:
    """Get a project by ID."""
    result = await db.execute(select(GeoProject).where(GeoProject.id == project_id))
    return result.scalar_one_or_none()


async def update_project(db: AsyncSession, project_id: str, **kwargs) -> GeoProject | None:
    """Update project fields."""
    project = await get_project(db, project_id)
    if not project:
        return None
    for key, value in kwargs.items():
        if hasattr(project, key):
            setattr(project, key, value)
    project.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(project)
    return project


async def list_projects(db: AsyncSession, user_id: str | None = None, limit: int = 20) -> list[GeoProject]:
    """List projects, optionally filtered by user."""
    query = select(GeoProject).order_by(GeoProject.created_at.desc()).limit(limit)
    if user_id:
        query = query.where(GeoProject.user_id == user_id)
    result = await db.execute(query)
    return list(result.scalars().all())


# ── Search Prompt CRUD ──────────────────────────────────────────────


SEARCH_TEMPLATES = {
    "brand_direct": {
        "label_zh": "品牌直接搜索",
        "label_en": "Brand Direct Search",
        "generate": lambda brand, **kw: f"{brand}",
    },
    "category_discovery": {
        "label_zh": "品类发现搜索",
        "label_en": "Category Discovery",
        "generate": lambda brand, industry="", **kw: f"best {industry} 推荐 2026" if industry else f"best {brand} alternatives 2026",
    },
    "competitor_comparison": {
        "label_zh": "竞品对比搜索",
        "label_en": "Competitor Comparison",
        "generate": lambda brand, competitors=None, **kw: f"{brand} vs {competitors[0]}" if competitors else f"{brand} vs competitors",
    },
    "brand_review": {
        "label_zh": "品牌评价搜索",
        "label_en": "Brand Review Search",
        "generate": lambda brand, **kw: f"{brand} review 评测 用户体验",
    },
}


async def generate_search_prompts(
    db: AsyncSession,
    project_id: str,
    lang: str = "zh",
) -> list[SearchPrompt]:
    """Generate search prompts for a project based on its brand profile."""
    project = await get_project(db, project_id)
    if not project:
        raise ValueError(f"Project {project_id} not found")

    prompts = []
    for template_type, template in SEARCH_TEMPLATES.items():
        query = template["generate"](
            brand=project.brand_name,
            industry=project.industry or "",
            competitors=project.competitors or [],
        )
        prompt = SearchPrompt(
            project_id=project_id,
            template_type=template_type,
            query_text=query,
            confirmed=False,
        )
        db.add(prompt)
        prompts.append(prompt)

    await db.commit()
    for p in prompts:
        await db.refresh(p)

    logger.info(f"Generated {len(prompts)} search prompts for project {project_id}")
    return prompts


async def get_project_prompts(db: AsyncSession, project_id: str) -> list[SearchPrompt]:
    """Get all search prompts for a project."""
    result = await db.execute(
        select(SearchPrompt).where(SearchPrompt.project_id == project_id)
    )
    return list(result.scalars().all())


async def confirm_prompts(db: AsyncSession, project_id: str) -> bool:
    """Mark all prompts as confirmed and update project status."""
    await db.execute(
        update(SearchPrompt)
        .where(SearchPrompt.project_id == project_id)
        .values(confirmed=True)
    )
    await update_project(db, project_id, status="prompts_ready")
    await db.commit()
    return True


async def update_prompt_query(
    db: AsyncSession, prompt_id: str, new_query: str
) -> SearchPrompt | None:
    """Update a single prompt's query text (user edit)."""
    result = await db.execute(select(SearchPrompt).where(SearchPrompt.id == prompt_id))
    prompt = result.scalar_one_or_none()
    if not prompt:
        return None
    prompt.query_text = new_query
    prompt.user_edited = True
    await db.commit()
    await db.refresh(prompt)
    return prompt


# ── Probe Result CRUD ───────────────────────────────────────────────


async def save_probe_result(
    db: AsyncSession,
    project_id: str,
    search_prompt_id: str,
    engine: str,
    total_results: int,
    brand_mentions: int,
    mention_positions: list[int],
    has_own_domain: bool,
    ai_answer_text: str | None,
    visibility_score: float,
    top_results: list[dict],
    raw_response: dict,
) -> ProbeResult:
    """Save a single probe result."""
    result_entry = ProbeResult(
        project_id=project_id,
        search_prompt_id=search_prompt_id,
        engine=engine,
        total_results=total_results,
        brand_mentions=brand_mentions,
        mention_positions=mention_positions,
        has_own_domain=has_own_domain,
        ai_answer_text=ai_answer_text,
        visibility_score=visibility_score,
        top_results=top_results,
        raw_response=raw_response,
    )
    db.add(result_entry)
    await db.commit()
    await db.refresh(result_entry)
    return result_entry


async def get_project_results(db: AsyncSession, project_id: str) -> list[ProbeResult]:
    """Get all probe results for a project."""
    result = await db.execute(
        select(ProbeResult).where(ProbeResult.project_id == project_id)
    )
    return list(result.scalars().all())


# ── Report CRUD ─────────────────────────────────────────────────────


async def generate_report(db: AsyncSession, project_id: str) -> ProjectReport:
    """Generate an aggregated visibility report from probe results."""
    results = await get_project_results(db, project_id)

    # Group results by template type
    by_template: dict[str, list[ProbeResult]] = {}
    for r in results:
        # Get template type from the associated prompt
        prompt = await db.get(SearchPrompt, r.search_prompt_id)
        template_type = prompt.template_type if prompt else "unknown"
        by_template.setdefault(template_type, []).append(r)

    def _avg_visibility(entries: list[ProbeResult]) -> float:
        if not entries:
            return 0.0
        return sum(e.visibility_score for e in entries) / len(entries)

    branded = _avg_visibility(by_template.get("brand_direct", []))
    category = _avg_visibility(by_template.get("category_discovery", []))
    competitor = _avg_visibility(by_template.get("competitor_comparison", []))
    review = _avg_visibility(by_template.get("brand_review", []))

    overall = (branded + category + competitor + review) / 4.0 * 100 if results else 0.0

    # Generate recommendations based on scores
    recommendations = []
    if branded < 0.3:
        recommendations.append("品牌直接搜索可见度低。建议加强品牌官网SEO基础建设（标题、描述、结构化数据）。")
    if category < 0.1:
        recommendations.append("品类搜索中品牌几乎不可见。需要创建行业相关内容和外部链接。")
    if competitor < 0.1:
        recommendations.append("竞品对比中品牌未被提及。建议创建与竞品的对比内容页面。")
    if review < 0.1:
        recommendations.append("品牌评价内容缺失。建议在知乎、小红书等平台引导用户生成评价。")

    report = ProjectReport(
        project_id=project_id,
        overall_visibility=round(overall, 1),
        branded_visibility=round(branded * 100, 1),
        category_visibility=round(category * 100, 1),
        competitor_visibility=round(competitor * 100, 1),
        review_visibility=round(review * 100, 1),
        recommendations=recommendations,
        competitor_benchmark={},  # To be populated with real competitor data
    )
    db.add(report)
    await db.commit()
    await db.refresh(report)

    return report


async def get_latest_report(db: AsyncSession, project_id: str) -> ProjectReport | None:
    """Get the latest report for a project."""
    result = await db.execute(
        select(ProjectReport)
        .where(ProjectReport.project_id == project_id)
        .order_by(ProjectReport.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()
