"""GEO project service — CRUD for audit projects, prompts, and results."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import httpx
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
        "generate": lambda brand, industry="", **kw: (
            f"best {industry} 推荐 2026" if industry else f"best {brand} alternatives 2026"
        ),
    },
    "competitor_comparison": {
        "label_zh": "竞品对比搜索",
        "label_en": "Competitor Comparison",
        "generate": lambda brand, competitors=None, **kw: (
            f"{brand} vs {competitors[0]}" if competitors else f"{brand} vs competitors"
        ),
    },
    "brand_review": {
        "label_zh": "品牌评价搜索",
        "label_en": "Brand Review Search",
        "generate": lambda brand, **kw: f"{brand} review 评测 用户体验",
    },
}

# User simulation prompts — what real users would actually search
USER_SIMULATION_TEMPLATES = {
    "user_discovery": {
        "label_zh": "用户发现类查询",
        "label_en": "User Discovery Query",
        "generate": lambda brand, industry="", **kw: (
            f"有没有好玩的{industry}推荐" if industry else f"类似{brand}的产品推荐"
        ),
    },
    "user_comparison": {
        "label_zh": "用户对比类查询",
        "label_en": "User Comparison Query",
        "generate": lambda brand, **kw: f"{brand} 怎么样 好不好用",
    },
    "user_alternatives": {
        "label_zh": "用户替代品查询",
        "label_en": "User Alternative Query",
        "generate": lambda brand, industry="", **kw: (
            f"{brand} 的替代品有哪些" if industry else f"除了{brand}还有什么选择"
        ),
    },
}


async def generate_search_prompts(
    db: AsyncSession,
    project_id: str,
    lang: str = "zh",
    include_user_simulation: bool = True,
) -> list[SearchPrompt]:
    """Generate search prompts for a project based on its brand profile.

    Creates both probe prompts (for visibility measurement) and
    user simulation prompts (to model real user behavior).
    """
    project = await get_project(db, project_id)
    if not project:
        raise ValueError(f"Project {project_id} not found")

    prompts = []

    # 1. Standard probe templates
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

    # 2. User simulation templates — model what real users would search
    if include_user_simulation:
        for template_type, template in USER_SIMULATION_TEMPLATES.items():
            query = template["generate"](
                brand=project.brand_name,
                industry=project.industry or "",
                competitors=project.competitors or [],
                value_proposition=project.value_proposition or "",
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

    logger.info(
        f"Generated {len(prompts)} search prompts for project {project_id}"
        f" ({len(SEARCH_TEMPLATES)} probe + {len(USER_SIMULATION_TEMPLATES)} user simulation)"
    )
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


# ── Competitor Discovery ─────────────────────────────────────────────


COMPETITOR_EXTRACTION_PROMPT = """You are analyzing search results to discover competitors.
Given the brand name "{brand}" in the {industry} industry, extract any competitor brand names
or product names mentioned in these search results.

Search results:
{results_text}

Return ONLY a JSON array of competitor names. If no competitors found, return empty array [].
Example: ["Competitor A", "Competitor B"]"""


async def extract_competitors_from_results(
    db: AsyncSession,
    project_id: str,
) -> list[str]:
    """Use LLM to extract competitor names from probe results.

    Analyzes AI answer texts across all engines to find brands
    the user might not have thought of.
    """
    project = await get_project(db, project_id)
    if not project:
        return []

    results = await get_project_results(db, project_id)
    if not results:
        return []

    # Collect all AI answer texts
    answer_texts = []
    for r in results:
        if r.ai_answer_text and len(r.ai_answer_text) > 50:
            answer_texts.append(f"[{r.engine}] {r.ai_answer_text[:800]}")

    if not answer_texts:
        return []

    combined = "\n---\n".join(answer_texts[:10])  # Max 10 answers

    # Use DeepSeek to extract competitors
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        logger.warning("No DeepSeek API key for competitor extraction")
        return []

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "You extract competitor names from search results. Return ONLY a JSON array."},
                        {
                            "role": "user",
                            "content": COMPETITOR_EXTRACTION_PROMPT.format(
                                brand=project.brand_name,
                                industry=project.industry or "unknown",
                                results_text=combined,
                            ),
                        },
                    ],
                    "temperature": 0.1,
                    "max_tokens": 500,
                    "response_format": {"type": "json_object"},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            import json
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        competitors = v
                        break
                else:
                    competitors = []
            elif isinstance(parsed, list):
                competitors = parsed
            else:
                competitors = []

            # Filter out the brand itself
            brand_lower = project.brand_name.lower()
            competitors = [
                c for c in competitors
                if isinstance(c, str) and brand_lower not in c.lower()
            ]

            logger.info(f"Extracted {len(competitors)} competitors for {project.brand_name}: {competitors}")
            return competitors

    except Exception as e:
        logger.warning(f"Failed to extract competitors for {project_id}: {e}")
        return []


# ── Report CRUD ─────────────────────────────────────────────────────


async def generate_report(db: AsyncSession, project_id: str) -> ProjectReport:
    """Generate an aggregated visibility report from multi-engine probe results."""
    results = await get_project_results(db, project_id)

    # Group results by template type AND engine
    by_template: dict[str, list[ProbeResult]] = {}
    by_engine: dict[str, list[ProbeResult]] = {}
    for r in results:
        prompt = await db.get(SearchPrompt, r.search_prompt_id)
        template_type = prompt.template_type if prompt else "unknown"
        by_template.setdefault(template_type, []).append(r)
        by_engine.setdefault(r.engine, []).append(r)

    def _avg_visibility(entries: list[ProbeResult]) -> float:
        if not entries:
            return 0.0
        return sum(e.visibility_score for e in entries) / len(entries)

    branded = _avg_visibility(by_template.get("brand_direct", []))
    category = _avg_visibility(by_template.get("category_discovery", []))
    competitor = _avg_visibility(by_template.get("competitor_comparison", []))
    review = _avg_visibility(by_template.get("brand_review", []))
    user_disc = _avg_visibility(by_template.get("user_discovery", []))
    user_comp = _avg_visibility(by_template.get("user_comparison", []))
    user_alt = _avg_visibility(by_template.get("user_alternatives", []))

    # Compute engine-specific scores
    engine_scores = {}
    for eng, entries in by_engine.items():
        engine_scores[eng] = round(_avg_visibility(entries) * 100, 1)

    # Overall: average of all template scores
    all_scores = [branded, category, competitor, review, user_disc, user_comp, user_alt]
    valid_scores = [s for s in all_scores if s > 0 or any(
        r.visibility_score > 0 for r in by_template.get(
            {0: "brand_direct", 1: "category_discovery", 2: "competitor_comparison",
             3: "brand_review", 4: "user_discovery", 5: "user_comparison",
             6: "user_alternatives"}.get(all_scores.index(s), ""), []
        )
    )]
    if valid_scores:
        overall = sum(valid_scores) / len(valid_scores) * 100
    else:
        overall = sum(all_scores) / len(all_scores) * 100 if all_scores else 0.0

    # Generate recommendations
    recommendations = []
    if branded < 0.3:
        recommendations.append(
            "品牌直接搜索可见度低。建议加强品牌官网SEO基础建设（标题、描述、结构化数据）。"
        )
    if category < 0.1:
        recommendations.append(
            "品类搜索中品牌几乎不可见。需要创建行业相关内容和外部链接。"
        )
    if competitor < 0.1:
        recommendations.append(
            "竞品对比中品牌未被提及。建议创建与竞品的对比内容页面。"
        )
    if review < 0.1:
        recommendations.append(
            "品牌评价内容缺失。建议在知乎、小红书等平台引导用户生成评价。"
        )

    # Add engine-specific insights
    llm_engines = [e for e in by_engine if e != "tavily"]
    web_engines = [e for e in by_engine if e == "tavily"]

    if llm_engines:
        llm_avg = sum(engine_scores.get(e, 0) for e in llm_engines) / len(llm_engines)
        if llm_avg > 30:
            recommendations.insert(
                0,
                f"✅ AI 大模型对品牌有认知 (LLM 可见度 {llm_avg:.0f}%)。品牌在训练数据中有较好覆盖。"
            )

    if web_engines:
        web_avg = engine_scores.get("tavily", 0)
        if web_avg < 20 and any(engine_scores.get(e, 0) > 30 for e in llm_engines):
            recommendations.append(
                f"⚠️ 网页搜索可见度 ({web_avg:.0f}%) 远低于 LLM 认知度。说明品牌缺乏独立网页内容，"
                "建议建设官网、撰写技术博客、增加外部媒体报道。"
            )

    # Discover competitors from results
    discovered_competitors = []
    try:
        discovered_competitors = await extract_competitors_from_results(db, project_id)
    except Exception as e:
        logger.warning(f"Competitor discovery failed: {e}")

    # Update project with discovered competitors
    if discovered_competitors:
        proj = await get_project(db, project_id)
        if proj:
            existing = set(c.lower() for c in (proj.competitors or []))
            new = [c for c in discovered_competitors if c.lower() not in existing]
            if new:
                await update_project(
                    db, project_id,
                    competitors=list(proj.competitors or []) + new,
                )

    report = ProjectReport(
        project_id=project_id,
        overall_visibility=round(overall, 1),
        branded_visibility=round(branded * 100, 1),
        category_visibility=round(category * 100, 1),
        competitor_visibility=round(competitor * 100, 1),
        review_visibility=round(review * 100, 1),
        recommendations=recommendations,
        competitor_benchmark={
            "discovered_competitors": discovered_competitors,
            "engine_scores": engine_scores,
            "user_simulation_visibility": {
                "user_discovery": round(user_disc * 100, 1),
                "user_comparison": round(user_comp * 100, 1),
                "user_alternatives": round(user_alt * 100, 1),
            },
        },
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
