"""GEO project service — CRUD for audit projects, prompts, and results."""

from __future__ import annotations

import json
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

# LLM-based query generation — turns project profile data into realistic search queries.
# Previously we used naive lambda concatenation (e.g. f"best {industry} 推荐 2026"),
# but `industry` contains raw interview free-text, producing garbage queries like
# "best https://symbol.science/moon 是 月球种子工厂... 推荐 2026".
# Now we use an LLM pass to extract a real query from the profile fields.

_QUERY_GEN_SYSTEM = """You are a search query generator for GEO (Generative Engine Optimization) visibility testing.

GIVEN a brand/product profile, generate NATURAL, LONG-FORM search queries that a REAL USER would type —
but NEVER include the brand name, website, or product name in the queries.

CRITICAL: These queries are used as "probe queries" — they test whether the brand's website appears
in AI search engines (Tavily, Gemini grounding) when users search for relevant CATEGORY terms.
Industry term: "GEO probe queries" / "AI visibility test queries" / "LLM search probes".

The test measures: "If a user searches for CATEGORY terms, does the brand appear?"
If the query contains the brand name, the test is invalid — we're measuring SEO, not GEO.

Return ONLY valid JSON:
{
  "category_query": "Long natural category query (8-15+ words, like '2026年最好的开源AI历史策略游戏推荐')",
  "comparison_query": "Long comparison query about competing options in the space",
  "review_query": "Long review/opinion query about the category or problem space",
  "user_discovery": "Long query a new user would type to discover this type of product",
  "user_comparison": "Long query comparing options with specific requirements",
  "user_alternatives": "Long query for alternatives with specific constraints"
}

FORMAT REQUIREMENTS:
- Output language must match the project language (zh or en)
- NEVER include the brand name, domain, or product name in ANY query
- NEVER include URLs in queries
- Queries MUST be 8-20 words — natural, conversational, like a real person typing into Google or an AI chatbot
- NOT short 3-word queries like '历史策略游戏' — make them SPECIFIC and DESCRIPTIVE
- Include qualifiers: use case, audience, feature requirements, budget constraints, etc.
- Chinese queries should use natural Chinese sentence patterns, not keyword stuffing
- Each query should be UNIQUE — don't just rephrase the same thing

EXAMPLE good queries:
  zh: "2025年有什么好玩的AI驱动的多人历史策略游戏推荐"
  zh: "类似文明6但带有真实历史模拟和LLM生成剧情的策略游戏"
  en: "best open source AI powered historical strategy games with multiplayer 2025"
  en: "strategy games where AI generates realistic historical narratives and scenarios"
"""


async def _generate_queries_with_llm(
    brand_name: str,
    website_url: str = "",
    industry: str = "",
    value_proposition: str = "",
    target_audience: str = "",
    competitors: list[str] | None = None,
    lang: str = "zh",
) -> dict[str, str]:
    """Use LLM to generate realistic search queries from project profile."""
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        return _fallback_queries(brand_name, website_url, industry, competitors, lang)

    # Use the full URL path as the "brand key" for more accurate searching
    brand_key = website_url.replace("https://", "").replace("http://", "").rstrip("/") if website_url else brand_name

    profile = f"""Brand: {brand_name}
Full URL: {website_url or 'N/A'}
Industry/Category: {industry or 'N/A'}
Value Proposition: {value_proposition or 'N/A'}
Target Audience: {target_audience or 'N/A'}
Competitors: {', '.join(competitors) if competitors else 'N/A'}
Language: {lang}"""

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
                        {"role": "system", "content": _QUERY_GEN_SYSTEM},
                        {"role": "user", "content": profile},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500,
                    "response_format": {"type": "json_object"},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return parsed
    except Exception as e:
        logger.warning(f"LLM query generation failed: {e}, using fallback")
        return _fallback_queries(brand_name, website_url, industry, competitors, lang)


def _fallback_queries(
    brand_name: str,
    website_url: str = "",
    industry: str = "",
    competitors: list[str] | None = None,
    lang: str = "zh",
) -> dict[str, str]:
    """Fallback query generation without LLM — uses category/industry keywords."""
    # Use industry as the category descriptor if available
    category_words = industry if industry else brand_name

    if lang == "zh":
        return {
            "category_query": f"2025年最好的{category_words}推荐和评测",
            "comparison_query": f"{category_words}有哪些选择 对比评测 优缺点",
            "review_query": f"{category_words}好不好用 真实用户评价和体验分享",
            "user_discovery": f"想找一个好用的{category_words} 有什么推荐吗",
            "user_comparison": f"{category_words}哪个最好 性价比对比 2025",
            "user_alternatives": f"{category_words}的替代方案 免费或便宜的选择",
        }
    else:
        return {
            "category_query": f"best {category_words} recommendations and reviews 2025",
            "comparison_query": f"{category_words} comparison which one is best pros and cons",
            "review_query": f"{category_words} honest review real user experience worth it",
            "user_discovery": f"looking for a good {category_words} what do you recommend",
            "user_comparison": f"best {category_words} compared side by side 2025",
            "user_alternatives": f"{category_words} alternatives free or cheaper options available",
        }


async def generate_search_prompts(
    db: AsyncSession,
    project_id: str,
    lang: str = "zh",
    include_user_simulation: bool = True,
) -> list[SearchPrompt]:
    """Generate search prompts for a project based on its brand profile.

    Uses LLM to turn interview data into realistic user search queries,
    NOT naive string concatenation. Falls back to URL-path-based queries
    if LLM is unavailable.
    """
    project = await get_project(db, project_id)
    if not project:
        raise ValueError(f"Project {project_id} not found")

    # Generate queries via LLM (with fallback)
    queries = await _generate_queries_with_llm(
        brand_name=project.brand_name,
        website_url=project.website_url or "",
        industry=project.industry or "",
        value_proposition=project.value_proposition or "",
        target_audience=project.target_audience or "",
        competitors=project.competitors or [],
        lang=lang,
    )

    # Map LLM output keys → standard template_type values
    _TYPE_MAP = {
        "category_query": "category_discovery",
        "comparison_query": "competitor_comparison",
        "review_query": "brand_review",
        "user_discovery": "user_discovery",
        "user_comparison": "user_comparison",
        "user_alternatives": "user_alternatives",
    }

    prompts = []
    probe_count = 0
    sim_count = 0

    for query_key, template_type in _TYPE_MAP.items():
        query_text = queries.get(query_key, "")
        if not query_text:
            continue

        is_simulation = query_key.startswith("user_")
        if is_simulation and not include_user_simulation:
            continue

        prompt = SearchPrompt(
            project_id=project_id,
            template_type=template_type,
            query_text=query_text,
            confirmed=False,
        )
        db.add(prompt)
        prompts.append(prompt)

        if is_simulation:
            sim_count += 1
        else:
            probe_count += 1

    await db.commit()
    for p in prompts:
        await db.refresh(p)

    logger.info(
        f"Generated {len(prompts)} search prompts for project {project_id}"
        f" ({probe_count} probe + {sim_count} user simulation)"
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

    # No brand_direct any more — GEO measures category, not branded search
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
    all_scores = [category, competitor, review]
    valid_scores = [s for s in all_scores if s > 0]
    if valid_scores:
        overall = sum(valid_scores) / len(valid_scores) * 100
    else:
        overall = 0.0

    # Generate recommendations
    recommendations = []

    # Search engine overall
    search_avg = (
        sum(engine_scores.get(e, 0) for e in by_engine) / len(by_engine)
        if by_engine else 0
    )

    if search_avg < 20:
        recommendations.append(
            "网页搜索可见度低。建议加强品牌官网 SEO 基础建设（标题、描述、结构化数据）。"
        )

    if category < 0.1:
        recommendations.append(
            "品牌直接搜索几乎不可见。需要创建品牌专属内容页面，至少让搜索引擎能找到品牌名。"
        )
    if category < 0.1:
        recommendations.append(
            "品类搜索中品牌不可见。需要创建行业相关内容（如「最佳{行业}工具推荐」）来吸引品类流量。"
        )
    if competitor < 0.1:
        recommendations.append(
            "竞品对比中品牌未被提及。建议创建与主流竞品的对比页面。"
        )
    if review < 0.1:
        recommendations.append(
            "品牌评价内容缺失。建议在知乎、小红书等平台鼓励用户生成评价内容。"
        )

    # Cap recommendations at 5
    recommendations = recommendations[:5]

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
        branded_visibility=0.0,  # Brand direct queries removed — GEO measures category, not branded search
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
