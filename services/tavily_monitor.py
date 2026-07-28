"""Tavily GEO Monitor — search-based brand visibility tracking.

Uses Tavily Search API to monitor how a brand/URL appears in AI-optimized
search results. Simpler and more reliable than per-LLM probing.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

from services.async_jobs import Job, job_store

logger = logging.getLogger("tavily_monitor")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_API_URL = "https://api.tavily.com/search"

# GEO monitoring search templates
_GEO_SEARCH_TEMPLATES: list[dict[str, str]] = [
    {
        "id": "brand_direct",
        "query": "{brand}",
        "label_zh": "品牌直接搜索",
        "label_en": "Brand Direct Search",
    },
    {
        "id": "brand_review",
        "query": "{brand} review 评测",
        "label_zh": "品牌评价搜索",
        "label_en": "Brand Review Search",
    },
    {
        "id": "brand_vs_competitors",
        "query": "{brand} vs {competitors}",
        "label_zh": "竞品对比搜索",
        "label_en": "Competitor Comparison",
    },
    {
        "id": "category_discovery",
        "query": "best {keywords} 推荐 2026",
        "label_zh": "品类发现搜索",
        "label_en": "Category Discovery",
    },
]


async def _tavily_search(query: str, search_depth: str = "basic") -> dict[str, Any]:
    """Execute a single Tavily search query."""
    if not TAVILY_API_KEY:
        return {"error": "TAVILY_API_KEY not configured", "results": []}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                TAVILY_API_URL,
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": search_depth,
                    "include_answer": True,
                    "include_raw_content": False,
                    "max_results": 10,
                },
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Tavily search failed for '{query}': {e}")
        return {"error": str(e), "results": []}


def _analyze_visibility(
    brand: str,
    url: str | None,
    search_results: dict[str, Any],
    template_id: str,
) -> dict[str, Any]:
    """Analyze Tavily results for brand visibility metrics."""
    results = search_results.get("results", [])
    answer = search_results.get("answer", "")

    brand_lower = brand.lower()
    mentioned_in_answer = brand_lower in answer.lower() if answer else False

    # Count results mentioning the brand
    brand_mentions = 0
    brand_mention_positions: list[int] = []
    competitor_mentions: dict[str, int] = {}

    for i, result in enumerate(results):
        content = (result.get("content", "") + " " + result.get("title", "")).lower()
        url_str = result.get("url", "").lower()

        if brand_lower in content or (url and url.lower() in url_str):
            brand_mentions += 1
            brand_mention_positions.append(i + 1)

    # Check if brand's URL appears in top results
    has_own_domain = False
    if url:
        domain = url.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0].lower()
        for result in results:
            if domain in result.get("url", "").lower():
                has_own_domain = True
                break

    total = len(results)
    visibility_score = round((brand_mentions / max(total, 1)) * 100, 1)

    return {
        "template_id": template_id,
        "total_results": total,
        "brand_mentions": brand_mentions,
        "mention_positions": brand_mention_positions,
        "mentioned_in_ai_answer": mentioned_in_answer,
        "has_own_domain_in_results": has_own_domain,
        "visibility_score": visibility_score,
        "top_result": results[0] if results else None,
    }


async def run_tavily_geo_monitor(
    job: Job,
    brand: str,
    url: str | None = None,
    keywords: list[str] | None = None,
    competitors: list[str] | None = None,
    lang: str = "zh",
):
    """Run Tavily GEO monitoring across multiple search templates."""
    if not TAVILY_API_KEY:
        await job_store.fail(job.id, "TAVILY_API_KEY not configured")
        return

    await job_store.start(job.id)
    await job_store.log(job.id, f"Starting Tavily GEO monitor for '{brand}'", phase="init", progress=0.05)

    kw_str = " ".join(keywords) if keywords else brand
    comp_str = " ".join(competitors[:3]) if competitors else ""

    template_results: list[dict[str, Any]] = []
    total = len(_GEO_SEARCH_TEMPLATES)

    for i, template in enumerate(_GEO_SEARCH_TEMPLATES):
        query = template["query"].format(
            brand=brand,
            keywords=kw_str,
            competitors=comp_str,
        )
        label = template["label_zh"] if lang == "zh" else template["label_en"]

        await job_store.log(
            job.id,
            f"Searching: [{label}] {query[:80]}...",
            phase=f"searching ({i + 1}/{total})",
            progress=0.1 + (0.6 * i / total),
        )

        search_results = await _tavily_search(query)
        visibility = _analyze_visibility(brand, url, search_results, template["id"])
        visibility["template_label"] = label
        visibility["query"] = query
        template_results.append(visibility)

        score = visibility["visibility_score"]
        await job_store.log(
            job.id,
            f"✓ {label}: {score}% visibility ({visibility['brand_mentions']}/{visibility['total_results']} mentions)",
            phase=f"searching ({i + 1}/{total})",
            progress=0.1 + (0.6 * (i + 1) / total),
        )

    # Compute aggregate score
    scores = [r["visibility_score"] for r in template_results]
    avg_visibility = round(sum(scores) / len(scores), 1) if scores else 0
    ai_answer_rate = sum(1 for r in template_results if r["mentioned_in_ai_answer"]) / len(template_results)
    domain_rate = sum(1 for r in template_results if r["has_own_domain_in_results"]) / len(template_results)

    result = {
        "brand": brand,
        "url": url,
        "keywords": keywords,
        "overall_visibility": avg_visibility,
        "ai_answer_citation_rate": round(ai_answer_rate, 2),
        "own_domain_coverage": round(domain_rate, 2),
        "search_templates": template_results,
        "recommendations": _generate_recommendations(template_results, avg_visibility, lang),
    }

    await job_store.complete(job.id, result)


def _generate_recommendations(results: list[dict], overall: float, lang: str) -> list[str]:
    """Generate actionable recommendations based on results."""
    recs: list[str] = []

    if overall < 30:
        recs.append(
            "品牌在 AI 搜索中几乎不可见。优先建立官网基础 SEO（标题、描述、结构化数据）。"
            if lang == "zh" else
            "Brand is nearly invisible in AI search. Prioritize basic on-site SEO."
        )
    elif overall < 60:
        recs.append(
            "品牌有一定可见度但仍有提升空间。建议增加高质量内容产出（博客、案例研究）。"
            if lang == "zh" else
            "Moderate visibility. Increase high-quality content output."
        )

    # Check brand review presence
    review_tpl = next((r for r in results if r["template_id"] == "brand_review"), None)
    if review_tpl and review_tpl["visibility_score"] < 20:
        recs.append(
            "品牌评价内容缺失。建议主动在知乎、小红书等平台引导用户评价。"
            if lang == "zh" else
            "Brand review content missing. Encourage user reviews on relevant platforms."
        )

    # Check competitor comparison presence
    comp_tpl = next((r for r in results if r["template_id"] == "brand_vs_competitors"), None)
    if comp_tpl and comp_tpl["visibility_score"] < 20:
        recs.append(
            "竞品对比中品牌未被提及。建议创建对比内容页面并优化 SEO。"
            if lang == "zh" else
            "Brand not mentioned in competitor comparisons. Create comparison content pages."
        )

    # Check category discovery
    cat_tpl = next((r for r in results if r["template_id"] == "category_discovery"), None)
    if cat_tpl and cat_tpl["has_own_domain_in_results"] is False:
        recs.append(
            "品类搜索中品牌域名未出现。需加强外部链接建设和行业媒体曝光。"
            if lang == "zh" else
            "Brand domain missing from category search results. Build external backlinks."
        )

    if not recs:
        recs.append(
            "品牌 GEO 表现良好。继续保持内容更新频率，关注竞品动态。"
            if lang == "zh" else
            "Brand GEO performance is good. Maintain content cadence and monitor competitors."
        )

    return recs
