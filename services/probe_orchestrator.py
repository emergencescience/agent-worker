"""Probe orchestrator — runs search prompts against Tavily and saves results."""

from __future__ import annotations

import logging
import os

import httpx

from services.database import async_session
from services.geo_projects import (
    generate_report,
    get_project,
    get_project_prompts,
    save_probe_result,
    update_project,
)

logger = logging.getLogger("probe_orchestrator")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_API_URL = "https://api.tavily.com/search"


async def run_project_probes(project_id: str):
    """Run Tavily searches for all confirmed prompts in a project."""
    if not TAVILY_API_KEY:
        logger.error("TAVILY_API_KEY not set — cannot run probes")
        return

    async with async_session() as db:
        project = await get_project(db, project_id)
        if not project:
            logger.error(f"Project {project_id} not found")
            return

        prompts = await get_project_prompts(db, project_id)
        if not prompts:
            logger.warning(f"No prompts for project {project_id}")
            return

        # Only run confirmed prompts
        prompts = [p for p in prompts if p.confirmed]
        if not prompts:
            logger.warning(f"No confirmed prompts for project {project_id}")
            return

        await update_project(db, project_id, status="probing")
        brand_name = project.brand_name.lower()
        website_url = project.website_url or ""
        own_domain = website_url.replace("https://", "").replace("http://", "").split("/")[0] if website_url else ""

        for prompt in prompts:
            try:
                result = await _run_tavily_search(prompt.query_text)
                if not result:
                    continue

                # Count brand mentions
                total = len(result.get("results", []))
                answer = result.get("answer", "")
                brand_mentions = 0
                mention_positions = []
                has_own_domain = False

                for i, r in enumerate(result.get("results", [])):
                    title = (r.get("title", "") or "").lower()
                    content = (r.get("content", "") or "").lower()
                    url = (r.get("url", "") or "").lower()

                    if brand_name in title or brand_name in content:
                        brand_mentions += 1
                        mention_positions.append(i + 1)

                    if own_domain and own_domain in url:
                        has_own_domain = True

                # Calculate visibility score
                if total > 0:
                    visibility_score = brand_mentions / total
                else:
                    visibility_score = 0.0

                # Extract top results (simplified)
                top_results = []
                for r in (result.get("results", []) or [])[:5]:
                    top_results.append({
                        "url": r.get("url", ""),
                        "title": r.get("title", ""),
                        "content": (r.get("content", "") or "")[:300],
                        "score": r.get("score", 0),
                    })

                await save_probe_result(
                    db=db,
                    project_id=project_id,
                    search_prompt_id=prompt.id,
                    engine="tavily",
                    total_results=total,
                    brand_mentions=brand_mentions,
                    mention_positions=mention_positions,
                    has_own_domain=has_own_domain,
                    ai_answer_text=answer,
                    visibility_score=visibility_score,
                    top_results=top_results,
                    raw_response=result,
                )

                logger.info(
                    f"Probe {prompt.template_type}: {brand_mentions}/{total} mentions, "
                    f"score={visibility_score:.2f}"
                )

            except Exception as e:
                logger.exception(f"Failed to probe {prompt.template_type}: {e}")

        # Generate report
        await generate_report(db, project_id)
        await update_project(db, project_id, status="completed")

        logger.info(f"Project {project_id} probing complete")


async def _run_tavily_search(query: str) -> dict | None:
    """Run a single Tavily search and return the result."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                TAVILY_API_URL,
                json={
                    "query": query,
                    "search_depth": "advanced",
                    "include_answer": True,
                    "include_raw_content": False,
                    "max_results": 10,
                },
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {TAVILY_API_KEY}",
                },
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Tavily search failed for '{query}': {e}")
        return None
