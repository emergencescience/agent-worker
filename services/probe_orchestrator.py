"""Probe orchestrator — runs search prompts against multiple AI engines.

Engine Registry:
  - tavily: Web search API — measures web presence
  - deepseek: DeepSeek Chat — measures LLM knowledge / brand recall
  - perplexity: Perplexity Sonar — measures AI search engine visibility
  - doubao: ByteDance Doubao — measures Chinese AI ecosystem visibility

Each engine receives the same search prompt and returns:
  - Whether the brand was mentioned
  - How prominently (position, count)
  - Raw response for audit trail
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

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

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "")
DOUBAO_API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
DOUBAO_MODEL = os.getenv("DOUBAO_MODEL", "doubao-pro-32k")


# ── Engine probe functions ─────────────────────────────────────────


async def _probe_tavily(query: str, brand_name: str, own_domain: str) -> dict[str, Any] | None:
    """Tavily web search probe."""
    if not TAVILY_API_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                TAVILY_API_URL,
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "advanced",
                    "include_answer": True,
                    "include_raw_content": False,
                    "max_results": 10,
                },
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.error(f"Tavily probe failed for '{query}': {e}")
        return None


async def _probe_deepseek(query: str, brand_name: str, own_domain: str) -> dict[str, Any] | None:
    """DeepSeek Chat probe — ask the LLM what it knows about the brand."""
    if not DEEPSEEK_API_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                DEEPSEEK_API_URL,
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a search engine simulating a user query. "
                                "Answer the search query as if you were an AI search engine. "
                                "Cite specific facts, names, and sources when possible. "
                                "If you don't know, say so honestly."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Search query: {query}\n\nProvide search results as if you're an AI search engine.",
                        },
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1024,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "answer": data["choices"][0]["message"]["content"],
                "results": [],  # LLMs don't return structured results
            }
    except Exception as e:
        logger.error(f"DeepSeek probe failed for '{query}': {e}")
        return None


async def _probe_perplexity(query: str, brand_name: str, own_domain: str) -> dict[str, Any] | None:
    """Perplexity Sonar probe — AI search engine with real-time citations."""
    if not PERPLEXITY_API_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                PERPLEXITY_API_URL,
                headers={
                    "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "sonar",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an AI search engine. Answer search queries accurately with citations.",
                        },
                        {"role": "user", "content": query},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 1024,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "answer": data["choices"][0]["message"]["content"],
                "results": [],
            }
    except Exception as e:
        logger.error(f"Perplexity probe failed for '{query}': {e}")
        return None


async def _probe_doubao(query: str, brand_name: str, own_domain: str) -> dict[str, Any] | None:
    """Doubao (豆包) probe — ByteDance AI engine for Chinese market."""
    if not DOUBAO_API_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                DOUBAO_API_URL,
                headers={
                    "Authorization": f"Bearer {DOUBAO_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": DOUBAO_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是一个AI搜索引擎。请准确回答用户的搜索查询，引用具体事实。如果不知道，诚实地说不知道。",
                        },
                        {"role": "user", "content": query},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1024,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "answer": data["choices"][0]["message"]["content"],
                "results": [],
            }
    except Exception as e:
        logger.error(f"Doubao probe failed for '{query}': {e}")
        return None


# ── Engine registry ──────────────────────────────────────────────────

_PROBE_ENGINES: dict[str, dict[str, Any]] = {
    "tavily": {
        "label": "Tavily",
        "description": "Web search engine",
        "probe_fn": _probe_tavily,
    },
    "deepseek": {
        "label": "DeepSeek",
        "description": "LLM knowledge probe",
        "probe_fn": _probe_deepseek,
    },
    "perplexity": {
        "label": "Perplexity",
        "description": "AI search engine",
        "probe_fn": _probe_perplexity,
    },
    "doubao": {
        "label": "豆包",
        "description": "Chinese AI search engine",
        "probe_fn": _probe_doubao,
    },
}


def get_available_engines() -> list[str]:
    """Return list of engine names that are configured (have API keys)."""
    return [
        name
        for name, engine in _PROBE_ENGINES.items()
        if _is_engine_available(name)
    ]


def _is_engine_available(name: str) -> bool:
    """Check if an engine has its API key configured."""
    keys = {
        "tavily": TAVILY_API_KEY,
        "deepseek": DEEPSEEK_API_KEY,
        "perplexity": PERPLEXITY_API_KEY,
        "doubao": DOUBAO_API_KEY,
    }
    return bool(keys.get(name, ""))


# ── Visibility analysis ──────────────────────────────────────────────


def _analyze_engine_visibility(
    engine: str,
    result: dict[str, Any] | None,
    brand_name: str,
    own_domain: str,
) -> dict[str, Any]:
    """Analyze one engine's probe result for brand visibility."""
    if result is None:
        return {
            "engine": engine,
            "status": "skipped",
            "total_results": 0,
            "brand_mentions": 0,
            "mention_positions": [],
            "has_own_domain": False,
            "ai_answer_text": None,
            "visibility_score": 0.0,
            "top_results": [],
            "raw_response": {},
        }

    brand_lower = brand_name.lower()

    if engine == "tavily":
        # Tavily returns structured web results
        results = result.get("results", [])
        answer = result.get("answer", "")
        total = len(results)
        brand_mentions = 0
        mention_positions = []
        has_own_domain = False

        for i, r in enumerate(results):
            title = (r.get("title", "") or "").lower()
            content = (r.get("content", "") or "").lower()
            url = (r.get("url", "") or "").lower()
            if brand_lower in title or brand_lower in content:
                brand_mentions += 1
                mention_positions.append(i + 1)
            if own_domain and own_domain in url:
                has_own_domain = True

        top_results = []
        for r in (results or [])[:5]:
            top_results.append({
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "content": (r.get("content", "") or "")[:300],
                "score": r.get("score", 0),
            })

        visibility_score = (brand_mentions / max(total, 1)) if total > 0 else 0.0

        return {
            "engine": engine,
            "status": "ok",
            "total_results": total,
            "brand_mentions": brand_mentions,
            "mention_positions": mention_positions,
            "has_own_domain": has_own_domain,
            "ai_answer_text": answer,
            "visibility_score": visibility_score,
            "top_results": top_results,
            "raw_response": result,
        }

    else:
        # LLM engines: analyze the answer text for brand mentions and knowledge depth
        answer = result.get("answer", "")
        answer_lower = answer.lower()

        # Count brand mentions in LLM answer
        brand_mentions = answer_lower.count(brand_lower)
        domain_mentioned = own_domain in answer_lower if own_domain else False

        # Estimate visibility from LLM knowledge
        # - 0 mentions → 0.0 (LLM doesn't know brand)
        # - 1-2 mentions → 0.3 (LLM has heard of it)
        # - 3-5 mentions → 0.6 (LLM is familiar)
        # - 6+ mentions → 0.9 (LLM knows it well)
        if brand_mentions == 0:
            visibility_score = 0.0
        elif brand_mentions <= 2:
            visibility_score = 0.3
        elif brand_mentions <= 5:
            visibility_score = 0.6
        else:
            visibility_score = 0.9

        # Also check if the LLM answer suggests it KNOWS the brand vs just repeating the query
        knows_brand = brand_mentions >= 2 or answer_lower.startswith(brand_lower)

        return {
            "engine": engine,
            "status": "ok",
            "total_results": 1,  # LLM answer is the "result"
            "brand_mentions": brand_mentions,
            "mention_positions": [1] if brand_mentions > 0 else [],
            "has_own_domain": domain_mentioned,
            "ai_answer_text": answer,
            "visibility_score": visibility_score if knows_brand else 0.0,
            "top_results": [
                {"content": answer[:500], "title": f"{engine} AI Answer"}
            ],
            "raw_response": result,
        }


# ── Orchestrator ─────────────────────────────────────────────────────


async def run_project_probes(project_id: str):
    """Run multi-engine probes for all confirmed prompts in a project."""
    available = get_available_engines()
    if not available:
        logger.error("No probe engines available — check API keys (TAVILY, DEEPSEEK, PERPLEXITY, DOUBAO)")
        return

    logger.info(f"Running probes for project {project_id} with engines: {available}")

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
        own_domain = (
            website_url.replace("https://", "").replace("http://", "").split("/")[0]
            if website_url
            else ""
        )

        total_probes = len(prompts) * len(available)
        completed = 0

        for prompt in prompts:
            query = prompt.query_text

            for engine_name in available:
                engine_config = _PROBE_ENGINES[engine_name]
                probe_fn = engine_config["probe_fn"]

                try:
                    raw = await probe_fn(query, brand_name, own_domain)
                    analysis = _analyze_engine_visibility(
                        engine_name, raw, brand_name, own_domain
                    )

                    await save_probe_result(
                        db=db,
                        project_id=project_id,
                        search_prompt_id=prompt.id,
                        engine=engine_name,
                        total_results=analysis["total_results"],
                        brand_mentions=analysis["brand_mentions"],
                        mention_positions=analysis["mention_positions"],
                        has_own_domain=analysis["has_own_domain"],
                        ai_answer_text=analysis["ai_answer_text"],
                        visibility_score=analysis["visibility_score"],
                        top_results=analysis["top_results"],
                        raw_response=analysis["raw_response"],
                    )

                    completed += 1
                    logger.info(
                        f"[{completed}/{total_probes}] {engine_name}/{prompt.template_type}: "
                        f"{analysis['brand_mentions']} mentions, score={analysis['visibility_score']:.2f}"
                    )

                except Exception as e:
                    logger.exception(
                        f"Probe failed: {engine_name}/{prompt.template_type}: {e}"
                    )
                    # Save a failed probe result
                    await save_probe_result(
                        db=db,
                        project_id=project_id,
                        search_prompt_id=prompt.id,
                        engine=engine_name,
                        total_results=0,
                        brand_mentions=0,
                        mention_positions=[],
                        has_own_domain=False,
                        ai_answer_text=f"ERROR: {str(e)}",
                        visibility_score=0.0,
                        top_results=[],
                        raw_response={"error": str(e)},
                    )

        # Generate aggregated report
        await generate_report(db, project_id)
        await update_project(db, project_id, status="completed")

        logger.info(f"Project {project_id} probing complete ({completed}/{total_probes} probes)")
