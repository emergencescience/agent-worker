"""Probe orchestrator — runs search prompts against multiple AI engines.

Engine types:
  SEARCH engines (real-time search):
    - tavily: Web search API → measures web presence
    - gemini: Google Search grounding → measures real AI search visibility
    - perplexity: AI search engine with citations → measures AI search visibility

  AWARENESS engines (base LLMs, no search):
    - deepseek: Base LLM → measures brand awareness in training data
    - doubao: Base LLM → measures Chinese-market brand awareness

Methodology:
  - Search engines: send user query → check if brand mentioned in results
  - Awareness engines: ask LLM what it knows about the query topic →
    check if brand is mentioned organically (NOT prompted by brand name)
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

# API keys
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_API_URL = "https://api.tavily.com/search"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "")
DOUBAO_API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
DOUBAO_MODEL = os.getenv("DOUBAO_MODEL", "doubao-pro-32k")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"


# ── Engine probe functions ─────────────────────────────────────────

async def _probe_tavily(query: str, brand_name: str, own_domain: str) -> dict[str, Any] | None:
    """Tavily web search probe — measures web presence."""
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


async def _probe_gemini(query: str, brand_name: str, own_domain: str) -> dict[str, Any] | None:
    """Gemini with Google Search grounding — measures real AI search visibility.

    Uses Google Search grounding to get real-time search results,
    then measures if the brand appears in the grounded response.

    The query is treated as a natural-language search question, not a keyword dump.
    """
    if not GEMINI_API_KEY:
        return None

    # Build a natural-language prompt that triggers Google Search grounding.
    # The query text (e.g. "月球种子工厂项目 推荐") is embedded in a full sentence
    # so Google interprets it as a search intent, not a URL or keyword list.
    search_prompt = (
        f"Search the web for: \"{query}\". "
        f"List what you find with URLs and brief descriptions. "
        f"Be specific — mention company names, project names, and website addresses."
    )

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": search_prompt}],
                        }
                    ],
                    "tools": [{"google_search": {}}],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 1024,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()

            # Extract answer text from Gemini response
            candidates = data.get("candidates", [])
            answer_text = ""
            grounding_sources = []

            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                for part in parts:
                    if "text" in part:
                        answer_text += part["text"]
                # Check for grounding metadata
                grounding_metadata = candidates[0].get("groundingMetadata", {})
                grounding_sources = grounding_metadata.get("groundingChunks", [])

            return {
                "answer": answer_text,
                "results": [
                    {
                        "title": chunk.get("web", {}).get("title", ""),
                        "url": chunk.get("web", {}).get("uri", ""),
                        "content": "",
                    }
                    for chunk in grounding_sources
                    if "web" in chunk
                ],
                "grounding_sources": len(grounding_sources),
            }
    except Exception as e:
        logger.error(f"Gemini probe failed for '{query}': {e}")
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
                            "content": "You are an AI search engine. Answer accurately with citations.",
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
                "results": [],  # Perplexity citations are inline
            }
    except Exception as e:
        logger.error(f"Perplexity probe failed for '{query}': {e}")
        return None


async def _probe_deepseek(query: str, brand_name: str, own_domain: str) -> dict[str, Any] | None:
    """DeepSeek awareness probe — measures brand recall in LLM training data.

    Unlike search engines, DeepSeek has no grounding. We measure:
    - Does the LLM organically mention this brand when asked about the TOPIC?
    - NOT: "what do you know about symbol.science?" (that triggers hallucination)
    - YES: "what are good {category} projects?" → check if brand mentioned

    IMPORTANT: If the query is just a bare brand name or URL (brand_direct template),
    we reformulate it as a topic question to avoid circular hallucination scoring.
    """
    if not DEEPSEEK_API_KEY:
        return None

    # Detect if query is just a bare brand/URL search (brand_direct style)
    # These cause circular hallucination: "tell me about brand X" → LLM invents X
    # → we count mentions of X → fake visibility score
    is_bare_brand_query = (
        " " not in query                          # single word/domain
        or query.startswith("http://")
        or query.startswith("https://")
        or query == brand_name.lower()
        or query == own_domain.lower()
    )

    if is_bare_brand_query:
        # Reformulate as a topic discovery question
        # Extract product name from URL path (e.g. "moon" from "symbol.science/moon")
        product_hint = ""
        if "/" in own_domain:
            product_hint = own_domain.split("/")[-1]

        awareness_prompt = (
            f"I'm researching {product_hint or brand_name}. "
            f"Tell me about notable companies, projects, and websites in this space. "
            f"Be specific with names, URLs, and what each one does."
        )
    else:
        # Query is already a topic/category search — use as-is
        awareness_prompt = query

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
                                "You are a helpful assistant. Answer the user's question "
                                "with specific examples, brand names, and product names where relevant. "
                                "Be factual and specific. If you don't know something, say so."
                            ),
                        },
                        {"role": "user", "content": awareness_prompt},
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
                "note": "base_llm_awareness",  # Mark as awareness probe, not search
            }
    except Exception as e:
        logger.error(f"DeepSeek probe failed for '{query}': {e}")
        return None


async def _probe_doubao(query: str, brand_name: str, own_domain: str) -> dict[str, Any] | None:
    """Doubao awareness probe — measures brand recall in Chinese AI ecosystem.

    Same principle as DeepSeek: reformulate bare brand queries as topic questions.
    """
    if not DOUBAO_API_KEY:
        return None

    # Same bare-brand detection as DeepSeek
    is_bare_brand_query = (
        " " not in query
        or query.startswith("http://")
        or query.startswith("https://")
        or query == brand_name.lower()
        or query == own_domain.lower()
    )

    if is_bare_brand_query:
        product_hint = ""
        if "/" in own_domain:
            product_hint = own_domain.split("/")[-1]

        awareness_prompt = (
            f"我正在研究{product_hint or brand_name}领域。"
            f"请告诉我这个领域有哪些知名的公司、项目和网站。"
            f"请具体说明名称、网址和各自的特点。"
        )
    else:
        awareness_prompt = query

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
                            "content": (
                                "你是一个有帮助的助手。回答用户问题时请提供具体的例子、品牌名称和产品名称。"
                                "实事求是，具体清晰。不知道就说不知道。"
                            ),
                        },
                        {"role": "user", "content": awareness_prompt},
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
                "note": "base_llm_awareness",
            }
    except Exception as e:
        logger.error(f"Doubao probe failed for '{query}': {e}")
        return None


# ── Engine registry ──────────────────────────────────────────────────

_PROBE_ENGINES: dict[str, dict[str, Any]] = {
    "tavily": {
        "label": "Tavily",
        "type": "search",
        "description": "Web search engine — measures web presence",
        "probe_fn": _probe_tavily,
    },
    "gemini": {
        "label": "Gemini",
        "type": "search",
        "description": "Google Search grounding — measures real AI search visibility",
        "probe_fn": _probe_gemini,
    },
    "perplexity": {
        "label": "Perplexity",
        "type": "search",
        "description": "AI search engine with citations",
        "probe_fn": _probe_perplexity,
    },
    # NOTE: DeepSeek removed — base LLM with no search API, hallucination
    # renders awareness scores unreliable for GEO measurements.
    # "doubao" also removed — same issue, base LLM without search grounding.
}


def get_available_engines() -> list[str]:
    """Return list of engine names that are configured (have API keys)."""
    keys = {
        "tavily": TAVILY_API_KEY,
        "perplexity": PERPLEXITY_API_KEY,
        "gemini": GEMINI_API_KEY,
    }
    available = []
    for name in _PROBE_ENGINES:
        if keys.get(name, ""):
            available.append(name)
    return available


def _is_engine_available(name: str) -> bool:
    keys = {
        "tavily": TAVILY_API_KEY,
        "perplexity": PERPLEXITY_API_KEY,
        "gemini": GEMINI_API_KEY,
    }
    return bool(keys.get(name, ""))


# ── Visibility analysis ──────────────────────────────────────────────

def _analyze_engine_visibility(
    engine: str,
    result: dict[str, Any] | None,
    brand_name: str,
    own_domain: str,
) -> dict[str, Any]:
    """Analyze one engine's probe result for brand visibility.

    For SEARCH engines: count mentions in search results.
    For AWARENESS engines: check if LLM organically mentions the brand
    when asked about the TOPIC (not the brand itself).
    """
    engine_config = _PROBE_ENGINES.get(engine, {})
    engine_type = engine_config.get("type", "search")

    if result is None:
        return {
            "engine": engine,
            "engine_type": engine_type,
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
    # Match both full path (e.g. "symbol.science/moon") and root domain ("symbol.science")
    domain_root = own_domain.split("/")[0] if own_domain else ""
    domain_roots = [own_domain, domain_root] if domain_root and domain_root != own_domain else [own_domain]

    if engine_type == "search" and engine == "tavily":
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
            if own_domain and any(d in url for d in domain_roots):
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
            "engine_type": engine_type,
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

    elif engine_type == "search":
        # Gemini/Perplexity — search engines with AI answers
        answer = result.get("answer", "")
        results = result.get("results", [])
        answer_lower = answer.lower()

        # For Gemini grounding: check both answer AND search sources
        brand_in_answer = brand_lower in answer_lower
        domain_in_answer = own_domain in answer_lower if own_domain else False

        # Check search result URLs for brand domain
        has_own_domain = False
        brand_in_sources = 0
        for r in results:
            url = (r.get("url", "") or "").lower()
            title = (r.get("title", "") or "").lower()
            if own_domain and any(d in url for d in domain_roots):
                has_own_domain = True
            if brand_lower in title:
                brand_in_sources += 1

        total_results = len(results) if results else 1
        brand_mentions = (1 if brand_in_answer else 0) + brand_in_sources

        # Search engine visibility: weighted by answer mention + source presence
        if brand_in_answer and has_own_domain:
            visibility_score = 0.9  # Brand owns the answer + domain in sources
        elif brand_in_answer:
            visibility_score = 0.7  # Brand mentioned in answer
        elif brand_in_sources > 0:
            visibility_score = 0.4 * min(brand_in_sources / max(len(results), 1), 1)
        elif domain_in_answer:
            visibility_score = 0.3  # Domain mentioned but not as brand
        else:
            visibility_score = 0.0

        return {
            "engine": engine,
            "engine_type": engine_type,
            "status": "ok",
            "total_results": total_results,
            "brand_mentions": max(brand_mentions, 0),
            "mention_positions": [1] if brand_mentions > 0 else [],
            "has_own_domain": has_own_domain,
            "ai_answer_text": answer,
            "visibility_score": visibility_score,
            "top_results": results[:5] if results else [{"content": answer[:500], "title": f"{engine} Answer"}],
            "raw_response": result,
        }

    else:
        # AWARENESS engines (DeepSeek, Doubao) — base LLMs without search
        # Measure: does the LLM organically mention the brand when asked about the TOPIC?
        answer = result.get("answer", "")
        answer_lower = answer.lower()
        domain_lower = own_domain.lower() if own_domain else ""

        # Count organic brand mentions
        brand_mentions = answer_lower.count(brand_lower)
        domain_mentioned = domain_lower and domain_lower in answer_lower

        # For awareness engines, the scoring is different:
        # - Only count as "visibility" if the LLM mentions the brand WITHOUT being prompted
        # - The query is a USER question about the category, not the brand
        if brand_mentions == 0:
            visibility_score = 0.0        # LLM has no awareness
        elif brand_mentions == 1:
            visibility_score = 0.2        # Barely aware — possibly just repeating the query
        elif brand_mentions <= 3:
            visibility_score = 0.4        # Some awareness
        elif brand_mentions <= 6:
            visibility_score = 0.6        # Good awareness
        else:
            visibility_score = 0.8        # Strong awareness

        return {
            "engine": engine,
            "engine_type": engine_type,
            "status": "ok",
            "total_results": 1,
            "brand_mentions": brand_mentions,
            "mention_positions": [1] if brand_mentions > 0 else [],
            "has_own_domain": domain_mentioned,
            "ai_answer_text": answer,
            "visibility_score": visibility_score,
            "top_results": [
                {"content": answer[:500], "title": f"{engine} Awareness Answer"}
            ],
            "raw_response": result,
        }


# ── Orchestrator ─────────────────────────────────────────────────────

async def run_project_probes(project_id: str):
    """Run multi-engine probes for all confirmed prompts in a project."""
    available = get_available_engines()
    if not available:
        logger.error("No probe engines available")
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

        prompts = [p for p in prompts if p.confirmed]
        if not prompts:
            logger.warning(f"No confirmed prompts for project {project_id}")
            return

        await update_project(db, project_id, status="probing")
        brand_name = project.brand_name.lower()

        # Fix URL handling: preserve the full path, not just domain
        # e.g. "https://symbol.science/moon" → "symbol.science/moon"
        website_url = project.website_url or ""
        own_domain = (
            website_url
            .replace("https://", "")
            .replace("http://", "")
            .rstrip("/")
            if website_url
            else ""
        )
        # Also keep a domain-only version for broad matching
        domain_root = own_domain.split("/")[0] if own_domain else ""

        total_probes = len(prompts) * len(available)
        completed = 0

        for prompt in prompts:
            query = prompt.query_text

            for engine_name in available:
                engine_config = _PROBE_ENGINES[engine_name]
                probe_fn = engine_config["probe_fn"]

                try:
                    # For matching, use the FULL path (e.g. "symbol.science/moon")
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
                    eng_type = engine_config.get("type", "?")
                    logger.info(
                        f"[{completed}/{total_probes}] {engine_name}({eng_type})/{prompt.template_type}: "
                        f"{analysis['brand_mentions']} mentions, score={analysis['visibility_score']:.2f}"
                    )

                except Exception as e:
                    logger.exception(f"Probe failed: {engine_name}/{prompt.template_type}: {e}")
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

        await generate_report(db, project_id)
        await update_project(db, project_id, status="completed")
        logger.info(f"Project {project_id} probing complete ({completed}/{total_probes} probes)")
