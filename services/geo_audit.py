"""GEO Audit service — runs multi-LLM probes against a target URL."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

from config.probe_registry import LLMProbe, ProbeTier, get_enabled_probes
from services.async_jobs import Job, job_store

logger = logging.getLogger("geo_audit")

# API key env var mapping (probe id → env var name)
_API_KEY_ENV_MAP: dict[str, str] = {
    "deepseek": "DEEPSEEK_API_KEY",
    "kimi": "MOONSHOT_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
}

# GEO audit prompt template
_AUDIT_PROMPT = """You are a Generative Engine Optimization (GEO) auditor.

Analyze the following URL for its visibility and citability by AI search engines:
URL: {url}
Target keywords: {keywords}

Evaluate across these 8 pillars:
1. Heading Structure — Are section headers clear and hierarchical?
2. Schema/JSON-LD — Are structured data schemas present and valid?
3. Evidence Chain — Are claims backed by citations and data?
4. FAQ Coverage — Does the page answer common user questions?
5. Multi-modality — Are images, videos, or interactive elements used?
6. Cross-source Backlinks — Are there external references pointing to this page?
7. Semantic Freshness — Is the content up-to-date?
8. Machine Readability — Is the page parseable by LLM crawlers?

Return a JSON object with:
{{
  "overall_score": <0-100>,
  "citation_likelihood": <0.0-1.0>,
  "mentioned": <true/false>,
  "snippet": "<how this URL would appear in an AI answer>",
  "pillar_scores": {{
    "heading_structure": <0-100>,
    "schema_jsonld": <0-100>,
    "evidence_chain": <0-100>,
    "faq_coverage": <0-100>,
    "multi_modality": <0-100>,
    "cross_source_backlinks": <0-100>,
    "semantic_freshness": <0-100>,
    "machine_readability": <0-100>
  }},
  "recommendations": ["<actionable recommendation 1>", "..."]
}}

Only return valid JSON, no markdown wrapping."""


async def _probe_api(probe: LLMProbe, url: str, keywords: list[str]) -> dict[str, Any]:
    """Run a Tier A API probe against a single LLM."""
    api_key_env = _API_KEY_ENV_MAP.get(probe.id, "")
    api_key = os.getenv(api_key_env, "")

    if not api_key:
        return {
            "probe_id": probe.id,
            "display_name": probe.display_name,
            "status": "skipped",
            "reason": f"API key not configured ({api_key_env})",
        }

    prompt = _AUDIT_PROMPT.format(url=url, keywords=", ".join(keywords))

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{probe.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": probe.model,
                    "messages": [
                        {"role": "system", "content": "You are a GEO auditor. Return only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]

            import json

            result = json.loads(content)
            result["probe_id"] = probe.id
            result["display_name"] = probe.display_name
            result["status"] = "completed"
            return result

    except httpx.HTTPStatusError as e:
        logger.error(f"Probe {probe.id} HTTP {e.response.status_code}: {e.response.text[:200]}")
        return {
            "probe_id": probe.id,
            "display_name": probe.display_name,
            "status": "error",
            "error": f"HTTP {e.response.status_code}",
        }
    except Exception as e:
        logger.error(f"Probe {probe.id} failed: {e}")
        return {
            "probe_id": probe.id,
            "display_name": probe.display_name,
            "status": "error",
            "error": str(e)[:200],
        }


async def run_geo_audit(job: Job, url: str, keywords: list[str], probes: list[str] | None = None):
    """Run GEO audit across all enabled Tier A probes."""
    await job_store.start(job.id)
    await job_store.log(job.id, f"Starting GEO audit for {url}", phase="init", progress=0.05)

    # Determine which probes to run
    enabled_probes = get_enabled_probes(tier=ProbeTier.API)
    if probes:
        enabled_probes = [p for p in enabled_probes if p.id in probes]

    if not enabled_probes:
        await job_store.fail(job.id, "No enabled Tier A probes configured")
        return

    await job_store.log(
        job.id,
        f"Running {len(enabled_probes)} probes: {', '.join(p.id for p in enabled_probes)}",
        phase="probing",
        progress=0.1,
    )

    # Run probes concurrently
    probe_results: list[dict[str, Any]] = []
    total = len(enabled_probes)

    tasks = [_probe_api(probe, url, keywords) for probe in enabled_probes]
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        probe_results.append(result)

        probe_id = result.get("probe_id", "unknown")
        status = result.get("status", "unknown")
        progress = 0.1 + (0.7 * (i + 1) / total)

        if status == "completed":
            score = result.get("overall_score", "?")
            await job_store.log(
                job.id,
                f"✓ {result.get('display_name', probe_id)} — score {score}/100",
                phase=f"probing ({i + 1}/{total})",
                progress=progress,
            )
        elif status == "skipped":
            await job_store.log(
                job.id,
                f"⊘ {result.get('display_name', probe_id)} — skipped ({result.get('reason', '')})",
                phase=f"probing ({i + 1}/{total})",
                progress=progress,
            )
        else:
            await job_store.log(
                job.id,
                f"✗ {result.get('display_name', probe_id)} — error: {result.get('error', 'unknown')[:80]}",
                phase=f"probing ({i + 1}/{total})",
                progress=progress,
            )

    # Compute aggregate score
    completed = [r for r in probe_results if r.get("status") == "completed"]
    if completed:
        avg_score = sum(r.get("overall_score", 0) for r in completed) / len(completed)
        citation_rate = sum(1 for r in completed if r.get("mentioned")) / len(completed)

        # Average pillar scores
        pillar_keys = [
            "heading_structure", "schema_jsonld", "evidence_chain",
            "faq_coverage", "multi_modality", "cross_source_backlinks",
            "semantic_freshness", "machine_readability",
        ]
        avg_pillars: dict[str, float] = {}
        for key in pillar_keys:
            scores = [r.get("pillar_scores", {}).get(key, 0) for r in completed]
            avg_pillars[key] = sum(scores) / len(scores) if scores else 0

        result = {
            "url": url,
            "keywords": keywords,
            "overall_score": round(avg_score, 1),
            "citation_rate": round(citation_rate, 2),
            "probes_run": len(enabled_probes),
            "probes_completed": len(completed),
            "pillar_scores": {k: round(v, 1) for k, v in avg_pillars.items()},
            "probe_results": probe_results,
        }
    else:
        result = {
            "url": url,
            "keywords": keywords,
            "overall_score": 0,
            "citation_rate": 0,
            "probes_run": len(enabled_probes),
            "probes_completed": 0,
            "error": "No probes completed successfully",
            "probe_results": probe_results,
        }

    await job_store.complete(job.id, result)
