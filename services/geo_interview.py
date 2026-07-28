"""GEO Interview service — progressive brand profile extraction for brands without a website."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import httpx

from schemas.geo_profile import GeoProfile
from services.async_jobs import Job, job_store

logger = logging.getLogger("geo_interview")

# Which LLM to use for interview extraction (DeepSeek by default — deterministic, cheap)
_INTERVIEW_LLM_API_BASE = "https://api.deepseek.com"
_INTERVIEW_LLM_MODEL = "deepseek-chat"

# Interview question sequence
_INTERVIEW_QUESTIONS = [
    {
        "field": "target_audience",
        "question_zh": "你的目标用户是谁？请描述他们的画像。",
        "question_en": "Who is your target audience? Describe their personas.",
    },
    {
        "field": "differentiators",
        "question_zh": "与竞品相比，你的核心差异化优势是什么？",
        "question_en": "What are your key differentiators compared to competitors?",
    },
    {
        "field": "geographic_markets",
        "question_zh": "你主要面向哪些地理市场？",
        "question_en": "Which geographic markets do you primarily target?",
    },
    {
        "field": "competitor_names",
        "question_zh": "你的主要竞争对手有哪些？（品牌名或域名）",
        "question_en": "Who are your main competitors? (brand names or domains)",
    },
    {
        "field": "content_channels",
        "question_zh": "你目前在哪些渠道发布内容？（如微信公众号、Twitter、官网博客等）",
        "question_en": "Which channels do you currently publish content on?",
    },
    {
        "field": "known_gaps",
        "question_zh": "你在 SEO/GEO 方面已经发现了哪些问题或短板？",
        "question_en": "What SEO/GEO gaps or weaknesses have you already identified?",
    },
]

_EXTRACTION_SYSTEM_PROMPT = """You are a GEO brand profile extractor. Given a user's answer to an interview question,
extract structured data for the specific field requested.

Return ONLY valid JSON with this exact structure:
{
  "values": ["extracted item 1", "extracted item 2"],
  "confidence": 0.0-1.0,
  "follow_up": "optional follow-up question if the answer is unclear, or null"
}

Rules:
- For target_audience: extract specific persona descriptions
- For differentiators: extract unique features, not generic claims
- For geographic_markets: extract country/region codes (CN, US, SG, etc.) or region names
- For competitor_names: extract brand names or domains
- For content_channels: extract platform names
- For known_gaps: extract specific weaknesses mentioned

Confidence:
- 0.9-1.0: clear, specific answer with multiple concrete items
- 0.7-0.89: somewhat clear but could use follow-up
- 0.5-0.69: vague or single-item answer
- 0.0-0.49: answer doesn't address the question

If the answer is too vague, set confidence low and add a specific follow_up question."""


async def _extract_field(field: str, question: str, user_answer: str, lang: str = "zh") -> dict[str, Any]:
    """Use LLM to extract structured field values from user's answer."""
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        # Fallback: basic extraction
        return _basic_extract(field, user_answer)

    user_prompt = f"""Field to extract: {field}
Interview question that was asked: {question}
User's answer: {user_answer}

Extract the structured values for {field} from this answer."""

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                f"{_INTERVIEW_LLM_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": _INTERVIEW_LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)

    except Exception as e:
        logger.warning(f"LLM extraction failed for {field}: {e}, falling back to basic extraction")
        return _basic_extract(field, user_answer)


def _basic_extract(field: str, user_answer: str) -> dict[str, Any]:
    """Fallback basic extraction without LLM."""
    # Split by common delimiters
    items = [s.strip() for s in user_answer.replace("、", ",").replace("，", ",").split(",") if s.strip()]

    if not items:
        items = [user_answer.strip()]

    confidence = 0.8 if len(items) >= 2 else 0.5

    return {
        "values": items,
        "confidence": confidence,
        "follow_up": None,
    }


async def run_geo_interview(job: Job, brand_name: str, current_question: str, user_answer: str, previous_answers: dict[str, Any] | None = None, lang: str = "zh"):
    """Process one interview turn, extract structured data, and determine next question.

    Returns the updated profile state so orchestrator can continue the conversation.
    """
    await job_store.start(job.id)

    previous = previous_answers or {}
    turn = previous.get("interview_turns", 0) + 1

    # Find the question being answered
    question_def = None
    next_question = None

    for i, qdef in enumerate(_INTERVIEW_QUESTIONS):
        if qdef["field"] == current_question:
            question_def = qdef
            if i + 1 < len(_INTERVIEW_QUESTIONS):
                next_question = _INTERVIEW_QUESTIONS[i + 1]
            break

    if not question_def:
        await job_store.fail(job.id, f"Unknown question field: {current_question}")
        return

    # Extract structured data from user's answer
    q_text = question_def["question_zh"] if lang == "zh" else question_def["question_en"]
    extraction = await _extract_field(current_question, q_text, user_answer, lang)

    # Update profile
    profile = {
        "brand_name": previous.get("brand_name", brand_name),
        "website_url": previous.get("website_url"),
        "target_audience": previous.get("target_audience", []),
        "differentiators": previous.get("differentiators", []),
        "geographic_markets": previous.get("geographic_markets", []),
        "competitor_names": previous.get("competitor_names", []),
        "content_channels": previous.get("content_channels", []),
        "known_gaps": previous.get("known_gaps", []),
        "interview_turns": turn,
        "extraction_confidence": previous.get("extraction_confidence", {}),
    }

    # Update the answered field
    profile[current_question] = extraction.get("values", [])
    profile["extraction_confidence"][current_question] = extraction.get("confidence", 0.5)

    # Check if complete
    geo = GeoProfile(**profile)
    is_complete = geo.is_complete

    result = {
        "profile": profile,
        "is_complete": is_complete,
        "current_field": current_question,
        "confidence": extraction.get("confidence", 0.5),
        "next_question": None,
        "follow_up": extraction.get("follow_up"),
    }

    if not is_complete and next_question:
        result["next_question"] = {
            "field": next_question["field"],
            "text": next_question["question_zh"] if lang == "zh" else next_question["question_en"],
        }
        await job_store.complete(job.id, result)
    elif not is_complete and extraction.get("follow_up"):
        # Ask follow-up on same field
        result["next_question"] = {
            "field": current_question,
            "text": extraction["follow_up"],
        }
        await job_store.complete(job.id, result)
    else:
        # All done — save to database if available
        try:
            await _save_profile(profile)
            result["saved"] = True
        except Exception as e:
            logger.warning(f"Failed to save profile to DB: {e}")
            result["saved"] = False

        await job_store.log(job.id, f"Profile extraction complete ({turn} turns)", phase="done", progress=1.0)
        await job_store.complete(job.id, result)


async def _save_profile(profile: dict[str, Any]):
    """Persist GeoProfile to PostgreSQL."""
    from services.database import GeoProfileRow, async_session

    async with async_session() as session:
        row = GeoProfileRow(
            brand_name=profile["brand_name"],
            website_url=profile.get("website_url"),
            target_audience=profile.get("target_audience", []),
            differentiators=profile.get("differentiators", []),
            geographic_markets=profile.get("geographic_markets", []),
            competitor_names=profile.get("competitor_names", []),
            content_channels=profile.get("content_channels", []),
            known_gaps=profile.get("known_gaps", []),
            interview_turns=profile.get("interview_turns", 0),
            extraction_confidence=profile.get("extraction_confidence", {}),
        )
        session.add(row)
        await session.commit()
        logger.info(f"Saved GeoProfile for {profile['brand_name']}")
