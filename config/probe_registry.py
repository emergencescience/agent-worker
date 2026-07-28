"""LLM Probe Registry — single versioned config for all GEO probes.

Adding/removing LLMs requires no pipeline code changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ProbeMarket(str, Enum):
    GLOBAL = "global"
    CHINA = "china"


class ProbeTier(str, Enum):
    API = "api"  # Automated via HTTP/OpenAI-compatible API
    PLAYWRIGHT = "playwright"  # Headless browser — rate limited, slower


@dataclass
class LLMProbe:
    id: str  # machine-readable key e.g. "doubao"
    display_name: str  # "豆包 (Doubao · 字节)"
    market: ProbeMarket
    tier: ProbeTier
    api_base: str | None  # For Tier A probes
    model: str | None  # For Tier A probes
    playwright_url: str | None  # For Tier B probes
    enabled: bool = True


PROBE_REGISTRY: list[LLMProbe] = [
    # ── Global · Tier A (API-automated) ──
    LLMProbe(
        id="perplexity",
        display_name="Perplexity.ai",
        market=ProbeMarket.GLOBAL,
        tier=ProbeTier.API,
        api_base="https://api.perplexity.ai",
        model="sonar",
        playwright_url=None,
    ),
    LLMProbe(
        id="deepseek",
        display_name="DeepSeek Chat",
        market=ProbeMarket.CHINA,
        tier=ProbeTier.API,
        api_base="https://api.deepseek.com",
        model="deepseek-chat",
        playwright_url=None,
    ),
    LLMProbe(
        id="kimi",
        display_name="Kimi (Moonshot)",
        market=ProbeMarket.CHINA,
        tier=ProbeTier.API,
        api_base="https://api.moonshot.cn/v1",
        model="moonshot-v1-8k",
        playwright_url=None,
    ),
    # ── China · Tier B (Playwright headless) — V2 ──
    LLMProbe(
        id="doubao",
        display_name="豆包 (Doubao · 字节)",
        market=ProbeMarket.CHINA,
        tier=ProbeTier.PLAYWRIGHT,
        api_base=None,
        model=None,
        playwright_url="https://www.doubao.com",
        enabled=False,  # V2
    ),
    LLMProbe(
        id="yuanbao",
        display_name="腾讯元宝 (Yuanbao)",
        market=ProbeMarket.CHINA,
        tier=ProbeTier.PLAYWRIGHT,
        api_base=None,
        model=None,
        playwright_url="https://yuanbao.tencent.com",
        enabled=False,  # V2
    ),
    LLMProbe(
        id="ernie",
        display_name="文心一言 (Ernie · Baidu)",
        market=ProbeMarket.CHINA,
        tier=ProbeTier.PLAYWRIGHT,
        api_base=None,
        model=None,
        playwright_url="https://yiyan.baidu.com",
        enabled=False,  # V2
    ),
]


def get_enabled_probes(tier: ProbeTier | None = None) -> list[LLMProbe]:
    """Return enabled probes, optionally filtered by tier."""
    probes = [p for p in PROBE_REGISTRY if p.enabled]
    if tier:
        probes = [p for p in probes if p.tier == tier]
    return probes


def get_probe_by_id(probe_id: str) -> LLMProbe | None:
    """Look up a probe by ID."""
    for p in PROBE_REGISTRY:
        if p.id == probe_id:
            return p
    return None
