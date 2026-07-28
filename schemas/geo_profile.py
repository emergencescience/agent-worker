"""GeoProfile Pydantic Schema — contract between interview conversation and structured storage."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GeoProfile(BaseModel):
    """Structured GEO profile extracted from multi-round brand interview.

    This is the canonical schema for geo_profiles table rows and
    the JSON contract between agent-worker and orchestrator.
    """

    brand_name: str = Field(..., description="Brand or product name")
    website_url: str | None = Field(None, description="URL if available, else None")

    target_audience: list[str] = Field(
        default_factory=list,
        description="Target user personas e.g. ['Web3 开发者', 'AI startup founders']",
    )
    differentiators: list[str] = Field(
        default_factory=list,
        description="Key unique features that distinguish from competitors",
    )
    geographic_markets: list[str] = Field(
        default_factory=list,
        description="Primary markets e.g. ['CN', 'SG', 'TH']",
    )
    competitor_names: list[str] = Field(
        default_factory=list,
        description="Competitor domains or brand names",
    )
    content_channels: list[str] = Field(
        default_factory=list,
        description="Where brand currently publishes: Twitter, WeChat, Medium, etc.",
    )
    known_gaps: list[str] = Field(
        default_factory=list,
        description="Identified GEO weaknesses from interview",
    )

    # Internal interview metadata (not shown in UI)
    interview_turns: int = Field(0, description="Number of conversation turns taken")
    extraction_confidence: dict[str, float] = Field(
        default_factory=dict,
        description="Per-field confidence scores 0.0-1.0",
    )

    @property
    def is_complete(self) -> bool:
        """True when all required fields have confidence >= 0.65."""
        required_fields = [
            "target_audience",
            "differentiators",
            "geographic_markets",
            "competitor_names",
        ]
        return all(
            self.extraction_confidence.get(f, 0.0) >= 0.65
            for f in required_fields
        )
