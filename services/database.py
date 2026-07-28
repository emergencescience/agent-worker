"""Database connection, models, and initialization for GEO audit system.

Models:
  - GeoProfileRow: Legacy interview-extracted brand profiles
  - GeoProject: Core audit project entity
  - SearchPrompt: Auto-generated visibility search prompts
  - ProbeResult: Per-prompt, per-engine probe result
  - ProjectReport: Aggregated report with scores and recommendations
"""

from __future__ import annotations

import logging
import os
import uuid as _uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

logger = logging.getLogger("database")

DATABASE_URL = os.getenv("DATABASE_URL", "").replace(
    "postgresql://", "postgresql+asyncpg://", 1
)

if not DATABASE_URL:
    logger.warning("DATABASE_URL not set — database features disabled")

engine = create_async_engine(
    DATABASE_URL, echo=False, pool_size=5, max_overflow=10
) if DATABASE_URL else None
async_session = (
    sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    if engine
    else None
)


class Base(DeclarativeBase):
    pass


# ── Legacy model (kept for backward compatibility) ───────────────────


class GeoProfileRow(Base):
    """geo_profiles table — stores extracted brand profiles from interview flow."""

    __tablename__ = "geo_profiles"

    id = Column(UUID(as_uuid=False), primary_key=True, server_default=func.gen_random_uuid())
    user_id = Column(UUID(as_uuid=False), nullable=True, index=True)
    session_id = Column(UUID(as_uuid=False), nullable=True, index=True)
    brand_name = Column(Text, nullable=False)
    website_url = Column(Text, nullable=True)

    target_audience = Column(JSONB, default=list, nullable=False)
    differentiators = Column(JSONB, default=list, nullable=False)
    geographic_markets = Column(JSONB, default=list, nullable=False)
    competitor_names = Column(JSONB, default=list, nullable=False)
    content_channels = Column(JSONB, default=list, nullable=False)
    known_gaps = Column(JSONB, default=list, nullable=False)

    interview_turns = Column(Integer, default=0, nullable=False)
    extraction_confidence = Column(JSONB, default=dict, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ── GEO Project models ───────────────────────────────────────────────


class GeoProject(Base):
    """Core GEO audit project — one per brand audit."""

    __tablename__ = "geo_projects"

    id = Column(UUID(as_uuid=False), primary_key=True, default=lambda: str(_uuid.uuid4()))
    user_id = Column(Text, nullable=True, index=True)  # External user ID or session anchor
    session_id = Column(Text, nullable=True, index=True)  # Chat session ID
    brand_name = Column(Text, nullable=False, index=True)
    industry = Column(Text, nullable=True)  # e.g. "SaaS", "游戏", "教育"
    value_proposition = Column(Text, nullable=True)  # Core value / positioning
    target_audience = Column(Text, nullable=True)  # Who the brand serves
    website_url = Column(Text, nullable=True)
    competitors = Column(JSONB, default=list, nullable=False)  # ["comp A", "comp B"]
    lang = Column(Text, default="zh", nullable=False)
    status = Column(
        Text,
        default="interview",
        nullable=False,
        # interview → prompts_ready → probing → completed
        # interview: brand profile being collected
        # prompts_ready: search prompts generated, awaiting confirmation
        # probing: multi-engine probes running
        # completed: report available
    )

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    prompts = relationship("SearchPrompt", back_populates="project", cascade="all, delete-orphan")
    results = relationship("ProbeResult", back_populates="project", cascade="all, delete-orphan")
    reports = relationship("ProjectReport", back_populates="project", cascade="all, delete-orphan")


class SearchPrompt(Base):
    """Auto-generated search prompt for brand visibility probing."""

    __tablename__ = "search_prompts"

    id = Column(UUID(as_uuid=False), primary_key=True, default=lambda: str(_uuid.uuid4()))
    project_id = Column(
        UUID(as_uuid=False), ForeignKey("geo_projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    template_type = Column(
        Text, nullable=False,
        # brand_direct: "{brand} review/评测"
        # category_discovery: "best {industry} 2026"
        # competitor_comparison: "{brand} vs {competitor}"
        # brand_review: "{brand} 怎么样 用户评价"
    )
    query_text = Column(Text, nullable=False)  # The actual search query sent to engines
    confirmed = Column(Boolean, default=False, nullable=False)
    user_edited = Column(Boolean, default=False, nullable=False)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    project = relationship("GeoProject", back_populates="prompts")
    probe_results = relationship("ProbeResult", back_populates="prompt", cascade="all, delete-orphan")


class ProbeResult(Base):
    """Single probe result — one prompt × one engine."""

    __tablename__ = "probe_results"

    id = Column(UUID(as_uuid=False), primary_key=True, default=lambda: str(_uuid.uuid4()))
    project_id = Column(
        UUID(as_uuid=False), ForeignKey("geo_projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    search_prompt_id = Column(
        UUID(as_uuid=False), ForeignKey("search_prompts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    engine = Column(Text, default="tavily", nullable=False)  # tavily, perplexity, doubao...

    total_results = Column(Integer, default=0, nullable=False)
    brand_mentions = Column(Integer, default=0, nullable=False)  # Times brand name appears in results
    mention_positions = Column(JSONB, default=list, nullable=False)  # [1, 3, 5] — positions where brand appears
    has_own_domain = Column(Boolean, default=False, nullable=False)  # Brand's domain in results
    ai_answer_text = Column(Text, nullable=True)  # AI-generated answer text if available
    visibility_score = Column(Float, default=0.0, nullable=False)  # 0.0–1.0
    top_results = Column(JSONB, default=list, nullable=False)  # Top N results with metadata
    raw_response = Column(JSONB, default=dict, nullable=False)  # Full API response for audit trail

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    project = relationship("GeoProject", back_populates="results")
    prompt = relationship("SearchPrompt", back_populates="probe_results")


class ProjectReport(Base):
    """Aggregated visibility report for a project — regenerated after each probe run."""

    __tablename__ = "project_reports"

    id = Column(UUID(as_uuid=False), primary_key=True, default=lambda: str(_uuid.uuid4()))
    project_id = Column(
        UUID(as_uuid=False), ForeignKey("geo_projects.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Overall score (0–100)
    overall_visibility = Column(Float, default=0.0, nullable=False)

    # Per-template scores
    branded_visibility = Column(Float, default=0.0, nullable=False)
    category_visibility = Column(Float, default=0.0, nullable=False)
    competitor_visibility = Column(Float, default=0.0, nullable=False)
    review_visibility = Column(Float, default=0.0, nullable=False)

    # AI-generated recommendations
    recommendations = Column(JSONB, default=list, nullable=False)

    # Competitive benchmark
    competitor_benchmark = Column(JSONB, default=dict, nullable=False)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    project = relationship("GeoProject", back_populates="reports")


# ── Initialization ───────────────────────────────────────────────────


async def init_db():
    """Create all tables on startup."""
    if not engine:
        logger.warning("No DATABASE_URL — skipping database init")
        return
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized")


async def get_db() -> AsyncSession:
    """Get an async database session (dependency injection)."""
    if not async_session:
        raise RuntimeError("Database not configured")
    async with async_session() as session:
        yield session
