"""Database connection and table initialization."""

from __future__ import annotations

import logging
import os

from sqlalchemy import Column, DateTime, Float, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger("database")

DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgresql://", "postgresql+asyncpg://", 1)

if not DATABASE_URL:
    logger.warning("DATABASE_URL not set — database features disabled")

engine = create_async_engine(DATABASE_URL, echo=False, pool_size=5, max_overflow=10)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class GeoProfileRow(Base):
    """geo_profiles table — stores extracted brand profiles."""

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


async def init_db():
    """Create all tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized")


async def get_session() -> AsyncSession:
    """Get an async database session."""
    async with async_session() as session:
        yield session
