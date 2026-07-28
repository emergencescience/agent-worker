"""agent-worker — GEO audit microservice for the Emergence Science platform.

Multi-LLM probe network + brand profiling pipeline.
Deployed on Railway as a sleep-on-idle pod.
"""

from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import jobs, projects
from services.async_jobs import job_store

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("agent-worker")

app = FastAPI(title="Agent Worker API", version="0.1.0")

# CORS (only needed for local dev; orchestrator handles CORS in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Initialize DB tables and start job cleanup loop."""
    # Start async job cleanup
    asyncio.create_task(job_store.cleanup_loop())

    # Initialize database tables
    try:
        from services.database import init_db

        await init_db()
        logger.info("Database tables initialized")
    except Exception as e:
        logger.warning(f"Database init skipped (no DATABASE_URL?): {e}")


@app.get("/health")
def health():
    """Health check endpoint for Railway."""
    from config.probe_registry import get_enabled_probes

    probes = get_enabled_probes()
    return {
        "status": "healthy",
        "service": "agent-worker",
        "version": "0.1.0",
        "probes_available": len(probes),
        "probes": [p.id for p in probes],
    }


# Include routers
app.include_router(jobs.router)
app.include_router(projects.router)

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)
