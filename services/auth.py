from __future__ import annotations

import os
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer(auto_error=False)

AGENT_WORKER_TOKEN = os.getenv("AGENT_WORKER_TOKEN", "")


async def authenticate(request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(security)):
    """Bearer token auth matching agent-render pattern.

    Returns without error if AGENT_WORKER_TOKEN is not configured (dev mode).
    """
    if not AGENT_WORKER_TOKEN:
        return

    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if credentials.credentials != AGENT_WORKER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")
