"""Rate limiting and circuit breaker for agent-worker.

Rate limiter: Token bucket per IP, protects pod from request floods.
Circuit breaker: Stops calling a failing external API after N consecutive failures.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger("agent_protection")

# ── Rate Limiter ───────────────────────────────────────────────

# Token bucket: max 30 requests per IP per 60 seconds
RATE_LIMIT_MAX_TOKENS = 30
RATE_LIMIT_REFILL_SECONDS = 60
RATE_LIMIT_TOKENS_PER_REFILL = 30  # Refill all tokens every 60s


class TokenBucket:
    """Simple token bucket rate limiter."""

    def __init__(self, max_tokens: int, refill_rate: float):
        self.max_tokens = max_tokens
        self.tokens = float(max_tokens)
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.monotonic()

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if allowed."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP rate limiting middleware.

    Exempts /health endpoint from rate limiting.
    """

    def __init__(self, app, max_requests: int = RATE_LIMIT_MAX_TOKENS,
                 window_seconds: int = RATE_LIMIT_REFILL_SECONDS):
        super().__init__(app)
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()
        refill_rate = RATE_LIMIT_TOKENS_PER_REFILL / window_seconds
        self._max_tokens = max_requests
        self._refill_rate = refill_rate

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path == "/health":
            return await call_next(request)

        # Get client IP
        client_ip = (
            request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or request.headers.get("X-Real-IP", "")
            or request.client.host if request.client else "unknown"
        )

        async with self._lock:
            bucket = self._buckets.get(client_ip)
            if bucket is None:
                bucket = TokenBucket(self._max_tokens, self._refill_rate)
                self._buckets[client_ip] = bucket

        if not bucket.consume():
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "status": "error",
                    "code": 429,
                    "message": "Too many requests. Try again in 60 seconds.",
                },
            )

        return await call_next(request)


# ── Circuit Breaker ────────────────────────────────────────────

class CircuitBreaker:
    """Circuit breaker for external API calls.

    States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing)
    """

    def __init__(self, name: str, failure_threshold: int = 5,
                 recovery_timeout: float = 60.0, half_open_max: int = 1):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout  # seconds before trying again
        self.half_open_max = half_open_max

        self._failure_count = 0
        self._last_failure_time = 0.0
        self._state = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
        self._half_open_requests = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        return self._state

    async def call(self, fn: Callable, *args, **kwargs) -> Any:
        """Call the function with circuit breaker protection.

        Raises RuntimeError if circuit is OPEN.
        """
        async with self._lock:
            if self._state == "OPEN":
                if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                    self._state = "HALF_OPEN"
                    self._half_open_requests = 0
                    logger.info(f"Circuit {self.name}: OPEN → HALF_OPEN")
                else:
                    raise RuntimeError(
                        f"Circuit {self.name} is OPEN. "
                        f"Retry in {self.recovery_timeout - (time.monotonic() - self._last_failure_time):.0f}s"
                    )

            if self._state == "HALF_OPEN":
                if self._half_open_requests >= self.half_open_max:
                    raise RuntimeError(f"Circuit {self.name} is HALF_OPEN (max probes reached)")

                self._half_open_requests += 1

        # Actually execute the call (outside lock to allow concurrent calls)
        try:
            result = await fn(*args, **kwargs)

            # Success! Reset.
            async with self._lock:
                if self._state == "HALF_OPEN":
                    logger.info(f"Circuit {self.name}: HALF_OPEN → CLOSED (recovered)")
                self._state = "CLOSED"
                self._failure_count = 0

            return result

        except Exception as e:
            async with self._lock:
                self._failure_count += 1
                self._last_failure_time = time.monotonic()

                if self._state == "HALF_OPEN" or self._failure_count >= self.failure_threshold:
                    if self._state != "OPEN":
                        logger.warning(
                            f"Circuit {self.name}: {self._state} → OPEN "
                            f"({self._failure_count} failures, threshold={self.failure_threshold})"
                        )
                    self._state = "OPEN"

            raise  # Re-raise the original exception


# ── Global circuit breakers ────────────────────────────────────

_circuits: dict[str, CircuitBreaker] = {}


def get_circuit(name: str) -> CircuitBreaker:
    """Get or create a circuit breaker for an external service."""
    if name not in _circuits:
        _circuits[name] = CircuitBreaker(name=name)
    return _circuits[name]
