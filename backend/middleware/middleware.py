"""
FastAPI middleware:
- Structured request/response logging
- HTTP metrics (Prometheus)
- API key authentication (optional)
- Request ID injection
"""

from __future__ import annotations

import time
import uuid

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from monitoring.metrics import HTTP_REQUESTS_TOTAL, HTTP_REQUEST_DURATION_SECONDS

logger = structlog.get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Logs every request with timing and status code."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        structlog.contextvars.bind_contextvars(request_id=request_id)

        start = time.time()
        log = logger.bind(
            method=request.method,
            path=request.url.path,
            request_id=request_id,
            client=request.client.host if request.client else "unknown",
        )
        log.info("request_started")

        try:
            response = await call_next(request)
        except Exception as exc:
            log.error("request_unhandled_error", error=str(exc), exc_info=True)
            raise
        finally:
            structlog.contextvars.unbind_contextvars("request_id")

        duration = time.time() - start
        log.info(
            "request_completed",
            status_code=response.status_code,
            duration_seconds=round(duration, 4),
        )

        # Prometheus
        endpoint = request.url.path.split("?")[0]
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=str(response.status_code),
        ).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(duration)

        response.headers["X-Request-ID"] = request_id
        return response


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Optional API key enforcement."""

    def __init__(self, app, api_key: str, header_name: str, enabled: bool):
        super().__init__(app)
        self.api_key = api_key
        self.header_name = header_name
        self.enabled = enabled
        self._exempt_paths = {"/health", "/metrics", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self.enabled or request.url.path in self._exempt_paths:
            return await call_next(request)

        provided = request.headers.get(self.header_name)
        if provided != self.api_key:
            return Response(
                content='{"detail":"Invalid or missing API key"}',
                status_code=401,
                media_type="application/json",
            )

        return await call_next(request)
