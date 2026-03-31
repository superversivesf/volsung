"""
Volsung Coordinator/Gateway - Service routing and health aggregation.

This is the main entry point for the Volsung audio generation system.
It acts as a lightweight gateway that routes requests to microservices:
- /voice/* → TTS service (port 8001)
- /music/* → Music service (port 8002)
- /sfx/* → SFX service (port 8003)

The coordinator provides:
- Request routing and forwarding
- Health aggregation from all services
- Service discovery and availability checking
- Helpful error messages when services are unavailable

Example:
    # Start all services
    docker-compose up

    # Or manually
    ./scripts/start-all.sh

    # Then use the coordinator
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/voice/design -d '{"text": "Hello", "instruct": "warm voice"}'

Environment Variables:
    TTS_SERVICE_URL: TTS service URL (default: http://localhost:8001)
    MUSIC_SERVICE_URL: Music service URL (default: http://localhost:8002)
    SFX_SERVICE_URL: SFX service URL (default: http://localhost:8003)
    COORDINATOR_HOST: Coordinator bind address (default: 0.0.0.0)
    COORDINATOR_PORT: Coordinator port (default: 8000)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from volsung.services.client import (
    ServiceClient,
    ServiceInfo,
    ServiceRegistry,
    discover_services,
    get_service_registry,
    close_service_registry,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://localhost:8001")
MUSIC_SERVICE_URL = os.getenv("MUSIC_SERVICE_URL", "http://localhost:8002")
SFX_SERVICE_URL = os.getenv("SFX_SERVICE_URL", "http://localhost:8003")
COORDINATOR_HOST = os.getenv("COORDINATOR_HOST", "0.0.0.0")
COORDINATOR_PORT = int(os.getenv("COORDINATOR_PORT", "8000"))

# ============================================================================
# Pydantic Schemas
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response with service aggregation."""

    status: str = Field(..., description="Overall coordinator status")
    coordinator: str = Field(default="healthy", description="Coordinator status")
    services: Dict[str, Any] = Field(
        default_factory=dict, description="Health status of each service"
    )
    available: List[str] = Field(
        default_factory=list, description="List of available services"
    )
    unavailable: List[str] = Field(
        default_factory=list, description="List of unavailable services"
    )


class PreloadRequest(BaseModel):
    """Request for preloading models across services."""

    models: List[str] = Field(
        default_factory=lambda: ["all"],
        description="Models to preload: 'tts', 'music', 'sfx', or 'all'",
    )


class PreloadResponse(BaseModel):
    """Preload response with per-service results."""

    status: str = Field(..., description="Overall preload status")
    results: Dict[str, Any] = Field(
        default_factory=dict, description="Preload results per service"
    )


class ServiceStatus(BaseModel):
    """Status of a single service."""

    name: str
    url: str
    healthy: bool
    response_time_ms: Optional[float] = None
    error: Optional[str] = None


class DocumentationResponse(BaseModel):
    """API documentation response."""

    name: str = Field(default="Volsung Coordinator")
    version: str = Field(default="2.0.0")
    description: str = Field(
        default="Gateway for Volsung audio generation microservices"
    )
    architecture: str = Field(default="microservices")
    services: Dict[str, Any] = Field(default_factory=dict)
    endpoints: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Global State
# ============================================================================

_service_registry: Optional[ServiceRegistry] = None


def get_registry() -> ServiceRegistry:
    """Get or create the service registry."""
    global _service_registry
    if _service_registry is None:
        _service_registry = ServiceRegistry(
            tts_url=TTS_SERVICE_URL,
            music_url=MUSIC_SERVICE_URL,
            sfx_url=SFX_SERVICE_URL,
        )
    return _service_registry


# ============================================================================
# FastAPI Application
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    logger.info("=" * 60)
    logger.info("Volsung Coordinator - Gateway for Audio Generation Services")
    logger.info("=" * 60)
    logger.info(f"TTS Service:    {TTS_SERVICE_URL}")
    logger.info(f"Music Service:  {MUSIC_SERVICE_URL}")
    logger.info(f"SFX Service:    {SFX_SERVICE_URL}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Endpoints:")
    logger.info("  GET  /health        - Health check (aggregates all services)")
    logger.info("  GET  /doc           - API documentation")
    logger.info("  POST /preload       - Preload models in all services")
    logger.info("  ANY  /voice/*       - Forward to TTS service")
    logger.info("  ANY  /music/*       - Forward to Music service")
    logger.info("  ANY  /sfx/*         - Forward to SFX service")
    logger.info("=" * 60)

    # Discover services on startup
    services = await discover_services(
        ports={
            "tts": 8001,
            "music": 8002,
            "sfx": 8003,
        }
    )

    for name, available in services.items():
        status = "✓" if available else "✗"
        logger.info(f"Service '{name}': {status}")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down coordinator...")
    await close_service_registry()
    _service_registry = None


app = FastAPI(
    title="Volsung Coordinator",
    description="Gateway for Volsung audio generation microservices",
    version="2.0.0",
    lifespan=lifespan,
)


# ============================================================================
# Health & Status Endpoints
# ============================================================================


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check health status of all services.

    Aggregates health information from:
    - TTS service (port 8001)
    - Music service (port 8002)
    - SFX service (port 8003)

    Returns:
        HealthResponse with overall status and per-service details
    """
    registry = get_registry()
    service_health = await registry.health_check_all()

    available = []
    unavailable = []
    services_info: Dict[str, Any] = {}

    for name, info in service_health.items():
        service_data = {
            "healthy": info.is_healthy,
            "url": info.url,
        }
        if info.response_time_ms:
            service_data["response_time_ms"] = round(info.response_time_ms, 2)
        if info.error:
            service_data["error"] = info.error

        services_info[name] = service_data

        if info.is_healthy:
            available.append(name)
        else:
            unavailable.append(name)

    overall_status = "healthy" if len(available) > 0 else "degraded"
    if len(unavailable) == 0:
        overall_status = "healthy"
    elif len(available) == 0:
        overall_status = "unavailable"

    return HealthResponse(
        status=overall_status,
        coordinator="healthy",
        services=services_info,
        available=available,
        unavailable=unavailable,
    )


@app.get("/doc")
async def documentation():
    """Get full API documentation.

    Returns:
        DocumentationResponse with service endpoints and usage examples
    """
    return {
        "name": "Volsung Coordinator",
        "version": "2.0.0",
        "description": "Gateway for Volsung audio generation microservices",
        "architecture": "microservices",
        "services": {
            "tts": {
                "url": TTS_SERVICE_URL,
                "endpoints": [
                    "GET  /health - Health check",
                    "POST /voice/design - Generate voice from description",
                    "POST /voice/synthesize - Synthesize with cloned voice",
                ],
            },
            "music": {
                "url": MUSIC_SERVICE_URL,
                "endpoints": [
                    "GET  /health - Health check",
                    "GET  /info - Service information",
                    "POST /music/generate - Generate music from description",
                ],
            },
            "sfx": {
                "url": SFX_SERVICE_URL,
                "endpoints": [
                    "GET  /health - Health check",
                    "POST /sfx/generate - Generate sound effects",
                    "POST /sfx/layer - Generate layered SFX",
                ],
            },
        },
        "endpoints": {
            "GET /health": {
                "description": "Check health of all services",
                "example_response": {
                    "status": "healthy",
                    "coordinator": "healthy",
                    "services": {
                        "tts": {"healthy": True, "response_time_ms": 12.5},
                        "music": {"healthy": True, "response_time_ms": 8.3},
                        "sfx": {"healthy": True, "response_time_ms": 15.1},
                    },
                    "available": ["tts", "music", "sfx"],
                    "unavailable": [],
                },
            },
            "POST /preload": {
                "description": "Preload models across all services",
                "request": {"models": ["all"]},
                "example_response": {
                    "status": "ok",
                    "results": {
                        "tts": {"status": "loaded"},
                        "music": {"status": "loaded"},
                        "sfx": {"status": "loaded"},
                    },
                },
            },
            "POST /voice/design": {
                "description": "Generate voice from description (proxied to TTS)",
                "request": {
                    "text": "Hello, I am John.",
                    "instruct": "A warm, elderly man's voice",
                    "language": "English",
                },
            },
            "POST /music/generate": {
                "description": "Generate music from description (proxied to Music)",
                "request": {
                    "description": "Peaceful acoustic guitar",
                    "duration": 10.0,
                },
            },
            "POST /sfx/generate": {
                "description": "Generate sound effects (proxied to SFX)",
                "request": {
                    "description": "Footsteps on gravel",
                    "duration": 3.0,
                },
            },
        },
        "environment_variables": {
            "TTS_SERVICE_URL": f"TTS service URL (default: {TTS_SERVICE_URL})",
            "MUSIC_SERVICE_URL": f"Music service URL (default: {MUSIC_SERVICE_URL})",
            "SFX_SERVICE_URL": f"SFX service URL (default: {SFX_SERVICE_URL})",
        },
    }


@app.post("/preload", response_model=PreloadResponse)
async def preload(request: PreloadRequest):
    """Preload models across all services.

    Forwards preload requests to each service that supports it.
    Services that are unavailable will be skipped.

    Args:
        request: PreloadRequest with list of models to preload

    Returns:
        PreloadResponse with results from each service
    """
    registry = get_registry()
    results: Dict[str, Any] = {}
    any_success = False

    # Services that support preloading
    preload_endpoints = {
        "tts": ("/preload", "tts"),
        "music": ("/preload", "music"),
        "sfx": ("/preload", "sfx"),
    }

    for service_name, (endpoint, client_name) in preload_endpoints.items():
        try:
            client = registry.get_client(client_name)
            response = await client.forward(
                endpoint,
                method="POST",
                json={"models": request.models},
                skip_retry=True,  # Don't retry preloads
            )

            if response.status_code == 200:
                results[service_name] = response.json()
                any_success = True
            else:
                results[service_name] = {
                    "status": "error",
                    "code": response.status_code,
                }

        except HTTPException as e:
            results[service_name] = {
                "status": "unavailable",
                "error": e.detail
                if isinstance(e.detail, str)
                else "Service unavailable",
            }
        except Exception as e:
            results[service_name] = {
                "status": "error",
                "error": str(e),
            }

    status = "ok" if any_success else "failed"
    return PreloadResponse(status=status, results=results)


# ============================================================================
# Service Routing Endpoints
# ============================================================================


async def forward_to_service(
    request: Request,
    service_name: str,
    path: str,
) -> Response:
    """Forward a request to the appropriate service.

    Args:
        request: Incoming FastAPI request
        service_name: Name of the service (tts, music, sfx)
        path: Path to forward to

    Returns:
        Response from the service

    Raises:
        HTTPException: If service is unavailable or returns an error
    """
    registry = get_registry()
    client = registry.get_client(service_name)

    # Read request body
    body = await request.body()
    json_data = None
    if body:
        try:
            import json

            json_data = json.loads(body)
        except json.JSONDecodeError:
            # Not JSON, pass as raw data
            pass

    # Prepare headers
    headers = dict(request.headers)
    headers.pop("host", None)  # Remove host header (will be set by client)

    try:
        response = await client.forward(
            path,
            method=request.method,
            json=json_data,
            data=body if not json_data else None,
            headers=headers,
        )

        # Forward response
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.headers.get("content-type"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error forwarding to {service_name}: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service unavailable",
                "service": service_name,
                "message": str(e),
                "suggestion": f"Check that the {service_name} service is running",
            },
        )


# TTS Service Routes
@app.api_route("/voice/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def tts_proxy(request: Request, path: str):
    """Proxy all voice-related requests to TTS service.

    Routes:
        POST /voice/design - Generate voice from description
        POST /voice/synthesize - Synthesize with cloned voice
        GET  /voice/* - Other TTS endpoints
    """
    return await forward_to_service(request, "tts", f"/voice/{path}")


# Music Service Routes
@app.api_route("/music/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def music_proxy(request: Request, path: str):
    """Proxy all music-related requests to Music service.

    Routes:
        POST /music/generate - Generate music from description
        GET  /music/info - Service information
        GET  /music/* - Other Music endpoints
    """
    return await forward_to_service(request, "music", f"/music/{path}")


# SFX Service Routes
@app.api_route("/sfx/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def sfx_proxy(request: Request, path: str):
    """Proxy all SFX-related requests to SFX service.

    Routes:
        POST /sfx/generate - Generate sound effects
        POST /sfx/layer - Generate layered SFX
        GET  /sfx/* - Other SFX endpoints
    """
    return await forward_to_service(request, "sfx", f"/sfx/{path}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info(
        f"Starting Volsung Coordinator on {COORDINATOR_HOST}:{COORDINATOR_PORT}"
    )
    uvicorn.run(
        "volsung.server:app",
        host=COORDINATOR_HOST,
        port=COORDINATOR_PORT,
        log_level="info",
        access_log=True,
    )
