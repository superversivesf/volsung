"""
Volsung Coordinator/Gateway - Service routing with smart model loading.

This is the main entry point for the Volsung audio generation system.
It acts as a lightweight gateway that routes requests to microservices
and manages smart model loading (only one model loaded at a time).

Service Registry:
- /voice/design → qwen-voice:8001
- /voice/synthesize → qwen-base:8002
- /voice/styletts → styletts:8003
- /music/generate → music:8004
- /sfx/generate → sfx:8005

Smart Loading:
- Tracks currently loaded service/model
- Unloads current before loading new (resource management)
- Fast path when target already loaded

Admin Endpoints:
- POST /admin/load - Force load specific service
- POST /admin/unload - Force unload
- GET /admin/status - Show loaded service

Example:
    # Start all services
    docker-compose up

    # Or manually
    ./scripts/start-all.sh

    # Then use the coordinator
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/voice/design -d '{"text": "Hello"}'
    curl http://localhost:8000/admin/status

Environment Variables:
    QWEN_VOICE_SERVICE_URL: Qwen voice service (default: http://localhost:8001)
    QWEN_BASE_SERVICE_URL: Qwen base service (default: http://localhost:8002)
    STYLETTS_SERVICE_URL: StyleTTS service (default: http://localhost:8003)
    MUSIC_SERVICE_URL: Music service (default: http://localhost:8004)
    SFX_SERVICE_URL: SFX service (default: http://localhost:8005)
    COORDINATOR_HOST: Coordinator bind address (default: 0.0.0.0)
    COORDINATOR_PORT: Coordinator port (default: 8000)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from volsung.services.client import ServiceClient, ServiceInfo, discover_services

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

QWEN_VOICE_SERVICE_URL = os.getenv("QWEN_VOICE_SERVICE_URL", "http://localhost:8001")
QWEN_BASE_SERVICE_URL = os.getenv("QWEN_BASE_SERVICE_URL", "http://localhost:8002")
STYLETTS_SERVICE_URL = os.getenv("STYLETTS_SERVICE_URL", "http://localhost:8003")
MUSIC_SERVICE_URL = os.getenv("MUSIC_SERVICE_URL", "http://localhost:8004")
SFX_SERVICE_URL = os.getenv("SFX_SERVICE_URL", "http://localhost:8005")
COORDINATOR_HOST = os.getenv("COORDINATOR_HOST", "0.0.0.0")
COORDINATOR_PORT = int(os.getenv("COORDINATOR_PORT", "8000"))

# ============================================================================
# Service Registry & Smart Loading
# ============================================================================


class ServiceName(str, Enum):
    """Service names for the registry."""

    QWEN_VOICE = "qwen-voice"
    QWEN_BASE = "qwen-base"
    STYLETTS = "styletts"
    MUSIC = "music"
    SFX = "sfx"


# Maps endpoints to service names
ENDPOINT_SERVICE_MAP = {
    # Voice services
    "/voice/design": ServiceName.QWEN_VOICE,
    "/voice/synthesize": ServiceName.QWEN_BASE,
    "/voice/styletts": ServiceName.STYLETTS,
    "/voice/styletts/design": ServiceName.STYLETTS,
    # Music services
    "/music/generate": ServiceName.MUSIC,
    # SFX services
    "/sfx/generate": ServiceName.SFX,
    "/sfx/layer": ServiceName.SFX,
}

# Service port mappings
SERVICE_PORTS = {
    ServiceName.QWEN_VOICE: 8001,
    ServiceName.QWEN_BASE: 8002,
    ServiceName.STYLETTS: 8003,
    ServiceName.MUSIC: 8004,
    ServiceName.SFX: 8005,
}


class SmartServiceRegistry:
    """Service registry with smart model loading.

    Only one service's model is kept loaded in memory at a time.
    When a request comes in for a different service:
    1. Unload the current service's model
    2. Load the target service's model
    3. Update currently_loaded tracking
    """

    def __init__(
        self,
        qwen_voice_url: str = QWEN_VOICE_SERVICE_URL,
        qwen_base_url: str = QWEN_BASE_SERVICE_URL,
        styletts_url: str = STYLETTS_SERVICE_URL,
        music_url: str = MUSIC_SERVICE_URL,
        sfx_url: str = SFX_SERVICE_URL,
    ):
        """Initialize the smart service registry.

        Args:
            qwen_voice_url: Qwen voice service URL
            qwen_base_url: Qwen base service URL
            styletts_url: StyleTTS service URL
            music_url: Music service URL
            sfx_url: SFX service URL
        """
        self._clients: Dict[ServiceName, ServiceClient] = {
            ServiceName.QWEN_VOICE: ServiceClient(qwen_voice_url),
            ServiceName.QWEN_BASE: ServiceClient(qwen_base_url),
            ServiceName.STYLETTS: ServiceClient(styletts_url),
            ServiceName.MUSIC: ServiceClient(music_url),
            ServiceName.SFX: ServiceClient(sfx_url),
        }

        self._service_urls: Dict[ServiceName, str] = {
            ServiceName.QWEN_VOICE: qwen_voice_url,
            ServiceName.QWEN_BASE: qwen_base_url,
            ServiceName.STYLETTS: styletts_url,
            ServiceName.MUSIC: music_url,
            ServiceName.SFX: sfx_url,
        }

        # Track which service currently has model loaded
        self._currently_loaded: Optional[ServiceName] = None

        # Lock for concurrent access
        self._loading = False

    @property
    def currently_loaded(self) -> Optional[ServiceName]:
        """Get the currently loaded service name."""
        return self._currently_loaded

    def get_client(self, service_name: ServiceName) -> ServiceClient:
        """Get client for a specific service.

        Args:
            service_name: Service name from ServiceName enum

        Returns:
            ServiceClient instance

        Raises:
            ValueError: If service name is unknown
        """
        if service_name not in self._clients:
            raise ValueError(
                f"Unknown service: {service_name}. "
                f"Available: {[s.value for s in self._clients.keys()]}"
            )
        return self._clients[service_name]

    def get_service_for_endpoint(self, endpoint: str) -> Optional[ServiceName]:
        """Get the service name responsible for an endpoint.

        Args:
            endpoint: API endpoint path

        Returns:
            ServiceName if mapped, None otherwise
        """
        # Exact match first
        if endpoint in ENDPOINT_SERVICE_MAP:
            return ENDPOINT_SERVICE_MAP[endpoint]

        # Prefix match for wildcard paths
        for path_prefix, service in ENDPOINT_SERVICE_MAP.items():
            if endpoint.startswith(path_prefix):
                return service

        return None

    async def ensure_loaded(self, target_service: ServiceName) -> Dict[str, Any]:
        """Ensure target service is loaded (smart loading).

        If target == currently_loaded: do nothing (fast path)
        If target != currently_loaded:
            - Send /unload to current service
            - Send /load to target service
            - Update currently_loaded

        Args:
            target_service: Service that needs to be loaded

        Returns:
            Dictionary with load operation results
        """
        result = {
            "target": target_service.value,
            "previously_loaded": self._currently_loaded.value
            if self._currently_loaded
            else None,
            "actions": [],
        }

        # Fast path: already loaded
        if target_service == self._currently_loaded:
            result["action"] = "noop"
            result["message"] = f"{target_service.value} already loaded"
            return result

        # Need to switch services
        actions = []

        # Step 1: Unload current service (if any)
        if self._currently_loaded is not None:
            try:
                current_client = self._clients[self._currently_loaded]
                unload_response = await current_client.forward(
                    "/unload",
                    method="POST",
                    skip_retry=True,
                )
                actions.append(
                    {
                        "action": "unload",
                        "service": self._currently_loaded.value,
                        "status": unload_response.status_code,
                    }
                )
                logger.info(f"Unloaded {self._currently_loaded.value}")
            except Exception as e:
                logger.warning(f"Failed to unload {self._currently_loaded.value}: {e}")
                actions.append(
                    {
                        "action": "unload",
                        "service": self._currently_loaded.value,
                        "status": "failed",
                        "error": str(e),
                    }
                )

        # Step 2: Load target service
        try:
            target_client = self._clients[target_service]
            load_response = await target_client.forward(
                "/load",
                method="POST",
                skip_retry=True,
            )
            actions.append(
                {
                    "action": "load",
                    "service": target_service.value,
                    "status": load_response.status_code,
                }
            )

            if load_response.status_code == 200:
                self._currently_loaded = target_service
                logger.info(f"Loaded {target_service.value}")
            else:
                error_msg = f"Failed to load {target_service.value}: HTTP {load_response.status_code}"
                logger.error(error_msg)
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "Service load failed",
                        "service": target_service.value,
                        "status_code": load_response.status_code,
                        "actions": actions,
                    },
                )

        except HTTPException:
            raise
        except Exception as e:
            error_msg = f"Failed to load {target_service.value}: {e}"
            logger.error(error_msg)
            actions.append(
                {
                    "action": "load",
                    "service": target_service.value,
                    "status": "failed",
                    "error": str(e),
                }
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Service load failed",
                    "service": target_service.value,
                    "message": str(e),
                    "actions": actions,
                },
            )

        result["action"] = "switched"
        result["actions"] = actions
        return result

    async def force_load(self, service_name: ServiceName) -> Dict[str, Any]:
        """Force load a specific service (admin operation).

        Args:
            service_name: Service to load

        Returns:
            Load operation results
        """
        return await self.ensure_loaded(service_name)

    async def force_unload(self) -> Dict[str, Any]:
        """Force unload currently loaded service (admin operation).

        Returns:
            Unload operation results
        """
        result = {
            "previously_loaded": self._currently_loaded.value
            if self._currently_loaded
            else None,
            "action": "unload",
        }

        if self._currently_loaded is None:
            result["message"] = "No service currently loaded"
            return result

        try:
            current_client = self._clients[self._currently_loaded]
            response = await current_client.forward(
                "/unload",
                method="POST",
                skip_retry=True,
            )
            result["status"] = response.status_code
            result["service"] = self._currently_loaded.value

            if response.status_code == 200:
                logger.info(f"Force unloaded {self._currently_loaded.value}")
                self._currently_loaded = None
            else:
                logger.warning(f"Unload returned {response.status_code}")

        except Exception as e:
            logger.error(f"Failed to unload {self._currently_loaded.value}: {e}")
            result["status"] = "failed"
            result["error"] = str(e)

        return result

    async def get_status(self) -> Dict[str, Any]:
        """Get current loading status.

        Returns:
            Status information
        """
        return {
            "currently_loaded": self._currently_loaded.value
            if self._currently_loaded
            else None,
            "available_services": [s.value for s in self._clients.keys()],
            "service_urls": {k.value: v for k, v in self._service_urls.items()},
        }

    async def health_check_all(self) -> Dict[str, ServiceInfo]:
        """Check health of all registered services.

        Returns:
            Dictionary of service name -> ServiceInfo
        """
        results: Dict[str, ServiceInfo] = {}

        for name, client in self._clients.items():
            try:
                results[name.value] = await client.health()
            except Exception as e:
                results[name.value] = ServiceInfo(
                    name=name.value,
                    url=client.base_url,
                    is_healthy=False,
                    error=str(e),
                )

        return results

    async def close_all(self) -> None:
        """Close all client connections."""
        import asyncio

        await asyncio.gather(
            *[client.close() for client in self._clients.values()],
            return_exceptions=True,
        )


# Global registry instance
_service_registry: Optional[SmartServiceRegistry] = None


def get_registry() -> SmartServiceRegistry:
    """Get or create the smart service registry."""
    global _service_registry
    if _service_registry is None:
        _service_registry = SmartServiceRegistry()
    return _service_registry


async def close_registry() -> None:
    """Close the service registry."""
    global _service_registry
    if _service_registry is not None:
        await _service_registry.close_all()
        _service_registry = None


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


class AdminLoadRequest(BaseModel):
    """Request to force load a service."""

    service: str = Field(..., description="Service name to load")


class AdminLoadResponse(BaseModel):
    """Response from admin load operation."""

    status: str = Field(..., description="Operation status")
    service: str = Field(..., description="Service that was loaded")
    previously_loaded: Optional[str] = Field(
        default=None, description="Previously loaded service"
    )
    actions: List[Dict[str, Any]] = Field(
        default_factory=list, description="Load/unload actions performed"
    )


class AdminUnloadResponse(BaseModel):
    """Response from admin unload operation."""

    status: str = Field(..., description="Operation status")
    previously_loaded: Optional[str] = Field(
        default=None, description="Service that was unloaded"
    )


class AdminStatusResponse(BaseModel):
    """Response from admin status endpoint."""

    currently_loaded: Optional[str] = Field(
        default=None, description="Currently loaded service"
    )
    available_services: List[str] = Field(
        default_factory=list, description="All available services"
    )


class DocumentationResponse(BaseModel):
    """API documentation response."""

    name: str = Field(default="Volsung Coordinator")
    version: str = Field(default="3.0.0")
    description: str = Field(
        default="Gateway for Volsung audio generation with smart model loading"
    )
    architecture: str = Field(default="microservices")
    services: Dict[str, Any] = Field(default_factory=dict)
    endpoints: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# FastAPI Application
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    logger.info("=" * 60)
    logger.info("Volsung Coordinator - Smart Model Loading Gateway")
    logger.info("=" * 60)
    logger.info(f"Qwen-Voice Service:  {QWEN_VOICE_SERVICE_URL}")
    logger.info(f"Qwen-Base Service:   {QWEN_BASE_SERVICE_URL}")
    logger.info(f"StyleTTS Service:    {STYLETTS_SERVICE_URL}")
    logger.info(f"Music Service:       {MUSIC_SERVICE_URL}")
    logger.info(f"SFX Service:         {SFX_SERVICE_URL}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Smart Loading Endpoints:")
    logger.info("  POST /voice/design       → Qwen-Voice (auto-load)")
    logger.info("  POST /voice/synthesize   → Qwen-Base (auto-load)")
    logger.info("  POST /voice/styletts     → StyleTTS (auto-load)")
    logger.info("  POST /music/generate     → Music (auto-load)")
    logger.info("  POST /sfx/generate       → SFX (auto-load)")
    logger.info("")
    logger.info("Admin Endpoints:")
    logger.info("  POST /admin/load         - Force load service")
    logger.info("  POST /admin/unload       - Force unload")
    logger.info("  GET  /admin/status       - Show loaded service")
    logger.info("")
    logger.info("Other Endpoints:")
    logger.info("  GET  /health             - Health check")
    logger.info("  GET  /doc                - API documentation")
    logger.info("=" * 60)

    # Discover services on startup
    services = await discover_services(
        ports={
            "qwen-voice": 8001,
            "qwen-base": 8002,
            "styletts": 8003,
            "music": 8004,
            "sfx": 8005,
        }
    )

    for name, available in services.items():
        status = "✓" if available else "✗"
        logger.info(f"Service '{name}': {status}")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down coordinator...")
    await close_registry()


app = FastAPI(
    title="Volsung Coordinator",
    description="Gateway for Volsung audio generation with smart model loading",
    version="3.0.0",
    lifespan=lifespan,
)


# ============================================================================
# Health & Documentation Endpoints
# ============================================================================


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check health status of all services.

    Aggregates health information from all services:
    - Qwen-Voice service (port 8001)
    - Qwen-Base service (port 8002)
    - StyleTTS service (port 8003)
    - Music service (port 8004)
    - SFX service (port 8005)

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
    registry = get_registry()
    status = await registry.get_status()

    return {
        "name": "Volsung Coordinator",
        "version": "3.0.0",
        "description": "Gateway for Volsung audio generation with smart model loading",
        "architecture": "microservices",
        "currently_loaded": status["currently_loaded"],
        "services": {
            "qwen-voice": {
                "url": QWEN_VOICE_SERVICE_URL,
                "endpoints": ["POST /voice/design - Generate voice from description"],
            },
            "qwen-base": {
                "url": QWEN_BASE_SERVICE_URL,
                "endpoints": ["POST /voice/synthesize - Synthesize with cloned voice"],
            },
            "styletts": {
                "url": STYLETTS_SERVICE_URL,
                "endpoints": ["POST /voice/styletts - Generate voice using StyleTTS2"],
            },
            "music": {
                "url": MUSIC_SERVICE_URL,
                "endpoints": ["POST /music/generate - Generate music from description"],
            },
            "sfx": {
                "url": SFX_SERVICE_URL,
                "endpoints": [
                    "POST /sfx/generate - Generate sound effects",
                    "POST /sfx/layer - Generate layered SFX",
                ],
            },
        },
        "endpoints": {
            "POST /voice/design": {
                "description": "Generate voice using Qwen-Voice (auto-loads model)",
                "service": "qwen-voice",
                "example_request": {
                    "text": "Hello, I am John.",
                    "instruct": "A warm, elderly man's voice",
                    "language": "English",
                },
            },
            "POST /voice/synthesize": {
                "description": "Synthesize with cloned voice (auto-loads model)",
                "service": "qwen-base",
                "example_request": {
                    "text": "Hello, I am John.",
                    "voice_id": "cloned_voice_123",
                },
            },
            "POST /voice/styletts": {
                "description": "Generate voice using StyleTTS2 (auto-loads model)",
                "service": "styletts",
                "example_request": {
                    "text": "Hello, I am John.",
                    "styletts_params": {"embedding_scale": 1.0},
                },
            },
            "POST /music/generate": {
                "description": "Generate music from description (auto-loads model)",
                "service": "music",
                "example_request": {
                    "description": "Peaceful acoustic guitar",
                    "duration": 10.0,
                },
            },
            "POST /sfx/generate": {
                "description": "Generate sound effects (auto-loads model)",
                "service": "sfx",
                "example_request": {
                    "description": "Footsteps on gravel",
                    "duration": 3.0,
                },
            },
            "GET /admin/status": {
                "description": "Show which service currently has model loaded",
            },
            "POST /admin/load": {
                "description": "Force load a specific service",
                "example_request": {"service": "music"},
            },
            "POST /admin/unload": {
                "description": "Force unload current service",
            },
        },
    }


# ============================================================================
# Admin Endpoints
# ============================================================================


@app.post("/admin/load", response_model=AdminLoadResponse)
async def admin_load(request: AdminLoadRequest):
    """Force load a specific service.

    This is useful for:
    - Pre-loading a model before requests come in
    - Testing that a service is working
    - Forcing a model switch

    Args:
        request: AdminLoadRequest with service name

    Returns:
        AdminLoadResponse with load results
    """
    registry = get_registry()

    try:
        service_name = ServiceName(request.service)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid service name",
                "requested": request.service,
                "available": [s.value for s in ServiceName],
            },
        )

    result = await registry.force_load(service_name)

    return AdminLoadResponse(
        status="success" if result.get("action") != "failed" else "failed",
        service=service_name.value,
        previously_loaded=result.get("previously_loaded"),
        actions=result.get("actions", []),
    )


@app.post("/admin/unload", response_model=AdminUnloadResponse)
async def admin_unload():
    """Force unload currently loaded service.

    This frees up GPU memory.

    Returns:
        AdminUnloadResponse with unload results
    """
    registry = get_registry()
    result = await registry.force_unload()

    return AdminUnloadResponse(
        status="success" if result.get("status") not in ["failed", None] else "noop",
        previously_loaded=result.get("previously_loaded"),
    )


@app.post("/admin/unload-all")
async def admin_unload_all():
    """Force unload all services.

    This frees up GPU memory by unloading models from all services.
    Useful for complete resource cleanup.

    Returns:
        Dictionary with unload results for all services
    """
    registry = get_registry()
    results = {}
    any_failed = False

    for service_name in ServiceName:
        try:
            client = registry.get_client(service_name)
            response = await client.forward(
                "/unload",
                method="POST",
                skip_retry=True,
            )
            results[service_name.value] = {
                "status": "unloaded" if response.status_code == 200 else "failed",
                "http_code": response.status_code,
            }
            if response.status_code != 200:
                any_failed = True
        except Exception as e:
            results[service_name.value] = {
                "status": "failed",
                "error": str(e),
            }
            any_failed = True

    # Also clear the currently_loaded tracking
    registry._currently_loaded = None

    return {
        "status": "partial_failure" if any_failed else "success",
        "results": results,
    }


@app.get("/admin/status", response_model=AdminStatusResponse)
async def admin_status():
    """Get current loading status.

    Shows which service currently has its model loaded.

    Returns:
        AdminStatusResponse with current status
    """
    registry = get_registry()
    status = await registry.get_status()

    return AdminStatusResponse(
        currently_loaded=status["currently_loaded"],
        available_services=status["available_services"],
    )


# ============================================================================
# Smart Forwarding with Auto-Loading
# ============================================================================


async def smart_forward(
    request: Request,
    service_name: ServiceName,
    path: str,
    skip_smart_load: bool = False,
) -> Response:
    """Forward request with smart model loading.

    Args:
        request: Incoming FastAPI request
        service_name: Target service
        path: Path to forward to
        skip_smart_load: If True, don't trigger load/unload

    Returns:
        Response from the service

    Raises:
        HTTPException: If service is unavailable or returns an error
    """
    registry = get_registry()
    client = registry.get_client(service_name)

    # Smart loading (unless skipped)
    if not skip_smart_load:
        try:
            load_result = await registry.ensure_loaded(service_name)
            if load_result.get("action") != "noop":
                logger.info(
                    f"Smart load: {load_result.get('previously_loaded')} → {service_name.value}"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Smart loading failed: {e}")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Smart loading failed",
                    "service": service_name.value,
                    "message": str(e),
                },
            )

    # Read request body
    body = await request.body()
    json_data = None
    if body:
        try:
            import json

            json_data = json.loads(body)
        except json.JSONDecodeError:
            pass

    # Prepare headers
    headers = dict(request.headers)
    headers.pop("host", None)

    try:
        response = await client.forward(
            path,
            method=request.method,
            json=json_data,
            data=body if not json_data else None,
            headers=headers,
        )

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.headers.get("content-type"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error forwarding to {service_name.value}: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service unavailable",
                "service": service_name.value,
                "message": str(e),
                "suggestion": f"Check that the {service_name.value} service is running",
            },
        )


# ============================================================================
# Service Routing Endpoints (with Smart Loading)
# ============================================================================


# Voice: Design (Qwen-Voice)
@app.api_route("/voice/design", methods=["GET", "POST", "PUT", "DELETE"])
@app.api_route("/voice/design/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def voice_design_proxy(request: Request, path: str = ""):
    """Proxy voice/design requests to Qwen-Voice service.

    Triggers smart loading to ensure Qwen-Voice model is loaded.
    """
    target_path = f"/voice/design/{path}" if path else "/voice/design"
    return await smart_forward(request, ServiceName.QWEN_VOICE, target_path)


# Voice: Synthesize (Qwen-Base)
@app.api_route("/voice/synthesize", methods=["GET", "POST", "PUT", "DELETE"])
@app.api_route(
    "/voice/synthesize/{path:path}", methods=["GET", "POST", "PUT", "DELETE"]
)
async def voice_synthesize_proxy(request: Request, path: str = ""):
    """Proxy voice/synthesize requests to Qwen-Base service.

    Triggers smart loading to ensure Qwen-Base model is loaded.
    """
    target_path = f"/voice/synthesize/{path}" if path else "/voice/synthesize"
    return await smart_forward(request, ServiceName.QWEN_BASE, target_path)


# Voice: StyleTTS
@app.api_route("/voice/styletts", methods=["GET", "POST", "PUT", "DELETE"])
@app.api_route("/voice/styletts/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def voice_styletts_proxy(request: Request, path: str = ""):
    """Proxy voice/styletts requests to StyleTTS service.

    Triggers smart loading to ensure StyleTTS model is loaded.
    """
    target_path = f"/voice/styletts/{path}" if path else "/voice/styletts"
    return await smart_forward(request, ServiceName.STYLETTS, target_path)


# Music
@app.api_route("/music/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def music_proxy(request: Request, path: str):
    """Proxy music requests to Music service.

    Triggers smart loading to ensure Music model is loaded.
    """
    return await smart_forward(request, ServiceName.MUSIC, f"/music/{path}")


# SFX
@app.api_route("/sfx/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def sfx_proxy(request: Request, path: str):
    """Proxy sfx requests to SFX service.

    Triggers smart loading to ensure SFX model is loaded.
    """
    return await smart_forward(request, ServiceName.SFX, f"/sfx/{path}")


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
