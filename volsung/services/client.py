"""Service client utilities for Volsung coordinator.

Provides HTTP client functionality for communicating with microservices,
including health checks, retries with exponential backoff, and service discovery.

Example:
    from volsung.services.client import ServiceClient, health_check

    # Create client for TTS service
    tts_client = ServiceClient(base_url="http://localhost:8001")

    # Forward a request
    response = await tts_client.forward("/voice/design", method="POST", json=data)

    # Check health
    is_healthy = await health_check("http://localhost:8001")
"""

from __future__ import annotations

import asyncio
import logging
import random
import socket
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# Service port mappings
DEFAULT_SERVICE_PORTS = {
    "tts": 8001,
    "music": 8002,
    "sfx": 8003,
}

DEFAULT_TIMEOUT = 60.0  # seconds for long-running ML inference
DEFAULT_CONNECT_TIMEOUT = 5.0  # seconds for connection establishment


@dataclass
class ServiceInfo:
    """Information about a service endpoint."""

    name: str
    url: str
    is_healthy: bool
    error: Optional[str] = None
    response_time_ms: Optional[float] = None


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt with exponential backoff."""
        delay = self.base_delay * (self.exponential_base**attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add random jitter (±25%)
            jitter_amount = delay * 0.25
            delay = delay + random.uniform(-jitter_amount, jitter_amount)

        return delay


class ServiceClient:
    """HTTP client for service-to-service communication.

    Provides:
    - Request forwarding to microservices
    - Retry logic with exponential backoff
    - Health checking
    - Proper error handling and HTTP status code propagation

    Example:
        client = ServiceClient(base_url="http://localhost:8001")
        response = await client.forward("/voice/design", method="POST", json=data)
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        retry_config: Optional[RetryConfig] = None,
    ):
        """Initialize the service client.

        Args:
            base_url: Base URL of the service (e.g., "http://localhost:8001")
            timeout: Request timeout in seconds
            connect_timeout: Connection timeout in seconds
            retry_config: Retry configuration (uses defaults if not provided)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.retry_config = retry_config or RetryConfig()
        self._client: Optional[httpx.AsyncClient] = None

    @asynccontextmanager
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with proper timeouts."""
        if self._client is None or self._client.is_closed:
            timeout = httpx.Timeout(
                self.timeout,
                connect=self.connect_timeout,
                read=self.timeout,
            )
            self._client = httpx.AsyncClient(
                timeout=timeout,
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
        try:
            yield self._client
        finally:
            # Keep client alive for connection pooling
            pass

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _build_url(self, path: str) -> str:
        """Build full URL from path."""
        path = path.lstrip("/")
        return f"{self.base_url}/{path}"

    async def forward(
        self,
        path: str,
        method: str = "GET",
        json: Optional[Dict[str, Any]] = None,
        data: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        skip_retry: bool = False,
    ) -> httpx.Response:
        """Forward a request to the service with retry logic.

        Args:
            path: API path (e.g., "/voice/design")
            method: HTTP method (GET, POST, etc.)
            json: JSON payload for POST/PUT requests
            data: Raw data payload
            headers: Additional headers
            params: Query parameters
            skip_retry: If True, don't retry on failure

        Returns:
            HTTP response from the service

        Raises:
            HTTPException: If the service returns an error or is unavailable
        """
        url = self._build_url(path)
        request_headers = headers or {}

        # Determine retry count
        max_attempts = 1 if skip_retry else (self.retry_config.max_retries + 1)

        last_error: Optional[Exception] = None

        for attempt in range(max_attempts):
            try:
                async with self._get_client() as client:
                    logger.debug(f"[{method}] {url} (attempt {attempt + 1})")

                    response = await client.request(
                        method=method.upper(),
                        url=url,
                        json=json,
                        content=data,
                        headers=request_headers,
                        params=params,
                    )

                    # If we get a 5xx error, retry (unless it's the last attempt)
                    if response.status_code >= 500 and attempt < max_attempts - 1:
                        delay = self.retry_config.calculate_delay(attempt)
                        logger.warning(
                            f"Service returned {response.status_code}, "
                            f"retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                        continue

                    return response

            except httpx.ConnectError as e:
                last_error = e
                if attempt < max_attempts - 1:
                    delay = self.retry_config.calculate_delay(attempt)
                    logger.warning(
                        f"Failed to connect to service at {self.base_url}, "
                        f"retrying in {delay:.1f}s... ({attempt + 1}/{max_attempts})"
                    )
                    await asyncio.sleep(delay)
                else:
                    break

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < max_attempts - 1:
                    delay = self.retry_config.calculate_delay(attempt)
                    logger.warning(
                        f"Request to {url} timed out, retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    break

            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:
                    delay = self.retry_config.calculate_delay(attempt)
                    logger.warning(f"Request failed: {e}, retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    break

        # All retries exhausted
        error_msg = f"Service unavailable at {self.base_url}"
        if last_error:
            error_msg += f": {last_error}"

        logger.error(error_msg)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service unavailable",
                "service": self.base_url,
                "message": str(last_error)
                if last_error
                else "Could not connect to service",
                "suggestion": "Check that the service is running and accessible",
            },
        )

    async def health(self) -> ServiceInfo:
        """Check if the service is healthy.

        Returns:
            ServiceInfo with health status
        """
        import time

        start_time = time.time()
        url = self._build_url("/health")

        try:
            async with self._get_client() as client:
                response = await client.get(url, timeout=5.0)
                response_time_ms = (time.time() - start_time) * 1000

                is_healthy = response.status_code == 200

                return ServiceInfo(
                    name=self.base_url,
                    url=self.base_url,
                    is_healthy=is_healthy,
                    response_time_ms=response_time_ms,
                    error=None if is_healthy else f"HTTP {response.status_code}",
                )

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return ServiceInfo(
                name=self.base_url,
                url=self.base_url,
                is_healthy=False,
                response_time_ms=response_time_ms,
                error=str(e),
            )


async def health_check(
    base_url: str,
    timeout: float = 5.0,
) -> bool:
    """Quick health check for a service.

    Args:
        base_url: Service URL (e.g., "http://localhost:8001")
        timeout: Timeout in seconds

    Returns:
        True if service is healthy, False otherwise
    """
    url = f"{base_url.rstrip('/')}/health"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            return response.status_code == 200
    except Exception:
        return False


async def discover_services(
    host: str = "localhost",
    ports: Optional[Dict[str, int]] = None,
    timeout: float = 2.0,
) -> Dict[str, bool]:
    """Discover which services are available.

    Checks if services are running on their expected ports.

    Args:
        host: Host to check (default: localhost)
        ports: Dictionary of service name -> port (uses defaults if None)
        timeout: Timeout per service check in seconds

    Returns:
        Dictionary of service name -> is_available
    """
    ports = ports or DEFAULT_SERVICE_PORTS
    results: Dict[str, bool] = {}

    async def check_service(name: str, port: int) -> Tuple[str, bool]:
        """Check if a service is available."""
        url = f"http://{host}:{port}"
        is_healthy = await health_check(url, timeout=timeout)
        return name, is_healthy

    # Check all services concurrently
    tasks = [check_service(name, port) for name, port in ports.items()]
    results_list = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results_list:
        if isinstance(result, Exception):
            continue
        name, is_healthy = result
        results[name] = is_healthy

    return results


def is_port_available(port: int, host: str = "localhost") -> bool:
    """Check if a port is available (not in use).

    Args:
        port: Port number to check
        host: Host to check (default: localhost)

    Returns:
        True if port is available, False if in use
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            result = sock.connect_ex((host, port))
            return result != 0  # 0 means connection succeeded (port is in use)
    except Exception:
        return False


def get_available_port(start_port: int = 8001, host: str = "localhost") -> int:
    """Find the next available port starting from start_port.

    Args:
        start_port: Starting port number to check
        host: Host to check (default: localhost)

    Returns:
        First available port number
    """
    port = start_port
    while not is_port_available(port, host):
        port += 1
        if port > 65535:
            raise RuntimeError("No available ports found")
    return port


class ServiceRegistry:
    """Registry of service endpoints with health tracking."""

    def __init__(
        self,
        tts_url: str = "http://localhost:8001",
        music_url: str = "http://localhost:8002",
        sfx_url: str = "http://localhost:8003",
    ):
        """Initialize the service registry.

        Args:
            tts_url: TTS service URL
            music_url: Music service URL
            sfx_url: SFX service URL
        """
        self.tts = ServiceClient(tts_url)
        self.music = ServiceClient(music_url)
        self.sfx = ServiceClient(sfx_url)

        self._clients: Dict[str, ServiceClient] = {
            "tts": self.tts,
            "music": self.music,
            "sfx": self.sfx,
        }

    async def health_check_all(self) -> Dict[str, ServiceInfo]:
        """Check health of all registered services.

        Returns:
            Dictionary of service name -> ServiceInfo
        """
        results: Dict[str, ServiceInfo] = {}

        tasks = [(name, client.health()) for name, client in self._clients.items()]

        for name, task in tasks:
            try:
                results[name] = await task
            except Exception as e:
                results[name] = ServiceInfo(
                    name=name,
                    url=self._clients[name].base_url,
                    is_healthy=False,
                    error=str(e),
                )

        return results

    def get_client(self, service_name: str) -> ServiceClient:
        """Get client for a specific service.

        Args:
            service_name: One of "tts", "music", or "sfx"

        Returns:
            ServiceClient instance

        Raises:
            ValueError: If service name is unknown
        """
        if service_name not in self._clients:
            raise ValueError(
                f"Unknown service: {service_name}. "
                f"Available: {list(self._clients.keys())}"
            )
        return self._clients[service_name]

    async def close_all(self) -> None:
        """Close all client connections."""
        await asyncio.gather(
            *[client.close() for client in self._clients.values()],
            return_exceptions=True,
        )


# Module-level singleton (for convenience)
_default_registry: Optional[ServiceRegistry] = None


def get_service_registry() -> ServiceRegistry:
    """Get the default service registry singleton.

    Returns:
        ServiceRegistry instance (creates one if needed)
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = ServiceRegistry()
    return _default_registry


async def close_service_registry() -> None:
    """Close the default service registry."""
    global _default_registry
    if _default_registry is not None:
        await _default_registry.close_all()
        _default_registry = None
