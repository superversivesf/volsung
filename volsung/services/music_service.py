"""Mustango Music Service for Volsung.

A standalone FastAPI service that provides music generation endpoints using Mustango.
Runs on port 8004 and handles music generation from text descriptions.

Apache 2.0 Licensed - Commercial use allowed!

Example:
    # Start the service
    python -m volsung.services.music_service

    # Or using uvicorn directly
    uvicorn volsung.services.music_service:app --host 0.0.0.0 --port 8004

Environment Variables:
    MUSIC_SERVICE_HOST: Server bind address (default: 0.0.0.0)
    MUSIC_SERVICE_PORT: Server port (default: 8004)
    MUSIC_DEVICE: Device override (default: auto-detected)
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================


class MusicServiceConfig(BaseModel):
    """Music service configuration."""

    host: str = Field(
        default_factory=lambda: os.getenv("MUSIC_SERVICE_HOST", "0.0.0.0")
    )
    port: int = Field(
        default_factory=lambda: int(os.getenv("MUSIC_SERVICE_PORT", "8004"))
    )
    model_id: str = Field(default="declare-lab/mustango")
    device: Optional[str] = Field(default_factory=lambda: os.getenv("MUSIC_DEVICE"))


# =============================================================================
# Pydantic Schemas
# =============================================================================


class GenerateRequest(BaseModel):
    """Request for generating music from text description."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "description": "Peaceful acoustic guitar for meditation audiobook",
                "duration": 30,
            }
        }
    )

    description: str = Field(
        ...,
        description="Natural language description of desired music",
        min_length=1,
        max_length=1000,
    )
    duration: int = Field(
        default=30,
        ge=10,
        le=60,
        description="Target duration in seconds (10-60, actual may vary)",
    )


class MusicMetadata(BaseModel):
    """Metadata for generated music."""

    duration: float = Field(..., description="Actual duration in seconds")
    sample_rate: int = Field(..., description="Audio sample rate in Hz")
    generation_time_ms: float = Field(
        ...,
        description="Time taken to generate in milliseconds",
    )
    model_used: str = Field(..., description="Model identifier used")


class GenerateResponse(BaseModel):
    """Response containing generated music and metadata."""

    audio: str = Field(..., description="Base64-encoded WAV audio data")
    sample_rate: int = Field(..., description="Sample rate in Hz")
    metadata: MusicMetadata = Field(..., description="Generation metadata")


class LoadRequest(BaseModel):
    """Request to load the model."""

    device: Optional[str] = Field(
        default=None,
        description="Device to load on (cuda, cpu). Auto-detected if not specified.",
    )


class LoadResponse(BaseModel):
    """Response from load operation."""

    status: str = Field(..., description="Status: loaded or already_loaded")
    model: str = Field(..., description="Model identifier")
    device: str = Field(..., description="Device model is loaded on")
    message: str = Field(..., description="Human-readable status message")
    license: str = Field(default="Apache 2.0", description="Model license")


class UnloadResponse(BaseModel):
    """Response from unload operation."""

    status: str = Field(..., description="Status: unloaded or not_loaded")
    model: str = Field(..., description="Model identifier")
    message: str = Field(..., description="Human-readable status message")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Overall service status")
    service: str = Field(default="music", description="Service name")
    model: dict = Field(default_factory=dict, description="Model status")


# =============================================================================
# Model Manager Class
# =============================================================================


class MustangoManager:
    """Manager for Mustango model."""

    def __init__(self, config: MusicServiceConfig):
        self.config = config
        self._model: Optional[Any] = None
        self._device = self._get_device()
        self._is_loaded = False
        self._weights_cached = False

    def is_weights_cached(self) -> bool:
        """Check if Mustango weights are cached in HuggingFace cache.

        Mustango uses diffusers, so we check for the model in HF cache.
        The model is identified as 'declare-lab/mustango'.

        Returns:
            True if weights are cached, False otherwise
        """
        try:
            # Get HuggingFace cache directory
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

            # Check for diffusers models cache
            # Mustango uses diffusers, which stores models with 'models--' prefix
            model_name = self.config.model_id.replace("/", "--")
            expected_pattern = f"models--{model_name}"

            if not cache_dir.exists():
                logger.debug(f"HF cache directory not found: {cache_dir}")
                return False

            # Look for the model in cache
            for item in cache_dir.iterdir():
                if item.is_dir() and expected_pattern in item.name:
                    # Check if there are actual model files
                    snapshots_dir = item / "snapshots"
                    if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
                        logger.debug(f"Found cached weights: {item.name}")
                        return True

            logger.debug(f"Weights not cached for {self.config.model_id}")
            return False

        except Exception as e:
            logger.warning(f"Error checking weights cache: {e}")
            return False

    def download_weights(self) -> Dict[str, Any]:
        """Download Mustango weights from HuggingFace.

        Uses diffusers to download the model weights without loading them
        into memory. This allows pre-caching before first generation.

        Returns:
            Dict with status, message, and cache location
        """
        import gc

        if self.is_weights_cached():
            return {
                "status": "cached",
                "message": f"Weights already cached for {self.config.model_id}",
                "cache_location": str(Path.home() / ".cache" / "huggingface" / "hub"),
            }

        logger.info(f"Downloading Mustango weights for {self.config.model_id}...")

        try:
            from diffusers import DiffusionPipeline

            # Download the model without loading into memory
            # Using from_pretrained with cache_dir will download and cache
            logger.info("Downloading model weights (this may take a while)...")

            # Just download by creating the pipeline
            # We don't keep it, just let it download to cache
            _ = DiffusionPipeline.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.float32,
                cache_dir=None,  # Use default HF cache
            )

            # Clear any loaded tensors
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            cache_location = Path.home() / ".cache" / "huggingface" / "hub"

            logger.info(f"Weights downloaded successfully to {cache_location}")

            return {
                "status": "downloaded",
                "message": f"Weights downloaded successfully for {self.config.model_id}",
                "cache_location": str(cache_location),
            }

        except Exception as e:
            logger.error(f"Failed to download weights: {e}")
            return {
                "status": "error",
                "message": f"Failed to download weights: {str(e)}",
                "cache_location": None,
            }

    def _get_device(self) -> str:
        """Get the best available device."""
        if self.config.device:
            return self.config.device
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load(self, device: Optional[str] = None) -> dict:
        """Load the Mustango model."""
        if self._is_loaded:
            return {
                "status": "already_loaded",
                "device": self._device,
                "message": f"Model already loaded on {self._device}",
            }

        try:
            from mustango import Mustango
        except ImportError:
            raise ImportError(
                "Mustango is required. Install with: pip install git+https://github.com/AMAAI-Lab/mustango.git"
            )

        if device:
            self._device = device

        logger.info(f"Loading Mustango on {self._device}...")

        try:
            self._model = Mustango(self.config.model_id, device=self._device)
            self._is_loaded = True
            logger.info("Mustango model loaded successfully")

            return {
                "status": "loaded",
                "device": self._device,
                "message": f"Model loaded successfully on {self._device}",
                "license": "Apache 2.0",
            }
        except Exception as e:
            logger.error(f"Failed to load Mustango: {e}")
            raise RuntimeError(f"Failed to load Mustango: {e}")

    def unload(self) -> dict:
        """Unload the model and free resources."""
        if not self._is_loaded:
            return {"status": "not_loaded", "message": "Model was not loaded"}

        logger.info("Unloading Mustango model...")
        self._model = None
        self._is_loaded = False

        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Mustango model unloaded")

        return {"status": "unloaded", "message": "Model unloaded successfully"}

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def device(self) -> str:
        return self._device

    def generate(self, prompt: str) -> Tuple[np.ndarray, int]:
        """Generate music from text prompt."""
        if not self._is_loaded:
            self.load()

        if self._model is None:
            raise RuntimeError("Mustango model not loaded")

        logger.info(f"[Mustango] Generating music for: '{prompt[:50]}...'")

        # Generate
        audio = self._model.generate(prompt)
        sample_rate = 16000  # Mustango outputs 16kHz

        logger.info(f"[Mustango] Generated: {len(audio)} samples @ {sample_rate}Hz")

        return audio, sample_rate


# =============================================================================
# Global State
# =============================================================================

_config = MusicServiceConfig()
_mustango_manager: Optional[MustangoManager] = None


def get_mustango_manager() -> MustangoManager:
    """Get or create the Mustango manager singleton."""
    global _mustango_manager
    if _mustango_manager is None:
        _mustango_manager = MustangoManager(_config)
    return _mustango_manager


# =============================================================================
# FastAPI Application
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    On startup, downloads model weights if not already cached.
    """
    logger.info("Music Service starting up...")

    # Initialize manager and download weights if needed
    manager = get_mustango_manager()
    if not manager.is_weights_cached():
        logger.info("Model weights not cached, downloading...")
        download_result = manager.download_weights()
        if download_result["status"] == "error":
            logger.warning(
                f"Failed to pre-download weights: {download_result['message']}"
            )
        else:
            logger.info(f"Weights status: {download_result['message']}")
    else:
        logger.info("Model weights already cached")

    yield

    logger.info("Music Service shutting down...")
    global _mustango_manager
    if _mustango_manager:
        _mustango_manager.unload()


app = FastAPI(
    title="Volsung Music Service (Mustango)",
    description="Standalone music generation service using Mustango (Apache 2.0 licensed)",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check() -> dict:
    """Check service health status.

    Returns status based on whether weights are cached:
    - ready: Weights are cached and available
    - downloading: Weights are being downloaded
    - unavailable: Weights not cached and mustango unavailable
    """
    model_status = {
        "name": "mustango",
        "available": True,
        "loaded": False,
        "device": None,
        "license": "Apache 2.0",
        "weights_cached": False,
    }

    try:
        from mustango import Mustango

        model_status["available"] = True
        if _mustango_manager is not None:
            model_status["loaded"] = _mustango_manager.is_loaded
            model_status["device"] = (
                _mustango_manager.device if _mustango_manager.is_loaded else None
            )
            model_status["weights_cached"] = _mustango_manager.is_weights_cached()
    except ImportError:
        model_status["available"] = False

    # Determine overall status based on cache state
    if model_status["weights_cached"]:
        overall_status = "ready"
    elif model_status["available"]:
        overall_status = "healthy"  # Available but not cached yet
    else:
        overall_status = "unavailable"

    return {
        "status": overall_status,
        "service": "music",
        "model": model_status,
    }


@app.post("/load", response_model=LoadResponse)
async def load_model(request: LoadRequest = None) -> LoadResponse:
    """Load the Mustango model."""
    manager = get_mustango_manager()
    device = request.device if request else None

    try:
        result = manager.load(device)
        return LoadResponse(
            status=result["status"],
            model="mustango",
            device=result["device"],
            message=result["message"],
            license="Apache 2.0",
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to load model: {str(e)}",
        )


@app.post("/unload", response_model=UnloadResponse)
async def unload_model() -> UnloadResponse:
    """Unload the Mustango model and free resources."""
    manager = get_mustango_manager()
    result = manager.unload()

    return UnloadResponse(
        status=result["status"],
        model="mustango",
        message=result["message"],
    )


@app.post("/music/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate music from text description using Mustango."""
    try:
        start_time = time.time()
        manager = get_mustango_manager()

        # Generate music
        audio_array, sample_rate = manager.generate(prompt=request.description)

        # Convert to base64 WAV
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, sample_rate, format="WAV")
        buffer.seek(0)
        audio_b64 = base64.b64encode(buffer.read()).decode()

        actual_duration = len(audio_array) / sample_rate
        elapsed_ms = (time.time() - start_time) * 1000

        metadata = MusicMetadata(
            duration=actual_duration,
            sample_rate=sample_rate,
            generation_time_ms=elapsed_ms,
            model_used="declare-lab/mustango",
        )

        return GenerateResponse(
            audio=audio_b64,
            sample_rate=sample_rate,
            metadata=metadata,
        )

    except Exception as e:
        logger.error(f"[MUSIC] Generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Music generation failed: {str(e)}",
        )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    config = MusicServiceConfig()
    logger.info(f"Starting Music Service on {config.host}:{config.port}")
    uvicorn.run(
        "volsung.services.music_service:app",
        host=config.host,
        port=config.port,
        log_level="info",
        reload=False,
    )
