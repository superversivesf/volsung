"""AudioLDM SFX Service for Volsung.

A standalone FastAPI service that provides AudioLDM2 sound effects generation endpoints.
Runs on port 8005 and handles SFX generation from text descriptions.

Example:
    # Start the service
    python -m volsung.services.sfx_service

    # Or using uvicorn directly
    uvicorn volsung.services.sfx_service:app --host 0.0.0.0 --port 8005

Environment Variables:
    SFX_SERVICE_HOST: Server bind address (default: 0.0.0.0)
    SFX_SERVICE_PORT: Server port (default: 8005)
    SFX_DEVICE: Device override (default: auto-detected)
"""

from __future__ import annotations

import base64
import logging
import os
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, status
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================


class SFXServiceConfig(BaseModel):
    """SFX service configuration."""

    host: str = Field(default_factory=lambda: os.getenv("SFX_SERVICE_HOST", "0.0.0.0"))
    port: int = Field(
        default_factory=lambda: int(os.getenv("SFX_SERVICE_PORT", "8004"))
    )
    model_size: str = Field(default="base")
    device: Optional[str] = Field(default_factory=lambda: os.getenv("SFX_DEVICE"))


# =============================================================================
# Pydantic Schemas
# =============================================================================


class GenerateRequest(BaseModel):
    """Request for generating sound effects from text description."""

    description: str = Field(
        ...,
        description="Natural language description of desired sound effect",
        min_length=1,
        max_length=1000,
    )
    duration: float = Field(
        default=5.0,
        ge=1.0,
        le=10.0,
        description="Target duration in seconds (1.0 to 10.0)",
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional category hint (e.g., 'nature', 'mechanical', 'urban')",
    )
    num_inference_steps: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Number of denoising steps (higher = better quality, slower)",
    )
    guidance_scale: float = Field(
        default=3.5,
        ge=1.0,
        le=20.0,
        description="Prompt adherence (higher = more faithful to prompt)",
    )


class SFXMetadata(BaseModel):
    """Metadata for generated sound effects."""

    duration: float = Field(..., description="Audio duration in seconds")
    sample_rate: int = Field(..., description="Sample rate in Hz")
    category: Optional[str] = Field(default=None, description="SFX category")
    generation_time_ms: float = Field(..., description="Time taken to generate (ms)")
    model_used: str = Field(..., description="Model identifier")
    num_inference_steps: int = Field(default=50, description="Denoising steps used")
    guidance_scale: float = Field(default=3.5, description="Guidance scale used")


class GenerateResponse(BaseModel):
    """Response containing generated sound effect and metadata."""

    audio: str = Field(..., description="Base64-encoded WAV audio data")
    sample_rate: int = Field(..., description="Sample rate in Hz")
    metadata: SFXMetadata = Field(..., description="Generation metadata")


class LoadRequest(BaseModel):
    """Request to load the model."""

    device: Optional[str] = Field(
        default=None,
        description="Device to load on (cuda, cpu, mps). Auto-detected if not specified.",
    )


class LoadResponse(BaseModel):
    """Response from load operation."""

    status: str = Field(..., description="Status: loaded or already_loaded")
    model: str = Field(..., description="Model identifier")
    device: str = Field(..., description="Device model is loaded on")
    message: str = Field(..., description="Human-readable status message")


class UnloadResponse(BaseModel):
    """Response from unload operation."""

    status: str = Field(..., description="Status: unloaded or not_loaded")
    model: str = Field(..., description="Model identifier")
    message: str = Field(..., description="Human-readable status message")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Overall service status")
    model: dict = Field(default_factory=dict, description="Model status")


# =============================================================================
# Model Manager Class
# =============================================================================


class AudioLDMManager:
    """Manager for AudioLDM2 model."""

    MODEL_CONFIGS = {
        "base": {
            "model_id": "cvssp/audioldm2",
            "vram_gb": 4.0,
        },
        "large": {
            "model_id": "cvssp/audioldm2-large",
            "vram_gb": 8.0,
        },
    }

    def __init__(self, config: SFXServiceConfig):
        self.config = config
        self._pipeline: Optional[Any] = None
        self._device = self._get_device()
        self._is_loaded = False
        self._model_id = self.MODEL_CONFIGS[config.model_size]["model_id"]

    def _get_device(self) -> str:
        """Get the best available device."""
        if self.config.device:
            return self.config.device
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self, device: Optional[str] = None) -> dict:
        """Load the AudioLDM2 model."""
        if self._is_loaded:
            return {
                "status": "already_loaded",
                "device": self._device,
                "message": f"Model already loaded on {self._device}",
            }

        try:
            from diffusers import AudioLDM2Pipeline
        except ImportError:
            raise ImportError(
                "AudioLDM requires diffusers. Install with: pip install diffusers"
            )

        if device:
            self._device = device

        dtype = "float16" if self._device == "cuda" else "float32"
        logger.info(f"Loading AudioLDM2 on {self._device} with {dtype}...")

        self._pipeline = AudioLDM2Pipeline.from_pretrained(
            self._model_id,
            torch_dtype=getattr(torch, dtype),
        )

        if self._device != "cpu":
            self._pipeline = self._pipeline.to(self._device)

        self._is_loaded = True
        logger.info("AudioLDM2 loaded successfully")

        return {
            "status": "loaded",
            "device": self._device,
            "message": f"Model loaded successfully on {self._device}",
        }

    def unload(self) -> dict:
        """Unload the model and free resources."""
        if not self._is_loaded:
            return {"status": "not_loaded", "message": "Model was not loaded"}

        logger.info("Unloading AudioLDM2...")
        self._pipeline = None
        self._is_loaded = False

        import gc

        gc.collect()
        if self._device and "cuda" in self._device:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("AudioLDM2 unloaded")

        return {"status": "unloaded", "message": "Model unloaded successfully"}

    def is_weights_cached(self) -> bool:
        """Check if AudioLDM model weights are cached locally.

        Uses HuggingFace hub to check if model files exist in the cache.
        Returns:
            True if model weights are cached, False otherwise
        """
        try:
            from huggingface_hub import scan_cache_dir

            # Scan cache for this specific model
            cache_info = scan_cache_dir()
            for repo in cache_info.repos:
                if repo.repo_id == self._model_id:
                    # Check if the cache entry has files
                    if len(repo.revisions) > 0:
                        return True
            return False
        except Exception as e:
            logger.warning(f"Could not check cache status: {e}")
            return False

    def download_weights(self) -> dict:
        """Download AudioLDM model weights from HuggingFace.

        Downloads the model using huggingface_hub snapshot_download.
        Returns:
            Dict with status and message
        """
        try:
            logger.info(f"Downloading AudioLDM model: {self._model_id}...")
            start_time = time.time()

            # Download the model using huggingface_hub
            cache_path = snapshot_download(
                repo_id=self._model_id,
                repo_type="model",
            )

            elapsed = time.time() - start_time
            logger.info(
                f"AudioLDM model downloaded successfully to {cache_path} "
                f"({elapsed:.1f}s)"
            )

            return {
                "status": "downloaded",
                "cache_path": cache_path,
                "elapsed_seconds": elapsed,
                "message": f"Model downloaded successfully ({elapsed:.1f}s)",
            }
        except Exception as e:
            logger.error(f"Failed to download model weights: {e}")
            raise RuntimeError(f"Failed to download model weights: {str(e)}")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def device(self) -> str:
        return self._device

    def generate(
        self,
        prompt: str,
        duration: float,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        negative_prompt: str = "",
    ) -> tuple[np.ndarray, int]:
        """Generate sound effects from text prompt."""
        if not self._is_loaded:
            self.load()

        if self._pipeline is None:
            raise RuntimeError("AudioLDM2 pipeline not loaded")

        # AudioLDM2 works with specific durations
        duration = max(1.0, min(10.0, duration))

        logger.info(
            f"[AudioLDM2] Generating SFX: '{prompt[:50]}...' "
            f"duration={duration}s steps={num_inference_steps}"
        )

        # Generate audio using AudioLDM2
        output = self._pipeline(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            audio_length_in_s=int(duration),
        )

        # Extract audio
        audio = output.audios[0]
        sample_rate = output.sample_rate

        # Ensure correct format
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        # Normalize to [-1, 1] range
        audio = audio.astype(np.float32)
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val

        logger.info(
            f"[AudioLDM2] Generated: duration={len(audio) / sample_rate:.2f}s "
            f"sample_rate={sample_rate}"
        )

        return audio, sample_rate


# =============================================================================
# Global State
# =============================================================================

_config = SFXServiceConfig()
_sfx_manager: Optional[AudioLDMManager] = None


def get_sfx_manager() -> AudioLDMManager:
    """Get or create the AudioLDM manager singleton."""
    global _sfx_manager
    if _sfx_manager is None:
        _sfx_manager = AudioLDMManager(_config)
    return _sfx_manager


# =============================================================================
# Model Download and Loading State
# =============================================================================

_model_cache_status = {
    "cached": False,
    "download_in_progress": False,
    "download_complete": False,
    "download_error": None,
    "cache_path": None,
}

# =============================================================================
# FastAPI Application
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _model_cache_status

    logger.info("SFX Service starting up...")

    # Initialize manager and check/download model weights
    manager = get_sfx_manager()

    try:
        # Check if weights are cached
        _model_cache_status["cached"] = manager.is_weights_cached()

        if not _model_cache_status["cached"]:
            logger.info("AudioLDM model not cached. Downloading...")
            _model_cache_status["download_in_progress"] = True

            try:
                result = manager.download_weights()
                _model_cache_status["download_complete"] = True
                _model_cache_status["cache_path"] = result.get("cache_path")
                _model_cache_status["cached"] = True
                logger.info(result["message"])
            except Exception as e:
                _model_cache_status["download_error"] = str(e)
                logger.error(f"Model download failed: {e}")
        else:
            logger.info("AudioLDM model already cached. Skipping download.")
            _model_cache_status["cached"] = True
            _model_cache_status["download_complete"] = True

    except Exception as e:
        logger.error(f"Error during startup weight check: {e}")

    yield

    logger.info("SFX Service shutting down...")
    global _sfx_manager
    if _sfx_manager:
        _sfx_manager.unload()


app = FastAPI(
    title="Volsung SFX Service",
    description="Standalone sound effects generation service using AudioLDM2",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check() -> dict:
    """Check service health status including model cache status."""
    global _model_cache_status

    model_status = {
        "available": True,
        "loaded": False,
        "device": None,
        "cached": _model_cache_status["cached"],
        "download_complete": _model_cache_status["download_complete"],
        "download_in_progress": _model_cache_status["download_in_progress"],
    }

    try:
        from diffusers import AudioLDM2Pipeline

        model_status["available"] = True
        if _sfx_manager is not None:
            model_status["loaded"] = _sfx_manager.is_loaded
            model_status["device"] = (
                _sfx_manager.device if _sfx_manager.is_loaded else None
            )
    except ImportError:
        model_status["available"] = False

    # Determine overall status based on cache and availability
    if not model_status["available"]:
        overall_status = "unavailable"
    elif _model_cache_status["download_error"]:
        overall_status = "degraded"
    elif _model_cache_status["download_in_progress"]:
        overall_status = "downloading"
    elif _model_cache_status["cached"]:
        overall_status = "healthy"
    else:
        overall_status = "unavailable"

    return {
        "status": overall_status,
        "model": model_status,
    }


@app.post("/load", response_model=LoadResponse)
async def load_model(request: LoadRequest = None) -> LoadResponse:
    """Load the AudioLDM2 model.

    Args:
        request: Optional device override

    Returns:
        LoadResponse with status and device info
    """
    manager = get_sfx_manager()
    device = request.device if request else None

    try:
        result = manager.load(device)
        return LoadResponse(
            status=result["status"],
            model="AudioLDM2-Base",
            device=result["device"],
            message=result["message"],
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to load model: {str(e)}",
        )


@app.post("/unload", response_model=UnloadResponse)
async def unload_model() -> UnloadResponse:
    """Unload the AudioLDM2 model and free resources.

    Returns:
        UnloadResponse with status
    """
    manager = get_sfx_manager()
    result = manager.unload()

    return UnloadResponse(
        status=result["status"], model="AudioLDM2-Base", message=result["message"]
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate sound effects from text description.

    Args:
        request: SFX generation request with description and parameters

    Returns:
        GenerateResponse with base64-encoded audio and metadata
    """
    # Validate duration
    if request.duration > 10.0:
        raise HTTPException(status_code=400, detail="Duration must be <= 10 seconds")
    if request.duration <= 0:
        raise HTTPException(status_code=400, detail="Duration must be positive")

    manager = get_sfx_manager()

    try:
        logger.info(
            f"[SFX_GENERATE] Request: description='{request.description[:50]}...' "
            f"duration={request.duration}s category={request.category}"
        )
        start_time = time.time()

        # Generate sound effect
        audio, sample_rate = manager.generate(
            prompt=request.description,
            duration=request.duration,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
        )

        elapsed_ms = (time.time() - start_time) * 1000
        duration_seconds = len(audio) / sample_rate

        logger.info(
            f"[SFX_GENERATE] Generated: duration={duration_seconds:.2f}s "
            f"sample_rate={sample_rate} elapsed={elapsed_ms:.0f}ms"
        )

        # Convert to base64
        buffer = BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        buffer.seek(0)
        audio_b64 = base64.b64encode(buffer.read()).decode()
        audio_size_kb = len(audio_b64) * 3 // 4 // 1024
        logger.info(f"[SFX_GENERATE] Response: audio_size={audio_size_kb}KB")

        # Build metadata
        metadata = SFXMetadata(
            duration=duration_seconds,
            sample_rate=sample_rate,
            category=request.category,
            generation_time_ms=elapsed_ms,
            model_used="cvssp/audioldm2",
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
        )

        return GenerateResponse(
            audio=audio_b64,
            sample_rate=sample_rate,
            metadata=metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SFX_GENERATE] Failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SFX generation failed: {str(e)}",
        )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    config = SFXServiceConfig()
    logger.info(f"Starting SFX Service on {config.host}:{config.port}")
    uvicorn.run(
        "volsung.services.sfx_service:app",
        host=config.host,
        port=config.port,
        log_level="info",
    )
