"""StyleTTS2 Service for Volsung.

A standalone FastAPI service that provides StyleTTS2 voice generation endpoints.
Runs on port 8003 and handles voice synthesis using StyleTTS2.

This service is isolated from Qwen-TTS to avoid dependency conflicts:
- transformers<4.40
- huggingface-hub<0.20

Example:
    # Start the service
    python -m volsung.services.styletts_service

    # Or using uvicorn directly
    uvicorn volsung.services.styletts_service:app --host 0.0.0.0 --port 8003

Environment Variables:
    STYLETTS_SERVICE_HOST: Server bind address (default: 0.0.0.0)
    STYLETTS_SERVICE_PORT: Server port (default: 8003)
    STYLETTS_DEVICE: Device override (default: auto-detected)
    STYLETTS_DTYPE: Data type (default: bfloat16 for CUDA, float32 otherwise)
"""

from __future__ import annotations

import base64
import logging
import os
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Literal, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, status
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


class StyleTTSConfig(BaseModel):
    """StyleTTS2 service configuration."""

    host: str = Field(
        default_factory=lambda: os.getenv("STYLETTS_SERVICE_HOST", "0.0.0.0")
    )
    port: int = Field(
        default_factory=lambda: int(os.getenv("STYLETTS_SERVICE_PORT", "8003"))
    )
    device: Optional[str] = Field(default_factory=lambda: os.getenv("STYLETTS_DEVICE"))
    dtype: Optional[str] = Field(default_factory=lambda: os.getenv("STYLETTS_DTYPE"))


# =============================================================================
# Pydantic Schemas
# =============================================================================


class StyleTTSParams(BaseModel):
    """Parameters for StyleTTS 2 voice generation."""

    embedding_scale: float = Field(
        default=1.0,
        ge=1.0,
        le=10.0,
        description="Emotion intensity (1.0-10.0)",
    )
    alpha: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Diffusion alpha parameter",
    )
    beta: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Diffusion beta parameter",
    )


class GenerateRequest(BaseModel):
    """Request to generate speech from text."""

    text: str = Field(
        ...,
        description="Text to synthesize",
        examples=["Hello, I am speaking with StyleTTS2."],
    )
    language: str = Field(
        default="English",
        description="Language for phonemization",
        examples=["English"],
    )
    styletts_params: Optional[StyleTTSParams] = Field(
        default=None,
        description="StyleTTS 2-specific parameters",
    )


class GenerateResponse(BaseModel):
    """Response containing generated audio."""

    audio: str = Field(..., description="Base64-encoded WAV audio data")
    sample_rate: int = Field(default=24000, description="Audio sample rate in Hz")


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


class StyleTTS2Manager:
    """Manager for StyleTTS 2 model."""

    def __init__(self, config: StyleTTSConfig):
        self.config = config
        self._generator: Optional[object] = None
        self._device = self._get_device()
        self._dtype = self._get_dtype()
        self._is_loaded = False

    def _get_device(self) -> str:
        """Get the best available device."""
        if self.config.device:
            return self.config.device
        if torch.cuda.is_available():
            return "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _get_dtype(self) -> str:
        """Get the optimal dtype for the device."""
        if self.config.dtype:
            return self.config.dtype
        if torch.cuda.is_available():
            return "bfloat16"
        return "float32"

    def load(self, device: Optional[str] = None) -> dict:
        """Load StyleTTS 2 model."""
        if self._is_loaded:
            return {
                "status": "already_loaded",
                "device": self._device,
                "message": f"Model already loaded on {self._device}",
            }

        try:
            from styletts2 import tts
        except ImportError:
            raise ImportError(
                "styletts2 is required. Install with: pip install styletts2"
            )

        if device:
            self._device = device

        logger.info(f"Loading StyleTTS 2 on {self._device} with {self._dtype}...")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._generator = tts.StyleTTS2()
        if hasattr(self._generator, "to"):
            self._generator.to(self._device)

        self._is_loaded = True
        logger.info("StyleTTS 2 loaded successfully")

        return {
            "status": "loaded",
            "device": self._device,
            "message": f"Model loaded successfully on {self._device}",
        }

    def unload(self) -> dict:
        """Unload model and free resources."""
        if not self._is_loaded:
            return {"status": "not_loaded", "message": "Model was not loaded"}

        logger.info("Unloading StyleTTS 2...")
        self._generator = None
        self._is_loaded = False

        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("StyleTTS 2 unloaded")

        return {"status": "unloaded", "message": "Model unloaded successfully"}

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def device(self) -> str:
        return self._device

    def generate(
        self,
        text: str,
        embedding_scale: float = 1.0,
        alpha: float = 0.3,
        beta: float = 0.7,
    ) -> GenerateResponse:
        """Generate speech with StyleTTS 2."""
        if not self._is_loaded:
            self.load()

        if self._generator is None:
            raise RuntimeError("StyleTTS 2 generator not loaded")

        logger.info(f"[STYLETTS2] text='{text[:50]}...'")

        try:
            with torch.no_grad():
                wav = self._generator.inference(
                    text=text,
                    target_voice_path=None,
                    embedding_scale=embedding_scale,
                    alpha=alpha,
                    beta=beta,
                )

            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()

            if wav.ndim > 1:
                wav = wav.mean(axis=0)

            if wav.max() > 1.0 or wav.min() < -1.0:
                wav = wav / max(abs(wav.max()), abs(wav.min()))

            sr = 24000
            audio_b64 = self._audio_to_base64(wav, sr)

            duration = len(wav) / sr
            logger.info(f"[STYLETTS2] Generated: {duration:.2f}s @ {sr}Hz")

            return GenerateResponse(audio=audio_b64, sample_rate=sr)

        except Exception as e:
            logger.error(f"[STYLETTS2] Failed: {e}", exc_info=True)
            raise RuntimeError(f"StyleTTS 2 generation failed: {e}")

    def _audio_to_base64(self, audio: np.ndarray, sample_rate: int) -> str:
        """Convert audio array to base64-encoded WAV."""
        buffer = BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode()


# =============================================================================
# Global State
# =============================================================================

_config = StyleTTSConfig()
_styletts2_manager: Optional[StyleTTS2Manager] = None


def get_styletts2_manager() -> StyleTTS2Manager:
    """Get or create the StyleTTS 2 manager singleton."""
    global _styletts2_manager
    if _styletts2_manager is None:
        _styletts2_manager = StyleTTS2Manager(_config)
    return _styletts2_manager


# =============================================================================
# FastAPI Application
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    logger.info("StyleTTS2 Service starting up...")
    yield
    # Cleanup on shutdown
    logger.info("StyleTTS2 Service shutting down...")
    global _styletts2_manager
    if _styletts2_manager:
        _styletts2_manager.unload()


app = FastAPI(
    title="Volsung StyleTTS2 Service",
    description="Standalone StyleTTS2 service for voice synthesis",
    version="0.1.0",
    lifespan=lifespan,
)


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check() -> dict:
    """Check service health status.

    Returns:
        Dictionary with status and model availability
    """
    model_status = {
        "available": True,
        "loaded": False,
        "device": None,
    }

    try:
        import styletts2

        model_status["available"] = True
        if _styletts2_manager is not None:
            model_status["loaded"] = _styletts2_manager.is_loaded
            model_status["device"] = (
                _styletts2_manager.device if _styletts2_manager.is_loaded else None
            )
    except ImportError:
        model_status["available"] = False

    overall_status = "healthy" if model_status["available"] else "unavailable"

    return {
        "status": overall_status,
        "model": model_status,
    }


@app.post("/load", response_model=LoadResponse)
async def load_model(request: LoadRequest = None) -> LoadResponse:
    """Load the StyleTTS2 model.

    Args:
        request: Optional device override

    Returns:
        LoadResponse with status and device info
    """
    manager = get_styletts2_manager()
    device = request.device if request else None

    try:
        result = manager.load(device)
        return LoadResponse(
            status=result["status"],
            model="StyleTTS2",
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
    """Unload the StyleTTS2 model and free resources.

    Returns:
        UnloadResponse with status
    """
    manager = get_styletts2_manager()
    result = manager.unload()

    return UnloadResponse(
        status=result["status"], model="StyleTTS2", message=result["message"]
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate speech using StyleTTS2.

    Creates synthesized speech from text using StyleTTS2.

    Args:
        request: Generation parameters

    Returns:
        GenerateResponse with base64-encoded audio
    """
    manager = get_styletts2_manager()
    params = request.styletts_params or StyleTTSParams()
    try:
        return manager.generate(
            text=request.text,
            embedding_scale=params.embedding_scale,
            alpha=params.alpha,
            beta=params.beta,
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {e}",
        )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    config = StyleTTSConfig()
    logger.info(f"Starting StyleTTS2 Service on {config.host}:{config.port}")
    uvicorn.run(
        "volsung.services.styletts_service:app",
        host=config.host,
        port=config.port,
        log_level="info",
    )
