"""Qwen VoiceDesign Service for Volsung.

A standalone FastAPI service that provides Qwen3-TTS VoiceDesign endpoint only.
Runs on port 8001 and handles voice design generation.

This service is isolated from Qwen Base and other models to avoid dependency conflicts:
- transformers==4.57.3
- huggingface-hub>=0.34.0

Example:
    # Start the service
    python -m volsung.services.qwen_voice_service

    # Or using uvicorn directly
    uvicorn volsung.services.qwen_voice_service:app --host 0.0.0.0 --port 8001

Environment Variables:
    QWEN_VOICE_SERVICE_HOST: Server bind address (default: 0.0.0.0)
    QWEN_VOICE_SERVICE_PORT: Server port (default: 8001)
    QWEN_VOICE_DEVICE: Device override (default: auto-detected)
    QWEN_VOICE_DTYPE: Data type (default: bfloat16 for CUDA, float32 otherwise)
"""

from __future__ import annotations

import base64
import logging
import os
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Optional

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


class QwenVoiceConfig(BaseModel):
    """Qwen VoiceDesign service configuration."""

    host: str = Field(
        default_factory=lambda: os.getenv("QWEN_VOICE_SERVICE_HOST", "0.0.0.0")
    )
    port: int = Field(
        default_factory=lambda: int(os.getenv("QWEN_VOICE_SERVICE_PORT", "8001"))
    )
    model_id: str = Field(default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    device: Optional[str] = Field(
        default_factory=lambda: os.getenv("QWEN_VOICE_DEVICE")
    )
    dtype: Optional[str] = Field(default_factory=lambda: os.getenv("QWEN_VOICE_DTYPE"))


# =============================================================================
# Pydantic Schemas
# =============================================================================


class VoiceDesignRequest(BaseModel):
    """Request to generate a voice sample from a description."""

    text: str = Field(
        ...,
        description="Sample text to speak (e.g., 'Hello, I am John.')",
        examples=["Hello, I am John. Nice to meet you."],
    )
    language: str = Field(
        default="English",
        description="Language: English, Chinese, Japanese, Korean, German, French, etc.",
        examples=["English"],
    )
    instruct: str = Field(
        ...,
        description="Natural language voice description",
        examples=["A warm, elderly man's voice with a slight Southern accent"],
    )


class VoiceDesignResponse(BaseModel):
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


class QwenVoiceManager:
    """Manager for Qwen3-TTS VoiceDesign model."""

    def __init__(self, config: QwenVoiceConfig):
        self.config = config
        self._model: Optional[torch.nn.Module] = None
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
        """Load the VoiceDesign model."""
        if self._is_loaded:
            return {
                "status": "already_loaded",
                "device": self._device,
                "message": f"Model already loaded on {self._device}",
            }

        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError:
            raise ImportError(
                "qwen_tts package is required. Install with: pip install qwen-tts"
            )

        if device:
            self._device = device

        logger.info(
            f"Loading VoiceDesign model on {self._device} with {self._dtype}..."
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._model = Qwen3TTSModel.from_pretrained(
            self.config.model_id,
            device_map=self._device,
            dtype=getattr(torch, self._dtype),
        )
        self._is_loaded = True

        logger.info("VoiceDesign model loaded successfully")

        return {
            "status": "loaded",
            "device": self._device,
            "message": f"Model loaded successfully on {self._device}",
        }

    def unload(self) -> dict:
        """Unload model and free resources."""
        if not self._is_loaded:
            return {"status": "not_loaded", "message": "Model was not loaded"}

        logger.info("Unloading VoiceDesign model...")
        self._model = None
        self._is_loaded = False

        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("VoiceDesign model unloaded")

        return {"status": "unloaded", "message": "Model unloaded successfully"}

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def device(self) -> str:
        return self._device

    def generate(self, request: VoiceDesignRequest) -> VoiceDesignResponse:
        """Generate voice from description."""
        if not self._is_loaded:
            self.load()

        if self._model is None:
            raise RuntimeError("VoiceDesign model not loaded")

        logger.info(
            f"[VOICE_DESIGN] text='{request.text[:50]}...' language='{request.language}'"
        )

        try:
            wavs, sr = self._model.generate_voice_design(
                text=request.text,
                language=request.language,
                instruct=request.instruct,
            )

            audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
            audio_b64 = self._audio_to_base64(audio, sr)

            duration = len(audio) / sr
            logger.info(f"[VOICE_DESIGN] Generated: {duration:.2f}s @ {sr}Hz")

            return VoiceDesignResponse(audio=audio_b64, sample_rate=sr)

        except Exception as e:
            logger.error(f"[VOICE_DESIGN] Failed: {e}", exc_info=True)
            raise RuntimeError(f"Voice design failed: {e}")

    def _audio_to_base64(self, audio: np.ndarray, sample_rate: int) -> str:
        """Convert audio array to base64-encoded WAV."""
        buffer = BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode()


# =============================================================================
# Global State
# =============================================================================

_config = QwenVoiceConfig()
_voice_manager: Optional[QwenVoiceManager] = None


def get_voice_manager() -> QwenVoiceManager:
    """Get or create the VoiceDesign manager singleton."""
    global _voice_manager
    if _voice_manager is None:
        _voice_manager = QwenVoiceManager(_config)
    return _voice_manager


# =============================================================================
# FastAPI Application
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    logger.info("Qwen VoiceDesign Service starting up...")
    yield
    # Cleanup on shutdown
    logger.info("Qwen VoiceDesign Service shutting down...")
    global _voice_manager
    if _voice_manager:
        _voice_manager.unload()


app = FastAPI(
    title="Volsung Qwen VoiceDesign Service",
    description="Standalone Qwen3-TTS VoiceDesign service for voice design",
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
        from qwen_tts import Qwen3TTSModel

        model_status["available"] = True
        if _voice_manager is not None:
            model_status["loaded"] = _voice_manager.is_loaded
            model_status["device"] = (
                _voice_manager.device if _voice_manager.is_loaded else None
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
    """Load the VoiceDesign model.

    Args:
        request: Optional device override

    Returns:
        LoadResponse with status and device info
    """
    manager = get_voice_manager()
    device = request.device if request else None

    try:
        result = manager.load(device)
        return LoadResponse(
            status=result["status"],
            model="Qwen3-TTS-VoiceDesign",
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
    """Unload the VoiceDesign model and free resources.

    Returns:
        UnloadResponse with status
    """
    manager = get_voice_manager()
    result = manager.unload()

    return UnloadResponse(
        status=result["status"],
        model="Qwen3-TTS-VoiceDesign",
        message=result["message"],
    )


@app.post("/generate", response_model=VoiceDesignResponse)
async def generate(request: VoiceDesignRequest) -> VoiceDesignResponse:
    """Generate voice from natural language description.

    Creates a unique voice character based on the instruction and
    generates audio for the provided text in that voice.

    Args:
        request: Voice design parameters

    Returns:
        VoiceDesignResponse with base64-encoded audio
    """
    manager = get_voice_manager()
    try:
        return manager.generate(request)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice design failed: {e}",
        )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    config = QwenVoiceConfig()
    logger.info(f"Starting Qwen VoiceDesign Service on {config.host}:{config.port}")
    uvicorn.run(
        "volsung.services.qwen_voice_service:app",
        host=config.host,
        port=config.port,
        log_level="info",
    )
