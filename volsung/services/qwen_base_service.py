"""Qwen Base Service for Volsung.

A standalone FastAPI service that provides Qwen3-TTS Base endpoint only.
Runs on port 8002 and handles voice cloning/synthesis.

This service is isolated from Qwen VoiceDesign and other models to avoid dependency conflicts:
- transformers==4.57.3
- huggingface-hub>=0.34.0

Example:
    # Start the service
    python -m volsung.services.qwen_base_service

    # Or using uvicorn directly
    uvicorn volsung.services.qwen_base_service:app --host 0.0.0.0 --port 8002

Environment Variables:
    QWEN_BASE_SERVICE_HOST: Server bind address (default: 0.0.0.0)
    QWEN_BASE_SERVICE_PORT: Server port (default: 8002)
    QWEN_BASE_DEVICE: Device override (default: auto-detected)
    QWEN_BASE_DTYPE: Data type (default: bfloat16 for CUDA, float32 otherwise)
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


class QwenBaseConfig(BaseModel):
    """Qwen Base service configuration."""

    host: str = Field(
        default_factory=lambda: os.getenv("QWEN_BASE_SERVICE_HOST", "0.0.0.0")
    )
    port: int = Field(
        default_factory=lambda: int(os.getenv("QWEN_BASE_SERVICE_PORT", "8002"))
    )
    model_id: str = Field(default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    device: Optional[str] = Field(default_factory=lambda: os.getenv("QWEN_BASE_DEVICE"))
    dtype: Optional[str] = Field(default_factory=lambda: os.getenv("QWEN_BASE_DTYPE"))


# =============================================================================
# Pydantic Schemas
# =============================================================================


class SynthesizeRequest(BaseModel):
    """Request to synthesize text with a cloned voice."""

    ref_audio: str = Field(
        ...,
        description="Base64-encoded reference WAV audio",
    )
    ref_text: str = Field(
        ...,
        description="Transcript of the reference audio",
        examples=["Hello, I am John. Nice to meet you."],
    )
    text: str = Field(
        ...,
        description="New text to synthesize in the cloned voice",
        examples=["The quick brown fox jumps over the lazy dog."],
    )
    language: str = Field(
        default="English",
        description="Language code",
        examples=["English"],
    )


class SynthesizeResponse(BaseModel):
    """Response containing synthesized audio."""

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


class QwenBaseManager:
    """Manager for Qwen3-TTS Base model (voice cloning)."""

    def __init__(self, config: QwenBaseConfig):
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
        """Load the Base model."""
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

        logger.info(f"Loading Base model on {self._device} with {self._dtype}...")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._model = Qwen3TTSModel.from_pretrained(
            self.config.model_id,
            device_map=self._device,
            dtype=getattr(torch, self._dtype),
        )
        self._is_loaded = True

        logger.info("Base model loaded successfully")

        return {
            "status": "loaded",
            "device": self._device,
            "message": f"Model loaded successfully on {self._device}",
        }

    def unload(self) -> dict:
        """Unload model and free resources."""
        if not self._is_loaded:
            return {"status": "not_loaded", "message": "Model was not loaded"}

        logger.info("Unloading Base model...")
        self._model = None
        self._is_loaded = False

        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("Base model unloaded")

        return {"status": "unloaded", "message": "Model unloaded successfully"}

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def device(self) -> str:
        return self._device

    def synthesize(self, request: SynthesizeRequest) -> SynthesizeResponse:
        """Synthesize text using cloned voice."""
        if not self._is_loaded:
            self.load()

        if self._model is None:
            raise RuntimeError("Base model not loaded")

        logger.info(f"[SYNTHESIZE] text='{request.text[:50]}...'")

        try:
            # Decode reference audio
            audio_bytes = base64.b64decode(request.ref_audio)
            buffer = BytesIO(audio_bytes)
            ref_audio, ref_sr = sf.read(buffer)

            ref_duration = len(ref_audio) / ref_sr
            logger.info(f"[SYNTHESIZE] Reference: {ref_duration:.2f}s @ {ref_sr}Hz")

            # Generate cloned voice
            wavs, sr = self._model.generate_voice_clone(
                ref_audio=(ref_audio, ref_sr),
                ref_text=request.ref_text,
                text=request.text,
                language=request.language,
            )

            audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
            audio_b64 = self._audio_to_base64(audio, sr)

            duration = len(audio) / sr
            logger.info(f"[SYNTHESIZE] Generated: {duration:.2f}s @ {sr}Hz")

            return SynthesizeResponse(audio=audio_b64, sample_rate=sr)

        except Exception as e:
            logger.error(f"[SYNTHESIZE] Failed: {e}", exc_info=True)
            raise RuntimeError(f"Synthesis failed: {e}")

    def _audio_to_base64(self, audio: np.ndarray, sample_rate: int) -> str:
        """Convert audio array to base64-encoded WAV."""
        buffer = BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode()


# =============================================================================
# Global State
# =============================================================================

_config = QwenBaseConfig()
_base_manager: Optional[QwenBaseManager] = None


def get_base_manager() -> QwenBaseManager:
    """Get or create the Base manager singleton."""
    global _base_manager
    if _base_manager is None:
        _base_manager = QwenBaseManager(_config)
    return _base_manager


# =============================================================================
# FastAPI Application
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    logger.info("Qwen Base Service starting up...")
    yield
    # Cleanup on shutdown
    logger.info("Qwen Base Service shutting down...")
    global _base_manager
    if _base_manager:
        _base_manager.unload()


app = FastAPI(
    title="Volsung Qwen Base Service",
    description="Standalone Qwen3-TTS Base service for voice cloning",
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
        if _base_manager is not None:
            model_status["loaded"] = _base_manager.is_loaded
            model_status["device"] = (
                _base_manager.device if _base_manager.is_loaded else None
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
    """Load the Base model.

    Args:
        request: Optional device override

    Returns:
        LoadResponse with status and device info
    """
    manager = get_base_manager()
    device = request.device if request else None

    try:
        result = manager.load(device)
        return LoadResponse(
            status=result["status"],
            model="Qwen3-TTS-Base",
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
    """Unload the Base model and free resources.

    Returns:
        UnloadResponse with status
    """
    manager = get_base_manager()
    result = manager.unload()

    return UnloadResponse(
        status=result["status"], model="Qwen3-TTS-Base", message=result["message"]
    )


@app.post("/generate", response_model=SynthesizeResponse)
async def generate(request: SynthesizeRequest) -> SynthesizeResponse:
    """Synthesize text using a cloned voice.

    Uses reference audio to clone a voice and synthesize new text
    in that voice character.

    Args:
        request: Synthesis parameters

    Returns:
        SynthesizeResponse with base64-encoded audio
    """
    manager = get_base_manager()
    try:
        return manager.synthesize(request)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Synthesis failed: {e}",
        )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    config = QwenBaseConfig()
    logger.info(f"Starting Qwen Base Service on {config.host}:{config.port}")
    uvicorn.run(
        "volsung.services.qwen_base_service:app",
        host=config.host,
        port=config.port,
        log_level="info",
    )
