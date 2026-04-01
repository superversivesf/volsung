"""Chatterbox TTS Service for Volsung.

A standalone FastAPI service that provides Chatterbox voice generation endpoints.
Runs on port 8007 and handles voice synthesis with emotion control via
exaggeration and cfg_weight parameters.

Example:
    python -m volsung.services.chatterbox_service
    uvicorn volsung.services.chatterbox_service:app --host 0.0.0.0 --port 8007

Environment Variables:
    CHATTERBOX_SERVICE_HOST: Server bind address (default: 0.0.0.0)
    CHATTERBOX_SERVICE_PORT: Server port (default: 8007)
    CHATTERBOX_DEVICE: Device override (default: auto-detected)
"""

from __future__ import annotations

import base64
import gc
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class ChatterboxConfig(BaseModel):
    host: str = Field(
        default_factory=lambda: os.getenv("CHATTERBOX_SERVICE_HOST", "0.0.0.0")
    )
    port: int = Field(
        default_factory=lambda: int(os.getenv("CHATTERBOX_SERVICE_PORT", "8007"))
    )
    device: Optional[str] = Field(
        default_factory=lambda: os.getenv("CHATTERBOX_DEVICE")
    )


# =============================================================================
# Pydantic Schemas
# =============================================================================


class ChatterboxParams(BaseModel):
    exaggeration: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Expressiveness (0.0=monotone, 1.0=very expressive)",
    )
    cfg_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Classifier-free guidance weight for pacing",
    )


class GenerateRequest(BaseModel):
    text: str = Field(
        ...,
        description="Text to synthesize",
        examples=["Hello, this is a test of Chatterbox TTS."],
    )
    ref_audio: Optional[str] = Field(
        default=None,
        description="Base64-encoded reference audio (WAV) for voice cloning. ~5-10s recommended.",
    )
    chatterbox_params: Optional[ChatterboxParams] = Field(
        default=None,
        description="Chatterbox-specific parameters",
    )


class GenerateResponse(BaseModel):
    audio: str = Field(..., description="Base64-encoded WAV audio data")
    sample_rate: int = Field(default=24000, description="Audio sample rate in Hz")


class LoadRequest(BaseModel):
    device: Optional[str] = Field(
        default=None,
        description="Device to load on (cuda, cpu). Auto-detected if not specified.",
    )


class LoadResponse(BaseModel):
    status: str = Field(..., description="Status: loaded or already_loaded")
    model: str = Field(..., description="Model identifier")
    device: str = Field(..., description="Device model is loaded on")
    message: str = Field(..., description="Human-readable status message")


class UnloadResponse(BaseModel):
    status: str = Field(..., description="Status: unloaded or not_loaded")
    model: str = Field(..., description="Model identifier")
    message: str = Field(..., description="Human-readable status message")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall service status")
    model: dict = Field(default_factory=dict, description="Model status")


# =============================================================================
# Model Manager
# =============================================================================


class ChatterboxManager:
    def __init__(self, config: ChatterboxConfig):
        self.config = config
        self._generator = None
        self._device = self._get_device()
        self._is_loaded = False
        self._sample_rate = 24000

    def _get_device(self) -> str:
        if self.config.device:
            return self.config.device
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load(self, device: Optional[str] = None) -> dict:
        if self._is_loaded:
            return {
                "status": "already_loaded",
                "device": self._device,
                "message": f"Model already loaded on {self._device}",
            }

        try:
            from chatterbox.tts import ChatterboxTTS
        except ImportError:
            raise ImportError(
                "chatterbox-tts is required. Install with: pip install chatterbox-tts"
            )

        if device:
            self._device = device

        logger.info(f"Loading Chatterbox on {self._device}...")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._generator = ChatterboxTTS.from_pretrained(device=self._device)
        self._sample_rate = self._generator.sr
        self._is_loaded = True
        logger.info(f"Chatterbox loaded successfully (sr={self._sample_rate})")

        return {
            "status": "loaded",
            "device": self._device,
            "message": f"Model loaded successfully on {self._device}",
        }

    def unload(self) -> dict:
        if not self._is_loaded:
            return {"status": "not_loaded", "message": "Model was not loaded"}

        logger.info("Unloading Chatterbox...")
        self._generator = None
        self._is_loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("Chatterbox unloaded")
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
        ref_audio_b64: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
    ):
        """Generate speech, optionally cloning from reference audio."""
        if not self._is_loaded:
            self.load()

        if self._generator is None:
            raise RuntimeError("Chatterbox not loaded")

        logger.info(
            f"[CHATTERBOX] text='{text[:50]}...', exaggeration={exaggeration}, "
            f"cfg_weight={cfg_weight}, cloning={ref_audio_b64 is not None}"
        )

        tmp_path = None
        try:
            audio_prompt_path = None
            if ref_audio_b64:
                ref_bytes = base64.b64decode(ref_audio_b64)
                ref_data, ref_sr = sf.read(BytesIO(ref_bytes))
                logger.info(
                    f"[CHATTERBOX] Reference audio: {len(ref_data)} samples @ {ref_sr}Hz"
                )
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(tmp.name, ref_data, ref_sr, format="WAV")
                tmp_path = tmp.name
                tmp.close()
                audio_prompt_path = tmp_path

            with torch.no_grad():
                wav = self._generator.generate(
                    text,
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                )

            if isinstance(wav, torch.Tensor):
                wav = wav.squeeze().cpu().numpy()

            if wav.ndim > 1:
                wav = wav.mean(axis=0)

            if wav.max() > 1.0 or wav.min() < -1.0:
                wav = wav / max(abs(wav.max()), abs(wav.min()))

            sr = self._sample_rate
            duration = len(wav) / sr
            logger.info(f"[CHATTERBOX] Generated: {duration:.2f}s @ {sr}Hz")

            return wav, sr

        except Exception as e:
            logger.error(f"[CHATTERBOX] Failed: {e}", exc_info=True)
            raise RuntimeError(f"Chatterbox generation failed: {e}")
        finally:
            if tmp_path:
                os.unlink(tmp_path)


# =============================================================================
# Global State
# =============================================================================

_config = ChatterboxConfig()
_manager: Optional[ChatterboxManager] = None


def get_manager() -> ChatterboxManager:
    global _manager
    if _manager is None:
        _manager = ChatterboxManager(_config)
    return _manager


# =============================================================================
# FastAPI Application
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Chatterbox Service starting up...")
    yield
    logger.info("Chatterbox Service shutting down...")
    global _manager
    if _manager:
        _manager.unload()


app = FastAPI(
    title="Volsung Chatterbox Service",
    description="Chatterbox TTS service with voice cloning and emotion control",
    version="0.1.0",
    lifespan=lifespan,
)


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check() -> dict:
    model_status = {
        "available": True,
        "loaded": False,
        "device": None,
    }

    try:
        import chatterbox  # noqa: F401

        model_status["available"] = True
        if _manager is not None:
            model_status["loaded"] = _manager.is_loaded
            model_status["device"] = _manager.device if _manager.is_loaded else None
    except ImportError:
        model_status["available"] = False

    overall_status = "healthy" if model_status["available"] else "unavailable"
    return {"status": overall_status, "model": model_status}


@app.post("/load", response_model=LoadResponse)
async def load_model(request: LoadRequest = None) -> LoadResponse:
    manager = get_manager()
    device = request.device if request else None
    try:
        result = manager.load(device)
        return LoadResponse(
            status=result["status"],
            model="Chatterbox",
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
    manager = get_manager()
    result = manager.unload()
    return UnloadResponse(
        status=result["status"], model="Chatterbox", message=result["message"]
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    manager = get_manager()
    params = request.chatterbox_params or ChatterboxParams()
    try:
        wav, sr = manager.generate(
            text=request.text,
            ref_audio_b64=request.ref_audio,
            exaggeration=params.exaggeration,
            cfg_weight=params.cfg_weight,
        )

        buffer = BytesIO()
        sf.write(buffer, wav, sr, format="WAV")
        buffer.seek(0)
        audio_b64 = base64.b64encode(buffer.read()).decode()

        return GenerateResponse(audio=audio_b64, sample_rate=sr)
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

    config = ChatterboxConfig()
    logger.info(f"Starting Chatterbox Service on {config.host}:{config.port}")
    uvicorn.run(
        "volsung.services.chatterbox_service:app",
        host=config.host,
        port=config.port,
        log_level="info",
    )
