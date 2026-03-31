"""Standalone TTS Service for Volsung.

A lightweight FastAPI service that provides only TTS (Text-to-Speech) endpoints.
Runs on port 8001 and handles voice design and voice synthesis using Qwen3-TTS and StyleTTS 2.

This service is designed to be run independently from the main Volsung server,
allowing TTS workloads to be scaled separately.

Example:
    # Start the service
    python -m volsung.services.tts_service

    # Or using uvicorn directly
    uvicorn volsung.services.tts_service:app --host 0.0.0.0 --port 8001

Environment Variables:
    TTS_SERVICE_HOST: Server bind address (default: 0.0.0.0)
    TTS_SERVICE_PORT: Server port (default: 8001)
    TTS_IDLE_TIMEOUT: Seconds before unloading models (default: 300)
    TTS_DEVICE: Device override (default: auto-detected)
    TTS_DTYPE: Data type (default: bfloat16 for CUDA, float32 otherwise)
"""

from __future__ import annotations

import base64
import logging
import os
import sys
from contextlib import asynccontextmanager
from io import BytesIO
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
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


class TTSConfig(BaseModel):
    """TTS service configuration."""

    host: str = Field(default_factory=lambda: os.getenv("TTS_SERVICE_HOST", "0.0.0.0"))
    port: int = Field(
        default_factory=lambda: int(os.getenv("TTS_SERVICE_PORT", "8001"))
    )
    idle_timeout: int = Field(
        default_factory=lambda: int(os.getenv("TTS_IDLE_TIMEOUT", "300"))
    )
    voice_design_model: str = Field(default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    base_model: str = Field(default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    device: Optional[str] = Field(default_factory=lambda: os.getenv("TTS_DEVICE"))
    dtype: Optional[str] = Field(default_factory=lambda: os.getenv("TTS_DTYPE"))


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
    diffusion_steps: int = Field(
        default=10,
        ge=3,
        le=20,
        description="Number of diffusion steps",
    )


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
    backend: Literal["qwen3", "styletts2"] = Field(
        default="qwen3",
        description="TTS backend to use",
    )
    styletts_params: Optional[StyleTTSParams] = Field(
        default=None,
        description="StyleTTS 2-specific parameters (only used when backend='styletts2')",
    )


class VoiceDesignResponse(BaseModel):
    """Response containing generated audio."""

    audio: str = Field(..., description="Base64-encoded WAV audio data")
    sample_rate: int = Field(default=24000, description="Audio sample rate in Hz")


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


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Overall service status")
    qwen3: dict = Field(default_factory=dict, description="Qwen3-TTS status")
    styletts2: dict = Field(default_factory=dict, description="StyleTTS 2 status")


# =============================================================================
# Model Manager Classes
# =============================================================================


class Qwen3TTSManager:
    """Manager for Qwen3-TTS models."""

    def __init__(self, config: TTSConfig):
        self.config = config
        self._voice_design_model: Optional[torch.nn.Module] = None
        self._base_model: Optional[torch.nn.Module] = None
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

    def load(self) -> None:
        """Load both TTS models."""
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError:
            raise ImportError(
                "qwen_tts package is required. Install with: pip install qwen-tts"
            )

        logger.info(f"Loading Qwen3-TTS models on {self._device} with {self._dtype}...")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load VoiceDesign model
        logger.info(f"Loading VoiceDesign model: {self.config.voice_design_model}")
        self._voice_design_model = Qwen3TTSModel.from_pretrained(
            self.config.voice_design_model,
            device_map=self._device,
            dtype=getattr(torch, self._dtype),
        )
        logger.info("VoiceDesign model loaded successfully")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load Base model
        logger.info(f"Loading Base model: {self.config.base_model}")
        self._base_model = Qwen3TTSModel.from_pretrained(
            self.config.base_model,
            device_map=self._device,
            dtype=getattr(torch, self._dtype),
        )
        logger.info("Base model loaded successfully")
        self._is_loaded = True

    def unload(self) -> None:
        """Unload models and free resources."""
        logger.info("Unloading Qwen3-TTS models...")
        self._voice_design_model = None
        self._base_model = None
        self._is_loaded = False

        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("Qwen3-TTS models unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def voice_design(self, request: VoiceDesignRequest) -> VoiceDesignResponse:
        """Generate voice from description."""
        if not self._is_loaded:
            self.load()

        if self._voice_design_model is None:
            raise RuntimeError("VoiceDesign model not loaded")

        logger.info(
            f"[VOICE_DESIGN] text='{request.text[:50]}...' language='{request.language}'"
        )

        try:
            wavs, sr = self._voice_design_model.generate_voice_design(
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

    def synthesize(self, request: SynthesizeRequest) -> SynthesizeResponse:
        """Synthesize text using cloned voice."""
        if not self._is_loaded:
            self.load()

        if self._base_model is None:
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
            wavs, sr = self._base_model.generate_voice_clone(
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


class StyleTTS2Manager:
    """Manager for StyleTTS 2 models."""

    def __init__(self, config: TTSConfig):
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

    def load(self) -> None:
        """Load StyleTTS 2 model."""
        try:
            from styletts2 import tts
        except ImportError:
            raise ImportError(
                "styletts2 is required. Install with: pip install styletts2"
            )

        logger.info(f"Loading StyleTTS 2 on {self._device} with {self._dtype}...")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._generator = tts.StyleTTS2()
        if hasattr(self._generator, "to"):
            self._generator.to(self._device)

        self._is_loaded = True
        logger.info("StyleTTS 2 loaded successfully")

    def unload(self) -> None:
        """Unload model and free resources."""
        logger.info("Unloading StyleTTS 2...")
        self._generator = None
        self._is_loaded = False

        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("StyleTTS 2 unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def generate(
        self,
        text: str,
        embedding_scale: float = 1.0,
        alpha: float = 0.3,
        beta: float = 0.7,
        diffusion_steps: int = 10,
    ) -> VoiceDesignResponse:
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
                    target_voice_path=None,  # Use default/reference voice
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

            return VoiceDesignResponse(audio=audio_b64, sample_rate=sr)

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

_config = TTSConfig()
_qwen3_manager: Optional[Qwen3TTSManager] = None
_styletts2_manager: Optional[StyleTTS2Manager] = None


def get_qwen3_manager() -> Qwen3TTSManager:
    """Get or create the Qwen3-TTS manager singleton."""
    global _qwen3_manager
    if _qwen3_manager is None:
        _qwen3_manager = Qwen3TTSManager(_config)
    return _qwen3_manager


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
    logger.info("TTS Service starting up...")
    yield
    # Cleanup on shutdown
    logger.info("TTS Service shutting down...")
    global _qwen3_manager, _styletts2_manager
    if _qwen3_manager:
        _qwen3_manager.unload()
    if _styletts2_manager:
        _styletts2_manager.unload()


app = FastAPI(
    title="Volsung TTS Service",
    description="Standalone Text-to-Speech service using Qwen3-TTS and StyleTTS 2",
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
    qwen3_status = {
        "available": True,
        "loaded": False,
    }
    styletts2_status = {
        "available": True,
        "loaded": False,
    }

    # Check Qwen3
    try:
        from qwen_tts import Qwen3TTSModel

        qwen3_status["available"] = True
        if _qwen3_manager is not None:
            qwen3_status["loaded"] = _qwen3_manager.is_loaded
    except ImportError:
        qwen3_status["available"] = False

    # Check StyleTTS2
    try:
        import styletts2

        styletts2_status["available"] = True
        if _styletts2_manager is not None:
            styletts2_status["loaded"] = _styletts2_manager.is_loaded
    except ImportError:
        styletts2_status["available"] = False

    overall_status = (
        "healthy"
        if (qwen3_status["available"] or styletts2_status["available"])
        else "unavailable"
    )

    return {
        "status": overall_status,
        "qwen3": qwen3_status,
        "styletts2": styletts2_status,
    }


@app.post("/voice/design", response_model=VoiceDesignResponse)
async def voice_design(request: VoiceDesignRequest) -> VoiceDesignResponse:
    """Generate voice from natural language description.

    Creates a unique voice character based on the instruction and
    generates audio for the provided text in that voice.

    Args:
        request: Voice design parameters

    Returns:
        VoiceDesignResponse with base64-encoded audio
    """
    if request.backend == "qwen3":
        manager = get_qwen3_manager()
        try:
            return manager.voice_design(request)
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

    elif request.backend == "styletts2":
        manager = get_styletts2_manager()
        params = request.styletts_params or StyleTTSParams()
        try:
            return manager.generate(
                text=request.text,
                embedding_scale=params.embedding_scale,
                alpha=params.alpha,
                beta=params.beta,
                diffusion_steps=params.diffusion_steps,
            )
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

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown backend: {request.backend}. Use 'qwen3' or 'styletts2'.",
        )


@app.post("/voice/synthesize", response_model=SynthesizeResponse)
async def synthesize(request: SynthesizeRequest) -> SynthesizeResponse:
    """Synthesize text using a cloned voice.

    Uses reference audio (typically from voice_design) to clone a voice
    and synthesize new text in that voice character.

    Args:
        request: Synthesis parameters

    Returns:
        SynthesizeResponse with base64-encoded audio
    """
    # Currently only Qwen3 supports voice cloning/synthesis
    manager = get_qwen3_manager()

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

    config = TTSConfig()
    logger.info(f"Starting TTS Service on {config.host}:{config.port}")
    uvicorn.run(
        "volsung.services.tts_service:app",
        host=config.host,
        port=config.port,
        log_level="info",
    )
