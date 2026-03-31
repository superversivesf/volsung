"""MusicGen Service for Volsung.

A standalone FastAPI service that provides MusicGen music generation endpoints.
Runs on port 8004 and handles music generation from text descriptions.

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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    model_id: str = Field(default="facebook/musicgen-small")
    device: Optional[str] = Field(default_factory=lambda: os.getenv("MUSIC_DEVICE"))


# =============================================================================
# Pydantic Schemas
# =============================================================================


class GenerateRequest(BaseModel):
    """Request for generating music from text description."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "description": "Peaceful acoustic guitar for meditation",
                "duration": 10.0,
                "genre": "acoustic",
                "mood": "calm",
                "tempo": "slow",
            }
        }
    )

    description: str = Field(
        ...,
        description="Natural language description of desired music",
        min_length=1,
        max_length=1000,
    )
    duration: float = Field(
        default=10.0,
        ge=1.0,
        le=30.0,
        description="Target duration in seconds (1-30)",
    )
    genre: Optional[str] = Field(
        default=None,
        description="Optional genre tag (acoustic, electronic, classical, etc.)",
    )
    mood: Optional[str] = Field(
        default=None,
        description="Optional mood descriptor (upbeat, calm, tense, etc.)",
    )
    tempo: Optional[str] = Field(
        default=None,
        description="Tempo hint: slow, medium, or fast",
    )
    top_k: int = Field(
        default=250,
        ge=1,
        le=1000,
        description="Top-k sampling (higher = more diverse)",
    )
    top_p: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling (0.0 = disabled)",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.1,
        le=2.0,
        description="Randomness/temperature (higher = more random)",
    )

    @field_validator("tempo")
    @classmethod
    def validate_tempo(cls, v: Optional[str]) -> Optional[str]:
        """Validate tempo is one of the allowed values."""
        if v is None:
            return v
        allowed = {"slow", "medium", "fast"}
        if v.lower() not in allowed:
            raise ValueError(f"Tempo must be one of: {allowed}")
        return v.lower()


class MusicMetadata(BaseModel):
    """Metadata for generated music."""

    duration: float = Field(..., description="Actual duration in seconds")
    sample_rate: int = Field(..., description="Audio sample rate in Hz")
    genre_tags: List[str] = Field(
        default_factory=list,
        description="Genre and mood tags applied to generation",
    )
    generation_time_ms: float = Field(
        ...,
        description="Time taken to generate in milliseconds",
    )
    model_used: str = Field(..., description="Model identifier used")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Generation parameters used",
    )


class GenerateResponse(BaseModel):
    """Response containing generated music and metadata."""

    audio: str = Field(..., description="Base64-encoded WAV audio data")
    sample_rate: int = Field(..., description="Sample rate in Hz")
    metadata: MusicMetadata = Field(..., description="Generation metadata")


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


class MusicGenManager:
    """Manager for MusicGen model."""

    def __init__(self, config: MusicServiceConfig):
        self.config = config
        self._generator: Optional[Any] = None
        self._device = self._get_device()
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
        if self._device.startswith("cuda"):
            return "bfloat16"
        return "float32"

    def load(self, device: Optional[str] = None) -> dict:
        """Load the MusicGen model."""
        if self._is_loaded:
            return {
                "status": "already_loaded",
                "device": self._device,
                "message": f"Model already loaded on {self._device}",
            }

        try:
            from audiocraft.models import MusicGen
        except ImportError:
            raise ImportError(
                "audiocraft is required for MusicGen. Install with: pip install audiocraft"
            )

        if device:
            self._device = device

        dtype = self._get_dtype()
        logger.info(f"Loading MusicGen on {self._device} with {dtype}...")

        try:
            self._generator = MusicGen.get_pretrained("musicgen-small")
            self._generator.set_generation_params(
                duration=10.0,
                top_k=250,
                top_p=0.0,
                temperature=1.0,
                use_sampling=True,
            )
            self._is_loaded = True
            logger.info("MusicGen model loaded successfully")

            return {
                "status": "loaded",
                "device": self._device,
                "message": f"Model loaded successfully on {self._device}",
            }
        except Exception as e:
            logger.error(f"Failed to load MusicGen: {e}")
            raise RuntimeError(f"Failed to load MusicGen: {e}")

    def unload(self) -> dict:
        """Unload the model and free resources."""
        if not self._is_loaded:
            return {"status": "not_loaded", "message": "Model was not loaded"}

        logger.info("Unloading MusicGen model...")
        self._generator = None
        self._is_loaded = False

        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("MusicGen model unloaded")

        return {"status": "unloaded", "message": "Model unloaded successfully"}

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
        top_k: int = 250,
        top_p: float = 0.0,
        temperature: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """Generate music from text prompt."""
        if not self._is_loaded:
            self.load()

        if self._generator is None:
            raise RuntimeError("MusicGen generator not loaded")

        logger.info(f"[MusicGen] Generating {duration}s: '{prompt[:50]}...'")

        # Set generation parameters
        self._generator.set_generation_params(
            duration=duration,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            use_sampling=True,
        )

        # Generate
        with torch.no_grad():
            output = self._generator.generate(
                descriptions=[prompt],
                progress=False,
            )

        # Convert to numpy array
        audio = output[0, 0].cpu().numpy()
        sample_rate = 32000

        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=0)

        logger.info(f"[MusicGen] Generated: {len(audio)} samples @ {sample_rate}Hz")

        return audio, sample_rate


# =============================================================================
# Global State
# =============================================================================

_config = MusicServiceConfig()
_music_manager: Optional[MusicGenManager] = None


def get_music_manager() -> MusicGenManager:
    """Get or create the MusicGen manager singleton."""
    global _music_manager
    if _music_manager is None:
        _music_manager = MusicGenManager(_config)
    return _music_manager


def _build_prompt(
    base_prompt: str,
    genre: Optional[str] = None,
    mood: Optional[str] = None,
    tempo: Optional[str] = None,
) -> str:
    """Build enhanced prompt from base prompt and tags."""
    parts = [base_prompt]

    if genre:
        parts.append(f"Genre: {genre}")
    if mood:
        parts.append(f"Mood: {mood}")
    if tempo:
        parts.append(f"Tempo: {tempo}")

    return ", ".join(parts)


# =============================================================================
# FastAPI Application
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Music Service starting up...")
    yield
    logger.info("Music Service shutting down...")
    global _music_manager
    if _music_manager:
        _music_manager.unload()


app = FastAPI(
    title="Volsung Music Service",
    description="Standalone music generation service using MusicGen",
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
    """Check service health status."""
    model_status = {
        "available": True,
        "loaded": False,
        "device": None,
    }

    try:
        from audiocraft.models import MusicGen

        model_status["available"] = True
        if _music_manager is not None:
            model_status["loaded"] = _music_manager.is_loaded
            model_status["device"] = (
                _music_manager.device if _music_manager.is_loaded else None
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
    """Load the MusicGen model.

    Args:
        request: Optional device override

    Returns:
        LoadResponse with status and device info
    """
    manager = get_music_manager()
    device = request.device if request else None

    try:
        result = manager.load(device)
        return LoadResponse(
            status=result["status"],
            model="MusicGen-Small",
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
    """Unload the MusicGen model and free resources.

    Returns:
        UnloadResponse with status
    """
    manager = get_music_manager()
    result = manager.unload()

    return UnloadResponse(
        status=result["status"], model="MusicGen-Small", message=result["message"]
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate music from text description.

    Args:
        request: Music generation request with description and parameters

    Returns:
        GenerateResponse with base64-encoded audio and metadata
    """
    try:
        start_time = time.time()
        manager = get_music_manager()

        # Build enhanced prompt
        enhanced_prompt = _build_prompt(
            request.description,
            request.genre,
            request.mood,
            request.tempo,
        )

        # Generate music
        audio_array, sample_rate = manager.generate(
            prompt=enhanced_prompt,
            duration=request.duration,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature,
        )

        # Convert to base64 WAV
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, sample_rate, format="WAV")
        buffer.seek(0)
        audio_b64 = base64.b64encode(buffer.read()).decode()

        actual_duration = len(audio_array) / sample_rate
        elapsed_ms = (time.time() - start_time) * 1000

        # Build metadata
        genre_tags = []
        if request.genre:
            genre_tags.append(request.genre)
        if request.mood:
            genre_tags.append(request.mood)
        if request.tempo:
            genre_tags.append(request.tempo)

        metadata = MusicMetadata(
            duration=actual_duration,
            sample_rate=sample_rate,
            genre_tags=genre_tags,
            generation_time_ms=elapsed_ms,
            model_used="facebook/musicgen-small",
            parameters={
                "top_k": request.top_k,
                "top_p": request.top_p,
                "temperature": request.temperature,
            },
        )

        return GenerateResponse(
            audio=audio_b64,
            sample_rate=sample_rate,
            metadata=metadata,
        )

    except ValueError as e:
        logger.warning(f"[MUSIC] Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
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
