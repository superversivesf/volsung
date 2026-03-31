"""Standalone Music Service using FastAPI and MusicGen.

A standalone microservice for music generation using Facebook's MusicGen model
via audiocraft. Runs on port 8002 and provides a simple HTTP API for
generating music from text descriptions.

This service is completely independent from the main Volsung server and
only imports music-related modules.

Example:
    Start the service:
        ./scripts/start-music.sh
    
    Generate music:
        curl -X POST http://localhost:8002/music/generate \\
            -H "Content-Type: application/json" \\
            -d '{"description": "Peaceful acoustic guitar", "duration": 10}'

Attributes:
    app: FastAPI application instance
    generator: MusicGen generator instance (lazy-loaded)

References:
    - https://github.com/facebookresearch/audiocraft
"""

import base64
import io
import logging
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

# Module-level state
_generator: Optional[Any] = None
_device: Optional[str] = None
_dtype: Optional[str] = None


# ============================================================================
# Pydantic Schemas
# ============================================================================


class MusicGenerateRequest(BaseModel):
    """Request for generating music from text description.

    Attributes:
        description: Natural language description of desired music
        duration: Target duration in seconds (1-30, default: 10)
        genre: Optional genre tag (e.g., "acoustic", "electronic")
        mood: Optional mood descriptor (e.g., "upbeat", "calm")
        tempo: Optional tempo hint ("slow", "medium", "fast")
        top_k: Top-k sampling parameter (higher = more diverse)
        top_p: Nucleus sampling parameter (0.0 = disabled)
        temperature: Randomness/temperature parameter
    """

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


class MusicGenerateResponse(BaseModel):
    """Response containing generated music and metadata."""

    audio: str = Field(..., description="Base64-encoded WAV audio data")
    sample_rate: int = Field(..., description="Sample rate in Hz")
    metadata: MusicMetadata = Field(..., description="Generation metadata")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    model: str = Field(..., description="Model name")
    loaded: bool = Field(..., description="Whether model is loaded")
    device: Optional[str] = Field(None, description="Device model is on")


class InfoResponse(BaseModel):
    """Service information response."""

    status: str = Field(..., description="Service status")
    model: str = Field(..., description="Model name")
    model_id: str = Field(..., description="Model identifier")
    loaded: bool = Field(..., description="Whether model is loaded")
    device: Optional[str] = Field(None, description="Device model is on")
    dtype: Optional[str] = Field(None, description="Data type used")
    version: str = Field(default="1.0.0", description="Service version")


# ============================================================================
# MusicGen Integration
# ============================================================================


def _get_device() -> str:
    """Determine the best device for model inference."""
    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_dtype(device: str) -> str:
    """Determine the best dtype for the device."""
    if device.startswith("cuda"):
        return "bfloat16"
    return "float32"


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


def load_model() -> None:
    """Load the MusicGen model.

    Called automatically on first generation request.
    """
    global _generator, _device, _dtype

    if _generator is not None:
        return

    try:
        from audiocraft.models import MusicGen
    except ImportError:
        raise ImportError(
            "audiocraft is required for MusicGen. Install with: pip install audiocraft"
        )

    _device = _get_device()
    _dtype = _get_dtype(_device)

    logger.info(f"Loading MusicGen model (facebook/musicgen-small)...")
    logger.info(f"Device: {_device}, Dtype: {_dtype}")

    try:
        _generator = MusicGen.get_pretrained("musicgen-small")
        _generator.set_generation_params(
            duration=10.0,
            top_k=250,
            top_p=0.0,
            temperature=1.0,
            use_sampling=True,
        )
        logger.info("MusicGen model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load MusicGen: {e}")
        raise


def unload_model() -> None:
    """Unload the model and free resources."""
    global _generator, _device, _dtype

    if _generator is not None:
        logger.info("Unloading MusicGen model...")
        del _generator
        _generator = None

        # Clear GPU cache
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        _device = None
        _dtype = None
        logger.info("MusicGen model unloaded")


def generate_music(
    prompt: str,
    duration: float,
    top_k: int = 250,
    top_p: float = 0.0,
    temperature: float = 1.0,
) -> Tuple[np.ndarray, int]:
    """Generate music from text prompt.

    Args:
        prompt: Text description of desired music
        duration: Target duration in seconds
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        temperature: Randomness/temperature parameter

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    global _generator

    if _generator is None:
        load_model()

    logger.info(f"[MusicGen] Generating {duration}s: '{prompt[:50]}...'")

    # Set generation parameters
    _generator.set_generation_params(
        duration=duration,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        use_sampling=True,
    )

    # Generate
    with torch.no_grad():
        output = _generator.generate(
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


# ============================================================================
# FastAPI Application
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Music Service starting up...")
    # Pre-load model on startup (optional - can also lazy-load)
    # load_model()  # Uncomment to load at startup
    yield
    logger.info("Music Service shutting down...")
    unload_model()


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


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns:
        HealthResponse with service status
    """
    return HealthResponse(
        status="healthy",
        model="MusicGen Small",
        loaded=_generator is not None,
        device=_device,
    )


@app.get("/info", response_model=InfoResponse)
async def info() -> InfoResponse:
    """Service information endpoint.

    Returns:
        InfoResponse with service metadata
    """
    return InfoResponse(
        status="ready",
        model="MusicGen Small",
        model_id="musicgen-small",
        loaded=_generator is not None,
        device=_device,
        dtype=_dtype,
    )


@app.post("/music/generate", response_model=MusicGenerateResponse)
async def music_generate(request: MusicGenerateRequest) -> MusicGenerateResponse:
    """Generate music from text description.

    Creates music up to 30 seconds from a natural language description.
    Useful for background music, ambience, etc.

    Args:
        request: Music generation request with description and parameters

    Returns:
        MusicGenerateResponse with base64-encoded audio and metadata

    Raises:
        HTTPException: If generation fails

    Example:
        ```python
        import requests

        response = requests.post(
            "http://localhost:8002/music/generate",
            json={
                "description": "Peaceful acoustic guitar",
                "duration": 10.0,
            }
        )
        result = response.json()
        # result["audio"] contains base64-encoded WAV
        ```
    """
    try:
        start_time = time.time()

        # Build enhanced prompt
        enhanced_prompt = _build_prompt(
            request.description,
            request.genre,
            request.mood,
            request.tempo,
        )

        # Generate music
        audio_array, sample_rate = generate_music(
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

        return MusicGenerateResponse(
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


@app.post("/music/unload")
async def music_unload() -> Dict[str, Any]:
    """Force unload the music model.

    Frees GPU memory by unloading the model immediately.
    Model will be reloaded on next generation request.

    Returns:
        Dictionary with unload status
    """
    was_loaded = _generator is not None
    unload_model()

    return {
        "unloaded": was_loaded,
        "status": "unloaded" if was_loaded else "already_unloaded",
    }


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "volsung.services.music_service:app",
        host="0.0.0.0",
        port=8002,
        log_level="info",
        reload=False,
    )
