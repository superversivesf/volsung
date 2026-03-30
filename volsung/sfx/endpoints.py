"""
FastAPI router for SFX (Sound Effects) generation endpoints.

Provides REST API endpoints for:
- POST /sfx/generate - Generate sound effects from text
- POST /sfx/layer - Generate layered/combined sound effects
- GET /sfx/health - Health check for SFX module
"""

import logging
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, status

from volsung.models.types import AudioResult, AudioType

from .schemas import (
    SFXGenerateRequest,
    SFXGenerateResponse,
    SFXHealthResponse,
    SFXLayerRequest,
    SFXLayerResponse,
    SFXMetadata,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/sfx", tags=["sfx"])

# Module-level manager instance (set during initialization)
_sfx_manager = None


def set_manager(manager) -> None:
    """Set the SFX manager instance for the router.

    This is called during application startup to inject the
    SFX model manager dependency.

    Args:
        manager: SFXModelManager instance
    """
    global _sfx_manager
    _sfx_manager = manager
    logger.info("SFX router manager set")


def _audio_array_to_base64(audio: np.ndarray, sample_rate: int) -> str:
    """Convert numpy audio array to base64-encoded WAV.

    Args:
        audio: Audio samples as numpy array
        sample_rate: Sample rate in Hz

    Returns:
        Base64-encoded WAV string
    """
    import base64
    from io import BytesIO

    import soundfile as sf

    buffer = BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()


def _combine_audio_layers(layers: list) -> tuple[np.ndarray, int]:
    """Combine multiple audio layers into a single track.

    Args:
        layers: List of (audio_array, sample_rate) tuples

    Returns:
        Tuple of (combined_audio, sample_rate)
    """
    if not layers:
        raise ValueError("No layers to combine")

    # Find max length and use first sample rate
    max_length = max(len(layer[0]) for layer in layers)
    sample_rate = layers[0][1]

    # Initialize output array
    combined = np.zeros(max_length, dtype=np.float32)

    # Mix all layers together
    for audio, _ in layers:
        # Pad if shorter
        if len(audio) < max_length:
            audio = np.pad(audio, (0, max_length - len(audio)))
        combined += audio

    # Normalize to prevent clipping
    max_val = np.max(np.abs(combined))
    if max_val > 1.0:
        combined = combined / max_val

    return combined.astype(np.float32), sample_rate


@router.post(
    "/generate",
    response_model=SFXGenerateResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate sound effects",
    description="Generate sound effects from text description using AudioLDM",
)
async def sfx_generate(request: SFXGenerateRequest) -> SFXGenerateResponse:
    """Generate sound effects from text description.

    Creates sound effects up to 10 seconds from natural language description.
    Uses AudioLDM for high-quality sound effect generation.

    Args:
        request: SFX generation request with description and parameters

    Returns:
        Generated audio with metadata

    Raises:
        HTTPException: If model not loaded or generation fails
    """
    if _sfx_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SFX module not initialized",
        )

    # Validate duration
    if request.duration > 10.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Duration must be <= 10 seconds",
        )
    if request.duration <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Duration must be positive",
        )

    try:
        logger.info(
            f"[SFX_GENERATE] Request: description='{request.description[:50]}...' "
            f"duration={request.duration}s category={request.category}"
        )
        start_time = time.time()

        # Generate audio using manager
        result: AudioResult = _sfx_manager.generate(
            prompt=request.description,
            duration=request.duration,
            category=request.category,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
        )

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"[SFX_GENERATE] Generated: duration={result.duration:.2f}s "
            f"sample_rate={result.sample_rate} elapsed={elapsed_ms:.0f}ms"
        )

        # Build metadata
        metadata = SFXMetadata(
            duration=result.duration,
            sample_rate=result.sample_rate,
            category=request.category,
            generation_time_ms=elapsed_ms,
            model_used=result.generator,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
        )

        return SFXGenerateResponse(
            audio=result.audio,
            sample_rate=result.sample_rate,
            metadata=metadata,
        )

    except Exception as e:
        logger.error(f"[SFX_GENERATE] Failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SFX generation failed: {str(e)}",
        )


@router.post(
    "/layer",
    response_model=SFXLayerResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate layered SFX",
    description="Generate multiple sound effects and layer them together",
)
async def sfx_layer(request: SFXLayerRequest) -> SFXLayerResponse:
    """Generate combined/layered sound effects.

    Creates multiple sound effects and layers them together into a single track.
    Useful for complex audio scenes like "thunder + rain + wind".

    Args:
        request: Layered SFX request with multiple generation requests

    Returns:
        Combined audio with metadata for each layer

    Raises:
        HTTPException: If model not loaded or generation fails
    """
    if _sfx_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SFX module not initialized",
        )

    # Validate layers
    if not request.layers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one layer required",
        )
    if len(request.layers) > 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 5 layers allowed",
        )

    for i, layer in enumerate(request.layers):
        if layer.duration > 10.0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Layer {i + 1}: Duration must be <= 10 seconds",
            )
        if layer.duration <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Layer {i + 1}: Duration must be positive",
            )

    try:
        logger.info(f"[SFX_LAYER] Request: {len(request.layers)} layers")
        start_time = time.time()

        # Generate each layer
        generated_layers = []
        layers_metadata = []
        max_duration = 0.0

        for i, layer_req in enumerate(request.layers):
            layer_start = time.time()

            result: AudioResult = _sfx_manager.generate(
                prompt=layer_req.description,
                duration=layer_req.duration,
                category=layer_req.category,
                num_inference_steps=layer_req.num_inference_steps,
                guidance_scale=layer_req.guidance_scale,
            )

            # Decode base64 audio back to array for mixing
            import base64
            from io import BytesIO

            import soundfile as sf

            audio_bytes = base64.b64decode(result.audio)
            audio_array, sr = sf.read(BytesIO(audio_bytes))

            layer_elapsed_ms = (time.time() - layer_start) * 1000

            generated_layers.append((audio_array, sr))
            layers_metadata.append(
                SFXMetadata(
                    duration=result.duration,
                    sample_rate=result.sample_rate,
                    category=layer_req.category,
                    generation_time_ms=layer_elapsed_ms,
                    model_used=result.generator,
                    num_inference_steps=layer_req.num_inference_steps,
                    guidance_scale=layer_req.guidance_scale,
                )
            )
            max_duration = max(max_duration, result.duration)

            logger.info(
                f"[SFX_LAYER] Layer {i + 1}: {result.duration:.2f}s "
                f"'{layer_req.description[:30]}...'"
            )

        # Combine layers
        combined_audio, sample_rate = _combine_audio_layers(generated_layers)

        elapsed_ms = (time.time() - start_time) * 1000

        # Encode combined audio to base64
        audio_base64 = _audio_array_to_base64(combined_audio, sample_rate)
        audio_size_kb = len(audio_base64) * 3 // 4 // 1024

        logger.info(
            f"[SFX_LAYER] Combined: duration={max_duration:.2f}s "
            f"elapsed={elapsed_ms:.0f}ms size={audio_size_kb}KB"
        )

        return SFXLayerResponse(
            audio=audio_base64,
            sample_rate=sample_rate,
            layers_metadata=layers_metadata,
            total_duration=max_duration,
        )

    except Exception as e:
        logger.error(f"[SFX_LAYER] Failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SFX layering failed: {str(e)}",
        )


@router.get(
    "/health",
    response_model=SFXHealthResponse,
    status_code=status.HTTP_200_OK,
    summary="SFX health check",
    description="Check SFX module status and model load state",
)
async def sfx_health() -> SFXHealthResponse:
    """Check SFX module health status.

    Returns:
        Health status including model load state
    """
    if _sfx_manager is None:
        return SFXHealthResponse(
            status="uninitialized",
            model_loaded=False,
            model_name="none",
            idle_seconds=None,
        )

    model_loaded = _sfx_manager.is_loaded
    idle_seconds = _sfx_manager.idle_seconds if model_loaded else None

    return SFXHealthResponse(
        status="healthy" if model_loaded else "unloaded",
        model_loaded=model_loaded,
        model_name=_sfx_manager.config.model_name,
        idle_seconds=idle_seconds,
    )
