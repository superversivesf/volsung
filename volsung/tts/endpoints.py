"""FastAPI router for TTS (Text-to-Speech) endpoints.

Provides REST API endpoints for voice design and voice synthesis.
Supports both Qwen3-TTS and StyleTTS 2 backends.
"""

import logging

from fastapi import APIRouter, HTTPException, status

from ..config import get_config
from .manager import TTSModelManager
from .schemas import (
    SynthesizeRequest,
    SynthesizeResponse,
    VoiceDesignRequest,
    VoiceDesignResponse,
)

# Import StyleTTS2Manager - may not exist yet, handle gracefully
try:
    from .managers.styletts2 import StyleTTS2Manager

    HAS_STYLETTS2 = True
except ImportError:
    StyleTTS2Manager = None  # type: ignore[misc, assignment]
    HAS_STYLETTS2 = False

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/voice", tags=["voice"])

# Global manager instances (lazy-loaded)
_qwen3_manager: TTSModelManager | None = None
_styletts2_manager: "StyleTTS2Manager | None" = None


def get_qwen3_manager() -> TTSModelManager:
    """Get or create the Qwen3-TTS manager singleton.

    Returns:
        TTSModelManager instance
    """
    global _qwen3_manager
    if _qwen3_manager is None:
        config = get_config()
        _qwen3_manager = TTSModelManager(
            voice_design_model_id=config.tts.voice_design_model,
            base_model_id=config.tts.base_model,
            idle_timeout=config.tts.idle_timeout,
        )
    return _qwen3_manager


def get_styletts2_manager() -> "StyleTTS2Manager":
    """Get or create the StyleTTS 2 manager singleton.

    Returns:
        StyleTTS2Manager instance

    Raises:
        HTTPException: If StyleTTS 2 is not available
    """
    global _styletts2_manager
    if not HAS_STYLETTS2:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="StyleTTS 2 backend not available. Please install styletts2 package.",
        )
    if _styletts2_manager is None:
        config = get_config()
        _styletts2_manager = StyleTTS2Manager(
            idle_timeout=config.tts.idle_timeout,
        )
    return _styletts2_manager


def reset_managers() -> None:
    """Reset all TTS managers (useful for testing).

    Forces recreation of all managers on next use.
    """
    global _qwen3_manager, _styletts2_manager

    if _qwen3_manager is not None:
        _qwen3_manager.shutdown()
    _qwen3_manager = None

    if _styletts2_manager is not None:
        _styletts2_manager.shutdown()
    _styletts2_manager = None


# Keep backward compatibility
reset_tts_manager = reset_managers


# =============================================================================
# Voice Design Endpoint
# =============================================================================


@router.post(
    "/design",
    response_model=VoiceDesignResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate voice from description",
    description="""
    Generate a voice sample from a natural language description.

    Uses the selected backend (Qwen3-TTS or StyleTTS 2) to create a unique
    voice character based on the instruction. The output should be stored
    (audio + transcript) for later use in synthesis.

    Example workflow:
    1. POST /voice/design with text and instruct
    2. Store the returned audio (base64 decode to WAV)
    3. Use that audio as ref_audio in /voice/synthesize

    **Backends:**
    - `qwen3` (default): High-quality voice design with natural prosody
    - `styletts2`: StyleTTS 2 with controllable emotion via embedding_scale

    **StyleTTS 2 Parameters:**
    - `embedding_scale`: Emotion intensity (1.0-10.0, default: 1.0)
    - `alpha`: Diffusion alpha (0.0-1.0, default: 0.3)
    - `beta`: Diffusion beta (0.0-1.0, default: 0.7)
    - `diffusion_steps`: Quality vs speed tradeoff (3-20, default: 10)
    """,
    responses={
        200: {
            "description": "Voice generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "audio": "base64encodedwavdata...",
                        "sample_rate": 24000,
                    }
                }
            },
        },
        503: {
            "description": "Model not available",
            "content": {
                "application/json": {
                    "example": {"detail": "VoiceDesign model not loaded"}
                }
            },
        },
        500: {
            "description": "Generation failed",
            "content": {
                "application/json": {"example": {"detail": "Voice design failed"}}
            },
        },
    },
)
async def voice_design(request: VoiceDesignRequest) -> VoiceDesignResponse:
    """Generate voice sample from natural language description.

    Args:
        request: Voice design parameters including text, voice description,
                 backend choice, and StyleTTS-specific parameters

    Returns:
        VoiceDesignResponse with base64-encoded audio

    Raises:
        HTTPException: If model not loaded or generation fails
    """
    # Route to appropriate backend
    if request.backend == "qwen3":
        manager = get_qwen3_manager()

        try:
            result = manager.voice_design(request)
            audio_base64 = manager.audio_to_base64(result)

            return VoiceDesignResponse(
                audio=audio_base64,
                sample_rate=result.sample_rate,
            )

        except RuntimeError as e:
            logger.error(f"Voice design failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e),
            )
        except Exception as e:
            logger.error(f"Unexpected error in voice design: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Voice design failed: {e}",
            )

    elif request.backend == "styletts2":
        if not HAS_STYLETTS2:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="StyleTTS 2 backend not available. Please install styletts2 package.",
            )

        manager = get_styletts2_manager()

        try:
            # Get StyleTTS parameters with defaults
            params = request.styletts_params
            embedding_scale = params.embedding_scale if params else 1.0
            alpha = params.alpha if params else 0.3
            beta = params.beta if params else 0.7
            diffusion_steps = params.diffusion_steps if params else 10

            # For StyleTTS 2, we need reference audio - generate a default voice
            # or use a predefined reference. The text is synthesized using StyleTTS 2.
            # Note: StyleTTS 2 requires reference audio for voice cloning.
            # We'll use the generate method which handles voice design internally.
            result = manager.generate(
                text=request.text,
                ref_audio=None,  # Will use default/reference voice
                embedding_scale=embedding_scale,
                alpha=alpha,
                beta=beta,
                diffusion_steps=diffusion_steps,
            )
            audio_base64 = result.audio

            return VoiceDesignResponse(
                audio=audio_base64,
                sample_rate=result.sample_rate,
            )

        except RuntimeError as e:
            logger.error(f"Voice design failed (StyleTTS 2): {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e),
            )
        except Exception as e:
            logger.error(
                f"Unexpected error in voice design (StyleTTS 2): {e}", exc_info=True
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Voice design failed: {e}",
            )

    else:
        # This should not happen due to Literal validation, but handle anyway
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown backend: {request.backend}. Use 'qwen3' or 'styletts2'.",
        )


# =============================================================================
# StyleTTS 2 Specific Voice Design Endpoint
# =============================================================================


@router.post(
    "/styletts/design",
    response_model=VoiceDesignResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate voice using StyleTTS 2",
    description="""
    Generate a voice sample using StyleTTS 2 with explicit parameter control.

    This endpoint provides direct access to StyleTTS 2's voice generation
    with fine-grained control over style and emotion parameters.

    **Parameters:**
    - `embedding_scale`: Emotion intensity (1.0-10.0, default: 1.0)
    - `alpha`: Diffusion alpha (0.0-1.0, default: 0.3)
    - `beta`: Diffusion beta (0.0-1.0, default: 0.7)
    - `diffusion_steps`: Quality vs speed tradeoff (3-20, default: 10)

    The returned audio can be used as reference audio for synthesis.
    """,
    responses={
        200: {
            "description": "Voice generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "audio": "base64encodedwavdata...",
                        "sample_rate": 24000,
                    }
                }
            },
        },
        503: {
            "description": "StyleTTS 2 not available",
            "content": {
                "application/json": {
                    "example": {"detail": "StyleTTS 2 backend not available"}
                }
            },
        },
    },
)
async def styletts_voice_design(request: VoiceDesignRequest) -> VoiceDesignResponse:
    """Generate voice sample using StyleTTS 2.

    Args:
        request: Voice design parameters with StyleTTS-specific options

    Returns:
        VoiceDesignResponse with base64-encoded audio

    Raises:
        HTTPException: If StyleTTS 2 is not available or generation fails
    """
    if not HAS_STYLETTS2:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="StyleTTS 2 backend not available. Please install styletts2 package.",
        )

    manager = get_styletts2_manager()

    try:
        # Get StyleTTS parameters with defaults
        params = request.styletts_params
        embedding_scale = params.embedding_scale if params else 1.0
        alpha = params.alpha if params else 0.3
        beta = params.beta if params else 0.7
        diffusion_steps = params.diffusion_steps if params else 10

        # Generate voice using StyleTTS 2
        result = manager.generate(
            text=request.text,
            ref_audio=None,  # Will use default/reference voice
            embedding_scale=embedding_scale,
            alpha=alpha,
            beta=beta,
            diffusion_steps=diffusion_steps,
        )
        audio_base64 = result.audio

        return VoiceDesignResponse(
            audio=audio_base64,
            sample_rate=result.sample_rate,
        )

    except RuntimeError as e:
        logger.error(f"Voice design failed (StyleTTS 2): {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Unexpected error in voice design (StyleTTS 2): {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice design failed: {e}",
        )


# =============================================================================
# Synthesis Endpoint
# =============================================================================


@router.post(
    "/synthesize",
    response_model=SynthesizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Synthesize text with cloned voice",
    description="""
    Synthesize text using a cloned voice from reference audio.

    Uses the Base model to clone the voice from reference audio and
    synthesize new text in that voice character.

    Example workflow:
    1. Get reference audio from /voice/design
    2. Call this endpoint with ref_audio, ref_text, and new text
    3. Receive synthesized audio in the cloned voice

    The ref_audio should be base64-encoded WAV from /voice/design.
    The ref_text must be the transcript of that reference audio.
    """,
    responses={
        200: {
            "description": "Audio synthesized successfully",
            "content": {
                "application/json": {
                    "example": {
                        "audio": "base64encodedwavdata...",
                        "sample_rate": 24000,
                    }
                }
            },
        },
        503: {
            "description": "Model not available",
            "content": {
                "application/json": {"example": {"detail": "Base model not loaded"}}
            },
        },
        500: {
            "description": "Synthesis failed",
            "content": {
                "application/json": {"example": {"detail": "Synthesis failed"}}
            },
        },
    },
)
async def synthesize(request: SynthesizeRequest) -> SynthesizeResponse:
    """Synthesize text using cloned voice from reference audio.

    Args:
        request: Synthesis parameters including reference audio and text

    Returns:
        SynthesizeResponse with base64-encoded audio

    Raises:
        HTTPException: If model not loaded or synthesis fails
    """
    # Currently only Qwen3 supports voice cloning/synthesis
    manager = get_qwen3_manager()

    try:
        result = manager.synthesize(request)
        audio_base64 = manager.audio_to_base64(result)

        return SynthesizeResponse(
            audio=audio_base64,
            sample_rate=result.sample_rate,
        )

    except RuntimeError as e:
        logger.error(f"Synthesis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error in synthesis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Synthesis failed: {e}",
        )


# =============================================================================
# Health Check Endpoint (TTS-specific)
# =============================================================================


@router.post("/unload")
async def voice_unload() -> dict:
    """Force unload all voice models.

    Frees GPU memory by unloading VoiceDesign and Base models.
    Models will be reloaded on next request.

    Returns:
        Dictionary with unload status
    """
    global _qwen3_manager, _styletts2_manager

    results = {}

    if _qwen3_manager is not None:
        was_loaded = _qwen3_manager.is_loaded
        _qwen3_manager.force_unload()
        results["qwen3"] = {"unloaded": was_loaded}
    else:
        results["qwen3"] = {"unloaded": False, "reason": "not_initialized"}

    if _styletts2_manager is not None:
        was_loaded = _styletts2_manager.is_loaded
        _styletts2_manager.force_unload()
        results["styletts2"] = {"unloaded": was_loaded}
    else:
        results["styletts2"] = {"unloaded": False, "reason": "not_initialized"}

    return {"unloaded": True, "results": results}


@router.get(
    "/health",
    summary="Check TTS health status",
    description="Check if TTS models are loaded and available.",
    responses={
        200: {
            "description": "Health status",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "qwen3": {
                            "voice_design_loaded": True,
                            "base_model_loaded": True,
                        },
                        "styletts2": {
                            "available": True,
                            "loaded": False,
                        },
                    }
                }
            },
        }
    },
)
async def tts_health() -> dict:
    """Check TTS model health status.

    Returns:
        Dictionary with status and model load states
    """
    qwen3_status = {
        "voice_design_loaded": False,
        "base_model_loaded": False,
    }

    styletts2_status = {
        "available": HAS_STYLETTS2,
        "loaded": False,
    }

    # Check Qwen3 manager
    if _qwen3_manager is not None:
        qwen3_status["voice_design_loaded"] = _qwen3_manager.is_loaded
        qwen3_status["base_model_loaded"] = _qwen3_manager.is_loaded
        qwen3_status["idle_seconds"] = _qwen3_manager.idle_seconds

    # Check StyleTTS 2 manager
    if _styletts2_manager is not None:
        styletts2_status["loaded"] = _styletts2_manager.is_loaded
        styletts2_status["idle_seconds"] = _styletts2_manager.idle_seconds

    overall_status = (
        "healthy"
        if (
            qwen3_status["voice_design_loaded"]
            or (styletts2_status["available"] and styletts2_status["loaded"])
        )
        else "not_loaded"
    )

    return {
        "status": overall_status,
        "qwen3": qwen3_status,
        "styletts2": styletts2_status,
    }
