"""FastAPI router for TTS (Text-to-Speech) endpoints.

Provides REST API endpoints for voice design and voice synthesis.
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

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/voice", tags=["voice"])

# Global manager instance (lazy-loaded)
_tts_manager: TTSModelManager | None = None


def get_tts_manager() -> TTSModelManager:
    """Get or create the TTS manager singleton.

    Returns:
        TTSModelManager instance
    """
    global _tts_manager
    if _tts_manager is None:
        config = get_config()
        _tts_manager = TTSModelManager(
            voice_design_model_id=config.tts.voice_design_model,
            base_model_id=config.tts.base_model,
            idle_timeout=config.tts.idle_timeout,
        )
    return _tts_manager


def reset_tts_manager() -> None:
    """Reset the TTS manager (useful for testing).

    Forces recreation of the manager on next use.
    """
    global _tts_manager
    if _tts_manager is not None:
        _tts_manager.shutdown()
    _tts_manager = None


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

    Uses the VoiceDesign model to create a unique voice character
    based on the instruction. The output should be stored (audio
    + transcript) for later use in synthesis.

    Example workflow:
    1. POST /voice/design with text and instruct
    2. Store the returned audio (base64 decode to WAV)
    3. Use that audio as ref_audio in /voice/synthesize
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
        request: Voice design parameters including text and voice description

    Returns:
        VoiceDesignResponse with base64-encoded audio

    Raises:
        HTTPException: If model not loaded or generation fails
    """
    manager = get_tts_manager()

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
    manager = get_tts_manager()

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
                        "voice_design_loaded": True,
                        "base_model_loaded": True,
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
    manager = get_tts_manager()

    return {
        "status": "healthy" if manager.is_loaded else "not_loaded",
        "voice_design_loaded": manager.is_loaded,
        "base_model_loaded": manager.is_loaded,
        "idle_seconds": manager.idle_seconds,
    }
