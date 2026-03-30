"""FastAPI router for Music generation endpoints.

Provides HTTP API for generating music from text descriptions.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from volsung.config import get_config
from volsung.models.base import ModelConfig
from .manager import MusicModelManager
from .schemas import (
    MusicGenerateRequest,
    MusicGenerateResponse,
    MusicMetadata,
    MusicInfoResponse,
)

logger = logging.getLogger(__name__)

# Module-level manager instance (lazy-loaded)
_manager: Optional[MusicModelManager] = None


def get_manager() -> MusicModelManager:
    """Get or create the music manager instance.

    Returns:
        MusicModelManager singleton
    """
    global _manager
    if _manager is None:
        config = get_config()
        model_config = ModelConfig(
            model_id="musicgen-small",
            model_name="MusicGen Small",
            device=config.music.generation.get("device", "auto"),
            dtype=config.music.generation.get("dtype", "auto"),
            idle_timeout_seconds=config.music.idle_timeout,
        )
        _manager = MusicModelManager(model_config)
        logger.info("MusicModelManager initialized")
    return _manager


router = APIRouter(
    prefix="/music",
    tags=["music"],
    responses={
        503: {"description": "Model not loaded"},
        500: {"description": "Generation failed"},
    },
)


@router.post("/generate", response_model=MusicGenerateResponse)
async def music_generate(request: MusicGenerateRequest) -> MusicGenerateResponse:
    """Generate music from text description.

    Creates background music up to 30 seconds from natural language description.
    Useful for audiobook background music, ambience, etc.

    Args:
        request: Music generation request with description and parameters

    Returns:
        MusicGenerateResponse with base64-encoded audio and metadata

    Raises:
        HTTPException: If generation fails or model unavailable

    Example:
        ```python
        import requests

        response = requests.post(
            "http://localhost:8000/music/generate",
            json={
                "description": "Peaceful acoustic guitar",
                "duration": 15.0,
                "mood": "calm",
            }
        )
        result = response.json()
        # result["audio"] contains base64-encoded WAV
        ```
    """
    manager = get_manager()

    try:
        result = manager.generate(
            prompt=request.description,
            duration=request.duration,
            genre=request.genre,
            mood=request.mood,
            tempo=request.tempo,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature,
        )

        # Build metadata
        genre_tags = []
        if request.genre:
            genre_tags.append(request.genre)
        if request.mood:
            genre_tags.append(request.mood)
        if request.tempo:
            genre_tags.append(request.tempo)

        metadata = MusicMetadata(
            duration=result.duration,
            sample_rate=result.sample_rate,
            genre_tags=genre_tags,
            generation_time_ms=result.metadata.get("generation_time_ms", 0.0),
            model_used=result.generator,
            parameters=result.metadata.get("parameters", {}),
        )

        return MusicGenerateResponse(
            audio=result.audio,
            sample_rate=result.sample_rate,
            metadata=metadata,
        )

    except ValueError as e:
        logger.warning(f"[MUSIC] Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[MUSIC] Generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Music generation failed: {str(e)}"
        )


@router.get("/info", response_model=MusicInfoResponse)
async def music_info() -> MusicInfoResponse:
    """Get music module information.

    Returns current model status, load state, and configuration.

    Returns:
        MusicInfoResponse with module information

    Example:
        ```python
        import requests

        response = requests.get("http://localhost:8000/music/info")
        info = response.json()
        # info["is_loaded"] tells if model is in memory
        ```
    """
    config = get_config()

    if _manager is None:
        return MusicInfoResponse(
            status="not_initialized",
            model_id="musicgen-small",
            model_name="MusicGen Small",
            is_loaded=False,
            device=None,
        )

    info = _manager.get_info()
    return MusicInfoResponse(
        status="ready" if info["is_loaded"] else "unloaded",
        model_id=info["model_id"],
        model_name=info["model_name"],
        is_loaded=info["is_loaded"],
        device=info.get("device"),
    )


@router.post("/unload")
async def music_unload() -> dict:
    """Force unload the music model.

    Frees GPU memory by unloading the model immediately.
    Model will be reloaded on next generation request.

    Returns:
        Dictionary with unload status

    Example:
        ```python
        import requests

        response = requests.post("http://localhost:8000/music/unload")
        result = response.json()
        # result["unloaded"] is True if model was unloaded
        ```
    """
    if _manager is None:
        return {"unloaded": False, "reason": "manager not initialized"}

    was_loaded = _manager.is_loaded
    _manager.force_unload()

    return {
        "unloaded": was_loaded,
        "status": "unloaded" if was_loaded else "already_unloaded",
    }
