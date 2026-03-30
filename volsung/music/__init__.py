"""
Volsung music package.

Handles music generation and processing.
"""

from .manager import MusicModelManager
from .schemas import (
    MusicGenerateRequest,
    MusicGenerateResponse,
    MusicInfoResponse,
    MusicMetadata,
)
from .endpoints import router as music_router

__all__ = [
    "MusicModelManager",
    "MusicGenerateRequest",
    "MusicGenerateResponse",
    "MusicInfoResponse",
    "MusicMetadata",
    "music_router",
]
