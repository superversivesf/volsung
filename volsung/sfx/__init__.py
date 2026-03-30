"""
Volsung SFX (Sound Effects) package.

Handles sound effects generation and processing.
"""

from .endpoints import router as sfx_router
from .manager import SFXModelManager
from .schemas import (
    SFXGenerateRequest,
    SFXGenerateResponse,
    SFXHealthResponse,
    SFXLayerRequest,
    SFXLayerResponse,
    SFXMetadata,
)

__all__ = [
    # Manager
    "SFXModelManager",
    # Router
    "sfx_router",
    # Schemas
    "SFXGenerateRequest",
    "SFXGenerateResponse",
    "SFXHealthResponse",
    "SFXLayerRequest",
    "SFXLayerResponse",
    "SFXMetadata",
]
