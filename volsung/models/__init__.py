"""Base models module for Volsung.

Provides abstract base classes and shared types for all model managers.
"""

from .base import ModelManagerBase, GeneratorBase, ModelConfig
from .preload_manager import PreloadManager, get_preload_manager
from .types import AudioResult, AudioType
from .registry import ModelRegistry, get_registry

__all__ = [
    "ModelManagerBase",
    "GeneratorBase",
    "ModelConfig",
    "PreloadManager",
    "get_preload_manager",
    "AudioResult",
    "AudioType",
    "ModelRegistry",
    "get_registry",
]
