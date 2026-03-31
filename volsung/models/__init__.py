"""Base models module for Volsung.

Provides abstract base classes and shared types for all model managers.
"""

from .base import ModelManagerBase, GeneratorBase, ModelConfig
from .types import AudioResult, AudioType
from .registry import ModelRegistry, get_registry

__all__ = [
    "ModelManagerBase",
    "GeneratorBase",
    "ModelConfig",
    "AudioResult",
    "AudioType",
    "ModelRegistry",
    "get_registry",
]
