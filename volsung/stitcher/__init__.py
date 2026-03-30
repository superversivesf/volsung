"""
Volsung stitcher package.

Handles audio stream stitching and seamless transitions
between audio segments (music, SFX, TTS).
"""

from volsung.stitcher.schemas import (
    TrackType,
    TransitionType,
    AudioSegment,
    Track,
    Timeline,
    CompositionRequest,
    CompositionResult,
    CompositionMetadata,
    CompositionStatus,
    CompositionJob,
)
from volsung.stitcher.composer import AudioComposer
from volsung.stitcher.endpoints import router

__all__ = [
    # Schemas
    "TrackType",
    "TransitionType",
    "AudioSegment",
    "Track",
    "Timeline",
    "CompositionRequest",
    "CompositionResult",
    "CompositionMetadata",
    "CompositionStatus",
    "CompositionJob",
    # Composer
    "AudioComposer",
    # Endpoints
    "router",
]
