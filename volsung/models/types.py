"""Shared types for Volsung models.

Contains Pydantic models for standardized data structures across modules.
"""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class AudioType(str, Enum):
    """Type of audio content."""

    TTS = "tts"  # Speech/dialogue
    MUSIC = "music"  # Background music
    SFX = "sfx"  # Sound effects


class AudioResult(BaseModel):
    """Standardized audio result from any generator.

    Used for TTS, Music, and SFX generation.

    Attributes:
        audio: Base64-encoded WAV audio data
        sample_rate: Sample rate in Hz (e.g., 24000)
        duration: Duration in seconds
        audio_type: Type of audio content (TTS, MUSIC, SFX)
        generator: Generator/model used
        prompt: Input prompt/description
        format: Audio format (default: wav)
        channels: Number of channels (1=mono, 2=stereo)
        metadata: Additional metadata dictionary
        track_id: Unique track identifier (for stitcher)
        start_time: Start time in composition (seconds, for stitcher)
        end_time: End time in composition (seconds, for stitcher)

    Example:
        ```python
        result = AudioResult(
            audio="base64encoded...",
            sample_rate=24000,
            duration=30.0,
            audio_type=AudioType.MUSIC,
            generator="musicgen-small",
            prompt="A calm, mysterious orchestral piece with strings",
        )
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "audio": "base64encoded...",
                "sample_rate": 24000,
                "duration": 30.0,
                "audio_type": "music",
                "generator": "musicgen-small",
                "prompt": "A calm, mysterious orchestral piece with strings",
                "format": "wav",
                "channels": 1,
                "metadata": {
                    "generation_time": 5.2,
                    "model_version": "1.0",
                },
            }
        }
    )

    # Core audio data
    audio: str = Field(..., description="Base64-encoded WAV audio")
    sample_rate: int = Field(..., description="Sample rate in Hz (e.g., 24000)")
    duration: float = Field(..., description="Duration in seconds")

    # Metadata
    audio_type: AudioType = Field(..., description="Type of audio content")
    generator: str = Field(..., description="Generator/model used")
    prompt: str = Field(..., description="Input prompt/description")

    # Optional fields
    format: str = Field(default="wav", description="Audio format")
    channels: int = Field(
        default=1, description="Number of channels (1=mono, 2=stereo)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    # For future stitcher
    track_id: Optional[str] = Field(default=None, description="Unique track identifier")
    start_time: Optional[float] = Field(
        default=None, description="Start time in composition (seconds)"
    )
    end_time: Optional[float] = Field(
        default=None, description="End time in composition (seconds)"
    )
