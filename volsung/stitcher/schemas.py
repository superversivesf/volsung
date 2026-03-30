"""
Pydantic schemas for the Stitcher module.

Defines data models for audio composition, tracks, timelines,
and composition requests/responses.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class TrackType(str, Enum):
    """Types of audio tracks that can be composed."""

    TTS = "tts"
    MUSIC = "music"
    SFX = "sfx"
    VOICE = "voice"


class TransitionType(str, Enum):
    """Types of transitions between audio segments."""

    NONE = "none"
    FADE = "fade"
    CROSSFADE = "crossfade"
    OVERLAP = "overlap"


class AudioSegment(BaseModel):
    """An audio segment with timing information."""

    audio: str = Field(description="Base64-encoded WAV audio data")
    duration: float = Field(gt=0, description="Duration in seconds")
    sample_rate: int = Field(
        default=24000, ge=8000, le=48000, description="Sample rate in Hz"
    )

    @field_validator("audio")
    @classmethod
    def validate_audio_base64(cls, v: str) -> str:
        """Ensure audio is valid base64."""
        import base64

        try:
            base64.b64decode(v)
        except Exception:
            raise ValueError("Audio must be valid base64-encoded data")
        return v


class Track(BaseModel):
    """
    A track represents a single audio stream with metadata.

    Tracks can be of different types (TTS, music, SFX) and can be
    positioned at specific times in a timeline with optional transitions.
    """

    track_id: str = Field(description="Unique identifier for this track")
    track_type: TrackType = Field(description="Type of audio content")
    audio: AudioSegment = Field(description="Audio segment data")
    start_time: float = Field(
        default=0.0, ge=0.0, description="Start time in seconds from timeline beginning"
    )
    end_time: float | None = Field(
        default=None, ge=0.0, description="End time in seconds (None = audio duration)"
    )
    volume: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Volume multiplier (0.0 = silent, 1.0 = normal, 2.0 = double)",
    )
    transition_in: TransitionType = Field(
        default=TransitionType.FADE, description="Transition type when track starts"
    )
    transition_out: TransitionType = Field(
        default=TransitionType.FADE, description="Transition type when track ends"
    )
    fade_in_duration: float = Field(
        default=0.5, ge=0.0, le=5.0, description="Fade-in duration in seconds"
    )
    fade_out_duration: float = Field(
        default=0.5, ge=0.0, le=5.0, description="Fade-out duration in seconds"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional track metadata (e.g., speaker_id, emotion, category)",
    )

    @field_validator("end_time")
    @classmethod
    def validate_end_time(cls, v: float | None, info: Any) -> float | None:
        """Ensure end_time is after start_time."""
        if v is not None and info.data.get("start_time") is not None:
            if v <= info.data["start_time"]:
                raise ValueError("end_time must be greater than start_time")
        return v


class Timeline(BaseModel):
    """
    A timeline contains multiple tracks arranged in time.

    Represents the complete audio composition structure before rendering.
    """

    timeline_id: str = Field(description="Unique identifier for this timeline")
    name: str = Field(default="Untitled Timeline", description="Human-readable name")
    duration: float = Field(gt=0.0, description="Total timeline duration in seconds")
    sample_rate: int = Field(
        default=24000,
        ge=8000,
        le=48000,
        description="Target sample rate for composition",
    )
    tracks: list[Track] = Field(
        default_factory=list, description="Audio tracks in this timeline"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )

    @field_validator("tracks")
    @classmethod
    def validate_tracks_fit_in_duration(cls, v: list[Track], info: Any) -> list[Track]:
        """Ensure all tracks fit within timeline duration."""
        duration = info.data.get("duration")
        if duration is not None:
            for track in v:
                if track.start_time > duration:
                    raise ValueError(
                        f"Track {track.track_id} starts after timeline ends"
                    )
        return v


class CompositionRequest(BaseModel):
    """
    Request to compose audio from a timeline.

    This is the input to the stitcher composition endpoint.
    """

    timeline: Timeline = Field(description="Timeline containing tracks to compose")
    output_format: Literal["wav", "mp3", "ogg", "flac"] = Field(
        default="wav", description="Output audio format"
    )
    normalize: bool = Field(
        default=True, description="Normalize output audio to prevent clipping"
    )
    normalize_target_db: float | None = Field(
        default=-1.0,
        ge=-20.0,
        le=0.0,
        description="Target dB for normalization (None = no normalization)",
    )
    request_id: str | None = Field(
        default=None, description="Optional client-provided request ID for tracking"
    )


class CompositionMetadata(BaseModel):
    """Metadata about a composition operation."""

    duration: float = Field(description="Output audio duration in seconds")
    sample_rate: int = Field(description="Output sample rate in Hz")
    num_tracks: int = Field(description="Number of tracks composed")
    peak_amplitude: float = Field(description="Peak amplitude of output (0.0 to 1.0)")
    processing_time_ms: float = Field(
        description="Time taken to compose in milliseconds"
    )
    applied_normalization: bool = Field(description="Whether normalization was applied")
    normalization_gain_db: float | None = Field(
        default=None, description="Gain applied during normalization in dB"
    )


class CompositionResult(BaseModel):
    """
    Result of a composition operation.

    Contains the composed audio and metadata about the operation.
    """

    audio: str = Field(description="Base64-encoded composed audio")
    sample_rate: int = Field(description="Sample rate of output audio in Hz")
    duration: float = Field(description="Duration of output audio in seconds")
    format: str = Field(description="Audio format (wav, mp3, etc.)")
    metadata: CompositionMetadata = Field(description="Composition metadata")
    request_id: str | None = Field(
        default=None, description="Echo of client request ID if provided"
    )


class CompositionStatus(str, Enum):
    """Status of a composition job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CompositionJob(BaseModel):
    """Represents a queued or active composition job."""

    job_id: str = Field(description="Unique job identifier")
    status: CompositionStatus = Field(description="Current job status")
    request: CompositionRequest = Field(description="Original composition request")
    result: CompositionResult | None = Field(
        default=None, description="Result if completed"
    )
    error: str | None = Field(default=None, description="Error message if failed")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Job creation timestamp"
    )
    started_at: datetime | None = Field(
        default=None, description="Processing start timestamp"
    )
    completed_at: datetime | None = Field(
        default=None, description="Completion timestamp"
    )
    progress_percent: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Current progress percentage"
    )


# Export all models
__all__ = [
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
]
