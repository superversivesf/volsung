"""Pydantic schemas for Music module API.

Request and response models for music generation endpoints.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MusicGenerateRequest(BaseModel):
    """Request for generating music from text description.

    Attributes:
        description: Natural language description of desired music
        duration: Target duration in seconds (1-30, default: 10)
        genre: Optional genre tag (e.g., "acoustic", "electronic", "classical")
        mood: Optional mood descriptor (e.g., "upbeat", "calm", "tense")
        tempo: Optional tempo hint ("slow", "medium", "fast")
        top_k: Top-k sampling (higher = more diverse, default: 250)
        top_p: Nucleus sampling (0.0 = disabled, default: 0.0)
        temperature: Randomness (higher = more random, default: 1.0)

    Example:
        ```python
        request = MusicGenerateRequest(
            description="Peaceful acoustic guitar for meditation",
            duration=15.0,
            genre="acoustic",
            mood="calm",
            tempo="slow",
        )
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "description": "Peaceful acoustic guitar for meditation audiobook",
                "duration": 15.0,
                "genre": "acoustic",
                "mood": "calm",
                "tempo": "slow",
            }
        }
    )

    description: str = Field(
        ...,
        description="Natural language description of desired music",
        min_length=1,
        max_length=1000,
    )
    duration: float = Field(
        default=10.0,
        ge=1.0,
        le=30.0,
        description="Target duration in seconds (1-30)",
    )
    genre: Optional[str] = Field(
        default=None,
        description="Optional genre tag (acoustic, electronic, classical, etc.)",
    )
    mood: Optional[str] = Field(
        default=None,
        description="Optional mood descriptor (upbeat, calm, tense, etc.)",
    )
    tempo: Optional[str] = Field(
        default=None,
        description="Tempo hint: slow, medium, or fast",
    )
    top_k: int = Field(
        default=250,
        ge=1,
        le=1000,
        description="Top-k sampling (higher = more diverse)",
    )
    top_p: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling (0.0 = disabled)",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.1,
        le=2.0,
        description="Randomness/temperature (higher = more random)",
    )

    @field_validator("tempo")
    @classmethod
    def validate_tempo(cls, v: Optional[str]) -> Optional[str]:
        """Validate tempo is one of the allowed values."""
        if v is None:
            return v
        allowed = {"slow", "medium", "fast"}
        if v.lower() not in allowed:
            raise ValueError(f"Tempo must be one of: {allowed}")
        return v.lower()


class MusicMetadata(BaseModel):
    """Metadata for generated music.

    Attributes:
        duration: Actual duration in seconds
        sample_rate: Audio sample rate in Hz
        genre_tags: List of genre/mood tags applied
        generation_time_ms: Time taken to generate in milliseconds
        model_used: Model identifier used for generation
        parameters: Generation parameters used

    Example:
        ```python
        metadata = MusicMetadata(
            duration=15.0,
            sample_rate=32000,
            genre_tags=["acoustic", "guitar", "calm"],
            generation_time_ms=2500.0,
            model_used="facebook/musicgen-small",
        )
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "duration": 15.0,
                "sample_rate": 32000,
                "genre_tags": ["acoustic", "guitar", "calm"],
                "generation_time_ms": 2500.0,
                "model_used": "facebook/musicgen-small",
            }
        }
    )

    duration: float = Field(..., description="Actual duration in seconds")
    sample_rate: int = Field(..., description="Audio sample rate in Hz")
    genre_tags: List[str] = Field(
        default_factory=list,
        description="Genre and mood tags applied to generation",
    )
    generation_time_ms: float = Field(
        ...,
        description="Time taken to generate in milliseconds",
    )
    model_used: str = Field(..., description="Model identifier used")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Generation parameters used",
    )


class MusicGenerateResponse(BaseModel):
    """Response containing generated music and metadata.

    Attributes:
        audio: Base64-encoded WAV audio data
        sample_rate: Sample rate in Hz
        metadata: Generation metadata

    Example:
        ```python
        response = MusicGenerateResponse(
            audio="base64encoded...",
            sample_rate=32000,
            metadata=MusicMetadata(...),
        )
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "audio": "base64encodedWAVdata...",
                "sample_rate": 32000,
                "metadata": {
                    "duration": 15.0,
                    "sample_rate": 32000,
                    "genre_tags": ["acoustic", "calm"],
                    "generation_time_ms": 2500.0,
                    "model_used": "facebook/musicgen-small",
                },
            }
        }
    )

    audio: str = Field(
        ...,
        description="Base64-encoded WAV audio data",
    )
    sample_rate: int = Field(
        ...,
        description="Sample rate in Hz",
    )
    metadata: MusicMetadata = Field(
        ...,
        description="Generation metadata",
    )


class MusicInfoResponse(BaseModel):
    """Response containing music module information.

    Attributes:
        status: Module status
        model_id: Current model identifier
        model_name: Human-readable model name
        is_loaded: Whether model is currently loaded
        device: Device the model is on

    Example:
        ```python
        info = MusicInfoResponse(
            status="ready",
            model_id="musicgen-small",
            model_name="MusicGen Small",
            is_loaded=True,
            device="cuda:0",
        )
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ready",
                "model_id": "musicgen-small",
                "model_name": "MusicGen Small",
                "is_loaded": True,
                "device": "cuda:0",
            }
        }
    )

    status: str = Field(..., description="Module status")
    model_id: str = Field(..., description="Model identifier")
    model_name: str = Field(..., description="Human-readable model name")
    is_loaded: bool = Field(..., description="Whether model is loaded")
    device: Optional[str] = Field(None, description="Device model is on")
