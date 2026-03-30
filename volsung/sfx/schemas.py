"""
Pydantic schemas for SFX (Sound Effects) API.

Request/response models for sound effects generation endpoints.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class SFXGenerateRequest(BaseModel):
    """Request for generating sound effects from text description.

    Attributes:
        description: Natural language description of desired sound effect
        duration: Target duration in seconds (1.0 to 10.0)
        category: Optional category hint (e.g., "nature", "mechanical", "urban")
        num_inference_steps: Number of denoising steps (higher = better quality)
        guidance_scale: How closely to follow the prompt (higher = more faithful)

    Example:
        ```python
        request = SFXGenerateRequest(
            description="Footsteps on gravel",
            duration=3.0,
            category="nature"
        )
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "description": "Thunder rumbling in the distance",
                "duration": 5.0,
                "category": "nature",
                "num_inference_steps": 50,
                "guidance_scale": 3.5,
            }
        }
    )

    description: str = Field(
        ...,
        description="Natural language description of desired sound effect",
        min_length=1,
        max_length=1000,
    )
    duration: float = Field(
        default=5.0,
        ge=1.0,
        le=10.0,
        description="Target duration in seconds (1.0 to 10.0)",
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional category hint (e.g., 'nature', 'mechanical', 'urban', 'fantasy')",
    )
    num_inference_steps: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Number of denoising steps (higher = better quality, slower)",
    )
    guidance_scale: float = Field(
        default=3.5,
        ge=1.0,
        le=20.0,
        description="Prompt adherence (higher = more faithful to prompt)",
    )


class SFXMetadata(BaseModel):
    """Metadata for generated sound effects.

    Attributes:
        duration: Actual duration in seconds
        sample_rate: Sample rate in Hz
        category: Category tag if provided
        generation_time_ms: Time taken to generate in milliseconds
        model_used: Model identifier used for generation
        num_inference_steps: Denoising steps used
        guidance_scale: Guidance scale used

    Example:
        ```python
        metadata = SFXMetadata(
            duration=5.0,
            sample_rate=16000,
            category="nature",
            generation_time_ms=2500,
            model_used="audioldm2",
        )
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "duration": 5.0,
                "sample_rate": 16000,
                "category": "nature",
                "generation_time_ms": 2500,
                "model_used": "audioldm2",
                "num_inference_steps": 50,
                "guidance_scale": 3.5,
            }
        }
    )

    duration: float = Field(..., description="Audio duration in seconds")
    sample_rate: int = Field(..., description="Sample rate in Hz")
    category: Optional[str] = Field(default=None, description="SFX category")
    generation_time_ms: float = Field(..., description="Time taken to generate (ms)")
    model_used: str = Field(..., description="Model identifier")
    num_inference_steps: int = Field(default=50, description="Denoising steps used")
    guidance_scale: float = Field(default=3.5, description="Guidance scale used")


class SFXGenerateResponse(BaseModel):
    """Response containing generated sound effect and metadata.

    Attributes:
        audio: Base64-encoded WAV audio data
        sample_rate: Sample rate in Hz
        metadata: Generation metadata

    Example:
        ```python
        response = SFXGenerateResponse(
            audio="base64encoded...",
            sample_rate=16000,
            metadata=SFXMetadata(...),
        )
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "audio": "base64encodedWAVdata...",
                "sample_rate": 16000,
                "metadata": {
                    "duration": 5.0,
                    "sample_rate": 16000,
                    "category": "nature",
                    "generation_time_ms": 2500,
                    "model_used": "audioldm2",
                    "num_inference_steps": 50,
                    "guidance_scale": 3.5,
                },
            }
        }
    )

    audio: str = Field(..., description="Base64-encoded WAV audio data")
    sample_rate: int = Field(..., description="Sample rate in Hz")
    metadata: SFXMetadata = Field(..., description="Generation metadata")


class SFXLayerRequest(BaseModel):
    """Request for generating layered/combined sound effects.

    Attributes:
        layers: List of SFX generation requests to combine
        mix_mode: How to combine layers ("sum" for simple addition)

    Example:
        ```python
        request = SFXLayerRequest(
            layers=[
                SFXGenerateRequest(description="Thunder", duration=5.0),
                SFXGenerateRequest(description="Rain", duration=5.0),
            ]
        )
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "layers": [
                    {
                        "description": "Thunder rumbling",
                        "duration": 5.0,
                        "category": "nature",
                    },
                    {
                        "description": "Rain falling",
                        "duration": 5.0,
                        "category": "nature",
                    },
                ],
                "mix_mode": "sum",
            }
        }
    )

    layers: List[SFXGenerateRequest] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="List of SFX generation requests to combine",
    )
    mix_mode: str = Field(
        default="sum",
        description="Mixing mode ('sum' for simple addition)",
    )


class SFXLayerResponse(BaseModel):
    """Response containing combined/layered sound effects.

    Attributes:
        audio: Base64-encoded combined WAV audio
        sample_rate: Sample rate in Hz
        layers_metadata: Metadata for each generated layer
        total_duration: Duration of combined audio in seconds

    Example:
        ```python
        response = SFXLayerResponse(
            audio="base64encoded...",
            sample_rate=16000,
            layers_metadata=[SFXMetadata(...), SFXMetadata(...)],
            total_duration=5.0,
        )
        ```
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "audio": "base64encodedWAVdata...",
                "sample_rate": 16000,
                "layers_metadata": [
                    {
                        "duration": 5.0,
                        "sample_rate": 16000,
                        "category": "nature",
                        "generation_time_ms": 2500,
                        "model_used": "audioldm2",
                    },
                ],
                "total_duration": 5.0,
            }
        }
    )

    audio: str = Field(..., description="Base64-encoded combined WAV audio")
    sample_rate: int = Field(..., description="Sample rate in Hz")
    layers_metadata: List[SFXMetadata] = Field(
        ..., description="Metadata for each generated layer"
    )
    total_duration: float = Field(
        ..., description="Duration of combined audio in seconds"
    )


class SFXHealthResponse(BaseModel):
    """Health check response for SFX module.

    Attributes:
        status: Module status ("healthy" or "unhealthy")
        model_loaded: Whether the SFX model is currently loaded
        model_name: Name of the configured model
        idle_seconds: Seconds since last access (if loaded)

    Example:
        ```python
        health = SFXHealthResponse(
            status="healthy",
            model_loaded=True,
            model_name="audioldm2",
            idle_seconds=45.5,
        )
        ```
    """

    status: str = Field(..., description="Module status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: str = Field(..., description="Configured model name")
    idle_seconds: Optional[float] = Field(
        default=None, description="Seconds since last access (if loaded)"
    )
