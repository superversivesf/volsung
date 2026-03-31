"""Pydantic schemas for TTS (Text-to-Speech) API requests and responses.

Provides data validation and serialization for voice design and synthesis endpoints.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


# =============================================================================
# StyleTTS 2 Parameters Schema
# =============================================================================


class StyleTTSParams(BaseModel):
    """Parameters for StyleTTS 2 voice generation.

    Controls the style, emotion, and quality of generated speech
    when using the StyleTTS 2 backend.

    Example:
        ```python
        params = StyleTTSParams(
            embedding_scale=1.5,
            alpha=0.3,
            beta=0.7,
            diffusion_steps=10,
        )
        ```
    """

    embedding_scale: float = Field(
        default=1.0,
        ge=1.0,
        le=10.0,
        description="Scale for speaker embedding, controls emotion intensity (1.0-10.0). Higher values = more emotional/expressive speech",
        examples=[1.0],
    )
    alpha: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Alpha parameter for diffusion sampling (0.0-1.0)",
        examples=[0.3],
    )
    beta: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Beta parameter for diffusion sampling (0.0-1.0)",
        examples=[0.7],
    )
    diffusion_steps: int = Field(
        default=10,
        ge=3,
        le=20,
        description="Number of diffusion steps for inference (3-20). More steps = higher quality but slower",
        examples=[10],
    )


# =============================================================================
# Voice Design Schemas
# =============================================================================


class VoiceDesignRequest(BaseModel):
    """Request for generating a voice sample from a description.

    Uses the VoiceDesign model to create a unique voice character
    based on natural language description. The output can be used
    as reference audio for synthesis.

    Example:
        ```python
        request = VoiceDesignRequest(
            text="Hello, I am John. Nice to meet you.",
            language="English",
            instruct="A warm, elderly man's voice with a slight Southern accent"
        )
        ```
    """

    text: str = Field(
        ...,
        description="Sample text to speak (e.g., 'Hello, I am John.')",
        examples=["Hello, I am John. Nice to meet you."],
    )
    language: str = Field(
        default="English",
        description="Language: English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian, or Auto",
        examples=["English"],
    )
    instruct: str = Field(
        ...,
        description="Natural language voice description (e.g., 'A warm, elderly man with a Southern accent')",
        examples=["A warm, elderly man's voice with a slight Southern accent"],
    )
    backend: Literal["qwen3", "styletts2"] = Field(
        default="qwen3",
        description="TTS backend to use: 'qwen3' for Qwen3-TTS, 'styletts2' for StyleTTS 2",
        examples=["qwen3"],
    )
    styletts_params: Optional[StyleTTSParams] = Field(
        default=None,
        description="StyleTTS 2-specific parameters. Only used when backend='styletts2'",
    )


class VoiceDesignResponse(BaseModel):
    """Response containing generated audio for use as reference.

    The returned audio is base64-encoded WAV data that should be stored
    along with the transcript (the text sent in the request) for later
    use in synthesis.

    Example:
        ```python
        response = VoiceDesignResponse(
            audio="base64encodedwavdata...",
            sample_rate=24000
        )
        ```
    """

    audio: str = Field(
        ...,
        description="Base64-encoded WAV audio data",
    )
    sample_rate: int = Field(
        default=24000,
        description="Audio sample rate in Hz",
        examples=[24000],
    )


# =============================================================================
# Synthesis Schemas
# =============================================================================


class SynthesizeRequest(BaseModel):
    """Request for synthesizing text with a cloned voice.

    Uses reference audio (typically from voice_design) to clone a voice
    and synthesize new text in that voice character.

    Example:
        ```python
        request = SynthesizeRequest(
            ref_audio="base64encodedwavdata...",
            ref_text="Hello, I am John. Nice to meet you.",
            text="The quick brown fox jumps over the lazy dog.",
            language="English"
        )
        ```
    """

    ref_audio: str = Field(
        ...,
        description="Base64-encoded reference WAV audio (from /voice/design output)",
    )
    ref_text: str = Field(
        ...,
        description="Transcript of the reference audio",
        examples=["Hello, I am John. Nice to meet you."],
    )
    text: str = Field(
        ...,
        description="New text to synthesize in the cloned voice",
        examples=["The quick brown fox jumps over the lazy dog."],
    )
    language: str = Field(
        default="English",
        description="Language code (default: English)",
        examples=["English"],
    )


class SynthesizeResponse(BaseModel):
    """Response containing synthesized audio.

    The returned audio is base64-encoded WAV data containing the
    synthesized speech in the cloned voice.

    Example:
        ```python
        response = SynthesizeResponse(
            audio="base64encodedwavdata...",
            sample_rate=24000
        )
        ```
    """

    audio: str = Field(
        ...,
        description="Base64-encoded WAV audio data",
    )
    sample_rate: int = Field(
        default=24000,
        description="Audio sample rate in Hz",
        examples=[24000],
    )
