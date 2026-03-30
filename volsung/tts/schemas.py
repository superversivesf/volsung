"""Pydantic schemas for TTS (Text-to-Speech) API requests and responses.

Provides data validation and serialization for voice design and synthesis endpoints.
"""

from pydantic import BaseModel, Field


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
