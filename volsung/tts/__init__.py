"""Volsung TTS (Text-to-Speech) package.

Handles text-to-speech synthesis using Qwen3-TTS models with voice design
and voice cloning capabilities.

Example:
    ```python
    from volsung.tts import TTSModelManager, VoiceDesignRequest, SynthesizeRequest

    # Initialize manager
    manager = TTSModelManager()

    # Design a voice
    design_req = VoiceDesignRequest(
        text="Hello, I am John.",
        language="English",
        instruct="A warm, elderly man's voice with a Southern accent"
    )
    design_result = manager.voice_design(design_req)

    # Synthesize with cloned voice
    synth_req = SynthesizeRequest(
        ref_audio=design_audio_base64,
        ref_text="Hello, I am John.",
        text="The quick brown fox jumps over the lazy dog."
    )
    synth_result = manager.synthesize(synth_req)
    ```
"""

from .manager import TTSModelManager
from .schemas import (
    SynthesizeRequest,
    SynthesizeResponse,
    VoiceDesignRequest,
    VoiceDesignResponse,
)

__all__ = [
    # Manager
    "TTSModelManager",
    # Schemas
    "VoiceDesignRequest",
    "VoiceDesignResponse",
    "SynthesizeRequest",
    "SynthesizeResponse",
]
