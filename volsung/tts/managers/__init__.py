"""Volsung TTS managers package.

Provides specific TTS model managers for voice synthesis.
Each manager wraps a specific TTS model and implements lazy loading
and idle timeout monitoring via ModelManagerBase.

Example:
    ```python
    from volsung.tts.managers import StyleTTS2Manager, get_styletts2_manager

    # Option 1: Create new manager instance
    manager = StyleTTS2Manager(idle_timeout=300)

    # Option 2: Use singleton getter (recommended)
    manager = get_styletts2_manager()

    # Extract style from reference audio
    style = manager.compute_style(ref_audio_b64)

    # Generate speech with cloned voice
    result = manager.generate(
        text="Hello, world!",
        ref_audio_b64=ref_audio_b64,
        embedding_scale=1.0,
        alpha=0.3,
        beta=0.7,
        diffusion_steps=10,
    )

    # Convert to base64 for transmission
    audio_b64 = result.audio
    ```
"""

from .styletts2 import StyleTTS2Manager, get_styletts2_manager

__all__ = [
    "StyleTTS2Manager",
    "get_styletts2_manager",
]
