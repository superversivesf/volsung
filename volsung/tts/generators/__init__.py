"""TTS generators package.

Provides specific TTS model generators for voice synthesis.
Each generator wraps a specific TTS model and implements the GeneratorBase interface.

Example:
    ```python
    from volsung.tts.generators import StyleTTS2Generator

    generator = StyleTTS2Generator()
    generator.load(device="cuda:0", dtype="float16")
    audio, sr = generator.generate(
        text="Hello, world!",
        ref_audio=base64_reference_audio,
    )
    ```
"""

from .styletts2 import StyleTTS2Generator

__all__ = [
    "StyleTTS2Generator",
]
