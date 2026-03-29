"""
Volsung - Voice synthesis server for Qwen3-TTS.

Named after the Völsung saga of Norse mythology, where heroes' deeds
were preserved through oral tradition. This server gives voice to text.
"""

__version__ = "1.0.0"

from volsung.server import app

__all__ = ["app", "__version__"]
