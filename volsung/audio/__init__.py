"""
Volsung audio package.

Handles audio processing, encoding, and utilities.
"""

from .effects import (
    apply_gain,
    fade_in,
    fade_out,
    get_duration,
    mix_tracks,
    normalize,
    pad_audio,
    resample,
    stereo_to_mono,
    trim_silence,
)
from .utils import audio_to_base64, base64_to_audio

__all__ = [
    # Utils
    "audio_to_base64",
    "base64_to_audio",
    # Effects
    "normalize",
    "fade_in",
    "fade_out",
    "mix_tracks",
    "resample",
    "get_duration",
    "trim_silence",
    "apply_gain",
    "stereo_to_mono",
    "pad_audio",
]
