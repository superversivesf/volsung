"""
Audio utility functions for encoding/decoding and format conversion.

Volsung - Voice synthesis server for Qwen3-TTS.
"""

import base64
from io import BytesIO

import numpy as np
import soundfile as sf


def audio_to_base64(wav_array: np.ndarray, sample_rate: int) -> str:
    """Convert audio array to base64-encoded WAV.

    Args:
        wav_array: Numpy array containing audio samples (float32, [-1, 1])
        sample_rate: Sample rate of the audio (e.g., 24000)

    Returns:
        Base64-encoded string of the WAV file

    Example:
        >>> audio = np.random.randn(24000).astype(np.float32)  # 1 second
        >>> b64 = audio_to_base64(audio, 24000)
    """
    buffer = BytesIO()
    sf.write(buffer, wav_array, sample_rate, format="WAV")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()


def base64_to_audio(b64: str) -> tuple[np.ndarray, int]:
    """Convert base64 WAV to audio array and sample rate.

    Args:
        b64: Base64-encoded string of a WAV file

    Returns:
        Tuple of (audio_array, sample_rate)
        - audio_array: Numpy array of float32 samples in [-1, 1]
        - sample_rate: Sample rate of the decoded audio

    Example:
        >>> audio, sr = base64_to_audio(encoded_string)
        >>> print(f"Audio duration: {len(audio) / sr:.2f}s")
    """
    audio_bytes = base64.b64decode(b64)
    buffer = BytesIO(audio_bytes)
    audio, sr = sf.read(buffer)
    return audio, sr
