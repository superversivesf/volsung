"""
Audio effects and processing functions.

Volsung - Voice synthesis server for Qwen3-TTS.
"""

import numpy as np
from numpy.typing import NDArray
from scipy import signal as scipy_signal


def normalize(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalize audio to target peak amplitude.

    Scales the entire audio signal so that the maximum absolute amplitude
    equals the target peak value. Prevents clipping while maximizing volume.

    Args:
        audio: Input audio array (float32 or float64)
        target_peak: Target maximum amplitude (default 0.95, range (0, 1])

    Returns:
        Normalized audio array with peak at target_peak

    Example:
        >>> audio = np.array([0.5, -0.3, 0.8, -0.2])
        >>> normalized = normalize(audio, target_peak=1.0)
        >>> np.max(np.abs(normalized))
        1.0
    """
    audio = np.asarray(audio, dtype=np.float32)
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    return audio * (target_peak / peak)


def fade_in(audio: np.ndarray, duration_samples: int) -> np.ndarray:
    """Apply linear fade-in to audio.

    Gradually increases volume from 0 to full amplitude over the specified
    duration at the beginning of the audio.

    Args:
        audio: Input audio array
        duration_samples: Length of fade-in in samples

    Returns:
        Audio with fade-in applied

    Example:
        >>> audio = np.ones(10000)  # 10k samples
        >>> faded = fade_in(audio, 1000)  # 1000 sample fade-in
        >>> faded[0]
        0.0  # Starts at zero
        >>> faded[999]
        1.0  # Full volume at end of fade
    """
    audio = np.asarray(audio, dtype=np.float32).copy()
    if duration_samples <= 0 or len(audio) == 0:
        return audio
    duration_samples = min(duration_samples, len(audio))
    fade_curve = np.linspace(0, 1, duration_samples, dtype=np.float32)
    audio[:duration_samples] *= fade_curve
    return audio


def fade_out(audio: np.ndarray, duration_samples: int) -> np.ndarray:
    """Apply linear fade-out to audio.

    Gradually decreases volume from full amplitude to 0 over the specified
    duration at the end of the audio.

    Args:
        audio: Input audio array
        duration_samples: Length of fade-out in samples

    Returns:
        Audio with fade-out applied

    Example:
        >>> audio = np.ones(10000)  # 10k samples
        >>> faded = fade_out(audio, 1000)  # 1000 sample fade-out
        >>> faded[-1]
        0.0  # Ends at zero
        >>> faded[-1000]
        1.0  # Full volume at start of fade
    """
    audio = np.asarray(audio, dtype=np.float32).copy()
    if duration_samples <= 0 or len(audio) == 0:
        return audio
    duration_samples = min(duration_samples, len(audio))
    fade_curve = np.linspace(1, 0, duration_samples, dtype=np.float32)
    audio[-duration_samples:] *= fade_curve
    return audio


def mix_tracks(
    tracks: list[tuple[np.ndarray, float]],
    normalize_output: bool = True,
) -> np.ndarray:
    """Mix multiple audio tracks together with optional per-track gain.

    Combines multiple audio arrays into a single mixed output. Longer tracks
    are truncated to match the shortest, and shorter tracks are padded with
    silence. Each track can have an individual gain applied.

    Args:
        tracks: List of (audio_array, gain) tuples where gain is a multiplier
        normalize_output: Whether to normalize the final mix to prevent clipping

    Returns:
        Mixed audio array

    Example:
        >>> track1 = (np.random.randn(24000), 1.0)  # Full volume
        >>> track2 = (np.random.randn(24000), 0.5)  # Half volume
        >>> mixed = mix_tracks([track1, track2])
    """
    if not tracks:
        return np.array([], dtype=np.float32)

    # Find minimum length
    min_len = min(len(t[0]) for t in tracks)

    # Mix with gain
    mixed = np.zeros(min_len, dtype=np.float64)  # Use double for accumulation
    for audio, gain in tracks:
        mixed += np.asarray(audio[:min_len], dtype=np.float64) * gain

    mixed = mixed.astype(np.float32)

    if normalize_output:
        mixed = normalize(mixed)

    return mixed


def resample(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
    filter: str = "kaiser_best",
) -> np.ndarray:
    """Resample audio to a different sample rate.

    Uses scipy's resample_poly for high-quality sample rate conversion.
    Maintains the duration of the audio while changing the sample rate.

    Args:
        audio: Input audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate
        filter: Resampling filter quality ("kaiser_best", "kaiser_fast")

    Returns:
        Resampled audio array

    Example:
        >>> audio = np.random.randn(24000)  # 24kHz, 1 second
        >>> resampled = resample(audio, 24000, 16000)
        >>> len(resampled)
        16000  # Now 1 second at 16kHz
    """
    audio = np.asarray(audio, dtype=np.float32)
    if orig_sr == target_sr:
        return audio

    # Calculate GCD for polyphase resampling
    from math import gcd

    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g

    return scipy_signal.resample_poly(audio, up, down, window=("kaiser", 5.0))


def get_duration(audio: np.ndarray, sample_rate: int) -> float:
    """Calculate audio duration in seconds.

    Args:
        audio: Audio array (single or multi-channel)
        sample_rate: Sample rate in Hz

    Returns:
        Duration in seconds

    Example:
        >>> audio = np.random.randn(48000)
        >>> get_duration(audio, 24000)
        2.0  # 2 seconds
    """
    if audio.ndim == 1:
        return len(audio) / sample_rate
    else:
        return audio.shape[0] / sample_rate


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -60.0,
    padding_ms: float = 100.0,
) -> np.ndarray:
    """Trim leading and trailing silence from audio.

    Removes silence below a specified threshold from the beginning and end
    of the audio signal, with optional padding to preserve context.

    Args:
        audio: Input audio array
        sample_rate: Sample rate in Hz
        threshold_db: Silence threshold in dB (default -60)
        padding_ms: Padding in milliseconds to add around non-silent audio

    Returns:
        Trimmed audio array

    Example:
        >>> audio = np.concatenate([np.zeros(1000), audio, np.zeros(1000)])
        >>> trimmed = trim_silence(audio, 24000, threshold_db=-50)
    """
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) == 0:
        return audio

    # Convert dB threshold to amplitude
    threshold = 10 ** (threshold_db / 20)

    # Find first and last samples above threshold
    above_threshold = np.abs(audio) > threshold
    if not np.any(above_threshold):
        return np.array([], dtype=np.float32)

    first = np.argmax(above_threshold)
    last = len(audio) - np.argmax(above_threshold[::-1])

    # Add padding
    padding_samples = int((padding_ms / 1000) * sample_rate)
    first = max(0, first - padding_samples)
    last = min(len(audio), last + padding_samples)

    return audio[first:last]


def apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply gain to audio in decibels.

    Multiplies the audio signal by a gain factor specified in dB.
    Positive values increase volume, negative values decrease it.

    Args:
        audio: Input audio array
        gain_db: Gain in decibels (e.g., 6.0 for +6dB, -12.0 for -12dB)

    Returns:
        Audio with gain applied

    Example:
        >>> audio = np.ones(1000) * 0.5
        >>> louder = apply_gain(audio, 6.0)  # +6dB
        >>> np.max(louder)  # Approximately doubled
        1.0
    """
    audio = np.asarray(audio, dtype=np.float32)
    gain_linear = 10 ** (gain_db / 20)
    return audio * gain_linear


def stereo_to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert stereo audio to mono by averaging channels.

    Args:
        audio: Input audio array, can be mono (1D) or stereo (2D)

    Returns:
        Mono audio array (1D)

    Example:
        >>> stereo = np.random.randn(1000, 2)  # 1000 samples, 2 channels
        >>> mono = stereo_to_mono(stereo)
        >>> mono.shape
        (1000,)
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def pad_audio(
    audio: np.ndarray,
    target_length: int,
    mode: str = "silence",
) -> np.ndarray:
    """Pad or truncate audio to target length.

    Args:
        audio: Input audio array
        target_length: Desired length in samples
        mode: Padding mode ("silence" for zeros, "repeat" for loop)

    Returns:
        Audio padded or truncated to target_length

    Example:
        >>> audio = np.random.randn(1000)
        >>> padded = pad_audio(audio, 2000, mode="repeat")
        >>> len(padded)
        2000
    """
    audio = np.asarray(audio, dtype=np.float32)
    current_len = len(audio)

    if current_len == target_length:
        return audio

    if current_len > target_length:
        return audio[:target_length]

    # Pad
    if mode == "repeat":
        repeats = (target_length // current_len) + 1
        padded = np.tile(audio, repeats)[:target_length]
    else:  # silence
        padded = np.zeros(target_length, dtype=np.float32)
        padded[:current_len] = audio

    return padded
