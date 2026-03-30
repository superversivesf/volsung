"""Tests for Audio utilities and effects.

Tests audio_to_base64, base64_to_audio, and audio effects functions
including normalization, fades, mixing, resampling, and more.
"""

import base64
from io import BytesIO

import numpy as np
import pytest
import soundfile as sf

from volsung.audio.utils import audio_to_base64, base64_to_audio
from volsung.audio.effects import (
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


# =============================================================================
# audio_to_base64 / base64_to_audio Tests
# =============================================================================


class TestAudioToBase64:
    """Test audio_to_base64 utility function."""

    def test_converts_array_to_base64(self, audio_array: np.ndarray, sample_rate: int):
        """Test converting numpy array to base64 string."""
        b64 = audio_to_base64(audio_array, sample_rate)

        # Should return a string
        assert isinstance(b64, str)
        # Should be non-empty
        assert len(b64) > 0
        # Should be valid base64
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    def test_base64_can_be_decoded(self, audio_array: np.ndarray, sample_rate: int):
        """Test that base64 output can be decoded back to audio."""
        b64 = audio_to_base64(audio_array, sample_rate)

        # Decode and verify it's valid WAV
        decoded = base64.b64decode(b64)
        buffer = BytesIO(decoded)
        audio, sr = sf.read(buffer)

        # Should have correct sample rate
        assert sr == sample_rate
        # Should have similar audio data (allowing for float precision)
        assert len(audio) == len(audio_array)

    def test_different_sample_rates(self):
        """Test conversion with different sample rates."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)

        for sr in [8000, 16000, 22050, 24000, 44100, 48000]:
            b64 = audio_to_base64(audio[:sr], sr)  # 1 second of audio
            assert isinstance(b64, str)


class TestBase64ToAudio:
    """Test base64_to_audio utility function."""

    def test_converts_base64_to_array(self, audio_array: np.ndarray, sample_rate: int):
        """Test converting base64 string back to numpy array."""
        # First encode
        b64 = audio_to_base64(audio_array, sample_rate)

        # Then decode
        audio, sr = base64_to_audio(b64)

        # Should return array and sample rate
        assert isinstance(audio, np.ndarray)
        assert isinstance(sr, int)
        assert sr == sample_rate

    def test_roundtrip_conversion(self, audio_array: np.ndarray, sample_rate: int):
        """Test roundtrip conversion preserves audio."""
        # Encode
        b64 = audio_to_base64(audio_array, sample_rate)

        # Decode
        audio, sr = base64_to_audio(b64)

        # Audio should be approximately preserved (WAV compression is lossless for float)
        assert len(audio) == len(audio_array)
        assert sr == sample_rate


class TestAudioUtilsRoundTrip:
    """Test round-trip conversions."""

    def test_sine_wave_roundtrip(self):
        """Test roundtrip with sine wave."""
        sample_rate = 24000
        t = np.linspace(0, 1, sample_rate)
        original = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        b64 = audio_to_base64(original, sample_rate)
        recovered, sr = base64_to_audio(b64)

        assert sr == sample_rate
        assert len(recovered) == len(original)
        # Audio should be very similar (allowing for tiny precision differences)
        np.testing.assert_array_almost_equal(original, recovered, decimal=5)


# =============================================================================
# normalize Tests
# =============================================================================


class TestNormalize:
    """Test normalize audio effect."""

    def test_normalizes_to_target_peak(self):
        """Test audio is normalized to target peak."""
        audio = np.array([0.5, -0.3, 0.8, -0.2], dtype=np.float32)
        normalized = normalize(audio, target_peak=1.0)

        assert np.max(np.abs(normalized)) == pytest.approx(1.0, abs=0.01)

    def test_default_target_peak(self):
        """Test default target peak is 0.95."""
        audio = np.array([0.5, -0.3, 0.8], dtype=np.float32)
        normalized = normalize(audio)

        assert np.max(np.abs(normalized)) == pytest.approx(0.95, abs=0.01)

    def test_handles_zero_audio(self):
        """Test normalize handles all-zero audio."""
        audio = np.zeros(100, dtype=np.float32)
        normalized = normalize(audio, target_peak=1.0)

        assert np.allclose(normalized, 0)

    def test_preserves_shape(self):
        """Test normalize preserves audio shape."""
        audio = np.random.randn(1000).astype(np.float32) * 0.5
        normalized = normalize(audio)

        assert normalized.shape == audio.shape


# =============================================================================
# fade_in Tests
# =============================================================================


class TestFadeIn:
    """Test fade_in audio effect."""

    def test_fade_in_applies_ramp(self):
        """Test fade-in applies linear ramp."""
        audio = np.ones(1000, dtype=np.float32)
        faded = fade_in(audio, duration_samples=100)

        # First sample should be 0
        assert faded[0] == pytest.approx(0.0, abs=0.01)
        # Last sample of fade should be 1
        assert faded[99] == pytest.approx(1.0, abs=0.01)
        # After fade, should be 1
        assert faded[100] == pytest.approx(1.0, abs=0.01)

    def test_zero_duration(self):
        """Test fade with zero duration does nothing."""
        audio = np.ones(100, dtype=np.float32)
        faded = fade_in(audio, duration_samples=0)

        np.testing.assert_array_almost_equal(audio, faded)

    def test_negative_duration(self):
        """Test fade with negative duration does nothing."""
        audio = np.ones(100, dtype=np.float32)
        faded = fade_in(audio, duration_samples=-10)

        np.testing.assert_array_almost_equal(audio, faded)

    def test_duration_longer_than_audio(self):
        """Test fade-in with duration longer than audio."""
        audio = np.ones(50, dtype=np.float32)
        faded = fade_in(audio, duration_samples=100)

        # Should fade entire audio
        assert faded[0] == pytest.approx(0.0, abs=0.01)
        assert faded[-1] == pytest.approx(1.0, abs=0.01)


# =============================================================================
# fade_out Tests
# =============================================================================


class TestFadeOut:
    """Test fade_out audio effect."""

    def test_fade_out_applies_ramp(self):
        """Test fade-out applies linear ramp."""
        audio = np.ones(1000, dtype=np.float32)
        faded = fade_out(audio, duration_samples=100)

        # Before fade, should be 1
        assert faded[-101] == pytest.approx(1.0, abs=0.01)
        # First sample of fade should be 1
        assert faded[-100] == pytest.approx(1.0, abs=0.01)
        # Last sample should be 0
        assert faded[-1] == pytest.approx(0.0, abs=0.01)

    def test_zero_duration(self):
        """Test fade with zero duration does nothing."""
        audio = np.ones(100, dtype=np.float32)
        faded = fade_out(audio, duration_samples=0)

        np.testing.assert_array_almost_equal(audio, faded)

    def test_duration_longer_than_audio(self):
        """Test fade-out with duration longer than audio."""
        audio = np.ones(50, dtype=np.float32)
        faded = fade_out(audio, duration_samples=100)

        # Should fade entire audio
        assert faded[0] == pytest.approx(1.0, abs=0.01)
        assert faded[-1] == pytest.approx(0.0, abs=0.01)


# =============================================================================
# mix_tracks Tests
# =============================================================================


class TestMixTracks:
    """Test mix_tracks audio effect."""

    def test_mixes_multiple_tracks(self):
        """Test mixing multiple tracks."""
        track1 = np.ones(1000, dtype=np.float32)
        track2 = np.ones(1000, dtype=np.float32) * 0.5

        mixed = mix_tracks([(track1, 1.0), (track2, 1.0)])

        assert len(mixed) == 1000
        # Should be sum of tracks: 1.0 + 0.5 = 1.5
        assert np.allclose(mixed, 1.5)

    def test_with_gains(self):
        """Test mixing with different gains."""
        track1 = np.ones(1000, dtype=np.float32)
        track2 = np.ones(1000, dtype=np.float32)

        mixed = mix_tracks([(track1, 0.5), (track2, 0.5)])

        # Should be normalized to 1.0
        assert np.max(np.abs(mixed)) <= 1.0

    def test_empty_tracks(self):
        """Test mixing empty list returns empty array."""
        mixed = mix_tracks([])
        assert len(mixed) == 0

    def test_tracks_different_lengths(self):
        """Test mixing tracks of different lengths."""
        track1 = np.ones(1000, dtype=np.float32)
        track2 = np.ones(500, dtype=np.float32)

        mixed = mix_tracks([(track1, 1.0), (track2, 1.0)])

        # Should be length of shortest track
        assert len(mixed) == 500

    def test_no_normalization(self):
        """Test mixing without normalization."""
        track1 = np.ones(100, dtype=np.float32)
        track2 = np.ones(100, dtype=np.float32)

        mixed = mix_tracks([(track1, 1.0), (track2, 1.0)], normalize_output=False)

        # Should be sum: 1.0 + 1.0 = 2.0
        assert np.allclose(mixed, 2.0)


# =============================================================================
# resample Tests
# =============================================================================


class TestResample:
    """Test resample audio effect."""

    def test_resample_same_rate(self):
        """Test resample with same rate returns original."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 24000)).astype(np.float32)
        resampled = resample(audio, 24000, 24000)

        np.testing.assert_array_almost_equal(audio, resampled)

    def test_resample_downsample(self):
        """Test downsampling to lower rate."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 24000)).astype(np.float32)
        resampled = resample(audio, 24000, 16000)

        # Duration should be preserved
        assert len(resampled) == 16000

    def test_resample_upsample(self):
        """Test upsampling to higher rate."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
        resampled = resample(audio, 16000, 24000)

        # Duration should be preserved
        assert len(resampled) == 24000


# =============================================================================
# get_duration Tests
# =============================================================================


class TestGetDuration:
    """Test get_duration function."""

    def test_mono_audio_duration(self):
        """Test duration calculation for mono audio."""
        audio = np.zeros(24000, dtype=np.float32)
        duration = get_duration(audio, 24000)

        assert duration == pytest.approx(1.0)

    def test_different_sample_rates(self):
        """Test duration with different sample rates."""
        audio = np.zeros(24000, dtype=np.float32)

        duration_24k = get_duration(audio, 24000)
        assert duration_24k == pytest.approx(1.0)

        duration_48k = get_duration(audio, 48000)
        assert duration_48k == pytest.approx(0.5)

    def test_stereo_audio_duration(self):
        """Test duration calculation for stereo audio."""
        audio = np.zeros((24000, 2), dtype=np.float32)
        duration = get_duration(audio, 24000)

        assert duration == pytest.approx(1.0)


# =============================================================================
# trim_silence Tests
# =============================================================================


class TestTrimSilence:
    """Test trim_silence audio effect."""

    def test_trims_leading_silence(self):
        """Test trimming leading silence."""
        # Create audio with leading silence
        audio = np.concatenate(
            [
                np.zeros(1000, dtype=np.float32),  # Silence
                np.ones(1000, dtype=np.float32) * 0.5,  # Audio
            ]
        )

        trimmed = trim_silence(audio, 24000, threshold_db=-60)

        # Should remove leading silence
        assert len(trimmed) < len(audio)

    def test_trims_trailing_silence(self):
        """Test trimming trailing silence."""
        audio = np.concatenate(
            [
                np.ones(1000, dtype=np.float32) * 0.5,  # Audio
                np.zeros(1000, dtype=np.float32),  # Silence
            ]
        )

        trimmed = trim_silence(audio, 24000, threshold_db=-60)

        # Should remove trailing silence
        assert len(trimmed) < len(audio)

    def test_preserves_non_silent_audio(self):
        """Test non-silent audio is preserved."""
        audio = np.ones(1000, dtype=np.float32) * 0.5

        trimmed = trim_silence(audio, 24000)

        # Should preserve the audio
        assert len(trimmed) > 0

    def test_all_silence_returns_empty(self):
        """Test all-silent audio returns empty."""
        audio = np.zeros(1000, dtype=np.float32)

        trimmed = trim_silence(audio, 24000)

        assert len(trimmed) == 0

    def test_empty_audio(self):
        """Test empty audio returns empty."""
        audio = np.array([], dtype=np.float32)

        trimmed = trim_silence(audio, 24000)

        assert len(trimmed) == 0


# =============================================================================
# apply_gain Tests
# =============================================================================


class TestApplyGain:
    """Test apply_gain audio effect."""

    def test_positive_gain_increases_volume(self):
        """Test positive gain increases volume."""
        audio = np.ones(100, dtype=np.float32) * 0.5
        gained = apply_gain(audio, 6.0)  # +6dB

        # +6dB should approximately double the amplitude
        assert np.allclose(gained, 1.0, atol=0.01)

    def test_negative_gain_decreases_volume(self):
        """Test negative gain decreases volume."""
        audio = np.ones(100, dtype=np.float32) * 0.5
        gained = apply_gain(audio, -6.0)  # -6dB

        # -6dB should approximately halve the amplitude
        assert np.allclose(gained, 0.25, atol=0.01)

    def test_zero_gain_no_change(self):
        """Test zero gain makes no change."""
        audio = np.random.randn(100).astype(np.float32) * 0.5
        gained = apply_gain(audio, 0.0)

        np.testing.assert_array_almost_equal(audio, gained)


# =============================================================================
# stereo_to_mono Tests
# =============================================================================


class TestStereoToMono:
    """Test stereo_to_mono audio effect."""

    def test_converts_stereo_to_mono(self):
        """Test stereo audio is converted to mono."""
        stereo = np.random.randn(1000, 2).astype(np.float32)
        mono = stereo_to_mono(stereo)

        # Should be 1D array
        assert mono.ndim == 1
        assert len(mono) == 1000

    def test_averages_channels(self):
        """Test channels are averaged."""
        stereo = np.ones((100, 2), dtype=np.float32)
        stereo[:, 0] = 1.0  # Left
        stereo[:, 1] = -1.0  # Right

        mono = stereo_to_mono(stereo)

        # Average of 1.0 and -1.0 should be 0.0
        assert np.allclose(mono, 0.0)

    def test_mono_passes_through(self):
        """Test mono audio passes through unchanged."""
        audio = np.ones(100, dtype=np.float32)
        result = stereo_to_mono(audio)

        np.testing.assert_array_equal(audio, result)


# =============================================================================
# pad_audio Tests
# =============================================================================


class TestPadAudio:
    """Test pad_audio function."""

    def test_pad_silence_mode(self):
        """Test padding with silence mode."""
        audio = np.ones(500, dtype=np.float32)
        padded = pad_audio(audio, target_length=1000, mode="silence")

        assert len(padded) == 1000
        # Original audio should be at beginning
        assert np.allclose(padded[:500], 1.0)
        # Padding should be zeros
        assert np.allclose(padded[500:], 0.0)

    def test_pad_repeat_mode(self):
        """Test padding with repeat mode."""
        audio = np.ones(500, dtype=np.float32)
        padded = pad_audio(audio, target_length=1000, mode="repeat")

        assert len(padded) == 1000
        # Should loop the audio
        assert np.allclose(padded[:500], 1.0)
        assert np.allclose(padded[500:1000], 1.0)

    def test_truncates_if_too_long(self):
        """Test truncation when audio is longer than target."""
        audio = np.ones(1500, dtype=np.float32)
        padded = pad_audio(audio, target_length=1000)

        assert len(padded) == 1000

    def test_no_change_if_same_length(self):
        """Test no change when audio is already target length."""
        audio = np.ones(1000, dtype=np.float32)
        padded = pad_audio(audio, target_length=1000)

        np.testing.assert_array_equal(audio, padded)


# =============================================================================
# Integration Tests
# =============================================================================


class TestAudioEffectsIntegration:
    """Integration tests for audio effects."""

    def test_normalize_then_fade(self):
        """Test applying normalize then fade."""
        audio = np.random.randn(1000).astype(np.float32) * 0.3

        # Normalize first
        normalized = normalize(audio, target_peak=1.0)
        # Then fade in
        faded = fade_in(normalized, duration_samples=100)

        # Should be normalized
        assert np.max(np.abs(faded[100:])) == pytest.approx(1.0, abs=0.05)
        # Should have fade applied
        assert faded[0] == pytest.approx(0.0, abs=0.01)

    def test_mix_with_effects(self):
        """Test mixing tracks with applied effects."""
        track1 = np.ones(1000, dtype=np.float32)
        track2 = np.ones(1000, dtype=np.float32)

        # Apply effects
        track1 = fade_in(track1, duration_samples=100)
        track2 = fade_out(track2, duration_samples=100)

        # Mix
        mixed = mix_tracks([(track1, 1.0), (track2, 1.0)])

        assert len(mixed) == 1000
        # Mixed audio should be normalized
        assert np.max(np.abs(mixed)) <= 1.0

    def test_audio_pipeline(self):
        """Test a complete audio processing pipeline."""
        # Generate test audio
        audio = np.random.randn(24000).astype(np.float32) * 0.3

        # Apply effects
        audio = normalize(audio, target_peak=0.9)
        audio = fade_in(audio, duration_samples=2400)
        audio = fade_out(audio, duration_samples=2400)

        # Mix with another track
        track2 = np.sin(2 * np.pi * 880 * np.linspace(0, 1, 24000)).astype(np.float32)
        track2 = normalize(track2, target_peak=0.5)

        mixed = mix_tracks([(audio, 1.0), (track2, 0.5)])

        # Convert to base64
        b64 = audio_to_base64(mixed, 24000)

        # Convert back
        recovered, sr = base64_to_audio(b64)

        assert sr == 24000
        assert len(recovered) == len(mixed)
