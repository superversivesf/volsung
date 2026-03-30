"""
Audio composer for the Stitcher module.

Provides the AudioComposer class for mixing and transitioning
between audio segments (TTS, music, SFX) into a unified output.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from volsung.stitcher.schemas import (
    CompositionRequest,
    CompositionResult,
    CompositionMetadata,
    Track,
    Timeline,
    TransitionType,
)

logger = logging.getLogger(__name__)


class AudioComposer:
    """
    Audio composer for mixing multiple tracks into a unified audio output.

    This class handles:
    - Track positioning and timing alignment
    - Volume control and normalization
    - Transitions (fade, crossfade, overlap)
    - Multi-track mixing
    - Output format conversion

    Example:
        >>> composer = AudioComposer()
        >>> result = composer.compose(request)
        >>> print(f"Composed {result.duration}s audio with {result.metadata.num_tracks} tracks")

    Attributes:
        sample_rate: Target sample rate for composition (Hz)
        max_tracks: Maximum number of tracks allowed per composition
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        max_tracks: int = 20,
    ):
        """
        Initialize the audio composer.

        Args:
            sample_rate: Target sample rate for composition (Hz)
            max_tracks: Maximum number of tracks allowed
        """
        self.sample_rate = sample_rate
        self.max_tracks = max_tracks
        logger.info(
            f"AudioComposer initialized: sample_rate={sample_rate}, max_tracks={max_tracks}"
        )

    def compose(self, request: CompositionRequest) -> CompositionResult:
        """
        Compose audio from a timeline of tracks.

        Mixes multiple audio tracks together, applying timing offsets,
        volume adjustments, and transitions as specified.

        Args:
            request: Composition request containing timeline and options

        Returns:
            CompositionResult with the composed audio and metadata

        Raises:
            ValueError: If timeline validation fails
            RuntimeError: If composition processing fails

        Example:
            >>> request = CompositionRequest(
            ...     timeline=Timeline(
            ...         timeline_id="demo",
            ...         duration=30.0,
            ...         tracks=[track1, track2],
            ...     )
            ... )
            >>> result = composer.compose(request)
        """
        start_time = time.time()
        logger.info(
            f"[COMPOSE] Starting composition: timeline_id={request.timeline.timeline_id}"
        )

        # Validate request
        self._validate_request(request)

        # TODO: Implement actual composition logic
        # This is a placeholder that returns silent audio
        duration = request.timeline.duration
        num_samples = int(duration * self.sample_rate)
        audio = np.zeros(num_samples, dtype=np.float32)

        # Apply normalization if requested
        peak = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
        normalization_gain_db = None

        if request.normalize and peak > 0:
            target_peak = 10 ** (request.normalize_target_db / 20)
            gain = target_peak / peak
            audio = audio * gain
            normalization_gain_db = 20 * np.log10(gain)
            logger.info(
                f"[COMPOSE] Applied normalization: {normalization_gain_db:.2f} dB"
            )

        # Convert to base64
        import base64
        from io import BytesIO
        import soundfile as sf

        buffer = BytesIO()
        sf.write(buffer, audio, self.sample_rate, format="WAV")
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode()

        # Build metadata
        processing_time_ms = (time.time() - start_time) * 1000
        metadata = CompositionMetadata(
            duration=duration,
            sample_rate=self.sample_rate,
            num_tracks=len(request.timeline.tracks),
            peak_amplitude=float(peak),
            processing_time_ms=processing_time_ms,
            applied_normalization=request.normalize,
            normalization_gain_db=normalization_gain_db,
        )

        result = CompositionResult(
            audio=audio_base64,
            sample_rate=self.sample_rate,
            duration=duration,
            format=request.output_format,
            metadata=metadata,
            request_id=request.request_id,
        )

        logger.info(
            f"[COMPOSE] Completed: duration={duration:.2f}s, "
            f"tracks={len(request.timeline.tracks)}, "
            f"processing_time={processing_time_ms:.1f}ms"
        )

        return result

    def _validate_request(self, request: CompositionRequest) -> None:
        """
        Validate a composition request.

        Args:
            request: Composition request to validate

        Raises:
            ValueError: If validation fails
        """
        timeline = request.timeline

        # Check track count
        if len(timeline.tracks) > self.max_tracks:
            raise ValueError(
                f"Timeline has {len(timeline.tracks)} tracks, "
                f"exceeds maximum of {self.max_tracks}"
            )

        # Check for duplicate track IDs
        track_ids = [t.track_id for t in timeline.tracks]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("Timeline contains duplicate track IDs")

        # Validate each track
        for track in timeline.tracks:
            if track.start_time >= timeline.duration:
                raise ValueError(
                    f"Track {track.track_id} starts at {track.start_time}s "
                    f"but timeline duration is {timeline.duration}s"
                )

        logger.debug(f"[COMPOSE] Request validated: {len(timeline.tracks)} tracks")

    def _apply_fade(
        self,
        audio: np.ndarray,
        fade_in_duration: float = 0.0,
        fade_out_duration: float = 0.0,
    ) -> np.ndarray:
        """
        Apply fade-in and fade-out to audio.

        Args:
            audio: Input audio array
            fade_in_duration: Fade-in duration in seconds
            fade_out_duration: Fade-out duration in seconds

        Returns:
            Audio with fades applied
        """
        if len(audio) == 0:
            return audio

        result = audio.copy()
        num_samples = len(result)

        # Fade in
        if fade_in_duration > 0:
            fade_in_samples = min(int(fade_in_duration * self.sample_rate), num_samples)
            fade_curve = np.linspace(0.0, 1.0, fade_in_samples)
            result[:fade_in_samples] *= fade_curve

        # Fade out
        if fade_out_duration > 0:
            fade_out_samples = min(
                int(fade_out_duration * self.sample_rate), num_samples
            )
            fade_curve = np.linspace(1.0, 0.0, fade_out_samples)
            result[-fade_out_samples:] *= fade_curve

        return result

    def _mix_tracks(
        self,
        tracks: list[tuple[np.ndarray, float, float]],
    ) -> np.ndarray:
        """
        Mix multiple audio tracks together.

        Args:
            tracks: List of (audio_array, start_time, volume) tuples

        Returns:
            Mixed audio array
        """
        # TODO: Implement proper multi-track mixing with crossfades
        # This is a placeholder that returns the first track or empty audio
        if not tracks:
            return np.array([], dtype=np.float32)

        # Find max duration
        max_duration = 0.0
        for audio, start_time, _ in tracks:
            end_time = start_time + len(audio) / self.sample_rate
            max_duration = max(max_duration, end_time)

        # Create output buffer
        num_samples = int(max_duration * self.sample_rate)
        output = np.zeros(num_samples, dtype=np.float32)

        # Mix each track
        for audio, start_time, volume in tracks:
            start_sample = int(start_time * self.sample_rate)
            end_sample = start_sample + len(audio)

            # Apply volume
            audio_with_volume = audio * volume

            # Add to output (handle bounds)
            actual_end = min(end_sample, num_samples)
            output[start_sample:actual_end] += audio_with_volume[
                : actual_end - start_sample
            ]

        return output

    def _normalize_audio(
        self,
        audio: np.ndarray,
        target_db: float = -1.0,
    ) -> tuple[np.ndarray, float]:
        """
        Normalize audio to target dB level.

        Args:
            audio: Input audio array
            target_db: Target peak level in dB (e.g., -1.0 dB)

        Returns:
            Tuple of (normalized_audio, gain_applied_in_db)
        """
        if len(audio) == 0:
            return audio, 0.0

        peak = np.max(np.abs(audio))
        if peak == 0:
            return audio, 0.0

        # Convert target dB to linear scale
        target_peak = 10 ** (target_db / 20)
        gain = target_peak / peak

        normalized = audio * gain
        gain_db = 20 * np.log10(gain)

        return normalized, gain_db


# Export the class
__all__ = ["AudioComposer"]
