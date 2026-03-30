"""Tests for Stitcher module.

Tests Track, Timeline, CompositionRequest schemas and related functionality
for audio composition and stitching.
"""

import base64
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from volsung.stitcher.schemas import (
    AudioSegment,
    CompositionJob,
    CompositionMetadata,
    CompositionRequest,
    CompositionResult,
    CompositionStatus,
    Timeline,
    Track,
    TrackType,
    TransitionType,
)


# =============================================================================
# TrackType Enum Tests
# =============================================================================


class TestTrackType:
    """Test TrackType enum."""

    def test_enum_values(self):
        """Test TrackType enum values."""
        assert TrackType.TTS.value == "tts"
        assert TrackType.MUSIC.value == "music"
        assert TrackType.SFX.value == "sfx"
        assert TrackType.VOICE.value == "voice"

    def test_string_comparison(self):
        """Test TrackType can be compared to strings."""
        assert TrackType.TTS == "tts"
        assert TrackType.MUSIC == "music"


# =============================================================================
# TransitionType Enum Tests
# =============================================================================


class TestTransitionType:
    """Test TransitionType enum."""

    def test_enum_values(self):
        """Test TransitionType enum values."""
        assert TransitionType.NONE.value == "none"
        assert TransitionType.FADE.value == "fade"
        assert TransitionType.CROSSFADE.value == "crossfade"
        assert TransitionType.OVERLAP.value == "overlap"


# =============================================================================
# AudioSegment Tests
# =============================================================================


class TestAudioSegment:
    """Test AudioSegment Pydantic model."""

    def test_valid_segment_creation(self, base64_audio: str):
        """Test creating a valid AudioSegment."""
        segment = AudioSegment(
            audio=base64_audio,
            duration=1.0,
            sample_rate=24000,
        )
        assert segment.audio == base64_audio
        assert segment.duration == 1.0
        assert segment.sample_rate == 24000

    def test_default_sample_rate(self, base64_audio: str):
        """Test default sample rate."""
        segment = AudioSegment(
            audio=base64_audio,
            duration=1.0,
        )
        assert segment.sample_rate == 24000

    def test_duration_validation(self, base64_audio: str):
        """Test duration must be positive."""
        with pytest.raises(ValidationError):
            AudioSegment(
                audio=base64_audio,
                duration=0.0,
                sample_rate=24000,
            )

        with pytest.raises(ValidationError):
            AudioSegment(
                audio=base64_audio,
                duration=-1.0,
                sample_rate=24000,
            )

    def test_sample_rate_bounds(self, base64_audio: str):
        """Test sample rate bounds."""
        # Too low
        with pytest.raises(ValidationError):
            AudioSegment(
                audio=base64_audio,
                duration=1.0,
                sample_rate=4000,
            )

        # Too high
        with pytest.raises(ValidationError):
            AudioSegment(
                audio=base64_audio,
                duration=1.0,
                sample_rate=96000,
            )

    def test_base64_validation(self):
        """Test audio must be valid base64."""
        with pytest.raises(ValidationError, match="valid base64"):
            AudioSegment(
                audio="not-valid-base64!!!",
                duration=1.0,
                sample_rate=24000,
            )

    def test_serialization(self, base64_audio: str):
        """Test AudioSegment serialization."""
        segment = AudioSegment(
            audio=base64_audio,
            duration=1.0,
            sample_rate=24000,
        )
        data = segment.model_dump()
        assert data["audio"] == base64_audio
        assert data["duration"] == 1.0
        assert data["sample_rate"] == 24000


# =============================================================================
# Track Tests
# =============================================================================


class TestTrack:
    """Test Track Pydantic model."""

    def test_valid_track_creation(self, audio_segment_data: dict):
        """Test creating a valid Track."""
        track = Track(
            track_id="track-001",
            track_type=TrackType.TTS,
            audio=audio_segment_data,
            start_time=0.0,
            end_time=5.0,
            volume=1.0,
        )
        assert track.track_id == "track-001"
        assert track.track_type == TrackType.TTS
        assert track.start_time == 0.0
        assert track.end_time == 5.0

    def test_track_with_audio_segment(self, base64_audio: str):
        """Test Track with AudioSegment object."""
        segment = AudioSegment(
            audio=base64_audio,
            duration=1.0,
            sample_rate=24000,
        )
        track = Track(
            track_id="track-002",
            track_type=TrackType.MUSIC,
            audio=segment,
            start_time=2.0,
        )
        assert track.track_type == TrackType.MUSIC
        assert track.start_time == 2.0

    def test_default_values(self, audio_segment_data: dict):
        """Test Track default values."""
        track = Track(
            track_id="track-003",
            track_type=TrackType.SFX,
            audio=audio_segment_data,
        )
        assert track.start_time == 0.0
        assert track.end_time is None
        assert track.volume == 1.0
        assert track.transition_in == TransitionType.FADE
        assert track.transition_out == TransitionType.FADE
        assert track.fade_in_duration == 0.5
        assert track.fade_out_duration == 0.5
        assert track.metadata == {}

    def test_volume_bounds(self, audio_segment_data: dict):
        """Test volume bounds (0.0-2.0)."""
        # Too low
        with pytest.raises(ValidationError):
            Track(
                track_id="track-004",
                track_type=TrackType.TTS,
                audio=audio_segment_data,
                volume=-0.1,
            )

        # Too high
        with pytest.raises(ValidationError):
            Track(
                track_id="track-005",
                track_type=TrackType.TTS,
                audio=audio_segment_data,
                volume=2.1,
            )

        # Valid boundaries
        track1 = Track(
            track_id="track-006",
            track_type=TrackType.TTS,
            audio=audio_segment_data,
            volume=0.0,
        )
        assert track1.volume == 0.0

        track2 = Track(
            track_id="track-007",
            track_type=TrackType.TTS,
            audio=audio_segment_data,
            volume=2.0,
        )
        assert track2.volume == 2.0

    def test_fade_duration_bounds(self, audio_segment_data: dict):
        """Test fade duration bounds (0-5 seconds)."""
        # Too long
        with pytest.raises(ValidationError):
            Track(
                track_id="track-008",
                track_type=TrackType.TTS,
                audio=audio_segment_data,
                fade_in_duration=6.0,
            )

        # Valid
        track = Track(
            track_id="track-009",
            track_type=TrackType.TTS,
            audio=audio_segment_data,
            fade_in_duration=3.0,
            fade_out_duration=4.0,
        )
        assert track.fade_in_duration == 3.0
        assert track.fade_out_duration == 4.0

    def test_start_time_must_be_non_negative(self, audio_segment_data: dict):
        """Test start_time must be >= 0."""
        with pytest.raises(ValidationError):
            Track(
                track_id="track-010",
                track_type=TrackType.TTS,
                audio=audio_segment_data,
                start_time=-1.0,
            )

    def test_end_time_after_start_time(self, audio_segment_data: dict):
        """Test end_time must be after start_time."""
        with pytest.raises(ValidationError, match="end_time must be greater"):
            Track(
                track_id="track-011",
                track_type=TrackType.TTS,
                audio=audio_segment_data,
                start_time=5.0,
                end_time=3.0,
            )

    def test_metadata_field(self, audio_segment_data: dict):
        """Test metadata field accepts arbitrary dict."""
        track = Track(
            track_id="track-012",
            track_type=TrackType.VOICE,
            audio=audio_segment_data,
            metadata={
                "speaker_id": "speaker-001",
                "emotion": "happy",
                "custom_key": "custom_value",
            },
        )
        assert track.metadata["speaker_id"] == "speaker-001"
        assert track.metadata["emotion"] == "happy"


# =============================================================================
# Timeline Tests
# =============================================================================


class TestTimeline:
    """Test Timeline Pydantic model."""

    def test_valid_timeline_creation(self, track_data: dict):
        """Test creating a valid Timeline."""
        timeline = Timeline(
            timeline_id="timeline-001",
            name="Test Timeline",
            duration=30.0,
            sample_rate=24000,
            tracks=[track_data],
        )
        assert timeline.timeline_id == "timeline-001"
        assert timeline.name == "Test Timeline"
        assert timeline.duration == 30.0
        assert len(timeline.tracks) == 1

    def test_default_values(self, track_data: dict):
        """Test Timeline default values."""
        timeline = Timeline(
            timeline_id="timeline-002",
            duration=60.0,
            tracks=[track_data],
        )
        assert timeline.name == "Untitled Timeline"
        assert timeline.sample_rate == 24000
        assert isinstance(timeline.created_at, datetime)
        assert isinstance(timeline.updated_at, datetime)

    def test_duration_must_be_positive(self, track_data: dict):
        """Test duration must be > 0."""
        with pytest.raises(ValidationError):
            Timeline(
                timeline_id="timeline-003",
                duration=0.0,
                tracks=[track_data],
            )

    def test_empty_tracks_allowed(self):
        """Test empty tracks list is allowed."""
        timeline = Timeline(
            timeline_id="timeline-004",
            duration=30.0,
            tracks=[],
        )
        assert timeline.tracks == []

    def test_sample_rate_bounds(self, track_data: dict):
        """Test sample rate bounds."""
        # Too low
        with pytest.raises(ValidationError):
            Timeline(
                timeline_id="timeline-005",
                duration=30.0,
                sample_rate=4000,
                tracks=[track_data],
            )

        # Too high
        with pytest.raises(ValidationError):
            Timeline(
                timeline_id="timeline-006",
                duration=30.0,
                sample_rate=96000,
                tracks=[track_data],
            )

    def test_track_start_after_timeline_end_validation(self, base64_audio: str):
        """Test track start time cannot exceed timeline duration."""
        track = Track(
            track_id="track-013",
            track_type=TrackType.TTS,
            audio=AudioSegment(
                audio=base64_audio,
                duration=1.0,
                sample_rate=24000,
            ),
            start_time=50.0,  # After timeline ends
        )

        with pytest.raises(ValidationError, match="starts after timeline ends"):
            Timeline(
                timeline_id="timeline-007",
                duration=30.0,
                tracks=[track],
            )

    def test_serialization(self, track_data: dict):
        """Test Timeline serialization."""
        timeline = Timeline(
            timeline_id="timeline-008",
            name="My Timeline",
            duration=60.0,
            tracks=[track_data],
        )
        data = timeline.model_dump()
        assert data["timeline_id"] == "timeline-008"
        assert data["name"] == "My Timeline"
        assert data["duration"] == 60.0


# =============================================================================
# CompositionRequest Tests
# =============================================================================


class TestCompositionRequest:
    """Test CompositionRequest Pydantic model."""

    def test_valid_request_creation(self, timeline_data: dict):
        """Test creating a valid CompositionRequest."""
        request = CompositionRequest(
            timeline=timeline_data,
            output_format="wav",
            normalize=True,
        )
        assert request.output_format == "wav"
        assert request.normalize is True

    def test_default_values(self, timeline_data: dict):
        """Test CompositionRequest default values."""
        request = CompositionRequest(
            timeline=timeline_data,
        )
        assert request.output_format == "wav"
        assert request.normalize is True
        assert request.normalize_target_db == -1.0
        assert request.request_id is None

    def test_output_format_validation(self, timeline_data: dict):
        """Test output_format must be valid."""
        # Valid formats
        for fmt in ["wav", "mp3", "ogg", "flac"]:
            request = CompositionRequest(
                timeline=timeline_data,
                output_format=fmt,
            )
            assert request.output_format == fmt

        # Invalid format
        with pytest.raises(ValidationError):
            CompositionRequest(
                timeline=timeline_data,
                output_format="invalid",
            )

    def test_normalize_target_db_bounds(self, timeline_data: dict):
        """Test normalize_target_db bounds."""
        # Too low
        with pytest.raises(ValidationError):
            CompositionRequest(
                timeline=timeline_data,
                normalize_target_db=-25.0,
            )

        # Too high
        with pytest.raises(ValidationError):
            CompositionRequest(
                timeline=timeline_data,
                normalize_target_db=1.0,
            )

        # Valid
        request = CompositionRequest(
            timeline=timeline_data,
            normalize_target_db=-10.0,
        )
        assert request.normalize_target_db == -10.0

    def test_with_timeline_object(self, base64_audio: str):
        """Test CompositionRequest with Timeline object."""
        segment = AudioSegment(
            audio=base64_audio,
            duration=1.0,
            sample_rate=24000,
        )
        track = Track(
            track_id="track-014",
            track_type=TrackType.TTS,
            audio=segment,
        )
        timeline = Timeline(
            timeline_id="timeline-009",
            duration=30.0,
            tracks=[track],
        )
        request = CompositionRequest(
            timeline=timeline,
            output_format="mp3",
        )
        assert request.timeline.timeline_id == "timeline-009"
        assert request.output_format == "mp3"


# =============================================================================
# CompositionMetadata Tests
# =============================================================================


class TestCompositionMetadata:
    """Test CompositionMetadata Pydantic model."""

    def test_valid_metadata_creation(self):
        """Test creating valid CompositionMetadata."""
        metadata = CompositionMetadata(
            duration=30.0,
            sample_rate=24000,
            num_tracks=3,
            peak_amplitude=0.95,
            processing_time_ms=1500.0,
            applied_normalization=True,
        )
        assert metadata.duration == 30.0
        assert metadata.num_tracks == 3
        assert metadata.applied_normalization is True

    def test_optional_normalization_gain(self):
        """Test normalization_gain_db is optional."""
        metadata = CompositionMetadata(
            duration=30.0,
            sample_rate=24000,
            num_tracks=1,
            peak_amplitude=0.8,
            processing_time_ms=500.0,
            applied_normalization=False,
        )
        assert metadata.normalization_gain_db is None


# =============================================================================
# CompositionResult Tests
# =============================================================================


class TestCompositionResult:
    """Test CompositionResult Pydantic model."""

    def test_valid_result_creation(self):
        """Test creating valid CompositionResult."""
        metadata = CompositionMetadata(
            duration=30.0,
            sample_rate=24000,
            num_tracks=2,
            peak_amplitude=0.9,
            processing_time_ms=1000.0,
            applied_normalization=True,
        )
        result = CompositionResult(
            audio="base64encodeddata...",
            sample_rate=24000,
            duration=30.0,
            format="wav",
            metadata=metadata,
        )
        assert result.audio == "base64encodeddata..."
        assert result.sample_rate == 24000
        assert result.format == "wav"
        assert result.metadata.num_tracks == 2


# =============================================================================
# CompositionStatus Tests
# =============================================================================


class TestCompositionStatus:
    """Test CompositionStatus enum."""

    def test_enum_values(self):
        """Test CompositionStatus enum values."""
        assert CompositionStatus.PENDING.value == "pending"
        assert CompositionStatus.PROCESSING.value == "processing"
        assert CompositionStatus.COMPLETED.value == "completed"
        assert CompositionStatus.FAILED.value == "failed"
        assert CompositionStatus.CANCELLED.value == "cancelled"


# =============================================================================
# CompositionJob Tests
# =============================================================================


class TestCompositionJob:
    """Test CompositionJob Pydantic model."""

    def test_valid_job_creation(self, composition_request_data: dict):
        """Test creating a valid CompositionJob."""
        job = CompositionJob(
            job_id="job-001",
            status=CompositionStatus.PENDING,
            request=composition_request_data,
        )
        assert job.job_id == "job-001"
        assert job.status == CompositionStatus.PENDING
        assert job.progress_percent == 0.0

    def test_default_values(self, composition_request_data: dict):
        """Test CompositionJob default values."""
        job = CompositionJob(
            job_id="job-002",
            status=CompositionStatus.PROCESSING,
            request=composition_request_data,
        )
        assert job.result is None
        assert job.error is None
        assert job.started_at is None
        assert job.completed_at is None
        assert job.progress_percent == 0.0

    def test_progress_bounds(self, composition_request_data: dict):
        """Test progress_percent bounds (0-100)."""
        # Too low
        with pytest.raises(ValidationError):
            CompositionJob(
                job_id="job-003",
                status=CompositionStatus.PROCESSING,
                request=composition_request_data,
                progress_percent=-10.0,
            )

        # Too high
        with pytest.raises(ValidationError):
            CompositionJob(
                job_id="job-004",
                status=CompositionStatus.PROCESSING,
                request=composition_request_data,
                progress_percent=150.0,
            )

        # Valid boundaries
        job1 = CompositionJob(
            job_id="job-005",
            status=CompositionStatus.PROCESSING,
            request=composition_request_data,
            progress_percent=0.0,
        )
        assert job1.progress_percent == 0.0

        job2 = CompositionJob(
            job_id="job-006",
            status=CompositionStatus.PROCESSING,
            request=composition_request_data,
            progress_percent=100.0,
        )
        assert job2.progress_percent == 100.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestStitcherIntegration:
    """Integration tests for stitcher schemas."""

    def test_complete_composition_workflow(self, base64_audio: str):
        """Test a complete composition workflow."""
        # Create audio segments
        tts_segment = AudioSegment(
            audio=base64_audio,
            duration=5.0,
            sample_rate=24000,
        )
        music_segment = AudioSegment(
            audio=base64_audio,
            duration=30.0,
            sample_rate=24000,
        )
        sfx_segment = AudioSegment(
            audio=base64_audio,
            duration=2.0,
            sample_rate=24000,
        )

        # Create tracks
        tts_track = Track(
            track_id="tts-001",
            track_type=TrackType.TTS,
            audio=tts_segment,
            start_time=0.0,
            end_time=5.0,
            fade_in_duration=0.1,
            fade_out_duration=0.1,
            metadata={"speaker_id": "speaker-001"},
        )
        music_track = Track(
            track_id="music-001",
            track_type=TrackType.MUSIC,
            audio=music_segment,
            start_time=0.0,
            end_time=30.0,
            volume=0.3,
            fade_in_duration=2.0,
            fade_out_duration=2.0,
        )
        sfx_track = Track(
            track_id="sfx-001",
            track_type=TrackType.SFX,
            audio=sfx_segment,
            start_time=3.0,
            end_time=5.0,
            volume=0.8,
        )

        # Create timeline
        timeline = Timeline(
            timeline_id="timeline-complete-001",
            name="Complete Test Timeline",
            duration=30.0,
            sample_rate=24000,
            tracks=[tts_track, music_track, sfx_track],
        )

        # Create composition request
        request = CompositionRequest(
            timeline=timeline,
            output_format="wav",
            normalize=True,
            normalize_target_db=-1.0,
            request_id="req-001",
        )

        # Create composition job
        job = CompositionJob(
            job_id="job-complete-001",
            status=CompositionStatus.PENDING,
            request=request,
        )

        # Verify everything is linked correctly
        assert job.request.timeline.timeline_id == "timeline-complete-001"
        assert len(job.request.timeline.tracks) == 3
        assert job.request.timeline.tracks[0].track_type == TrackType.TTS

    def test_timeline_with_multiple_track_types(self, base64_audio: str):
        """Test timeline with different track types."""
        segment = AudioSegment(
            audio=base64_audio,
            duration=10.0,
            sample_rate=24000,
        )

        tracks = [
            Track(
                track_id=f"track-{i}",
                track_type=tt,
                audio=segment,
                start_time=i * 2.0,
            )
            for i, tt in enumerate(
                [
                    TrackType.TTS,
                    TrackType.MUSIC,
                    TrackType.SFX,
                    TrackType.VOICE,
                ]
            )
        ]

        timeline = Timeline(
            timeline_id="multi-type-timeline",
            duration=20.0,
            tracks=tracks,
        )

        assert timeline.tracks[0].track_type == TrackType.TTS
        assert timeline.tracks[1].track_type == TrackType.MUSIC
        assert timeline.tracks[2].track_type == TrackType.SFX
        assert timeline.tracks[3].track_type == TrackType.VOICE

    def test_serialization_roundtrip(self, base64_audio: str):
        """Test serialization and deserialization roundtrip."""
        segment = AudioSegment(
            audio=base64_audio,
            duration=5.0,
            sample_rate=24000,
        )
        track = Track(
            track_id="track-rt",
            track_type=TrackType.TTS,
            audio=segment,
            start_time=0.0,
        )
        timeline = Timeline(
            timeline_id="timeline-rt",
            duration=10.0,
            tracks=[track],
        )
        request = CompositionRequest(
            timeline=timeline,
            output_format="wav",
        )

        # Serialize
        data = request.model_dump()

        # Deserialize
        restored = CompositionRequest.model_validate(data)

        assert restored.timeline.timeline_id == "timeline-rt"
        assert restored.timeline.tracks[0].track_id == "track-rt"
        assert restored.output_format == "wav"
