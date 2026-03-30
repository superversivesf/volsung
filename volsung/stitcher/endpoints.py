"""
FastAPI router for the Stitcher module.

Provides REST endpoints for audio composition operations.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from volsung.stitcher.schemas import (
    CompositionRequest,
    CompositionResult,
    CompositionJob,
    CompositionStatus,
    Timeline,
    Track,
)
from volsung.stitcher.composer import AudioComposer

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/stitcher",
    tags=["stitcher"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"},
    },
)

# Global composer instance (lazy-loaded)
_composer: AudioComposer | None = None


def get_composer() -> AudioComposer:
    """Get or create the global AudioComposer instance."""
    global _composer
    if _composer is None:
        _composer = AudioComposer()
    return _composer


class ComposeResponse(BaseModel):
    """Response wrapper for compose endpoint."""

    success: bool = Field(description="Whether the composition was successful")
    data: CompositionResult | None = Field(
        default=None, description="Composition result if successful"
    )
    error: str | None = Field(
        default=None, description="Error message if not successful"
    )


class JobStatusResponse(BaseModel):
    """Response for job status endpoint."""

    job_id: str = Field(description="Job identifier")
    status: CompositionStatus = Field(description="Current job status")
    progress_percent: float = Field(description="Completion percentage (0-100)")
    result: CompositionResult | None = Field(
        default=None, description="Result if job is complete"
    )
    error: str | None = Field(default=None, description="Error message if job failed")


class HealthResponse(BaseModel):
    """Health check response for stitcher module."""

    status: str = Field(description="Overall status: healthy, degraded, or unhealthy")
    composer_ready: bool = Field(description="Whether the composer is initialized")
    max_tracks: int = Field(description="Maximum tracks supported per composition")
    supported_formats: list[str] = Field(description="List of supported output formats")


# ============================================================================
# Stitcher Endpoints
# ============================================================================


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Check stitcher module health status.

    Returns information about the composer's readiness and capabilities.
    """
    composer = get_composer()

    return HealthResponse(
        status="healthy",
        composer_ready=True,
        max_tracks=composer.max_tracks,
        supported_formats=["wav", "mp3", "ogg", "flac"],
    )


@router.post(
    "/compose",
    response_model=CompositionResult,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Invalid request"},
        422: {"description": "Validation error"},
        500: {"description": "Composition failed"},
    },
)
async def compose(request: CompositionRequest) -> CompositionResult:
    """
    Compose audio from a timeline of tracks.

    Takes a timeline containing multiple audio tracks (TTS, music, SFX)
    and mixes them into a single output audio file with proper
    timing, transitions, and volume control.

    Example request:
        ```json
        {
            "timeline": {
                "timeline_id": "demo-scene-1",
                "name": "Scene 1",
                "duration": 30.0,
                "sample_rate": 24000,
                "tracks": [
                    {
                        "track_id": "music-bg",
                        "track_type": "music",
                        "audio": {
                            "audio": "<base64-audio-data>",
                            "duration": 30.0,
                            "sample_rate": 24000
                        },
                        "start_time": 0.0,
                        "volume": 0.5
                    },
                    {
                        "track_id": "tts-narration",
                        "track_type": "tts",
                        "audio": {
                            "audio": "<base64-audio-data>",
                            "duration": 5.0,
                            "sample_rate": 24000
                        },
                        "start_time": 2.0,
                        "volume": 1.0,
                        "fade_in_duration": 0.3,
                        "fade_out_duration": 0.3
                    }
                ]
            },
            "output_format": "wav",
            "normalize": true,
            "normalize_target_db": -1.0
        }
        ```

    Args:
        request: Composition request with timeline and options

    Returns:
        CompositionResult with the composed audio and metadata

    Raises:
        HTTPException: If composition fails or request is invalid
    """
    logger.info(
        f"[STITCHER] Compose request: timeline_id={request.timeline.timeline_id}, "
        f"tracks={len(request.timeline.tracks)}, "
        f"duration={request.timeline.duration}s"
    )

    try:
        composer = get_composer()
        result = composer.compose(request)

        logger.info(
            f"[STITCHER] Compose complete: request_id={result.request_id}, "
            f"processing_time={result.metadata.processing_time_ms:.1f}ms"
        )

        return result

    except ValueError as e:
        logger.warning(f"[STITCHER] Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid request: {str(e)}"
        )

    except Exception as e:
        logger.error(f"[STITCHER] Composition failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Composition failed: {str(e)}",
        )


@router.post(
    "/compose/async",
    response_model=JobStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"description": "Invalid request"},
        422: {"description": "Validation error"},
    },
)
async def compose_async(request: CompositionRequest) -> JobStatusResponse:
    """
    Queue an async composition job.

    Submits a composition request for background processing.
    Returns a job ID that can be used to check status and retrieve results.

    **Note**: This is a placeholder endpoint. Async processing
    will be implemented in a future update.

    Args:
        request: Composition request with timeline and options

    Returns:
        JobStatusResponse with job ID and initial status
    """
    # TODO: Implement actual async job queueing
    # For now, this is a placeholder that returns a mock job

    import uuid

    job_id = str(uuid.uuid4())

    logger.info(f"[STITCHER] Async compose queued: job_id={job_id}")

    return JobStatusResponse(
        job_id=job_id,
        status=CompositionStatus.PENDING,
        progress_percent=0.0,
        result=None,
        error=None,
    )


@router.get(
    "/compose/status/{job_id}",
    response_model=JobStatusResponse,
    responses={
        404: {"description": "Job not found"},
    },
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """
    Get the status of an async composition job.

    Returns the current status, progress, and result (if complete)
    for a previously submitted async composition job.

    **Note**: This is a placeholder endpoint. Async processing
    will be implemented in a future update.

    Args:
        job_id: Job identifier returned by /compose/async

    Returns:
        JobStatusResponse with current job status

    Raises:
        HTTPException: If job not found
    """
    # TODO: Implement actual job status retrieval
    # For now, this is a placeholder that returns "not found"

    logger.warning(f"[STITCHER] Job status requested for non-existent job: {job_id}")

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found"
    )


@router.delete(
    "/compose/cancel/{job_id}",
    status_code=status.HTTP_200_OK,
    responses={
        404: {"description": "Job not found"},
        409: {"description": "Job cannot be cancelled"},
    },
)
async def cancel_job(job_id: str) -> dict[str, Any]:
    """
    Cancel a pending or running composition job.

    **Note**: This is a placeholder endpoint. Async processing
    will be implemented in a future update.

    Args:
        job_id: Job identifier to cancel

    Returns:
        Confirmation of cancellation

    Raises:
        HTTPException: If job not found or cannot be cancelled
    """
    # TODO: Implement actual job cancellation

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found"
    )


@router.post(
    "/validate",
    response_model=dict[str, Any],
    status_code=status.HTTP_200_OK,
)
async def validate_timeline(request: CompositionRequest) -> dict[str, Any]:
    """
    Validate a timeline without performing composition.

    Checks if the timeline is valid and would succeed if composed.
    Useful for client-side validation before submitting actual composition.

    Args:
        request: Composition request to validate

    Returns:
        Validation result with any errors or warnings
    """
    logger.info(
        f"[STITCHER] Validate request: timeline_id={request.timeline.timeline_id}"
    )

    try:
        composer = get_composer()
        composer._validate_request(request)

        # Additional validation
        warnings = []

        # Check for overlapping tracks
        tracks = sorted(request.timeline.tracks, key=lambda t: t.start_time)
        for i, track in enumerate(tracks):
            track_end = track.end_time or (track.start_time + track.audio.duration)
            for other in tracks[i + 1 :]:
                other_end = other.end_time or (other.start_time + other.audio.duration)
                if track.start_time < other_end and other.start_time < track_end:
                    warnings.append(
                        f"Tracks '{track.track_id}' and '{other.track_id}' overlap"
                    )

        return {
            "valid": True,
            "timeline_id": request.timeline.timeline_id,
            "num_tracks": len(request.timeline.tracks),
            "duration": request.timeline.duration,
            "warnings": warnings if warnings else None,
        }

    except ValueError as e:
        return {
            "valid": False,
            "timeline_id": request.timeline.timeline_id,
            "error": str(e),
        }


# Export the router
__all__ = ["router"]
