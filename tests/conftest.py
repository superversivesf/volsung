"""Pytest fixtures for Volsung tests.

Provides shared fixtures for mocking models, generating test data,
and setting up test environments.
"""

import base64
import io
from datetime import datetime
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf


# =============================================================================
# Audio Fixtures
# =============================================================================


@pytest.fixture
def sample_rate() -> int:
    """Default sample rate for test audio."""
    return 24000


@pytest.fixture
def sample_duration() -> float:
    """Default duration for test audio in seconds."""
    return 1.0


@pytest.fixture
def audio_array(sample_rate: int, sample_duration: float) -> np.ndarray:
    """Generate a simple sine wave as test audio.

    Returns:
        Numpy array containing a 1-second sine wave at 440Hz.
    """
    t = np.linspace(0, sample_duration, int(sample_rate * sample_duration))
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture
def stereo_audio_array(sample_rate: int, sample_duration: float) -> np.ndarray:
    """Generate a stereo test audio array.

    Returns:
        Numpy array of shape (samples, 2) containing stereo audio.
    """
    t = np.linspace(0, sample_duration, int(sample_rate * sample_duration))
    left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    right = np.sin(2 * np.pi * 880 * t).astype(np.float32)
    return np.stack([left, right], axis=1)


@pytest.fixture
def base64_audio(audio_array: np.ndarray, sample_rate: int) -> str:
    """Convert audio array to base64-encoded WAV string.

    Returns:
        Base64-encoded WAV audio data.
    """
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format="WAV")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()


@pytest.fixture
def invalid_base64() -> str:
    """Invalid base64 string for testing validation errors."""
    return "not-valid-base64!!!"


# =============================================================================
# Model Config Fixtures
# =============================================================================


@pytest.fixture
def model_config() -> dict:
    """Default model configuration for tests.

    Returns:
        Dictionary containing ModelConfig-compatible settings.
    """
    return {
        "model_id": "test-model",
        "model_name": "Test Model",
        "device": "cpu",
        "dtype": "float32",
        "idle_timeout_seconds": 300,
    }


@pytest.fixture
def model_config_no_timeout() -> dict:
    """Model configuration with no idle timeout.

    Returns:
        Dictionary containing ModelConfig with timeout disabled.
    """
    return {
        "model_id": "test-model",
        "model_name": "Test Model",
        "device": "cpu",
        "dtype": "float32",
        "idle_timeout_seconds": 0,
    }


# =============================================================================
# Mock Generator Fixtures
# =============================================================================


@pytest.fixture
def mock_generator() -> MagicMock:
    """Create a mock generator for testing.

    Returns:
        Mock object simulating a GeneratorBase implementation.
    """
    mock = MagicMock()
    mock.model_id = "mock-generator"
    mock.model_name = "Mock Generator"
    mock.required_vram_gb = 4.0
    mock.model = None

    def mock_load(device: str, dtype: str) -> None:
        mock.model = MagicMock()
        mock._device = device
        mock._dtype = dtype

    def mock_unload() -> None:
        mock.model = None

    def mock_generate(prompt: str, duration: float, **kwargs) -> tuple:
        sample_rate = 24000
        samples = int(sample_rate * duration)
        audio = np.random.randn(samples).astype(np.float32) * 0.1
        return audio, sample_rate

    def mock_get_info() -> dict:
        return {
            "model_id": mock.model_id,
            "model_name": mock.model_name,
            "required_vram_gb": mock.required_vram_gb,
            "loaded": mock.model is not None,
        }

    mock.load.side_effect = mock_load
    mock.unload.side_effect = mock_unload
    mock.generate.side_effect = mock_generate
    mock.get_info.side_effect = mock_get_info

    return mock


@pytest.fixture
def mock_music_generator(mock_generator: MagicMock) -> MagicMock:
    """Create a mock music generator."""
    mock_generator.model_id = "musicgen-small"
    mock_generator.model_name = "MusicGen Small"
    mock_generator.required_vram_gb = 6.0
    return mock_generator


@pytest.fixture
def mock_sfx_generator(mock_generator: MagicMock) -> MagicMock:
    """Create a mock SFX generator."""
    mock_generator.model_id = "audioldm2-base"
    mock_generator.model_name = "AudioLDM2 Base"
    mock_generator.required_vram_gb = 4.0
    return mock_generator


@pytest.fixture
def mock_tts_model() -> MagicMock:
    """Create a mock TTS model.

    Returns:
        Mock object simulating a Qwen3-TTS model.
    """
    mock = MagicMock()

    def mock_voice_design(text: str, language: str, instruct: str) -> tuple:
        sample_rate = 24000
        samples = int(sample_rate * 2)  # 2 seconds
        audio = np.random.randn(samples).astype(np.float32) * 0.1
        return [audio], sample_rate

    def mock_voice_clone(
        ref_audio: tuple, ref_text: str, text: str, language: str
    ) -> tuple:
        sample_rate = 24000
        samples = int(sample_rate * 2)
        audio = np.random.randn(samples).astype(np.float32) * 0.1
        return [audio], sample_rate

    mock.generate_voice_design.side_effect = mock_voice_design
    mock.generate_voice_clone.side_effect = mock_voice_clone

    return mock


# =============================================================================
# Schema Fixtures
# =============================================================================


@pytest.fixture
def voice_design_request_data() -> dict:
    """Valid VoiceDesignRequest data."""
    return {
        "text": "Hello, I am a test voice.",
        "language": "English",
        "instruct": "A warm, friendly voice",
    }


@pytest.fixture
def synthesize_request_data(base64_audio: str) -> dict:
    """Valid SynthesizeRequest data."""
    return {
        "ref_audio": base64_audio,
        "ref_text": "Hello, I am a test voice.",
        "text": "This is synthesized speech.",
        "language": "English",
    }


@pytest.fixture
def music_generate_request_data() -> dict:
    """Valid MusicGenerateRequest data."""
    return {
        "description": "Peaceful acoustic guitar",
        "duration": 10.0,
        "genre": "acoustic",
        "mood": "calm",
        "tempo": "slow",
        "top_k": 250,
        "top_p": 0.0,
        "temperature": 1.0,
    }


@pytest.fixture
def sfx_generate_request_data() -> dict:
    """Valid SFXGenerateRequest data."""
    return {
        "description": "Thunder rumbling",
        "duration": 5.0,
        "category": "nature",
        "num_inference_steps": 50,
        "guidance_scale": 3.5,
    }


@pytest.fixture
def audio_segment_data(base64_audio: str) -> dict:
    """Valid AudioSegment data."""
    return {
        "audio": base64_audio,
        "duration": 1.0,
        "sample_rate": 24000,
    }


@pytest.fixture
def track_data(audio_segment_data: dict) -> dict:
    """Valid Track data."""
    return {
        "track_id": "track-001",
        "track_type": "tts",
        "audio": audio_segment_data,
        "start_time": 0.0,
        "volume": 1.0,
    }


@pytest.fixture
def timeline_data(track_data: dict) -> dict:
    """Valid Timeline data."""
    return {
        "timeline_id": "timeline-001",
        "name": "Test Timeline",
        "duration": 10.0,
        "sample_rate": 24000,
        "tracks": [track_data],
    }


@pytest.fixture
def composition_request_data(timeline_data: dict) -> dict:
    """Valid CompositionRequest data."""
    return {
        "timeline": timeline_data,
        "output_format": "wav",
        "normalize": True,
    }


# =============================================================================
# Mock Environment Fixtures
# =============================================================================


@pytest.fixture
def mock_torch_cuda_available() -> Generator[MagicMock, None, None]:
    """Mock torch.cuda.is_available() to return True."""
    with patch("torch.cuda.is_available") as mock:
        mock.return_value = True
        yield mock


@pytest.fixture
def mock_torch_cuda_unavailable() -> Generator[MagicMock, None, None]:
    """Mock torch.cuda.is_available() to return False."""
    with patch("torch.cuda.is_available") as mock:
        mock.return_value = False
        yield mock


@pytest.fixture
def mock_mps_available() -> Generator[MagicMock, None, None]:
    """Mock torch.backends.mps.is_available() to return True."""
    with patch("torch.backends.mps.is_available") as mock:
        mock.return_value = True
        yield mock


@pytest.fixture
def mock_gc_collect() -> Generator[MagicMock, None, None]:
    """Mock garbage collection."""
    with patch("gc.collect") as mock:
        yield mock


@pytest.fixture(autouse=True)
def reset_logging() -> Generator[None, None, None]:
    """Reset logging handlers after each test to avoid duplicate handlers."""
    import logging

    # Store original handlers
    root_handlers = logging.root.handlers[:]

    yield

    # Restore original handlers
    logging.root.handlers = root_handlers
