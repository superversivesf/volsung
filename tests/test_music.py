"""Tests for Music module.

Tests MusicModelManager, MusicGenGenerator, MusicGenerateRequest/Response schemas,
and related functionality.
"""

import base64
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from volsung.models.base import ModelConfig
from volsung.models.types import AudioResult, AudioType
from volsung.music.manager import MusicModelManager
from volsung.music.schemas import (
    MusicGenerateRequest,
    MusicGenerateResponse,
    MusicInfoResponse,
    MusicMetadata,
)
from volsung.music.generators.musicgen import MusicGenGenerator


# =============================================================================
# MusicGenerateRequest Tests
# =============================================================================


class TestMusicGenerateRequest:
    """Test MusicGenerateRequest Pydantic model."""

    def test_valid_request_creation(self):
        """Test creating a valid MusicGenerateRequest."""
        request = MusicGenerateRequest(
            description="Peaceful acoustic guitar",
            duration=15.0,
            genre="acoustic",
            mood="calm",
            tempo="slow",
        )
        assert request.description == "Peaceful acoustic guitar"
        assert request.duration == 15.0
        assert request.genre == "acoustic"
        assert request.mood == "calm"
        assert request.tempo == "slow"

    def test_default_values(self):
        """Test default parameter values."""
        request = MusicGenerateRequest(description="Test music")
        assert request.duration == 10.0
        assert request.genre is None
        assert request.mood is None
        assert request.tempo is None
        assert request.top_k == 250
        assert request.top_p == 0.0
        assert request.temperature == 1.0

    def test_required_description(self):
        """Test description is required."""
        with pytest.raises(ValidationError):
            MusicGenerateRequest(duration=10.0)

    def test_description_length_validation(self):
        """Test description length constraints."""
        # Too short
        with pytest.raises(ValidationError):
            MusicGenerateRequest(description="")

        # Too long
        with pytest.raises(ValidationError):
            MusicGenerateRequest(description="x" * 1001)

    def test_duration_bounds(self):
        """Test duration bounds (1-30 seconds)."""
        # Too short
        with pytest.raises(ValidationError):
            MusicGenerateRequest(description="Test", duration=0.5)

        # Too long
        with pytest.raises(ValidationError):
            MusicGenerateRequest(description="Test", duration=31.0)

        # Valid boundaries
        request1 = MusicGenerateRequest(description="Test", duration=1.0)
        assert request1.duration == 1.0

        request2 = MusicGenerateRequest(description="Test", duration=30.0)
        assert request2.duration == 30.0

    def test_tempo_validation(self):
        """Test tempo field validation."""
        # Valid values
        for tempo in ["slow", "medium", "fast", "SLOW", "Medium", "FAST"]:
            request = MusicGenerateRequest(description="Test", tempo=tempo)
            assert request.tempo == tempo.lower()

        # Invalid value
        with pytest.raises(ValidationError):
            MusicGenerateRequest(description="Test", tempo="veryfast")

    def test_top_k_bounds(self):
        """Test top_k bounds (1-1000)."""
        with pytest.raises(ValidationError):
            MusicGenerateRequest(description="Test", top_k=0)

        with pytest.raises(ValidationError):
            MusicGenerateRequest(description="Test", top_k=1001)

        request = MusicGenerateRequest(description="Test", top_k=500)
        assert request.top_k == 500

    def test_top_p_bounds(self):
        """Test top_p bounds (0-1)."""
        with pytest.raises(ValidationError):
            MusicGenerateRequest(description="Test", top_p=-0.1)

        with pytest.raises(ValidationError):
            MusicGenerateRequest(description="Test", top_p=1.1)

        request = MusicGenerateRequest(description="Test", top_p=0.5)
        assert request.top_p == 0.5

    def test_temperature_bounds(self):
        """Test temperature bounds (0.1-2.0)."""
        with pytest.raises(ValidationError):
            MusicGenerateRequest(description="Test", temperature=0.05)

        with pytest.raises(ValidationError):
            MusicGenerateRequest(description="Test", temperature=2.1)

        request = MusicGenerateRequest(description="Test", temperature=1.5)
        assert request.temperature == 1.5

    def test_serialization(self):
        """Test MusicGenerateRequest serialization."""
        request = MusicGenerateRequest(
            description="Peaceful guitar",
            duration=15.0,
            genre="acoustic",
        )
        data = request.model_dump()
        assert data["description"] == "Peaceful guitar"
        assert data["duration"] == 15.0
        assert data["genre"] == "acoustic"

    def test_json_schema(self):
        """Test JSON schema generation."""
        schema = MusicGenerateRequest.model_json_schema()
        assert "description" in schema["properties"]
        assert "duration" in schema["properties"]
        assert "genre" in schema["properties"]


# =============================================================================
# MusicMetadata Tests
# =============================================================================


class TestMusicMetadata:
    """Test MusicMetadata Pydantic model."""

    def test_valid_metadata_creation(self):
        """Test creating valid MusicMetadata."""
        metadata = MusicMetadata(
            duration=15.0,
            sample_rate=32000,
            generation_time_ms=2500.0,
            model_used="facebook/musicgen-small",
        )
        assert metadata.duration == 15.0
        assert metadata.sample_rate == 32000
        assert metadata.generation_time_ms == 2500.0

    def test_optional_fields(self):
        """Test optional fields."""
        metadata = MusicMetadata(
            duration=10.0,
            sample_rate=24000,
            generation_time_ms=1000.0,
            model_used="test-model",
        )
        assert metadata.genre_tags == []
        assert metadata.parameters == {}

    def test_serialization(self):
        """Test MusicMetadata serialization."""
        metadata = MusicMetadata(
            duration=15.0,
            sample_rate=32000,
            genre_tags=["acoustic", "calm"],
            generation_time_ms=2500.0,
            model_used="musicgen",
            parameters={"top_k": 250},
        )
        data = metadata.model_dump()
        assert data["genre_tags"] == ["acoustic", "calm"]
        assert data["parameters"]["top_k"] == 250


# =============================================================================
# MusicGenerateResponse Tests
# =============================================================================


class TestMusicGenerateResponse:
    """Test MusicGenerateResponse Pydantic model."""

    def test_valid_response_creation(self):
        """Test creating valid MusicGenerateResponse."""
        metadata = MusicMetadata(
            duration=15.0,
            sample_rate=32000,
            generation_time_ms=2500.0,
            model_used="musicgen",
        )
        response = MusicGenerateResponse(
            audio="base64encodeddata...",
            sample_rate=32000,
            metadata=metadata,
        )
        assert response.audio == "base64encodeddata..."
        assert response.sample_rate == 32000
        assert response.metadata.duration == 15.0

    def test_required_fields(self):
        """Test required fields."""
        metadata = MusicMetadata(
            duration=10.0,
            sample_rate=24000,
            generation_time_ms=1000.0,
            model_used="test",
        )
        # Missing audio
        with pytest.raises(ValidationError):
            MusicGenerateResponse(sample_rate=24000, metadata=metadata)

        # Missing metadata
        with pytest.raises(ValidationError):
            MusicGenerateResponse(audio="base64", sample_rate=24000)


# =============================================================================
# MusicInfoResponse Tests
# =============================================================================


class TestMusicInfoResponse:
    """Test MusicInfoResponse Pydantic model."""

    def test_valid_response_creation(self):
        """Test creating valid MusicInfoResponse."""
        response = MusicInfoResponse(
            status="ready",
            model_id="musicgen-small",
            model_name="MusicGen Small",
            is_loaded=True,
            device="cuda:0",
        )
        assert response.status == "ready"
        assert response.is_loaded is True
        assert response.device == "cuda:0"

    def test_optional_device(self):
        """Test device is optional."""
        response = MusicInfoResponse(
            status="ready",
            model_id="test",
            model_name="Test",
            is_loaded=False,
        )
        assert response.device is None


# =============================================================================
# MusicGenGenerator Tests (Mocked)
# =============================================================================


class TestMusicGenGenerator:
    """Test MusicGenGenerator with mocked audiocraft."""

    @pytest.fixture
    def mock_musicgen_class(self):
        """Create a mock MusicGen class."""
        mock_model = MagicMock()
        mock_model.set_generation_params = MagicMock()
        mock_model.generate.return_value = MagicMock()
        # Shape: (batch, channels, samples)
        mock_model.generate.return_value.cpu.return_value.numpy.return_value = (
            np.random.randn(1, 1, 320000).astype(np.float32) * 0.1
        )

        mock_musicgen = MagicMock()
        mock_musicgen.get_pretrained.return_value = mock_model

        return mock_musicgen, mock_model

    def test_generator_properties(self):
        """Test generator properties."""
        generator = MusicGenGenerator()
        assert generator.model_id == "musicgen-small"
        assert generator.model_name == "MusicGen Small"
        assert generator.required_vram_gb == 6.0

    def test_initial_state(self):
        """Test initial state of generator."""
        generator = MusicGenGenerator()
        assert generator.model is None
        assert generator._device is None
        assert generator._dtype is None

    def test_load_success(self, mock_musicgen_class):
        """Test successful model loading."""
        mock_musicgen, mock_model = mock_musicgen_class

        with patch("volsung.music.generators.musicgen.MusicGen", mock_musicgen):
            generator = MusicGenGenerator()
            generator.load("cpu", "float32")

            assert generator.model is mock_model
            assert generator._device == "cpu"
            assert generator._dtype == "float32"
            mock_musicgen.get_pretrained.assert_called_once_with("musicgen-small")

    def test_unload(self, mock_musicgen_class):
        """Test model unloading."""
        mock_musicgen, _ = mock_musicgen_class

        with patch("volsung.music.generators.musicgen.MusicGen", mock_musicgen):
            with patch("gc.collect"):
                with patch("torch.cuda.is_available", return_value=False):
                    generator = MusicGenGenerator()
                    generator.load("cpu", "float32")
                    assert generator.model is not None

                    generator.unload()
                    assert generator.model is None

    def test_generate_not_loaded(self):
        """Test generate raises error when not loaded."""
        generator = MusicGenGenerator()
        with pytest.raises(RuntimeError, match="not loaded"):
            generator.generate("test prompt", 10.0)

    def test_get_info_not_loaded(self):
        """Test get_info when not loaded."""
        generator = MusicGenGenerator()
        info = generator.get_info()

        assert info["model_id"] == "musicgen-small"
        assert info["loaded"] is False
        assert info["device"] is None

    def test_get_info_loaded(self, mock_musicgen_class):
        """Test get_info when loaded."""
        mock_musicgen, _ = mock_musicgen_class

        with patch("volsung.music.generators.musicgen.MusicGen", mock_musicgen):
            generator = MusicGenGenerator()
            generator.load("cuda:0", "bfloat16")

            info = generator.get_info()
            assert info["loaded"] is True
            assert info["device"] == "cuda:0"
            assert info["dtype"] == "bfloat16"


# =============================================================================
# MusicModelManager Tests (Mocked)
# =============================================================================


class TestMusicModelManager:
    """Test MusicModelManager with mocked generator."""

    @pytest.fixture
    def manager_with_mock(self):
        """Create a MusicModelManager with mocked generator."""
        config = ModelConfig(
            model_id="musicgen-test",
            model_name="MusicGen Test",
            device="cpu",
            dtype="float32",
            idle_timeout_seconds=0,
        )

        with patch("volsung.music.manager.MusicGenGenerator") as mock_gen_class:
            mock_generator = MagicMock()
            mock_generator.load = MagicMock()
            mock_generator.unload = MagicMock()
            mock_generator.generate.return_value = (
                np.random.randn(320000).astype(np.float32) * 0.1,
                32000,
            )
            mock_generator.model_id = "musicgen-small"

            mock_gen_class.return_value = mock_generator

            manager = MusicModelManager(config)
            yield manager, mock_generator

    def test_manager_initialization(self, manager_with_mock):
        """Test MusicModelManager initialization."""
        manager, _ = manager_with_mock
        assert manager.config.model_id == "musicgen-test"
        assert manager.config.model_name == "MusicGen Test"

    def test_generate_success(self, manager_with_mock):
        """Test successful music generation."""
        manager, mock_generator = manager_with_mock

        result = manager.generate(
            prompt="Peaceful acoustic guitar",
            duration=10.0,
        )

        assert isinstance(result, AudioResult)
        assert result.audio_type == AudioType.MUSIC
        assert result.generator == "musicgen-small"
        assert result.prompt == "Peaceful acoustic guitar"
        assert result.sample_rate == 32000
        mock_generator.load.assert_called_once()
        mock_generator.generate.assert_called_once()

    def test_generate_duration_validation(self, manager_with_mock):
        """Test duration validation in generate method."""
        manager, _ = manager_with_mock

        with pytest.raises(ValueError, match="Duration must be between"):
            manager.generate(prompt="Test", duration=0.5)

        with pytest.raises(ValueError, match="Duration must be between"):
            manager.generate(prompt="Test", duration=31.0)

    def test_generate_with_tags(self, manager_with_mock):
        """Test generation with genre/mood/tempo tags."""
        manager, mock_generator = manager_with_mock

        result = manager.generate(
            prompt="Peaceful guitar",
            duration=10.0,
            genre="acoustic",
            mood="calm",
            tempo="slow",
        )

        # Check that tags were included in enhanced prompt
        call_args = mock_generator.generate.call_args
        enhanced_prompt = call_args[1]["prompt"]
        assert "acoustic" in enhanced_prompt
        assert "calm" in enhanced_prompt
        assert "slow" in enhanced_prompt

    def test_generate_metadata(self, manager_with_mock):
        """Test generation includes correct metadata."""
        manager, _ = manager_with_mock

        result = manager.generate(
            prompt="Peaceful guitar",
            duration=10.0,
            genre="acoustic",
            mood="calm",
            tempo="slow",
            top_k=500,
            top_p=0.5,
            temperature=1.2,
        )

        assert result.metadata["genre"] == "acoustic"
        assert result.metadata["mood"] == "calm"
        assert result.metadata["tempo"] == "slow"
        assert result.metadata["parameters"]["top_k"] == 500
        assert result.metadata["parameters"]["top_p"] == 0.5
        assert result.metadata["parameters"]["temperature"] == 1.2

    def test_get_info_not_loaded(self, manager_with_mock):
        """Test get_info when model not loaded."""
        manager, _ = manager_with_mock

        info = manager.get_info()
        assert info["model_id"] == "musicgen-test"
        assert info["is_loaded"] is False
        assert info["device"] == "cpu"

    def test_get_info_loaded(self, manager_with_mock):
        """Test get_info when model is loaded."""
        manager, mock_generator = manager_with_mock

        # Trigger load
        manager.generate(prompt="Test", duration=5.0)

        info = manager.get_info()
        assert info["is_loaded"] is True
        assert "generator" in info

    def test_build_prompt(self, manager_with_mock):
        """Test prompt enhancement with tags."""
        manager, _ = manager_with_mock

        # Base prompt only
        prompt1 = manager._build_prompt("Peaceful guitar")
        assert prompt1 == "Peaceful guitar"

        # With genre
        prompt2 = manager._build_prompt("Guitar", genre="acoustic")
        assert "acoustic" in prompt2

        # With all tags
        prompt3 = manager._build_prompt(
            "Guitar", genre="acoustic", mood="calm", tempo="slow"
        )
        assert "Guitar" in prompt3
        assert "acoustic" in prompt3
        assert "calm" in prompt3
        assert "slow" in prompt3


class TestMusicModelManagerDeviceSelection:
    """Test device and dtype auto-selection."""

    def test_device_auto_selection_cuda(self):
        """Test CUDA device auto-selection."""
        config = ModelConfig(
            model_id="test",
            model_name="Test",
            device="auto",
            idle_timeout_seconds=0,
        )

        with patch("torch.cuda.is_available", return_value=True):
            with patch("volsung.music.manager.MusicGenGenerator") as mock_gen_class:
                mock_generator = MagicMock()
                mock_generator.load = MagicMock()
                mock_generator.generate.return_value = (
                    np.random.randn(320000).astype(np.float32) * 0.1,
                    32000,
                )
                mock_gen_class.return_value = mock_generator

                manager = MusicModelManager(config)
                manager.generate(prompt="Test", duration=5.0)

                # Check that cuda was selected
                call_args = mock_generator.load.call_args
                assert "cuda:0" in str(call_args)

    def test_device_auto_selection_mps(self):
        """Test MPS device auto-selection."""
        config = ModelConfig(
            model_id="test",
            model_name="Test",
            device="auto",
            idle_timeout_seconds=0,
        )

        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=True):
                with patch("volsung.music.manager.MusicGenGenerator") as mock_gen_class:
                    mock_generator = MagicMock()
                    mock_generator.load = MagicMock()
                    mock_generator.generate.return_value = (
                        np.random.randn(320000).astype(np.float32) * 0.1,
                        32000,
                    )
                    mock_gen_class.return_value = mock_generator

                    manager = MusicModelManager(config)
                    manager.generate(prompt="Test", duration=5.0)

                    call_args = mock_generator.load.call_args
                    assert "mps" in str(call_args)


# =============================================================================
# Import Error Tests
# =============================================================================


class TestMusicImportErrors:
    """Test handling of missing audiocraft package."""

    def test_audiocraft_import_error(self):
        """Test ImportError when audiocraft is not available."""
        with patch.dict("sys.modules", {"audiocraft": None}):
            with patch("audiocraft.models.MusicGen", side_effect=ImportError):
                generator = MusicGenGenerator()
                with pytest.raises(ImportError, match="audiocraft"):
                    generator.load("cpu", "float32")
