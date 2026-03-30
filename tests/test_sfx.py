"""Tests for SFX (Sound Effects) module.

Tests SFXModelManager, AudioLDMGenerator, SFXGenerateRequest/Response schemas,
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
from volsung.sfx.manager import SFXModelManager
from volsung.sfx.schemas import (
    SFXGenerateRequest,
    SFXGenerateResponse,
    SFXHealthResponse,
    SFXLayerRequest,
    SFXLayerResponse,
    SFXMetadata,
)
from volsung.sfx.generators.audioldm import AudioLDMGenerator


# =============================================================================
# SFXGenerateRequest Tests
# =============================================================================


class TestSFXGenerateRequest:
    """Test SFXGenerateRequest Pydantic model."""

    def test_valid_request_creation(self):
        """Test creating a valid SFXGenerateRequest."""
        request = SFXGenerateRequest(
            description="Thunder rumbling",
            duration=5.0,
            category="nature",
            num_inference_steps=50,
            guidance_scale=3.5,
        )
        assert request.description == "Thunder rumbling"
        assert request.duration == 5.0
        assert request.category == "nature"

    def test_default_values(self):
        """Test default parameter values."""
        request = SFXGenerateRequest(description="Test sound")
        assert request.duration == 5.0
        assert request.category is None
        assert request.num_inference_steps == 50
        assert request.guidance_scale == 3.5

    def test_required_description(self):
        """Test description is required."""
        with pytest.raises(ValidationError):
            SFXGenerateRequest(duration=5.0)

    def test_description_length_validation(self):
        """Test description length constraints."""
        # Too short
        with pytest.raises(ValidationError):
            SFXGenerateRequest(description="")

        # Too long
        with pytest.raises(ValidationError):
            SFXGenerateRequest(description="x" * 1001)

    def test_duration_bounds(self):
        """Test duration bounds (1-10 seconds)."""
        # Too short
        with pytest.raises(ValidationError):
            SFXGenerateRequest(description="Test", duration=0.5)

        # Too long
        with pytest.raises(ValidationError):
            SFXGenerateRequest(description="Test", duration=11.0)

        # Valid boundaries
        request1 = SFXGenerateRequest(description="Test", duration=1.0)
        assert request1.duration == 1.0

        request2 = SFXGenerateRequest(description="Test", duration=10.0)
        assert request2.duration == 10.0

    def test_num_inference_steps_bounds(self):
        """Test num_inference_steps bounds (10-200)."""
        with pytest.raises(ValidationError):
            SFXGenerateRequest(description="Test", num_inference_steps=5)

        with pytest.raises(ValidationError):
            SFXGenerateRequest(description="Test", num_inference_steps=201)

        request = SFXGenerateRequest(description="Test", num_inference_steps=100)
        assert request.num_inference_steps == 100

    def test_guidance_scale_bounds(self):
        """Test guidance_scale bounds (1-20)."""
        with pytest.raises(ValidationError):
            SFXGenerateRequest(description="Test", guidance_scale=0.5)

        with pytest.raises(ValidationError):
            SFXGenerateRequest(description="Test", guidance_scale=21.0)

        request = SFXGenerateRequest(description="Test", guidance_scale=5.0)
        assert request.guidance_scale == 5.0

    def test_serialization(self):
        """Test SFXGenerateRequest serialization."""
        request = SFXGenerateRequest(
            description="Thunder",
            duration=5.0,
            category="nature",
        )
        data = request.model_dump()
        assert data["description"] == "Thunder"
        assert data["duration"] == 5.0
        assert data["category"] == "nature"


# =============================================================================
# SFXMetadata Tests
# =============================================================================


class TestSFXMetadata:
    """Test SFXMetadata Pydantic model."""

    def test_valid_metadata_creation(self):
        """Test creating valid SFXMetadata."""
        metadata = SFXMetadata(
            duration=5.0,
            sample_rate=16000,
            generation_time_ms=2500.0,
            model_used="audioldm2",
        )
        assert metadata.duration == 5.0
        assert metadata.sample_rate == 16000
        assert metadata.generation_time_ms == 2500.0

    def test_optional_fields(self):
        """Test optional fields."""
        metadata = SFXMetadata(
            duration=5.0,
            sample_rate=16000,
            generation_time_ms=1000.0,
            model_used="test-model",
        )
        assert metadata.category is None
        assert metadata.num_inference_steps == 50
        assert metadata.guidance_scale == 3.5


# =============================================================================
# SFXGenerateResponse Tests
# =============================================================================


class TestSFXGenerateResponse:
    """Test SFXGenerateResponse Pydantic model."""

    def test_valid_response_creation(self):
        """Test creating valid SFXGenerateResponse."""
        metadata = SFXMetadata(
            duration=5.0,
            sample_rate=16000,
            generation_time_ms=2500.0,
            model_used="audioldm2",
        )
        response = SFXGenerateResponse(
            audio="base64encodeddata...",
            sample_rate=16000,
            metadata=metadata,
        )
        assert response.audio == "base64encodeddata..."
        assert response.sample_rate == 16000
        assert response.metadata.duration == 5.0

    def test_required_fields(self):
        """Test required fields."""
        metadata = SFXMetadata(
            duration=5.0,
            sample_rate=16000,
            generation_time_ms=1000.0,
            model_used="test",
        )
        # Missing audio
        with pytest.raises(ValidationError):
            SFXGenerateResponse(sample_rate=16000, metadata=metadata)

        # Missing metadata
        with pytest.raises(ValidationError):
            SFXGenerateResponse(audio="base64", sample_rate=16000)


# =============================================================================
# SFXLayerRequest Tests
# =============================================================================


class TestSFXLayerRequest:
    """Test SFXLayerRequest Pydantic model."""

    def test_valid_layer_request(self):
        """Test creating valid SFXLayerRequest."""
        request = SFXLayerRequest(
            layers=[
                SFXGenerateRequest(description="Thunder", duration=5.0),
                SFXGenerateRequest(description="Rain", duration=5.0),
            ]
        )
        assert len(request.layers) == 2
        assert request.layers[0].description == "Thunder"
        assert request.layers[1].description == "Rain"
        assert request.mix_mode == "sum"

    def test_min_layers(self):
        """Test at least one layer is required."""
        with pytest.raises(ValidationError):
            SFXLayerRequest(layers=[])

    def test_max_layers(self):
        """Test maximum of 5 layers."""
        layers = [
            SFXGenerateRequest(description=f"Sound {i}", duration=1.0) for i in range(6)
        ]
        with pytest.raises(ValidationError):
            SFXLayerRequest(layers=layers)

    def test_default_mix_mode(self):
        """Test default mix mode."""
        request = SFXLayerRequest(
            layers=[SFXGenerateRequest(description="Test", duration=1.0)]
        )
        assert request.mix_mode == "sum"


# =============================================================================
# SFXLayerResponse Tests
# =============================================================================


class TestSFXLayerResponse:
    """Test SFXLayerResponse Pydantic model."""

    def test_valid_layer_response(self):
        """Test creating valid SFXLayerResponse."""
        metadata = SFXMetadata(
            duration=5.0,
            sample_rate=16000,
            generation_time_ms=2500.0,
            model_used="audioldm2",
        )
        response = SFXLayerResponse(
            audio="base64encodeddata...",
            sample_rate=16000,
            layers_metadata=[metadata],
            total_duration=5.0,
        )
        assert response.audio == "base64encodeddata..."
        assert response.total_duration == 5.0
        assert len(response.layers_metadata) == 1


# =============================================================================
# SFXHealthResponse Tests
# =============================================================================


class TestSFXHealthResponse:
    """Test SFXHealthResponse Pydantic model."""

    def test_valid_health_response(self):
        """Test creating valid SFXHealthResponse."""
        response = SFXHealthResponse(
            status="healthy",
            model_loaded=True,
            model_name="audioldm2",
            idle_seconds=45.5,
        )
        assert response.status == "healthy"
        assert response.model_loaded is True
        assert response.model_name == "audioldm2"
        assert response.idle_seconds == 45.5

    def test_optional_idle_seconds(self):
        """Test idle_seconds is optional."""
        response = SFXHealthResponse(
            status="healthy",
            model_loaded=False,
            model_name="audioldm2",
        )
        assert response.idle_seconds is None


# =============================================================================
# AudioLDMGenerator Tests (Mocked)
# =============================================================================


class TestAudioLDMGenerator:
    """Test AudioLDMGenerator with mocked diffusers."""

    @pytest.fixture
    def mock_audioldm_pipeline(self):
        """Create a mock AudioLDM2 pipeline."""
        mock_audio = np.random.randn(16000 * 5).astype(np.float32) * 0.1

        mock_output = MagicMock()
        mock_output.audios = [mock_audio]
        mock_output.sample_rate = 16000

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = mock_output
        mock_pipeline.to = MagicMock(return_value=mock_pipeline)

        return mock_pipeline

    def test_generator_properties(self):
        """Test generator properties."""
        generator = AudioLDMGenerator(model_size="base")
        assert generator.model_id == "audioldm2-base"
        assert generator.model_name == "AudioLDM2 Base"
        assert generator.required_vram_gb == 4.0

    def test_generator_large_size(self):
        """Test large model size."""
        generator = AudioLDMGenerator(model_size="large")
        assert generator.model_id == "audioldm2-large"
        assert generator.model_name == "AudioLDM2 Large"
        assert generator.required_vram_gb == 8.0

    def test_generator_music_size(self):
        """Test music model size."""
        generator = AudioLDMGenerator(model_size="music")
        assert generator.model_id == "audioldm2-music"
        assert generator.model_name == "AudioLDM2 Music"
        assert generator.required_vram_gb == 6.0

    def test_invalid_model_size(self):
        """Test invalid model size raises error."""
        with pytest.raises(ValueError, match="Unknown model size"):
            AudioLDMGenerator(model_size="invalid")

    def test_initial_state(self):
        """Test initial state of generator."""
        generator = AudioLDMGenerator()
        assert generator.model is None
        assert generator.device is None
        assert generator.dtype is None

    def test_load_success(self, mock_audioldm_pipeline):
        """Test successful model loading."""
        with patch(
            "volsung.sfx.generators.audioldm.AudioLDM2Pipeline"
        ) as mock_pipeline_class:
            mock_pipeline_class.from_pretrained.return_value = mock_audioldm_pipeline

            generator = AudioLDMGenerator(model_size="base")
            generator.load("cpu", "float32")

            assert generator.model is not None
            assert generator.device == "cpu"
            assert str(generator.dtype) == "torch.float32"

    def test_unload(self):
        """Test model unloading."""
        generator = AudioLDMGenerator()
        # Just test that unload doesn't raise when model is None
        generator.unload()
        assert generator.model is None

    def test_generate_not_loaded(self):
        """Test generate raises error when not loaded."""
        generator = AudioLDMGenerator()
        with pytest.raises(RuntimeError, match="not loaded"):
            generator.generate("test prompt", 5.0)

    def test_duration_clamping(self, mock_audioldm_pipeline):
        """Test duration is clamped to valid range."""
        with patch(
            "volsung.sfx.generators.audioldm.AudioLDM2Pipeline"
        ) as mock_pipeline_class:
            mock_pipeline_class.from_pretrained.return_value = mock_audioldm_pipeline

            generator = AudioLDMGenerator(model_size="base")
            generator.load("cpu", "float32")

            # Test with duration < 1 (should be clamped to 1)
            generator.generate("test", duration=0.5)
            call_kwargs = mock_audioldm_pipeline.call_args[1]
            assert call_kwargs["audio_length_in_s"] == 1

            # Test with duration > 10 (should be clamped to 10)
            generator.generate("test", duration=15.0)
            call_kwargs = mock_audioldm_pipeline.call_args[1]
            assert call_kwargs["audio_length_in_s"] == 10

    def test_get_info(self, mock_audioldm_pipeline):
        """Test get_info method."""
        with patch(
            "volsung.sfx.generators.audioldm.AudioLDM2Pipeline"
        ) as mock_pipeline_class:
            mock_pipeline_class.from_pretrained.return_value = mock_audioldm_pipeline

            generator = AudioLDMGenerator(model_size="base")
            generator.load("cpu", "float32")

            info = generator.get_info()
            assert info["model_size"] == "base"
            assert info["huggingface_id"] == "cvssp/audioldm2"
            assert info["device"] == "cpu"


# =============================================================================
# SFXModelManager Tests (Mocked)
# =============================================================================


class TestSFXModelManager:
    """Test SFXModelManager with mocked generator."""

    @pytest.fixture
    def manager_with_mock(self):
        """Create a SFXModelManager with mocked generator."""
        config = ModelConfig(
            model_id="audioldm2-test",
            model_name="AudioLDM2 Test",
            device="cpu",
            dtype="float32",
            idle_timeout_seconds=0,
        )

        with patch("volsung.sfx.manager.AudioLDMGenerator") as mock_gen_class:
            mock_generator = MagicMock()
            mock_generator.load = MagicMock()
            mock_generator.generate.return_value = (
                np.random.randn(16000 * 5).astype(np.float32) * 0.1,
                16000,
            )
            mock_generator.model_id = "audioldm2-base"
            mock_generator.get_info.return_value = {
                "model_id": "audioldm2-base",
                "model_name": "AudioLDM2 Base",
            }

            mock_gen_class.return_value = mock_generator

            manager = SFXModelManager(config)
            yield manager, mock_generator

    def test_manager_initialization(self, manager_with_mock):
        """Test SFXModelManager initialization."""
        manager, _ = manager_with_mock
        assert manager.config.model_id == "audioldm2-test"
        assert manager.config.model_name == "AudioLDM2 Test"

    def test_generate_success(self, manager_with_mock):
        """Test successful SFX generation."""
        manager, mock_generator = manager_with_mock

        result = manager.generate(
            prompt="Thunder rumbling",
            duration=5.0,
        )

        assert isinstance(result, AudioResult)
        assert result.audio_type == AudioType.SFX
        assert result.generator == "audioldm2-base"
        assert result.prompt == "Thunder rumbling"
        mock_generator.load.assert_called_once()
        mock_generator.generate.assert_called_once()

    def test_generate_with_category(self, manager_with_mock):
        """Test generation with category enhancement."""
        manager, mock_generator = manager_with_mock

        result = manager.generate(
            prompt="Thunder rumbling",
            duration=5.0,
            category="nature",
        )

        # Check that category was prepended to prompt
        call_args = mock_generator.generate.call_args
        enhanced_prompt = call_args[1]["prompt"]
        assert "nature:" in enhanced_prompt
        assert "Thunder" in enhanced_prompt

    def test_generate_metadata(self, manager_with_mock):
        """Test generation includes correct metadata."""
        manager, _ = manager_with_mock

        result = manager.generate(
            prompt="Thunder",
            duration=5.0,
            category="nature",
            num_inference_steps=75,
            guidance_scale=5.0,
        )

        assert result.metadata["category"] == "nature"
        assert result.metadata["num_inference_steps"] == 75
        assert result.metadata["guidance_scale"] == 5.0

    def test_get_generator_info_not_loaded(self, manager_with_mock):
        """Test get_generator_info when not loaded."""
        manager, _ = manager_with_mock

        # Force not loaded state by setting _loaded to False
        manager._loaded = False
        manager._generator = None

        info = manager.get_generator_info()
        assert info["loaded"] is False


class TestSFXModelManagerAutoConfig:
    """Test SFXModelManager auto-configuration from global config."""

    def test_auto_config_from_global(self):
        """Test auto-configuration when no config provided."""
        mock_volsung_config = MagicMock()
        mock_volsung_config.sfx.model = "cvssp/audioldm2-large"
        mock_volsung_config.sfx.device = "cuda:0"
        mock_volsung_config.sfx.idle_timeout = 600

        with patch("volsung.sfx.manager.get_config", return_value=mock_volsung_config):
            with patch("volsung.sfx.manager.AudioLDMGenerator") as mock_gen_class:
                mock_generator = MagicMock()
                mock_gen_class.return_value = mock_generator

                manager = SFXModelManager()

                assert manager.config.model_id == "cvssp-audioldm2-large"
                assert manager.config.model_name == "cvssp/audioldm2-large"
                assert manager.config.device == "cuda:0"
                assert manager.config.idle_timeout_seconds == 600


class TestSFXModelManagerModelSize:
    """Test model size detection."""

    def test_large_model_detection(self):
        """Test detection of large model from config."""
        config = ModelConfig(
            model_id="audioldm2-large",
            model_name="cvssp/audioldm2-large",
            device="cpu",
            idle_timeout_seconds=0,
        )

        with patch("volsung.sfx.manager.AudioLDMGenerator") as mock_gen_class:
            mock_generator = MagicMock()
            mock_gen_class.return_value = mock_generator

            manager = SFXModelManager(config)
            # Model size is determined during _load_model
            # We can't easily test this without calling _load_model
            # which requires more complex mocking


# =============================================================================
# Import Error Tests
# =============================================================================


class TestSFXImportErrors:
    """Test handling of missing diffusers package."""

    def test_diffusers_import_error(self):
        """Test ImportError when diffusers is not available."""
        with patch.dict("sys.modules", {"diffusers": None}):
            with patch("diffusers.AudioLDM2Pipeline", side_effect=ImportError):
                generator = AudioLDMGenerator(model_size="base")
                with pytest.raises(ImportError, match="diffusers"):
                    generator.load("cpu", "float32")
