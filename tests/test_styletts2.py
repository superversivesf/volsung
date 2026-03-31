"""Tests for StyleTTS 2 module.

Tests StyleTTS2Manager, StyleTTS2Generator, StyleTTSParams schema,
and related functionality.
"""

import base64
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch
from pydantic import ValidationError

from volsung.models.base import ModelConfig
from volsung.models.types import AudioResult, AudioType
from volsung.tts.generators.styletts2 import StyleTTS2Generator
from volsung.tts.managers.styletts2 import StyleTTS2Manager
from volsung.tts.schemas import StyleTTSParams


# =============================================================================
# StyleTTSParams Tests
# =============================================================================


class TestStyleTTSParams:
    """Test StyleTTSParams Pydantic model."""

    def test_valid_params_creation(self):
        """Test creating valid StyleTTSParams."""
        params = StyleTTSParams(
            embedding_scale=1.5,
            alpha=0.3,
            beta=0.7,
            diffusion_steps=10,
        )
        assert params.embedding_scale == 1.5
        assert params.alpha == 0.3
        assert params.beta == 0.7
        assert params.diffusion_steps == 10

    def test_default_values(self):
        """Test default parameter values."""
        params = StyleTTSParams()
        assert params.embedding_scale == 1.0
        assert params.alpha == 0.3
        assert params.beta == 0.7
        assert params.diffusion_steps == 10

    def test_embedding_scale_range(self):
        """Test embedding_scale bounds (1.0-10.0)."""
        # Too low
        with pytest.raises(ValidationError):
            StyleTTSParams(embedding_scale=0.5)

        # Too high
        with pytest.raises(ValidationError):
            StyleTTSParams(embedding_scale=10.5)

        # Valid boundaries
        params1 = StyleTTSParams(embedding_scale=1.0)
        assert params1.embedding_scale == 1.0

        params2 = StyleTTSParams(embedding_scale=10.0)
        assert params2.embedding_scale == 10.0

    def test_alpha_beta_range(self):
        """Test alpha and beta bounds (0.0-1.0)."""
        # Alpha too low
        with pytest.raises(ValidationError):
            StyleTTSParams(alpha=-0.1)

        # Alpha too high
        with pytest.raises(ValidationError):
            StyleTTSParams(alpha=1.1)

        # Beta too low
        with pytest.raises(ValidationError):
            StyleTTSParams(beta=-0.1)

        # Beta too high
        with pytest.raises(ValidationError):
            StyleTTSParams(beta=1.1)

        # Valid boundaries
        params1 = StyleTTSParams(alpha=0.0, beta=0.0)
        assert params1.alpha == 0.0
        assert params1.beta == 0.0

        params2 = StyleTTSParams(alpha=1.0, beta=1.0)
        assert params2.alpha == 1.0
        assert params2.beta == 1.0

    def test_diffusion_steps_range(self):
        """Test diffusion_steps bounds (3-20)."""
        # Too low
        with pytest.raises(ValidationError):
            StyleTTSParams(diffusion_steps=2)

        # Too high
        with pytest.raises(ValidationError):
            StyleTTSParams(diffusion_steps=21)

        # Valid boundaries
        params1 = StyleTTSParams(diffusion_steps=3)
        assert params1.diffusion_steps == 3

        params2 = StyleTTSParams(diffusion_steps=20)
        assert params2.diffusion_steps == 20

    def test_serialization(self):
        """Test StyleTTSParams serialization."""
        params = StyleTTSParams(
            embedding_scale=1.5,
            alpha=0.5,
            beta=0.5,
            diffusion_steps=15,
        )
        data = params.model_dump()
        assert data["embedding_scale"] == 1.5
        assert data["alpha"] == 0.5
        assert data["beta"] == 0.5
        assert data["diffusion_steps"] == 15


# =============================================================================
# StyleTTS2Generator Tests (Mocked)
# =============================================================================


class TestStyleTTS2Generator:
    """Test StyleTTS2Generator with mocked styletts2."""

    @pytest.fixture
    def mock_styletts2_model(self):
        """Create a mock StyleTTS2 model."""
        mock_model = MagicMock()

        # Mock inference to return a numpy array
        def mock_inference(text, target_voice_path, embedding_scale, alpha, beta):
            # Return a 2-second audio at 24000 Hz
            return np.random.randn(48000).astype(np.float32) * 0.1

        mock_model.inference.side_effect = mock_inference
        mock_model.to = MagicMock()

        return mock_model

    @pytest.fixture
    def mock_styletts2_module(self, mock_styletts2_model):
        """Create a mock styletts2 module."""
        mock_tts = MagicMock()
        mock_tts.StyleTTS2.return_value = mock_styletts2_model

        mock_module = MagicMock()
        mock_module.tts = mock_tts

        return mock_module

    def test_generator_properties(self):
        """Test generator properties."""
        generator = StyleTTS2Generator()
        assert generator.model_id == "styletts2"
        assert generator.model_name == "StyleTTS2"
        assert generator.required_vram_gb == 4.0

    def test_initial_state(self):
        """Test initial state of generator."""
        generator = StyleTTS2Generator()
        assert generator.model is None
        assert generator._device is None
        assert generator._dtype is None

    def test_load_unload(self, mock_styletts2_module):
        """Test model loading and unloading."""
        with patch.dict("sys.modules", {"styletts2": mock_styletts2_module}):
            with patch("styletts2.tts", mock_styletts2_module.tts):
                generator = StyleTTS2Generator()

                # Test load
                generator.load("cpu", "float32")
                assert generator.model is not None
                assert generator._device == "cpu"
                assert generator._dtype == "float32"

                # Test unload
                with patch("gc.collect"):
                    with patch("torch.cuda.is_available", return_value=False):
                        generator.unload()
                        assert generator.model is None
                        assert generator._device is None
                        assert generator._dtype is None

    def test_load_success(self, mock_styletts2_module):
        """Test successful model loading."""
        with patch.dict("sys.modules", {"styletts2": mock_styletts2_module}):
            with patch("styletts2.tts", mock_styletts2_module.tts):
                generator = StyleTTS2Generator()
                generator.load("cuda:0", "float16")

                assert generator.model is not None
                assert generator._device == "cuda:0"
                assert generator._dtype == "float16"
                mock_styletts2_module.tts.StyleTTS2.assert_called_once()

    def test_generate_mock(self, mock_styletts2_module, base64_audio):
        """Test generation with mocked model."""
        with patch.dict("sys.modules", {"styletts2": mock_styletts2_module}):
            with patch("styletts2.tts", mock_styletts2_module.tts):
                generator = StyleTTS2Generator()
                generator.load("cpu", "float32")

                audio, sr = generator.generate(
                    text="Hello, world!",
                    ref_audio=base64_audio,
                    embedding_scale=1.0,
                    alpha=0.3,
                    beta=0.7,
                    diffusion_steps=10,
                )

                assert isinstance(audio, np.ndarray)
                assert sr == 24000
                assert len(audio) > 0
                generator.model.inference.assert_called_once()

    def test_generate_with_emotion_params(self, mock_styletts2_module, base64_audio):
        """Test generation with different emotion parameters."""
        with patch.dict("sys.modules", {"styletts2": mock_styletts2_module}):
            with patch("styletts2.tts", mock_styletts2_module.tts):
                generator = StyleTTS2Generator()
                generator.load("cpu", "float32")

                # Test with high emotion
                audio1, sr1 = generator.generate(
                    text="Excited speech!",
                    ref_audio=base64_audio,
                    embedding_scale=5.0,
                    alpha=0.1,
                    beta=0.9,
                    diffusion_steps=15,
                )

                # Check inference was called with correct parameters
                call_kwargs = mock_styletts2_module.tts.StyleTTS2.return_value.inference.call_args[
                    1
                ]
                assert call_kwargs["embedding_scale"] == 5.0
                assert call_kwargs["alpha"] == 0.1
                assert call_kwargs["beta"] == 0.9

    def test_generate_not_loaded(self):
        """Test generate raises error when not loaded."""
        generator = StyleTTS2Generator()
        with pytest.raises(RuntimeError, match="not loaded"):
            generator.generate("test", "base64audio")

    def test_get_info_not_loaded(self):
        """Test get_info when not loaded."""
        generator = StyleTTS2Generator()
        info = generator.get_info()

        assert info["model_id"] == "styletts2"
        assert info["loaded"] is False
        assert info["device"] is None

    def test_get_info_loaded(self, mock_styletts2_module):
        """Test get_info when loaded."""
        with patch.dict("sys.modules", {"styletts2": mock_styletts2_module}):
            with patch("styletts2.tts", mock_styletts2_module.tts):
                generator = StyleTTS2Generator()
                generator.load("cuda:0", "bfloat16")

                info = generator.get_info()
                assert info["loaded"] is True
                assert info["device"] == "cuda:0"
                assert info["dtype"] == "bfloat16"
                assert info["sample_rate"] == 24000


# =============================================================================
# StyleTTS2Manager Tests (Mocked)
# =============================================================================


class TestStyleTTS2Manager:
    """Test StyleTTS2Manager with mocked generator."""

    @pytest.fixture
    def manager_with_mock(self):
        """Create a StyleTTS2Manager with mocked generator."""
        with patch("torch.cuda.is_available", return_value=False):
            manager = StyleTTS2Manager(
                device="cpu",
                dtype="float32",
                idle_timeout=0,
            )

            # Mock the generator
            mock_generator = MagicMock()
            mock_generator.load = MagicMock()
            mock_generator.unload = MagicMock()
            mock_generator.generate.return_value = (
                np.random.randn(48000).astype(np.float32) * 0.1,
                24000,
            )
            mock_generator.model_id = "styletts2"
            mock_generator.get_info.return_value = {
                "model_id": "styletts2",
                "model_name": "StyleTTS2",
            }

            manager._generator = mock_generator
            manager._loaded = True

            yield manager, mock_generator

    def test_manager_initialization(self, manager_with_mock):
        """Test StyleTTS2Manager initialization."""
        manager, _ = manager_with_mock
        assert manager._model_id == "styletts2"
        assert manager.device == "cpu"
        assert manager.dtype == "float32"

    def test_compute_style_mock(self, manager_with_mock, base64_audio):
        """Test compute_style with mocked style extraction."""
        manager, mock_generator = manager_with_mock

        # Mock compute_style method
        mock_style = torch.randn(1, 256)
        mock_generator.compute_style.return_value = mock_style

        style = manager.compute_style(base64_audio)

        assert isinstance(style, torch.Tensor)
        assert style.shape == (1, 256)
        mock_generator.compute_style.assert_called_once_with(base64_audio)

    def test_generate_with_emotion(self, manager_with_mock, base64_audio):
        """Test generation with emotion parameters."""
        manager, mock_generator = manager_with_mock

        result = manager.generate(
            text="Hello with emotion!",
            ref_audio=base64_audio,
            embedding_scale=2.0,
            alpha=0.2,
            beta=0.8,
            diffusion_steps=15,
        )

        assert isinstance(result, AudioResult)
        assert result.audio_type == AudioType.TTS
        assert result.generator == "styletts2"
        assert result.prompt == "Hello with emotion!"

        # Check metadata includes emotion params
        assert result.metadata["embedding_scale"] == 2.0
        assert result.metadata["alpha"] == 0.2
        assert result.metadata["beta"] == 0.8
        assert result.metadata["diffusion_steps"] == 15

    def test_generate_metadata(self, manager_with_mock, base64_audio):
        """Test generation includes correct metadata."""
        manager, _ = manager_with_mock

        result = manager.generate(
            text="Test text",
            ref_audio=base64_audio,
            embedding_scale=1.5,
            alpha=0.4,
            beta=0.6,
            diffusion_steps=12,
        )

        assert result.metadata["model"] == "styletts2"
        assert result.metadata["text"] == "Test text"
        assert result.metadata["embedding_scale"] == 1.5
        assert result.metadata["alpha"] == 0.4
        assert result.metadata["beta"] == 0.6
        assert result.metadata["diffusion_steps"] == 12

    def test_get_info_loaded(self, manager_with_mock):
        """Test get_info when model is loaded."""
        manager, _ = manager_with_mock

        info = manager.get_info()
        assert info["model_id"] == "styletts2"
        assert info["device"] == "cpu"
        assert info["dtype"] == "float32"
        assert info["loaded"] is True

    def test_get_info_not_loaded(self):
        """Test get_info when model is not loaded."""
        with patch("torch.cuda.is_available", return_value=False):
            manager = StyleTTS2Manager(
                device="cpu",
                dtype="float32",
                idle_timeout=0,
            )
            # Don't load the model
            manager._loaded = False
            manager._generator = None

            info = manager.get_info()
            assert info["model_id"] == "styletts2"
            assert info["loaded"] is False


class TestStyleTTS2ManagerDeviceSelection:
    """Test device and dtype auto-selection."""

    def test_device_auto_selection_cuda(self):
        """Test CUDA device auto-selection."""
        with patch("torch.cuda.is_available", return_value=True):
            manager = StyleTTS2Manager(idle_timeout=0)
            assert manager.device == "cuda:0"

    def test_device_auto_selection_mps(self):
        """Test MPS device auto-selection."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=True):
                manager = StyleTTS2Manager(idle_timeout=0)
                assert manager.device == "mps"

    def test_device_auto_selection_cpu(self):
        """Test CPU device auto-selection."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                manager = StyleTTS2Manager(idle_timeout=0)
                assert manager.device == "cpu"

    def test_dtype_auto_selection_cuda(self):
        """Test bfloat16 dtype auto-selection with CUDA."""
        with patch("torch.cuda.is_available", return_value=True):
            manager = StyleTTS2Manager(idle_timeout=0)
            assert manager.dtype == "bfloat16"

    def test_dtype_auto_selection_no_cuda(self):
        """Test float32 dtype auto-selection without CUDA."""
        with patch("torch.cuda.is_available", return_value=False):
            manager = StyleTTS2Manager(idle_timeout=0)
            assert manager.dtype == "float32"


# =============================================================================
# Import Error Tests
# =============================================================================


class TestStyleTTS2ImportErrors:
    """Test handling of missing styletts2 package."""

    def test_styletts2_import_error(self):
        """Test ImportError when styletts2 is not available."""
        with patch.dict("sys.modules", {"styletts2": None}):
            generator = StyleTTS2Generator()
            with pytest.raises(ImportError, match="styletts2"):
                generator.load("cpu", "float32")
