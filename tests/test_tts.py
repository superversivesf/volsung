"""Tests for TTS (Text-to-Speech) module.

Tests TTSModelManager, VoiceDesignRequest, VoiceDesignResponse,
SynthesizeRequest, and SynthesizeResponse schemas.
"""

import base64
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from volsung.models.base import ModelConfig
from volsung.models.types import AudioResult, AudioType
from volsung.tts.manager import TTSModelManager
from volsung.tts.schemas import (
    SynthesizeRequest,
    SynthesizeResponse,
    VoiceDesignRequest,
    VoiceDesignResponse,
)


# =============================================================================
# VoiceDesignRequest Tests
# =============================================================================


class TestVoiceDesignRequest:
    """Test VoiceDesignRequest Pydantic model."""

    def test_valid_request_creation(self):
        """Test creating a valid VoiceDesignRequest."""
        request = VoiceDesignRequest(
            text="Hello, I am John.",
            language="English",
            instruct="A warm, elderly man's voice with a Southern accent",
        )
        assert request.text == "Hello, I am John."
        assert request.language == "English"
        assert "warm" in request.instruct.lower()

    def test_default_language(self):
        """Test default language is English."""
        request = VoiceDesignRequest(
            text="Hello.",
            instruct="A friendly voice",
        )
        assert request.language == "English"

    def test_required_fields(self):
        """Test required fields must be provided."""
        with pytest.raises(ValidationError):
            VoiceDesignRequest(language="English")

        with pytest.raises(ValidationError):
            VoiceDesignRequest(text="Hello.")

    def test_serialization(self):
        """Test VoiceDesignRequest serialization."""
        request = VoiceDesignRequest(
            text="Hello, I am John.",
            language="English",
            instruct="A warm voice",
        )
        data = request.model_dump()
        assert data["text"] == "Hello, I am John."
        assert data["language"] == "English"
        assert "warm" in data["instruct"]

    def test_json_schema(self):
        """Test JSON schema generation."""
        schema = VoiceDesignRequest.model_json_schema()
        assert "text" in schema["properties"]
        assert "language" in schema["properties"]
        assert "instruct" in schema["properties"]


# =============================================================================
# VoiceDesignResponse Tests
# =============================================================================


class TestVoiceDesignResponse:
    """Test VoiceDesignResponse Pydantic model."""

    def test_valid_response_creation(self):
        """Test creating a valid VoiceDesignResponse."""
        response = VoiceDesignResponse(
            audio="base64encodedwavdata...",
            sample_rate=24000,
        )
        assert response.audio == "base64encodedwavdata..."
        assert response.sample_rate == 24000

    def test_default_sample_rate(self):
        """Test default sample rate is 24000."""
        response = VoiceDesignResponse(audio="base64data...")
        assert response.sample_rate == 24000

    def test_required_audio_field(self):
        """Test audio field is required."""
        with pytest.raises(ValidationError):
            VoiceDesignResponse(sample_rate=24000)

    def test_serialization(self):
        """Test VoiceDesignResponse serialization."""
        response = VoiceDesignResponse(
            audio="base64encoded...",
            sample_rate=24000,
        )
        data = response.model_dump()
        assert data["audio"] == "base64encoded..."
        assert data["sample_rate"] == 24000


# =============================================================================
# SynthesizeRequest Tests
# =============================================================================


class TestSynthesizeRequest:
    """Test SynthesizeRequest Pydantic model."""

    def test_valid_request_creation(self, base64_audio: str):
        """Test creating a valid SynthesizeRequest."""
        request = SynthesizeRequest(
            ref_audio=base64_audio,
            ref_text="Hello, I am John.",
            text="The quick brown fox.",
            language="English",
        )
        assert request.ref_audio == base64_audio
        assert request.ref_text == "Hello, I am John."
        assert request.text == "The quick brown fox."

    def test_default_language(self, base64_audio: str):
        """Test default language is English."""
        request = SynthesizeRequest(
            ref_audio=base64_audio,
            ref_text="Hello.",
            text="Testing.",
        )
        assert request.language == "English"

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            SynthesizeRequest(ref_text="Hello.", text="Testing.")

        with pytest.raises(ValidationError):
            SynthesizeRequest(ref_audio="base64", text="Testing.")

        with pytest.raises(ValidationError):
            SynthesizeRequest(ref_audio="base64", ref_text="Hello.")

    def test_serialization(self, base64_audio: str):
        """Test SynthesizeRequest serialization."""
        request = SynthesizeRequest(
            ref_audio=base64_audio,
            ref_text="Hello.",
            text="Testing.",
            language="English",
        )
        data = request.model_dump()
        assert data["ref_audio"] == base64_audio
        assert data["ref_text"] == "Hello."
        assert data["text"] == "Testing."


# =============================================================================
# SynthesizeResponse Tests
# =============================================================================


class TestSynthesizeResponse:
    """Test SynthesizeResponse Pydantic model."""

    def test_valid_response_creation(self):
        """Test creating a valid SynthesizeResponse."""
        response = SynthesizeResponse(
            audio="base64encodedwavdata...",
            sample_rate=24000,
        )
        assert response.audio == "base64encodedwavdata..."
        assert response.sample_rate == 24000

    def test_default_sample_rate(self):
        """Test default sample rate."""
        response = SynthesizeResponse(audio="base64data...")
        assert response.sample_rate == 24000

    def test_required_audio_field(self):
        """Test audio field is required."""
        with pytest.raises(ValidationError):
            SynthesizeResponse(sample_rate=24000)


# =============================================================================
# TTSModelManager Tests (Mocked)
# =============================================================================


class TestTTSModelManager:
    """Test TTSModelManager with mocked dependencies."""

    @pytest.fixture
    def manager_no_timeout(self):
        """Create a TTSModelManager with no idle timeout."""
        with patch("torch.cuda.is_available", return_value=False):
            manager = TTSModelManager(
                voice_design_model_id="Qwen/Qwen3-TTS-Test",
                base_model_id="Qwen/Qwen3-TTS-Base-Test",
                device="cpu",
                dtype="float32",
                idle_timeout=0,
            )
            yield manager

    def test_manager_initialization(self, manager_no_timeout: TTSModelManager):
        """Test TTSModelManager initialization."""
        assert manager_no_timeout.voice_design_model_id == "Qwen/Qwen3-TTS-Test"
        assert manager_no_timeout.base_model_id == "Qwen/Qwen3-TTS-Base-Test"
        assert manager_no_timeout.device == "cpu"
        assert manager_no_timeout.dtype == "float32"
        assert not manager_no_timeout.is_loaded

    def test_default_model_ids(self):
        """Test default model IDs are set correctly."""
        with patch("torch.cuda.is_available", return_value=False):
            manager = TTSModelManager(
                device="cpu",
                dtype="float32",
                idle_timeout=0,
            )
            assert "Qwen3-TTS-12Hz-1.7B-VoiceDesign" in manager.voice_design_model_id
            assert "Qwen3-TTS-12Hz-1.7B-Base" in manager.base_model_id

    def test_device_auto_detection_cuda(self):
        """Test CUDA device auto-detection."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.backends.mps.is_available", return_value=False):
                manager = TTSModelManager(idle_timeout=0)
                assert manager.device == "cuda:0"

    def test_device_auto_detection_mps(self):
        """Test MPS device auto-detection."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=True):
                manager = TTSModelManager(idle_timeout=0)
                assert manager.device == "mps"

    def test_device_auto_detection_cpu(self):
        """Test CPU device auto-detection."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch.object(
                type(type("mock", (), {"is_available": lambda: False})),
                "mps",
                create=True,
            ):
                manager = TTSModelManager(idle_timeout=0)
                assert manager.device == "cpu"

    def test_dtype_auto_detection_cuda(self):
        """Test bfloat16 dtype auto-detection with CUDA."""
        with patch("torch.cuda.is_available", return_value=True):
            manager = TTSModelManager(idle_timeout=0)
            assert manager.dtype == "bfloat16"

    def test_dtype_auto_detection_no_cuda(self):
        """Test float32 dtype auto-detection without CUDA."""
        with patch("torch.cuda.is_available", return_value=False):
            manager = TTSModelManager(idle_timeout=0)
            assert manager.dtype == "float32"

    def test_generate_not_implemented(self, manager_no_timeout: TTSModelManager):
        """Test generate method raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="voice_design.*synthesize"):
            manager_no_timeout.generate()


class TestTTSModelManagerVoiceDesign:
    """Test TTSModelManager voice design with mocked model."""

    @pytest.fixture
    def manager_with_mock(self):
        """Create a TTSModelManager with mocked Qwen3-TTS model."""
        with patch("torch.cuda.is_available", return_value=False):
            manager = TTSModelManager(
                device="cpu",
                dtype="float32",
                idle_timeout=0,
            )

            # Mock the models
            mock_voice_design_model = MagicMock()
            mock_voice_design_model.generate_voice_design.return_value = (
                [np.random.randn(24000).astype(np.float32) * 0.1],
                24000,
            )

            mock_base_model = MagicMock()

            manager._voice_design_model = mock_voice_design_model
            manager._base_model = mock_base_model
            manager._loaded = True

            yield manager, mock_voice_design_model, mock_base_model

    def test_voice_design_success(self, manager_with_mock):
        """Test successful voice design."""
        manager, mock_model, _ = manager_with_mock

        request = VoiceDesignRequest(
            text="Hello, I am John.",
            language="English",
            instruct="A warm voice",
        )

        result = manager.voice_design(request)

        assert isinstance(result, AudioResult)
        assert result.sample_rate == 24000
        assert result.metadata["language"] == "English"
        assert result.metadata["text"] == "Hello, I am John."
        mock_model.generate_voice_design.assert_called_once()

    def test_voice_design_metadata(self, manager_with_mock):
        """Test voice design includes correct metadata."""
        manager, _, _ = manager_with_mock

        request = VoiceDesignRequest(
            text="Test text.",
            language="Chinese",
            instruct="A friendly voice",
        )

        result = manager.voice_design(request)

        assert result.metadata["language"] == "Chinese"
        assert result.metadata["text"] == "Test text."
        assert "Qwen3-TTS" in result.metadata["model"]

    def test_voice_design_model_not_loaded(self, manager_with_mock):
        """Test voice design when model is not loaded."""
        manager, _, _ = manager_with_mock
        manager._voice_design_model = None

        request = VoiceDesignRequest(
            text="Hello.",
            language="English",
            instruct="A voice",
        )

        with pytest.raises(RuntimeError, match="VoiceDesign model not loaded"):
            manager.voice_design(request)

    def test_voice_design_error_handling(self, manager_with_mock):
        """Test voice design error handling."""
        manager, mock_model, _ = manager_with_mock
        mock_model.generate_voice_design.side_effect = Exception("Generation failed")

        request = VoiceDesignRequest(
            text="Hello.",
            language="English",
            instruct="A voice",
        )

        with pytest.raises(RuntimeError, match="Voice design failed"):
            manager.voice_design(request)


class TestTTSModelManagerSynthesize:
    """Test TTSModelManager synthesize with mocked model."""

    @pytest.fixture
    def manager_with_mock(self, base64_audio: str):
        """Create a TTSModelManager with mocked Qwen3-TTS model."""
        with patch("torch.cuda.is_available", return_value=False):
            manager = TTSModelManager(
                device="cpu",
                dtype="float32",
                idle_timeout=0,
            )

            # Mock the models
            mock_voice_design_model = MagicMock()

            mock_base_model = MagicMock()
            mock_base_model.generate_voice_clone.return_value = (
                [np.random.randn(24000).astype(np.float32) * 0.1],
                24000,
            )

            manager._voice_design_model = mock_voice_design_model
            manager._base_model = mock_base_model
            manager._loaded = True

            yield manager, mock_base_model

    def test_synthesize_success(self, manager_with_mock, base64_audio: str):
        """Test successful synthesis."""
        manager, mock_model = manager_with_mock

        request = SynthesizeRequest(
            ref_audio=base64_audio,
            ref_text="Hello, I am John.",
            text="The quick brown fox.",
            language="English",
        )

        result = manager.synthesize(request)

        assert isinstance(result, AudioResult)
        assert result.sample_rate == 24000
        assert result.metadata["language"] == "English"
        assert result.metadata["text"] == "The quick brown fox."
        mock_model.generate_voice_clone.assert_called_once()

    def test_synthesize_metadata(self, manager_with_mock, base64_audio: str):
        """Test synthesis includes correct metadata."""
        manager, _ = manager_with_mock

        request = SynthesizeRequest(
            ref_audio=base64_audio,
            ref_text="Hello.",
            text="Test synthesis.",
            language="Spanish",
        )

        result = manager.synthesize(request)

        assert result.metadata["language"] == "Spanish"
        assert result.metadata["text"] == "Test synthesis."
        assert "ref_duration" in result.metadata

    def test_synthesize_model_not_loaded(self, manager_with_mock, base64_audio: str):
        """Test synthesis when model is not loaded."""
        manager, _ = manager_with_mock
        manager._base_model = None

        request = SynthesizeRequest(
            ref_audio=base64_audio,
            ref_text="Hello.",
            text="Test.",
            language="English",
        )

        with pytest.raises(RuntimeError, match="Base model not loaded"):
            manager.synthesize(request)

    def test_synthesize_invalid_audio(self, manager_with_mock):
        """Test synthesis with invalid base64 audio."""
        manager, _ = manager_with_mock

        request = SynthesizeRequest(
            ref_audio="invalid-base64!!!",
            ref_text="Hello.",
            text="Test.",
            language="English",
        )

        with pytest.raises((RuntimeError, Exception)):
            manager.synthesize(request)


class TestTTSModelManagerAudioToBase64:
    """Test audio to base64 conversion."""

    def test_audio_to_base64(self, audio_array: np.ndarray):
        """Test converting AudioResult to base64."""
        with patch("torch.cuda.is_available", return_value=False):
            manager = TTSModelManager(idle_timeout=0)

            result = AudioResult(
                audio=audio_array,
                sample_rate=24000,
                duration=1.0,
                audio_type=AudioType.TTS,
                generator="test",
                prompt="test",
            )

            b64 = manager.audio_to_base64(result)
            assert isinstance(b64, str)
            assert len(b64) > 0

            # Verify it's valid base64
            decoded = base64.b64decode(b64)
            assert len(decoded) > 0


# =============================================================================
# Import Error Tests
# =============================================================================


class TestTTSImportErrors:
    """Test handling of missing qwen_tts package."""

    def test_qwen_tts_import_error(self):
        """Test ImportError when qwen_tts is not available."""
        with patch.dict("sys.modules", {"qwen_tts": None}):
            with patch("torch.cuda.is_available", return_value=False):
                manager = TTSModelManager(idle_timeout=0)

                # Mock _loaded to trigger _load_model
                manager._loaded = False

                with pytest.raises(ImportError, match="qwen_tts"):
                    manager._load_model()
