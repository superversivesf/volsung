"""Tests for base model classes.

Tests ModelManagerBase and GeneratorBase abstract classes including
lazy loading, idle timeout, thread safety, and abstract method enforcement.
"""

import threading
import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from volsung.models.base import GeneratorBase, ModelConfig, ModelManagerBase
from volsung.models.types import AudioResult, AudioType


# =============================================================================
# ModelConfig Tests
# =============================================================================


class TestModelConfig:
    """Test ModelConfig Pydantic model."""

    def test_valid_config_creation(self):
        """Test creating a valid ModelConfig."""
        config = ModelConfig(
            model_id="test-model",
            model_name="Test Model",
            device="cpu",
            dtype="float32",
            idle_timeout_seconds=300,
        )
        assert config.model_id == "test-model"
        assert config.model_name == "Test Model"
        assert config.device == "cpu"
        assert config.dtype == "float32"
        assert config.idle_timeout_seconds == 300
        assert config.max_memory_gb is None

    def test_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig(
            model_id="test-model",
            model_name="Test Model",
        )
        assert config.device == "auto"
        assert config.dtype == "auto"
        assert config.idle_timeout_seconds == 300

    def test_config_with_max_memory(self):
        """Test ModelConfig with memory limit."""
        config = ModelConfig(
            model_id="test-model",
            model_name="Test Model",
            max_memory_gb=8.0,
        )
        assert config.max_memory_gb == 8.0

    def test_config_zero_timeout(self):
        """Test ModelConfig with zero timeout (never unload)."""
        config = ModelConfig(
            model_id="test-model",
            model_name="Test Model",
            idle_timeout_seconds=0,
        )
        assert config.idle_timeout_seconds == 0

    def test_config_serialization(self):
        """Test ModelConfig can be serialized to dict."""
        config = ModelConfig(
            model_id="test-model",
            model_name="Test Model",
            device="cuda:0",
            dtype="bfloat16",
        )
        data = config.model_dump()
        assert data["model_id"] == "test-model"
        assert data["device"] == "cuda:0"

    def test_config_json_schema(self):
        """Test ModelConfig JSON schema generation."""
        schema = ModelConfig.model_json_schema()
        assert "model_id" in schema["properties"]
        assert "model_name" in schema["properties"]
        assert schema["properties"]["device"]["default"] == "auto"


# =============================================================================
# Concrete Test Implementations
# =============================================================================


class ConcreteModelManager(ModelManagerBase):
    """Concrete implementation for testing ModelManagerBase."""

    def __init__(self, config: ModelConfig):
        self._load_called = False
        self._unload_called = False
        super().__init__(config)

    def _load_model(self) -> None:
        """Track that load was called."""
        self._load_called = True
        self._model = {"loaded": True}

    def _unload_model(self) -> None:
        """Track that unload was called."""
        self._unload_called = True
        self._model = None

    def generate(self, *args, **kwargs) -> AudioResult:
        """Generate test audio."""
        self._ensure_loaded()
        return AudioResult(
            audio="base64encodeddata",
            sample_rate=24000,
            duration=1.0,
            audio_type=AudioType.TTS,
            generator=self.config.model_id,
            prompt="test",
        )


class ConcreteGenerator(GeneratorBase):
    """Concrete implementation for testing GeneratorBase."""

    def __init__(self):
        self.model = None
        self._device = None
        self._dtype = None

    @property
    def model_id(self) -> str:
        return "test-generator"

    @property
    def model_name(self) -> str:
        return "Test Generator"

    @property
    def required_vram_gb(self) -> float:
        return 4.0

    def load(self, device: str, dtype: str) -> None:
        self._device = device
        self._dtype = dtype
        self.model = MagicMock()

    def unload(self) -> None:
        self.model = None
        self._device = None
        self._dtype = None

    def generate(self, prompt: str, duration: float, **kwargs) -> tuple:
        import numpy as np

        samples = int(24000 * duration)
        audio = np.random.randn(samples).astype(np.float32)
        return audio, 24000


# =============================================================================
# ModelManagerBase Tests
# =============================================================================


class TestModelManagerBase:
    """Test ModelManagerBase abstract class."""

    def test_cannot_instantiate_abstract(self):
        """Test that ModelManagerBase cannot be instantiated directly."""
        config = ModelConfig(model_id="test", model_name="Test")
        with pytest.raises(TypeError):
            ModelManagerBase(config)

    def test_concrete_implementation(self):
        """Test concrete implementation can be instantiated."""
        config = ModelConfig(model_id="test", model_name="Test", idle_timeout_seconds=0)
        manager = ConcreteModelManager(config)
        assert manager.config.model_id == "test"
        assert not manager.is_loaded

    def test_lazy_loading(self):
        """Test lazy loading on first access."""
        config = ModelConfig(model_id="test", model_name="Test", idle_timeout_seconds=0)
        manager = ConcreteModelManager(config)

        # Not loaded initially
        assert not manager.is_loaded
        assert not manager._load_called

        # Load on first access
        manager._ensure_loaded()
        assert manager.is_loaded
        assert manager._load_called

    def test_lazy_loading_via_generate(self):
        """Test lazy loading via generate method."""
        config = ModelConfig(model_id="test", model_name="Test", idle_timeout_seconds=0)
        manager = ConcreteModelManager(config)

        # Generate should trigger loading
        result = manager.generate()
        assert manager.is_loaded
        assert isinstance(result, AudioResult)

    def test_idle_seconds_not_loaded(self):
        """Test idle_seconds returns None when not loaded."""
        config = ModelConfig(model_id="test", model_name="Test", idle_timeout_seconds=0)
        manager = ConcreteModelManager(config)
        assert manager.idle_seconds is None

    def test_idle_seconds_after_access(self):
        """Test idle_seconds is tracked after access."""
        config = ModelConfig(model_id="test", model_name="Test", idle_timeout_seconds=0)
        manager = ConcreteModelManager(config)

        manager._ensure_loaded()
        # Should be very close to 0
        assert manager.idle_seconds is not None
        assert 0 <= manager.idle_seconds < 1

    def test_force_unload(self):
        """Test force_unload method."""
        config = ModelConfig(model_id="test", model_name="Test", idle_timeout_seconds=0)
        manager = ConcreteModelManager(config)

        manager._ensure_loaded()
        assert manager.is_loaded

        manager.force_unload()
        assert not manager.is_loaded
        assert manager._unload_called

    def test_unload_if_idle_not_loaded(self):
        """Test unload_if_idle when not loaded."""
        config = ModelConfig(model_id="test", model_name="Test", idle_timeout_seconds=0)
        manager = ConcreteModelManager(config)
        result = manager.unload_if_idle()
        assert result is False

    def test_unload_if_idle_zero_timeout(self):
        """Test unload_if_idle with zero timeout never unloads."""
        config = ModelConfig(model_id="test", model_name="Test", idle_timeout_seconds=0)
        manager = ConcreteModelManager(config)

        manager._ensure_loaded()
        # Should not unload with timeout=0
        result = manager.unload_if_idle()
        assert result is False
        assert manager.is_loaded

    def test_thread_safety(self):
        """Test thread-safe access to model."""
        config = ModelConfig(model_id="test", model_name="Test", idle_timeout_seconds=0)
        manager = ConcreteModelManager(config)

        results = []

        def access_model():
            manager._ensure_loaded()
            results.append(manager.is_loaded)

        threads = [threading.Thread(target=access_model) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(results)
        assert manager.is_loaded

    def test_shutdown(self):
        """Test shutdown method."""
        config = ModelConfig(model_id="test", model_name="Test", idle_timeout_seconds=0)
        manager = ConcreteModelManager(config)

        manager._ensure_loaded()
        manager.shutdown()

        assert not manager.is_loaded
        assert manager._shutdown is True

    def test_idle_monitor_not_started_with_zero_timeout(self):
        """Test idle monitor is not started when timeout is 0."""
        config = ModelConfig(model_id="test", model_name="Test", idle_timeout_seconds=0)
        manager = ConcreteModelManager(config)
        assert manager._idle_timer is None


class TestModelManagerBaseIdleTimeout:
    """Test idle timeout monitoring behavior."""

    def test_idle_unload_after_timeout(self):
        """Test model unloads after idle timeout."""
        config = ModelConfig(model_id="test", model_name="Test", idle_timeout_seconds=1)
        manager = ConcreteModelManager(config)

        manager._ensure_loaded()
        assert manager.is_loaded

        # Wait for timeout
        time.sleep(1.5)

        # Trigger unload check
        result = manager.unload_if_idle()
        assert result is True
        assert not manager.is_loaded

    def test_access_resets_idle_timer(self):
        """Test that access resets the idle timer."""
        config = ModelConfig(model_id="test", model_name="Test", idle_timeout_seconds=2)
        manager = ConcreteModelManager(config)

        manager._ensure_loaded()

        # Access multiple times
        for _ in range(3):
            time.sleep(0.5)
            manager._ensure_loaded()  # Should reset timer

        # Should still be loaded
        assert manager.is_loaded


class TestModelManagerBaseClearMemory:
    """Test memory clearing functionality."""

    def test_clear_memory_calls_gc(self):
        """Test _clear_memory calls garbage collection."""
        config = ModelConfig(model_id="test", model_name="Test", idle_timeout_seconds=0)
        manager = ConcreteModelManager(config)

        with patch("gc.collect") as mock_gc:
            manager._clear_memory()
            mock_gc.assert_called_once()

    def test_clear_memory_with_cuda(self):
        """Test _clear_memory clears CUDA cache."""
        config = ModelConfig(model_id="test", model_name="Test", idle_timeout_seconds=0)
        manager = ConcreteModelManager(config)

        with patch("gc.collect"):
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.empty_cache = MagicMock()
            mock_torch.cuda.synchronize = MagicMock()

            with patch.dict("sys.modules", {"torch": mock_torch}):
                manager._clear_memory()
                mock_torch.cuda.empty_cache.assert_called_once()
                mock_torch.cuda.synchronize.assert_called_once()


# =============================================================================
# GeneratorBase Tests
# =============================================================================


class TestGeneratorBase:
    """Test GeneratorBase abstract class."""

    def test_cannot_instantiate_abstract(self):
        """Test that GeneratorBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            GeneratorBase()

    def test_concrete_implementation(self):
        """Test concrete implementation can be instantiated."""
        generator = ConcreteGenerator()
        assert generator.model_id == "test-generator"
        assert generator.model_name == "Test Generator"
        assert generator.required_vram_gb == 4.0

    def test_generator_properties(self):
        """Test abstract properties are enforced."""
        generator = ConcreteGenerator()
        assert isinstance(generator.model_id, str)
        assert isinstance(generator.model_name, str)
        assert isinstance(generator.required_vram_gb, float)

    def test_generator_load_unload(self):
        """Test load and unload methods."""
        generator = ConcreteGenerator()
        assert generator.model is None

        generator.load("cpu", "float32")
        assert generator.model is not None
        assert generator._device == "cpu"
        assert generator._dtype == "float32"

        generator.unload()
        assert generator.model is None

    def test_generator_generate(self):
        """Test generate method returns audio."""
        import numpy as np

        generator = ConcreteGenerator()
        generator.load("cpu", "float32")

        audio, sample_rate = generator.generate("test prompt", 1.0)
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert sample_rate == 24000

    def test_generator_get_info(self):
        """Test get_info method."""
        generator = ConcreteGenerator()
        info = generator.get_info()

        assert isinstance(info, dict)
        assert info["model_id"] == "test-generator"
        assert info["model_name"] == "Test Generator"
        assert info["required_vram_gb"] == 4.0
        assert info["loaded"] is False

        generator.load("cpu", "float32")
        info = generator.get_info()
        assert info["loaded"] is True

    def test_get_info_includes_loaded_state(self):
        """Test get_info reflects loaded state."""
        generator = ConcreteGenerator()

        # Not loaded
        info = generator.get_info()
        assert info["loaded"] is False

        # Loaded
        generator.load("cpu", "float32")
        info = generator.get_info()
        assert info["loaded"] is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestBaseIntegration:
    """Integration tests for base classes."""

    def test_manager_uses_generator(self):
        """Test manager can use a generator."""

        class GeneratorUsingManager(ModelManagerBase):
            def __init__(self, config: ModelConfig, generator: GeneratorBase):
                self._generator = generator
                super().__init__(config)

            def _load_model(self) -> None:
                self._generator.load(self.config.device, self.config.dtype)
                self._model = self._generator

            def _unload_model(self) -> None:
                self._generator.unload()
                self._model = None

            def generate(self, prompt: str, duration: float) -> AudioResult:
                self._ensure_loaded()
                audio, sr = self._generator.generate(prompt, duration)
                return AudioResult(
                    audio="base64data",
                    sample_rate=sr,
                    duration=duration,
                    audio_type=AudioType.MUSIC,
                    generator=self._generator.model_id,
                    prompt=prompt,
                )

        config = ModelConfig(
            model_id="test-integration",
            model_name="Test Integration",
            device="cpu",
            dtype="float32",
            idle_timeout_seconds=0,
        )
        generator = ConcreteGenerator()
        manager = GeneratorUsingManager(config, generator)

        result = manager.generate("test", 1.0)
        assert isinstance(result, AudioResult)
        assert result.audio_type == AudioType.MUSIC
