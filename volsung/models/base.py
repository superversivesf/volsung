"""Abstract base classes for Volsung model managers.

Provides the foundation for lazy loading, idle timeout monitoring,
and thread-safe access across all model types (TTS, Music, SFX).
"""

import gc
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field

from .types import AudioResult

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Configuration for a model manager.

    Attributes:
        model_id: Unique identifier for this model
        model_name: Human-readable name
        device: Device to use ("auto", "cuda", "cuda:0", "cpu", "mps")
        dtype: Data type ("auto", "float16", "float32", "bfloat16")
        max_memory_gb: Maximum memory to use in GB (None for unlimited)
        idle_timeout_seconds: Seconds before unloading model (0 = never unload)

    Example:
        ```python
        config = ModelConfig(
            model_id="musicgen-small",
            model_name="MusicGen Small",
            idle_timeout_seconds=300
        )
        ```
    """

    model_id: str = Field(..., description="Unique identifier for this model")
    model_name: str = Field(..., description="Human-readable name")
    device: str = Field(
        default="auto", description="Device to use (auto, cuda, cpu, mps)"
    )
    dtype: str = Field(
        default="auto", description="Data type (auto, float16, float32, bfloat16)"
    )
    max_memory_gb: Optional[float] = Field(
        default=None, description="Maximum memory to use in GB"
    )
    idle_timeout_seconds: int = Field(
        default=300, description="Seconds before unloading model"
    )


class ModelManagerBase(ABC):
    """Abstract base class for all model managers.

    Provides:
    - Lazy loading (models load on first use)
    - Idle timeout monitoring (auto-unload after inactivity)
    - Thread-safe access via reentrant locks
    - Resource cleanup (GPU cache clearing)

    Subclasses must implement:
    - _load_model(): Load the actual model
    - _unload_model(): Unload and free resources
    - generate(): Generate audio from inputs

    Example:
        ```python
        class MusicModelManager(ModelManagerBase):
            def _load_model(self):
                self._model = load_my_model()

            def _unload_model(self):
                self._model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            def generate(self, prompt: str, duration: float) -> AudioResult:
                self._ensure_loaded()
                # ... generate audio
                return AudioResult(...)
        ```
    """

    def __init__(self, config: ModelConfig):
        """Initialize the model manager.

        Args:
            config: Model configuration including timeout settings
        """
        self.config = config
        self._model: Optional[Any] = None
        self._loaded: bool = False
        self._last_access: float = 0.0
        self._lock: threading.RLock = threading.RLock()
        self._idle_timer: Optional[threading.Timer] = None
        self._shutdown: bool = False

        self._start_idle_monitor()
        logger.info(
            f"Initialized {config.model_name} manager (idle_timeout={config.idle_timeout_seconds}s)"
        )

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded.

        Returns:
            True if model is loaded, False otherwise
        """
        with self._lock:
            return self._loaded

    @property
    def idle_seconds(self) -> Optional[float]:
        """Get seconds since last access.

        Returns:
            Seconds since last access, or None if not loaded
        """
        with self._lock:
            if not self._loaded:
                return None
            return time.time() - self._last_access

    def _ensure_loaded(self) -> None:
        """Load model if not already loaded.

        This is called automatically before any generation.
        Updates the last access time on successful load.
        """
        with self._lock:
            if not self._loaded:
                logger.info(f"Lazy loading {self.config.model_name}...")
                self._load_model()
                self._loaded = True
                logger.info(f"{self.config.model_name} loaded successfully")
            self._last_access = time.time()

    def unload_if_idle(self) -> bool:
        """Unload model if idle for too long.

        Called by the idle monitor periodically. Can also be called
        manually to force unload.

        Returns:
            True if model was unloaded, False if still in use or not loaded
        """
        with self._lock:
            if not self._loaded:
                return False

            # If timeout is 0, never unload
            if self.config.idle_timeout_seconds <= 0:
                return False

            idle_duration = time.time() - self._last_access
            if idle_duration >= self.config.idle_timeout_seconds:
                logger.info(
                    f"Unloading {self.config.model_name} after "
                    f"{idle_duration:.0f}s idle"
                )
                self._unload_model()
                self._loaded = False
                self._model = None
                self._last_access = 0.0
                return True
            return False

    def force_unload(self) -> None:
        """Force unload the model immediately.

        Use this when you need to free resources immediately,
        regardless of idle timeout settings.
        """
        with self._lock:
            if self._loaded:
                logger.info(f"Force unloading {self.config.model_name}...")
                self._unload_model()
                self._loaded = False
                self._model = None
                self._last_access = 0.0

    def _start_idle_monitor(self) -> None:
        """Start the idle timeout monitoring timer.

        Called automatically during initialization.
        """
        if self.config.idle_timeout_seconds <= 0:
            return

        def check_idle():
            if self._shutdown:
                return
            self.unload_if_idle()
            # Schedule next check (every 30 seconds or half timeout, whichever is smaller)
            interval = min(30, max(10, self.config.idle_timeout_seconds // 2))
            self._idle_timer = threading.Timer(interval, check_idle)
            self._idle_timer.daemon = True
            self._idle_timer.start()

        # Start first check
        check_idle()

    def shutdown(self) -> None:
        """Shutdown the manager and cleanup resources.

        Call this when the application is shutting down.
        """
        self._shutdown = True
        if self._idle_timer:
            self._idle_timer.cancel()
        self.force_unload()

    @abstractmethod
    def _load_model(self) -> None:
        """Load the actual model.

        Implementations should:
        - Load the model from disk/network
        - Move to appropriate device
        - Set up any required preprocessing
        - Store in self._model

        Raises:
            Any exception if loading fails
        """
        pass

    @abstractmethod
    def _unload_model(self) -> None:
        """Unload the model and free resources.

        Implementations should:
        - Delete model references
        - Run garbage collection
        - Clear GPU cache if applicable

        Note:
            This is called within a lock, so it's thread-safe.
        """
        pass

    @abstractmethod
    def generate(self, *args, **kwargs) -> AudioResult:
        """Generate audio.

        Subclasses must implement this method. Implementations should:
        - Call self._ensure_loaded() before generation
        - Return a standardized AudioResult
        - Handle errors appropriately

        Returns:
            AudioResult with generated audio and metadata

        Raises:
            Exception on generation failure
        """
        pass

    def _clear_memory(self) -> None:
        """Clear memory after model operations.

        Helper method that can be called by subclasses after
        generation to free up memory.
        """
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass


class GeneratorBase(ABC):
    """Abstract base for specific model generators.

    Each generator wraps a specific model (e.g., MusicGen, AudioLDM).
    This is a lower-level abstraction than ModelManagerBase - generators
    are managed by the manager.

    Example:
        ```python
        class MusicGenGenerator(GeneratorBase):
            @property
            def model_id(self) -> str:
                return "musicgen-small"

            @property
            def model_name(self) -> str:
                return "MusicGen Small"

            def load(self, device: str, dtype: str):
                self.model = load_musicgen(device, dtype)

            def unload(self):
                self.model = None

            def generate(self, prompt: str, duration: float, **kwargs) -> Tuple[np.ndarray, int]:
                audio = self.model.generate(prompt, duration)
                return audio, 32000
        ```
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique identifier for this generator.

        Returns:
            Short unique ID (e.g., "musicgen-small", "audioldm-m-full")
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable name.

        Returns:
            Display name (e.g., "MusicGen Small", "AudioLDM-M-Full")
        """
        pass

    @property
    @abstractmethod
    def required_vram_gb(self) -> float:
        """Estimated VRAM requirement in GB.

        Returns:
            Approximate GB of VRAM needed
        """
        pass

    @abstractmethod
    def load(self, device: str, dtype: str) -> None:
        """Load the model.

        Args:
            device: Device to load on ("cuda", "cuda:0", "cpu", "mps")
            dtype: Data type ("float16", "float32", "bfloat16")

        Raises:
            Exception if loading fails
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload and cleanup.

        Should free all resources and clear GPU memory.
        """
        pass

    @abstractmethod
    def generate(self, prompt: str, duration: float, **kwargs) -> Tuple[Any, int]:
        """Generate audio.

        Args:
            prompt: Text description of desired audio
            duration: Target duration in seconds
            **kwargs: Generator-specific parameters

        Returns:
            Tuple of (audio_array, sample_rate)
            audio_array should be numpy array or torch tensor

        Raises:
            Exception if generation fails
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """Get generator information.

        Returns:
            Dictionary with generator metadata
        """
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "required_vram_gb": self.required_vram_gb,
            "loaded": hasattr(self, "model") and self.model is not None,
        }
