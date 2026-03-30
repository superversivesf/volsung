"""Music module manager for Volsung.

Provides high-level music generation with lazy loading and idle timeout.
Uses MusicGen model for background music generation.
"""

import logging
from typing import Any, Dict, Optional

from volsung.models.base import ModelConfig, ModelManagerBase
from volsung.models.types import AudioResult, AudioType

logger = logging.getLogger(__name__)


class MusicModelManager(ModelManagerBase):
    """Music generation model manager.

    Wraps a music generator (e.g., MusicGen) with lazy loading,
    idle timeout, and thread-safe access.

    Attributes:
        config: Model configuration
        _generator: The underlying generator instance

    Example:
        ```python
        from volsung.config import get_config

        config = get_config()
        manager = MusicModelManager(ModelConfig(
            model_id="musicgen-small",
            model_name="MusicGen Small",
            device="auto",
            idle_timeout_seconds=config.music.idle_timeout,
        ))

        result = manager.generate(
            prompt="Upbeat acoustic guitar for audiobook",
            duration=10.0
        )
        ```
    """

    def __init__(self, config: ModelConfig):
        """Initialize the music model manager.

        Args:
            config: Model configuration including timeout settings
        """
        self._generator: Optional[Any] = None
        super().__init__(config)

    def _load_model(self) -> None:
        """Load the music generation model.

        Imports and instantiates the appropriate generator based on config.
        Called automatically by _ensure_loaded().
        """
        from .generators.musicgen import MusicGenGenerator

        logger.info(f"Loading {self.config.model_name}...")

        # Determine device
        device = self.config.device
        if device == "auto":
            import torch

            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # Determine dtype
        dtype = self.config.dtype
        if dtype == "auto":
            import torch

            if torch.cuda.is_available():
                dtype = "bfloat16"
            else:
                dtype = "float32"

        # Create and load generator
        self._generator = MusicGenGenerator()
        self._generator.load(device=device, dtype=dtype)

        logger.info(f"{self.config.model_name} loaded on {device} with {dtype}")

    def _unload_model(self) -> None:
        """Unload the model and free resources.

        Called automatically by unload_if_idle() or force_unload().
        """
        if self._generator is not None:
            logger.info(f"Unloading {self.config.model_name}...")
            self._generator.unload()
            self._generator = None

        # Clear GPU memory
        self._clear_memory()
        logger.info(f"{self.config.model_name} unloaded")

    def generate(
        self,
        prompt: str,
        duration: float,
        genre: Optional[str] = None,
        mood: Optional[str] = None,
        tempo: Optional[str] = None,
        top_k: int = 250,
        top_p: float = 0.0,
        temperature: float = 1.0,
    ) -> AudioResult:
        """Generate music from text description.

        Args:
            prompt: Text description of desired music
            duration: Target duration in seconds (1-30)
            genre: Optional genre tag (e.g., "acoustic", "electronic")
            mood: Optional mood tag (e.g., "upbeat", "calm")
            tempo: Optional tempo hint ("slow", "medium", "fast")
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            temperature: Randomness/temperature parameter

        Returns:
            AudioResult with generated audio and metadata

        Raises:
            ValueError: If duration is invalid
            Exception: If generation fails

        Example:
            ```python
            result = manager.generate(
                prompt="Peaceful acoustic guitar for meditation",
                duration=15.0,
                mood="calm",
                tempo="slow"
            )
            ```
        """
        import time
        import base64
        from io import BytesIO

        import numpy as np
        import soundfile as sf

        # Validate duration
        if duration <= 0 or duration > 30:
            raise ValueError("Duration must be between 1 and 30 seconds")

        # Ensure model is loaded
        self._ensure_loaded()

        start_time = time.time()
        logger.info(
            f"[MUSIC] Generating: duration={duration}s prompt='{prompt[:50]}...'"
        )

        # Enhance prompt with optional tags
        enhanced_prompt = self._build_prompt(prompt, genre, mood, tempo)

        # Generate audio
        audio_array, sample_rate = self._generator.generate(
            prompt=enhanced_prompt,
            duration=duration,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

        # Convert to base64 WAV
        buffer = BytesIO()
        sf.write(buffer, audio_array, sample_rate, format="WAV")
        buffer.seek(0)
        audio_b64 = base64.b64encode(buffer.read()).decode()

        actual_duration = len(audio_array) / sample_rate
        elapsed_ms = (time.time() - start_time) * 1000

        logger.info(
            f"[MUSIC] Generated: duration={actual_duration:.2f}s "
            f"sample_rate={sample_rate} elapsed={elapsed_ms:.0f}ms"
        )

        # Build metadata
        metadata: Dict[str, Any] = {
            "generation_time_ms": elapsed_ms,
            "model_version": self._generator.model_id,
            "parameters": {
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
            },
        }
        if genre:
            metadata["genre"] = genre
        if mood:
            metadata["mood"] = mood
        if tempo:
            metadata["tempo"] = tempo

        return AudioResult(
            audio=audio_b64,
            sample_rate=sample_rate,
            duration=actual_duration,
            audio_type=AudioType.MUSIC,
            generator=self._generator.model_id,
            prompt=prompt,
            metadata=metadata,
        )

    def _build_prompt(
        self,
        base_prompt: str,
        genre: Optional[str] = None,
        mood: Optional[str] = None,
        tempo: Optional[str] = None,
    ) -> str:
        """Build enhanced prompt from base prompt and tags.

        Args:
            base_prompt: Original text description
            genre: Optional genre tag
            mood: Optional mood tag
            tempo: Optional tempo tag

        Returns:
            Enhanced prompt string
        """
        parts = [base_prompt]

        if genre:
            parts.append(f"Genre: {genre}")
        if mood:
            parts.append(f"Mood: {mood}")
        if tempo:
            parts.append(f"Tempo: {tempo}")

        return ", ".join(parts)

    def get_info(self) -> Dict[str, Any]:
        """Get manager information.

        Returns:
            Dictionary with manager metadata
        """
        info = {
            "model_id": self.config.model_id,
            "model_name": self.config.model_name,
            "is_loaded": self.is_loaded,
            "device": self.config.device,
            "dtype": self.config.dtype,
        }

        if self._generator is not None:
            info["generator"] = self._generator.get_info()

        return info
