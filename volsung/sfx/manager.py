"""
SFX Model Manager - coordinates sound effect generation.

Manages AudioLDM and other SFX generators with lazy loading,
idle timeout monitoring, and thread-safe access.
"""

import base64
import logging
import time
from io import BytesIO
from typing import Any, Optional

import numpy as np
import soundfile as sf

from volsung.config import get_config
from volsung.models.base import ModelConfig, ModelManagerBase
from volsung.models.types import AudioResult, AudioType

from .generators.audioldm import AudioLDMGenerator

logger = logging.getLogger(__name__)


class SFXModelManager(ModelManagerBase):
    """Manager for sound effect generation models.

    Wraps AudioLDM and other SFX generators with:
    - Lazy loading (models load on first use)
    - Idle timeout monitoring (auto-unload after inactivity)
    - Thread-safe access via reentrant locks
    - Resource cleanup (GPU cache clearing)

    Example:
        ```python
        config = ModelConfig(
            model_id="audioldm2-base",
            model_name="AudioLDM2 Base",
            idle_timeout_seconds=300
        )
        manager = SFXModelManager(config)

        # Generate sound effect
        result = manager.generate(
            prompt="Thunder rumbling",
            duration=5.0
        )
        ```
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize the SFX model manager.

        Args:
            config: Model configuration. If None, loads from global config.
        """
        if config is None:
            # Load from global configuration
            volsung_config = get_config()
            sfx_config = volsung_config.sfx

            config = ModelConfig(
                model_id=sfx_config.model.replace("/", "-"),
                model_name=sfx_config.model,
                device=sfx_config.device or "auto",
                dtype="float16",  # SFX models work well with float16
                idle_timeout_seconds=sfx_config.idle_timeout,
            )

        super().__init__(config)
        self._generator: Optional[AudioLDMGenerator] = None

    def _load_model(self) -> None:
        """Load the AudioLDM model.

        Determines model size from config and loads appropriate generator.
        """
        # Determine model size from model name
        model_name = self.config.model_name.lower()
        if "large" in model_name:
            model_size = "large"
        elif "music" in model_name:
            model_size = "music"
        else:
            model_size = "base"

        logger.info(f"Initializing AudioLDM generator (size: {model_size})")

        # Create generator
        self._generator = AudioLDMGenerator(model_size=model_size)

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
                dtype = "float16"
            else:
                dtype = "float32"

        # Load the model
        self._generator.load(device=device, dtype=dtype)
        logger.info(f"AudioLDM loaded on {device} with {dtype}")

    def _unload_model(self) -> None:
        """Unload the model and free resources."""
        if self._generator is not None:
            self._generator.unload()
            self._generator = None
            self._clear_memory()
            logger.info("AudioLDM unloaded")

    def generate(
        self,
        prompt: str,
        duration: float,
        category: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        **kwargs,
    ) -> AudioResult:
        """Generate sound effects from text prompt.

        Args:
            prompt: Text description of desired sound effect
            duration: Target duration in seconds (1.0 to 10.0)
            category: Optional category hint (e.g., "nature", "mechanical")
            num_inference_steps: Number of denoising steps
            guidance_scale: Prompt adherence level
            **kwargs: Additional generation parameters

        Returns:
            AudioResult with base64-encoded audio and metadata

        Raises:
            Exception: If generation fails
        """
        # Ensure model is loaded
        self._ensure_loaded()

        # Enhance prompt with category if provided
        enhanced_prompt = prompt
        if category:
            enhanced_prompt = f"{category}: {prompt}"

        logger.info(
            f"[SFX_GENERATE] prompt='{enhanced_prompt[:50]}...' "
            f"duration={duration}s steps={num_inference_steps}"
        )

        # Generate audio
        audio_array, sample_rate = self._generator.generate(
            prompt=enhanced_prompt,
            duration=duration,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs,
        )

        # Convert to base64
        audio_base64 = self._audio_to_base64(audio_array, sample_rate)

        # Calculate actual duration
        actual_duration = len(audio_array) / sample_rate

        # Build result
        result = AudioResult(
            audio=audio_base64,
            sample_rate=sample_rate,
            duration=actual_duration,
            audio_type=AudioType.SFX,
            generator=self._generator.model_id,
            prompt=prompt,
            format="wav",
            channels=1,
            metadata={
                "category": category,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            },
        )

        return result

    def _audio_to_base64(self, audio: np.ndarray, sample_rate: int) -> str:
        """Convert audio array to base64-encoded WAV.

        Args:
            audio: Audio samples as numpy array
            sample_rate: Sample rate in Hz

        Returns:
            Base64-encoded WAV string
        """
        buffer = BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode()

    def get_generator_info(self) -> dict[str, Any]:
        """Get information about the loaded generator.

        Returns:
            Dictionary with generator metadata
        """
        if self._generator is None:
            return {"loaded": False}
        return self._generator.get_info()
