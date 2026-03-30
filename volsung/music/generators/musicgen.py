"""MusicGen generator implementation.

Wraps Facebook's MusicGen model for music generation from text prompts.
Part of the Volsung music generation system.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from volsung.models.base import GeneratorBase

logger = logging.getLogger(__name__)


class MusicGenGenerator(GeneratorBase):
    """MusicGen model generator.

    Wraps Facebook's MusicGen model for conditional music generation.
    Supports text-to-music generation with configurable sampling parameters.

    Attributes:
        model: The underlying MusicGen model instance
        processor: Audio processor for the model

    Example:
        ```python
        generator = MusicGenGenerator()
        generator.load(device="cuda:0", dtype="bfloat16")

        audio, sample_rate = generator.generate(
            prompt="Upbeat acoustic guitar",
            duration=10.0,
        )
        ```

    References:
        - https://github.com/facebookresearch/audiocraft
        - https://huggingface.co/facebook/musicgen-small
    """

    MODEL_ID = "musicgen-small"
    MODEL_NAME = "MusicGen Small"
    DEFAULT_SAMPLE_RATE = 32000
    REQUIRED_VRAM_GB = 6.0

    def __init__(self):
        """Initialize the MusicGen generator."""
        self.model: Optional[Any] = None
        self._device: Optional[str] = None
        self._dtype: Optional[str] = None

    @property
    def model_id(self) -> str:
        """Unique identifier for this generator.

        Returns:
            Model ID string
        """
        return self.MODEL_ID

    @property
    def model_name(self) -> str:
        """Human-readable name.

        Returns:
            Display name
        """
        return self.MODEL_NAME

    @property
    def required_vram_gb(self) -> float:
        """Estimated VRAM requirement in GB.

        Returns:
            Approximate GB of VRAM needed
        """
        return self.REQUIRED_VRAM_GB

    def load(self, device: str, dtype: str) -> None:
        """Load the MusicGen model.

        Args:
            device: Device to load on ("cuda", "cuda:0", "cpu", "mps")
            dtype: Data type ("float16", "float32", "bfloat16")

        Raises:
            ImportError: If audiocraft is not installed
            Exception: If model loading fails
        """
        try:
            from audiocraft.models import MusicGen
        except ImportError:
            raise ImportError(
                "audiocraft is required for MusicGen. "
                "Install with: pip install audiocraft"
            )

        logger.info(f"Loading MusicGen model (facebook/{self.MODEL_ID})...")

        try:
            # Load the model
            self.model = MusicGen.get_pretrained(self.MODEL_ID)

            # Configure generation parameters
            self.model.set_generation_params(
                duration=10.0,  # Will be overridden per-generation
                top_k=250,
                top_p=0.0,
                temperature=1.0,
                use_sampling=True,
            )

            self._device = device
            self._dtype = dtype

            logger.info(f"MusicGen loaded on {device} with {dtype}")

        except Exception as e:
            logger.error(f"Failed to load MusicGen: {e}")
            raise

    def unload(self) -> None:
        """Unload and cleanup resources.

        Frees GPU memory and clears model references.
        """
        if self.model is not None:
            logger.info("Unloading MusicGen model...")

            # Delete model reference
            del self.model
            self.model = None

            # Clear GPU cache
            try:
                import gc
                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass

            self._device = None
            self._dtype = None

            logger.info("MusicGen unloaded")

    def generate(
        self,
        prompt: str,
        duration: float,
        top_k: int = 250,
        top_p: float = 0.0,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, int]:
        """Generate music from text prompt.

        Args:
            prompt: Text description of desired music
            duration: Target duration in seconds (1-30)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter (0.0 = disabled)
            temperature: Randomness/temperature
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (audio_array, sample_rate)
            audio_array is numpy array of shape (samples,)

        Raises:
            RuntimeError: If model not loaded
            Exception: If generation fails
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        logger.info(f"[MusicGen] Generating {duration}s: '{prompt[:50]}...'")

        # Set generation parameters
        self.model.set_generation_params(
            duration=duration,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            use_sampling=True,
        )

        # Generate
        try:
            with torch.no_grad():
                # Generate from text
                output = self.model.generate(
                    descriptions=[prompt],
                    progress=False,
                )

            # Convert to numpy array
            # Output shape: (1, 1, samples) or (batch, channels, samples)
            audio = output[0, 0].cpu().numpy()

            # Ensure mono
            if audio.ndim > 1:
                audio = audio.mean(axis=0)

            logger.info(
                f"[MusicGen] Generated: {len(audio)} samples "
                f"@ {self.DEFAULT_SAMPLE_RATE}Hz"
            )

            return audio, self.DEFAULT_SAMPLE_RATE

        except Exception as e:
            logger.error(f"[MusicGen] Generation failed: {e}")
            raise

    def get_info(self) -> Dict[str, Any]:
        """Get generator information.

        Returns:
            Dictionary with generator metadata
        """
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "required_vram_gb": self.required_vram_gb,
            "loaded": self.model is not None,
            "device": self._device,
            "dtype": self._dtype,
            "sample_rate": self.DEFAULT_SAMPLE_RATE,
        }

    def generate_continuation(
        self,
        audio: np.ndarray,
        prompt: Optional[str] = None,
        duration: float = 10.0,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, int]:
        """Generate continuation of existing audio.

        Extends existing audio by generating a continuation.
        Useful for seamless looping or extending tracks.

        Args:
            audio: Existing audio to continue from (numpy array)
            prompt: Optional new prompt (None = continue same style)
            duration: Duration to generate
            **kwargs: Generation parameters

        Returns:
            Tuple of (continued_audio, sample_rate)
            Note: Returns the CONTINUATION only, not original + continuation

        Raises:
            RuntimeError: If model not loaded
            Exception: If generation fails
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        logger.info(f"[MusicGen] Continuing audio for {duration}s")

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)

        # Set parameters
        self.model.set_generation_params(
            duration=duration,
            use_sampling=True,
        )

        try:
            with torch.no_grad():
                # Generate continuation
                output = self.model.generate_continuation(
                    prompt=audio_tensor.to(self._device or "cpu"),
                    prompt_sample_rate=self.DEFAULT_SAMPLE_RATE,
                    descriptions=[prompt] if prompt else None,
                    progress=False,
                )

            # Extract continuation (skip the prompt audio)
            continuation = output[0, 0, len(audio) :].cpu().numpy()

            return continuation, self.DEFAULT_SAMPLE_RATE

        except Exception as e:
            logger.error(f"[MusicGen] Continuation failed: {e}")
            raise
