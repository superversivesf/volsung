"""
AudioLDM generator for sound effects synthesis.

Wraps the AudioLDM2 model for text-to-sound-effect generation.
Supports the AudioLDM2 model family from cvssp/audioldm2.
"""

import logging
from typing import Any, Tuple

import numpy as np
import torch

from volsung.models.base import GeneratorBase

logger = logging.getLogger(__name__)


class AudioLDMGenerator(GeneratorBase):
    """AudioLDM-based sound effects generator.

    Wraps the AudioLDM2 diffusion model for high-quality sound effect generation.
    Uses latent diffusion in the audio space for efficient generation.

    Models supported:
    - cvssp/audioldm2 (default)
    - cvssp/audioldm2-large
    - cvssp/audioldm2-music

    Example:
        ```python
        generator = AudioLDMGenerator(model_size="base")
        generator.load(device="cuda", dtype="float16")
        audio, sr = generator.generate(
            prompt="Thunder rumbling in the distance",
            duration=5.0,
        )
        ```
    """

    # Model configurations
    MODEL_CONFIGS = {
        "base": {
            "model_id": "cvssp/audioldm2",
            "vram_gb": 4.0,
        },
        "large": {
            "model_id": "cvssp/audioldm2-large",
            "vram_gb": 8.0,
        },
        "music": {
            "model_id": "cvssp/audioldm2-music",
            "vram_gb": 6.0,
        },
    }

    def __init__(self, model_size: str = "base"):
        """Initialize AudioLDM generator.

        Args:
            model_size: Model variant ("base", "large", "music")

        Raises:
            ValueError: If model_size is not supported
        """
        if model_size not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model size: {model_size}. "
                f"Supported: {list(self.MODEL_CONFIGS.keys())}"
            )

        self.model_size = model_size
        self.config = self.MODEL_CONFIGS[model_size]
        self.model = None
        self.vocoder = None
        self.device = None
        self.dtype = None

    @property
    def model_id(self) -> str:
        """Unique identifier for this generator.

        Returns:
            Model ID string (e.g., "audioldm2-base")
        """
        return f"audioldm2-{self.model_size}"

    @property
    def model_name(self) -> str:
        """Human-readable name.

        Returns:
            Display name (e.g., "AudioLDM2 Base")
        """
        return f"AudioLDM2 {self.model_size.title()}"

    @property
    def required_vram_gb(self) -> float:
        """Estimated VRAM requirement in GB.

        Returns:
            Approximate GB of VRAM needed
        """
        return self.config["vram_gb"]

    def load(self, device: str, dtype: str) -> None:
        """Load the AudioLDM model.

        Args:
            device: Device to load on ("cuda", "cuda:0", "cpu", "mps")
            dtype: Data type ("float16", "float32", "bfloat16")

        Raises:
            ImportError: If diffusers or transformers not installed
            Exception: If model loading fails
        """
        try:
            from diffusers import AudioLDM2Pipeline
        except ImportError:
            raise ImportError(
                "AudioLDM requires diffusers. Install with: pip install diffusers"
            )

        self.device = device
        self.dtype = getattr(torch, dtype)

        logger.info(f"Loading {self.model_name} on {device} with {dtype}...")

        # Load pipeline
        self.model = AudioLDM2Pipeline.from_pretrained(
            self.config["model_id"],
            torch_dtype=self.dtype,
        )

        # Move to device
        if device != "cpu":
            self.model = self.model.to(device)

        logger.info(f"{self.model_name} loaded successfully")

    def unload(self) -> None:
        """Unload the model and free resources.

        Clears GPU memory and removes model references.
        """
        if self.model is not None:
            logger.info(f"Unloading {self.model_name}...")

            # Delete model components
            self.model = None
            self.vocoder = None

            # Clear GPU cache
            if self.device and "cuda" in self.device:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logger.info(f"{self.model_name} unloaded")

    def generate(
        self,
        prompt: str,
        duration: float,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        negative_prompt: str = "",
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Generate sound effects from text prompt.

        Args:
            prompt: Text description of desired sound effect
            duration: Target duration in seconds
            num_inference_steps: Number of denoising steps (10-200)
            guidance_scale: Prompt adherence (1.0-20.0)
            negative_prompt: What to avoid in generation
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (audio_array, sample_rate)
            audio_array is numpy float32 array

        Raises:
            RuntimeError: If model not loaded
            Exception: If generation fails
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # AudioLDM2 works with specific durations (must be multiple of model config)
        # Round to nearest valid duration
        duration = max(1.0, min(10.0, duration))

        logger.info(
            f"Generating SFX: '{prompt[:50]}...' "
            f"duration={duration}s steps={num_inference_steps}"
        )

        try:
            # Generate audio using AudioLDM2
            output = self.model(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                audio_length_in_s=int(duration),
            )

            # Extract audio
            audio = output.audios[0]  # shape: (samples,)
            sample_rate = output.sample_rate

            # Ensure correct format
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()

            # Normalize to [-1, 1] range
            audio = audio.astype(np.float32)
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                audio = audio / max_val

            logger.info(
                f"Generated: duration={len(audio) / sample_rate:.2f}s "
                f"sample_rate={sample_rate}"
            )

            return audio, sample_rate

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def get_info(self) -> dict[str, Any]:
        """Get generator information.

        Returns:
            Dictionary with generator metadata
        """
        info = super().get_info()
        info.update(
            {
                "model_size": self.model_size,
                "huggingface_id": self.config["model_id"],
                "device": self.device,
                "dtype": str(self.dtype) if self.dtype else None,
            }
        )
        return info
