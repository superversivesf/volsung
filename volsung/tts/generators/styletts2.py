"""StyleTTS 2 generator implementation.

Wraps StyleTTS 2 model for text-to-speech synthesis with voice cloning.
Part of the Volsung TTS generation system.

References:
    - https://github.com/yl4579/StyleTTS2
    - https://huggingface.co/yl4579/StyleTTS2-LibriTTS
"""

import gc
import logging
import os
import tempfile
from typing import Any, Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from volsung.audio.utils import base64_to_audio
from volsung.models.base import GeneratorBase

logger = logging.getLogger(__name__)


class StyleTTS2Generator(GeneratorBase):
    """StyleTTS 2 model generator.

    Wraps StyleTTS 2 for conditional text-to-speech synthesis.
    Supports voice cloning from reference audio with emotion control.

    Attributes:
        model: The underlying StyleTTS2 model instance
        _device: Device the model is loaded on
        _dtype: Data type used for model

    Example:
        ```python
        generator = StyleTTS2Generator()
        generator.load(device="cuda:0", dtype="float16")

        # Generate with voice cloning
        audio, sr = generator.generate(
            text="Hello, this is a test.",
            ref_audio=base64_reference_audio,
            embedding_scale=1.5,
            alpha=0.3,
            beta=0.7,
        )
        ```

    References:
        - https://github.com/yl4579/StyleTTS2
        - https://huggingface.co/yl4579/StyleTTS2-LibriTTS
    """

    MODEL_ID = "styletts2"
    MODEL_NAME = "StyleTTS2"
    DEFAULT_SAMPLE_RATE = 24000
    REQUIRED_VRAM_GB = 4.0

    def __init__(self):
        """Initialize the StyleTTS2 generator."""
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
        """Load the StyleTTS2 model from HuggingFace.

        Args:
            device: Device to load on ("cuda", "cuda:0", "cpu", "mps")
            dtype: Data type ("float16", "float32", "bfloat16")

        Raises:
            ImportError: If styletts2 package is not installed
            Exception: If model loading fails
        """
        try:
            from styletts2 import tts
        except ImportError:
            raise ImportError(
                "styletts2 is required for StyleTTS2. "
                "Install with: pip install styletts2"
            )

        logger.info("Loading StyleTTS2 model (yl4579/StyleTTS2-LibriTTS)...")

        try:
            # Initialize StyleTTS2 model
            # The model auto-downloads from HuggingFace on first use
            self.model = tts.StyleTTS2()

            self._device = device
            self._dtype = dtype

            # Move to device if needed
            if hasattr(self.model, "to"):
                self.model.to(device)

            logger.info(f"StyleTTS2 loaded on {device}")

        except Exception as e:
            logger.error(f"Failed to load StyleTTS2: {e}")
            raise

    def unload(self) -> None:
        """Unload and cleanup resources.

        Frees GPU memory and clears model references.
        """
        if self.model is not None:
            logger.info("Unloading StyleTTS2 model...")

            # Delete model reference
            del self.model
            self.model = None

            # Clear GPU cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            self._device = None
            self._dtype = None

            logger.info("StyleTTS2 unloaded")

    def generate(
        self,
        text: str,
        ref_audio: str,
        embedding_scale: float = 1.0,
        alpha: float = 0.3,
        beta: float = 0.7,
        diffusion_steps: int = 10,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech with voice cloning.

        Args:
            text: Text to synthesize
            ref_audio: Base64-encoded reference audio for voice cloning
            embedding_scale: Emotion intensity (1.0-10.0, default: 1.0)
            alpha: Voice similarity control (0-1, default: 0.3)
            beta: Emotion similarity control (0-1, default: 0.7)
            diffusion_steps: Number of diffusion steps for style diversity (default: 10)
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (audio_array, sample_rate)
            audio_array is numpy array of float32 samples in [-1, 1]

        Raises:
            RuntimeError: If model not loaded
            ValueError: If reference audio is invalid
            Exception: If generation fails
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        logger.info(f"[StyleTTS2] Generating: '{text[:50]}...'")

        # Decode reference audio from base64
        try:
            ref_audio_array, ref_sr = base64_to_audio(ref_audio)
        except Exception as e:
            raise ValueError(f"Failed to decode reference audio: {e}")

        # Write reference audio to temporary file
        # StyleTTS2 requires a file path for target voice
        ref_temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_temp:
                sf.write(ref_temp, ref_audio_array, ref_sr, format="WAV")
                ref_temp_path = ref_temp.name

            # Run inference
            # StyleTTS2 inference signature:
            # inference(text, target_voice_path, embedding_scale, alpha, beta)
            with torch.no_grad():
                wav = self.model.inference(
                    text=text,
                    target_voice_path=ref_temp_path,
                    embedding_scale=embedding_scale,
                    alpha=alpha,
                    beta=beta,
                    # diffusion_steps may be supported in newer versions
                )

            # Ensure output is numpy array
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()

            # Ensure mono and correct shape
            if wav.ndim > 1:
                wav = wav.mean(axis=0)

            # Normalize to [-1, 1] if needed
            if wav.max() > 1.0 or wav.min() < -1.0:
                wav = wav / max(abs(wav.max()), abs(wav.min()))

            duration = len(wav) / self.DEFAULT_SAMPLE_RATE
            logger.info(
                f"[StyleTTS2] Generated: {len(wav)} samples "
                f"@ {self.DEFAULT_SAMPLE_RATE}Hz, {duration:.2f}s"
            )

            return wav, self.DEFAULT_SAMPLE_RATE

        except Exception as e:
            logger.error(f"[StyleTTS2] Generation failed: {e}")
            raise

        finally:
            # Cleanup temporary file
            if ref_temp_path and os.path.exists(ref_temp_path):
                try:
                    os.unlink(ref_temp_path)
                except OSError:
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
            "loaded": self.model is not None,
            "device": self._device,
            "dtype": self._dtype,
            "sample_rate": self.DEFAULT_SAMPLE_RATE,
        }
