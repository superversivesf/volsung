"""StyleTTS 2 model manager for Volsung.

Extends ModelManagerBase to provide StyleTTS 2 voice cloning capabilities.
Supports lazy loading and idle timeout monitoring.
"""

import logging
from typing import Optional

import torch

from volsung.models.base import ModelConfig, ModelManagerBase
from volsung.models.types import AudioResult, AudioType

logger = logging.getLogger(__name__)


class StyleTTS2Manager(ModelManagerBase):
    """Manager for StyleTTS 2 models with voice cloning and synthesis capabilities.

    Extends ModelManagerBase to provide:
    - Style extraction: Extract style vectors from reference audio
    - Voice synthesis: Clone voices from reference audio and synthesize text

    Uses StyleTTS2Generator for actual model inference.

    Attributes:
        model_id: Unique identifier for this model
        device: Computed device (cuda, mps, or cpu)
        dtype: Computed dtype (bfloat16 for CUDA, float32 otherwise)

    Example:
        ```python
        from volsung.tts.managers.styletts2 import StyleTTS2Manager

        manager = StyleTTS2Manager(
            device="cuda:0",
            idle_timeout=300,
        )

        # Extract style from reference audio
        style = manager.compute_style(ref_audio_b64)

        # Generate speech with cloned voice
        result = manager.generate(
            text="Hello, world!",
            ref_audio_b64=ref_audio_b64,
            embedding_scale=1.0,
            alpha=0.3,
            beta=0.7,
            diffusion_steps=10,
        )
        ```
    """

    def __init__(
        self,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        idle_timeout: int = 300,
    ):
        """Initialize the StyleTTS 2 model manager.

        Args:
            device: Device override (auto-detected if None)
            dtype: Dtype override (auto-detected if None)
            idle_timeout: Seconds before unloading model (0 = never unload)
        """
        self._model_id = "styletts2"

        # Auto-detect device and dtype
        self.device = device or self._get_device()
        self.dtype = dtype or self._get_dtype()

        # Initialize base class with config
        config = ModelConfig(
            model_id=self._model_id,
            model_name="StyleTTS 2",
            device=self.device,
            dtype=self.dtype,
            idle_timeout_seconds=idle_timeout,
        )
        super().__init__(config)

        # This will be set by _load_model
        self._generator: Optional[object] = None

    def _get_device(self) -> str:
        """Get the best available device.

        Returns:
            Device string: "cuda:0" for CUDA, "mps" for Apple Silicon, "cpu" otherwise
        """
        if torch.cuda.is_available():
            return "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _get_dtype(self) -> str:
        """Get the optimal dtype for the device.

        Returns:
            "bfloat16" for CUDA, "float32" otherwise
        """
        if torch.cuda.is_available():
            return "bfloat16"
        return "float32"

    def _load_model(self) -> None:
        """Load the StyleTTS 2 model.

        Loads the StyleTTS2Generator and initializes it.
        Called automatically by _ensure_loaded() when needed.
        """
        from ..generators.styletts2 import StyleTTS2Generator

        logger.info(f"Loading StyleTTS 2 model on {self.device} with {self.dtype}...")

        # Clear GPU cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Create and load generator
        self._generator = StyleTTS2Generator()
        self._generator.load(device=self.device, dtype=self.dtype)

        self._model = self._generator

        logger.info("StyleTTS 2 model loaded successfully")

    def _unload_model(self) -> None:
        """Unload the StyleTTS 2 model and free resources."""
        logger.info("Unloading StyleTTS 2 model...")

        if self._generator is not None:
            self._generator.unload()
            self._generator = None

        self._model = None

        self._clear_memory()
        logger.info("StyleTTS 2 model unloaded")

    def load(self) -> None:
        """Load the StyleTTS 2 model.

        Preloads the model into memory. Safe to call multiple times
        (idempotent - will not reload if already loaded).
        """
        self._ensure_loaded()

    def compute_style(self, ref_audio_b64: str) -> torch.Tensor:
        """Extract style vector from reference audio.

        Args:
            ref_audio_b64: Base64-encoded reference audio (WAV format)

        Returns:
            Style vector tensor

        Raises:
            RuntimeError: If model fails to extract style
        """
        self._ensure_loaded()

        if self._generator is None:
            raise RuntimeError("Generator not loaded")

        logger.info("[STYLE_EXTRACT] Extracting style from reference audio")

        try:
            style_vector = self._generator.compute_style(ref_audio_b64)
            logger.info(
                f"[STYLE_EXTRACT] Style vector extracted: shape={style_vector.shape}"
            )
            return style_vector
        except Exception as e:
            logger.error(f"[STYLE_EXTRACT] Failed: {e}", exc_info=True)
            raise RuntimeError(f"Style extraction failed: {e}") from e

    def generate(
        self,
        text: str,
        ref_audio_b64: str,
        embedding_scale: float = 1.0,
        alpha: float = 0.3,
        beta: float = 0.7,
        diffusion_steps: int = 10,
        **kwargs,
    ) -> AudioResult:
        """Generate speech from text with voice cloning.

        Clones the voice from the reference audio and synthesizes
        the text in that voice character.

        Args:
            text: Text to synthesize
            ref_audio_b64: Base64-encoded reference audio for voice cloning
            embedding_scale: Scale for speaker embedding (1.0 = normal)
            alpha: Style mixing parameter (0-1)
            beta: Prosody control parameter (0-1)
            diffusion_steps: Number of diffusion steps (higher = better quality)
            **kwargs: Additional generation parameters

        Returns:
            AudioResult with generated audio and metadata

        Raises:
            RuntimeError: If generation fails
            ValueError: If reference audio is invalid
        """
        self._ensure_loaded()

        if self._generator is None:
            raise RuntimeError("Generator not loaded")

        logger.info(f"[GENERATE] text='{text[:50]}...'")

        try:
            # Generate audio
            audio, sr = self._generator.generate(
                text=text,
                ref_audio=ref_audio_b64,
                embedding_scale=embedding_scale,
                alpha=alpha,
                beta=beta,
                diffusion_steps=diffusion_steps,
                **kwargs,
            )

            duration = len(audio) / sr

            logger.info(
                f"[GENERATE] Generated: duration={duration:.2f}s, sample_rate={sr}"
            )

            # Convert to base64 for AudioResult
            import base64
            from io import BytesIO

            import soundfile as sf

            buffer = BytesIO()
            sf.write(buffer, audio, sr, format="WAV")
            buffer.seek(0)
            audio_b64 = base64.b64encode(buffer.read()).decode()

            return AudioResult(
                audio=audio_b64,
                sample_rate=sr,
                duration=duration,
                audio_type=AudioType.TTS,
                generator=self._model_id,
                prompt=text,
                metadata={
                    "model": self._model_id,
                    "text": text,
                    "embedding_scale": embedding_scale,
                    "alpha": alpha,
                    "beta": beta,
                    "diffusion_steps": diffusion_steps,
                },
            )

        except Exception as e:
            logger.error(f"[GENERATE] Failed: {e}", exc_info=True)
            raise RuntimeError(f"Generation failed: {e}") from e

    def get_info(self) -> dict:
        """Get manager information.

        Returns:
            Dictionary with manager metadata
        """
        info = {
            "model_id": self._model_id,
            "device": self.device,
            "dtype": self.dtype,
            "loaded": self.is_loaded,
            "idle_seconds": self.idle_seconds,
        }
        if self._generator:
            info.update(self._generator.get_info())
        return info


# Singleton instance
_styletts2_manager: Optional[StyleTTS2Manager] = None


def get_styletts2_manager() -> StyleTTS2Manager:
    """Get the singleton StyleTTS2Manager instance.

    Returns:
        Singleton StyleTTS2Manager instance
    """
    global _styletts2_manager
    if _styletts2_manager is None:
        _styletts2_manager = StyleTTS2Manager()
    return _styletts2_manager
