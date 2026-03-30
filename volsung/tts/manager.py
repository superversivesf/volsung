"""TTS (Text-to-Speech) model manager for Volsung.

Extends ModelManagerBase to provide voice design and voice cloning capabilities
using Qwen3-TTS models. Supports lazy loading and idle timeout monitoring.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch

from volsung.models.base import ModelConfig, ModelManagerBase
from volsung.models.types import AudioResult
from .schemas import VoiceDesignRequest, SynthesizeRequest

logger = logging.getLogger(__name__)


class TTSModelManager(ModelManagerBase):
    """Manager for TTS models with voice design and synthesis capabilities.

    Extends ModelManagerBase to provide:
    - Voice design: Generate unique voice characters from text descriptions
    - Voice synthesis: Clone voices from reference audio and synthesize new text

    Uses two Qwen3-TTS models:
    - VoiceDesign model: Creates new voice samples from descriptions
    - Base model: Clones voices and synthesizes text

    Attributes:
        voice_design_model_id: HuggingFace model ID for voice design
        base_model_id: HuggingFace model ID for voice cloning/synthesis
        device: Computed device (cuda, mps, or cpu)
        dtype: Computed dtype (bfloat16 for CUDA, float32 otherwise)

    Example:
        ```python
        from volsung.config import get_config
        from volsung.tts.manager import TTSModelManager

        config = get_config()
        manager = TTSModelManager(
            voice_design_model_id=config.tts.voice_design_model,
            base_model_id=config.tts.base_model,
            idle_timeout=config.tts.idle_timeout,
        )

        # Design a voice
        request = VoiceDesignRequest(...)
        result = manager.voice_design(request)

        # Synthesize with cloned voice
        synth_request = SynthesizeRequest(...)
        result = manager.synthesize(synth_request)
        ```
    """

    def __init__(
        self,
        voice_design_model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        base_model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        idle_timeout: int = 300,
    ):
        """Initialize the TTS model manager.

        Args:
            voice_design_model_id: HuggingFace model ID for voice design
            base_model_id: HuggingFace model ID for voice cloning/synthesis
            device: Device override (auto-detected if None)
            dtype: Dtype override (auto-detected if None)
            idle_timeout: Seconds before unloading model (0 = never unload)
        """
        self.voice_design_model_id = voice_design_model_id
        self.base_model_id = base_model_id

        # Auto-detect device and dtype
        self.device = device or self._get_device()
        self.dtype = dtype or self._get_dtype()

        # Initialize base class with config
        config = ModelConfig(
            model_id="tts-manager",
            model_name="TTS Manager",
            device=self.device,
            dtype=self.dtype,
            idle_timeout_seconds=idle_timeout,
        )
        super().__init__(config)

        # These will be set by _load_model
        self._voice_design_model: Optional[torch.nn.Module] = None
        self._base_model: Optional[torch.nn.Module] = None

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
        """Load both TTS models.

        Loads the VoiceDesign and Base models from HuggingFace.
        Called automatically by _ensure_loaded() when needed.
        """
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError:
            raise ImportError(
                "qwen_tts package is required. Install with: pip install qwen-tts"
            )

        logger.info(f"Loading TTS models on {self.device} with {self.dtype}...")

        # Clear GPU cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load VoiceDesign model
        logger.info(f"Loading VoiceDesign model: {self.voice_design_model_id}")
        self._voice_design_model = Qwen3TTSModel.from_pretrained(
            self.voice_design_model_id,
            device_map=self.device,
            dtype=getattr(torch, self.dtype),
        )
        logger.info("VoiceDesign model loaded successfully")

        # Clear cache between models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load Base model
        logger.info(f"Loading Base model: {self.base_model_id}")
        self._base_model = Qwen3TTSModel.from_pretrained(
            self.base_model_id,
            device_map=self.device,
            dtype=getattr(torch, self.dtype),
        )
        logger.info("Base model loaded successfully")

        self._model = {
            "voice_design": self._voice_design_model,
            "base": self._base_model,
        }

    def _unload_model(self) -> None:
        """Unload both TTS models and free resources."""
        logger.info("Unloading TTS models...")

        self._voice_design_model = None
        self._base_model = None
        self._model = None

        self._clear_memory()
        logger.info("TTS models unloaded")

    def generate(self, *args, **kwargs) -> AudioResult:
        """Not implemented - use voice_design() or synthesize() instead."""
        raise NotImplementedError(
            "Use voice_design() or synthesize() methods instead of generate()"
        )

    def voice_design(self, request: VoiceDesignRequest) -> AudioResult:
        """Generate a voice sample from a text description.

        Creates a unique voice character based on the instruction and
        generates audio for the provided text in that voice.

        Args:
            request: VoiceDesignRequest with text, language, and instruct

        Returns:
            AudioResult with generated audio and metadata

        Raises:
            RuntimeError: If voice design fails
        """
        self._ensure_loaded()

        if self._voice_design_model is None:
            raise RuntimeError("VoiceDesign model not loaded")

        logger.info(
            f"[VOICE_DESIGN] text='{request.text[:50]}...' "
            f"language='{request.language}' instruct='{request.instruct[:50]}...'"
        )

        try:
            wavs, sr = self._voice_design_model.generate_voice_design(
                text=request.text,
                language=request.language,
                instruct=request.instruct,
            )

            audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
            duration = len(audio) / sr

            logger.info(
                f"[VOICE_DESIGN] Generated: duration={duration:.2f}s, sample_rate={sr}"
            )

            return AudioResult(
                audio=audio,
                sample_rate=sr,
                duration=duration,
                metadata={
                    "model": self.voice_design_model_id,
                    "language": request.language,
                    "text": request.text,
                },
            )

        except Exception as e:
            logger.error(f"[VOICE_DESIGN] Failed: {e}", exc_info=True)
            raise RuntimeError(f"Voice design failed: {e}") from e

    def synthesize(self, request: SynthesizeRequest) -> AudioResult:
        """Synthesize text using a cloned voice from reference audio.

        Clones the voice from the reference audio and synthesizes
        the new text in that voice character.

        Args:
            request: SynthesizeRequest with ref_audio, ref_text, text, and language

        Returns:
            AudioResult with synthesized audio and metadata

        Raises:
            RuntimeError: If synthesis fails
            ValueError: If reference audio is invalid
        """
        import base64
        from io import BytesIO

        import soundfile as sf

        self._ensure_loaded()

        if self._base_model is None:
            raise RuntimeError("Base model not loaded")

        logger.info(
            f"[SYNTHESIZE] text='{request.text[:50]}...' language='{request.language}'"
        )

        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(request.ref_audio)
            buffer = BytesIO(audio_bytes)
            ref_audio, ref_sr = sf.read(buffer)

            ref_duration = len(ref_audio) / ref_sr
            logger.info(
                f"[SYNTHESIZE] Reference: duration={ref_duration:.2f}s, sample_rate={ref_sr}"
            )

            # Generate cloned voice
            wavs, sr = self._base_model.generate_voice_clone(
                ref_audio=(ref_audio, ref_sr),
                ref_text=request.ref_text,
                text=request.text,
                language=request.language,
            )

            audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
            duration = len(audio) / sr

            logger.info(
                f"[SYNTHESIZE] Generated: duration={duration:.2f}s, sample_rate={sr}"
            )

            return AudioResult(
                audio=audio,
                sample_rate=sr,
                duration=duration,
                metadata={
                    "model": self.base_model_id,
                    "language": request.language,
                    "text": request.text,
                    "ref_duration": ref_duration,
                },
            )

        except Exception as e:
            logger.error(f"[SYNTHESIZE] Failed: {e}", exc_info=True)
            raise RuntimeError(f"Synthesis failed: {e}") from e

    def audio_to_base64(self, audio_result: AudioResult) -> str:
        """Convert AudioResult to base64-encoded WAV string.

        Args:
            audio_result: AudioResult containing audio array

        Returns:
            Base64-encoded WAV string
        """
        import base64
        from io import BytesIO

        import soundfile as sf

        buffer = BytesIO()
        sf.write(buffer, audio_result.audio, audio_result.sample_rate, format="WAV")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode()
