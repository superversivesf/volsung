"""
Standalone SFX (Sound Effects) Service - AudioLDM via diffusers.

This service provides sound effects generation using AudioLDM2 from the diffusers library.
It runs as a standalone FastAPI service on port 8003.

Volsung SFX Service - Standalone sound effects generation using AudioLDM2.
"""

import base64
import logging
import time
from io import BytesIO
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Schemas
# ============================================================================


class SFXGenerateRequest(BaseModel):
    """Request for generating sound effects from text description."""

    description: str = Field(
        ...,
        description="Natural language description of desired sound effect",
        min_length=1,
        max_length=1000,
    )
    duration: float = Field(
        default=5.0,
        ge=1.0,
        le=10.0,
        description="Target duration in seconds (1.0 to 10.0)",
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional category hint (e.g., 'nature', 'mechanical', 'urban')",
    )
    num_inference_steps: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Number of denoising steps (higher = better quality, slower)",
    )
    guidance_scale: float = Field(
        default=3.5,
        ge=1.0,
        le=20.0,
        description="Prompt adherence (higher = more faithful to prompt)",
    )


class SFXMetadata(BaseModel):
    """Metadata for generated sound effects."""

    duration: float = Field(..., description="Audio duration in seconds")
    sample_rate: int = Field(..., description="Sample rate in Hz")
    category: Optional[str] = Field(default=None, description="SFX category")
    generation_time_ms: float = Field(..., description="Time taken to generate (ms)")
    model_used: str = Field(..., description="Model identifier")
    num_inference_steps: int = Field(default=50, description="Denoising steps used")
    guidance_scale: float = Field(default=3.5, description="Guidance scale used")


class SFXGenerateResponse(BaseModel):
    """Response containing generated sound effect and metadata."""

    audio: str = Field(..., description="Base64-encoded WAV audio data")
    sample_rate: int = Field(..., description="Sample rate in Hz")
    metadata: SFXMetadata = Field(..., description="Generation metadata")


class HealthResponse(BaseModel):
    """Health check response for SFX service."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the SFX model is loaded")
    model_name: str = Field(..., description="Name of the model")
    device: str = Field(..., description="Device being used (cuda, cpu, mps)")


# ============================================================================
# AudioLDM Generator
# ============================================================================


class AudioLDMGenerator:
    """AudioLDM-based sound effects generator using diffusers.

    Wraps the AudioLDM2 diffusion model for high-quality sound effect generation.
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
        """
        if model_size not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model size: {model_size}. "
                f"Supported: {list(self.MODEL_CONFIGS.keys())}"
            )

        self.model_size = model_size
        self.config = self.MODEL_CONFIGS[model_size]
        self.model = None
        self.device = None
        self.dtype = None

    @property
    def model_id(self) -> str:
        """Unique identifier for this generator."""
        return f"audioldm2-{self.model_size}"

    @property
    def model_name(self) -> str:
        """Human-readable name."""
        return f"AudioLDM2 {self.model_size.title()}"

    def load(self, device: Optional[str] = None, dtype: Optional[str] = None) -> None:
        """Load the AudioLDM model.

        Args:
            device: Device to load on ("cuda", "cpu", "mps")
            dtype: Data type ("float16", "float32", "bfloat16")
        """
        try:
            from diffusers import AudioLDM2Pipeline
        except ImportError:
            raise ImportError(
                "AudioLDM requires diffusers. Install with: pip install diffusers"
            )

        # Auto-detect device if not specified
        if device is None:
            device = self._get_device()

        if dtype is None:
            dtype = "float16" if device == "cuda" else "float32"

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
        """Unload the model and free resources."""
        if self.model is not None:
            logger.info(f"Unloading {self.model_name}...")
            self.model = None

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
    ) -> tuple[np.ndarray, int]:
        """Generate sound effects from text prompt.

        Args:
            prompt: Text description of desired sound effect
            duration: Target duration in seconds
            num_inference_steps: Number of denoising steps
            guidance_scale: Prompt adherence
            negative_prompt: What to avoid in generation

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # AudioLDM2 works with specific durations
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
            audio = output.audios[0]
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

    def _get_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Volsung SFX Service",
    description="Standalone sound effects generation service using AudioLDM2",
    version="1.0.0",
)

# Global generator instance
sfx_generator: Optional[AudioLDMGenerator] = None


def get_generator() -> AudioLDMGenerator:
    """Get or create the SFX generator."""
    global sfx_generator
    if sfx_generator is None:
        sfx_generator = AudioLDMGenerator(model_size="base")
    return sfx_generator


def audio_to_base64(audio: np.ndarray, sample_rate: int) -> str:
    """Convert audio array to base64-encoded WAV."""
    buffer = BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    logger.info("=" * 60)
    logger.info("Volsung SFX Service")
    logger.info("=" * 60)
    logger.info("Using AudioLDM2 via diffusers")
    logger.info("Model will load on first generation request")
    logger.info("")
    logger.info("Endpoints:")
    logger.info("  GET  /health        - Health check")
    logger.info("  POST /sfx/generate  - Generate sound effects")
    logger.info("=" * 60)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check service health status."""
    generator = get_generator()
    return HealthResponse(
        status="healthy",
        model_loaded=generator.model is not None,
        model_name=generator.model_name,
        device=generator.device or "not_loaded",
    )


@app.post("/sfx/generate", response_model=SFXGenerateResponse)
async def sfx_generate(req: SFXGenerateRequest):
    """
    Generate sound effects from text description.

    Creates sound effects up to 10 seconds from natural language description.
    Uses AudioLDM2 diffusion model via diffusers library.

    Example:
        ```json
        {
            "description": "Thunder rumbling in the distance",
            "duration": 5.0,
            "category": "nature",
            "num_inference_steps": 50,
            "guidance_scale": 3.5
        }
        ```
    """
    # Validate duration
    if req.duration > 10.0:
        raise HTTPException(status_code=400, detail="Duration must be <= 10 seconds")
    if req.duration <= 0:
        raise HTTPException(status_code=400, detail="Duration must be positive")

    generator = get_generator()

    # Lazy load model on first request
    if generator.model is None:
        try:
            generator.load()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(
                status_code=503, detail=f"Failed to load SFX model: {str(e)}"
            )

    try:
        logger.info(
            f"[SFX_GENERATE] Request: description='{req.description[:50]}...' "
            f"duration={req.duration}s category={req.category}"
        )
        start_time = time.time()

        # Generate sound effect
        audio, sample_rate = generator.generate(
            prompt=req.description,
            duration=req.duration,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
        )

        elapsed_ms = (time.time() - start_time) * 1000
        duration_seconds = len(audio) / sample_rate

        logger.info(
            f"[SFX_GENERATE] Generated: duration={duration_seconds:.2f}s "
            f"sample_rate={sample_rate} elapsed={elapsed_ms:.0f}ms"
        )

        # Convert to base64
        audio_base64 = audio_to_base64(audio, sample_rate)
        audio_size_kb = len(audio_base64) * 3 // 4 // 1024
        logger.info(f"[SFX_GENERATE] Response: audio_size={audio_size_kb}KB")

        # Build metadata
        metadata = SFXMetadata(
            duration=duration_seconds,
            sample_rate=sample_rate,
            category=req.category,
            generation_time_ms=elapsed_ms,
            model_used=generator.model_id,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
        )

        return SFXGenerateResponse(
            audio=audio_base64,
            sample_rate=sample_rate,
            metadata=metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SFX_GENERATE] Failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"SFX generation failed: {str(e)}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        log_level="info",
        access_log=True,
    )
