"""
FastAPI server for Volsung - Voice synthesis, music, SFX, and audio stitching.

Volsung - Voice synthesis server for Qwen3-TTS with music, SFX, and composition.
"""

import os
from pathlib import Path

# Set HF_HOME before importing transformers/huggingface_hub
# This ensures models are stored in the project's models/ directory
project_root = Path(__file__).parent.parent
models_dir = project_root / "models"
models_dir.mkdir(exist_ok=True)
os.environ.setdefault("HF_HOME", str(models_dir))
os.environ.setdefault("HF_CACHE_HOME", str(models_dir))

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Literal, Union
import torch
import numpy as np
import soundfile as sf
from io import BytesIO
import base64
import logging
import time
import threading

from qwen_tts import Qwen3TTSModel

# Import module routers
from volsung.tts.endpoints import router as tts_router
from volsung.music.endpoints import router as music_router
from volsung.sfx.endpoints import router as sfx_router
from volsung.sfx.schemas import (
    SFXGenerateRequest,
    SFXGenerateResponse,
    SFXLayerRequest,
    SFXLayerResponse,
    SFXMetadata,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Volsung",
    description="Voice synthesis, music generation, SFX, and audio composition server",
    version="1.0.0",
)

# Register module routers
app.include_router(tts_router)
app.include_router(music_router)
app.include_router(sfx_router)

voice_design_model = None
base_model = None
music_model = None
sfx_model = None
models_loaded = False

# Thread-safe idle monitoring globals
last_access_time = 0.0
idle_lock = threading.Lock()
idle_monitor_thread = None
IDLE_TIMEOUT_SECONDS = 300  # 5 minutes
IDLE_CHECK_INTERVAL = 60  # Check every 60 seconds
idle_monitor_running = False


class VoiceDesignRequest(BaseModel):
    """Request for generating a voice sample from a description."""

    text: str
    language: str = "English"
    instruct: str


class VoiceDesignResponse(BaseModel):
    """Response containing generated audio for use as reference."""

    audio: str
    sample_rate: int


class SynthesizeRequest(BaseModel):
    """Request for synthesizing text with a cloned voice."""

    ref_audio: str
    ref_text: str
    text: str
    language: str = "English"


class SynthesizeResponse(BaseModel):
    """Response containing synthesized audio."""

    audio: str
    sample_rate: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    voice_design_model: bool
    base_model: bool
    music_model: bool
    sfx_model: bool
    tts_router: bool
    music_router: bool
    sfx_router: bool


class PreloadRequest(BaseModel):
    """Request for preloading models.

    Accepts either a single model name or a list of models to preload.
    """

    models: Union[str, List[str]] = "all"


class PreloadResponse(BaseModel):
    """Preload response with detailed status."""

    status: str
    models: List[str]
    loaded: List[str]
    unloaded: List[str]


# ============================================================================
# Music Generation Models
# ============================================================================


class MusicGenerateRequest(BaseModel):
    """Request for generating music from text description."""

    description: str
    duration: float = 30.0
    genre: str | None = None
    mood: str | None = None
    tempo: str | None = None  # slow, medium, fast


class MusicMetadata(BaseModel):
    """Metadata for generated music."""

    duration: float
    sample_rate: int
    genre_tags: list[str]
    generation_time_ms: float
    model_used: str


class MusicGenerateResponse(BaseModel):
    """Response containing generated music and metadata."""

    audio: str
    sample_rate: int
    metadata: MusicMetadata


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def update_last_access_time():
    """Update the last access time for idle monitoring (thread-safe)."""
    global last_access_time
    with idle_lock:
        last_access_time = time.time()


def unload_models_if_idle():
    """Unload models if they've been idle for too long."""
    global voice_design_model, base_model, models_loaded

    with idle_lock:
        if not models_loaded:
            return

        idle_duration = time.time() - last_access_time
        if idle_duration < IDLE_TIMEOUT_SECONDS:
            return

        logger.info(
            f"Models idle for {idle_duration:.0f}s, unloading to free GPU memory..."
        )

        try:
            # Unload models
            voice_design_model = None
            base_model = None
            models_loaded = False

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")

            logger.info("Models unloaded due to idle timeout")
        except Exception as e:
            logger.error(f"Error unloading models: {e}", exc_info=True)


def idle_monitor_loop():
    """Background thread loop that monitors for idle models."""
    global idle_monitor_running
    logger.info("Idle monitor thread started")

    while idle_monitor_running:
        try:
            unload_models_if_idle()
        except Exception as e:
            logger.error(f"Error in idle monitor: {e}", exc_info=True)

        # Sleep with periodic checks to allow quick shutdown
        for _ in range(IDLE_CHECK_INTERVAL):
            if not idle_monitor_running:
                break
            time.sleep(1)

    logger.info("Idle monitor thread stopped")


def start_idle_monitor():
    """Start the idle monitor background thread."""
    global idle_monitor_thread, idle_monitor_running

    with idle_lock:
        if idle_monitor_running:
            return

        idle_monitor_running = True
        idle_monitor_thread = threading.Thread(target=idle_monitor_loop, daemon=True)
        idle_monitor_thread.start()
        logger.info(f"Started idle monitor (timeout: {IDLE_TIMEOUT_SECONDS}s)")


def stop_idle_monitor():
    """Stop the idle monitor background thread."""
    global idle_monitor_running

    with idle_lock:
        idle_monitor_running = False

    if idle_monitor_thread and idle_monitor_thread.is_alive():
        idle_monitor_thread.join(timeout=5.0)
        logger.info("Idle monitor stopped")


def audio_to_base64(wav_array: np.ndarray, sample_rate: int) -> str:
    """Convert audio array to base64-encoded WAV."""
    buffer = BytesIO()
    sf.write(buffer, wav_array, sample_rate, format="WAV")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()


def base64_to_audio(b64: str) -> tuple[np.ndarray, int]:
    """Convert base64 WAV to audio array and sample rate."""
    audio_bytes = base64.b64decode(b64)
    buffer = BytesIO(audio_bytes)
    audio, sr = sf.read(buffer)
    return audio, sr


def load_models():
    """Load models lazily on first request or via preload endpoint."""
    global voice_design_model, base_model, models_loaded

    with idle_lock:
        if models_loaded:
            return

    try:
        device = get_device()
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        logger.info(f"Loading models on {device} with {dtype}...")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("-" * 60)
        logger.info("Loading VoiceDesign model...")
        start = time.time()
        voice_design_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            device_map=device,
            dtype=dtype,
        )
        logger.info(f"VoiceDesign model loaded in {time.time() - start:.1f}s")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("-" * 60)
        logger.info("Loading Base model...")
        start = time.time()
        base_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map=device,
            dtype=dtype,
        )
        logger.info(f"Base model loaded in {time.time() - start:.1f}s")

        with idle_lock:
            models_loaded = True
            update_last_access_time()

        logger.info("-" * 60)
        logger.info("All models loaded successfully!")

    except Exception as e:
        logger.error(f"Failed to load models: {e}", exc_info=True)
        raise


@app.on_event("startup")
async def startup_event():
    """Lazy load models - don't load until first request."""
    logger.info("=" * 60)
    logger.info("Volsung - Voice Synthesis Server")
    logger.info("=" * 60)
    logger.info(f"Model cache: {models_dir}")
    logger.info("Models will load on first request to save GPU memory")
    logger.info("POST /preload to load models manually")
    logger.info("")
    logger.info("Endpoints:")
    logger.info("  GET  /health        - Health check")
    logger.info("  GET  /doc           - API documentation")
    logger.info("  POST /preload       - Load models now")
    logger.info("  POST /voice/design     - Generate voice from description")
    logger.info("  POST /voice/synthesize - Synthesize with cloned voice")
    logger.info("  POST /music/generate   - Generate music from description")
    logger.info("  POST /sfx/generate     - Generate sound effects")
    logger.info("  POST /sfx/layer        - Generate layered SFX")
    logger.info("  GET  /music/info       - Music module info")
    logger.info("  GET  /sfx/health       - SFX module health")
    logger.info("=" * 60)
    start_idle_monitor()


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check server health status including module status."""
    return HealthResponse(
        status="healthy",
        voice_design_model=voice_design_model is not None,
        base_model=base_model is not None,
        music_model=music_model is not None,
        sfx_model=sfx_model is not None,
        tts_router=True,
        music_router=True,
        sfx_router=True,
    )


@app.get("/doc")
async def documentation() -> Dict[str, Any]:
    """Get full API documentation with examples."""
    return {
        "name": "Volsung",
        "version": "1.0.0",
        "description": "Voice synthesis server for Qwen3-TTS",
        "endpoints": {
            "GET /health": {
                "description": "Check server status and model load state",
                "output": {
                    "status": "healthy",
                    "voice_design_model": True,
                    "base_model": True,
                    "music_model": True,
                    "sfx_model": True,
                },
            },
            "POST /preload": {
                "description": "Preload specific models into GPU memory. Models parameter accepts: ['qwen3', 'styletts2', 'music', 'sfx'] or 'all'",
                "input": {
                    "models": "Array of model names or 'all' (default: ['qwen3']). Options: 'qwen3', 'styletts2', 'music', 'sfx'"
                },
                "behavior": {
                    "default": "Unloads existing models, loads requested models",
                    "smart_loading": "Skips models already in memory (no reload unless needed)",
                    "gpu_conservation": "Only requested models are loaded to minimize GPU memory usage",
                },
                "output": {
                    "status": "ok",
                    "loaded": ["List of newly loaded models"],
                    "unloaded": ["List of models unloaded to make room"],
                    "already_loaded": ["List of models already in memory"],
                },
                "examples": {
                    "preload_qwen3_only": {"models": ["qwen3"]},
                    "preload_styletts2": {"models": ["styletts2"]},
                    "preload_all_models": {"models": "all"},
                    "preload_multiple_specific": {"models": ["qwen3", "music"]},
                },
                "workflow_note": "Preload models before first request to avoid wait time. For example, call POST /preload with {'models': ['qwen3']} at server startup or when you know which models you'll need.",
            },
            "POST /music/generate": {
                "description": "Generate music from text description (up to 30 seconds)",
                "input": {
                    "description": "Natural language music description (e.g., 'Upbeat acoustic guitar for audiobook background')",
                    "duration": "Duration in seconds (default: 30.0, max: 30.0)",
                    "genre": "Optional: Genre tag (e.g., 'acoustic', 'electronic', 'classical')",
                    "mood": "Optional: Mood tag (e.g., 'upbeat', 'calm', 'tense')",
                    "tempo": "Optional: Tempo hint - 'slow', 'medium', or 'fast'",
                },
                "output": {
                    "audio": "Base64-encoded WAV audio",
                    "sample_rate": 24000,
                    "metadata": {
                        "duration": "Audio duration in seconds",
                        "sample_rate": 24000,
                        "genre_tags": ["acoustic", "guitar", "upbeat"],
                        "generation_time_ms": "Time taken to generate",
                        "model_used": "Model identifier",
                    },
                },
                "example": {
                    "description": "Peaceful acoustic guitar for meditation audiobook background",
                    "duration": 15.0,
                    "genre": "acoustic",
                    "mood": "calm",
                    "tempo": "slow",
                },
            },
            "POST /sfx/generate": {
                "description": "Generate sound effects from text description (up to 10 seconds)",
                "input": {
                    "description": "Natural language SFX description (e.g., 'footsteps on gravel')",
                    "duration": "Duration in seconds (default: 5.0, max: 10.0)",
                    "category": "Optional: Category tag (e.g., 'nature', 'mechanical', 'urban', 'fantasy')",
                },
                "output": {
                    "audio": "Base64-encoded WAV audio",
                    "sample_rate": 24000,
                    "metadata": {
                        "duration": "Audio duration in seconds",
                        "sample_rate": 24000,
                        "category": "SFX category",
                        "generation_time_ms": "Time taken to generate",
                        "model_used": "Model identifier",
                    },
                },
                "example": {
                    "description": "Footsteps on gravel path",
                    "duration": 3.0,
                    "category": "nature",
                },
            },
            "POST /sfx/layer": {
                "description": "Generate combined/layered sound effects",
                "input": {
                    "layers": "Array of SFX generation requests to combine",
                },
                "output": {
                    "audio": "Base64-encoded combined WAV audio",
                    "sample_rate": 24000,
                    "layers_metadata": "Metadata for each layer",
                    "total_duration": "Duration of combined audio",
                },
                "example": {
                    "layers": [
                        {
                            "description": "Thunder rumbling",
                            "duration": 5.0,
                            "category": "nature",
                        },
                        {
                            "description": "Rain falling",
                            "duration": 5.0,
                            "category": "nature",
                        },
                    ],
                },
            },
            "POST /voice/design": {
                "description": "Generate voice sample from natural language description (Qwen3-TTS or StyleTTS 2)",
                "input": {
                    "text": "Sample text to speak (e.g., 'Hello, I am John. Nice to meet you.')",
                    "language": "Language: English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian, or Auto",
                    "instruct": "Natural language voice description (e.g., 'A warm, elderly man with a Southern accent')",
                    "backend": "Optional: TTS backend to use - 'qwen3' (default) or 'styletts2'",
                    "styletts_params": "Optional (StyleTTS 2 only): embedding_scale (1.0-10.0, emotion intensity), alpha (0.0-1.0, voice similarity), beta (0.0-1.0, emotion similarity), diffusion_steps (3-20, style diversity)",
                },
                "output": {"audio": "Base64-encoded WAV audio", "sample_rate": 24000},
                "example": {
                    "text": "Hello, I am John. Nice to meet you.",
                    "language": "English",
                    "instruct": "A warm, elderly man's voice with a slight Southern accent and gravelly tone",
                },
                "example_styletts": {
                    "text": "This is an exciting announcement!",
                    "language": "English",
                    "instruct": "A passionate speaker with high energy",
                    "backend": "styletts2",
                    "styletts_params": {
                        "embedding_scale": 2.0,
                        "alpha": 0.3,
                        "beta": 0.7,
                        "diffusion_steps": 5,
                    },
                },
                "note": "StyleTTS 2 uses text-driven emotion control - describe emotions in the instruct field and tune with embedding_scale",
            },
            "POST /voice/styletts/design": {
                "description": "Generate voice sample using StyleTTS 2 (dedicated endpoint)",
                "input": {
                    "text": "Sample text to speak",
                    "language": "Language code (English, Chinese, etc.)",
                    "instruct": "Voice and emotion description",
                    "styletts_params": "Optional: embedding_scale (1.0-10.0, emotion intensity), alpha (0.0-1.0, voice similarity), beta (0.0-1.0, emotion similarity), diffusion_steps (3-20, style diversity)",
                },
                "output": {"audio": "Base64-encoded WAV audio", "sample_rate": 24000},
                "example": {
                    "text": "Welcome to the future of speech synthesis!",
                    "language": "English",
                    "instruct": "An enthusiastic presenter with dynamic energy",
                    "styletts_params": {
                        "embedding_scale": 2.5,
                        "alpha": 0.3,
                        "beta": 0.7,
                        "diffusion_steps": 5,
                    },
                },
            },
            "POST /voice/synthesize": {
                "description": "Synthesize text using cloned voice from reference audio",
                "input": {
                    "ref_audio": "Base64-encoded WAV (from /voice/design output)",
                    "ref_text": "Transcript of the reference audio",
                    "text": "New text to synthesize in the cloned voice",
                    "language": "Language code (default: English)",
                },
                "output": {"audio": "Base64-encoded WAV audio", "sample_rate": 24000},
                "workflow": "1. Call /voice/design to get audio sample\n2. Store the audio and the text you sent\n3. Call /voice/synthesize with that audio + transcript + new text",
            },
            "GET /music/info": {
                "description": "Get music module information and status",
                "output": {
                    "status": "ready|unloaded|not_initialized",
                    "model_id": "Music model identifier",
                    "model_name": "Music model name",
                    "is_loaded": "Whether model is loaded",
                    "device": "Device used (cuda, cpu, etc.)",
                },
            },
            "GET /sfx/health": {
                "description": "Check SFX module health status",
                "output": {
                    "status": "healthy|unloaded|uninitialized",
                    "model_loaded": "Whether SFX model is loaded",
                    "model_name": "SFX model name",
                    "idle_seconds": "Time since last use",
                },
            },
        },
        "workflows": {
            "voice_cloning": {
                "name": "Voice Cloning (Qwen3-TTS or StyleTTS 2)",
                "steps": [
                    "1. Choose backend: Qwen3-TTS (text-to-voice) or StyleTTS 2 (emotion-rich)",
                    "2. POST /voice/design with text, instruct, and optional backend/styletts_params",
                    "3. Save the returned audio as reference",
                    "4. POST /voice/synthesize with ref_audio, ref_text, and new text",
                    "5. Result: Audio in the cloned voice",
                ],
                "note": "StyleTTS 2 adds emotion control via embedding_scale parameter (1.0-10.0). Describe emotions in the instruct field.",
            },
            "audiobook_production": {
                "name": "Audiobook Production",
                "steps": [
                    "1. Generate TTS audio using /voice/design + /voice/synthesize",
                    "2. Generate background music using /music/generate",
                    "3. Generate sound effects using /sfx/generate or /sfx/layer",
                    "4. Result: Mixed audiobook chapter",
                ],
            },
        },
        "models": {
            "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            "base_clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "styletts2": "yl4579/StyleTTS2-LibriTTS",
            "music": "facebook/musicgen-small",
            "sfx": "audiogen-medium",
        },
        "routers": {
            "/voice": "TTS module - voice design and synthesis",
            "/music": "Music generation module",
            "/sfx": "Sound effects module",
        },
    }


@app.post("/preload", response_model=PreloadResponse)
async def preload(req: PreloadRequest):
    """Download and cache models.

    Args:
        req: PreloadRequest containing models to preload.
            Can be a single model name or list of names:
            - "qwen3": Qwen3-TTS voice models
            - "styletts2": StyleTTS 2 voice cloning
            - "music": MusicGen music generation
            - "sfx": AudioLDM sound effects
            - "all": All models (default)

    Returns:
        PreloadResponse with status indicating which models were loaded/unloaded.

    Examples:
        {"models": ["qwen3"]} - Load only Qwen3-TTS
        {"models": ["styletts2", "music"]} - Load StyleTTS 2 and MusicGen
        {"models": "all"} - Load all models
    """
    from volsung.models.preload_manager import get_preload_manager

    try:
        manager = get_preload_manager()

        # Handle single string or list
        if isinstance(req.models, str):
            requested = [req.models]
        else:
            requested = req.models

        result = manager.preload(requested)

        update_last_access_time()
        return PreloadResponse(
            status=result["status"],
            models=result["models"],
            loaded=result.get("loaded", []),
            unloaded=result.get("unloaded", []),
        )
    except ValueError as e:
        logger.error(f"Invalid preload request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Preload failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to preload models: {str(e)}"
        )


# ============================================================================
# Music Generation Endpoints
# ============================================================================


@app.post("/music/generate", response_model=MusicGenerateResponse)
async def music_generate(req: MusicGenerateRequest):
    """
    Generate music from text description.

    Creates background music up to 30 seconds from natural language description.
    Useful for audiobook background music, ambience, etc.
    """
    # Validate duration
    if req.duration > 30.0:
        raise HTTPException(status_code=400, detail="Duration must be <= 30 seconds")
    if req.duration <= 0:
        raise HTTPException(status_code=400, detail="Duration must be positive")

    # Lazy load models on first request
    if not models_loaded:
        load_models()

    update_last_access_time()

    if music_model is None:
        raise HTTPException(status_code=503, detail="Music model not loaded")

    try:
        logger.info(
            f"[MUSIC_GENERATE] Request: description='{req.description[:50]}...' duration={req.duration}s"
        )
        start_time = time.time()

        wavs, sr = music_model.generate_music(
            description=req.description,
            duration=req.duration,
            genre=req.genre,
            mood=req.mood,
            tempo=req.tempo,
        )

        elapsed_ms = (time.time() - start_time) * 1000
        duration_seconds = len(wavs[0]) / sr
        logger.info(
            f"[MUSIC_GENERATE] Generated: duration={duration_seconds:.2f}s sample_rate={sr} elapsed={elapsed_ms:.0f}ms"
        )

        audio_base64 = audio_to_base64(wavs[0], sr)
        audio_size_kb = len(audio_base64) * 3 // 4 // 1024
        logger.info(f"[MUSIC_GENERATE] Response: audio_size={audio_size_kb}KB")

        # Build genre tags from request
        genre_tags = []
        if req.genre:
            genre_tags.append(req.genre)
        if req.mood:
            genre_tags.append(req.mood)
        if req.tempo:
            genre_tags.append(req.tempo)

        metadata = MusicMetadata(
            duration=duration_seconds,
            sample_rate=sr,
            genre_tags=genre_tags,
            generation_time_ms=elapsed_ms,
            model_used="musicgen-large",  # Placeholder - actual model TBD
        )

        return MusicGenerateResponse(
            audio=audio_base64, sample_rate=sr, metadata=metadata
        )

    except Exception as e:
        logger.error(f"[MUSIC_GENERATE] Failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Music generation failed: {str(e)}"
        )


# ============================================================================
# SFX Generation Endpoints
# ============================================================================


@app.post("/sfx/generate", response_model=SFXGenerateResponse)
async def sfx_generate(req: SFXGenerateRequest):
    """
    Generate sound effects from text description.

    Creates sound effects up to 10 seconds from natural language description.
    Useful for audiobook sound effects, ambient sounds, etc.
    """
    # Validate duration
    if req.duration > 10.0:
        raise HTTPException(status_code=400, detail="Duration must be <= 10 seconds")
    if req.duration <= 0:
        raise HTTPException(status_code=400, detail="Duration must be positive")

    # Lazy load models on first request
    if not models_loaded:
        load_models()

    update_last_access_time()

    if sfx_model is None:
        raise HTTPException(status_code=503, detail="SFX model not loaded")

    try:
        logger.info(
            f"[SFX_GENERATE] Request: description='{req.description[:50]}...' duration={req.duration}s category={req.category}"
        )
        start_time = time.time()

        wavs, sr = sfx_model.generate_sfx(
            description=req.description,
            duration=req.duration,
            category=req.category,
        )

        elapsed_ms = (time.time() - start_time) * 1000
        duration_seconds = len(wavs[0]) / sr
        logger.info(
            f"[SFX_GENERATE] Generated: duration={duration_seconds:.2f}s sample_rate={sr} elapsed={elapsed_ms:.0f}ms"
        )

        audio_base64 = audio_to_base64(wavs[0], sr)
        audio_size_kb = len(audio_base64) * 3 // 4 // 1024
        logger.info(f"[SFX_GENERATE] Response: audio_size={audio_size_kb}KB")

        metadata = SFXMetadata(
            duration=duration_seconds,
            sample_rate=sr,
            category=req.category,
            generation_time_ms=elapsed_ms,
            model_used="audiogen-medium",  # Placeholder - actual model TBD
        )

        return SFXGenerateResponse(
            audio=audio_base64, sample_rate=sr, metadata=metadata
        )

    except Exception as e:
        logger.error(f"[SFX_GENERATE] Failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"SFX generation failed: {str(e)}")


@app.post("/sfx/layer", response_model=SFXLayerResponse)
async def sfx_layer(req: SFXLayerRequest):
    """
    Generate combined/layered sound effects.

    Creates multiple sound effects and layers them together.
    Useful for complex audio scenes.
    """
    # Validate layers
    if not req.layers:
        raise HTTPException(status_code=400, detail="At least one layer required")
    if len(req.layers) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 layers allowed")

    for i, layer in enumerate(req.layers):
        if layer.duration > 10.0:
            raise HTTPException(
                status_code=400, detail=f"Layer {i + 1}: Duration must be <= 10 seconds"
            )
        if layer.duration <= 0:
            raise HTTPException(
                status_code=400, detail=f"Layer {i + 1}: Duration must be positive"
            )

    # Lazy load models on first request
    if not models_loaded:
        load_models()

    update_last_access_time()

    if sfx_model is None:
        raise HTTPException(status_code=503, detail="SFX model not loaded")

    try:
        logger.info(f"[SFX_LAYER] Request: {len(req.layers)} layers")
        start_time = time.time()

        # Generate each layer
        generated_layers = []
        layers_metadata = []
        max_duration = 0.0

        for i, layer_req in enumerate(req.layers):
            layer_start = time.time()

            wavs, sr = sfx_model.generate_sfx(
                description=layer_req.description,
                duration=layer_req.duration,
                category=layer_req.category,
            )

            layer_duration = len(wavs[0]) / sr
            layer_elapsed_ms = (time.time() - layer_start) * 1000

            generated_layers.append((wavs[0], sr))
            layers_metadata.append(
                SFXMetadata(
                    duration=layer_duration,
                    sample_rate=sr,
                    category=layer_req.category,
                    generation_time_ms=layer_elapsed_ms,
                    model_used="audiogen-medium",
                )
            )
            max_duration = max(max_duration, layer_duration)

            logger.info(
                f"[SFX_LAYER] Layer {i + 1}: {layer_duration:.2f}s '{layer_req.description[:30]}...'"
            )

        # Combine layers (simple mix for now - can be enhanced)
        combined_audio = combine_audio_layers(generated_layers)

        elapsed_ms = (time.time() - start_time) * 1000
        audio_base64 = audio_to_base64(combined_audio, sr)
        audio_size_kb = len(audio_base64) * 3 // 4 // 1024

        logger.info(
            f"[SFX_LAYER] Combined: duration={max_duration:.2f}s elapsed={elapsed_ms:.0f}ms size={audio_size_kb}KB"
        )

        return SFXLayerResponse(
            audio=audio_base64,
            sample_rate=sr,
            layers_metadata=layers_metadata,
            total_duration=max_duration,
        )

    except Exception as e:
        logger.error(f"[SFX_LAYER] Failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"SFX layering failed: {str(e)}")


def combine_audio_layers(layers: list[tuple]) -> np.ndarray:
    """
    Combine multiple audio layers into a single audio track.

    Args:
        layers: List of (audio_array, sample_rate) tuples

    Returns:
        Combined audio array
    """
    if not layers:
        return np.array([])

    # Find max length
    max_length = max(len(layer[0]) for layer in layers)
    sr = layers[0][1]

    # Initialize output array
    combined = np.zeros(max_length)

    # Mix all layers together
    for audio, layer_sr in layers:
        # Pad if shorter
        if len(audio) < max_length:
            audio = np.pad(audio, (0, max_length - len(audio)))
        combined += audio

    # Normalize to prevent clipping
    max_val = np.max(np.abs(combined))
    if max_val > 1.0:
        combined = combined / max_val

    return combined.astype(np.float32)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
    )
