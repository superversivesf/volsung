"""
FastAPI server for Qwen3-TTS with voice design and cloning endpoints.

Volsung - Voice synthesis server for Qwen3-TTS.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import torch
import numpy as np
import soundfile as sf
from io import BytesIO
import base64
import logging
import time

from qwen_tts import Qwen3TTSModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Volsung",
    description="Voice synthesis server for Qwen3-TTS",
    version="1.0.0",
)

voice_design_model = None
base_model = None
models_loaded = False


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


class PreloadResponse(BaseModel):
    """Preload response."""

    status: str


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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

        models_loaded = True
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
    logger.info("Models will load on first request to save GPU memory")
    logger.info("POST /preload to load models manually")
    logger.info("")
    logger.info("Endpoints:")
    logger.info("  GET  /health       - Health check")
    logger.info("  GET  /doc          - API documentation")
    logger.info("  POST /preload      - Load models now")
    logger.info("  POST /voice_design - Generate voice from description")
    logger.info("  POST /synthesize   - Clone voice from reference audio")
    logger.info("=" * 60)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check server health status."""
    return HealthResponse(
        status="healthy",
        voice_design_model=voice_design_model is not None,
        base_model=base_model is not None,
    )


@app.get("/doc")
async def documentation() -> Dict[str, Any]:
    """Get full API documentation with examples."""
    return {
        "name": "Volsung",
        "version": "1.0.0",
        "description": "Voice synthesis server for Qwen3-TTS",
        "endpoints": {
            "POST /voice_design": {
                "description": "Generate voice sample from natural language description",
                "input": {
                    "text": "Sample text to speak (e.g., 'Hello, I am John. Nice to meet you.')",
                    "language": "Language: English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian, or Auto",
                    "instruct": "Natural language voice description (e.g., 'A warm, elderly man with a Southern accent')",
                },
                "output": {"audio": "Base64-encoded WAV audio", "sample_rate": 24000},
                "example": {
                    "text": "Hello, I am John. Nice to meet you.",
                    "language": "English",
                    "instruct": "A warm, elderly man's voice with a slight Southern accent and gravelly tone",
                },
            },
            "POST /synthesize": {
                "description": "Synthesize text using cloned voice from reference audio",
                "input": {
                    "ref_audio": "Base64-encoded WAV (from /voice_design output)",
                    "ref_text": "Transcript of the reference audio",
                    "text": "New text to synthesize in the cloned voice",
                    "language": "Language code (default: English)",
                },
                "output": {"audio": "Base64-encoded WAV audio", "sample_rate": 24000},
                "workflow": "1. Call /voice_design to get audio sample\n2. Store the audio and the text you sent\n3. Call /synthesize with that audio + transcript + new text",
            },
            "GET /health": {
                "description": "Check server status and model load state",
                "output": {
                    "status": "healthy",
                    "voice_design_model": True,
                    "base_model": True,
                },
            },
            "POST /preload": {
                "description": "Download and load models (call at startup or before first use)",
                "output": {"status": "preloaded"},
            },
        },
        "workflow": {
            "voice_design_step": "Use /voice_design to create a character voice sample from a description",
            "storage": "Store the returned audio (base64 decode to WAV file) and the text you sent",
            "synthesis_step": "Use /synthesize with the stored audio + transcript + new dialogue to generate story audio",
            "example_flow": [
                "1. POST /voice_design with text='Hello, I am Alice.' and instruct='A cheerful young woman's voice'",
                "2. Save the resulting audio as alice_voice.wav",
                "3. POST /synthesize with ref_audio=alice_voice.wav (base64), ref_text='Hello, I am Alice.', text='The quick brown fox jumps over the lazy dog.'",
                "4. Result: Audio of 'The quick brown fox' in Alice's cloned voice",
            ],
        },
        "models": {
            "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            "base_clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        },
    }


@app.post("/preload", response_model=PreloadResponse)
async def preload():
    """Download and cache models."""
    try:
        load_models()
        return PreloadResponse(status="preloaded")
    except Exception as e:
        logger.error(f"Preload failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to preload models: {str(e)}"
        )


@app.post("/voice_design", response_model=VoiceDesignResponse)
async def voice_design(req: VoiceDesignRequest):
    """
    Generate voice sample from text description.

    Uses VoiceDesign model to create audio. The output should be stored
    (audio + transcript) for later use as reference in synthesis.
    """
    if voice_design_model is None:
        raise HTTPException(status_code=503, detail="VoiceDesign model not loaded")

    try:
        logger.info(
            f"[VOICE_DESIGN] Request: text='{req.text[:50]}...' language='{req.language}' instruct='{req.instruct[:50]}...'"
        )
        start_time = time.time()

        wavs, sr = voice_design_model.generate_voice_design(
            text=req.text, language=req.language, instruct=req.instruct
        )

        elapsed = time.time() - start_time
        logger.info(
            f"[VOICE_DESIGN] Generated audio: duration={len(wavs[0]) / sr:.2f}s sample_rate={sr} elapsed={elapsed:.2f}s"
        )

        audio_base64 = audio_to_base64(wavs[0], sr)
        audio_size_kb = len(audio_base64) * 3 // 4 // 1024
        logger.info(f"[VOICE_DESIGN] Response: audio_size={audio_size_kb}KB")

        return VoiceDesignResponse(audio=audio_base64, sample_rate=sr)

    except Exception as e:
        logger.error(f"[VOICE_DESIGN] Failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Voice design failed: {str(e)}")


@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(req: SynthesizeRequest):
    """
    Synthesize text using cloned voice from reference audio.

    Uses Base model's generate_voice_clone() directly with raw audio.
    The ref_audio should come from /voice_design output.
    The ref_text must be the transcript of that audio.
    """
    if base_model is None:
        raise HTTPException(status_code=503, detail="Base model not loaded")

    try:
        logger.info(
            f"[SYNTHESIZE] Request: text='{req.text[:50]}...' language='{req.language}'"
        )
        start_time = time.time()

        ref_audio, ref_sr = base64_to_audio(req.ref_audio)
        ref_duration = len(ref_audio) / ref_sr
        logger.info(
            f"[SYNTHESIZE] Reference audio: duration={ref_duration:.2f}s sample_rate={ref_sr}"
        )

        wavs, sr = base_model.generate_voice_clone(
            ref_audio=(ref_audio, ref_sr),
            ref_text=req.ref_text,
            text=req.text,
            language=req.language,
        )

        elapsed = time.time() - start_time
        output_duration = len(wavs[0]) / sr
        logger.info(
            f"[SYNTHESIZE] Generated audio: duration={output_duration:.2f}s sample_rate={sr} elapsed={elapsed:.2f}s"
        )

        audio_base64 = audio_to_base64(wavs[0], sr)
        audio_size_kb = len(audio_base64) * 3 // 4 // 1024
        logger.info(f"[SYNTHESIZE] Response: audio_size={audio_size_kb}KB")

        return SynthesizeResponse(audio=audio_base64, sample_rate=sr)

    except Exception as e:
        logger.error(f"[SYNTHESIZE] Failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
    )
