"""IndexTTS-2 Service for Volsung.

A standalone FastAPI service that provides IndexTTS-2 voice generation endpoints.
Runs on port 8006 and handles voice synthesis with fine-grained emotion control
via emotion vectors, emotion text, or emotion reference audio.

Example:
    python -m volsung.services.indextts_service
    uvicorn volsung.services.indextts_service:app --host 0.0.0.0 --port 8006

Environment Variables:
    INDEXTTS_SERVICE_HOST: Server bind address (default: 0.0.0.0)
    INDEXTTS_SERVICE_PORT: Server port (default: 8006)
    INDEXTTS_DEVICE: Device override (default: auto-detected)
    INDEXTTS_MODEL_DIR: Model checkpoint directory (default: checkpoints)
"""

from __future__ import annotations

import base64
import gc
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from io import BytesIO
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, status
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

INDEXTTS2_REPO_ID = "IndexTeam/IndexTTS-2"


class IndexTTSConfig(BaseModel):
    host: str = Field(
        default_factory=lambda: os.getenv("INDEXTTS_SERVICE_HOST", "0.0.0.0")
    )
    port: int = Field(
        default_factory=lambda: int(os.getenv("INDEXTTS_SERVICE_PORT", "8006"))
    )
    device: Optional[str] = Field(
        default_factory=lambda: os.getenv("INDEXTTS_DEVICE")
    )
    model_dir: str = Field(
        default_factory=lambda: os.getenv("INDEXTTS_MODEL_DIR", "checkpoints")
    )


# =============================================================================
# Pydantic Schemas
# =============================================================================


class IndexTTSParams(BaseModel):
    emo_alpha: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Emotion intensity (0.0-1.0)",
    )
    emo_vector: Optional[List[float]] = Field(
        default=None,
        min_length=8,
        max_length=8,
        description="Emotion vector [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]",
    )
    emo_text: Optional[str] = Field(
        default=None,
        description="Text description of desired emotion",
    )
    use_emo_text: bool = Field(
        default=False,
        description="Derive emotion from text content or emo_text",
    )


class GenerateRequest(BaseModel):
    text: str = Field(
        ...,
        description="Text to synthesize",
        examples=["Hello, this is a test of IndexTTS-2."],
    )
    ref_audio: str = Field(
        ...,
        description="Base64-encoded reference audio (WAV) for voice cloning. Required.",
    )
    emo_audio: Optional[str] = Field(
        default=None,
        description="Base64-encoded emotion reference audio (WAV).",
    )
    indextts_params: Optional[IndexTTSParams] = Field(
        default=None,
        description="IndexTTS-2-specific parameters",
    )


class GenerateResponse(BaseModel):
    audio: str = Field(..., description="Base64-encoded WAV audio data")
    sample_rate: int = Field(default=32000, description="Audio sample rate in Hz")


class LoadRequest(BaseModel):
    device: Optional[str] = Field(
        default=None,
        description="Device to load on (cuda, cpu). Auto-detected if not specified.",
    )


class LoadResponse(BaseModel):
    status: str = Field(..., description="Status: loaded or already_loaded")
    model: str = Field(..., description="Model identifier")
    device: str = Field(..., description="Device model is loaded on")
    message: str = Field(..., description="Human-readable status message")


class UnloadResponse(BaseModel):
    status: str = Field(..., description="Status: unloaded or not_loaded")
    model: str = Field(..., description="Model identifier")
    message: str = Field(..., description="Human-readable status message")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall service status")
    model: dict = Field(default_factory=dict, description="Model status")
    weights: dict = Field(default_factory=dict, description="Weights cache status")


# =============================================================================
# Model Cache Utilities
# =============================================================================


def is_weights_cached(model_dir: str) -> bool:
    from pathlib import Path

    cfg = Path(model_dir) / "config.yaml"
    return cfg.exists()


def download_weights(model_dir: str) -> str:
    logger.info(f"Downloading IndexTTS-2 weights to {model_dir}...")
    try:
        path = snapshot_download(
            repo_id=INDEXTTS2_REPO_ID,
            local_dir=model_dir,
        )
        logger.info(f"IndexTTS-2 weights downloaded to: {path}")
        return path
    except Exception as e:
        logger.error(f"Failed to download IndexTTS-2 weights: {e}")
        raise


# =============================================================================
# Model Manager
# =============================================================================


class IndexTTS2Manager:
    def __init__(self, config: IndexTTSConfig):
        self.config = config
        self._generator = None
        self._device = self._get_device()
        self._is_loaded = False

    def _get_device(self) -> str:
        if self.config.device:
            return self.config.device
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load(self, device: Optional[str] = None) -> dict:
        if self._is_loaded:
            return {
                "status": "already_loaded",
                "device": self._device,
                "message": f"Model already loaded on {self._device}",
            }

        try:
            from indextts.infer_v2 import IndexTTS2
        except ImportError:
            raise ImportError(
                "indextts is required. Install IndexTTS-2 from source."
            )

        if device:
            self._device = device

        model_dir = self.config.model_dir
        cfg_path = os.path.join(model_dir, "config.yaml")

        if not os.path.exists(cfg_path):
            logger.info("Weights not found, downloading...")
            download_weights(model_dir)

        logger.info(f"Loading IndexTTS-2 from {model_dir} on {self._device}...")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._generator = IndexTTS2(
            cfg_path=cfg_path,
            model_dir=model_dir,
            use_fp16=(self._device != "cpu"),
        )

        self._is_loaded = True
        logger.info("IndexTTS-2 loaded successfully")

        return {
            "status": "loaded",
            "device": self._device,
            "message": f"Model loaded successfully on {self._device}",
        }

    def unload(self) -> dict:
        if not self._is_loaded:
            return {"status": "not_loaded", "message": "Model was not loaded"}

        logger.info("Unloading IndexTTS-2...")
        self._generator = None
        self._is_loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("IndexTTS-2 unloaded")
        return {"status": "unloaded", "message": "Model unloaded successfully"}

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def device(self) -> str:
        return self._device

    def generate(
        self,
        text: str,
        ref_audio_b64: str,
        emo_audio_b64: Optional[str] = None,
        emo_alpha: float = 1.0,
        emo_vector: Optional[List[float]] = None,
        emo_text: Optional[str] = None,
        use_emo_text: bool = False,
    ):
        """Generate speech with voice cloning and optional emotion control."""
        if not self._is_loaded:
            self.load()

        if self._generator is None:
            raise RuntimeError("IndexTTS-2 not loaded")

        logger.info(
            f"[INDEXTTS2] text='{text[:50]}...', "
            f"emo_alpha={emo_alpha}, has_emo_audio={emo_audio_b64 is not None}, "
            f"has_emo_vector={emo_vector is not None}"
        )

        tmp_files = []
        try:
            # Write reference audio to temp file
            ref_bytes = base64.b64decode(ref_audio_b64)
            ref_data, ref_sr = sf.read(BytesIO(ref_bytes))
            logger.info(f"[INDEXTTS2] Reference audio: {len(ref_data)} samples @ {ref_sr}Hz")
            ref_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(ref_tmp.name, ref_data, ref_sr, format="WAV")
            ref_tmp.close()
            tmp_files.append(ref_tmp.name)

            # Write emotion audio to temp file if provided
            emo_audio_path = None
            if emo_audio_b64:
                emo_bytes = base64.b64decode(emo_audio_b64)
                emo_data, emo_sr = sf.read(BytesIO(emo_bytes))
                emo_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(emo_tmp.name, emo_data, emo_sr, format="WAV")
                emo_tmp.close()
                tmp_files.append(emo_tmp.name)
                emo_audio_path = emo_tmp.name

            # Output temp file
            out_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            out_tmp.close()
            tmp_files.append(out_tmp.name)

            # Build kwargs for infer
            infer_kwargs = {
                "spk_audio_prompt": ref_tmp.name,
                "text": text,
                "output_path": out_tmp.name,
            }

            # Add emotion parameters if provided
            if emo_audio_path:
                infer_kwargs["emo_audio_prompt"] = emo_audio_path
            if emo_vector is not None:
                infer_kwargs["emo_vector"] = emo_vector
            if emo_text is not None:
                infer_kwargs["emo_text"] = emo_text
            if use_emo_text:
                infer_kwargs["use_emo_text"] = True
            if emo_alpha != 1.0 or emo_vector is not None or emo_audio_path or emo_text:
                infer_kwargs["emo_alpha"] = emo_alpha

            with torch.no_grad():
                self._generator.infer(**infer_kwargs)

            # Read back the output
            wav, sr = sf.read(out_tmp.name)
            if isinstance(wav, np.ndarray) and wav.ndim > 1:
                wav = wav.mean(axis=1)

            wav = wav.astype(np.float32)
            if wav.max() > 1.0 or wav.min() < -1.0:
                wav = wav / max(abs(wav.max()), abs(wav.min()))

            duration = len(wav) / sr
            logger.info(f"[INDEXTTS2] Generated: {duration:.2f}s @ {sr}Hz")

            return wav, sr

        except Exception as e:
            logger.error(f"[INDEXTTS2] Failed: {e}", exc_info=True)
            raise RuntimeError(f"IndexTTS-2 generation failed: {e}")
        finally:
            for f in tmp_files:
                try:
                    os.unlink(f)
                except OSError:
                    pass


# =============================================================================
# Global State
# =============================================================================

_config = IndexTTSConfig()
_manager: Optional[IndexTTS2Manager] = None


def get_manager() -> IndexTTS2Manager:
    global _manager
    if _manager is None:
        _manager = IndexTTS2Manager(_config)
    return _manager


# =============================================================================
# FastAPI Application
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("IndexTTS-2 Service starting up...")

    model_dir = _config.model_dir
    if not is_weights_cached(model_dir):
        logger.info("IndexTTS-2 weights not cached, downloading...")
        try:
            download_weights(model_dir)
            logger.info("IndexTTS-2 weights downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download IndexTTS-2 weights on startup: {e}")
    else:
        logger.info("IndexTTS-2 weights already cached")

    yield

    logger.info("IndexTTS-2 Service shutting down...")
    global _manager
    if _manager:
        _manager.unload()


app = FastAPI(
    title="Volsung IndexTTS-2 Service",
    description="IndexTTS-2 service with voice cloning and emotion control",
    version="0.1.0",
    lifespan=lifespan,
)


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check() -> dict:
    model_status = {
        "available": True,
        "loaded": False,
        "device": None,
    }

    try:
        from indextts.infer_v2 import IndexTTS2  # noqa: F401

        model_status["available"] = True
        if _manager is not None:
            model_status["loaded"] = _manager.is_loaded
            model_status["device"] = _manager.device if _manager.is_loaded else None
    except ImportError:
        model_status["available"] = False

    weights_cached = is_weights_cached(_config.model_dir)
    weights_status = {
        "cached": weights_cached,
        "model_id": INDEXTTS2_REPO_ID,
    }

    overall_status = "healthy" if weights_cached or model_status["available"] else "weights_not_cached"
    return {"status": overall_status, "model": model_status, "weights": weights_status}


@app.post("/load", response_model=LoadResponse)
async def load_model(request: LoadRequest = None) -> LoadResponse:
    manager = get_manager()
    device = request.device if request else None
    try:
        result = manager.load(device)
        return LoadResponse(
            status=result["status"],
            model="IndexTTS-2",
            device=result["device"],
            message=result["message"],
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to load model: {str(e)}",
        )


@app.post("/unload", response_model=UnloadResponse)
async def unload_model() -> UnloadResponse:
    manager = get_manager()
    result = manager.unload()
    return UnloadResponse(
        status=result["status"], model="IndexTTS-2", message=result["message"]
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    manager = get_manager()
    params = request.indextts_params or IndexTTSParams()
    try:
        wav, sr = manager.generate(
            text=request.text,
            ref_audio_b64=request.ref_audio,
            emo_audio_b64=request.emo_audio,
            emo_alpha=params.emo_alpha,
            emo_vector=params.emo_vector,
            emo_text=params.emo_text,
            use_emo_text=params.use_emo_text,
        )

        buffer = BytesIO()
        sf.write(buffer, wav, sr, format="WAV")
        buffer.seek(0)
        audio_b64 = base64.b64encode(buffer.read()).decode()

        return GenerateResponse(audio=audio_b64, sample_rate=sr)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {e}",
        )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    config = IndexTTSConfig()
    logger.info(f"Starting IndexTTS-2 Service on {config.host}:{config.port}")
    uvicorn.run(
        "volsung.services.indextts_service:app",
        host=config.host,
        port=config.port,
        log_level="info",
    )
