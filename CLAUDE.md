# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Volsung is the TTS/audio generation component of [Skaldforge](https://github.com/anomaly/skaldforge), an audiobook generation pipeline. It provides voice design, voice cloning, music generation, and sound effects via a microservices architecture.

## Architecture

**Coordinator pattern**: A lightweight FastAPI gateway (`volsung/server.py`, port 8000) routes requests to isolated ML services. The coordinator has no ML dependencies (only `fastapi`, `httpx`, `pydantic`, `numpy`).

**Services** (each is a standalone FastAPI app with its own Dockerfile and requirements):
- **qwen-voice** (8001): Voice design via Qwen3-TTS VoiceDesign model
- **qwen-base** (8002): Voice synthesis/cloning via Qwen3-TTS base model
- **styletts** (8003): TTS via StyleTTS2 (requires `espeak-ng` system package)
- **indextts** (8006): TTS with emotion control via IndexTTS-2 (Bilibili). Supports emotion vectors, emotion text, and emotion reference audio
- **chatterbox** (8007): TTS with voice cloning via Chatterbox (Resemble AI, MIT). Emotion via `exaggeration` (0-1) and `cfg_weight` parameters
- **sfx** (8004): Sound effects via AudioLDM2
- **music** (8005): Music generation via Mustango (Apache 2.0)

**Smart model loading**: The coordinator tracks which service has its model loaded and unloads before switching, so only one model occupies GPU VRAM at a time. Services expose `/load` and `/unload` endpoints for this.

**Route mapping** (in `server.py`):
- `/voice/design` -> qwen-voice:8001
- `/voice/synthesize` -> qwen-base:8002
- `/voice/styletts` -> styletts:8003
- `/voice/indextts` -> indextts:8006
- `/voice/chatterbox` -> chatterbox:8007
- `/sfx/generate` -> sfx:8004
- `/music/generate` -> music:8005

All voice/TTS services expose `/generate` internally. The coordinator translates public paths (e.g. `/voice/chatterbox`) to `/generate` on the target service.

**Key env vars for coordinator**: `QWEN_VOICE_SERVICE_URL`, `QWEN_BASE_SERVICE_URL`, `STYLETTS_SERVICE_URL`, `INDEXTTS_SERVICE_URL`, `CHATTERBOX_SERVICE_URL`, `SFX_SERVICE_URL`, `MUSIC_SERVICE_URL`, `COORDINATOR_HOST`, `COORDINATOR_PORT`.

## Build & Run Commands

```bash
# Docker (recommended) - starts all 8 services
docker-compose up -d

# Local coordinator only
pip install -r requirements.txt
python -m volsung

# With StyleTTS2 support
pip install -e ".[styletts]"

# Start all services locally
./scripts/start-all.sh
./scripts/start-all.sh --daemon          # background mode
SKIP_SERVICES=music ./scripts/start-all.sh  # skip specific services
```

## Testing

```bash
pytest                          # all tests
pytest tests/test_music.py      # single module
pytest tests/test_music.py::TestMusicManager::test_generate  # single test
pytest -vv                      # verbose
pytest --cov=volsung            # with coverage
```

Tests use mocked torch/cuda environments (no GPU required). Shared fixtures in `tests/conftest.py` provide sample audio arrays, base64 encoded audio, and mock models.

## Dependencies

Two separate dependency graphs:
- `requirements.txt`: Coordinator only (lightweight, no ML libs)
- `requirements/services/*.txt`: Per-service ML dependencies (torch, transformers, etc.)
- `pyproject.toml`: Full package deps including all ML libs; optional extra `[styletts]`

Note: IndexTTS-2 is not pip-installable; its Dockerfile clones from source and installs via `pip install -e .`. Chatterbox installs via `pip install chatterbox-tts`.

## Code Patterns

- **Service files** (`volsung/services/*_service.py`): Each is a complete FastAPI app that can run standalone. Not imported by the coordinator. All expose `/health`, `/load`, `/unload`, `/generate` endpoints.
- **Model managers** (within each service file or `volsung/{tts,music,sfx}/manager.py`): Handle lazy loading, device detection, and resource cleanup. Thread-safe via RLock where applicable.
- **Generators** (`volsung/{tts,music,sfx}/generators/`): Subclass `GeneratorBase`. Wrap specific ML models with a uniform interface.
- **Schemas** (Pydantic request/response models in each service or `volsung/{tts,music,sfx}/schemas.py`).
- **Configuration** (`volsung/config.py`): `VolsungConfig` with env var override (`VOLSUNG_<SECTION>__<KEY>`), YAML file support, and defaults.

## Important Notes

- First startup downloads models (~15 min). Subsequent starts use cached models.
- Audio output is always base64-encoded WAV. Sample rates vary by service (TTS: 24000, SFX: 16000, Music: 32000, IndexTTS-2: 32000).
- Python 3.10+ required. CUDA 11.8+ recommended for GPU.
- `diffusers` is pinned to 0.24.0 for torch 2.1.0 compatibility in the SFX service.
- MusicGen was replaced with Mustango for Apache 2.0 licensing.
- `server.py` has a known bug: `SFX_SERVICE_URL` defaults to port 8005 instead of 8004.
- StyleTTS2 requires a `torch.load` monkey-patch for PyTorch 2.6+ compatibility and NLTK `punkt_tab` data.
