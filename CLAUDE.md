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
- **sfx** (8004): Sound effects via AudioLDM2
- **music** (8005): Music generation via Mustango (Apache 2.0)

**Smart model loading**: The coordinator tracks which service has its model loaded and unloads before switching, so only one model occupies GPU VRAM at a time. Services expose `/load` and `/unload` endpoints for this.

**Route mapping** (in `server.py`):
- `/voice/design` -> qwen-voice:8001
- `/voice/synthesize` -> qwen-base:8002
- `/voice/styletts` -> styletts:8003
- `/sfx/generate` -> sfx:8004
- `/music/generate` -> music:8005

**Key env vars for coordinator**: `QWEN_VOICE_SERVICE_URL`, `QWEN_BASE_SERVICE_URL`, `STYLETTS_SERVICE_URL`, `SFX_SERVICE_URL`, `MUSIC_SERVICE_URL`, `COORDINATOR_HOST`, `COORDINATOR_PORT`.

## Build & Run Commands

```bash
# Docker (recommended) - starts all 6 services
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

## Code Patterns

- **Service files** (`volsung/services/*_service.py`): Each is a complete FastAPI app that can run standalone. Not imported by the coordinator.
- **Model managers** (`volsung/{tts,music,sfx}/manager.py`): Subclass `ModelManagerBase` from `volsung/models/base.py`. Handle lazy loading, idle timeout, thread-safe via RLock.
- **Generators** (`volsung/{tts,music,sfx}/generators/`): Subclass `GeneratorBase`. Wrap specific ML models with a uniform interface.
- **Schemas** (`volsung/{tts,music,sfx}/schemas.py`): Pydantic request/response models.
- **Configuration** (`volsung/config.py`): `VolsungConfig` with env var override (`VOLSUNG_<SECTION>__<KEY>`), YAML file support, and defaults.

## Important Notes

- First startup downloads models (~15 min). Subsequent starts use cached models.
- Audio output is always base64-encoded WAV. Sample rates vary by service (TTS: 24000, SFX: 16000, Music: 32000).
- Python 3.10+ required. CUDA 11.8+ recommended for GPU.
- `diffusers` is pinned to 0.24.0 for torch 2.1.0 compatibility in the SFX service.
- MusicGen was replaced with Mustango for Apache 2.0 licensing.
- `server.py` line 77 has a known bug: `SFX_SERVICE_URL` defaults to port 8005 instead of 8004.
