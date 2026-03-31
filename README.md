# Volsung

> Norse mythology: The Völsung saga tells of legendary heroes and their deeds. Just as the saga preserves stories through voice, Volsung gives voice to your text.

FastAPI server for Qwen3-TTS with voice design and voice cloning capabilities.

## Features

- **Voice Design**: Generate voice samples from natural language descriptions
- **Voice Cloning**: Synthesize text in a cloned voice from reference audio

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (recommended) or CPU
- ~8GB VRAM for 1.7B models
- **espeak-ng** (system package required for StyleTTS 2 phonemization)

### System Dependencies

StyleTTS 2 requires the `espeak-ng` system package for phonemization:

```bash
# Ubuntu/Debian
sudo apt-get install espeak-ng

# macOS
brew install espeak

# Fedora/RHEL
sudo dnf install espeak-ng

# Windows
# Download from https://github.com/espeak-ng/espeak-ng/releases
```

## Quick Start

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

### 2. Install Dependencies

```bash
# Basic installation (Qwen-TTS only)
pip install -r requirements.txt

# Or install with StyleTTS 2 support
pip install -r requirements.txt
pip install -e ".[styletts]"
```

**Note**: StyleTTS 2 requires `espeak-ng` to be installed on your system first. See [System Dependencies](#system-dependencies) above.

### 3. Run Server

```bash
# First time: models download automatically on startup
python -m volsung

# Or use uvicorn directly with custom port
uvicorn volsung.server:app --host 0.0.0.0 --port 8000
```

### 4. Preload Models (Optional)

Models load on first request, but you can preload:

```bash
curl -X POST http://localhost:8000/preload
```

## API Endpoints

### POST /voice_design

Generate a voice sample from a natural language description.

**Request:**
```json
{
  "text": "Hello, I am John. Nice to meet you.",
  "language": "English",
  "instruct": "A warm, elderly man's voice with a slight Southern accent and gravelly tone"
}
```

**Response:**
```json
{
  "audio": "base64-encoded-wav-data...",
  "sample_rate": 24000
}
```

### POST /synthesize

Synthesize text using a cloned voice from reference audio.

**Request:**
```json
{
  "ref_audio": "base64-encoded-audio-from-voice-design...",
  "ref_text": "Hello, I am John. Nice to meet you.",
  "text": "The quick brown fox jumps over the lazy dog.",
  "language": "English"
}
```

**Response:**
```json
{
  "audio": "base64-encoded-wav-data...",
  "sample_rate": 24000
}
```

### GET /doc

Get full API documentation with examples.

### GET /health

Check server status and model load state.

### POST /preload

Download and cache models before first use.

## Microservices Architecture

Volsung uses a microservices architecture with a lightweight coordinator/gateway that routes requests to specialized services:

```
┌─────────────────────────────────────────────────────────────┐
│                      Coordinator                             │
│                    (Port 8000)                               │
│  - Routes requests to appropriate services                   │
│  - Aggregates health status from all services                │
│  - Returns 503 if service unavailable                        │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
      ┌────────▼────────┐          ┌────────▼────────┐
      │   TTS Service   │          │  Music Service  │
      │   (Port 8001)   │          │   (Port 8002)   │
      │                 │          │                 │
      │ • /voice/design │          │ • /music/generate│
      │ • /voice/       │          │ • /music/*      │
      │   synthesize    │          │                 │
      └────────┬────────┘          └────────┬────────┘
               │                              │
      ┌────────▼────────┐                     │
      │   SFX Service   │                     │
      │   (Port 8003)   │◄────────────────────┘
      │                 │
      │ • /sfx/generate │
      │ • /sfx/layer    │
      │ • /sfx/*        │
      └─────────────────┘
```

### Services

| Service | Port | Description | Endpoints |
|---------|------|-------------|-----------|
| **Coordinator** | 8000 | API Gateway & routing | `/health`, `/doc`, `/preload`, `/voice/*`, `/music/*`, `/sfx/*` |
| **TTS** | 8001 | Text-to-Speech (Qwen3-TTS, StyleTTS 2) | `/voice/design`, `/voice/synthesize` |
| **Music** | 8002 | Music generation (MusicGen) | `/music/generate`, `/music/info` |
| **SFX** | 8003 | Sound effects (AudioLDM2) | `/sfx/generate`, `/sfx/layer` |

### Starting All Services

Use the convenience script to start all services:

```bash
# Start all services
./scripts/start-all.sh

# Start in daemon mode (background)
./scripts/start-all.sh --daemon

# Skip specific services
SKIP_SERVICES=music ./scripts/start-all.sh

# Custom log directory
./scripts/start-all.sh --log-dir /var/log/volsung
```

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f coordinator
docker-compose logs -f tts

# Stop all services
docker-compose down
```

### Health Aggregation

Check the health of all services:

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "coordinator": "healthy",
  "services": {
    "tts": {"healthy": true, "response_time_ms": 12.5},
    "music": {"healthy": true, "response_time_ms": 8.3},
    "sfx": {"healthy": true, "response_time_ms": 15.1}
  },
  "available": ["tts", "music", "sfx"],
  "unavailable": []
}
```

### Service Unavailability

If a service is unavailable, the coordinator returns HTTP 503:

```json
{
  "error": "Service unavailable",
  "service": "music",
  "message": "Could not connect to service",
  "suggestion": "Check that the music service is running"
}
```

## Workflow

1. Use `/voice_design` to create character voice samples
2. Store the audio and transcript
3. Use `/synthesize` with stored audio to generate dialogue in that voice

## Running in tmux/screen

```bash
# Create new tmux session
tmux new -s volsung

# Run server
python -m volsung

# Detach: Ctrl+B then D
# Reattach: tmux attach -t volsung
```

## Environment Variables

- `CUDA_VISIBLE_DEVICES`: Specify GPU (e.g., `CUDA_VISIBLE_DEVICES=0`)
- `DEVICE`: Override device (e.g., `DEVICE=cpu`)

## Hardware Requirements

| Model | Min VRAM | Recommended |
|-------|----------|-------------|
| 1.7B | 8GB | 12GB |
| 0.6B | 4GB | 8GB |

## Part of Skaldforge

Volsung is the TTS component of [Skaldforge](https://github.com/anomaly/skaldforge), a pipeline for converting prose manuscripts into audiobooks with character voice synthesis.

## License

This server uses Qwen-TTS models which have their own license terms.