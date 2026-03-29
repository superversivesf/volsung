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
pip install -r requirements.txt
```

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