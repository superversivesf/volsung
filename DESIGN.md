# Music & SFX Module Integration Guide

## Volsung Audio Generation Platform

This document provides comprehensive technical guidance for integrating MusicGen and AudioLDM modules into the Volsung TTS server.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture Overview](#architecture-overview)
4. [Installation & Setup](#installation--setup)
5. [Usage Examples](#usage-examples)
6. [Implementation Roadmap](#implementation-roadmap)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### Purpose

The Music and SFX modules extend Volsung from a pure Text-to-Speech (TTS) server into a comprehensive audio generation platform. These modules provide:

- **Music Module**: Generate background music and ambient soundscapes using MusicGen
- **SFX Module**: Generate sound effects and audio textures using AudioLDM

### Integration Philosophy

The modules follow Volsung's existing architectural patterns:

- **Lazy Loading**: Models load on first request, minimizing startup time and memory usage
- **Idle Timeout**: Models automatically unload after period of inactivity (configurable)
- **FastAPI Integration**: RESTful endpoints with Pydantic validation
- **Base64 Encoding**: Audio data transmitted as base64-encoded WAV files
- **Resource Management**: Careful GPU memory management with cache clearing

### Module Capabilities

| Module | Model | Use Cases | Output |
|--------|-------|-----------|--------|
| TTS (Existing) | Qwen3-TTS | Voice synthesis, voice cloning | 24kHz WAV |
| Music | MusicGen | Background music, ambience | 32kHz WAV |
| SFX | AudioLDM | Sound effects, textures | 16kHz WAV |

---

## Prerequisites

### Hardware Requirements

#### Minimum Requirements

```
GPU: NVIDIA GPU with 8GB+ VRAM
RAM: 16GB system RAM
Storage: 50GB free space for models
```

#### Recommended Specifications

| Configuration | VRAM | Use Case |
|--------------|------|----------|
| Single Module | 12GB | Run one module at a time |
| TTS + Music | 16GB | Full audiobook production |
| TTS + SFX | 14GB | Interactive applications |
| All Modules | 24GB | Production server |

#### VRAM Breakdown by Model

```
Qwen3-TTS (VoiceDesign):     ~6GB
Qwen3-TTS (Base):            ~6GB
MusicGen (Small):            ~4GB
MusicGen (Medium):           ~8GB
AudioLDM 2:                  ~5GB
──────────────────────────────────
Total (all models):          ~29GB
```

**Note**: With lazy loading and idle timeouts, you can run on less VRAM by loading only what you need.

### Software Dependencies

#### Python Version

```bash
# Required
Python >= 3.10

# Recommended
Python 3.11 or 3.12
```

#### Core Dependencies

Add to `pyproject.toml`:

```toml
dependencies = [
    # Existing dependencies
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.23.0",
    "torch>=2.0.0",
    "transformers>=4.52.0",
    "qwen-tts",
    "soundfile>=0.12.0",
    "pydantic>=2.0.0",
    "numpy>=1.24.0",
    # New dependencies for Music & SFX
    "audiocraft>=1.3.0",        # MusicGen
    "diffusers>=0.29.0",        # AudioLDM
    "accelerate>=0.30.0",       # Model loading optimization
]
```

#### CUDA Requirements

```bash
# Check CUDA version
nvidia-smi

# Required: CUDA >= 11.8
# PyTorch with CUDA support installed automatically via requirements
```

### Model Access

#### HuggingFace Authentication

Both MusicGen and AudioLDM require HuggingFace access:

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Login (required for some models)
huggingface-cli login
```

#### Model Downloads

Models download automatically on first use, but can be pre-downloaded:

```bash
# MusicGen
python -c "from audiocraft.models import musicgen; musicgen.MusicGen.get_pretrained('facebook/musicgen-small')"

# AudioLDM
python -c "from diffusers import AudioLDM2Pipeline; AudioLDM2Pipeline.from_pretrained('cvssp/audioldm2')"
```

---

## Architecture Overview

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   TTS Module │  │ Music Module │  │   SFX Module │    │
│  │   (Existing) │  │   (New)      │  │   (New)      │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                 │              │
│         │    ┌────────────┴─────────────────┘              │
│         │    │                                            │
│         │    ▼                                            │
│         │  ┌─────────────────────┐                        │
│         │  │   Model Manager   │                        │
│         │  │  (Lazy Loading)   │                        │
│         │  └─────────────────────┘                        │
│         │                │                                │
│         ▼                ▼                ▼              │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐     │
│  │ Qwen3-TTS    │ │   MusicGen   │ │   AudioLDM   │     │
│  │ Models       │ │   Model      │ │   Model      │     │
│  └──────────────┘ └──────────────┘ └──────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   GPU Memory     │
                    │   (Shared)       │
                    └──────────────────┘
```

### Module Communication

#### Shared Resources

All modules share:
- **Device Management**: Automatic CUDA/CPU selection
- **Memory Pool**: GPU cache cleared between model switches
- **Idle Monitor**: Single background thread tracks all module activity
- **Logger**: Unified logging with module prefixes

#### Isolation Strategy

Each module maintains:
- **Independent Model Instances**: No shared weights between modules
- **Separate Pipelines**: Different generation strategies
- **Module-Specific Config**: Individual timeout and batch settings

### Resource Sharing Strategy

#### Memory Management

```python
# Global model references (one per module)
music_model: Optional[MusicGen] = None
sfx_model: Optional[AudioLDM2Pipeline] = None

# Shared idle monitoring
last_access_time: float = 0.0
idle_lock: threading.Lock()
IDLE_TIMEOUT_SECONDS = 300  # 5 minutes
```

#### Model Lifecycle

```
Request Received ──► Check if model loaded ──► [No] ──► Load Model
        │                                          │
        │                                         [Yes]
        │                                          │
        └──────────────────────────────────────────┘
                          │
                          ▼
                Generate Audio
                          │
                          ▼
              Update last_access_time
                          │
                          ▼
         Idle Monitor checks: if timeout exceeded
                          │
                          ▼
              Unload model, clear GPU cache
```

---

## Installation & Setup

### Step 1: Update Dependencies

```bash
# Add to requirements.txt
echo "audiocraft>=1.3.0" >> requirements.txt
echo "diffusers>=0.29.0" >> requirements.txt
echo "accelerate>=0.30.0" >> requirements.txt

# Install
pip install -r requirements.txt
```

### Step 2: Create Module Structure

```bash
mkdir -p volsung/modules
touch volsung/modules/__init__.py
touch volsung/modules/music.py
touch volsung/modules/sfx.py
```

### Step 3: Configure Environment

Create `.env` file:

```bash
# Model Selection
MUSIC_MODEL_SIZE=small  # small, medium, large, melody
SFX_MODEL_NAME=cvssp/audioldm2  # or audioldm2-l, audioldm2-m

# Resource Management
MUSIC_IDLE_TIMEOUT=300
SFX_IDLE_TIMEOUT=300
MAX_CONCURRENT_GENERATIONS=1

# Generation Defaults
MUSIC_DURATION_SECONDS=10
SFX_DURATION_SECONDS=5
MUSIC_TOP_K=250
SFX_GUIDANCE_SCALE=3.5

# Device Override (optional)
# DEVICE=cuda:0
# DEVICE=cpu
```

### Step 4: Verify Installation

```bash
# Test imports
python -c "from audiocraft.models import musicgen; print('MusicGen OK')"
python -c "from diffusers import AudioLDM2Pipeline; print('AudioLDM OK')"

# Test CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Usage Examples

### Example 1: Generate Background Music

```python
import requests
import base64

# Request background music for a scene
response = requests.post(
    "http://localhost:8000/generate_music",
    json={
        "description": "Epic orchestral fantasy music, building tension, suitable for a climactic battle scene",
        "duration": 15,
        "top_k": 250,
        "top_p": 0.0,
        "temperature": 1.0
    }
)

result = response.json()
audio_data = base64.b64decode(result["audio"])

# Save to file
with open("battle_music.wav", "wb") as f:
    f.write(audio_data)
```

### Example 2: Generate Sound Effects

```python
import requests
import base64

# Generate sword clash sound
response = requests.post(
    "http://localhost:8000/generate_sfx",
    json={
        "description": "Metallic sword clash, sharp impact, resonant",
        "duration": 3,
        "num_inference_steps": 50,
        "guidance_scale": 3.5
    }
)

result = response.json()
audio_data = base64.b64decode(result["audio"])

# Save to file
with open("sword_clash.wav", "wb") as f:
    f.write(audio_data)
```

### Example 3: Combine with TTS

```python
import requests
import base64

base_url = "http://localhost:8000"

# Step 1: Generate character voice
voice_response = requests.post(
    f"{base_url}/voice_design",
    json={
        "text": "I am the guardian of this realm.",
        "language": "English",
        "instruct": "A deep, resonant voice of an ancient guardian, wise and powerful"
    }
)
voice_data = voice_response.json()

# Step 2: Generate ambient background
music_response = requests.post(
    f"{base_url}/generate_music",
    json={
        "description": "Mysterious ambient dungeon atmosphere, dripping water, distant echoes",
        "duration": 20
    }
)
music_data = music_response.json()

# Step 3: Generate door creaking sound
sfx_response = requests.post(
    f"{base_url}/generate_sfx",
    json={
        "description": "Heavy wooden door creaking open slowly, rusty hinges",
        "duration": 4
    }
)
sfx_data = sfx_response.json()

# Step 4: Synthesize dialogue
speech_response = requests.post(
    f"{base_url}/synthesize",
    json={
        "ref_audio": voice_data["audio"],
        "ref_text": "I am the guardian of this realm.",
        "text": "Who dares enter the sacred chamber?",
        "language": "English"
    }
)
speech_data = speech_response.json()

# Results: voice, background music, SFX, and synthesized speech
# Combine in post-processing with your audio tools
```

### Example 4: Preload All Models

```bash
# Load all models at startup (useful for production)
curl -X POST http://localhost:8000/preload

# Or preload specific modules
curl -X POST http://localhost:8000/preload?modules=music,sfx
```

---

## Implementation Roadmap

### Phase 1: Music Module

**Week 1-2: Core Implementation**

- [ ] Create `volsung/modules/music.py`
- [ ] Implement MusicGen wrapper class
- [ ] Add lazy loading support
- [ ] Create `/generate_music` endpoint
- [ ] Add request/response Pydantic models
- [ ] Implement idle timeout integration
- [ ] Write unit tests

**Week 3: Integration & Testing**

- [ ] Integrate with main server
- [ ] Add model preloading support
- [ ] Test memory management
- [ ] Performance benchmarking
- [ ] Documentation updates

**Deliverable**: Working music generation endpoint

### Phase 2: SFX Module

**Week 4-5: Core Implementation**

- [ ] Create `volsung/modules/sfx.py`
- [ ] Implement AudioLDM wrapper class
- [ ] Add lazy loading support
- [ ] Create `/generate_sfx` endpoint
- [ ] Add request/response Pydantic models
- [ ] Implement idle timeout integration
- [ ] Write unit tests

**Week 6: Integration & Testing**

- [ ] Integrate with main server
- [ ] Test concurrent module usage
- [ ] Memory pressure testing
- [ ] API documentation

**Deliverable**: Working SFX generation endpoint

### Phase 3: Stitcher Integration

**Week 7-8: Audio Stitching**

- [ ] Design audio stitching API
- [ ] Implement timing/positioning logic
- [ ] Create `/stitch_audio` endpoint
- [ ] Support layering (background + voice + SFX)
- [ ] Add volume control per layer
- [ ] Write integration tests

**Week 9: Advanced Features**

- [ ] Silence detection and trimming
- [ ] Crossfade between segments
- [ ] Batch processing support
- [ ] Export to multiple formats

**Deliverable**: Complete audio composition pipeline

### Phase 4: Optimization

**Week 10: Performance**

- [ ] Implement model quantization
- [ ] Add streaming generation support
- [ ] Optimize GPU memory usage
- [ ] Add request queuing for concurrent requests

**Deliverable**: Production-ready implementation

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MUSIC_MODEL_SIZE` | `small` | MusicGen model size: small, medium, large, melody |
| `SFX_MODEL_NAME` | `cvssp/audioldm2` | AudioLDM model name |
| `MUSIC_IDLE_TIMEOUT` | `300` | Seconds before unloading music model |
| `SFX_IDLE_TIMEOUT` | `300` | Seconds before unloading SFX model |
| `MAX_CONCURRENT_GENERATIONS` | `1` | Max parallel generation requests |
| `MUSIC_DURATION_SECONDS` | `10` | Default music generation duration |
| `SFX_DURATION_SECONDS` | `5` | Default SFX generation duration |
| `MUSIC_TOP_K` | `250` | Top-k sampling for music |
| `SFX_GUIDANCE_SCALE` | `3.5` | Guidance scale for SFX |
| `DEVICE` | `auto` | Override device (cuda:0, cpu) |

### Configuration File

Create `config.yaml`:

```yaml
modules:
  music:
    enabled: true
    model: facebook/musicgen-small
    idle_timeout: 300
    default_duration: 10
    generation:
      top_k: 250
      top_p: 0.0
      temperature: 1.0
      
  sfx:
    enabled: true
    model: cvssp/audioldm2
    idle_timeout: 300
    default_duration: 5
    generation:
      num_inference_steps: 50
      guidance_scale: 3.5
      
  tts:
    enabled: true
    idle_timeout: 300

resource_management:
  max_concurrent_requests: 1
  clear_cache_between_requests: true
  log_memory_usage: true
```

### Runtime Configuration

Modify behavior at runtime via API:

```bash
# Update idle timeout
curl -X POST http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"music_idle_timeout": 600}'

# Get current config
curl http://localhost:8000/config
```

---

## Troubleshooting

### Common Issues

#### Issue: CUDA Out of Memory

**Symptoms:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size** (if implemented):
   ```python
   # In config
   MAX_CONCURRENT_GENERATIONS = 1
   ```

2. **Use smaller models**:
   ```bash
   MUSIC_MODEL_SIZE=small
   ```

3. **Enable aggressive caching**:
   ```python
   torch.cuda.empty_cache()  # Called after each generation
   ```

4. **Force CPU fallback**:
   ```bash
   export DEVICE=cpu
   ```

#### Issue: Model Download Fails

**Symptoms:**
```
OSError: Model not found
```

**Solutions:**

1. **Login to HuggingFace**:
   ```bash
   huggingface-cli login
   ```

2. **Verify model access**:
   ```bash
   python -c "from huggingface_hub import model_info; model_info('facebook/musicgen-small')"
   ```

3. **Check internet connection**:
   ```bash
   curl -I https://huggingface.co
   ```

#### Issue: Slow First Request

**Symptoms:**
First request takes 30-60 seconds

**Solutions:**

1. **Preload at startup**:
   ```bash
   # Add to server startup
   curl -X POST http://localhost:8000/preload
   ```

2. **Increase idle timeout**:
   ```bash
   MUSIC_IDLE_TIMEOUT=1800  # 30 minutes
   ```

3. **Keep models loaded** (development only):
   ```bash
   MUSIC_IDLE_TIMEOUT=0  # Never unload
   ```

#### Issue: Audio Quality Poor

**Symptoms:**
Noisy, distorted, or incorrect audio

**Solutions:**

1. **For MusicGen - adjust sampling**:
   ```json
   {
     "top_k": 250,
     "temperature": 0.8,
     "description": "More specific prompt with genre and mood"
   }
   ```

2. **For AudioLDM - adjust guidance**:
   ```json
   {
     "guidance_scale": 5.0,
     "num_inference_steps": 100
   }
   ```

3. **Check input audio** (for cloning):
   - Sample rate should match model (24kHz for TTS)
   - Audio should be clear with minimal background noise

### Memory Optimization Tips

#### 1. Sequential Module Usage

Don't run all modules simultaneously:

```python
# Good: Sequential usage
for scene in scenes:
    music = generate_music(scene.music_prompt)
    save(music)
    unload_music_model()
    
    speech = synthesize(scene.dialogue)
    save(speech)
    unload_tts_model()
```

#### 2. Monitor GPU Memory

```python
# Add to your code
import torch

def log_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
```

#### 3. Use Model Quantization

```python
# Load models with quantization
model = MusicGen.get_pretrained(
    "facebook/musicgen-small",
    device='cuda',
    dtype=torch.float16  # Use FP16 instead of FP32
)
```

#### 4. Batch Processing

Process multiple prompts in batches:

```python
# Instead of multiple API calls
response = requests.post(
    "http://localhost:8000/generate_music_batch",
    json={
        "descriptions": [
            "Happy music",
            "Sad music",
            "Action music"
        ],
        "duration": 10
    }
)
```

### Debugging Checklist

- [ ] Check GPU availability: `nvidia-smi`
- [ ] Verify CUDA version: `python -c "import torch; print(torch.version.cuda)"`
- [ ] Test model loading: `python -c "from audiocraft.models import musicgen"`
- [ ] Check disk space: `df -h`
- [ ] Review server logs for errors
- [ ] Verify environment variables are set
- [ ] Test with minimal configuration first

### Support Resources

- **MusicGen Documentation**: https://github.com/facebookresearch/audiocraft
- **AudioLDM Documentation**: https://github.com/haoheliu/AudioLDM
- **HuggingFace Audio Models**: https://huggingface.co/models?pipeline_tag=text-to-audio
- **Volsung Issues**: https://github.com/anomaly/volsung/issues

---

## Appendix

### API Reference Summary

#### Music Generation

```http
POST /generate_music
Content-Type: application/json

{
  "description": "string",      // Required: Text description of music
  "duration": 10,               // Optional: Seconds (1-30)
  "top_k": 250,                 // Optional: Sampling parameter
  "top_p": 0.0,                 // Optional: Nucleus sampling
  "temperature": 1.0          // Optional: Randomness
}
```

#### SFX Generation

```http
POST /generate_sfx
Content-Type: application/json

{
  "description": "string",      // Required: Text description of sound
  "duration": 5,                // Optional: Seconds (1-10)
  "num_inference_steps": 50,   // Optional: Quality vs speed
  "guidance_scale": 3.5        // Optional: Prompt adherence
}
```

#### Model Preload

```http
POST /preload
Content-Type: application/json

{
  "modules": ["music", "sfx"]  // Optional: Specific modules
}
```

### File Structure

```
volsung/
├── __init__.py
├── __main__.py
├── server.py              # Main FastAPI application
└── modules/
    ├── __init__.py
    ├── music.py            # MusicGen wrapper
    └── sfx.py              # AudioLDM wrapper
```

### Version Compatibility

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10 | 3.11 |
| PyTorch | 2.0.0 | 2.2.0 |
| Transformers | 4.52.0 | 4.52.0+ |
| audiocraft | 1.3.0 | 1.3.0+ |
| diffusers | 0.29.0 | 0.29.0+ |
| CUDA | 11.8 | 12.1 |

---

*Document Version: 1.0*
*Last Updated: 2026-03-31*
*For: Volsung Audio Generation Platform*
