# Volsung Architecture: Music & SFX Integration

## Executive Summary

This document outlines the architecture for integrating MusicGen (music generation) and AudioLDM (SFX generation) into volsung alongside the existing Qwen3-TTS voice synthesis system. The architecture follows clean separation of concerns, modular design, and extensibility principles.

## Design Goals

1. **Separation of Concerns**: Music, SFX, and TTS modules are independent
2. **Unified API**: Single FastAPI server with clear endpoint separation
3. **Resource Management**: Independent lazy loading with idle timeout per module
4. **Extensibility**: Easy to add new model types (voice, music, SFX)
5. **Audio Composition**: Support for future "stitcher" that combines TTS + Music + SFX

---

## Module Structure

```
volsung/
├── __init__.py              # Package exports
├── __main__.py              # Entry point
├── server.py                # FastAPI app (minimal, delegates to modules)
├── config.py                # Configuration management
├── models/                  # Abstract base classes
│   ├── __init__.py
│   ├── base.py              # ModelManagerBase abstract class
│   ├── registry.py          # Model registry for discovery
│   └── types.py             # Shared types (AudioResult, etc.)
├── tts/                     # TTS Module (refactored existing code)
│   ├── __init__.py
│   ├── manager.py           # TTSModelManager
│   ├── endpoints.py         # TTS routes
│   └── schemas.py           # Pydantic models for TTS
├── music/                   # Music Module (NEW)
│   ├── __init__.py
│   ├── manager.py           # MusicModelManager
│   ├── endpoints.py         # Music routes
│   ├── schemas.py           # Pydantic models for music
│   └── generators/          # Specific generators
│       ├── __init__.py
│       └── musicgen.py      # MusicGen integration
├── sfx/                     # SFX Module (NEW)
│   ├── __init__.py
│   ├── manager.py           # SFXModelManager
│   ├── endpoints.py         # SFX routes
│   ├── schemas.py           # Pydantic models for SFX
│   └── generators/          # Specific generators
│       ├── __init__.py
│       └── audioldm.py      # AudioLDM integration
├── audio/                   # Audio utilities
│   ├── __init__.py
│   ├── utils.py             # Audio conversion utilities
│   └── effects.py           # Audio effects (fade, normalize, etc.)
└── stitcher/                # Future: Audio composition (NEW)
    ├── __init__.py
    ├── composer.py          # Timeline-based composition
    ├── schemas.py           # Track, Timeline, etc.
    └── endpoints.py         # Composition routes
```

---

## Component Boundaries

### Module Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Server                          │
│                    (volsung/server.py)                         │
└──────────────────┬──────────────────────────────────────────────┘
                   │
        ┌──────────┼──────────┬──────────┐
        │          │          │          │
        ▼          ▼          ▼          ▼
┌────────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
│   TTS      │ │ Music  │ │  SFX   │ │ Stitcher │
│  Module    │ │ Module │ │ Module │ │ (future) │
└─────┬──────┘ └───┬────┘ └───┬────┘ └────┬─────┘
      │            │          │           │
      ▼            ▼          ▼           ▼
┌────────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
│  Qwen3-    │ │MusicGen│ │AudioLDM│ │Audio     │
│    TTS     │ │ (Meta) │ │(ICML)  │ │Composer  │
└────────────┘ └────────┘ └────────┘ └──────────┘
```

### Module Responsibilities

| Module | Responsibility | Key Classes |
|--------|---------------|-------------|
| `models/` | Abstract base classes, shared types | `ModelManagerBase`, `ModelRegistry`, `AudioResult` |
| `tts/` | Voice synthesis (existing) | `TTSModelManager`, `TTSEndpoints` |
| `music/` | Music generation (NEW) | `MusicModelManager`, `MusicEndpoints` |
| `sfx/` | Sound effects generation (NEW) | `SFXModelManager`, `SFXEndpoints` |
| `audio/` | Audio utilities | `audio_to_base64`, `base64_to_audio`, `normalize` |
| `stitcher/` | Audio composition (future) | `AudioComposer`, `Timeline`, `Track` |

---

## Class Hierarchy

### Abstract Base Classes (models/base.py)

```python
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import threading
import time
from pydantic import BaseModel


class AudioResult(BaseModel):
    """Standardized audio result format."""
    audio: str  # base64-encoded
    sample_rate: int
    duration: float
    format: str = "wav"
    metadata: Dict[str, Any] = {}


class ModelConfig(BaseModel):
    """Configuration for a model."""
    model_id: str
    model_name: str
    device: str = "auto"
    dtype: str = "auto"
    max_memory_gb: Optional[float] = None
    idle_timeout_seconds: int = 300  # 5 minutes default


class ModelManagerBase(ABC):
    """
    Abstract base class for all model managers.
    
    Provides:
    - Lazy loading
    - Idle timeout monitoring
    - Thread-safe access
    - Resource cleanup
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._model = None
        self._loaded = False
        self._last_access = 0.0
        self._lock = threading.RLock()
        self._idle_timer = None
        self._start_idle_monitor()
    
    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._loaded
    
    def _ensure_loaded(self):
        """Load model if not already loaded."""
        with self._lock:
            if not self._loaded:
                self._load_model()
                self._loaded = True
            self._last_access = time.time()
    
    def unload_if_idle(self):
        """Called by idle monitor to unload if idle too long."""
        with self._lock:
            if not self._loaded:
                return
            idle_duration = time.time() - self._last_access
            if idle_duration >= self.config.idle_timeout_seconds:
                self._unload_model()
                self._loaded = False
    
    @abstractmethod
    def _load_model(self):
        """Implement: Load the actual model."""
        pass
    
    @abstractmethod
    def _unload_model(self):
        """Implement: Unload the model and free resources."""
        pass
    
    @abstractmethod
    def generate(self, *args, **kwargs) -> AudioResult:
        """Implement: Generate audio."""
        pass


class GeneratorBase(ABC):
    """
    Abstract base for specific model generators.
    
    Each generator wraps a specific model (e.g., MusicGen, AudioLDM).
    """
    
    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique identifier for this generator."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable name."""
        pass
    
    @property
    @abstractmethod
    def required_vram_gb(self) -> float:
        """Estimated VRAM requirement."""
        pass
    
    @abstractmethod
    def load(self, device: str, dtype: str):
        """Load the model."""
        pass
    
    @abstractmethod
    def unload(self):
        """Unload and cleanup."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, duration: float, **kwargs) -> tuple:
        """
        Generate audio.
        
        Returns: (audio_array, sample_rate)
        """
        pass
```

### Concrete Implementations

```python
# tts/manager.py
class TTSModelManager(ModelManagerBase):
    """Manages Qwen3-TTS models (VoiceDesign + Base)."""
    
    def __init__(self):
        super().__init__(ModelConfig(
            model_id="qwen3-tts",
            model_name="Qwen3-TTS",
            idle_timeout_seconds=300
        ))
        self.voice_design_model = None
        self.base_model = None
    
    def _load_model(self):
        from qwen_tts import Qwen3TTSModel
        device = get_device()
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        self.voice_design_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            device_map=device,
            dtype=dtype
        )
        self.base_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map=device,
            dtype=dtype
        )
    
    def _unload_model(self):
        self.voice_design_model = None
        self.base_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# music/generators/musicgen.py
class MusicGenGenerator(GeneratorBase):
    """MusicGen model from Meta."""
    
    @property
    def model_id(self) -> str:
        return "musicgen-small"
    
    @property
    def model_name(self) -> str:
        return "MusicGen-Small"
    
    @property
    def required_vram_gb(self) -> float:
        return 6.0  # 6-8GB
    
    def load(self, device: str, dtype: str):
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        
        self.processor = AutoProcessor.from_pretrained(
            "facebook/musicgen-small"
        )
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-small"
        )
        self.model.to(device)
    
    def unload(self):
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def generate(self, prompt: str, duration: float, **kwargs) -> tuple:
        """Generate music from text prompt."""
        # Implementation details
        pass


# music/manager.py
class MusicModelManager(ModelManagerBase):
    """Manages music generation models."""
    
    def __init__(self, default_generator: str = "musicgen-small"):
        super().__init__(ModelConfig(
            model_id="music",
            model_name="Music Generation",
            idle_timeout_seconds=300
        ))
        self.default_generator = default_generator
        self._generators: Dict[str, GeneratorBase] = {}
        self._current_generator: Optional[GeneratorBase] = None
    
    def register_generator(self, generator: GeneratorBase):
        """Register a new music generator."""
        self._generators[generator.model_id] = generator
    
    def _load_model(self):
        """Load default generator."""
        if self.default_generator not in self._generators:
            # Auto-register MusicGen
            from .generators.musicgen import MusicGenGenerator
            self.register_generator(MusicGenGenerator())
        
        self._current_generator = self._generators[self.default_generator]
        device = get_device()
        dtype = "float16" if torch.cuda.is_available() else "float32"
        self._current_generator.load(device, dtype)
    
    def _unload_model(self):
        if self._current_generator:
            self._current_generator.unload()
            self._current_generator = None
    
    def generate(self, prompt: str, duration: float = 30.0) -> AudioResult:
        """Generate music from text description."""
        self._ensure_loaded()
        
        audio, sr = self._current_generator.generate(prompt, duration)
        
        return AudioResult(
            audio=audio_to_base64(audio, sr),
            sample_rate=sr,
            duration=len(audio) / sr,
            metadata={
                "generator": self._current_generator.model_id,
                "prompt": prompt,
                "duration": duration
            }
        )


# sfx/generators/audioldm.py
class AudioLDMGenerator(GeneratorBase):
    """AudioLDM for SFX generation."""
    
    @property
    def model_id(self) -> str:
        return "audioldm-m-full"
    
    @property
    def model_name(self) -> str:
        return "AudioLDM-M-Full"
    
    @property
    def required_vram_gb(self) -> float:
        return 8.0  # 8-12GB
    
    def load(self, device: str, dtype: str):
        from diffusers import AudioLDMPipeline
        
        self.model = AudioLDMPipeline.from_pretrained(
            "cvssp/audioldm-m-full",
            torch_dtype=torch.float16 if dtype == "float16" else torch.float32
        )
        self.model.to(device)
    
    def unload(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def generate(self, prompt: str, duration: float, **kwargs) -> tuple:
        """Generate SFX from text description."""
        # Implementation
        pass
```

---

## Resource Management Strategy

### Memory Requirements

| Model | VRAM Required | CPU RAM | Max Duration | Notes |
|-------|---------------|---------|--------------|-------|
| Qwen3-TTS 1.7B | 8GB | 4GB | N/A (streaming) | VoiceDesign + Base |
| MusicGen-Small | 6-8GB | 4GB | 30s | Meta, Transformers |
| AudioLDM-M-Full | 8-12GB | 6GB | 10s | ICML 2023, Diffusers |

### Loading Strategy

```
┌────────────────────────────────────────────────────────────────┐
│                    Resource Manager                            │
└────────────────────────────────────────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  TTS Manager    │   │ Music Manager   │   │  SFX Manager    │
│  ├─ VoiceDesign │   │  ├─ MusicGen    │   │  ├─ AudioLDM    │
│  └─ Base        │   │  └─ (extensible)│   │  └─ (extensible)│
│                 │   │                 │   │                 │
│  Status: IDLE   │   │  Status: IDLE   │   │  Status: IDLE   │
│  ├─ Loaded: No  │   │  ├─ Loaded: No  │   │  ├─ Loaded: No  │
│  ├─ Idle: N/A   │   │  ├─ Idle: N/A   │   │  ├─ Idle: N/A   │
│  └─ Timer: Off  │   │  └─ Timer: Off  │   │  └─ Timer: Off  │
└─────────────────┘   └─────────────────┘   └─────────────────┘
```

### Concurrent Loading Considerations

```python
# config.py - Resource allocation rules
RESOURCE_RULES = {
    # Can TTS and Music run simultaneously?
    "concurrent_tts_music": False,  # No: 8GB + 6GB = 14GB > typical GPU
    "concurrent_tts_sfx": False,    # No: 8GB + 8GB = 16GB > typical GPU  
    "concurrent_music_sfx": False,  # No: 6GB + 8GB = 14GB > typical GPU
    
    # Priority order when multiple requests arrive
    "priority_order": ["tts", "sfx", "music"],
    
    # Queue strategy: "fifo" or "priority"
    "queue_strategy": "priority",
    
    # Timeout for queued requests
    "max_queue_wait_seconds": 60,
}


class ResourceManager:
    """
    Coordinates resource usage across model managers.
    
    Prevents OOM by ensuring only compatible models are loaded
    simultaneously.
    """
    
    def __init__(self):
        self._managers: Dict[str, ModelManagerBase] = {}
        self._active_models: Set[str] = set()
        self._request_queue = []
        self._lock = threading.Lock()
    
    def register(self, name: str, manager: ModelManagerBase):
        self._managers[name] = manager
    
    def request_access(self, model_type: str) -> bool:
        """
        Request access to a model type.
        
        Returns True if granted, False if queued.
        """
        with self._lock:
            if model_type in self._active_models:
                return True  # Already active
            
            # Check if we need to unload others
            if self._would_cause_oom(model_type):
                self._unload_incompatible(model_type)
            
            self._active_models.add(model_type)
            return True
    
    def _would_cause_oom(self, new_model: str) -> bool:
        """Check if adding this model would cause OOM."""
        total_vram = 0
        for model in self._active_models:
            total_vram += self._get_vram_requirement(model)
        total_vram += self._get_vram_requirement(new_model)
        return total_vram > self._get_available_vram()
    
    def release_access(self, model_type: str):
        """Release access, allowing others to be loaded."""
        with self._lock:
            self._active_models.discard(model_type)
```

---

## Data Flow: Generation to Stitcher

### Audio Generation Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Request    │────▶│   Module     │────▶│   Manager    │
│   (prompt)   │     │   Router     │     │   (lazy load)│
└──────────────┘     └──────────────┘     └──────────────┘
                                                    │
                                                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Response   │◀────│   AudioResult│◀────│   Generator  │
│   (base64)   │     │   (metadata) │     │   (inference)│
└──────────────┘     └──────────────┘     └──────────────┘
```

### Stitcher Preparation Flow (Future)

```
┌─────────────────────────────────────────────────────────────────┐
│                         Stitcher Pipeline                        │
└─────────────────────────────────────────────────────────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
            ▼                     ▼                     ▼
     ┌────────────┐        ┌────────────┐        ┌────────────┐
     │ TTS Track  │        │ Music Track│        │  SFX Track │
     │ (dialogue) │        │ (bg music) │        │ (effects)  │
     └─────┬──────┘        └─────┬──────┘        └─────┬──────┘
           │                     │                     │
           └─────────────────────┼─────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────┐
                    │     Timeline       │
                    │ (temporal layout)  │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │   AudioComposer    │
                    │ (mix + render)     │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │  Final Audiobook   │
                    │  (mixed output)    │
                    └────────────────────┘
```

### AudioResult Design

```python
# models/types.py

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from enum import Enum


class AudioType(str, Enum):
    """Type of audio content."""
    TTS = "tts"           # Speech/dialogue
    MUSIC = "music"       # Background music
    SFX = "sfx"           # Sound effects


class AudioResult(BaseModel):
    """
    Standardized audio result from any generator.
    
    Used for TTS, Music, and SFX generation.
    """
    # Core audio data
    audio: str = Field(..., description="Base64-encoded WAV audio")
    sample_rate: int = Field(..., description="Sample rate in Hz (e.g., 24000)")
    duration: float = Field(..., description="Duration in seconds")
    
    # Metadata
    audio_type: AudioType = Field(..., description="Type of audio content")
    generator: str = Field(..., description="Generator/model used")
    prompt: str = Field(..., description="Input prompt/description")
    
    # Optional fields
    format: str = Field(default="wav", description="Audio format")
    channels: int = Field(default=1, description="Number of channels (1=mono, 2=stereo)")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")
    
    # For future stitcher
    track_id: Optional[str] = Field(default=None, description="Unique track identifier")
    start_time: Optional[float] = Field(default=None, description="Start time in composition (seconds)")
    end_time: Optional[float] = Field(default=None, description="End time in composition (seconds)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "audio": "base64encoded...",
                "sample_rate": 24000,
                "duration": 30.0,
                "audio_type": "music",
                "generator": "musicgen-small",
                "prompt": "A calm, mysterious orchestral piece with strings",
                "format": "wav",
                "channels": 1,
                "metadata": {
                    "generation_time": 5.2,
                    "model_version": "1.0"
                }
            }
        }


# Future: Stitcher schemas
class Track(BaseModel):
    """A single track in the composition."""
    track_id: str
    audio_type: AudioType
    audio_result: AudioResult
    start_time: float
    end_time: Optional[float] = None
    fade_in: float = 0.0
    fade_out: float = 0.0
    volume: float = 1.0


class Timeline(BaseModel):
    """Timeline containing multiple tracks."""
    timeline_id: str
    tracks: List[Track]
    total_duration: float
    sample_rate: int = 24000


class CompositionRequest(BaseModel):
    """Request to compose multiple audio tracks."""
    timeline: Timeline
    output_format: str = "wav"
    normalize: bool = True


class CompositionResult(BaseModel):
    """Result of audio composition."""
    audio: str  # base64
    sample_rate: int
    duration: float
    tracks_mixed: int
    peak_db: float
```

---

## API Endpoint Design

### Current Endpoints (Existing)

```
POST /voice_design   → TTS Module (VoiceDesign)
POST /synthesize     → TTS Module (Base)
POST /preload        → All Modules
GET  /health         → All Modules status
GET  /doc            → API documentation
```

### New Endpoints

```
# Music Module
POST /music/generate
  Body: { "prompt": "calm orchestral music", "duration": 30, "model": "musicgen-small" }
  Response: AudioResult

GET /music/models
  Response: { "models": [{ "id": "musicgen-small", "name": "MusicGen-Small", "max_duration": 30 }] }

POST /music/preload
  Response: { "status": "loaded" }


# SFX Module  
POST /sfx/generate
  Body: { "prompt": "thunder rumbling", "duration": 5, "model": "audioldm-m-full" }
  Response: AudioResult

GET /sfx/models
  Response: { "models": [{ "id": "audioldm-m-full", "name": "AudioLDM-M-Full", "max_duration": 10 }] }

POST /sfx/preload
  Response: { "status": "loaded" }


# Future: Stitcher Module
POST /stitcher/compose
  Body: { "timeline": Timeline, "output_format": "wav" }
  Response: CompositionResult

POST /stitcher/preview
  Body: { "timeline": Timeline, "start": 0, "duration": 30 }
  Response: AudioResult (preview segment)
```

### Updated Health Endpoint

```python
# Returns status for all modules

class ModuleStatus(BaseModel):
    loaded: bool
    idle_seconds: Optional[float]
    last_access: Optional[str]  # ISO timestamp


class HealthResponse(BaseModel):
    status: str
    modules: Dict[str, ModuleStatus]
    resources: Dict[str, Any]


# Example response:
{
    "status": "healthy",
    "modules": {
        "tts": {
            "loaded": True,
            "idle_seconds": 45.2,
            "last_access": "2024-01-15T10:30:00Z"
        },
        "music": {
            "loaded": False,
            "idle_seconds": null,
            "last_access": null
        },
        "sfx": {
            "loaded": False,
            "idle_seconds": null,
            "last_access": null
        }
    },
    "resources": {
        "gpu_available": True,
        "gpu_memory_used_gb": 8.5,
        "gpu_memory_total_gb": 24.0
    }
}
```

---

## Configuration Management

### Configuration Hierarchy

```python
# config.py

from pydantic import BaseModel, Field
from typing import Dict, Optional, List
import os
from functools import lru_cache


class TTSConfig(BaseModel):
    """TTS-specific configuration."""
    voice_design_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    base_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    idle_timeout_seconds: int = 300
    max_audio_length_seconds: int = 60  # Maximum TTS generation length


class MusicConfig(BaseModel):
    """Music generation configuration."""
    default_model: str = "musicgen-small"
    idle_timeout_seconds: int = 300
    max_duration_seconds: int = 30
    available_models: Dict[str, Dict] = {
        "musicgen-small": {
            "repo": "facebook/musicgen-small",
            "vram_gb": 6,
            "max_duration": 30,
            "source": "huggingface"
        },
        "musicgen-medium": {
            "repo": "facebook/musicgen-medium", 
            "vram_gb": 12,
            "max_duration": 60,
            "source": "huggingface"
        }
    }


class SFXConfig(BaseModel):
    """SFX generation configuration."""
    default_model: str = "audioldm-m-full"
    idle_timeout_seconds: int = 300
    max_duration_seconds: int = 10
    available_models: Dict[str, Dict] = {
        "audioldm-m-full": {
            "repo": "cvssp/audioldm-m-full",
            "vram_gb": 8,
            "max_duration": 10,
            "source": "huggingface_diffusers"
        }
    }


class ResourceConfig(BaseModel):
    """Resource management configuration."""
    # If True, only one model type can be loaded at a time
    exclusive_mode: bool = True
    
    # Priority when multiple requests compete (lower = higher priority)
    priority_order: List[str] = ["tts", "sfx", "music"]
    
    # Max VRAM usage before unloading others (GB)
    vram_threshold_gb: float = 20.0
    
    # Check interval for idle monitor (seconds)
    idle_check_interval: int = 60


class VolsungConfig(BaseModel):
    """Root configuration."""
    tts: TTSConfig = TTSConfig()
    music: MusicConfig = MusicConfig()
    sfx: SFXConfig = SFXConfig()
    resources: ResourceConfig = ResourceConfig()
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    
    # Device overrides
    device: Optional[str] = None  # "cuda", "cpu", "mps", or None for auto
    dtype: Optional[str] = None   # "float16", "float32", or None for auto


@lru_cache()
def get_config() -> VolsungConfig:
    """
    Load configuration from environment variables and defaults.
    
    Environment variable format: VOLSUNG_<SECTION>_<KEY>
    Examples:
        VOLSUNG_TTS_IDLE_TIMEOUT_SECONDS=600
        VOLSUNG_MUSIC_DEFAULT_MODEL=musicgen-medium
        VOLSUNG_RESOURCES_EXCLUSIVE_MODE=false
    """
    config_dict = {}
    
    # Load from env vars
    for key, value in os.environ.items():
        if key.startswith("VOLSUNG_"):
            parts = key[8:].lower().split("_")
            # Parse into nested dict
            # e.g., VOLSUNG_TTS_IDLE_TIMEOUT_SECONDS → config["tts"]["idle_timeout_seconds"]
            target = config_dict
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            
            # Convert value type
            try:
                # Try int
                target[parts[-1]] = int(value)
            except ValueError:
                try:
                    # Try float
                    target[parts[-1]] = float(value)
                except ValueError:
                    # Try bool
                    if value.lower() in ("true", "false"):
                        target[parts[-1]] = value.lower() == "true"
                    else:
                        target[parts[-1]] = value
    
    return VolsungConfig(**config_dict)
```

### Configuration File Support

```python
# Optional: Load from YAML/JSON file

def load_config_from_file(path: str) -> VolsungConfig:
    """Load configuration from file."""
    import json
    import yaml
    
    with open(path) as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            config_dict = yaml.safe_load(f)
        else:
            config_dict = json.load(f)
    
    return VolsungConfig(**config_dict)
```

Example `config.yaml`:

```yaml
# Volsung Configuration

tts:
  idle_timeout_seconds: 300
  max_audio_length_seconds: 60

music:
  default_model: "musicgen-small"
  idle_timeout_seconds: 300
  max_duration_seconds: 30

sfx:
  default_model: "audioldm-m-full"
  idle_timeout_seconds: 300
  max_duration_seconds: 10

resources:
  exclusive_mode: true
  priority_order: ["tts", "sfx", "music"]
  vram_threshold_gb: 20.0

server:
  host: "0.0.0.0"
  port: 8000
```

---

## Thread Safety & Concurrency

### Lock Strategy

```python
# Each manager uses RLock for thread safety

class ModelManagerBase(ABC):
    def __init__(self, config: ModelConfig):
        self._lock = threading.RLock()  # Reentrant lock
        # ...
    
    def generate(self, *args, **kwargs) -> AudioResult:
        with self._lock:
            self._ensure_loaded()
            return self._do_generate(*args, **kwargs)
    
    @abstractmethod
    def _do_generate(self, *args, **kwargs) -> AudioResult:
        """Actual generation, called within lock."""
        pass
```

### Request Queue

```python
from queue import PriorityQueue
import uuid


class GenerationRequest:
    def __init__(self, model_type: str, priority: int, callback):
        self.id = str(uuid.uuid4())
        self.model_type = model_type
        self.priority = priority
        self.callback = callback
        self.created_at = time.time()
    
    def __lt__(self, other):
        # Higher priority = lower number = first in queue
        if self.priority != other.priority:
            return self.priority < other.priority
        # Same priority: FIFO
        return self.created_at < other.created_at


class RequestQueue:
    """Priority queue for generation requests."""
    
    def __init__(self):
        self._queue = PriorityQueue()
        self._pending: Dict[str, GenerationRequest] = {}
        self._lock = threading.Lock()
    
    def enqueue(self, request: GenerationRequest) -> str:
        with self._lock:
            self._queue.put(request)
            self._pending[request.id] = request
        return request.id
    
    def dequeue(self) -> Optional[GenerationRequest]:
        with self._lock:
            if self._queue.empty():
                return None
            request = self._queue.get()
            del self._pending[request.id]
            return request
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1)

1. Create `models/` directory with base classes
2. Refactor existing TTS code into `tts/` module
3. Verify TTS still works end-to-end
4. Add configuration system

### Phase 2: Music Module (Week 2)

1. Create `music/` module structure
2. Implement `MusicGenGenerator`
3. Implement `MusicModelManager`
4. Add `/music/*` endpoints
5. Test lazy loading and idle timeout

### Phase 3: SFX Module (Week 3)

1. Create `sfx/` module structure
2. Implement `AudioLDMGenerator`
3. Implement `SFXModelManager`
4. Add `/sfx/*` endpoints
5. Test concurrent resource management

### Phase 4: Integration & Stitcher (Week 4)

1. Update health endpoint with all modules
2. Add resource manager for coordination
3. Implement stitcher foundation
4. End-to-end testing

---

## Migration Guide: Existing Code

### Current State (Single File)

```python
# volsung/server.py (current)
from fastapi import FastAPI
from qwen_tts import Qwen3TTSModel

app = FastAPI()

# Global model variables
voice_design_model = None
base_model = None
models_loaded = False

# Idle monitoring
last_access_time = 0.0
idle_lock = threading.Lock()
...

@app.post("/voice_design")
async def voice_design(req: VoiceDesignRequest):
    if not models_loaded:
        load_models()
    # ... generate audio
```

### Target State (Modular)

```python
# volsung/server.py (refactored)
from fastapi import FastAPI
from volsung.config import get_config
from volsung.tts.endpoints import router as tts_router
from volsung.music.endpoints import router as music_router
from volsung.sfx.endpoints import router as sfx_router

app = FastAPI()
config = get_config()

# Register all module routers
app.include_router(tts_router, prefix="/voice", tags=["TTS"])
app.include_router(music_router, prefix="/music", tags=["Music"])
app.include_router(sfx_router, prefix="/sfx", tags=["SFX"])

# Module managers initialized in routers
# Each manages its own lazy loading and idle timeout
```

```python
# volsung/tts/endpoints.py
from fastapi import APIRouter, HTTPException
from volsung.models.registry import get_registry

router = APIRouter()
registry = get_registry()

# Lazy init on first request
_tts_manager = None

def get_tts_manager():
    global _tts_manager
    if _tts_manager is None:
        from .manager import TTSModelManager
        _tts_manager = TTSModelManager()
        registry.register("tts", _tts_manager)
    return _tts_manager

@router.post("/design")
async def voice_design(req: VoiceDesignRequest):
    manager = get_tts_manager()
    result = manager.generate_voice_design(req.text, req.language, req.instruct)
    return result

@router.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    manager = get_tts_manager()
    result = manager.synthesize(req.ref_audio, req.ref_text, req.text, req.language)
    return result
```

---

## Error Handling & Logging

### Module-Level Logging

```python
# Each module has its own logger

import logging

logger = logging.getLogger(f"volsung.{__name__}")

class MusicModelManager(ModelManagerBase):
    def _load_model(self):
        logger.info("Loading music generation model...")
        try:
            # ... load
            logger.info(f"Music model loaded successfully (VRAM: {vram_used:.1f}GB)")
        except Exception as e:
            logger.error(f"Failed to load music model: {e}", exc_info=True)
            raise
```

### Structured Errors

```python
class VolsungError(Exception):
    """Base exception for volsung."""
    def __init__(self, message: str, error_code: str, details: dict = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}

class ModelLoadError(VolsungError):
    """Failed to load model."""
    pass

class GenerationError(VolsungError):
    """Failed to generate audio."""
    pass

class ResourceExhaustedError(VolsungError):
    """Not enough resources (VRAM, etc.)."""
    pass

# In endpoints:
@app.exception_handler(VolsungError)
async def volsung_exception_handler(request: Request, exc: VolsungError):
    return JSONResponse(
        status_code=500,
        content={
            "error": exc.error_code,
            "message": str(exc),
            "details": exc.details
        }
    )
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_music_manager.py
import pytest
from volsung.music.manager import MusicModelManager

class TestMusicModelManager:
    def test_lazy_loading(self):
        manager = MusicModelManager()
        assert not manager.is_loaded
        
        # Mock generator to avoid loading real model
        manager._current_generator = MockGenerator()
        manager._loaded = True
        
        assert manager.is_loaded
    
    def test_idle_timeout(self):
        manager = MusicModelManager()
        # ... test idle detection
```

### Integration Tests

```python
# tests/test_endpoints.py
import pytest
from fastapi.testclient import TestClient
from volsung.server import app

client = TestClient(app)

def test_music_generation_endpoint():
    response = client.post("/music/generate", json={
        "prompt": "calm piano music",
        "duration": 10
    })
    assert response.status_code == 200
    result = response.json()
    assert "audio" in result
    assert result["sample_rate"] == 24000
```

---

## Performance Considerations

### Memory Optimization

```python
class ModelManagerBase:
    def _unload_model(self):
        """Unload with memory optimization."""
        # 1. Delete model references
        self._model = None
        
        # 2. Force garbage collection
        import gc
        gc.collect()
        
        # 3. Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
```

### Batch Generation (Future)

```python
class MusicModelManager:
    def generate_batch(self, prompts: List[str], duration: float) -> List[AudioResult]:
        """
        Generate multiple music pieces in batch.
        More efficient than sequential calls.
        """
        self._ensure_loaded()
        # ... batch inference
```

---

## Security Considerations

### Input Validation

```python
from fastapi import HTTPException

class MusicGenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    duration: float = Field(default=30.0, ge=1.0, le=30.0)
    model: Optional[str] = None
    
    @validator('prompt')
    def validate_prompt(cls, v):
        # Check for potentially harmful content
        if any(bad in v.lower() for bad in BLOCKED_WORDS):
            raise ValueError("Prompt contains disallowed content")
        return v
```

### Resource Limits

```python
# Prevent abuse via rate limiting / resource limits
MAX_GENERATIONS_PER_MINUTE = 10
MAX_AUDIO_DURATION_MINUTES = 5
```

---

## Conclusion

This architecture provides:

1. **Clean Separation**: Each module (TTS, Music, SFX) is independent
2. **Unified Interface**: Common base classes and standardized audio results
3. **Resource Management**: Lazy loading, idle timeout, and VRAM coordination
4. **Extensibility**: Easy to add new model types via generator pattern
5. **Future-Proof**: Stitcher foundation for audio composition

The key insight is that all three generation types (voice, music, SFX) share the same pattern:
- Lazy model loading on first use
- Idle timeout to free resources
- Text/audio prompt → model inference → audio output

By extracting this commonality into `ModelManagerBase`, we avoid code duplication and ensure consistent behavior across all modules.

---

## Appendix A: Dependencies

Update `pyproject.toml`:

```toml
[project]
dependencies = [
    # Existing
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.23.0",
    "torch>=2.0.0",
    "transformers>=4.52.0",
    "qwen-tts",
    "soundfile>=0.12.0",
    "pydantic>=2.0.0",
    "numpy>=1.24.0",
    # New for Music/SFX
    "diffusers>=0.25.0",      # For AudioLDM
    "audioldm",              # AudioLDM wrapper
    "torchaudio>=2.0.0",     # Audio processing
]
```

## Appendix B: Directory Structure

```
volsung/
├── __init__.py
├── __main__.py
├── server.py           # FastAPI app setup
├── config.py             # Configuration management
├── models/               # Abstract base classes
│   ├── __init__.py
│   ├── base.py
│   ├── registry.py
│   └── types.py
├── tts/                  # TTS module
│   ├── __init__.py
│   ├── manager.py
│   ├── endpoints.py
│   └── schemas.py
├── music/                # Music module
│   ├── __init__.py
│   ├── manager.py
│   ├── endpoints.py
│   ├── schemas.py
│   └── generators/
│       ├── __init__.py
│       └── musicgen.py
├── sfx/                  # SFX module
│   ├── __init__.py
│   ├── manager.py
│   ├── endpoints.py
│   ├── schemas.py
│   └── generators/
│       ├── __init__.py
│       └── audioldm.py
├── audio/                # Audio utilities
│   ├── __init__.py
│   ├── utils.py
│   └── effects.py
└── stitcher/             # Future composition
    ├── __init__.py
    ├── composer.py
    ├── schemas.py
    └── endpoints.py
```

## Appendix C: Model Registry

```python
# models/registry.py

from typing import Dict, Type
from .base import ModelManagerBase

class ModelRegistry:
    """Registry of all model managers."""
    
    def __init__(self):
        self._managers: Dict[str, ModelManagerBase] = {}
    
    def register(self, name: str, manager: ModelManagerBase):
        self._managers[name] = manager
    
    def get(self, name: str) -> ModelManagerBase:
        return self._managers[name]
    
    def list_loaded(self) -> Dict[str, bool]:
        return {name: m.is_loaded for name, m in self._managers.items()}
    
    def unload_all(self):
        for manager in self._managers.values():
            if manager.is_loaded:
                manager.unload_if_idle()


# Singleton registry
_registry = ModelRegistry()

def get_registry() -> ModelRegistry:
    return _registry
```
