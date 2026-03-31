"""
Volsung - Voice synthesis server for Qwen3-TTS.

Named after the Völsung saga of Norse mythology, where heroes' deeds
were preserved through oral tradition. This server gives voice to text.
"""

import os
from pathlib import Path

__version__ = "1.0.0"


def _setup_model_cache():
    """Configure HF_HOME to use local models directory.

    This ensures all Hugging Face model downloads are stored in the
    project's models/ directory, keeping the installation self-contained.
    """
    # Get the project root (parent of volsung package)
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    # Set HF_HOME to the local models directory
    os.environ.setdefault("HF_HOME", str(models_dir))
    os.environ.setdefault("HF_CACHE_HOME", str(models_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(models_dir / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(models_dir / "transformers"))

    return str(models_dir)


# Set up model cache on module import
_MODEL_CACHE_DIR = _setup_model_cache()


def get_model_cache_dir() -> str:
    """Get the path to the local model cache directory.

    Returns:
        Absolute path to the models/ directory.
    """
    return _MODEL_CACHE_DIR


# Lazy import server to avoid heavy dependencies on submodule imports
# Use get_app() to access the FastAPI application
_app = None


def get_app():
    """Get the FastAPI application instance.

    This is lazily loaded to avoid importing heavy dependencies
    (torch, transformers, etc.) when importing submodules.
    """
    global _app
    if _app is None:
        from volsung.server import app as _app
    return _app


# Backward compatibility: app will be None until get_app() is called
app = None

__all__ = ["get_app", "app", "get_model_cache_dir", "__version__"]
