"""
Volsung - Voice synthesis server for Qwen3-TTS.

Named after the Völsung saga of Norse mythology, where heroes' deeds
were preserved through oral tradition. This server gives voice to text.
"""

__version__ = "1.0.0"

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

__all__ = ["get_app", "app", "__version__"]
