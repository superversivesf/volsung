"""Unified Preload Manager for Volsung models.

Provides centralized model loading and unloading with GPU memory optimization.
Only unloads models when switching to a different set to avoid unnecessary reloads.

Example:
    ```python
    from volsung.models.preload_manager import PreloadManager, get_preload_manager

    # Use singleton instance (recommended)
    manager = get_preload_manager()

    # Preload specific models
    result = manager.preload(["qwen3", "styletts2"])

    # Preload all models
    result = manager.preload(["all"])

    # Get currently loaded models
    loaded = manager.get_loaded_models()

    # Unload all models
    manager.unload_all()

    # Unload all except specific ones
    manager.unload_models(except_models=["qwen3"])
    ```
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Set

import torch

logger = logging.getLogger(__name__)


class PreloadManager:
    """Unified manager for loading and unloading models across all Volsung modules.

    Provides centralized control with the following key behaviors:
    - Smart loading: Only loads models not already loaded
    - Smart unloading: Only unloads when switching to different model set
    - GPU memory tracking: Logs VRAM usage before/after operations
    - Thread-safe: All operations use reentrant locks

    Supported models:
        - "qwen3": Qwen3-TTS voice models (VoiceDesign + Base)
        - "styletts2": StyleTTS 2 voice cloning model
        - "music": MusicGen music generation model
        - "sfx": AudioLDM sound effects model
        - "all": All of the above models

    Example:
        ```python
        manager = PreloadManager()

        # Load qwen3 and music models only
        result = manager.preload(["qwen3", "music"])
        # Returns: {"status": "loaded", "models": ["qwen3", "music"]}

        # Try to load same models - no-op since already loaded
        result = manager.preload(["qwen3", "music"])
        # Returns: {"status": "already_loaded", "models": ["qwen3", "music"]}

        # Switch to different set - unloads qwen3, loads styletts2 and sfx
        result = manager.preload(["styletts2", "sfx", "music"])
        # Unloads: qwen3
        # Loads: styletts2, sfx (music already loaded, no-op)
        ```
    """

    # Model name constants
    MODEL_QWEN3 = "qwen3"
    MODEL_STYLETTS2 = "styletts2"
    MODEL_MUSIC = "music"
    MODEL_SFX = "sfx"
    MODEL_ALL = "all"

    # All valid individual models (excluding "all")
    VALID_MODELS = {MODEL_QWEN3, MODEL_STYLETTS2, MODEL_MUSIC, MODEL_SFX}

    def __init__(self):
        """Initialize the preload manager."""
        self._lock = threading.RLock()
        self._qwen3_loaded = False
        self._styletts2_loaded = False
        self._music_loaded = False
        self._sfx_loaded = False

        # Store references to models for direct unload
        self._qwen3_models = None
        self._styletts2_manager = None
        self._music_manager = None
        self._sfx_manager = None

        logger.info("PreloadManager initialized")

    def _log_vram_usage(self, label: str) -> None:
        """Log current GPU VRAM usage.

        Args:
            label: Label to identify this log entry
        """
        if not torch.cuda.is_available():
            return

        try:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(
                f"[VRAM {label}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
            )
        except Exception as e:
            logger.debug(f"Could not log VRAM usage: {e}")

    def _clear_gpu_cache(self) -> None:
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU cache cleared")

    def _expand_models(self, models: List[str]) -> Set[str]:
        """Expand "all" to individual model names.

        Args:
            models: List of model names, may include "all"

        Returns:
            Set of individual model names
        """
        if self.MODEL_ALL in models:
            return self.VALID_MODELS.copy()
        return set(models)

    def _get_vram_before(self) -> Optional[float]:
        """Get current VRAM allocated for comparison.

        Returns:
            GB allocated, or None if CUDA not available
        """
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return None

    def _log_vram_delta(self, label: str, before: Optional[float]) -> None:
        """Log VRAM change from before value.

        Args:
            label: Operation label
            before: VRAM in GB before operation
        """
        if before is None or not torch.cuda.is_available():
            return

        after = torch.cuda.memory_allocated() / 1024**3
        delta = after - before
        sign = "+" if delta > 0 else ""
        logger.info(f"[VRAM {label}] Change: {sign}{delta:.2f} GB (now {after:.2f} GB)")

    def preload(self, requested_models: List[str]) -> Dict[str, any]:
        """Load requested models, only unloading if switching to different set.

        This is the main entry point for model loading. It implements smart
        loading logic that:
        1. Expands "all" to individual model names
        2. Checks if requested models match currently loaded
        3. Only unloads models not in the requested set
        4. Only loads models not already loaded
        5. Clears GPU cache after unload operations
        6. Logs VRAM usage before/after

        Args:
            requested_models: List of model names to load.
                Valid values: "qwen3", "styletts2", "music", "sfx", "all"

        Returns:
            Dictionary with:
                - "status": "already_loaded", "loaded", or "partial"
                - "models": List of models now loaded
                - "loaded": List of models that were loaded this call (if any)
                - "unloaded": List of models that were unloaded this call (if any)

        Raises:
            ValueError: If invalid model names provided
            RuntimeError: If model loading fails

        Example:
            ```python
            # Load music and sfx
            result = manager.preload(["music", "sfx"])

            # Try loading same models - returns immediately
            result = manager.preload(["music", "sfx"])
            # Returns: {"status": "already_loaded", "models": ["music", "sfx"]}

            # Switch to TTS models - music/sfx unloaded, qwen3/styletts2 loaded
            result = manager.preload(["qwen3", "styletts2"])
            ```
        """
        with self._lock:
            # Expand "all" to individual models
            expanded_requested = self._expand_models(requested_models)

            # Validate model names
            invalid = expanded_requested - self.VALID_MODELS
            if invalid:
                raise ValueError(
                    f"Invalid model names: {invalid}. Valid: {self.VALID_MODELS}"
                )

            current_loaded = self._get_loaded_models_set()

            # If requested models match current loaded, skip everything
            if expanded_requested == current_loaded:
                logger.info(f"Models already loaded: {sorted(current_loaded)}")
                return {
                    "status": "already_loaded",
                    "models": sorted(current_loaded),
                    "loaded": [],
                    "unloaded": [],
                }

            # Track what we actually do
            loaded_now = []
            unloaded_now = []

            # Log VRAM before operations
            vram_before = self._get_vram_before()
            self._log_vram_usage("before preload")

            # Unload models not in requested list
            for model in list(current_loaded):
                if model not in expanded_requested:
                    self._unload_single_model(model)
                    unloaded_now.append(model)

            if unloaded_now:
                self._clear_gpu_cache()
                self._log_vram_delta("after unload", vram_before)
                vram_before = self._get_vram_before()

            # Load requested models that aren't loaded
            for model in expanded_requested:
                if model not in current_loaded:
                    self._load_single_model(model)
                    loaded_now.append(model)

            if loaded_now:
                self._log_vram_delta("after load", vram_before)

            final_loaded = self._get_loaded_models_set()

            status = (
                "loaded"
                if loaded_now and not unloaded_now
                else "partial"
                if loaded_now or unloaded_now
                else "already_loaded"
            )

            result = {
                "status": status,
                "models": sorted(final_loaded),
                "loaded": loaded_now,
                "unloaded": unloaded_now,
            }

            logger.info(f"Preload complete: {result}")
            return result

    def _get_loaded_models_set(self) -> Set[str]:
        """Get set of currently loaded models.

        Returns:
            Set of loaded model names
        """
        loaded = set()
        if self._qwen3_loaded:
            loaded.add(self.MODEL_QWEN3)
        if self._styletts2_loaded:
            loaded.add(self.MODEL_STYLETTS2)
        if self._music_loaded:
            loaded.add(self.MODEL_MUSIC)
        if self._sfx_loaded:
            loaded.add(self.MODEL_SFX)
        return loaded

    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models.

        Returns:
            List of loaded model names in sorted order
        """
        with self._lock:
            return sorted(self._get_loaded_models_set())

    def _load_single_model(self, model: str) -> None:
        """Load a single model by name.

        Args:
            model: Model name to load

        Raises:
            RuntimeError: If loading fails
        """
        logger.info(f"Loading model: {model}")
        start_time = time.time()

        try:
            if model == self.MODEL_QWEN3:
                self._load_qwen3()
            elif model == self.MODEL_STYLETTS2:
                self._load_styletts2()
            elif model == self.MODEL_MUSIC:
                self._load_music()
            elif model == self.MODEL_SFX:
                self._load_sfx()

            elapsed = time.time() - start_time
            logger.info(f"Loaded {model} in {elapsed:.1f}s")

        except Exception as e:
            logger.error(f"Failed to load {model}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load {model}: {e}") from e

    def _unload_single_model(self, model: str) -> None:
        """Unload a single model by name.

        Args:
            model: Model name to unload
        """
        logger.info(f"Unloading model: {model}")

        if model == self.MODEL_QWEN3:
            self._unload_qwen3()
        elif model == self.MODEL_STYLETTS2:
            self._unload_styletts2()
        elif model == self.MODEL_MUSIC:
            self._unload_music()
        elif model == self.MODEL_SFX:
            self._unload_sfx()

    def _load_qwen3(self) -> None:
        """Load Qwen3-TTS models (VoiceDesign + Base)."""
        from volsung.server import load_models as load_qwen3_models

        # Call the server's load_models function
        # Note: This loads both VoiceDesign and Base models
        load_qwen3_models()
        self._qwen3_loaded = True

        # Import and store references for unload
        import volsung.server as server_module

        self._qwen3_models = {
            "voice_design": getattr(server_module, "voice_design_model", None),
            "base": getattr(server_module, "base_model", None),
        }

    def _unload_qwen3(self) -> None:
        """Unload Qwen3-TTS models."""
        import volsung.server as server_module

        # Clear references in server module
        server_module.voice_design_model = None
        server_module.base_model = None
        server_module.models_loaded = False

        self._qwen3_models = None
        self._qwen3_loaded = False
        logger.info("Qwen3 models unloaded")

    def _load_styletts2(self) -> None:
        """Load StyleTTS 2 model."""
        from volsung.tts.managers.styletts2 import get_styletts2_manager

        manager = get_styletts2_manager()
        manager._ensure_loaded()
        self._styletts2_manager = manager
        self._styletts2_loaded = True

    def _unload_styletts2(self) -> None:
        """Unload StyleTTS 2 model."""
        if self._styletts2_manager:
            self._styletts2_manager.force_unload()
            self._styletts2_manager = None
        self._styletts2_loaded = False
        logger.info("StyleTTS 2 model unloaded")

    def _load_music(self) -> None:
        """Load MusicGen model."""
        from volsung.music.manager import MusicModelManager
        from volsung.models.base import ModelConfig

        # Create config for music model
        config = ModelConfig(
            model_id="musicgen-small",
            model_name="MusicGen Small",
            device="auto",
            dtype="auto",
            idle_timeout_seconds=0,  # Disable idle timeout (managed by PreloadManager)
        )

        manager = MusicModelManager(config)
        manager._ensure_loaded()
        self._music_manager = manager
        self._music_loaded = True

    def _unload_music(self) -> None:
        """Unload MusicGen model."""
        if self._music_manager:
            self._music_manager.force_unload()
            self._music_manager = None
        self._music_loaded = False
        logger.info("Music model unloaded")

    def _load_sfx(self) -> None:
        """Load AudioLDM SFX model."""
        from volsung.sfx.manager import SFXModelManager

        manager = SFXModelManager()
        manager._ensure_loaded()
        self._sfx_manager = manager
        self._sfx_loaded = True

    def _unload_sfx(self) -> None:
        """Unload AudioLDM SFX model."""
        if self._sfx_manager:
            self._sfx_manager.force_unload()
            self._sfx_manager = None
        self._sfx_loaded = False
        logger.info("SFX model unloaded")

    def unload_all(self) -> None:
        """Unload all models and clear GPU cache.

        Example:
            ```python
            manager = PreloadManager()
            manager.preload(["qwen3", "music"])
            # ... use models ...
            manager.unload_all()  # Free all GPU memory
            ```
        """
        with self._lock:
            logger.info("Unloading all models...")
            vram_before = self._get_vram_before()
            self._log_vram_usage("before unload_all")

            self._unload_qwen3()
            self._unload_styletts2()
            self._unload_music()
            self._unload_sfx()

            self._clear_gpu_cache()
            self._log_vram_delta("after unload_all", vram_before)
            logger.info("All models unloaded")

    def unload_models(self, except_models: List[str]) -> None:
        """Unload all models except the specified ones.

        This is useful when switching between workloads while keeping
        certain models resident.

        Args:
            except_models: List of model names to keep loaded

        Example:
            ```python
            manager = PreloadManager()
            manager.preload(["qwen3", "styletts2", "music", "sfx"])

            # Unload everything except qwen3
            manager.unload_models(except_models=["qwen3"])
            # Now only qwen3 is loaded
            ```
        """
        with self._lock:
            except_set = self._expand_models(except_models)
            current_loaded = self._get_loaded_models_set()

            to_unload = current_loaded - except_set
            if not to_unload:
                logger.info("No models to unload (all are in except list)")
                return

            logger.info(
                f"Unloading models except {sorted(except_set)}: {sorted(to_unload)}"
            )
            vram_before = self._get_vram_before()
            self._log_vram_usage("before unload_models")

            for model in to_unload:
                self._unload_single_model(model)

            self._clear_gpu_cache()
            self._log_vram_delta("after unload_models", vram_before)

    def is_model_loaded(self, model: str) -> bool:
        """Check if a specific model is loaded.

        Args:
            model: Model name to check

        Returns:
            True if model is loaded, False otherwise

        Example:
            ```python
            if manager.is_model_loaded("qwen3"):
                print("Qwen3 is ready")
            ```
        """
        with self._lock:
            if model == self.MODEL_ALL:
                return (
                    self._qwen3_loaded
                    and self._styletts2_loaded
                    and self._music_loaded
                    and self._sfx_loaded
                )
            elif model == self.MODEL_QWEN3:
                return self._qwen3_loaded
            elif model == self.MODEL_STYLETTS2:
                return self._styletts2_loaded
            elif model == self.MODEL_MUSIC:
                return self._music_loaded
            elif model == self.MODEL_SFX:
                return self._sfx_loaded
            return False


# Singleton instance
_preload_manager: Optional[PreloadManager] = None


def get_preload_manager() -> PreloadManager:
    """Get the singleton PreloadManager instance.

    Returns:
        Singleton PreloadManager instance

    Example:
        ```python
        from volsung.models.preload_manager import get_preload_manager

        manager = get_preload_manager()
        result = manager.preload(["qwen3", "music"])
        ```
    """
    global _preload_manager
    if _preload_manager is None:
        _preload_manager = PreloadManager()
    return _preload_manager
