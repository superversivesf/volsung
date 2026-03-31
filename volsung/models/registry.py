"""Model registry for Volsung.

Provides a central registry for discovering and managing all model managers.
"""

import logging
from typing import Any, Dict, List, Optional

from .base import ModelManagerBase

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry of all model managers.

    Provides a central location for registering and discovering
    model managers across different modules (TTS, Music, SFX).

    The registry maintains references to all managers and provides
    utility methods for querying their status and managing resources.

    Example:
        ```python
        # Get the singleton registry
        registry = get_registry()

        # Register a manager
        registry.register("music", music_manager)

        # Check what's loaded
        loaded = registry.list_loaded()
        # {"tts": True, "music": False, "sfx": False}

        # Get a specific manager
        music_mgr = registry.get("music")

        # Unload all models
        registry.unload_all()
        ```
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._managers: Dict[str, ModelManagerBase] = {}
        logger.debug("ModelRegistry initialized")

    def register(self, name: str, manager: ModelManagerBase) -> None:
        """Register a model manager.

        Args:
            name: Unique name for this manager (e.g., "tts", "music", "sfx")
            manager: The model manager instance

        Raises:
            ValueError: If name is already registered
        """
        if name in self._managers:
            raise ValueError(f"Manager '{name}' is already registered")

        self._managers[name] = manager
        logger.info(f"Registered model manager: {name}")

    def unregister(self, name: str) -> Optional[ModelManagerBase]:
        """Unregister a model manager.

        Args:
            name: Name of the manager to unregister

        Returns:
            The removed manager, or None if not found
        """
        if name not in self._managers:
            logger.warning(f"Attempted to unregister unknown manager: {name}")
            return None

        manager = self._managers.pop(name)
        logger.info(f"Unregistered model manager: {name}")
        return manager

    def get(self, name: str) -> ModelManagerBase:
        """Get a registered model manager.

        Args:
            name: Name of the manager to retrieve

        Returns:
            The model manager instance

        Raises:
            KeyError: If manager is not registered
        """
        if name not in self._managers:
            raise KeyError(f"No model manager registered for '{name}'")
        return self._managers[name]

    def get_optional(self, name: str) -> Optional[ModelManagerBase]:
        """Get a registered model manager, returning None if not found.

        Args:
            name: Name of the manager to retrieve

        Returns:
            The model manager instance, or None if not registered
        """
        return self._managers.get(name)

    def list_all(self) -> List[str]:
        """List all registered manager names.

        Returns:
            List of registered manager names
        """
        return list(self._managers.keys())

    def list_loaded(self) -> Dict[str, bool]:
        """Get loading status of all managers.

        Returns:
            Dictionary mapping manager names to loaded status
        """
        return {name: manager.is_loaded for name, manager in self._managers.items()}

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed status of all managers.

        Returns:
            Dictionary with status information for each manager:
            {
                "tts": {
                    "loaded": True,
                    "idle_seconds": 45.2,
                    "model_name": "Qwen3-TTS"
                },
                ...
            }
        """
        status = {}
        for name, manager in self._managers.items():
            status[name] = {
                "loaded": manager.is_loaded,
                "idle_seconds": manager.idle_seconds,
                "model_name": manager.config.model_name,
                "model_id": manager.config.model_id,
            }
        return status

    def unload_all(self) -> Dict[str, bool]:
        """Unload all loaded models.

        Returns:
            Dictionary mapping manager names to whether they were unloaded
        """
        results = {}
        for name, manager in self._managers.items():
            if manager.is_loaded:
                manager.force_unload()
                results[name] = True
            else:
                results[name] = False
        logger.info(f"Unloaded {sum(results.values())} models")
        return results

    def unload_idle(self) -> Dict[str, bool]:
        """Unload all models that are idle.

        Returns:
            Dictionary mapping manager names to whether they were unloaded
        """
        results = {}
        for name, manager in self._managers.items():
            results[name] = manager.unload_if_idle()
        unloaded_count = sum(results.values())
        if unloaded_count > 0:
            logger.info(f"Unloaded {unloaded_count} idle models")
        return results

    def shutdown_all(self) -> None:
        """Shutdown all managers.

        This should be called when the application is shutting down.
        It unloads all models and cleans up resources.
        """
        for name, manager in self._managers.items():
            logger.info(f"Shutting down manager: {name}")
            manager.shutdown()
        self._managers.clear()
        logger.info("All managers shut down")

    def __contains__(self, name: str) -> bool:
        """Check if a manager is registered.

        Args:
            name: Manager name to check

        Returns:
            True if registered, False otherwise
        """
        return name in self._managers

    def __len__(self) -> int:
        """Get number of registered managers.

        Returns:
            Number of registered managers
        """
        return len(self._managers)


# Singleton registry instance
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get the singleton model registry.

    Returns:
        The global ModelRegistry instance

    Example:
        ```python
        from volsung.models.registry import get_registry

        registry = get_registry()
        registry.register("music", my_music_manager)
        ```
    """
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def reset_registry() -> None:
    """Reset the singleton registry (mainly for testing).

    This clears all registered managers and creates a fresh registry.
    Use with caution in production.
    """
    global _registry
    if _registry is not None:
        _registry.shutdown_all()
    _registry = ModelRegistry()
    logger.info("Registry reset")
