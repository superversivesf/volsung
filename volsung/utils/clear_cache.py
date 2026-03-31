"""Utility module for clearing corrupted model cache directories.

Provides functions to safely remove corrupted model cache directories
with support for checking active models and logging operations.

Example:
    # Clear all corrupted caches
    from volsung.utils.clear_cache import clear_model_cache
    result = clear_model_cache()
    print(result)

    # Clear specific model
    result = clear_model_cache(model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    print(result)
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


def get_model_cache_dir() -> str:
    """Get the model cache directory from volsung config.

    Returns:
        Absolute path to the models cache directory.
    """
    # Try to import from volsung.config first
    try:
        from volsung.config import get_model_cache_dir as config_cache_dir

        return config_cache_dir()
    except ImportError:
        # Fallback: compute relative to this file
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
        return str(project_root / "models")


def get_loaded_models() -> List[str]:
    """Get list of currently loaded models to avoid clearing them.

    Returns:
        List of loaded model names/identifiers.
    """
    loaded = []
    try:
        from volsung.models.preload_manager import get_preload_manager

        manager = get_preload_manager()
        loaded = manager.get_loaded_models()
    except ImportError:
        logger.warning(
            "Could not import preload manager, unable to check loaded models"
        )
    return loaded


def get_model_cache_subdirectories(cache_dir: str) -> List[Path]:
    """List all subdirectories in the model cache.

    Args:
        cache_dir: Path to the cache directory.

    Returns:
        List of subdirectory paths.
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        logger.warning(f"Cache directory does not exist: {cache_dir}")
        return []

    subdirs = [d for d in cache_path.iterdir() if d.is_dir()]
    return sorted(subdirs)


def is_corrupted_cache(cache_path: Path) -> bool:
    """Check if a cache directory appears corrupted.

    Heuristics for corruption:
    - Directory is empty
    - Directory contains incomplete download markers
    - Directory is significantly smaller than expected

    Args:
        cache_path: Path to the cache directory.

    Returns:
        True if directory appears corrupted.
    """
    if not cache_path.exists():
        return False

    # Check if directory is empty
    contents = list(cache_path.iterdir())
    if not contents:
        logger.info(f"Cache directory is empty: {cache_path.name}")
        return True

    # Check for incomplete download markers
    incomplete_markers = [".incomplete", ".tmp", ".downloading", ".part"]
    for item in contents:
        if any(marker in item.name.lower() for marker in incomplete_markers):
            logger.info(f"Incomplete download marker found in: {cache_path.name}")
            return True

    return False


def clear_model_cache(
    model_name: Optional[str] = None, confirm: bool = False, force: bool = False
) -> Dict[str, Any]:
    """Clear corrupted model cache directories.

    Safely removes corrupted model cache directories. If a model is currently
    loaded, it will not be cleared unless force=True.

    Args:
        model_name: Specific model to clear. If None, clears all corrupted caches.
        confirm: If True, clears all model caches (not just corrupted ones).
        force: If True, clears even if model is currently loaded.

    Returns:
        Dictionary with:
            - "success": True if operation completed
            - "cleared": List of directories cleared
            - "skipped_loaded": List of directories skipped (loaded models)
            - "skipped_not_corrupted": List of directories skipped (not corrupted)
            - "errors": List of error messages
    """
    cache_dir = get_model_cache_dir()
    logger.info(f"Checking cache directory: {cache_dir}")

    result = {
        "success": True,
        "cleared": [],
        "skipped_loaded": [],
        "skipped_not_corrupted": [],
        "errors": [],
    }

    # Get currently loaded models
    loaded_models = get_loaded_models()
    logger.info(f"Currently loaded models: {loaded_models}")

    # Get subdirectories
    subdirs = get_model_cache_subdirectories(cache_dir)

    if not subdirs:
        logger.info("No cache subdirectories found")
        return result

    for subdir in subdirs:
        dir_name = subdir.name

        # If specific model requested, skip others
        if model_name and model_name not in dir_name:
            continue

        # Check if this directory corresponds to a loaded model
        is_loaded = any(
            loaded in dir_name.lower() or dir_name.lower() in loaded
            for loaded in loaded_models
        )

        if is_loaded and not force:
            logger.info(f"Skipping loaded model cache: {dir_name}")
            result["skipped_loaded"].append(str(subdir))
            continue

        # Check if corrupted (or confirm=True for all)
        if not confirm and not is_corrupted_cache(subdir):
            logger.debug(f"Skipping non-corrupted cache: {dir_name}")
            result["skipped_not_corrupted"].append(str(subdir))
            continue

        # Clear the directory
        try:
            if confirm or is_corrupted_cache(subdir):
                logger.info(f"Clearing cache: {dir_name}")
                shutil.rmtree(subdir)
                result["cleared"].append(str(subdir))
        except Exception as e:
            error_msg = f"Failed to clear {dir_name}: {str(e)}"
            logger.error(error_msg)
            result["errors"].append(error_msg)

    # Log summary
    cleared_count = len(result["cleared"])
    skipped_loaded_count = len(result["skipped_loaded"])
    skipped_corrupted_count = len(result["skipped_not_corrupted"])
    error_count = len(result["errors"])

    logger.info(
        f"Cache clear complete: {cleared_count} cleared, "
        f"{skipped_loaded_count} skipped (loaded), "
        f"{skipped_corrupted_count} skipped (not corrupted), "
        f"{error_count} errors"
    )

    result["success"] = error_count == 0
    return result


def get_cache_status() -> Dict[str, Any]:
    """Get status of model cache directories.

    Returns:
        Dictionary with cache status information.
    """
    cache_dir = get_model_cache_dir()
    cache_path = Path(cache_dir)

    result = {
        "cache_dir": cache_dir,
        "exists": cache_path.exists(),
        "total_size_mb": 0.0,
        "directories": [],
        "loaded_models": get_loaded_models(),
    }

    if not cache_path.exists():
        return result

    subdirs = get_model_cache_subdirectories(cache_dir)
    total_size = 0

    for subdir in subdirs:
        dir_size = sum(f.stat().st_size for f in subdir.rglob("*") if f.is_file())
        total_size += dir_size

        dir_info = {
            "name": subdir.name,
            "path": str(subdir),
            "size_mb": round(dir_size / (1024 * 1024), 2),
            "is_corrupted": is_corrupted_cache(subdir),
        }
        result["directories"].append(dir_info)

    result["total_size_mb"] = round(total_size / (1024 * 1024), 2)
    return result


if __name__ == "__main__":
    # Command-line interface
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Clear corrupted model cache directories"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to clear (default: all corrupted)",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Clear all caches, not just corrupted ones",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear even if model is currently loaded",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show cache status without clearing",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.status:
        status = get_cache_status()
        print(f"Cache directory: {status['cache_dir']}")
        print(f"Exists: {status['exists']}")
        print(f"Total size: {status['total_size_mb']:.2f} MB")
        print(f"Loaded models: {status['loaded_models']}")
        print("\nDirectories:")
        for dir_info in status["directories"]:
            corrupted = " (CORRUPTED)" if dir_info["is_corrupted"] else ""
            print(f"  - {dir_info['name']}: {dir_info['size_mb']:.2f} MB{corrupted}")
        sys.exit(0)

    result = clear_model_cache(
        model_name=args.model,
        confirm=args.confirm,
        force=args.force,
    )

    print(f"\nCache clear result:")
    print(f"  Success: {result['success']}")
    print(f"  Cleared: {len(result['cleared'])} directories")
    for cleared in result["cleared"]:
        print(f"    - {cleared}")

    if result["skipped_loaded"]:
        print(f"  Skipped (loaded): {len(result['skipped_loaded'])} directories")

    if result["skipped_not_corrupted"]:
        print(
            f"  Skipped (not corrupted): {len(result['skipped_not_corrupted'])} directories"
        )

    if result["errors"]:
        print(f"  Errors: {len(result['errors'])}")
        for error in result["errors"]:
            print(f"    - {error}")

    sys.exit(0 if result["success"] else 1)
