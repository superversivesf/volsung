"""
Configuration module for Volsung.

Provides centralized configuration management with support for:
- Environment variables (VOLSUNG_* prefix)
- YAML configuration files
- Pydantic validation
- Sensible defaults
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class TTSConfig(BaseModel):
    """Text-to-Speech configuration."""

    enabled: bool = Field(default=True, description="Enable TTS module")
    voice_design_model: str = Field(
        default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        description="Voice design model name",
    )
    base_model: str = Field(
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        description="Base voice cloning model name",
    )
    idle_timeout: int = Field(
        default=300,
        ge=0,
        description="Seconds before unloading TTS model (0 = never unload)",
    )
    dtype: Literal["float32", "bfloat16", "float16"] = Field(
        default="bfloat16",
        description="Model precision (bfloat16 recommended for CUDA)",
    )
    device: Optional[str] = Field(
        default=None,
        description="Device override (auto-detected if not set)",
    )

    @field_validator("device", mode="before")
    @classmethod
    def _validate_device(cls, v: Optional[str]) -> Optional[str]:
        """Allow empty string to mean None."""
        if v == "" or v == "auto":
            return None
        return v


class MusicGenerationConfig(BaseModel):
    """Music generation parameters."""

    top_k: int = Field(default=250, ge=1, le=1000, description="Top-k sampling")
    top_p: float = Field(default=0.0, ge=0.0, le=1.0, description="Nucleus sampling")
    temperature: float = Field(default=1.0, ge=0.1, le=2.0, description="Randomness")


class MusicConfig(BaseModel):
    """Music generation module configuration."""

    enabled: bool = Field(default=True, description="Enable music module")
    model: str = Field(
        default="facebook/musicgen-small",
        description="MusicGen model name",
    )
    idle_timeout: int = Field(
        default=300,
        ge=0,
        description="Seconds before unloading music model (0 = never unload)",
    )
    default_duration: float = Field(
        default=10.0,
        ge=1.0,
        le=30.0,
        description="Default generation duration in seconds",
    )
    generation: MusicGenerationConfig = Field(
        default_factory=MusicGenerationConfig,
        description="Generation parameters",
    )


class SFXGenerationConfig(BaseModel):
    """SFX generation parameters."""

    num_inference_steps: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Denoising steps (higher = better quality, slower)",
    )
    guidance_scale: float = Field(
        default=3.5,
        ge=1.0,
        le=20.0,
        description="Prompt adherence (higher = more faithful to prompt)",
    )


class SFXConfig(BaseModel):
    """Sound effects module configuration."""

    enabled: bool = Field(default=True, description="Enable SFX module")
    model: str = Field(
        default="cvssp/audioldm2",
        description="AudioLDM model name",
    )
    idle_timeout: int = Field(
        default=300,
        ge=0,
        description="Seconds before unloading SFX model (0 = never unload)",
    )
    default_duration: float = Field(
        default=5.0,
        ge=1.0,
        le=10.0,
        description="Default generation duration in seconds",
    )
    generation: SFXGenerationConfig = Field(
        default_factory=SFXGenerationConfig,
        description="Generation parameters",
    )


class ResourceConfig(BaseModel):
    """Resource management configuration."""

    max_concurrent_requests: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Max parallel generation requests",
    )
    clear_cache_between_requests: bool = Field(
        default=True,
        description="Clear GPU cache between model switches",
    )
    log_memory_usage: bool = Field(
        default=True,
        description="Log GPU memory usage",
    )
    idle_check_interval: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Seconds between idle checks",
    )


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = Field(default="0.0.0.0", description="Server bind address")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    log_level: Literal["debug", "info", "warning", "error"] = Field(
        default="info",
        description="Logging level",
    )
    access_log: bool = Field(default=True, description="Enable access logging")


class VolsungConfig(BaseModel):
    """Root Volsung configuration.

    Supports loading from:
    1. Environment variables (VOLSUNG_* prefix)
    2. YAML configuration file
    3. Programmatic construction (direct instantiation)

    Priority (highest to lowest):
    1. Environment variables
    2. YAML file values
    3. Default values

    Examples:
        # Load from environment only
        config = VolsungConfig.from_env()

        # Load from YAML file
        config = VolsungConfig.from_yaml("config.yaml")

        # Load with automatic discovery
        config = VolsungConfig.load()

        # Direct construction
        config = VolsungConfig(
            tts=TTSConfig(enabled=True),
            server=ServerConfig(port=8080)
        )
    """

    tts: TTSConfig = Field(default_factory=TTSConfig, description="TTS module config")
    music: MusicConfig = Field(
        default_factory=MusicConfig, description="Music module config"
    )
    sfx: SFXConfig = Field(default_factory=SFXConfig, description="SFX module config")
    resources: ResourceConfig = Field(
        default_factory=ResourceConfig, description="Resource management config"
    )
    server: ServerConfig = Field(
        default_factory=ServerConfig, description="Server config"
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> VolsungConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            VolsungConfig instance

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the YAML is invalid
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Merge with environment variables (env takes precedence)
        env_data = cls._load_env_vars()
        merged = cls._deep_merge(data, env_data)

        return cls.model_validate(merged)

    @classmethod
    def from_env(cls) -> VolsungConfig:
        """Load configuration from environment variables only.

        Environment variables use VOLSUNG_* prefix with double underscore
        as nested separator:
        - VOLSUNG_TTS__ENABLED=true
        - VOLSUNG_MUSIC__MODEL=facebook/musicgen-medium
        - VOLSUNG_SERVER__PORT=8080

        Returns:
            VolsungConfig instance
        """
        data = cls._load_env_vars()
        return cls.model_validate(data)

    @classmethod
    def load(
        cls, config_path: Optional[str | Path] = None, env: bool = True
    ) -> VolsungConfig:
        """Load configuration with automatic discovery.

        Tries (in order):
        1. Explicit config_path if provided
        2. Default locations: ./volsung.yaml, ./config.yaml, /etc/volsung/config.yaml
        3. Environment variables
        4. Default values

        Args:
            config_path: Explicit path to YAML config file
            env: Whether to also load environment variables (takes precedence)

        Returns:
            VolsungConfig instance
        """
        data: dict = {}

        # Try explicit path first
        if config_path:
            path = Path(config_path)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}

        # Try default locations
        if not data:
            default_paths = [
                Path("volsung.yaml"),
                Path("config.yaml"),
                Path("/etc/volsung/config.yaml"),
            ]
            for default_path in default_paths:
                if default_path.exists():
                    with open(default_path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                    break

        # Merge with environment variables if requested
        if env:
            env_data = cls._load_env_vars()
            data = cls._deep_merge(data, env_data)

        return cls.model_validate(data)

    @classmethod
    def _load_env_vars(cls) -> dict:
        """Load configuration from environment variables.

        Converts VOLSUNG_* env vars to nested dict structure:
        - VOLSUNG_TTS__ENABLED=true -> tts.enabled=true
        - VOLSUNG_SERVER__PORT=8000 -> server.port=8000

        Returns:
            Nested dictionary of config values
        """
        prefix = "VOLSUNG_"
        result: dict = {}

        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue

            # Remove prefix and split by __ for nesting
            config_key = key[len(prefix) :].lower()
            parts = config_key.split("__")

            # Convert value to appropriate type
            parsed_value = cls._parse_env_value(value)

            # Build nested dict
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = parsed_value

        return result

    @staticmethod
    def _parse_env_value(value: str) -> bool | int | float | str:
        """Parse environment variable value to appropriate type.

        Order of parsing attempts:
        1. Boolean (true/false, yes/no, 1/0)
        2. Integer
        3. Float
        4. String (as fallback)

        Args:
            value: Raw environment variable value

        Returns:
            Parsed value
        """
        value = value.strip()
        value_lower = value.lower()

        # Boolean
        if value_lower in ("true", "yes", "1"):
            return True
        if value_lower in ("false", "no", "0"):
            return False

        # Integer
        try:
            return int(value)
        except ValueError:
            pass

        # Float
        try:
            return float(value)
        except ValueError:
            pass

        # String
        return value

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Deep merge two dictionaries (override takes precedence).

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = VolsungConfig._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save YAML file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.model_dump(mode="json"),
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    def get_idle_timeout(self, module: str) -> int:
        """Get idle timeout for a specific module.

        Args:
            module: Module name ("tts", "music", "sfx")

        Returns:
            Idle timeout in seconds
        """
        timeouts = {
            "tts": self.tts.idle_timeout,
            "music": self.music.idle_timeout,
            "sfx": self.sfx.idle_timeout,
        }
        return timeouts.get(module, 300)

    def is_module_enabled(self, module: str) -> bool:
        """Check if a module is enabled.

        Args:
            module: Module name ("tts", "music", "sfx")

        Returns:
            True if module is enabled
        """
        modules = {
            "tts": self.tts.enabled,
            "music": self.music.enabled,
            "sfx": self.sfx.enabled,
        }
        return modules.get(module, False)


# Global configuration instance (lazy-loaded)
_config: Optional[VolsungConfig] = None


def get_config() -> VolsungConfig:
    """Get the global configuration instance.

    The configuration is loaded once and cached. Subsequent calls
    return the same instance.

    Returns:
        Global VolsungConfig instance
    """
    global _config
    if _config is None:
        _config = VolsungConfig.load()
    return _config


def set_config(config: VolsungConfig) -> None:
    """Set the global configuration instance.

    This is useful for testing or when configuration needs to be
    changed at runtime.

    Args:
        config: New configuration instance
    """
    global _config
    _config = config


def reload_config(config_path: Optional[str | Path] = None) -> VolsungConfig:
    """Reload the global configuration.

    Args:
        config_path: Optional explicit config path

    Returns:
        New global configuration instance
    """
    global _config
    _config = VolsungConfig.load(config_path=config_path)
    return _config
