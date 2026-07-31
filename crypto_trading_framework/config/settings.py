import os
import re
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


def _interpolate_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        pattern = re.compile(r"\$\{([^}]+)\}")

        def replacer(match: re.Match) -> str:
            env_var = match.group(1)
            default = ""
            if ":" in env_var:
                env_var, default = env_var.split(":", 1)
            return os.getenv(env_var, default)

        return pattern.sub(replacer, value)
    if isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env_vars(item) for item in value]
    return value


def _load_yaml(path: str) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, overlay: dict) -> dict:
    merged = base.copy()
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="allow",
    )

    config_path: str = "config/base.yaml"
    overlay_path: str | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        config = _load_yaml(self.config_path)
        if self.overlay_path and Path(self.overlay_path).exists():
            overlay = _load_yaml(self.overlay_path)
            config = _deep_merge(config, overlay)
        config = _interpolate_env_vars(config)
        self._raw_config = config

    @property
    def raw_config(self) -> dict:
        return self._raw_config

    def validate(self) -> dict:
        from crypto_trading_framework.core.config_schema import validate_config

        return validate_config(self._raw_config)

    @classmethod
    def load(cls, config_path: str = "config/base.yaml", overlay_path: str | None = None) -> "Settings":
        return Settings(config_path=config_path, overlay_path=overlay_path)


def load_config(config_path: str = "config/base.yaml", overlay_path: str | None = None) -> dict:
    settings = Settings(config_path=config_path, overlay_path=overlay_path)
    return settings.validate()
