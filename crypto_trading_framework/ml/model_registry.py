"""
Model registry and versioning for tracking trained models.

Stores metadata in JSON format under the configured registry directory.
Supports versioning, A/B rollout, and rollback.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from crypto_trading_framework.core.logging import get_logger

logger = get_logger("model_registry")


@dataclass
class ModelRegistryConfig:
    """Configuration for model registry."""

    model_config: dict | None = None
    enabled: bool = True
    registry_dir: str = "models/registry"
    active_version: str = "latest"
    canary: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelMetadata:
    """Metadata for a single trained model version."""

    version: str
    model_type: str
    symbol: str
    timeframe: str
    training_date: str
    hyperparameters: dict[str, Any]
    metrics: dict[str, Any]
    model_path: str
    scaler_path: str
    meta_path: str
    tags: list[str] = field(default_factory=list)
    status: str = "active"
    rollout: str = "production"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "model_type": self.model_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "training_date": self.training_date,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "model_path": self.model_path,
            "scaler_path": self.scaler_path,
            "meta_path": self.meta_path,
            "tags": self.tags,
            "status": self.status,
            "rollout": self.rollout,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        return cls(
            version=data["version"],
            model_type=data["model_type"],
            symbol=data["symbol"],
            timeframe=data["timeframe"],
            training_date=data["training_date"],
            hyperparameters=data.get("hyperparameters", {}),
            metrics=data.get("metrics", {}),
            model_path=data["model_path"],
            scaler_path=data["scaler_path"],
            meta_path=data["meta_path"],
            tags=data.get("tags", []),
            status=data.get("status", "active"),
            rollout=data.get("rollout", "production"),
        )


class ModelRegistry:
    """Lightweight model registry backed by JSON files."""

    def __init__(self, config: ModelRegistryConfig):
        self.config = config
        self.registry_dir = Path(config.registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.registry_dir / "index.json"
        self._index: dict[str, Any] = self._load_index()

    def _load_index(self) -> dict[str, Any]:
        if self.index_path.exists():
            with open(self.index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"models": {}, "aliases": {"latest": None}}

    def _save_index(self):
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2, default=str)

    def register(
        self,
        model_type: str,
        symbol: str,
        timeframe: str,
        model_path: str,
        scaler_path: str,
        meta_path: str,
        hyperparameters: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        rollout: str = "production",
    ) -> ModelMetadata:
        """
        Register a new model version.

        :return: Registered ModelMetadata
        """
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        metadata = ModelMetadata(
            version=version,
            model_type=model_type,
            symbol=symbol,
            timeframe=timeframe,
            training_date=datetime.now(timezone.utc).isoformat(),
            hyperparameters=hyperparameters or {},
            metrics=metrics or {},
            model_path=model_path,
            scaler_path=scaler_path,
            meta_path=meta_path,
            tags=tags or [],
            rollout=rollout,
        )

        key = f"{symbol}:{timeframe}"
        self._index["models"][key] = metadata.to_dict()
        self._index["aliases"]["latest"] = version
        self._save_index()

        logger.info(f"[Registry] Registered {key} version={version}")
        return metadata

    def get(self, symbol: str, timeframe: str, version: str | None = None) -> ModelMetadata | None:
        """
        Retrieve model metadata by symbol/timeframe and optional version.

        :param version: If None, uses active_version config or 'latest' alias.
        """
        key = f"{symbol}:{timeframe}"
        model_entry = self._index["models"].get(key)
        if model_entry is None:
            return None

        target_version = version or self.config.active_version
        if target_version == "latest":
            target_version = self._index["aliases"].get("latest")

        if target_version is None or model_entry["version"] == target_version:
            return ModelMetadata.from_dict(model_entry)

        return None

    def list_versions(self, symbol: str, timeframe: str) -> list[str]:
        """List all registered versions for a symbol/timeframe."""
        key = f"{symbol}:{timeframe}"
        model_entry = self._index["models"].get(key)
        if model_entry is None:
            return []
        return [model_entry["version"]]

    def set_active(self, version: str):
        """Set the active version alias."""
        self._index["aliases"]["active"] = version
        self._save_index()

    def promote(self, symbol: str, timeframe: str, version: str, rollout: str = "production"):
        """Promote a model version to a specific rollout stage."""
        key = f"{symbol}:{timeframe}"
        model_entry = self._index["models"].get(key)
        if model_entry is None:
            raise KeyError(f"No model registered for {key}")

        if model_entry["version"] != version:
            raise ValueError(f"Version {version} not found for {key}")

        model_entry["rollout"] = rollout
        self._save_index()
        logger.info(f"[Registry] Promoted {key}:{version} to {rollout}")

    def rollback(self, symbol: str, timeframe: str, version: str) -> ModelMetadata:
        """
        Rollback to a previous model version.

        :return: The rolled-back model metadata
        """
        key = f"{symbol}:{timeframe}"
        model_entry = self._index["models"].get(key)
        if model_entry is None:
            raise KeyError(f"No model registered for {key}")

        if model_entry["version"] != version:
            raise ValueError(f"Version {version} not found for {key}")

        model_entry["status"] = "active"
        self._index["aliases"]["latest"] = version
        self._save_index()
        logger.info(f"[Registry] Rolled back {key} to {version}")
        return ModelMetadata.from_dict(model_entry)

    def delete(self, symbol: str, timeframe: str, version: str):
        """Remove a model version from registry."""
        key = f"{symbol}:{timeframe}"
        model_entry = self._index["models"].get(key)
        if model_entry is None:
            return

        if model_entry["version"] == version:
            del self._index["models"][key]
            if self._index["aliases"].get("latest") == version:
                self._index["aliases"]["latest"] = None
            self._save_index()
            logger.info(f"[Registry] Deleted {key}:{version}")

    def list_all(self) -> list[dict[str, Any]]:
        """List all registered models."""
        return [copy.deepcopy(v) for v in self._index["models"].values()]
