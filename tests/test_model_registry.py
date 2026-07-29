"""
Tests for model registry.
"""

import os
import tempfile

import pytest

from crypto_trading_framework.ml.model_registry import ModelRegistry, ModelRegistryConfig


@pytest.fixture
def registry_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def registry(registry_dir):
    return ModelRegistry(ModelRegistryConfig(registry_dir=registry_dir))


class TestModelRegistryConfig:
    def test_defaults(self):
        cfg = ModelRegistryConfig()
        assert cfg.enabled is True
        assert cfg.registry_dir == "models/registry"
        assert cfg.active_version == "latest"

    def test_custom(self):
        cfg = ModelRegistryConfig(enabled=False, registry_dir="tmp", active_version="v1")
        assert cfg.enabled is False
        assert cfg.registry_dir == "tmp"
        assert cfg.active_version == "v1"


class TestModelRegistry:
    def test_register_and_get(self, registry):
        meta = registry.register(
            model_type="lstm",
            symbol="BTC/USDT",
            timeframe="h1",
            model_path="models/best.pt",
            scaler_path="models/scaler.pkl",
            meta_path="models/meta.json",
            hyperparameters={"hidden_size": 64},
            metrics={"accuracy": 0.85},
        )
        assert meta.version is not None
        retrieved = registry.get("BTC/USDT", "h1")
        assert retrieved is not None
        assert retrieved.symbol == "BTC/USDT"
        assert retrieved.model_type == "lstm"
        assert retrieved.hyperparameters["hidden_size"] == 64

    def test_register_creates_index_file(self, registry, registry_dir):
        registry.register(
            model_type="lstm",
            symbol="BTC/USDT",
            timeframe="h1",
            model_path="models/best.pt",
            scaler_path="models/scaler.pkl",
            meta_path="models/meta.json",
        )
        assert os.path.exists(os.path.join(registry_dir, "index.json"))

    def test_list_versions(self, registry):
        registry.register(
            model_type="lstm",
            symbol="BTC/USDT",
            timeframe="h1",
            model_path="models/best.pt",
            scaler_path="models/scaler.pkl",
            meta_path="models/meta.json",
        )
        versions = registry.list_versions("BTC/USDT", "h1")
        assert len(versions) == 1

    def test_get_unknown_symbol_returns_none(self, registry):
        assert registry.get("UNKNOWN", "h1") is None

    def test_promote(self, registry):
        registry.register(
            model_type="lstm",
            symbol="BTC/USDT",
            timeframe="h1",
            model_path="models/best.pt",
            scaler_path="models/scaler.pkl",
            meta_path="models/meta.json",
            rollout="staging",
        )
        version = registry.list_versions("BTC/USDT", "h1")[0]
        registry.promote("BTC/USDT", "h1", version, rollout="production")
        meta = registry.get("BTC/USDT", "h1")
        assert meta.rollout == "production"

    def test_rollback(self, registry):
        registry.register(
            model_type="lstm",
            symbol="BTC/USDT",
            timeframe="h1",
            model_path="models/best.pt",
            scaler_path="models/scaler.pkl",
            meta_path="models/meta.json",
        )
        version = registry.list_versions("BTC/USDT", "h1")[0]
        rolled = registry.rollback("BTC/USDT", "h1", version)
        assert rolled.version == version
        assert rolled.status == "active"

    def test_delete(self, registry):
        registry.register(
            model_type="lstm",
            symbol="BTC/USDT",
            timeframe="h1",
            model_path="models/best.pt",
            scaler_path="models/scaler.pkl",
            meta_path="models/meta.json",
        )
        version = registry.list_versions("BTC/USDT", "h1")[0]
        registry.delete("BTC/USDT", "h1", version)
        assert registry.get("BTC/USDT", "h1") is None

    def test_list_all(self, registry):
        registry.register(
            model_type="lstm",
            symbol="BTC/USDT",
            timeframe="h1",
            model_path="models/best.pt",
            scaler_path="models/scaler.pkl",
            meta_path="models/meta.json",
        )
        registry.register(
            model_type="lstm",
            symbol="ETH/USDT",
            timeframe="h1",
            model_path="models/eth.pt",
            scaler_path="models/eth_scaler.pkl",
            meta_path="models/eth_meta.json",
        )
        all_models = registry.list_all()
        assert len(all_models) == 2

    def test_set_active(self, registry):
        registry.register(
            model_type="lstm",
            symbol="BTC/USDT",
            timeframe="h1",
            model_path="models/best.pt",
            scaler_path="models/scaler.pkl",
            meta_path="models/meta.json",
        )
        version = registry.list_versions("BTC/USDT", "h1")[0]
        registry.set_active(version)
        assert registry._index["aliases"]["active"] == version
