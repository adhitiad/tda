"""
Tests for observability data provider.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from crypto_trading_framework.db.database import create_schema, reset_database
from crypto_trading_framework.ml.feature_store import (
    create_feature_store_schema,
    drop_feature_store_schema,
)
from crypto_trading_framework.observability import ObservabilityDataProvider


def _use_sqlite_fallback(monkeypatch):
    monkeypatch.setenv("TIMESCALE_SERVICE_URL", "sqlite:///:memory:")


def _make_ohlcv_df(rows=10):
    now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return pl.DataFrame({
        "timestamp": [now + timedelta(hours=i) for i in range(rows)],
        "open": [100.0 + i for i in range(rows)],
        "high": [101.0 + i for i in range(rows)],
        "low": [99.0 + i for i in range(rows)],
        "close": [100.5 + i for i in range(rows)],
        "volume": [1000.0 + i * 10 for i in range(rows)],
    })


@pytest.fixture
def obs_db(monkeypatch, tmp_path):
    _use_sqlite_fallback(monkeypatch)
    reset_database()
    create_schema()
    drop_feature_store_schema()
    create_feature_store_schema()

    backtest_dir = tmp_path / "backtests"
    backtest_dir.mkdir(parents=True, exist_ok=True)
    (backtest_dir / "h1_20260727_212900.json").write_text(
        '{"symbol":"BTC-USDTUSDT","timestamp":"2026-07-27T21:29:00","total_trades":10,"wins":5,"losses":5,"winrate":50.0,"total_pnl":0.0,"return_pct":0.0,"profit_factor":1.0,"max_drawdown_pct":0.0,"final_capital":10000.0,"total_fees":0.0,"total_slippage":0.0,"trades":[]}',
        encoding="utf-8",
    )

    models_dir = tmp_path / "models"
    registry_dir = models_dir / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    (registry_dir / "index.json").write_text(
        '{"models":{"BTC-USDTUSDT:h1":{"version":"v1","model_type":"lstm","symbol":"BTC-USDTUSDT","timeframe":"h1","training_date":"2024-01-01","hyperparameters":{},"metrics":{},"model_path":"models/best.pth","scaler_path":"models/scaler.pkl","meta_path":"models/meta.json","tags":[],"status":"active","rollout":"production"}},"aliases":{"latest":"v1"}}',
        encoding="utf-8",
    )

    yield {"backtest_dir": str(backtest_dir), "models_dir": str(models_dir)}


def test_get_ohlcv_empty(obs_db):
    provider = ObservabilityDataProvider(config={"paths": obs_db, "data": {"symbols": [], "timeframes": []}})
    df = provider.get_ohlcv("BTC/USDT:USDT", "h1")
    assert df.is_empty()


def test_get_ohlcv_with_data(obs_db, monkeypatch):
    _use_sqlite_fallback(monkeypatch)
    pdf = _make_ohlcv_df().to_pandas()
    pdf["symbol"] = "BTC-USDTUSDT"
    pdf["timeframe"] = "h1"
    pdf["source"] = "ccxt"
    from crypto_trading_framework.db.database import upsert_ohlcv_dataframe
    upsert_ohlcv_dataframe(pdf, source="ccxt")

    provider = ObservabilityDataProvider(config={"paths": obs_db, "data": {"symbols": [], "timeframes": []}})
    df = provider.get_ohlcv("BTC/USDT:USDT", "h1")
    assert not df.is_empty()
    assert df.height == 10


def test_get_indicators_with_data(obs_db, monkeypatch):
    _use_sqlite_fallback(monkeypatch)
    pdf = _make_ohlcv_df(rows=250).to_pandas()
    from crypto_trading_framework.ml.feature_store import compute_and_store_features
    compute_and_store_features(pdf, symbol="BTC-USDTUSDT", timeframe="h1")

    provider = ObservabilityDataProvider(config={"paths": obs_db, "data": {"symbols": [], "timeframes": []}})
    df = provider.get_indicators("BTC/USDT:USDT", "h1")
    assert not df.is_empty()
    assert "rsi" in df.columns


def test_get_model_registry(obs_db):
    provider = ObservabilityDataProvider(config={"paths": obs_db, "data": {"symbols": [], "timeframes": []}})
    models = provider.get_model_registry()
    assert len(models) == 1
    assert models[0]["symbol"] == "BTC-USDTUSDT"


def test_get_backtest_results(obs_db):
    provider = ObservabilityDataProvider(config={"paths": obs_db, "data": {"symbols": [], "timeframes": []}})
    results = provider.get_backtest_results()
    assert len(results) == 1
    assert results[0]["total_trades"] == 10


def test_get_latest_backtest(obs_db):
    provider = ObservabilityDataProvider(config={"paths": obs_db, "data": {"symbols": [], "timeframes": []}})
    result = provider.get_latest_backtest("BTC-USDTUSDT", "h1")
    assert result is not None
    assert result["winrate"] == 50.0


def test_get_db_health_healthy(obs_db):
    provider = ObservabilityDataProvider(config={"paths": obs_db, "data": {"symbols": [], "timeframes": []}})
    health = provider.get_db_health()
    assert health["status"] == "healthy"


def test_get_redis_health_unavailable(obs_db):
    provider = ObservabilityDataProvider(config={"paths": obs_db, "data": {"symbols": [], "timeframes": []}})
    health = provider.get_redis_health()
    assert health["status"] in ("unavailable", "unhealthy")


def test_get_smart_money_snapshot(obs_db):
    provider = ObservabilityDataProvider(config={"paths": obs_db, "data": {"symbols": [], "timeframes": []}})
    snapshot = provider.get_smart_money_snapshot("BTC/USDT:USDT")
    assert "composite_signal" in snapshot
    assert "veto_status" in snapshot


def test_get_shadow_trader_performance_empty(obs_db):
    provider = ObservabilityDataProvider(config={"paths": obs_db, "data": {"symbols": [], "timeframes": []}})
    perf = provider.get_shadow_trader_performance()
    assert isinstance(perf, dict)
