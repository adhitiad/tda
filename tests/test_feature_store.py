"""
Tests for feature store module.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl

from crypto_trading_framework.ml.feature_store import (
    _get_engine,
    compute_and_store_features,
    create_feature_store_schema,
    drop_feature_store_schema,
    get_features,
    get_latest_features,
    invalidate_feature_cache,
)


def _make_ohlcv_df(rows=10):
    now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    df = pl.DataFrame({
        "timestamp": [now + timedelta(hours=i) for i in range(rows)],
        "open": [100.0 + i for i in range(rows)],
        "high": [101.0 + i for i in range(rows)],
        "low": [99.0 + i for i in range(rows)],
        "close": [100.5 + i for i in range(rows)],
        "volume": [1000.0 + i * 10 for i in range(rows)],
    })
    return df


def _use_sqlite_fallback(monkeypatch):
    monkeypatch.setenv("TIMESCALE_SERVICE_URL", "sqlite:///:memory:")


class TestFeatureStoreSchema:
    def test_create_schema_sqlite(self, monkeypatch):
        _use_sqlite_fallback(monkeypatch)
        create_feature_store_schema()
        from sqlalchemy import inspect
        inspector = inspect(_get_engine())
        assert "feature_store" in inspector.get_table_names()

    def test_drop_schema(self, monkeypatch):
        _use_sqlite_fallback(monkeypatch)
        create_feature_store_schema()
        drop_feature_store_schema()
        from sqlalchemy import inspect
        inspector = inspect(_get_engine())
        assert "feature_store" not in inspector.get_table_names()


class TestComputeAndStoreFeatures:
    def test_store_features(self, monkeypatch):
        _use_sqlite_fallback(monkeypatch)
        create_feature_store_schema()
        df = _make_ohlcv_df().to_pandas()
        rows = compute_and_store_features(df, "BTC/USDT", "1h")
        assert rows == 10

    def test_store_features_empty_df(self, monkeypatch):
        _use_sqlite_fallback(monkeypatch)
        create_feature_store_schema()
        df = pd.DataFrame()
        rows = compute_and_store_features(df, "BTC/USDT", "1h")
        assert rows == 0


class TestGetFeatures:
    def test_get_features_empty(self, monkeypatch):
        _use_sqlite_fallback(monkeypatch)
        drop_feature_store_schema()
        create_feature_store_schema()
        df = get_features("BTC/USDT", "1h")
        assert df.empty

    def test_get_features_roundtrip(self, monkeypatch):
        _use_sqlite_fallback(monkeypatch)
        drop_feature_store_schema()
        create_feature_store_schema()
        df = _make_ohlcv_df().to_pandas()
        compute_and_store_features(df, "BTC/USDT", "1h")
        result = get_features("BTC/USDT", "1h")
        assert not result.empty
        assert len(result) == 10
        assert "timestamp" in result.columns
        assert "rsi" in result.columns
        assert "macd_hist" in result.columns

    def test_get_features_with_time_range(self, monkeypatch):
        _use_sqlite_fallback(monkeypatch)
        drop_feature_store_schema()
        create_feature_store_schema()
        df = _make_ohlcv_df(rows=10).to_pandas()
        compute_and_store_features(df, "BTC/USDT", "1h")
        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 12, 4, 0, tzinfo=timezone.utc)
        result = get_features("BTC/USDT", "1h", start=start, end=end)
        assert len(result) == 1


class TestGetLatestFeatures:
    def test_latest_features_from_db(self, monkeypatch):
        _use_sqlite_fallback(monkeypatch)
        create_feature_store_schema()
        df = _make_ohlcv_df(rows=10).to_pandas()
        compute_and_store_features(df, "BTC/USDT", "1h")
        result = get_latest_features("BTC/USDT", "1h", n_rows=5)
        assert result is not None
        assert len(result) == 5

    def test_latest_features_cache_miss(self, monkeypatch):
        _use_sqlite_fallback(monkeypatch)
        drop_feature_store_schema()
        create_feature_store_schema()
        result = get_latest_features("BTC/USDT", "1h", n_rows=5)
        assert result is None


class TestInvalidateCache:
    def test_invalidate_no_error(self, monkeypatch):
        _use_sqlite_fallback(monkeypatch)
        with patch("crypto_trading_framework.redis_cache.get_redis_cache") as mock_cache:
            mock_cache.return_value._client = MagicMock()
            invalidate_feature_cache("BTC/USDT", "1h")
            mock_cache.return_value._client.keys.assert_called_once()
