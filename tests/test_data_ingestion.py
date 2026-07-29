"""
Tests for data ingestion database and cache integration.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from crypto_trading_framework.data_ingestion import DataIngestion


def _make_polars_df(rows=5):
    now = datetime(2024, 1, 1, 12, 0, 0)
    return pl.DataFrame({
        "timestamp": [now + timedelta(hours=i) for i in range(rows)],
        "open": [100.0 + i for i in range(rows)],
        "high": [101.0 + i for i in range(rows)],
        "low": [99.0 + i for i in range(rows)],
        "close": [100.5 + i for i in range(rows)],
        "volume": [1000.0 + i * 10 for i in range(rows)],
    })


@pytest.fixture
def mock_db():
    with patch("crypto_trading_framework.data_ingestion.DB_AVAILABLE", True):
        with patch("crypto_trading_framework.data_ingestion.upsert_ohlcv_dataframe") as mock_upsert, \
             patch("crypto_trading_framework.data_ingestion.update_ingestion_state") as mock_state, \
             patch("crypto_trading_framework.data_ingestion.get_last_ingested_at", return_value=None) as mock_last, \
             patch("crypto_trading_framework.data_ingestion.get_redis_cache") as mock_cache_factory:
            mock_cache = MagicMock()
            mock_cache_factory.return_value = mock_cache
            yield {
                "upsert": mock_upsert,
                "state": mock_state,
                "last": mock_last,
                "cache": mock_cache,
            }


def test_persist_to_db(mock_db):
    ingestion = DataIngestion.__new__(DataIngestion)
    ingestion.symbol = "BTC/USDT"
    df = _make_polars_df()
    mock_db["upsert"].return_value = 5
    rows = ingestion.persist_to_db(df, source="ccxt", timeframe="1h")
    assert rows == 5
    mock_db["upsert"].assert_called_once()
    mock_db["state"].assert_called_once()


def test_persist_to_db_empty(mock_db):
    ingestion = DataIngestion.__new__(DataIngestion)
    ingestion.symbol = "BTC/USDT"
    rows = ingestion.persist_to_db(None, source="ccxt")
    assert rows == 0
    mock_db["upsert"].assert_not_called()


def test_normalize_symbol():
    ingestion = DataIngestion.__new__(DataIngestion)
    ingestion.symbol = "BTC/USDT:USDT"
    assert ingestion._normalize_symbol() == "BTC-USDTUSDT"


def test_cache_data(mock_db):
    ingestion = DataIngestion.__new__(DataIngestion)
    ingestion.symbol = "BTC/USDT"
    df = _make_polars_df()
    mock_db["cache"].cache_dataframe.return_value = True
    result = ingestion.cache_data(df, timeframe="1h")
    assert result is True
    mock_db["cache"].cache_ohlcv.assert_called_once()


def test_get_cached_data(mock_db):
    mock_db["cache"].get_ohlcv.return_value = None
    ingestion = DataIngestion.__new__(DataIngestion)
    ingestion.symbol = "BTC/USDT"
    result = ingestion.get_cached_data(timeframe="1h")
    assert result is None
    mock_db["cache"].get_ohlcv.assert_called_once_with("BTC-USDT", "1h")


def test_get_cached_data_returns_polars(mock_db):
    mock_df = _make_polars_df().to_pandas()
    mock_db["cache"].get_ohlcv.return_value = mock_df
    ingestion = DataIngestion.__new__(DataIngestion)
    ingestion.symbol = "BTC/USDT"
    result = ingestion.get_cached_data(timeframe="1h")
    assert result is not None
    assert isinstance(result, pl.DataFrame)
    mock_db["cache"].get_ohlcv.assert_called_once_with("BTC-USDT", "1h")


def test_fetch_ohlcv_with_persistence_uses_cache(mock_db):
    ingestion = DataIngestion.__new__(DataIngestion)
    ingestion.symbol = "BTC/USDT"
    cached_df = _make_polars_df().to_pandas()
    mock_db["cache"].get_ohlcv.return_value = cached_df

    result = ingestion.get_cached_data(timeframe="1h")
    assert result is not None
    mock_db["cache"].get_ohlcv.assert_called_once()
