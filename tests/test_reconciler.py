"""
Tests for data reconciliation module.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import polars as pl
import pytest

from crypto_trading_framework.db.database import (
    create_schema,
    query_ohlcv,
    reset_database,
)
from crypto_trading_framework.ml.feature_store import (
    create_feature_store_schema,
    drop_feature_store_schema,
)
from crypto_trading_framework.reconciler import DataReconciler


def _use_sqlite_fallback(monkeypatch):
    monkeypatch.setenv("TIMESCALE_SERVICE_URL", "sqlite:///:memory:")


def _make_exchange_df(rows=5):
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
def reconciled_db(monkeypatch):
    _use_sqlite_fallback(monkeypatch)
    reset_database()
    create_schema()
    from crypto_trading_framework.db.database import drop_schema
    drop_schema()
    create_schema()
    drop_feature_store_schema()
    create_feature_store_schema()
    yield
    reset_database()


@pytest.mark.asyncio
async def test_reconcile_no_gaps(reconciled_db, monkeypatch):
    config = {
        "data": {
            "exchange_id": "okx",
            "symbols": ["BTC/USDT:USDT"],
            "timeframes": ["h1"],
            "lookback": 2000,
        }
    }

    with patch("crypto_trading_framework.reconciler.DataIngestion") as MockIngestion:
        mock_ingestion = MagicMock()
        exchange_df = _make_exchange_df(rows=5)
        mock_ingestion.fetch_ohlcv_async = AsyncMock(return_value=exchange_df)
        mock_ingestion.__enter__ = MagicMock(return_value=mock_ingestion)
        mock_ingestion.__exit__ = MagicMock(return_value=False)
        MockIngestion.return_value = mock_ingestion

        reconciler = DataReconciler(config)

        all_pdf = exchange_df.to_pandas()
        all_pdf["symbol"] = "BTC-USDTUSDT"
        all_pdf["timeframe"] = "h1"
        all_pdf["source"] = "ccxt"
        all_pdf["updated_at"] = pd.Timestamp.now("UTC")
        all_pdf["is_interpolated"] = 0.0

        from crypto_trading_framework.db.database import upsert_ohlcv_dataframe
        upsert_ohlcv_dataframe(all_pdf, source="ccxt")

        summary = await reconciler.reconcile("BTC/USDT:USDT", "h1")

    assert summary["gaps_found"] == 0
    assert summary["candles_interpolated"] == 0
    assert summary["candles_inserted"] == 0


@pytest.mark.asyncio
async def test_reconcile_fills_gaps_and_flags(reconciled_db, monkeypatch):
    config = {
        "data": {
            "exchange_id": "okx",
            "symbols": ["BTC/USDT:USDT"],
            "timeframes": ["h1"],
            "lookback": 2000,
        }
    }

    with patch("crypto_trading_framework.reconciler.DataIngestion") as MockIngestion:
        mock_ingestion = MagicMock()
        exchange_df = _make_exchange_df(rows=5)
        mock_ingestion.fetch_ohlcv_async = AsyncMock(return_value=exchange_df)
        mock_ingestion.__enter__ = MagicMock(return_value=mock_ingestion)
        mock_ingestion.__exit__ = MagicMock(return_value=False)
        MockIngestion.return_value = mock_ingestion

        reconciler = DataReconciler(config)

        # Pre-persist only candles 0, 2, 4 (missing 1 and 3)
        partial = exchange_df.filter(
            pl.col("timestamp").is_in([
                exchange_df["timestamp"][0],
                exchange_df["timestamp"][2],
                exchange_df["timestamp"][4],
            ])
        )
        partial_pdf = partial.to_pandas()
        partial_pdf["symbol"] = "BTC-USDTUSDT"
        partial_pdf["timeframe"] = "h1"
        partial_pdf["source"] = "ccxt"
        partial_pdf["updated_at"] = pd.Timestamp.now("UTC")
        partial_pdf["is_interpolated"] = 0.0

        from crypto_trading_framework.db.database import upsert_ohlcv_dataframe
        upsert_ohlcv_dataframe(partial_pdf, source="ccxt")

        summary = await reconciler.reconcile("BTC/USDT:USDT", "h1")

    assert summary["gaps_found"] == 2
    assert summary["candles_interpolated"] == 2
    assert summary["candles_inserted"] >= 2

    result = query_ohlcv(symbol="BTC-USDTUSDT", timeframe="h1")
    assert len(result) == 5
    interpolated = result[result["is_interpolated"] == 1.0]
    assert len(interpolated) == 2


@pytest.mark.asyncio
async def test_reconcile_marks_feature_store(reconciled_db, monkeypatch):
    config = {
        "data": {
            "exchange_id": "okx",
            "symbols": ["BTC/USDT:USDT"],
            "timeframes": ["h1"],
            "lookback": 2000,
        }
    }

    with patch("crypto_trading_framework.reconciler.DataIngestion") as MockIngestion:
        mock_ingestion = MagicMock()
        exchange_df = _make_exchange_df(rows=5)
        mock_ingestion.fetch_ohlcv_async = AsyncMock(return_value=exchange_df)
        mock_ingestion.__enter__ = MagicMock(return_value=mock_ingestion)
        mock_ingestion.__exit__ = MagicMock(return_value=False)
        MockIngestion.return_value = mock_ingestion

        reconciler = DataReconciler(config)

        partial = exchange_df.filter(
            pl.col("timestamp").is_in([
                exchange_df["timestamp"][0],
                exchange_df["timestamp"][2],
                exchange_df["timestamp"][4],
            ])
        )
        partial_pdf = partial.to_pandas()
        partial_pdf["symbol"] = "BTC-USDTUSDT"
        partial_pdf["timeframe"] = "h1"
        partial_pdf["source"] = "ccxt"
        partial_pdf["updated_at"] = pd.Timestamp.now("UTC")
        partial_pdf["is_interpolated"] = 0.0

        from crypto_trading_framework.db.database import upsert_ohlcv_dataframe
        upsert_ohlcv_dataframe(partial_pdf, source="ccxt")

        await reconciler.reconcile("BTC/USDT:USDT", "h1")

    from crypto_trading_framework.ml.feature_store import get_features
    features = get_features(symbol="BTC-USDTUSDT", timeframe="h1")
    assert not features.empty
    assert "is_interpolated" in features.columns
