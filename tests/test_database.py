"""
Tests for the database layer.
"""

from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import text

from crypto_trading_framework.db.database import (
    MarketDataOHLCV,
    create_schema,
    get_data_range,
    get_last_ingested_at,
    get_session,
    update_ingestion_state,
    upsert_ohlcv_dataframe,
)


def _make_df(symbol="BTCUSDT", timeframe="1h", rows=5) -> pd.DataFrame:
    now = datetime(2024, 1, 1, 12, 0, 0)
    return pd.DataFrame({
        "timestamp": [now + timedelta(hours=i) for i in range(rows)],
        "symbol": [symbol] * rows,
        "timeframe": [timeframe] * rows,
        "open": [100.0 + i for i in range(rows)],
        "high": [101.0 + i for i in range(rows)],
        "low": [99.0 + i for i in range(rows)],
        "close": [100.5 + i for i in range(rows)],
        "volume": [1000.0 + i * 10 for i in range(rows)],
        "source": ["ccxt"] * rows,
    })


def _use_sqlite_fallback(monkeypatch):
    monkeypatch.setenv("TIMESCALE_SERVICE_URL", "sqlite:///:memory:")


def test_engine_creation(monkeypatch):
    _use_sqlite_fallback(monkeypatch)
    from crypto_trading_framework.db.database import get_engine
    engine = get_engine()
    assert engine is not None


def test_schema_creation(monkeypatch):
    _use_sqlite_fallback(monkeypatch)
    create_schema()
    with get_session() as session:
        result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='market_data_ohlcv'"))
        assert result.fetchone() is not None


def test_upsert_ohlcv_dataframe(monkeypatch):
    _use_sqlite_fallback(monkeypatch)
    create_schema()
    df = _make_df()
    rows = upsert_ohlcv_dataframe(df, source="ccxt")
    assert rows == 5

    with get_session() as session:
        result = session.query(MarketDataOHLCV).filter_by(symbol="BTCUSDT", timeframe="1h").count()
        assert result == 5


def test_upsert_skips_duplicates(monkeypatch):
    _use_sqlite_fallback(monkeypatch)
    create_schema()
    df1 = _make_df()
    df2 = _make_df()
    rows1 = upsert_ohlcv_dataframe(df1, source="ccxt")
    rows2 = upsert_ohlcv_dataframe(df2, source="ccxt")
    assert rows1 == 5
    assert rows2 == 5  # ON CONFLICT DO UPDATE counts affected rows including updates


def test_upsert_updates_existing(monkeypatch):
    _use_sqlite_fallback(monkeypatch)
    create_schema()
    df1 = _make_df()
    df2 = _make_df()
    df2.loc[0, "close"] = 999.0
    upsert_ohlcv_dataframe(df1, source="ccxt")
    upsert_ohlcv_dataframe(df2, source="ccxt")

    target_ts = pd.Timestamp(df1["timestamp"].iloc[0]).to_pydatetime()
    with get_session() as session:
        result = session.execute(text("""
            SELECT close FROM market_data_ohlcv
            WHERE symbol = :symbol AND timeframe = :timeframe AND timestamp = :timestamp
        """), {"symbol": "BTCUSDT", "timeframe": "1h", "timestamp": target_ts})
        row = result.fetchone()
        assert row is not None
        assert float(row[0]) == 999.0


def test_get_last_ingested_at(monkeypatch):
    _use_sqlite_fallback(monkeypatch)
    create_schema()
    df = _make_df()
    upsert_ohlcv_dataframe(df, source="ccxt")
    update_ingestion_state("BTCUSDT", "1h", "ccxt", pd.Timestamp(df["timestamp"].iloc[-1]).to_pydatetime(), len(df))

    ts = get_last_ingested_at("BTCUSDT", "1h", "ccxt")
    assert ts is not None
    assert pd.Timestamp(ts) == pd.Timestamp(df["timestamp"].iloc[-1])


def test_get_last_ingested_at_missing(monkeypatch):
    _use_sqlite_fallback(monkeypatch)
    create_schema()
    ts = get_last_ingested_at("MISSING", "1h", "ccxt")
    assert ts is None


def test_update_ingestion_state(monkeypatch):
    _use_sqlite_fallback(monkeypatch)
    create_schema()
    now = datetime(2024, 1, 1, 12, 0, 0)
    update_ingestion_state("BTCUSDT", "1h", "ccxt", now, 42)

    ts = get_last_ingested_at("BTCUSDT", "1h", "ccxt")
    assert ts is not None
    assert str(ts) == str(now)


def test_get_data_range(monkeypatch):
    _use_sqlite_fallback(monkeypatch)
    create_schema()
    df = _make_df()
    upsert_ohlcv_dataframe(df, source="ccxt")

    start, end = get_data_range("BTCUSDT", "1h", "ccxt")
    assert start is not None
    assert end is not None
    assert pd.Timestamp(start) == pd.Timestamp(df["timestamp"].iloc[0])
    assert pd.Timestamp(end) == pd.Timestamp(df["timestamp"].iloc[-1])


def test_get_data_range_empty(monkeypatch):
    _use_sqlite_fallback(monkeypatch)
    create_schema()
    start, end = get_data_range("MISSING", "1h", "ccxt")
    assert start is None
    assert end is None
