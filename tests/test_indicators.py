from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from crypto_trading_framework.core.indicators import (
    add_all_indicators,
    compute_atr,
    compute_bollinger_bands,
    compute_ema,
    compute_ichimoku,
    compute_macd,
    compute_rsi,
    compute_stochastic,
    compute_volume_profile,
)


@pytest.fixture
def sample_ohlcv():
    """Membuat sample OHLCV data untuk testing."""
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.random.rand(n) * 2
    low = close - np.random.rand(n) * 2
    volume = np.random.randint(100, 1000, n).astype(float)
    open_price = close + np.random.randn(n) * 0.3

    return pl.DataFrame({
        "timestamp": pl.datetime_range(
            datetime(2023, 1, 1), datetime(2023, 1, 1) + timedelta(minutes=n-1),
            interval="1m",
            eager=True,
        ),
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_compute_ema(sample_ohlcv):
    df = compute_ema(sample_ohlcv, column="close", periods=[20, 50])
    assert "ema_20" in df.columns
    assert "ema_50" in df.columns
    assert df["ema_20"].null_count() == 0
    assert df["ema_50"].null_count() == 0


def test_compute_bollinger_bands(sample_ohlcv):
    df = compute_bollinger_bands(sample_ohlcv, column="close", period=20, multiplier=2.0)
    assert "bb_upper" in df.columns
    assert "bb_middle" in df.columns
    assert "bb_lower" in df.columns
    assert "bb_width" in df.columns
    assert (df["bb_upper"] >= df["bb_middle"]).all()
    assert (df["bb_middle"] >= df["bb_lower"]).all()


def test_compute_rsi(sample_ohlcv):
    df = compute_rsi(sample_ohlcv, column="close", period=14)
    assert "rsi" in df.columns
    valid = df.filter(pl.col("rsi").is_not_null())
    assert valid["rsi"].min() >= 0
    assert valid["rsi"].max() <= 100


def test_compute_stochastic(sample_ohlcv):
    df = compute_stochastic(sample_ohlcv)
    assert "stoch_k" in df.columns
    assert "stoch_d" in df.columns
    valid = df.filter(pl.col("stoch_k").is_not_null())
    assert valid["stoch_k"].min() >= 0
    assert valid["stoch_k"].max() <= 100


def test_compute_atr(sample_ohlcv):
    df = compute_atr(sample_ohlcv, period=14)
    assert "atr" in df.columns
    valid = df.filter(pl.col("atr").is_not_null())
    assert valid["atr"].null_count() == 0
    assert valid["atr"].min() > 0


def test_compute_macd(sample_ohlcv):
    df = compute_macd(sample_ohlcv, column="close", fast=12, slow=26, signal=9)
    assert "macd" in df.columns
    assert "macd_signal" in df.columns
    assert "macd_hist" in df.columns


def test_compute_ichimoku(sample_ohlcv):
    df = compute_ichimoku(sample_ohlcv)
    assert "ichimoku_tenkan" in df.columns
    assert "ichimoku_kijun" in df.columns
    assert "ichimoku_senkou_a" in df.columns
    assert "ichimoku_senkou_b" in df.columns


def test_compute_volume_profile(sample_ohlcv):
    df = compute_volume_profile(sample_ohlcv, bins=20)
    assert "volume_profile_p50" in df.columns
    assert "volume_profile_skew" in df.columns


def test_add_all_indicators(sample_ohlcv):
    df = add_all_indicators(sample_ohlcv)
    expected_cols = [
        "ema_20", "ema_50", "ema_200",
        "bb_upper", "bb_middle", "bb_lower", "bb_width",
        "rsi", "stoch_k", "stoch_d",
        "atr", "macd", "macd_signal", "macd_hist",
        "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a", "ichimoku_senkou_b",
        "volume_profile_p50", "volume_profile_skew", "volume_profile_kurtosis",
        "volatility", "autocorrelation", "kurtosis",
        "tick_imbalance", "trade_velocity", "spread_dynamics",
        "returns_1d", "returns_5d",
    ]
    for col in expected_cols:
        assert col in df.columns
