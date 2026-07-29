from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from crypto_trading_framework.ml.backtest import Backtester


@pytest.fixture
def sample_backtest_df():
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.random.rand(n) * 2
    low = close - np.random.rand(n) * 2
    prob = np.clip(np.random.beta(2, 5, n), 0, 1)

    return pl.DataFrame({
        "timestamp": pl.datetime_range(
            datetime(2023, 1, 1), datetime(2023, 1, 1) + timedelta(hours=n-1),
            interval="1h",
            eager=True,
        ),
        "close": close,
        "high": high,
        "low": low,
        "prob": prob,
        "atr": np.random.rand(n) * 2,
    })


def test_backtester_basic_run(sample_backtest_df):
    bt = Backtester(tp_pct=0.03, sl_pct=0.015, initial_capital=10000.0)
    result = bt.run_backtest(sample_backtest_df, threshold=0.8)
    assert "total_trades" in result
    assert "winrate" in result
    assert "final_capital" in result


def test_backtester_transaction_costs(sample_backtest_df):
    bt = Backtester(
        tp_pct=0.03, sl_pct=0.015,
        transaction_fee_pct=0.001,
        slippage_pct=0.0005,
    )
    result = bt.run_backtest(sample_backtest_df, threshold=0.8)
    assert result["total_fees"] >= 0
    assert result["total_slippage"] >= 0


def test_backtester_trailing_stop(sample_backtest_df):
    bt = Backtester(
        tp_pct=0.03, sl_pct=0.015,
        trailing_stop_enabled=True,
        trailing_stop_activation_pct=0.02,
        trailing_stop_distance_pct=0.01,
    )
    result = bt.run_backtest(sample_backtest_df, threshold=0.8)
    assert "total_trades" in result


def test_backtester_max_drawdown_circuit_breaker(sample_backtest_df):
    bt = Backtester(
        tp_pct=0.03, sl_pct=0.015,
        max_drawdown_pct=0.05,
    )
    result = bt.run_backtest(sample_backtest_df, threshold=0.8)
    # Toleransi margin 1% untuk fee dan slippage saat posisi ditutup oleh circuit breaker
    assert result["max_drawdown_pct"] <= 6.0 or result["total_trades"] == 0


def test_backtester_atr_sizing(sample_backtest_df):
    bt = Backtester(
        tp_pct=0.03, sl_pct=0.015,
        position_size_method="atr",
        atr_multiplier=1.0,
        max_risk_per_trade=0.02,
    )
    result = bt.run_backtest(sample_backtest_df, threshold=0.8)
    assert "total_trades" in result
